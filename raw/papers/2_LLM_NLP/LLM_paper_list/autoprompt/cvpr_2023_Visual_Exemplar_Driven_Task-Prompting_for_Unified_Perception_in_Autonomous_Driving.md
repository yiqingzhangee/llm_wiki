# **Visual Exemplar Driven Task-Prompting for Unified Perception in Autonomous Driving** 

Xiwen Liang[1] , Minzhe Niu[2] , Jianhua Han[2] , Hang Xu[2] , Chunjing Xu[2] , Xiaodan Liang[1] _[†]_ 1Shenzhen Campus of Sun Yat-sen University, 2Huawei Noah’s Ark Lab 

_{_ liangxw29@mail2, liangxd9@mail _}_ .sysu.edu.cn, 

_{_ niuminzhe1, hanjianhua4, xu.hang, xuchunjing _}_ @huawei.com 

## **Abstract** 

_Multi-task learning has emerged as a powerful paradigm to solve a range of tasks simultaneously with good efficiency in both computation resources and inference time. However, these algorithms are designed for different tasks mostly not within the scope of autonomous driving, thus making it hard to compare multi-task methods in autonomous driving. Aiming to enable the comprehensive evaluation of present multi-task learning methods in autonomous driving, we extensively investigate the performance of popular multi-task methods on the large-scale driving dataset, which covers four common perception tasks, i.e., object detection, semantic segmentation, drivable area segmentation, and lane detection. We provide an in-depth analysis of current multitask learning methods under different common settings and find out that the existing methods make progress but there is still a large performance gap compared with single-task baselines. To alleviate this dilemma in autonomous driving, we present an effective multi-task framework, VE-Prompt, which introduces visual exemplars via task-specific prompting to guide the model toward learning high-quality taskspecific representations. Specifically, we generate visual exemplars based on bounding boxes and color-based markers, which provide accurate visual appearances of target categories and further mitigate the performance gap. Furthermore, we bridge transformer-based encoders and convolutional layers for efficient and accurate unified perception in autonomous driving. Comprehensive experimental results on the diverse self-driving dataset BDD100K show that the VE-Prompt improves the multi-task baseline and further surpasses single-task models._ 

## **1. Introduction** 

Multi-task learning (MTL) has been the source of a number of breakthroughs in autonomous driving over the last 

> _†_ Corresponding author. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving_images/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving.pdf-0001-10.png)


**----- Start of picture text -----**<br>
A photo of  a [class]. Text   ... ...<br>Encoder Learnable<br>... [CLASS] Prompts Embed<br>Image EncoderImage  Alignment Image Patch ...<br>(a) Textual Prompt (b) Visual Prompt<br>... ... GeneratorPrompt Task Prompts Head BHead A<br>... Head C<br>Image  Task<br>... Image Encoder Prompting Head D<br>(c) Visual Exemplar Driven Prompt (Ours)<br>Transformer<br>**----- End of picture text -----**<br>


Figure 1. **Comparison of different prompts in computer vision.** (a) Extracting textual prompts from a text encoder to perform image-text alignment [64]. (b) Prepend learnable prompts to the embeddings of image patches [22]. (c) Visual exemplar driven prompts for multi-task learning (ours). The generated task prompts encode high-quality task-specific knowledge for downstream tasks. 

few years [25, 52, 58] and general vision tasks recently [2, 13, 28, 36, 57]. As the foundation of autonomous driving, a robust vision perception system is required to provide critical information, including the position of traffic participants, traffic signals like lights, signs, lanes, and obstacles that influence the drivable space, to ensure driving safety and comfort. These tasks gain knowledge from the same data source and present prominent relationships between each other, like traffic participants, are more likely to appear within drivable spaces and traffic signs may appear near traffic lights, etc. Training these tasks independently is time costing and fails to mine the latent relationship among them. Therefore, it is crucial to solve these multiple tasks simultaneously, which can improve data efficiency and reduce training and inference time. 

Some recent works have attempted to apply unified training on multiple tasks in autonomous training. Uncertainty [23] trains per-pixel depth prediction, semantic segmenta- 

1 

tion, and instance segmentation in a single model. CIL [21] introduces an extra traffic light classifier to learn different traffic patterns following traffic light changes. CP-MTL [4] learns object detection and depth prediction together to identify dangerous traffic scenes. However, these works differ in task types, evaluation matrix, and dataset, making it hard to compare their performances. For example, most of them are developed upon dense prediction [2, 57] and natural language understanding [8,49], rather than being tailored for more common perception tasks for autonomous driving, thus these methods may produce poor results when applied to a self-driving system. As a result, there is an emerging demand for a thorough evaluation of existing multi-task learning methods covering common tasks in autonomous driving. 

In this paper, we focus on heterogeneous multi-task learning in common scenarios of autonomous driving and cover popular self-driving tasks, i.e., object detection, semantic segmentation, drivable area segmentation, and lane detection. We provide a systematic study of present MTL methods on large-scale driving dataset BDD100K [60]. Specifically, we find that task scheduling [28] is better than zeroing loss [53], but worse than pseudo labeling [16] on most tasks. Interestingly, in task-balancing methods, Uncertainty [23] produces satisfactory results on most tasks, while MGDA [43] only performs well on lane detection. This indicates that negative transfer [9], which is a phenomenon that increasing the performance of a model on one task will hurt the performance on another task with different needs, is common among these approaches. 

To mitigate the negative transfer problem, we introduce the visual exemplar-driven task-prompting (shorten as **VEPrompt** ) based on the following motivations: (1) Given the visual clues of each task, the model can extract task-related information from the pre-trained model. Different from current prompting methods which introduce textual prompts [6, 40, 64, 65] or learnable context [22], we leverage exemplars containing information of target objects to generate task-specific prompts by considering that the visual clues should represent the specific task to some extent, and give hints for learning task-specific information; (2) Transformer has achieved competitive performance on many vision tasks but usually requires long training time, thus tackling four tasks simultaneously on a pure transformer is resourceintensive. To overcome this challenge, we efficiently bridge transformer encoders and convolutional layers to build the hybrid multi-task architecture. Extensive experiments show that VE-Prompt surpasses multi-task baselines by a large margin. 

We summarize the main contributions of our work below: 

- We provide an in-depth analysis of current multi-task learning approaches under multiple settings that comply 

with real-world scenarios, consisting of three common multi-task data split settings, two partial-label learning approaches, three task scheduling techniques, and three task balancing strategies. 

- We propose an effective framework VE-Prompt, which utilizes visual exemplars to provide task-specific visual clues and guide the model toward learning high-quality task-specific representations. 

- The VE-Prompt framework is constructed in a computationally efficient way and outperforms competitive multitask methods on all tasks. 

## **2. Related Work** 

**Multi-task Learning** Multi-task learning jointly trains shared parameters on multiple tasks, mining latent information among them to improve efficiency and accuracy. Famous multi-task learning models include Mask R-CNN [17], which applies Faster R-CNN [41] as the backbone and conducts instance segmentation and object detection at the same time. Other methods like Eigen _et al._ [12] address depth prediction, surface normal estimation, semantic labeling tasks, and MultiNet [46] provide prediction on classification, detection, semantic segmentation tasks within a single model. YOLOP [52] leverages CSPDarknet as the backbone, which branches out three task-specific heads for object detection, drivable area segmentation, and lane detection prediction. Standley _et al._ [44] and Christopher _et al._ [13] improves previous multi-task training schema by grouping proper tasks together rather than naively training all tasks together. In this paper, we focus on developing general and effective approaches for multi-task learning in autonomous driving scenarios. 

**Visual Perception for Autonomous Driving** Autonomous driving relies on a perception system to gather information and understand the environment. Visual perception, as the most similar sensing modality to humans, provides high-resolution images that satisfy almost all tasks required for autonomous driving. Some of the tasks have long been studied beyond autonomous driving scenarios. Chen _et al._ [3] predicts 2D object detection from images while Semantic FPN [24] performs semantic segmentation and Lanenet [50] implements lane detection respectively using visual inputs. Though these models are designed for different tasks, they all adopt the backbone-header architecture, some of which even share the same backbone structure like ResNet [18] or transformer [11]. Running independent models for perception tasks separately is a waste of time and computation resources, making an emerging call for the development of a unified perception system. 

**Prompt-based Learning** Prompt-based learning [19,32, 51] is put forward to bridge the gap between pre-training 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving_images/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Multi-task Architecture<br>Multi-task Settings<br>Head A<br>Data Split Task Scheduling Task Balancing Partial-label<br>Head B<br>Shared Learning<br>Encoder Disjoint-normal Uniform sampler Fixed<br>Head C Zeroing loss<br>Disjoint-balance Weighted sampler Uncertainty<br>Full Round-robin GradNorm Pseudo labeling<br>Head D<br>MGDA<br>**----- End of picture text -----**<br>


Figure 2. **The multi-task architecture and settings in our investigation.** We follow the common multi-task architecture where each task shares the same encoder and has its specific head. The multi-task settings focus on three types of task scheduling, four task balancing methods, and two partial-label learning techniques and cover three common data split settings. 

and model tuning in the field of natural language processing. GPT-3 [1] first designs various text prompts according to the property of tasks and treats the downstream task as a masked language modeling problem. Meanwhile, other approaches like [26,34,63] train learnable continuous prompts in the embedding space of the model and achieve competitive performance compared with finetuning. Recently, CLIP [39], which is trained on multi-modality vision-language pairs data, achieved impressive performance on zero-shot image classification by injecting visual categories into the text input as prompts. Subsequent works [14,59,64] further tune CLIP with learnable soft prompts by few-shot supervisions in the field of computer vision, or leverage text features from CLIP to enhance visual representations [6, 40]. Prompt tuning without textual information is introduced by injecting learnable vectors in the input space [22] or inserting lightweight blocks to learn prompts [38]. However, these approaches are tailored for solving downstream tasks independently and are inapplicable to heterogeneous multitask learning. In this work, we design the visual exemplardriven task-prompting to inject task-specific knowledge for heterogeneous multi-task learning. 

## **3. Empirical Study** 

**Multi-task Architectures** Multi-task learning (MTL) architectures apply parameter sharing to learn shared information between different tasks. MTL architectures can be divided into encoder-focused architectures [15, 33, 37, 42] and decoder-focused ones [47, 56, 62] according to parameter sharing scope. Encoder-focused architectures can be further categorized into hard and soft parameter sharing. In this paper, we select the hard parameter-sharing structure as our backbone due to its simplicity and stability. Parameters are only shared in the encoder part of the model followed by task-specific heads. As Figure 2 shows, the image inputs first go through the shared encoder, and then the feature map is fed into different heads to produce corresponding predictions. 

**Task Scheduling** Task scheduling is the process of choos- 

ing which task or tasks to train on at each training step. Some scheduling methods arrange the task orders during the training process in a fixed order like Round-Robin [60], while others may sample tasks following specific distributions [29], like Uniform sampler and Weighted sampler. Specifically, Uniform sampler samples tasks from a uniform distribution and Weighted sampler samples tasks with weight proportional to the number of training epochs of each task. We test the above three task scheduling methods in our investigation and compare their performances. **Task Balancing** Task balancing is designed to deal with the gradients between tasks for the shared parameters in the network. When dealing with multiple tasks, the shared parameters are likely to be dominated by the one with a large gradient magnitude or confused by conflict gradients. It is intuitive to apply weights over these gradients to balance among tasks, and several methods have been proposed, including 1) Fixed weighting, which fixes all loss weights during training; 2) Uncertainty weighting [23], introducing the task-dependent Homoscedastic uncertainty as the basis for weighting losses by maximizing the Gaussian likelihood of the uncertainty; 3) GradNorm [5], calculating the product of _L_ 2 norm of task gradient and the relative inverse learning rate as the indicator of the task learning pace, and then setting task weights to minimize the learning pace difference among tasks to balance the training process; 4) MGDA [43], treating the Multi-Task Learning problem as a multiobjective optimization problem by using multiple gradient descent algorithm [10]; 5) ParetoMTL [31], which finds a solution called Pareto optimal solution where all task losses can decrease without increasing the loss on other tasks. 

**Learning on Partial Labels** Image segmentation task requires annotations of labels to every pixel of the image, which costs a great time, and as a result hard to get enough annotations. To process the missing annotation problem, two different methods are introduced, including Zeroing loss [25, 54] and Pseudo labeling [16]. Zeroing loss [25, 54] simply zero losses for a particular task if the input image does not have the corresponding annotation. Pseudo labeling [16] first trains a teacher model on fully labeled data. 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving_images/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Head B<br>Visual Prompt key & value Task-Prompting<br>Exemplars Generator Block Head C<br>query<br>4 8 16 [32] Head D<br>Image Transformer Unflatten<br>Encoder<br>Encoder<br>4 8 16 [32] [64] Detection Head<br>Flatten<br>Positional Embeddings<br>**----- End of picture text -----**<br>


Figure 3. **The architecture of the proposed VE-Prompt.** VE-Prompt consists of (1) the image encoder to extract image features; (2) a shared transformer encoder for feature enhancement; (3) task-specific prompts generated by the prompt generator with visual exemplars; (4) a task-prompting block to integrate the visual representation with task-specific prompts; and (5) task-specific heads for different tasks. 

Then the teacher model is used to label the missing annotations to create a multitask pseudo-labeled dataset. 

We focus on four major tasks in autonomous driving, i.e., object detection, semantic segmentation, drivable area segmentation, and lane detection. The in-depth analysis of current multi-task methods is shown in Section 5.3. 

## **4. VE-Prompt** 

The key to multi-task learning is to learn high-quality task-specific representations among tasks, which can explore relationships between tasks. Therefore, a good multitask learning framework should take full advantage of taskspecific priors, and guide the model to learn better representations. To this end, we introduce our proposed multi-task framework with VE-Prompt, which consists of five components: (1) an image feature encoder to extract image features; (2) a lightweight shared transformer encoder for feature enhancement; (3) task-specific prompts which encodes task-specific information from visual exemplars; (4) a visual exemplar driven task-prompting block to integrate the visual representation with task-specific prompts; (5) taskspecific heads for predicting results simultaneously. 

The whole pipeline of the proposed multi-task framework is shown in Figure 3. In the following sections, we first delve into the overall multi-task framework in Section 4.1 and elaborate on the visual exemplar-driven prompt in Section 4.2 and the task-prompting module in Section 4.3 respectively. 

## **4.1. Bridging CNN and Transformer** 

The multi-task framework aims to learn more effective representations for all tasks via bridging CNN and Transformer efficiently. The neck of the image encoder and segmentation heads of the framework are CNN-based, reducing the overall training time. The shared transformer encoder is built upon the transformer architecture to capture the long-range dependency [55]. 

**Image Encoder** The image encoder consists of a backbone network and a neck network. We choose the Swin transformer [35] as the backbone to extract features of the 

input image. The output of the backbone is denoted as _{C_ 2 _, C_ 3 _, C_ 4 _, C_ 5 _}_ . Then we adopt Feature Pyramid Network (FPN) [30] module for the neck network to fuse features generated by the backbone. The pyramidal features are of 5 scales, and the detection head only processes the last four-scale features for reducing the computation cost. Here we denote the output of the neck as _{P_ 2 _, P_ 3 _, P_ 4 _, P_ 5 _, P_ 6 _}_ , which have strides of _{_ 4, 8, 16, 32, 64 _}_ pixels. 

**Shared Transformer Encoder** The shared transformer encoder TransEncoder receives multi-scale outputs from the neck and enhances features for following taskspecific heads. We first flatten the feature maps from _{P_ 3 _, P_ 4 _, P_ 5 _, P_ 6 _}_ and concatenate them to obtain a 1D sequence _P_ . Since flattening the features leads to losing the spatial information critical for segmentation, we supplement positional embeddings _pl_ to the flattened features. For the model not considering prompts, we obtain the enhanced feature as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving_images/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving.pdf-0004-12.png)


After feature enhancement, _O_ is passed to the detection head directly, while unflattened to multi-scale features _{z_ 3 _, z_ 4 _, z_ 5 _, z_ 6 _}_ for segmentation heads. 

**Detection Head** The detection head consists of 4 multiscale deformable decoder layers which are elaborated in DINO [61]. Following DINO, we adopt the mixed query selection strategy to initialize anchors as positional queries for the decoder and use the contrastive denoising training approach by taking into account hard negative samples. 

**Segmentation Head** For segmentation-based tasks, we choose Semantic FPN [24] as the segmentation head. In the model without considering prompts, segmentation heads take in multi-layer features from both the neck and shared transformer encoder _{P_ 2 _, z_ 3 _, z_ 4 _, z_ 5 _}_ . The resolution of _P_ 2 is larger and thus provides more image information for the following heads. Then the multi-layer features are upsampled and summed element-wisely. This merged feature map is again upsampled 4 _×_ followed by softmax to produce the classification score for every pixel at the original resolution. 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving_images/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Object Detection Semantic Seg. Drivable Area Lane Detection<br>... ... ... ...<br>... ... ...<br>CLIP Image Encoder<br>... ...<br>**----- End of picture text -----**<br>


Figure 4. **Process of generating visual exemplar-driven prompts.** For the box-wise task, we crop class-related image regions to generate visual exemplars. For pixel-wise tasks, we mask class-related image regions with colored segmentation masks to produce exemplars. Then the fixed CLIP image encoder is adopted to extract task-specific prompts. 

## **4.2. Prompt Generation with Visual Exemplar** 

In order to motivate the model to learn more high-quality task-specific knowledge and handle all tasks better, VEPrompt is introduced to provide more task-specific information with visual clues. The process of generating visual exemplar-driven prompts is shown in Figure 4. The key idea of task-specific prompts is to let the model know how to solve different tasks and what categories to focus on in advance of each task. Therefore, task-specific prompts should contain object-level information which helps the model understand tasks better, and we leverage visual exemplars to generate prompts. 

We first sample a few examples from the training set to generate object-level image regions and segmentation masks as in Figure 4. There are a few generated exemplars for each category in each task, thus the generated prompts cover all classes. Then we adopt CLIP [39] to generate taskspecific prompts since it is a robust feature extractor pretrained with a huge amount of image-text data pairs. For visual perception, the ground-truth annotations can provide hints of the shapes and sizes of different objects, motivating the model to learn high-quality task-specific representations. The task-specific prompt is object-level for object detection and aims to represent relevant image regions. Only the generated prompts are used during training and inference, and no new exemplars will be included, thus it will not lead to training data leakage. 

For the box-wise task (object detection), we use the annotated bounding boxes to crop sampled images and obtain raw object-level image regions for generating prompts. We choose the image encoder with ViT [11] backbone and pass _n_ image regions _{ri[k][}]_[ of] _[ K]_[classes to get the initial prompt] 

as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving_images/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving.pdf-0005-07.png)


where IE and _D_ represent the image encoder of CLIP and the feature dimension respectively. _n_ stands for the numberˆ of visual exemplars for each category. _{p[k] i[}]_[ and] _[ p]_[ indicate] all prompts from image regions and the averaged version respectively. Specifically, class numbers for detection, semantic segmentation, drivable area segmentation, and lane detection are denoted as _K_ 1, _K_ 2, _K_ 3, and _K_ 4. The number of visual exemplars for different tasks is further denoted as _n_ 1, _n_ 2, _n_ 3, and _n_ 4. 

For pixel-wise tasks (i.e., semantic segmentation, drivable area segmentation, and lane detection), image regions of specific classes are marked with colored segmentation masks, and different colors indicate different object categories of different tasks. Similar to object detection, after obtaining _n_ images with colored segmentation masks of _K_ classes, we adopt CLIP to extract features by Equation 2. In this way, we get task-specific prompts _p_ det, _p_ sem, _p_ driv, _p_ lane for object detection, semantic segmentation, drivable area segmentation, and lane detection, respectively. 

## **4.3. Visual Exemplar Driven Task Prompting** 

Task prompting aims to integrate the image features with task-specific prompts to obtain high-quality task-specific representations. It receives additional task-specific prompts as inputs and generates task-specific features for following task-specific heads. 

Here we design two prompting methods to improve task-specific representations. The first strategy is pre-head prompting. The last feature map _P_ 6 from the neck and taskspecific prompt _p_ are fused via a transformer decoder: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving_images/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving.pdf-0005-13.png)


where _q_ , _k_ and _v_ stand for query, key and value. In this way, we get task-specific features _fpre[det]_[,] _[f] pre[ sem]_[,] _[f] pre[ driv]_[,][and] _fpre[lane]_ for object detection, semantic segmentation, drivable area segmentation, and lane detection, respectively. For object detection, we flatten _{P_ 3 _, P_ 4 _, P_ 5 _, fpre[det][}]_[to][a][1D][se-] quence and combine it with positional embeddings. Following Equation 1, we obtain features for the detection head _Odet_ . For segmentation-based tasks, we first flatten _{P_ 3 _, P_ 4 _, P_ 5 _}_ to a 1D sequence and get the enhanced features through Equation 1. Then unflatten the output features as _{z_ 3 _[′][, z]_ 4 _[′][, z]_ 5 _[′][}]_[,][and][pass] _[{][P]_[2] _[, z]_ 3 _[′][, z]_ 4 _[′][, z]_ 5 _[′][, f][pre]_[]][to][spe-] cific segmentation-based heads. Note that _fpre_ is marked as _fpre[sem]_[,] _[ f] pre[ driv]_[, or] _[ f] pre[ lane]_[according to the task type.] 

5 

Another choice is to refine predicted results with taskspecific prompts, namely post-head prompting. In this variant, we obtain class-related features (R _[K][×][D]_ ) similar as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving_images/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving.pdf-0006-01.png)


Here class-related outputs from task-specific heads are denoted as _v_ . Then the final output is calculated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving_images/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving.pdf-0006-03.png)


Empirical results of these two strategies are presented in Section 5.5 and show that pre-head prompting performs better than post-head prompting. 

## **4.4. Optimization** 

Since there are four different perception tasks in our network, our multi-task loss contains four parts. For object detection, we adopt the same objective as in DINO [61]. For segmentation-based tasks, i.e., semantic segmentation, drivable area segmentation, and lane detection, we employ the same loss as in Semantic FPN [24]. Therefore, the total loss for the multi-task model is formulated as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving_images/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving.pdf-0006-07.png)


where _L_ det, _L_ sem, _L_ driv, _L_ lane represent objectives for object detection, semantic segmentation, drivable area segmentation, and lane detection, respectively. _λdet_ , _λsem_ , _λdriv_ and _λlane_ stand for different loss weights. 

## **5. Experiments** 

## **5.1. Dataset Settings** 

Our experiments are tested on the BDD100K dataset. BDD100K dataset has _∼_ 74k training images and covers both object detection (OD), semantic segmentation (SS), drivable area segmentation (DA), and lane detection (LD). We follow [27] to consider three dataset split settings complying with real-world scenarios, i.e., Disjoint-normal setting, Disjoint-balance setting, and Full setting: 

**Disjoint-normal Setting** The number of labeled images for each task is as follows: object detection (10k), semantic segmentation (7k), drivable area segmentation (20k), and lane detection (20k). 

**Disjoint-balance Setting** There are 21k images in this set and each task has 7k labeled images that are not overlapped with other tasks. 

**Full Setting** Full-setting refers to experimenting on all available annotations on _∼_ 74k images in BDD100K and can be used to analyze the upper bound of different methods. 

## **5.2. Evaluation and Implementation Details** 

**Evaluation Metric** In addition to reporting performance on every individual task, we follow [48] to evaluate the whole multi-task performance: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving_images/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving.pdf-0006-17.png)


where _Mm,i_ is the performance of multi-task model on task _i_ , and _Mb,i_ indicates the result of single-task baseline. Since we choose Sparse R-CNN as the detection head for re-implementing current multi-task methods, we regard it as the baseline for object detection. For object detection, we adopt mAP as the main metric. While for segmentation tasks, we use mIoU to evaluate the model. We also compute the average performance (Avg.) of all tasks to compare experimental results more intuitively. 

**Implementation Details** The default training setting is that epoch and batch size are fixed as 36 and 16, the learning rate is set to 1 _×_ 10 _[−]_[5] , and weight decay is 1 _×_ 10 _[−]_[4] . We adopt the AdamW optimizer, for which the warmup length is 1 epoch and the warmup factor is 0.001. We choose SwinTiny [35] as the backbone by default. More details are provided in Appendix. 

## **5.3. Comparison of Multi-task Methods** 

We study the performances of popular existing multitask methods under three settings on BDD100K. We adopt Sparse R-CNN to construct the detection head for efficiency. 

**Partial-label Learning** As shown in Table 1, pseudo labeling [16] can improve performances, especially in object detection and semantic segmentation compared with zeroing loss [53]. Pseudo-labeling achieves satisfactory performance in all settings. 

**Task Scheduling** As shown in Table 1, three task sampling methods (i.e., Uniform sampler [28], Weighted sampler [28] and Round-robin [28]) perform better than Zeroing loss [53] by a large margin on segmentation-based tasks, but get worse in object detection. We hypothesize that training one task per step may lead to forgetting to some extent. **Task Balancing** We choose pseudo labeling as the baseline since task-balancing methods are more suitable in settings with complete labels. Fixed denotes fixed loss weights for all tasks during training. As shown in Table 2, Uncertainty performs better than Fixed on semantic segmentation and drivable area segmentation under the full and disjointnormal settings, while performances of other approaches (i.e., GradNorm and MGDA) degrade significantly. Especially, GradNorm uses the last shared layer of weights to compute gradient norm in its paper, thus we adopt the last layer of _P_ 5 in the neck. When we use the full image encoder to compute the gradient norm, GradNorm achieves 

6 

Table 1. Comparisons of popular task scheduling strategies and partial-label learning methods. 

|Setting|Methods|mAP<br>AP50<br>AP75|mIoU (SS)|mIoU (DA)|IoU (LD)|Avg.|∆_MT L_(%)|
|---|---|---|---|---|---|---|---|
|Full|Zeroing loss [53]<br>Pseudo labeling [16]|36.2<br>61.6<br>35.9<br>36.3<br>61.6<br>36.1|58.6<br>60.9|89.3<br>89.3|23.8<br>23.8|52.0<br>52.6|-2.68<br>-1.65|
||VE-Prompt (Ours)|**39.2**<br>**64.9**<br>**39.0**|**63.2**|**89.4**|**24.0**|**54.0**|**+1.52**|
|Disjoint-normal|Zeroing loss [53]<br>Uniform sampler [28]<br>Weighted sampler [28]<br>Round-robin [28]<br>Pseudo labeling [16]|31.1<br>54.3<br>30.2<br>30.1<br>52.8<br>29.0<br>29.3<br>51.9<br>28.7<br>30.2<br>53.1<br>29.7<br>32.6<br>54.6<br>32.3|55.7<br>60.6<br>58.5<br>61.0<br>59.7|88.0<br>88.6<br>**88.9**<br>88.7<br>88.2|22.2<br>23.4<br>**23.8**<br>23.5<br>23.0|49.3<br>50.7<br>50.1<br>50.9<br>50.9|-2.64<br>-0.10<br>-1.19<br>+2.87<br>+1.19|
||VE-Prompt (Ours)|**34.2**<br>**56.9**<br>**33.9**|**62.2**|88.3|23.3|**52.0**|**+3.95**|
|Disjoint-balance|Zeroing loss [53]<br>Uniform sampler [28]<br>Round-robin [28]<br>Pseudo labeling [16]|29.7<br>52.3<br>29.2<br>28.1<br>50.2<br>27.5<br>28.4<br>50.8<br>27.8<br>31.3<br>52.8<br>30.8|57.5<br>60.4<br>60.0<br>60.2|86.7<br>87.1<br>87.1<br>87.0|21.4<br>**22.6**<br>**22.6**<br>22.2|48.8<br>50.0<br>49.5<br>50.2|-1.61<br>-0.44<br>-0.34<br>+1.87|
||VE-Prompt (Ours)|**33.9**<br>**56.6**<br>**33.7**|**61.2**|**87.4**|22.2|**51.2**|**+4.72**|



Table 2. Comparisons of task balancing strategies with pseudo labels. _[∗]_ means using the full image encoder to compute the gradient norm. 

|Setting|Method|mAP<br>AP50<br>AP75|mIoU (SS)|mIoU (DA)|IoU (LD)|Avg.|∆_MT L_(%)|
|---|---|---|---|---|---|---|---|
|Full|Fixed [16]<br>Uncertainty [23]<br>GradNorm [5]|36.3<br>61.6<br>36.1<br>36.2<br>61.6<br>35.5<br>23.4<br>40.9<br>22.8|60.9<br>61.2<br>25.8|89.3<br>**89.5**<br>51.3|23.8<br>**24.6**<br>13.0|52.6<br>52.9<br>28.4|-1.65<br>-0.76<br>-46.24|
||VE-Prompt (Ours)|**39.2**<br>**64.9**<br>**39.0**|**63.2**|89.4|24.0|**54.0**|**+1.52**|
|Disjoint-normal|Fixed [16]<br>Uncertainty [23]<br>GradNorm [5]<br>MGDA [43]|32.6<br>54.6<br>32.3<br>32.2<br>54.1<br>31.5<br>25.9<br>43.2<br>26.1<br>25.9<br>44.6<br>26.0|59.7<br>59.8<br>39.2<br>50.1|88.2<br>**88.6**<br>39.6<br>85.4|23.0<br>23.8<br>3.7<br>**25.2**|50.9<br>51.1<br>27.1<br>46.7|+1.19<br>+1.79<br>-46.18<br>-7.26|
||VE-Prompt (Ours)|**34.2**<br>**56.9**<br>**33.9**|**62.2**|88.3|23.3|**52.0**|**+3.95**|
|Disjoint-balance|Fixed [16]<br>Uncertainty [23]<br>GradNorm [5]<br>GradNorm_∗_[5]<br>MGDA [43]|31.3<br>52.8<br>30.8<br>31.2<br>53.1<br>30.9<br>28.9<br>49.0<br>28.7<br>30.7<br>51.8<br>30.4<br>21.0<br>38.0<br>20.3|60.2<br>59.9<br>46.8<br>56.6<br>45.5|87.0<br>87.0<br>57.4<br>86.9<br>82.7|22.2<br>22.2<br>19.6<br>21.7<br>**24.3**|50.2<br>50.1<br>38.2<br>49.0<br>43.4|+1.87<br>+1.66<br>-17.26<br>-0.73<br>-12.48|
||VE-Prompt (Ours)|**33.9**<br>**56.6**<br>**33.7**|**61.2**|**87.4**|22.2|**51.2**|**+4.72**|



much better results but still lags behind the baseline. Interestingly, MGDA achieves the best result on lane detection, indicating that it suffers from heavy negative transfer. 

For efficiency and effectiveness, we choose pseudo labeling with fixed loss weights as our baseline, which achieves competitive performance compared with other complicated multi-task methods, to verify the effectiveness of VE-Prompt. 

## **5.4. Compare VE-Prompt with Previous Methods** 

As shown in Table 3, our VE-Prompt surpasses the baseline consistently on almost all metrics in all three settings and achieves significant overall multi-task performance. We conclude that VE-Prompt can learn high-quality task- 

specific knowledge during training, and further improve performance. VE-Prompt also achieves the best results on three tasks compared with single-task models. We also compare VE-Prompt with LV-Adapter [27] in Appendix. 

## **5.5. Ablation Study** 

We conduct all ablation studies under the disjointbalance setting for efficiency. 

**Module Components** We present detailed comparisons on each module to validate our VE-Prompt as in Table 4. The introduced shared transformer encoder alleviates this imbalance to some extent (row 1 vs. row 2). Equipped with taskspecific prompts through task prompting, the model gets better results on all tasks (row 2 vs. row 3), confirming 

7 

Table 3. Comparison between single-task and multi-task learning baselines under different settings. 

|Setting|Methods|mAP<br>AP50<br>AP75|mIoU (SS)|mIoU (DA)|IoU (LD)|Avg.|∆_MT L_(%)|
|---|---|---|---|---|---|---|---|
|Full|Sparse R-CNN [45]<br>DINO [61]<br>Semantic FPN [24]<br>Semantic FPN [24]<br>Semantic FPN [24]|36.5<br>61.5<br>36.1<br>38.6<br>64.2<br>38.2<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|-<br>-<br>59.8<br>-<br>-|-<br>-<br>-<br>89.1<br>-|-<br>-<br>-<br>-<br>25.9|-<br>-<br>-<br>-<br>-|-<br>-<br>-<br>-<br>-|
||Sparse R-CNN based<br>DINO based<br>VE-Prompt (Ours)|36.3<br>61.6<br>36.1<br>**39.4**<br>64.5<br>**39.8**<br>39.2<br>**64.9**<br>39.0|60.9<br>61.5<br>**63.2**|89.3<br>84.9<br>**89.4**|23.8<br>22.0<br>**24.0**|52.6<br>52.0<br>**54.0**|-1.65<br>-2.25<br>**+1.52**|
|Disjoint-normal|Sparse R-CNN [45]<br>DINO [61]<br>Semantic FPN [24]<br>Semantic FPN [24]<br>Semantic FPN [24]|28.8<br>50.4<br>28.0<br>31.2<br>53.0<br>30.5<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|-<br>-<br>59.8<br>-<br>-|-<br>-<br>-<br>87.8<br>-|-<br>-<br>-<br>-<br>25.2|-<br>-<br>-<br>-<br>-|-<br>-<br>-<br>-<br>-|
||Sparse R-CNN based<br>DINO based<br>VE-Prompt (Ours)|32.6<br>54.6<br>32.3<br>33.1<br>55.9<br>32.2<br>**34.2**<br>**56.9**<br>**33.9**|59.7<br>59.2<br>**62.2**|88.2<br>87.2<br>**88.3**|23.0<br>22.7<br>**23.3**|50.9<br>50.6<br>**52.0**|+1.19<br>+0.83<br>**+3.95**|
|Disjoint-balance|Sparse R-CNN [45]<br>DINO [61]<br>Semantic FPN [24]<br>Semantic FPN [24]<br>Semantic FPN [24]|28.1<br>49.2<br>26.7<br>29.4<br>50.8<br>28.1<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|-<br>-<br>59.8<br>-<br>-|-<br>-<br>-<br>85.5<br>-|-<br>-<br>-<br>-<br>23.7|-<br>-<br>-<br>-<br>-|-<br>-<br>-<br>-<br>-|
||Sparse R-CNN based<br>DINO based<br>VE-Prompt (Ours)|31.3<br>52.8<br>30.8<br>33.5<br>55.6<br>33.1<br>**33.9**<br>**56.6**<br>**33.7**|60.2<br>58.1<br>**61.2**|87.0<br>85.2<br>**87.4**|**22.2**<br>21.4<br>**22.2**|50.2<br>50.0<br>**51.2**|+1.87<br>+1.58<br>**+4.72**|



Table 4. Ablation study of modules in our proposed VE-Prompt. TE means transformer encoder. 

||mAP|mIoU (SS)|mIoU (DA)|IoU (LD)|
|---|---|---|---|---|
|DINO based<br>w/ shared TE<br>+Prompt|33.5<br>32.2<br>**33.9**|58.1<br>60.5<br>**61.2**|85.2<br>86.5<br>**87.4**|21.4<br>21.4<br>**22.2**|



Table 5. Ablation study of task-specific prompts. Post and Pre indicate post-head prompting and pre-head prompting respectively. 

|#|Prompt<br>Post<br>Pre|mAP|mIoU (SS)|mIoU (DA)|
|---|---|---|---|---|
|1|<br><br>|32.2|60.5|86.5|
|2<br>3|<br><br><br><br><br>|33.2<br>**33.9**|58.9<br>**61.2**|86.4<br>**87.4**|



Table 6. Ablation study of initialization for prompt vectors. 

|CLIP Initialization|mAP|mIoU (SS)|mIoU (DA)|IoU (LD)|
|---|---|---|---|---|
|<br>|33.5<br>**33.9**|61.0<br>**61.2**|87.2<br>**87.4**|21.9<br>**22.2**|



that task-specific prompts can motivate the model to learn useful task-specific knowledge for specific tasks. **Task Prompting** We conduct an ablation study to compare the proposed two prompting strategies as in Table 5. We can see that both post-head and pre-head prompting improve the performance of object detection. However, post-head 

prompting gets inferior results on segmentation-based tasks (#1 vs. #2), indicating that the post-head process is not suitable for dense prediction tasks. On the contrary, pre-head prompting helps the model make full use of task-specific knowledge and improves all tasks consistently (#3). 

**Prompt Initialization** The task-specific prompts are initialized with the pre-trained image encoder of CLIP. We compare it with random initialization as in Table 6. Results show that prompts with CLIP initialization improve the multi-task model on all metrics. More comparisons and analyses on prompting are presented in Appendix. 

## **6. Conclusion and Discussion** 

In this paper, we first provide an in-depth analysis of popular multi-task learning methods under the realistic scenarios of self-driving, which covers four common perception tasks, i.e., object detection, semantic segmentation, drivable area segmentation, and lane detection. We find that existing methods cannot solve all tasks satisfactorily due to the negative transfer. To mitigate the negative transfer, we propose visual exemplar driven task-prompting (VEPrompt), which incorporates visual exemplars of different tasks to provide high-quality task-specific knowledge. Besides, the proposed framework bridges transformer and convolutional layers for efficient and accurate unified percep- 

8 

tion in autonomous driving. Experimental results show that VE-Prompt can achieve superior performance on largescale driving dataset BDD100K. 

**Limitations** Although our method has achieved substantial improvement in the overall multi-task metric, we find the model gets worse results on lane detection compared with the single-task model. We conjecture that it is because the lane detection task is quite different from other tasks, thus it is difficult for the multi-task model to solve all four tasks satisfactorily. Our VE-Prompt gets better results on lane detection compared with multi-task baselines, but there is still room for improvement. We believe VE-Prompt can be further improved by introducing more robust pseudo-labeling methods or designing specific heads for lane detection. The proposed prompt generation with visual exemplar is general and can be applied to other applications in computer vision. Our method does not directly involve societal issues. 

## **Acknowledgements** 

We gratefully acknowledge the support of MindSpore[1] , CANN (Computer Architecture for Neural Networks) and Ascend AI Processor used for this research. 

## **References** 

- [1] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In _Advances in Neural Information Processing Systems (NeurIPS)_ , volume 33, pages 1877– 1901, 2020. 3 

- [2] David Br¨uggemann, Menelaos Kanakis, Anton Obukhov, Stamatios Georgoulis, and Luc Van Gool. Exploring relational context for multi-task dense prediction. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 15869–15878, 2021. 1, 2 

- [3] Rui Chen, Haizhou Ai, Chong Shang, Long Chen, and Zijie Zhuang. Learning lightweight pedestrian detector with hierarchical knowledge distillation. In _2019 IEEE International Conference on Image Processing (ICIP)_ , pages 1645–1649. IEEE, 2019. 2 

- [4] Yaran Chen, Dongbin Zhao, Le Lv, and Qichao Zhang. Multi-task learning for dangerous object detection in autonomous driving. _Information Sciences_ , 432:559–571, 2018. 2 

- [5] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for 

- 1https://www.mindspore.cn/ 

adaptive loss balancing in deep multitask networks. In _International Conference on Machine Learning_ , pages 794–803. PMLR, 2018. 3, 7, 12 

- [6] Zhiyang Chen, Yousong Zhu, Zhaowen Li, Fan Yang, Wei Li, Haixin Wang, Chaoyang Zhao, Liwei Wu, Rui Zhao, Jinqiao Wang, et al. Obj2seq: Formatting objects as sequences with class prompt for visual tasks. _arXiv preprint arXiv:2209.13948_ , 2022. 2, 3 

- [7] Bowen Cheng, Alex Schwing, and Alexander Kirillov. Perpixel classification is not all you need for semantic segmentation. _Advances in Neural Information Processing Systems_ , 34:17864–17875, 2021. 13 

- [8] Kevin Clark, Minh-Thang Luong, Urvashi Khandelwal, Christopher D Manning, and Quoc V Le. Bam! bornagain multi-task networks for natural language understanding. _arXiv preprint arXiv:1907.04829_ , 2019. 2 

- [9] Michael Crawshaw. Multi-task learning with deep neural networks: A survey. _arXiv preprint arXiv:2009.09796_ , 2020. 2 

- [10] Jean-Antoine D´esid´eri. Multiple-gradient descent algorithm (mgda) for multiobjective optimization. _Comptes Rendus Mathematique_ , 350(5-6):313–318, 2012. 3 

- [11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv preprint arXiv:2010.11929_ , 2020. 2, 5 

- [12] David Eigen and Rob Fergus. Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture. In _Proceedings of the IEEE International Conference on Computer Vision (ICCV)_ , pages 2650– 2658, 2015. 2 

- [13] Christopher Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, and Chelsea Finn. Efficiently identifying task groupings for multi-task learning. _Advances in Neural Information Processing Systems (NeurIPS)_ , 34, 2021. 1, 2 

- [14] Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Qiao. Clip-adapter: Better vision-language models with feature adapters. _arXiv preprint arXiv:2110.04544_ , 2021. 3 

- [15] Yuan Gao, Jiayi Ma, Mingbo Zhao, Wei Liu, and Alan L Yuille. Nddr-cnn: Layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 3205–3214, 2019. 3 

- [16] Golnaz Ghiasi, Barret Zoph, Ekin D. Cubuk, Quoc V. Le, and Tsung-Yi Lin. Multi-task self-training for learning general representations. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , pages 8856–8865, 2021. 2, 3, 6, 7, 12 

- [17] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Girshick. Mask r-cnn. In _Proceedings of the IEEE International Conference on Computer Vision (ICCV)_ , pages 2961–2969, 2017. 2 

- [18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceed-_ 

9 

   - _ings of the IEEE conference on computer vision and pattern recognition_ , pages 770–778, 2016. 2, 13 

- [19] Yun He, Steven Zheng, Yi Tay, Jai Gupta, Yu Du, Vamsi Aribandi, Zhe Zhao, YaGuang Li, Zhao Chen, Donald Metzler, et al. Hyperprompt: Prompt-based task-conditioning of transformers. In _International Conference on Machine Learning_ , pages 8678–8690. PMLR, 2022. 2 

- [20] Yuenan Hou, Zheng Ma, Chunxiao Liu, and Chen Change Loy. Learning lightweight lane detection cnns by self attention distillation. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 1013–1021, 2019. 12 

- [21] Keishi Ishihara, Anssi Kanervisto, Jun Miura, and Ville Hautamaki. Multi-task learning with attention for end-to-end autonomous driving. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 2902–2911, 2021. 2 

- [22] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. _arXiv preprint arXiv:2203.12119_ , 2022. 1, 2, 3 

- [23] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 7482–7491, 2018. 1, 2, 3, 7, 12 

- [24] Alexander Kirillov, Ross Girshick, Kaiming He, and Piotr Doll´ar. Panoptic feature pyramid networks. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 6399–6408, 2019. 2, 4, 6, 8, 13 

- [25] Iasonas Kokkinos. Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 6129–6138, 2017. 1, 3 

- [26] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. _arXiv preprint arXiv:2101.00190_ , 2021. 3 

- [27] Xiwen Liang, Yangxin Wu, Jianhua Han, Hang Xu, Chunjing Xu, and Xiaodan Liang. Effective adaptation in multitask co-training for unified autonomous driving. _arXiv preprint arXiv:2209.08953_ , 2022. 6, 7, 13 

- [28] Valerii Likhosherstov, Anurag Arnab, Krzysztof Choromanski, Mario Lucic, Yi Tay, Adrian Weller, and Mostafa Dehghani. Polyvit: Co-training vision transformers on images, videos and audio. _arXiv preprint arXiv:2111.12993_ , 2021. 1, 2, 6, 7, 12 

- [29] Valerii Likhosherstov, Anurag Arnab, Krzysztof Choromanski, Mario Lucic, Yi Tay, Adrian Weller, and Mostafa Dehghani. Polyvit: Co-training vision transformers on images, videos and audio. _arXiv preprint arXiv:2111.12993_ , 2021. 3 

- [30] Tsung-Yi Lin, Piotr Doll´ar, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2117–2125, 2017. 4 

- [31] Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qing-Fu Zhang, and Sam Kwong. Pareto multi-task learning. _Advances in neural information processing systems_ , 32, 2019. 3 

- [32] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. _arXiv preprint arXiv:2107.13586_ , 2021. 2 

- [33] Shikun Liu, Edward Johns, and Andrew J Davison. Endto-end multi-task learning with attention. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 1871–1880, 2019. 3 

- [34] Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang. Gpt understands, too. _arXiv preprint arXiv:2103.10385_ , 2021. 3 

- [35] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 10012–10022, 2021. 4, 6, 13 

- [36] Othmane Marfoq, Giovanni Neglia, Aur´elien Bellet, Laetitia Kameni, and Richard Vidal. Federated multi-task learning under a mixture of distributions. _Advances in Neural Information Processing Systems_ , 34, 2021. 1 

- [37] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 3994–4003, 2016. 3 

- [38] Xing Nie, Bolin Ni, Jianlong Chang, Gaomeng Meng, Chunlei Huo, Zhaoxiang Zhang, Shiming Xiang, Qi Tian, and Chunhong Pan. Pro-tuning: Unified prompt tuning for vision tasks. _arXiv preprint arXiv:2207.14381_ , 2022. 3 

- [39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In _International Conference on Machine Learning (ICML)_ , pages 8748–8763, 2021. 3, 5 

- [40] Yongming Rao, Wenliang Zhao, Guangyi Chen, Yansong Tang, Zheng Zhu, Guan Huang, Jie Zhou, and Jiwen Lu. Denseclip: Language-guided dense prediction with contextaware prompting. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18082–18091, 2022. 2, 3 

- [41] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. _Advances in Neural Information Processing Systems (NeurIPS)_ , 28, 2015. 2 

- [42] Sebastian Ruder, Joachim Bingel, Isabelle Augenstein, and Anders Søgaard. Latent multi-task architecture learning. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 33, pages 4822–4829, 2019. 3 

- [43] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. _Advances in neural information processing systems_ , 31, 2018. 2, 3, 7, 12 

- [44] Trevor Scott Standley, Amir Roshan Zamir, Dawn Chen, Leonidas J. Guibas, Jitendra Malik, and Silvio Savarese. 

10 

Which tasks should be learned together in multi-task learning? In _International Conference on Machine Learning (ICML)_ , pages 9120–9132, 2020. 2 

- [45] Peize Sun, Rufeng Zhang, Yi Jiang, Tao Kong, Chenfeng Xu, Wei Zhan, Masayoshi Tomizuka, Lei Li, Zehuan Yuan, Changhu Wang, and Ping Luo. Sparse r-cnn: End-to-end object detection with learnable proposals. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 14454–14463, 2021. 8 

- [46] Marvin Teichmann, Michael Weber, Marius Zoellner, Roberto Cipolla, and Raquel Urtasun. Multinet: Realtime joint semantic reasoning for autonomous driving. In _IEEE Intelligent Vehicles Symposium (IV)_ , pages 1013–1020, 2018. 2 

- [47] Simon Vandenhende, Stamatios Georgoulis, and Luc Van Gool. Mti-net: Multi-scale task interaction networks for multi-task learning. In _European Conference on Computer Vision_ , pages 527–543. Springer, 2020. 3 

- [48] Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, and Luc Van Gool. Multi-task learning for dense prediction tasks: A survey. _IEEE transactions on pattern analysis and machine intelligence_ , 2021. 6 

- [49] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. _arXiv preprint arXiv:1804.07461_ , 2018. 2 

- [50] Ze Wang, Weiqiang Ren, and Qiang Qiu. Lanenet: Realtime lane detection networks for autonomous driving. _arXiv preprint arXiv:1807.01726_ , 2018. 2 

- [51] Zifeng Wang, Zizhao Zhang, Sayna Ebrahimi, Ruoxi Sun, Han Zhang, Chen-Yu Lee, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, et al. Dualprompt: Complementary prompting for rehearsal-free continual learning. _arXiv preprint arXiv:2204.04799_ , 2022. 2 

- [52] Dong Wu, Manwen Liao, Weitian Zhang, and Xinggang Wang. Yolop: You only look once for panoptic driving perception. _arXiv preprint arXiv:2108.11250_ , 2021. 1, 2, 12 

   - [57] Yangyang Xu, Xiangtai Li, Haobo Yuan, Yibo Yang, Jing Zhang, Yunhai Tong, Lefei Zhang, and Dacheng Tao. Multitask learning with multi-query transformer for dense prediction. _arXiv preprint arXiv:2205.14354_ , 2022. 1, 2 

   - [58] Zhengyuan Yang, Yixuan Zhang, Jerry Yu, Junjie Cai, and Jiebo Luo. End-to-end multi-modal multi-task vehicle control for self-driving cars with visual perceptions. _24th International Conference on Pattern Recognition (ICPR)_ , pages 2289–2294, 2018. 1 

   - [59] Yuan Yao, Ao Zhang, Zhengyan Zhang, Zhiyuan Liu, TatSeng Chua, and Maosong Sun. Cpt: Colorful prompt tuning for pre-trained vision-language models. _arXiv preprint arXiv:2109.11797_ , 2021. 3 

   - [60] Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell. Bdd100k: A diverse driving dataset for heterogeneous multitask learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 2636–2645, 2020. 2, 3, 12 

   - [61] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. _arXiv preprint arXiv:2203.03605_ , 2022. 4, 6, 8 

   - [62] Zhenyu Zhang, Zhen Cui, Chunyan Xu, Zequn Jie, Xiang Li, and Jian Yang. Joint task-recursive learning for semantic segmentation and depth estimation. In _Proceedings of the European Conference on Computer Vision (ECCV)_ , pages 235–251, 2018. 3 

   - [63] Zexuan Zhong, Dan Friedman, and Danqi Chen. Factual probing is [mask]: Learning vs. learning to recall. _arXiv preprint arXiv:2104.05240_ , 2021. 3 

   - [64] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. _arXiv preprint arXiv:2109.01134_ , 2021. 1, 2, 3 

   - [65] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 16816–16825, 2022. 2 

- [53] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In _European Conference on Computer Vision (ECCV)_ , pages 418–434, 2018. 2, 6, 7, 12 

- [54] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In _Proceedings of the European Conference on Computer Vision (ECCV)_ , pages 418–434, 2018. 3 

- [55] Yutong Xie, Jianpeng Zhang, Chunhua Shen, and Yong Xia. Cotr: Efficiently bridging cnn and transformer for 3d medical image segmentation. In _International conference on medical image computing and computer-assisted intervention_ , pages 171–180, 2021. 4 

- [56] Dan Xu, Wanli Ouyang, Xiaogang Wang, and Nicu Sebe. Pad-net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 675–684, 2018. 3 

11 

## **A. Experimental Setup** 

## **A.1. Implementation Details** 

Here we further provide detailed experimental settings in our paper. Class numbers for object detection, semantic segmentation, drivable area segmentation, and lane detection are 9, 19, 2, and 1 respectively. We remove the train class as in [52] for object detection. For lane detection, we follow [20] to preprocess lane line annotations. Loss weights for object detection, semantic segmentation, drivable area segmentation, and lane detection are fixed as 1, 2, 2, and 2 respectively. All experiments are conducted on servers with 8 Nvidia V100 GPUs and Intel Xeon Platinum 8168 CPU (2.70GHz). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving_images/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving.pdf-0012-03.png)


**----- Start of picture text -----**<br>
10 Fixed<br>MGDA<br>8 VE-Prompt<br>6<br>4<br>20000 30000 40000 50000 60000<br>Number of iterations<br>Value of loss<br>**----- End of picture text -----**<br>


## **A.2. More Details on Dataset** 

BDD100k dataset [60] contains multiple tasks. Here we focus on object detection (OD), semantic segmentation (SS), drivable area segmentation (DA), and lane detection (LD). In BDD100K, 70k training images are labeled for object detection, drivable area segmentation, and lane detection, and only 7k training images are labeled for semantic segmentation. 

## **B. More Investigations** 

In this section, we present more analyses of popular multi-task learning methods. 

## **B.1. Task Scheduling** 

Here we analyze task scheduling methods on disjointbalance settings, whose results are shown in Table 1 in the main paper. Note that the full setting contains almost complete annotations except semantic segmentation, thus it is not suitable for task scheduling. Since the data of all tasks in the disjoint-balance setting is balanced and non-overlapped, Uniform sampler [28] and Weighted sampler [28] are equivalent. As shown in Table 1 in the main paper, task sampling methods (i.e., Uniform sampler [28] and Round-robin [28]) perform better than Zeroing loss [53] by a large margin on segmentation-based tasks, but get worse in object detection. We hypothesize that negative transfer still exists among these approaches, and training one task per step may lead to forgetting to some extent. 

## **B.2. Partial-label Learning** 

As shown in Table 1 in the main paper, pseudo labeling [16] surpasses Zeroing loss [53] on almost all tasks, especially on semantic segmentation. However, the improvement in drivable area segmentation and lane detection under the full setting is not obvious, since there are less unlabeled data on these tasks. 

Figure 5. We provide the loss changes of all tasks during training under the disjoint-balance setting. From the curve, we find out that our VE-Prompt can achieve faster and better convergence. 

## **B.3. Task Balancing** 

We choose pseudo labeling as the baseline since taskbalancing methods are more suitable in settings with complete labels. Fixed denotes fixed loss weights for all tasks during training. As shown in Table 2 in the main paper, Uncertainty [23] performs better than Fixed [16] under the full settings overall, while the performance of GradNorm [5] degrades significantly. Interestingly, Fixed performs slightly better than Uncertainty under the disjoint-balance setting, which indicates that Uncertainty is not suitable for all data split settings. GradNorm and MGDA [43] perform poorly overall, showing that these task-balancing methods are not suitable for autonomous driving. Especially, GradNorm uses the last shared layer of weights to compute gradient norm in its paper, thus we adopt the last layer of _P_ 5 in the neck. We also choose the whole shared encoder to implement GradNorm, which is denoted as GradNorm _[∗]_ , improving the original one by a large margin under the disjointbalance setting (+10.8 in Avg.). This indicates that the selection of network weights for computing GradNorm is important. Interestingly, MGDA consistently achieves the best result on lane detection, indicating that it suffers from the heavy negative transfer. Since it takes a long time to train MGDA, we did not implement it under the full setting for the time limit. 

In summary, most existing multi-task learning methods suffer from poor performances under the real-world scenarios of autonomous driving since they are not designed to handle unified perception in self-driving. Therefore, it is extremely important and urgent to develop applicable multitask methods for autonomous driving. 

12 

Table 7. Compare with LV-Adapter under the disjoint-balance setting. 

|Method|mAP<br>AP50<br>AP75|mIoU (SS)|mIoU (DA)|IoU (LD)|GFLOPs|Params|
|---|---|---|---|---|---|---|
||||||||
|LV-Adapter [27]<br>VE-Prompt (Ours)|24.6<br>47.4<br>21.9<br>**26.8**<br>**51.2**<br>**23.8**|**61.8**<br>58.3|80.6<br>**86.8**|-<br>**22.1**|415<br>401|200M<br>60M|



Table 8. Comparison of fixed and trainable task-specific prompts under the disjoint-balance setting. 

|Fixed|mAP<br>AP50<br>AP75|mIoU (SS)|mIoU (DA)|IoU (LD)|
|---|---|---|---|---|
||||||
|<br>|33.3<br>55.3<br>32.5<br>**33.9**<br>**56.6**<br>**33.7**|61.1<br>**61.2**|87.2<br>**87.4**|22.1<br>**22.2**|



## **C. Compare with LV-Adapter** 

We conduct experiments to compare with recent LVAdapter [27], which tackles three tasks, as in Table 7. The class number of object detection is 10 in LV-Adapter, and the backbone is Res50 [18]. Here we use the same setting as in LV-Adapter, and present results under the disjointbalance setting in Table 7. Note that data splits of object detection, semantic segmentation, and drivable area segmentation are the same as in LV-Adapter. Results show that our proposed VE-Prompt performs better than LV-Adapter on object detection and semantic segmentation by a large margin (+2.2 in mAP and +6.2 in mIoU (DA)). Meanwhile, our method gets competitive results in lane detection compared with the Swin-Tiny [35] backbone as in Table 3 in the main paper. LV-Adapter adopts MaskFormer [7], which is a stronger baseline, to generate pseudo labels for semantic segmentation, while VE-Prompt chooses Semantic FPN [24] as the teacher model. Therefore, the improvement of semantic segmentation for LV-Adapter may come from high-quality pseudo labels. The number of parameters in the proposed VE-Prompt is much less than that of LVAdapter as in Table 7. We also report GFLOPs on the same V100 NVIDIA GPU for a fair comparison. Results show 

better when _n_ 1 = _n_ 2 = _n_ 3 = _n_ 5 = 5, thus we set the number of visual exemplars as 5 in our final model. 

## **D.3. Loss Analysis** 

We also analyze the loss changes of VE-Prompt and the baseline under the disjoint-balance setting as in Figure 5. From the loss curves, we conclude that our VE-Prompt achieves consistent faster and better convergence during training. Note that loss weights for all tasks in Fixed and VE-Prompt here are set as 1 for a fair comparison. 

## **D.4. Comparisons with Alternative Options** 

We present the results of VE-Prompt with some alternative options as in Table 9. Results show that VE-Prompt with Uncertainty can improve Uncertainty on all tasks, and VE-Prompt with Fixed performs better than VE-Prompt with Uncertainty. 

Table 9. Results of VE-Prompt with alternative options under the disjoint-balance setting. 

|Model|mAP|mIoU (SS)|mIoU (DA)|IoU (LD)|∆_MT L_(%)|
|---|---|---|---|---|---|
|Uncertainty<br>VE-Prompt (Uncertainty)|31.2<br>**32.9**|59.9<br>**60.6**|87.0<br>**87.9**|22.2<br>**22.5**|+1.66<br>**+4.04**|
|Fixed<br>VE-Prompt (Fixed)|31.3<br>**33.9**|60.2<br>**61.2**|87.0<br>**87.4**|**22.2**<br>**22.2**|+1.87<br>**+4.72**|



## **D. More Ablation Studies** 

## **D.1. Influence of Fixed Prompts** 

The task-specific prompts are not fixed during training in VE-Prompt. We also conduct experiments to verify the effectiveness of trainable task-specific prompts as in Table 8. Results show that the model with trainable task-specific prompts performs better on all four tasks. 

## **E. Experiments on NuImages Dataset** 

We also conduct experiments on nuImages dataset[2] , which covers two tasks, object detection and semantic segmentation. Results are shown in Table 10, indicating that VE-Prompt performs much better than baselines. 

## **D.2. Number of Exemplars** 

Here we compare different configurations of the number of visual exemplars. The number of visual exemplars for different tasks is _n_ 1, _n_ 2, _n_ 3, and _n_ 4. We keep them equal for simplification. As shown in Figure 6, the model performs 

> 2https://www.nuscenes.org/nuimages 

13 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving_images/cvpr_2023_Visual_Exemplar_Driven_Task-Prompting_for_Unified_Perception_in_Autonomous_Driving.pdf-0014-00.png)


**----- Start of picture text -----**<br>
34 61.4 22.4<br>61.2 22.2<br>33.5 61 22<br>60.8 21.8<br>33<br>60.6 21.6<br>32.5 Object Detection 60.4 Semantic Segmentation 21.4 Lane Detection<br>1 3 5 7 9 11 1 3 5 7 9 11 1 3 5 7 9 11<br>**----- End of picture text -----**<br>


Figure 6. Ablation study of different numbers of visual exemplars under the disjoint-balance setting. The x-axis represents the number of visual exemplars, and the y-axis indicates mAP or mIoU. 

Table 10. Comparisons with multi-task baselines on nuImages. 

|Model|mAP<br>AP50<br>AP75|mIoU|Avg.|
|---|---|---|---|
|Sparse R-CNN based<br>DINO based<br>VE-Prompt (Ours)|50.4<br>76.8<br>54.5<br>55.5<br>81.6<br>60.6<br>**55.8**<br>**81.9**<br>**60.7**|53.8<br>56.7<br>**59.1**|52.1<br>56.1<br>**57.5**|



14 

