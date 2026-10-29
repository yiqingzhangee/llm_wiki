# **Compositional 4D Dynamic Scenes Understanding with Physics Priors for Video Question Answering** 

**Xingrui Wang**[1] **Wufei Ma**[1] **Angtian Wang**[1] **Shuo Chen**[2] **Adam Kortylewski**[3] _[,]_[ 4] **Alan Yuille**[1] 

1 Johns Hopkins University 2 Tsinghua University 

3 Max Planck Institute for Informatics 4 University of Freiburg 

```
{xwang378,wma27,angtianwang,ayuille1}@jhu.edu
chenshuo20@mails.tsinghua.edu.cnakortyle@mpi-inf.mpg.de
```

## **Abstract** 

For vision-language models (VLMs), understanding the dynamic properties of objects and their interactions within 3D scenes from video is crucial for effective reasoning. Cognitive science research has suggested that humans are adept at understanding these properties by constructing 3D and temporal (4D) representations of the world. In this work, we introduce a video question answering dataset **SuperCLEVR-Physics** that focuses on the dynamics properties of objects. We concentrate on physical concepts — _velocity_ , _acceleration_ , and _collisions_ within 4D scenes, where the model needs to fully understand these dynamics properties and answer the questions built on top of them. From the evaluation of a variety of current VLMs, we find that these models struggle with understanding these dynamic properties due to the lack of explicit knowledge about the spatial structure in 3D and world dynamics in time variants. 

To demonstrate the importance of an explicit 4D dynamics representation of the scenes in understanding world dynamics, we further propose **NS-4Dynamics** , a **N** eural- **S** ymbolic model **for** reasoning on **4D Dynamics** properties under explicit scene representation from videos. Using scene rendering likelihood combining physical prior distribution, the 4D scene parser can estimate the dynamics properties of objects over time to and interpret the observation into 4D scene representation as world states. By further incorporating neural-symbolic reasoning, our approach enables advanced applications in future prediction, factual reasoning, and counterfactual reasoning. Our experiments show that our NS-4Dynamics suppresses previous VLMs in understanding the dynamics properties and answering questions about factual queries, future prediction, and counterfactual reasoning. Moreover, based on the explicit 4D scene representation, our model is effective in reconstructing the 4D scenes and re-simulate the future or counterfactual events. Our dataset and model will be released at `https://github.com/XingruiWang/SuperCLEVR-Physics` 

## **1 Introduction** 

Visual question answering (VQA) is commonly regarded as a comprehensive method to assess how well machine learning models can identify objects, understand their relationships, and perform reasoning over the scenes. When it comes to dynamic scenes in video question answering (VideoQA), we argue that a model should incorporate a dynamic representation that captures how objects change and interact over time. This is partly motivated by studies in cognitive science [11, 27] which found that humans excel at understanding the dynamics and interactions in the physical world. This allows humans to perform complex tasks such as planning and manipulation in the 3D world. 

Preprint. Under review. 

However, current video understanding models in the computer vision community focus on VQA tasks in large benchmark datasets of real images. These datasets lack annotations of 3D/4D properties (which are notoriously hard to annotate), which makes it impractical to train these models to estimate 3D/4D annotations. Although these LVMs may have the capability of learning knowledge of 3D/4D properties, perhaps encoded implicitly, it is impossible to test this on these datasets. This applies also to the most recent algorithms which exploit the enormous strengths of large language models. 

Cognitive scientists have developed an impressive variety of synthetic datasets for physical reasoning which enable the development and testing of models to understand dynamics, physical properties, and interactions. This has led in many important conceptual findings [39, 6, 4, 26, 1, 22, 9, 40]. Due to the nature of these datasets and in particular the simplicity of the objects and environments, it is nontrivial to extend their models to the complexity of real world. As listed in Table 1, their environments are limited either to 2D flat planes or overly simplified objects, leading to a big gap between controlled synthetic datasets and real image datasets that computer vision community favors. 

To address the understanding of the 4D dynamic in Video Question Answering (VideoQA) tasks, we introduce _SuperCLEVRPhysics_ . This dataset focuses on the dynamics and interactions of objects in 3D space — velocities, accelerations, and collisions, and design questions around these properties in the factual, future prediction, and counterfactual scenario. As opposed to the simple objects in previous datasets [39, 16], we improve the realism of the appearances of objects and backgrounds with new materials and textures.. For video generation, we first design the dynamics properties in the initial state and use PyBullet [7] to simulate the motion and interaction 

Table 1: Comparisons with previous dynamical reasoning datasets. From left to right: **3D** space, **phy** sical dynamics, **lang** uage understanding, **real** istic objects. 

||3D<br>Phy.<br>Lang.<br>Real|
|---|---|
|CRAFT [1]<br>✗<br>✗<br>✓<br>✗||
|||
|CLEVRER [39]|✗<br>✗<br>✓<br>✗|
|||
|Comphy [6]|✗<br>✓<br>✓<br>✗|
|||
|Physion [4]|✓<br>✗<br>✗<br>✗|
|||
|Physion++ [26]|✓<br>✓<br>✗<br>✗|
|||
|**Ours**|✓<br>✓<br>✓<br>✓|



in the following frames. Then we use Blender [5] to render the videos. Questions are formulated around dynamic properties and interactions from three perspectives: factual, predictive, and counterfactual. Factual questions are answered using the full video frames, predictive questions are answered based only on the initial portion of the video, and counterfactual questions rely on modified properties that differ from the original video. Fig.1 presents examples of video frames and their corresponding factual, predictive, and counterfactual questions. As a compositional reasoning dataset, we also provide new operation programs to answer these questions step by step using a 4D dynamic scene representation. We test the existing VideoQA models and their variants on SuperCLEVR-Physics and find they are struggling to understand these dynamic features, due to the lack of explicit knowledge of the 3D world or 3D dynamics of objects. 

This motivates us to develop _NS-4Dynamics_ , a neural-symbolic model that reasons about dynamics by first estimating explicit 4D scene representation. Our NS-4Dynamics consists of two key modules, a dynamic 3D compositional model following by a symbolic reasoning module. The dynamic 3D compositional model learns a 3D generative model [20, 30] of the scene and captures the dynamics of objects with a physical prior to output a 4D parse of the video.. Then based on the compositional 4D scene representation, the symbolic reasoning module executes the reasoning program and obtains answers step-by-step with high interpretability. 

We demonstrate that NS-4Dynamics outperforms previous state-of-the-art models on video question answering regarding the dynamic properties of objects, including velocity, acceleration, and collision events on our SuperCLEVR-Physics. Furthermore, with an explicit 4D scene representation, our model not only infers objects’ dynamic properties and reconstruct the scenes, but also predicts future states and reasons about counterfactual conditions. 

**Our contributions** :(1) First, we introduce a VideoQA dataset regarding the dynamics properties of objects (velocity, acceleration) and multi-object interactions (collisions). We also improve the visual complexity of objects and backgrounds with new color or textures for materials, and further design questions about these properties in factual, future prediction, and counterfactual scenarios. (2) Second, we propose NS-4Dynamics, a neural-symbolic model that first reconstructs the 4D scene with a dynamic 3D generative model with physical priors as the perception module, and then reasons about 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering_images/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Input Videos Questions Answers<br>Factual Question 1<br>Is the suv moving faster than the  No<br>yellow bus?<br>(a) …<br>Factual Question 2<br>What is the color of the bus  Green<br>which collides with the suv?<br>Predictive Question<br>(b) … ? Will the red bus collide with the  Yes<br>yellow object?<br>Counterfactual Question<br>(c) Would the bus collide with the<br>… minivan if it were standing still from the beginning? No<br>Frame 0 Frame 1 Frame N<br>**----- End of picture text -----**<br>


Figure 1: Examples of video frames and their corresponding factual, predictive, and counterfactual questions. 

dynamics over the explicit 4D scene representation. Third, we demonstrate that NS-4Dynamics can outperform previous state-of-the-art models VLMs on SuperCLEVR-Physics, showing the advantages of our approach to infer objects’ dynamic properties, reconstruct the scenes, imagine future states, and reason about counterfactual conditions. 

## **2 Related Work** 

**Video question answering.** Video questions answering (VideoQA) is a challenging task because models must not only detect and identify objects from static images, but also track and infer objects’ changes and interactions over a sequence of frames. A number of VideoQA datasets annotate questionanswer pairs on natural videos [37, 38, 21]. However these datasets are not suitable for studying dynamic physical properties as natural videos contain limited object interactions and the annotated questions are hindered by shortcuts and biases. Another line of works focused on physical reasoning in simulated environments [39, 6, 8, 26]. However, these datasets are either built in restricted settings [39, 6] or lack of real dynamics and forces common in real world [26] (see Table 1). In order to further challenge and explore the limitations of current video-text models on dynamics reasoning, we develop SuperCLEVR-Physics with improved realism for both video quality and physics simulation. 

**Video-text models.** With the availability web-scale image-text or video-text paired datasets [25, 2], recent video-text models [32, 17, 36] adopted heavy multi-modal pretraining and achieved improved results on a wide range video understanding tasks. However, these models often exploit biases and shortcuts for dynamic reasoning, failing to capture the physical intrinsics that drive object movements and interactions. Our NS-4Dynamics incorporated a 4D dynamic scene representation in our scene parser, allowing compositional reasoning of various physical events in the video sequence. **Physical scene understanding.** Understanding the physical events in a dynamic 4D scene or inferring the future state is hard as it requires a model to apply physical principles to explain the observed events or simulate the future outcomes. Previous studies explored physics engines for simulation and learning [13, 3], or integrating a differentiable physics engine in deep learning models [34, 33, 15]. Another line of works learned physical properties from a compositional scene [11, 27, 8, 6, 40]. However, previous methods were limited to simple scenes where physics models provide good approximations of the world. We extend our scope to more realistic scenes with real-world objects and complex simulations, and enable physical scene reasoning with our dynamic 3D compositional reasoning. 

**3D compositional models.** Our work is also related to previous 3D compositional models for image classification [12], 6D pose estimation [20], and 3D-aware visual questions answering [29]. 3D compositional models learns a probabilistic generative model of feature activations of a scene and enables scene understanding with analysis-by-synthesis. We extended previous 3D compositional models with a dynamic scene representation, allowing us to reconstruct object trajectories and reason about the physical events. 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering_images/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Frame 0 Frame 20 Frame 40<br>(a) Realistic material for objects.<br>Frame 60 Frame 80 Frame 100<br>(b) Environment map in dome<br>shape as background. (c) Sample SuperCLEVR-Physics videos.<br>**----- End of picture text -----**<br>


Figure 2: (a) We keep the same shape from SuperCLEVR but repaint the color of the material for objects to make it more realistic. (b) We project the environment map from HDRI into a dome shape. (c) After assigning dynamics properties, we use physical simulation (PyBullet) and render (Blender) to generate the video data, where objects are moving and interacting in the dynamic 3D space. 

## **3 Dataset** 

To study the understanding of dynamics in visual question answering, we present the SuperCLEVRPhysicsdataset. This dataset is designed to evaluate the understanding of physical object dynamics — including _velocity_ , _acceleration_ , and _collision_ events in Video Question Answering (VideoQA). Our work builds on [16], which introduced complex scenes with multiple object categories to image-based VQA and [31] which expanded the former with 3D aware scene understanding questions. We follow the established setting in terms of objects classes and shapes, and extend this line of work to video data, where we include the dynamic properties of objects to build a 4D scene representation. 

In this section, we first introduce the three key components for designing the video data in our proposed dataset: (1) A more realistic appearance for the objects and the background in Sec.3.1; (2) The introduction of object dynamics over time in Sec.3.2; and (3) the introduction of dynamic events caused by object interactions in Sec.3.3. Building on this setting, we will then discuss the physics simulation and video rendering in Sec.3.4 and the automated generation of questions in Sec.3.5. 

## **3.1 Realistic Appearance** 

**Objects appearances** We use the same object shapes as in SuperCLEVR, including 5 vehicle categories (car, plane, bicycle, motorbike, bus) with 18 sub-types. However, like other synthetic datasets in VQA [39], the original SuperCLEVR colors are simple monochrome. To achieve higher realism while retaining distinct color labels, we replace the 3D model material with a more realistic colormap by following steps. (1) We generate a new colormap for all color labels (gray, red, brown, yellow, green, cyan, blue, and purple), with each color including six randomly generated HEX codes created by an LLM. (2) We then apply these new colors to the main body of the objects. For example, as shown in Figure 2 (a), we choose the largest material area, like a car’s body, and apply a randomly selected HEX code from the colormap. For other parts, we use colors that enhance realism: transparent for windows, black for wheels, and red for lights. 

**Backgrounds** We use highly realistic background images and lights in our dataset. We utilize real HDRI images ( `https://polyhaven.com/hdris` ), containing real captured images with environment maps in 509 different scenes. We map the image as material into a dome shape as background to introduce more variability into data (see Fig. 2 (b)). 

## **3.2 Dynamics Properties** 

A critical feature of our dataset is the dynamic scene properties designed to test the 4D physical scene understanding of vision-language models. We define these properties to accurately reflect real-world physics, thereby creating a realistic and controlled setting for VideoQA challenges. This section details how we establish and manipulate each dynamic feature to create diverse and complex scenarios for models to interpret. Detailed settings are provided in the Appendix. 

**Position and Orientation.** (1) **Position** : Objects are placed randomly in the scene. we use a beta distribution along the x and y axes to make then closer to the edge. The z-axis, which controls height 

4 

relative to the ground, is set differently to classes. For airplanes, they have the option to start in the air, where the z-axis positions range from 2 to 4.5; otherwise, they are positioned on the ground, same as the other classes. (2) **Orientation** : All objects are initially facing the center of the scene and we add a random noise for variability. 

**Velocity.** Each object begins with a predefined velocity state — static (0 m/s), slow (3 m/s), or fast (6 m/s) — setting the initial conditions for movement and subsequent interactions. 

**Internal Forces and Acceleration.** In our dataset, the velocities of objects are further influenced by both internal and external forces, which directly impact their acceleration and subsequent motion, akin to real-world scenarios as follows. (1) **Engine Force for Vehicles:** Vehicles are equipped with engines that provide forward acceleration. We apply a uniform acceleration of 1 m/s[2] to vehicles with an active engine. We also set speed limits for objects with different velocity states — 8 m/s for "fast" objects and 5.5 m/s for others. (2) **Floating Force for Airplanes:** Airplanes may have a floating force to either maintain a static height or descend due to gravity without the floating. 

**External Forces and other physical properties.** The environment also includes external forces such as gravity, friction, and elastic forces during collisions, along with additional physical properties of the objects. For further details, please refer to the Appendix. 

## **3.3 Collision Events** 

In our dataset, we extend prior benchmarks [39, 6] by focusing on more complex shapes and motions in full 3D space, which makes the motions of objects harder to track after the collision. The collision event will be calculated by a physics simulator. For each collision, we will record the objects involved and the frame index, allowing us to develop questions from these annotations. 

## **3.4 Physical Simulation and Rendering** 

The video data is generated using the pipeline from the Kubric [10] package, which uses the physical engine (PyBullet [7]) for simulation and Blender [5] to render videos frame by frame. In our dataset, each video is 2000 ms long, resulting in 120 frames with a corresponding frame rate of 60. In the final dataset, we generate 1000 videos for training, 100 for validation and 100 for testing. 

## **3.5 Question Generation** 

To study 3D dynamic reasoning in visual question-answering tasks, we design the questions for the SuperCLEVR-physics dataset to address three aspects: factual, predictive, and counterfactual reasoning, each focused on dynamics like velocity, acceleration, and collision events. The questions are generated by predefined templates with functional programs. We have 12752 questions totally, where 7850 are factual, 2750 are predictive, 989 are counterfactual. We provide all the question templates and the corresponding programs in the Appendix. 

**Factual Questions** These questions revolve around the velocity, acceleration, and collisions of objects. Examples include "Is the object moving?" and "Is object A moving faster than object B?" We also incorporate questions by defining a specific moment from the events of collisions happens or objects coming into frames and ask much like how physics defines instantaneous velocity such as "when object A collides with object B", or "when object A comes into the frame". Vertical and horizontal accelerations are also explored, with questions like "Is object A floating in the air?" or "Is the wagon accelerating?" 

**Predictive Questions** Predictive questions are designed to infer future interactions, especially collisions, based on current dynamics and positioning. An example question is “Will object A collide with object B next?”. For predictive questions, only first 30 frames of videos are given to the model. **Counterfactual Questions** Counterfactual questions explore changes in object dynamics and how these affect interactions, such as changes in velocity or acceleration. For instance, "If Object A were moving fast, would it collide with Object B?" or "If Object A’s engine stops, would it collide with Object B?" Such questions allow us to assess how models respond to hypothetical modifications, providing deeper insight into their understanding of physical reasoning. 

## **4 Model** 

In this section, we introduce NS-4Dynamics, to tackle the VideoQA task about object dynamics in 3D space. As shown in Fig. 3, our model consists of three main components: a **neural symbolic** 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering_images/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering.pdf-0006-00.png)


**----- Start of picture text -----**<br>
Input Video<br>Image  𝑻+ 𝟏 I. Neural Symbolic Scene Parser 4D Symbolic<br>Scene<br>Representation<br>Scene<br>Image  𝑻 Likelihood State  𝑻<br>CNN<br>Physical<br>Prior<br>Image  𝑻−𝟏 II. Physical Prior<br>State  𝑻−𝟏 Simulation 𝜖~ 𝑁(0, 𝜎) State  𝑻−𝟏<br>III. Program Execution<br>Question: Is the suv faster than  Question<br>the yellow bus? Parser Programs Answer: False<br>…<br>**----- End of picture text -----**<br>


Figure 3: Our NS-4Dynamicshas three main components. I: A Neural symbolic scene parser that parses a 4D dynamic scene representation from the video. II: A physical prior providing a prior distribution for current states. III: A program executor that parses questions into reasoning programs and then executes the program over the predicted scene representation to answer the questions. 

**scene parser** , a **physical prior module** and a **program executor** with **question parser** . The neural symbolic scene parser and the physics prior are combined to first parsing the video sequences into a 4D scene representation. Then, we integrate the language model and reasoning program to answer the questions about the dynamic properties of objects, including velocity, acceleration, and deduce collisions. In this part, we will introduce the definition of the 4D symbolic scene representation in Section 4.1, then the scene parser with physical prior in Section 4.2, and the language model and reasoning program in Section 4.3 respectively. 

## **4.1 4D Symbolic Scene Representation** 

For a given video, we denote the image frame at timestep _t_ as _It_ . The 4D symbolic scene representation _S_ is a sequence of objects states _St_ = _{Ot_[1] _[, O] t_[2] _[, . . . , O] t[N][}]_[ over time, where] _[ N]_[is the number] of objects in the scene. Object _Ot[i]_[is defined by its translation] _[ T][ i] t[∈]_[R][3][, rotation] _[ R] t[i][∈]_[R][3][, velocity] _vt[i][∈]_[R][3][, acceleration] _[ a][i] t[∈]_[R][3][, class] _[ cls][i]_[and color] _[ c][i]_[.] 

## **4.2 Dynamic Scene Parser with Physics Prior** 

The scene parser aims to estimate the world state _St_ from the corresponding observation _It_ and the previous states _S<t_ with a probabilistic model. Similar to the pose estimation on images [28, 20], we first use render-and-compare on the feature level for each frame, ( _i.e_ . _F_ ( _It_ )) and maximizing a _scene likelihood_ to estimate the states, where _F_ ( _·_ ) is a CNN backbone. We further consider the temporal consistency and physical plausibility between frames by adding a _physical prior_ in the likelihood function for correction. This maximizing likelihood for world states estimation can be written as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering_images/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering.pdf-0006-07.png)


**Preliminaries.** Like pose estimation, the scene likelihood compares the features of images ( _It_ ) and the neural mesh model ( _Mcls_ ) from soft rasterization [18], to estimate the 3D location ( _Tt_ ) and rotation ( _Rt_ ) by maximizing likelihood ( _i.e_ . world states _St_ ). The neural mesh model represents 3D CAD model as a vertex set and is trained with constructive learning with images to obtain a representation of each vertex. The details of the definition of the neural mesh model and calculation of scene likelihood are put in the Appendix. During inference, with a given translation _Tt_ and rotation _Rt_ , the neural mesh model of category _cls_ can be projected to have its features _fcls_ ( _R, T_ ). **Inference as scene parsing** As the input images are processed by the CNN and get a feature _Ft_ := _F_ ( _It_ ), following the similar structure [28, 20], we are able to compare the features and 

6 

(2) 

construct a likelihood function as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering_images/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering.pdf-0007-02.png)


where _B_ is a background parameter learned from training. 

**Physical prior** Scene likelihood alone is insufficient to reconstruct a physically plausible 4D scene. Inspired by the idea from bayesian inference area[24], we integrate a physics prior _p_ ( _St|S<t_ ) into the likelihood function as a correction term as in Eq. 4 

However, it’s nontrivial to directly model the physical functions. Many previous studies on physical engines within neural networks only focus on simple shapes like cubes as the complexity in shapes of objects and the interaction process is hard to model. On the other hand, computational physical engine[7] excel at modeling physical functions for any known shapes but are not differentiable and challenging to integrate into the de-rendering process during inference. 

In our method, we modify the discriminative physics engineering into a probabilistic model by introducing an uncertainty term _ϵ ∼N_ (0 _, σ_[2] ). Denoted a physics engine as PE( _·_ ), then the prior _p_ ( _St|S<t_ ) can be computed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering_images/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering.pdf-0007-07.png)


where _µ_ = PE( _R_[ˆ] _t−_ 1 _, T_[ˆ] _t−_ 1), _C_ = 1 _/_ �(2 _π_ ) _[k] |σ_[2] _I|_ . Finally, the rotation _Rt_ and translation _Tt_ can be estimated by maximizing the joint likelihood of the rendering likelihood and the physical prior likelihood: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering_images/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering.pdf-0007-09.png)


**Other attributes** By having the estimated world states ( _Rt, Tt_ ), here we introduce how the other properties and interactions are predicted: (1) **Velocity** and **acceleration** can be calculated from the difference between the translation and further between the computed velocities at two consecutive time steps. To avoid the potential noise in the translation estimation, we use a moving average filter to smooth the velocity and acceleration estimation with the window size 5. (2) **Object class** can be obtained from the category of mesh model achieving the highest likelihood[31]. To predict the **color** attribute of objects, we crop the region containing the object from the RGB image and train an additional CNN classifier. 

**Collisions and resimulation Collisions** are obtained from the physics engine simulator as the time of estimating the physical prior. For each collision, we record the time and objects involved. **Predictive and counterfactual resimulation** : From the explicit 4D scene representation, we can easily simulate the future or counterfactual states of the objects. For either two cases, we apply the physics engine and assign the predicted translation, rotation, velocity or counterfactual condition to all objects as initial conditions and simulate. We visualize the resimulation results in Sec. 5.4 

## **4.3 Program execution** 

After having the 4D scene representation from the scene parser, we can parse the question into reasoning programs and execute the program on the scene representation to predict the answer. The question parsing follows previous work [39, 6], where an LSTM sequence-to-sequence model is trained to parse the question into its corresponding program. 

## **5 Experiments** 

## **5.1 Experiment Setup** 

**Baseline models.** We select 7 representation models from the following 3 categories as baseline models on SuperCLEVR-Physics. (1) Simple classification-based methods: _**CNN+LSTM**_ and _**FiLM**_ . (2) Neural symbolic models: _**NS-DR**_ and _**PO3D-VQA**_ , (3) large pretrained video-text models: _**Video-LLaVA**_ , _**PLLaVA**_ , and _**InternVideo**_ . Please refer to the appendix for detailed implementations. 

7 

**Dynamic scene parser implementation.** The dynamic scene parser and CNN classifier are trained on the images and 120k images are used for training. The dynamic scene parser uses ResNeXT for the feature extractor and is trained for 50 epochs with batch size 4 on 4 GPUs. The attributes classifier uses ResNet50 and is trained with the cropped images by the ground truth bounding box. During inference, we set the variance of the physical prior as 3. 

## **5.2 Video Question Answering Results** 

We first compare the NS-4Dynamics with baseline models for the VideoQA task of SuperCLEVRPhysics, the results are shown in Tab. 2. 

Table 2: Performance on the SuperCLEVR-Physics testing split for each question type, _i.e_ . factual, predictive and counterfactual. The factual questions contain sub-type about **Vel** ocity, **Acc** eleration, **Col** lision and "All" means its overall accuracy.[�] indicates GPT-assisted zero-shot evaluation. 

||**Average**|**Average**|**Factual**<br>**All**<br>Vel.<br>Acc.<br>Col.|**Predictive**|**Counterfactual**|
|---|---|---|---|---|---|
|CNN+LSTM<br>FiLM [23]||48.03<br>50.18|40.63<br>41.71<br>56.79<br>25.37<br>44.07<br>48.58<br>53.09<br>26.87|56.04<br>54.94|47.42<br>51.54|
|NS-DR [39]<br>PO3D-VQA [31]||51.44<br>62.93|51.44<br>55.63<br>46.34<br>46.86<br>61.22<br>62.21<br>73.17<br>51.20|-<br>65.33|-<br>62.24|
|Video-LLaVA �[17]<br>PLLaVA �[36]<br>InternVideo [32]||36.04<br>62.18<br>52.62|33.12<br>31.90<br>56.44<br>17.31<br>51.45<br>51.90<br>60.12<br>43.75<br>51.07<br>59.29<br>49.08<br>36.06|42.34<br>65.69<br>54.74|32.65<br>69.39<br>59.18|
|**w/o physics**<br>**NS-4Dynamics**||75.97<br>**82.64**|79.68<br>81.40<br>81.30<br>74.88<br>**87.70**<br>**88.66**<br>**83.73**<br>**88.46**|78.83<br>**85.71**|69.39<br>**74.51**|



**Comparison with classification-based methods.** The accuracy of **CNN+LSTM** only reaches 40.63%, 56.04% and 47.42% for factual, predictive, and counterfactual questions; and **FiLM** achieves 44.07%, 54.94% and 51.54%. The performance of these models is notably lower than that of NS4Dynamics, which achieves 87.70%, 85.71% and 74.51% for factual, respectively. As classificationbased methods generally rely on extracting features through CNNs, we demonstrates significant advancements in handling dynamic content with explicit scene representation. 

**Comparison with the Neural-Symbolic methods** The comparison between the Neural-Symbolic models shows significant performance improvement as the dimension of scene representation increases. PO3D-VQA constructs explicit 3D scene representation, leading to a better estimation to the objects’ position and poses, which is beneficial to infer the dynamics and interaction compared with the 2D presentation in VR-DR (9.78%) for the factual questions. As pushing the dimension into 4D and considering the physical prior knowledge, our NS-4Dynamics achieves 82.64% overall accuracy which is 19.71% higher than PO3D-VQA. 

**Comparison with the Video-LLM** Although Video-LLMs are found to have strong video understanding abilities and generalization abilities, they do not perform as well on SuperCLEVR-Physics. In zero-shot settings, Video-LLaVA obtains an overall accuracy of 36.04%, while PLLaVA achieves a better but limited performance of 62.18%. Even with finetuning, the accuracy of InternVideo is only 52.62%, falling far behind our NS-4Dynamics. Results show existing video foundation models fail to capture physical principles or perform reasoning on complex dynamic scenes. 

## **5.3 4D Scene Reconstruction Results** 

To evaluate the accuracy of reconstructed scene representation with the observation, we using the metric is 6D pose estimation [35]to compare the accuracy of the translation and rotation estimation, and compare with the image level 6D pose estimation in PO3D-VQA, and the dynamics scene representation without the physics prior (w/o physics). The results are shown in Tab. 3. 

## **5.4 Qualitative results** 

We visualize how our model and baseline w/o physics reconstruct the scenes and re-simulate the future scene in predictive questions. In Fig. 4, the questions ask about the collision, but in (c) the baseline misses the position of the car and makes mistakes. In Fig. 5, from the input videos (a1), our model predicts a better position for the blur bus (b1); By resimulation, the result of our model (b2) is closer to the ground truth future frames (a2), compared with the results of baseline (c2). 

8 

Table 3: Evaluation of scene representation by the metric in 6D pose estimation. 

|||FLOPs|Translations<br>RMSE|Rotations<br>_Acc_ _π_<br>18<br>_Acc π_<br>6<br>Med_err_||
|---|---|---|---|---|---|
||PO3D-VQA<br>w/o physics<br>|0.18<br>**0.63**<br>|0.233<br>0.055<br>|62.21<br>67.21<br>3.498<br>78.73<br>96.18<br>2.730<br><br><br>||
|(a)<br>(b)<br>(c)<br>Input<br>NS-4Dyanmics<br>w/o physics|**NS-4Dynamics**|0.62|**0.040**|**89.71**<br>**98.34**<br>**1.419**||



Figure 4: Qualitative examples of factual questions. (a) shows the input video; (b)We show our NS-4Dynamics can have a better estimation for the motion with the physical prior. (c) The error of position predicted by baseline w/o physics in the red box leads to the mistake in answering the question. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering_images/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering.pdf-0009-03.png)


**----- Start of picture text -----**<br>
Question:<br>Will the red suv collide<br>(a1) (a2) with the school next?<br>Answer: No<br>NS-4Dyanmics:<br>(b2) Prediction:  No<br>(b1)<br>w/o physics:<br>Prediction:  Yes<br>(c1) (c2)<br>**----- End of picture text -----**<br>


Figure 5: Qualitative examples of predictive questions. (a1) The first 30 frames are given to models as input video and (a2) the following are hidden as ground truth future states; (b1) our NS-4Dynamics has a better estimation of the poses of objects, and (b2) a plausible imagination by re-simulation. (c1) The red box shows the error of poses estimation of bus from the baselinew/o physics, which (c2) makes the red suv collide with the school bus in the future. 

## **6 Conclusion and Discussion** 

In this work, we study the 3D dynamics properties in video question answering. We present a new dataset SuperCLEVR-Physics which builds a realistic 3D simulation with various physical events. We find that existing video-text models, even with large-scale pretraining, fail to capture important physical properties that are crucial for temporal or predictive reasoning. Hence we develop NS-4Dynamics, a neural-symbolic model built on explicit 4D scene representation. We capture the physical principles by incorporating physical priors in our neural symbolic scene parser. Experimental results on SuperCLEVR-Physics show that our approach outperforms previous state-ofthe-art methods by a wide margin, demonstrating the advantages of our approach to infer dynamic properties and to predict future states. In ongoing work we are making our dataset more realistic by exploiting big data techniques, such as generating 3D-DST data with diffusion models [19] and developing our neural-symbolic parser to work on real images by extending [14]. 

**Limitations.** Our neural symbolic scene parser integrates physics as a prior when predicting future states. For future work we hope to explore differentiable physics engines that enable scene parsing with multi-frame reasoning. This is a significant extension of previous methods as we are involving real-world objects and complex simulations, posing challenges to the differentiable physics simulator. 

9 

## **References** 

- [1] Tayfun Ates, M. Ate¸so˘glu, Ça˘gatay Yi˘git, Ilker Kesen, Mert Kobas, Erkut Erdem, Aykut Erdem, Tilbe Goksun, and Deniz Yuret. CRAFT: A benchmark for causal reasoning about forces and inTeractions. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, _Findings of the Association for Computational Linguistics: ACL 2022_ , pages 2602–2627, Dublin, Ireland, May 2022. Association for Computational Linguistics. 

- [2] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1728–1738, 2021. 

- [3] Peter W Battaglia, Jessica B Hamrick, and Joshua B Tenenbaum. Simulation as an engine of physical scene understanding. _Proceedings of the National Academy of Sciences_ , 110(45):18327– 18332, 2013. 

- [4] Daniel M Bear, Elias Wang, Damian Mrowca, Felix J Binder, Hsiao-Yu Fish Tung, RT Pramod, Cameron Holdaway, Sirui Tao, Kevin Smith, Fan-Yun Sun, et al. Physion: Evaluating physical prediction from vision in humans and machines. _arXiv preprint arXiv:2106.08261_ , 2021. 

- [5] Blender Online Community. _Blender - a 3D modelling and rendering package_ , 2021. Blender Foundation, Stichting Blender Foundation, Amsterdam, The Netherlands. 

- [6] Zhenfang Chen, Kexin Yi, Yunzhu Li, Mingyu Ding, Antonio Torralba, Joshua B Tenenbaum, and Chuang Gan. Comphy: Compositional physical reasoning of objects and events from videos. _arXiv preprint arXiv:2205.01089_ , 2022. 

- [7] Erwin Coumans and Yunfei Bai. Pybullet, a python module for physics simulation for games, robotics and machine learning. 2016. 

- [8] Mingyu Ding, Zhenfang Chen, Tao Du, Ping Luo, Josh Tenenbaum, and Chuang Gan. Dynamic visual reasoning by learning differentiable physics models from video and language. _Advances In Neural Information Processing Systems_ , 34:887–899, 2021. 

- [9] Rohit Girdhar and Deva Ramanan. CATER: A diagnostic dataset for Compositional Actions and TEmporal Reasoning. In _ICLR_ , 2020. 

- [10] Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, Thomas Kipf, Abhijit Kundu, Dmitry Lagun, Issam Laradji, Hsueh-Ti (Derek) Liu, Henning Meyer, Yishu Miao, Derek Nowrouzezahrai, Cengiz Oztireli, Etienne Pot, Noha Radwan, Daniel Rebain, Sara Sabour, Mehdi S. M. Sajjadi, Matan Sela, Vincent Sitzmann, Austin Stone, Deqing Sun, Suhani Vora, Ziyu Wang, Tianhao Wu, Kwang Moo Yi, Fangcheng Zhong, and Andrea Tagliasacchi. Kubric: a scalable dataset generator. 2022. 

- [11] Jessica B Hamrick, Peter W Battaglia, Thomas L Griffiths, and Joshua B Tenenbaum. Inferring mass in complex scenes by mental simulation. _Cognition_ , 157:61–76, 2016. 

- [12] Artur Jesslen, Guofeng Zhang, Angtian Wang, Alan Yuille, and Adam Kortylewski. Robust 3d-aware object classification via discriminative render-and-compare. _arXiv preprint arXiv:2305.14668_ , 2023. 

- [13] Achuta Kadambi, Celso de Melo, Cho-Jui Hsieh, Mani Srivastava, and Stefano Soatto. Incorporating physics into data-driven computer vision. _Nature Machine Intelligence_ , 5(6):572–580, 2023. 

- [14] Prakhar Kaushik, Aayush Mishra, Adam Kortylewski, and Alan Yuille. Source-free and imageonly unsupervised domain adaptation for category level object pose estimation. _arXiv preprint arXiv:2401.10848_ , 2024. 

- [15] Nikolaos Kyriazis and Antonis Argyros. Physically plausible 3d scene tracking: The single actor hypothesis. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 9–16, 2013. 

10 

- [16] Zhuowan Li, Xingrui Wang, Elias Stengel-Eskin, Adam Kortylewski, Wufei Ma, Benjamin Van Durme, and Alan L Yuille. Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 14963–14973, 2023. 

- [17] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. _arXiv preprint arXiv:2311.10122_ , 2023. 

- [18] Shichen Liu, Tianye Li, Weikai Chen, and Hao Li. Soft rasterizer: A differentiable renderer for image-based 3d reasoning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 7708–7717, 2019. 

- [19] Wufei Ma, Qihao Liu, Jiahao Wang, Angtian Wang, Yaoyao Liu, Adam Kortylewski, and Alan Yuille. Adding 3d geometry control to diffusion models. _arXiv preprint arXiv:2306.08103_ , 2023. 

- [20] Wufei Ma, Angtian Wang, Alan Yuille, and Adam Kortylewski. Robust category-level 6d pose estimation with coarse-to-fine rendering of neural features. In _European Conference on Computer Vision_ , pages 492–508. Springer, 2022. 

- [21] Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, et al. OpenEQA: Embodied Question Answering in the Era of Foundation Models. In _CVPR_ , 2024. 

- [22] Maitreya Patel, Tejas Gokhale, Chitta Baral, and Yezhou Yang. Cripp-vqa: Counterfactual reasoning about implicit physical properties via video question answering. _arXiv preprint arXiv:2211.03779_ , 2022. 

- [23] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 32, 2018. 

- [24] Mathieu Salzmann and Raquel Urtasun. Physically-based motion models for 3d tracking: A convex formulation. In _2011 International Conference on Computer Vision_ , pages 2064–2071, 2011. 

- [25] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion5b: An open large-scale dataset for training next generation image-text models. _Advances in Neural Information Processing Systems_ , 35:25278–25294, 2022. 

- [26] Fish Tung, Mingyu Ding, Zhenfang Chen, Daniel M. Bear, Chuang Gan, Joshua B. Tenenbaum, Daniel L. K. Yamins, Judith Fan, and Kevin A. Smith. Physion++: Evaluating physical scene understanding that requires online inference of different physical properties. _arXiv_ , 2023. 

- [27] Tomer D Ullman, Andreas Stuhlmüller, Noah D Goodman, and Joshua B Tenenbaum. Learning physical parameters from dynamic scenes. _Cognitive psychology_ , 104:57–82, 2018. 

- [28] Angtian Wang, Adam Kortylewski, and Alan Yuille. Nemo: Neural mesh models of contrastive features for robust 3d pose estimation. _arXiv preprint arXiv:2101.12378_ , 2021. 

- [29] Xingrui Wang, Wufei Ma, Zhuowan Li, Adam Kortylewski, and Alan Yuille. 3d-aware visual question answering about parts, poses and occlusions. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. 

- [30] Xingrui Wang, Wufei Ma, Zhuowan Li, Adam Kortylewski, and Alan L Yuille. 3d-aware visual question answering about parts, poses and occlusions. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, _Advances in Neural Information Processing Systems_ , volume 36, pages 58717–58735. Curran Associates, Inc., 2023. 

- [31] Xingrui Wang, Wufei Ma, Zhuowan Li, Adam Kortylewski, and Alan L Yuille. 3d-aware visual question answering about parts, poses and occlusions. _Advances in Neural Information Processing Systems_ , 36, 2024. 

11 

- [32] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. _arXiv preprint arXiv:2212.03191_ , 2022. 

- [33] Jiajun Wu, Erika Lu, Pushmeet Kohli, Bill Freeman, and Josh Tenenbaum. Learning to see physics via visual de-animation. _Advances in neural information procesing systems_ , 30, 2017. 

- [34] Jiajun Wu, Ilker Yildirim, Joseph J Lim, Bill Freeman, and Josh Tenenbaum. Galileo: Perceiving physical object properties by integrating a physics engine with deep learning. _Advances in neural information processing systems_ , 28, 2015. 

- [35] Yu Xiang, Roozbeh Mottaghi, and Silvio Savarese. Beyond pascal: A benchmark for 3d object detection in the wild. In _IEEE winter conference on applications of computer vision_ , pages 75–82. IEEE, 2014. 

- [36] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. _arXiv preprint arXiv:2404.16994_ , 2024. 

- [37] Hongyang Xue, Zhou Zhao, and Deng Cai. Unifying the video and question attentions for openended video question answering. _IEEE Transactions on Image Processing_ , 26(12):5656–5666, 2017. 

- [38] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Learning to answer visual questions from web videos. _IEEE TPAMI_ , 2022. 

- [39] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. _arXiv preprint arXiv:1910.01442_ , 2019. 

- [40] Zhicheng Zheng, Xin Yan, Zhenfang Chen, Jingzhou Wang, Qin Zhi Eddie Lim, Joshua B Tenenbaum, and Chuang Gan. Contphy: Continuum physical concept learning and reasoning from videos. _arXiv preprint arXiv:2402.06119_ , 2024. 

12 

## **A Dynamic properties setting** 

Table 4: Dynamic settings of SuperCLEVR-Physics 

|**Attribute**|**Description**|**Aeroplanes**|**Others**|
|---|---|---|---|
|||||
|Mass|Calculated from volume|_ρV_||
|Position||||
||(_x, y_)|Beta distribution||
|||||
||_z_|Uniform distribution|0|
|Orientation|Faces to the center with noises<br>-|||
|Velocity|Initial state: static, slow, fast<br>_{_0_,_3_,_6_}_m/s|||
|Internal Force|Engine force (Forwards)<br>_{_1_,_0_}_m/s2|||
|||||
||Floating force(Upwards)|{10, 0} Mass_×_m/s2|0 Mass_×_m/s2|



## **External Forces and other physical properties** 

- **Gravity:** All objects in the dataset are subject to gravity, which influences their vertical movements and impacts when in flight or during falls. The gravitational constant is set as 10. 

- **Friction:** Frictional forces affect the movement of all objects, especially when they interact with the ground or each other, slowing them down and eventually bringing them to a stop. 

- **Elastic Force during Collisions:** When objects collide, the elastic forces come into play, defined by elasticity coefficients. These forces affect how objects bounce off each other or rebound from barriers, significantly altering their trajectories and speeds. 

**Physical Properties** (1) **Mass:** The mass for each object is calculated based on its specific shape model, which is linearly related to its volume with a density of 2.7; (2) **Friction** : The friction of objects are set as xx; the friction floor (the dome shaped background) is set as xx. (2) **Restitution** : The restitution of objects are xx and the floor’s are set to xx. 

## **B Question and program executor** 

We design new program for the 4D dynamcis reasoning VideoQA. For the factual question, we use `query_moving` , `query_moving_fast` , `query_moving_direction` for get the moving status. `greater_linear_velocities` , `less_linear_velocities` for comparing the linear velocities. `query_engine` for querying the physics engine. `query_floating` for querying the floating objects. `filter_collision` for filtering the collision events of given objects. `get_col_partner` for getting the collision partner of the object. `get_all_col_partners` for getting all the collision partners of the object. 

## **C Details of Dynamic Scene Parser with Physics Prior** 

The ultimate goal of the scene parser is to estimate the all dynamic state _St_ from the corresponding observation _It_ and the previous states _S<t_ using a probabilistic model. The previous work in 6D pose estimation has proposed to use render-and-compare to maximizing scene likelihood for static images. For the video data, we further consider the temporal consistency and physical plausibility of the dynamics estimation. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering_images/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering.pdf-0013-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering_images/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering.pdf-0013-13.png)


13 

Inspired by the neural-meshed based generative mode for pose estimation methods on static images [28, 20], we develop a de-rendering based generative model to reconstruct the 4D dynamic scene representation frame by frame. To ensure the predictions are not only plausible in 3D positions but also in physical rules, we incorporate a physical likelihood model and a 3D generative model with rendering likelihood, as shown in Fig. 3. 

**Preliminaries.** In the previous work for static images[20], Neural Meshes model were introduced for 6D pose estimation through inverse rendering. For that task, the goal is to jointly estimate the 6D pose (2D location ( _x, y_ ), distance _d_ to the camera and 3D pose ( _α, β, γ_ ) of objects in an image by comparing the Neural Meshes feature after rendering with the input image feature and maximizing the rendering likelihood. More formally, the mesh for a given object in category _c_ is represented as _Mc_ = _{vi ∈_ R[3] _|i_ = 1 _. . . N }_ where _vi_ means the vertex. The corresponding neural texture of the mesh _Mc_ is _Tc ∈_ R _[N][×][l]_ where _l_ is the dimension of the feature. So that the neural mesh model for category _c_ is their aggregation _Oc_ = _{Mc, Tc}_ . 

In this work, we use transform the position into a world coordinate instead. Given the object 3D rotation _R_ = ( _α, β, γ_ ) and translation _T_ = ( _x, y, z_ ), we can render the neural mesh model _Oc_ into a feature map _Fc_ with soft rasterization [18]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering_images/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering.pdf-0014-03.png)


**Inference as scene parsing** For static images, the previous work [28] has shown that the rendering likelihood can be used to estimate the 6D pose of objects. In our dynamics scene, similar strategies can be applied to each frame at time step _t_ . As the neural mesh models are probabilistic generative model of neural feature activation, we can first define the rendering likelihood of the feature map _Ft_ given any 6D pose _Rt, Tt_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering_images/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering.pdf-0014-05.png)


where _FG_ and _BG_ are the set of foreground and background locations on the 2D feature map and _ft_[(] _[i]_[)] is the feature vector of _F_ at location _i_ at timestep _t_ . B is a background parameter learned from training. Here the foreground and background likelihoods are modeled as Gaussian distributions. 

**Physical prior** (correction) 

Rendering likelihood alone is insufficient to reconstruct a physically plausible 4D dynamic scene representation. We also integrate a physics prior into our likelihood model. The physical process can be modeled as a Markov model [24], where the physical prior distribution of rotation _Rt_ and translation _Tt_ at time step _t_ can be expressed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering_images/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering.pdf-0014-09.png)


However, it’s nontrivial to directly model the physical functions. Most current studies on physical engines within neural networks only focus on simple shapes like cubes as the complexity in shapes of objects and the interaction process is hard to model. On the other hand, computational physical engines such as Bullet excel at modeling physical functions for any known shapes but are not differentiable and challenging to integrate into the de-rendering process during inference. 

In our method, we modify the discriminative physics engineering into a probabilistic model by introducing an uncertainty termˆ ˆ _ϵ ∼N_ (0 _, σ_[2] ). Denoted a physics engine as PE( _·_ ), then the _q_ ( _Rt, Tt | Rt−_ 1 _, Tt−_ 1) can be assumed by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering_images/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering.pdf-0014-12.png)


14 

where _µ_ = PE( _R_[ˆ] _t−_ 1 _, T_[ˆ] _t−_ 1), _C_ = 1 _/_ �(2 _π_ ) _[k] |σ_[2] _I|_ . Finally, the rotation _Rt_ and translation _Tt_ can be estimated by maximizing the joint likelihood of the rendering likelihood and the physical prior likelihood: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering_images/iclr_2025_Compositional_4D_Dynamic_Scenes_Understanding_with_Physics_Priors_for_Video_Question_Answering.pdf-0015-01.png)


**Relationship to Bayes-Kalman filtering** The conceptual ideas are directly inspired by the classic ideas of Bayes-Kalman filtering where the goal is to estimate a hidden state based on a sequence of observations. Bayes-Kalman filtering consists of updating a probability distribution of the hidden state by a prediction step followed by a correction step which incorporates evidence from a new observation. It uses a dynamic model for how the hidden state changes with time which is directly analogous to our physical prior. It has an observation model, corresponding to our scene parser, for how new observations give evidence for the hidden state. Bayes-Kalman, however, is difficult to implement in complex applications, like ours, because it requires us to represent and update a complex probability distribution. The standard approach for doing this is particle filtering where the probability distribution is represented by a set of point particles which are updated during the prediction and correction steps. This is also challenging so instead we use a simple approximation which essentially uses a single particle. In future work we will experiments to see if our model gives even better results if instead of this approximation we use particle filtering. 

## **D Baseline Models** 

We select 7 representative models from the following 3 categories as baseline models on SuperCLEVRPhysics. **(1) Simple classification-based methods** encode videos with a CNN backbone and predict answers with a classifier head. Specifically we consider _**CNN+LSTM**_ that aggregates frame-level CNN features and encode the question by LSTM, and _**FiLM**_ [23] that incorportes a feature-level linear modulation module for question answering. **(2) Neural symbolic models** first parse the scene into object instances and then executes a program for question answering. _**NS-DR**_ [39] adopts a Mask R-CNN for object detection. We modify it with the new program in SuperCLEVR-Physics and evaluate it on factual questions. As their 2D simulator is unable to reason the objects in 3D, we do not compare it on predictive and counterfactual questions. _**PO3D-VQA**_ [31] uses a 3D detector and reconstructs an explicit 3D scene representation for each frame. We extend the model for VideoQA by computing dynamic properties from object locations and predict collision by filtering the distance between objects. **(3) Large pretrained video-text models** adopt large-scale image-text or video-text data for pretraining and achieve strong generalization abilites to various video-text tasks. We consider _**Video-LLaVA**_ [17] and _**PLLaVA**_ [36] for zero-shot evaluation and choose the model sizes as 7B. For evaluation, we use a GPT model to justify the correctness of free-form answers. Moreover, we finetune a pretrained _**InternVideo**_ [32] model that predicts answers with a classifier head. 

15 

