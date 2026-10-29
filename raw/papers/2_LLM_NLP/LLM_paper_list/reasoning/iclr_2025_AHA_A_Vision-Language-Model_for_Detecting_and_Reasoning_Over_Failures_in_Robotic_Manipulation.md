# **AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation** 

**Jiafei Duan**[1] _[,]_[2] **Wilbert Pumacay**[3] **Nishanth Kumar**[1] _[,]_[4] **Yi Ru Wang**[1] _[,]_[2] **Shulin Tian**[5] **Wentao Yuan**[1] _[,]_[2] **Ranjay Krishna**[2] _[,]_[6] **Dieter Fox**[1] _[,]_[2] **Ajay Mandlekar** _[∗]_[1] **Yijie Guo** _[∗]_[1] 

1NVIDIA 2University of Washington 3Universidad Católica San Pablo 

4MIT 

5Nanyang Technological University 6Allen Institute for Artificial Intelligence 

## **aha-vlm.github.io** 

**Abstract:** Robotic manipulation in open-world settings requires not only task execution but also the ability to detect and learn from failures. While recent advances in vision-language models (VLMs) and large language models (LLMs) have improved robots’ spatial reasoning and problem-solving abilities, they still struggle with failure recognition, limiting their real-world applicability. We introduce AHA, an open-source VLM designed to detect and reason about failures in robotic manipulation using natural language. By framing failure detection as a free-form reasoning task, AHA identifies failures and provides detailed, adaptable explanations across different robots, tasks, and environments. We fine-tuned AHA using `FailGen` , a scalable framework that generates the first large-scale dataset of robotic failure trajectories, the AHA dataset. `FailGen` achieves this by procedurally perturbing successful demonstrations from simulation. Despite being trained solely on the AHA dataset, AHA generalizes effectively to real-world failure datasets, robotic systems, and unseen tasks. It surpasses the second-best model (GPT-4o in-context learning) by 10.3% and exceeds the average performance of six compared models—including five state-of-the-art VLMs—by 35.3% across multiple metrics and datasets. We integrate AHA into three manipulation frameworks that utilize LLMs/VLMs for reinforcement learning, task and motion planning, and zero-shot trajectory generation. AHA ’s failure feedback enhances these policies’ performances by refining dense reward functions, optimizing task planning, and improving sub-task verification, boosting task success rates by an average of 21.4% across all three tasks compared to GPT-4 models. 

**Keywords:** Failure detection and reasoning, Foundation models for robotics, Data generation, Zero-shot manipulation, robotic manipulation 

## **1 Introduction** 

In recent years, foundation models have made remarkable progress across various domains, demonstrating their ability to handle open-world tasks[1, 2, 3, 4]. These models, including large language models (LLMs) and vision-language models (VLMs), have shown proficiency in interpreting and executing human language instructions[5], producing accurate predictions and achieving strong task performance. However, despite these advancements, key challenges remain—particularly with hallucinations, where models generate responses that deviate from truth. Unlike humans, who can intuitively detect and adjust for such errors, these models often lack the mechanisms for recognizing their own mistakes[6, 7, 8]. 

> _∗_ Equal advising 

Learning from failure is a fundamental aspect of human intelligence. Whether it’s a child learning to skate or perfecting a swing, the ability reason over failures is essential for improvement[9, 10, 8]. The concept of improvement through failures is widely applied in training foundation models and is exemplified by techniques such as Reinforcement Learning with Human Feedback (RLHF)[5, 11], where human oversight and feedback steers models toward desired outcomes. This feedback loop plays a critical role in aligning generative models with real-world objectives. However, a crucial question persists: How can we enable these models to autonomously detect and reason about their own failures, particularly in robotics, where interactions and environments are stochastic and unpredictable? 

This need is particularly pressing in robotics, where foundation models such as VLMs and LLMs are increasingly used to address open-world tasks. Recent advancements have enabled these models to tackle spatial reasoning, object recognition, and multimodal problem-solving—skills vital for robotic manipulation[12, 13, 14, 15, 16]. VLMs and LLMs are already being integrated to automate reward generation for reinforcement learning[17, 18], develop task plans for motion planning[19], and even generate zero-shot robot trajectories[20, 21, 22, 23]. While these models excel at task execution, they often face challenges in detecting and reasoning over failures—skills that are crucial for navigating dynamic and complex environments. For example, if a robot drops an object mid-task, a human observer would immediately recognize the error and take corrective action. How can we empower robots with similar capabilities, allowing them not only to perform tasks but also to detect and learn from their mistakes? 

To learn from their mistakes, robots must first detect and understand why they failed. We introduce AHA, an open-source vision-language model (VLM) that uses natural language to detect and reason about failures in robotic manipulation. Unlike prior work that treats failure reasoning as a binary detection problem, we frame it as a free-form reasoning task, offering deeper insights into failure mode reasoning. Our model not only identifies failures but also generates detailed explanations. This approach enables AHA to adapt to various robots, camera viewpoints, tasks, and environments in both simulated and real-world scenarios. It can also be integrated into downstream robotic applications leveraging VLMs and LLMs. We make the following three major contributions: 

**1. We introduce** `FailGen` **, a data generation pipeline for the procedural generation of failure demonstration data for robotic manipulation tasks across simulators.** To instruction-tune AHA, we developed `FailGen` , the first automated data generation pipeline that procedurally creates the AHA dataset—a large-scale collection of robotic manipulation failures with over 49K+ image-query pairs across 79 diverse simulated tasks. Despite being fine-tuned only on the AHA dataset, AHA demonstrates strong generalization to real-world failure datasets, different robotic systems, and unseen tasks, as evaluated on three separate datasets not included in the fine-tuning. `FailGen` is also flexible data generation pipeline integrates seamlessly with various simulators, enabling scalable procedural generation of failure demonstrations. 

**2. We demonstrate that AHA excels in failure reasoning, generalizing across different embodiments, unseen environments, and novel tasks, outperforming both open-source and proprietary VLMs.** Upon fine-tuning AHA, we benchmarked it against six state-of-the-art VLMs, both opensource and proprietary, evaluating performance across four metrics on three diverse evaluation datasets, each featuring different embodiments, tasks, and environments out-of-distribution from the training data. AHA outperformed GPT-4o model by more than 20.0% on average across datasets and metrics, and by over 43.0% compared to LLaVA-v1.5-13B [24], the base model from which AHA is derived. This demonstrates AHA’s exceptional ability to detect and reason about failures in robotic manipulation across embodiment and domains. 

**3. We show that AHA enhances downstream robotic applications by providing failure reasoning feedback.** We demonstrate that AHA can be seamlessly integrated into robotic applications that utilize VLMs and LLMs. By providing failure feedback, AHA improves reward functions through Eureka reflection, enhances task and motion planning, and verifies sub-task success in zero-shot robotic manipulation. Across three downstream tasks, our approach achieved an average success rate 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_AHA_A_Vision-Language-Model_for_Detecting_and_Reasoning_Over_Failures_in_Robotic_Manipulation_images/iclr_2025_AHA_A_Vision-Language-Model_for_Detecting_and_Reasoning_Over_Failures_in_Robotic_Manipulation.pdf-0003-00.png)


Figure 1: AHA is a Vision-Language Model designed to detect and reason about failures in robotic manipulation. As an instruction-tuned VLM, it can enhance task performance in robotic applications that utilize VLMs for reward generation, task planning, or sub-task verification. By incorporating AHA into the reasoning pipeline, these applications can achieve accelerated and improved performance. 

21.4% higher than GPT-4 models, highlighting AHA’s effectiveness in delivering accurate natural language failure feedback to improve task performance through error correction. 

## **2 Related Work** 

AHA enables language reasoning for failure detection in robotic manipulation, enhancing downstream robotics applications. To provide context, we review progress in: 1) failure detection in robotic manipulation, 2) data generation in robotics, and 3) foundation models for robotic manipulation. 

**Failure Detection in Robotic Manipulation** . Failure detection and reasoning have long been studied in the Human-Robot Interaction (HRI) community [25, 26] and in works leveraging Task and Motion Planning (TAMP) [27]. With the recent widespread adoption of LLMs and VLMs in robot manipulation systems—either for generating reward functions or synthesizing robot trajectories [17, 18] in a zero-shot manner—the importance of detecting task failures has regained prominence [20, 22, 28, 29]. Most modern approaches focus on using off-the-shelf VLMs or LLMs as success detectors [30, 29, 31, 22], and some employ instruction-tuning of VLMs to detect failures [32]. However, these methods are often limited to binary success detection and does not provide language explanations for why failures occur. Our framework introduces failure reasoning in a new formulation, generating language-based explanations of failures to aid robotics systems that leverage VLMs and LLMs in downstream tasks. 

**Data Generation in Robotics** There have been many methods in robotic manipulation that automate data generation of task demonstrations at scale [33, 34], whether for training behavior cloning policies, instruction-tuning VLMs [14], or curating benchmarks for evaluating robotic policies in simulation 

3 

Table 1: **AHA datasets for instruction-tuning.** We combined the AHA dataset, our large-scale robotic manipulation failure dataset, with VQA and object detection data. By incorporating this diverse data mix into the fine-tuning process, AHA is able to reason about failures in robotic manipulation across different domains, embodiments, and tasks. 

|Source|The AHAdataset (Train)|VQA [24]|LVIS [45]|
|---|---|---|---|
|||||
|Quantity|49K|665K|100K|
|Query|For the given sub-tasks, frst determine|What is the cat doing in the image?|Find all instances of drawer.|
||it has succeed by choosing from ["yes",|||
||"no"] and then explain the reason why|||
||the current sub-tasks has failed.|||
|Answer|No, The robot gripper rotated with an|The cat is sticking its head into a vase|[(0.41, 0.68, 0.03, 0.05), (0.42, 0.73,|
||incorrect roll angle|or container, possibly drinking water or|0.04, 0.08), ...]|
|||investigating the interior of the item.||



[35, 36]. A well-known example is MimicGen [33], which automates task demonstration generation via trajectory adaptation by leveraging known object poses. Additionally, works like RoboPoint use simulation to generate general-purpose representations for robotic applications, specifically for fine-tuning VLMs. Similarly, systems like The Colosseum [36] automate data generation for curating benchmarks in robotic manipulation. Our approach aligns closely with RoboPoint, as we also leverage simulation to generate data for instruction-tuning VLMs. However, unlike RoboPoint, we focus on synthesizing robotic actions in simulation rather than generating representations like bounding boxes or points. 

**Foundation Models for Robotic Manipulation.** In recent years, leveraging foundation models for robotic manipulation has gained significant attention due to the effectiveness of LLMs/VLMs in interpreting open-world semantics and their ability to generalize across tasks [37, 38, 39, 40]. Two main approaches have emerged: the first uses VLMs and LLMs in a promptable manner, where visual prompts guide low-level action generation based on visual inputs [41, 21, 23]. The second focuses on instruction-tuning VLMs for domain-specific tasks [42]. For example, RoboPoint [14] is tuned for spatial affordance prediction, and Octopi [43] for physical reasoning using tactile images. These models generalize beyond their training data and integrate seamlessly into manipulation pipelines. Our approach follows this second path, developing a scalable method for generating instructiontuning data in simulation and fine-tuning VLMs specialized in detecting and reasoning about robotic manipulation failures, with applications that extend beyond manipulation tasks to other robotic domains. 

## **3 The AHA Dataset** 

We leveraged `FailGen` to procedurally generate the AHA dataset from RLBench tasks [44] and used it for the instruction-tuning of AHA. In this section, we begin by categorizing common failure modes in robotics manipulation and defining a taxonomy of failures in Section 3.1. Next, we explain how this taxonomy is used with `FailGen` to automate the data generation for the AHA dataset in simulation in Section 3.2. 

## **3.1 Failure Modes in Robotic Manipulation** 

To curate an instruction-tuning dataset of failure trajectories for robotic manipulation tasks, we began by systematically identifying prevalent failure modes. Our approach involved a review of existing datasets, including DROID [46] and Open-X Embodiment [47], as well as an analysis of policy 

4 

rollouts from behavior cloning models. We examined failures occurring in both teleoperated and autonomous policies. Building upon prior works, such as REFLECT [48], we formalized a taxonomy encompassing seven distinct failure modes commonly observed in robotic manipulation: incomplete grasp, inadequate grip retention, misaligned keyframe, incorrect rotation, missing rotation, wrong action sequence, and wrong target object. 

**Incomplete Grasp (** `No_Grasp` **) Failure:** `No_Grasp` is an object-centric failure that occurs when the gripper reaches the desired grasp pose but fails to close before proceeding to the next keyframe. 

**Inadequate Grip Retention (** `Slip` **) Failure:** `Slip` is an object-centric failure that happens after the object has been successfully grasped. As the gripper moves the object to the next task-specific keyframe, the grip loosens, causing the object to slip from the gripper. 

**Misaligned keyframe (** `Translation` **) Failure:** This action-centric failure occurs when the gripper moves toward a task keyframe, but a translation offset along the X, Y, or Z axis causes the task to fail. 

**Incorrect Rotation (** `Rotation` **) Failure:** `Rotation` is an action-centric failure that occurs when the gripper reaches the desired translation pose for the sub-task keyframe, but there is an offset in roll, yaw, or pitch, leading to task failure. 

**Missing Rotation (** `No_Rotation` **) Failure:** `No_Rotation` is an action-centric failure that happens when the gripper reaches the desired translation pose but fails to achieve the necessary rotation (roll, yaw, or pitch) for the sub-task, resulting in task failure. 

**Wrong Action Sequence (** `Wrong_action` **) Failure:** `Wrong_action` is an action-centric failure that occurs when the robot executes actions out of order, performing an action keyframe before the correct one. For example, in the task `put_cube_in_drawer` , the robot moves the cube toward the drawer before opening it, leading to task failure. 

**Wrong Target Object (** `Wrong_object` **) Failure:** `Wrong_object` is an object-centric failure that occurs when the robot acts on the wrong target object, not matching the language instruction. For example, in the task `pick_the_red_cup` , the gripper picks up the green cup, causing failure. 

## **3.2 Implementation of the AHA dataset** 

The AHA dataset is generated with RLBench [44], utilizing its keyframe-based formulation to dynamically induce failure modes during task execution. RLBench natively provides keyframes for task demonstrations, which enables flexibility in both object manipulation (handling tasks with varying objects) and the sequence of actions (altering the execution order of keyframes). Building on this foundation, we leverage `FailGen` , our custom environment wrapper to wrap around RLBench that allows for task-specific trajectory modifications through keyframes perturbations, object substitutions, and reordering of keyframe sequences. `FailGen` systematically generates failure trajectories aligned with the taxonomy defined in Section 3.1, yielding a curated dataset of 49k failure-question pairs. 

To generate the AHA dataset, we systematically sweep through all keyframes in each RLBench task, considering all potential configurations of the seven failure modes that could result in overall task failure. By leveraging the success condition checker in the simulation, we procedurally generate `YAML` -based configuration files by sweeping through each failure mode across all keyframes. These files provide details on potential failure modes, parameters (such as distance, task sequence, gripper retention strength, etc.), and corresponding keyframes that `FailGen` should perturb to induce failure. Additionally, we incorporate language templates to describe what the robot is doing between consecutive keyframes. Using these descriptions along with the failure modes, we can systematically curate question-answer pairs for each corresponding failure mode. 

For specific failure modes, `No_Grasp` is implemented by omitting gripper open/close commands at the relevant keyframes, effectively disabling gripper control. `Slip` introduces a timed release of the gripper shortly after activation. `Translation` and `Rotation` perturb the position and orientation of a keyframe, respectively, while `No_Rotation` constrains the keyframe’s rotational axis. `Wrong_Action` reorders keyframe activations to simulate incorrect sequencing, and `Wrong_Object` 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_AHA_A_Vision-Language-Model_for_Detecting_and_Reasoning_Over_Failures_in_Robotic_Manipulation_images/iclr_2025_AHA_A_Vision-Language-Model_for_Detecting_and_Reasoning_Over_Failures_in_Robotic_Manipulation.pdf-0006-00.png)


Figure 2: **Overview of AHA Pipeline** . (Top) The data generation for AHA is accomplished by taking a normal task trajectory in simulation and procedurally perturbing all keyframes using our taxonomy of failure modes. Through `FailGen` , we systematically alter keyframes to synthesize failure demonstrations conditioned on the original tasks. Simultaneously, we generate corresponding query and answer prompts for each task and failure mode, which are used for instruction-tuning. (Bottom) The instruction-tuning pipeline follows the same fine-tuning procedure as LLaVA-v1.5 [24], where we fine-tune only the LLM base model—in this case, LLaMA-2-13B and the projection linear layers, while freezing the image encoder and tokenizer. 

reassigns the keyframes intended for one object to another, maintaining the relative pose to mimic improper object manipulation. Using this pipeline, we also successfully generated a failure dataset from ManiSkill [49] and adapted RoboFail [48] for the evaluation of AHA. This further demonstrates the generalizability and versatility of `FailGen` in generating failure cases across different simulation environments. 

## **4 Method** 

This section outlines the failure reasoning problem formulation (Sec.4.1) used to fine-tune and evaluate AHA. Next, we discuss the curated data mix used for co-finetuning AHA (Sec.4.2). Finally, we detail the instruction fine-tuning pipeline and the model architecture selection for AHA (Sec.4.3). 

## **4.1 Failure Reasoning Formulation** 

Unlike previous works [48, 28, 22] that primarily focus on detecting task success as binary classification problem, we approach failure reasoning by first predicting a binary success condition ("Yes" or "No") of the given sub-task based on a language specification and an input image prompt. If the answer is "No", the VLM is expected to generate a concise, free-form natural language explanation detailing why the task is perceived as a failure. To formulate failure reasoning, we prompt the VLMs to analyze the trajectory failures at the current sub-task and provide reasoning for _why_ or _what_ led to the failure. We define manipulation task trajectories as a series of sub-tasks _{S_ 0 _, S_ 1 _, S_ 2 _, . . . , St}_ , where each sub-task is represented by two consecutive keyframes. For example, in a task like "stacking cubes", a sub-task could represent a primitive action, such as ’picking up the cube’. For the input formulation in VLMs for instruction fine-tuning and evaluation, we required a query prompt 

6 

with an input image for prompting the VLMs. The query prompt was generated using a template corresponding to the current sub-task the robot is performing. To capture the temporal relationships within the action sequence, the input image was constructed by selecting a single frame that represents the robot’s trajectory up to the current sub-task and concatenating it with frames from other viewpoints in the rollout sequence, as shown in Table 1. 

This input frame is built by concatenating all keyframes up to the current sub-task in temporal order, from left to right, with any remaining keyframes replaced by white image patches. To mitigate occlusions, we also included all the available camera viewpoints, concatenating them alongside the temporal sequence, and provide a detailed task description in the prompt, as illustrated in Table 1 (left image). The image data is structured as a matrix **I** , where each row corresponds to a different camera viewpoint _{V_ 0 _, V_ 1 _, . . . , Vn}_ and each column captures the temporal sequence of keyframes _{S_ 0 _, S_ 1 _, S_ 2 _, . . . , St}_ . This formulation for curating images serves as a general approach for formatting all datasets used for fine-tuning and evaluation. This structured input enables consistent handling of data across different tasks and viewpoints. Overall, our failure reasoning problem is to prompt VLM with sub-task discription and keyframe trajectory image to predict the success condition and language description of failure reason for each sub-task, as shown in Table 1. 

## **4.2 Synthetic Data for Instruction-tuning** 

To facilitate the instruction-tuning of AHA, we needed to systematically generate failure demonstration data. To achieve this, we developed `FailGen` , an environment wrapper that can be easily applied to any robot manipulation simulator. `FailGen` systematically perturbs successful robot trajectories for manipulation tasks, transforming them into failure trajectories with various modes of failure as depicted in Figure 2 (Top image). Using `FailGen` , we curated the AHA dataset (Train) dataset by alternating across 79 different tasks in the RLBench simulator, resulting in 49k failure image-text pairs. Furthermore, following proper instruction-tuning protocols for VLMs [24] and building on prior works [50, 14], co-finetuning is crucial to the success of instruction fine-tuning of VLMs. Therefore, in addition to the AHA dataset, we co-finetuned AHA with general visual question-answering (VQA) datasets sourced from internet data, which helps models retain pre-trained knowledge. Specifically, we included the VQA dataset [24], containing 665k conversation pairs, and the LVIS dataset [45], which comprises 100k instances with predicted bounding box centers and dimensions, as summarized in Table 1. 

## **4.3 Instruction Fine-tuning** 

We followed the instruction-tuning pipeline outlined by [51]. As depicted in Fig. 2, our model architecture includes an image encoder, a linear projector, a language tokenizer, and a transformerbased language model. The image encoder processes images into tokens, projected by a 2-layer linear projector into the same space as the language tokens. These multimodal tokens are then concatenated and passed through the language transformer. All components are initialized with pre-trained weights. During fine-tuning, only the projector and transformer weights are updated, while the vision encoder and tokenizer remain frozen. The model operates autoregressively, predicting response tokens and a special token marking the boundary between instruction and response. 

## **5 Experimental Results** 

In this section, we evaluate AHA’s detection and reasoning performance against six state-of-the-art VLMs, including both open-source and proprietary models, some utilizing in-context learning. The evaluation spans three diverse datasets, covering out-of-domain tasks, various simulation environments, and cross-embodiment scenarios. We then assess AHA’s ability to retain general world knowledge after fine-tuning on domain-specific data. Finally, we explore its potential to improve downstream robotic manipulation tasks. 

7 

Table 2: **Quantitative Evaluation on Failure Detection and Reasoning.** AHA-13B was evaluated and benchmarked against three open and three proprietary VLMs and one visual prompting baseline across three evaluation datasets. AHA-13B outperformed all other VLMs on every evaluation dataset and nearly every evaluation metric, with the exception of the AHA (Test) dataset, where GPT-4o exceeded by less than 3%. 

|**Models**|**Evaluation Datasets**|**Evaluation Metrics**<br>ROUGE_L ↑_<br>Cosine Similarity_↑_<br>Binary Success(%)_↑_<br>LLM Fuzzy Match_↑_|
|---|---|---|
|**LLaVA-v1.5-13B**[24]<br>**LLaVA-NeXT-34B**[52]<br>**Qwen-VL**[53]|**AHA dataset (Test set)**<br>**ManiSkill-Fail**<br>**RoboFail**[48]|0.061<br>0.208<br>0.080<br>0.648<br>0.000<br>0.208<br>0.022<br>0.270<br>0.000<br>0.203<br>0.000<br>0.404<br>0.013<br>0.231<br>0.017<br>0.626<br>0.001<br>0.195<br>0.007<br>0.277<br>0.018<br>0.188<br>0.017<br>0.351<br>0.000<br>0.161<br>0.000<br>0.426<br>0.037<br>0.301<br>0.116<br>0.034<br>0.000<br>0.159<br>0.000<br>0.050|
||**AHA dataset (Test set)**<br>**ManiSkill-Fail**<br>**RoboFail**[48]||
||**AHA dataset (Test set)**<br>**ManiSkill-Fail**<br>**RoboFail**[48]||
|**Gemini-1.5 Flash**[12]<br>**GPT-4o**<br>**GPT-4o-ICL (5-shot)**|**AHA dataset (Test set)**<br>**ManiSkill-Fail**<br>**RoboFail**[48]|0.120<br>0.231<br>0.371<br>0.566<br>0.003<br>0.121<br>0.014<br>0.032<br>0.000<br>0.042<br>0.000<br>0.393<br>0.251<br>0.308<br>0.500<br>**0.784**<br>0.142<br>0.335<br>0.688<br>0.453<br>0.114<br>0.318<br>0.554<br>0.438<br>0.226<br>0.380<br>0.611<br>0.776<br>0.341<br>0.429<br>0.971<br>0.630<br>0.236<br>0.429<br>0.571<br>0.418|
||**AHA dataset (Test set)**<br>**ManiSkill-Fail**<br>**RoboFail**[48]||
||**AHA dataset (Test set)**<br>**ManiSkill-Fail**<br>**RoboFail**[48]||
|**AHA-7B**<br>**AHA-13B (Ours)**|**AHA dataset (Test set)**<br>**ManiSkill-Fail**<br>**RoboFail**[48]|0.434<br>0.574<br>0.691<br>0.695<br>**0.609**<br>0.680<br>1.000<br>0.532<br>0.204<br>0.394<br>0.625<br>0.439<br>**0.446**<br>**0.583**<br>**0.702**<br>0.768<br>0.600<br>**0.681**<br>**1.000**<br>**0.633**<br>**0.280**<br>**0.471**<br>**0.643**<br>**0.465**|
||**AHA dataset (Test set)**<br>**ManiSkill-Fail**<br>**RoboFail**[48]||



Table 3: **Quantitative Evaluation on Standard VQA Benchmarks.** AHA-13B performs on par with LLaVA-13B [24], the VLM from which AHA adapts its fine-tuning strategy. 

||MMBench [54]|ScienceQA [55]|TextVQA [56]|POPE [57]|VizWiz[58]|
|---|---|---|---|---|---|
|LLaVA-13B (LLama-2) [24]|**67.70**|**73.21**|**67.40**|**88.00**|53.01|
|AHA-13B (LLama-2)|65.20|71.94|65.20|85.74|**53.45**|



## **5.1 Experimental Setup** 

To quantitatively evaluate AHA’s detection and reasoning capabilities for failures in robotic manipulation, we curated two datasets and adapted an existing failure dataset for benchmarking. To ensure a fair comparison of free-form language reasoning, we also employed four different evaluation metrics to measure semantic similarity between sentences. 

**Benchmarks.** We curated three datasets to evaluate AHA’s reasoning and failure detection capabilities, benchmarking against other state-of-the-art VLMs. The first dataset, AHA dataset (Test), includes 11k image-question pairs from 10 RLBench tasks, generated similarly to the fine-tuning data via `FailGen` (Section 3.2) but without overlapping with the tasks from the finetuning dataset. It evaluates AHA’s ability to generalize to novel, out-of-domain tasks. The second dataset, ManiSkill-Fail, comprises 130 image-question pairs across four tasks in ManiSkill [49], generated using `Failgen` wrapper on Maniskill simulator. This dataset assesses AHA’s performance in a different simulator and under changing viewpoints. Lastly, we adapted a failure benchmark from the RoboFail dataset [48], which features real-world robot failures in seven UR5 robot tasks. This allows for evaluation across simulation and real-world trajectories and across different embodiments. 

**Evaluation Metrics.** To fairly evaluate success detection and free language reasoning across all datasets and baselines, we employ four metrics. First, the **ROUGE-L score** measures the quality of generated text by focusing on the longest common subsequence between candidate and reference texts. Second, we use **Cosine Similarity** to assess similarity between texts or embeddings, avoiding the "curse of dimensionality". Third, **LLM Fuzzy Matching** utilizes an external language 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_AHA_A_Vision-Language-Model_for_Detecting_and_Reasoning_Over_Failures_in_Robotic_Manipulation_images/iclr_2025_AHA_A_Vision-Language-Model_for_Detecting_and_Reasoning_Over_Failures_in_Robotic_Manipulation.pdf-0009-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_AHA_A_Vision-Language-Model_for_Detecting_and_Reasoning_Over_Failures_in_Robotic_Manipulation_images/iclr_2025_AHA_A_Vision-Language-Model_for_Detecting_and_Reasoning_Over_Failures_in_Robotic_Manipulation.pdf-0009-01.png)


Figure 3: (Left) **Scaling law with the AHA dataset** . Scaling of effect of model performance with varying domain specific fine-tuning data. (Right) **Downstream Robotic Application Performance.** AHA-13B outperforms GPT-4o in reasoning about failures within these robotic applications, leading to improved performance of the downstream tasks. 

model—specifically, Anthropic’s unseen model, `claude-3-sonnet` —to evaluate semantic similarity in a teacher-student prompting format. Lastly, we calculate a **Binary success rate** by comparing the model’s predictions directly against the ground truth for success detection. 

## **5.2 Quantitative Experimental Results** 

We contextualize the performance of AHA by conducting a systematic evaluation of failure reasoning and detection across these three datasets, general VQA datasets, and performed ablation studies. 

**AHA generalizes across embodiments, unseen environments, and novel tasks.** To ensure fairness and eliminate bias in the detection and reasoning capabilities of AHA, we evaluated it on three different datasets that were never seen during fine-tuning, each designed to test a specific form of generalization. First, on the AHA dataset (test) dataset, AHA demonstrated its ability to **generalize reasoning across tasks and new behaviors within the same domain, outperforming the secondbest performing VLM, GPT-4o ICL** , by an average margin of 12.6% difference across all evaluation metrics. Second, we assessed AHA-13B on a dataset generated by the `Failgen` wrapper in a **different simulation domain** , ManiSkill, showing that our model outperforms GPT-4o-ICL by an average of 13.4% difference across all metrics as depicted in Table 2. Lastly, to demonstrate **generalization to real-world robots and different embodiments** , we evaluated AHA-13B on RoboFail [48], where it outperforms GPT-4o-ICL by 4.9% difference. 

**AHA retains common sense knowledge.** We evaluated AHA-13B’s performance on various VQA benchmarks and present the results in Table 3 . AHA-13B **performs comparably to LLaVAv1.5-13B (LLama-2) [24]** , with only a 1.5% margin difference as depicted in Table 3. Notably, LLaVA-v1.5-13B is a VLM trained on the same pre-trained weights as AHA-13B but fine-tuned on VQA data. This indicates that AHA-13B is capable of functioning as a general purpose VLM, in addition to excelling at failure reasoning. 

**AHA’s performance scales with data size.** We evaluated Aha’s performance using a range of AHA data for instruction fine-tuning, spanning [3k, 6k, 12k, 34k, 48k, 60k], and co-trained individual checkpoints corresponding to these data sizes as shown in Figure 3 (Left). The model was then assessed on the ManiSkill-Fail dataset across four evaluation metrics. An average quadratic fit gradient of 0.0022 across all four metrics demonstrates a **scaling effect with fine-tuning on our procedurally generated data pipeline** . This suggests that further scaling of the generated data may lead to improved model performance. 

## **5.3 Downstream Robotics Tasks** 

We demonstrate that AHA’s failure detection and reasoning capabilities are useful across a wide spectrum of downstream robotics applications. This includes automatic reward generation for reinforcement learning applications [17], automatic task plan generation for task and motion planning 

9 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_AHA_A_Vision-Language-Model_for_Detecting_and_Reasoning_Over_Failures_in_Robotic_Manipulation_images/iclr_2025_AHA_A_Vision-Language-Model_for_Detecting_and_Reasoning_Over_Failures_in_Robotic_Manipulation.pdf-0010-00.png)


Figure 4: **Downstream Robotic Application.** We demonstrated that AHA can be integrated into existing LLM/VLM-assisted robotic applications to provide failure reasoning and feedback, helping to accelerate and improve task success rates in these systems. 

applications [19], and as an improved verification step for automatic data generation systems [22]. Videos and detailed improved reward function, task plan, example videos from each applications and etc can be found on the project page: aha-vlm.github.io. 

**AHA enables efficient reward synthesis for reinforcement learning.** To evaluate this downstream task, we adapted Eureka’s [17] implementation to the ManiSkill simulator, which offers more statebased manipulation tasks. We strictly followed the Eureka reward function generation and reflection pipeline, modifying it by incorporating perception failure feedback via either AHA-13B or GPT-4o (acting as a baseline) to enhance the original LLM reflection mechanism. Instead of only including a textual summary of reward quality based on policy training statistics for automated reward editing, we further incorporated explanations of policy failures based on evaluation rollouts. We evaluated our approach on five reinforcement learning tasks from ManiSkill, ranging from tabletop to mobile manipulation. To systematically assess the reasoning capabilities of different VLMs under budget constraints, we sampled one reward function initially and allowed for iterations over two sessions of GPT API calls. Each policy was trained using PPO over task-specific training steps and evaluated across 1,000 test steps. During policy rollouts, we employed either AHA-13B or GPT-4o for reward reflection to improve the reward function. Comparing the evaluated policy success rates using different failure feedback VLMs, we observed that AHA-13B provided intuitive, human-level failure reasoning that aided in modifying and improving generated dense reward functions. This resulted in success across all five tasks within the budget constraints, and our approach **outperformed GPT4o by a significant margin of 22.34% in task success rate** shown in Figure 3 (Right). 

**AHA refines task-plan generation for TAMP.** To demonstrate AHA’s utility within a planning system, we incorporated our approach into PRoC3S [19]. The PRoC3S system solves tasks specified in natural language by prompting an LLM for a Language-Model Program (LMP) that generates plans, and then testing a large number of these plans within a simulator before executing valid plans on a robot. If no valid plan can be found within a certain number of samples (100 in our experiments), the LLM is re-prompted for a new LMP given failure information provided by the environment. Importantly, as is typical of TAMP methods, the original approach checks for a finite set of failures (inverse kinematics, collisions, etc.) from the environment, and returns any sampled plan that does not fail in any of these ways. We incorporated a VLM into this pipeline in two ways: (1) we prompt the VLM with visualizations of failed plan executions within the simulator, ask it to return an explanation for the failure, and feed this back to PRoC3S’ LLM during the LMP feedback stage, (2) after PRoC3S returns a valid plan, we provide a visualization of this to the VLM and ask it to return whether this plan truly achieves the natural language goal, with replanning triggered if not. We compared GPT-4o and AHA-13B as the VLM-based failure reasoning modules within this implementation of PRoC3S across three tasks (shown in Figure 4). Each task was evaluated over 10 trials, with a 

10 

maximum of 100 sampling steps and three feedback cycles provided by either GPT-4o or AHA-13B. The success rate for each task was recorded. As shown in Figure Figure 3 (Right), utilizing AHA-13B for **failure reasoning significantly improved the task success rate and outperforming GPT-4o by a substantial margin of 36.7%** . 

**AHA improves task verification for zero-shot robot data generation.** To demonstrate AHA’s utility in zero-shot robot demonstration generation, we integrated our approach into the `Manipulate-Anything` framework. This open-ended system employs various Vision-Language Models (VLMs) to generate diverse robot trajectories and perform a wide range of manipulation tasks without being constrained by predefined actions or scenarios. A critical component of `Manipulate-Anything` is its sub-task verification module, which analyzes past and current frames to decide whether a sub-task has been achieved before proceeding or re-iterating over the previous sub-task. We replaced the original VLM (GPT-4V) in the sub-task verification module with AHA-13B and evaluated performance across four RLBench tasks (Figure 4), conducting 25 episodes for each task. Our results show that **substituting the sub-task verification module’s VLM with AHA improved reasoning accuracy and overall task success by an average of 5%** . 

## **6 Conclusion** 

**Limitations.** AHA currently outputs language reasoning that is closely aligned with the failure scenarios in the fine-tuning data. However, there is an opportunity to output more open-ended failures, to cover those arising from modes outside of the ones included in the failure taxonomy. Additionally, while `FailGen` systematically curates failure data from simulations, distilling large pretrained policies to perform diverse tasks in simulation and sampling failure modes would allow us to generate more open-ended failure examples, potentially enhancing the instruction-tuned performance of AHA. 

**Conclusion.** We introduce AHA, an open-source vision-language model that significantly enhances robots’ ability to detect and reason about manipulation task failures using natural language. By framing failure detection as a free-form reasoning task, AHA not only identifies failures but also provides detailed explanations adaptable to various robots, tasks, and environments. Leveraging `FailGen` and the curated AHA dataset, we trained AHA on a diverse set of robotic failure trajectories. Our evaluations show that AHA outperforms existing models across multiple metrics and datasets. When integrated into manipulation frameworks, its natural language feedback greatly improves error recovery and policy performance compared to GPT-4 models. These results demonstrate AHA’s effectiveness in enhancing task performance through accurate error detection and correction. 

## **7 Acknowledgement** 

Jiafei Duan is supported by the Agency for Science, Technology and Research (A*STAR) National Science Fellowship. Wilbert Pumacay is supported by grant 234-2015-FONDECYT from Cienciactiva of the National Council for Science, Technology and Technological Innovation (CONCYTEC-PERU). 

## **References** 

- [1] D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu, et al. Palm-e: An embodied multimodal language model. _arXiv preprint arXiv:2303.03378_ , 2023. 

- [2] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al. Flamingo: a visual language model for few-shot learning. _Advances in neural information processing systems_ , 35:23716–23736, 2022. 

- [3] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

11 

- [4] L. Zhang, A. Rao, and M. Agrawala. Adding conditional control to text-to-image diffusion models. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 3836–3847, 2023. 

- [5] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. _Advances in neural information processing systems_ , 35:27730–27744, 2022. 

- [6] S. Lin, J. Hilton, and O. Evans. Truthfulqa: Measuring how models mimic human falsehoods. _arXiv preprint arXiv:2109.07958_ , 2021. 

- [7] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. D. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, et al. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ , 2021. 

- [8] G. D. Heyman. Children’s critical thinking when learning from others. _Current directions in psychological science_ , 17(5):344–347, 2008. 

- [9] H. P. Young. Learning by trial and error. _Games and economic behavior_ , 65(2):626–643, 2009. 

- [10] A. Gopnik. Childhood as a solution to explore–exploit tensions. _Philosophical Transactions of the Royal Society B_ , 375(1803):20190502, 2020. 

- [11] P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei. Deep reinforcement learning from human preferences. _Advances in neural information processing systems_ , 30, 2017. 

- [12] M. Reid, N. Savinov, D. Teplyashin, D. Lepikhin, T. Lillicrap, J.-b. Alayrac, R. Soricut, A. Lazaridou, O. Firat, J. Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_ , 2024. 

- [13] OpenAI. Hello gpt-4o, May 2024. URL `https://openai.com/index/hello-gpt-4o` . 

- [14] W. Yuan, J. Duan, V. Blukis, W. Pumacay, R. Krishna, A. Murali, A. Mousavian, and D. Fox. Robopoint: A vision-language model for spatial affordance prediction for robotics. _arXiv preprint arXiv:2406.10721_ , 2024. 

- [15] B. Chen, Z. Xu, S. Kirmani, B. Ichter, D. Driess, P. Florence, D. Sadigh, L. Guibas, and F. Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. _arXiv preprint arXiv:2401.12168_ , 2024. 

- [16] Y. R. Wang, J. Duan, D. Fox, and S. Srinivasa. Newton: Are large language models capable of physical reasoning? _arXiv preprint arXiv:2310.07018_ , 2023. 

- [17] Y. J. Ma, W. Liang, G. Wang, D.-A. Huang, O. Bastani, D. Jayaraman, Y. Zhu, L. Fan, and A. Anandkumar. Eureka: Human-level reward design via coding large language models. _arXiv preprint arXiv:2310.12931_ , 2023. 

- [18] Y. J. Ma, W. Liang, H.-J. Wang, S. Wang, Y. Zhu, L. Fan, O. Bastani, and D. Jayaraman. Dreureka: Language model guided sim-to-real transfer. _arXiv preprint arXiv:2406.01967_ , 2024. 

- [19] A. Curtis, N. Kumar, J. Cao, T. Lozano-Pérez, and L. P. Kaelbling. Trust the proc3s: Solving long-horizon robotics problems with llms and constraint satisfaction, 2024. URL `https: //arxiv.org/abs/2406.05572` . 

- [20] W. Huang, C. Wang, R. Zhang, Y. Li, J. Wu, and L. Fei-Fei. Voxposer: Composable 3d value maps for robotic manipulation with language models. _arXiv preprint arXiv:2307.05973_ , 2023. 

- [21] H. Huang, F. Lin, Y. Hu, S. Wang, and Y. Gao. Copa: General robotic manipulation through spatial constraints of parts with foundation models. _arXiv preprint arXiv:2403.08248_ , 2024. 

12 

- [22] J. Duan, W. Yuan, W. Pumacay, Y. R. Wang, K. Ehsani, D. Fox, and R. Krishna. Manipulateanything: Automating real-world robots using vision-language models. _arXiv preprint arXiv:2406.18915_ , 2024. 

- [23] W. Huang, C. Wang, Y. Li, R. Zhang, and L. Fei-Fei. Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. _arXiv preprint arXiv:2409.01652_ , 2024. 

- [24] H. Liu, C. Li, Y. Li, and Y. J. Lee. Improved baselines with visual instruction tuning, 2023. 

- [25] S. Ye, G. Neville, M. Schrum, M. Gombolay, S. Chernova, and A. Howard. Human trust after robot mistakes: Study of the effects of different forms of robot communication. In _2019 28th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN)_ , pages 1–7. IEEE, 2019. 

- [26] P. Khanna, E. Yadollahi, M. Björkman, I. Leite, and C. Smith. User study exploring the role of explanation of failures by robots in human robot collaboration tasks. _arXiv preprint arXiv:2303.16010_ , 2023. 

- [27] C. R. Garrett, T. Lozano-Pérez, and L. P. Kaelbling. Pddlstream: Integrating symbolic planners and blackbox samplers via optimistic adaptive planning. In _Proceedings of the international conference on automated planning and scheduling_ , volume 30, pages 440–448, 2020. 

- [28] M. Skreta, Z. Zhou, J. L. Yuan, K. Darvish, A. Aspuru-Guzik, and A. Garg. Replan: Robotic replanning with perception and language models. _arXiv preprint arXiv:2401.04157_ , 2024. 

- [29] H. Ha, P. Florence, and S. Song. Scaling up and distilling down: Language-guided robot skill acquisition. In _Conference on Robot Learning_ , pages 3766–3777. PMLR, 2023. 

- [30] Y. J. Ma, S. Sodhani, D. Jayaraman, O. Bastani, V. Kumar, and A. Zhang. Vip: Towards universal visual reward and representation via value-implicit pre-training. _arXiv preprint arXiv:2210.00030_ , 2022. 

- [31] L. Wang, Y. Ling, Z. Yuan, M. Shridhar, C. Bao, Y. Qin, B. Wang, H. Xu, and X. Wang. Gensim: Generating robotic simulation tasks via large language models. _arXiv preprint arXiv:2310.01361_ , 2023. 

- [32] Y. Du, K. Konyushkova, M. Denil, A. Raju, J. Landon, F. Hill, N. de Freitas, and S. Cabi. Vision-language models as success detectors. _arXiv preprint arXiv:2303.07280_ , 2023. 

- [33] A. Mandlekar, S. Nasiriany, B. Wen, I. Akinola, Y. Narang, L. Fan, Y. Zhu, and D. Fox. Mimicgen: A data generation system for scalable robot learning using human demonstrations. _arXiv preprint arXiv:2310.17596_ , 2023. 

- [34] R. Hoque, A. Mandlekar, C. Garrett, K. Goldberg, and D. Fox. Intervengen: Interventional data generation for robust and data-efficient robot imitation learning. _arXiv preprint arXiv:2405.01472_ , 2024. 

- [35] A. Xie, L. Lee, T. Xiao, and C. Finn. Decomposing the generalization gap in imitation learning for visual robotic manipulation. In _2024 IEEE International Conference on Robotics and Automation (ICRA)_ , pages 3153–3160. IEEE, 2024. 

- [36] W. Pumacay, I. Singh, J. Duan, R. Krishna, J. Thomason, and D. Fox. The colosseum: A benchmark for evaluating generalization for robotic manipulation. _arXiv preprint arXiv:2402.08191_ , 2024. 

- [37] J. Duan, S. Yu, H. L. Tan, H. Zhu, and C. Tan. A survey of embodied ai: From simulators to research tasks. _IEEE Transactions on Emerging Topics in Computational Intelligence_ , 6(2): 230–244, 2022. 

13 

- [38] Y. Hu, Q. Xie, V. Jain, J. Francis, J. Patrikar, N. Keetha, S. Kim, Y. Xie, T. Zhang, Z. Zhao, et al. Toward general-purpose robots via foundation models: A survey and meta-analysis. _arXiv preprint arXiv:2312.08782_ , 2023. 

- [39] R. Firoozi, J. Tucker, S. Tian, A. Majumdar, J. Sun, W. Liu, Y. Zhu, S. Song, A. Kapoor, K. Hausman, et al. Foundation models in robotics: Applications, challenges, and the future. _arXiv preprint arXiv:2312.07843_ , 2023. 

- [40] J. Urain, A. Mandlekar, Y. Du, M. Shafiullah, D. Xu, K. Fragkiadaki, G. Chalvatzaki, and J. Peters. Deep generative models in robotics: A survey on learning from multimodal demonstrations. _arXiv preprint arXiv:2408.04380_ , 2024. 

- [41] F. Liu, K. Fang, P. Abbeel, and S. Levine. Moka: Open-vocabulary robotic manipulation through mark-based visual prompting. _arXiv preprint arXiv:2403.03174_ , 2024. 

- [42] X. Li, C. Mata, J. Park, K. Kahatapitiya, Y. S. Jang, J. Shang, K. Ranasinghe, R. Burgert, M. Cai, Y. J. Lee, et al. Llara: Supercharging robot learning data for vision-language policy. _arXiv preprint arXiv:2406.20095_ , 2024. 

- [43] S. Yu, K. Lin, A. Xiao, J. Duan, and H. Soh. Octopi: Object property reasoning with large tactile-language models. _arXiv preprint arXiv:2405.02794_ , 2024. 

- [44] S. James, Z. Ma, D. R. Arrojo, and A. J. Davison. Rlbench: The robot learning benchmark & learning environment. _IEEE Robotics and Automation Letters_ , 5(2):3019–3026, 2020. 

- [45] A. Gupta, P. Dollar, and R. Girshick. Lvis: A dataset for large vocabulary instance segmentation. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 5356–5364, 2019. 

- [46] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. _arXiv preprint arXiv:2403.12945_ , 2024. 

- [47] A. Padalkar, A. Pooley, A. Jain, A. Bewley, A. Herzog, A. Irpan, A. Khazatsky, A. Rai, A. Singh, A. Brohan, et al. Open x-embodiment: Robotic learning datasets and rt-x models. _arXiv preprint arXiv:2310.08864_ , 2023. 

- [48] Z. Liu, A. Bahety, and S. Song. Reflect: Summarizing robot experiences for failure explanation and correction. _arXiv preprint arXiv:2306.15724_ , 2023. 

- [49] T. Mu, Z. Ling, F. Xiang, D. Yang, X. Li, S. Tao, Z. Huang, Z. Jia, and H. Su. Maniskill: Generalizable manipulation skill benchmark with large-scale demonstrations. _arXiv preprint arXiv:2107.14483_ , 2021. 

- [50] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. _arXiv preprint arXiv:2307.15818_ , 2023. 

- [51] H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning, 2023. 

- [52] H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 

- [53] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. _arXiv preprint arXiv:2308.12966_ , 2023. 

- [54] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, et al. Mmbench: Is your multi-modal model an all-around player? _arXiv preprint arXiv:2307.06281_ , 2023. 

14 

- [55] P. Lu, S. Mishra, T. Xia, L. Qiu, K.-W. Chang, S.-C. Zhu, O. Tafjord, P. Clark, and A. Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In _The 36th Conference on Neural Information Processing Systems (NeurIPS)_ , 2022. 

- [56] A. Singh, V. Natarjan, M. Shah, Y. Jiang, X. Chen, D. Parikh, and M. Rohrbach. Towards vqa models that can read. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 8317–8326, 2019. 

- [57] Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen. Evaluating object hallucination in large vision-language models. _arXiv preprint arXiv:2305.10355_ , 2023. 

- [58] D. Gurari, Q. Li, A. J. Stangl, A. Guo, C. Lin, K. Grauman, J. Luo, and J. P. Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 3608–3617, 2018. 

15 

