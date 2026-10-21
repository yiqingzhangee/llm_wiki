## **Situational Awareness Matters in 3D Vision Language Reasoning** 

Yunze Man Liang-Yan Gui Yu-Xiong Wang University of Illinois Urbana-Champaign 

_{_ yunzem2,lgui,yxw _}_ @illinois.edu 

## **Abstract** 

_Being able to carry out complicated vision language reasoning tasks in 3D space represents a significant milestone in developing household robots and human-centered embodied AI. In this work, we demonstrate that a critical and distinct challenge in 3D vision language reasoning is situational awareness, which incorporates two key components: (1) The autonomous agent grounds its self-location based on a language prompt. (2) The agent answers open-ended questions from the perspective of its calculated position. To address this challenge, we introduce SIG3D, an end-to-end Situation-Grounded model for 3D vision language reasoning. We tokenize the 3D scene into sparse voxel representation and propose a language-grounded situation estimator, followed by a situated question answering module. Experiments on the SQA3D and ScanQA datasets show that SIG3D outperforms state-of-the-art models in situation estimation and question answering by a large margin (e.g., an enhancement of over 30% on situation estimation accuracy). Subsequent analysis corroborates our architectural design choices, explores the distinct functions of visual and textual tokens, and highlights the importance of situational awareness in the domain of 3D question answering. The project page is available at https://yunzeman. github.io/situation3d._ 

## **1. Introduction** 

Humans learn knowledge efficiently through the interactions with the 3D world and the integration of multi-modal information, such as verbal guidance or instructions. Similarly, introducing language guidance into the visual comprehension task can greatly enhance the learning efficiency of models [3, 42]. Nonetheless, despite considerable advancements in linguistic understanding [7, 11, 31, 58] and visionlanguage integration [3, 37, 54, 61], current methodologies remain deficient in accurately perceiving and rationalizing within real-world 3D environments, which is largely attributed to the lack of 3D situational reasoning capabilities. 

Compared to machine learning models, humans put 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning_images/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning.pdf-0001-09.png)


**----- Start of picture text -----**<br>
<3D Scene> <Situation> “I am sitting on an armchair facing the center of the<br>room, and there is a brown sofa on my right.”<br><Question> “Describe to me: what I will see if I stand up from the<br>armchair and turn right?”<br>Previous Models Our SIG3D<br>Tokenization Tokenization<br>Q K, V<br>TransformerMulti-modal + HeadsTask  Cross-attentionSelf-attention Multi-modalLarge<br>Head Transformer<br>Situation<br>Response  <text> Situation < vector >  Estimation Response  <text><br>**----- End of picture text -----**<br>


Figure 1. Previous methods perform direct 3D vision language reasoning without modeling the situation of an embodied agent in the 3D environment. Our method, SIG3D, grounds the situational description in the 3D space, and then re-encodes the visual tokens from the agent’s intended perspective before vision-language fusion, resulting in a more comprehensive and generalized 3D vision language (VL) representation and reasoning framework. Q, K, V stand for query, key, and value, respectively. 

themselves inside the 3D world and then perceive and interact with the surrounding environment from their egoperspective (Figure 1). Such situational awareness is a crucial difference between 2D and 3D visual understanding, and a key to achieving seamless understanding of _spatial_ concepts in more complex real-world environments. Several existing methods recognize the lack of positional understanding in 3D and propose new benchmarks and joint optimization functions [44], or positional embedding methods [25] to enhance the overall reasoning performance. 

However, the lack of an _explicit_ situation modeling and situation-grounded 3D reasoning method restricts them from obtaining a generalizable and consistent 3D visionlanguage (VL) representation. As shown in Figure 2, the situation prediction of the state-of-the-art method [44] (in blue) diverges significantly from the ground truth vectors (in red) in almost all scenes in the dataset [13]. Moreover, our pilot study in Section 3 also reveals that situational understanding, despite being very crucial in comprehending the context of questions, only plays a minor role in the final 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning_images/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning.pdf-0002-00.png)


**----- Start of picture text -----**<br>
I am facing a trash can, while there is a  I am facing the paper towel dispenser with<br>door on my right. a chair in my six o’clock direction.<br>I am facing a pool table, taking out the<br>white ball from the side pocket. I am  I am sitting on a couch with a pillow facing<br>also facing a painting on the wall. another couch and the pillow is on my right.<br>**----- End of picture text -----**<br>


Figure 2. Situation estimation in existing methods [44] fails in most scenarios, indicating the missing registration between the situational descriptions and 3D embeddings. Red: Ground truth (GT) vector. Blue: Estimated vector. 

question answering (QA) performance of existing methods. 

In this work, we propose SIG3D, a novel approach designed to _precisely model and estimate an embodied agent’s ego-location and orientation from a textual description_ , before performing multi-modal QA tasks from the agent’s ego-centric perspective, as shown in Figure 1. Specifically, we leverage large-scale pretrained language and visual encoders to process the input text and 3D data, and fuse the tokens with attention modules to predict a situational vector. Previous attempts to directly predict the ego-situation are hindered by the expansive search space inherent in 3D environments. To address this challenge, we re-conceptualize the task as an _anchor-based classification_ , where visual tokens are regarded as anchor points, and a likelihood of position together with a set of rotation parameters are concurrently regressed for each visual token. After obtaining the situation estimation, we propose a situational alignment and a situation-guided tokens re-encoding strategy, to perceive the environment from the agent’s intended perspective. These strategies enhance the visual tokens with more accurate situational awareness for subsequent QA tasks. 

Experiments on two challenging 3D visual question answering (VQA) datasets [5, 44] demonstrate the significant improvement in situation estimation and QA tasks of our model. In particular, we improve the accuracy of situation estimation by more than 30%, and subsequent QA performance by up to 3%. Further qualitative and quantitative analysis verifies our design choices and highlights the significance of situational awareness in 3D reasoning tasks. 

To sum up, our paper has the following contributions: (1) We recognize the lack of situational awareness as a sig- 

nificant oversight in existing research. To address this, we introduce SIG3D, a situation-grounded 3D VL reasoning architecture, specifically designed to fill this void. (2) We propose an anchor-based approach to situation estimation, which effectively narrows the extensive search space in 3D environments for precise grounding of 3D positions and orientations with textual descriptions. Additionally, we investigate situational alignment and visual re-encoding mechanisms to leverage situational awareness for enhanced QA performance. (3) Our model demonstrates superior performance on two challenging datasets, SQA3D and ScanQA, surpassing the state of the art in both situation estimation and QA metrics. Ablation studies highlight the importance of situation-guided encoding, revealing its beneficial impact on general QA tasks. 

## **2. Related Work** 

**Vision Language Models (VLMs).** Early transformerdriven [59] textual and visual encoders [17, 31] have facilitated great progress in recent vision language learning. Text-image contrastive models [30, 54] propose to align the feature space of two modalities with large-scale pretraining, fueling numerous downstream tasks from generalized openvocabulary visual perception [21, 34, 36] to text-to-image generation [56]. Concurrently, some work uses text and vision encoders on separate modalities followed by feature fusion [18, 33] for multi-modal reasoning tasks. Since the emergence of Large Language Models (LLMs) [7, 58, 69], VLMs have experienced huge improvement with the help of LLMs as building blocks for multi-modal learning architectures. Specifically, recent work directly projects visual embeddings into language-space tokens as input to LLMs [40, 47, 70], or use the latent bottleneck structure for cross-modal visual decoding [3, 25, 37, 38], or treat LLM layers as encoder blocks for various visual tasks [49]. 

In the domain of visual question answering (VQA) [4, 73], recent work has pushed the frontier towards video understanding [14, 28, 29, 40, 62], knowledge-based understanding [20, 23, 41, 46, 57, 63], and commonsense reasoning [68]. Despite the outstanding performance in 2D image interpretation, most existing methods lack the capability to generalize to 3D scenarios. In contrast, our work studies the representation of visual information and its fusion with language embeddings in the 3D domain by targeting on the 3D situation-guided visual language interpretation. 

**Grounding Language in 3D Space.** Compared with 2D images, knowledge such as spatial relationships, interactive exploration, and topological analysis – which only exists in the 3D world – provides additional challenges and opportunities to develop better language models with stronger commonsense reasoning capability grounded in the realworld 3D scenarios. In this direction, early work seeks 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning_images/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Encode GT situational vector in the input 47.2<br>Corrupt situational supervision with noise 47<br>Remove situational description from input 45.3<br>Baseline 47.2<br>0 10 20 30 40 Acc. / %<br>**----- End of picture text -----**<br>


Figure 3. Results on variants of the representative SQA3D baseline method [44] demonstrate that situational understanding, despite being indispensable in perceiving the context of questions, makes negligible contribution in existing methods. This motivates a situation-guided 3D encoding mechanism in our model. 

to ground isolated objects [1, 9] or objects within more complex scenes [2, 8, 19, 27] using natural language descriptions. Recently, with more collected 3D vision language benchmarks, some work starts to explore languageguided 3D visual interpretation and reasoning on a diverse set of datasets, including 3D scene captioning [10], openvocabulary segmentation [16, 32, 51], and question answering [5, 15, 24, 26, 65, 74]. 

The success of LLMs also elicits the usage of them in 3D vision language reasoning for task decomposition [64], data generation, and multi-modal feature fusion [25]. Motivated by ScanQA [5], SQA3D [44] takes the first step to explore the challenging 3D situational reasoning problem by developing a situated question answering benchmark, and proposing the first joint learning baseline on this benchmark. Our work highlights the uniqueness and significance of situational awareness in the 3D vision language learning paradigm, leading to notably better 3D situational grounding and question answering performance. 

## **3. Pilot Study on Situational Reasoning** 

Despite highlighting the importance of situational understanding and reasoning, existing methods [44] fall short in providing effective situation estimation, as illustrated in Figure 2. This section delves into a pilot study that examines the impact of situational understanding on downstream reasoning tasks. The SQA3D baseline [44] incorporates situational descriptions and uses ground truth (GT) situational vectors for supervision in a direct regression task. We investigate three variants of this baseline to assess the effect of situational understanding. In the first variant, we _remove the situational description and supervision_ from the model, by passing in empty situational tokens. In another variant, we _corrupt the situation supervision_ by introducing very large Gaussian noise to the GT vectors to effectively randomize them. Finally, we try to _encode the GT situational vector in the input_ with learnable multi-layer perceptron (MLP) layers to form a GT situational token. 

Figure 3 demonstrates the results of this study, revealing negligible changes in performance across these variants. Notably, corrupting the GT situational information or di- 

rectly incorporating it results in only marginal alterations in the QA outcomes. Omitting the situational description entirely from the input results in a slight 2% decrease in precision. However, in the absence of this information, the model resorts to random guessing when determining the correct answer, as all responses depend on the situation. The findings from Figures 2 and 3 collectively indicate a deficiency in existing methods regarding situation estimation and the application of situational understanding in subsequent reasoning tasks. These unresolved challenges motivate the development of our proposed method. 

## **4. Method** 

An overview of our approach SIG3D is illustrated in Figure 4. Our method begins with a set of points that represent a 3D scene, accompanied by a situational description and a question that define the overall context of the problem. We tokenize them into separate token embeddings (Section 4.1), and ground the textual description in the 3D scene with a vector comprising location and orientation. We find direct single vector estimation to be challenging due to the vast and complex nature of the 3D search space, so we propose an anchor-based situation estimation strategy (Section 4.2). Subsequently, we re-encode the visual tokens from the perspective of situational vectors, enhancing the situational awareness for downstream reasoning tasks (Section 4.3). The finalized visual and textual tokens are fused by a transformer decoder to generate the final response. 

## **4.1. Visual and Textual Tokenization** 

Leveraging input scene point clouds and textual prompts, our objective is to generate three distinct types of tokens: 3D visual tokens _z_[3D] _∈_ R _[N][v][×][C][v]_ , situational tokens _z_[S] _∈_ R _[N][s][×][C][s]_ , and question tokens _z_[Q] _∈_ R _[N][q][×][C][q]_ . Each type of token is characterized by two primary components: _N_ , representing the number of tokens, and _C_ , encapsulating the feature embeddings. To tokenize and capture feature embeddings for situational input and questions, we employ a shared text tokenizer E[TXT] following prior methods [5, 44]. We assume that situation and question prompts are separated in the input data. If not, LLMs [7] can be used to parse the textual input without changing the semantic meaning of the sentences. However, there is a lack of consensus on a standard 3D visual tokenization method E[3D] that is appropriate for the 3D VL reasoning task, prompting a more detailed exploration in the following paragraphs. 

**Visual Tokenization.** Given an input point cloud **p** _∈_ R _[N][×]_[3] , most prior methods [5, 15, 44] adopt a VoteNet [53] detector to acquire object-level tokens _z_[3D] _∈_ R _[N]_[obj] _[×][C]_[obj] as the visual representation, where _N_ obj is the number of object proposals, and _C_ obj is the object-level feature embeddings. However, we point out several problems with 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning_images/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning.pdf-0004-00.png)


**----- Start of picture text -----**<br>
“Describe to me: what I will<br><Question> see if I stand up and turn<br>right.”<br>Large<br>Text  Multi-modal Output Text<br>Tokenizer  Transformer <Response><br>“I am sitting on an armchair<br><Situation> facing … there is a brown  Situation<br>sofa on my right.” Interpreter<br>K,V<br>3D Situational<br>Encoding<br>3D Visual   Q Multi-layer<br>Tokenizer Transformer<br><3D Scene><br>represent<br>3D Positional Encoding<br>Position Likelihood 6D Rotation<br>**----- End of picture text -----**<br>


Figure 4. **Overview of our SIG3D model** , which includes 3D scene and text encoding, anchor-based situation estimation, situation-guided visual re-encoding, and multi-modal decoder modules. We tokenize the 3D scene into voxels, treat each token as an anchor point, and query the text tokens to predict a token-level position likelihood and rotation matrix to locate the situational vector associated with the textual description. Then we update the scene tokens with situational position encoding (PE), and finally perform the 3D VL reasoning task with a large transformer decoder. 

this abstraction strategy: (1) A detection-based tokenization method tends to ignore the non-object regions in the scene, which can be indispensable in some reasoning scenarios ( _e.g_ ., carpets on the ground, ceiling, walls). (2) After object-level abstraction, the visual representation losses the high-level information of the scene ( _e.g_ ., the shape of the living room, the corner of the kitchen). (3) A supervised detector trained from scratch can only recognize objects within the training set ( _e.g_ ., only 20 categories for ScanNet [13]), meaning that the method does not have zero-shot capability to reason about novel unseen objects that are inevitably common in real-world scenarios. 

In light of these, we adopt a pretrained open-vocabulary voxel-based tokenization method from OpenScene [51]. The scene is first discretized into regular small 3D voxels and fed into a visual encoder for feature extraction: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning_images/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning.pdf-0004-04.png)


where _V_ represents the voxelization process, and E[3D] is a Minkowski sparse 3D convolutional network [12]. The sparse network is pretrained by distillation from CLIP [54] embeddings of rendered multi-view 2D images, resulting in a feature map with better language alignment and 3D awareness. We take the upsampled bottleneck-layer feature embeddings from the encoder network, and compute the mean average over the _z_ -axis (vertical) to project the voxels onto the _x_ - _y_ plane and treat the feature grids in the resulting 2D feature map as our _Nv_ visual tokens. We find that this bird’s-eye-view projection results in a more compact representation and improves the final performance. 

## **4.2. Situation Estimation** 

Given 3D visual tokens _z_[3D] and situational tokens _z_[S] , our objective is to estimate the situational vector _⃗s_ referred to by the situational description, which comprises a position component _s_[pos] represented by coordinates ( _x, y, z_ ), and a rotation component _s_[rot] represented by Euler angles ( _θ, ψ, ϕ_ ), where pitch angles _ψ_ are always defined as 0, meaning that situational vectors are defined to be parallel with the ground plane. The prior method [44] utilizes a transformer block to calculate the cross-attention feature between visual and language tokens, and directly regress a final situational vector from the averaged attention map. We find such a strategy producing very inaccurate estimates, as shown in Figure 2, due to the large search space in the entire 3D volume. Inspired by recent 3D object detection methods [45, 66, 71], we reduce the search space by _turning the localization problem into a classification problem_ . 

**Positional Embedding and Feature Fusion.** After the voxelization and 3D encoding process, each 3D token associates with a 3D position ( _x, y, z_ ) representing the center of its voxel. We first provide positional information to the model by generating learnable positional embeddings (PE) using a two-layer perceptron for each of the _Nv_ visual tokens, and add learnable positional embeddings to the token features _z_[3D] . We use a situation interpreter [55] to extract situational information, and ask the updated visual tokens to attend to these situational tokens with several transformer layers to produce the joint feature embeddings. 

**Anchor-based Situation Estimation.** We treat each out- 

4 

put token of the feature fusion module as an anchor point, and use it to predict a position likelihood _p ∈_ [0 _,_ 1] and a rotation estimation. Since each token has an associated 3D position ( _x, y, z_ ), the position likelihood _p_ indicates how likely the situational vector locates at the center of this token (voxel). We define a soft ground truth for this classification task with a Gaussian kernel, meaning that the closer a token is to the actual situational vector _s_[pos] , a higher ground truth probability _p_ will be assigned to that token. In order to counteract the sparse supervisory signal and increase the positive supervision around the vector position, we adopt the peak enlarging technique in CenterPoint [66], where the size of the Gaussian kernel is increased (meaning that the _σ_ is increased) to allow denser supervision around the vector position. Furthermore, we explore different rotation representation and find that compared with quaternion and (sin _θ,_ cos _θ_ ) representations, the 6D vector proposed by [72] achieves the best performance. Hence, we adopt a situation estimation head with MLP layers to output 7-dimensional vector for each of the tokens, where the first channel represents the position likelihood and the other six channels represent the 6D rotation matrix. We take the center of the token with the peak position likelihood as our estimated _s_[pos] , and convert its corresponding 6D rotation vector as our estimated _s_[rot] . The estimation can be equivalently represented as a rotation matrix _R_ and a translation matrix _T_ . More discussion about the architecture and design choices is in Section 5.3. 

## **4.3. Situation-guided Visual Encoding** 

After obtaining the situation estimation, we investigate a better approach to enhancing the generation of downstream responses, inspired by human cognitive processes. Intuitively, humans typically comprehend their immediate 3D environment by first interpreting their own situation in space, and then discerning their surroundings from an appropriate viewpoint. Our model is designed to emulate this natural strategy. Using the situational vector _⃗s_ , we adjust the coordinate system by repositioning the origin at _s_[pos] , and reorienting the axes according to _s_[rot] so that the new y- axis is aligned with the indicated direction. We keep the _z_ -axis vertically oriented and project the situational vectors onto the _x_ - _y_ plane. This is in line with the format of the dataset [44], where situational vectors are assumed to be parallel with the ground plane. Subsequently, we compute a new situation-guided PE for each of the _Nv_ visual tokens, similar to the learnable 3D PE outlined in Section 4.2. They allow the model to grasp the positional interrelations from the perspective of the current situation. These situational embeddings are added to the output embeddings of the situation estimation module, which consists of blocks featuring self-attention layers for visual tokens, succeeded by cross-attention layers that bridge visual and 

situational information. This structure allows for the reencoding of visual tokens under the influence of situation and question context, guiding the model to assign higher weights to situation-related and question-related visual tokens. The output, termed _situation-guided visual tokens_ , embodies this re-contextualized understanding. 

## **4.4. Question Answering Head** 

We follow existing methods [25] to use a large visionlanguage decoder to fuse the final visual and textual tokens and generate textual response to the input question. We explore both auto-regressive response generation and classification-based answer prediction [5, 44]. For classification, we predict a vector _v_[ans] _∈_ R _[n][a]_ for the candidates of _na_ answers in the training set following [5]. 

## **5. Analysis in 3D VQA Task** 

We evaluated SIG3D for 3D VL reasoning on two challenging benchmarks, addressing both visually-oriented situation estimation and textual-focused QA tasks. We present a detailed examination of the implementation strategies adopted, the datasets employed, and the metrics applied in our research. For exhaustive understanding, implementation, training details, and other additional information are available in the **supplementary material** . 

**Datasets.** We evaluate our method on SQA3D [44] and ScanQA [5], two challenging indoor 3D VQA datasets. Both datasets are derived from the ScanNet dataset [13], serving as the foundational source for their 3D scenes. SQA3D features over 33K question-answer pairs for the 3D VQA task and 26K unique situational descriptions for the situation estimation task. Each entry in this dataset includes a 3D scene point cloud, a situational description, a question, and pertinent annotations. ScanQA consists of over 41K question-answer pairs, without situational descriptions and situational annotations. We use it to demonstrate the generalizability of our method on general QA tasks. We use the splits provided by these datasets. 

**Evaluation Metrics.** For SQA3D, in order to compare with baseline methods [44, 49, 74], we use a shallow transformer decoder task head to perform the answer classification task, and evaluate the performance with exact matches (EM@1), which is equivalent to Top-1 answer accuracy. We also provide EM@1 on a breakdown of question types, including “What,” “Is,” “How,” “Can,” “Which,” and “Other,” based on the first word in the question sentence. Additionally, we evaluate situation estimation performance with localization accuracy and orientation accuracy. In both tasks, we use accuracy within different distance or angle thresholds as our metrics. For example, “Acc@0.5m” means accuracy of location estimation when positive threshold is set to 0.5 meter. For ScanQA, we perform auto-regressive an- 

5 

|Model|Question Breakdown<br>Overall<br>What<br>Is<br>How<br>Can<br>Which<br>Other|
|---|---|
|GPT-3 [7]<br>ClipBERT [35]<br>MCAN [67]<br>ScanQA [5]<br>SQA3D [44]<br>Multi-CLIP [15]<br>LM4Vision [49]<br>3D-LLM [25]<br>3D-VisTA [74]|**39.7**<br>46.0<br>40.5<br>45.6<br>36.1<br>38.4<br>41.0<br>30.2<br>60.1<br>38.7<br>63.3<br>42.5<br>42.7<br>43.3<br>28.9<br>59.7<br>44.1<br>68.3<br>40.7<br>40.5<br>43.4<br>28.6<br>65.0<br>47.3<br>66.3<br>43.9<br>42.9<br>45.3<br>33.5<br>66.1<br>42.4<br>69.5<br>43.0<br>46.4<br>47.2<br>-<br>-<br>-<br>-<br>-<br>-<br>48.0<br>34.3<br>67.1<br>48.2<br>68.3<br>48.9<br>45.6<br>48.1<br>36.5<br>65.6<br>47.2<br>68.8<br>48.0<br>46.3<br>48.1<br>34.8<br>63.3<br>45.4<br>69.8<br>47.2<br>**48.1**<br>48.5|
|SIG3D (Ours)|35.6<br>**67.2**<br>**48.5**<br>**71.4**<br>**49.1**<br>45.8<br>**52.6**|



Table 1. Our proposed SIG3D achieves state-of-the-art performance on the SQA3D benchmark [44]. We perform the best on “Is,” “How,” and “Can” breakdown types of questions, as well as the average accuracy with EM@1 metric. The results are reported on the test set. 

|Model|Localization|Orientation|
|---|---|---|
||Acc@0.5m Acc@1.0m|Acc@15° Acc@30°|
|Random<br>SQA3D [44]<br>SQA3D (_separate_)<br>3D-VisTA [74]|7.2<br>25.8<br>9.5<br>29.6<br>10.3<br>31.4<br>11.7<br>34.5|8.4<br>16.9<br>8.7<br>16.5<br>17.1<br>22.8<br>16.9<br>24.2|
|SIG3D (Ours)|**27.4**<br>**59.1**|**28.7**<br>**42.5**|



Table 2. Our proposed method SIG3D performs significantly better than prior methods [44] in the situation estimation task. “Acc@0.5m” stands for localization accuracy with 0.5m threshold. “Acc@15°” represents orientation accuracy with 15° threshold. _separate_ means disabling other tasks to let the model focus on situation estimation only. 

|Model|BLEU-1|BLEU-4|ROUGE|METEOR|CIDEr|
|---|---|---|---|---|---|
|BLIP2 [38]|29.7|5.9|26.6|11.3|45.7|
|Flamingo [3]|25.6|8.4|31.1|11.3|55.0|
|VN+MCAN [67]|28.0|6.2|29.8|11.4|54.7|
|SR+MCAN [67]|26.9|7.9|30.0|11.5|55.4|
|ScanQA [5]|30.2|10.1|33.3|13.1|64.9|
|3D-LLM [25]|39.3|12.0|35.7|**14.5**|**69.4**|
|SIG3D|**39.5**|**12.4**|**35.9**|13.4|68.8|



Table 3. Performance of SIG3D on the ScanQA dataset [5] is on-par with the state of the art with large-scale text-3D pretraining. VN and SR stand for VoteNet and ScanRefer, respectively. 3D-LLM [74] leverages pretrained 2D VL foundation models and LLM models [3, 7, 37, 38], and is pretrained on a large-scale heldin 3D-text dataset before the finetuning on ScanQA. 

swer generation with large transformer decoder [25], and evaluate with BLEU [50], ROUGE [39], METEOR [6], and CIDEr [60] metrics. 

## **5.1. Situated Question Answering** 

**Baselines.** Our study involves a comparative analysis with a range of representative baselines on the SQA3D dataset. In particular, we evaluate against **GPT-3** [7], **Clip-** 

**BERT** [35], and **MCAN** [67], which are, as reported in prior work [44], baselines focused on language-only, 2D video, and 2D image QA, respectively. For GPT-3, we follow SQA3D [44] to convert the visual input into a caption using Scan2Cap [10] for LLMs to process. **ScanQA** [5] represents a 3D QA baseline that ignores the situational input. Both **SQA3D** [44] and **Multi-CLIP** [15] employ situational descriptions and annotations for direct regression tasks. **LM4Vision** [49] utilizes LLMs as visual and textual encoders. Additionally, **3D-VisTA** [74] undergoes a pretraining procedure on their large-scale 3D scene-text dataset, ScanScribe, prior to the finetuning on this dataset. 

**Situation Estimation.** As shown in Table 2, our work performs significantly better than the state of the art [44, 74] in both localization and orientation estimation tasks. For 3DVisTA [74], we use a pretrained model and finetune a new situation head with the SQA3D dataset following [44]. We also report a _random_ baseline, in which we randomly sample position and orientation from a uniform distribution as a lower-bound performance. Note that the original SQA3D performs only marginally better than the random baseline, meaning that it does not acquire any situational awareness, despite having the situation estimation loss. Disabling the QA task and asking the model to exclusively focus on the situation estimation task results in a slight better performance. Our method, with the anchor-based position likelihood estimation, results in much better understanding of the 3D situational relationship. Our method also outperforms 3D-VisTA, which is pretrained on a large-scale 3Dtext dataset, indicating that large pretraining alone is not enough to address the situational awareness problem. Note that we do not include the random baseline performance reported in [44], because each value is obtained by generating three random values and taking the _closest_ one to the ground true, and thus it does not reflect a true “random” baseline. 

**Situated Question Answering.** SIG3D outperforms prior 

6 

|(a)Number of Visual Tokens<br>Acc@1.0m<br>Acc@30°<br>EM@1<br>128<br>48.9<br>38.2<br>49.2<br>256<br>**59.1**<br>**42.5**<br>**50.9**<br>512<br>57.8<br>42.1<br>50.7|(b)Voxel size(_in meters_)<br>Acc@1.0m<br>EM@1<br>0.01<br>54.1<br>49.5<br>0.02<br>**59.1**<br>**50.9**<br>0.05<br>47.3<br>48.8|(c)Rotation representation|
|---|---|---|
|||Acc@30°<br>EM@1|
|||Quaternion<br>31.4<br>50.0|
|||6D vector<br>42.5<br>**50.9**|
|||sin_θ,_cos_θ_<br>**42.6**<br>50.6|



Table 4. Ablation study validates that our various design choices improve the performance. “Acc@1.0m,” “Acc@30°,” and “EM@1” are accuracy (%) for localization estimation, orientation estimation, and QA tasks, respectively. Our settings are marked in gray . 

||Acc@1.0m<br>Acc@30°<br>EM@1|
|---|---|
|_3D Vision Encoder_||
|Text-only (_no vision input_)<br>VoteNet [53]<br>3DETR [48]<br>OpenScene - OpenSeg [51]<br>OpenScene - LSeg [51]|-<br>-<br>47.5<br>37.4<br>28.2<br>49.1<br>47.2<br>29.1<br>49.4<br>57.5<br>41.6<br>50.2<br>**59.1**<br>**42.5**<br>**50.9**|
|_Language Tokenizer / Encoder_||
|GloVe + LSTM [22,52]<br>SBERT - MiniLM [55]<br>SBERT - MPNet [55]<br>SBERT - MPNet (_fnetune_)|44.3<br>30.9<br>48.7<br>56.1<br>38.6<br>49.4<br>55.9<br>40.6<br>49.7<br>**59.1**<br>**42.5**<br>**50.9**|



Table 5. Performance of SIG3D improves with stronger visual and language encoders. We find that the open-vocabulary point encoder and MPNet-based sentence BERT (SBERT) leads to the best performance. “Acc@1.0m” and “Acc@30°” stand for localization and orientation accuracy in the situation estimation task, respectively. “EM@1” demonstrates the exact match metric in the QA task. 

methods in most question breakdown categories and overall accuracy, as shown in Table 1. Our work achieves leading results _without large-scale pretraining_ (compared with 3D-VisTA) and LLMs (compared with GPT-3), indicating its superiority in situational awareness. Note that the LLM baseline GPT-3 achieves the best performance on the “What” category, suggesting the potential of a stronger language encoder in interpreting complicated questions. 

## **5.2. General Question Answering on ScanQA** 

**Baselines.** We compare with 2D image VQA **MCAN** - based baselines [67], **ScanQA** [5], **3D-LLM** [25] which leverages large-scale pretrained 2D VLMs and LLMs as backbone models, and **3D-VisTA** [74] pretrained on their proposed large-scale 3D-text dataset. 

**Question Answering.** As shown in Table 3, despite that the questions do not explicitly require situational understanding to answer in ScanQA, SIG3D achieves comparable results with state-of-the-art methods without the large-scale 3D-text pretraining and powerful 2D VLM and LLM backbone models. Our work pretrained on SQA3D [44] leads to higher performance on BLEU-1, BLEU-4, and ROUGE metrics, showing its generalizability on general 3D QA scenarios. 

||Acc@1.0m<br>Acc@30°<br>EM@1|
|---|---|
|Baseline(_joint optimization_)|29.5<br>23.1<br>47.7|
|_How to achieve better situation estimation_||
|+ 3D PE<br>+ 6D Representation<br>+ Anchor-based Estimation|38.8<br>23.6<br>47.8<br>38.5<br>27.4<br>47.7<br>58.8<br>41.9<br>48.2|
|_How to utilize situation estimation for better QA_<br> <br><br><br>||
|+ 3D Situational PE<br>+ Visual Token Re-encoding|58.9<br>41.8<br>50.0<br>**59.1**<br>**42.5**<br>**50.9**|
|_Oracle Model (Ground Truth Situation Information)_||
|Situation as direct input<br>Situation as intermediate input|100<br>100<br>47.7<br>100<br>100<br>53.9|



Table 6. Ablation study verifies that our proposed modules lead to better situation estimation and better QA performance. 

## **5.3. Ablation Study and Analysis** 

**Vision and Language Encoders.** We study the impact of different visual and textual tokenizers in Table 5. It is observed that the open-vocabulary visual encoder (OpenScene) outperforms detection-based encoders (such as VoteNet and 3DETR) across all metrics. This superior performance of OpenScene is attributed to the limitations of 3D detectors, which are typically trained on a limited set of object categories, rendering them less effective in recognizing novel objects mentioned in textual prompts. Regarding language encoders, our findings indicate that a stronger backbone correlates with better performance, primarily due to its improved capability to interpret complex textual inputs. This leads to the suggestion of integrating LLMs with our method to potentially further enhance performance, an avenue we intend to explore in future research. 

**Situational Awareness.** In Table 6 we verify the crucial role of situational awareness in the 3D VL task. Firstly, we show that 3D PE, 6D rotation estimation, and anchor-based position estimation all lead to much better position and orientation estimation performance. We further establish that situational PE and visual token re-encoding modules lead to better utilization of the predicted situational vector for the QA task. Additionally, We design two oracle models under the assumption of having access to the ground truth situational vector as input. The outcomes from these models reveal a critical insight: the model fails to effectively interpret situational information when it is directly incorporated into the input visual embeddings. This underlines the 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning_images/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning.pdf-0008-00.png)


**----- Start of picture text -----**<br>
Situational Description : After an  Situational Description:  I am sitting  Situational Description:  I am sitting on  Situational Description:  I am leaning<br>exhausted day, I am lying on the bed  in an armchair with a lamp behind me. a chair in between a shelf on my right  on the door facing the blinds.<br>closest to the window with my head  and a box on the other side.<br>on the pillows.<br>Question : Can I see the television  Question : Which direction should I go  Question : Is the picture on the shelf or  Question : I need to make a call so<br>from my position on the bed? if I want to open the window? on the wall? where can I find the phone?<br>SQA3D<br>Answer : No Answer : Right Answer : Wall Answer : Bed<br>SIG3D<br>(Ours)<br>Answer : Yes Answer : Left Answer : Shelf Answer : Desk<br>**----- End of picture text -----**<br>


Figure 5. Qualitative results demonstrate significant improvement of SIG3D over prior methods. The first row is results from SQA3D [44], and the second row is results from our method. In the 3D scene, red: Ground truth (GT) vector, and blue: Estimated vector. 

necessity of the intermediate representation and encoding mechanism we have proposed, affirming its importance in achieving optimal 3D VL task performance. 

**Architectural Design.** We explore different architectural design choices of our model in Table 4. We find that the number of visual tokens sampled from the visual feature embeddings affects the performance of both situation estimation and QA tasks. Sampling fewer visual tokens increases the risk of missing the region of significance, while sampling more does not lead to a better performance as well. We study the size of voxels and find 0.02m to be the most effective choice, as the OpenScene [51] backbone is pretrained with the same voxel size. We also find that the (sin _θ,_ cos _θ_ ) and 6D vector representations perform a lot better than quaternion in the rotation estimation task. This is consistent with the finding reported in [72]. 

## **5.4. Qualitative Analysis** 

Finally, we demonstrate some qualitative results of our SIG3D in Figure 5. We show the ground truth and estimated situational vectors in red and blue, respectively, in their corresponding 3D scenes. We also print the answers with a red cross or green checkmark indicating the correctness. It is clear that our method performs significantly better in situation estimation tasks, resulting in vectors very close to the ground truth in both position and orientation perspectives. Better situational awareness also aids the complicated 

embodied navigation and commonsense QA activities. This further demonstrates great potential of our method in the development of indoor robotics and/or conversational agents. 

**Supplementary Material.** The supplementary section offers an extensive analysis, encompassing a detailed examination of the _3D visual token activation changes_ preand post-situational re-encoding. Additionally, it includes a comprehensive collection of _positive and negative samples_ , an insightful _failure case analysis_ , and a forward-looking discussion on _limitations and future work_ . 

## **6. Conclusion** 

In this paper, we introduce SIG3D, a situation-aware vision language model for 3D reasoning tasks. We propose to represent 3D scenes as feature tokens, treat tokens as anchor points to estimate a situational vector from a textual description, and use the estimated situation as guidance to align and re-encode the visual tokens to enhance the features for reasoning tasks. We observe consistent and significant performance gains on both situation estimation and question answering tasks. 

**Acknowledgement.** This work was supported in part by NSF Grant 2106825, NIFA Award 2020-67021-32799, the Jump ARCHES endowment, and the IBM-Illinois Discovery Accelerator Institute. This work used NVIDIA GPUs at NCSA Delta through allocations CIS220014, CIS230012, and CIS230013 from the ACCESS program. 

8 

## **References** 

- [1] Panos Achlioptas, Judy Fan, Robert Hawkins, Noah Goodman, and Leonidas J Guibas. ShapeGlot: Learning language for shape differentiation. In _ICCV_ , 2019. 3 

- [2] Panos Achlioptas, Ahmed Abdelreheem, Fei Xia, Mohamed Elhoseiny, and Leonidas Guibas. ReferIt3D: Neural listeners for fine-grained 3D object identification in real-world scenes. In _ECCV_ , 2020. 3 

- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: A visual language model for few-shot learning. In _NeurIPS_ , 2022. 1, 2, 6 

- [4] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. VQA: Visual question answering. In _ICCV_ , 2015. 2 

- [5] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. ScanQA: 3D question answering for spatial scene understanding. In _CVPR_ , 2022. 2, 3, 5, 6, 7 

- [6] Satanjeev Banerjee and Alon Lavie. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In _Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization_ , 2005. 6 

- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In _NeurIPS_ , 2020. 1, 2, 3, 6 

- [8] Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. ScanRefer: 3D object localization in RGB-D scans using natural language. In _ECCV_ , 2020. 3 

- [9] Kevin Chen, Christopher B Choy, Manolis Savva, Angel X Chang, Thomas Funkhouser, and Silvio Savarese. Text2shape: Generating shapes from natural language by learning joint embeddings. In _ACCV_ , 2019. 3 

- [10] Zhenyu Chen, Ali Gholami, Matthias Nießner, and Angel X Chang. Scan2Cap: Context-aware dense captioning in RGBD scans. In _CVPR_ , 2021. 3, 6 

- [11] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, 

   - Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. PaLM: Scaling language modeling with pathways. _arXiv preprint arXiv:2204.02311_ , 2022. 1 

- [12] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4D spatio-temporal convNets: Minkowski convolutional neural networks. In _CVPR_ , 2019. 4, 1 

- [13] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richly-annotated 3D reconstructions of indoor scenes. In _CVPR_ , 2017. 1, 4, 5 

- [14] Samyak Datta, Sameer Dharur, Vincent Cartillier, Ruta Desai, Mukul Khanna, Dhruv Batra, and Devi Parikh. Episodic memory question answering. In _CVPR_ , 2022. 2 

- [15] Alexandros Delitzas, Maria Parelli, Nikolas Hars, Georgios Vlassis, Sotirios Anagnostidis, Gregor Bachmann, and Thomas Hofmann. Multi-CLIP: Contrastive vision-language pre-training for question answering tasks in 3D scenes. In _BMVC_ , 2023. 3, 6 

- [16] Runyu Ding, Jihan Yang, Chuhui Xue, Wenqing Zhang, Song Bai, and Xiaojuan Qi. PLA: Language-driven openvocabulary 3D scene understanding. In _CVPR_ , 2023. 3 

- [17] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_ , 2021. 2 

- [18] Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang Zhu, Pengchuan Zhang, Lu Yuan, Nanyun Peng, Zicheng Liu, and Michael Zeng. An empirical study of training end-to-end vision-and-language transformers. In _CVPR_ , 2022. 2 

- [19] Mingtao Feng, Zhen Li, Qi Li, Liang Zhang, XiangDong Zhang, Guangming Zhu, Hui Zhang, Yaonan Wang, and Ajmal Mian. Free-form description guided 3D visual graph network for object grounding in point cloud. In _ICCV_ , 2021. 3 

- [20] Franc¸ois Gard`eres, Maryam Ziaeefard, Baptiste Abeloos, and Freddy Lecue. ConceptBERT: Concept-aware representation for visual question answering. In _EMNLP_ , 2020. 2 

- [21] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Scaling open-vocabulary image segmentation with image-level labels. In _ECCV_ , 2022. 2 

- [22] Alex Graves and Alex Graves. Long short-term memory. _Supervised Sequence Labelling with Recurrent Neural Networks_ , pages 37–45, 2012. 7 

9 

- [23] Liangke Gui, Borui Wang, Qiuyuan Huang, Alex Hauptmann, Yonatan Bisk, and Jianfeng Gao. KAT: A knowledge augmented transformer for vision-and-language. In _NAACL_ , 2022. 2 

- [24] Yining Hong, Chunru Lin, Yilun Du, Zhenfang Chen, Joshua B Tenenbaum, and Chuang Gan. 3D concept learning and reasoning from multi-View images. In _CVPR_ , 2023. 3 

- [25] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3D-LLM: Injecting the 3D world into large language models. In _NeurIPS_ , 2023. 1, 2, 3, 5, 6, 7 

- [26] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3D world. In _ICML_ , 2024. 3 

- [27] Pin-Hao Huang, Han-Hung Lee, Hwann-Tzong Chen, and Tyng-Luh Liu. Text-guided graph neural networks for referring 3D instance segmentation. In _AAAI_ , 2021. 3 

- [28] Baoxiong Jia, Yixin Chen, Siyuan Huang, Yixin Zhu, and Song-Chun Zhu. LEMMA: A multi-view dataset for learning multi-agent multi-task activities. In _ECCV_ , 2020. 2 

- [29] Baoxiong Jia, Ting Lei, Song-Chun Zhu, and Siyuan Huang. Egotaskqa: Understanding human tasks in egocentric videos. In _NeurIPS_ , 2022. 2 

- [30] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _ICML_ , 2021. 2 

- [31] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In _NAACL_ , 2019. 1, 2 

- [32] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. LERF: Language embedded radiance fields. In _ICCV_ , 2023. 3 

- [33] Wonjae Kim, Bokyung Son, and Ildoo Kim. ViLT: Visionand-language transformer without convolution or region supervision. In _ICML_ , 2021. 2 

- [34] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross B. Girshick. Segment anything. In _ICCV_ , 2023. 2 

- [35] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, Mohit Bansal, and Jingjing Liu. Less is more: Clipbert for video-and-language learning via sparse sampling. In _CVPR_ , 2021. 6 

- [36] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and Rene Ranftl. Language-driven semantic segmentation. In _ICLR_ , 2022. 2 

- [37] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In _ICML_ , 2022. 1, 2, 6 

- [38] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _ICML_ , 2023. 2, 6, 1 

- [39] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In _ACL_ , 2004. 6 

- [40] Xudong Lin, Simran Tiwari, Shiyuan Huang, Manling Li, Mike Zheng Shou, Heng Ji, and Shih-Fu Chang. Towards fast adaptation of pretrained contrastive models for multichannel video-language retrieval. In _CVPR_ , 2023. 2, 1 

- [41] Yuanze Lin, Yujia Xie, Dongdong Chen, Yichong Xu, Chenguang Zhu, and Lu Yuan. REVIVE: Regional visual representation matters in knowledge-based visual question answering. In _NeurIPS_ , 2022. 2 

- [42] Zhiqiu Lin, Samuel Yu, Zhiyi Kuang, Deepak Pathak, and Deva Ramanan. Multimodality helps unimodality: Crossmodal few-shot learning with multimodal models. In _CVPR_ , 2023. 1 

- [43] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2017. 1 

- [44] Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, Song-Chun Zhu, and Siyuan Huang. SQA3D: Situated question answering in 3D scenes. In _ICLR_ , 2023. 1, 2, 3, 4, 5, 6, 7, 8 

- [45] Yunze Man, Liang-Yan Gui, and Yu-Xiong Wang. BEVguided multi-modality fusion for driving perception. In _CVPR_ , 2023. 4 

- [46] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. OK-VQA: A visual question answering benchmark requiring external knowledge. In _CVPR_ , 2019. 2 

- [47] Jack Merullo, Louis Castricato, Carsten Eickhoff, and Ellie Pavlick. Linearly mapping from image to text space. _ICLR_ , 2023. 2 

- [48] Ishan Misra, Rohit Girdhar, and Armand Joulin. An endto-end transformer model for 3D object detection. In _ICCV_ , 2021. 7 

- [49] Ziqi Pang, Ziyang Xie, Yunze Man, and Yu-Xiong Wang. Frozen transformers in language models are effective visual encoder layers. In _ICLR_ , 2024. 2, 5, 6 

- [50] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. BLEU: A method for automatic evaluation of machine translation. In _ACL_ , 2002. 6 

- [51] Songyou Peng, Kyle Genova, Chiyu ”Max” Jiang, Andrea Tagliasacchi, Marc Pollefeys, and Thomas Funkhouser. OpenScene: 3D scene understanding with open vocabularies. In _CVPR_ , 2023. 3, 4, 7, 8, 1 

- [52] Jeffrey Pennington, Richard Socher, and Christopher D Manning. Glove: Global vectors for word representation. In _EMNLP_ , 2014. 7 

- [53] Charles R Qi, Or Litany, Kaiming He, and Leonidas J Guibas. Deep hough voting for 3D object detection in point clouds. In _ICCV_ , 2019. 3, 7 

- [54] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 1, 2, 4 

- [55] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using siamese BERT-networks. In _EMNLP_ , 2019. 4, 7, 1 

10 

- [56] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In _CVPR_ , 2022. 2 

- [57] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-OKVQA: A benchmark for visual question answering using world knowledge. In _ECCV_ , 2022. 2 

- [58] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. LLaMA: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 1, 2 

   - [71] Brady Zhou and Philipp Kr¨ahenb¨uhl. Cross-view transformers for real-time map-view semantic segmentation. In _CVPR_ , 2022. 4 

   - [72] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In _CVPR_ , 2019. 5, 8 

   - [73] Yuke Zhu, Oliver Groth, Michael Bernstein, and Li FeiFei. Visual7W: Grounded question answering in images. In _CVPR_ , 2016. 2 

   - [74] Ziyu Zhu, Xiaojian Ma, Yixin Chen, Zhidong Deng, Siyuan Huang, and Qing Li. 3D-VisTA: Pre-trained transformer for 3D vision and text alignment. In _ICCV_ , 2023. 3, 5, 6, 7 

- [59] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _NeurIPS_ , 2017. 2, 1 

- [60] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. CIDER: Consensus-based image description evaluation. In _CVPR_ , 2015. 6 

- [61] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, and Jifeng Dai. VisionLLM: Large language model is also an open-ended decoder for vision-centric tasks. In _NeurIPS_ , 2024. 1 

- [62] Bo Wu, Shoubin Yu, Zhenfang Chen, Joshua B Tenenbaum, and Chuang Gan. STAR: A benchmark for situated reasoning in real-world videos. In _NeurIPS (Datasets and Benchmarks Track)_ , 2021. 2 

- [63] Jialin Wu, Jiasen Lu, Ashish Sabharwal, and Roozbeh Mottaghi. Multi-modal answer validation for knowledge-based vqa. In _AAAI_ , 2022. 2 

- [64] Jianing Yang, Xuweiyi Chen, Shengyi Qian, Nikhil Madaan, Madhavan Iyengar, David F Fouhey, and Joyce Chai. LLMGrounder: Open-vocabulary 3D visual grounding with large language model as an agent. In _ICRA_ , 2024. 3 

- [65] Shuquan Ye, Dongdong Chen, Songfang Han, and Jing Liao. 3D question answering. _IEEE Transactions on Visualization and Computer Graphics_ , 30(3):1772–1786, 2024. 3 

- [66] Tianwei Yin, Xingyi Zhou, and Philipp Krahenbuhl. Centerbased 3D object detection and tracking. In _CVPR_ , 2021. 4, 5 

- [67] Zhou Yu, Jun Yu, Yuhao Cui, Dacheng Tao, and Qi Tian. Deep modular co-attention networks for visual question answering. In _CVPR_ , 2019. 6, 7, 1 

- [68] Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In _CVPR_ , 2019. 2 

- [69] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. OPT: Open pre-trained transformer language models. _arXiv preprint arXiv:2205.01068_ , 2022. 2 

- [70] Yaqi Zhang, Di Huang, Bin Liu, Shixiang Tang, Yan Lu, Lu Chen, Lei Bai, Qi Chu, Nenghai Yu, and Wanli Ouyang. MotionGPT: Finetuned LLMs are general-purpose motion generators. In _AAAI_ , 2024. 2 

11 

## **Situational Awareness Matters in 3D Vision Language Reasoning** Supplementary Material 

## **A. Implementation Details** 

Here we provide more details about our model. 

**Visual and Textual Encoders.** We use OpenScene [51] (the 3D distilled variant) as our visual encoder, which incorporates distilled CLIP features into a 3D Minkowski CNN backbone originally designed for the 3D semantic segmentation task. We use the default 0.02m voxel size to discretize a point cloud into 3D voxels and disable the scaling and elastic distortion augmentation methods during the voxelization process. The 3D architecture is the predefined _MinkUNet18A_ [12]. The number of visual tokens _Nv_ is 256. Additionally, we use the Sentence-BERT [55] MPNet variant as our text tokenizer and encoder. We use the fixed batch padding strategy and set the length to be 100 for both situational tokens _Ns_ and question tokens _Nq_ . The feature embedding sizes for all three types of tokens are set to 768. The 256 dimensional output of the OpenScene backbone is projected to the 768 hidden size with a 1x1 Conv layer. We freeze the OpenScene backbone, and finetune only the last layer of the textual backbone during our training process. 

**Fusion and Decoder Models.** We use a 4-layer MCAN Transformer [59, 67] as the fusion block of the visual and situational tokens. The learnable positional embeddings and situational embeddings are composed of a 2-layer MLP: first from dimension 3 to 128, and then from 128 to the target dimension. We use BLIP-2 [38] as the large multimodal tranformer for final response generation, simiar to 3D-LLM [25]. 

**Training Details.** We train our model with the AdamW [43] optimizer with _β_ 1 = 0 _._ 9, _β_ 2 = 0 _._ 999, and _ϵ_ = 1e-8. We use a batch size of 16 and set the initial learning rate to be 2e-5. The weight decay is set to 0.05, and we disable the weight decay on the layernorm layers and all bias parameters following [40]. We decrease the learning rate by 10 times after the 10-th and the 20-th epochs. We train the model for a total of 50 epochs on a single NVIDIA A100 GPU. 

## **B. Enhanced Vision Token Activation Through Situational Re-encoding** 

In Figure B, we provide an insightful visualization of the activation changes in 3D visual tokens _z_[3D] , before and after undergoing our situation-guided visual re-encoding process. This visualization employs the _viridis_ colormap, where a brighter token representation indicates a higher activation value. The effectiveness of situational guidance in amplifying the relevance of crucial tokens is evident from 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning_images/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning.pdf-0012-08.png)


**----- Start of picture text -----**<br>
Situation : I am facing… behind me. Situation : I am … picture… the other side.<br>Question : Is the table round, square, or rectangular? Question : Is the picture on my left on the shelf or wall?<br>SQA3D : rectangular. SQA3D : wall.<br>SIG3D : rectangular. SIG3D : shelf.<br>[Easy] :  Only one table in the scene. [Hard] :  Multiple pictures, shelves in the room.<br>**----- End of picture text -----**<br>


Figure A. Visualization of cases where situation prediction benefits the most. Arrow colors: GT, SIG3D, and SQA3D predictions. 

this depiction. 

For example, the visualization in the second row reveals a notable shift in focus. Initially, the tokens predominantly concentrate on the bed area. However, after re-encoding, there is a discernible shift in attention towards areas closely aligned with the situational vector and those directly related to the query. Similarly, in the third row, the situational reencoding process results in the window region “on the left” receiving increased emphasis. In the fourth row, the attention initially focuses on the vanity region. Then it shifts to the toilet on the left of the agent, as suggested by the situational vector and the question prompt. This experiment provides a clear demonstration of how our method, using enhanced situational awareness, contributes to improved performance in downstream reasoning tasks in an explainable manner. The ability of our model to dynamically adjust focus in response to situational cues is a key factor in its enhanced reasoning capabilities. 

## **C. More Qualitative Results** 

We show more qualitative results of our model in Figures C and D. Visualization encompasses a diverse array of tasks, including queries about object orientation, characteristics of specific objects, the count of objects within a scene, and yes/no questions based on commonsense reasoning. A key observation from these results is that, in numerous instances, absolute precision in situation estimation is not a prerequisite for our model to accurately deduce the answers to the posed questions. This finding highlights the model’s robustness and its capacity to effectively handle a variety of query types, even with less optimal situational awareness. 

## **D. Performance on Hard Cases** 

An example of our case study of easy-hard samples is shown in Figure A. We find that simple examples in the dataset allow existing models to guess the correct answer without any 3D situational understanding. However, our method effectively improves the hard examples with com- 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning_images/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning.pdf-0013-00.png)


**----- Start of picture text -----**<br>
3D Scene for Reference Before  Situational Re-encoding After  Situational Re-encoding<br>Situational Description:  I am sitting on a couch with a<br>pillow facing another couch and the pillow is on my right. Question : How many couches are behind me? Answer : One<br>Situational Description:  I am walking into the room with<br>the bathroom vanity on my right. Question : What is hung on the nook to my right side? Answer : Ironing board<br>Situational Description:  I am looking at my backpack<br>and desk is behind me. Question : How many windows to my left? Answer : One<br>Situational Description:  I am facing a bathroom vanity,<br>while having a ladder on my right within reach.  Question : Is the toilet seat up or down to my left? Answer : Up<br>**----- End of picture text -----**<br>


Figure B. **3D vision token activation before and after situational re-encoding** . We can notice that higher weights are assigned to question and situation-related tokens after our proposed situational re-encoding mechanism. 

plicated and entangled questions and situations. 

## **E. Analysis of Failure Cases** 

We perform a failure case analysis on our model in Figure E. We categorize and visualize three types of failure cases. 

## **Accurate Situation Estimation, Incorrect Question An-** 

**swering.** This scenario demonstrates that accurate situational understanding does not necessarily guarantee correct responses to queries. A significant proportion of failures within this category can be attributed to complex question prompts that demand multi-stage reasoning or the integration of commonsense knowledge. For instance, the initial 

example necessitates the model’s comprehension of the spatial relationship between the viewer’s perspective and the couch, followed by an additional reasoning phase focused on the couch to accurately respond to the query. The subsequent example demands an understanding of the concepts of “odd” and “even,” and their application to the count of objects in a 3D environment. 

## **Inaccurate Situation Estimation, Correct Question An-** 

**swering.** This category reveals that errors in situation estimation are more likely when the scene description involves minor or less common objects. Furthermore, it is observed that the model might incidentally arrive at the correct answer without fully grasping the complex situational 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning_images/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning.pdf-0014-00.png)


**----- Start of picture text -----**<br>
Situational Description:  I am sitting  Situational Description:  I just walked Situational Description:  I am standing  Situational Description:  I am picking<br>on the left cushion of the couch and  into the room through the doors. in front of the door and facing the file  up my jacket on the chair while facing<br>to my right is a pillow. cabinet. the blackboard and there is a desk with<br>no monitor in my six o’clock direction.<br>Question : Which direction should I go  Question : How many armchairs are  Question : Is the door behind me open  Question : What color is the desk<br>if I want to open the curtain? directly in front of me? or closed? behind me?<br>Answer : Left Answer : Zero Answer : Closed Answer : Brown<br>Situational Description : I am fixing  Situational Description:  I am taking out  Situational Description:  I am facing a  Situational Description:  I am looking<br>the cabinet with a few bags by my  the white ball from the side pocket while  chair and there is a printer on my right. at mirror and combing my hair with a<br>right foot. my beer on the table is being drunk by  hairbrush that was in the bathroom<br>me. I am also facing a white board. vanity.<br>Question : If I want to take a break and  Question : Can I see chair if I turn  Question : If I turned directly around and  Question : Is the door to my right open<br>take a nap, is there a bed I could sleep  around? walked straight, what would I hit first? or closed?<br>on?<br>Answer : Yes Answer : Right Answer : Bed Answer : Wall<br>Situational Description : I am  Situational Description:  I am opening  Situational Description:  I am sitting  Situational Description:  I am turning<br>standing in the middle of the kitchen  the door. on the toilet facing a bathtub. on the lamp by the chair with the chair<br>and the stove is on my left. on my right within reach.<br>Question : What is to my left that can  Question : Can I see a window if I turn  Question : Does the bathtub in front of me  Question : If I look to my right, can I see<br>be used to cook food? my head rightwards? have a shower curtain? my reflection?<br>Answer : Stove Answer : Yes Answer : No Answer : Wall<br>Situational Description : I am Situational Description:  I am sitting on  Situational Description:  I am facing  Situational Description:  I am opening<br>leaning on the door facing the blinds. the toilet facing a bathtub. the shelf closest to the armchair. the door.<br>Question : Where should I walk if I  Question : Can I reach the sink from  Question : What color is the ledge in front  Question : Are there pillows on the<br>want to find a book to read during my  where I am sitting? of me? couches?<br>break?<br>Answer : Forward Answer : Right Answer : White Answer : Wall<br>**----- End of picture text -----**<br>


Figure C. We demonstrate more successful examples of our method. 

3 

**Situational Description:** I am facing a **Situational Description:** I am picking table, while there is a chair on my up my backpack with a chair to my right and two chairs behind me. left within reach. **Question** : Is the room clean or messy? **Question** : How many chairs are behind me? **Answer** : Messy **Answer** : Two 

**Situational Description:** I am facing the whiteboard, and there are some chairs and a table behind me. 

**Situational Description** : I am standing by the paper cutter on my right within reach. 

**Situational Description:** I am looking for a cold drink to have and there is a cabinet on my right side. **Question** : What is on top of the cabinet that is on my 1 o’clock? **Answer** : TV **Situational Description:** I am throwing trash with a chair very close to me behind me. 

**Situational Description:** I am facing the toilet, and the sink is behind me. **Question** : Is the amount of bar I am facing odd or even? **Answer** : Odd 

**Situational Description:** I am standing right in front of the sink, and I can see table across the room. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning_images/cvpr_2024_Situational_Awareness_Matters_in_3D_Vision_Language_Reasoning.pdf-0015-06.png)


**----- Start of picture text -----**<br>
Question : Is the number of chairs on  Question : What is to my left that you  Question : Is the shape of table behind me  Question : Can I see the TV without<br>my left odd or even? can use to see the outside? round, square, or rectangular? moving much?<br>Answer : Even Answer : Windows Answer : Rectangular Answer : No<br>Situational Description : I am facing  Situational Description:  I am opening  Situational Description:  I am  Situational Description:  I just entered<br>the wall with large windows, and  one door while there is another one  standing in front of a counter, and  the room, while having a desk with a<br>there is a trashcan behind me. adjacent on my right. there is a recycling bin to my left  shelf on top of it at my right and a door<br>within reach. behind me.<br>Question : How many chairs are  Question : Is the door in front of me  Question : Which direction should I go  Question : Where do I go to get to the<br>accounted for in the room total? closed or open? if I want to open the mailbox? other bed?<br>Answer : Eight Answer : Closed Answer : Right Answer : Right<br>Situational Description : I am placing  Situational Description:  I am facing the  Situational Description:  I am opening  Situational Description:  I am looking<br>garbage into a trash can, while  toilet, and the sink is behind me. the door. at mirror and combing my hair with a<br>having a toilet to my left within reach. hairbrush that was in the bathroom<br>vanity.<br>Question : What is the color of the  Question : Which direction should I go  Question : Does the toilet to my right have  Question : What color is the sink in front<br>towel behind me? if I want to exit the room? an open or closed lid? of me?<br>Answer : White Answer : Right Answer : Open Answer : White<br>**----- End of picture text -----**<br>


Figure D. We demonstrate more successful examples of our method. 

4 

(a) Situational estimation is correct, (b) Situational estimation is wrong, (c) Situational estimation is wrong, and QA is wrong. but QA is correct. and QA is wrong. **Situational Description** : After an **Situational Description:** I am standing **Situational Description:** I am standing exhausted day, I am lying on the bed in front of the door and facing the file by backpack on my left side and the bed closest to the window with my head cabinet. is behind me while the curtain is in my on the pillows. six o’clock direction. **Question** : What is to the right of the **Question** : Is the door behind me open **Question** : Are there more doors to my couch in front of me? or closed? left than there are to my right? **GT Answer** : Desk **GT Answer** : Closed **GT Answer** : No **Pred Answer** : End table **Pred Answer** : Closed **Pred Answer** : Yes **Situational Description** : Standing by **Situational Description:** I am facing a **Situational Description:** I am sitting on the table with a tray rack to my vending machine that is next to a chair facing the table with the immediate right. another one, while there is a trash can blackboard behind me and a chair on behind me directly. my left within reach. **Question** : Is the amount of chair on my left side odd of even? **Question** : Can I see clothes where I **Question** : Is the amount of table I am am standing? facing odd or even? **GT Answer** : Odd **GT Answer** : Yes **GT Answer** : Odd **Pred Answer** : Even **Pred Answer** : Yes **Pred Answer** : Even 

Figure E. We demonstrate three different categories of failure cases. 

and multi-modal context, particularly in cases where the question involves choosing between two or among multiple given options. Therefore, a blend of qualitative and quantitative assessments is crucial for a comprehensive evaluation of the model’s performance. 

**Both Situation Estimation and Question Answering are Incorrect.** This group contains the most challenging examples from the dataset, typically encompassing multiple complexities identified in the preceding categories. These cases present a compounded difficulty level, highlighting the model’s limitations in scenarios that require an intricate understanding of both situational context and question in- 

terpretation. 

## **F. Limitations and Future Work** 

**Selection of 3D Scenes.** The SQA3D [44] and ScanQA [5] datasets, both derived from the ScanNet [13] dataset, exclusively feature indoor household environments. These static scenes limit the model’s applicability to dynamic tasks like manipulation and exploration. Consequently, our current model is tailored to static household settings. This scalability problem is a long-standing challenge for all existing 3D VL reasoning work [5, 25, 44, 74]. We believe that with a more scalable visual representation 

5 

( _e.g_ ., scene graphs, sparse learnable embeddings), we can extend our model to support larger 3D environments in the future work. 

**More Comprehensive Visual Encoding.** In our approach, the utilization of a voxel-based, open-vocabulary 3D encoder achieves much better overall performance. Nevertheless, for specific queries involving counting or referencing, a detection-based encoder may yield a more advantageous visual token set, owing to its capacity to provide instance-level information pertinent to the questions. This indicates the potential benefits of a multifaceted visual tokenization system that amalgamates the strengths of various encoder types. 

6 

