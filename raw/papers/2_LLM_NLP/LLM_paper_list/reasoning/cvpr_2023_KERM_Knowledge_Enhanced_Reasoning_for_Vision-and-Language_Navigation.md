## **KERM: Knowledge Enhanced Reasoning for Vision-and-Language Navigation** 

Xiangyang Li[1,2] , Zihan Wang[1,2] , Jiahao Yang[1,2] , Yaowei Wang[3] , Shuqiang Jiang[1,2,3] 1Key Lab of Intelligent Information Processing Laboratory of the Chinese Academy of Sciences (CAS), Institute of Computing Technology, Beijing, 100190, China 

2University of Chinese Academy of Sciences, Beijing, 100049, China 

3Peng Cheng Laboratory, Shenzhen, 518055, China 

lixiangyang@ict.ac.cn, _{_ zihan.wang, jiahao.yang _}_ @vipl.ict.ac.cn, wangyw@pcl.ac.cn, sqjiang@ict.ac.cn 

## **Abstract** 

_Vision-and-language navigation (VLN) is the task to enable an embodied agent to navigate to a remote location following the natural language instruction in real scenes. Most of the previous approaches utilize the entire features or object-centric features to represent navigable candidates. However, these representations are not efficient enough for an agent to perform actions to arrive the target location. As knowledge provides crucial information which is complementary to visible content, in this paper, we propose a Knowledge Enhanced Reasoning Model (KERM) to leverage knowledge to improve agent navigation ability. Specifically, we first retrieve facts (i.e., knowledge described by language descriptions) for the navigation views based on local regions from the constructed knowledge base. The retrieved facts range from properties of a single object (e.g., color, shape) to relationships between objects (e.g., action, spatial position), providing crucial information for VLN. We further present the KERM which contains the purification, fact-aware interaction, and instruction-guided aggregation modules to integrate visual, history, instruction, and fact features. The proposed KERM can automatically select and gather crucial and relevant cues, obtaining more accurate action prediction. Experimental results on the REVERIE, R2R, and SOON datasets demonstrate the effectiveness of the proposed method. The source code is available at https://github.com/XiangyangLi20/KERM ._ 

## **1. Introduction** 

Vision-and-language navigation (VLN) [3, 12, 23, 24, 36, 38] is one of the most attractive embodied AI tasks, where agents should be able to understand natural language instructions, perceive visual content in dynamic 3D environments, and perform actions to navigate to the target location. Most previous methods [9,15,22,31,34] depend on se- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation_images/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation.pdf-0001-10.png)


**----- Start of picture text -----**<br>
Go up the stairs through the outdoor living area and …<br>Instruction the kitchen … between the kitchen island and pantry.<br>Panoramic View<br>✔<br>Navigable Navigable Navigable<br>Candidate 1 Candidate 2 Candidate 3<br><lamp to right of sofa> <workspace kitchen island><br>Knowledge<br>database <white wall partition> <contemporary table><br><sofa next to wall> <switch above sofa> …<br>✔<br>Retrieved Facts<br><sofa next to wall> <light hanging over  <ceiling recessed light><br><lamp to right of sofa> kitchen island> <wood countertop><br><a living room area> <laminate countertop> <contemporary table><br>… … …<br>**----- End of picture text -----**<br>


Figure 1. Illustration of knowledge related navigable candidates, which provides crucial information such as attributes and relationships between objects for VLN. Best viewed in color. 

quential models ( _e.g.,_ LSTMs and Transformers) to continuously receive visual observations and align them with the instructions to predict actions at each step. More recently, transformer-based architectures [5, 7, 25] which make use of language instructions, current observations, and historical information have been widely used. 

Most of the previous approaches utilize the entire features [5, 12, 13, 25] or object-centric features [1, 7, 10, 20] to represent navigable candidates. For example, Qi _et al._ [22] and Gao _et al._ [10] encode discrete images within each panorama with detected objects. Moudgil _et al._ [20] utilize both object-level and scene-level features to represent visual observations. However, these representations are not efficient enough for an agent to navigate to the target location. For example, as shown in Figure 1, there are three candi- 

dates. According to the instruction and the current location, candidate2 is the correct navigation. Based on the entire features of a candidate view, it is hard to select the correct one, as candidate2 and candidate3 belong to the same category ( _i.e.,_ “dining room”). Meanwhile, it is also hard to differentiate them from individual objects, as “lamp” and “light” are the common components for them. 

As humans make inferences under their knowledge [11], it is important to incorporate knowledge related to navigable candidates for VLN tasks. First, knowledge provides crucial information which is complementary to visible content. In addition to visual information, high-level abstraction of the objects and relationships contained by knowledge provides essential information. Such information is indispensable to align the visual objects in the view image with the concepts mentioned in the instruction. As shown in Figure 1, with the knowledge related to candidate2 ( _i.e., <_ light hanging over kitchen island _>_ ), the agent is able to navigate to the target location. Second, the knowledge improves the generalization ability of the agent. As the alignment between the instruction and the navigable candidate is learned in limited-seen environments, leveraging knowledge benefits the alignment in the unseen environment, as there is no specific regularity for target object arrangement. Third, knowledge increases the capability of VLN models. As rich conceptual information is injected into VLN models, the correlations among numerous concepts are learned. The learned correlations are able to benefit visual and language alignment, especially for tasks with high-level instructions. 

In this work, we incorporate knowledge into the VLN task. To obtain knowledge for view images, facts ( _i.e.,_ knowledge described by language descriptions) are retrieved from the knowledge base constructed on the Visual Genome dataset [16]. The retrieved facts by CLIP [26] provide rich and complementary information for visual view images. And then, a knowledge enhanced reasoning model (KERM) which leverages knowledge for sufficient interaction and better alignment between vision and language information is proposed. Especially, the proposed KERM consists of a purification module, a fact-aware interaction module, and an instruction-guided aggregation module. The purification model aims to extract key information in the fact representations, the visual region representations, and the historical representations respectively guided by the instruction. The fact-aware interaction module allows visual and historical representations to obtain the interaction of the facts with cross-attention encoders. And the instruction-guided aggregation module extracts the most relevant components of the visual and historical representations according to the instruction for fusion. 

We conduct the experiments on three VLN datasets, _i.e.,_ the REVERIE [24], SOON [36], and R2R [3]. Our approach outperforms state-of-the-art methods on all splits of 

these datasets under most metrics. The further experimental analysis demonstrates the effectiveness of our method. In summary, we make the following contributions: 

- We incorporate region-centric knowledge to comprehensively depict navigation views in VLN tasks. For each navigable candidate, the retrieved facts ( _i.e.,_ knowledge described by language descriptions) are complementary to visible content. 

- We propose the knowledge enhanced reasoning model (KERM) to inject fact features into the visual representations of navigation views for better action prediction. 

- We conduct extensive experiments to validate the effectiveness of our method and show that it outperforms existing methods with a better generalization ability. 

## **2. Related Work** 

**Vision-and-Language Navigation.** VLN [3, 6, 7, 14, 25, 31, 33, 34] has drawn significant research interests because of its practicality for potential applications ( _e.g.,_ healthcare robots and personal assistants). Given natural language instructions ( _e.g.,_ step-by-step instructions [3, 17], high-level instructions [24, 36], and dialog-based instructions [32]), VLN tasks require the agent to navigate to the goal location. Early methods [9, 34, 37] usually utilize recurrent neural networks (RNNs) to encode historical observations and actions, which are represented as a state vector. To make further advances, numerous approaches focus on incorporating more visual information. For example, Hong _et al._ [14] propose to learn the relationships among the scene, objects, and directional clues. In order to capture environment layouts, Wang _et al._ [33] employ topological maps to memorize the percepts during navigation, supporting long-term planning. 

Inspired by the big success of vision-and-language pretraining (VLP) [26, 30], more recent approaches utilize transformer-based architectures. Hao _et al._ [13] propose PREVALENT which pretrains the navigation model under the self-learning paradigm. To learn general navigationoriented textual representations, AirBERT [12] and HM3DAutoVLN [6] create large-scale VLN data to improve the interaction between different modalities. VLNBERT [15] injects a recurrent function into cross-modal structures to maintain time-dependent state information. HAMT [5] concatenates language instructions, historical and current observations, and then feed them into a cross-modal transformer to derive multi-modal action prediction. DUET [7] combines local observations and the global topological map via graph transformers, facilitating action planning and cross-modal understanding. In this work, we leverage mined knowledge ( _i.e.,_ retrieved facts) of viewpoint images to improve the generalization ability and facilitate the alignment between vision and language for VLN. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation_images/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Action Prediction<br>Instruction Fine-scale<br>Encoder<br>(a) Panoramic  Topological Turn left and walk out …<br>View Map the door to the right of<br>the TV … the closet. Coarse-scale  Candidate 1 Candidate 2 Candidate K<br>Encoder<br>Purification Fact-aware Interaction Instruction-guided Aggregation<br>Panoramic<br>View Update [Turn][left][and] [the][closet] Knowledge-enhanced<br>1 … History Features<br>Topological 2 …<br>Map 3 … View<br>Representations<br><Cabinet surrounds  N …<br>television> Instruction-<br>(b) <Console table next to television> … Row-Wise Attentionguided<br>Max-pooling<br>Retrieved Facts<br>… Score<br>1 2 3 … N<br>Single View … Knowledge-enhancedVisual Features<br>Cropped Image<br>…<br>…<br>…<br>…<br>… … Interaction History-Fact …<br>… …<br>…<br>…<br>… Interaction Visual-Fact<br>…<br>**----- End of picture text -----**<br>


Figure 2. The overall pipeline. (a) The baseline method uses a dual-scale graph transformer to encode the panoramic view, the topological map, and the instruction for action prediction. (b) Our approach incorporates retrieved facts as additional input. The knowledge enhanced representations of each candidate view are obtained with the purification module, the fact-aware interaction module, and the instructionguided aggregation module. Best viewed in color. 

**Vision-Language Reasoning with External Knowledge.** 

There has been growing attention to incorporating external knowledge for multi-modal understanding and reasoning [10, 21, 28]. ConceptNet [29] and DBpedia [4] are widely used knowledge bases where concepts and relationships are represented with nodes and edges respectively. The structured knowledge can be represented with graph neural networks or graph transformers, enabling interaction within visual and linguistic features. Based on acquired knowledge, KE-GAN [21] utilizes the knowledge relation loss to re-optimize parsing results. CKR [10] leverages relationships among concepts according to ConceptNet to learn correlations among rooms and object entities. Another line of work employs unstructured knowledge which is retrieved from knowledge bases such as Wiktionary [19]. For example, K-LITE [28] enriches natural language supervision with knowledge descriptions, thus learning image representations that can understand both visual concepts and their knowledge. In this work, we leverage the knowledge that provides complementary and necessary information for the view images to benefit VLN tasks. 

## **3. Method** 

We address VLN in discrete environments [3, 24, 36]. The environment is provided with a navigation connectivity graph _G,_ = _{V, E}_ , where _V_ denotes navigable nodes 

and _E_ denotes edges. Given natural language instructions, the agent needs to explore the environment to reach the target locations. _L_[ˆ] = _{l_[ˆ] _i}[M] i_ =1[denotes the word embeddings of] the instruction containing _M_ words. At step _t_ , the agent observes a panoramic view of its current node _Vt_ . The panoramic view contains 36 single views _St_ = _{si}_[36] _i_ =1[,] and each single view is represented by a vector _si_ accompanied with its orientation. The navigable views of _Vt_ are a subset of _St_ . For tasks with object grounding [24, 36], _N_ object features _Ot_ = _{oi}[N] i_ =1[for a panorama are extracted] with annotated bounding boxes or object detectors [2]. 

## **3.1. Overview of Our Approach** 

The baseline follows the architecture of DUET [7], as illustrated in Figure 2(a). A topological map is constructed over time by adding newly observed information to the map and updating visual representations of its nodes. Specifically, _Gt_ = _{Vt, Et}_ denotes the topological map at step _t_ , _Gt ∈G_ is the map of the environment after _t_ navigation steps. As shown in Figure 2(a), there are three kinds of nodes in _Vt_ : visited nodes ( _i.e.,_ nodes with solid black border), the current node ( _i.e.,_ the node with solid red border), and navigable nodes ( _i.e.,_ nodes with dashed black border). Then at each step, all the features of the 36 single views for the current location, the topological map, and the instruction are fed into the dual-scale graph transformer to predict the next location in the map or the stop action. 

Our model which leverages knowledge to obtain contextualized representation is illustrated in Figure 2(b). First, facts that provide complementary information to the visual features of each discrete view are retrieved. Then, the instruction, the topological map, the visual view features, and the corresponding facts are fed into the purification module, the fact-aware interaction module, and the instructionguided aggregation module gradually to form knowledge enhanced representations. At last, the dual-scale graph transformer [7] is utilized to predict the action. 

## **3.2. Fact Acquisition** 

To navigate to the target location, high-level abstraction of the objects and relationships by natural language provides essential information which is complementary to visual features. Such information is indispensable to model the relevance between the visual objects and concepts mentioned in instructions, facilitating the matching between candidate views and instructions. We first construct a knowledge base and then use the pretrained multi-modal model CLIP [26] to retrieve facts for each view. 

**Knowledge Base Construction.** The knowledge base is the source of relevant facts that describe the visual region. In order to obtain the rich and varied descriptions, similar to [18], we parse the region descriptions from the Visual Genome [16] dataset to construct the knowledge base. Specifically, in the Visual Genome dataset, the parsed attribute annotations take the form of “attribute-object” pairs and the parsed relationship annotations take the form of “subject-predicate-object” triplets. We convert all of the “attribute-object” pairs and “subject-predicate-object” triplets to their synset canonical forms. After removing duplicates, we totally get 630K facts expressed by language descriptions that are used to build our knowledge base. 

**Fact Retrieval.** Our main goal is to obtain facts for the view images when navigating in the environment. To this end, we crop each view image into five sub-regions (as shown in Figure 3) and retrieve facts for these sub-regions from the knowledge base. In order to retrieve relevant facts for the visual sub-region, we use the pretrained model CLIP [26]. It consists of an image encoder CLIP-I and a text encoder CLIP-T that encode image and text into a joint embedding space. We use the CLIP-T to encode all facts in the knowledge base as the search keys. The visual sub-regions are encoded by the CLIP-I as the queries. We then search in the knowledge base for the facts with the top-k highest cosine similarity scores. For each sub-region, we keep five facts with the top-5 highest cosine scores as the knowledge. Some examples of the top-5 results are shown in Figure 3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation_images/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation.pdf-0004-05.png)


**----- Start of picture text -----**<br>
<cabinet surrounds television> <shelf are filled with glassware><br><console table by television> <displayed liquor><br><on built-in shelf television> <bottle sitting barroom><br><console table next to television> <glassware on shelf><br><drawer on entertainment center> <barroom in display><br>**----- End of picture text -----**<br>


Figure 3. Five cropped sub-regions of the view image and the retrieved top-5 facts for the sub-region in the blue box. 

## **3.3. Knowledge Enhanced Reasoning** 

At each step _t_ , the visual features, the history features, the instruction features, and the fact features are fed into our proposed knowledge enhanced reasoning model (KERM) to support better action prediction, as shown in Figure 2(b). Specifically, for each single view _n_ , the visual features _Rn_ = _{ri}_[5] _i_ =1[are][composed][of][5][sub-regions][where] each sub-region _ri_ is encoded by CLIP-I. The fact features _En_ = _{ei}_[25] _i_ =1[are the total fact representations, where each] fact _ei_ is encoded by CLIP-T. To represent historical information and the layout of the explored environment, the history features _Hn_ which are composed of all the node representations of the topological map _Gt_ are also employed. As the same in [7], word embeddings _L_[ˆ] are fed into a multilayer transformer to obtain the instruction features _L_ . 

## **3.3.1 Purification Module** 

Due to the complexity of the navigation environment, a large number of fact features within each view are not all needed by the agent to complete the current navigation task. The agent needs more critical, highly correlated knowledge with the current navigation instruction to understand and characterize the environment image. Therefore, we propose an instruction-aware fact purification module that aims to filter out less relevant information and capture critical information that is closely related to the current navigation task. Specifically, we first evaluate the relevance of each fact in the view to each token of navigation instruction by computing the fact-instruction relevance matrix _A_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation_images/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation.pdf-0004-11.png)


where _W_ 1 and _W_ 2 are learned parameters and _d_ is the dimension of features. After this, we compute the row-wise max-pooling on _A_ to evaluate the relevance of each fact to the instruction as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation_images/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation.pdf-0004-13.png)


The fact features are purified by the guidance of instructionaware attention as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation_images/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation.pdf-0005-01.png)


turesThe _Hn_ purified _′_[are obtained in the same way.] visual region features _Rn′_[and][history][fea-] 

## **3.3.2 Fact-aware Interaction Module** 

After obtaining the purified features, we use a multi-layer cross-modal transformer for vision-fact interaction. Each transformer layer consists of a cross-attention layer and a self-attention layer. The cross-attention between visual features and fact features is calculated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation_images/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation.pdf-0005-05.png)


And then,the knowledge _R_[˜] _n_ are fed into the self-attention layer to obtainenhanced visual features _Rn′′_[.][The][history-] fact interaction is also conducted in this same way with anotherknowledge enhanced history featuresmulti-layer cross-modal transformer, _Hn′′_[.] forming the 

## **3.3.3 Instruction-guided Aggregation Module** 

At last, we use the instruction-guided attention to aggregate the vision-fact features into a contextual vector ¯ _rn_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation_images/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation.pdf-0005-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation_images/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation.pdf-0005-10.png)


where _W_ 3 and _W_ 4 are learned parameters and _d_ is the dimension of features, _K_ is the number of sub-regions ( _i.e., K_ =5), _L_ 0 are the feature from the [CLS] token representing the entire instruction. With the same mechanism, we aggregate the history-fact features into _h_[¯] _n_ . Then we use an FFN to fuse _r_ ¯ _n_ and _h_[¯] _n_ into the knowledge enhanced representations of view _n_ , which are fed into the dual-scale graph transformer [7] for action prediction. 

## **3.4. Training and Inference** 

**Pretraining.** As demonstrated in [5, 7, 25], it is beneficial to pretrain transformer-based VLN models. We utilize the following four tasks to pretrain our KERM model. 

1) Masked language modeling (MLM). We randomly mask out the words of the instruction with a probability of 15%, and then the masked words _Lm_ are predicted with the corresponding contextual features. 

2) Masked view classification (MVC). The MVC requires the model to predict the semantic labels of masked view images. We randomly mask out view images with a 

probability of 15%. Similar to [7], the target labels for view images are obtained by an image classification model [8] pretrained on ImageNet. 

3) Single-step action prediction (SAP). Given a demonstration path _P[∗]_ and its partial _P<t[∗]_[, the] _[ SAP]_[loss in behav-] ior cloning is defined as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation_images/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation.pdf-0005-18.png)


where _at_ is the expert action of _P<t[∗]_[.] 

4) Object grounding (OG). OG is used when object annotations are available: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation_images/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation.pdf-0005-21.png)


where _o[∗]_ is the ground-truth object at the destination location _PD_ . More details are presented in the supplementary materials. 

**Fine-tuning and Inference.** For fine-tuning, we use the imitation learning method [27] and the SAP loss _LSAP_ . Different from the pretrain process using the demonstration path, the supervision of fine-tuning is from a pseudointeractive demonstrator. It selects a navigable node as the next target node with the overall shortest distance from the current node to the destination. For tasks with object annotations, we also use the object grounding loss _LOG_ . 

For inference, the model predicts an action at each step. If the action is not the stop action, the agent performs this action and moves to the predicted node, otherwise, the agent stops at the current node. The agent will be forced to stop if it exceeds the maximum action steps and then returns to the node with maximum stop probability as its final prediction. At the predicted location, the object with the highest prediction score is selected as the target. 

## **4. Experiment** 

## **4.1. Datasets and Evaluation Metrics** 

**Datasets.** We evaluate our model on the REVERIE [24], SOON [36], and R2R [3] datasets. REVERIE contains high-level instructions and the instructions contain 21 words on average. The predefined object bounding boxes are provided for each panorama, and the agent should select the correct object bounding box from candidates at the end of the navigation path. The path length is between 4 and 7 steps. SOON also provides instructions that describe the target locations and target objects. The average length of instructions is 47 words. However, the object bounding boxes are not provided, while the agent needs to predict the center location of the target object. Similar to the settings in [7], we use object detectors [2] to obtain candidate object boxes. The path length of SOON is between 2 and 21 steps. R2R 

Table 1. Comparison with state-of-the-art methods on the REVERIE dataset. 

|Methods|Val Seen|Val Seen|Val Unseen|Val Unseen|Test Unseen|Test Unseen|
|---|---|---|---|---|---|---|
||Navigation|Grounding|Navigation|Grounding|Navigation|Grounding|
||TL↓OSR↑SR↑SPL↑|RGS↑RGSPL↑|TL↓OSR↑SR↑SPL↑|RGS↑RGSPL↑|TL↓OSR↑SR↑SPL↑|RGS↑RGSPL↑|
|Seq2Seq [3]|12.88 35.70 29.59 24.01|18.97<br>14.96|11.07 8.07 4.20 2.84|2.16<br>1.63|10.89 6.88 3.99 3.09|2.00<br>1.58|
|RCM[34]|10.70 29.44 23.33 21.82|13.23<br>15.36|11.98 14.23 9.29 6.97|4.89<br>3.89|10.60 11.68 7.84 6.67|3.67<br>3.14|
|VLNBERT[15]|13.44 53.90 51.79 47.96|38.23<br>35.61|16.78 35.02 30.67 24.90|18.77<br>15.27|15.68 32.91 29.61 23.99|16.50<br>13.51|
|AirBERT[12]|15.16 49.98 47.01 42.34|32.75<br>30.01|18.71 34.51 27.89 21.88|18.23<br>14.18|17.91 34.20 30.28 23.61|16.83<br>13.28|
|HOP[25]|13.80 54.88 53.76 47.19|38.65<br>33.85|16.46 36.24 31.78 26.11|18.85<br>15.73|16.38 33.06 30.17 24.34|17.69<br>14.34|
|HAMT[5]|12.79 47.65 43.29 40.19|27.20<br>15.18|14.08 36.84 32.95 30.20|18.92<br>17.28|13.62 33.41 30.40 26.67|14.88<br>13.08|
|DUET[7]|13.86 73.68 71.75 63.94|57.41<br>51.14|22.11 51.07 46.98 33.73|32.15<br>23.03|21.30 56.91 52.51 36.06|31.88<br>22.06|
|KERM-pt(Ours)|14.25**74.49 71.89 64.04**|**57.55**<br>**51.22**|22.47**53.65 49.02 34.83**|**33.97**<br>**24.14**|18.38**57.44**52.26**37.46**|**32.69**<br>**23.15**|
|KERM(Ours)|12.84**79.20 76.88 70.45**|**61.00**<br>**56.07**|21.85**55.21 50.44 35.38**|**34.51**<br>**24.45**|17.32**57.58 52.43 39.21**|**32.39**<br>**23.64**|



provides step-by-step instructions and is not required to predict object locations. The average length of instructions is 32 words and the average length of paths is 6 steps. 

**Evaluation Metrics.** We utilize the standard evaluation metrics [3, 24] for VLN tasks to compare our method with previous approaches, including (1) Trajectory Length (TL): the agent’s average path length in meters; (2) Navigation Error (NE): average distance in meters between the agent’s final location and the target one; (3) Success Rate (SR): the proportion of successfully executed instructions with the NE less than 3 meters; (4) Oracle SR (OSR): SR given the oracle stop policy; (5) SPL: SR penalized by Path Length. For the REVERIE and SOON datasets that require object grounding, we adopt metrics including (1) Remote Grounding Success (RGS): proportion of successfully executed instructions; (2) RGSPL: RGS penalized by Path Length. Except for TL and NE, all metrics are higher the better. 

## **4.2. Implementation Details** 

## **4.2.1 Model Architectures** 

We adopt the pretrained CLIP-ViT-B/16 [26] to retrieve facts for each view. For the fact-aware interaction module, the number of layers for the cross-modal transformer is set as 2, with a hidden size of 768. The parameters of this module are initialized with the pretrained LXMERT [30]. 

We use the ViT-B/16 [8] pretrained on ImageNet to extract object features on the REVERIE dataset as it provides bounding boxes. The BUTD object detector [2] is utilized on the SOON dataset to extract object bounding boxes. For the dual-scale graph transformer [7], the number of layers for the language encoder, panorama encoder, coarse-scale cross-modal encoder, and fine-scale cross-modal encoder are set as 9,2,4 and 4, respectively. And the parameters are also initialized with the pretrained LXMERT [30]. 

## **4.2.2 Training Details** 

For pretraining, we set the batch size as 16 using 4 NVIDIA RTX3090 GPUs. For the REVERIE dataset, we combine the original dataset with augmented data synthesized by DUET [7] to pretrain our model with 100k iterations. Then we fine-tune the pretrained model with the batch size of 2 for 20k iterations on 4 NVIDIA RTX3090 GPUs. For the SOON dataset, we only use the original data with automatically cleaned object bounding boxes, sharing the same settings in DUET [7]. We pretrain the model with 40k iterations. And then we fine-tune the pretrained model with the batch size of 1 for 20k iterations on 4 NVIDIA RTX3090 GPUs. For the R2R dataset, additional augmented R2R data in [13] is used in pretraining. We pretrain our model with 200k iterations. Then we fine-tune the pretrained model with the batch size of 2 for 20k iterations on 4 NVIDIA RTX3090 GPUs. For all the datasets, the best epoch is selected by SPL on the val unseen split. 

## **4.3. Comparison to State-of-the-Art Methods** 

Table 1,2,3 illustrate the results of our method on the REVERIE, R2R and SOON datasets. Our approach achieves state-of-the-art performance under most metrics on all the seen/unseen splits of all three datasets, demonstrating the effectiveness of the proposed model. In particular, as shown in Table 1, our model outperforms the previous DUET [7] by 5.52% on SR, 6.51% on SPL, and 4.93% on RGSPL for val seen split of the REVERIE dataset. As shown in Table 2 and 3, our approach also shows performance gains on the R2R and SOON datasets, but not as pronounced on the REVERIE dataset. It is mainly because of the smaller size of data and more complex instructions, which make the generalization ability of our knowledge enhancement method not fully exploited. 

In addition, as shown in Table 1, we show the results of our model with two training strategies. KERM denotes the full pipeline which is composed of pretraining and fine-tuning. For KERM, our model is first pretrained 

Table 2. Comparison with state-of-the-art methods on the R2R dataset. _[†]_ indicates reproduced results. 

|Methods|Val Seen|Val Unseen|Test Unseen|
|---|---|---|---|
||TL↓<br>NE↓<br>SR↑<br>SPL↑|TL↓<br>NE↓<br>SR↑<br>SPL↑|TL↓<br>NE↓<br>SR↑<br>SPL↑|
|Seq2Seq [3]|11.33<br>6.01<br>39<br>-|8.39<br>7.81<br>22<br>-|8.13<br>7.85<br>20<br>18|
|AuxRN[37]|-<br>3.33<br>70<br>67|-<br>5.28<br>55<br>50|-<br>5.15<br>55<br>51|
|PREVALENT[13]|10.32<br>3.67<br>69<br>65|10.19<br>4.71<br>58<br>53|10.51<br>5.30<br>54<br>51|
|EntityGraph[14]|10.13<br>3.47<br>67<br>65|9.99<br>4.73<br>57<br>53|10.29<br>4.75<br>55<br>52|
|VLNBERT[15]|11.13<br>2.90<br>72<br>68|12.01<br>3.93<br>63<br>57|12.35<br>4.09<br>63<br>57|
|AirBERT[12]|11.09<br>2.68<br>75<br>70|11.78<br>4.01<br>62<br>56|12.41<br>4.13<br>62<br>67|
|HOP[25]|11.26<br>2.72<br>75<br>70|12.27<br>3.80<br>64<br>57|12.68<br>3.83<br>64<br>59|
|HAMT[5]|-<br>-<br>-<br>-|11.46<br>2.29<br>66<br>61|12.27<br>3.93<br>65<br>60|
|DUET[7]|-<br>-<br>-<br>-|13.94<br>3.31<br>72<br>60|14.73<br>3.65<br>69<br>59|
|DUET_†_|12.30<br>2.28<br>78.84<br>72.89|13.94<br>3.31<br>71.52<br>60.42|14.74<br>3.65<br>69.25<br>58.68|
|KERM(Ours)|12.16<br>**2.19**<br>**79.73**<br>**73.79**|13.54<br>**3.22**<br>**71.95**<br>**60.91**|14.60<br>**3.61**<br>**69.73**<br>**59.25**|



|ble 3. Comparison with state-of-the-art methods on the val un-<br>n split of the SOON dataset.<br>Method<br>TL↓OSR↑SR↑SPL↑RGSPL↑<br>GBE[36]<br>28.96 28.54 19.52 13.34<br>1.16<br>DUET[7]<br>36.20 50.91 36.28 22.58<br>3.75<br>KERM(Ours)<br>35.83 **51.62 38.05 23.16**<br>**4.04**<br><br><br> <br>|ble 3. Comparison with state-of-the-art methods on the val un-<br>n split of the SOON dataset.<br>Method<br>TL↓OSR↑SR↑SPL↑RGSPL↑<br>GBE[36]<br>28.96 28.54 19.52 13.34<br>1.16<br>DUET[7]<br>36.20 50.91 36.28 22.58<br>3.75<br>KERM(Ours)<br>35.83 **51.62 38.05 23.16**<br>**4.04**<br><br><br> <br>|Table 4.<br>Ablation study results on val unseen split of the<br>REVERIE dataset. “Pur.” denotes the purifcation module. “VF-<br>Int.” and “HF-Int.” denote the vision-fact and history-fact interac-<br>tion module respectively.|Table 4.<br>Ablation study results on val unseen split of the<br>REVERIE dataset. “Pur.” denotes the purifcation module. “VF-<br>Int.” and “HF-Int.” denote the vision-fact and history-fact interac-<br>tion module respectively.|
|---|---|---|---|
|Method|TL↓OSR↑SR↑SPL↑RGSPL↑|||
|GBE[36]|28.96 28.54 19.52 13.34<br>1.16|||
|DUET[7]|36.20 50.91 36.28 22.58<br>3.75|Pur. VF-Int. HF-Int.|OSR↑SR↑SPL↑RGS↑RGSPL↑<br><br>|
|KERM(Ours)|35.83 **51.62 38.05 23.16**<br>**4.04**||51.07 46.98 33.73 32.15<br>23.03|
|||✓<br>✓|5378493335013380<br>2389|
||||. . . .<br>.|
|||✓<br>✓|53.53 48.96 34.70 33.63<br>23.58|
|||✓<br>✓|54.01 49.43 34.79 33.22<br>23.80|
|||✓<br>✓<br>✓|**55.21 50.44 35.38 34.51**<br>**24.45**|



Table 3. Comparison with state-of-the-art methods on the val unseen split of the SOON dataset. 

|Pur.|VF-Int.|HF-Int.|OSR↑SR↑SPL↑RGS↑RGSPL↑|OSR↑SR↑SPL↑RGS↑RGSPL↑|
|---|---|---|---|---|
||||51.07 46.98 33.73 32.15|23.03|
|✓||✓|53.78 49.33 35.01 33.80|23.89|
|✓|✓||53.53 48.96 34.70 33.63|23.58|
||✓|✓|54.01 49.43 34.79 33.22|23.80|
|✓|✓|✓|**55.21 50.44 35.38 34.51**|**24.45**|



with augmented data, where the parameters are initialized with LXMERT [30]. And then, it is fine-tuned with standard data. KERM-pt denotes the pipeline which only finetunes our knowledge enhanced reasoning model with standard data, based on the baseline method ( _i.e.,_ the pretrained model in DUET [7]). Both models outperform the previous methods, further illustrating the effectiveness of the proposed knowledge enhancement method. Then, with the help of pretraining, KERM achieves better performance, reflecting that the generalization ability of our method can be further strengthened by pretraining with larger data. 

**Fact vs. Object.** In order to illustrate that region-based knowledge is superior to the object-centric information, we compare our method with the model utilizing object enhanced representations. Specifically, for each cropped region image, we use VinVL [35] to detect objects and take five objects with the highest scores as substitutes for facts. For fair comparison with retrieved facts, we prompt each object label by adding “an image of a” before the label to form a short sentence and then use CLIP-T to obtain objectcentric features. The purification module and fact-aware interaction module are the same as the fact-based approach. As shown in Table 5, we compare the baseline method and the methods of adding object features and fact features on the val unseen split of REVERIE. We can see that the object features are beneficial to navigation, but they are still inferior to fact representations. This is consistent with our assumption that object tags are not sufficient to provide strong generalization performance compared to facts which have a much larger semantic feature space. 

## **4.4. Further Analysis** 

**Ablation Study.** As shown in Table 4, we evaluate the impact of the key components of our model on the val unseen split of REVERIE. First, the results in row 2-4 are better than row 1, which demonstrate that all these modules are beneficial for navigation. Then, the results in row 5 are better than row 2,3, validating the effect of the knowledge enhanced representations in the vision-fact interaction or the history-fact interaction. Finally, the results in row 5 are better than row 4, validating the effectiveness of purification operations. With the purification, interaction, and aggregation modules, our method obtains the best performance. 

**Number of Regions.** As shown in Table 6, we explore the impact of the number of cropped sub-regions on navigation performance on the val unseen split of REVERIE. It 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation_images/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation.pdf-0008-00.png)


**----- Start of picture text -----**<br>
KERM (Our) DUET<br>✔ ×<br>(a)<br><artwork above cabinet>, <picture above cabinet>, <paneled cabinetwork><br><light brown kitchen cabinet> , <light wood paneling><br>✔<br>(b)<br>Go to the bathroom of the bedroom on the left at the end of the hallway by the display cabinet and bring me the photo hanging on the wall opposite the shower door.<br>**----- End of picture text -----**<br>


Figure 4. Visualization of navigation examples. The sentence within the yellow box is the navigation instruction for the agent. (a) shows a comparison where our KERM chooses the right location while the baseline model makes the wrong choice. (b) illustrates the weights for the 25 facts after the fact purification when the agent navigates to the right location. The facts for the sub-region within the blue box are annotated in red. The results show that our model can automatically select the relevant facts. 

Table 5. Comparison between fact and object. 

||OSR↑SR↑SPL↑RGS↑RGSPL↑|
|---|---|
|Baseline|51.07 46.98 33.73 32.15<br>23.03|
|+ Object|54.27 49.73 33.95 33.91<br>23.40|
|+ Fact(KERM)|**55.21 50.44 35.38 34.51**<br>**24.45**|



Table 6. The effect of the number of cropped sub-regions. 

|Number|OSR↑<br>SR↑<br>SPL↑<br>RGS↑<br>RGSPL↑|
|---|---|
|1|55.69<br>49.67<br>34.46<br>34.05<br>23.89|
|5|**55.21**<br>**50.44**<br>**35.38**<br>**34.51**<br>**24.45**|
|9|53.59<br>48.99<br>34.87<br>33.14<br>23.79|



is demonstrated that the strategy of five sub-regions yields the best results. It is mainly because fewer but larger sub-regions are difficult to retrieve more fine-grained facts, while more but smaller regions are too fragmented to contain all the complete parts that can retrieve accurate facts. 

## **4.5. Qualitative Results** 

Examples in the val unseen split of REVERIE are illustrated in Figure 4. The sentence within the yellow box is the navigation instruction of these examples and the red color words are the landmarks ( _e.g.,_ “cabinet” and “shower door”) or target objects ( _e.g.,_ “photo”). Figure 4(a) shows an example where our KERM chooses the correct navigation direction by capturing the facts associated with the “cabinet”, as shown in the left bottom of Figure 4(a). The baseline model makes the wrong choice without the facts. The left of Figure 4(b) shows the cropped region in the blue box where target objects ( _i.e.,_ objects belong to “photo”) are contained. In the right of Figure 4(b), a heat map is utilized to visualize 

the weight distribution after the purification process of the 25 facts mined from the target view image. Particularly, we use red color to highlight the five facts only mined from the region in the blue box. We can see that the facts annotated in red contain the words related to the target object ( _i.e.,_ “photo hanging on the wall opposite the shower door”) in the navigation instruction, such as “photograph” and “bathroom”. It is worth noting that these facts related to the target object have almost the highest purification weights, which demonstrates that our model can automatically select the relevant facts, thus obtain better performance. 

## **5. Conclusion** 

In this paper, we propose KERM, a knowledge enhanced reasoning model to improve the generalization ability and performance for VLN. Our work utilizes a large number of external fact descriptions to build a knowledge base and introduces the CLIP model to retrieve the facts ( _i.e.,_ knowledge described by language descriptions) for view images. We further design a knowledge enhanced reasoning approach with the process of purification, interaction, and aggregation that automatically obtains contextual information for action prediction. We illustrate the good interpretability of KERM and provide case study in deep insights. Our approach achieves excellent improvement on many VLN tasks, demonstrating that leveraging knowledge is a promising direction in improving VLN and Embodied AI. For future work, we will improve our KERM with larger training data and employ it on VLN in continuous environments. 

**Acknowledgment.** This work was supported in part by the National Natural Science Foundation of China under Grants 62125207, 62102400, 62272436, U1936203, and 

U20B2052, in part by the National Postdoctoral Program for Innovative Talents under Grant BX20200338, and in part by the Peng Cheng Laboratory Research Project under Grant PCL2021A07. 

## **References** 

- [1] Dong An, Yuankai Qi, Yan Huang, Qi Wu, Liang Wang, and Tieniu Tan. Neighbor-view enhanced model for vision and language navigation. In _ACM MM_ , pages 5101–5109, 2021. 1 

- [2] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In _CVPR_ , pages 6077–6086, 2018. 3, 4.1, 4.2.1, A 

- [3] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko S¨underhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In _CVPR_ , pages 3674–3683, 2018. 1, 1, 2, 3, 4.1, 1, 4.1, 2 

- [4] S¨oren Auer, Christian Bizer, Georgi Kobilarov, Jens Lehmann, Richard Cyganiak, and Zachary Ives. Dbpedia: A nucleus for a web of open data. In _The Semantic Web_ , pages 722–735, 2007. 2 

- [5] Shizhe Chen, Pierre-Louis Guhur, Cordelia Schmid, and Ivan Laptev. History aware multimodal transformer for vision-and-language navigation. In _NeurIPS_ , pages 5834– 5847, 2021. 1, 1, 2, 3.4, 1, 2 

- [6] Shizhe Chen, Pierre-Louis Guhur, Makarand Tapaswi, Cordelia Schmid, and Ivan Laptev. Learning from unlabeled 3d environments for vision-and-language navigation. In _ECCV_ , pages 638–655, 2022. 2 

- [7] Shizhe Chen, Pierre-Louis Guhur, Makarand Tapaswi, Cordelia Schmid, and Ivan Laptev. Think global, act local: Dual-scale graph transformer for vision-and-language navigation. In _CVPR_ , pages 16537–16547, 2022. 1, 1, 2, 3.1, 3.3, 3.3.3, 3.4, 4.1, 1, 4.2.1, 4.2.2, 4.3, 2, 3, A, B 

- [8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_ , 2020. 3.4, 4.2.1, A 

- [9] Daniel Fried, Ronghang Hu, Volkan Cirik, Anna Rohrbach, Jacob Andreas, Louis-Philippe Morency, Taylor BergKirkpatrick, Kate Saenko, Dan Klein, and Trevor Darrell. Speaker-follower models for vision-and-language navigation. In _NeurIPS_ , 2018. 1, 2 

- [10] Chen Gao, Jinyu Chen, Si Liu, Luting Wang, Qiong Zhang, and Qi Wu. Room-and-object aware knowledge reasoning for remote embodied referring expression. In _CVPR_ , pages 3064–3073, 2021. 1, 2 

- [11] Gerd Gigerenzer and Daniel G Goldstein. Reasoning the fast and frugal way: models of bounded rationality. _Psychological review_ , 103(4):650, 1996. 1 

- [12] Pierre-Louis Guhur, Makarand Tapaswi, Shizhe Chen, Ivan Laptev, and Cordelia Schmid. Airbert: In-domain pretraining for vision-and-language navigation. In _CVPR_ , pages 1634–1643, 2021. 1, 1, 2, 1, 2 

- [13] Weituo Hao, Chunyuan Li, Xiujun Li, Lawrence Carin, and Jianfeng Gao. Towards learning a generic agent for visionand-language navigation via pre-training. In _CVPR_ , pages 13137–13146, 2020. 1, 2, 4.2.2, 2, A 

- [14] Yicong Hong, Cristian Rodriguez, Yuankai Qi, Qi Wu, and Stephen Gould. Language and visual entity relationship graph for agent navigation. In _NeurIPS_ , pages 7685–7696, 2020. 2, 2 

- [15] Yicong Hong, Qi Wu, Yuankai Qi, Cristian RodriguezOpazo, and Stephen Gould. Vln bert: A recurrent visionand-language bert for navigation. In _CVPR_ , pages 1643– 1653, 2021. 1, 2, 1, 2 

- [16] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _IJCV_ , 123(1):32–73, 2017. 1, 3.2, A 

- [17] Alexander Ku, Peter Anderson, Roma Patel, Eugene Ie, and Jason Baldridge. Room-across-room: Multilingual visionand-language navigation with dense spatiotemporal grounding. In _EMNLP_ , pages 4392–4412, 2020. 2 

- [18] Chia-Wen Kuo and Zsolt Kira. Beyond a pre-trained object detector: Cross-modal textual and visual context for image captioning. In _CVPR_ , pages 17969–17979, 2022. 3.2 

- [19] Christian M Meyer and Iryna Gurevych. _Wiktionary: A new rival for expert-built lexicons? Exploring the possibilities of collaborative lexicography_ . na, 2012. 2 

- [20] Abhinav Moudgil, Arjun Majumdar, Harsh Agrawal, Stefan Lee, and Dhruv Batra. SOAT: A scene-and object-aware transformer for vision-and-language navigation. In _NeurIPS_ , pages 7357–7367, 2021. 1 

- [21] Mengshi Qi, Yunhong Wang, Jie Qin, and Annan Li. Kegan: Knowledge embedded generative adversarial networks for semi-supervised scene parsing. In _CVPR_ , pages 5237– 5246, 2019. 2 

- [22] Yuankai Qi, Zizheng Pan, Yicong Hong, Ming-Hsuan Yang, Anton van den Hengel, and Qi Wu. The road to knowwhere: An object-and-room informed sequential bert for indoor vision-language navigation. In _ICCV_ , pages 1655– 1664, 2021. 1, 1 

- [23] Yuankai Qi, Zizheng Pan, Shengping Zhang, Anton van den Hengel, and Qi Wu. Object-and-action aware model for visual language navigation. In _ECCV_ , pages 303–317, 2020. 1 

- [24] Yuankai Qi, Qi Wu, Peter Anderson, Xin Wang, William Yang Wang, Chunhua Shen, and Anton van den Hengel. Reverie: Remote embodied visual referring expression in real indoor environments. In _CVPR_ , pages 9982– 9991, 2020. 1, 1, 2, 3, 4.1, 4.1 

- [25] Yanyuan Qiao, Yuankai Qi, Yicong Hong, Zheng Yu, Peng Wang, and Qi Wu. HOP: History-and-order aware pretraining for vision-and-language navigation. In _CVPR_ , pages 15418–15427, 2022. 1, 1, 2, 3.4, 1, 2, A 

- [26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _ICML_ , pages 8748–8763, 2021. 1, 2, 3.2, 3.2, 4.2.1 

- [27] St´ephane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to noregret online learning. In _AISTATS_ , pages 627–635, 2011. 3.4 

- [28] Sheng Shen, Chunyuan Li, Xiaowei Hu, Yujia Xie, Jianwei Yang, Pengchuan Zhang, Anna Rohrbach, Zhe Gan, Lijuan Wang, Lu Yuan, et al. K-lite: Learning transferable visual models with external knowledge. In _NeurIPS_ , pages 15558– 15573, 2022. 2 

- [29] Robyn Speer, Joshua Chin, and Catherine Havasi. Conceptnet 5.5: An open multilingual graph of general knowledge. In _AAAI_ , pages 4444–4451, 2017. 2 

- [30] Hao Tan and Mohit Bansal. Lxmert: Learning crossmodality encoder representations from transformers. In _EMNLP_ , pages 5103–5114, 2019. 2, 4.2.1, 4.3 

- [31] Hao Tan, Licheng Yu, and Mohit Bansal. Learning to navigate unseen environments: Back translation with environmental dropout. In _NAACL_ , pages 2610–2621, 2019. 1, 2 

- [32] Jesse Thomason, Michael Murray, Maya Cakmak, and Luke Zettlemoyer. Vision-and-dialog navigation. In _CoRL_ , pages 394–406, 2019. 2 

- [33] Hanqing Wang, Wenguan Wang, Wei Liang, Caiming Xiong, and Jianbing Shen. Structured scene memory for visionlanguage navigation. In _CVPR_ , pages 8455–8464, 2021. 2 

- [34] Xin Wang, Qiuyuan Huang, Asli Celikyilmaz, Jianfeng Gao, Dinghan Shen, Yuan-Fang Wang, William Yang Wang, and Lei Zhang. Reinforced cross-modal matching and selfsupervised imitation learning for vision-language navigation. In _CVPR_ , pages 6629–6638, 2019. 1, 2, 1 

- [35] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Revisiting visual representations in vision-language models. In _CVPR_ , pages 5579–5588, 2021. 4.4 

- [36] Fengda Zhu, Xiwen Liang, Yi Zhu, Qizhi Yu, Xiaojun Chang, and Xiaodan Liang. Soon: Scenario oriented object navigation with graph-based exploration. In _CVPR_ , pages 12689–12699, 2021. 1, 1, 2, 3, 4.1, 3 

- [37] Fengda Zhu, Yi Zhu, Xiaojun Chang, and Xiaodan Liang. Vision-language navigation with self-supervised auxiliary reasoning tasks. In _CVPR_ , pages 10012–10022, 2020. 2, 2 

- [38] Wanrong Zhu, Yuankai Qi, Pradyumna Narayana, Kazoo Sone, Sugato Basu, Xin Wang, Qi Wu, Miguel Eckstein, and William Yang Wang. Diagnosing vision-and-language navigation: What really matters. In _NAACL_ , pages 5981–5993, 2022. 1 

## **Appendix** 

Section A provides additional details for the pretraining of our overall framework. The experimental results for 

the improvement based on the fine-scale encoder and the coarse-scale encoder are illustrated in Section B. Section C presents more qualitative examples. 

## **A. Pretraining Details** 

To pretrain our proposed KERM, we utilize four auxiliary tasks. Besides the behavior cloning tasks, _i.e.,_ singlestep action prediction (SAP) and object grounding (OG), the masked language modeling (MLM) and masked region classification (MRC) are utilized. In the following, these two tasks are described. 

- Masked language modeling (MLM). MLM aims to learn language representations by masking parts of the text and predicting them with the contextual information. The inputs of this task are pairs of language instruction _L_ and the corresponding demonstration path _P_ . As our method utilizes the dual-scale graph transformer [7] for action prediction, we also average the embeddings of the fine-scale and coarsescale encoders and then a network with two fullyconnected layers is used to predict the target word. Similar to previous approaches [13, 25], we randomly mask out the instruction words with a probability of 15%. This task is optimized by minimizing the negative log-likelihood of the original words: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation_images/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation.pdf-0010-19.png)


where _Lm_ is the masked instruction and _wi_ is the label of the masked word. 

- Masked region classification (MRC). MRC requires the model to predict the semantic labels of masked view images according to the instruction, unmasked view images, and the corresponding features in the topological map. With the same settings in DUET [7], we randomly mask out view images and objects in the last observation of the corresponding demonstration path _P_ with a probability of 15% in the fine-scale encoder. The visual features for the masked images or objects are set to zero, while their position embeddings are preserved. The target semantic labels for view images are predicted by an image classification model [8] pretrained on ImageNet, and the labels for the objects are obtained by an object detector [2] pretrained on the Visual Genome dataset [16]. Similar to [7], we use a two-layer fully-connected network to predict the semantic labels of masked visual tokens, and the KL divergence between the predicted and target probability distribution of each mask token is minimized. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation_images/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation.pdf-0011-00.png)


**----- Start of picture text -----**<br>
<office type desk> <artwork centered over headboard> <pictured bookcase> <gray-brown bathroom><br><clutter with desk> <artwork above sofa> <bookcase next to furniture> <area in bathroom><br><desk with a surface> <painting above sofa> <medium bookcase> <bathroom to right of kitchen><br><indoors office> <picture above sofa> <display hutch hovel> <room has faucet><br><door desk> <painting over sofa> <wall-mounted bookcase> <wall separating bathroom><br>**----- End of picture text -----**<br>


Figure 5. Examples of the retrieved facts for view images. Each view image is cropped into five sub-regions. We show the top-5 facts for the sub-regions in the blue box. 

Table 7. The results of different scales and dual-scale fusion on the val unseen split of the REVERIE dataset. 

||OSR↑SR↑SPL↑RGS↑RGSPL↑|
|---|---|
|DUET-fne|30.96 28.86 23.57 20.39<br>16.64|
|KERM-fne|34.40 30.80 24.83 21.68<br>17.56|
|DUET-coarse|46.44 36.52 25.98<br>-<br>-|
|KERM-coarse|46.38 37.38 26.32<br>-<br>-|
|DUET|51.07 46.98 33.73 32.15<br>23.03|
|KERM|55.21 50.44 35.38 34.51<br>24.45|



## **B. Improvements on Different Scales** 

We also evaluate our proposed method on the settings with only the fine-scale encoder and the coarse-scale encoder separately, as illustrated in Table 7. When only using the fine-scale encoder to predict the action, our KERMfine significantly outperforms DUET-fine. For example, the Success Rate (SR) is improved from 28.86% to 30.80%. The trend on the coarse-scale encoder is also the same. With both the fine-scale and the coarse-scale encoders, our KERM improves the SR by 3.3%. The results demonstrate the effectiveness of our method. 

Furthermore, we investigate the effect of the strategy that the agent selects the most likely viewpoint in the navigation history if the timesteps go above a certain threshold. For the fair comparison with previous approaches, we apply the same settings as [7]. Specifically, we set the maximum action steps as 15 for REVERIE and R2R, and 20 for SOON. Table 8 illustrates the proportion of the episodes terminated with the maximum action steps. For example, on the val unseen split of REVERIE, “560/3521 (15.90%)” represents that this split has 3521 episodes, and 560 of them are terminated by the criterion of the termination policy (TP), making up a proportion of 15.90%. Moreover, Table 9 shows the results of our KERM and the policy that the agent stops at the last visited location ( _i.e._ , KERM w/o TP). The re- 

Table 8. Statistics of episodes with the maximum action steps. 

||Val Seen|Val Unseen|Test|
|---|---|---|---|
|REVERIE|31/1423(2.18%)|560/3521(15.90%)|898/6292(14.27%)|
|R2R|10/1021(0.98%)|51/2349(2.17%)|132/4173(3.16%)|
|SOON|-|888/2261(39.27%)|1423/3080(46.20%)|



Table 9. Influence of the termination policy on the val unseen split. 

|||OSR|SR|SPL|RGSPL|
|---|---|---|---|---|---|
|REVERIE|KERM|**55.21**|**50.44**|**35.38**|**24.45**|
||KERM w/o TP|55.21|49.96|35.29|24.25|
|R2R|KERM|**80.42**|**71.95**|**60.91**|-|
||KERM w/o TP|80.42|71.90|60.77|-|
|SOON|KERM|**51.62**|**38.05**|**23.16**|**4.04**|
||KERM w/o TP|51.62|35.23|21.53|3.52|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation_images/cvpr_2023_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation.pdf-0011-11.png)


**----- Start of picture text -----**<br>
wall mounted fixture<br>wall mounted light<br>sconce hanging above mirror<br>wall-mounted light<br>sconce hangs over mirror<br>tile are behind bathroom fixture<br>side by side bathroom fixture<br>design in bathroom<br>bathroom painted color<br>painting on bathroom<br>bathroom in photograph<br>photographed bathroom<br>photograph of bathroom<br>photograph in bathroom<br>design in bathroom<br>tile are behind bathroom fixture<br>design in bathroom<br>painting above sink<br>side by side bathroom fixture<br>design in bathroom<br>wall separating bathroom<br>colonial style mirror<br>side by side bathroom fixture<br>tile are behind bathroom fixture<br>paneled barroom<br>[CLS] Go to the bathroom of the  bedroom  on  the  left  at  the  end  of  the  hallway  by  the  display  cabinet and  bring  me  the  photo hanging  on  the  wall opposite  the  shower door [END]<br>**----- End of picture text -----**<br>


Figure 6. Illustration of the weights for the 25 facts during fact purification. Best viewed in color. 

sults illustrate that the employed TP has small influence on REVERIE and R2R, while has great influence on SOON. This is because that the average hop of the trajectories on SOON is longer with more complex language instructions. 

## **C. More Qualitative Results** 

Figure 5 illustrates the retrieved facts for view images. The retrieved facts provide crucial information ( _e.g.,_ attributes and relationships between objects) which are complementary to visual features. Figure 6 demonstrates the weights for each fact corresponding to each word in the instruction during fact purification. It is illustrated that our model can automatically select the relevant facts to make better action prediction. 

