# **ReasonNet: End-to-End Driving with Temporal and Global Reasoning** 

Hao Shao[1] Letian Wang[2] Ruobing Chen[1] Steven L. Waslander[2] Hongsheng Li[3] Yu Liu[1] _[,]_[4][*] 

1SenseTime Research 2University of Toronto 3CUHK MMLab 

4Shanghai Artificial Intelligence Laboratory 

## **Abstract** 

_The large-scale deployment of autonomous vehicles is yet to come, and one of the major remaining challenges lies in urban dense traffic scenarios. In such cases, it remains challenging to predict the future evolution of the scene and future behaviors of objects, and to deal with rare adverse events such as the sudden appearance of occluded objects. In this paper, we present ReasonNet, a novel endto-end driving framework that extensively exploits both temporal and global information of the driving scene. By reasoning on the temporal behavior of objects, our method can effectively process the interactions and relationships among features in different frames. Reasoning about the global information of the scene can also improve overall perception performance and benefit the detection of adverse events, especially the anticipation of potential danger from occluded objects. For comprehensive evaluation on occlusion events, we also release publicly a driving simulation benchmark DriveOcclusionSim consisting of diverse occlusion events. We conduct extensive experiments on multiple CARLA benchmarks, where our model outperforms all prior methods, ranking first on the sensor track of the public CARLA Leaderboard [53]._ 

## **1. Introduction** 

Despite significant recent progress in the field of autonomous driving, truly large-scale deployment of autonomous vehicles (AVs) on public roads has yet to be established. The majority of the remaining issues lie in navigating dense urban traffic scenes, where a large number of different dynamic objects (e.g. vehicles, bicycles, pedestrians), complex road geometries and road user interactions are involved. In such circumstances, currently deployed or tested solutions could make incorrect or unexpected decisions , resulting in severe accidents or traffic infractions [4, 24, 53]. Two of the major challenges behind 

- *Corresponding author 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning_images/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning.pdf-0001-10.png)


**----- Start of picture text -----**<br>
Potential Danger<br>Temporal Reasoning<br>Brake<br>Occluded<br>Area<br>Global Reasoning<br>**----- End of picture text -----**<br>


Figure 1. Temporal reasoning on the historic behaviors of surrounding objects can benefit the prediction of the scene evolution and objects’ future behaviors. Global reasoning on the interaction among objects and the environment allows for inference about unobservable space and occluded objects, anticipating potential danger and enhancing perception/driving performance. 

such autonomous incompetence include 1) how to achieve a comprehensive understanding of the driving scene and, more importantly, how to make high-fidelity predictions on the future evolution of the driving scene; 2) how to deal with rare adverse events in long-tail distributions, such as undetected but relevant objects in occluded regions. 

Comprehensive scene understanding and high-fidelity prediction of how objects in the scene will move in the future are vital for autonomous vehicles to take safe and reliable actions. Toward this end, modularized methods were proposed to decompose the task into three sequential subtasks: detection [37–40, 42], tracking [5, 65], and forecasting [6, 26, 28, 32, 33, 59–61]. While more interpretability is provided by developing each module independently, these sub-tasks are still regarded as open research questions and errors in each sub-task can propagate and accumulate, leading to unstable overall performance. In contrast, end-toend driving methods [9, 10, 51] have recently emerged as a promising method to solve these subtasks in a monolithic manner directly. However, a high-fidelity future prediction necessitates sufficient **temporal reasoning** over the historic 

information of the scene [22, 50], which is usually only somewhat considered in previous end-to-end driving methods if not completely ignored. For example, [10, 15] only exploited scene information in the current frame, and [54] simply concatenated features in historic frames for temporal reasoning. In such cases, the interactions and relationships amongst features in different frames and objects cannot be sufficiently modeled. Thus in this paper, we propose a temporal reasoning module to effectively fuse information from different frames for better driving performance. 

On the other hand, rare adverse events in long-tail distributions remain a notoriously challenging issue on the way toward large scale deployment of autonomous vehicles. For example, one such challenge is the difficulty in detecting occluded but relevant objects in the scene. While a large amount of research has focused on improving perception performance [37, 57], the occluded objects essentially lie out of the scope of observable elements, and failure to consider such objects can result in either dangerous or overly cautious driving behavior. Our observation is that, while humans also suffer from similar limitations to autonomous vehicles regarding occluded objects, they are able to reason about these unobservable spaces by exploiting global information of the scene such as road geometry and driving interaction patterns, to anticipate potential danger even under occlusion. For example, when one human driver notices another vehicle braking abruptly, the driver may reason the presence of an occluded object (e.g., a pedestrian) ahead, reminding himself to drive cautiously. Thus, our insight is that, a safe and intelligent autonomous vehicle should also master the **global reasoning** capability to have a better perception of the scene. In this paper, we propose a transformer-based global reasoning module to sufficiently fuse information of the environment and objects, and analyze their interactions for better scene understanding. Such global reasoning capability not only benefits interaction modeling with occluded objects, but also improves overall perception performance. Examples of such performance gains include better traffic light status identification by reasoning over other vehicles’ actions and more accurate future trajectory forecasting by reasoning over interactions among objects. Besides, considering the fact that the occlusion events lie in the long-tail distribution and have been rare in currently available datasets, we also construct a Driving in Occlusion Simulation benchmark (DOS) consisting of 4 occlusion scenarios, each with 25 cases, as a comprehensive occlusion event evaluation benchmark in the field of end-to-end autonomous driving. 

In this paper, we propose a novel end-to-end driving framework named temporal and global reasoning network (ReasonNet), which provides enhanced reasoning on the temporal evolution and the global information of the scene, for better perception performance and driving quality. Our 

contributions are three-fold: 

- We propose a novel temporal and global reasoning Network (ReasonNet) to enhance historic scene reasoning for high-fidelity prediction of the scene’s future evolution and improve global contextual perception performance even under occlusion. 

- We present a new benchmark called **D** riving in **O** cclusion **S** imulation benchmark (DOS), which consists of diverse occlusion scenarios in urban driving for systematic evaluation in occlusion events, and make the benchmark publicly available. 

- We experimentally validate our method on multiple benchmarks with complex and adversarial urban scenarios. Our model ranks first on the sensor track of the CARLA autonomous driving leaderboard. 

## **2. Related work** 

**End-to-end Autonomous Driving** End-to-end autonomous driving in urban scenarios has become more studied recently thanks to the CARLA simulator and leaderboard [21]. Recent works mainly consist of reinforcement learning (RL) and imitation learning (IL) methods. The reinforcement Learning methods train the agents by constantly interacting with simulated environments and learning from these experiences. Latent DRL [54] first trains an embedding space as a latent representation of the environment observation, and then conducts reinforcement learning with the latent observation. Roach [66] utilizes an RL agent with privileged information of the environment to distill a model only with regular information (e.g. sensor) as the final agent. WOR [9] builds a model-based RL agent along with the world model and reward model. The final agent is distilled from the expert knowledge acquired from these pretrained models. Imitation learning methods aim at learning from an expert agent to bypass interacting with the environment. Early IL methods include CIL [17] and CILRS [18], which apply a conditional architecture with different network branches for different navigation commands. LBC [11] first trains an imitation learning agent with privileged information, which is then distilled into a model using sensor data. Transfuser [15, 47] designs a multi-modal transformer to fuse information between the front camera image and LiDAR data. LAV [10] exploits data of not only the ego vehicle but also surrounding vehicles for data augmentation by learning a viewpoint-invariant spatial intermediate representation. TCP [63] proposes a network with two branches which generates the control signal and waypoints respectively. An adaptive ensemble is applied to fuse the two output signals. InterFuser [51] uses a transformer to fuse and process multimodal multi-view sensors for comprehensive scene understanding. 

**Attention for Autonomous Driving** The attention mechanism has been demonstrated to be a powerful module in many areas of deep learning, including the context of driving. The classic attention-based Transformer architecture [55] was originally established in Natural Language Processing. Transformer (VIT) was then applied in Computer Vision (vision Transformer, VIT [20, 49]) and attains excellent performance on Imagenet classification. Later generations move on to generalize the attention mechanism to the driving domain, including motion forecasting [23, 36, 58], driver attention prediction [25, 34] and object tracking [45,52]. In the field of end-to-end autonomous driving, TransFuser [15, 47] exploits several transformer modules for the fusion of data from the front view camera and LiDAR. NEAT [14] uses intermediate attention maps to iteratively compress 2D image features into a compact birdeye-view (BEV) representation for driving. InterFuser [51] utilizes a transformer encoder and decoder to fuse information and decode the feature into interpretable embeddings. **Multi-task Learning** Our end-to-end driving framework adopts multi-task learning, with a joint objective of object detection, occupancy forecasting, traffic sign prediction and waypoint prediction. MotionNet [62] proposes a spatio-temporal pyramid network for joint perception and motion prediction based on BEV maps. PnPNet [41] proposes a new object trajectory representation and multiobject tracker to handle occlusion and false positives. IntentNet [7] predicts the high-level intentions of each agent from semantic HD maps building. ST-P3 [29] proposes an egocentric-aligned accumulation technique to preserve geometry information in 3D space and utilize a dual pathway modeling to consider past motion variations. 

## **3. Method** 

We aim at learning a driving policy _π_ that generates raw control commands by taking multi-view multi-modal sensor readings, vehicle measurements, and navigation commands as inputs. As shown in Figure 2, the proposed ReasonNet consists of three parts: 1) a perception module that extracts bird’s-eye-view (BEV) features from LiDAR and RGB data; 2) a temporal reasoning module that processes temporal information and maintains a memory bank storing historic features; 3) a global reasoning module that captures the interaction/relationship amongst objects and the environment, to detect adverse events (e.g. occlusion) and improve overall perception performance. This section will introduce these modules in detail. 

## **3.1. Perception Module** 

The perception module is responsible for processing and fusing different sensor data at the early stage of our framework, based on which temporal and global reasoning can be conducted by later modules. Specifically, five sensors 

are utilized: four RGB cameras (left, front, right and rear) _I_ rgb = _I_ 0 _,_ 1 _,_ 2 _,_ 3 and one LiDAR sensor _I_ lidar = _I_ 4. Four image inputs are obtained from the four cameras, and an additional focus-view image input is center-cropped from the front image to capture distant traffic lights. Point cloud data is retrieved from the LiDAR sensor. Our perception module includes a 2D backbone to embed image input into keys and values, a 3D backbone to embed LiDAR input into queries, and a BEV decoder that utilizes these keys, values, and queries to obtain features of the bird’s-eye view (BEV) map, waypoints, and traffic signs. 

**Image Input** For each image input of _Irgb_ , a 2D CNN backbone ResNet [27] is applied to generate a feature map _fi_ . Then, we use a convolution layer to map the channels of _fi_ to _Cv_ and flatten it to one-dimensional tokens. A sinusoidal positional encoding and learnable sensor embeddings are added to the tokens, so that the following network can distinguish them from different cameras and relative positions. Finally, tokens of different images are passed through a standard transformer encoder with _Ke_ layers. Each layer consists of Multi-Headed Self-Attention [55], MLP blocks and layer normalization [3]. This image fusion operation can contribute to a better perception of global context from multi-view inputs, generating keys and values for the image-LiDAR fusion in BEV decoder. 

**LiDAR Input** For the LiDAR input, we use PointPillars [35] as our 3D perception backbone to process points in the ego-vehicle-centered area _x ∈_ [ _−Hb, H − Hb_ ] and _y ∈_ [ _−W/_ 2 _, W/_ 2]. Specifically, we use a simplified version of PointNet [48] to encode the information of raw LiDAR points. Each pillar includes the points in a 0.25m _×_ 0.25m area. The extracted feature map is downsampled to _Cv × H × W_ for computation reduction and then serves as BEV queries used in the BEV decoder and memory bank. 

**Sensor-Fusion BEV Decoder** The BEV decoder follows a standard transformer architecture design with _Kbev_ layers to fuse tokens from different sensors. Tokens from the RGB images are fed as values and keys into the decoder, and tokens from the LiDAR points are fed as the _H × W_ queries to generate BEV features. In addition, two other kinds of queries for the prediction of traffic signs and waypoints **w** are also fed into the decoder. Following InterFuser [51], we use a 2-layer MLP as the traffic sign classifier to predict the traffic light state and whether there is a stop sign ahead; we then use a single-layer GRU [16] to auto-regressively generate consecutive waypoints _{_ **w** _t}[T] t_ =1 _[f]_[conditioned][on][the] goal location of the ego vehicle. _Tf_ denotes the number of the predicted time steps. To pretrain the perception module in the first training stage, the generated BEV feature is passed through a one-stage CenterPoint [64] to generate the _H × W ×_ 7 BEV map covering an _H_ m _× W_ m spatial region, where the seven channels represent object existence 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning_images/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Temporal Reasoning<br>Memory Bank<br>Update<br>Short-term Long-term Memory value Cv  ×  T  × H  × W<br>memory value memory value v<br>Short-term Long-term Memory key Ck  ×  T  × H  × W New memorykey New memoryvalue<br>memory key memory key k<br>Copy<br>LiDAR Input<br>Ck  ×  H  ×  W HW  ×  THW Cv  ×  T  ×  H  ×  W Value<br>3D Cv  ×  H  ×  W Query Affinity M Readout GRU encoder<br>BackBone q S features  M<br>M fused<br>query BEV features Cv  ×  H  ×  W Cv  ×  H  ×  W BEV Map<br>Multi-view Input<br>Interaction<br>Modeling<br>2D value Traffic sign<br>BackBone key Occupancy queries<br>Waypoints<br>consistency loss<br>Perception Global Reasoning<br>Occupancy Map<br>encoder Query<br>BEV<br>Decoder<br>Decoder<br>Occupancy<br>**----- End of picture text -----**<br>


Figure 2. The proposed ReasonNet consists of three modules: 1) the perception module fuses different sensor data to generate the BEV feature, traffic sign feature, and waypoints in the early stage of our framework; 2) the temporal reasoning module processes current and historic features and maintains a memory bank to store historic features; 3) the global reasoning module models the interaction and relationship among objects and the environment to detect adverse events (e.g. occlusion) and improve overall perception performance. 

probability, offset from grid center, bounding box extent, heading angle and velocity for objects at each grid cell. 

## **3.2. Temporal Reasoning module** 

Compared to existing end-to-end driving methods that only exploit scene information of the current frame [10, 15] or simply concatenate features of historic frames [54], we propose a temporal reasoning module that can sufficiently store and fuse temporal information to benefit the motion forecasting of traffic participants and the tracking of intermittently occluded objects. As shown in Figure 2, our temporal reasoning module includes temporal processing to fuse current and historic features through an attention mechanism, and maintains a memory bank which stores historic short-term and long-term feature keys and values. 

**Temporal Processing** Considering that information in different historic frames could have different relevance to the current scene, we apply an attention-based memory reading from the historic features. Specifically, for each historic frame _t_ stored in the memory, we first measure its relevance by calculating the normalized similarity **S** between the historic-frame feature key **k** _∈_ R _[C][k][×][T][h][×][H][×][W]_[ 1] and the current-frame feature query **q** _∈_ R _[C][k][×][H][×][W]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning_images/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning.pdf-0004-06.png)


> 1 _C_ , _H_ , _W_ denotes the channel, height, and width of the feature respectively, _Th_ denotes the number of the frames stored in the memory bank. 

We map every query element to a distribution over _H × W_ memory elements and correspondingly aggregate their values v to obtain the readout feature **M** _∈_ R _[C][v][×][T][h][×][H][×][W]_ for each frame _t_ stored in the memory: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning_images/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning.pdf-0004-09.png)


The aggregated features from all historic frames are then concatenated **M** _′ ∈_ R _Cv×_ (with _Th_ +1)the _×H×_ current-frame _W_ , which is thenfeaturepassedvaluethroughto get a GRU to progressively fuse temporal information and get **M** _fused ∈_ R _[C][v][×][H][×][W]_ as the final output of the module. Technically, we take the L2 similarity proposed in STCN [13] as the similarity measure function, which is more stable than the dot product [46]. The current-frame feature query **q** is obtained by passing the features from the 3D backbone **F** _∈_ R _[C][v][×][H][×][W]_ through a query encoder (several convolution layers). The historic-frame feature key **k** and value **v** are taken from the temporal memory bank. 

**Memory Bank Maintaining** As above, we have introduced the temporal processing at one single frame. After every _τ_ frame, the obtained feature key and value at that frame will be used to update the memory bank. Specifically, the current-frame feature query **q** is directly copied and fed into the memory bank as the memory key without extra computation. The final output **M** _fused_ will first be encoded 

to a BEV map **M** _p_ . The BEV map **M** _p_ will be concatenated with the final output **M** _fused_ and passed through a value encoder to obtain the memory value **v** , which is fed into the memory back. With the above key-value pairs, the memory bank maintains two kinds of buffer: the short-term and long-term buffer. On the one hand, the new key-value pair will be appended to the short-term buffer, as a highresolution memory of the scene in the past few seconds for accurate feature matching. Considering the limited GPU memory resources, we limit the buffer size and older keyvalue pairs will be discarded when the limit number _Ts_ is reached. However, when these older features are discarded, the long-term behavior of the traffic participants is missing, which can be crucial for motion forecasting in complex traffic scenarios. Thus on the other hand, inspired by XMem [12], the memory bank also maintains a long-term buffer that selectively stores important/representative keyvalue feature pairs discarded by the short-term buffer. Considering the fact that the objects surrounding the ego vehicle are sparse most of the time[2] , the long-term buffer selectively stores key-value features ( **k** and **v** ) which meet one of the two criteria: 1) their corresponding location in the BEV map **M** _p_ has a high probability of object existence; 2) their usage frequency is in top- _K_ of all candidate key-value features. The usage frequency is defined by its cumulative normalized similarity (Eq. 1). The features selected by the above criteria are appended to the last frame of memory. And if the last memory frame is full, we will initialize one new frame with the zero vector and set it as the last frame to store new features. When the number of frames reaches the limit _Tl_ , the obsolete memory will be removed. Such a compact storing strategy can efficiently track long-term representative features and intermittently occluded objects, while balancing the resources required. 

## **3.3. Global Reasoning module** 

Rare adverse events such as occluded objects are a notorious issue for the practical deployment of AVs. Our insight is that humans perceive their surroundings not only through sensors, but also by exploiting global information on the scene to reason over the unobservable spaces. For instance, when a vehicle performs an emergency stop without a clear reason, humans can infer that there is potentially an occluded object ahead of the vehicle and will drive more cautiously. Thus we propose the global reasoning module to capture the interaction and relationship between objects and the environment to detect adverse events (e.g. occlusion) and improve overall perception performance. The module consists of three parts: 1) an object-environment and objectobject interaction modeling process; 2) an occupancy decoder to generate the occupancy map; 3) a consistency loss 

> 2Based on the data collected from the CARLA simulator, only 7% of the ego-vehicle-centered BEV map area is occupied by active objects. 

to encourage consistent prediction of waypoints and the occupancy map. 

**Interaction Modeling** The object-environment and objectobject interaction modeling process aims at reasoning about the relationship among objects and the environment. On the one hand, **M** _fused_ features whose corresponding location in the BEV map **M** _p_ has a high probability of object existence will be extracted to represent object features. On the other hand, **M** _fused_ features will also be downsampled to represent the environment features. All object and environment features are used to construct a graph, which is passed through a graph attention network (GAT) [56] for interaction modeling. 

**Occupancy Decoder** Taking the features outputted by the GAT as keys and values, and the learnable positional embeddings as queries, the occupancy decoder utilizes a transformer decoder with _Kopy_ layers to generate: 1) the traffic sign feature, which is then concatenated with the traffic sign feature from the BEV decoder to generate the final traffic sign prediction; 2) the occupancy map feature, which is then applied with convolution operation to generate the occupancy map **O** _t ∈_ R _[T][f][ ×][H][×][W]_ . At a future time _t_ , each cell in the occupancy map contains a value in the range [0,1] representing the probability that the cell is occupied. 

**Consistency Loss** Currently, our framework predicts the waypoints and the occupancy map independently, which are not necessarily consistent. For example, the waypoints could overlap some obstacles in the occupancy map. Thus we propose a consistency loss to discourage waypoints’ crossing the high-probability region of the occupancy map. Further, the consistency loss also encourages generating longer waypoint trajectories for efficient driving. Specifically, the consistency loss aims at minimizing the average object existence probability of the cells that cover the predicted waypoints, and maximizing the average _l_ 2 length of the waypoint trajectory **w** : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning_images/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning.pdf-0005-08.png)


,where _Nc[t]_[denotes the number of covered cell at step] _[ t]_[,] **[ O]** _[t,i]_ denotes the object existence probability at cell _i_ at time _t_ . 

## **3.4. Control** 

Following [11], we use two PID controllers for latitudinal and longitudinal control, to track the heading and velocity of predicted waypoints respectively. If a red traffic light or stop sign is detected, the ego-vehicle will brake. Additionally, an emergency stop will also be performed if the ego vehicle’s current bounding box crosses the area in the occupancy map that has a high object existence probability or if the future waypoints overlap with objects in the BEV map. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning_images/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning.pdf-0006-00.png)


**----- Start of picture text -----**<br>
Parked Cars Sudden Brake<br>occluded<br>pedestrian<br>occluded  collision<br>pedestrian<br>collision<br>braking!<br>ego car ego car<br>occluded occluded<br>visible visible<br>Left Turn Red Light Infraction<br>occluded car<br>occluded car<br>collision<br>braking!<br>collision<br>ego car ego car<br>occluded occluded<br>visible visible<br>**----- End of picture text -----**<br>


Figure 3. An illustration for the four types of occlusion scenarios included in the proposed DOS benchmark. The orange color denotes the ego car. The blue/green dots denote the occluded/visible trajectory of the occluded dangerous object. 

## **3.5. Training Setup** 

The training of our framework consists of two stages. In the first stage, we train the perception module to predict BEV features, traffic sign features, and waypoints. Specifically, the loss of BEV features and traffic sign features is computed with additional prediction heads, which are discarded in the next stage. In the second stage, we freeze the perception module and train the other two modules. Five loss terms are considered: 1) the waypoints loss _Lw_ that minimizes the error between predicted waypoints and expert waypoints; 2) the BEV map loss _LBEV_ that follows [10, 64] to minimize the current-frame BEV map prediction error; 3) the traffic sign loss _Lsign_ for the traffic regulation prediction; 4) occupancy map loss _Lopy_ that minimizes the occupancy prediction error in a future horizon; 5) the consistency loss _L_ consistency which encourages a consistent generation of the waypoints and occupancy map. These loss terms are balanced by corresponding loss weights. 

## **4. Drive in Occlusion Sim (DOS) Benchmark** 

In order to address the issue that occlusion events are rare in existing datasets and benchmarks, we present the Drive in Occlusion Simulation benchmark (DOS), a CARLAbased framework providing diverse driving scenarios with occluded objects. As shown in Figure 3, the proposed DOS benchmark includes four types of challenging occlusion driving scenarios: 

**Parked Cars** (#1) The ego vehicle is driving in a straight 

|Rank|Method|DS _↑_|RC _↑_|IS _↑_|
|---|---|---|---|---|
|1|ReasonNet (Ours)|**79.95**|89_._89|**0.89**|
|2|InterFuser [51]|76_._18|88_._23|0_._84|
|3|TCP [63]|75_._14|85_._63|0_._87|
|4|LAV [10]|61_._85|**94.46**|0_._64|
|5|TransFuser [15]|61_._18|86_._69|0_._71|
|6|Latent TransFuser [15]|45_._20|66_._31|0_._72|
|7|GRIAD [8]|36_._79|61_._85|0_._60|
|8|TransFuser+ [1]|34_._58|69_._84|0_._56|



Table 1. Performance comparison on the public CARLA leaderboard [53] (accessed Nov 2022). For all three metrics, higher is better. Our method ranks first overall on the leaderboard, with the highest driving score (DS) and infraction score (IS), and the second highest route completion (RC). 

lane with parked cars on the side. Pedestrians can first appear on the sidewalk (visible) and then suddenly emerge through the occluded areas between parked cars (occluded). **Sudden Brake** (#2) The ego vehicle is driving in a straight lane along with other vehicles ahead. Pedestrians can suddenly emerge from the sidewalks, causing the other vehicles to brake while remaining invisible to the ego vehicle. 

**Left Turn** (#3) The ego vehicle intends to perform an unprotected left turn at an intersection, but a truck in the opposite lane blocks the view of oncoming traffic, intermittently obscuring vehicles driving straight through the intersection. **Red Light Infraction** (#4) The ego vehicle is crossing an intersection after some trucks. A left-to-right vehicle running a red light suddenly appears, forcing the trucks to brake promptly. But the ego vehicle’s view toward the runninglight vehicle is blocked by the trucks, so it remains invisible to the ego vehicle. 

Each of the four scenarios in the DOS benchmark comprises 25 different cases varying in the road environment and background traffic. Compared to a previous occlusion benchmark AUTOCASTSIM [19], the DOS benchmark: 1) includes occlusions of both vehicles and pedestrians, instead of only vehicles; 2) includes 100 cases of 4 scenarios, instead of only 3 cases of 3 scenarios; 3) considers specific occlusions that can potentially be resolved by temporal reasoning (intermittent occlusion, #1, #3) and global reasoning (constant occlusion but with interaction clues, #2, #4) about the scene, instead of random occlusions as in AUTOCASTSIM. Thus our scenarios can also serve as a good tracking-with-intermittent-occlusion benchmark and a People-as-Sensor [2, 31] benchmark. 

## **5. Experiments** 

## **5.1. Experiment Setup** 

**Implementation** We implement and evaluate our approach on the open-source CARLA simulator with version 

|Setting|Town 05 Long|DOS|
|---|---|---|
|_Ts_<br>_Tl_|DS_↑_<br>RC_↑_<br>IS_↑_<br>CR_↓_<br>Red_↓_<br>Blocked_↓_|SR#1_↑_<br>SR#2_↑_<br>SR#3_↑_<br>SR#4_↑_|
|0<br>0<br>1<br>0<br>2<br>0<br>2<br>1<br>4<br>0<br>4<br>2|66.7_±_3.8<br>**97.6**_±_**2.7**<br>0.68_±_0.03<br>0.18_±_0.03<br>0.05_±_0.02<br>**0.03**_±_**0.03**<br>67.9_±_3.4<br>96.8_±_2.3<br>0.70_±_0.02<br>0.16_±_0.04<br>0.04_±_0.03<br>0.05_±_0.02<br>68.1_±_3.1<br>96.9_±_3.4<br>0.70_±_0.03<br>0.16_±_0.03<br>0.04_±_0.02<br>0.05_±_0.03<br>70.9_±_2.0<br>95.7_±_3.1<br>0.74_±_0.02<br>0.13_±_0.02<br>0.04_±_0.02<br>0.06_±_0.04<br>70.5_±_2.1<br>96.4_±_2.5<br>0.73_±_0.04<br>0.14_±_0.03<br>0.03_±_0.02<br>0.06_±_0.03<br>**73.2**_±_**1.9**<br>95.9_±_2.3<br>**0.76**_±_**0.03**<br>**0.11**_±_**0.02**<br>**0.03**_±_**0.01**<br>0.07_±_0.03|22_±_1.6<br>28_±_3.4<br>26_±_2.1<br>25_±_1.6<br>30_±_3.6<br>38_±_3.6<br>32_±_2.8<br>32_±_3.4<br>28_±_5.5<br>48_±_4.1<br>38_±_4.4<br>52_±_3.9<br>55_±_4.4<br>57_±_4.1<br>48_±_4.1<br>55_±_5.5<br>32_±_5.4<br>58_±_4.4<br>40_±_5.5<br>55_±_4.9<br>**63**_±_**4.2**<br>**73**_±_**3.6**<br>**80**_±_**4.2**<br>**70**_±_**5.5**|



Table 2. Ablation study on different short-term buffer size _Ts_ and long-term buffer size _Tl_ , on the Town 05 Long benchmark and the proposed DOS benchmark. Performance is evaluated over three runs. CR: Collision rate, Red: Red light violation, Blocked: Vehicle blocked, SR: Success rate. SR#1 denotes the first kind of scenario in the DOS benchmark. As the two buffer sizes increase, improvement is witnessed in all metrics but the road completion. 

0.9.10.1 [21]. We use ResNet-50 pretrained on ImageNet as the 2D backbone and PointPillars trained from scratch as the 3D backbone. We predict _Tf_ = 4 time steps for the waypoints and occupancy map, and the interval between each time step is 0.5 seconds. The memory bank maintains _Ts_ = 4 frames in the short-term buffer and _Tl_ = 2 frames in the long-term buffer. The memory bank is updated every _τ_ = 2 frame. We refer readers to Appendix A for more details. 

**Dataset Collection** We collect an expert dataset of 2M frames by running a rule-based expert agent on all 8 public towns and 21 types of weather, with the access to the privileged information in the CARLA simulator. We randomly set routes, spawn dynamic objects and adversarial scenarios provided in [47], to diversify the collected data. To ensure the temporal continuity of collected data, the data are collected at a high frequency of 10HZ. 

**Metrics** We consider three major metrics introduced by the CARLA LeaderBoard: route completion ratio (RC), infraction score (IS), and driving score (DS). The route completion ratio is the percentage of the route completed. The infraction score measures infractions triggered. When collisions or traffic rule violations occur, the infraction score will decay by a discount factor. The driving score is the product of the route completion ratio and the infraction score, describing both driving progress and safety, and thus is the primary ranking metric in the CARLA Leaderboard. 

## **5.2. Comparison to the state of the art** 

Table 1 shows the top 8 entries on the public CARLA Leaderboard. Readers can refer to Sec 2 for descriptions of these methods. Our method outperforms all prior methods, with the highest driving score and infraction score, and the second highest route completion. The previous leading method InterFuser uses a transformer for sensor fusion but lacks temporal and global reasoning. Compared to InterFuser, our method improved the driving score, road completion, and infraction score by 5%, 2%, and 6% respectively. 

## **5.3. Ablation study** 

We investigate the effect of the temporal and global reasoning modules on the Town05 Long benchmark and the DOS benchmark. For each scenario in DOS, we take 5 cases for training and 20 cases for evaluation. In addition to the three metrics mentioned earlier, we also present four more metrics for detailed analysis: collision rate (CR), red light violation (Red), ego vehicle blocked frequency (Blocked), and success rate (SR). The first three metrics are normalized by the driven distance (km). Visualizations of how the temporal reasoning and global reasoning work can be found at Figure 4 and Figure 5 respectively. 

**Memory Size** Table 2 studies the effect of different shortterm buffer size _Ts_ and long-term buffer size _Tl_ . The overall observation is that, as the two buffer sizes increase, improvement is witnessed in all metrics but road completion. Specifically, when the long-term memory is removed ( _Tl_ = 0), the average success rates drop sharply from 71.5 to 36 on DOS scenarios that require keeping track of intermittently occluded objects (#1 and #3). If we remove the temporal reasoning module ( _Ts_ = _Tl_ = 0), the driving score on the Town05 benchmark drops by 9%, and the average success rate on the DOS benchmark drops by 46%. We hypothesize that the drop in performance is because 1) it can be really hard to accurately estimate the objects’ future motion based only on single-frame data; 2) temporal information can help keep track of objects that are intermittently occluded; 3) the global reasoning module may also work poorly when historic information is missing. 

**Long-Term Memory Selection Strategy** Table 3 studies the performance of different long-term memory selection strategies. Specifically, the proposed strategy in Sec 3.2 includes two selection criteria. So here we ablate the effect of the two criteria by 1) only selecting the short-term feature with top- _K_ usages (usage-based); 2) only selecting the feature with a high probability of the existence of an object (object-based). Besides, we also compared a random selection strategy. As in Table 3, the random selection strategy has the poorest performance especially on the DOS benchmark, as random selection could miss important and repre- 

|Setting|Town 05 Long|DOS|
|---|---|---|
||DS_↑_<br>RC_↑_<br>IS_↑_<br>CR_↓_<br>Red_↓_<br>Blocked_↓_|SR#1_↑_<br>SR#2_↑_<br>SR#3_↑_<br>SR#4_↑_|
|Random<br>Usage-based<br>Object-based<br>Full (Ours)|71.2_±_5.4<br>**96.6**_±_**2.4**<br>0.74_±_0.04<br>0.13_±_0.04<br>0.03_±_0.01<br>0.06_±_0.02<br>72.0_±_3.9<br>95.9_±_2.2<br>0.75_±_0.04<br>0.12_±_0.03<br>0.03_±_0.01<br>0.06_±_0.02<br>72.2_±_3.7<br>96.1_±_3.0<br>0.75_±_0.03<br>0.12_±_0.03<br>0.03_±_0.01<br>**0.05**_±_**0.02**<br>**73.2**_±_**1.9**<br>95.9_±_2.3<br>**0.76**_±_**0.03**<br>**0.11**_±_**0.02**<br>**0.03**_±_**0.01**<br>0.07_±_0.03|33_±_4.4<br>55_±_6.2<br>42_±_5.5<br>53_±_5.4<br>45_±_4.2<br>62_±_3.4<br>53_±_2.8<br>62_±_3.9<br>57_±_4.1<br>65_±_3.6<br>73_±_4.4<br>60_±_3.7<br>**63**_±_**4.2**<br>**73**_±_**3.6**<br>**80**_±_**4.2**<br>**70**_±_**5.5**|



Table 3. Ablation study on different long-term memory selection strategies. Our proposed strategy considering both the usage and object criteria outperforms the random selection strategy and the two methods with only one criteria, especially on the DOS benchmark. 

|Setting|Town 05 Long|DOS|
|---|---|---|
||DS_↑_<br>RC_↑_<br>IS_↑_<br>CR_↓_<br>Red_↓_<br>Blocked_↓_|SR#1_↑_<br>SR#2_↑_<br>SR#3_↑_<br>SR#4_↑_|
|No global reasoning<br>No consistency loss<br>No traffc sign prediction<br>Full (Ours)|68.9_±_4.6<br>**97.4**_±_**2.9**<br>0.71_±_0.04<br>0.15_±_0.04<br>0.05_±_0.02<br>**0.05**_±_**0.02**<br>72.2_±_3.4<br>96.1_±_3.2<br>0.75_±_0.03<br>0.12_±_0.02<br>**0.03**_±_**0.02**<br>0.06_±_0.03<br>71.1_±_2.7<br>96.0_±_4.1<br>0.74_±_0.03<br>**0.11**_±_**0.03**<br>0.05_±_0.03<br>0.07_±_0.03<br>**73.2**_±_**1.9**<br>95.9_±_2.3<br>**0.76**_±_**0.03**<br>**0.11**_±_**0.02**<br>**0.03**_±_**0.01**<br>0.07_±_0.03|28_±_2.8<br>34_±_3.4<br>29_±_2.0<br>27_±_3.6<br>60_±_4.1<br>72_±_3.9<br>77_±_4.9<br>68_±_4.2<br>62_±_4.4<br>72_±_4.0<br>**82**_±_**2.8**<br>**70**_±_**4.1**<br>**63**_±_**4.2**<br>**73**_±_**3.6**<br>80_±_4.2<br>**70**_±_**5.5**|



Table 4. Ablation study on the global reasoning module. The performance would drop when 1) the entire global reasoning module is removed; 2) the consistency loss is not applied; 3) the traffic sign feature from the reasoning module is not utilized. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning_images/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning.pdf-0008-04.png)


**----- Start of picture text -----**<br>
T - 1 T - 2 T - 3 T - 4<br>1.0<br>T - 1 T - 2 T - 3 T - 4 0.5<br>0<br>**----- End of picture text -----**<br>


Figure 4. Visualization of the attention map between one object’s current-frame feature query and the historic-frame feature stored in the short-term buffer, in two cases. The object’s current-frame feature consistently attends to its corresponding region in the historic feature map. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning_images/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning.pdf-0008-06.png)


**----- Start of picture text -----**<br>
Previous Frame Current Frame T T + 1 T + 2<br>occluded<br>stopping! stopping!<br>occluded occluded<br>**----- End of picture text -----**<br>


Figure 5. We show two cases of how our framework reasons the presence of the occluded object. In the first case, a pedestrian first appeared on the sidewalk (visible) and then emerges between two parked cars (occluded). In the second case, a vehicle runs the red light, forcing trucks to brake abruptly. But the ego vehicle’s view toward the running-light vehicle is blocked by the front trucks, so the running-light vehicle remains invisible to the ego vehicle. The rectangles mark the occluded objects. 

sentative features on the scene. Compared to our strategy utilizing both criteria, the two ablations omitting one of the criteria have a performance drop, especially on the DOS benchmark. The usage-based strategy performs worse than the object-based, showing that the features of objects could be more informative for capturing historic behaviors. 

**Global Reasoning Design** Table 4 studies the performance when different designs of the global reasoning module are applied. First, we remove the entire module and observe a significant drop in all metrics but the road completion. For instance, the average success rate on the DOS benchmark dropped from 71.5 to 29.5. This demonstrates the effectiveness of global reasoning, especially in occlusion events. Second, we ablated the consistency loss, which could alleviate the sub-optimal issues in collected expert data. A removal of consistency loss leads to a lower driving score and higher collision rate on the Town 05 benchmark and a lower success rate on the DOS benchmark. Third, excluding the traffic sign feature from the global reasoning model results in an increase on the red light violation. One explanation is that the traffic sign feature from the global reasoning module could help reason the distant traffic light state according to other road participants’ behavior. 

## **6. Conclusion** 

We present ReasonNet, a novel end-to-end autonomous driving framework including two major components: a temporal reasoning module and a global reasoning module. The temporal reasoning module processes the historic information on the driving scene for high-fidelity forecasting of other road participants and dynamically maintains a temporal memory bank. The global reasoning module models the interaction and relationship among the objects and environment to detect adverse events, especially occlusion, and improve overall perception performance. Our method pushes the state-of-the-art performance of the CARLA leaderboard by a considerable margin. Moreover, we also publicly release a new benchmark DOS consisting of diverse occlusion scenarios, to facilitate the study of occlusion detection in the field of end-to-end autonomous driving. 

## **References** 

- [1] Expert drivers for autonomous driving. urlhttps://kait0.github.io/files/master ~~t~~ hesis ~~b~~ ernhard ~~j~~ aeger.pdf, 2021. 6, 13 

- [2] Oladapo Afolabi, Katherine Driggs-Campbell, Roy Dong, Mykel J Kochenderfer, and S Shankar Sastry. People as sensors: Imputing maps from human actions. In _2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_ , pages 2342–2348. IEEE, 2018. 6 

- [3] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. _arXiv preprint arXiv:1607.06450_ , 2016. 3 

- [4] Neal E. Boudette. Teslas self-driving system cleared in deadly crash. The New York Times, 2017. 1 

- [5] Jinkun Cao, Xinshuo Weng, Rawal Khirodkar, Jiangmiao Pang, and Kris Kitani. Observation-centric sort: Rethinking sort for robust multi-object tracking. _arXiv preprint arXiv:2203.14360_ , 2022. 1 

- [6] Sergio Casas, Cole Gulino, Simon Suo, Katie Luo, Renjie Liao, and Raquel Urtasun. Implicit latent variable model for scene-consistent motion forecasting. In _Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXIII 16_ , pages 624– 641. Springer, 2020. 1 

- [7] Sergio Casas, Wenjie Luo, and Raquel Urtasun. Intentnet: Learning to predict intention from raw sensor data. In _Conference on Robot Learning_ , pages 947–956. PMLR, 2018. 3 

- [8] Raphael Chekroun, Marin Toromanoff, Sascha Hornauer, and Fabien Moutarde. Gri: General reinforced imitation and its application to vision-based autonomous driving. _arXiv preprint arXiv:2111.08575_ , 2021. 6, 13 

- [9] Dian Chen, Vladlen Koltun, and Philipp Kr¨ahenb¨uhl. Learning to drive from a world on rails. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 15590–15599, 2021. 1, 2, 13, 14 

- [10] Dian Chen and Philipp Kr¨ahenb¨uhl. Learning from all vehicles. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 17222–17231, 2022. 1, 2, 4, 6, 13 

- [11] Dian Chen, Brady Zhou, Vladlen Koltun, and Philipp Kr¨ahenb¨uhl. Learning by cheating. In _Conference on Robot Learning_ , pages 66–75. PMLR, 2020. 2, 5, 13, 14 

- [12] Ho Kei Cheng and Alexander G Schwing. Xmem: Longterm video object segmentation with an atkinson-shiffrin memory model. In _European Conference on Computer Vision_ , pages 640–658. Springer, 2022. 5 

- [13] Ho Kei Cheng, Yu-Wing Tai, and Chi-Keung Tang. Rethinking space-time networks with improved memory coverage for efficient video object segmentation. _Advances in Neural Information Processing Systems_ , 34:11781–11794, 2021. 4 

- [14] Kashyap Chitta, Aditya Prakash, and Andreas Geiger. Neat: Neural attention fields for end-to-end autonomous driving. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 15793–15803, 2021. 3, 12, 13, 14 

- [15] Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. _arXiv preprint arXiv:2205.15997_ , 2022. 2, 3, 4, 6, 13 

- [16] Kyunghyun Cho, Bart Van Merri¨enboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. _arXiv preprint arXiv:1406.1078_ , 2014. 3 

- [17] Felipe Codevilla, Matthias M¨uller, Antonio L´opez, Vladlen Koltun, and Alexey Dosovitskiy. End-to-end driving via conditional imitation learning. In _2018 IEEE International Conference on Robotics and Automation (ICRA)_ , pages 4693– 4700. IEEE, 2018. 2 

- [18] Felipe Codevilla, Eder Santana, Antonio M L´opez, and Adrien Gaidon. Exploring the limitations of behavior cloning for autonomous driving. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 9329–9338, 2019. 2, 13, 14 

- [19] Jiaxun Cui, Hang Qiu, Dian Chen, Peter Stone, and Yuke Zhu. Coopernaut: End-to-end driving with cooperative perception for networked vehicles. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 17252–17262, 2022. 6 

- [20] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv preprint arXiv:2010.11929_ , 2020. 3 

- [21] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. Carla: An open urban driving simulator. In _Conference on robot learning_ , pages 1–16. PMLR, 2017. 2, 7, 13 

- [22] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 6202–6211, 2019. 2 

- [23] Jiyang Gao, Chen Sun, Hang Zhao, Yi Shen, Dragomir Anguelov, Congcong Li, and Cordelia Schmid. Vectornet: Encoding hd maps and agent dynamics from vectorized representation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 11525– 11533, 2020. 3 

- [24] Samuel Gibbs. Ubers self-driving car saw the pedestrian but didnt swerve–report. The Guardian, 2018. 1 

- [25] Chao Gou, Yuchen Zhou, and Dan Li. Driver attention prediction based on convolution and transformers. _The Journal of Supercomputing_ , 78(6):8268–8284, 2022. 3 

- [26] Junru Gu, Chen Sun, and Hang Zhao. Densetnt: End-to-end trajectory prediction from dense goal sets. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 15303–15312, 2021. 1 

- [27] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 770–778, 2016. 3, 12 

- [28] Anthony Hu, Zak Murez, Nikhil Mohan, Sof´ıa Dudas, Jeffrey Hawke, Vijay Badrinarayanan, Roberto Cipolla, and Alex Kendall. Fiery: future instance prediction in bird’seye view from surround monocular cameras. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 15273–15282, 2021. 1 

- [29] Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, and Dacheng Tao. St-p3: End-to-end vision-based autonomous driving via spatial-temporal feature learning. In _Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXVIII_ , pages 533–549. Springer, 2022. 3 

- [30] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In _International conference on machine learning_ , pages 448–456. PMLR, 2015. 12 

- [31] Masha Itkina, Ye-Ji Mun, Katherine Driggs-Campbell, and Mykel J Kochenderfer. Multi-agent variational occlusion inference using people as sensors. In _2022 International Conference on Robotics and Automation (ICRA)_ , pages 4585– 4591. IEEE, 2022. 6 

- [32] Xiaosong Jia, Li Chen, Penghao Wu, Jia Zeng, Junchi Yan, Hongyang Li, and Yu Qiao. Towards capturing the temporal dynamics for trajectory prediction: a coarse-to-fine approach. In _Conference on Robot Learning_ , pages 910–920. PMLR, 2023. 1 

- [33] Xiaosong Jia, Penghao Wu, Li Chen, Hongyang Li, Yu Liu, and Junchi Yan. Hdgt: Heterogeneous driving graph transformer for multi-agent trajectory prediction via scene encoding. _arXiv preprint arXiv:2205.09753_ , 2022. 1 

- [34] Jinkyu Kim, Teruhisa Misu, Yi-Ting Chen, Ashish Tawari, and John Canny. Grounding human-to-vehicle advice for self-driving vehicles. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10591–10599, 2019. 3 

- [35] Alex H Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. Pointpillars: Fast encoders for object detection from point clouds. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 12697–12705, 2019. 3 

- [36] Lingyun Luke Li, Bin Yang, Ming Liang, Wenyuan Zeng, Mengye Ren, Sean Segal, and Raquel Urtasun. End-toend contextual perception and prediction with interaction transformer. In _2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_ , pages 5784–5791. IEEE, 2020. 3 

- [37] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. _arXiv preprint arXiv:2203.17270_ , 2022. 1, 2 

- [38] Qing Lian, Peiliang Li, and Xiaozhi Chen. Monojsg: Joint semantic and geometric cost volume for monocular 3d object detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 1070– 1079, 2022. 1 

- [39] Qing Lian, Yanbo Xu, Weilong Yao, Yingcong Chen, and Tong Zhang. Semi-supervised monocular 3d object detec- 

   - tion by multi-view consistency. In _Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part VIII_ , pages 715–731. Springer, 2022. 1 

- [40] Qing Lian, Botao Ye, Ruijia Xu, Weilong Yao, and Tong Zhang. Exploring geometric consistency for monocular 3d object detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 1685–1694, 2022. 1 

- [41] Ming Liang, Bin Yang, Wenyuan Zeng, Yun Chen, Rui Hu, Sergio Casas, and Raquel Urtasun. Pnpnet: End-to-end perception and prediction with tracking in the loop. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 11553–11562, 2020. 3 

- [42] Zhijian Liu, Haotian Tang, Alexander Amini, Xinyu Yang, Huizi Mao, Daniela Rus, and Song Han. Bevfusion: Multitask multi-sensor fusion with unified bird’s-eye view representation. _arXiv preprint arXiv:2205.13542_ , 2022. 1 

- [43] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. _arXiv preprint arXiv:1608.03983_ , 2016. 12 

- [44] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _International Conference on Learning Representations_ , 2018. 12 

- [45] Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, and Christoph Feichtenhofer. Trackformer: Multi-object tracking with transformers. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8844–8854, 2022. 3 

- [46] Mandela Patrick, Dylan Campbell, Yuki Asano, Ishan Misra, Florian Metze, Christoph Feichtenhofer, Andrea Vedaldi, and Jo˜ao F Henriques. Keeping your eye on the ball: Trajectory attention in video transformers. _Advances in neural information processing systems_ , 34:12493–12506, 2021. 4 

- [47] Aditya Prakash, Kashyap Chitta, and Andreas Geiger. Multimodal fusion transformer for end-to-end autonomous driving. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 7077–7087, 2021. 2, 3, 7, 12, 13, 14 

- [48] Charles R Qi, Wei Liu, Chenxia Wu, Hao Su, and Leonidas J Guibas. Frustum pointnets for 3d object detection from rgbd data. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 918–927, 2018. 3, 12 

- [49] Shengju Qian, Hao Shao, Yi Zhu, Mu Li, and Jiaya Jia. Blending anti-aliasing into vision transformer. _Advances in Neural Information Processing Systems_ , 34:5416–5429, 2021. 3 

- [50] Hao Shao, Shengju Qian, and Yu Liu. Temporal interlacing network. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 34, pages 11966–11973, 2020. 2 

- [51] Hao Shao, Letian Wang, Ruobing Chen, Hongsheng Li, and Yu Liu. Safety-enhanced autonomous driving using interpretable sensor fusion transformer. _arXiv preprint arXiv:2207.14024_ , 2022. 1, 2, 3, 6, 13, 14 

- [52] Peize Sun, Jinkun Cao, Yi Jiang, Rufeng Zhang, Enze Xie, Zehuan Yuan, Changhu Wang, and Ping Luo. Transtrack: Multiple object tracking with transformer. _arXiv preprint arXiv:2012.15460_ , 2020. 3 

- [53] CARLA team. Carla autonomous driving leaderboard. https://leaderboard.carla.org/, 2020. Accessed: 2021-02-11. 1, 6, 12, 13 

_IEEE/CVF International Conference on Computer Vision_ , pages 15222–15232, 2021. 2, 13, 14 

- [54] Marin Toromanoff, Emilie Wirbel, and Fabien Moutarde. End-to-end model-free reinforcement learning for urban driving using implicit affordances. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 7153–7162, 2020. 2, 4, 13 

- [55] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 3 

- [56] Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. _stat_ , 1050:20, 2017. 5 

- [57] Sourabh Vora, Alex H Lang, Bassam Helou, and Oscar Beijbom. Pointpainting: Sequential fusion for 3d object detection. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 4604–4612, 2020. 2 

- [58] Letian Wang, Yeping Hu, Liting Sun, Wei Zhan, Masayoshi Tomizuka, and Changliu Liu. Hierarchical adaptable and transferable networks (hatn) for driving behavior prediction. _arXiv preprint arXiv:2111.00788_ , 2021. 3 

- [59] Letian Wang, Yeping Hu, Liting Sun, Wei Zhan, Masayoshi Tomizuka, and Changliu Liu. Transferable and adaptable driving behavior prediction. _arXiv preprint arXiv:2202.05140_ , 2022. 1 

- [60] Letian Wang, Liting Sun, Masayoshi Tomizuka, and Wei Zhan. Socially-compatible behavior design of autonomous vehicles with verification on real human data. _IEEE Robotics and Automation Letters_ , 6(2):3421–3428, 2021. 1 

- [61] Bob Wei, Mengye Ren, Wenyuan Zeng, Ming Liang, Bin Yang, and Raquel Urtasun. Perceive, attend, and drive: Learning spatial attention for safe self-driving. In _2021 IEEE International Conference on Robotics and Automation (ICRA)_ , pages 4875–4881. IEEE, 2021. 1 

- [62] Pengxiang Wu, Siheng Chen, and Dimitris N Metaxas. Motionnet: Joint perception and motion prediction for autonomous driving based on bird’s eye view maps. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 11385–11395, 2020. 3 

- [63] Penghao Wu, Xiaosong Jia, Li Chen, Junchi Yan, Hongyang Li, and Yu Qiao. Trajectory-guided control prediction for end-to-end autonomous driving: A simple yet strong baseline. _arXiv preprint arXiv:2206.08129_ , 2022. 2, 6, 13 

- [64] Tianwei Yin, Xingyi Zhou, and Philipp Krahenbuhl. Centerbased 3d object detection and tracking. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 11784–11793, 2021. 3, 6 

- [65] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. Bytetrack: Multi-object tracking by associating every detection box. In _European Conference on Computer Vision_ , pages 1–21. Springer, 2022. 1 

- [66] Zhejun Zhang, Alexander Liniger, Dengxin Dai, Fisher Yu, and Luc Van Gool. End-to-end urban driving by imitating a reinforcement learning coach. In _Proceedings of the_ 

## **A. Implementation Details** 

**Model Details** The feature dimension of all decoders in our framework is set as 256. We use _Ke_ = 1, _Kbev_ = 3, _Kopy_ = 3, _C[K]_ = 64, _C[V]_ = 256 for the feature dimensions mentioned in Sec 3. The feature of the 5th stage in Resnet was used as the feature map _fi_ in the 2D backbone. We use Fully Connected Layer and Batch Normalization [30] to construct a simplified version of PointNet [48] to encode the information of raw LiDAR points in the 3D backbone. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning_images/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning.pdf-0012-02.png)


**----- Start of picture text -----**<br>
LiDAR Input<br>3D Cv  ×  H  ×  W<br>BackBone<br>BEV Map<br>Multi-view Input query<br>2D value<br>BackBone key Traffic sign<br>Waypoints<br>Perception<br>BEV<br>Decoder<br>**----- End of picture text -----**<br>


Figure 6. Overview of our pipeline for pretraining the perception module in the first training stage. 

**Training** We train our models using the AdamW optimizer [44] and a cosine learning rate scheduler [43]. In the first training stage, the initial learning rate is set to 5 _e[−]_[4] _×[BatchSize]_ for the transformer encoder and the 3D 512 backbone, and 2 _e[−]_[4] _×[BatchSize]_ 512 for the 2D backbones. The weight decay is 0.07. We train the models for 35 epochs with the first 5 epochs for warm-up [27]. We used random scaling from 0 _._ 9 to 1 _._ 1 and color jittering to augment the collected RGB images. The overview of the first-stage framework can be found in Figure 6. In the second training stage, we freeze the perception module. The training schedule of the other two modules are similar to that in the first stage. 

**Sensors** The RGB images are collected and cropped from one front-facing camera, two side-facing cameras, and one back-facing camera with a resolution of 800 _×_ 600. Each camera has a 100 _[◦]_ horizontal field of view (FOV), and the side cameras are angled at 60 _[◦]_ . For the front image, we scale the shorter side of the front camera input to 256 and crop its center patch of 224 _×_ 224. For the focusing-view image, we directly crop the center of the front camera input to get a 128 _×_ 128 patch. For the other images, the shorter side of the camera input is scaled to 160 and a center patch of 128 _×_ 128 is taken. 

**Other hyper-parameter values** Some other hyperparameter values used in ReasonNet are listed in Table 8. 

## **B. Benchmark details** 

We evaluate our method on the CARLA public leaderboard [53], Town05 benchmark [47], and our proposed DOS benchmark. Adversarial events[3] are included in the first two benchmarks, and occlusion events are included in the last benchmark. In these benchmarks, the ego vehicle is required to complete a given route without collision or traffic rules violation. 

**CARLA Leaderboard** The CARLA Autonomous Driving Leaderboard [53] is to evaluate the driving proficiency of autonomous agents in realistic traffic situations with a variety of weather conditions. The CARLA leaderboard provides a set of 76 routes for training and verifying agents and contains a secret set of 100 routes to evaluate the driving performance of the submitted agents. 

**Town05 benchmark** In this benchmark, we use Town05 for evaluation and other towns for training. Following [47], the benchmark includes two evaluation settings: 1) Town05 Short: 10 short routes of 100-500m, each comprising 3 intersections, 2) Town05 Long: 10 long routes of 10002000m, each comprising 10 intersections. Town05 is a complex town with multi-lane roads, single-lane roads, bridges, highways and exits. The core challenge of the benchmark is how to handle dynamic dense agents and adversarial events. **CARLA 42 routes benchmark** The CARLA 42 routes benchmark was proposed in NEAT [14], including six towns covering a variety of areas such as US-style intersections, EU-style intersections, freeways, roundabouts, stop signs, urban scenes and residential districts. The traffic density of each town is set to be comparable to busy traffic setting. We take the same configuration open-sourced by [47] when we evaluated the methods. 

## **C. More Experimental results** 

In this section we report additional experimental results, including the CARLA leaderboard and two other benchmarks. 

## **C.1. CARLA leaderboard** 

Table 5 shows the detailed comparison between our method and the baselines on the CARLA public Leaderboard [53]. Our method also leads the vehicle collision and offroad infraction numbers among all the methods. 

## **C.2. Town05 and CARLA 42 routes** 

Table 6 and Table 7 additionally compare the driving score, road completion, and infraction score of the presented approach to prior state-of-the-art on the CARLA 

> 3Adversarial events include unexpected agents rushing into the road from occluded regions, vehicles running red traffic lights, etc. Please refer to https://leaderboard.carla.org/scenarios/ for detailed descriptions. 

|Rank<br>Method|Driving<br>Score<br>Route<br>Completion<br>Infraction<br>Score<br>Vehicle<br>Collisions<br>Pedestrian<br>Collisions<br>Layout<br>Collisions<br>Red light<br>Violations<br>Offroad<br>Infractions<br>Blocked<br>Infractions|
|---|---|
|1<br>ReasonNet (Ours)<br>2<br>InterFuser [51]<br>3<br>TCP [63]<br>4<br>LAV [10]<br>5<br>TransFuser [15]<br>6<br>Latent TransFuser [15]<br>7<br>GRIAD [8]<br>8<br>TransFuser+ [1]<br>9<br>Rails [9]<br>10<br>IARL [54]<br>11<br>NEAT [14]|**79.95**<br>89.89<br>**0.89**<br>**0.13**<br>0.02<br>0.01<br>0.08<br>**0.04**<br>0.33<br>76.18<br>88.23<br>0.84<br>0.37<br>0.04<br>0.14<br>0.22<br>0.13<br>0.43<br>75.14<br>85.63<br>0.87<br>0.32<br>**0.00**<br>**0.00**<br>0.09<br>**0.04**<br>0.54<br>61.85<br>**94.46**<br>0.64<br>0.70<br>0.04<br>0.02<br>0.17<br>0.25<br>**0.10**<br>61.18<br>86.69<br>0.04<br>0.71<br>0.81<br>0.01<br>**0.05**<br>0.23<br>0.43<br>45.20<br>66.31<br>0.72<br>1.11<br>0.02<br>0.02<br>**0.05**<br>0.16<br>1.82<br>36.79<br>61.85<br>0.60<br>2.77<br>**0.00**<br>0.41<br>0.48<br>1.39<br>0.84<br>34.58<br>69.84<br>0.56<br>0.70<br>0.04<br>0.03<br>0.75<br>0.18<br>2.41<br>31.37<br>57.65<br>0.56<br>1.35<br>0.61<br>1.02<br>0.79<br>0.96<br>0.47<br>24.98<br>46.97<br>0.52<br>2.33<br>**0.00**<br>2.47<br>0.55<br>1.82<br>0.94<br>21.83<br>41.71<br>0.65<br>0.74<br>0.04<br>0.62<br>0.70<br>2.68<br>5.22|



Table 5. Comparison of our method and the state-of-the-art on the public CARLA leaderboard [53] (accessed Nov 2022). Methods are ranked by the driving score as the main metric. Driving Score, Route Completion, Infraction Score are higher the better, and the other metrics are lower the better. We outperform all other methods by a wide margin. We also lead the vehicle collision, offroad infraction numbers among all the methods. 

||**Town05 Short**|**Town05 Long**|
|---|---|---|
|**Method**|DrivingScore _↑_<br>Road Completion _↑_|DrivingScore _↑_<br>Road Completion _↑_|
|CILRS [18]<br>LBC [11]<br>TransFuser [47]<br>NEAT [14]<br>Roach [66]<br>WOR [9]<br>InterFuser[51]|7.47_±_2.51<br>13.40_±_1.09<br>30.97_±_4.17<br>55.01_±_5.14<br>54.52_±_4.29<br>78.41_±_3.75<br>58.70_±_4.11<br>77.32_±_4.91<br>65.26_±_3.63<br>88.24_±_5.16<br>64.79_±_5.53<br>87.47_±_4.68<br>94.95_±_1.91<br>95.19_±_2.57|3.68_±_2.16<br>7.19_±_2.95<br>7.05_±_2.13<br>32.09_±_7.40<br>33.15_±_4.04<br>56.36_±_7.14<br>37.72_±_3.55<br>62.13_±_4.66<br>43.64_±_3.95<br>80.37_±_5.68<br>44.80_±_3.69<br>82.41_±_5.01<br>68.31_±_1.86<br>94.97_±_2.87|
|ReasonNet(Ours)|**95.71**_±_1.88<br>**96.23**_±_3.17|**73.22**_±_1.91<br>**95.88**_±_2.31|



Table 6. Comparison of our ReasonNet with six state-of-the-art methods in Town05 benchmark. Our method outperformed other strong methods in all metrics and scenarios. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning_images/cvpr_2023_ReasonNet_End-to-End_Driving_With_Temporal_and_Global_Reasoning.pdf-0013-04.png)


**----- Start of picture text -----**<br>
WetDawn WetCloudyTwilight WetCloudyNight<br>MidRainNoon ClearTwilight HardRainDawn<br>**----- End of picture text -----**<br>


towns. For the ablation studies, we train the models on five towns (Town01, Town03, Town04, Town06 ,and Town07). 

## **E. License of Assets** 

We use the open-source CARLA driving simulator [21]. CARLA is released under the MIT license. Its assets are under the CC-BY license. The pretrained ResNet model is under the MIT license. The source code for our work will be publicly available once accepted and they are under the CC-BY-NC 4.0 license. 

Figure 7. Different types of weather in our dataset. 

Town05 benchmark [47] and CARLA 42 routes benchmark [14]. 

## **D. Data statistics** 

We describe the detailed statistics for each town and their corresponding maps in Table 9. In Figure 7, we show six types of weathers among our dataset. For the submission for the online leaderboard, the model is trained in all eight 

|**Method**|DrivingScore _↑_<br>Road Completion _↑_<br>Infraction Score _↑_|
|---|---|
|CILRS [18]<br>LBC [11]<br>AIM [47]<br>TransFuser [47]<br>NEAT [14]<br>Roach [66]<br>WOR [9]<br>InterFuser[51]|22.97_±_0.90<br>35.46_±_0.41<br>0.66_±_0.02<br>29.07_±_0.67<br>61.35_±_2.26<br>0.57_±_0.02<br>51.25_±_0.17<br>70.04_±_2.31<br>0.73_±_0.03<br>53.40_±_4.54<br>72.18_±_4.17<br>0.74_±_0.04<br>65.17_±_1.75<br>79.17_±_3.25<br>0.82_±_0.01<br>65.08_±_0.99<br>85.16_±_4.20<br>0.77_±_0.02<br>67.64_±_1.26<br>90.16_±_3.81<br>0.75_±_0.02<br>91.84_±_2.17<br>**97.12**_±_**1.95**<br>0.95_±_0.02|
|ReasonNet(Ours)|**93.25**_±_**2.91**<br>96.84_±_2.17<br>**0.96**_±_**0.02**|



Table 7. Comparison of our ReasonNet with other methods in CARLA 42 routes benchmark. Our method outperformed other strong methods in driving score and infraction score. 

|**Notation**|**Description**|**Value**|
|---|---|---|
||BEV Map and Controller||
|_amax_|Maximum acceleration|1.0 m/s|
|_vmax_|Maximum velocity|7.5 m/_s_2|
|H, W|Size of the BEV map|50, 50|
||Size of the BEV area|50 meter_×_50 meter|
|_Hb_|The detection range for the backward of the ego vehicle|20|
||Scale factor for bounding box size of pedestrians and bicycles|2|
||||
||Learning Process||
||Number of epochs|35|
||Number of warm-up epochs|5|
|_λsign_|Weight for the traffc sign loss|0.2|
|_λw_|Weight for the waypoints loss|0.4|
|_λBEV_|Weight for the BEV map loss|0.4|
|_λopy_|Weight for the occupancy map loss|0.2|
|_λconsistency_|Weight for the consistency loss|0.05|
||Max norm for gradient clipping|10.0|
||Weight decay|0.07|
||Batch size|256|



Table 8. The parameter used for ReasonNet. 

|Town Name|#Frames|Description|
|---|---|---|
|Town01|342846|A basic town layout consisting of “T junctions”|
|Town02|197240|Similar to Town01, but smaller|
|Town03|469115|The most complex town, with a 5-lane junction, a roundabout, unevenness, a tunnel, and more|
|Town04|429979|An infnite loop with a highway and a small town|
|Town05|297140|Squared-grid town with cross junctions and a bridge. It has multiple lanes per direction.|
|Town06|148495|Long highways with many highway entrances and exits. It also has a Michigan left|
|Town07|55299|A rural environment with narrow roads, barns and hardly any traffc lights|
|Town10|69039|A city environment with different environments such as an avenue or promenade|



Table 9. Detailed statistics of the number of frames and a brief description of each town. 

