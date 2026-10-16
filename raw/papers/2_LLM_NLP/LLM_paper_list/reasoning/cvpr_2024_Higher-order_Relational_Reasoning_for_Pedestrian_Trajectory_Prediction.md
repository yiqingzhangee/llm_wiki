## **TrafficPredict: Trajectory Prediction for Heterogeneous Traffic-Agents** 

**Yuexin Ma**[1] _[,]_[2] **, Xinge Zhu**[3] **, Sibo Zhang**[1] **, Ruigang Yang**[1] **, Wenping Wang**[2] **, Dinesh Manocha**[4] 

Baidu Research, Baidu Inc.[1] , The University of Hong Kong[2] , 

The Chinese University of Hong Kong[3] , University of Maryland at College Park[4] yxma@cs.hku.hk, zhuxinge123@gmail.com, sibozhang@baidu.com, yangruigang@baidu.com, wenping@cs.hku.hk, dm@cs.umd.edu 

## **Abstract** 

To safely and efficiently navigate in complex urban traffic, autonomous vehicles must make responsible predictions in relation to surrounding traffic-agents (vehicles, bicycles, pedestrians, etc.). A challenging and critical task is to explore the movement patterns of different traffic-agents and predict their future trajectories accurately to help the autonomous vehicle make reasonable navigation decision. To solve this problem, we propose a long short-term memory-based (LSTM-based) realtime traffic prediction algorithm, TrafficPredict. Our approach uses an instance layer to learn instances’ movements and interactions and has a category layer to learn the similarities of instances belonging to the same type to refine the prediction. In order to evaluate its performance, we collected trajectory datasets in a large city consisting of varying conditions and traffic densities. The dataset includes many challenging scenarios where vehicles, bicycles, and pedestrians move among one another. We evaluate the performance of TrafficPredict on our new dataset and highlight its higher accuracy for trajectory prediction by comparing with prior prediction methods. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0001-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0001-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0001-09.png)


## **Introduction** 

Autonomous driving is a significant and difficult task that has the potential to impact peoples day-to-day lives. The goal is to make a vehicle perceive the environment and safely and efficiently navigate any traffic situation without human intervention. Some of the challenges arise in dense urban environments, where the traffic consists of different kinds of _traffic-agents_ , including cars, bicycles, buses, pedestrians, etc.. These traffic-agents have different shapes, dynamics, and varying behaviors and can be regarded as an instance of a heterogeneous multi-agent system. To guarantee the safety of autonomous driving, the system should be able to analyze the motion patterns of other traffic-agents and predict their future trajectories so that the autonomous vehicle can make appropriate navigation decisions. 

Driving in an urban environment is much more challenging than driving on a highway. Urban traffic is riddled with more uncertainties, complex road conditions, and diverse traffic-agents, especially on some cross-roads. Different traffic-agents have different motion patterns. At the same 

Copyright _⃝_ c 2019, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved. 

Figure 1: **Heterogeneous urban traffic scenario:** We demonstrate the improved trajectory prediction accuracy of our method over prior approaches (top). The green solid lines denote ground truth trajectories (GT), pink solid lines are for our method (TP) and dashed lines are the predicted trajectories for other methods (ED, SL, SA). We observe 20% improvement in accuracy using TP. Traffic corresponding point cloud captured by LiDAR of the acquisition car is shown on the left bottom. Original trajectories of trafficagents in the scenario are shown on the right bottom: blue for cars, green for bicycles, and red for pedestrians. 

time, traffic-agents behaviors are deeply affected by other traffic-agents. It is necessary to consider the interaction between the agent to improve the accuracy of trajectory prediction. 

The problem of predicting trajectories for moving agents has been studied extensively. Some traditional algorithms are based on motion models like kinematic and dynamic models (Toledo-Moreo and Zamora-Izquierdo 2009), Bayesian filters (Kalman 1960), Gaussian Processes (Rasmussen and Williams 2006), etc. These methods do not take into account interactions between the traffic-agents and the environment, making it difficult to analyze complicated sce- 

narios or perform long-term predictions. With the success of LSTM networks in modeling non-linear temporal dependencies (Ma _et al._ 2017) in sequence learning and generation tasks, more and more works have been using these networks to predict trajectories of human crowds(Alahi _et al._ 2016) and vehicles trajectories (Lee _et al._ 2017). The common limitation of these works is the focus on predicting one type of group (only pedestrians or cars, for example). These methods may not work in heterogeneous traffic, where different vehicles and pedestrians coexist and interact with each other(Chandra _et al._ 2018b). 

**Main Results:** For the task of trajectory prediction in heterogeneous traffic, we propose a novel LSTM-based algorithm, TrafficPredict. Given a sequence of trajectory data, we construct a _4D Graph_ , where two dimensions are for instances and their interactions, one dimension is for time series, and one dimension is for high-level categorization. In this graph, all the valid instances and categories of trafficagents are denoted as nodes, and all the relationships in spatial and temporal space is represented as edges. Sequential movement information and interaction information are stored and transferred by these nodes and edges. Our LSTM network architecture is constructed on the _4D Graph_ , which can be divided into two main layers: one is the instance layer and the other is the category layer. The former is designed to capture dynamic properties and and interactions between the traffic-agents at a micro level. The latter aims to conclude the behavior similarities of instances of the same category using a macroscopic view and guide the prediction for instances in turn. We also use a self attention mechanism in the category layer to capture the historical movement patterns and highlight the category difference. Our method is the first to integrate the trajectory prediction for different kinds of trafficagents in one unified framework. 

To better expedite research progress on prediction and navigation in challenging scenarios for autonomous driving, we provide a new trajectory dataset for complex urban traffic with heterogeneous traffic-agents during rush hours. Scenario and data sample of our dataset is shown in Fig. 1. In practice, TrafficPredict takes about a fraction of a second on a single CPU core and exhibits 20% accuracy improvement over prior prediction schemes. The novel components of our work include: 

- Propose a new approach for trajectory prediction in heterogeneous traffic. 

- Collect a new trajectory dataset in urban traffic with much interaction between different categories of traffic-agents. 

- Our method has smaller prediction error compared with other state-of-art approaches. 

The rest of the paper is organized as follows. We give a brief overview of related prior work in Section 2. In Section 3, we define the problem and give details of our prediction algorithm. We introduce our new traffic dataset and show the performance of our methods in Section 4. 

## **Related Work** 

## **Classical methods for trajectory prediction** 

The problem of trajectory prediction or path prediction has been extensively studied. There are many classical approaches, including Bayesian networks (Lef`evre _et al._ 2011), Monte Carlo Simulation (Danielsson _et al._ 2007), Hidden Markov Models (HMM) (Firl _et al._ 2012), Kalman Filters (Kalman 1960), linear and non-linear Gaussian Process regression models (Rasmussen and Williams 2006), etc. These methods focus on analyzing the inherent regularities of objects themselves based on their previous movements. They can be used in simple traffic scenarios in which there are few interactions among cars, but these methods may not work well when different kinds of vehicles and pedestrians appear at the same time. 

## **Behavior modeling and interactions** 

There is considerable work on human behavior and interactions. The Social Force model (Helbing and Molnar 1995) presents a pedestrian motion model with attractive and repulsive forces, which has been extended by (Yamaguchi _et al._ 2011). Some similar methods have also been proposed that use continuum dynamics (Treuille _et al._ 2006), Gaussian processes (Wang _et al._ 2008), etc. Bera et al. (2016; 2017) combine an Ensemble Kalman Filter and human motion model to predict the trajectories for crowds. These methods are useful for analyzing motions of pedestrians in different scenarios, such as shopping malls, squares, and pedestrian streets. There are also some approaches to classify group emotions or identify driver behaviors (Cheung _et al._ 2018). To extend these methods to general traffic scenarios, (Ma _et al._ 2018) predicts the trajectories of multiple traffic-agents by considering kinematic and dynamic constraints. However, this model assumes perfect sensing and shape and dynamics information for all of the traffic agents. 

## **RNN networks for sequence prediction** 

In recent years, the concept of the deep neural network (DNN) has received a huge amount of attention due to its good performance in many areas (Goodfellow _et al._ 2016). Recurrent neural network (RNN) is one of the DNN architectures and is widely used for sequence generation in many domains, including speech recognition (Graves and Jaitly 2014), machine translation (Chung _et al._ 2015), and image captioning (Vinyals _et al._ 2015). Many methods based on long short-term Memory (LSTM), one variant of RNN, have been proposed for maneuver classification (Khosroshahi 2017) and trajectory prediction (Altch´e and Fortelle 2017). Some methods (Kim _et al._ 2017; Park _et al._ 2018; Lee _et al._ 2017) produce the probabilistic information about the future locations of vehicles over an occupancy grid map or samples by making use of an encoder-decoder structure. However, these sampling-based methods suffer from inherent inaccuracies due to discretization limits. Another method (Deo and Trivedi 2018) presents a model that outputs the multimodal distribution and then generates trajectories. Nevertheless, most of these methods require clear road lanes 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Instance Layer Category Layer<br>Pedestrian Bicycle Car<br>Instance node<br>Super node frame n frame n+1 frame n frame n+1<br>（a） （b） （c）<br>**----- End of picture text -----**<br>


Figure 2: Our _4D Graph_ for a traffic sequence. (a) Icons for instances and categories are shown on the left table. (b) The instance layer of the _4D Graph_ with spatial edges as solid lines and temporal edges as dashed lines. (c) The category layer with temporal edges of super nodes drawn by dashed lines. 

and simple driving scenarios without other types of trafficagents passing through. Based on images, (Chandra _et al._ 2018a) models the interactions between different trafficagents by a LSTM-CNN hybrid network for trajectory prediction. Taking into account the human-human interactions, some approaches (Alahi _et al._ 2016; Gupta _et al._ 2018; Vemula _et al._ 2017) use LSTM for predicting trajectories of pedestrians in a crowd and they show good performance on public crowd datasets. However, these methods are also limited in terms of trajectory prediction in complex traffic scenarios where the interactions are among not only pedestrians but also heterogeneous traffic-agents. 

## 

We assume each scene is preprocessed to get the categories and spatial coordinates of traffic-agents. At any time _t_ , the feature of the _i_ th traffic-agent _A[t] i_[can][be][denoted][as] _[f] i[ t]_[=] ( _x[t] i[, y] i[t][, c][t] i_[)][,][where][the][first][two][items][are][coordinates][in][the] x-axis and y-axis respectively, and the last item is the category of the traffic-agent. In our dataset, we currently take into account three types of traffic-agents, _ci ∈{_ 1 _,_ 2 _,_ 3 _}_ , where 1 stands for pedestrians, 2 represents bicycles and 3 denotes cars. Our approach can be easily extended to take into account more agent types. Our task is to observe features of all the traffic-agents in the time interval [1 : _Tobs_ ] and then predict their discrete positions at [ _Tobs_ +1 : _Tpred_ ]. 

## 

## _**4D Graph**_ **Generation** 

There are several datasets related to traffic scenes. Cityscapes (Cordts _et al._ 2016) contains 2D semantic, instance-wise, dense pixel annotations for 30 classes. ApolloScape (Huang _et al._ 2018) is a large-scale comprehensive dataset of street views that contains higher scene complexities, 2D/3D annotations and pose information, lane markings and video frames. However these two dataset do not provide trajectories information. The Simulation (NGSIM) dataset (Administration 2005) has trajectory data for cars, but the scene is limited to highways with similar simple road conditions. KITTI (Geiger _et al._ 2013) is a dataset for different computer vision tasks such as stereo, optical ow, 2D/3D object detection, and tracking. However, the total time of the dataset with tracklets is about 22 minutes. In addition, there are few intersection between vehicles, pedestrians and cyclists in KITTI, which makes it insufficient for exploring the motion patterns of traffic-agents in challenging traffic conditions. There are some pedestrian trajectory datasets like ETH (Pellegrini _et al._ 2009), UCY (Lerner _et al._ 2007), etc., but such datasets only focus on human crowds without any vehicles. 

## 

In this section, we present our novel algorithm to predict the trajectories of different traffic-agents. 

In urban traffic scenarios where various traffic-agents are interacting with others, each instance has its own state in relation to the interaction with others at any time and they also have continuous information in time series. Considering traffic-agents as instance nodes and relationships as edges, we can construct a graph in the instance layer, shown in Fig.2 (b). The edge between two instance nodes in one frame is called spatial edge (Jain _et al._ 2016; Vemula _et al._ 2017), which can transfer the interaction information between two traffic-agents in spatial space. The edge between the same instance in adjacent frames is the temporal edge, which is able to pass the historic information frame by frame in temporal space. The feature of the spatial edge ( _A[t] i[, A][t] j_[)][for] _[A][t] i_[can][be][computed][as] _[f] ij[ t]_[=][(] _[x][t] ij[, y] ij[t][, c][t] ij_[)][,] where _x[t] ij_[=] _[x][t] j[−][x][t] i_[,] _[y] ij[t]_[=] _[y] j[t][−][y] i[t]_[stands][for][the][rela-] tive position from _A[t] j_[to] _[A][t] i_[,] _[c][t] ij_[is][an][unique][encoding][for] ( _A[t] i[, A][t] j_[)][. When traffic-agent] _[ A][j]_[considers the spatial edge,] the spatial edge is represented as ( _A[t] j[, A] i[t]_[)][.][The][feature][of] the temporal edge ( _A[t] i[, A][t] i_[+1] ) is computed in the same way. It is normally observed that the same kind of trafficagents have similar behavior characteristics. For example, the pedestrians have not only similar velocities but also similar reactions to other nearby traffic-agents. These similarities will be directly reflected in their trajectories. We construct a super node _Cu[t][, u][∈{]_[1] _[,]_[ 2] _[,]_[ 3] _[}]_[for][each][kind][of][traffic-agent] 

to learn the similarities of their trajectories and then utilize that super node to refine the prediction for instances. Fig.2 (c) shows the graph in the category layer. All instances of the same type are integrated into one group and each group has an edge oriented toward the corresponding super node. After summarizing the motion similarities, the super node passes the guidance through an oriented edge to the group of instances. There are also temporal edges between the same super node in sequential frames. This category layer is specially designed for heterogeneous traffic and can make full use of the data to extract valuable information to improve the prediction results. This layer is very flexible and can be easily degenerated to situations when several categories disappear in some frames. 

Finally, we get the _4D Graph_ for a traffic sequence with two dimensions for traffic-agents and their interactions, one dimension for time series, and one dimension for high-level categories. By this _4D Graph_ , we construct an information network for the entire traffic. All the information can be delivered and utilized through the nodes and edges of the graph. 

## **Model Architecture** 

Our TrafficPredict algorithm is based on the _4D Graph_ , which consists of two main layers: the instance layer and the category layer. Details are given below. 

**Instance Layer** The instance layer aims to capture the movement pattern of instances in traffic. For each instance node _Ai_ , we have an LSTM, represented as _Li_ . Because different kinds of traffic-agents have different dynamic properties and motion rules, only instances of the same type share the same parameters. There are three types of traffic-agents in our dataset: vehicles, bicycles, and pedestrians. Therefore, we have three different LSTMs for instance nodes. We also distribute LSTM _Lij_ for each edge ( _Ai, Aj_ ) of the graph. All the spatial edges share the same parameters and all the temporal edges are classified into three types according to corresponding node type. 

For edge LSTM _Lij_ at any time _t_ , we embed the feature _fij[t]_[into][a][fixed][vector] _[e][t] ij_[,][which][is][used][as][the][input] to LSTM: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0004-06.png)


where _φ_ ( _·_ ) is an embedding function, _h[t] ij_[is the hidden state] also the output of LSTM _Lij_ , and _Wspa[e]_[are the embedding] weights, and _Wspa[r]_[are][LSTM][cell][weights,][which][contains] the movement pattern of the instance itself. LSTMs for temporal edges _Lii_ are defined in a similar way with parameters _Wtem[e]_[and] _[ W][ r] tem_[.] 

Each instance node may connect with several other instance nodes via spatial edges. However, each of the other instances has different impacts on the node’s behavior. We use a soft attention mechanism (Vemula _et al._ 2017) to distribute various weights for all the spatial edges of one instance node 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0004-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0004-10.png)


**----- Start of picture text -----**<br>
h2 Φ concat h1 Fuut ht-1uu<br>Instance c softmax  Temporal<br>  LSTM      LSTM<br>htu<br>h2 Φ concat h1 mean htuu<br>Φ<br>Instance<br>  LSTM c softmax concat Super node<br>     LSTM<br>h2 Φ concat h1 ht-1u<br>Instance<br>c softmax<br>  LSTM<br>**----- End of picture text -----**<br>


Figure 3: Architecture of the network for one super node in the category layer. 

where _Wi_ and _Wij_ are embedding weights, _Dot_ ( _·_ ) is the dot product, and ~~_√_~~ _mde_[is][a][scaling][factor][(][Vaswani] _[et][al.]_[2017][).] The final weights are ratios of _w_ ( _h[t] ij_[)][to][the][sum.][The][out-] put vector _Hi[t]_[is][computed][as][a][weighted][sum][of] _[h][t] ij_[.] _[H] i[t]_ stands for the influence exhibited on an instance’s trajectory by surrounding traffic-agents and _h[t] ii_[denotes][the][informa-] tion passing by temporal edges. We concatenate them and embed the result into a fixed vector _a[t] i_[.][The][node][features] _fi[t]_[and] _[ a][t] i_[can finally concatenate with each other to feed the] instance LSTM _Li_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0004-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0004-14.png)


where _Wins[e]_[and] _[ W][ a] ins_[are the embedding weights,] _[ W][ r] ins_[is] the LSTM cell weight for the instance node, _h_ 1 _[t] i_[is the first] hidden state of the instance LSTM. _h_ 2 _[t] i[−]_[1] is the final hidden state of the instance LSTM in the last frame, which will be described in next section. 

**Category Layer** Usually traffic-agents of the same category have similar dynamic properties, including the speed, acceleration, steering, etc., and similar reactions to other kinds of traffic-agents or the whole environment. If we can learn the movement patterns from the same category of instances, we can better predict trajectories for the entire instances. The category layer is based on the graph in Fig. 2(c). There are four important components: the super node for a specified category, the directed edge from a group of instances to the super node, the directed edge from the super node to instances, and the temporal edges for super nodes. 

Taking one super node with three instances as the example, the architecture in the category layer is shown in Fig. 3. Assume there are _n_ instances belonging to the same category in the current frame. We have already gotten the hidden state _h_ 1 and the cell state _c_ from the instance LSTM, which are the input for the category layer. Because the cell state _c_ contains the historical trajectory information of the instance, self-attention mechanism (Vaswani _et al._ 2017) is used on _c_ by softmax operation to explore the pattern of the 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0005-00.png)


Figure 4: Scenarios used for data collection: (a) Normal lanes with various traffic-agents. (b) Crossroads with different traffic-agents. 

internal sequence. At time _t_ , the movement feature _d_ for the _m_ th instance in the category is captured as follows. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0005-03.png)


Then, we obtain the feature _Fu[t]_[for the corresponding su-] per node _Cu[t]_[by computing the average of all the instances’] movement feature of the category. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0005-05.png)


_Fu[t]_[captures][valid][trajectory][information][from][instances] and learn the internal movement law of the category. Equation (7)-(8) show the process of transferring information on the directed edge from a group of instances to the super node. 

The feature _Fuu[t]_[of][the][temporal][edge][of][super][node][is] computed by _Fu[t][−][F][ t] u[−]_[1] . Take _Wst[e]_[as][embedding][weights] and _Wst[r]_[as the LSTM cell weights. The LSTM of the tem-] poral edge between the same super node in adjacent frames can be computed as follows. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0005-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0005-09.png)


Next, we integrate the information from the group of instances and the temporal edge as the input to the super node. We embed the feature _Fu[t]_[into fixed-length vectors and then] concatenate with _h[t] uu_[together. The hidden state] _[ h][t] u_[of super] node can be gotten by follows. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0005-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0005-12.png)


Finally, we describe the process of transferring guidance on the directed edge from the super node to instances. For the _m_ th instance in the group, the hidden state of the super node is concatenated with the first hidden state _h_ 1 _[t] m_[and then em-] bedded into a vector with the same length of _h_ 1 _[t] m_[. The sec-] ond hidden state _h_ 2 _[t] m_[is the final output of the instance node.] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0005-14.png)


where _Ws[r]_[is the embedding weights. By the network of the] category layer, we use the similarity inside the same type of instances to refine the prediction of trajectories for instances. 

Table 1: The acquisition time, total frames, total instances (count ID), average instances per frame, acquisition devices of NGSIM, KITTI (with tracklets) and our dataset. 

|Count||NGSIM|KITTI|Our|
|---|---|---|---|---|
|||||Dataset|
|duration(min)||45|22|**155**|
|frames(_×_103)||11.2|13.1|**93.0**|
||pedestrian|0|0.09|**16.2**|
|total (_×_103)|bicycle|0|0.04|**5.5**|
||vehicle|2.91|0.93|**60.1**|
||pedestrian|0|1.3|**1.6**|
|average (1/f)|bicycle|0|0.24|**1.9**|
||vehicle|**845**|3.4|12.9|
||camera|yes|yes|yes|
|device|lidar|no|yes|yes|
||GPS|no|yes|yes|



**Position estimation** We assume the position of the trafficagent in next frame meets a bivariate Gaussian distribution as (Alahi _et al._ 2016) with parameters including the mean _µ[t] i_[=][(] _[µ][x][, µ][y]_[)] _[t] i_[, standard deviation] _[ σ] i[t]_[=][(] _[σ][x][, σ][y]_[)] _[t] i_[and cor-] relation coefficient _ρ[t] i_[.][The][corresponding][position][can][be] represented as follows. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0005-19.png)


The second hidden state of traffic-agents at any time is used to to predict these parameters by linear projection. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0005-21.png)


The loss function is defined by the negative log-Likelihood _Li_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0005-23.png)


We train the model by minimizing the loss for all the trajectories in the training dataset. We jointly back-propagated through instance nodes, super nodes and spatial and temporal edges to update all the parameters to minimize the loss at each time-step. 

## **Experiments** 

## **Dataset** 

We use Apollo acquisition car (BaiduApollo 2018) to collect traffic data, including camera-based images and LiDARbased point clouds, and generate trajectories by detection and tracking. 

Our new dataset is a large-scale dataset for urban streets, which focuses on trajectories of heterogeneous traffic-agents for planning, prediction and simulation tasks. Our acquisition car runs in urban areas in rush hours in those scenarios shown in Fig. 4. The data is generated from a variety of sensors, including LiDAR (Velodyne HDL-64E S3), radar (Continental ARS408-21), camera, high definition maps and 

Table 2: The average displacement error and the final displacement error of the prior methods (ED, SL, SA) and variants of our method (TP) on our new dataset. For each evaluation metric, we show the values on pedestrians, bicycles, vehicles, and all the traffic-agents. We set the observation time as 2 seconds and the prediction time as 3 seconds for these measurements. 

|Metric|Methods|ED|SL|SA|TP-NoCL|TP-NoSA|TraffcPredict|
|---|---|---|---|---|---|---|---|
|Avg. disp. error|pedestrian<br>bicycle<br>vehicle<br>total|0.121<br>0.112<br>0.122<br>0.120|0.135<br>0.142<br>0.147<br>0.145|0.112<br>0.111<br>0.108<br>0.110|0.125<br>0.115<br>0.101<br>0.113|0.118<br>0.110<br>0.096<br>0.108|**0.091**<br>**0.083**<br>**0.080**<br>**0.085**|
|Final disp. error|pedestrian<br>bicycle<br>vehicle<br>total|0.255<br>0.190<br>0.195<br>0.214|0.173<br>0.184<br>0.202<br>0.198|0.160<br>0.170<br>0.189<br>0.178|0.188<br>0.193<br>0.172<br>0.187|0.178<br>0.169<br>0.150<br>0.165|**0.150**<br>**0.139**<br>**0.131**<br>**0.141**|



a localization system at 10HZ. We provide camera images and trajectory files in the dataset. The perception output information includes the timestamp, and the traffic-agent’s ID, category, position, velocity, heading angle, and bounding polygon. The dataset includes RGB videos with 100K 1920 _×_ 1080 images and around 1000km trajectories for all kinds of moving traffic agents. A comparison of NGSIM, KITTI (with tracklets), and our dataset is shown in Table. 1. Because NGSIM has a very large, top-down view, it has a large number of vehicles per frame. In this paper, each period of sequential sequences of the dataset was isometrically normalized for experiments. Our new dataset has been released over the WWW (Apolloscape 2018). 

## **Evaluation Metrics and Baselines** 

We use the following metrics (Pellegrini _et al._ 2009; Vemula _et al._ 2017) to measure the performance of algorithms used for predicting the trajectories of traffic-agents. 

1. _Average displacement error_ : The mean Euclidean distance over all the predicted positions and real positions during the prediction time. 

2. _Final displacement error_ : The mean Euclidean distance between the final predicted positions and the corresponding true locations. 

We compare our approach with these methods below: 

- _RNN ED (ED)_ : An RNN encoder-decoder model, which is widely used in motion and trajectory prediction for vehicles. 

- _Social LSTM (SL)_ : An LSTM-based network with social pooling of hidden states (Alahi _et al._ 2016). The model performs better than traditional methods, including the linear model, the Social force model, and Interacting Gaussian Processes. 

- _Social Attention (SA)_ : An attention-based S-RNN architecture (Vemula _et al._ 2017), which learn the relative influence in the crowd and predict pedestrian trajectories. 

- _TrafficPredict-NoCL (TP-NoCL)_ : The proposed method without the category layer. 

- _TrafficPredict-NoSA (TP-NoSA)_ : The proposed method without the self-attention mechanism of the category layer. 

## **Implementation Details** 

In our evaluation benchmarks, the dimension of hidden state of spatial and temporal edge cell is set to 128 and that of node cell is 64 (for both instance layer and category layer). We also apply the fixed input dimension of 64 and attention layer of 64. During training, Adam (Kingma and Ba 2014) optimization is applied with _β_ 1=0.9 and _β_ 2=0.999. Learning rate is scheduled as 0.001 and a staircase weight decay is applied. The model is trained on a single Tesla K40 GPU with a batch size of 8. For the training stability, we clip the gradients with the range -10 to 10. During the computation of predicted trajectories, we observe trajectories of 2 seconds and predict the future trajectories in next 3 seconds. 

## **Analysis** 

The performance of all the prior methods and our algorithm on heterogeneous traffic datasets is shown in Table. 2. We compute the average displacement error and the final displacement error for all the instances and we also count the error for pedestrians, bicycles and vehicles, respectively. The social attention (SA) model considers the spatial relations of instances and has smaller error than RNN ED and Social LSTM. Our method without category layer (TP-NoCL) not only considers the interactions between instances but also distinguishes between instances by using different LSTMs. Its error is similar to SA. By adding the category layer without self attention, the prediction results of TP-NoSA are more accurate in terms of both metrics. The accuracy improvement becomes is more evident after we use the selfattention mechanism in the design of category layer. Our algorithm, TrafficPredict, performs better in terms of all the metrics with about 20% improvement of accuracy. It means the category layer has learned the inbuilt movement patterns for traffic-agents of the same type and provides good guidance for prediction. The combination of the instance layer and the category layer makes our algorithm more applicable in heterogeneous traffic conditions. 

We illustrate some prediction results on corresponding 2D images in Fig. 5. The scenario in the image captured by the front-facing camera does not show the entire scenario. However, it is more intrinsic to project the trajectory results on the image. In most heterogeneous traffic scenarios, our algorithm computes a reasonably accurate predicted trajectory 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0007-00.png)


Figure 5: Illustration of our TrafficPredict (TP) method on camera-based images. There are six scenarios with different road conditions and traffic situations. We only show the trajectories of several instances in each scenario. The ground truth (GT) is drawn in green and the prediction results of other methods (ED,SL,SA) are shown with different dashed lines. The prediction trajectories of our TP algorithm (pink lines) are the closest to ground truth in most of the cases. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0007-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_images/cvpr_2024_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction.pdf-0007-03.png)


**----- Start of picture text -----**<br>
(a) (b)<br>**----- End of picture text -----**<br>


Figure 6: Illustration of some prediction results by our method. The ground truth of trajectories of vehicles, bicycles and pedestrians are drawn by blue, green and red respectively. Predicted locations are all represented by yellow stars. For each instance, first five discrete points are observed positions, but there are some overlaps in the illustration of pedestrian trajectories. 

and is close to the ground truth. If we have prior trajectories over a longer duration, the prediction accuracy increases. 

When traffic-agents are moving on straight lanes, it is easy to predict their trajectories because almost all the trafficagents are moving in straight direction. It is more challenging to provide accurate prediction in cross roads, as the agents are turning. Fig. 5 shows 2D experimental results of two sequences in cross areas. There are some overlaps on trajectories. In these scenarios, there are many curves with high curvature because of left turn. Our algorithm can compute accurate predicted trajectories in these cases. 

## **Conclusion** 

In this paper, we have presented a novel LSTM-based algorithm, TrafficPredict, for predicting trajectories for heterogeneous traffic-agents in urban environment. We use a in- 

stance layer to capture the trajectories and interactions for instances and use a category layer to summarize the similarities of movement pattern of instances belong to the same type and guide the prediction algorithm. All the information in spatial and temporal space can be leveraged and transferred in our designed _4D Graph_ . Our method outperforms previous state-of-the-art approaches in improving the accuracy of trajectory prediction on our new collected dataset for heterogeneous traffic. We have evaluated our algorithm in traffic datasets corresponding to urban dense scenarios and observe good accuracy. Our algorithm is realtime and makes no assumption about the traffic conditions or the number of agents. 

Our approach has some limitations. Its accuracy varies based on traffic conditions and the duration of past trajectories. In the future, we will consider more constraints, like the lane direction, the traffic signals and traffic rules, etc. to further improve the accuracy of trajectory prediction. Furthermore, we would like to evaluate the performance in more dense scenarios. 

## **Acknowledgement** 

Dinesh Manocha is supported in part by ARO Contract W911NF16-1-0085, and Intel. We appreciate all the people who offered help for collecting the dataset. 

## **References** 

- [Administration 2005] U.S. Federal Highway Administration. Us highway 101 dataset. 2005. 3 

- [Alahi _et al._ 2016] A. Alahi, K. Goel, V. Ramanathan, A. Robicquet, F.F. Li, and S. Savarese. Social lstm: Human trajectory prediction in crowded spaces. In _CVPR_ , pages 961–971, 2016. 2, 5, 6 

- [Altch´e and Fortelle 2017] F. Altch´e and A. De La Fortelle. An lstm network for highway trajectory prediction. In _ITSC_ , pages 353– 359. IEEE, 2017. 2 

- [Apolloscape 2018] Apolloscape. Trajectory dataset for urban traffic. 2018. http://apolloscape.auto/trajectory. html. 5 

- [BaiduApollo 2018] BaiduApollo. 2018. https://github. com/ApolloAuto/apollo. 5 

- [Bera _et al._ 2016] A. Bera, S. Kim, T. Randhavane, S. Pratapa, and D. Manocha. Glmp-realtime pedestrian path prediction using global and local movement patterns. In _ICRA_ , pages 5528–5535. IEEE, 2016. 2 

- [Bera _et al._ 2017] A. Bera, T. Randhavane, and D. Manocha. Aggressive, tense, or shy? identifying personality traits from crowd videos. In _IJCAI_ , volume 17, pages 112–118, 2017. 2 

- [Chandra _et al._ 2018a] R. Chandra, U. Bhattacharya, A. Bera, and D. Manocha. Traphic: Trajectory prediction in dense and heterogeneous traffic using weighted interactions. _arXiv:1812.04767_ , 2018. 2 

- [Chandra _et al._ 2018b] R. Chandra, T. Randhavane, U. Bhattacharya, A. Bera, , and D. Manocha. Deeptagent: Realtime tracking of highly heterogeneous agents in very dense traffic. 2018. http://gamma.cs.unc.edu/HTI/. 2 

- [Cheung _et al._ 2018] E. Cheung, A. Bera, E. Kubin, K. Gray, and Dinesh D. Manocha. Identifying driver behaviors using trajectory features for vehicle navigation. _IROS_ , 2018. 2 

- [Chung _et al._ 2015] J. Chung, K. Kastner, L. Dinh, K. Goel, A.C. Courville, and Y. Bengio. A recurrent latent variable model for sequential data. In _NIPS_ , pages 2980–2988, 2015. 2 

- [Cordts _et al._ 2016] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele. The cityscapes dataset for semantic urban scene understanding. In _CVPR_ , pages 3213–3223, 2016. 3 

- [Danielsson _et al._ 2007] S. Danielsson, L. Petersson, and A. Eidehall. Monte carlo based threat assessment: Analysis and improvements. In _IV_ , pages 233–238. IEEE, 2007. 2 

- [Deo and Trivedi 2018] N. Deo and M.M. Trivedi. Multi-modal trajectory prediction of surrounding vehicles with maneuver based lstms. _arXiv:1805.05499_ , 2018. 2 

- [Firl _et al._ 2012] J. Firl, H. St¨ubing, S.A. Huss, and C. Stiller. Predictive maneuver evaluation for enhancement of car-to-x mobility data. In _IV_ , pages 558–564. IEEE, 2012. 2 

- [Geiger _et al._ 2013] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun. Vision meets robotics: The kitti dataset. _IJRR_ , 32(11):1231–1237, 2013. 3 

- [Goodfellow _et al._ 2016] I. Goodfellow, Y. Bengio, A. Courville, and Y. Bengio. _Deep learning_ , volume 1. MIT press Cambridge, 2016. 2 

- [Graves and Jaitly 2014] A. Graves and N. Jaitly. Towards end-toend speech recognition with recurrent neural networks. In _ICML_ , pages 1764–1772, 2014. 2 

- [Gupta _et al._ 2018] A. Gupta, J. Johnson, F.F. Li, S. Savarese, and A. Alahi. Social gan: Socially acceptable trajectories with generative adversarial networks. In _CVPR_ , number CONF, 2018. 2 

- [Helbing and Molnar 1995] D. Helbing and P. Molnar. Social force model for pedestrian dynamics. _Physical review E_ , 51(5):4282, 1995. 2 

- [Huang _et al._ 2018] X. Huang, X. Cheng, Q. Geng, B. Cao, D. Zhou, P. Wang, Y. Lin, and R. Yang. The apolloscape dataset for autonomous driving. _CVPR Workshop_ , 2018. 3 

- [Kalman 1960] R. E. Kalman. A new approach to linear filtering and prediction problems. _Journal of basic Engineering_ , 82(1):35– 45, 1960. 1, 2 

- [Khosroshahi 2017] A. Khosroshahi. _Learning, Classification and Prediction of Maneuvers of Surround Vehicles at Intersections using LSTMs_ . PhD thesis, UC San Diego, 2017. 2 

- [Kim _et al._ 2017] B. Kim, C.M. Kang, S.H. Lee, H. Chae, J. Kim, C.C. Chung, and J.W. Choi. Probabilistic vehicle trajectory prediction over occupancy grid map via recurrent neural network. _ITSC_ , 2017. 2 

- [Kingma and Ba 2014] D. Kingma and J. Ba. Adam: A method for stochastic optimization. _arXiv:1412.6980_ , 2014. 6 

- [Lee _et al._ 2017] N. Lee, W. Choi, P. Vernaza, C.B. Choy, P.H. Torr, and M. Chandraker. Desire: Distant future prediction in dynamic scenes with interacting agents. In _CVPR_ , pages 336–345, 2017. 2 

- [Lef`evre _et al._ 2011] S. Lef`evre, C. Laugier, and J.Iba˜nez-Guzm´an. Exploiting map information for driver intention estimation at road intersections. In _IV_ , pages 583–588. IEEE, 2011. 2 

- [Lerner _et al._ 2007] A. Lerner, Y. Chrysanthou, and D. Lischinski. Crowds by example. In _CGF_ , volume 26, pages 655–664. Wiley, 2007. 3 

- [Ma _et al._ 2017] Y. Ma, X. Zhu, Y. Sun, and B. Yan. Image tagging by joint deep visual-semantic propagation. In _Pacific Rim Conference on Multimedia_ , pages 25–35. Springer, 2017. 2 

- [Ma _et al._ 2018] Y. Ma, D. Manocha, and W. Wang. Autorvo: Local navigation with dynamic constraints in dense heterogeneous traffic. _CSCS_ , 2018. 2 

- [Park _et al._ 2018] S. Park, B. Kim, C.M. Kang, C.C. Chung, and J.W. Choi. Sequence-to-sequence prediction of vehicle trajectory via lstm encoder-decoder architecture. _arXiv:1802.06338_ , 2018. 2 

- [Pellegrini _et al._ 2009] S. Pellegrini, A. Ess, K. Schindler, and L. Van Gool. You’ll never walk alone: Modeling social behavior for multi-target tracking. In _ICCV_ , pages 261–268. IEEE, 2009. 3, 6 

[Rasmussen and Williams 2006] C.E. Rasmussen and C. K. Williams. Gaussian processes for machine learning. 2006. _The MIT Press, Cambridge, MA, USA_ , 38:715–719, 2006. 1, 2 

   - [Toledo-Moreo and Zamora-Izquierdo 2009] R. Toledo-Moreo and M.A. Zamora-Izquierdo. Imm-based lane-change prediction in highways with low-cost gps/ins. _ITS_ , 10(1):180–185, 2009. 1 

   - [Treuille _et al._ 2006] A. Treuille, S. Cooper, and Z. Popovi´c. Continuum crowds. _TOG_ , 25(3):1160–1168, 2006. 2 

   - [Vaswani _et al._ 2017] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _NIPS_ , pages 5998–6008, 2017. 4 

   - [Vemula _et al._ 2017] A. Vemula, K. Muelling, and J. Oh. Social attention: Modeling attention in human crowds. _arXiv:1710.04689_ , 2017. 2, 3, 4, 6 

   - [Vinyals _et al._ 2015] O. Vinyals, A. Toshev, S. Bengio, and D. Erhan. Show and tell: A neural image caption generator. In _CVPR_ , pages 3156–3164, 2015. 2 

   - [Wang _et al._ 2008] J.M. Wang, D.J. Fleet, and A. Hertzmann. Gaussian process dynamical models for human motion. _TPAMI_ , 30(2):283–298, 2008. 2 

   - [Yamaguchi _et al._ 2011] K. Yamaguchi, A.C. Berg, L.E. Ortiz, and T.L. Berg. Who are you with and where are you going? In _CVPR_ , pages 1345–1352. IEEE, 2011. 2 

- [Jain _et al._ 2016] A. Jain, A.R. Zamir, S. Savarese, and A. Saxena. Structural-rnn: Deep learning on spatio-temporal graphs. In _CVPR_ , pages 5308–5317, 2016. 3 

