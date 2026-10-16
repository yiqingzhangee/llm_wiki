# **GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks** 

**Zhonghang Li**[1] _[,]_[2] **Lianghao Xia**[2] **Yong Xu**[1] _[,]_[3] _[,]_[4] **Chao Huang**[2] _[∗]_ 1 South China University of Technology, 2 University of Hong Kong, 3 PAZHOU LAB, 4 Guangdong Key Lab of Communication and Computer Network `{bjdwh.zzh,chaohuang75}@gmail.com aka_xia@foxmail.com yxu@scut.edu.cn` 

## **Abstract** 

In recent years, there has been a rapid development of spatio-temporal prediction techniques in response to the increasing demands of traffic management and travel planning. While advanced end-to-end models have achieved notable success in improving predictive performance, their integration and expansion pose significant challenges. This work aims to address these challenges by introducing a spatio-temporal pre-training framework that seamlessly integrates with downstream baselines and enhances their performance. The framework is built upon two key designs: (i) We propose a spatio-temporal mask autoencoder as a pre-training model for learning spatio-temporal dependencies. The model incorporates customized parameter learners and hierarchical spatial pattern encoding networks. These modules are specifically designed to capture spatio-temporal customized representations and intra- and inter-cluster region semantic relationships, which have often been neglected in existing approaches. (ii) We introduce an adaptive mask strategy as part of the pre-training mechanism. This strategy guides the mask autoencoder in learning robust spatio-temporal representations and facilitates the modeling of different relationships, ranging from intra-cluster to inter-cluster, in an easy-to-hard training manner. Extensive experiments conducted on representative benchmarks demonstrate the effectiveness of our proposed method. We have made our model implementation publicly available at `https://github.com/HKUDS/GPT-ST` . 

## **1 Introduction** 

Intelligent Transportation Systems (ITS) is a rapidly growing research field driven by advancements in sensors and communications technology [1, 12]. Spatio-Temporal (ST) prediction, including traffic flow and ride demand, aims to forecast future trends, facilitating decision-making in traffic management and risk response [43], and enhancing people’s lives with future travel plans [6]. Deep learning methods, such as RNNs and CNNs, have been widely adopted to model temporal and spatial correlations [5, 51, 50, 45]. Recently, GNNs have gained attention for spatial and temporal prediction tasks, achieving remarkable results. Initially, researchers built adjacency matrices based on region attributes, such as distance [48, 54]. Subsequent studies introduced learnable graph structures to capture spatial correlations [40, 39], while others focused on dynamic relationships between regions, proposing methods for constructing dynamic graphs [14, 40]. These advancements have significantly contributed to the development of spatio-temporal prediction techniques. 

Despite the remarkable results achieved by existing methods in spatio-temporal prediction, there are several issues that remain inadequately addressed. **i) Lack of customized representations of specific spatio-temporal patterns.** Customization can be categorized into two key aspects: _**timedynamic**_ and _**node-specific**_ properties in both the time and spatial domains. Time-dynamic patterns exhibit variations across different time periods, such as contrasting patterns between weekends and 

> _∗_ Corresponding author. 

37th Conference on Neural Information Processing Systems (NeurIPS 2023). 

weekdays within the same region. Moreover, correlations between regions dynamically evolve over time, surpassing the capabilities of static graph representations. Node-specific patterns highlight the distinct time series observed in different regions, rather than a shared pattern. Additionally, it is important to ensure that different regions retain their individual node characteristics even after message aggregation, to prevent interference from prominent nodes in the spatial domain. We believe that encoding all of these customized characteristics is essential to guarantee the model robustness. However, existing works often fall short in providing a comprehensive consideration of these factors. **ii) Insufficient consideration of spatial dependencies at different levels.** Most approaches primarily focus on pairwise associations between regions when modeling spatial dependency, but they overlook the semantic relevance at different spatial levels. In the real world, regions with similar functions tend to exhibit similar spatio-temporal patterns. By performing clustering analysis on different regions, the model can explore common characteristics among similar regions, thereby facilitating improved spatial representation learning. Moreover, there is a lack of adequate modeling of high-level regional relationships across time in current studies. Spatio-temporal patterns between different types of high-level regions may exhibit dynamic transfer relationships. For instance, during work hours, there is a notable movement of people from residential areas to work areas, as illustrated in the right part of Figure 1. In this context, changes in people flow within residential areas can provide valuable auxiliary signals for predicting people flow in work areas. It highlights the importance of incorporating both fine-grained and coarse-grained correlations among different levels of regions in order to enhance the predictive capabilities of spatio-temporal prediction models. 

An intuitive approach to address the aforementioned challenges is to develop an end-to-end model. However, the current models face difficulties in benefiting from such an approach, as each module in the state-of-the-art (SOTA) model is intricately refined, and any disassembly and integration may lead to a degradation in prediction performance. So, _is there a strategy that can allow existing spatiotemporal methods to leverage the advantages of an end-to-end model?_ The answer is affirmative. Recent pre-training frameworks, such as ChatGPT [3, 27] in the field of natural language processing (NLP) and MAE [15] in computer vision (CV), have been proposed by pioneers and widely studied in their respective domains. These pre-training architectures construct unsupervised training tasks, involving masking or other techniques, to learn better representations and improve 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0002-02.png)


**----- Start of picture text -----**<br>
CD 7-9am 9-12amWeek day<br>Work day<br>6-8pm<br>0.7<br>0.8 0.9 0.2 0.8 0.1<br>0.1<br>A C<br>B D<br>**----- End of picture text -----**<br>


Figure 1: The motivations behind GPT-ST. The upper left figure demonstrates that traffic patterns in the same region exhibit variations across different time periods. Similarly, the lower left figure depicts how this variation can occur between different regions during the same time period. In contrast, existing works typically construct static graphs, failing to capture the dynamic nature of region relationships over time, as shown in the middle figures (lower), where region relationships change dynamically. Lastly, the right figure illustrates the commuting phenomenon between regions belonging to different categories. 

downstream task performance. However, the application of such scalable pre-training models in the field of spatio-temporal prediction has been limited thus far. To address the aforementioned challenges, we propose a novel framework called Generative Pre-Training for Spatio-Temporal prediction (GPT-ST). This framework is designed to seamlessly integrate into existing spatio-temporal prediction models, enhancing their performance. Our contributions are summarized as follows: 

- We present GPT-ST, a novel pre-training framework designed for spatio-temporal prediction. This framework seamlessly integrates into existing spatio-temporal neural networks, resulting in performance enhancements. Our approach combines a model parameter customization scheme with a self-supervised masking autoencoding, enabling effective spatio-temporal pre-training. 

- GPT-ST ingeniously utilizes a hierarchical hypergraph structure to capture spatial dependencies at different levels from a global perspective. Through collaboration with a thoughtfully designed adaptive mask strategy, the model acquires the capability to model both intra-cluster and intercluster spatial relations among regions, thereby generating robust spatio-temporal representations. 

- We conduct extensive experiments on real-world datasets, and the demonstrated improvement in the performance of diverse downstream baselines showcases the superior performance of GPT-ST. 

2 

## **2 Related Work** 

**Deep Spatio-temporal Prediction Techniques.** Numerous studies have focused on designing deep neural networks for spatio-temporal data prediction. To capture temporal dependencies, D- LSTM [49] and GWN [40] have developed predictive frameworks based on Recurrent Neural Networks (RNNs) and Temporal Convolutional Networks (TCNs), respectively. GMAN [55] and STGNN [36] have utilized temporal self-attention networks to facilitate long-range temporal learning. In spatial relation learning, DMVST-Net [44] adopted convolutional network to model region associations. STGCN [48] and DCRNN [23] exploited Graph Neural Networks (GNNs) for message passing between regions. STAN [26] consider the interrelationships between regions based on Graph Attention Networks (GAT). Temporal graph networks serve as a similar research baseline aimed at reasoning about dynamic graph structures, such as CAWs [37] and PINT [33]. More recently, some works have explored the combination of ST models with advanced methods, such as Neural-ODEbased [10, 18] and self-supervised learning-based [24], for better spatial and temporal dependencies modelling. Compared to these works, our proposed GPT-ST framework aims at empower diverse deep spati-temporal models with the pre-training stage. 

**Pre-trainning methods.** Pre-training models have made significant advancements in recent years, with notable examples including BERT [7, 28], vision transformers [9, 8], masked autoencoders [15, 20], and language models [3, 27, 34]. The success of the autoencoder-based approach, exemplified by MAE, has demonstrated the powerful expressive capability of the mask-reconstruction pre-training framework. Building upon this, STEP [31] proposed a pre-training model specifically for time series prediction. In other domains, CMSF [41] introduced a master-slave framework for identifying urban villages. The master model pre-trains region representations, while the slave model fine-tunes specific regions for accurate identification. Researchers have also recognized the need for more efficient mask mechanisms than random masking. AttMask [19] and SemMAE [21] leverage attention mechanisms and semantic information, respectively, to guide the mask operation. AdaMAE [2] proposes adaptively selecting visible signals with greater reconstruction loss, as they contain more information, and using a higher proportion of masking strategy accordingly. Despite the significant progress of pre-training models in the fields of natural language processing (NLP) and computer vision (CV), their application to spatio-temporal prediction remains relatively limited. Additionally, computational efficiency has been a major bottleneck for transformer architectures in general. 

**Self-supervised learning methods for GNNs.** In recent years, self-supervised learning methods for graph data have gained significant attention [53, 38]. Contrastive learning-based GNNs generate different views of the original graph through data augmentation techniques [42, 29]. A loss function is then employed to maximize the consistency of positive sample pairs while minimizing the consistency of negative sample pairs across views. For example, GraphCL [47] generates two views of the graph by applying node dropping and edge shuffling, and performs contrastive learning between them. Another research direction focuses on generative graph neural networks, where the graph data itself serves as a natural supervisory signal for representation learning through reconstruction. GPTGNN [17] conducts pre-training by reconstructing graph features and edges, while GraphMAE [16] utilizes node feature masking in both the graph encoder and decoder to reconstruct features and learn graph representations. However, spatio-temporal forecasting tasks require simultaneous consideration of complex temporal evolution patterns and spatial correlation mechanisms. Pre-training paradigms specifically designed for such tasks are still an area of exploration and research. 

## **3 Preliminaries** 

**Spatio-Temporal Data X** . To capture the ST information, we represent it as a three-way tensor **X** _∈_ R _[R][×][T][ ×][F]_ , where _R_ , _T_ , and _F_ correspond to the numbers of regions, time slots, and feature dimensions, respectively. Each entry **X** _r,t,f_ is the _f_ -th feature of the _r_ -th region in the _t_ -th time slot. 

**Spatio-Temporal Hypergraphs** . We employ ST hypergraphs for ST modeling. A hypergraph _H_ = _{V, E,_ **H** _}_ is composed of three parts: **i)** Vertices _V_ = _{vr,t_ : _r ∈ R, t ∈ T }_ , each of which represents a region _r_ in a specific time slot _t_ . **ii)** _H_ hyperedges _E_ = _{e_ 1 _, ..., eH }_ , each of which connects multiple vertices to reflect the multipartite region-wise relations ( _e_ . _g_ ., all residential regions can be connected by one hyperedge). **iii)** The vertex-hyperedge connections **H** _∈_ R _[N][×][H]_ , where _N_ denotes the number of vertices. To fully excavate the potential of hypergraphs in region-wise relation learning, we adopt a learnable hypergraph scheme where **H** is derived from trainable parameters. 

3 

**Spatio-Temporal Pre-training Paradigm** . Our GPT-ST framework aims at developing a pre-trained ST representation method that facilitates the accuracy on downstream ST prediction tasks such as traffic volume forecasting. As depicted in Figure 2, the workflow of GPT-ST can be divided into the _pre-training stage_ and the _downstream task stage_ . We formulate the two stages as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Pre-training Stage Downstream Tasks Stage<br>෡Y<br>Mask AutoEncoder<br>Spatial-temporal Neural<br>Networks<br>Mask Operation Mask AutoEncoder<br>**----- End of picture text -----**<br>


i) The _pre-training stage_ of our GPT-ST framework adopts the masked autoencoding (MAE) task as the trainFigure 2: Overall workflow of GPT-ST. ing objective. The goal of this MAE task is to reconstruct masked information of the spatio-temporal data according to the unmasked ones by learning a ST representation function _f_ . For the training ST data during the period of [ _tK−L_ +1 _, tK_ ], we aim to minimize the following objective: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0004-03.png)


where _L_ denotes the measure of prediction deviation, **M** _∈{_ 0 _,_ 1 _}_ represents the mask tensor, and _⊙_ denotes the element-wise multiplication operation. **W** denotes a predictive linear layer. 

ii) After the pre-training stage, the results of our GPT-ST are then utilized in the _downstream task stage_ . The pre-trained model yields high-quality ST representations to facilitate the downstream prediction tasks such as traffic forecasting. Specifically, the downstream stage is formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0004-06.png)


where _**ζ**_ refers to the representation generated by _f_ according to the historical spatio-temporal data from the previous _L_ time slots before the _K_ -th time slot. The output is the predictions for the next _P_ time slots. Various existing spatio-temporal neural networks can serve as the prediction function _g_ . 

## **4 Methodology** 

This section introduces the technical details of the proposed ST pre-training framework GPT-ST. It develops a customized temporal encoder and a hierarchical spatial encoder to obtain large modeling capacity for the pre-trained model. An adaptive mask strategy is proposed to facilitate effective MAE training. Figure 3 demonstrates the model design of our pre-trained spatio-temporal encoding model. 

## **4.1 Customized Temporal Pattern Encoding** 

**Initial Embedding Layer** . We begin by constructing an embedding layer to initialize the representation of the spatio-temporal data **X** . The original data undergoes normalization using the Z-Score function [1, 39] and is then masked using the mask operation. Subsequently, a linear transformation is applied to augment the representation as **E** _r,t_ = **M** _r,t ⊙_ **X**[¯] _r,t ·_ **E** 0, where **E** _r,t ∈_ R _[d]_ , **M** _r,t ∈_ R _[F]_ , and **X**[¯] _r,t ∈_ R _[F]_ represent the representation, mask operation, and normalized spatio-temporal data for the _r_ -th region in the _t_ -th time slot. The variable _d_ denotes the number of hidden units. Additionally, **E** 0 _∈_ R _[F][ ×][d]_ represents the learnable embedding vectors for the _F_ feature dimensions. 

**Temporal Hypergraph Neural Network** . To facilitate global relation learning, we employ the hypergraph neural network for the temporal pattern encoding, specifically as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0004-13.png)


where **Γ** _t,_ **E**[¯] _t,_ **E** _t ∈_ R _[R][×][d]_ denote the result, intermediate, and initial region embeddings for the _t_ -th time slot, respectively. **W** _t ∈_ R _[d][×][d] ,_ **b** _t ∈_ R _[d]_ represent the _t_ -th time slot-specific parameters. _σ_ ( _·_ ) denotes the LeakyReLU activation. The intermediate embeddings **E**[¯] _∈_ R _[R][×][T][ ×][d]_ are calculated by the hypergraph information propagation. It employs the region-specific hypergraph **H** _r ∈_ R _[H][T][ ×][T]_ to propagate the temporal embeddings **E** _r ∈_ R _[T][ ×][d]_ for the _r_ -th region along the connections between time slots and _HT_ hyperedges, so as to capture the multipartite relations among time periods. 

**Customized Parameter Learner** . To characterize the diversity of temporal patterns, our temporal encoder conducts model parameter customization for both different regions and different time periods. Specifically, the aforementioned time-specific parameters **W** _t,_ **b** _t_ and the region-specific hypergraph 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Customized Parameter Learner Hypergraph Capsule Clustering Network Cross-class Relation Learning<br>𝐃𝑡 Time-dynamic parameters Inject t1t1t2 t1 … tn 𝐂𝑟 Region-specific parameters Inject ഥ𝐇 R1R2Rr Primary capsules … 𝐯𝑟,𝑡 = squash(𝐕𝚪 … 𝑟,𝑡 + 𝐜) … Regions clusteringLow-level  High-level Patterns  migration<br>Specific  t2 … Specific  …<br>Customized Temporal Pattern EncodingTime-HyperPropagate ത𝐄𝑟 = 𝜎 … 𝐇𝑟 [⊺] ⋅𝜎(𝐇𝑟 ⋅𝐄𝑟) 𝐬𝑖,𝑡 = ෍𝑟=1𝑅 𝑐𝑖,𝑟,𝑡 ⋅ത𝐯𝑖|𝑟,𝑡 𝑐𝑖,𝑟,𝑡Dynamic routing= σ𝑏exp(𝑏𝑟𝑖,𝑟,𝑡exp(𝑏𝑖,𝑟,𝑡𝑖,𝑟,𝑡) ) 𝑢= 𝐇Generate𝑖,𝑟,𝑡′ ⋅𝐯𝑟,𝑡 Clustering  Back to regions<br>R1 R2 … Rr ത𝐯𝑖|𝑟,𝑡 = 𝑠𝑞𝑢𝑎𝑠ℎ(𝑢) [⊺] ⋅𝐯𝑟,𝑡 High-level capsules Temporal<br>**----- End of picture text -----**<br>


Figure 3: The detailed framework of the proposed GPT-ST. 

parameters **H** _r_ are generated through a learnable process, rather than directly utilizing independent parameters. Specifically, the customized parameters are learned as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0005-03.png)


where **H**[¯] _∈_ R _[d][′][×][H][T][ ×][T] ,_ **W**[¯] _∈_ R _[d][′][×][d][×][d] ,_ **b**[¯] _∈_ R _[d][′][×][d]_ are independent parameters for the three generated parameters, respectively. **c** _r,_ **d** _t ∈_ R _[d][′]_ denote representations for the _r_ -th region and the _t_ -th time slot, respectively. **c** _r_ for _r ∈ R_ is a free-form parameter, while **d** _t_ for _t ∈ T_ is calculated from the normalized time-of-day features ¯ **z**[(] _t[d]_[)] and day-of-week features ¯ **z**[(] _t[w]_[)] . **e** 1 _,_ **e** 2 _∈_ R _[d][′]_ are their corresponding learnable embeddings. This parameter learner realizes spatial and temporal customization by generating parameters according to the features of specific time slots and regions. 

## **4.2 Hierarchical Spatial Pattern Encoding** 

## **4.2.1 Hypergraph Capsule Clustering Network** 

Current spatial encoders primarily focus on capturing local adjacent region-wise relations while disregarding the widespread similarities that exist between distant regions. For example, business districts that are geographically apart can still exhibit similar ST patterns. In light of this, our GPT-ST introduces a hypergraph capsule clustering network to capture global region-wise similarities. This network explicitly learns multiple cluster centers as hyperedges, characterizing the global region-wise similarities. To further enhance the hypergraph structure learning, we incorporate the dynamic routing mechanism of capsule network. This mechanism iteratively updates hyperedge representations and region-hyperedge connections based on semantic similarities. As a result, it improves the clustering capability of hyperedges and facilitates the global modeling of dependencies among regions. 

Specifically, we firstly obtain a normalized region embedding **v** _r,t ∈_ R _[d]_ for each region _r_ in time slot _t_ , using the previous embedding **Γ** _r,t_ and the squash function [30]. This embedding is then used to calculate the transferred information ¯ **v** _i|r,t ∈_ R _[d]_ from each region _r_ to each cluster center (hyperedge) _i_ , within the _t_ -th time slot. Formally, these two variables are calculated by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0005-09.png)


where **V** _∈_ R _[d][×][d]_ and **c** _∈_ R _[d]_ are free-form learnable parameters. The hypergraph connectivity matrix **H** _[′] t[∈]_[R] _[H][S][×][R]_[records the relations between] _[ R]_[ regions and] _[ H][S]_[hyperedges as the cluster centroids.][It] is tailored to the _t_ -th time slot using the aforementioned customized parameter learner, specifically by **H** _[′] t_[=][ softmax][(] **[d]** _′t⊤_ **[H]**[¯] _′_ ). Here, **d** _′t_[and] **[H]**[¯] _′_ are temporal features and hypergraph embeddings. 

**Iterative Hypergraph Structure Learning** . With the initialized region embeddings **v** _r,t_ and hypergraph connection embeddings ¯ **v** _i|r,t_ , we follow the dynamic routing mechanism of capsule network to enhance the clustering effect for the hyperedges. The _j_ -th iteration is described as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0005-12.png)


where **s** _i,t ∈_ R _[d]_ represents the iterative hyperedge embedding. It is calculated utilizing the iterative hyperedge-region weights _ci,r,t ∈_ R. The weight _ci,r,t_ is in turn calculated from the hyperedge embeddings **s** _i,t_ from the last iteration. With this iterative process, the relation scores and the hyperedge representations are mutually adjusted by each other, to better reflect the semantic similarities between regions and the spatial cluster centroids represented by hyperedges. 

5 

After the iterations of the dynamic routing algorithm, to make use of both _bi,r,t_ and **H** _[′] i,r,t_[for better] region-hyperedge relation learning, GPT-ST combines the two sets of weights and generates the final embeddings ¯ **s** _i,t ∈_ R _[d]_ . We begin by replacing _bi,r,t_ with ( _bi,r,t_ + **H** _[′] i,r,t_[)][to][obtain][a][new][weight] vector ¯ _ci,r,t ∈_ R, and then utilize ¯ _ci,r,t ∈_ R to calculate the final embeddings ¯ **s** _i,t_ . 

## **4.2.2 Cross-Cluster Relation Learning** 

With the clustered embeddings ¯ **s** _i,t_ , we propose to model the inter-cluster relations via a high-level hypergraph neural network. Specifically, the refined cluster embeddings **S**[ˆ] _∈_ R _[H][S][×][T][ ×][d]_ are calculated by message passing between the _HS_ cluster centroids and _HM_ high-level hyperedges as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0006-03.png)


where **S**[˜] _∈_ R _[H][S][T][ ×][d]_ denotes the reshaped embedding matrix obtained from ¯ **s** _i,t_ for _i ∈ HS_ and _t ∈ T_ . **H** _[′′] ∈_ R _[H][M][×][H][S][T]_ denotes the high-level hypergraph structure, which is obtained through the aforementioned personalized parameter learner. This parameter customization aggregates the temporal features ¯ **z**[(] _t[d]_[)] _[,]_[¯] **[z]**[(] _t[w]_[)] for all _t ∈ T_ as the input, and generates the parameters as in Eq 4. 

After refining the cluster representations ˆ **s** _i,t ∈_ R _[d]_ for _i ∈ HS, t ∈ T_ , we propagate the clustered embeddings back to the regional embeddings with the low-level hypergraph structure, as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0006-06.png)


where **Ψ** _r,t ∈_ R _[d]_ denotes the new region embedding for the _r_ -th region in the _t_ -th time slot. _ci,r,t ∈_ R denotes the weight for the low-level hypergraph capsule network. **W** _[′′] r[∈]_[R] _[d][×][d]_[and] **[ b]** _[′′] r[∈]_[R] _[d]_[denotes] the region-specific transformation and bias parameters generated by the customized parameter learner. 

## **4.3 Cluster-aware Masking Mechanism** 

Inspired by semantic-guided MAE [21], we design a cluster-aware masking mechanism to enhance the intra- and inter-cluster relation learning for our GPT-ST. The adaptive masking strategy incorporates the foregoing learned clustering information ¯ _ci,r,t_ to develop an easy-to-hard masking process. In particular, at the beginning of the training, we randomly mask a portion of regions for each cluster, in which case the masked values can be easily predicted by referring to in-cluster regions that share similar ST patterns. Subsequently, we gradually increase the masking proportion for certain categories, to increase the prediction difficulty for these clusters by reducing their relevant information. Finally, we completely mask the signals of some clusters, facilitating the ability of cross-cluster knowledge transfer for the pre-trained model. This adaptive masking process is depicted in Figure 4. 

However, directly utilizing the learned cluster information _c_ ¯ _i,r,t_ to **Cluster-aware Masking Strategy** generate the cluster-aware mask **M** is not feasible. This is because the cluster information is calculated by deeper layers of our GPTCategory Unmasked ST network, while the generated mask is required as input to the network ( **M** _⊙_ **X** ). To address this, we employ a two-layer MLP network with customized parameters to predict the learned results of _c_ ¯ _i,r,t_ . In particular, the transformations and bias vectors in this MLP network are replaced with time-dynamic and node-specific parameters generated by the customized parameter learner (Eq 4). Subsequently, a linear layer and softmax( _·_ ) function are used to obtain the predictions _qi,r,t ∈_ R for _i ∈ HS, r ∈ R, t ∈ T_ . To optimize the distribution of _q_ truthis _i,r,t_ prevented, we utilize the KL divergence loss function _c_ ¯ _i,r,t_ . It is important to note that the backpropagation ofin this step. The category is the maximum _Lkl_ with the groundprobability _c_ ¯ _i,r,t_ Figure 4:adaptive masking strategy.adaptive masking strategy.Illustration for the according to _qi,r,t_ is taken as the classification result. See Algorithm 1 in A.2 for more details. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0006-11.png)


**----- Start of picture text -----**<br>
Cluster-aware Masking Strategy<br>Category Unmasked Masked<br>From-easy-to-hard<br>**----- End of picture text -----**<br>


Figure 4:adaptive masking strategy.adaptive masking strategy.Illustration for the 

## **5 Evaluation** 

In this section, we verify the effectiveness of GPT-ST by answering the following five questions: 

- **RQ1** : How does our proposed pre-training framework GPT-ST benefit different downstream baseline methods, in terms of prediction accuracy on diverse spatio-temporal prediction tasks? 

6 

- **RQ2** : How effective are the designed technical modules of our GPT-ST framework? 

- **RQ3** : How are the interpretation ability of the learned global region clusters and the inter-cluster transition patterns in the hierarchical spatial pattern encoding of our GPT-ST framework. 

- **RQ4** : How efficient is GPT-ST in both the pre-training stage and the downstream task stage? 

- **RQ5** : How does varying the mask ratio setting affect the performance of GPT-ST? 

## **5.1 Experimental Setting** 

To evaluate the effectiveness of our proposed method, we conduct experiments on four real-world ST datasets including: **PEMS08** [32], **METR-LA** [23], **NYC Taxi** and **NYC Citi Bike** [46], which records the traffic flow, traffic speed, taxi order records, and bike order records, respectively. 

We choose Mean Absolute Error (MAE), Root Mean Square Error (RMSE) and Mean Absolute Percentage Error (MAPE) as the evaluation metrics, which are widely used in ST prediction tasks [52]. The lower value of the metrics, the better performance of the model. Following previous works [13, 32, 22], the number of time slots _L_ is set to 12. We divide METR-LA dataset into training set, validation set and test set in a ratio of 7:1:2 and 6:2:2 for others datasets. The final model parameters are selected according to the optimal effect (minimum MAE value) of the validation set. 

For optimization, in the pre-trainning phase, the absolute error loss function _Lr_ is used to optimize the parameters and a hyperparameter _λ_ is adopted to balance the weight of _Lr_ and _Lkl_ . In the downstream task stage, we firstly fuse the embedding _**ζ**_ and the raw signal **X** as the input to the downstream model, and the optimization strategies are distinct for different downstream models. Detailed descriptions of datasets and evaluation are presented in the supplementary material A.1. 

## **5.2 Main Results (RQ1)** 

In this section, we mainly investigate whether GPT-ST improves the forecasting for downstream tasks. To achieve this, we evaluate both the original performance and enhanced performance (w/ GPT-ST) of different baselines on four datasets, and the results are shown in Table 1. Note that since DMVSTGCN, STMGCN and CCRNN are specially designed for demand forecasting, we eliminate them in traffic forecasting tasks. The results indicate that the proposed model significantly improves the prediction performance of different downstream baselines on all datasets, which confirms the effectiveness of the our framework. We analyze the promotion effect of GPT-ST from three dimensions: 

**Consistent improvements by GPT-ST** . We observe that GPT-ST improves different types of methods ( _e_ . _g_ ., GNN-based, or attention-based models), and such positive effect does not partialize to a certain category of baselines, which verifies the generalization ability of GPT-ST. We attribute this improvement to the MAE pre-training with the awareness of intra- and inter-cluster ST relations. 

**Difference among baseline models.** When compared to recently proposed methods like MSDR, our GPT-ST exhibits more substantial improvements when applied to classical baselines such as STGCN. Specifically, in traffic flow prediction using the PEMS08 dataset, MSDR shows improvements of 0.55 in MAE, 0.60 in RMSE, and 0.21% in MPAE, while STGCN demonstrates improvements of 1.61 in MAE, 2.71 in RMSE, and 0.63% in MPAE. One possible explanation for these findings is that advanced baselines like MSDR are already well-designed and comprehensive in modeling various factors. Consequently, they are capable of encoding abundant knowledge independently, which may diminish the utility of the additional signals provided by the pre-training model. Conversely, classical baselines like STGCN may derive greater benefits from the insights offered by our GPT-ST due to their simpler design and potentially limited capacity to capture complex relationships. 

**Comparison with pre-training methods.** In addition to evaluating our pre-training method against spatio-temporal prediction methods without pre-training, we also compare it with the competitive pre-training baseline STEP. This method utilizes long-term time series as input for pre-training to enhance the performance of the downstream model (GWN). Despite our GPT-ST using only short-term data during the pre-training stage, it outperforms STEP on several metrics. Furthermore, our GPT-ST demonstrates an even larger performance advantage in scenarios where long-term data is insufficient, _e_ . _g_ ., the taxi and bike demand prediction tasks. This highlights the broader applicability of our proposed GPT-ST framework, emphasizing its ability to excel in various settings. 

7 

Table 1: Overall performance comparison on different datasets in terms of _MAE_ , _RMSE_ and _MAPE_ . 

|Model|Dataset<br>PEMS08<br>METR-LA<br>NYC Taxi<br>NYC Citi Bike|Dataset<br>PEMS08<br>METR-LA<br>NYC Taxi<br>NYC Citi Bike|Dataset<br>PEMS08<br>METR-LA<br>NYC Taxi<br>NYC Citi Bike|Dataset<br>PEMS08<br>METR-LA<br>NYC Taxi<br>NYC Citi Bike|
|---|---|---|---|---|
||Metrics<br>MAE<br>RMSE<br>MAPE<br>MAE<br>RMSE<br>MAPE<br>MAE<br>RMSE<br>MAPE<br>MAE<br>RMSE<br>MAPE||||
|ST<br>w/ G|GCN<br>17.85<br>28.64<br>11.08%<br>PT-ST<br>**16.24**<br>**25.93**<br>**10.45%**|3.55<br>7.39<br>10.02%<br>**3.29**<br>**6.85**<br>**9.14%**|5.68<br>11.55<br>38.10%<br>**5.09**<br>**10.05**<br>**35.21%**|2.04<br>3.34<br>51.10%<br>**1.82**<br>**2.82**<br>**49.15%**|
|DMV<br>w/ G|STNET<br>-<br>-<br>-<br>PT-ST<br>-<br>-<br>-|-<br>-<br>-<br>-<br>-<br>-|7.73<br>14.26<br>69.86%<br>**5.27**<br>**9.18**<br>**40.56%**|2.27<br>3.65<br>61.89%<br>**1.89**<br>**2.86**<br>**55.01%**|
|T<br>w/ G|GCN<br>21.44<br>31.82<br>15.67%<br>PT-ST<br>**17.34**<br>**26.53**<br>**12.26%**|3.70<br>7.28<br>10.65%<br>**3.16**<br>**6.41**<br>**8.88%**|9.73<br>19.39<br>83.50%<br>**6.88**<br>**13.80**<br>**51.84%**|2.23<br>3.72<br>62.90%<br>**2.00**<br>**3.22**<br>**58.84%**|
|AS<br>w/ G|TGCN<br>17.55<br>26.97<br>11.78%<br>PT-ST<br>**15.89**<br>**24.87**<br>**10.28%**|3.57<br>7.13<br>10.19%<br>**3.10**<br>**6.26**<br>**8.74%**|6.54<br>11.81<br>52.17%<br>**5.90**<br>**10.38**<br>**45.78%**|2.06<br>3.35<br>55.18%<br>**1.90**<br>**2.93**<br>**53.60%**|
|G<br>w/ G|WN<br>15.29<br>24.59<br>9.98%<br>PT-ST<br>**14.72**<br>**24.12**<br>**9.42%**|3.15<br>6.44<br>8.67%<br>**3.02**<br>**6.18**<br>**8.29%**|5.64<br>9.92<br>42.19%<br>**5.12**<br>**8.89**<br>**37.49%**|1.90<br>2.98<br>53.00%<br>**1.80**<br>**2.70**<br>**51.69%**|
|ST<br>w/ G|MGCN<br>-<br>-<br>-<br>PT-ST<br>-<br>-<br>-|-<br>-<br>-<br>-<br>-<br>-|6.53<br>12.20<br>49.38%<br>**5.18**<br>**9.00**<br>**38.38%**|1.96<br>3.12<br>55.99%<br>**1.81**<br>**2.73**<br>**52.95%**|
|MT<br>w/ G|GNN<br>15.28<br>24.70<br>9.72%<br>PT-ST<br>**14.93**<br>**24.46**<br>**9.67%**|3.12<br>6.44<br>8.60%<br>**2.99**<br>**6.19**<br>**8.21%**|5.15<br>9.19<br>36.02%<br>**4.90**<br>**8.55**<br>**34.97%**|1.84<br>2.82<br>51.45%<br>**1.78**<br>**2.67**<br>**51.16%**|
|ST<br>w/ G|SGCN<br>17.82<br>27.88<br>11.49%<br>PT-ST<br>**16.29**<br>**26.14**<br>**10.59%**|3.58<br>7.36<br>9.95%<br>**3.16**<br>**6.51**<br>**8.74%**|6.02<br>10.98<br>46.87%<br>**5.00**<br>**8.77**<br>**36.28%**|2.04<br>3.19<br>56.73%<br>**1.88**<br>**2.79**<br>**55.23%**|
|ST<br>w/ G|FGNN<br>17.13<br>27.32<br>11.10%<br>PT-ST<br>**15.89**<br>**25.78**<br>**10.53%**|3.29<br>6.69<br>9.14%<br>**3.13**<br>**6.47**<br>**8.84%**|5.91<br>10.64<br>46.32%<br>**4.97**<br>**8.69**<br>**36.37%**|2.03<br>3.17<br>56.51%<br>**1.86**<br>**2.74**<br>**54.84%**|
|ST<br>w/ G|GODE<br>17.73<br>27.53<br>11.39%<br>PT-ST<br>**15.68**<br>**24.84**<br>**10.46%**|3.47<br>7.02<br>9.93%<br>**3.19**<br>**6.44**<br>**9.30%**|6.62<br>12.14<br>55.03%<br>**5.61**<br>**9.91**<br>**43.09%**|2.08<br>3.27<br>56.47%<br>**1.90**<br>**2.89**<br>**52.86%**|
|CC<br>w/ G|RNN<br>-<br>-<br>-<br>PT-ST<br>-<br>-<br>-|-<br>-<br>-<br>-<br>-<br>-|5.41<br>10.51<br>38.47%<br>**5.08**<br>**9.47**<br>**37.04%**|1.92<br>3.01<br>52.44%<br>**1.82**<br>**2.77**<br>**51.33%**|
|M<br>w/ G|SDR<br>16.47<br>25.63<br>10.54%<br>PT-ST<br>**15.92**<br>**25.03**<br>**10.33%**|3.19<br>6.42<br>8.83%<br>**3.07**<br>**6.27**<br>**8.50%**|5.73<br>10.45<br>42.33%<br>**5.27**<br>**9.33**<br>**38.90%**|1.93<br>2.98<br>53.44%<br>**1.82**<br>**2.74**<br>**52.78%**|
|S<br>w/ G|TWA<br>16.16<br>25.80<br>10.55%<br>PT-ST<br>**15.29**<br>**24.50**<br>**9.81%**|3.19<br>6.56<br>8.68%<br>**3.09**<br>**6.32**<br>**8.59%**|5.25<br>9.34<br>38.08%<br>**5.06**<br>**8.82**<br>**36.63%**|1.89<br>2.96<br>50.07%<br>**1.83**<br>**2.80**<br>**49.91%**|
|G<br>w/<br>w/ G|WN*<br>13.97<br>22.94<br>9.28%<br>STEP<br>14.08<br>23.33<br>**9.07%**<br>PT-ST<br>**13.96**<br>**22.83**<br>9.16%|2.95<br>5.99<br>8.01%<br>2.92<br>**5.91**<br>7.98%<br>**2.92**<br>5.97<br>**7.93%**|5.18<br>9.08<br>35.75%<br>5.33<br>9.24<br>36.17%<br>**5.07**<br>**8.82**<br>**34.97%**|2.02<br>3.12<br>55.16%<br>2.02<br>3.12<br>53.86%<br>**1.98**<br>**3.07**<br>**51.44%**|



## **5.3 Model Ablation Study (RQ2)** 

This section investigates the impact of major components in our GPT-ST. To evaluate the impact of different components, we re-conduct pre-training for multiple ablated variants, and evaluate the performance of the downstream method using the newly pre-trained ablated models. GWN is utilized as the downstream approach, and two datasets METR-LA and NYC Taxi are employed for evaluation. The results are shown in Figure 5. From the results, we make the following observations: 

**Impact of basic components.** i) -P. We remove the customized parameter learner in GPT-ST. ii) -C. We disable the hypergraph capsule clustering network to cancel the clustering operation. In this case, the spatial hypergraph NNs works on fine-grained regions directly. iii) -T. We leave the crosscluster relation learning unexplored by removing the high-level spatial hypergraph. The noticeable performance decay demonstrates the positive benefits brought by all three components, indicating that generating personalized ST parameters and modeling the intra- and inter-cluster ST dependencies can effectively capture complex ST correlations to benefit the predictions. Among the three components, the removal of the hypergraph capsule clustering network causes the most significant performance drop. This is because the clustering results also play an important role in many other components, including the cross-cluster dependencies and the cluster-aware adaptive masking. 

**Impact of mask mechanism.** i) Ran0.25 and Ran0.75. We replace our adaptive mask strategy with random masking with a mask ratio of 0.25 (mask ratio same as ours) and 0.75 (mask ratio used in MAE [15] and STEP [31]), respectively. The results consistently demonstrate that our proposed mask strategy outperforms the random mask strategies. This can be attributed to the fact that our mask strategy effectively promotes the learning of intra- and inter-cluster relationships by GPT-ST, resulting in the generation of high-quality representations. Similar results are obtained in additional experiments conducted on the other two datasets, as shown in Table 2. ii) GMAE and AdaMAE. We compare our approach to two variants using the mask strategies proposed by GraphMAE [16] and AdaMAE [2]. Both variants perform worse than our methods, highlighting the importance of considering spatial and temporal patterns in the masking strategy. This further confirms the superiority of our adaptive masking approach that leverages clustering information. 

**Impact of pre-training strategy.** To further investigate the effectiveness of the mask-reconstruction pre-training approach in spatio-temporal pre-training, we compare it to other pre-training methods: local-global infomax and contrastive pre-training. We used DGI [35] and GraphCL [47] as baselines, which are widely recognized pre-training strategies for GNNs. By replacing GPT-ST’s pre-training strategy with these methods (denoted as "r/"), we observe improvements in performance compared to 

8 

models without pre-training. This indicates the adaptability and benefits of infomax and contrastive pre-training for representation learning in our model. However, our approach, which utilizes the maskreconstruction task, achieved the most significant performance enhancement. This can be attributed to the higher correlation between the mask-reconstruction task and the downstream regression task, leading to more effective learning of spatio-temporal representations. Additionally, our adaptive mask strategy plays a crucial role in facilitating the model to learn robust spatio-temporal representations by increasing the difficulty of the pre-training task. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0009-01.png)


**----- Start of picture text -----**<br>
-P -C -T Ran0.25 Ran0.75 GMAE AdaMAE r/ DGI r/ GCL Ours Table 2: Comparison of Ran & Ada mask.<br>3.1 6.4 5.45.3 9.2 Datasetmodel GWN PeMS08(MSDRMAE/ RMSE) STWA<br>6.3 9.0 Ran mask 14.80/ 24.34 16.50/ 25.63 15.39/ 24.85<br>3.0 6.2 5.25.1 8.8 Ada maskDataset 14.72 /  24.12 NYC Citi Bike 15.92 (/  25.03 MAE/ RMSE 15.29 ) /  24.50<br>6.1 8.6 model DMVSTNET STMGCN CCRNN<br>2.9 METR-LA METR-LA 5.0 NYC TAXI NYC TAXI Ran mask 1.92/ 2.89 1.82/ 2.74 1.85/ 2.86<br>Ada mask 1.89 /  2.86 1.81 /  2.73 1.82 /  2.77<br>MAE RMSE MAE RMSE<br>**----- End of picture text -----**<br>


Figure 5: Ablation study of GPT-ST. 

## **5.4 Investigation on Clustering Effect (RQ3)** 

To comprehensively demonstrate the effectiveness of the clustering process, we assess the interpretability of our GPT-ST framework by analyzing the embeddings generated by the Hypergraph Capsule Clustering Network (HCCN), as illustrated in Figure 6. We employ the T-SNE algorithm to visualize the high-dimensional embeddings produced by HCCN, mapping them to 2-dimensional vectors. Each category is represented by a distinct color, with 10 categories defined for hyperparameter _HS_ . The region clustering is determined based on the probabilities of belonging to different categories. Upon examining the visualized embeddings, we observe that regions belonging to the same class exhibit tight clustering within a confined space, providing empirical evidence of the robust clustering capability of our hypergraph capsule clustering network. 

In another case study conducted on the METR-LA dataset, we investigate the in-cluster region relations derived from the hypergraph capsule clustering network, as well as the cross-class dependencies obtained from the cross-cluster hypergraph network. The results, illustrated in Figure 7, demonstrate that the top-3 regions within the same category, as depicted in Figure 7(a) and 7(b), exhibit similar traffic patterns, indicating shared functionalities. For instance, regions near commercial areas (7(a)) experience an evening peak, while those near residential areas (7(b)) remain relatively stable. These observations align with real-world scenarios. Furthermore, focusing on Figure 7(c), we analyze the top-2 regions from two categories that have undergone shifts in traffic patterns over a specific time period, while sharing similar hyperedge weights in the cross-class transition. The results reveal that regions undergoing pattern shifts exhibit distinct traffic patterns while maintaining close interconnections within a short driving distance. These findings provide further evidence of the cross-cluster transition learning’s ability to capture semantic-level relationships between regions, reflecting real-world traffic scenarios. These advantages contribute to the generation of high-quality representations in our GPT-ST framework, leading to improved performance in downstream tasks. 

## **5.5 Model Efficiency Study (RQ4)** 

In this section, we assess the efficiency of our model. Table 3: Computational time cost investigation. Specifically, we measure the per-epoch training time of all methods on the PEMS08 dataset, and STGCNModel time/epoch7.72(s) ModelGWN time/epoch9.87(s) the results are summarized in Table 3. To ensure w/ GPT-ST 12.52 w/ GPT-ST 13.46 fairness, all experiments are conducted on a system w/ GPT-STASTGCN 12.4718.20 w/ GPT-STTGCN 14.047.63 equipped with a GTX 3090 GPU and an Intel Core MTGNN 9.40 STSGCN 18.71 i9-12900K CPU. The batch size for all methods, exw/ GPT-ST 13.02 w/ GPT-ST 21.43 STFGNN 16.03 STGODE 22.93 cept those tagged with ’b8’, is set to 64 (where ’b8’ w/ GPT-ST 19.54 w/ GPT-ST 19.45 denotes a batch size of 8). By combining the results MSDR 29.15 STWA 46.72 w/ GPT-ST 33.23 w/ GPT-ST 49.90 from Table 1, we can conclude that while most of Downstream tasks stage Pre-training stage the baselines experience a minor decrease in trainGWN* b8 210.98 STEP b8 327.78 ing efficiency when enhanced by GPT-ST, there is a w/ GPT-ST b8w/ STEP b8 425.12220.29 GPT-ST b8 _GPT-ST_ 12.5040.18 significant improvement in their performance. For instance, some previous methods ( _e_ . _g_ ., STGCN) equipped with GPT-ST achieve comparable perfor- 

9 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0010-00.png)


Figure 6: Visualized embbeddings of the hypergraph capsule clustering network. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0010-02.png)


**----- Start of picture text -----**<br>
Residential district<br>Commercial district<br>Commercial district Residential district<br>(a) Case A for learned clusters (b) Case B for learned clusters (c) Case for cross-class transitions<br>Traffic speed<br>**----- End of picture text -----**<br>


Figure 7: Case study for the spatial encoder. (a) and (b) present the real traffic conditions and their geographic locations belonging to two specific categories learned by the hypergraph capsule network; (c) illustrates the learned traffic transition relation between two categories of regions. 

mance to advanced methods ( _e_ . _g_ ., MSDR), while maintaining superior training efficiency. In the comparison between GPT-ST and STEP, we observe that STEP incurs a substantial time cost, both in the pre-training stage and the downstream task stage, whereas GPT-ST enhances existing baselines in a lightweight manner. These cases demonstrate that our framework achieves a win-win situation in terms of performance and training efficiency, making it well-suited for practical applications. 

## **5.6 Model Hyperparameter Experiment (RQ5)** 

To examine the impact of different hyperparameters on GPT-ST, we conduct parameter experiments on the NYC Taxi dataset using GWN as the downstream model. The results of these experiments are depicted in Figure 8, where the evaluation metrics are plotted as relative values. 

In our investigation, we specifically focus on the total mask ratio _rt_ and adaptive mask ratio _ra_ as the main parameters of interest. Based on the findings presented in Figure 8, we observe that the optimal effect is achieved when the total mask ratio _rt_ is set to 0.25. Increasing the mask ratio beyond this value does not improve the model’s ability to learn better representations; instead, it leads to a decline in prediction performance. Furthermore, we explore the impact of the adaptive mask ratio _rr_ , which determines the number of clusters that are entirely masked, thereby influencing the entire training process. The best performance is obtained when the adaptive mask ratio is set to 1. This suggests that this specific configuration enhances the model’s ability to learn regional relations, both within individual clusters and across different clusters. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0010-08.png)


**----- Start of picture text -----**<br>
MAE<br>RMSE<br>0.0 0.2 0.4 0.6 0.8 1.0<br>(a) Total mask ratio  mt<br>MAE<br>RMSE<br>0.25 0.50 0.75 1.00<br>(b) Adaptive mask ratio  ma<br>.06<br>.04<br>.02<br>.00<br>.02<br>.01<br>.00<br>Relative values<br>Relative values<br>**----- End of picture text -----**<br>


Figure 8: Hyperparameter study of _mt_ and _ma_ . 

## **6 Conclusion** 

In this work, we introduce a scalable and effective pre-training framework tailored for spatio-temporal prediction tasks. Our framework starts with a foundational pre-training model that focuses on capturing spatio-temporal dependencies. It leverages a customized parameter learner and hierarchical hypergraph networks to extract customized spatio-temporal features and region-wise semantic associations, respectively. To further improve the model’s performance, we propose an adaptive masking strategy. This strategy guides the model to learn inferential reasoning capabilities by considering both intra-class and inter-class relations during the pre-training stage. We conduct extensive experiments on four real-world datasets, demonstrating the efficacy of our proposed GPT-ST framework in enhancing the performance of downstream baselines across different spatio-temporal tasks. 

10 

## **References** 

- [1] Lei Bai, Lina Yao, Can Li, Xianzhi Wang, and Can Wang. Adaptive graph convolutional recurrent network for traffic forecasting. In _NeurIPS_ , pages 17804–17815, 2020. 

- [2] Wele Gedara Chaminda Bandara, Naman Patel, Ali Gholami, Mehdi Nikkhah, Motilal Agrawal, and Vishal M. Patel. Adamae: Adaptive masking for efficient spatiotemporal learning with masked autoencoders. In _NeurIPS_ , 2022. 

- [3] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, et al. Language models are few-shot learners. In _NeurIPS_ , 2020. 

- [4] Razvan-Gabriel Cirstea, Bin Yang, Chenjuan Guo, Tung Kieu, and Shirui Pan. Towards spatio- temporal aware traffic time series forecasting. In _ICDE_ , pages 2900–2913, 2022. 

- [5] Zhiyong Cui, Kristian Henrickson, Ruimin Ke, and Yinhai Wang. Traffic graph convolutional recurrent neural network: A deep learning framework for network-scale traffic learning and forecasting. _TITS_ , pages 4883–4894, 2020. 

- [6] Rui Dai, Shenkun Xu, Qian Gu, Chenguang Ji, and Kaikui Liu. Hybrid spatio-temporal graph convolutional network improving traffic prediction with navigation data. In _KDD_ , pages 3074–3082, 2020. 

- [7] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In _NAACL_ , pages 4171–4186, 2019. 

- [8] Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Weiming Zhang, Nenghai Yu, Lu Yuan, Dong Chen, and Baining Guo. Cswin transformer: A general vision transformer backbone with cross-shaped windows. In _CVPR_ , pages 12124–12134, 2022. 

- [9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_ , 2021. 

- [10] Zheng Fang, Qingqing Long, Guojie Song, and Kunqing Xie. Spatial-temporal graph ode networks for traffic flow forecasting. In _KDD_ , pages 364–373, 2021. 

- [11] Xu Geng, Yaguang Li, Leye Wang, Lingyu Zhang, Qiang Yang, Jieping Ye, and Yan Liu. Spatiotemporal multi-graph convolution network for ride-hailing demand forecasting. In _AAAI_ , pages 3656–3663, 2019. 

- [12] Kan Guo, Yongli Hu, Yanfeng Sun, Sean Qian, Junbin Gao, and Baocai Yin. Hierarchical graph convolution network for traffic forecasting. In _AAAI_ , pages 151–159, 2021. 

- [13] Shengnan Guo, Youfang Lin, Ning Feng, Chao Song, and Huaiyu Wan. Attention based spatial-temporal graph convolutional networks for traffic flow forecasting. In _AAAI_ , pages 922–929, 2019. 

- [14] Liangzhe Han, Bowen Du, Leilei Sun, Yanjie Fu, Yisheng Lv, and Hui Xiong. Dynamic and multi-faceted spatio-temporal deep learning for traffic speed forecasting. In _KDD_ , pages 547–555, 2021. 

- [15] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In _CVPR_ , pages 15979–15988, 2022. 

- [16] Zhenyu Hou, Xiao Liu, Yukuo Cen, Yuxiao Dong, Hongxia Yang, Chunjie Wang, and Jie Tang. Graphmae: Self-supervised masked graph autoencoders. In _KDD_ , pages 594–604, 2022. 

- [17] Ziniu Hu, Yuxiao Dong, Kuansan Wang, Kai-Wei Chang, and Yizhou Sun. Gpt-gnn: Generative pre-training of graph neural networks. In _KDD_ , pages 1857–1867, 2020. 

- [18] Ming Jin, Yu Zheng, Yuan-Fang Li, Siheng Chen, Bin Yang, and Shirui Pan. Multivariate time series forecasting with dynamic graph neural odes. _TKDE_ , pages 1–14, 2022. 

- [19] Ioannis Kakogeorgiou, Spyros Gidaris, Bill Psomas, Yannis Avrithis, Andrei Bursuc, Konstantinos Karantzalos, and Nikos Komodakis. What to hide from your students: Attention-guided masked image modeling. In _ECCV_ , pages 300–318, 2022. 

- [20] Chaoliu Li, Lianghao Xia, Xubin Ren, Yaowen Ye, Yong Xu, and Chao Huang. Graph transformer for recommendation. _arXiv preprint arXiv:2306.02330_ , 2023. 

- [21] Gang Li, Heliang Zheng, Daqing Liu, Chaoyue Wang, Bing Su, and Changwen Zheng. SemMAE: Semantic-guided masking for learning masked autoencoders. In _NeurIPS_ , 2022. 

- [22] Mengzhang Li and Zhanxing Zhu. Spatial-temporal fusion graph neural networks for traffic flow forecasting. In _AAAI_ , pages 4189–4196, 2021. 

- [23] Yaguang Li, Rose Yu, Cyrus Shahabi, and Yan Liu. Diffusion convolutional recurrent neural network: data-driven traffic forecasting. In _ICLR_ , 2018. 

- [24] Zhonghang Li, Chao Huang, Lianghao Xia, Yong Xu, and Jian Pei. Spatial-temporal hypergraph selfsupervised learning for crime prediction. In _ICDE_ , pages 2984–2996, 2022. 

11 

- [25] Dachuan Liu, Jin Wang, Shuo Shang, and Peng Han. Msdr: Multi-step dependency relation networks for spatial temporal forecasting. In _KDD_ , pages 1042–1050, 2022. 

- [26] Yingtao Luo, Qiang Liu, and Zhaocheng Liu. Stan: Spatio-temporal attention network for next location recommendation. In _WWW_ , pages 2177–2185, 2021. 

- [27] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, et al. Training language models to follow instructions with human feedback. _arXiv preprint arXiv:2203.021555_ , 2022. 

- [28] Zhaopeng Qiu, Xian Wu, Jingyue Gao, and Wei Fan. U-bert: Pre-training user representations for improved recommendation. In _AAAI_ , number 5, pages 4320–4327, 2021. 

- [29] Xubin Ren, Lianghao Xia, Jiashu Zhao, Dawei Yin, and Chao Huang. Disentangled contrastive collaborative filtering. _arXiv preprint arXiv:2305.02759_ , 2023. 

- [30] Sara Sabour, Nicholas Frosst, and Geoffrey E Hinton. Dynamic routing between capsules. In _NeurIPS_ , 2017. 

- [31] Zezhi Shao, Zhao Zhang, Fei Wang, and Yongjun Xu. Pre-training enhanced spatial-temporal graph neural network for multivariate time series forecasting. In _KDD_ , pages 1567–1577, 2022. 

- [32] Chao Song, Youfang Lin, Shengnan Guo, and Huaiyu Wan. Spatial-temporal synchronous graph convolutional networks: A new framework for spatial-temporal network data forecasting. In _AAAI_ , pages 914–921, 2020. 

- [33] Amauri Souza, Diego Mesquita, Samuel Kaski, and Vikas Garg. Provably expressive temporal graph networks. In _NeurIPS_ , pages 32257–32269, 2022. 

- [34] Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, and Chao Huang. Graphgpt: Graph instruction tuning for large language models. _arXiv preprint arXiv:2310.13023_ , 2023. 

- [35] Petar Veliˇckovi´c, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio, and R Devon Hjelm. Deep graph infomax. In _International Conference on Learning Representations_ , 2019. 

- [36] Xiaoyang Wang, Yao Ma, Yiqi Wang, Wei Jin, Xin Wang, Jiliang Tang, Caiyan Jia, and Jian Yu. Traffic flow prediction via spatial temporal graph neural network. In _WWW_ , pages 1082–1092, 2020. 

- [37] Yanbang Wang, Yen-Yu Chang, Yunyu Liu, Jure Leskovec, and Pan Li. Inductive representation learning in temporal networks via causal anonymous walks. In _ICLR_ , 2021. 

- [38] Wei Wei, Chao Huang, Lianghao Xia, and Chuxu Zhang. Multi-modal self-supervised learning for recommendation. In _WWW_ , pages 790–800, 2023. 

- [39] Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Xiaojun Chang, and Chengqi Zhang. Connecting the dots: Multivariate time series forecasting with graph neural networks. In _KDD_ , pages 753–763, 2020. 

- [40] Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, and Chengqi Zhang. Graph wavenet for deep spatial-temporal graph modeling. In _IJCAI_ , 2019. 

- [41] Congxi Xiao, Jingbo Zhou, Jizhou Huang, Hengshu Zhu, Tong Xu, Dejing Dou, and Hui Xiong. A contextual master-slave framework on urban region graph for urban village detection. In _ICDE_ , pages 736–748, 2023. 

- [42] Dongkuan Xu, Wei Cheng, Dongsheng Luo, Haifeng Chen, and Xiang Zhang. Infogcl: Information-aware graph contrastive learning. _NeurIPS_ , pages 30414–30425, 2021. 

- [43] Huaxiu Yao, Xianfeng Tang, Hua Wei, Guanjie Zheng, and Zhenhui Li. Revisiting spatial-temporal similarity: A deep learning framework for traffic prediction. In _AAAI_ , pages 5668–5675, 2019. 

- [44] Huaxiu Yao, Fei Wu, Jintao Ke, Xianfeng Tang, Yitian Jia, Siyu Lu, Pinghua Gong, Jieping Ye, Didi Chuxing, and Zhenhui Li. Deep multi-view spatial-temporal network for taxi demand prediction. In _AAAI_ , pages 2588–2595, 2018. 

- [45] Junchen Ye, Leilei Sun, Bowen Du, Yanjie Fu, Xinran Tong, and Hui Xiong. Co-prediction of multiple transportation demands based on deep spatio-temporal neural network. In _KDD_ , pages 305–313, 2019. 

- [46] Junchen Ye, Leilei Sun, Bowen Du, Yanjie Fu, and Hui Xiong. Coupled layer-wise graph convolution for transportation demand prediction. In _AAAI_ , pages 4617–4625, 2021. 

- [47] Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, and Yang Shen. Graph contrastive learning with augmentations. In _NeurIPS_ , pages 5812–5823, 2020. 

- [48] Bing Yu, Haoteng Yin, et al. Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting. In _IJCAI_ , 2018. 

- [49] Rose Yu, Yaguang Li, Cyrus Shahabi, Ugur Demiryurek, and Yan Liu. Deep learning: A generic approach for extreme condition traffic forecasting. In _SDM_ , pages 777–785, 2017. 

- [50] Jinlei Zhang, Feng Chen, Zhiyong Cui, Yinan Guo, and Yadi Zhu. Deep learning architecture for short-term passenger flow forecasting in urban rail transit. _TITS_ , pages 7004–7014, 2021. 

12 

- [51] Junbo Zhang, Yu Zheng, and Dekang Qi. Deep spatio-temporal residual networks for citywide crowd flows prediction. In _AAAI_ , 2017. 

- [52] Qianru Zhang, Chao Huang, Lianghao Xia, Zheng Wang, Zhonghang Li, and Siuming Yiu. Automated spatio-temporal graph contrastive learning. In _WWW_ , pages 295–305, 2023. 

- [53] Zaixi Zhang, Qi Liu, Hao Wang, Chengqiang Lu, and Chee-Kong Lee. Motif-based graph self-supervised learning for molecular property prediction. _NeurIPS_ , pages 15870–15882, 2021. 

- [54] Ling Zhao, Yujiao Song, Chao Zhang, Yu Liu, Pu Wang, Tao Lin, Min Deng, and Haifeng Li. T-gcn: A temporal graph convolutional network for traffic prediction. _TITS_ , pages 3848–3858, 2020. 

- [55] Chuanpan Zheng, Xiaoliang Fan, Cheng Wang, and Jianzhong Qi. Gman: A graph multi-attention network for traffic prediction. In _AAAI_ , pages 1234–1241, 2020. 

13 

## **A Supplementary Material** 

In the supplementary material, we provide additional information and details in A.1. This section covers the introduction of data, key parameter settings, comparisons with baselines, optimization methods, and the algorithm process of our method. Furthermore, A.2 presents supplementary experiments for our model, including visualization experiments and replication studies. Additionally, we discuss the reasons behind utilizing hypergraphs as the temporal encoder in A.3. Finally, the limitations and broader impacts of our work are discussed in A.4. 

## **A.1 Data and Implementation Details** 

**Data** . The statistical information of the aforementioned four real-world datasets is presented in Table 4. These datasets primarily consist of daily spatio-temporal statistics in the United States. Specifically, the **PeMS08** and **METR-LA** datasets were collected from 7 roads in the San Bernardino area and the road network of Los Angeles County, respectively. On the other hand, the **NYC Taxi** and **NYC Citi Bike** datasets were obtained from New York City. 

**Parameters** . The latent representation dimensionality ( _d_ ) and the customized parameter ( _d[′]_ ) are both set to 64 and 16, respectively. The number of hyperedges ( _HT_ , _HS_ , _HM_ ) is set to 8, 10, and 16, respectively. We perform 2 dynamic routing iterations. Additionally, the balance ratio ( _λ_ ) for _Lr_ and _Lkl_ is set to 0.1, and the total mask ratio ( _rt_ ) is set to 0.25. 

Table 4: Statistical Information of Experimental Datasets. 

|Dataset|Data Record|# Node|Time Steps|Sample Rate|Sample Date|
|---|---|---|---|---|---|
|PEMS08<br>METR-LA<br>NYC Taxi<br>NYC Citi Bike|traffc fow<br>traffc speed<br>taxicab records<br>bike orders|170<br>207<br>266<br>250|17856<br>34272<br>4368<br>4368|5min<br>5min<br>30min<br>30min|1/Jul/2016 - 31/Aug/2016<br>1/Mar/2012 - 30/Jun/2012<br>1/Apr/2016 - 30/Jun/2016<br>1/Apr/2016 - 30/Jun/2016|



**Baselines** . We selected 13 methods as baselines and categorized them into distinct groups: 

## **Hybrid spatio-temporal prediction method:** 

- **DMVSTNET** [44]: This framework for demand forecasting utilizes RNNs, convolutional networks, and fully connected layers. RNNs capture temporal correlation, convolutional networks handle spatial correlation, and fully connected layers address regional semantic correlation. 

## **Spatio-temporal prediction method based on GNNs:** 

- **STGCN** [48]: It employs convolution operations to model both temporal and spatial dependencies. 

- **GWN** [40]: This method utilizes a learnable graph structure to capture spatial dependencies and incorporates the diffuse convolution technique to capture temporal dependencies. 

- **GWN*** [40]: To evaluate the effectiveness of GWN, we utilize long-term time series data as input. Our evaluation focuses on predicting the data for the next hour using the data from the preceding two weeks. We employ a simple linear layer to process the long-term features in this prediction. 

- **TGCN** [54]: In this method, both RNNs and GNNs are employed to model temporal dependence and capture spatial correlation, respectively. 

- **MTGNN** [39]: This method utilizes a learnable graph structure to model associations between multiple variables. It incorporates dilated convolution and skip connections to jointly capture spatio-temporal dependencies. 

- **MSDR** [25]: This model proposes a variant of RNNs to make full use of historical time step information, and combines GNNs to model long-range spatio-temporal dependence. 

- **STMGCN** [11]: This method is a spatio-temporal framework designed for demand forecasting. It incorporates region association from various aspects and combines the power of RNNs and GNNs to effectively model both spatial and temporal relations. 

- **CCRNN** [46]: This is an approach specifically designed for traffic demand prediction. It leverages multiple layers of GNNs, each assigned with different adjacency matrices, to capture hierarchical spatial associations. RNNs are employed to establish temporal dependencies within the network. 

1 

- **STSGCN** [32]: This work captures spatio-temporal correlations by constructing a local spatiotemporal graph, enabling the synchronous modeling of these correlations. 

- **STFGNN** [22]: This work proposes a data-driven approach that utilizes a gated convolution method to generate spatio-temporal graphs. By learning spatial and temporal dependencies, the approach effectively captures the correlations within the data. 

## **Attention-based spatio-temporal prediction method:** 

- **ASTGCN** [13]: It utilizes attention and GNNs to capture spatio-temporal periodic dependencies. 

- **STWA** [4]: It integrates location-specific and time-varying parameters into the attention network to effectively capture dynamic spatio-temporal correlations. 

## **Spatio-temporal prediction via differential equation:** 

- **STGODE** [10]: This method captures spatial dependencies by enhancing GNNs through the use of ordinary differential equations. Additionally, temporal dependencies are modeled using a dilated temporal convolutional network. 

## **A.1.1 Optimization Method** 

In the GPT-STframework, multiple temporal encoders (Section 4.1) and spatial encoders (Section 4.2) are stacked together to generate the final embeddings. The architecture includes two temporal encoders followed by a spatial encoder, forming a spatio-temporal (ST) encoding block. The final embeddings are generated by passing the data through two spatio-temporal encoding blocks. 

> During the pre-training phase, the predictions **Y**[ˆ] _∈_ R _[R][×][T][ ×][F]_ are computed by applying a linear layer to the hidden dimension. To optimize the parameters, the absolute error loss function is utilized, following the approach employed in prior works such as [55, 1]. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0015-10.png)


Let **X** represent the input, consisting of _F_ features across _R_ regions in the previous _T_ time slots. The two layers of the MLP network discussed in Sec 4.3 can be formalized as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0015-12.png)


In the given formulation, **W** _[p] r[∈]_[R] _[d][×][d]_[ and] **[ b]** _[p] r[∈]_[R] _[d]_[ represent the region-specific parameters, while] **W** _[p] t[∈]_[R] _[d][×][d]_[ and] **[ b]** _[t][p][∈]_[R] _[d]_[ are the time-dynamic parameters created as in Equation 4.][The predictions] _q ∈_ R _[H][s][×][R][×][T]_ are obtained from **Q** through a linear layer followed by a softmax function. The KL divergence loss function _Lkl_ can be formulated as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0015-14.png)


In this case, _c_ ¯ _∈_ R _[H][s][×][R][×][T]_ is considered as the ground truth classification result. To balance the contribution of the two loss functions, we introduce a parameter _λ_ to adjust their ratio, as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0015-16.png)


In the downstream task stage, we utilize the pre-trained embeddings _**ζ** ∈_ R _[R][×][T][ ×][d]_ along with the raw data representation **E** _[′]_ = **X**[¯] _·_ **e** 0 (without any mask operation) as the input for the downstream model. To fuse these two inputs, we employ a simple gated fusion layer proposed by [55], which can be formalized as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0015-18.png)


In the given formulation, **W** _h,_ 1, **W** _h,_ 2 _∈_ R _[d][×][d]_ and **b** _h ∈_ R _[d]_ are learnable parameters. The variable _z_ represents the gate operation, and _δ_ ( _·_ ) denotes the sigmoid activation function. It is important to note that we prevent the backpropagation of _**ζ**_ at this stage. By fusing _**ζ**_ and **E** _[′]_ using the gated fusion layer, downstream models can leverage the knowledge gained during the pre-training stage for improved predictions. The specific optimization methods employed may vary depending on the downstream models, such as mean absolute error [1, 25] or Huber loss [22, 32]. 

2 

## **A.1.2 Algorithm Process** 

Algorithm 1 represents the process of the adaptive mask strategy described in Section 4.3. On the other hand, Algorithm 2 illustrates the algorithmic process during the pre-training stage. 

## **Algorithm 1:** Adaptive Mask Strategy 

- **Input:** ST data **X** _∈_ R _[R][×][T][ ×][F]_ , maximum epoch number _E_ , total mask ratio _rt_ , number of **X** elements _J_ **Output:** mask matrix **M** _∈_ R _[R][×][T][ ×][F]_ 

- **1 for** _e_ = 1 _to E_ **do 2** Calculate the classification results _q_ from **X** according to Sec 4.3 **3** Calculate adaptive mask ratio _ra_ by _ra_ = ( _e/E_ ) _[γ]_ **4** Calculate the number of masked elements by _mt_ = _Jrt_ , and then obtain the adaptive masked number _ma_ and random masked number _mr_ by _ma_ = _Jra_ ; _mr_ = _mt − ma_ 

- **5** Randomly select _n_ categories from the classification list until the total number of elements of these categories is greater than _ma_ . Mask all the elements of the first _n −_ 1 categories, and randomly mask the elements of the _n_ -th category with the residual masked number in _ma_ 

- **6** Randomly mask the remaining elements with random masked number _mr_ **7 end** 

## **Algorithm 2:** Learning Process of GPT-ST Framework 

**Input:** Spatio-temporal data **X** _∈_ R _[R][×][T][ ×][F]_ , mask matrix **M** _∈_ R _[R][×][T][ ×][F]_ , dynamic routing iterations number _R_ , maximum epoch number _E_ , learning rate _η_ **Output:** Trained parameters in **Θ 1** Initialize all parameters in **Θ** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0016-06.png)


**----- Start of picture text -----**<br>
2 for  e  = 1  to E do<br>3 Calculate approximate classification results  q  according to Eq 10 and then generate the mask matrix  M<br>according to Alg 1<br>4 Mask the elements in  X  with  M  and then calculate the initial representation  E  of the masked traffic data<br>for each region in each time slot<br>5 Calculate the raw temporal features  d t according to Eq 4 and initialize the free-form region embedding<br>matrices  c r<br>6 Encode the temporal traffic pattern with  Γ  by integrating customized parameters  d t and  c r into the<br>temporal hypergraph neural network according to Eq 3<br>7 Generate the normalized region embedding  v  and calculate the transferred information ¯ v i|r,t ∈ R [d] from<br>each region  r  to each cluster center (hyperedge)  i  according to Eq 5<br>8 for  rn = 0  to R  do<br>9 Perform the dynamic routing algorithm to characterize the semantic similarities between the regions<br>(low-level capsules) and the spatial cluster centroids (high-level capsules) according to Eq 6<br>10 end<br>11 Generate the final cluster embedding ¯ s  for cross-class relationships learning<br>12 Generate the personalized high-level hypergraph structure and conduct it on the reshaped embedding ˜ s<br>to generate ˆ s  to model inter-classes relationships according to Eq 7<br>13 Conduct the customized low-level hypergraph structure to propagate the clustered embeddings ˆ s  back to<br>the regional embeddings  Ψ  according to Eq 8<br>14 Make predictions Y [ˆ]  and calculate the absolute error loss  Lr according to Eq 9<br>15 Calculate the KL divergence loss  Lkl based on Eq 11<br>16 Calculate the final loss  L  according to Eq 12<br>17 for  θ ∈ Θ do<br>18 θ =  θ − η · ∂L/∂θ<br>19 end<br>20 end<br>21 return  all parameters  Θ<br>**----- End of picture text -----**<br>


## **A.2 Additional experiments** 

## **A.2.1 Visualization Study** 

Figure 9 presents the reconstruction results of the model during the pre-training stage. In the figure, gray, red, and blue lines respectively represent the masked signals, predicted values, and visible 

3 

signals. The results demonstrate that GPT-ST can accurately predict the masked signals based on the visible signals, regardless of whether a random mask or an adaptive mask is used. Furthermore, it can be observed that the masked signals generated by the adaptive strategy exhibit greater temporal consistency compared to those generated by the random strategy. This is because the category attribute of a region is less likely to change within a short period of time. The continuous mask and cluster mask are effective in increasing the difficulty of the reconstruction task, enabling GPT-ST to learn robust spatio-temporal representations even with low masking ratios. 

Figure 10 summarizes the enhancement effect of your model on downstream baselines. The figure showcases the predicted performance of four baselines on the PEMS08 dataset, where the blue, green, and red lines respectively represent the ground truth, the original performance, and the enhanced performance of the baselines. With the assistance of GPT-ST, the prediction performance of the baseline models experiences significant improvements on certain subsets, as highlighted by the red box. This confirms that the pre-trained model, equipped with customized parameter learners and mechanisms for encoding intra- and inter-class spatial patterns, can provide a discriminative and semantic representation for downstream tasks. As a result, it effectively compensates for the limitations of different baselines. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0017-02.png)


**----- Start of picture text -----**<br>
600 600 400 y_mask 400 y_mask<br>500 500 350300 yy_unmask_pred 350300 yy_unmask_pred<br>400 400 250 250<br>300 300 200 200<br>150 150<br>200 y_mask 200 y_mask 100 100<br>100 y_predy_unmask 100 y_predy_unmask 50 50<br>0 100 200 Time of Day300 400 500 600 0 100 200 Time of Day300 400 500 600 2000 2100 2200 Time of Day2300 2400 2500 2600 2000 2100 2200 Time of Day2300 2400 2500 2600<br>(a) Case A for Ran mask (b) Case A for Ada mask (c) Case B for Ran mask (d) Case B for Ada mask<br>Traffic flow Traffic flow Traffic flow Traffic flow<br>**----- End of picture text -----**<br>


Figure 9: Visualization experiments of reconstruction performance in the pre-training stage. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks_images/nips_2023_GPT-ST_Generative_Pre-Training_of_Spatio-Temporal_Graph_Neural_Networks.pdf-0017-04.png)


**----- Start of picture text -----**<br>
GPST GPST<br>GPST GPST<br>(a) STGCN (b) TGCN (c) GWN (d) MSDR<br>**----- End of picture text -----**<br>


Figure 10: Visualization experiments of different baselines in the downstream stage. 

## **A.2.2 Replication Study** 

In this section, the deviation of GPT-ST’s performance over random parameter initialization is investigated. The statistical results are presented in Table 5. For each performance evaluation on the PEMS08 dataset, we randomly selected 5 seeds. The results indicate that GPT-ST exhibits strong adaptability to different parameter initialization settings, consistently providing stable and high-quality spatio-temporal representations for downstream models. 

Table 5: Replication study on PEMS08 dataset. 

|Model<br>MAE<br>RMSE|MAPE<br>Model<br>MAE<br>RMSE|
|---|---|
|STGCN<br>17.99±0.14<br>28.56±0.18<br>w/ GPT-ST<br>**16.40±0.16**<br>**26.27±0.29**|11.22±0.24<br>GWN<br>15.37±0.28<br>24.61±0.26<br><br>**10.62±0.20**<br>w/ GPT-ST<br>**14.77±0.05**<br>**24.19±0.08**<br>|
|ASTGCN<br>18.05±0.30<br>27.76±0.47<br>w/ GPT-ST<br>**16.83±0.50**<br>**26.40±0.71**|11.40±0.26<br>TGCN<br>21.47±0.04<br>31.82±0.03<br>1<br>**10.78±0.45**<br>w/ GPT-ST<br>**17.33±0.08**<br>**26.50±0.11**<br>**1**|
|MTGNN<br>15.34±0.04<br>24.55±0.09<br>w/ GPT-ST<br>**14.96±0.05**<br>**24.47±0.10**|9.72±0.04<br>STSGCN<br>17.96±0.10<br>28.19±0.20<br>1<br>**9.66±0.04**<br>w/ GPT-ST<br>**16.28±0.07**<br>**26.05±0.09**<br>**1**|
|STFGNN<br>17.20±0.07<br>27.55±0.18<br>w/ GPT-ST<br>**15.90±0.03**<br>**25.86±0.06**|11.09±0.10<br>STGODE<br>17.81±0.06<br>27.64±0.08<br>1<br>**10.42±0.16**<br>w/ GPT-ST<br>**15.89±0.16**<br>**25.05±0.16**<br>**1**|
|MSDR<br>16.52±0.16<br>25.70±0.19<br>w/ GPT-ST<br>**15.96±0.06**<br>**25.09±0.08**|10.69±0.35<br>STWA<br>15.80±0.22<br>25.16±0.36<br>1<br>**10.33±0.08**<br>w/ GPT-ST<br>**15.30±0.05**<br>**24.64±0.12**<br>|



4 

## **A.3 Analysis of Hypergraph as Temporal Encoder** 

Existing time encoders in spatio-temporal prediction schemes, such as recurrent neural networks (RNNs), temporal convolutional networks (TCNs), and attention mechanisms, have certain limitations. RNNs are prone to losing long-term information due to the vanishing gradient problem. TCNs can only capture temporal information within a limited neighborhood defined by the size of the convolution kernel. Although attention mechanisms consider the interrelationship between time steps, their computational efficiency decreases significantly when the length of the time series ( _T_ ) is large, resulting in a time complexity of _O_ ( _T_[2] ). These limitations hinder the ability of existing approaches to effectively capture and model the temporal dynamics in spatio-temporal data. 

To address the aforementioned limitations, we propose to utilize the hypergraph neural network as a temporal encoder for time-dependent modeling. A hypergraph is composed of multiple sets of hyperedges, where each hyperedge serves as a learnable information hub connecting time-step information with different weights. By treating different hyperedges as a means of integrating temporal relationships across different dimensions, the hypergraph neural network offers a flexible control over complexity by adjusting the number of hyperedges. This approach allows for modeling long-term temporal dependencies while maintaining low computational costs. The hypergraph neural network combines the advantages of capturing temporal dynamics effectively and efficiently, making it a promising solution for spatio-temporal prediction tasks. 

## **A.4 Limitations and Broader Impacts** 

The proposed GPT-ST has demonstrated its effectiveness in improving the prediction performance of downstream models. However, it also has two main limitations: i) Task-specific pre-training: GPT-ST requires pre-training for each specific downstream task. This is because different prediction tasks often have distinct data formats and distributions, necessitating task-specific pre-training. For instance, a GPT-ST pre-trained on task A cannot be directly applied to task B. ii) Increased time cost: Both the pre-training process and the enhancement process in the downstream task stage add to the prediction time cost. These additional computations can impact real-time applications or scenarios with stringent time constraints. In future research, we aim to explore more generalized and versatile spatio-temporal pre-training frameworks, along with lightweight algorithms, to address the task-specific pre-training requirement and further reduce the computational overhead. 

5 

