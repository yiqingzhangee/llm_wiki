# **Open-Book Neural Algorithmic Reasoning** 

**Hefei Li, Chao Peng,** _[∗]_ **Chenyang Xu,** _[∗]_ **Zhengfeng Yang** Shanghai Key Laboratory of Trustworthy Computing Software Engineering Institute East China Normal University, Shanghai, China `5` 1255902127@stu.ecnu.edu.cn, 

{cpeng, cyxu, zfyang}@sei.ecnu.edu.cn 

## **Abstract** 

Neural algorithmic reasoning is an emerging area of machine learning that focuses on building neural networks capable of solving complex algorithmic tasks. Recent advancements predominantly follow the standard supervised learning paradigm – feeding an individual problem instance into the network each time and training it to approximate the execution steps of a classical algorithm. We challenge this mode and propose a novel open-book learning framework. In this framework, whether during training or testing, the network can access and utilize all instances in the training dataset when reasoning for a given instance. 

Empirical evaluation is conducted on the challenging CLRS Algorithmic Reasoning Benchmark, which consists of 30 diverse algorithmic tasks. Our open-book learning framework exhibits a significant enhancement in neural reasoning capabilities. Further, we notice that there is recent literature suggesting that multi-task training on CLRS can improve the reasoning accuracy of certain tasks, implying intrinsic connections between different algorithmic tasks. We delve into this direction via the open-book framework. When the network reasons for a specific task, we enable it to aggregate information from training instances of other tasks in an attentionbased manner. We show that this open-book attention mechanism offers insights into the inherent relationships among various tasks in the benchmark and provides a robust tool for interpretable multi-task training. 

## **1 Introduction** 

Deep neural networks have achieved remarkable advancements in various areas, such as image processing [18, 6] and natural language processing [16, 21]. In recent years, as deep learning continues to evolve, there has been an increasing desire to see deep neural networks take on more complex tasks. Algorithmic reasoning tasks [27, 5, 28] have emerged as a particularly crucial category. In classical domains, deep neural networks have demonstrated their ability to learn predictive patterns from training data. The aspiration now is to extend this capability to the field of algorithmic reasoning, which motivates a burgeoning domain — _Neural Algorithmic Reasoning_ (NAR). 

Neural algorithmic reasoning was initially coined by [30]. The central objective of this domain is to develop and train neural networks with the capability to imitate classical rule-based algorithms, such as sorting algorithms and graph algorithms. Networks built in this manner demonstrate the ability to perform algorithmic computations similar to traditional algorithms in reasoning tasks, while showcasing improved computational efficiency compared to them [17]. Moreover, recent literature [31, 22] shows that owing to the characteristics of deep learning, these networks exhibit flexibility in handling diverse input formats, making them robust even in scenarios where certain input features are missing. 

> _∗_ Correspondence to Chao Peng and Chenyang Xu. The authors are ordered alphabetically. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 

**Challenging Benchmark for NAR.** CLRS Algorithmic Reasoning Benchmark proposed by [26] is currently the most popular and definitive benchmark for evaluating the algorithmic capabilities of neural networks. This benchmark comprises 30 diverse algorithmic reasoning tasks extracted from the foundational algorithms textbook “Introduction to Algorithms” [4], including sorting, searching, dynamic programming, graph algorithms, string algorithms, and more. Beyond the task diversity, another notable challenge of this benchmark is the _significant differences_ in scale between problem instances in the training and test sets. The test instances are substantially larger in scale compared to those in the training set. 

There have been many recent advances in exploring CLRS [7, 19, 2, 8, 24, 3]. As classical algorithms can often be represented by graph structures, several successful approaches leverage the Graph Neural Network (GNN) framework, including models such as PGN [29] and MPNN [9]. In addition to directly applying these classical GNNs, the literature has observed that the execution of some classical algorithms often relies on specific data structures. Consequently, there have been proposals to integrate classical GNNs with data structures like priority queues [12] or stacks [14] to enhance neural reasoning capabilities. 

However, we notice that all prior approaches predict algorithmic executions based solely on their parameters and the features of a single input. Although this mode is commonly used in traditional supervised learning tasks [20, 1], it may not be well-suited for NAR due to the inherent difference between complicated reasoning tasks and traditional tasks like image processing. In practical scenarios, when recognizing images, extensive background knowledge is typically not required; but when faced with complex reasoning tasks, a substantial amount of background knowledge is often necessary to complete various aspects of the reasoning process. In such situations, having real-time illustrative examples or formulas available for reference can significantly reduce our memory burden, thereby enhancing task completion. This naturally raises a question: 

_If allowing a neural network to access additional examples for reference during reasoning, will its reasoning capability improve as a result?_ 

## **1.1 Our Contributions** 

We explore the aforementioned question and introduce open-book neural algorithmic reasoning. In this model, the neural architecture is enhanced with an additional memory component that stores representations of instances in the training dataset. Whether during training or testing, whenever the network engages in reasoning for a specific instance, it has the capability to leverage this supplementary memory to aggregate information from other instances within the training set, akin to an open-book exam. The main results of the paper are summarized as follows: 

- We present a general framework for open-book NAR. This framework builds upon the foundation of previous NAR architectures by introducing two additional modules for embedding and information aggregation from the training set, and can seamlessly integrate with existing methods. We further provide a detailed implementation of the framework, which is grounded in the cross-attention mechanism. This design not only caters to single-task training but also proves to be highly effective in scenarios involving multi-task training. 

- Empirical evaluations are conducted on the challenging CLRS Benchmark [26]. We incorporate the proposed framework with three popular network architectures in the literature. The results demonstrate that each architecture’s reasoning capability can be improved significantly when utilizing the training instances through the framework. Across the majority of the reasoning tasks within the benchmark, the framework yields state-of-the-art results. 

- Multi-task training is also investigated in the paper. As highlighted in [11], on certain reasoning tasks, a generalist network trained on all datasets in CLRS outperforms the networks trained in a single-task manner. We provide an interpretation of this observation using the proposed open-book framework. Specifically, when training a neural network to solve a task, we input information from other task datasets into the framework for its use. The results show that our open-book framework can nearly replicate the effects of multi-task training for each algorithmic task, while in some tasks, it even achieves higher accuracies. Additionally, our attention-based implementation enables us to analyze the attention weights of various tasks, facilitating a deeper understanding of the intrinsic relationships among 

2 

tasks. A “paired training” experiment is further conducted to verify the effectiveness of the learned attention weights. 

## **1.2 Other Related Work** 

Our work is closely aligned with the exploration of non-parametric models [23, 25, 13], where models abstain from training specific parameters and, instead, utilize dependencies among training data points for predictions. Our framework can be viewed as a fusion of deep neural networks and non-parametric models. We have noted analogous efforts in recent work within the field of image processing [15]. This work focuses on the CIFAR-10 dataset, employing self-attention mechanisms among different points in the dataset to finish image classification tasks. 

## **2 Preliminaries** 

This section introduces the setting of an NAR dataset formally and outlines the standard paradigm employed in NAR. 

**NAR Dataset.** The objective of an NAR task is to train a neural network such that it can imitate each execution step of a classical algorithm on given problem instances. Hence, a NAR dataset is labeled by a specific problem and the algorithm employed to solve it. Each data point includes a problem instance, represented by a graph structure, and the corresponding algorithm execution on that instance, conveyed through a sequence of graph-structured states. Denote by **x** the problem instance and by **y** = _{_ **y**[(1)] _, ...,_ **y**[(] _[t]_[)] _, ...}_ the algorithm execution, where **y**[(] _[t]_[)] signifies the graph-structured states (e.g., the current nodes in the queue of breadth-first search) at the _t_ -th step of the algorithm. 

**Training Objective.** The training objective of the neural network is to perform sequential reasoning tasks over a given problem instance. At each step _t_ , the network takes as input the pair � **x** _,_ **y**[(] _[t][−]_[1)][�] and produces the output **y**[(] _[t]_[)] . This process enables the neural network to learn and predict the evolution of the algorithmic execution on the problem instance in a step-wise fashion. 

**Encode-Processor-Decode Paradigm.** To achieve the aforementioned step-wise objective, the literature follows the standard _encode-process-decode_ paradigm [10], which consists of three modules: Encoder, Processor, and Decoder. At each step _t_ , the input � **x** _,_ **y**[(] _[t][−]_[1)][�] traverses through these modules sequentially[2] : 

- The encoder module encompasses multiple neural networks that operate on � **x** _,_ **y**[(] _[t][−]_[1)][�] , thereby transforming it into a collection of graph-structured hidden states. Use _G_ = ( _V, E_ ) to denote the graph structure. Following this module, we obtain _hv_ corresponding to each node _v ∈ V_ , _hvu_ associated with each edge ( _v, u_ ) _∈ E_ , and _hg_ representing the hidden state of the entire graph _G_ . 

- The processor module usually consists of a graph neural network. This module maintains the historical hidden states of nodes, edges, and the graph: _{h_[(] _v[t][−]_[1)] _}v∈V_ , _{h_[(] _vu[t][−]_[1)] _}_ ( _v,u_ ) _∈E_ , _h_[(] _g[t][−]_[1)] , and integrate them with the newly generated states _{hv}_ , _{hv,u}_ , _hg_ to yield updated states. We borrow the language of the message-passing architecture [9] to formalize this process. For brevity, the following focuses only on updating the state of each node _v_ . At each step _t_ , the node computes and aggregates messages _muv_ from its incoming edges, updating its own hidden state: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Open-Book_Neural_Algorithmic_Reasoning_images/nips_2024_Open-Book_Neural_Algorithmic_Reasoning.pdf-0003-10.png)


Different processors employ different layers _f_ 1, _f_ 2, _f_ 3, and aggregation function[�] . 

- The decoder module utilizes the states **h**[(] _[t]_[)] as input to forecast the algorithmic execution **y**[(] _[t]_[)] at step _t_ . It is noteworthy that recent literature [11] also incorporates **x** and **y**[(] _[t][−]_[1)] within this module. 

> 2 ( _t−_ 1) For simplicity, we abuse the notion slightly, allowing **y** to represent the outcome of the last step. 

3 

## **3 Open-Book Reasoning** 

The paradigm above can be denoted by a function _F_ mapping **x** to **y** for each data point. Given a NAR dataset, this function implies a standard supervised learning mode: during a training step, a (or a mini-batch of) random datapoint ( **x** _,_ **y** ) is selected. The loss between _F_ ( **x** ) and **y** is then computed, and the parameters in _F_ are updated accordingly. In this section, we go beyond the individual **x** _→_ **y** mode in conventional supervised learning, exploring a more general and practical learning paradigm. 

## **3.1 Framework** 

We introduce an open-book reasoning framework. Within the framework, when the network is tasked with solving problem instance **x** and deducing **y** , it not only utilizes **x** as input but is also allowed to leverage information from other data points within the training set during the reasoning process. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Open-Book_Neural_Algorithmic_Reasoning_images/nips_2024_Open-Book_Neural_Algorithmic_Reasoning.pdf-0004-04.png)


Figure 1: An illustration of the open-book framework. At each reasoning step _t_ , we simultaneously input ( **x** _,_ **y**[(] _[t][−]_[1)] ) and instances from the training set **T** , yielding **y**[(] _[t]_[)] . 

The intuition behind the open-book framework is analogous to our real-world scenario of solving algorithmic problems or engaging in other reasoning tasks. In practical situations, we often consult textbooks and refer to example problems to aid in completing tasks. Typically, the structure and solutions of these examples provide substantial assistance in our reasoning process. Denoting the training set as **T** , the framework essentially aims to learn a comprehensive function _F_ : **x** _∪_ **T** _→_ **y** . 

An illustration of the framework is present in Figure 1. In addition to the original three modules, we introduce two new modules: _Dataset Encoder_ and _Open-Book Processor_ : 

- The dataset encoder module employs an encoding function _fE_ to compute the latent feature of each data point **d** _i_ = ( **x** _i,_ **y** _i_ ) in the training set: **r** _i ← fE_ ( **x** _i,_ **y** _i_ ) _._ It is worth noting that this encoder module is essentially different from the original one. It maps an entire data point **d** _i_ = ( **x** _i,_ **y** _i_ ), encompassing the ground truth of each node (and edge) at each step, into a single representation **r** _i_ . 

- The open-book processor module is incorporated between the original processor and decoder modules. The output **h**[(] _[t]_[)] from the processor no longer directly feeds into the decoder; instead, it passes through the open-book processor, where it undergoes information aggregation with the training data representation **R** = _{_ **r** 1 _, ...,_ **r** _i, ...}_ (generated by the dataset encoder). Subsequently, the open-book processor produces the latent features **h**[�][(] _[t]_[)] required by the decoder. Formally, for each node _v ∈ V_ ,[�] _h_[(] _v[t]_[)] _← fP h_[(] _v[t]_[)] _[,]_ **[ R]** _._ � � 

The central component of the framework is the open-book processor module. Within this module, the current problem instance to be solved is integrated with examples from the training set. It is crucial to acknowledge that this integration has both advantages and disadvantages. While it often enhances the architecture’s reasoning capabilities, there are instances when it may lead to counterproductive effects, particularly during multi-task training. We will elaborate on this in the experiments. 

4 

## **3.2 Attention-Based Implementation** 

Diverse implementations within the framework can be achieved by employing different functions for _fE_ and _fP_ . For the ease of investigating multitask training, we adopt an attention mechanism-based implementation. A description of the network implementation and training is given in Algorithm 1. 

**Algorithm 1** Attention-Based Implementation of Open-Book Reasoning 

**Input:** Training set **T** . 1: **while** current epochs _≤_ maximum training epochs **do** 2: Initialize the hidden state **h**[(0)] and **h**[�][(0)] . 3: Randomly pick a _target_ data point **d** = ( **x** _,_ **y** ) _∈_ **T** and several _auxiliary_ data points **d** 1 _,_ **d** 2 _, ..._ **d** _ℓ ∈_ **T** . _▷_ **Dataset Encoder** 4: **for** each auxiliary data point **d** _i_ = ( **x** _i,_ **y** _i_ ) **do** 5: Let _G_ = ( _V, E_ ) represent the underlying graph of **d** _i_ . This data point encompasses a state sequence _<_ **x** _i_ = **y** _i_[(0)] _,_ **y** _i_[(1)] _, ...,_ **y** _i_[(] _[t]_[)] _[, ... >]_[ on the graph.] 6: Randomly select two adjacent states **y** _i_[(] _[p]_[)] _,_ **y** _i_[(] _[p]_[+1)] from the sequence. 7: **for** each node _v ∈ V_ **do** 8: Let _yv_ and _yv[′]_[be node] _[ v]_[’s states in] **[ y]** _i_[(] _[p]_[)] and **y** _i_[(] _[p]_[+1)] respectively. 1 9: Employ a linear layer _zv ←_ linear � 2 _[·]_[ (] _[y][v]_[ +] _[ y] v[′]_[)] �. 10: **end for** 1 11: **r** _i ← |V |[·]_[ �] _v∈V[z][v]_[.] 12: **end for** 13: Define **R** := [ **r** 1 _, ...,_ **r** _ℓ_ ]. 14: **for** each algorithm execution step _t_ of the target data point **do** 15: Feed **x** and the outcome from the previous step into the encoder and processor sequentially to obtain the hidden states **h**[(] _[t]_[)] . 16: Employ a linear layer: **R**[(] _[t]_[)] _←_ **R** _∥_ linear( **h**[(] _[t]_[)] ). _▷_ **Open-Book Processor** 17: Set a QKV attention function with linear layers query( _·_ ) _,_ key( _·_ ) _,_ value( _·_ ). 18: Compute a cross-attention between **h**[(] _[t]_[)] and **R**[(] _[t]_[)] : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Open-Book_Neural_Algorithmic_Reasoning_images/nips_2024_Open-Book_Neural_Algorithmic_Reasoning.pdf-0005-04.png)


where _dk_ is the dimension of the key vectors. 19: Add a gate function: **h**[�][(] _[t]_[)] _←_ gate **h**[(] _[t]_[)] _,_ **h**[�][(] _[t]_[)][�] _._ � 20: Feed **h**[�][(] _[t]_[)] into the decoder module, yielding the predictions of this step. 21: **end for** 22: Compute the prediction loss and update the network parameters. 23: **end while** 

From the description, a _target_ data point and several _auxiliary_ data points are randomly selected in each training iteration. The target data point serves as the focal point for neural optimization in this iteration: the network predicts its ground truths, computes the loss, and consequently updates the network parameters. The auxiliary data points assist the network in reasoning for the target data point. Their latent features are obtained through the dataset encoder, and they subsequently influence predictions through the open-book processor. 

At each algorithmic step _t_ , the hidden states **h**[(] _[t]_[)] are computed conventionally. However, these states are not directly input into the decoder. Instead, they need to undergo cross-attention with the representations of auxiliary data points within the open-book processor module. This design allows the network to incorporate hints provided by the auxiliary data points during the reasoning process. 

The construction of the dataset encoder is a bit subtle. We observe a crucial aspect that all these pieces of information ultimately serve the decoder module. In a single algorithmic step, the decoder’s role is to facilitate the transition between two adjacent states throughout the entire reasoning process. Therefore, to better provide effective hints to the final decoder, for each auxiliary data point, we 

5 

randomly sample a pair of adjacent states from its corresponding state sequence. Subsequently, we employ a linear layer to yield the latent representations of these data points. 

**Remark.** The testing process is essentially similar to the training. It is worth noting that during the testing phase, the target data points are sourced from the testing set, while the auxiliary data points must still originate from the training set. 

## **4 Experiments** 

This section evaluates the open-book implementation empirically on the CLRS benchmark. We aim to investigate the following three questions during the experiments: 

- For various processor architectures present in the literature, can the open-book framework consistently enhance their empirical performances across the majority of the algorithmic tasks within the CLRS benchmark? 

- There is a recent literature [11] proposing a multi-task training approach for CLRS. They train a common network for various tasks in the benchmark and find that some tasks benefit from the multi-task approach, achieving higher accuracy than when trained individually. In the context of the open-book setting, does this phenomenon imply that incorporating training sets from various tasks into the open-book framework may enhance the network’s performance on certain tasks? 

- Can the attention-based implementation serve as a robust tool for interpretable multi-task training? When integrating training sets from various tasks into the open-book framework for a specific task, the network eventually learns attention weights in the open-book processor, signifying the task’s relevance to other tasks. Does this imply that if a task performs better in multi-task training than in single-task training, retaining only those tasks with prominent attention for multi-task training can still outperform single-task training? 

To tackle these questions, we conduct three types of experiments[3] : single-task augmenting, multi-task augmenting, and multi-task interpretation. Note that our “multi-task augmenting” experiment differs essentially from traditional multi-task training; here, we still train the network for a specific task, but with the inclusion of datasets from other tasks in the dataset encoder. Additional ablation experiments are also conducted (see the appendix). We initially outline the experimental setup and subsequently delve into each experiment. 

## **4.1 Setup** 

**Baselines.** We incorporate the open-book framework into three existing processor architectures: PGN [29], MPNN [9] and Triplet-GMPNN [11]. Given that the feature dimension of hidden states is set to 128 in the literature, we adjust the parameters of the dataset encoder and open-book processor to ensure seamless integration. The results (F1 scores) achieved by open-book reasoning are compared with them. Moreover, we also compare the performance with other recent architectures like Memnet [26] and NPQ [12]. 

**Computational Details.** The experiments are conducted on a machine equipped with an i7-13700K CPU, an RTX 4090 GPU, and an RTX A6000 GPU. The results are averaged over 4 runs. To ensure fair comparisons, we follow the widely-used experimental hyperparameter settings in [11], where the batch size is 32 and the network is trained for 10,000 steps by Adam optimizer with a learning rate of 0.001. During each training and testing iteration, we allow Algorithm 1 to sample 240 auxiliary data points and use only one attention head. The average training time for each reasoning task is approximately 0.5 GPU hours. 

## **4.2 Single-Task Augmenting** 

This subsection considers a single-task environment: for each reasoning task in CLRS, both target and auxiliary data points in Algorithm 1 are sourced from its own dataset. We create comparison charts for results on three existing architectures. We present one chart Figure 2 in the main body, 

> 3The codes are provided in `https://github.com/Hoferlee1/Open-Book` 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Open-Book_Neural_Algorithmic_Reasoning_images/nips_2024_Open-Book_Neural_Algorithmic_Reasoning.pdf-0007-00.png)


Figure 2: Comparison of the MPNN architecture’s performance before and after augmentation with the open-book framework. The 30 tasks are arranged in descending order of improvement magnitude. 

Table 1: The summary of our results on each task category in CLRS. The best-performing results in each row are highlighted in bold. To save space, we use the column “Prior Best” to denote the best results among four existing approaches: Memnet [26], PGN [26], MPNN [26], and NPQ [12], and the column “Ours” to denote the best results achieved by applying the open-book framework to the three existing architectures. 

|Task Category|Prior Best|Triplet-GMPNN|Ours|
|---|---|---|---|
|Graphs|64.98%±2.59|81.41%±1.53|**85.37%±1.73**|
|Geometry|92.48%±1.35|94.09%±0.77|**96.55%±0.50**|
|Strings|4.08%±0.57|49.09%±4.78|**72.41%±2.66**|
|Dynamic Programming|76%±2.47|81.99%±1.30|**82.14%±1.45**|
|Divide and Conquer|65.23%±2.56|**76.36%±0.43**|74.52%±1.88|
|Greedy|84.13%±2.59|91.22%±0.40|**93.40%±2.12**|
|Search|56.11%±0.36|58.61%±1.05|**63.15%±0.90**|
|Sorting|71.53%±0.97|60.38%±5.25|**83.65%±3.06**|



while the other two are deferred to Appendix A.The figure uses bar charts to illustrate average scores for each task, with standard deviations denoted by black lines. Additionally, we arrange the tasks in descending order of improvement magnitude to better illustrate trends. 

We also provide tables to comprehensively compare the accuracies that the open-book framework yields with existing results. In CLRS, the 30 tasks are partitioned into 8 categories: Divide and Conquer, Dynamic Programming, Geometry, Graphs, Greedy, Search, Sorting, and Strings. So we present two tables (Table 4 and Table 1): one showcasing the performance on the 30 individual tasks and another displaying the average performance for each of the 8 task categories. 

From the figures and tables, we observe that our approach outperforms the original architectures in the majority of tasks. The improvements provided by the open-book framework are particularly significant for certain tasks, such as the Naive String Matcher task (see Figure 2). However, we also notice a relatively large standard deviation in performance for some tasks. We attribute this variability to the fact that during testing, we sample data from the training set and input it into the dataset encoder each time. The quality of the sampled data influences the final inference results, leading to performance fluctuations. 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Open-Book_Neural_Algorithmic_Reasoning_images/nips_2024_Open-Book_Neural_Algorithmic_Reasoning.pdf-0008-00.png)


Figure 3: Comparisons between our multi-task augmented approach and Triplet-GMPNN. The 30 tasks are arranged in descending order of improvement magnitude. 

## **4.3 Multi-Task Augmenting** 

This subsection considers a “multi-task” environment: for each task in CLRS, Algorithm 1 selects target points from its own dataset, while the sampled auxiliary points are drawn from all datasets in CLRS. Since CLRS comprises 30 datasets, in each iteration, we randomly sample 8 instances from each dataset, ensuring that the total number of auxiliary points remains the same as in the singletask experiment, i.e., 240. Given that Triplet-GMPNN is the only architecture used for multi-task training in the literature, both this subsection and the following one “multi-task interpretation” focus exclusively on the results obtained by integrating the open-book framework with Triplet-GMPNN. 

The results are present in Figure 3. We find that incorporating data from different tasks into the open-book processor indeed replicates multi-task training. Our multi-task augmented method closely matches the previous multi-task training results, and even outperforms them on the vast majority of tasks. It is worth noting that multi-task training requires simultaneous training on all 30 algorithmic tasks, which is extremely time-consuming. If the goal is simply to enhance performance on a specific task using multi-task training, the cost is substantial. However, with the open-book framework, we can nearly achieve the effects of multi-task training on a target task in approximately the same amount of time it takes to train a single algorithm. 

## **4.4 Multi-Task Interpretation** 

This subsection delves into interpreting multi-task training. In our multi-task augmenting experiments, the acquired attention weights in the open-book processor reveal the significance of each task in relation to others. Specifically, for each task, we aggregate the attention weights of each node at every algorithmic step on each test instance. The resulting 30-dimensional vector is then normalized, serving as the total attention vector for that task relative to other tasks in the benchmark. Table 2 shows the task with the highest attention weight for each task. Moreover, we present a heatmap regarding the attention weights among CLRS tasks in Appendix B. 

Surprisingly, the table indicates that the majority of tasks exhibit a preference for attention toward tasks outside their own categories, contrary to our initial expectations. Only four bolded pairs show high attention to tasks within the same category, with most of these being graph algorithms. An intuitive explanation for this phenomenon is that tasks within the same category might not contribute additional information compared to the dataset used for training the task itself. Instead, tasks from other categories seem to play a crucial role in improving training accuracy. 

8 

Table 2: For each target (task), we show the task with the highest attention weight among other tasks in column “Auxiliary”. We use bold text to indicate when the paired tasks belong to the same algorithmic category. 

|Target<br>Auxiliary<br>Activity Selector<br>Topological Sort<br>Articulation Points<br>Knuth-Morris-Pratt<br>Bellman-Ford<br>Bridges<br>BFS<br>Task Scheduling<br>**Binary Search**<br>**Quickselect**<br>Bridges<br>Optimal BST<br>Bubble Sort<br>Task Scheduling<br>DAG Shortest Paths<br>Naïve String Matcher<br>DFS<br>Binary Search<br>**Dijkstra**<br>**Bellman-Ford**<br>Find Max. Subarray<br>Jarvis’ March<br>Floyd-Warshall<br>Heapsort<br>Graham Scan<br>Quicksort<br>Heapsort<br>Activity Selector<br>Insertion Sort<br>Minimum|Target<br>Auxiliary|
|---|---|
||Jarvis’ March<br>MST-Kruskal<br>Knuth-Morris-Pratt<br>Quicksort<br>LCS Length<br>Dijkstra<br>Matrix Chain Order<br>Jarvis’ March<br>Minimum<br>Quicksort<br>MST-Kruskal<br>Heapsort<br>**MST-Prim**<br>**Bridges**<br>Naïve String Matcher<br>LCS Length<br>Optimal BST<br>Find Max. Subarray<br>Quickselect<br>Dijkstra<br>Quicksort<br>BFS<br>Segments Intersect<br>Topological Sort<br>SCC<br>Task Scheduling<br>Task Scheduling<br>Heapsort<br>**Topological Sort**<br>**DAG Shortest Paths**|



Table 3: Comparisons among three training manners under Triplet-GMPNN. 

|Task|Single-Task|Multi-Task|Paired-Task|
|---|---|---|---|
|Heapsort|31.04%±5.82|**55.62%**±15.91|46.63%±10.43|
|Knuth-Morris-Pratt|19.51%±4.57|51.61%±8.63|**65.67%±12.36**|
|Insertion Sort|78.14%±4.64|87.00%±4.16|**95.78%±0.80**|
|LCS Length|80.51%±1.84|83.43%±1.19|**85.86%±1.47**|
|Quicksort|64.64%±5.12|75.10%±9.52|**88.43%±6.25**|
|SCC|43.43%±3.15|48.48%±9.96|**73.39%±3.00**|
|Jarvis’March|91.01%±1.30|74.51%±10.71|**94.44%±0.63**|
|MST-Kruskal|89.80%±0.77|89.08%±1.64|**90.55%±1.12**|
|MST-Prim|86.39%±1.33|86.26%±2.08|**92.56%±0.99**|
|Topological Sort|87.27%±2.67|81.65%±2.53|**87.30%±4.62**|
|Dijkstra|96.05%±0.60|94.29%±1.04|**97.44%±0.50**|
|Binary Search|77.58%±2.35|69.30%±5.65|**79.17%±2.79**|
|Bubble Sort|67.68%±5.50|52.94%±9.96|**70.30%±6.77**|
|Graham Scan|93.62%±0.91|87.74%±3.87|**94.58%±0.87**|
|Minimum|97.78%±0.55|92.50%±2.53|**98.32%±0.14**|



We proceed to a more in-depth examination of the relationships among tasks learned by the framework. We select a partner for each task according to Table 2 – namely, the task it pays the most attention to. We conduct training and testing in a multi-task manner for each task paired with its chosen partner, and refer to this type of training as paired-task training. In this experiment, we only focus on tasks that either demonstrate accuracy improvements or slight declines in multi-task training compared to single-task training, and train them in the paired manner. The results are given in Table 3. The table validates our hypothesis. On these tasks, paired-task training achieves improvements compared to single-task training, with most tasks even surpassing the performance of multi-task training. 

## **4.5 Experimental Summary** 

The experiments address the three questions posed at the beginning of the section. 

- The open-book framework can significantly enhance the reasoning capabilities of various existing architectures, yielding state-of-the-art results across the majority of tasks in CLRS. 

- By feeding data from various tasks into the dataset encoder, the framework can successfully replicate the effects of multi-task training, and in most datasets, even outperform it. 

9 

- The attention-based implementation provides a valuable tool for the interpretability of multitask training. By examining the learned attention weights, we can gain insights into the influences and intrinsic relationships among tasks during multi-task training. 

## **5 Conclusion** 

This paper considers open-book neural algorithmic reasoning, introducing a novel open-book framework accompanied by an attention-based implementation. Through empirical evaluations, we demonstrate that this implementation not only enhances the reasoning capabilities of the existing architecture but also functions as an effective tool for interpretable learning. 

Several interesting direction for future research exist, such as exploring more effective implementations within the open-book framework. Note that although our current implementation demonstrates performance improvements for the majority of tasks in CLRS, there are instances where the openbook approach may yield counterproductive results. Refining the current architecture to ensure performance enhancements across all tasks remains a significant challenge. 

10 

## **Acknowledgements** 

This work is supported by the National Key Research Project of China under Grant No. 2023YFA1009402, the Scientific and Technological Innovation 2030 Major Projects under Grant 2018AAA0100902, NSFC Programs (62161146001, 62302166, 62372176), Shanghai Key Lab of Trustworthy Computing, Henan Key Laboratory of Oracle Bone Inscription Information Processing (AnYang Normal University), and the Key Laboratory of Interdisciplinary Research of Computation and Economics (SUFE), Ministry of Education. 

## **References** 

- [1] Abeer Aljuaid and Mohd Anwar. Survey of supervised learning for medical image processing. _SN Comput. Sci._ , 3(4):292, 2022. 

- [2] Beatrice Bevilacqua, Kyriacos Nikiforou, Borja Ibarz, Ioana Bica, Michela Paganini, Charles Blundell, Jovana Mitrovic, and Petar Velickovic. Neural algorithmic reasoning with causal regularisation. In _ICML_ , volume 202 of _Proceedings of Machine Learning Research_ , pages 2272–2288. PMLR, 2023. 

- [3] Wilfried Bounsi, Borja Ibarz, Andrew Dudzik, Jessica B. Hamrick, Larisa Markeeva, Alex Vitvitskyi, Razvan Pascanu, and Petar Velickovic. Transformers meet neural algorithmic reasoners. _CoRR_ , abs/2406.09308, 2024. 

- [4] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. _Introduction to Algorithms, 3rd Edition_ . MIT Press, 2009. 

- [5] Andreea Deac, Petar Velickovic, Ognjen Milinkovic, Pierre-Luc Bacon, Jian Tang, and Mladen Nikolic. Neural algorithmic reasoners are implicit planners. In _NeurIPS_ , pages 15529–15542, 2021. 

- [6] Shisheng Deng, Dongping Liao, Xitong Gao, Juanjuan Zhao, and Kejiang Ye. A survey on cross-domain few-shot image classification. In _BigData_ , volume 14203 of _Lecture Notes in Computer Science_ , pages 3–17. Springer, 2023. 

- [7] Cameron Diao and Ricky Loynd. Relational attention: Generalizing transformers for graphstructured tasks. In _ICLR_ . OpenReview.net, 2023. 

- [8] Andrew Joseph Dudzik and Petar Velickovic. Graph neural networks are dynamic programmers. In _NeurIPS_ , 2022. 

- [9] Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural message passing for quantum chemistry. In _ICML_ , volume 70 of _Proceedings of Machine Learning Research_ , pages 1263–1272. PMLR, 2017. 

- [10] Jessica B. Hamrick, Kelsey R. Allen, Victor Bapst, Tina Zhu, Kevin R. McKee, Josh Tenenbaum, and Peter W. Battaglia. Relational inductive bias for physical construction in humans and machines. In _CogSci_ . cognitivesciencesociety.org, 2018. 

- [11] Borja Ibarz, Vitaly Kurin, George Papamakarios, Kyriacos Nikiforou, Mehdi Bennani, Róbert Csordás, Andrew Joseph Dudzik, Matko Bosnjak, Alex Vitvitskyi, Yulia Rubanova, Andreea Deac, Beatrice Bevilacqua, Yaroslav Ganin, Charles Blundell, and Petar Velickovic. A generalist neural algorithmic learner. In _LoG_ , volume 198 of _Proceedings of Machine Learning Research_ , page 2. PMLR, 2022. 

- [12] Rishabh Jain, Petar Velickovic, and Pietro Liò. Neural priority queues for graph neural networks. In _The 2023 ICML, Workshop on Knowledge and Logical Reasoning in the Era of Data-driven Learning_ , volume 202. PMLR, 2023. 

- [13] Zhiying Jiang, Yiqin Dai, Ji Xin, Ming Li, and Jimmy Lin. Few-shot non-parametric learning with deep latent variable model. In _NeurIPS_ , 2022. 

- [14] Jonas Jürß, Dulhan Hansaja Jayalath, and Petar Veliˇckovi´c. Recursive algorithmic reasoning. In _The Second Learning on Graphs Conference_ , 2023. 

- [15] Jannik Kossen, Neil Band, Clare Lyle, Aidan N. Gomez, Thomas Rainforth, and Yarin Gal. Self-attention between datapoints: Going beyond individual input-output pairs in deep learning. In _NeurIPS_ , pages 28742–28756, 2021. 

11 

- [16] Ivano Lauriola, Alberto Lavelli, and Fabio Aiolli. An introduction to deep learning in natural language processing: Models, techniques, and tools. _Neurocomputing_ , 470:443–456, 2022. 

- [17] Yujia Li, Felix Gimeno, Pushmeet Kohli, and Oriol Vinyals. Strong generalization and efficiency in neural programs. _CoRR_ , abs/2007.03629, 2020. 

- [18] Weibo Liu, Zidong Wang, Xiaohui Liu, Nianyin Zeng, Yurong Liu, and Fuad E. Alsaadi. A survey of deep neural network architectures and their applications. _Neurocomputing_ , 234:11–26, 2017. 

- [19] Sadegh Mahdavi, Kevin Swersky, Thomas Kipf, Milad Hashemi, Christos Thrampoulidis, and Renjie Liao. Towards better out-of-distribution generalization of neural algorithmic reasoning tasks. _Trans. Mach. Learn. Res._ , 2023, 2023. 

- [20] Mohammed Amine El Mrabet, Khalid El Makkaoui, and Ahmed Faize. Supervised machine learning: A survey. In _CommNet_ , pages 1–10. IEEE, 2021. 

- [21] Wongyung Nam and Beakcheol Jang. A survey on multimodal bidirectional machine learning translation of image and natural language processing. _Expert Syst. Appl._ , 235:121168, 2024. 

- [22] Danilo Numeroso, Davide Bacciu, and Petar Velickovic. Dual algorithmic reasoning. In _ICLR_ . OpenReview.net, 2023. 

- [23] Carl Edward Rasmussen. Gaussian processes in machine learning. In _Advanced Lectures on Machine Learning_ , volume 3176 of _Lecture Notes in Computer Science_ , pages 63–71. Springer, 2003. 

- [24] Gleb Rodionov and Liudmila Prokhorenkova. Neural algorithmic reasoning without intermediate supervision. In _NeurIPS_ , 2023. 

- [25] Bing Shuai, Gang Wang, Zhen Zuo, Bing Wang, and Lifan Zhao. Integrating parametric and non-parametric models for scene labeling. In _CVPR_ , pages 4249–4258. IEEE Computer Society, 2015. 

- [26] Petar Velickovic, Adrià Puigdomènech Badia, David Budden, Razvan Pascanu, Andrea Banino, Misha Dashevskiy, Raia Hadsell, and Charles Blundell. The CLRS algorithmic reasoning benchmark. In _ICML_ , volume 162 of _Proceedings of Machine Learning Research_ , pages 22084–22102. PMLR, 2022. 

- [27] Petar Velickovic and Charles Blundell. Neural algorithmic reasoning. _Patterns_ , 2(7):100273, 2021. 

- [28] Petar Velickovic, Matko Bosnjak, Thomas Kipf, Alexander Lerchner, Raia Hadsell, Razvan Pascanu, and Charles Blundell. Reasoning-modulated representations. In _LoG_ , volume 198 of _Proceedings of Machine Learning Research_ , page 50. PMLR, 2022. 

- [29] Petar Velickovic, Lars Buesing, Matthew C. Overlan, Razvan Pascanu, Oriol Vinyals, and Charles Blundell. Pointer graph networks. In _NeurIPS_ , 2020. 

- [30] Petar Velickovic, Rex Ying, Matilde Padovano, Raia Hadsell, and Charles Blundell. Neural execution of graph algorithms. In _ICLR_ . OpenReview.net, 2020. 

- [31] Louis-Pascal A. C. Xhonneux, Andreea Deac, Petar Velickovic, and Jian Tang. How to transfer algorithmic reasoning knowledge to learn new algorithms? In _NeurIPS_ , pages 19500–19512, 2021. 

12 

## **A Additional Experimental Results for Single-Task Augmenting** 

The comparison charts of PGN and Triplet-GMPNN are shown in Figure 4 and Figure 5 respectively. The table that summarizes the results for all 30 tasks is present in Table 4. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Open-Book_Neural_Algorithmic_Reasoning_images/nips_2024_Open-Book_Neural_Algorithmic_Reasoning.pdf-0013-02.png)


Figure 4: Comparison of the PGN architecture’s performance before and after augmentation with the open-book framework. The 30 tasks are arranged in descending order of improvement magnitude. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Open-Book_Neural_Algorithmic_Reasoning_images/nips_2024_Open-Book_Neural_Algorithmic_Reasoning.pdf-0013-04.png)


Figure 5: Comparison of the Triplet-GMPNN architecture’s performance before and after augmentation with the open-book framework. The 30 tasks are arranged in descending order of improvement magnitude. 

13 

Table 4: The summary of our obtained test accuracies on CLRS. The best-performing results in each row are highlighted in bold. The column “Prior Best” in the table represents the best results among four approaches in the literature: Memnet [26], PGN [26], MPNN [26], and NPQ [12], and the column “Ours” to denote the best results achieved by applying the open-book framework to the three existing architectures. 

|Task|Prior Best|Triplet-GMPNN|Ours|
|---|---|---|---|
|Activity Selector|83.36%±4.27|95.18%±0.45|**95.86%±0.61**|
|Articulation Points|50.91%±2.18|88.32%±2.01|**98.30%±0.35**|
|Bellman-Ford|92.99%±0.34|**97.39%±0.19**|95.18%±1.68|
|BFS|99.89%±0.05|99.73%±0.04|**99.99%±0.05**|
|Binary Search|76.95%±0.13|77.58%±2.35|**87.44%±1.07**|
|Bridges|72.69%±4.78|93.99%±2.07|**99.26%±0.04**|
|Bubble Sort|73.58%±0.78|67.68%±5.50|**73.16%±1.06**|
|DAG Shortest Paths|96.94%±0.16|**98.19%±0.30**|97.79%±0.34|
|DFS|13.36%±1.61|**47.79%±4.19**|40.79%±1.44|
|Dijkstra|91.50%±0.50|96.05%±0.60|**98.29%±0.42**|
|Find Max.Subarray|65.23%±2.56|**76.36%±0.43**|74.52%±1.88|
|Floyd-Warshall|28.76%±0.51|48.52%±1.04|**51.52%±1.74**|
|Graham Scan|91.04%±0.31|93.62%±0.91|**96.19%±0.06**|
|Heapsort|68.00%±1.57|31.04%±5.82|**85.71%±4.82**|
|Insertion Sort|71.42%±0.86|78.14%±4.64|**92.61%±2.82**|
|Jarvis’March|92.88%±2.87|91.01%±1.30|**94.74%±1.27**|
|Knuth-Morris-Pratt|3.91%±0.15|19.51%±4.57|**71.24%±3.7**|
|LCS Length|72.05%±5.72|80.51%±1.84|**85.54%±0.87**|
|Matrix Chain Order|83.91%±0.49|**91.68%±0.59**|90.85%±1.11|
|Minimum|87.71%±0.52|97.78%±0.55|**98.65%±0.27**|
|MST-Kruskal|70.97%±1.50|89.80%±0.77|**91.26%±0.56**|
|MST-Prim|69.08%±7.56|86.39%±1.33|**93.41%±0.52**|
|Naive String Matcher|4.24%±0.98|**78.67%±4.99**|73.57%±1.62|
|Optimal BST|72.03%±1.21|**73.77%±1.48**|70.04%±2.38|
|Quickselect|**3.66%±0.42**|0.47%±0.25|3.37%±1.37|
|Quicksort|73.10%±0.67|64.64%±5.12|**83.13%±3.52**|
|Segments Intersect|93.53%±0.88|97.64%±0.09|**98.71%±0.16**|
|SCC|32.19%±9.23|43.43%±3.15|**65.83%±5.51**|
|Task Scheduling|84.89%±0.91|87.25%±0.35|**90.93%±3.63**|
|Topological Sort|60.45%±2.69|87.27%±2.67|**92.80%±8.09**|
|Overall Average|66.04%|75.98%|**82.91%**|



14 

## **B Additional Experimental Results for Multi-Task Interpretation** 

We redraw the table that shows the learned task relationships and assign an index to each task in the new table (Table 5). Using the indices, we present a heatmap regarding the attention weights in Figure 6. 

Table 5: For each (target) task, we show the task with the highest attention weight among other tasks in column “Auxiliary Task”. The last column indicates whether the two tasks belong to the same category, displaying _[√]_ if they do. 

||Target Task|AuxiliaryTask|CategoryMembership|
|---|---|---|---|
|1|ActivitySelector|Topological Sort||
|2|Articulation Points|Knuth-Morris-Pratt||
|3|Bellman-Ford|Bridges||
|4|BFS|Task Scheduling||
|5|BinarySearch|Quickselect|~~_√_~~|
|6|Bridges|Optimal BST||
|7|Bubble Sort|Task Scheduling||
|8|DAG Shortest Paths|Naïve StringMatcher||
|9|DFS|BinarySearch||
|10|Dijkstra|Bellman-Ford|~~_√_~~|
|11|Find Max. Subarray|Jarvis’ March||
|12|Floyd-Warshall|Heapsort||
|13|Graham Scan|Quicksort||
|14|Heapsort|ActivitySelector||
|15|Insertion Sort|Minimum||
|16|Jarvis’ March|MST-Kruskal||
|17|Knuth-Morris-Pratt|Quicksort||
|18|LCS Length|Dijkstra||
|19|Matrix Chain Order|Jarvis’ March||
|20|Minimum|Quicksort||
|21|MST-Kruskal|Heapsort||
|22|MST-Prim|Bridges|~~_√_~~|
|23|Naïve StringMatcher|LCS Length||
|24|Optimal BST|Find Max. Subarray||
|25|Quickselect|Dijkstra||
|26|Quicksort|BFS||
|27|Segments Intersect|Topological Sort||
|28|SCC|Task Scheduling||
|29|Task Scheduling|Heapsort||
|30|Topological Sort|DAG Shortest Paths|~~_√_~~|



15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Open-Book_Neural_Algorithmic_Reasoning_images/nips_2024_Open-Book_Neural_Algorithmic_Reasoning.pdf-0016-00.png)


Figure 6: A heatmap where each row represents the attention vector of a (target) task in CLRS. 

16 

## **C Ablation Study for The Number of Auxiliary Data Points** 

This section presents the performance of our framework when the number of auxiliary data points varies. Figure 7 demonstrates the robustness. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Open-Book_Neural_Algorithmic_Reasoning_images/nips_2024_Open-Book_Neural_Algorithmic_Reasoning.pdf-0017-02.png)


Figure 7: The summary of our results when the number of auxiliary data points varies from 60 to 240. We blur the performance curves for each task, with the solid curve representing the average value. 

17 

## **D Scalability of Our Framework** 

This section further investigates the scalability of our framework. For each dataset in the CLRS benchmark, the training set graphs contain approximately 12 nodes on average and the test set graphs contain 64 nodes. We design two additional experiments (in the context of single-task augmenting). Note that due to memory constraints caused by the increased graph size, we omit the string category tasks and the quickselect algorithm tasks in these two experiments. 

- **Training Data Scaling:** In this experiment, we fix the original testing graph size and vary the graph size in the training data set from 4 to 20. The results are present in Figure 8. 

- **Test Data Scaling:** In this experiment, we fix the original training graph size and vary the graph size in the test data set from 64 to 128. The results are present in Figure 9. 

From the figures, we can see that as the graph size of the training data and test data varies, our approach maintains robust out-of-distribution performance. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Open-Book_Neural_Algorithmic_Reasoning_images/nips_2024_Open-Book_Neural_Algorithmic_Reasoning.pdf-0018-05.png)


Figure 8: The results of training data scaling. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Open-Book_Neural_Algorithmic_Reasoning_images/nips_2024_Open-Book_Neural_Algorithmic_Reasoning.pdf-0018-07.png)


Figure 9: The results of test data scaling. 

18 

