Published as a conference paper at ICLR 2024 

# - ON THE MARKOV PROPERTY OF NEURAL ALGORITH MIC REASONING: ANALYSES AND METHODS 

**Montgomery Bohde** _[∗]_ **, Meng Liu** _[∗]_ **, Alexandra Saxton, Shuiwang Ji** Department of Computer Science & Engineering Texas A&M University College Station, TX 77843, USA 

_{_ mbohde,mengliu,allie.saxton,sji _}_ @tamu.edu 

## ABSTRACT 

Neural algorithmic reasoning is an emerging research direction that endows neural networks with the ability to mimic algorithmic executions step-by-step. A common paradigm in existing designs involves the use of historical embeddings in predicting the results of future execution steps. Our observation in this work is that such historical dependence intrinsically contradicts the Markov nature of algorithmic reasoning tasks. Based on this motivation, we present our ForgetNet, which does not use historical embeddings and thus is consistent with the Markov nature of the tasks. To address challenges in training ForgetNet at early stages, we further introduce G-ForgetNet, which uses a gating mechanism to allow for the selective integration of historical embeddings. Such an enhanced capability provides valuable computational pathways during the model’s early training phase. Our extensive experiments, based on the CLRS-30 algorithmic reasoning benchmark, demonstrate that both ForgetNet and G-ForgetNet achieve better generalization capability than existing methods. Furthermore, we investigate the behavior of the gating mechanism, highlighting its degree of alignment with our intuitions and its effectiveness for robust performance. 

## 1 INTRODUCTION 

Neural algorithmic reasoning stands at the intersection of neural networks and classical algorithm research. It involves training neural networks to reason like classical algorithms, typically through learning to execute step-by-step algorithmic operations (Velickoviˇ c & Blundell, 2021; Veli´ ckoviˇ c´ et al., 2022a). Since classical algorithms inherently possess the power to generalize across inputs of varying sizes and act as “building blocks” for complicated reasoning pathways, learning to mimic algorithmic execution can confirm and amplify the generalization and reasoning abilities of neural network models (Xu et al., 2020; Deac et al., 2021; Numeroso et al., 2023; Velickoviˇ c et al., 2022b).´ 

Existing works based on the CLRS-30 benchmark (Velickoviˇ c et al., 2022a) have demonstrated the´ effectiveness of mimicking algorithmic operations in high-dimensional latent space (Velickoviˇ c et al.,´ 2020b; Georgiev & Lio, 2020; Veli´ ckoviˇ c et al., 2020a; 2022a; Ibarz et al., 2022; Diao & Loynd,´ 2023). As detailed in Section 3.1, they typically employ an encoder-processor-decoder framework to learn the step-by-step execution of algorithms. At each step, the current algorithm state is first embedded in a high-dimensional latent space via the encoder. The embedding is then given to the processor to perform one step of computation in the latent space. The processed embeddings are then decoded to predict the updated algorithm state, namely hints. Within this paradigm, a common practice is to use historical embeddings in the current execution step. Our insight in this work is that such historical dependence contradicts the intrinsic nature of classical algorithms. 

Our work is motivated by the Markov property of algorithmic reasoning tasks; that is, the present state is sufficient to fully determine the execution output of the current step. This observation led us to investigate if the use of historical embeddings in the existing paradigm is indeed useful as it does not align with the underlying Markov property. Such a misalignment introduces noise, thus hindering 

> _∗_ Equal contribution 

1 

Published as a conference paper at ICLR 2024 

the model’s generalization ability, especially in out-of-distribution scenarios. To be consistent with the Markov property, we present ForgetNet, which removes the dependency on historical embeddings and explicitly embraces the Markov nature of algorithmic reasoning tasks. Such a modification, while simple, fundamentally realigns the computational graph of the neural model with the inherent structure of algorithmic processes. We observe that, although ForgetNet shows improvements across a wide range of tasks, training such models may be challenging due to inaccurate intermediate state predictions, especially at the early stages of training. To improve training, we further enhance our design with G-ForgetNet, in which a regularized gating mechanism is introduced in order to align with the Markov property during testing while still allowing for beneficial computational pathways during training. 

Our extensive experimental evaluations on the widely used CLRS-30 algorithmic reasoning benchmark demonstrate that both ForgetNet and G-ForgetNet outperform established baselines. In particular, G-ForgetNet achieves robust and promising performance in many different tasks, showing the benefit of the proposed gating mechanism. Further in-depth analyses of the training dynamics and gate behavior shed light on our understanding of the advantages of the proposed approaches. Overall, the findings in this work demonstrate the importance of aligning model design with the underlying Markov nature to achieve better generalization performance in neural algorithmic reasoning tasks. 

## 2 RELATED WORK 

Equipping neural networks with algorithmic reasoning abilities has gained increasing attention in recent research. Early attempts (Zaremba & Sutskever, 2014; Graves et al., 2014; Kaiser & Sutskever, 2015; Graves et al., 2016; Joulin & Mikolov, 2015) in this direction typically use recurrent neural networks with augmented memory mechanisms to mimic algorithms, showing that neural models could learn algorithmic patterns from data. With the use of graph-based representations, graph neural networks (GNNs) (Gilmer et al., 2017; Battaglia et al., 2018) can be applied naturally to algorithmic reasoning tasks (Velickoviˇ c et al., 2020b; Georgiev & Li´ o, 2020; Veli´ ckoviˇ c et al., 2020a;´ Xu et al., 2020; Yan et al., 2020; Dudzik & Velickoviˇ c,´ 2022; Dwivedi et al., 2023). Intuitively, the message passing schema in various GNNs can naturally model the propagation and iterative nature of many classical algorithms. Recently, Velickoviˇ c & Blundell (2021) outlines a blueprint´ for neural algorithmic reasoning, proposing a general encoder-processor-decoder framework. The framework, trained in algorithmic tasks, produces processors with potential applicability in real-world applications. Such generalization and transferable reasoning capabilities have been showcased in a few prior studies (Deac et al., 2021; Numeroso et al., 2023; Velickoviˇ c et al., 2022b; Beurer-Kellner´ et al., 2022). In addition, Xhonneux et al. (2021); Ibarz et al. (2022) have explored the generalization ability of the processor across multiple algorithms. 

To provide a comprehensive testbed for algorithm reasoning tasks, Velickoviˇ c et al. (2022a) presents´ the CLRS-30 benchmark, which covers 30 classical algorithms that span sorting, searching, dynamic programming, geometry, graphs, and strings (Cormen et al., 2022). The CLRS-30 benchmark is known for its out-of-distribution (OOD) testing setup (Mahdavi et al., 2022), where the input size of testing samples is much larger than those during training. Such a setup provides a rigorous test for the generalization capability of models, serving as a standard testbed for algorithmic reasoning tasks. In the benchmark study, multiple representative neural models, including Deep Sets (Zaheer et al., 2017), GAT (Velickoviˇ c et al., 2018), MPNN (Gilmer et al., 2017), PGN (Veli´ ckoviˇ c et al., 2020a),´ and Memnet (Sukhbaatar et al., 2015), have been evaluated as the processor network within the encoder-processor-decoder framework. Based on this benchmark, Ibarz et al. (2022) further proposes Triplet-GMPNN, which employs a triplet message passing schema (Dudzik & Velickoviˇ c, 2022)´ and multiple improvements for the training stability of the encoder-processor-decoder framework. Recently, Bevilacqua et al. (2023) proposes an additional self-supervised objective to learn similar representations for inputs that result in identical intermediate computation. a common design in the above methods is the incorporation of historical embeddings into current execution steps. In this work, we highlight that such historical dependence poses a misalignment with the Markov nature of the algorithmic execution. This insight motivates our proposed ForgetNet and its enhanced version G-ForgetNet, which more faithfully align with the Markov nature by reconsidering the use of historical embeddings. 

2 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods_images/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods.pdf-0003-01.png)


Figure 1: An illustration of (a) the baseline, (b) ForgetNet, and (c) G-ForgetNet methods. _E_ , _P_ , and _D_ represent the encoder, processor, and decoder module, respectively. 

## 3 ANALYSES ON THE MARKOV PROPERTY 

In this section, we first recap the existing encoder-processor-decoder paradigm on the algorithmic reasoning tasks given in the CLRS-30 benchmark. Then, we emphasize the Markov characteristic of algorithmic executions. In addition, we highlight the existing misalignment between the use of historical embeddings and this Markov property. Motivated by this observation, we present ForgetNet, which removes such historical embeddings to achieve a closer alignment with the task’s nature. An empirical study validates that ForgetNet achieves better generalization capability. 

## 3.1 ENCODER-PROCESSOR-DECODER PARADIGM 

Following prior research, we consider the algorithmic reasoning tasks as formulated in the CLRS-30 benchmark (Velickoviˇ c et al., 2022a).´ For a certain algorithm, a single execution trajectory serves as a data sample, which is composed of the input, output, and hints. Here, hints are a time series of intermediate states of the algorithm execution. Typically, a data sample is represented as a graph with _n_ nodes, where _n_ reflects the size of a particular sample. For example, in sorting algorithms, elements in the input list of length _n_ are denoted as _n_ nodes. With such a graph representation, the input, output, and hints at a particular time step are either located in node-level, edge-level, or graph-level features. As detailed in Velickoviˇ c et al. (2022a), there are five possible types of features,´ including scalar, categorical, mask, mask ~~o~~ ne, and pointer, each accompanied by its encoding/decoding strategies and associated loss functions. 

Let us denote the node-level, edge-level, and graph-level features at time step _t_ as _{_ _**x**_[(] _i[t]_[)] _[}]_[,] _[ {]_ _**[e]**_[(] _ij[t]_[)] _[}]_[, and] _**g**_[(] _[t]_[)] , respectively. Here, _i_ indicates the node index and _ij_ specifies the index of the edge between node _i_ and _j_ . Note that in addition to the input, hints are also included in these features when they are available. Most existing neural algorithmic learners (Velickoviˇ c´ et al., 2020b; Georgiev & Lio, 2020; Veli´ ckoviˇ c et al., 2020a; 2022a; Ibarz et al., 2022; Diao & Loynd, 2023) adopt the´ encoder-processor-decoder paradigm (Hamrick et al., 2018). Specifically, at each time step _t_ , the encoder first embeds the current features into high-dimensional representations as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods_images/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods.pdf-0003-08.png)


Here, _fn_ ( _·_ ), _fe_ ( _·_ ), and _fg_ ( _·_ ) are the encoder layers, typically parameterized as linear layers. The embeddings are then fed into a processor, which is parameterized as a graph neural network _f_ GNN( _·_ ), to perform one step of computation. The processor can be formulated as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods_images/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods.pdf-0003-10.png)


where [ _·_ ] denotes concatenation. It is worth noting that the processed node embeddings from the previous step, _{_ _**h**_[(] _i[t][−]_[1)] _}_ , are used at each time step _t_ . Initially, _**h**_[(0)] _i_ = **0** for all nodes. Subsequently, the decoder, a linear model, uses the processed node embeddings _{_ _**h**_[(] _i[t]_[)] _[}]_[ to either predict the][ hints] for the next time step, or the output if it is at the final time step. Note that the encoder and decoder should be task-tailored based on the feature types in the particular task. Additionally, the learnable parameters of all neural modules are shared over time steps. To train the described encoder-processordecoder model, the loss is calculated based on the decoded hints at every step and the output at the end. 

3 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods_images/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Intermediate States  5 2 4 3 1 5 2 4 3 1 5 2 4 3 1<br>(Hints)<br>Partially-Sorted  5 2 4 3 1 2 5 4 3 1 2 4 5 3 1<br>Lists<br>**----- End of picture text -----**<br>


Figure 2: Illustration of the execution steps in insertion sort. The top row represents the intermediate states, while the bottom row shows the corresponding partially sorted lists. At a specific step, the present state, denoted as the hints, includes the current order (the black pointers), the recently inserted element (the green pointer), and the current iterator (the blue pointer). The present state can fully determine the next intermediate state. The figure is adapted from Veliˇckovi´c et al. (2022a). 

During training, either the ground truth hints or the hints predicted from the previous step can be fed into the encoder, depending on whether teacher forcing is used. During inference, the step-by-step hints are not available, and the encoder always receives the predicted hints from the previous step. In the benchmark study by Velickoviˇ c et al. (2022a), the ground truth´ hints are used with 50% probability during training, given that the training process would become unstable without teacher forcing. While using the actual hints can stabilize training, it introduces discrepancies between training and inference modes. Recently, Ibarz et al. (2022) proposes several techniques to improve training stability, such as using soft hint prediction, specific initialization, and gradient clipping tricks. More importantly, it demonstrates that, with such training stability, it is possible to completely remove teacher forcing and enforce the model to rely on the hints predicted from the previous step, thus aligning the training with inference and achieving better performance. Therefore, as illustrated in Figure 1 (a), our study in this work specifically adopts and builds on this pipeline that operates without relying on teacher forcing. 

## 3.2 ALGORITHMIC NECESSITY OF HISTORICAL EMBEDDINGS 

**Markov nature of algorithmic executions.** The Markov property refers to the principle that future states depend only on the current state and not on the sequence of states that preceded it. It is important to note that such fundamental property holds in the context of algorithmic reasoning tasks formulated in the CLRS-30 benchmark because the entire algorithm state is given in each hints. To be specific, within an algorithm’s sequential execution, the state at a time step _t_ encompasses all necessary information to unambiguously determine the state at the subsequent time step _t_ + 1, preventing the need to refer to any states preceding time step _t_ . Let us take the insertion sort in Figure 2 as an example. At any specific step, the intermediate state, represented as the hints, completely determines the algorithmic execution output of that particular step, _i.e._ , the next intermediate state. 

**Misalignment with the use of historical embeddings.** Given the Markov nature of the task, we revisit the necessity of using historical embeddings in the existing paradigm for algorithm reasoning. As described in Section 3.1, a prevalent practice in the existing encoder-processor-decoder framework is the incorporation of historical embeddings from previous steps into the current processor input. This practice, which might seem to naturally borrow from design principles in graph neural networks (GNNs) and recurrent neural networks (RNNs), intends to capture and propagate potentially relevant information across time steps. However, it intrinsically contradicts the Markov nature of the task as highlighted above. Given the Markov property of tasks within the CLRS-30 benchmark, the progression of the algorithm should depend solely on the current state, given by the current hints. The incorporation of historical embeddings from previous steps, while seemingly advantageous, might inadvertently add unnecessary complexity to the model. Such an addition not only complicates the model architecture but also introduces potential discrepancies and noise that might misguide our neural learners away from the desired algorithmic trajectory, consequently compromising the generalization ability. 

**ForgetNet: removing the use of historical embeddings.** As studied by Xu et al. (2020), it is easier for neural networks to learn reasoning tasks where the computational graph of the neural network aligns with the algorithmic structure of the task since the network only needs to learn simple algorithm steps. Motivated by this intuition and the identified misalignment between the use of historical embeddings and the Markov nature of neural algorithmic reasoning tasks, we suggest removing the use of historical embeddings to align the computational graph of the neural model 

4 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods_images/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods.pdf-0005-01.png)


**----- Start of picture text -----**<br>
100 ForgetNet<br>Baseline<br>80<br>60<br>40<br>20<br>0<br>Segments IntersectBFSMinimumBellman-FordDAG Shortest PathsDijkstraActivity SelectorGraham ScanDFSInsertion SortMatrix Chain OrderBridgesTopological SortMST-PrimArticulation PointsTask SchedulingJarvis' MarchMST-KruskalLCS LengthOptimal BSTBubble SortBinary SearchFind Max. SubarrayNaïve String MatcherQuicksort SCCHeapsortFloyd-WarshallKnuth-Morris-PrattQuickselectOverall Average<br>Average Score [%]<br>**----- End of picture text -----**<br>


Figure 3: Comparison between ForgetNet and the baseline. Reported results are the average of 10 runs with random seeds. Numerical results can be found in Table 1. 

with the task’s Markov nature. Specifically, following the notation in Eq. (2), we remove the use of _{_ _**h**_[(] _i[t][−]_[1)] _}_ and only use the encoded node embeddings _{_ _**x**_ **¯**[(] _i[t]_[)] _[}]_[ as the input node embeddings for the] processor. Formally, the processor as in Eq. (2) is replaced with 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods_images/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods.pdf-0005-04.png)


While the modification of the model architecture seems simple, it non-trivially enables the updated model to have a direct and coherent alignment with the underlying Markov nature of the neural algorithmic reasoning task. The parameterized processor can thus focus on learning the one-step execution of the algorithm, without the potential discrepancies introduced by using historical embeddings. This new streamlined framework, as illustrated in Figure 1 (b), is termed ForgetNet. 

**Empirical validation.** To verify our insight, using the full set of algorithms from the CLRS-30 benchmark, we train our ForgetNet alongside the existing architecture as a baseline ( _i.e._ , Figure 1 (b) _vs._ Figure 1 (a)). The only difference between these two models is that the historical embeddings are removed in ForgetNet. Using the standard OOD splits in the CLRS-30 benchmark, we perform 10 runs for each model on each algorithm task with a single set of hyperparameters. As demonstrated in Figure 3, ForgetNet improves the performance over the baseline across 23/30 algorithmic reasoning tasks. The improvements brought by removing historical embeddings are quite significant on several tasks. For example, the absolute margins of improvement on DFS, insertion sort, and bubble sort are 66 _._ 79%, 24 _._ 57%, and 13 _._ 19% respectively. By focusing purely on the relevant signals at the current step, ForgetNet can generalize better to OOD testing samples, fitting more useful signals for improved performance. In Appendix B.1, we further evaluate the performance of ForgetNet on the multi-task setup following Ibarz et al. (2022). These empirical studies directly verify our insight that it is effective to explicitly enforce the Markov property in neural algorithmic learners. 

## 4 IMPROVED TRAINING VIA ADAPTIVE ALIGNMENT 

In this section, we first identify the limitation of completely removing historical embeddings as suggested in ForgetNet. In particular, inaccurate intermediate state predictions at the early stage of the training will potentially lead to sub-optimal convergence. To alleviate this, we propose the G-ForgetNet model, which uses a learnable gating mechanism and an additional regularization term in order to capture the Markov property of ForgetNet without the subsequent training limitations. 

## 4.1 LIMITATIONS OF ENTIRELY REMOVING HISTORICAL EMBEDDINGS 

While our ForgetNet model demonstrates effectiveness on a diverse set of tasks, it underperforms the baseline on several tasks, such as the Floyd-Warshall algorithm. A closer examination suggests that during the early stage of training, the model struggles with producing accurate intermediate predictions for certain algorithm tasks, which could lead the model towards suboptimal convergence. To make this clear, with a slight abuse of notations, we let _x_ and _y_[(] _[t]_[)] denote the input state and the _t_ -th 

5 

Published as a conference paper at ICLR 2024 

intermediate state, _i.e._ , the hints, of an algorithmic trajectory sample, respectively. Accordingly, _y_ �[(] _[t]_[)] represents the intermediate state predicted at timestep _t_ . In addition, _E_ , _P_ , and _D_ represent the computation included in the encoder, processor, and decoder, respectively. In our ForgetNet model, the computational pathway _x →E →P →D → y_[(1)] naturally emerges as a desirable pathway for training. This is because both _x_ and _y_[(1)] are accurate, facilitating a high-quality back-propagation signal. However, as we progress into extended paths for subsequent execution steps, we expose the model to the pitfalls of inaccurate intermediate state predictions. For example, the computational pathway associated with the loss function for the second intermediate state, _x →E →P →D → y_ �[(1)] _→E →P →D → y_[(2)] , is impacted by the inaccuracies of the prediction � _y_[(1)] . Intuitively, this introduces noise for the processor _P_ to learn the one-step execution of the algorithm, since the processor receives inaccurate input at the second time step. Such inaccuracies accumulate over time steps. This indicates that, during the early stages of training, the model is primarily navigating these sub-optimal pathways, hindering its optimization. Additionally, by removing the hidden states in ForgetNet, we have essentially removed a residual connection between processor layers, making it more difficult for the model to backpropagate signals through many consecutive processor layers. As an empirical illustration, Figure 4 shows the training loss of ForgetNet and that of the baseline model for the Floyd-Warshall algorithm task. It indeed shows that the training losses for ForgetNet are elevated during the early stage of training, leading to sub-optimal convergence. We provide more results and deeper analysis of the training difficulties in ForgetNet in Appendix B.2, where we observe that elevated losses in ForgetNet are primarily incurred during the later steps of the hints time series, indicating the difficulties the model has with accumulation of inaccurate intermediate predictions. 

## 4.2 G-FORGETNET: ADAPTIVE USE OF HISTORICAL EMBEDDINGS 

In light of the aforementioned limitation, we further introduce G-ForgetNet with a regularized gating mechanism that restores important computational pathways during training and learns to align with the Markov property. The core motivation behind this proposal is that while inclusion of information from previous layers does not align with the inherent Markov nature of the task, it can provide helpful support, especially during the early stage of training, where it can mitigate the effects of inaccurate intermediate predictions and facilitate higher quality backpropagation signals. Further, the added loss penalty encourages the model to obey the Markov property that was shown to be beneficial in Section 3.2. Specifically, by including _**h**_[(] _i[t][−]_[1)] in Eq. (2) as a component of the input for the processor at time step _t_ , it can enrich the model with useful computational pathways, such as _x →E →P →P →D → y_[(2)] associated with the loss function for the second intermediate state. In general, the introduced computational pathways _x →E →P →· · · →P →D → y_[(] _[t]_[)] , where there are _t_ sequentially applied processors, are valuable for training the processor _P_ to capture onestep algorithmic execution. This is because _y_[(] _[t]_[)] is the accurate output after executing the algorithm for _t_ steps from the input _x_ . In essence, these pathways create an alternative route, circumventing the challenges posed by inaccurate intermediate state predictions, especially at the early stage of training. 

Based on the above intuition, in G-ForgetNet, we further introduce a learnable gating mechanism that modulates the use of historical embeddings. Formally, Eq. (2) is replaced with 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods_images/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods.pdf-0006-05.png)


where the gate _**g** i_[(] _[t]_[)] has the same dimensions as _**h**_[(] _i[t][−]_[1)] and _⊙_ denotes element-wise product. Here, we employ a simple multi-layer perceptron (MLP) to obtain the gate as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods_images/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods.pdf-0006-07.png)


where _σ_ ( _·_ ) is the sigmoid function. An illustration of G-ForgetNet is in Figure 1 (c). Finally, we introduce a modified hints loss function that includes a regularization term on the magnitude of _**g** i_[(] _[t]_[)] as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods_images/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods.pdf-0006-09.png)


Where _L_ � _y_ ˆ[(] _[t]_[)] _, y_[(] _[t]_[)][�] is the standard hints loss functions used in the CLRS-30 benchmark, which depends on the type and location of features contained in _y_[(] _[t]_[)] . At the early stage of training, we 

6 

Published as a conference paper at ICLR 2024 

Table 1: Test OOD micro-F1 score of the baseline, ForgetNet, and G-ForgetNet methods. The cells are highlighted if their corresponding results are better than the baseline. 

|**Algorithm**<br>**Baseline**<br>**ForgetNet**<br>**G-ForgetNet**|**Algorithm**<br>**Baseline**<br>**ForgetNet**<br>**G-ForgetNet**|
|---|---|
|Activity Selector<br>93_._02%_±_1_._62<br>97_._17%_±_0_._20<br>**99.03**%_±_0_._10<br>Articulation Points<br>95_._01%_±_2_._09<br>90_._16%_±_2_._25<br>**97.97**%_±_0_._58<br>Bellman-Ford<br>97_._67%_±_0_._28<br>98_._45%_±_0_._13<br>**99.18**%_±_0_._11<br>BFS<br>99_._45%_±_0_._11<br>99_._32%_±_0_._16<br>**99.96**%_±_0_._01<br>Binary Search<br>62_._79%_±_4_._31<br>74_._41%_±_2_._11<br>**85.96**%_±_1_._59<br>Bridges<br>89_._58%_±_4_._79<br>94_._74%_±_2_._00<br>**99.43**%_±_0_._15<br>Bubble Sort<br>61_._93%_±_6_._24<br>75_._12%_±_2_._97<br>**83.19**%_±_2_._59<br>DAG Shortest Paths<br>97_._92%_±_0_._28<br>98_._18%_±_0_._24<br>**99.37**%_±_0_._03<br>DFS<br>30_._33%_±_4_._77<br>**97.12**%_±_1_._68<br>74_._31%_±_5_._03<br>Dijkstra<br>96_._78%_±_0_._72<br>98_._32%_±_0_._13<br>**99.14**%_±_0_._06<br>Find Max. Subarray<br>63_._67%_±_1_._70<br>64_._57%_±_1_._42<br>**78.97**%_±_0_._70<br>Floyd-Warshall<br>53_._00%_±_1_._17<br>38_._14%_±_1_._09<br>**56.32**%_±_0_._86<br>Graham Scan<br>91_._82%_±_1_._20<br>95_._49%_±_0_._27<br>**97.67**%_±_0_._14<br>Heapsort<br>48_._09%_±_5_._42<br>46_._90%_±_5_._96<br>**57.47**%_±_6_._08<br>Insertion Sort<br>70_._62%_±_8_._26<br>95_._19%_±_0_._77<br>**98.40**%_±_0_._21|Jarvis’ March<br>85_._44%_±_3_._26<br>85_._21%_±_2_._83<br>**88.53**%_±_2_._96|
||Knuth-Morris-Pratt<br>3_._96%_±_1_._33<br>**18.96**%_±_4_._19<br>12_._45%_±_3_._12|
||LCS Length<br>76_._24%_±_1_._38<br>84_._60%_±_0_._47<br>**85.43**%_±_0_._47|
||Matrix Chain Order<br>86_._31%_±_0_._86<br>**94.83**%_±_0_._43<br>91_._08%_±_0_._51|
||Minimum<br>96_._42%_±_1_._35<br>99_._01%_±_0_._10<br>**99.26**%_±_0_._08|
||MST-Kruskal<br>87_._42%_±_1_._12<br>85_._08%_±_1_._12<br>**91.25**%_±_0_._40|
||MST-Prim<br>92_._35%_±_0_._87<br>94_._14%_±_0_._50<br>**95.19**%_±_0_._33|
||Na¨ıve String Matcher<br>80_._32%_±_6_._66<br>62_._22%_±_3_._55<br>**97.02**%_±_0_._77|
||Optimal BST<br>78_._14%_±_1_._26<br>80_._19%_±_0_._76<br>**83.58**%_±_0_._49|
||Quickselect<br>1_._45%_±_0_._34<br>1_._61%_±_0_._38<br>**6.30**%_±_0_._85|
||Quicksort<br>47_._86%_±_6_._34<br>61_._92%_±_6_._25<br>**73.28**%_±_6_._25|
||Segments Intersect<br>97_._83%_±_0_._11<br>98_._55%_±_0_._09<br>**99.06**%_±_0_._39|
||SCC<br>44_._83%_±_2_._74<br>50_._80%_±_2_._67<br>**53.53**%_±_2_._48|
||Task Scheduling<br>80_._93%_±_0_._21<br>**86.31**%_±_0_._46<br>84_._55%_±_0_._35|
||Topological Sort<br>84_._67%_±_3_._93<br>92_._63%_±_1_._55<br>**99.92**%_±_0_._02|



intuitively anticipate the gate to be more “open”, _i.e._ , the magnitude of _**g** i_[(] _[t]_[)] to be large, thus enriching the model with the aforementioned beneficial pathways. As training progresses and the model starts predicting more reliable intermediate predictions, the dependence on historical embeddings should diminish, _i.e._ , the gate becomes more “closed”, to honor the Markov nature. Since the scale of the hints losses varies drastically for each algorithm, we use a heuristic to select the value of _λ_ for each algorithm based on the loss values; further details can be found in Appendix A.1. 

## 5 EXPERIMENTS 

In this section, we perform comprehensive experiments to evaluate the proposed G-ForgetNet model, by addressing the following questions. (1) _Can our G-ForgetNet model, equipped with the regularized gating mechanism, consistently perform better than the baseline model? How does it perform in the several tasks where ForgetNet underperforms the baseline? Does it help the early stage of training?_ (2) _How is the G-ForgetNet model compared to a boarder range of prior methods?_ (3) _What are the dynamics of the gating mechanism within G-ForgetNet? To what extent does it align with our expectations?_ 

**Datasets and setup.** We perform experiments on the standard out-of-distribution (OOD) splits present in the CLRS-30 algorithmic reasoning benchmark (Velickoviˇ c et al., 2022a).´ To be specific, we train on inputs with 16 or fewer nodes, and use inputs with 16 nodes for validation. During testing, for most algorithms, there are 32 trajectories with inputs of 64 nodes. For algorithms where the ouputs are associated with graph-level features, rather than node-level or edge-level, there are 64 _×_ more trajectories, ensuring a consistent number of targets across all tasks. 

For the baseline, ForgetNet, and G-ForgetNet introduced in Section 3.1, 3.2, and 4.2 respectively, we conduct 10 runs for each model in each task, with a single set of hyperparameters. Specifically, we employ the Adam optimizer (Kingma & Ba, 2015) with a cosine learning rate scheduler and an initial learning rate of 0.0015. The models are trained for 10,000 steps with a batch size of 32. 

**More baselines.** Beyond the baseline paradigm introduced in Section 3.1, we further include more existing state-of-the-art methods for comparison. Specifically, we first involve the notable methods studied in the CLRS-30 benchmark, including Memnet (Sukhbaatar et al., 2015), MPNN (Gilmer et al., 2017), and PGN (Velickoviˇ c et al., 2020a).´ These models serve as the processor in the encoderprocessor-decoder framework, which is trained with noisy teacher forcing in the benchmark setup. Furthermore, we include the recently proposed Triplet-GMPNN method (Ibarz et al., 2022), which develops a set of techniques to stabilize training, thus removing teacher forcing completely to align the training and inference. Moreover, the processor network in Triplet-GMPNN is a message passing network which incorporates messages from triplets of nodes. Our introduced baseline, ForgetNet, and G-ForgetNet, _i.e._ , the three methods in Figure 1, are built on the framework as developed by Ibarz et al. (2022). Differing from the baseline described in Section 3.1, Triplet-GMPNN has an additional update gate. It is worth noting that such a gate is different from our introduced gating mechanism in G-ForgetNet in terms of both motivation and architectural design. In particular, the update gate in Triplet-GMPNN is placed ahead of the decoder and aims to update the processed embeddings for a subset of nodes at each time step and keep the remaining unchanged, whereas our gate mechanism is 

7 

Published as a conference paper at ICLR 2024 

Table 2: Test OOD micro-F1 score of G-ForgetNet and existing methods. The highest scores are highlighted in bold, while the second-highest scores are underlined. Results for individual algorithms can be found in Table 3. 

|**Algorithm**|**Memnet**|**MPNN**|**PGN**|**Triplet-GMPNN**|**G-ForgetNet**|
|---|---|---|---|---|---|
|Div. & C.|13_._05%_±_0_._08|20_._30%_±_0_._49|65_._23%_±_2_._56|76_._36<br>%_±_0_._43|**78.97**%_±_0_._70|
|DP|67_._95%_±_2_._19|65_._10%_±_0_._73|70_._58%_±_0_._84|81_._99<br>%_±_1_._30|**86.70**%_±_0_._49|
|Geometry|45_._14%_±_2_._36|73_._11%_±_4_._27|61_._19%_±_1_._14|94_._09<br>%_±_0_._77|**95.09**%_±_1_._16|
|Graphs|24_._12%_±_1_._46|62_._80%_±_2_._55|60_._25%_±_1_._57|81_._41<br>%_±_1_._53|**88.80**%_±_0_._84|
|Greedy|53_._42%_±_1_._13|82_._39%_±_1_._74|75_._85%_±_1_._27|91_._22<br>%_±_0_._40|**91.79**%_±_0_._23|
|Search|34_._35%_±_0_._20|41_._20%_±_0_._61|56_._11%_±_0_._36|58_._61<br>%_±_1_._05|**63.84**%_±_0_._84|
|Sorting|71_._53<br>%_±_0_._97|11_._83%_±_0_._91|15_._46%_±_1_._18|60_._38%_±_5_._27|**78.09**%_±_3_._78|
|Strings|1_._52%_±_0_._24|3_._21%_±_0_._58|2_._04%_±_0_._16|49_._09<br>%_±_4_._78|**54.74**%_±_1_._95|
|Overall Average|38_._03%|51_._02%|52_._31%|75_._98<br>%|**82.89**%|
|_>_99%|0_/_30|1_/_30|1_/_30|1_/_30|**9/30**|
|_>_97%|0_/_30|1_/_30|1_/_30|5_/_30|**13/30**|
|_>_95%|0_/_30|2_/_30|2_/_30|7_/_30|**14/30**|



designed to enforce the Markov property of algorithmic reasoning and is supported by a regularization term in the loss function. Another recent model, Hint-ReLIC (Bevilacqua et al., 2023), uses an additional self-supervised learning objective based on data augmentations. Given such augmentations and different setups, our model and Hint-ReLIC are not directly comparable. We expect that a fusion of our model and Hint-ReLIC could further boost the performance, and we leave such an evaluation to future work as the code of Hint-ReLIC is not yet publicly available. 

**G-ForgetNet** _**vs.**_ **ForgetNet** _**vs.**_ **the baseline.** In Section 3.2, we have demonstrated the effectiveness of our ForgetNet model which removes historical embeddings to honor the Markov nature of algorithmic reasoning. Here, we further compare the three methods included in Figure 1 to evaluate the effectiveness of our proposed gating mechanism in the G-ForgetNet model. In Table 1, we report the average test results over 10 runs for each model in each algorithm. While ForgetNet surpasses the baseline across 23/30 tasks, G-ForgetNet consistently achieves improved performance over the baseline on all 30 tasks. In the several tasks where ForgetNet underperforms the baseline, such as the Floyd-Warshall and na¨ıve string matcher tasks, G-ForgetNet demonstrates consistent improvements over the baseline. For example, in the na¨ıve string matcher task, while ForgetNet performs worse than the baseine, G-ForgetNet outperforms the baseline by an absolute margin of 16 _._ 70%. This demonstrates the effectiveness of the proposed gating mechanism, which is able to capture the benefits of honoring the Markov property without the training difficulties of ForgetNet. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods_images/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods.pdf-0008-05.png)


**----- Start of picture text -----**<br>
3.0<br>Baseline<br>2.5 ForgetNet<br>G-ForgetNet<br>2.0<br>1.5<br>1.0<br>0.5<br>0.0<br>0 2000 4000 6000 8000 10000<br># Training Steps<br>Training Loss<br>**----- End of picture text -----**<br>


Figure 4: Training curves for the baseline, ForgetNet, and G-ForgetNet methods on the Floyd-Warshall task. The shaded region indicates the standard deviation. Figure is smoothed for clarity. 

As clarified in Section 4.2, the proposed gating structure is expected to enhance the early stage of training, thus improving the final convergence in many tasks. To empirically verify such intuition, in Figure 4, we illustrate the training losses of the baseline, ForgetNet, and G-ForgetNet models in the Floyd-Warshall task. We observe that ForgetNet indeed faces challenges during the early stages, leading to sub-optimal convergence compared to the baseline in this task. The G-ForgetNet model, can effectively sidestep the early training pitfalls, thereby leading to a better convergence at the end of training in this task. This verifies our intuition that the additional computational pathways in G-ForgetNet can help enhance the early stages of training. In Appendix B we dive deeper into the loss curves corresponding to different execution steps for several algorithms and demonstrate that the loss experienced by ForgetNet at each execution step tends to escalate more sharply as the algorithmic execution progresses than G-ForgetNet. This observation validates our earlier intuition in Section 4.2 that the gating mechanism in G-ForgetNet introduces computational pathways that act as corrective 

8 

Published as a conference paper at ICLR 2024 

signals against accumulated errors. By offering these pathways, G-ForgetNet can circumvent the pitfalls posed by inaccurate intermediate predictions, thereby facilitating the optimization of the losses corresponding to later execution steps. Overall, G-ForgetNet outperforms ForgetNet in 26/30 tasks and improves the overall average score from 78 _._ 98% in ForgetNet to 82 _._ 89% in G-ForgetNet. 

**Compared to more existing methods.** We further extend our comparison of G-ForgetNet to more existing methods, including the aforementioned Memnet, MPNN, PGN, and Triplet-GMPNN methods. The results of these methods are obtained from the respective literature (Velickoviˇ c et al., 2022a;´ Ibarz et al., 2022). As summarized in Table 2, G-ForgetNet emerges as the top performer in 25/30 algorithmic tasks. Compared to the previous state-of-the-art method Triplet-GMPNN, G-ForgetNet improves the mean test performance across all 30 tasks from 75.98% to 82.89%. Additionally, G-ForgetNet surpasses the 99% threshold on 9/30 algorithms, compared to the prior best of just 1/30. Further, G-ForgetNet achieves large performance increases on several algorithms. For example, ¨ G-ForgetNet achieves a test score of 97 _._ 02% in the naıve string matcher task, while the previous state-of-the-art performance is 78 _._ 67%, and G-ForgetNet achieves a test score of 98 _._ 40% on insertion sort, compared to the previous state-of-the-art of 78 _._ 14%. This comparison further demonstrates the effectiveness of our proposed method. 

**Dynamics of the gating mechanism.** In order to understand the behavior of the gating mechanism and gauge its alignment with our anticipations, we empirically investigate its dynamics during training. Specifically, we compute the L2 norm of the hidden states, _**h**_[(] _i[t]_[)] before being passed to the processor and then normalize by dividing by the square root of the hidden state dimension. In G- ForgetNet, the L2 norm is taken after gating _**g** i_[(] _[t]_[)] _⊙_ _**h**_[(] _i[t][−]_[1)] , so we are effectively measuring how much of the hidden states are allowed to pass into the processor. For every sample in the validation set, we consistently monitor the average L2 norm over both nodes and algorithmic execution steps, along the training iterations. In Figure 5, we illustrate the average L2 norm over all samples in the validation set during the training process for the FloydWarshall task for the baseline and for G-ForgetNet. We observe that the baseline hidden state norm is fairly constant and has a relatively large magnitude, indicating that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods_images/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods.pdf-0009-04.png)


**----- Start of picture text -----**<br>
1.0<br>0.8<br>0.6 Baseline<br>G-ForgetNet<br>0.4<br>0.2<br>0.0<br>0 2000 4000 6000 8000 10000<br># Training Steps<br>Hidden State Norm<br>**----- End of picture text -----**<br>


Figure 5: Average L2 norm value throughout the training process on the Floyd-Warshall task. The shaded region indicates the standard deviation. 

it is fitting historical noise during training, whereas G-ForgetNet declines to nearly zero. This empirically validates that the dynamics of the gating mechanism align with our intuition in this task. That is, the gating mechanism is open during the beginning of training, thus enhancing early training while progressively focusing on the Markov nature of algorithmic tasks. We generally observe similar trends across all of the CLRS-30 algorithms, with more tasks shown in Appendix A.2. We further validate the importance of the loss penalty included in G-ForgetNet in Appendix A.3, where we investigate the behavior of the G-ForgetNet model without the loss penalty. We observe that without the loss penalty, the model still exhibits declining trends in the hidden state norm, however it will not converge to 0 as desired. The performance of G-ForgetNet without the penalty is still better than the baselines, however the performance is significantly improved with the penalty. This aligns with our intuitions since the penalty ensures that G-ForgetNet is consistent with the Markov property. 

## 6 CONCLUSION 

In this work, we highlight a key misalignment between the prevalent practice of incorporating historical embeddings and the intrinsic Markov characteristics of algorithmic reasoning tasks. In response, we propose ForgetNet, which explicitly honors the Markov nature by removing the use of historical embeddings, and its adaptive variant, G-ForgetNet, equipped with a gating mechanism and subsequent loss penalty in order to capture the benefits of the Markov property without the training difficulties found in ForgetNet. Our comprehensive experiments on the CLRS-30 benchmark demonstrate the superior generalization capabilities of both models compared to established baselines. In summary, this work reveals the importance of aligning model design with the Markov nature in neural algorithmic reasoning tasks, paving the way for more advancements in future research. 

9 

Published as a conference paper at ICLR 2024 

## ACKNOWLEDGMENTS 

This work was supported in part by National Science Foundation grants IIS-2243850 and IIS-2006861. 

## REFERENCES 

- Peter W Battaglia, Jessica B Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, et al. Relational inductive biases, deep learning, and graph networks. _arXiv preprint arXiv:1806.01261_ , 2018. 

- Luca Beurer-Kellner, Martin Vechev, Laurent Vanbever, and Petar Velickoviˇ c.´ Learning to configure computer networks with neural algorithmic reasoning. _Advances in Neural Information Processing Systems_ , 35:730–742, 2022. 

- Beatrice Bevilacqua, Kyriacos Nikiforou, Borja Ibarz, Ioana Bica, Michela Paganini, Charles Blundell, Jovana Mitrovic, and Petar Velickoviˇ c.´ Neural algorithmic reasoning with causal regularisation. In _International Conference on Machine Learning_ , pp. 2272–2288. PMLR, 2023. 

- Thomas H Cormen, Charles E Leiserson, Ronald L Rivest, and Clifford Stein. _Introduction to algorithms_ . MIT press, 2022. 

- Andreea-Ioana Deac, Petar Velickoviˇ c, Ognjen Milinkovic, Pierre-Luc Bacon, Jian Tang, and Mladen´ Nikolic. Neural algorithmic reasoners are implicit planners. _Advances in Neural Information Processing Systems_ , 34:15529–15542, 2021. 

- Cameron Diao and Ricky Loynd. Relational attention: Generalizing transformers for graph-structured tasks. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- Andrew J Dudzik and Petar Velickoviˇ c.´ Graph neural networks are dynamic programmers. _Advances in Neural Information Processing Systems_ , 35:20635–20647, 2022. 

- Vijay Prakash Dwivedi, Chaitanya K Joshi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Benchmarking graph neural networks. _Journal of Machine Learning Research_ , 24 (43):1–48, 2023. 

Dobrik Georgiev and Pietro Lio.´ Neural bipartite matching. _arXiv preprint arXiv:2005.11304_ , 2020. 

- Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In _International conference on machine learning_ , pp. 1263–1272. PMLR, 2017. 

- Alex Graves, Greg Wayne, and Ivo Danihelka. Neural turing machines. _arXiv preprint arXiv:1410.5401_ , 2014. 

- Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka GrabskaBarwinska, Sergio G´ omez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou, et al.´ Hybrid computing using a neural network with dynamic external memory. _Nature_ , 538(7626): 471–476, 2016. 

- Jessica B Hamrick, Kelsey R Allen, Victor Bapst, Tina Zhu, Kevin R McKee, Joshua B Tenenbaum, and Peter W Battaglia. Relational inductive bias for physical construction in humans and machines. _arXiv preprint arXiv:1806.01203_ , 2018. 

- Borja Ibarz, Vitaly Kurin, George Papamakarios, Kyriacos Nikiforou, Mehdi Bennani, Robert´ Csordas,´ Andrew Joseph Dudzik, Matko Bosnjak,ˇ Alex Vitvitskyi, Yulia Rubanova, et al. A generalist neural algorithmic learner. In _Learning on Graphs Conference_ , pp. 2–1. PMLR, 2022. 

- Armand Joulin and Tomas Mikolov. Inferring algorithmic patterns with stack-augmented recurrent nets. _Advances in neural information processing systems_ , 28, 2015. 

- Łukasz Kaiser and Ilya Sutskever. Neural gpus learn algorithms. _arXiv preprint arXiv:1511.08228_ , 2015. 

10 

Published as a conference paper at ICLR 2024 

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _The Eleventh International Conference on Learning Representations_ , 2015. 

- Sadegh Mahdavi, Kevin Swersky, Thomas Kipf, Milad Hashemi, Christos Thrampoulidis, and Renjie Liao. Towards better out-of-distribution generalization of neural algorithmic reasoning tasks. _Transactions on Machine Learning Research_ , 2022. 

- Danilo Numeroso, Davide Bacciu, and Petar Velickoviˇ c.´ Dual algorithmic reasoning. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, et al. End-to-end memory networks. _Advances in neural information processing systems_ , 28, 2015. 

- Petar Veliˇckovi´c and Charles Blundell. Neural algorithmic reasoning. _Patterns_ , 2(7), 2021. 

- Petar Velickoviˇ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li´ o, and Yoshua` Bengio. Graph attention networks. In _International Conference on Learning Representations_ , 2018. 

- Petar Velickoviˇ c,´ Lars Buesing, Matthew Overlan, Razvan Pascanu, Oriol Vinyals, and Charles Blundell. Pointer graph networks. _Advances in Neural Information Processing Systems_ , 33: 2232–2244, 2020a. 

- Petar Velickoviˇ c, Rex Ying, Matilde Padovano, Raia Hadsell, and Charles Blundell.´ Neural execution of graph algorithms. In _International Conference on Learning Representations_ , 2020b. 

- Petar Velickoviˇ c, Adri´ a Puigdom` enech Badia, David Budden, Razvan Pascanu, Andrea Banino, Misha` Dashevskiy, Raia Hadsell, and Charles Blundell. The CLRS algorithmic reasoning benchmark. In _International Conference on Machine Learning_ , pp. 22084–22102. PMLR, 2022a. 

- Petar Velickoviˇ c, Matko Bo´ snjak, Thomas Kipf, Alexander Lerchner, Raia Hadsell, Razvan Pascanu,ˇ and Charles Blundell. Reasoning-modulated representations. In _Learning on Graphs Conference_ , pp. 50–1. PMLR, 2022b. 

- Louis-Pascal Xhonneux, Andreea-Ioana Deac, Petar Velickoviˇ c,´ and Jian Tang. How to transfer algorithmic reasoning knowledge to learn new algorithms? _Advances in Neural Information Processing Systems_ , 34:19500–19512, 2021. 

- Keyulu Xu, Jingling Li, Mozhi Zhang, Simon S Du, Ken-ichi Kawarabayashi, and Stefanie Jegelka. What can neural networks reason about? In _International Conference on Learning Representations_ , 2020. 

- Yujun Yan, Kevin Swersky, Danai Koutra, Parthasarathy Ranganathan, and Milad Hashemi. Neural execution engines: Learning to execute subroutines. _Advances in Neural Information Processing Systems_ , 33:17298–17308, 2020. 

- Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Russ R Salakhutdinov, and Alexander J Smola. Deep sets. _Advances in neural information processing systems_ , 30, 2017. 

Wojciech Zaremba and Ilya Sutskever. Learning to execute. _arXiv preprint arXiv:1410.4615_ , 2014. 

11 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods_images/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods.pdf-0012-01.png)


**----- Start of picture text -----**<br>
100<br>80<br>60<br>40<br>20 G-ForgetNet<br>ForgetNet<br>Baseline<br>0<br>Topological SortBFSDAG Shortest PathsBridges MinimumBellman-FordSegments IntersectDijkstraActivity SelectorInsertion SortArticulation PointsNaïve String MatcherGraham Scan MST-PrimMatrix Chain OrderMST-KruskalJarvis' MarchBinary SearchLCS LengthTask SchedulingOptimal BSTFind Max. SubarrayBubble Sort DFSQuicksortHeapsortFloyd-WarshallKnuth-Morris-PrattSCCQuickselectOverall Average<br>Average Score [%]<br>**----- End of picture text -----**<br>


Figure 6: Comparison between the baseline, ForgetNet, and G-ForgetNet. Reported results are the average of 10 runs with random seeds. Numerical results can be found in Table1 

## A FURTHER G-FORGETNET ANALYSIS 

In Figure 6 we compare G-ForgetNet with ForgetNet and the baseline model, and in Table 3, we provide the full numeric results for G-ForgetNet compared with Memnet, MPNN, PGN, and TripletGMPNN. 

## A.1 LAMBDA HEURISTIC 

Since the scale of the loss function varies drastically between algorithms, it is not possible to use a single value for _λ_ across all algorithms. In general, it is non-trivial to select optimal values for this parameter, and in this work, we use a heuristic to select reasonable values for _λ_ . Specifically, we select _λ_ such that the gate penalty makes up approximately half of the total training loss after 6,000 training steps. Intuitively, with such a schedule, the model will spend the first 6,000 steps simply learning to execute the algorithm, then during the remaining 4,000 steps, the model will focus on learning single-step executions in accordance with the Markov property. As demonstrated in Table 2, this simple heuristic has quite robust performance across the entire set of CLRS-30 algorithms. We acknowledge that such a simple penalty schedule and heuristic is unlikely to be optimal and will be approved upon by future works. 

## A.2 GATE ANALYSIS 

In Figure 7 we include more figures of the hidden state’s L2 norm for G-ForgetNet and the baseline, as in Section 5. These further support that our G-ForgetNet model does enforce the Markov property during testing as we observe the L2 norm converge to 0. 

12 

Published as a conference paper at ICLR 2024 

Table 3: Test OOD micro-F1 score of G-ForgetNet and existing methods. The highest scores are highlighted in bold, while the second-highest scores are underlined. 

|**Algorithm**|**Memnet**|**MPNN**|**PGN**<br>**Triplet-GMPNN**|**PGN**<br>**Triplet-GMPNN**|**PGN**<br>**Triplet-GMPNN**|**G-ForgetNet**|
|---|---|---|---|---|---|---|
|Activity Selector<br>Articulation Points<br>Bellman-Ford<br>BFS<br>Binary Search<br>Bridges<br>Bubble Sort<br>DAG Shortest Paths<br>DFS<br>Dijkstra<br>Find Max. Subarray<br>Floyd-Warshall<br>Graham Scan<br>Heapsort<br>Insertion Sort<br>Jarvis’ March<br>Knuth-Morris-Pratt<br>LCS Length<br>Matrix Chain Order<br>Minimum<br>MST-Kruskal<br>MST-Prim<br>Na¨ıve String Matcher<br>Optimal BST<br>Quickselect<br>Quicksort<br>Segments Intersect<br>SCC<br>Task Scheduling<br>Topological Sort|24_._10%_±_2_._22<br>1_._50%_±_0_._61<br>40_._04%_±_1_._46<br>43_._34%_±_0_._04<br>14_._37%_±_0_._46<br>30_._26%_±_0_._05<br>73_._58<br>%_±_0_._78<br>66_._15%_±_1_._92<br>13_._36%_±_1_._61<br>22_._48%_±_2_._39<br>13_._05%_±_0_._08<br>14_._17%_±_0_._13<br>40_._62%_±_2_._31<br>**68.00**%_±_1_._57<br>71_._42%_±_0_._86<br>22_._99%_±_3_._87<br>1_._81%_±_0_._00<br>49_._84%_±_4_._34<br>81_._96%_±_1_._03<br>86_._93%_±_0_._11<br>28_._84%_±_0_._61<br>10_._29%_±_3_._77<br>1_._22%_±_0_._48<br>72_._03%_±_1_._21<br>1_._74%_±_0_._03<br>73_._10<br>%_±_0_._67<br>71_._80%_±_0_._90<br>16_._32%_±_4_._78<br>82_._74%_±_0_._04<br>2_._73%_±_0_._11|80_._66%_±_3_._16<br>50_._91%_±_2_._18<br>92_._01%_±_0_._28<br>99_._89<br>%_±_0_._05<br>36_._83%_±_0_._26<br>72_._69%_±_4_._78<br>5_._27%_±_0_._60<br>96_._24%_±_0_._56<br>6_._54%_±_0_._51<br>91_._50%_±_0_._50<br>20_._30%_±_0_._49<br>26_._74%_±_1_._77<br>91_._04%_±_0_._31<br>10_._94%_±_0_._84<br>19_._81%_±_2_._08<br>34_._86%_±_12_._39<br>2_._49%_±_0_._86<br>53_._23%_±_0_._36<br>79_._84%_±_1_._40<br>85_._34%_±_0_._88<br>70_._97%_±_1_._50<br>69_._08%_±_7_._56<br>3_._92%_±_0_._30<br>62_._23%_±_0_._44<br>1_._43%_±_0_._69<br>11_._30%_±_0_._10<br>93_._44%_±_0_._10<br>24_._37%_±_4_._88<br>84_._11%_±_0_._32<br>52_._60%_±_6_._24|66_._80%_±_1_._62<br>49_._53%_±_2_._09<br>92_._99%_±_0_._34<br>99_._63%_±_0_._29<br>76_._95%_±_0_._13<br>51_._42%_±_7_._82<br>6_._01%_±_1_._95<br>96_._94%_±_0_._16<br>8_._71%_±_0_._24<br>83_._45%_±_1_._75<br>65_._23%_±_2_._56<br>28_._76%_±_0_._51<br>56_._87%_±_1_._61<br>5_._27%_±_0_._18<br>44_._37%_±_2_._43<br>49_._19%_±_1_._07<br>2_._00%_±_0_._12<br>56_._82%_±_0_._21<br>83_._91%_±_0_._49<br>87_._71%_±_0_._52<br>66_._96%_±_1_._36<br>63_._33%_±_0_._98<br>2_._08%_±_0_._20<br>71_._01%_±_1_._82<br>3_._66<br>%_±_0_._42<br>6_._17%_±_0_._15<br>77_._51%_±_0_._75<br>20_._80%_±_0_._64<br>84_._89<br>%_±_0_._91<br>60_._45%_±_2_._69|95_._18<br>%_±_0_._45<br>88_._32<br>%_±_2_._01<br>97_._39<br>%_±_0_._19<br>99_._73<br>%_±_0_._04<br>77_._58<br>%_±_2_._35<br>93_._99<br>%_±_2_._07<br>67_._68<br>%_±_5_._50<br>98_._19<br>%_±_0_._30<br>47_._79<br>%_±_4_._19<br>96_._05<br>%_±_0_._60<br>76_._36<br>%_±_0_._43<br>48_._52<br>%_±_1_._04<br>93_._62<br>%_±_0_._91<br>31_._04%_±_5_._82<br>78_._14<br>%_±_4_._64<br>**91.01**%_±_1_._30<br>**19.51**%_±_4_._57<br>80_._51<br>%_±_1_._84<br>**91.68**%_±_0_._59<br>97_._78<br>%_±_0_._55<br>89_._80<br>%_±_0_._77<br>86_._39<br>%_±_1_._33<br>78_._67<br>%_±_4_._99<br>73_._77<br>%_±_1_._48<br>0_._47%_±_0_._25<br>64_._64%_±_5_._12<br>97_._64<br>%_±_0_._09<br>43_._43<br>%_±_3_._15<br>**87.25**%_±_0_._35<br>87_._27<br>%_±_2_._67||**99.03**%_±_0_._10<br>**97.97**%_±_0_._46<br>**99.18**%_±_0_._11<br>**99.96**%_±_0_._01<br>**85.96**%_±_1_._59<br>**99.43**%_±_0_._15<br>**83.19**%_±_2_._59<br>**99.37**%_±_0_._03<br>**74.31**%_±_5_._03<br>**99.14**%_±_0_._06<br>**78.97**%_±_0_._70<br>**56.32**%_±_0_._86<br>**97.67**%_±_0_._14<br>57_._47<br>%_±_6_._08<br>**98.40**%_±_0_._21<br>88_._53<br>%_±_2_._96<br>12_._45<br>%_±_3_._12<br>**85.43**%_±_0_._47<br>91_._08<br>%_±_0_._51<br>**99.26**%_±_0_._08<br>**91.25**%_±_0_._40<br>**95.19**%_±_0_._33<br>**97.02**%_±_0_._77<br>**83.58**%_±_0_._49<br>**6.30**%_±_0_._85<br>**73.28**%_±_6_._25<br>**99.06**%_±_0_._39<br>**53.53**%_±_2_._48<br>84_._55%_±_0_._35<br>**99.92**%_±_0_._02|
||||||||
|Overall Average|38_._03%|51_._02%|52_._31%||75_._98<br>%|**82.89**%|
|_>_99%<br>_>_97%<br>_>_95%|0_/_30<br>0_/_30<br>0_/_30|1_/_30<br>1_/_30<br>2_/_30|1_/_30<br>1_/_30<br>2_/_30||1_/_30<br>5_/_30<br>7_/_30|**9/30**<br>**13/30**<br>**14/30**|



13 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods_images/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods.pdf-0014-01.png)


**----- Start of picture text -----**<br>
Activity Selector Bridges<br>1.0 1.0<br>0.8 0.8<br>0.6 Baseline 0.6 Baseline<br>G-ForgetNet G-ForgetNet<br>0.4 0.4<br>0.2 0.2<br>0.0 0.0<br>0 2000 4000 6000 8000 10000 0 2000 4000 6000 8000 10000<br># Training Steps # Training Steps<br>Binary Search Topological Sort<br>1.0 Baseline 1.0<br>G-ForgetNet<br>0.8 0.8<br>0.6 0.6 Baseline<br>G-ForgetNet<br>0.4 0.4<br>0.2 0.2<br>0.0 0.0<br>0 2000 4000 6000 8000 10000 0 2000 4000 6000 8000 10000<br># Training Steps # Training Steps<br>Hidden State Norm Hidden State Norm<br>Hidden State Norm Hidden State Norm<br>**----- End of picture text -----**<br>


Figure 7: Average L2 norm value throughout the training process on the activity selector, bridges, binary search, and topological sort algorithms. Shaded regions indicate standard deviation. 

14 

Published as a conference paper at ICLR 2024 

Table 4: Test OOD micro-F1 score of the baseline, G-ForgetNet without penalty, and G-ForgetNet methods. The cells are highlighted if their corresponding results are better than the baseline. The highest scores are highlighted in bold. 

|highest scores are highlighted in bold.||
|---|---|
|**Algorithm**<br>**Baseline**<br>**G-ForgetNet w/o penalty**<br>**G-ForgetNet**|**Algorithm**<br>**Baseline**<br>**G-ForgetNet w/o penalty**<br>**G-ForgetNet**|
|Activity Selector<br>93_._02%_±_1_._62<br>96_._76%_±_0_._34<br>**99.03**%_±_0_._10<br>Articulation Points<br>95_._01%_±_2_._09<br>**98.97**%_±_0_._15<br>97_._97%_±_0_._58<br>Bellman-Ford<br>97_._67%_±_0_._28<br>98_._85%_±_0_._14<br>**99.18**%_±_0_._11<br>BFS<br>99_._45%_±_0_._11<br>99_._57%_±_0_._09<br>**99.96**%_±_0_._01<br>Binary Search<br>62_._79%_±_4_._31<br>82_._49%_±_1_._66<br>**85.96**%_±_1_._59<br>Bridges<br>89_._58%_±_4_._79<br>96_._38%_±_1_._46<br>**99.43**%_±_0_._15<br>Bubble Sort<br>61_._93%_±_6_._24<br>64_._78%_±_3_._71<br>**83.19**%_±_2_._59<br>DAG Shortest Paths<br>97_._92%_±_0_._28<br>98_._70%_±_0_._20<br>**99.37**%_±_0_._03<br>DFS<br>30_._33%_±_4_._77<br>47_._97%_±_4_._42<br>**74.31**%_±_5_._03<br>Dijkstra<br>96_._78%_±_0_._72<br>98_._46%_±_0_._23<br>**99.14**%_±_0_._06<br>Find Max. Subarray<br>63_._67%_±_1_._70<br>77_._65%_±_1_._05<br>**78.97**%_±_0_._70<br>Floyd-Warshall<br>53_._00%_±_1_._17<br>54_._60%_±_1_._14<br>**56.32**%_±_0_._86<br>Graham Scan<br>91_._82%_±_1_._20<br>97_._32%_±_0_._26<br>**97.67**%_±_0_._14<br>Heapsort<br>48_._09%_±_5_._42<br>**59.09**%_±_8_._80<br>57_._47%_±_6_._08<br>Insertion Sort<br>70_._62%_±_8_._26<br>87_._90%_±_2_._59<br>**98.40**%_±_0_._21|Jarvis’ March<br>85_._44%_±_3_._26<br>**88.61**%_±_2_._58<br>88_._53%_±_2_._96|
||Knuth-Morris-Pratt<br>3_._96%_±_1_._33<br>3_._84%_±_0_._79<br>**12.45**%_±_3_._12|
||LCS Length<br>76_._24%_±_1_._38<br>80_._33%_±_1_._68<br>**85.43**%_±_0_._47|
||Matrix Chain Order<br>86_._31%_±_0_._86<br>86_._83%_±_1_._23<br>**91.08**%_±_0_._51|
||Minimum<br>96_._42%_±_1_._35<br>**99.56**%_±_0_._03<br>99_._26%_±_0_._08|
||MST-Kruskal<br>87_._42%_±_1_._12<br>**91.38**%_±_0_._58<br>91_._25%_±_0_._40|
||MST-Prim<br>92_._35%_±_0_._87<br>94_._51%_±_0_._88<br>**95.19**%_±_0_._33|
||Na¨ıve String Matcher<br>80_._32%_±_6_._66<br>93_._79%_±_1_._63<br>**97.02**%_±_0_._77|
||Optimal BST<br>78_._14%_±_1_._26<br>79_._14%_±_1_._73<br>**83.58**%_±_0_._49|
||Quickselect<br>1_._45%_±_0_._34<br>2_._06%_±_0_._52<br>**6.30**%_±_0_._85|
||Quicksort<br>47_._86%_±_6_._34<br>70_._17%_±_3_._98<br>**73.28**%_±_6_._25|
||Segments Intersect<br>97_._83%_±_0_._11<br>**99.27**%_±_0_._05<br>99_._06%_±_0_._39|
||SCC<br>44_._83%_±_2_._74<br>**59.78**%_±_2_._80<br>53_._53%_±_2_._48|
||Task Scheduling<br>80_._93%_±_0_._21<br>80_._28%_±_0_._22<br>**84.55**%_±_0_._35|
||Topological Sort<br>84_._67%_±_3_._93<br>90_._36%_±_2_._84<br>**99.92**%_±_0_._02|



## A.3 PENALTY ANALYSIS 

In Table 4, we report the performance of our G-ForgetNet model without the loss penalty, i.e., with just the gate mechanism. We observe that, even without the loss penalty, the model still outperforms the baseline on 28/30 algorithms, however the average score is only 79 _._ 31% compared to 82 _._ 88% with the penalty. Additionally, G-ForgetNet without penalty outperforms G-ForgetNet with penalty on 7 algorithms, however these cases are very small improvements. Overall, this study empirically shows us the importance of the penalty in the G-ForgetNet model, which aligns with our intuition that the penalty is necessary in order to enforce the Markov property. Finally, we provide a comparison of the L2 norm of the gated hidden states in G-ForgetNet with and without the penalty in Figure 8. On the activity selector algorithm, the G-ForgetNet model without the penalty still consistently decreases during training, however it does not reach the same final convergence as the model with the penalty, and on Floyd-Warshall, G-ForgetNet without the penalty is fairly constant throughout training, again demonstrating the necessity of the loss penalty to enforce the Markov property in G-ForgetNet. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods_images/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods.pdf-0015-05.png)


**----- Start of picture text -----**<br>
Activity Selector Floyd-Warshall<br>1.4 Baseline 1.4 Baseline<br>G-ForgetNet w/o Penalty G-ForgetNet w/o Penalty<br>1.2 1.2<br>G-ForgetNet G-ForgetNet<br>1.0 1.0<br>0.8 0.8<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2<br>0.0 0.0<br>0 2000 4000 6000 8000 10000 0 2000 4000 6000 8000 10000<br># Training Steps # Training Steps<br>Hidden State Norm Hidden State Norm<br>**----- End of picture text -----**<br>


Figure 8: Average hidden state L2 norm value throughout the training process on the activity selector and Floyd-Warshall algorithms. Shaded regions indicate standard deviation. 

## B MORE EXPERIMENTAL RESULTS 

## B.1 MULTI-TASK EXPERIMENTS 

Prior works (Xhonneux et al., 2021; Ibarz et al., 2022) have investigated jointly learning multiple algorithms using a single processor. We follow the multi-task setup in Ibarz et al. (2022) and train a single ForgetNet processor on all 30 CLRS algorithms while keeping separate encoders and decoders 

15 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods_images/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods.pdf-0016-01.png)


**----- Start of picture text -----**<br>
100 ForgetNet<br>Baseline<br>80<br>60<br>40<br>20<br>0<br>Matrix Chain OrderBFSBridges MinimumJarvis' MarchGraham ScanActivity SelectorDijkstraInsertion SortTopological SortDAG Shortest PathsSegments IntersectArticulation PointsMST-PrimMST-KruskalBinary SearchQuicksortLCS LengthTask SchedulingOptimal BSTBellman-FordHeapsortBubble SortFind Max. SubarraySCCNaïve String MatcherDFSKnuth-Morris-PrattFloyd-WarshallQuickselectOverall Average<br>Average Score [%]<br>**----- End of picture text -----**<br>


Figure 9: Comparison between the ForgetNet multi-task model and the baseline multi-task model. Average of 10 runs with random seeds. Numerical results can be found in Table 5. 

Table 5: Test OOD micro-F1 scores of the baseline and ForgetNet on the multi-task setting. 

|**Algorithm**<br>**Baseline**<br>**ForgetNet**|**Algorithm**<br>**Baseline**<br>**ForgetNet**|
|---|---|
|Activity Selector<br>88_._14%_±_0_._69<br>**94.16**%_±_0_._55<br>Articulation Points<br>**94.70**%_±_0_._55<br>92_._69%_±_0_._69<br>Bellman-Ford<br>**87.21**%_±_1_._56<br>76_._41%_±_3_._45<br>BFS<br>**99.77**%_±_0_._05<br>99_._32%_±_0_._29<br>Binary Search<br>69_._63%_±_2_._85<br>**89.52**%_±_1_._23<br>Bridges<br>96_._38%_±_1_._02<br>**98.00**%_±_0_._98<br>Bubble Sort<br>53_._24%_±_5_._01<br>**74.40**%_±_3_._09<br>DAG Shortest Paths<br>91_._49%_±_0_._76<br>**93.87**%_±_0_._36<br>DFS<br>36_._35%_±_2_._99<br>**38.51**%_±_1_._48<br>Dijkstra<br>**94.49**%_±_0_._47<br>94_._34%_±_0_._40<br>Find Max. Subarray<br>**51.15**%_±_1_._23<br>35_._19%_±_0_._61<br>Floyd-Warshall<br>**24.99**%_±_1_._42<br>20_._39%_±_0_._40<br>Graham Scan<br>88_._08%_±_1_._90<br>**95.42**%_±_0_._25<br>Heapsort<br>55_._83%_±_7_._91<br>**78.39**%_±_4_._92<br>Insertion Sort<br>87_._48%_±_2_._03<br>**93.98**%_±_0_._92|Jarvis’ March<br>74_._84%_±_5_._34<br>**95.79**%_±_0_._20<br>Knuth-Morris-Pratt<br>**51.93**%_±_4_._33<br>25_._16%_±_0_._85<br>LCS Length<br>83_._68%_±_0_._57<br>**86.85**%_±_0_._16<br>Matrix Chain Order<br>89_._60%_±_0_._49<br>**97.83**%_±_0_._13<br>Minimum<br>92_._73%_±_1_._24<br>**95.87**%_±_0_._50<br>MST-Kruskal<br>89_._30%_±_0_._81<br>**89.56**%_±_1_._68<br>MST-Prim<br>86_._54%_±_0_._93<br>**90.38**%_±_0_._21<br>Na¨ıve String Matcher<br>**46.74**%_±_3_._51<br>31_._90%_±_2_._00<br>Optimal BST<br>74_._23%_±_2_._18<br>**82.55**%_±_0_._59<br>Quickselect<br>1_._71%_±_0_._49<br>**9.15**%_±_0_._57<br>Quicksort<br>75_._31%_±_4_._68<br>**88.48**%_±_2_._13<br>Segments Intersect<br>**95.26**%_±_0_._25<br>92_._71%_±_0_._35<br>SCC<br>48_._73%_±_4_._98<br>**66.93**%_±_2_._95<br>Task Scheduling<br>**86.61**%_±_0_._21<br>84_._02%_±_0_._13<br>Topological Sort<br>81_._85%_±_1_._21<br>**93.88**%_±_0_._48|



for each algorithm. As shown in Figure 9 and in Table 5, ForgetNet performs better than the baseline on 20/30 algorithms and has a higher average score across all algorithms. This confirms that enforcing the Markov property is also beneficial in the multi-task setting. 

## B.2 MORE TRAINING CURVES 

**More training curves of the baseline, ForgetNet, and G-ForgetNet methods.** In Figure 10, we illustrate the training curves, including the total loss and losses at different execution steps, of the three methods in the Floyd-Warshall and activity selector tasks. We observe that ForgetNet does not converge to the same total loss value, shown in plot (a) on each figure. Further, we observe that the gap between ForgetNet and the baseline widens at later points in the hints time series, showing how the removal of connections between consecutive layers in ForgetNet introduces training difficulties. The inaccurate intermediate predictions cause the model to struggle to optimize losses corresponding to later execution steps. 

16 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods_images/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods.pdf-0017-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods_images/iclr_2024_On_the_Markov_Property_of_Neural_Algorithmic_Reasoning_Analyses_and_Methods.pdf-0017-02.png)


Figure 10: Training curves for the baseline, ForgetNet, and G-ForgetNet methods in the FloydWarshall and activity selector tasks, tasks (from top to bottom): (a) total loss and (b) losses at different execution steps, i.e. the losses incurred at different points in the hints time series. 

17 

