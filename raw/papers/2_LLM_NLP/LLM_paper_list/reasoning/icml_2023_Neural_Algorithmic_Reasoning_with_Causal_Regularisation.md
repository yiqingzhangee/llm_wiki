**Neural Algorithmic Reasoning with Causal Regularisation** 

**Beatrice Bevilacqua**[1] **Kyriacos Nikiforou**[* 2] **Borja Ibarz**[* 2] **Ioana Bica**[2] **Michela Paganini**[2] **Charles Blundell**[2] **Jovana Mitrovic**[† 2] **Petar Veliˇckovi´c**[† 2] 

## **Abstract** 

Recent work on neural algorithmic reasoning has investigated the reasoning capabilities of neural networks, effectively demonstrating they can learn to execute classical algorithms on unseen data coming from the train distribution. However, the performance of existing neural reasoners significantly degrades on out-of-distribution (OOD) test data, where inputs have larger sizes. In this work, we make an important observation: there are many _different_ inputs for which an algorithm will perform certain intermediate computations _identically_ . This insight allows us to develop data augmentation procedures that, given an algorithm’s intermediate trajectory, produce inputs for which the target algorithm would have _exactly_ the same next trajectory step. We ensure invariance in the next-step prediction across such inputs, by employing a self-supervised objective derived by our observation, formalised in a causal graph. We prove that the resulting method, which we call Hint-ReLIC, improves the OOD generalisation capabilities of the reasoner. We evaluate our method on the CLRS algorithmic reasoning benchmark, where we show up to 3 _×_ improvements on the OOD test data. 

## **1. Introduction** 

Recent works advocate for building neural networks that can reason (Xu et al., 2020; 2021; Velickoviˇ c & Blundell´ , 2021; Velickoviˇ c et al.´ , 2022a). Therein, it is posited that combining the robustness of algorithms with the flexibility of neural networks can help us accelerate progress towards models that can tackle a wide range of tasks with real world impact (Davies et al., 2021; Deac et al., 2021; Velickoviˇ c´ 

> *Equal contribution †Equal Advising 1Purdue University 

> 2DeepMind. Work done while Beatrice Bevilacqua was at DeepMind. Correspondence to: Beatrice Bevilacqua _<_ bbevilac@purdue.edu _>_ . 

_Proceedings of the 40[th] International Conference on Machine Learning_ , Honolulu, Hawaii, USA. PMLR 202, 2023. Copyright 2023 by the author(s). 

et al., 2022b; Bansal et al., 2022; Beurer-Kellner et al., 2022). The rationale is that, if a model learns how to reason, or learns to execute an algorithm, it should be able to apply that reasoning, or algorithm, to a completely novel problem, even in a different domain. Specifically, if a model has learnt an algorithm, it should be gracefully applicable on out-of-distribution (OOD) examples, which are substantially different from the examples in the training set, and return correct outputs for them. This is because an algorithm—and reasoning in general—is a sequential, step-by-step process, where a simple decision is made in each step based on outputs of the previous computation. 

Prior work (Diao & Loynd, 2022; Dudzik & Velickoviˇ c´, 2022; Ibarz et al., 2022; Mahdavi et al., 2022) has explored this setup, using the CLRS-30 benchmark (Velickoviˇ c et al.´ , 2022a), and showed that while many algorithmic tasks can be learned by Graph Neural Network (GNN) processors in a way that generalises to larger problem instances, there are still several algorithms where this could not be achieved. 

Importantly, CLRS-30 also provides ground-truth _hints_ for every algorithm. Hints correspond to the state of different variables employed to solve the algorithm (e.g. positions, pointers, colouring of nodes) along its trace. Such hints can optionally be used during training, but are not available during evaluation. In previous work, they have mainly been used as auxiliary targets together with the algorithm output. The prevailing hypothesis is that gradients coming from predicting these additional relevant signals will help constrain the representations in the neural algorithmic executor and prevent overfitting. Predicted hints can also be optionally fed back into the model to provide additional context and aid their prediction at the next step. 

In practice, while utilising hints in this way does lead to models that _follow_ the algorithmic trajectory better, they have had a less substantial impact on the accuracy of the predicted _final output_ . This is likely due to the advent of powerful strategies such as recall (Bansal et al., 2022), wherein the input is fed back to the model at every intermediate step, constantly “reminding” the model of the problem that needs to be solved. The positive effect of recall on the final output accuracy has been observed on many occasions (Mahdavi et al., 2022), and outweighs the contribution from directly 

1 

**Neural Algorithmic Reasoning with Causal Regularisation** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation_images/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation.pdf-0002-01.png)


**----- Start of picture text -----**<br>
1 1 1 1 1<br>6<br>3 5 3 5 3 5 3 5 9 3 5<br>8<br>2 4 2 4 2 4 2 4 2 4 7<br>**----- End of picture text -----**<br>


_Figure 1._ An illustration of the key observation of our work, on the depth-first search (DFS) algorithm as implemented in CLRS-30 (Velickoviˇ c et al.´ , 2022a). On the left, the first four steps of DFS are visualised. At each step, DFS explores the unvisited neighbour with the smallest index, and backtracks if no unexplored neighbours exist. The next computational step— _assigning_ 2 _as the parent of_ 4—is bound to happen, even under many _transformations_ of this graph. For example, if we were to insert new (dashed) nodes and edges into the graph, this step would still proceed as expected. Capturing this computational invariance property is the essence of our paper. 

## predicting hints and feeding them back. 

In this work, we propose a method, namely Hint-ReLIC, that decisively demonstrates an advantage to using hints. We base our work on the observation that there are many different inputs for which an algorithm will make _identical_ computations at a certain step (Figure 1). For example, applying the bubble sort algorithm from the left on [2 _,_ 1 _,_ 3] or [2 _,_ 1 _,_ 5 _,_ 3] will result in the same first step computation: a comparison of 2 and 1, followed by swapping them. Conversely, the first step of execution would be different for inputs [2 _,_ 1 _,_ 3] and [2 _,_ 5 _,_ 1 _,_ 3]; the latter input would trigger a comparison of 2 and 5 without swapping them. This observation allows us to move beyond the conventional way of using hints, i.e. autoregressively predicting them (Velickoviˇ c´ et al., 2022a). Instead, we design a novel way that learns more informative representations that enable the networks to more faithfully execute algorithms. Specifically, we learn representations that are similar for inputs that result in identical intermediate computation. First, we design a causal graph in order to formally model an algorithmic execution trajectory. Based on this, we derive a self-supervised objective for learning hint representations that are invariant across inputs having the same computational step. Moreover, we prove that this procedure will result in stronger causally-invariant representations. 

**Contributions.** Our three key contributions are as follows: 

1. We design a _causal graph_ capturing the observation that the execution of an algorithm at a certain step is determined only by a subset of the input; 

2. Motivated by our causal graph, we present a _selfsupervised objective_ to learn representations that are _provably_ invariant to changes in the input subset that does not affect the computational step; 

3. We test our model, dubbed Hint-ReLIC, on the CLRS30 algorithmic reasoning benchmark (Velickoviˇ c et al.´ , 

2022a), demonstrating a _significant improvement_ in out-of-distribution generalisation over the recently published state-of-the-art (Ibarz et al., 2022). 

## **2. Related Work** 

**GNNs and invariance to size shifts.** Graph Neural Networks (GNNs) constitute a popular class of methods for learning representations of graph data, and they have been successfully applied to solve a variety of problems. We refer the reader to Bronstein et al. (2021); Jegelka (2022) for a thorough understanding of GNN concepts. While GNNs are designed to work on graphs of any size, recent work has empirically shown poor size-generalisation capabilities of standard methods, mainly in the context of molecular modeling (Gasteiger et al., 2022), graph property prediction (Corso et al., 2020), and in executing specific graph algorithms (Velickoviˇ c et al.´ , 2020; Joshi et al., 2020). A theoretical study of failure cases has been recently provided in Xu et al. (2021), with a focus on a geometrical interpretation of OOD generalisation. In order to learn models performing equally well in- and out-of-distribution, Bevilacqua et al. (2021); Chen et al. (2022); Zhou et al. (2022) designed ad-hoc solutions satisfying assumed causal assumptions. However, these models are not applicable to our setting, as the assumptions on our data generation process are significantly different. With the same motivation, Buffelli et al. (2022) introduced a regularisation strategy to improve generalisation to larger sizes, while Yehudai et al. (2021) proposed a semi-supervised and a self-supervised objective that assume access to the test distribution. However, these models are not designed to work on algorithmic data, where OOD generalisation is still underexplored. 

**Neural Algorithmic Reasoning.** In order to learn to execute algorithmic tasks, a neural network must include a _recurrent_ component simulating the individual algorithmic steps. This component is applied a variable number of times, as required by the size of the input and the problem at hand. 

2 

**Neural Algorithmic Reasoning with Causal Regularisation** 

The recurrent component can be an LSTM (Gers & Schmidhuber, 2001), possibly augmented with a memory as in Neural Turing Machines (Graves et al., 2014; 2016); it could exploit spatial invariances in the algorithmic task through a convolutional architecture (Bansal et al., 2022); it could be based on the transformer self-attentional architecture, as in the Universal Transformer (Dehghani et al., 2019); or it could be a Graph Neural Network (GNN). GNNs are particularly well suited for algorithmic execution (Velickoviˇ c´ et al., 2020; Xu et al., 2020), and they have been applied to algorithmic problems before with a focus on extrapolation capabilities (Palm et al., 2017; Selsam et al., 2019; Joshi et al., 2020; Tang et al., 2020). Recently, Velickoviˇ c´ & Blundell (2021) have proposed a general framework for algorithmic learning with GNNs. To reconcile different data encodings and provide a unified evaluation procedure, Velickoviˇ c´ et al. (2022a) have presented a benchmark of algorithmic tasks covering a variety of areas. This benchmark, namely the CLRS algorithmic benchmark, represents data as graphs, showing that the graph formulation is general enough to include several algorithms, and not just the graph-based ones. On the CLRS benchmark, Ibarz et al. (2022) has recently presented several improvements in the architecture and learning procedure in order to obtain better performances. However, even the latest state-of-the-art models suffer from performance drops in certain algorithms when going out-of-distribution, an aspect we wish to improve upon here. 

**Self-supervised learning.** Recently, many self-supervised representation learning methods that achieve good performance on a wide range of downstream vision tasks without access to labels have been proposed. One of the most popular approaches relies on contrastive objectives that make use of data augmentations to solve the instance discrimination task (Wu et al., 2018; Chen et al., 2020; He et al., 2020; Mitrovic et al., 2021). Other approaches that rely on target networks and clustering have also been explored (Grill et al., 2020; Caron et al., 2020). Our work is similar in spirit to Mitrovic et al. (2021), which examines representation learning through the lens of causality and employs techniques from invariant prediction to make better use of data augmentations. This approach has been demonstrated to be extremely successful on vision tasks (Tomasev et al., 2022). In the context of graphs, You et al. (2020); Suresh et al. (2021); You et al. (2022) have studied how to learn contrastive representations, with particular attention paid to data augmentations. Moreover, Velickoviˇ c et al.´ (2019); Zhu et al. (2020) proposed novel objectives based on mutual information maximization in the graph domain to learn representations. Several other self-supervised methods (e.g. Thakoor et al. (2022)) have also been studied, and we refer the reader to Xie et al. (2022) for a review of existing literature on self-supervision with GNNs. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation_images/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation.pdf-0003-03.png)


_Figure 2._ The causal graph formalising our assumption about the outcome of a step depends only on a subset _Xt[s]_[of the snapshot] _[ X] t_[,] while the remainder _Xt[c]_[of the snapshot can be arbitrarily different.] 

## **3. Causal Model for Algorithmic Trajectories** 

An algorithm’s execution trajectory is described in terms of the _inputs_ , _outputs_ and _hints_ , which represent intermediate steps in the execution. We consider a graph-oriented way of representing this data (Velickoviˇ c et al.´ , 2022a): inputs and outputs are presented as data on nodes and edges of a graph, and hints are encoded as node, edge or graph features changing over time steps. 

To better understand the data at hand, we propose to formalise the data generation process for an algorithmic trajectory using a _causal graph_ . In such a causal graph, nodes represent random variables, and incoming arrows indicate that the node is a function of its parents (Pearl, 2009). The causal graph we use can be found in Figure 2. Note that this graph does not represent input data for the model, but a way of describing how any such data is generated. 

Let us consider the execution trajectory of a certain algorithm of interest, at a particular time step _t_ . Assume _X_ 1 to be the observed input, and let _Xt_ be the random variable denoting the “snapshot” at step _t_ of the algorithm execution on the input. For example, in bubble sort, _X_ 1 will be the initial (unsorted) array, and _Xt_ the array after _t_ steps of the sorting procedure (thus a partially-sorted array). 

The _key contribution_ of our causal graph is modelling the assumption that outcomes of a particular execution step depend only on a subset of the current snapshot, while the remainder of the snapshot can be arbitrarily different. 

Accordingly, we assume the snapshot _Xt_ to be generated from _two_ random variables, _Xt[c]_[and] _[X] t[s]_[,][with] _[X] t[c]_[repre-] senting the part of the snapshot that does not influence the current execution step (what can be changed without affecting the execution), while _Xt[s]_[the one that determines it] (what needs to be stable). 

Let us now revisit our bubble sort example from this perspective (see Figure 3). At each execution step, bubble sort compares two adjacent elements of the input list, and swaps them if they are not correctly ordered. Hence, in this par- 

3 

**Neural Algorithmic Reasoning with Causal Regularisation** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation_images/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation.pdf-0004-01.png)


**----- Start of picture text -----**<br>
X 1 X 2 X 3 X 4 X 5<br>3 2 4 1 5 3 2 4 1 5 3 2 4 1 5 3 2 4 1 5 3 2 4 1 5<br>X 1 [c] X 1 [s] X 2 [c] X 2 [s] X 3 [c] X 3 [s] X 4 [c] X 4 [s] X 5 [c] X 5 [s]<br>**----- End of picture text -----**<br>


_Figure 3._ Example of values of _Xt[c]_[and] _[ X] t[s]_[on an input array in the execution of the bubble sort algorithm.][At every step of computation,] bubble sort compares and possibly swaps exactly _two_ nodes—those nodes are the only ones determining the outcome of the current step, and hence they constitute _Xt[s]_[.][All other nodes are part of] _[ X] t[c]_[.] 

ticular example, _Xt[s]_[constitutes these two elements being] compared at step _t_ , while the remaining elements—which do not affect whether or not a swap is going to happen at time _t_ —form _Xt[c]_[.][By definition this implies that the next] algorithm state is a function of _only Xt[s]_[.] 

The data encoding used by Velickoviˇ c´ et al. (2022a) prescribes that hints have values provided in _all_ relevant parts of the graph. That is, in a graph of _n_ nodes, an _m_ -dimensional node hint has shape R _[n][×][m]_ , and an _m_ -dimensional edge hint has shape R _[n][×][n][×][m]_ . However, in order to keep our causal model simple, we choose to track the next-step hint in _only one_ of those values, using an _index_ , _It_ , to decide which. Specifically, _It ∈{_ 1 _,_ 2 _, . . . , n}_ are possible indices for node-level hints, and _It ∈ {_ (1 _,_ 1) _,_ (1 _,_ 2) _, . . . ,_ (1 _, n_ ) _,_ (2 _,_ 1) _, . . . ,_ (2 _, n_ ) _, . . . ,_ ( _n, n_ ) _}_ are possible indices for edge-level hints. For the indexed node/edge only, our causal graph then tracks the next-step value of the hint (either no change from the previous step or the new value), which we denote by _Yt_ +1. 

Returning once again to our bubble sort example: one specific hint being tracked by the algorithm is which two nodes in the input list are currently considered for a swap. If _I_ 2 = 4, then _Y_ 3 will track whether node 4 is being considered for a swap, immediately after two steps of the bubble sort algorithm have been executed. 

Once step _t_ of the algorithm has been executed, a new snapshot _Xt_ +1 is produced, and it can be decomposed into _Xt[c]_ +1 and _Xt[s]_ +1[, just as before.][Note that the execution in CLRS-] 30 is assumed _Markovian_ (Velickoviˇ c´ et al., 2022a): the snapshot at step _t_ contains all the information to determine the snapshot at the next step. Finally, the execution terminates after _T_ steps, and the final output is produced. We can then represent the output in a particular node/edge—indexed by _IT_ , just as before—by _YT[o]_ +1[:=] _[ g]_[(] _[X] T[s][, I][T]_[ )][, with] _[ g]_[ being] the function producing the algorithm output. 

As can be seen in Figure 2, _Xt[s]_[has all the necessary informa-] tion to predict _Yt_ +1, since our causal model encodes the conditional independence assumption _Yt_ +1 _⊥ Xt[c][|][ X] t[s]_[.][More] importantly, using the independence of mechanisms (Peters 

et al., 2017) we can conclude that under this causal model, performing interventions on _Xt[c]_[by changing its value, does] not change the conditional distribution _P_ ( _Yt_ +1 _| Xt[s]_[)][.][Note] that this is exactly the formalisation of our initial intuition: _the output of a particular step of the algorithm (i.e., Yt_ +1 _) depends only on a subset of the current snapshot (i.e., Xt[s][),] and thus it is not affected by the addition of input items that do not interfere with it_ (which we formalise as an **intervention** on _Xt[c]_[).][1][Therefore, given a step] _[ t][∈]_[[1] _[ . . . T]_[]][, for all] _x, x[′] ∈Xt[c]_[, where] _[ X][ c] t_[denotes the domain of] _[ X] t[c]_[, we have] that _Xt[s]_[is an] _[ invariant]_[ predictor of] _[ Y][t]_[+1][under interventions] on _Xt[c]_[:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation_images/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation.pdf-0004-09.png)


where _p_[do][(] _[X] t[c]_[)=] _[x]_ denotes the distribution obtained from assigning _Xt[c]_[the value of] _[ x]_[, i.e. the interventional distribution.] 

Note, however, that Equation (1) does not give us a practical way of ensuring that our neural algorithmic reasoner respects these causal invariances, because it only has access to the entirety of the current snapshot _Xt_ , without knowing its specific subsets _Xt[c]_[and] _[ X] t[s]_[.][More precisely, it is gener-] ally not known _which_ input elements constitute _Xt[s]_[.][For this] reason, _Xt[c]_[and] _[ X] t[s]_[are represented as unobserved random] variables (white nodes) in Figure 2. In the next section, we will describe how to ensure invariant predictions for our reasoner, leveraging only _Xt_ . 

## **4. Size-Invariance through Self-Supervision in Neural Algorithmic Reasoning** 

Given a step _t_ , to ensure invariant predictions of _Yt_ +1 without access to _Xt[s]_[, we construct a] _[ refinement]_[ task] _[ Y] t[R]_ +1[and] learn a representation _f_ ( _Xt, It_ ) to predict _Yt[R]_ +1[,][as][orig-] inally proposed for images in Mitrovic et al. (2021). A refinement for a task (Chalupka et al., 2014) represents a more fine-grained version of the initial task. 

More formally, given two tasks _R_ : _A →B_ and _T_ : _A →_ 

1In bubble sort, adding sorted keys at the end of the array does not affect whether we are swapping the current entries. 

4 

**Neural Algorithmic Reasoning with Causal Regularisation** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation_images/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation.pdf-0005-01.png)


_Figure 4._ Our causal graph with the inclusion of the representation learning components as in Mitrovic et al. (2021). Solid arrows represent the causal relationships. Dashed arrows represent what is used to learn (in the case of _f_ ( _Xt, It_ )) or predict (in the case of _Yt[R]_ +1[) the corresponding random variables.] 

_B[′]_ , task _R_ is more (or equally) fine-grained than task _T_ if, for any two elements _a, a[′] ∈A_ , _R_ ( _a_ ) = _R_ ( _a[′]_ ) = _⇒ T_ ( _a_ ) = _T_ ( _a[′]_ ). 

We will use this concept to show that a representation learned on the refinement task can be effectively used in the original task. Note that, as for _Yt_ +1, we assume _f_ ( _Xt, It_ ) to be the representation learned from _Xt_ of a predefined hint value—indexed by _It_ —for example, the representation of the predecessor of a specific element of the input list. 

Given a step _t_ , let _Yt[R]_ +1[be][a][refinement][of] _[Y][t]_[+1][,][and][let] _f_ ( _Xt, It_ ) be a representation learned from _Xt_ , used for the prediction of the refinement (see Figure 4). As we will formally prove, a representation that is invariant in the prediction of the refinement task across changes in _Xt[c]_[is] also invariant in the prediction of the algorithmic step under these changes. Therefore, optimising _f_ ( _Xt, It_ ) to be an invariant predictor for the refinement task _Yt[R]_ +1[represents] a _sufficient_ condition for the invariance in the prediction of the next algorithmic state, _Yt_ +1. 

In the next subsection we present how to learn _f_ ( _Xt, It_ ) in order to be an invariant predictor of _Yt[R]_ +1[under changes in] _Xt[c]_[. Then, we show that this represents a sufficient condition] for _f_ ( _Xt, It_ ) to be an invariant predictor of _Yt_ +1 across changes in _Xt[c]_[.] 

## **4.1. Learning an invariant predictor of the refinement** 

We consider _Yt[R]_ +1[to][be][the][most-fine-grained][refinement] task, which corresponds to classifying each (hint) instance individually, that is, a contrastive learning objective where 

we want to distinguish each hint from all others. This represents the _most-fine-grained refinement_ , because _Yt[R]_ +1[(] _[a]_[) =] _Yt[R]_ +1[(] _[a][′]_[)] _[⇐⇒][a]_[=] _[a][′]_[,][by definition.][Our goal is to learn] _f_ ( _Xt, It_ ) to be an invariant predictor of _Yt[R]_ +1[under changes] (interventions) of _Xt[c]_[.][Thus, given a step] _[ t][ ∈]_[[1] _[ . . . T]_[]][, for] all _x, x[′] ∈Xt[c]_[, we want] _[ f]_[(] _[X][t][, I][t]_[)][ such that] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation_images/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation.pdf-0005-10.png)


where _p_[do][(] _[X] t[c]_[)] is the interventional distribution and _Xt[c]_[de-] notes the domain of _Xt[c]_[.][Since we do not have access to] _[ X] t[c]_[,] as it is unobserved (it is a white node in Figures 2 and 4), we cannot explicitly intervene on it. Thus, we simulate interventions on _Xt[c]_[through data augmentation.] 

As we are interested in being invariant to appropriate size changes, we design a data augmentation procedure tailored for neural algorithmic reasoning, which mimics interventions changing the size of the input. Given a current snapshot of the algorithm on a given input, the data augmentation procedure should produce an augmented input which is larger, but on which the execution of the current step is going to proceed identically. 

For example, a valid augmentation in bubble sort at a certain step consists of adding new elements to the tail of the input list, since the currently-considered swap will occur (or not) regardless of any elements added there. Thus, the valid augmentations for the bubble sort algorithm at a given step are all possible ways to add items in such a way that ensures that the one-step execution is unaffected by this addition. 

To learn an encoder _f_ ( _Xt, It_ ) that satisfies Equation (2), we propose to explicitly enforce invariance under valid augmentations. Such augmentations, as discussed, provide us with diverse inputs with an identical intermediate execution step. 

Specifically, we use the ReLIC objective (Mitrovic et al., 2021) as a regularisation term, which we adapt to our causal graph as follows. Consider a time step, _t_ , and let _Dt_ be the dataset containing the snapshots at time _t_ for all the inputs. Let _it, jt ∈ It_ be two indices, and denote by _alk_ = ( _al, ak_ ) _∈Axt × Axt_ a pair of augmentations, with _Axt_ the set of all possible valid augmentations at _t_ for _xt_ (which simulate the interventions on _Xt[c]_[).] 

The objective function to optimise becomes: 

_Lt_ = 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation_images/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation.pdf-0005-18.png)


with _x[a] t_[the][data][augmented][with][augmentation] _[a]_[,][and] _[α]_ a weighting of the KL divergence penalty. The first term 

5 

**Neural Algorithmic Reasoning with Causal Regularisation** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation_images/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation.pdf-0006-01.png)


**----- Start of picture text -----**<br>
1 1<br>3 5 3 5<br>6<br>2 4 2 4 7<br>**----- End of picture text -----**<br>


_Figure 5._ Example of applying our data augmentation and contrastive loss, following the example in Figure 1. An input graph (left) is augmented by adding nodes and edges (right), such that the next step—making 2 the parent of 4, i.e. _π_ 4 = 2—remains the same. The representation of the pair (4 _,_ 2) is hence contrasted against all other representations of pairs (4 _, u_ ) in the augmented graph. In other words, the green edge is the _positive_ pair to the blue edge, with other edges (in red) being _negative_ pairs to it. 

represents a contrastive objective where we compare a hint representation in _x[a] t[l]_[,][namely] _[f]_[(] _[x][a] t[l][, i][t]_[)][,][with][all][the][pos-] sible representations in _x[a] t[k]_[,] _[f]_[(] _[x][a] t[k][, j][t]_[)][.][Note][that][this][is] different from standard contrastive objectives, where negative examples are taken from the batch. Due to space constraints, we expand on the derivation of Equation (3) in Appendix A. 

In practice, we consider only one augmentation per graph, which is equivalent to setting _al_ to the identity transformation. Consequently, the hint representation in the original graph _f_ ( _x[a] t[l][, i][t]_[)][ is regularised to be similar to the hint rep-] resentation in the augmentation _f_ ( _x[a] t[k][, i][t]_[)][and][dissimilar] to all other possible representations in the augmentation _f_ ( _x[a] t[k][, j][t]_[)][,] _[j][t]_[=] _[i][t]_[.][Similarly,][the][hint][representation][in] the augmentation _f_ ( _x[a] t[k][, i][t]_[)][ is regularised to be similar to] the hint representation in the original graph _f_ ( _x[a] t[l][, i][t]_[)][ and] dissimilar to all other possible representations in the original graph _f_ ( _x[a] t[l][, j][t]_[)][,] _[ j][t]_[=] _[ i][t]_[.] 

We follow the standard setup in contrastive learning and implement _ϕ_ ( _f_ ( _x[a] t[l][, i][t]_[)] _[, f]_[(] _[x][a] t[k][, i][t]_[))] = _⟨ h_ ( _f_ ( _x[a] t[l]_[)] _[, i][t]_[)] _[, h]_[(] _[f]_[(] _[x][a] t[k][, i][t]_[))] _[ ⟩][/τ]_[with] _[ h]_[ a fully-connected] neural network and _τ_ a temperature parameter. Finally, we use a KL penalty to ensure invariance in the probability distribution across augmentations. This is a requirement for satisfying the assumptions of our key theoretical result. 

**Example.** To better understand Equation (3), we provide an example illustrated in Figure 5. We will consider one of the algorithms in CLRS-30—Kosaraju’s strongly connected component (SCC) algorithm (Aho et al., 1974)—which consists of two invocations of depth-first search (DFS). 

Let _G_ = ( _V, E_ ) be an input graph to the SCC algorithm. Further, assume that at step _t_ , the algorithm is visiting a node _v ∈ V_ . We will focus on the prediction of the _parent_ 

of _v_ : the node from which we have reached _v_ in the current DFS invocation. Note that, in practice, this is a classification task where node _v_ decides which of the other nodes is its parent. Accordingly, given a particular node _v_ , our model computes a representation for every other node _u ∈ V_ . This representation is which is then passed through a final classifier, outputting the (unnormalised) probability of _u_ being the parent of _v_ . 

Now, consider any augmentation of _G_ ’s nodes and edges that does not disrupt the current step of the search algorithm, denoted by _G[a]_ = ( _V[a] , E[a]_ ). For example, as the DFS implementation in CLRS-30 prefers nodes with a smaller id value, a valid augmentation can be obtained by adding nodes with a larger id than _v_ to _V[a]_ , and adding edges from them to _v_ in _E[a]_ (dashed nodes and edges in Figure 5). Note that this augmentation does not change the predicted parent of _v_ . We can enforce that our representations respect this constraint by using our regularisation loss in Equation (3). 

Given a node _v ∈ V_ , we denote the representation of its parent node, _πv ∈ V_ by _f_ ( _G,_ ( _v, πv_ )). This representation is contrasted to _all_ other representations of nodes _w ∈ V[a]_ in the augmented graph, that is _f_ ( _G[a] ,_ ( _v, w_ )).[2] 

More precisely, the most similar representation of _f_ ( _G,_ ( _v, πv_ )) is the representation _in the augmentation_ of the parent of _v_ , _f_ ( _G[a] ,_ ( _v, πv_ )), while the representations associated to all other nodes (including the added ones) represent the negative examples _f_ ( _G[a] ,_ ( _v, w_ )), for _w_ = _πv_ . 

Figure 5 illustrates the prediction of the parent of node _v_ = 4. In this case, _it_ in Equation (3) indexes the true parent of node 4, namely _π_ 4 = 2, and therefore _it_ = (4 _,_ 2), while _jt_ iterates over all other possible indices of the form (4 _, u_ ), _u ∈ V[a]_ , indeed representing _all other possible parents_ of 4. The objective of Equation (3) is to make the true parent representation in the original graph _f_ ( _G,_ (4 _,_ 2)) similar to the true parent representation in the augmentation _f_ ( _G[a] ,_ (4 _,_ 2)), and dissimilar to the representations of the other possible parents in the augmentation _f_ ( _G[a] ,_ (4 _, u_ )), _u ∈ V[a]_ . The same process applies to the augmentation. 

## **4.2. Implications of the invariance** 

In the previous subsection, we have presented a selfsupervised objective, justified by our assumed causal graph, in order to learn invariant predictors for a refinement task _Yt[R]_ +1[under changes of] _[ X] t[c]_[.][However, our initial goal was] to ensure invariance in the prediction of algorithmic hints _Yt_ +1 across _Xt[c]_[.][Now we will bridge these two aims.] 

In the following, we show how learning a representation that is an invariant predictor of _Yt[R]_ +1[under changes of] _[ X] t[c]_ 

> 2Note that, in this case, _It_ is a two-dimensional index, choosing two nodes—i.e., an edge—at once. 

6 

**Neural Algorithmic Reasoning with Causal Regularisation** 

represents a _sufficient_ condition for this representation to be invariant to _Xt[c]_[when predicting] _[ Y][t]_[+1][.] 

**Theorem 4.1.** _Consider an algorithm and let t ∈_ [1 _. . . T_ ] _be one of its steps. Let Yt_ +1 _be the task representing a prediction of the algorithm step and let Yt[R]_ +1 _[be a refinement] of such task. If f_ ( _Xt, It_ ) _is an invariant representation for Yt[R]_ +1 _[under changes in][ X] t[c][, then][ f]_[(] _[X][t][, I][t]_[)] _[ is an invariant] representation for Yt_ +1 _under changes in Xt[c][, that is, for all] x, x[′] ∈Xt[c][, the following holds:]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation_images/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation.pdf-0007-03.png)


We prove Theorem 4.1 in Appendix D. Note that this justifies our self-supervised objective: by learning invariant representations though a refinement task, we can also guarantee invariance in the hint prediction. In other words, we can _provably_ ensure that the prediction of an algorithm step is not affected by changes in the input that do not interfere with the current execution step. Since we can express these changes in the form of _addition_ of input nodes, we are ensuring that the hint prediction is the same on two inputs of different sizes, but identical current algorithmic step. 

## **5. Experiments** 

We conducted an extensive set of experiments to answer the following main questions: 

1. _Can our model, Hint-ReLIC, which relies on the addition of our causality-inspired self-supervised objective, outperform the corresponding base model in practice?_ 

2. _What is the importance of such objective when compared to other changes made with respect to the previous state-of-the-art model?_ 

3. _How does Hint-ReLIC compare to a model which does not leverage hints at all, directly predicting the output from the input? Are hints necessary?_ 

**Model.** As a base model, we use the Triplet-GMPNN architecture proposed by Ibarz et al. (2022), which consists of a fully-connected MPNN (Gilmer et al., 2017) where the input graph is encoded in the edge features, augmented with gating and triplet reasoning (Dudzik & Veliˇckovi´c, 2022). 

We replace the loss for predicting the next-step hint in the base model with our regularisation objective (Equation (3)), which aims at learning hint representations that are _invariant to size changes that are irrelevant to the current step_ via constrastive and KL losses. 

We make an additional change with respect to the base model, consisting of including the _reversal_ of hints of pointer type. More specifically, given an input graph, if a node _A_ points to another node _B_ in the graph, we include an additional (edge-based) hint representing the pointer from _B_ to _A_ . This change (which we refer to as **reversal** in the results) consists simply in the inclusion of these additional hints, and we study the impact of this addition in Section 5.1. The resulting model is what we call Hint-ReLIC. 

**Data augmentations.** To simulate interventions on _Xt[c]_ and learn invariant representations, we design augmentation procedures which construct augmented data given an input and an algorithm step, such that the step of the algorithm is the same on the original input and on the augmented data. 

We consider simple augmentations, which we describe in detail in Appendix E. To reduce the computational overhead, given an input graph, instead of sampling an augmentation at each algorithm step, we sample a single step, _t_[˜] _∼U{_ 1 _, T }_ , and construct an augmentation only for the sampled step. Then, we use the (same) constructed augmentation in all the steps _until_ the sampled one, _t ≤ t_[˜] . This follows from the consideration that, if augmentations are carefully constructed, the execution of the algorithm is the same not only in the next step but in all steps leading up to that. 

Whenever possible, we relax the requirement of having the augmentation with _exactly_ the same execution, and we allow for approximate augmentations, in order to avoid over-engineering the methodology and obtain a more robust model. This results in more general and simpler augmentations, though we expect more tailored ones to perform better. We refer the reader to Appendix E for more details. 

We end this paragraph by stressing that we _never_ run the target algorithm on the augmented inputs: rather, we directly construct them to have the same next execution step as the corresponding inputs. As a result, our method does not require direct access to the algorithm used to generate the inputs. Furthermore, the number of nodes in our augmentations is at most one more than the number of nodes in the largest training input example. This means that, in all of our experiments, we still never significantly cross the intended test size distribution shift during training. 

**Datasets.** We run our method on a diverse subset of the algorithms present in the CLRS benchmark consisting of: 1. _DFS-based algorithms_ (Articulation Points, Bridges, Strongly Connected Components (Aho et al., 1974), Topological Sort (Knuth, 1973)); 2. _Other graph-based algorithms_ (Bellman-Ford (Bellman, 1958), BFS (Moore, 1959), DAG Shortest Paths, Dijkstra (Dijkstra et al., 1959), Floyd-Warshall (Floyd, 1962), MST-Kruskal (Kruskal, 1956), MST-Prim (Prim, 1957)); 3. _Sorting algorithms_ (Bubble 

7 

**Neural Algorithmic Reasoning with Causal Regularisation** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation_images/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation.pdf-0008-01.png)


**----- Start of picture text -----**<br>
Hint-ReLIC (ours)<br>Baseline<br>100<br>80<br>60<br>40<br>20<br>0<br>Average score [%]<br>Heapsort Strongly Conn. Comps. Floyd-Warshall Quicksort Insertion Sort Binary Search Bubble Sort DAG Shortest Paths Articulation Points Topological Sort Bridges MST Kruskal Dijkstra Minimum MST Prim BFS Bellman-Ford Overall Average<br>**----- End of picture text -----**<br>


_Figure 6._ Per-algorithm comparison of the Triplet-GMPNN baseline (Ibarz et al., 2022) and our Hint-ReLIC. Error bars represent the standard error of the mean across three random seeds. The final column shows the average and standard error of the mean performances across the different algorithms. 

Sort, Heapsort (Williams, 1964), Insertion Sort, Quicksort (Hoare, 1962)); 4. _Searching algorithms_ (Binary-search, Minimum). This subset is chosen as it contains most algorithms suffering from out-of-distribution performance drops in current state-of-the-art; see Ibarz et al. (2022, Table 2). 

**Results.** Figure 6 compares the out-of-distribution (OOD) performances of the Triplet-GMPNN baseline, which we have re-trained and evaluated in our experiments, to our model Hint-ReLIC, as described above. Hint-ReLIC performs better or comparable to the existing state-of-the-art baseline, showcasing how the proposed procedure appears to be beneficial not only theoretically, but also in practice. The most significant improvements can be found in the sorting algorithms, where we obtain up to 3 _×_ increased performance. 

_Table 1. Effect of the inclusion of pointers’ reversal on each algorithm._ The table shows mean and stderr of the OOD micro-F1 score after 10 _,_ 000 training steps, across different seeds. 

|**Alg.**|**Baseline**|**Baseline + reversal**|**Hint-ReLIC (ours)**|
|---|---|---|---|
|Articulation points|88_._93%_±_1_._92|91_._04%_±_0_._92|**98**_._**45**%_±_0_._60|
|Bridges|93_._75%_±_2_._73|97_._70%_±_0_._34|**99**_._**32**%_±_0_._09|
|SCC|38_._53%_±_0_._45|31_._40%_±_8_._80|**76**_._**79**%_±_3_._04|
|Topological sort|87_._27%_±_2_._67|88_._83%_±_7_._29|**96**_._**59**%_±_0_._20|
|Bellman-Ford|**96**_._**67**%_±_0_._81|95_._02%_±_0_._49|95_._54%_±_1_._06|
|BFS|99_._64%_±_0_._05|**99**_._**93**%_±_0_._03|99_._00%_±_0_._21|
|DAG Shortest Paths|88_._12%_±_5_._70|96_._61%_±_0_._61|**98**_._**17**%_±_0_._26|
|Dijkstra|93_._41%_±_1_._08|91_._50%_±_1_._85|**97**_._**74**%_±_0_._50|
|Floyd-Warshall|46_._51%_±_1_._30|46_._28%_±_0_._80|**72**_._**23**%_±_4_._84|
|MST-Kruskal|91_._18%_±_1_._05|89_._93%_±_0_._43|**96**_._**01**%_±_0_._45|
|MST-Prim|87_._64%_±_1_._79|86_._95%_±_2_._34|**87**_._**97**%_±_2_._94|
|Insertion sort|75_._28%_±_5_._62|87_._21%_±_2_._80|**92**_._**70**%_±_1_._29|
|Bubble sort|79_._87%_±_6_._85|80_._51%_±_9_._10|**92**_._**94**%_±_1_._23|
|Quicksort|70_._53%_±_11_._59|85_._69%_±_4_._53|**93**_._**30**%_±_1_._96|
|Heapsort|32_._12%_±_5_._20|49_._13%_±_10_._35|**95**_._**16**%_±_1_._27|
|Binary Search|74_._60%_±_3_._61|50_._42%_±_8_._45|**89**_._**68**%_±_2_._13|
|Minimum|97_._78%_±_0_._63|98_._43%_±_0_._01|**99**_._**37**%_±_0_._20|



## **5.1. Ablation study** 

In this section we study the contribution and importance of two main components of our methodology. First, we consider the impact of the change we made with respect to the original baseline proposed in Ibarz et al. (2022), namely the inclusion of the reversal of hint pointers. Second, as we propose a novel way to leverage hints through our selfsupervised objective, which is different from the direct supervision in the baseline, one may wonder whether completely removing hints can achieve even better scores. Thus, we also study the performance when completely disregarding hints and directly going from input to output. Finally, we refer the reader to Appendix F.1 for additional ablation experiments, including the removal of the KL component in Equation (3)—which is necessary for the theoretical results but may not always be needed in practice. 

**The effect of the inclusion of pointers’ reversal.** As discussed above, pointers’ reversal simply consists of adding an additional hint for each hint of pointer type (if any), such that a node not only has the information representing which other node it points to, but also from which nodes it is pointed by. We study the impact of this inclusion by running the baseline with these additional hints, and evaluate its performance against both the baseline and our Hint-ReLIC. Table 1 shows that this addition, which we refer to as **Baseline + reversal** , indeed leads to improved results for certain algorithms, but does not obtain the predictive performances we reached with our regularisation objective. 

**The removal of hints.** While previous works directly included the supervision on the hint predictions, we argue in 

8 

**Neural Algorithmic Reasoning with Causal Regularisation** 

_Table 2. Importance of hint usage in the final performance._ The table shows mean and stderr of the OOD micro-F1 score after 10 _,_ 000 training steps, across different seeds. 

|**Alg.**|**No Hints**|**Hint-ReLIC (ours)**|
|---|---|---|
|Articulation points|81_._97%_±_5_._08|**98**_._**45**%_±_0_._60|
|Bridges|95_._62%_±_1_._03|**99**_._**32**%_±_0_._09|
|SCC|57_._63%_±_0_._68|**76**_._**79**%_±_3_._04|
|Topological sort|84_._29%_±_1_._16|**96**_._**59**%_±_0_._20|
|Bellman-Ford|93_._26%_±_0_._04|**95**_._**54**%_±_1_._06|
|BFS|**99**_._**89**%_±_0_._03|99_._00%_±_0_._21|
|DAG Shortest Paths|97_._62%_±_0_._62|**98**_._**17**%_±_0_._26|
|Dijkstra|95_._01%_±_1_._14|**97**_._**74**%_±_0_._50|
|Floyd-Warshall|40_._80%_±_2_._90|**72**_._**23**%_±_4_._84|
|MST-Kruskal|92_._28%_±_0_._82|**96**_._**01**%_±_0_._45|
|MST-Prim|85_._33%_±_1_._21|**87**_._**97**%_±_2_._94|
|Insertion sort|77_._29%_±_7_._42|**92**_._**70**%_±_1_._29|
|Bubble sort|81_._32%_±_6_._50|**92**_._**94**%_±_1_._23|
|Quicksort|71_._60%_±_2_._22|**93**_._**30**%_±_1_._96|
|Heapsort|68_._50%_±_2_._81|**95**_._**16**%_±_1_._27|
|Binary Search|**93**_._**21**%_±_1_._10|89_._68%_±_2_._13|
|Minimum|99_._24%_±_0_._21|**99**_._**37**%_±_0_._20|



favour of a novel way of leveraging hints. We use hints first to construct the augmentations representing the same algorithm step, and then we employ their representations in the self-supervised objective. An additional valid model might consist of a model that directly goes from input to output and completely ignores hints. In Table 2 we show that this **No Hints** model can achieve very good performances, but it is still generally outperformed by Hint-ReLIC. 

## **6. Conclusions** 

In this work we propose a self-supervised learning objective that employs augmentations derived from available hints, which represent intermediate steps of an algorithm, as a way to better ground the execution of GNN-based algorithmic reasoners on the computation that the target algorithm performs. Our Hint-ReLIC model, based on such self-supervised objective, leads to algorithmic reasoners that produce more robust outputs of the target algorithms, especially compared to autoregressive hint prediction. In conclusion, hints can take you a long way, if used in the right way. 

## **Acknowledgements** 

The authors would like to thank Andrew Dudzik and Daan Wierstra for valuable feedback on the paper. They would also like to show their gratitude to the Learning at Scale team at DeepMind for a supportive atmosphere. 

## **References** 

- Aho, A. V., Hopcroft, J. E., and Ullman, J. D. The design and analysis of computer algorithms. _Reading_ , 1974. 

Alet, F., Doblar, D., Zhou, A., Tenenbaum, J., Kawaguchi, 

- K., and Finn, C. Noether networks: meta-learning useful conserved quantities. _Advances in Neural Information Processing Systems_ , 34:16384–16397, 2021. 

- Bansal, A., Schwarzschild, A., Borgnia, E., Emam, Z., Huang, F., Goldblum, M., and Goldstein, T. End-toend algorithm synthesis with recurrent networks: Logical extrapolation without overthinking. _arXiv preprint arXiv:2202.05826_ , 2022. 

- Bellman, R. On a routing problem. _Quarterly of applied mathematics_ , 16(1):87–90, 1958. 

- Beurer-Kellner, L., Vechev, M., Vanbever, L., and Velickoviˇ c, P.´ Learning to configure computer networks with neural algorithmic reasoning. In _Advances in Neural Information Processing Systems_ , 2022. 

- Bevilacqua, B., Zhou, Y., and Ribeiro, B. Size-invariant graph representations for graph classification extrapolations. In _International Conference on Machine Learning_ , pp. 837–851. PMLR, 2021. 

- Bronstein, M. M., Bruna, J., Cohen, T., and Velickoviˇ c,´ P. Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. _arXiv preprint arXiv:2104.13478_ , 2021. 

- Buffelli, D., Lio,` P., and Vandin, F. Sizeshiftreg: a regularization method for improving size-generalization in graph neural networks. In _Advances in Neural Information Processing Systems_ , 2022. 

- Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., and Joulin, A. Unsupervised learning of visual features by contrasting cluster assignments. _Advances in Neural Information Processing Systems_ , 33:9912–9924, 2020. 

- Chalupka, K., Perona, P., and Eberhardt, F. Visual causal feature learning. _arXiv preprint arXiv:1412.2309_ , 2014. 

- Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In _International conference on machine learning_ , pp. 1597–1607. PMLR, 2020. 

- Chen, Y., Zhang, Y., Bian, Y., Yang, H., KAILI, M., Xie, B., Liu, T., Han, B., and Cheng, J. Learning causally invariant representations for out-of-distribution generalization on graphs. In _Advances in Neural Information Processing Systems_ , 2022. 

- Corso, G., Cavalleri, L., Beaini, D., Lio, P., and Veli` ckoviˇ c,´ P. Principal neighbourhood aggregation for graph nets. _Advances in Neural Information Processing Systems_ , 33: 13260–13271, 2020. 

9 

**Neural Algorithmic Reasoning with Causal Regularisation** 

- Davies, A., Velickoviˇ c,´ P., Buesing, L., Blackwell, S., Zheng, D., Tomasev, N., Tanburn, R., Battaglia, P., Blun-ˇ dell, C., Juhasz,´ A., et al. Advancing mathematics by guiding human intuition with ai. _Nature_ , 600(7887):70– 74, 2021. 

- Deac, A.-I., Velickoviˇ c,´ P., Milinkovic, O., Bacon, P.-L., Tang, J., and Nikolic, M. Neural algorithmic reasoners are implicit planners. _Advances in Neural Information Processing Systems_ , 34:15529–15542, 2021. 

- Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., and Kaiser, L. Universal transformers. In _International Conference on Learning Representations_ , 2019. 

- Diao, C. and Loynd, R. Relational attention: Generalizing transformers for graph-structured tasks. _arXiv preprint arXiv:2210.05062_ , 2022. 

- Dijkstra, E. W. et al. A note on two problems in connexion with graphs. _Numerische mathematik_ , 1(1):269–271, 1959. 

- Dudzik, A. J. and Velickoviˇ c, P.´ Graph neural networks are dynamic programmers. In _Advances in Neural Information Processing Systems_ , 2022. 

- Floyd, R. W. Algorithm 97: shortest path. _Communications of the ACM_ , 5(6):345, 1962. 

- Gasteiger, J., Shuaibi, M., Sriram, A., Gunnemann,¨ S., Ulissi, Z. W., Zitnick, C. L., and Das, A. How do graph networks generalize to large and diverse molecular systems? _ArXiv_ , abs/2204.02782, 2022. 

- Gers, F. A. and Schmidhuber, J. Lstm recurrent networks learn simple context-free and context-sensitive languages. _IEEE transactions on neural networks_ , 12 6:1333–40, 2001. 

- Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and Dahl, G. E. Neural message passing for quantum chemistry. In _Proceedings of the 34th International Conference on Machine Learning_ , ICML’17, pp. 1263–1272. JMLR.org, 2017. 

- Graves, A., Wayne, G., and Danihelka, I. Neural turing machines. _ArXiv_ , abs/1410.5401, 2014. 

- Graves, A., Wayne, G., Reynolds, M., Harley, T., Danihelka, I., Grabska-Barwinska,´ A., Colmenarejo, S. G., Grefenstette, E., Ramalho, T., Agapiou, J., Badia, A. P., Hermann, K. M., Zwols, Y., Ostrovski, G., Cain, A., King, H., Summerfield, C., Blunsom, P., Kavukcuoglu, K., and Hassabis, D. Hybrid computing using a neural network with dynamic external memory. _Nature_ , 538 (7626):471–476, October 2016. ISSN 00280836. 

- Grill, J.-B., Strub, F., Altche, F., Tallec, C., Richemond, P.,´ Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z., Gheshlaghi Azar, M., et al. Bootstrap your own latent-a new approach to self-supervised learning. _Advances in neural information processing systems_ , 33:21271–21284, 2020. 

- He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. Momentum contrast for unsupervised visual representation learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 9729–9738, 2020. 

- Hoare, C. A. Quicksort. _The Computer Journal_ , 5(1):10–16, 1962. 

- Ibarz, B., Kurin, V., Papamakarios, G., Nikiforou, K., Bennani, M., Csordas, R., Dudzik, A. J., Bo´ snjak, M., Vitvit-ˇ skyi, A., Rubanova, Y., Deac, A., Bevilacqua, B., Ganin, Y., Blundell, C., and Velickoviˇ c, P.´ A generalist neural algorithmic learner. In _The First Learning on Graphs Conference_ , 2022. 

- Jegelka, S. Theory of graph neural networks: Representation and learning. _arXiv preprint arXiv:2204.07697_ , 2022. 

- Joshi, C. K., Cappart, Q., Rousseau, L.-M., and Laurent, T. Learning the travelling salesperson problem requires rethinking generalization. _Constraints_ , 27:70 – 98, 2020. 

Knuth, D. E. Fundamental algorithms. 1973. 

- Kruskal, J. B. On the shortest spanning subtree of a graph and the traveling salesman problem. _Proceedings of the American Mathematical society_ , 7(1):48–50, 1956. 

- Mahdavi, S., Swersky, K., Kipf, T., Hashemi, M., Thrampoulidis, C., and Liao, R. Towards better out-ofdistribution generalization of neural algorithmic reasoning tasks. _arXiv preprint arXiv:2211.00692_ , 2022. 

- Mitrovic, J., McWilliams, B., Walker, J. C., Buesing, L. H., and Blundell, C. Representation learning via invariant causal mechanisms. In _International Conference on Learning Representations_ , 2021. 

- Moore, E. F. The shortest path through a maze. In _Proc. Int. Symp. Switching Theory, 1959_ , pp. 285–292, 1959. 

- Palm, R. B., Paquet, U., and Winther, O. Recurrent relational networks. In _Neural Information Processing Systems_ , 2017. 

- Pearl, J. _Causality_ . Cambridge university press, 2009. 

- Peters, J., Janzing, D., and Scholkopf, B.¨ _Elements of causal inference: foundations and learning algorithms_ . The MIT Press, 2017. 

10 

**Neural Algorithmic Reasoning with Causal Regularisation** 

- Prim, R. C. Shortest connection networks and some generalizations. _The Bell System Technical Journal_ , 36(6): 1389–1401, 1957. 

- Selsam, D., Lamm, M., Bunz, B., Liang, P., de Moura, L.,¨ and Dill, D. L. Learning a SAT solver from single-bit supervision. In _International Conference on Learning Representations_ , 2019. 

- Suresh, S., Li, P., Hao, C., and Neville, J. Adversarial graph augmentation to improve graph contrastive learning. In _Advances in Neural Information Processing Systems_ , 2021. 

- Tang, H., Huang, Z., Gu, J., Lu, B.-L., and Su, H. Towards scale-invariant graph-related problem solving by iterative homogeneous gnns. _Advances in Neural Information Processing Systems_ , 33:15811–15822, 2020. 

- Thakoor, S., Tallec, C., Azar, M. G., Azabou, M., Dyer, E. L., Munos, R., Velickoviˇ c, P., and Valko, M.´ Largescale representation learning on graphs via bootstrapping. In _International Conference on Learning Representations_ , 2022. 

- Tomasev, N., Bica, I., McWilliams, B., Buesing, L., Pascanu, R., Blundell, C., and Mitrovic, J. Pushing the limits of self-supervised resnets: Can we outperform supervised learning without labels on imagenet? _arXiv preprint arXiv:2201.05119_ , 2022. 

- Velickoviˇ c, P. and Blundell, C. Neural algorithmic reasoning.´ _Patterns_ , 2(7):100273, 2021. 

- Velickoviˇ c, P., Badia, A. P., Budden, D., Pascanu, R., Ban-´ ino, A., Dashevskiy, M., Hadsell, R., and Blundell, C. The CLRS algorithmic reasoning benchmark. In _International Conference on Machine Learning_ , 2022a. 

- Velickoviˇ c, P., Bo´ snjak, M., Kipf, T., Lerchner, A., Hadsell,ˇ R., Pascanu, R., and Blundell, C. Reasoning-modulated representations. In _The First Learning on Graphs Conference_ , 2022b. 

   - Xie, Y., Xu, Z., Zhang, J., Wang, Z., and Ji, S. Selfsupervised learning of graph neural networks: A unified review. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2022. 

   - Xu, K., Li, J., Zhang, M., Du, S. S., ichi Kawarabayashi, K., and Jegelka, S. What can neural networks reason about? In _International Conference on Learning Representations_ , 2020. 

   - Xu, K., Zhang, M., Li, J., Du, S. S., Kawarabayashi, K.-I., and Jegelka, S. How neural networks extrapolate: From feedforward to graph neural networks. In _International Conference on Learning Representations_ , 2021. 

   - Yehudai, G., Fetaya, E., Meirom, E., Chechik, G., and Maron, H. From local structures to size generalization in graph neural networks. In _International Conference on Machine Learning_ , pp. 11975–11986. PMLR, 2021. 

   - You, Y., Chen, T., Sui, Y., Chen, T., Wang, Z., and Shen, Y. Graph contrastive learning with augmentations. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M. F., and Lin, H. (eds.), _Advances in Neural Information Processing Systems_ , volume 33, pp. 5812–5823. Curran Associates, Inc., 2020. 

   - You, Y., Chen, T., Wang, Z., and Shen, Y. Bringing your own view: Graph contrastive learning without prefabricated data augmentations. In _Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining_ , pp. 1300–1309, 2022. 

   - Zhou, Y., Kutyniok, G., and Ribeiro, B. OOD link prediction generalization capabilities of message-passing GNNs in larger test graphs. In _Advances in Neural Information Processing Systems_ , 2022. 

   - Zhu, Y., Xu, Y., Yu, F., Liu, Q., Wu, S., and Wang, L. Deep Graph Contrastive Representation Learning. In _ICML Workshop on Graph Representation Learning and Beyond_ , 2020. 

- Velickoviˇ c, P., Fedus, W., Hamilton, W. L., Li´ o, P., Bengio,` Y., and Hjelm, R. D. Deep graph infomax. In _International Conference on Learning Representations_ , 2019. 

- Velickoviˇ c,´ P., Ying, R., Padovano, M., Hadsell, R., and Blundell, C. Neural execution of graph algorithms. In _International Conference on Learning Representations_ , 2020. 

- Williams, J. W. J. Algorithm 232: heapsort. _Commun. ACM_ , 7:347–348, 1964. 

- Wu, Z., Xiong, Y., Yu, S. X., and Lin, D. Unsupervised feature learning via non-parametric instance discrimination. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 3733–3742, 2018. 

11 

**Neural Algorithmic Reasoning with Causal Regularisation** 

## **A. Derivation of our Self-Supervised Objective** 

Equation (3) represents our objective function to optimise, which we derived by adapting the ReLIC objective (Mitrovic et al., 2021) to our causal graph. Note that we propose a unique and novel way of employing a contrastive-learning based objective, which is also different from Mitrovic et al. (2021), as we consider positive and negative examples for each hint representation in an input graph to be hint representations in valid augmentations of the graph. To better understand the equation, in this section we expand the derivation of our objective. 

Recall that our goal is to learn _f_ ( _Xt, It_ ) to be an invariant predictor of _Yt[R]_ +1[under changes (interventions) of] _[ X] t[c]_[.][As we] do not have access to _Xt[c]_[, because we do not know which subset of the input forms] _[ X] t[c]_[, we simulate interventions on] _[ X] t[c]_ through data augmentations. Therefore, our goal becomes to learn _f_ ( _Xt, It_ ) to be an invariant predictor of _Yt[R]_ +1[under] (valid) augmentations, that is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation_images/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation.pdf-0012-04.png)


where _Axt_ contains all possible valid augmentations at _t_ for _xt_ , and _p_[do][(] _[a]_[)] represents the simulation of the intervention on _Xt[c]_[through data augmentation] _[ a]_[.][Following][ Mitrovic et al.][ (][2021][), we enforce this invariance through a regularisation] objective, which for every time step _t_ has the following form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation_images/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation.pdf-0012-06.png)


for some small number _ρ_ . Since we consider _L_[ˆ] to be a contrastive learning objective, we take pairs of hint representations, indexed by _it_ and _jt_ , to compute similarity scores and use pairs of augmentations _alk_ = ( _al, ak_ ) _∈Axt × Axt_ , that is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation_images/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation.pdf-0012-08.png)


where _f_ is a neural network and _ϕ_ is a (learnable) function to compute the similarity between two representations. Note that, in words, we are computing the similarity of two hint representations (indexed by _it_ and _jt_ , respectively) in two data augmentations (obtained from _al_ and _ak_ ). Now, note that we want the representations of the same hint to be similar in the two augmentations. This means that the hint representation indexed by _it_ in _x[a] t[l]_[must be similar to the hint representation indexed] by _it_ in _x[a] t[k]_[.][Obviously, the same must be true when considering] _[ j][t]_[instead of] _[ i][t]_[.][Furthermore, we want representations] of different hints to be dissimilar in the two augmentations. Putting all together, our objective function at time _t_ can be rewritten as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation_images/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation.pdf-0012-10.png)


where _Dt_ is the dataset containing the snapshots at time _t_ for all the inputs, _it, jt ∈ It_ are two indices, _alk_ = ( _al, ak_ ) _∈ Axt × Axt_ is a pair of augmentations, with _Axt_ the set of all possible valid augmentations at _t_ for _xt_ (which simulate the interventions on _Xt[c]_[).][Finally,] _[α]_[is][the][weighting][of][the][KL][divergence][penalty][and] _[p]_[do][(] _[a][lk]_[)][is][a][shorthand][for] _p_[do][(] _[a][lk]_[)] ( _Yt[R]_ +1 _[|][f]_[(] _[x][t][, i][t]_[))][.] 

We note here that _jt_ is such that the hint representations of _it_ and _jt_ are actually different. In the SCC example in the main text (visualised in Figure 5), the sum over _jt_ is a sum over all other possible parents of the node _v_ under consideration. In a DFS’s execution, when the hint under consideration is the colour of a node in an input graph, its representation is regularised to be similar to the hint representation of the same node in the augmentation, and dissimilar to all other hint representations in the augmentation corresponding to _different_ colours. 

## **B. Causal Graph and Representation Learning Components** 

In this section we expand on the definition of our causal graph and on the representation learning components, justifying design choices that were left implicit in the main paper due to space constraints. Specifically, we first explain thoroughly the causal relationships among the random variables and then stress how we use those variables in a learning setting. 

Recall that our causal graph (Figure 2) describes the data generation process of an algorithmic trajectory. We denote by _X_ 1 the random variable representing the input to our algorithm, and refer by _Xt_ the snapshot at time step _t_ of the algorithm execution on such input, for every time step in the trajectory _t ∈_ [1 _. . . T_ ]. 

12 

**Neural Algorithmic Reasoning with Causal Regularisation** 

We assume _Xt_ to be generated by two random variables, _Xt[c]_[and] _[ X] t[s]_[, which we assume to be] _[ distinct]_[ parts (or splits) of] _Xt_ , which _together_ form the whole _Xt_ . We consider _Xt[c]_[to be the part of the snapshot that does not influence the current] execution of the algorithm at time step _t_ , and can therefore be arbitrarily different without affecting it. We instead denote by _Xt[s]_[the part of the snapshot that determines the current execution of the algorithm at time step] _[ t]_[, and therefore should not be] changed if we do not want to alter the current step execution. 

The current execution is represented as hint values on all nodes and/or edges. We call _Yt_ +1 the execution (or hint) _on a specific node or edge_ chosen accordingly to an index _It_ . Note that the current step hint _Yt_ +1 is represented with an increment of the time step, following a convention we adopt to indicate that we first need an execution and only then (in the next time step) we materialise its results. Further, note that _Yt_ +1 represents the algorithmic step in any node or edge, indicating whether or not it is involved in the current execution, thus either encoding that there is no change (and thus it is not involved in the current step) or representing what is its new hint value. 

By definition of _Xt[c]_[and] _[ X] t[s]_[, the current step of the algorithm on the node or edge indexed by] _[ I][t]_[, namely] _[ Y][t]_[+1][, is determined] by _Xt[s]_[only.] 

Assuming a Markovian execution, executing one algorithm step gives us _Xt[c]_ +1[and] _[ X] t[s]_ +1[, which form the snapshot we] observe _Xt_ +1. Note that _Xt[c]_ +1[and] _[ X] t[s]_ +1[are potentially different from] _[ X] t[c]_[and] _[ X] t[s]_[, because the current execution might] now be determined by very different subsets. Finally, note that we do not need an arrow from _Yt_ +1 to _Xt[c]_[or] _[ X] t[s]_[, because] _Yt_ +1 is _deterministically_ determined by _Xt[s]_[, and therefore all its information can be recovered from] _[ X] t[s]_[.] 

Recall now that our goal is to learn an invariant predictor for the refinement task across changes of _Xt[c]_[, as this represents a] sufficient condition for the invariance in the prediction of _Yt_ +1 (see Theorem 4.1). We denote by _Yt[R]_ +1[the refinement task of] the execution on a specific node or edge chosen according to _It_ . We omit the arrow from _It_ to _Yt[R]_ +1[as the dependency is] already implicit through _Yt_ +1. 

We denote by _f_ ( _Xt, It_ ) the representation of the execution step on a particular node or edge indexed by _It_ , which we learn to predict _Yt[R]_ +1[across changes of] _[ X] t[c]_[.][Finally, note that] _[ f]_[(] _[X][t][, I][t]_[)][ is used by the network to determine the predicted next] snapshot, which is determined by the next step prediction, and therefore it has a (dashed) arrow to _Xt_ +1. 

## **C. Assumptions on Prior Knowledge of the Unobserved** _Xt[c]_ 

## _t_ 

Given a time step _t_ , to ensure invariant predictions of _Yt_ +1 we learn predictors of _Yt[R]_ +1[that are invariant across interventions] on _Xt[c]_[,][simulated][through][data][augmentations.][However,][since] _[X] t[c]_[is][an][unobserved][random][variable,][we][must][make] assumptions about its properties to create valid data augmentations. In this section, we clarify our assumptions about prior knowledge of _Xt[c]_[and propose potential methods to eliminate this assumption in future work.] 

We start by remarking that our neural network is not assumed to have any prior knowledge about _Xt[c]_[.][This knowledge is] enforced into the network through our regularisation objective (Equation (3)), which is driven by appropriately chosen data augmentations. Indeed, those data augmentations do rely on priors that assume something about what _Xt[c]_[might look like.] This is similar to how the choice of data augmentation in image CNNs governs which parts of the image we consider to be “content” and “style”. 

However, for most algorithms of interest, the required priors are conceptually very simple, and a single augmentation may be reused for many algorithms. As an example, for many graph algorithms, it is an entirely safe operation to add disconnected subgraphs – an augmentation we repeatedly employ. Similarly in several sorting tasks, adding elements to the tail end of the list represent a valid augmentation. We provide an exhaustive list of the augmentation for each algorithm in Appendix E. 

Performing data augmentations without knowledge of _Xt[c]_[represents an interesting but challenging direction, that can be] explored in future work. A simple, computationally-expensive, data-augmentation procedure that does not require any knowledge of _Xt[c]_[could consist in randomly augmenting the input graph, run the actual algorithm and consider the generated] graph as a valid augmentation _only if_ the next step execution of the algorithm remains unaltered. A more interesting approach would consist in learning valid augmentations of a given input, perhaps by meta-learning conserved quantities in the spirit of Noether Networks (Alet et al., 2021). Investigating these cases remains an important avenue for future research. 

13 

**Neural Algorithmic Reasoning with Causal Regularisation** 

## **D. Theoretical Analysis** 

**Theorem 4.1.** _Consider an algorithm and let t ∈_ [1 _. . . T_ ] _be one of its steps. Let Yt_ +1 _be the task representing a prediction of the algorithm step and let Yt[R]_ +1 _[be a refinement of such task.][If][ f]_[(] _[X][t][, I][t]_[)] _[ is an invariant representation for][ Y] t[R]_ +1 _[under] changes in Xt[c][, then][ f]_[(] _[X][t][, I][t]_[)] _[ is an invariant representation for][ Y][t]_[+1] _[under changes in][ X] t[c][, that is, for all][ x, x][′][∈X] t[ c][, the] following holds:_ 

_Proof of Theorem 4.1._ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation_images/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation.pdf-0014-04.png)


The first equality is obtained by marginalising over _Yt[R]_ +1[and using the assumption of] _[ Y] t[R]_ +1[being a refinement of] _[ Y][t]_[+1][, which] implies that _Yt[R]_ +1[has all the necessary information to predict] _[ Y][t]_[+1][(and thus we can drop the conditioning on] _[ f]_[(] _[X][t][, I][t]_[)][).] The second equality follows from the fact that the mechanism _Yt_ +1 _|Yt[R]_ +1[is independent of interventions on] _[ X] t[c]_[under our] assumptions. Finally, the third equality follows from the assumption that _f_ ( _Xt, It_ ) is an invariant predictor of _Yt[R]_ +1[under] changes in _Xt[c]_[.] 

## **E. Data Augmentations** 

In this section we expand on our proposed augmentations, which simulate interventions on _Xt[c]_[, valid] _[ until]_[ step] _[ t]_[.][Further,] we report which hints we use in our objective (see Equation (3)) using the naming convention in Veliˇckovi´c et al. (2022a). 

**DFS-based algorithms (Articulation Points, Bridges, Strongly Connected Components, Topological Sort).** We construct exact augmentations for these kinds of problems. First, we sample a step by choosing uniformly at random amongst those where we enter a node for the first time (in case multiple DFSs are being executed for an input, we only consider the first one). Then, we construct a subgraph of nodes with larger node-ids, and we randomly determine connectivity between subgraph’s nodes. Finally, we connect all the subgraph’s nodes to the node we are entering in the sampled step. We contrast the following hints up to the sampled step (we mask out the contrastive loss on later steps): pi ~~h~~ in Articulation Points and Bridges; scc ~~i~~ d ~~h~~ , color and s ~~p~~ rev in Strongly Connected Components; and topo ~~h~~ , color, and s ~~p~~ rev in Topological Sort. 

**Graph-based algorithms (Bellman-Ford, BFS, DAG Shortest Path, Dijkstra, Floyd-Warshall, MST-Kruskal, MSTPrim).** We construct simple but exact augmentations consisting of adding a disconnected subgraph to each input’s graph. The subgraph consists of nodes with larger nodes ids and whose connectivity is randomly generated. We contrast until the end of the input’s trajectory the following hints: pi ~~h~~ in Bellman-Ford, BFS, Dijkstra and MST-Prim; pi ~~h~~ , topo ~~h~~ , color in DAG shortest path; Pi ~~h~~ in Floyd-Warshall; pi in MST-Kruskal. 

**Sorting algorithms (Insertion sort, Bubble Sort, Quicksort, Heapsort).** We construct general augmented inputs obtained by simply adding items at the end of each input array. We consider as trajectories for those augmentations the ones of the corresponding inputs. We note that those do not correspond to exact augmentations for all sorting algorithms, but only for Insertion Sort. Indeed, running the executor of one of the other algorithms would yield potentially different trajectories than those we consider. However, since we use the inputs’ trajectories, our regularisation aims at learning to be invariant to 

14 

**Neural Algorithmic Reasoning with Causal Regularisation** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation_images/icml_2023_Neural_Algorithmic_Reasoning_with_Causal_Regularisation.pdf-0015-01.png)


**----- Start of picture text -----**<br>
Hint-ReLIC (ours)<br>Baseline + reversal<br>100<br>Baseline<br>80<br>60<br>40<br>20<br>0<br>Average score [%]<br>Heapsort Strongly Conn. Comps. Floyd-Warshall Quicksort Insertion Sort Binary Search Bubble Sort DAG Shortest Paths Articulation Points Topological Sort Bridges MST Kruskal Dijkstra Minimum MST Prim BFS Bellman-Ford Overall Average<br>**----- End of picture text -----**<br>


_Figure 7._ Per-algorithm comparison of the Triplet-GMPNN baseline (Ibarz et al., 2022), its augmented version which includes pointers’ reversal and our Hint-ReLIC. Error bars represent the standard error of the mean across three random seeds. The final column shows the average and standard error of the mean performances across the different algorithms. 

added nodes that do not contribute to each currently considered step. We contrast until the end of the input’s trajectories the hints pred ~~h~~ , and parent for Heapsort, and pred ~~h~~ for all the other algorithms. 

**Searching algorithms (Binary Search, Minimum).** We construct general augmented inputs obtained by simply adding random numbers (different than the searched one) at the end of the input array. For those augmentations, we consider as trajectories the ones of the corresponding input arrays, whose hints are contrasted until the end of the trajectories themselves. We remark that running the searching algorithms on such augmentations could potentially lead to ground-truth trajectories different than those of the inputs. However, since we consider as trajectories for the augmentations the inputs’ ones, the contrastive objective is still valid, and can be seen as pushing the hint representations to be invariant to messages coming from nodes that are not involved in the current computation. We run our model by allowing the network to predict the predecessor of every array’s item, namely pred h, at every time step and use its representation in our regularisation loss (in other words, we do not run with the static hint elimination of Ibarz et al. (2022)). 

## **F. Experiments** 

## **F.1. Additional experiments** 

Table 3 contains a comprehensive set of experiments, including the performances of the **No Hints** , **Baseline** , **Baseline + reversal** models, as discussed in the main text. The column **Baseline + reversal + contr. + KL** represents our Hint-ReLIC model, which is obtained with the additional inclusion of the contrastive and KL losses (see Equation (3)). Additionally, we report performances of our model when _removing_ the KL divergence loss (setting _α_ = 0 in Equation (3)), namely **Baseline + reversal + contr.** . By comparing **Baseline + reversal + contr.** to **Baseline + reversal + contr. + KL** , we can see that, even if the KL penalty produces some gain for certain algorithms, it does not represent the component leading to the most improvement. Finally, Table 3 also reports the scores obtained in the DFS algorithm, which appears to be solved by the inclusion of the pointers’ reversal. We do not run our contrastive objective on such algorithm as there is no additional improvement to be made. 

Finally, to further evaluate the impact of the pointers’ reversal, we report the performances of Hint-ReLIC without the inclusion of such additional hints. As can be seen in Table 4, the pointers’ reversal helps stabilise our model, especially in the sorting algorithms. We remark however how only including those pointers’ reversal into a baseline model does not produce the performances of our model (see column **Baseline + reversal** in Table 3 and Figure 7). 

15 

**Neural Algorithmic Reasoning with Causal Regularisation** 

_Table 3. Comparison of performances for different models, with last column representing our proposed method Hint-ReLIC._ Table shows mean and stderr of OOD micro-F1 score after 10 _,_ 000 training steps, across different seeds. 

|**Alg.**|**No Hints**|**Baseline**|**Baseline**|**Baseline**|**Baseline**|
|---|---|---|---|---|---|
||||**+ reversal**|**+ reversal + contr.**|**+ reversal + contr. + KL**|
|Articulation points|81_._97%_±_5_._08|88_._93%_±_1_._92|91_._04%_±_0_._92|**98**_._**91**%_±_0_._34|98_._45%_±_0_._60|
|Bridges|95_._62%_±_1_._03|93_._75%_±_2_._73|97_._70%_±_0_._34|98_._14%_±_2_._00|**99**_._**32**%_±_0_._09|
|DFS|33_._94%_±_2_._57|39_._71%_±_1_._34|**100**_._**00**%_±_0_._00|_−_|_−_|
|SCC|57_._63%_±_0_._68|38_._53%_±_0_._45|31_._40%_±_8_._80|75_._78%_±_1_._25|**76**_._**79**%_±_3_._04|
|Topological sort|84_._29%_±_1_._16|87_._27%_±_2_._67|88_._83%_±_7_._29|95_._44%_±_0_._52|**96**_._**59**%_±_0_._20|
|Bellman-Ford|93_._26%_±_0_._04|**96**_._**67**%_±_0_._81|95_._02%_±_0_._49|95_._26%_±_0_._92|95_._54%_±_1_._06|
|BFS|99_._89%_±_0_._03|99_._64%_±_0_._05|**99**_._**93**%_±_0_._03|98_._41%_±_0_._39|99_._00%_±_0_._21|
|DAG Shortest Paths|97_._62%_±_0_._62|88_._12%_±_5_._70|96_._61%_±_0_._61|97_._31%_±_0_._51|**98**_._**17**%_±_0_._26|
|Dijkstra|95_._01%_±_1_._14|93_._41%_±_1_._08|91_._50%_±_1_._85|97_._22%_±_0_._12|**97**_._**74**%_±_0_._50|
|Floyd-Warshall|40_._80%_±_2_._90|46_._51%_±_1_._30|46_._28%_±_0_._80|71_._43%_±_2_._64|**72**_._**23**%_±_4_._84|
|MST-Kruskal|92_._28%_±_0_._82|91_._18%_±_1_._05|89_._93%_±_0_._43|95_._18%_±_1_._29|**96**_._**01**%_±_0_._45|
|MST-Prim|85_._33%_±_1_._21|87_._64%_±_1_._79|86_._95%_±_2_._34|**89**_._**23**%_±_1_._23|87_._97%_±_2_._94|
|Insertion sort|77_._29%_±_7_._42|75_._28%_±_5_._62|87_._21%_±_2_._80|**95**_._**06**%_±_1_._33|92_._70%_±_1_._29|
|Bubble sort|81_._32%_±_6_._50|79_._87%_±_6_._85|80_._51%_±_9_._10|**94**_._**09**%_±_0_._80|92_._94%_±_1_._23|
|Quicksort|71_._60%_±_2_._22|70_._53%_±_11_._59|85_._69%_±_4_._53|90_._54%_±_2_._49|**93**_._**30**%_±_1_._96|
|Heapsort|68_._50%_±_2_._81|32_._12%_±_5_._20|49_._13%_±_10_._35|89_._41%_±_4_._79|**95**_._**16**%_±_1_._27|
|Binary Search|**93**_._**21**%_±_1_._10|74_._60%_±_3_._61|50_._42%_±_8_._45|87_._50%_±_3_._62|89_._68%_±_2_._13|
|Minimum|99_._24%_±_0_._21|97_._78%_±_0_._63|98_._43%_±_0_._01|99_._54%_±_0_._05|**99**_._**37**%_±_0_._20|



## **F.2. Implementation details** 

We use the best hyperparameters of the Triplet-GMPNN (Ibarz et al., 2022) base model, and we only reduce the batch size to 16. We set the temperature parameter _τ_ to 1 _e −_ 1 and the weight of the KL loss _α_ to 1. We implement the similarity function as _ϕ_ ( _f_ ( _x[a] t[l][, i][t]_[)] _[, f]_[(] _[x][a] t[k][, i][t]_[)) =] _[ ⟨][h]_[(] _[f]_[(] _[x][a] t[l]_[)] _[, i][t]_[)] _[, h]_[(] _[f]_[(] _[x][a] t[k][, i][t]_[))] _[ ⟩][/τ]_[with] _[ h]_[ a two-layers MLP with hidden and output] dimensions equal to the input one, and ReLU non-linearities. 

16 

**Neural Algorithmic Reasoning with Causal Regularisation** 

_Table 4. Importance of the inclusion of the pointers’ reversal in our Hint-ReLIC._ The table shows mean and stderr of the OOD micro-F1 score after 10 _,_ 000 training steps, across different seeds. 

|**Alg.**|**Hint-ReLIC**|**Hint-ReLIC**|
|---|---|---|
|||(no reversal)|
|Articulation points|**98**_._**45**%_±_0_._60|97_._33%_±_1_._32|
|Bridges|99_._32%_±_0_._09|**99**_._**42**%_±_0_._20|
|SCC|76_._79%_±_3_._04|**81**_._**42**%_±_2_._68|
|Topological sort|**96**_._**59**%_±_0_._20|80_._25%_±_3_._03|
|Bellman-Ford|**95**_._**54**%_±_1_._06|95_._27%_±_0_._97|
|BFS|**99**_._**00**%_±_0_._21|98_._23%_±_0_._17|
|DAG Shortest Paths|**98**_._**17**%_±_0_._26|89_._23%_±_7_._11|
|Dijkstra|**97**_._**74**%_±_0_._50|96_._70%_±_0_._92|
|Floyd-Warshall|**72**_._**23**%_±_4_._84|57_._38%_±_1_._75|
|MST-Kruskal|**96**_._**01**%_±_0_._45|94_._53%_±_0_._40|
|MST-Prim|**87**_._**97**%_±_2_._94|74_._24%_±_10_._85|
|Insertion sort|**92**_._**70**%_±_1_._29|67_._80%_±_10_._86|
|Bubble sort|**92**_._**94**%_±_1_._23|82_._36%_±_6_._88|
|Quicksort|**93**_._**30**%_±_1_._96|74_._32%_±_10_._12|
|Heapsort|**95**_._**16**%_±_1_._27|77_._15%_±_4_._73|
|Binary Search|**89**_._**68**%_±_2_._13|86_._65%_±_2_._38|
|Minimum|**99**_._**37**%_±_0_._20|98_._91%_±_0_._23|



17 

