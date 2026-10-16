# **DIFUSCO: Graph-based Diffusion Solvers for Combinatorial O timization p** 

**Zhiqing Sun** _[∗]_ Language Technologies Institute Carnegie Mellon University `zhiqings@cs.cmu.edu` 

**Yiming Yang** Language Technologies Institute Carnegie Mellon University `yiming@cs.cmu.edu` 

## **Abstract** 

Neural network-based Combinatorial Optimization (CO) methods have shown promising results in solving various NP-complete (NPC) problems without relying on hand-crafted domain knowledge. This paper broadens the current scope of neural solvers for NPC problems by introducing a new graph-based diffusion framework, namely DIFUSCO. Our framework casts NPC problems as discrete _{_ 0 _,_ 1 _}_ -vector optimization problems and leverages graph-based denoising diffusion models to generate high-quality solutions. We investigate two types of diffusion models with Gaussian and Bernoulli noise, respectively, and devise an effective inference schedule to enhance the solution quality. We evaluate our methods on two well-studied NPC combinatorial optimization problems: Traveling Salesman Problem (TSP) and Maximal Independent Set (MIS). Experimental results show that DIFUSCO strongly outperforms the previous state-of-the-art neural solvers, improving the performance gap between ground-truth and neural solvers from 1.76% to **0.46%** on TSP-500, from 2.46% to **1.17%** on TSP-1000, and from 3.19% to **2.58%** on TSP-10000. For the MIS problem, DIFUSCO outperforms the previous state-of-the-art neural solver on the challenging SATLIB benchmark. 

## **1 Introduction** 

Combinatorial Optimization (CO) problems are mathematical problems that involve finding the optimal solution in a discrete space. They are fundamental challenges in computer science, especially the NP-Complete (NPC) class of problems, which are believed to be intractable in polynomial time. Traditionally, NPC solvers rely on integer programming (IP) or hand-crafted heuristics, which demand significant expert efforts to approximate near-optimal solutions [4, 31]. 

Recent development in deep learning has shown new promise in solving NPC problems. Existing neural CO solvers for NPC problems can be roughly classified into three categories based on how the solutions are generated, i.e., the autoregressive constructive solvers, the non-autoregressive constructive solvers, and the improvement heuristics solvers. Methods in the first category use autoregressive factorization to sequentially grow a valid partial solution [6, 64]. Those methods typically suffer from the costly computation in their sequential decoding parts and hence are difficult to scale up to large problems [27]. Methods in the second category rely on non-autoregressive modeling for scaling up, with a conditional independence assumption among variables as typical [53, 55, 92]. Such an assumption, however, unavoidably limits the capability of those methods to capture the multimodal nature of the problems [57, 33], for example, when multiple optimal solutions exists for the same graph. Methods in the third category (improvement heuristics solvers) use a Markov decision process (MDP) to iteratively refines an existing feasible solution with neural network-guided local operations such as 2-opt [71, 2] and node swap [17, 113]. These methods 

> _∗_ Our code is available at `https://github.com/Edward-Sun/DIFUSCO` . 

37th Conference on Neural Information Processing Systems (NeurIPS 2023). 

have also suffered from the difficulty in scaling up and the latency in inference, partly due to the sparse rewards and sample efficiency issues when learning improvement heuristics in a reinforcement learning (RL) framework [113, 79]. 

Motivated by the recent remarkable success of diffusion models in probabilistic generation [102, 40, 94, 120, 96], we introduce a novel approach named DIFUSCO, which stands for the graph-based DIFfUsion Solvers for Combinatorial Optimization. To apply the iterative denoising process of diffusion models to graph-based settings, we formulate each NPC problem as to find a _{_ 0 _,_ 1 _}_ -valued vector that indicates the optimal selection of nodes or edges in a candidate solution for the task. Then we use a message passing-based graph neural network [61, 36, 29, 107] to encode each instance graph and to denoise the corrupted variables. Such a graph-based diffusion model overcomes the limitations of previous neural NPC solvers from a new perspective. Firstly, DIFUSCO can perform inference on all variables in parallel with a few ( _≪ N_ ) denoising steps (Sec. 3.3), avoiding the sequential generation problem of autoregressive constructive solvers. Secondly, DIFUSCO can model a multimodal distribution via iterative refinements, which alleviates the expressiveness limitation of previous non-autoregressive constructive models. Last but not least, DIFUSCO is trained in an efficient and stable manner with supervised denoising (Sec. 3.2), which solves the training scalability issue of RL-based improvement heuristics methods. 

We should point out that the idea of utilizing a diffusion-based generative model for NPC problems has been explored recently in the literature. In particular, Graikos et al. [32] proposed an image-based diffusion model to solve Euclidean Traveling Salesman problems by projecting each TSP instance onto a 64 _×_ 64 greyscale image space and then using a Convolutional Neural Network (CNN) to generate the predicted solution image. The main difference between such _image-based_ diffusion solver and our _graph-based_ diffusion solver is that the latter can explicitly model the node/edge selection process via the corresponding random variables, which is a natural design choice for formulating NPC problems (since most of them are defined over a graph), while the former does not support such a desirable formalism. Although graph-based modeling has been employed with both constructive [64] and improvement heuristics [20] solvers, how to use graph-based diffusion models for solving NPC problems has not been studied before, to the best of our knowledge. 

We investigate two types of probabilistic diffusion modeling within the DIFUSCO framework: continuous diffusion with Gaussian noise [16] and discrete diffusion with Bernoulli noise [5, 44]. These two types of diffusion models have been applied to image processing but not to NPC problems so far. We systematically compare the two types of modeling and find that discrete diffusion performs better than continuous diffusion by a significant margin (Section 4). We also design an effective inference strategy to enhance the generation quality of the discrete diffusion solvers. 

Finally, we demonstrate that a single graph neural network architecture, namely the Anisotropic Graph Neural Network [9, 54], can be used as the backbone network for two different NP-complete combinatorial optimization problems: Traveling Salesman Problem (TSP) and Maximal Independent Set (MIS). Our experimental results show that DIFUSCO outperforms previous probabilistic NPC solvers on benchmark datasets of TSP and MIS problems with various sizes. 

## **2 Related Work** 

## **2.1 Autoregressive Construction Heuristics Solvers** 

Autoregressive models have achieved state-of-the-art results as constructive heuristic solvers for combinatorial optimization (CO) problems, following the recent success of language modeling in the text generation domain [106, 11]. The first approach proposed by Bello et al. [6] uses a neural network with reinforcement learning to append one new variable to the partial solution at each decoding step until a complete solution is generated. However, autoregressive models [64] face high time and space complexity challenges for large-scale NPC problems due to their sequential generation scheme and quadratic complexity in the self-attention mechanism [106]. 

## **2.2 Non-autoregressive Construction Heuristics Solvers** 

Non-autoregressive (or heatmap) constructive heuristics solvers [53, 27, 28, 92] are recently proposed to address this scalability issue by assuming conditional independence among variables in NPC problems, but this assumption limits the ability to capture the multimodal nature [57, 33] of high- 

2 

quality solution distributions. Therefore, additional active search [6, 92] or Monte-Carlo Tree Search (MCTS) [27, 98] are needed to further improve the expressive power of the non-autoregressive scheme. 

DIFUSCO can be regarded as a member in the non-autoregressive constructive heuristics category and thus can be benefited from heatmap search techniques such as MCTS. But DIFUSCO uses an iterative denoising scheme to generate the final heatmap, which significantly enhances its expressive power compared to previous non-autoregressive methods. 

## **2.3 Diffusion Models for Discrete Data** 

Typical diffusion models [100, 102, 40, 103, 85, 56] operate in the continuous domain, progressively adding Gaussian noise to the clean data in the forward process, and learning to remove noises in the reverse process in a discrete-time framework. 

Discrete diffusion models have been proposed for the generation of discrete image bits or texts using binomial noises [100] and multinomial/categorical noises [5, 44]. Recent research has also shown the potential of discrete diffusion models in sound generation [118], protein structure generation [77], molecule generation [108], and better text generation [52, 38]. 

Another line of work studies diffusion models for discrete data by applying continuous diffusion models with Gaussian noise on the embedding space of discrete data [30, 69, 24], the _{−_ 1 _._ 0 _,_ 1 _._ 0 _}_ real-number vector space [16], and the simplex space [37]. The most relevant work might be Niu et al. [86], which proposed a continuous score-based generative framework for graphs, but they only evaluated simple non-NP-hard CO tasks such as Shortest Path and Maximum Spanning Tree. 

## **3 DIFUSCO: Proposed Approach** 

## **3.1 Problem Definition** 

Following a conventional notation [88], we define _Xs_ = _{_ 0 _,_ 1 _}[N]_ as the space of candidate solutions _{_ **x** _}_ for a CO problem instance _s_ , and _cs_ : _Xs →_ R as the objective function for solution **x** _∈Xs_ : _cs_ ( **x** ) = cost( **x** _, s_ ) + valid( **x** _, s_ ) _._ (1) Here cost( _·_ ) is the task-specific cost of a candidate solution (e.g., the tour length in TSP), which is often a simple linear function of **x** in most NP-complete problems, and valid( _·_ ) is the validation term that returns 0 for a feasible solution and + _∞_ for an invalid one. The optimization objective is to find the optimal solution **x** _[s][∗]_ for a given instance _s_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0003-09.png)


This framework is generically applicable to different NPC problems. For example, for the Traveling Salesman Problem (TSP), **x** _∈{_ 0 _,_ 1 _}[N]_ is the indicator vector for selecting a subset from _N_ edges; the cost of this subset is calculated as: costTSP( **x** _, s_ ) =[�] _i[x][i][ ·][ d]_[(] _i[s]_[)][, where] _[ d]_[(] _i[s]_[)] denotes the weight of the _i_ -th edge in problem instance _s_ , and the valid( _·_ ) part of Formula (1) ensures that **x** is a tour that visits each node exactly once and returns to the starting node at the end. For the Maximal Independent Set (MIS) problem, **x** _∈{_ 0 _,_ 1 _}[N]_ is the indicator vector for selecting a subset from _N_ nodes; the cost of the subset is calculated as: costMIS( **x** _, s_ ) =[�] _i_[(1] _[ −][x][i]_[)] _[,]_[, and the corresponding][ valid(] _[·]_[)][ validates] **x** is an independent set where each node in the set has no connection to any other node in the set. 

Probabilistic neural NPC solvers [6] tackle instance problem _s_ by defining a parameterized conditional distributionprobabilistic generative models are usually optimized by reinforcement learning algorithms [ _p_ _**θ**_ ( **x** _|s_ ), such that the expected cost[�] **x** _∈Xs[c][s]_[(] **[x]**[)] _[ ·][ p]_[(] **[x]** _[|][s]_[)][ is minimized.] 111[Such] , 63]. In this paper, assuming the optimal (or high-quality) solution **x** _[∗] s_[is available for each training in-] stance _s_ , we optimize the model through supervised learning. Let _S_ = _{si}_ 1 _[N]_[be independently and] identically distributed (IID) training samples for a type of NPC problem, we aim to maximize the likelihood of optimal (or high-quality) solutions, where the loss function _L_ is defined as: 

_L_ ( _**θ**_ ) = E _s∈S_ [ _−_ log _p_ _**θ**_ ( **x** _[s][∗] |s_ )] (3) 

Next, we describe how to use diffusion models to parameterize the generative distribution _p_ _**θ**_ . For brevity, we omit the conditional notations of _s_ and denote **x** _[s][∗]_ as **x** 0 as a convention in diffusion models for all formulas in the the rest of the paper. 

3 

## **3.2 Diffusion Models in DIFUSCO** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0004-01.png)


where _C_ is a constant. 

**Discrete Diffusion** In discrete diffusion models with multinomial noises [5, 44], the forward process is defined as: _q_ ( **x** _t|_ **x** _t−_ 1) = Cat ( **x** _t_ ; **p** = **x** ˜ _t−_ 1 **Q** _t_ ) _,_ where **Q** _t_ = �(1 _−βtβt_ ) (1 _−βtβt_ )� is with a one-hot vector per row; andthe transition probability matrix; **x** ˜ **xQ** _∈{_ ˜ computes a row-wise vector-matrix product.0 _,_ 1 _}[N][×]_[2] is converted from the original vector **x** _∈{_ 0 _,_ 1 _}[N]_ Here, _βt_ denotes the corruption ratio. Also, we want[�] _[T] t_ =1[(1] _[ −][β][t]_[)] _[ ≈]_[0][ such that] **[ x]** _[T][∼]_[Uniform(] _[·]_[)][.] ˜ The _t_ -step marginal can thus be written as: _q_ ( **x** _t|_ **x** 0) = Cat � **x** _t_ ; **p** = **x** 0 **Q** _t_ � _,_ where **Q** _t_ = **Q** 1 **Q** 2 _. . ._ **Q** _t_ . And the posterior at time _t −_ 1 can be obtained by Bayes’ theorem: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0004-04.png)


where _⊙_ denotes the element-wise multiplication. 

According _p_ _**θ**_ ( **x** �0 _|_ **x** _t_ ), and the reverse process is obtained by substituting the predictedto Austin et al. [5], the denoising neural network is trained to � **x** predict0 as **x** 0 in Eq. 5:the clean data 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0004-07.png)


**Continuous Diffusion for Discrete Data** The continuous diffusion models [102, 40] can also be directly applied to discrete data by lifting the discrete input into a continuous space [16]. Since the continuous diffusion models usually start from a standard Gaussian distribution _**ϵ** ∼N_ ( **0** _,_ **I** ), Chen et al. **x** ˆ0, and then treat them as real values. [16] proposed to first rescale theThe forward process in continuous diffusion is defined as: _{_ 0 _,_ 1 _}_ -valued variables **x** 0 to the _{−_ 1 _,_ 1 _}_ domain as ˆ ˆ _q_ (ˆ **x** _t|_ **x** _t−_ 1) := _N_ (ˆ **x** _t_ ; _[√]_ 1 _− βt_ **x** _t−_ 1 _, βt_ **I** ) _._ 

Again, _βt_ denotes the corruption ratio, and we wantˆ[�] _[T] t_ =1[(1] _[ −]_ ¯ _[β]_ ˆ _[t]_[)] _[ ≈]_[0][ such that] ¯ **[ x]** _[T][∼N]_[(] _[·]_[)][.][The] _t_ -step marginal can thus be written as: _q_ (ˆ **x** _t|_ **x** 0) := _N_ (ˆ **x** _t_ ; _[√] αt_ **x** 0 _,_ (1 _− αt_ ) **I** ) where _αt_ = 1 _− βt_ and _α_ ¯ _t_ =[�] _[t] τ_ =1 _[α][τ]_[.][Similar to Eq. 5, the posterior at time] _[ t][ −]_[1][ can be obtained by Bayes’ theorem:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0004-10.png)


which is a closed-form Gaussian distribution [40]. In continuous diffusion, the denoising neural ¯ ˆ ¯ The reverse process [network is trained to predict the unscaled Gaussian noise40] can use a point estimation of **x** ˆ0 � _**ϵ**_ in the posterior: _t_ = (ˆ **x** _t −[√] αt_ **x** 0) _/[√]_ 1 _− αt_ = _f_ _**θ**_ (ˆ **x** _t, t_ ). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0004-12.png)


For generating discrete data, after the continuous data **x** ˆ0 is generated, a thresholding/quantization operation is applied to convert them back to _{_ 0 _,_ 1 _}_ -valued variables **x** 0 as the model’s prediction. 

4 

## **3.3 Denoising Schedule for Fast Inference** 

One way to speed up the inference of denoising diffusion models is to reduce the number of steps in the reverse diffusion process, which also reduces the number of neural network evaluations. The denoising diffusion implicit models (DDIMs) [101] are a class of models that apply this strategy in the continuous domain, and a similar approach can be used for discrete diffusion models [5]. 

Formally, when the forward process is defined not on all the latent variables **x** 1: _T_ , but on a subset _{_ **x** _τ_ 1 _, . . . ,_ **x** _τM }_ , where _τ_ is an increasing sub-sequence of [1 _, . . . , T_ ] with length _M_ , **x** _τ_ 1 = 1 and **x** _τM_ = _T_ , the fast sampling algorithms directly models _q_ ( **x** _τi−_ 1 _|_ **x** _τi,_ **x** 0). Due to the space limit, the detailed algorithms are described in the appendix. 

We consider two types of denoising scheduled for _τ_ given the desired card( _τ_ ) _< T_ : `linear` and `cosine` . The former uses timesteps such that _τi_ = _⌊ci⌋_ for some _c_ , and the latter uses timesteps such that _τi_ = _⌊_ cos([(][1] _[−]_ 2 _[ci]_[)] _[π]_ ) _· T ⌋_ for some _c_ . The intuition for the cosine schedule is that diffusion models can achieve better generation quality when iterating more steps in the low-noise regime [85, 121, 13]. 

## **3.4 Graph-based Denoising Network** 

Thepredicts the clean datadenoising network **x** �takes0. To balance both scalability and performance considerations, we adoptas input a set of noisy variables **x** _t_ and the problem instance _s_ and an anisotropic graph neural network with edge gating mechanisms [9, 54] as the backbone network for both discrete and continuous diffusion models, and the variables in the network output can be the states of either nodes, as in the case of Maximum Independent Set (MIS) problems, or edges, as in the case of Traveling Salesman Problems (TSP). Our choice of network mainly follows previous work [54, 92], as AGNN can produce the embeddings for both nodes and edges, unlike typical GNNs such as GCN [62] or GAT [107], which are designed for node embedding only. This design choice is particularly beneficial for tasks that require the prediction of edge variables. 

**Anisotropic Graph Neural Networks** Let _**h**[ℓ] i_[and] _**[ e]**[ℓ] ij_[denote the node and edge features at layer] _ℓ_ associated with node _i_ and edge _ij_ , respectively. **t** is the sinusoidal features [106] of denoising timestep _t_ . The features at the next layer is propagated with an anisotropic message passing scheme: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0005-07.png)


where _**U**[ℓ] ,_ _**V**[ℓ] ,_ _**P**[ℓ] ,_ _**Q**[ℓ] ,_ _**R**[ℓ] ∈_ R _[d][×][d]_ are the learnable parameters of layer _ℓ_ , _α_ denotes the ReLU [66] activation, BN denotes the Batch Normalization operator [51], _A_ denotes the aggregation function SUM pooling [116], _σ_ is the sigmoid function, _⊙_ is the Hadamard product, _Ni_ denotes the neighborhoods of node _i_ , and MLP( _·_ ) denotes a 2-layer multi-layer perceptron. 

For TSP, _**e**_[0] _ij_[are initialized as the corresponding values in] **[ x]** _[t]_[, and] _**[ h]** i_[0][are initialized as sinusoidal] features of the nodes. For MIS, _**e**_[0] _ij_[are initialized as zeros, and] _**[ h]** i_[0][are initialized as the corresponding] values in **x** _t_ . A 2-neuron and 1-neuron classification/regression head is applied to the final embeddings of **x** _t_ ( _{_ **e** _ij}_ for edges and _{_ **h** _i}_ for nodes) for discrete and continuous diffusion models, respectively. 

**Hyper-parameters** For all TSP and MIS benchmarks, we use a 12-layer Anisotropic GNN with a width of 256 as described above. 

## **3.5 Decoding Strategies for Diffusion-based Solvers** 

After the training of the parameterized denoising network according to Eq. 4, the solutions are sampled from the diffusion models _p_ _**θ**_ ( **x** 0 _|s_ ) for final evaluation. However, probabilistic generative models such as DIFUSCO cannot guarantee that the sampled solutions are feasible according to the definition of CO problems. Therefore, specialized decoding strategies are designed for the two CO problems studied in this paper. 

5 

**Heatmap Generation** The diffusion models _p_ _**θ**_ ( _·|s_ ) produce discrete variables **x** as the final predictions by applying Bernoulli sampling (Eq. 6) for discrete diffusion or quantization for continuous diffusion. However, this process discards the comparative information that reflects the confidence of the predicted variables, which is crucial for resolving conflicts in the decoding process. To preserve this information, we adapt the diffusion models to generate heatmaps [53, 92] by making the following appropriate modifications: 1) For discrete diffusion, the final score of _p_ _**θ**_ ( **x0** = 1 _|s_ ) is preserved as the heatmap scores; 2) For continuous diffusion, we remove the final quantization and use 0 _._ 5(ˆ **x** 0 + 1) as the heatmap scores. Note that different from previous heatmap approaches [53, 92] that produce a single conditionally independent distribution for all variables, DIFUSCO can produce diverse multimodal output distribution by using different random seeds. 

**TSP Decoding** Let _{Aij}_ be the heatmap scores generated by DIFUSCO denoting the confidence of each edge. We evaluate two approaches as the decoding method following previous work [32, 92]: 1) Greedy decoding [32], where all the edges are ranked by ( _Aij_ + _Aji_ ) _/∥_ **c** _i −_ **c** _j∥_ , and are inserted into the partial solution if there are no conflicts. 2-opt heuristics [71] are optionally applied. 2) Monte Carlo Tree Search (MCTS) [27], where _k_ -opt transformation actions are sampled guided by the heatmap scores to improve the current solutions. Due to the space limit, a detailed description of two decoding strategies can be found in the appendix. 

**MIS Decoding** Let _{ai}_ be the heatmap scores generated by DIFUSCO denoting the confidence of each node. A greedy decoding strategy is used for the MIS problem, where the nodes are ranked by _ai_ and inserted into the partial solution if there are no conflicts. Recent research [8] pointed out that the graph reduction and 2-opt search [2] can find near-optimal solutions even starting from a randomly generated solution, so we do not use any post-processing for the greedy-decoded solutions. **Solution Sampling** A common practice for probabilistic CO solvers [64] is to sample multiple solutions and report the best one. For DIFUSCO, we follow this practice by sampling multiple heatmaps from _p_ _**θ**_ ( **x** 0 _|s_ ) with different random seeds and then applying the greedy decoding algorithm described above to each heatmap. 

## **4 Experiments with TSP** 

We use 2-D Euclidean TSP instances to test our models. We generate these instances by randomly sampling nodes from a uniform distribution over the unit square. We use TSP-50 (with 50 nodes) as the main benchmark to compare different model configurations. We also evaluate our method on larger TSP instances with 100, 500, 1000, and 10000 nodes to demonstrate its scalability and performance against other state-of-the-art methods. 

## **4.1 Experimental Settings** 

**Datasets** We generate and label the training instances using the Concorde exact solver [3] for TSP-50/100 and the LKH-3 heuristic solver [39] for TSP-500/1000/10000. We use the same test instances as [54, 64] for TSP-50/100 and [27] for TSP-500/1000/10000. 

**Graph Sparsification** We use sparse graphs for large-scale TSP problems to reduce the computational complexity. We sparsify the graphs by limiting each node to have only _k_ edges to its nearest neighbors based on the Euclidean distances. We set _k_ to 50 for TSP-500 and 100 for TSP-1000/10000. This way, we avoid the quadratic growth of edges in dense graphs as the number of nodes increases. 

**Model Settings** _T_ = 1000 denoising steps are used for the training of DIFUSCO on all datasets. Following Ho et al. [40], Graikos et al. [32], we use a simple linear noise schedule for _{βt}[T] t_ =1[, where] _β_ 1 = 10 _[−]_[4] and _βT_ = 0 _._ 02. We follow Graikos et al. [32] and use the Greedy decoding + 2-opt scheme (Sec. 3.5) as the default decoding scheme for experiments. **Evaluation Metrics** In order to compare the performance of different models, we present three metrics: average tour length (Length), average relative performance gap (Gap), and total run time (Time). The detailed description can be found in the appendix. 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0007-00.png)


Table 1: Comparing results on TSP-50 and TSP-100. _∗_ denotes the baseline for computing the performance gap. _†_ indicates that the diffusion model samples a single solution as its greedy decoding scheme. Please refer to Sec. 4 for details. 

||ALGORITHM|TYPE|TSP-50<br>TSP-100<br>LENGTH_↓_GAP(%)_↓_LENGTH_↓_GAP(%)_↓_|TSP-50<br>TSP-100<br>LENGTH_↓_GAP(%)_↓_LENGTH_↓_GAP(%)_↓_|TSP-50<br>TSP-100<br>LENGTH_↓_GAP(%)_↓_LENGTH_↓_GAP(%)_↓_|TSP-50<br>TSP-100<br>LENGTH_↓_GAP(%)_↓_LENGTH_↓_GAP(%)_↓_|
|---|---|---|---|---|---|---|
||CONCORDE_∗_|EXACT|5.69|0.00|7.76|0.00|
||2-OPT|HEURISTICS|5.86|2.95|8.03|3.54|
||AM|GREEDY|5.80|1.76|8.12|4.53|
||GCN|GREEDY|5.87|3.10|8.41|8.38|
||TRANSFORMER|GREEDY|5.71|0.31|7.88|1.42|
||POMO|GREEDY|5.73|0.64|7.84|1.07|
||SYM-NCO|GREEDY|-|-|7.84|0.94|
||DPDP|1_k_-IMPROVEMENTS|5.70|0.14|7.89|1.62|
||IMAGEDIFFUSION|GREEDY_†_|5.76|1.23|7.92|2.11|
||**OURS**|GREEDY_†_|**5.70**|**0.10**|**7.78**|**0.24**|
||AM|1_k×_SAMPLING|5.73|0.52|7.94|2.26|
||GCN|2_k×_SAMPLING|5.70|0.01|7.87|1.39|
||TRANSFORMER|2_k×_SAMPLING|5.69|0.00|7.76|0.39|
||POMO<br>SYM-NCO<br>MDAM|8_×_AUGMENT<br>100_×_SAMPLING<br>50_×_SAMPLING|5.69<br>-<br>5.70|0.03<br>-<br>0.03|7.77<br>7.79<br>7.79|0.14<br>0.39<br>0.38|
||DPDP<br>**OURS**|100_k_-IMPROVEMENTS<br>16_×_SAMPLING|5.70<br>**5.69**|0.00<br>**-0.01**|7.77<br>**7.76**|0.00<br>**-0.01**|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0007-03.png)


Figure 1: Comparison of continuous (Gaussian noise) and discrete (Bernoulli noise) diffusion models with different inference diffusion steps and inference schedule ( `linear` v.s. `cosine` ). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0007-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0007-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0007-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0007-08.png)


**----- Start of picture text -----**<br>
(a) Continuous diffusion (b) Discrete diffusion (c) Runtime<br>**----- End of picture text -----**<br>


Figure 2: The performance Gap (%) are shown for continuous diffusion **(a)** and discrete diffusion **(b)** models on TSP-50 with different diffusion steps and number of samples. Their corresponding per-instance run-time (sec) are shown in **(c)** , where the decomposed analysis (neural network + greedy decoding + 2-opt) can be found in the appendix. 

## **4.2 Design Analysis** 

**Discrete Diffusion v.s. Continuous Diffusion** We first investigate the suitability of two diffusion approaches for combinatorial optimization, namely continuous diffusion with Gaussian noise and discrete diffusion with Bernoulli noise (Sec. 3.2). Additionally, we explore the effectiveness of different denoising schedules, such as `linear` and `cosine` schedules (Sec. 3.3), on CO problems. To efficiently evaluate these model choices, we utilize the TSP-50 benchmark. 

Note that although all the diffusion models are trained with a _T_ = 1000 noise schedule, the inference schedule can be shorter than _T_ , as described in Sec. 3.3. Specifically, we are interested in diffusion models with 1, 2, 5, 10, 20, 50, 100, and 200 diffusion steps. 

Fig. 1 demonstrates the performance of two types of diffusion models with two types of inference schedules and various diffusion steps. We can see that discrete diffusion consistently outperforms the continuous diffusion models by a large margin when there are more than 5 diffusion steps[2] . Besides, the `cosine` schedule is superior to `linear` on discrete diffusion and performs similarly on continuous diffusion. Therefore, we use `cosine` for the rest of the paper. 

**More Diffusion Iterations v.s. More Sampling** By utilizing effective denoising schedules, diffusion models are able to adaptively infer based on the available computation budget by predetermining the total number of diffusion steps. This is similar to changing the number of samples in previous probabilistic neural NPC solvers [64]. Therefore, we investigate the trade-off between the number of diffusion iterations and the number of samples for diffusion-based NPC solvers. 

> 2We also observe similar patterns on TSP-100, where the results are reported in the appendix. 

7 

Table 2: Results on large-scale TSP problems. RL, SL, AS, G, S, BS, and MCTS denotes Reinforcement Learning, Supervised Learning, Active Search, Greedy decoding, Sampling decoding, Beam-search, and Monte Carlo Tree Search, respectively. * indicates the baseline for computing the performance gap. Results of baselines are taken from Fu et al. [27] and Qiu et al. [92], so the runtime may not be directly comparable. See Section 4 and appendix for detailed descriptions. 

|ALGORITHM<br>TYPE|TSP-500<br>LENGTH_↓_<br>GAP_↓_<br>TIME_↓_|TSP-1000<br>LENGTH_↓_<br>GAP_↓_<br>TIME_↓_|TSP-10000<br>LENGTH_↓_<br>GAP_↓_<br>TIME_↓_|
|---|---|---|---|
|CONCORDE<br>EXACT<br>GUROBI<br>EXACT<br>LKH-3 (DEFAULT)<br>HEURISTICS<br>LKH-3 (LESS TRAILS) HEURISTICS<br>FARTHESTINSERTION HEURISTICS|16.55_∗_<br>—<br>37.66m<br>16.55<br>0.00%<br>45.63h<br>16.55<br>0.00% 46.28m<br>16.55<br>0.00%<br>3.03m<br>18.30<br>10.57%<br>0s|23.12_∗_<br>—<br>6.65h<br>N/A<br>N/A<br>N/A<br>23.12<br>0.00%<br>2.57h<br>23.12<br>0.00%<br>7.73m<br>25.72<br>11.25%<br>0s|N/A<br>N/A<br>N/A<br>N/A<br>N/A<br>N/A<br>71.77_∗_<br>—<br>8.8h<br>71.79<br>—<br>51.27m<br>80.59<br>12.29%<br>6s|
|AM<br>RL+G<br>GCN<br>SL+G<br>POMO+EAS-EMB<br>RL+AS+G<br>POMO+EAS-TAB<br>RL+AS+G<br>DIMES<br>RL+G<br>DIMES<br>RL+AS+G<br>OURS(DIFUSCO)<br>SL+G_†_<br>OURS(DIFUSCO)<br>SL+G_†_+2-OPT|20.02<br>20.99% 1.51m<br>29.72<br>79.61% 6.67m<br>19.24<br>16.25% 12.80h<br>24.54<br>48.22% 11.61h<br>18.93<br>14.38% 0.97m<br>17.81<br>7.61%<br>2.10h<br>18.35<br>10.85% 3.61m<br>**16.80**<br>**1.49%**<br>**3.65**m|31.15<br>34.75%<br>3.18m<br>48.62<br>110.29% 28.52m<br>N/A<br>N/A<br>N/A<br>49.56<br>114.36% 63.45h<br>26.58<br>14.97%<br>2.08m<br>24.91<br>7.74%<br>4.49h<br>26.14<br>13.06% 11.86m<br>**23.56**<br>**1.90%**<br>**12.06**m|141.68<br>97.39% 5.99m<br>N/A<br>N/A<br>N/A<br>N/A<br>N/A<br>N/A<br>N/A<br>N/A<br>N/A<br>86.44<br>20.44% 4.65m<br>80.45<br>12.09%<br>3.07h<br>98.15<br>36.75% 28.51m<br>**73.99**<br>**3.10% 35.38**m|
|EAN<br>RL+S+2-OPT<br>AM<br>RL+BS<br>GCN<br>SL+BS<br>DIMES<br>RL+S<br>DIMES<br>RL+AS+S<br>OURS(DIFUSCO)<br>SL+S<br>OURS(DIFUSCO)<br>SL+S+2-OPT|23.75<br>43.57% 57.76m<br>19.53<br>18.03% 21.99m<br>30.37<br>83.55% 38.02m<br>18.84<br>13.84% 1.06m<br>17.80<br>7.55%<br>2.11h<br>17.23<br>4.08% 11.02m<br>**16.65**<br>**0.57% 11.46**m|47.73<br>106.46%<br>5.39h<br>29.90<br>29.23%<br>1.64h<br>51.26<br>121.73% 51.67m<br>26.36<br>14.01%<br>2.38m<br>24.89<br>7.70%<br>4.53h<br>25.19<br>8.95%<br>46.08m<br>**23.45**<br>**1.43%**<br>**48.09**m|N/A<br>N/A<br>N/A<br>129.40<br>80.28%<br>1.81h<br>N/A<br>N/A<br>N/A<br>85.75<br>19.48% 4.80m<br>80.42<br>12.05%<br>3.12h<br>95.52<br>33.09%<br>6.59h<br>**73.89**<br>**2.95%**<br>**6.72**h|
|ATT-GCN<br>SL+MCTS<br>DIMES<br>RL+MCTS<br>DIMES<br>RL+AS+MCTS<br>OURS(DIFUSCO)<br>SL+MCTS|16.97<br>2.54%<br>2.20m<br>16.87<br>1.93%<br>2.92m<br>16.84<br>1.76%<br>2.15h<br>**16.63**<br>**0.46% 10.13**m|23.86<br>3.22%<br>4.10m<br>23.73<br>2.64%<br>6.87m<br>23.69<br>2.46%<br>4.62h<br>**23.39**<br>**1.17%**<br>**24.47**m|74.93<br>4.39% 21.49m<br>74.63<br>3.98% 29.83m<br>74.06<br>3.19%<br>3.57h<br>**73.62**<br>**2.58% 47.36**m|



Fig. 2 shows the results of continuous diffusion and discrete diffusion with various diffusion steps and number of parallel sampling, as well as their corresponding total run-time. The `cosine` denoising schedule is used for fast inference. Again, we find that discrete diffusion outperforms continuous diffusion across various settings. Besides, we find performing more diffusion iterations is generally more effective than sampling more solutions, even when the former uses less computation. For example, 20 (diffusion steps) _×_ 4 (samples) performs competitive to 1 (diffusion steps) _×_ 1024 (samples), while the runtime of the former is 18 _._ 5 _×_ less than the latter. 

In general, we find that 50 (diffusion steps) _×_ 1 (samples) policy and 10 (diffusion steps) _×_ 16 (samples) policy make a good balance between exploration and exploitation for discrete DIFUSCO models and use them as the **Greedy** and **Sampling** strategies for the rest of the experiments. 

## **4.3 Main Results** 

**Comparison to SOTA Methods** We compare discrete DIFUSCO to other state-of-the-art neural NPC solvers on TSP problems across various scales. Due to the space limit, the description of other baseline models can be found in the appendix. 

Tab. 1 compare discrete DIFUSCO with other models on TSP-50 and TSP-100, where DIFUSCO achieves the state-of-the-art performance in both greedy and sampling settings for probabilistic solvers. 

Tab. 2 compare discrete DIFUSCO with other models on the larger-scale TSP-500, TSP-1000, and TSP-10000 problems. Most previous probabilistic solvers (except DIMES [92]) becomes untrainable on TSP problems of these scales, so the results of these models are reported with TSP-100 trained models. The results are reported with greedy, sampling, and MCTS decoding strategies, respectively. For fair comparisons [27, 92], MCTS decoding for TSP is always evaluated with only one sampled heatmap. From the table, we can see that DIFUSCO strongly outperforms the previous neural solvers on all three settings. In particular, with MCTS-based decoding, DIFUSCO significantly improving the performance gap between ground-truth and neural solvers from 1.76% to **0.46%** on TSP-500, from 2.46% to **1.17%** on TSP-1000, and from 3.19% to **2.58%** on TSP-10000. 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0009-00.png)


Figure 3: Generalization tests of DIFUSCO trained and evaluated on TSP problems across various scales. The performance Gap (%) with greedy decoding and 2-opt is reported. 

Table 3: Results on MIS problems. _[∗]_ indicates the baseline for computing the optimality gap. RL, SL, G, S, and TS denote Reinforcement Learning, Supervised Learning, Greedy decoding, Sampling decoding, and Tree Search, respectively. Please refer to Sec. 5 and appendix for details. 

||METHOD|TYPE|SIZE_↑_|SATLIB<br>GAP_↓_|TIME_↓_|ER-[700-800]<br>SIZE_↑_<br>GAP_↓_<br>TIME_↓_|ER-[700-800]<br>SIZE_↑_<br>GAP_↓_<br>TIME_↓_|ER-[700-800]<br>SIZE_↑_<br>GAP_↓_<br>TIME_↓_|
|---|---|---|---|---|---|---|---|---|
||KAMIS|HEURISTICS|425.96_∗_|—|37.58m|44.87_∗_|—|52.13m|
||GUROBI|EXACT|425.95|0.00%|26.00m|41.38|7.78%|50.00m|
||INTEL|SL+G|420.66|1.48%|23.05m|34.86|22.31%|6.06m|
||INTEL|SL+TS|N/A|N/A|N/A|38.80|13.43%|20.00M|
||DGL<br>LWD|SL+TS<br>RL+S|N/A<br>422.22|N/A<br>0.88%|N/A<br> 18.83m|37.26<br>41.17|16.96% <br>8.25%|22.71m<br>6.33m|
||DIMES|RL+G|421.24|1.11%|24.17m|38.24|14.78%|6.12m|
||DIMES|RL+S|423.28|0.63%|20.26m|**42.06**|**6.26%**|12.01m|
||OURS<br>OURS|SL+G<br>SL+S|**424.50**<br>**425.13**|**0.34%**<br>**0.21%**|8.76m<br> 23.74m|38.83<br>41.12|12.40%<br>8.36%|8.80m<br>26.67m|



**Generalization Tests** Finally, we study the generalization ability of discrete DIFUSCO trained on a set of TSP problems of a specific problem scale and evaluated on other problem scales. From Fig. 3, we can see that DIFUSCO has a strong generalization ability. In particular, the model trained with TSP-50 perform well on even TSP-1000 and TSP0-10000. This pattern is different from the bad generalization ability of RL-trained or SL-trained non-autoregressive methods as reported in previous work [54]. 

## **5 Experiments with MIS** 

For Maximal Independent Set (MIS), we experiment on two types of graphs that recent work [70, 1, 8, 92] shows struggles against, i.e., SATLIB [46] and Erd˝os-Rényi (ER) graphs [26]. The former is a set of graphs reduced from SAT instances in CNF, while the latter are random graphs. We use ER-[700-800] for evaluation, where ER-[ _n_ - _N_ ] indicates the graph contains _n_ to _N_ nodes. Following Qiu et al. [92], the pairwise connection probability _p_ is set to 0 _._ 15. 

**Datasets** The training instances of labeled by the KaMIS[3] heuristic solver. The split of test instances on SAT datasets and the random-generated ER test graphs are taken from Qiu et al. [92]. 

**Model Settings** The training schedule is the same as the TSP solver (Sec. 4.1). For SATLIB, we use discrete diffusion with 50 (diffusion steps) _×_ 1 (samples) policy and 50 (diffusion steps) _×_ 4 (samples) policy as the **Greedy** and **Sampling** strategies, respectively. For ER graphs, we use continuous diffusion with 50 (diffusion steps) _×_ 1 (samples) policy and 20 (diffusion steps) _×_ 8 (samples) policy as the **Greedy** and **Sampling** strategies, respectively. 

**Evaluation Metrics** We report the average size of the independent set (Size), average optimality gap (Gap), and latency time (Time). The detailed description can be found in the appendix. Notice that we disable graph reduction and 2-opt local search in all models for a fair comparison since it is pointed out by [8] that all models would perform similarly with local search post-processing. 

**Results and Analysis** Tab. 3 compare discrete DIFUSCO with other baselines on SATLIB and ER-[700-800] benchmarks. We can see that DIFUSCO strongly outperforms previous state-of-the-art methods on SATLIB benchmark, reducing the gap between ground-truth and neural solvers from 0.63% to **0.21%** . However, we also found that DIFUSCO (especially with discrete diffusion in our preliminary experiments) does not perform well on the ER-[700-800] data. We hypothesize that this is because the previous methods usually use the node-based graph neural networks such as GCN [62] or GraphSage [36] as the backbone network, while we use an edge-based Anisotropic GNN (Sec. 3.4), whose inductive bias may be not suitable for ER graphs. 

> 3 `https://github.com/KarlsruheMIS/KaMIS` (MIT License) 

9 

## **6 Concluding Remarks** 

We proposed DIFUSCO, a novel graph-based diffusion model for solving NP-complete combinatorial optimization problems. We compared two variants of graph-based diffusion models: one with continuous Gaussian noise and one with discrete Bernoulli noise. We found that the discrete variant performs better than the continuous one. Moreover, we designed a `cosine` inference schedule that enhances the effectiveness of our model. DIFUSCO achieves state-of-the-art results on TSP and MIS problems, surpassing previous probabilistic NPC solvers in both accuracy and scalability. 

For future work, we would like to explore the potential of DIFUSCO in solving a broader range of NPC problems, including Mixed Integer Programming (discussed in the appendix). We would also like to explore the use of equivariant graph neural networks [117, 45] for further improvement of the diffusion models on geometrical NP-complete combinatorial optimization problems such as Euclidean TSP. Finally, we are interested in utilizing (higher-order) accelerated inference techniques for diffusion model-based solvers, such as those inspired by the continuous time framework for discrete diffusion [12, 105]. 

## **References** 

- [1] Sungsoo Ahn, Younggyo Seo, and Jinwoo Shin. Learning what to defer for maximum independent sets. In _International Conference on Machine Learning_ , pages 134–144. PMLR, 2020. 

- [2] Diogo V Andrade, Mauricio GC Resende, and Renato F Werneck. Fast local search for the maximum independent set problem. _Journal of Heuristics_ , 18(4):525–547, 2012. 

- [3] David Applegate, Ribert Bixby, Vasek Chvatal, and William Cook. Concorde TSP solver. `https://www.math.uwaterloo.ca/tsp/concorde/index.html` , 2006. 

- [4] Sanjeev Arora. Polynomial time approximation schemes for euclidean tsp and other geometric problems. In _Proceedings of 37th Conference on Foundations of Computer Science_ , pages 2–11. IEEE, 1996. 

- [5] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. _Advances in Neural Information Processing Systems_ , 34:17981–17993, 2021. 

- [6] Irwan Bello, Hieu Pham, Quoc V Le, Mohammad Norouzi, and Samy Bengio. Neural combinatorial optimization with reinforcement learning. _arXiv preprint arXiv:1611.09940_ , 2016. 

- [7] Jieyi Bi, Yining Ma, Jiahai Wang, Zhiguang Cao, Jinbiao Chen, Yuan Sun, and Yeow Meng Chee. Learning generalizable models for vehicle routing problems via knowledge distillation. In _Advances in Neural Information Processing Systems_ , 2022. 

- [8] Maximilian Böther, Otto Kißig, Martin Taraz, Sarel Cohen, Karen Seidel, and Tobias Friedrich. What’s wrong with deep learning in tree search for combinatorial optimization. In _International Conference on Learning Representations_ , 2022. URL `https://openreview.net/forum? id=mk0HzdqY7i1` . 

- [9] Xavier Bresson and Thomas Laurent. An experimental study of neural networks for variable graphs. 2018. 

- [10] Xavier Bresson and Thomas Laurent. The transformer network for the traveling salesman problem. _arXiv preprint arXiv:2103.03012_ , 2021. 

- [11] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877– 1901, 2020. 

10 

- [12] Andrew Campbell, Joe Benton, Valentin De Bortoli, Tom Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, _Advances in Neural Information Processing Systems_ , 2022. URL `https://openreview.net/forum? id=DmT862YAieY` . 

- [13] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 11315–11325, 2022. 

- [14] Nanxin Chen, Yu Zhang, Heiga Zen, Ron J Weiss, Mohammad Norouzi, and William Chan. Wavegrad: Estimating gradients for waveform generation. In _International Conference on Learning Representations_ , 2020. 

- [15] Nanxin Chen, Yu Zhang, Heiga Zen, Ron J Weiss, Mohammad Norouzi, Najim Dehak, and William Chan. Wavegrad 2: Iterative refinement for text-to-speech synthesis. _arXiv preprint arXiv:2106.09660_ , 2021. 

- [16] Ting Chen, Ruixiang Zhang, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. _arXiv preprint arXiv:2208.04202_ , 2022. 

- [17] Xinyun Chen and Yuandong Tian. Learning to perform local rewriting for combinatorial optimization. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- [18] Jinho Choo, Yeong-Dae Kwon, Jihoon Kim, Jeongwoo Jae, André Hottung, Kevin Tierney, and Youngjune Gwon. Simulation-guided beam search for neural combinatorial optimization. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, _Advances in Neural Information Processing Systems_ , 2022. URL `https://openreview.net/forum? id=tYAS1Rpys5` . 

- [19] Georges A Croes. A method for solving traveling-salesman problems. _Operations research_ , 6 (6):791–812, 1958. 

- [20] Paulo R d O Costa, Jason Rhuggenaath, Yingqian Zhang, and Alp Akcay. Learning 2-opt heuristics for the traveling salesman problem via deep reinforcement learning. In _Asian Conference on Machine Learning_ , pages 465–480. PMLR, 2020. 

- [21] Paulo R de O da Costa, Jason Rhuggenaath, Yingqian Zhang, and Alp Akcay. Learning 2-OPT heuristics for the traveling salesman problem via deep reinforcement learning. _arXiv preprint arXiv:2004.01608_ , 2020. 

- [22] Michel Deudon, Pierre Cournut, Alexandre Lacoste, Yossiri Adulyasak, and Louis-Martin Rousseau. Learning heuristics for the TSP by policy gradient. In _International conference on the integration of constraint programming, artificial intelligence, and operations research_ , pages 170–181. Springer, 2018. 

- [23] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. _Advances in Neural Information Processing Systems_ , 34:8780–8794, 2021. 

- [24] Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, et al. Continuous diffusion for categorical data. _arXiv preprint arXiv:2211.15089_ , 2022. 

- [25] Iddo Drori, Anant Kharkar, William R Sickinger, Brandon Kates, Qiang Ma, Suwen Ge, Eden Dolev, Brenda Dietrich, David P Williamson, and Madeleine Udell. Learning to solve combinatorial optimization problems on real-world graphs in linear time. In _2020 19th IEEE International Conference on Machine Learning and Applications (ICMLA)_ , pages 19–24. IEEE, 2020. 

- [26] Paul Erd˝os, Alfréd Rényi, et al. On the evolution of random graphs. _Publ. Math. Inst. Hung. Acad. Sci_ , 5(1):17–60, 1960. 

11 

- [27] Zhang-Hua Fu, Kai-Bin Qiu, and Hongyuan Zha. Generalize a small pre-trained model to arbitrarily large tsp instances. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pages 7474–7482, 2021. 

- [28] Simon Geisler, Johanna Sommer, Jan Schuchardt, Aleksandar Bojchevski, and Stephan Günnemann. Generalization of neural combinatorial solvers through the lens of adversarial robustness. In _International Conference on Learning Representations_ , 2022. URL `https://openreview.net/forum?id=vJZ7dPIjip3` . 

- [29] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In _International conference on machine learning_ , pages 1263–1272. PMLR, 2017. 

- [30] Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and LingPeng Kong. Diffuseq: Sequence to sequence text generation with diffusion models. _arXiv preprint arXiv:2210.08933_ , 2022. 

- [31] Teofilo F Gonzalez. _Handbook of approximation algorithms and metaheuristics_ . Chapman and Hall/CRC, 2007. 

- [32] Alexandros Graikos, Nikolay Malkin, Nebojsa Jojic, and Dimitris Samaras. Diffusion models as plug-and-play priors. In _Thirty-Sixth Conference on Neural Information Processing Systems_ , 2022. URL `https://arxiv.org/pdf/2206.09012.pdf` . 

- [33] Jiatao Gu, James Bradbury, Caiming Xiong, Victor OK Li, and Richard Socher. Nonautoregressive neural machine translation. In _International Conference on Learning Representations_ , 2018. 

- [34] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10696–10706, 2022. 

- [35] LLC Gurobi Optimization. Gurobi optimizer reference manual, 2018. 

- [36] Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. _Advances in neural information processing systems_ , 30, 2017. 

- [37] Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. Ssd-lm: Semi-autoregressive simplexbased diffusion language model for text generation and modular control. _arXiv preprint arXiv:2210.17432_ , 2022. 

- [38] Zhengfu He, Tianxiang Sun, Kuanning Wang, Xuanjing Huang, and Xipeng Qiu. Diffusionbert: Improving generative masked language models with diffusion models. _arXiv preprint arXiv:2211.15029_ , 2022. 

- [39] K. Helsgaun. An extension of the Lin-Kernighan-Helsgaun TSP solver for constrained traveling salesman and vehicle routing problems. Technical report, Roskilde University, 2017. 

- [40] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. _Advances in Neural Information Processing Systems_ , 33:6840–6851, 2020. 

- [41] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. _arXiv preprint arXiv:2210.02303_ , 2022. 

- [42] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. _J. Mach. Learn. Res._ , 23:47–1, 2022. 

- [43] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. _arXiv preprint arXiv:2204.03458_ , 2022. 

12 

- [44] Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. _Advances in Neural Information Processing Systems_ , 34:12454–12465, 2021. 

- [45] Emiel Hoogeboom, Vıctor Garcia Satorras, Clément Vignac, and Max Welling. Equivariant diffusion for molecule generation in 3d. In _International Conference on Machine Learning_ , pages 8867–8887. PMLR, 2022. 

- [46] Holger H Hoos and Thomas Stützle. SATLIB: An online resource for research on SAT. _Sat_ , 2000:283–292, 2000. 

- [47] André Hottung and Kevin Tierney. Neural large neighborhood search for the capacitated vehicle routing problem. _arXiv preprint arXiv:1911.09539_ , 2019. 

- [48] André Hottung, Yeong-Dae Kwon, and Kevin Tierney. Efficient active search for combinatorial optimization problems. _arXiv preprint arXiv:2106.05126_ , 2021. 

- [49] Benjamin Hudson, Qingbiao Li, Matthew Malencia, and Amanda Prorok. Graph neural network guided local search for the traveling salesperson problem. In _International Conference on Learning Representations_ , 2022. URL `https://openreview.net/forum?id= ar92oEosBIg` . 

- [50] Capgemini Research Institute. Capgemini research institute, the last-mile delivery challenge, 2023. URL `https://www.capgemini.com/wp-content/uploads/2019/01/ Report-Digital-âĂŞ-Last-Mile-Delivery-Challenge1.pdf` . 

- [51] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In _International conference on machine learning_ , pages 448–456. PMLR, 2015. 

- [52] Daniel D Johnson, Jacob Austin, Rianne van den Berg, and Daniel Tarlow. Beyond inplace corruption: Insertion and deletion in denoising probabilistic models. _arXiv preprint arXiv:2107.07675_ , 2021. 

- [53] Chaitanya K Joshi, Thomas Laurent, and Xavier Bresson. An efficient graph convolutional network technique for the travelling salesman problem. _arXiv preprint arXiv:1906.01227_ , 2019. 

- [54] Chaitanya K Joshi, Quentin Cappart, Louis-Martin Rousseau, and Thomas Laurent. Learning the travelling salesperson problem requires rethinking generalization. _Constraints_ , pages 1–29, 2022. 

- [55] Nikolaos Karalias and Andreas Loukas. Erdos goes neural: an unsupervised learning framework for combinatorial optimization on graphs. _Advances in Neural Information Processing Systems_ , 33:6659–6672, 2020. 

- [56] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. _arXiv preprint arXiv:2206.00364_ , 2022. 

- [57] Elias Khalil, Hanjun Dai, Yuyu Zhang, Bistra Dilkina, and Le Song. Learning combinatorial optimization algorithms over graphs. _Advances in neural information processing systems_ , 30, 2017. 

- [58] Minsu Kim, Jinkyoo Park, et al. Learning collaborative policies to solve NP-hard routing problems. _Advances in Neural Information Processing Systems_ , 34, 2021. 

- [59] Minsu Kim, Junyoung Park, and Jinkyoo Park. Sym-NCO: Leveraging symmetricity for neural combinatorial optimization. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, _Advances in Neural Information Processing Systems_ , 2022. URL `https://openreview.net/forum?id=kHrE2vi5Rvs` . 

- [60] Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. _Advances in neural information processing systems_ , 34:21696–21707, 2021. 

13 

- [61] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. _arXiv preprint arXiv:1609.02907_ , 2016. 

- [62] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In _International Conference on Learning Representations_ , 2017. URL `https: //openreview.net/forum?id=SJU4ayYgl` . 

- [63] Vijay R Konda and John N Tsitsiklis. Actor-critic algorithms. In _Advances in neural information processing systems_ , pages 1008–1014, 2000. 

- [64] Wouter Kool, Herke van Hoof, and Max Welling. Attention, learn to solve routing problems! In _International Conference on Learning Representations_ , 2019. 

- [65] Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 REINFORCE samples, get a baseline for free! In _Deep Reinforcement Learning Meets Structured Prediction, ICLR 2019 Workshop_ , 2019. 

- [66] Alex Krizhevsky and Geoff Hinton. Convolutional deep belief networks on cifar-10. _Unpublished manuscript_ , 40(7):1–9, 2010. 

- [67] Yeong-Dae Kwon, Jinho Choo, Byoungjip Kim, Iljoo Yoon, Youngjune Gwon, and Seungjai Min. POMO: Policy optimization with multiple optima for reinforcement learning. _arXiv preprint arXiv:2010.16011_ , 2020. 

- [68] Yeong-Dae Kwon, Jinho Choo, Iljoo Yoon, Minah Park, Duwon Park, and Youngjune Gwon. Matrix encoding networks for neural combinatorial optimization. _Advances in Neural Information Processing Systems_ , 34, 2021. 

- [69] Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori Hashimoto. Diffusion-LM improves controllable text generation. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, _Advances in Neural Information Processing Systems_ , 2022. URL `https://openreview.net/forum?id=3s9IrEsjLyk` . 

- [70] Zhuwen Li, Qifeng Chen, and Vladlen Koltun. Combinatorial optimization with graph convolutional networks and guided tree search. _Advances in neural information processing systems_ , 31, 2018. 

- [71] Shen Lin and Brian W Kernighan. An effective heuristic algorithm for the traveling-salesman problem. _Operations research_ , 21(2):498–516, 1973. 

- [72] Jinglin Liu, Chengxi Li, Yi Ren, Feiyang Chen, and Zhou Zhao. Diffsinger: Singing voice synthesis via shallow diffusion mechanism. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 36, pages 11020–11028, 2022. 

- [73] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In _International Conference on Learning Representations_ , 2021. 

- [74] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. _arXiv preprint arXiv:2206.00927_ , 2022. 

- [75] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpmsolver++: Fast solver for guided sampling of diffusion probabilistic models. _arXiv preprint arXiv:2211.01095_ , 2022. 

- [76] Hao Lu, Xingwen Zhang, and Shuang Yang. A learning-based iterative method for solving vehicle routing problems. In _International Conference on Learning Representations_ , 2020. 

- [77] Shitong Luo, Yufeng Su, Xingang Peng, Sheng Wang, Jian Peng, and Jianzhu Ma. Antigenspecific antibody design and optimization with diffusion-based generative models for protein structures. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, _Advances in Neural Information Processing Systems_ , 2022. URL `https://openreview. net/forum?id=jSorGn2Tjg` . 

14 

- [78] Qiang Ma, Suwen Ge, Danyang He, Darshan Thaker, and Iddo Drori. Combinatorial optimization by graph pointer networks and hierarchical reinforcement learning. _arXiv preprint arXiv:1911.04936_ , 2019. 

- [79] Yining Ma, Jingwen Li, Zhiguang Cao, Wen Song, Le Zhang, Zhenghua Chen, and Jing Tang. Learning to iteratively solve routing problems with dual-aspect collaborative transformer. _Advances in Neural Information Processing Systems_ , 34:11096–11107, 2021. 

- [80] Cedric Malherbe, Antoine Grosnit, Rasul Tutunov, Haitham Bou Ammar, and Jun Wang. Optimistic tree searches for combinatorial black-box optimization. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, _Advances in Neural Information Processing Systems_ , 2022. URL `https://openreview.net/forum?id=JGLW4DvX11F` . 

- [81] Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Image synthesis and editing with stochastic differential equations. _arXiv preprint arXiv:2108.01073_ , 2021. 

- [82] Vinod Nair, Sergey Bartunov, Felix Gimeno, Ingrid von Glehn, Pawel Lichocki, Ivan Lobov, Brendan O’Donoghue, Nicolas Sonnerat, Christian Tjandraatmadja, Pengming Wang, et al. Solving mixed integer programs using neural networks. _arXiv preprint arXiv:2012.13349_ , 2020. 

- [83] Mohammadreza Nazari, Afshin Oroojlooy, Lawrence Snyder, and Martin Takác. Reinforcement learning for solving the vehicle routing problem. _Advances in neural information processing systems_ , 31, 2018. 

- [84] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. _arXiv preprint arXiv:2112.10741_ , 2021. 

- [85] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In _International Conference on Machine Learning_ , pages 8162–8171. PMLR, 2021. 

- [86] Chenhao Niu, Yang Song, Jiaming Song, Shengjia Zhao, Aditya Grover, and Stefano Ermon. Permutation invariant graph generation via score-based generative modeling. In _International Conference on Artificial Intelligence and Statistics_ , pages 4474–4484. PMLR, 2020. 

- [87] Wenbin Ouyang, Yisen Wang, Shaochen Han, Zhejian Jin, and Paul Weng. Improving generalization of deep reinforcement learning-based tsp solvers. _arXiv preprint arXiv:2110.02843_ , 2021. 

- [88] Christos H Papadimitriou and Kenneth Steiglitz. _Combinatorial optimization: algorithms and complexity_ . Courier Corporation, 1998. 

- [89] Junyoung Park, Jaehyeong Chun, Sang Hun Kim, Youngkook Kim, and Jinkyoo Park. Learning to schedule job-shop problems: representation and policy learning using graph neural network and reinforcement learning. _International Journal of Production Research_ , 59(11):3360–3377, 2021. 

- [90] Bo Peng, Jiahai Wang, and Zizhen Zhang. A deep reinforcement learning algorithm using dynamic attention model for vehicle routing problems. In _International Symposium on Intelligence Computation and Applications_ , pages 636–650. Springer, 2019. 

- [91] pitney bowes. Pitney bowes parcel shipping index, 2023. URL `https://www.pitneybowes. com/us/shipping-index.html` . 

- [92] Ruizhong Qiu, Zhiqing Sun, and Yiming Yang. Dimes: A differentiable meta solver for combinatorial optimization problems. In _Advances in Neural Information Processing Systems_ , 2022. 

- [93] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. _arXiv preprint arXiv:2204.06125_ , 2022. 

15 

- [94] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10684–10695, 2022. 

- [95] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In _ACM SIGGRAPH 2022 Conference Proceedings_ , pages 1–10, 2022. 

- [96] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. _arXiv preprint arXiv:2205.11487_ , 2022. 

- [97] Paul Shaw. A new local search algorithm providing high quality solutions to vehicle routing problems. _APES Group, Dept of Computer Science, University of Strathclyde, Glasgow, Scotland, UK_ , 46, 1997. 

- [98] David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. _nature_ , 529(7587): 484–489, 2016. 

- [99] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. _arXiv preprint arXiv:2209.14792_ , 2022. 

- [100] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In _International Conference on Machine Learning_ , pages 2256–2265. PMLR, 2015. 

- [101] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In _International Conference on Learning Representations_ , 2021. URL `https://openreview. net/forum?id=St1giarCHLP` . 

- [102] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- [103] Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. _Advances in neural information processing systems_ , 33:12438–12448, 2020. 

- [104] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In _International Conference on Learning Representations_ , 2021. 

- [105] Haoran Sun, Lijun Yu, Bo Dai, Dale Schuurmans, and Hanjun Dai. Score-based continuoustime discrete diffusion models. _arXiv preprint arXiv:2211.16750_ , 2022. 

- [106] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Advances in neural information processing systems_ , pages 5998–6008, 2017. 

- [107] Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. Graph attention networks. In _International Conference on Learning Representations_ , 2018. 

- [108] Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. Digress: Discrete denoising diffusion for graph generation. _arXiv preprint arXiv:2209.14734_ , 2022. 

- [109] Chenguang Wang, Yaodong Yang, Oliver Slumbers, Congying Han, Tiande Guo, Haifeng Zhang, and Jun Wang. A game-theoretic approach for improving generalization ability of TSP solvers. _arXiv preprint arXiv:2110.15105_ , 2021. 

16 

- [110] Runzhong Wang, Zhigang Hua, Gan Liu, Jiayi Zhang, Junchi Yan, Feng Qi, Shuang Yang, Jun Zhou, and Xiaokang Yang. A bi-level framework for learning to solve combinatorial optimization on graphs. _arXiv preprint arXiv:2106.04927_ , 2021. 

- [111] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine learning_ , 8(3):229–256, 1992. 

- [112] Lemeng Wu, Chengyue Gong, Xingchao Liu, Mao Ye, and Qiang Liu. Diffusion-based molecule generation with informative prior bridges. _arXiv preprint arXiv:2209.00865_ , 2022. 

- [113] Yaoxin Wu, Wen Song, Zhiguang Cao, Jie Zhang, and Andrew Lim. Learning improvement heuristics for solving routing problems.. _IEEE transactions on neural networks and learning systems_ , 2021. 

- [114] Liang Xin, Wen Song, Zhiguang Cao, and Jie Zhang. Multi-decoder attention model with embedding glimpse for solving vehicle routing problems. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pages 12042–12049, 2021. 

- [115] Liang Xin, Wen Song, Zhiguang Cao, and Jie Zhang. NeuroLKH: Combining deep learning model with Lin–Kernighan–Helsgaun heuristic for solving the traveling salesman problem. _Advances in Neural Information Processing Systems_ , 34, 2021. 

- [116] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In _International Conference on Learning Representations_ , 2019. URL `https://openreview.net/forum?id=ryGs6iA5Km` . 

- [117] Minkai Xu, Lantao Yu, Yang Song, Chence Shi, Stefano Ermon, and Jian Tang. Geodiff: A geometric diffusion model for molecular conformation generation. In _International Conference on Learning Representations_ , 2021. 

- [118] Dongchao Yang, Jianwei Yu, Helin Wang, Wen Wang, Chao Weng, Yuexian Zou, and Dong Yu. Diffsound: Discrete diffusion model for text-to-sound generation. _arXiv preprint arXiv:2207.09983_ , 2022. 

- [119] Emre Yolcu and Barnabás Póczos. Learning local search heuristics for boolean satisfiability. In _NeurIPS_ , pages 7990–8001, 2019. 

- [120] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. _Transactions on Machine Learning Research_ . 

- [121] Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. _arXiv preprint arXiv:2212.05199_ , 2022. 

- [122] Cong Zhang, Wen Song, Zhiguang Cao, Jie Zhang, Puay Siew Tan, and Chi Xu. Learning to dispatch for job shop scheduling via deep reinforcement learning. _arXiv preprint arXiv:2010.12367_ , 2020. 

17 

## **A Frequently Asked Questions** 

## **A.1 Does DIFUSCO rely on high-quality solutions collected in advance?** 

DIFUSCO does not require exact optimal solutions to train the diffusion model-based solver. For instance, in TSP-500/1000/10000, we utilize the heuristic solver LKH-3 to generate near-optimal solutions, which is nearly as fast as the neural solvers themselves (refer to Table 2 for details). Empirically (in TSP-500/1000/10000), DIFUSCO still attains state-of-the-art performance even when trained on such near-optimal solutions rather than on exact optimal ones. The training data sizes are reported in Appendix E. 

## **A.2 This work simply applies discrete diffusion models to combinatorial optimization problems. Could you clarify the significance of this work in the field?** 

It is essential to highlight that our work introduces a novel approach to solving NP-complete problems using graph-based diffusion models. This approach has not been previously explored in the context of NP-complete problems, and we believe that the significant improvements in performance over the state-of-the-art on benchmark datasets emphasize the value of our work. 

Drawing a parallel to the application of diffusion models in image generation and subsequent extension to videos, our work explores a similar expansion of the scope. While denoising diffusion models have been extensively researched and applied to image generation, DIFUSCO represents the first instance where these models are applied to NP-complete CO problems. Such novelty in problem formulation is fundamental for solving NP-complete problems instead of incremental ones from a methodological point of view. 

The formulation of NP-complete problems into a discrete _{_ 0 _,_ 1 _}_ -vector space and the use of graphbased denoising diffusion models to generate high-quality solutions make our approach unique. Additionally, we explore diffusion models with Gaussian and Bernoulli noise and introduce an effective inference schedule to improve the generation quality, which further underlines the significance of our work. 

In conclusion, we believe that our work makes a significant and novel contribution to the field by introducing a new graph-based diffusion framework for solving NP-complete problems. The application of diffusion models, which have been primarily used for image and video generation, to the realm of combinatorial optimization highlights the innovative aspects of our research. We hope that this clarification demonstrates its importance in the field. 

## **A.3 How can DIFUSCO produce diverse multimodal output distribution?** 

Such ability comes from the characteristics of diffusion models, which are known for generating a wide variety of distributions as generative models (like diverse images). A more comprehensive understanding can be found in the related literature, such as DDPM. 

## **A.4 Why is there no demonstration for the time-cost in the results shown in Table 1, and why are there some conflicts in the metrics of Length and Gaps?** 

The time is not reported in Table 1 as these baseline methods are evaluated on very different hardware, which makes the runtime comparison not too meaningful. As for the “conflicts”, the results (both length and optimality gap) of all the baseline methods are directly copied from previous papers. The “conflicts” may be caused by rounding issues when reporting numerical numbers in previous papers, as our best guess. 

The -0.01 gap in DIFUSCO is because the Concorde solver only accepts integer coordinates as the inputs, which may lead it to produce inaccurate non-optimal solutions due to rounding issues. 

## **A.5 How can we effectively evaluate the usefulness of the proposed method when the training time is also a consideration?** 

It is important to clarify that in the context of neural combinatorial optimization solvers and general machine learning research, the primary focus is typically on inference time. This is because, after 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0019-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0019-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0019-02.png)


**----- Start of picture text -----**<br>
(a) (b)<br>**----- End of picture text -----**<br>


Figure 4: The performance Gap (%) of continuous diffusion **(a)** and discrete diffusion **(b)** models on TSP-50 with different diffusion steps and number of samples. **(c)** : The results are reported with greedy decoding without 2-opt post-processing. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0019-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0019-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0019-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0019-07.png)


**----- Start of picture text -----**<br>
(a) (b) (c)<br>**----- End of picture text -----**<br>


Figure 5: The inference per-instance run-time (sec) of diffusion models on TSP-50, where the total run-time is decomposed to neural network **(a)** + greedy decoding **(b)** + 2-opt **(c)** . 

the model has been trained, it can be applied to a virtually unlimited number of unseen examples (graphs) during deployment. On the other hand, the training time is often considered negligible in comparative evaluations, partly due to the fact that traditional NCP solvers are hand-crafted instead of learning-based, and partly because the training cost for learnable models is out weighted by the benefits of their numerous applications. 

## **A.5.1 Can you name a real application that has many test instances and require such a model?** 

It is worth noting that the routing problem is a fundamental and perhaps the most important combinatorial optimization problem in real-world scenarios. One example is the food delivery problem, where a pizza shop is planning to deliver pizzas to four different addresses and then return to the store. The routing problem has significant implications for industries such as retail, quick service restaurants (QSRs), consumer packaged goods (CPG), and manufacturing. In 2020, parcel shipping exceeded 131 billion in volume globally and is projected to more than double by 2026 [91]. With the changing economic and geopolitical landscape within the transport and logistics industry, Last Mile Delivery (LMD) has become the most expensive portion of the logistics fulfillment chain, representing over 41% of overall supply chain costs [50]. 

## **B Extended Related Work** 

**Autoregressive Constructive Solvers** Since Bello et al. [6] proposed the first autoregressive CO solver, more advanced models have been developed in the years since [22, 64, 90, 25, 68], including better network backbones [106, 64, 9]), more advanced deep reinforcement learning algorithms [57, 78, 65, 67, 87, 114, 109, 18], improved training strategies [59, 7], and for a wider range of NPC problems such as Capacitated Vehicle Routing Problem (CVRP) [83], Job Shop Scheduling Problem (JSSP) [122, 89], Maximal Independent Set (MIS) problem [57, 1, 80, 92], and boolean satisfiability problem (SAT) [119]. 

19 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0020-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0020-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0020-02.png)


**----- Start of picture text -----**<br>
(a) (b)<br>**----- End of picture text -----**<br>


Figure 6: Generalization tests of discrete DIFUSCO trained and evaluated on TSP problems across various scales. The results are reported with **(a)** and without **(b)** 2-opt post-processing. 

Table 4: Comparing discrete diffusion and continuous diffusion on TSP-100 with various diffusion steps and numbers of parallel sampling. `cosine` schedule is used for fast sampling. 

|DIFFUSION|STEPS|#SAMPLE|DISCRETE <br>W/ 2OPT|DIFFUSION(Gap%)<br>W/O2OPT|CONTINUOUS <br>W/ 2OPT|DIFFUSION(Gap%)<br>W/O2OPT|PER-INSTANCE RUNTIME(sec)<br>NN<br>GD<br>2-OPT|PER-INSTANCE RUNTIME(sec)<br>NN<br>GD<br>2-OPT|PER-INSTANCE RUNTIME(sec)<br>NN<br>GD<br>2-OPT|
|---|---|---|---|---|---|---|---|---|---|
|50||1|0.23869|1.45574|1.46146|7.66379|0.50633|0.00171|0.00210|
|100||1|0.23366|1.48161|1.32573|7.02117|1.00762|0.00170|0.00207|
|50||4|0.02253|0.09280|0.42741|1.65264|1.52401|0.00643|0.00575|
|10||16|-0.01870|0.00519|0.13015|0.54983|1.12550|0.02581|0.02228|
|50||16|-0.02322|-0.00699|0.09407|0.30712|5.63712|0.02525|0.02037|



**Improvement Heuristics Solvers** Unlike construction heuristics, DRL-based improvement heuristics solvers use neural networks to iteratively enhance the quality of the current solution until the computational budget is exhausted. Such DRL-based improvement heuristics methods are usually inspired by classical local search algorithms such as 2-opt [19] and the large neighborhood search (LNS) [97], and have been demonstrated with outstanding results by many previous works [17, 47, 21, 20, 115, 79, 113, 76, 110, 58, 49]. 

Improvement heuristics methods, while showing superior performance compared to construction heuristics methods, come at the cost of increased computational time, often requiring thousands of actions even for small-scale problems with hundreds of nodes [20, 110]. This is due to the sequential application of local operations, such as 2-opt, on existing solutions, resulting in a bottleneck for latency. On the other hand, DIFUSCO has the advantage of denoising all variables in parallel, which leads to a reduction in the number of network evaluations required. 

**Continuous Diffusion Models** Diffusion models were first proposed by Sohl-Dickstein et al. [100] and recently achieved impressive success on various tasks, such as high-resolution image synthesis [23, 42], image editing [81, 95], text-to-image generation [84, 96, 94, 93, 34], waveform generation [14, 15, 72], video generation [43, 41, 99], and molecule generation [117, 45, 112]. Recent works have also drawn connections to stochastic differential equations (SDEs) [104] and ordinary differential equations (ODEs) [101] in a continuous time framework, leading to improved sampling algorithms by solving discretized SDEs/ODEs with higher-order solvers [73, 75, 74] or implicit diffusion [101]. 

## **C Additional Results** 

**Discrete Diffusion v.s. Continuous Diffusion on TSP-100** We also compare discrete diffusion and continuous diffusion on the TSP-100 benchmark and report the results in Tab. 4. We can see that on TSP-100, discrete diffusion models still consistently outperform their continuous counterparts in various settings. 

**More Diffusion Steps v.s. More Sampling (w/o 2-opt)** Fig. 4 report the results of continuous diffusion and discrete diffusion with various diffusion steps and numbers of parallel sampling, without using 2-opt post-processing. The `cosine` denoising schedule is used for fast inference. Again, we find that discrete diffusion outperforms continuous diffusion across various settings. 

20 

Table 5: Comparison to DIMES w/ 2-opt 

|||TSP-500|TSP-500|TSP-1000|TSP-1000|TSP-10000|TSP-10000|
|---|---|---|---|---|---|---|---|
|||Length|Gap|Length|Gap|Length|Gap|
|DIMES|RL+S+2OPT|17.63|6.52%|24.81|7.31%|77.18|7.54%|
|DIMES|RL+AS+S+2OPT|17.30|4.53%|24.32|5.19%|75.94|5.81%|
|DIFUSCO|SL+G+2OPT|16.80|1.49%|23.56|1.90%|73.99|3.10%|
|DIFUSCO|SL+S+2OPT|16.65|0.57%|23.45|1.43%|73.89|2.95%|



Table 6: Solution quality and computation time for learning-based methods using models trained on synthetic data (all the other baselines are trained with TSP-100) and evaluated on TSPLIB instances with 50 to 200 nodes and 2D Euclidean distances. Other baseline results are taken from Hudson et al. [49]. 

|Method|Kool|et al.|Joshi|et al.|O. da Costa et al.|O. da Costa et al.|Hudson et al.|Hudson et al.|Ours (TSP-50)|Ours (TSP-50)|Ours (TSP-100)|Ours (TSP-100)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Instance|Time (s)|Gap (%)|Time (s)|Gap (%)|Time (s)|Gap (%)|Time (s) Gap (%)||Time (s) Gap (%)||Time (s)|Gap (%)|
|eil51|0.125|1.628|3.026|8.339|28.051|0.067|10.074|**0.000**|0.482|**0.000**|0.519|0.117|
|berlin52|0.129|4.169|3.068|33.225|31.874|0.449|10.103|0.142|0.527|**0.000**|0.526|**0.000**|
|st70|0.200|1.737|4.037|24.785|23.964|0.040|10.053|0.764|0.663|**0.000**|0.670|**0.000**|
|eil76|0.225|1.992|4.303|27.411|26.551|0.096|10.155|0.163|0.788|**0.000**|0.788|0.174|
|pr76|0.226|0.816|4.378|27.793|39.485|1.228|10.049|0.039|0.765|**0.000**|0.785|0.187|
|rat99|0.347|2.645|5.559|17.633|32.188|0.123|9.948|0.550|1.236|1.187|1.192|**0.000**|
|kroA100|0.352|4.017|5.705|28.828|42.095|18.313|10.255|0.728|1.259|0.741|1.217|**0.000**|
|kroB100|0.352|5.142|5.712|34.686|35.137|1.119|10.317|0.147|1.252|0.648|1.235|**0.742**|
|kroC100|0.352|0.972|5.641|35.506|34.333|0.349|10.172|1.571|1.199|1.712|1.168|**0.000**|
|kroD100|0.352|2.717|5.621|38.018|25.772|0.866|10.375|0.572|1.226|**0.000**|1.175|0.000|
|kroE100|0.352|1.470|5.650|26.589|34.475|1.832|10.270|1.216|1.208|0.274|1.197|**0.274**|
|rd100|0.352|3.407|5.737|50.432|28.963|0.003|10.125|0.459|1.191|**0.000**|1.172|0.000|
|eil101|0.359|2.994|5.790|26.701|23.842|0.387|10.276|0.201|1.222|0.576|1.215|**0.000**|
|lin105|0.380|1.739|5.938|34.902|39.517|1.867|10.330|0.606|1.321|**0.000**|1.280|**0.000**|
|pr107|0.391|3.933|5.964|80.564|29.039|0.898|9.977|0.439|1.381|**0.228**|1.378|0.415|
|pr124|0.499|3.677|7.059|70.146|29.570|10.322|10.360|0.755|1.803|0.925|1.782|**0.494**|
|bier127|0.522|5.908|7.242|45.561|39.029|3.044|10.260|1.948|1.938|**1.011**|1.915|0.366|
|ch130|0.550|3.182|7.351|39.090|34.436|0.709|10.032|3.519|1.989|**1.970**|1.967|0.077|
|pr136|0.585|5.064|7.727|58.673|31.056|0.000|10.379|3.387|2.184|**2.490**|2.142|0.000|
|pr144|0.638|7.641|8.132|55.837|28.913|1.526|10.276|3.581|2.478|**0.519**|2.446|0.261|
|ch150|0.697|4.584|8.546|49.743|35.497|0.312|10.109|2.113|2.608|**0.376**|2.555|0.000|
|kroA150|0.695|3.784|8.450|45.411|29.399|0.724|10.331|2.984|2.617|3.753|2.601|**0.000**|
|kroB150|0.696|2.437|8.573|56.745|29.005|0.886|10.018|3.258|2.626|**1.839**|2.592|0.067|
|pr152|0.708|7.494|8.632|33.925|29.003|0.029|10.267|3.119|2.716|**1.751**|2.712|0.481|
|u159|0.764|7.551|9.012|38.338|28.961|0.054|10.428|1.020|2.963|3.758|2.892|**0.000**|
|rat195|1.114|6.893|11.236|24.968|34.425|0.743|12.295|1.666|4.400|1.540|4.428|**0.767**|
|d198|1.153|373.020|11.519|62.351|30.864|0.522|12.596|4.772|4.615|4.832|4.153|**3.337**|
|kroA200|1.150|7.106|11.702|40.885|33.832|1.441|11.088|2.029|4.710|6.187|4.686|**0.065**|
|kroB200|1.150|8.541|11.689|43.643|31.951|2.064|11.267|2.589|4.606|6.605|4.619|**0.590**|
|Mean|0.532|16.767|7.000|40.025|31.766|1.725|10.420|1.529|1.999|1.480|1.966|**0.290**|



Besides, we find that without the 2-opt post-processing, performing more diffusion iterations is much more effective than sampling more solutions, even when the former uses less computation. For example, 20 (diffusion steps) _×_ 4 (samples) not only significantly outperforms 1 (diffusion steps) _×_ 1024 (samples), but also has a 18 _._ 5 _×_ less runtime. 

**Runtime Analysis** We report the decomposed runtime (neural network + greedy decoding + 2-opt) for diffusion models on TSP-50 in Fig. 5. We can see that while neural network execution takes the majority of total runtime, 2-opt also takes a non-negligible portion of the runtime, especially when only a few diffusion steps (like 1 or 2) are used. 

**Generalization Tests (w/o 2opt)** We also report the generalization tests of discrete DIFUSCO without 2-opt post-processing in Fig.6 (b). 

**Comparison to DIMES w/ 2opt** We compare DIMES with 2-opt to DIFUSCO in Tab. 5. We can see that while 2-opt can further improve the performance of DIMES on large-scale TPS problem instances, there is still a significant gap between DIEMS and DIFUSCO when both are equipped with 2-opt post-processing. 

21 

Table 7: Results of DIFUSCO with multiple samples for MCTS 

|||TSP-500|TSP-500|TSP-1000|TSP-1000|TSP-10000|TSP-10000|
|---|---|---|---|---|---|---|---|
|||Length|Gap|Length|Gap|Length|Gap|
|PREVIOUS|SOTA|16.84|1.76%|23.69|2.46%|74.06|3.19%|
|DIFUSCO|1XMCTS|16.64|0.46%|23.39|1.17%|73.62|2.58%|
|DIFUSCO|2XMCTS|16.57|0.09%|23.32|0.56%|73.60|2.55%|
|DIFUSCO|4XMCTS|16.56|0.02%|23.30|0.46%|73.54|2.48%|



**Generalization to Real-World Instances** In many cases, there might not be enough real-world problem instances available to train a model. As a result, the model needs to be trained using synthetically generated data. Hence, it becomes crucial to assess the effectiveness of transferring knowledge from the simulated environment to the real world. we evaluated the DIFUSCO models trained on TSP-50 and TSP-100 on TSPLib graphs with 50-200 nodes and a comparison to other methods (trained TSP-100) [64, 53, 20, 49] is shown in Tab. 6. 

We can see that DIFUSCO achieves significant improvements over other baselines on the real-world TSPLIB data (0.29% gap v.s. 1.48% in the best baseline). Besides, DIFUSCO also shows strong generalization ability across problem scales, where DIFUSCO trained on TSP-50 data achieves competitive performance with the best baseline Hudson et al. [49] trained on TSP-100. 

**MCTS with Multiple Diffusion Samples** One may wonder what if DIFUSCO with MCTS is allowed to have more than one sample. To evaluate the impact of using multiple samples in DIFUSCO with MCTS, we conducted experiments with 1x, 2x, and 4x greedy heatmap generation (50 diffusion steps) combined with MCTS using different random seeds, and report the results in Tab. 7. 

Our findings show that DIFUSCO’s performance significantly improves when MCTS decoding is applied to diverse heatmaps. We would like to highlight that the diverse heatmap + MCTS decoding approach is unique to diffusion model-based neural solvers like DIFUSCO, as previous methods (e.g., Att-GCN and DIMES) that employ MCTS only yield a unimodal distribution. While it is true that using 2x or 4x MCTS decoding would increase the runtime of DIFUSCO, we believe there is a similar exploitation versus exploration trade-off between MCTS time and MCTS seeds, akin to the trade-off between diffusion steps and the number of samples demonstrated in Figure 2 of our paper. 

## **D Discussion on the** _{_ 0 _,_ 1 _}[N]_ **Vector Space of CO Problems** 

The design of the _{_ 0 _,_ 1 _}[N]_ vector space can also represent non-graph-based NP-complete combinatorial optimization problems. For example, on the more general Mixed Integer Programming (MIP) problem, we can let _Xs_ be a 0/1 indication set of all extended variables[4] . cost( _·_ ) can be defined as a linear/quadratic function of **x** , and valid( _·_ ) is a function validating all linear/quadratic constraints, bound constraints, and integrality constraints. 

## **E Additional Experiment Details** 

**Metrics: TSP** For TSP, Length is defined as the average length of the system-predicted tour for each test-set instance. Gap is the average of the relative decrease in performance compared to a baseline method. Time is the total clock time required to generate solutions for all test instances, and is presented in seconds (s), minutes (m), or hours (h). 

**Metrics: MIS** For MIS, Size (the larger, the better) is the average size of the system-predicted maximal independent set for each test-set graph, Gap and Time are defined similarly as in the TSP case. 

**Hardware** All the methods are trained with 8 _×_ NVIDIA Tesla V100 Volta GPUs and evaluated on a single NVIDIA Tesla V100 Volta GPU, with 40 _×_ Intel(R) Xeon(R) Gold 6248 CPUs @ 2.50GHz. 

4 For an integer variable _z_ that can be assigned values from a finite set with cardinality card( _z_ ), any target value can be represented as a sequence of _⌈_ log2(card( _z_ )) _⌉_ bits [82, 16]. 

22 

**Random Seeds** Since the diffusion models can generate an arbitrary sample from its distribution even with the greedy decoding scheme, we report the averaged results across 5 different random seeds when reporting all results. 

**Training Details** All DIFUSCO models are trained with a cosine learning rate schedule starting from 2e-4 and ending at 0. 

- TSP-50: We use 1502000 random instances and train DIFUSCO for 50 epochs with a batch size of 512. 

- TSP-100: We use 1502000 random instances and train DIFUSCO for 50 epochs with a batch size of 256. 

- TSP-500: We use 128000 random instances and train DIFUSCO for 50 epochs with a batch size of 64. We also apply curriculum learning and initialize the model from the TSP-100 checkpoint. 

- TSP-1000: We use 64000 random instances and train DIFUSCO for 50 epochs with a batch size of 64. We also apply curriculum learning and initialize the model from the TSP-100 checkpoint. 

- TSP-10000: We use 6400 random instances and train DIFUSCO for 50 epochs with a batch size of 8. We also apply curriculum learning and initialize the model from the TSP-500 checkpoint. 

- SATLIB: We use the training split of 49500 examples from [46, 92] and train DIFUSCO for 50 epochs with a batch size of 128. 

- ER-[700-800]: We use 163840 random instances and train DIFUSCO for 50 epochs with a batch size of 32. 

## **F Decoding Strategies** 

**Greedy Decoding** We use a simple greedy decoding scheme for diffusion models, where we sample one solution from the learned distribution _p_ _**θ**_ ( **x** 0). We compare this scheme with other autoregressive constructive solvers that also use greedy decoding. 

**Sampling** Following Kool et al. [64], we also use a sampling scheme where we sample multiple solutions in parallel (i.e., each diffusion model starts with a different noise **xT** ) and report the best one. 

**Monte Carlo Tree Search** For the TSP task, we adopt a more advanced reinforcement learningbased search approach, i.e., Monte Carlo tree search (MCTS), to find high-quality solutions. In MCTS, we sample _k_ -opt transformation actions guided by the heatmap generated by the diffusion model to improve the current solutions. The MCTS repeats the simulation, selection, and back-propagation steps until no improving actions are found in the sampling pool. For more details, please refer to [27]. 

## **G Fast Inference for Continuous and Discrete Diffusion Models** 

We first describe the denoising diffusion implicit models (DDIMs) [101] algorithm for accelerating inference for continuous diffusion models. 

Formally, consider the forward process defined not on all the latent variables **x** 1: _T_ , but on a subset _{_ **x** _τ_ 1 _, . . . ,_ **x** _τM }_ , where _τ_ is an increasing sub-sequence of [1 _, . . . , T_ ] with length _M_ , **x** _τ_ 1 = 1 and **x** _τM_ = _T_ . 

For continuous diffusion, the marginal can still be defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0023-17.png)


And it’s (deterministic) posterior is defined by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0023-19.png)


¯ ¯ where � _**ϵ** τi_ = ( **x** _τi −[√] ατi_ **x** 0) _/[√]_ 1 _− ατi_ is the (predicted) diffusion noise. 

Next, we describe its analogy in the discrete domain, which is first proposed by Austin et al. [5]. 

23 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0024-00.png)


Figure 7: Qualitative illustration of how diffusion steps affect the generation quality of **continuous** diffusion models. The results are reported without any post-processing. **Continuous** DIFUSCO with 50 (first row), 20 (second row), 10 (third row), and 5 (last row) diffusion steps in `cosine` schedule are shown. 

For discrete diffusion, the marginal can still be defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0024-03.png)


while the posterior becomes: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0024-05.png)


where **Q** _t′,t_ = **Q** _t′_ +1 _. . ._ **Q** _t_ . 

## **H Experiment Baselines** 

**TSP-50/100** We evaluate DIFUSCO on TSP-50 and TSP-100 against 10 baselines, which belong to two categories: traditional Operations Research (OR) methods and learning-based methods. 

- Traditional OR methods include Concorde [3], an exact solver, and 2-opt [71], a heuristic method. 

- Learning-based methods include AM [64], GCN [53], Transformer [10], POMO [67], SymNCO[59], DPDP [79], Image Diffusion [32], and MDAM [114]. These are the state-of-the-art methods in recent benchmark studies. 

**TSP-500/1000/10000** We compare DIFUSCO on large-scale TSP problems with 9 baseline methods, which fall into two categories: traditional Operations Research (OR) methods and learning-based methods. 

- Traditional OR methods include Concorde [3] and Gurobi [35], which are exact solvers, and LKH-3 [39], which is a strong heuristic solver. We use two settings for LKH-3: (i) _default_ : 10000 trials (the default configuration of LKH-3); (ii) _less trials_ : 500 trials for TSP-500/1000 and 250 trials for TSP-10000. We also include Farthest Insertion, a simple heuristic method, as a baseline. 

- Learning-based methods include EAN [22], AM [64], GCN [53], POMO+EAS [67, 48], Att-GCN 

- [27], and DIMES [92]. These are the state-of-the-art methods in recent benchmark studies. They 

24 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0025-00.png)


Figure 8: Qualitative illustration of how diffusion steps affect the generation quality of **discrete** diffusion models. The results are reported without any post-processing. **Discrete** DIFUSCO with 50 (first row), 20 (second row), 10 (third row), and 5 (last row) diffusion steps in `cosine` schedule are shown. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0025-02.png)


Figure 9: Qualitative illustration of how diffusion schedules affect the generation quality of **continuous** diffusion models. The results are reported without any post-processing. **Continuous** DIFUSCO with `linear` schedule (first row) and `cosine` schedule (second row) with 20 diffusion steps are shown. 

can be further divided into reinforcement learning (RL) and supervised learning (SL) methods. Some RL methods can also use an Active Search (AS) stage [6] to fine-tune each instance. We take the results of the baselines from Fu et al. [27] and Qiu et al. [92]. Note that except for Att-GCN and DIMES, the baselines are trained on small graphs and tested on large graphs. 

**MIS** For MIS, we compare DIFUSCO with 6 other MIS solvers on the same test sets, including two traditional OR methods (i.e., Gurobi and KaMIS) and four learning-based methods. Gurobi solves MIS as an integer linear program, while KaMIS is a heuristics solver for MIS. The four learning-based methods can be divided into the reinforcement learning (RL) category, i.e., LwD [1]) and DIMES [92], and the supervised learning (SL) category, i.e., Intel [70] and DGL [8]. 

25 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0026-00.png)


Figure 10: Qualitative illustration of how diffusion schedules affect the generation quality of **discrete** diffusion models. The results are reported without any post-processing. **Discrete** DIFUSCO with `linear` schedule (first row) and `cosine` schedule (second row) with 20 diffusion steps are shown. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0026-02.png)


Figure 11: Qualitative illustration of discrete DIFUSCO on TSP-50, TSP-100 and TSP-500 with 50 diffusion steps and `cosine` schedule. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization_images/nips_2023_DIFUSCO_Graph-based_Diffusion_Solvers_for_Combinatorial_Optimization.pdf-0026-04.png)


Figure 12: Success (left) and failure (right) examples on TSP-100, where the latter fails to form a single tour that visits each node exactly once. The results are reported without any post-processing. 

26 

