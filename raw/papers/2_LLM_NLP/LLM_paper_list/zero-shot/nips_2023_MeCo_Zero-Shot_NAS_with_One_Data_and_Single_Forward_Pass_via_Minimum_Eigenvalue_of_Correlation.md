## **MeCo: Zero-Shot NAS with One Data and Single Forward Pass via Minimum Eigenvalue of Correlation** 

**Tangyu Jiang** _[†]_ School of Artificial Intelligence Beijing Normal University `jty@mail.bnu.edu.cn` 

**Haodi Wang** _[†]_ School of Artificial Intelligence Beijing Normal University `whd@mail.bnu.edu.cn` 

**Rongfang Bie** _[∗]_ School of Artificial Intelligence Beijing Normal University `rfbie@bnu.edu.cn` 

## **Abstract** 

Neural Architecture Search (NAS) is a promising paradigm in automatic architecture engineering. Zero-shot NAS can evaluate the network without training via some specific metrics called zero-cost proxies. Though effective, the existing zero-cost proxies either invoke at least one backpropagation or depend highly on the data and labels. To alleviate the above issues, in this paper, we first reveal how the Pearson correlation matrix of the feature maps impacts the convergence rate and the generalization capacity of an over-parameterized neural network. Enlightened by the theoretical analysis, we propose a novel zero-cost proxy called MeCo, which requires only one random data for a single forward pass. We further propose an optimization approach MeCoopt to improve the performance of our method. We design comprehensive experiments and extensively evaluate MeCo on multiple popular benchmarks. MeCo achieves the highest correlation with the ground truth (e.g., 0.89 on NATS-Bench-TSS with CIFAR-10) among all the state-of-the-art proxies, which is also fully independent of the data and labels. Moreover, we integrate MeCo with the existing generation method to comprise a complete NAS. The experimental results illustrate that MeCo-based NAS can select the architecture with the highest accuracy and a low search cost. For instance, the best network searched by MeCo-based NAS achieves 97.31% on CIFAR-10, which is 0.04% higher than the baselines under the same settings. Our code is available at https://github.com/HamsterMimi/MeCo. 

## **1 Introduction** 

Deep Neural Networks (DNNs) have been ubiquitously adopted in various fields due to their ability in hierarchical feature extraction [1–5]. It is a common consensus that the architecture of DNN is essential in model performance. Unfortunately, the manual trial-and-error method is non-scalable and inefficient, making it infeasible to find the globally optimal structure. Thanks to the development of the Neural Architecture Search (NAS), it is possible to automatically select the network architecture with the best performance. 

- *Corresponding author. 

- †These authors contributed equally to this work. 

37th Conference on Neural Information Processing Systems (NeurIPS 2023). 

There are abundant works proposed to improve the performance of NAS. The most fundamental observation is to fully or partially train the candidate networks with specific data and select the network with the highest accuracy. Though effective, these multi-shot NAS methods are inevitably resource-consuming, thus, are non-trivial to be adopted in real-world applications. To address this issue, the one-shot NAS has been proposed in which a supernet is designed and only requires one-time full training [6]. In 2020, Mellor et al. [7] proposed the zero-shot NAS and realized an efficient approach in which a zero-cost proxy is proposed to evaluate the networks without training. 

Despite the effectiveness of the existing zero-cost proxies, there are some critical issues that remain to be solved. (i) Almost all current proxies are established from the network gradients. As a consequence, computing these proxies requires at least one backpropagation, which is resource-consuming. (ii) The majority of the currently used proxies rely highly on the input data and labels. The reliance on the input samples underestimates the evaluation of the network’s intrinsic characteristics, and the incorrect labels (which are commonly occurred in real-world datasets) may obfuscate the ranking results. (iii) Although various zero-cost proxies have been proposed, the performance of those proxies can still be promoted in network evaluation. 

**Objectives.** The main objective of this paper is to design a novel zero-cost proxy with better performance than the current methods. The proposed proxy should be easily computed by a small number of data without backpropagation. Furthermore, to estimate the intrinsic trait of the architectures and avoid the label influence, our proxy should be label-independent. 

**Our Contributions.** To this end, in this paper, we craft a novel zero-cost proxy for zero-shot NAS from a new perspective. The proposed proxy, called MeCo, outperforms the State-Of-The-Art (SOTA) zero-cost proxies in NAS and only requires one data sample for a single forward pass. Specifically, we harness a novel observation on the multi-channel convolutional layers and view them as the “multi-sample fully-connected layers with constraints”. This insight permits MeCo to be computed by only one data. Moreover, we rigorously analyze how the Pearson correlation matrix of the feature maps impacts the training convergence rate and the generalization capacity of the networks, which theoretically proves the effectiveness of MeCo. Unlike the existing methods [7–9], MeCo is established upon each layer of the feature maps and hence requires only a single forward pass. Based on our design, we further propose an optimization approach MeCoopt to promote the performance of our method. The experimental results show that both MeCo and MeCoopt achieve a higher correlation with the network ground truth than the SOTA methods. For example, the Spearman correlation coefficients of MeCo on NATS-Bench-TSS with CIFAR-10 and CIFAR-100 are 0.89 and 0.88, respectively, which are 0.09 and 0.08 higher than the best baseline method. Our MeCo-based zero-shot NAS also acquires the highest accuracy among all the existing approaches, e.g., MeCobased NAS selects the network with 97.31% accuracy, which is 0.04% higher than the SOTA work. In all, our major contributions are as follows. 

- We theoretically analyze how the Pearson correlation matrix of the feature maps impact the training convergence rate and the generalization capacity of the networks, based on the characteristics of the multi-channel convolutional layers. 

- We propose MeCo , a novel zero-cost proxy for zero-shot NAS that achieves better performance than SOTA methods. Through systematic analysis, we further proposed MeCoopt as an optimization approach. Our proxy is fully training-free and data-independent, which requires only one data sample with a single forward pass. 

- We rigorously implement our MeCo and extensively design the experiments to evaluate its performance on several popular benchmarks (e.g., NAS-Bench-101, NATS-Bench-TSS, NATS-Bench-SSS, NAS-Bench-301, and Transbench-101) with diverse datasets and random data samples. The experimental results show that our zero-cost proxy outperforms the existing methods, and MeCo-based NAS can select the network with the highest accuracy. 

## **2 Related work** 

## **2.1 Zero-shot NAS and zero-cost proxy** 

Zero-shot NAS uses specific metrics called zero-cost proxies to evaluate the networks. Unlike the traditional multi-shot and one-shot strategies [10, 6, 11–19], zero-shot NAS eliminates the training procedure thus significantly improves the efficiency. Various studies have been brought forward to 

2 

promote the quality of the zero-cost proxies, and most of them are based on the gradients. Mellor et al. [7] firstly evaluated the performance of the initialized networks using the activation overlaps between data points. Abdelfattah et al. [8] proposed a series of parameter-pruning based proxies including _snip_ [20], _grasp_ [21], _synflow_ [22], and _fisher_ [23]. Recently, Li et al. [9] proposed _ZiCo_ , which firstly works better than the parameter amounts. These approaches need at least one backpropagation and rely on the data labels to compute the proxies. Another line of work [24–26] is based on the theory of deep learning such as NTK, which is independent of the labels yet still needs backpropagation. For example, Chen et al. [24] utilized the condition number of NTK to evaluate the trainability of the networks. Finally, there are a few works that do not require backpropagation or data labels. For instance, Lin et al. [27] proposed Zen-Score, in which they design an efficient zero-cost proxy with Gaussian random inputs. However, Zen-Score requires multiple forward passes while our MeCo only needs a single forward pass. 

## **2.2 Over-parameterized networks** 

Over-parameterized networks have received a lot of attention due to their outstanding effect and ease of optimization. Jacot et al. [28] demonstrated that the training dynamic of an infinite-width network follows a kernel called NTK [29, 30]. For finite-width networks, the training dynamic can be illustrated by a gram matrix [31, 32]. Further, Du et al. [31] proved that the loss of an overparameterized network could converge to a global minimum [33–35], and the training convergence rate could be characterized by the minimum eigenvalue of the gram matrix. Some works also discuss the optimization of the training convergence rate [36] and the topological properties of DNNs [37]. 

## **3** MeCo **: minimum eigenvalue of correlation on feature maps** 

## **3.1 Preliminaries** 

**Notations.** We define [ _n_ ] = _{_ 1 _,_ 2 _, . . . , n}_ . The lower and uppercase bold font represent vectors and matrices, respectively, e.g., **x** is a vector with entry _xi_ , and **M** is a matrix with entry [ **M** ] _ij_ . The minimum eigenvalue of **M** is denoted as _λ_ min( **M** ). We define **A** _◦_ **B** as Hadamard product between two same-sized matrices **A** and **B** . **1** _m×n_ represents a _m × n_ matrix filled by ones and _∥· ∥_ 2 is used to represent the _l_ 2 norm of a vector. _N_ ( **0** _,_ **I** ) and _U {S}_ represent the standard Gaussian distribution and uniform distribution over a set _S_ , respectively. We denote by **X** = _{_ ( **x** _i, yi_ ) _|_ **x** _i ∈_ R _[d][×]_[1] _, yi ∈_ R _, i ∈_ [ _n_ ] _}_ the training set, where **x** _i_ and _yi_ represent the _i_ -th data and label. I _{·}_ is defined as the indicator function that demonstrates the event occurrence, such that for event _A_ , I _{A}_ = 1 if and only if _A_ happened, otherwise it equals to 0. 

For input **x** _∈_ R _[d][×]_[1] , weight vector **w** _∈_ R _[d][×]_[1] in the weight matrix **W** _∈_ R _[d][×][m]_ , and output weight **a** _∈_ R _[m][×]_[1] , we denote _f_ ( **W** _,_ **a** _,_ **x** ) as a neural network with a single hidden layer such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0003-07.png)


where _σ_ is the activation function. In this paper, we mainly consider the ReLU function due to its effectiveness, i.e., _σ_ ( _z_ ) = _z_ I _{z >_ 0 _}_ . Given a training set **X** , the optimization goal is to minimize the empirical risk loss function 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0003-09.png)


We follow the definitions in [31] and define the matrices **H** ( _t_ ) and **H** _[∞]_ as follows. 

**Definition 1** (Gram Matrix) **.** _For a neural network with a single hidden layer, the gram matrix_ **H** ( _t_ ) _∈_ R _[n][×][n] induced by the ReLU activation function on a training set_ **X** := _{_ ( **x** _i, yi_ ) _}[n] i_ =1 _[with] entry_ [ **H** ( _t_ )] _ij is defined as:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0003-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0003-13.png)


_We denote λ_ 0 := _λ_ min( **H** _[∞]_ ) _._ 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Zero Operation<br>Full Connection<br>Sum operation<br>Input Filters Feature map Output Inputs Hidden nodes Output<br>**----- End of picture text -----**<br>


Figure 1: Conversion between the multi-channel convolution and multi-sample fully-connected operation. The input size is 3 _× w × h_ and each filter size is 3 _×_ 3 _×_ 3. Each input channel can be flattened with size _d ×_ 1, _d_ = _w × h_ . We collect the three flattened samples to obtain the output with constrained fully-connected operations (dot lines for zero operations and solid lines for full connection). The final output is computed by a sum operation (dashed lines) from the hidden nodes. 

**Remark 1.** _Gram matrix_ **H** ( _t_ ) _reflects the prediction dynamic at the t-th iteration of the network training. Du et al. [31] proved that for all t ≥_ 0 _, ∥_ **H** ( _t_ ) _−_ **H** _[∞] ∥_ 2 _→_ 0 _as m →∞. Moreover, the gram matrix_ **H** _[∞] has the following key property: if_ **x** _i_ ∦ **x** _j, ∀i ̸_ = _j, then λ_ 0 _>_ 0 _. The proof of this property can be found in Theorem 3.1 in [31]._ 

**Definition 2** (Pearson Correlation Matrix) **.** _P_ ( **X** ) _is a Pearson correlation matrix with the_ ( _i, j_ ) _-th entry_ [ _P_ ( **X** )] _ij as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0004-04.png)


_where µ_ **x** _and σ_ **x** _are the mean and standard deviation of_ **x** _, respectively._ 

**Remark 2.** _Pearson correlation matrix P_ ( **X** ) _is closely related to_ **H** ( _t_ ) _. For instance, suppose_ **x** 1 _,_ **x** 2 _∼ N_ ( **0** _,_ **I** ) _,_ **x** _i ∈_ R _[d][×]_[1] _and_ **w** _r[T]_[(] _[t]_[)] **[x]** _[i][≥]_[0] _[,][for][r][∈]_[[] _[m]_[]] _[and][i][∈]_[[2]] _[,][then]_[[] **[H]**[(] _[t]_[)]][12][=] ( _d −_ 1)[ _P_ ( **X** )]12 _. We will use this characteristic to approximate the training convergence rate and generation capacity of the networks when designing our proxy._ 

## **3.2 The construction of** MeCo 

To present the theoretical analysis, we first approximate the Convolutional Neural Network (CNN) to an over-parameterized Neural Network (NN) and then craft our proxy. 

## **3.2.1 Back to convolution** 

The convolutional layers are the fundamental parts of CNN. Suppose we have the input **x** in _∈_ R _[c]_[in] _[×][w][×][h]_ and the filter **w** _∈_ R _[c]_[in] _[×][k][×][k]_ . We set the stride size to one and retain the size of the feature maps via zero padding, then the convolutional layer can be formally expressed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0004-11.png)


where **x** _[c]_ in[(] _[resp.]_ **[w]** _[c]_[) represents the] _[ c]_[-th channel of] **[ x]**[in][ (] _[resp.]_ **[w]**[),] _[ p]_[ = (] _[k][ −]_[1)] _[/]_[2][,] _[ i][ ∈]_[[] _[w]_[]][,] _[ j][∈]_[[] _[h]_[]][.][In] this paper, we observe that **x** out and **x** _[c]_ in[can be flattened to one-dimensional vectors][ ˜] **[x]**[out] _[∈]_[R] _[d][×]_[1][ and] [˜ **x** _[c]_ in[]] _[ ∈]_[R] _[d][×]_[1][, where] _[ c][ ∈]_[[] _[c]_[in][]][,] _[ d]_[ =] _[ w][ ×][ h]_[.][More concretely, we have the following transformations:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0004-13.png)


4 

where _dh_ = _c_ in _× d_ , **W** _∈_ R _[d][×][d][h]_ is the weight matrix. Hence, the convolutional layer in CNN is equivalent to a fully-connected layer with constraints, i.e., shown in Equation 8. To further simplify theconsequence, we obtain a multi-sample fully-connected network in which each flattened channelproblem, we relax the second constraint to **B** _c_ = **1** _dh×d_ and ignore the last constraint. As **x** ˜ _[c]_ ina is regarded as an independent data sample, and the total number of samples is _c_ in. To better illustrate our insight, we present an example of the conversion procedure in Figure 1. 

## **3.2.2 Over-parameterized neural networks** 

The over-parameterized NNs are competitive in hierarchical feature extraction due to a large number of parameters. One of the typical architectures is the networks with wide hidden layers, which is proved to be tractable in training [38]. In the previous subsection, we convert a multi-channel convolutional layer to a multi-sample fully-connected layer with constraints. We further argue that if the number of hidden nodes in the transformed fully-connected layer is large enough, then it can be viewed as an over-parameterized NN layer. Therefore, the characteristics of the overparameterized NN can be transferred to CNN. To this end, we present the following theorem [31] about the convergence rate of the fully-connected NN with a single hidden layer as follows. 

**Theorem 1.** _If gram matrix_ **H** _[∞] ≻_ 0 _, ∥_ **x** _i∥_ 2 = 1 _, |yi| < C for some constant C and i ∈_ [ _n_ ] _, hidden nodes m_ = Ω � _λn_[4] 0 _[δ]_[6][3] � _, and i.i.d. initialize_ **w** _r ∼ N_ ( **0** _,_ **I** ) _, ar ∼ U {_ [ _−_ 1 _,_ 1] _} for r ∈_ [ _m_ ] _, then with probability at least_ 1 _− δ over the initialization, the following inequality holds: ∥f_ ( **W** ( _t_ ) _,_ **a** _,_ **X** ) _−_ **y** _∥_ 2[2] _[≤]_[exp (] _[−][λ]_[0] _[t]_[)] _[∥][f]_[(] **[W]**[(0)] _[,]_ **[ a]** _[,]_ **[ X]**[)] _[ −]_ **[y]** _[∥]_ 2[2] (9) 

**Remark 3.** _This inequality shows that λ_ 0 _positively affects the training convergence rate of the network. We further point out that the convergence rate of the network is independent of the label_ **y** _, which makes it possible to measure the network performance with only a single forward pass. Based on this observation, we establish our proxy on the intermediate feature maps to measure the performance of the network._ 

Based on Theorem 1 and Remark 2, we further analyze the relationship between the minimum eigenvalue of the matrix _P_ ( **X** ) and the training convergence rate of the over-parameterized network. We summarize our results in the following theorem: 

**Theorem 2.** _Suppose f is an NN with a single hidden layer and ReLU activation function. Assume_ **X** _∈_ R _[d][×][n] ,_ **w** (0) _∼ N_ ( **0** _,_ **I** ) _, P_ ( **X** ) _≻_ 0 _, p_ 0 := _λ_ min( _P_ ( **X** )) _, and hidden nodes m_ = Ω � _nλ_[6][4] 0 _d[δ]_[3][2] � _, then the following formula holds with probability at least_ 1 _− δ over the initialization_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0005-07.png)


_where c is a constant depending on m, and d._ 

_Proof sketch._ The key of the proof is to find the relationship between _λ_ 0 and _p_ 0 by Hoeffding’s inequality and Weyl inequalities [39]. We provide the full proof in Appendix A.1. Theorem 2 shows that _p_ 0 can reflect the convergence rate of the network to a certain extent. The advantage of _P_ ( **X** ) over **H** ( _t_ ) is that its computation only depends on the data. Therefore, it is feasible to estimate the training convergence rate of the network only through the feature maps. 

Other than the convergence rate, we also analyze the relationship between _p_ 0 and the _generalization capacity_ of the over-parameterized NN. We present the results in the following theorem: 

**Theorem 3.** _For an over-parameterized neural network with the loss on the testing set as L_ ( **W** ) _. Let_ **y** = ( _y_ 1 _, ..., yN_ ) _[T] , and γ be the step of SGD, γ_ = _κC_ 1� **y** _[T]_ ( **H** _[∞]_ ) _[−]_[1] **y** _/_ ( _m√N_ ) _for some small enough absolute constant κ. Under the assumption of Theorem 2, for any δ ∈_ (0 _, e[−]_[1] ] _, there exists m[∗]_ ( _δ, N, λ_ 0) _, such that if m ≥ m[∗] , then with probability at least_ 1 _− δ, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0005-12.png)


_where C, C[′] , δ are constants._ 

_Proof sketch._ The proof of the above theorem derives from the Corollary 3.10 of [40] and Section D.2 of [26]. We present the detailed proof in Appendix A.2. 

5 

## **3.2.3 New zero-cost proxy:** MeCo 

The above theorem and analysis indicate the following key insights: 

- A multi-channel CNN layer can be viewed as a multi-sample over-parameterized NN layer. 

- The minimum eigenvalue of _P_ ( **X** ) positively affects the training convergence rate and the generalization capacity of an NN, which is independent of the labels. 

Enlightened by these observations, we harness the Minimum eigenvalue of the Pearson Correlation matrix upon each layer of the feature maps to craft our novel zero-cost proxy called MeCo, such that **Definition 3** (MeCo) **.** _Assume the NN f_ ( _·_ ; _θ_ ) _has a total of D layers, then_ MeCo _is defined as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0006-05.png)


_where f[i]_ ( **X** ; _θ_ ) _represents the i-th feature map with the initialized parameters θ on dataset_ **X** _._ 

**Remark 4.** _Note that the Pearson correlation matrix contains the process of data normalization, thus our_ MeCo _eliminates the deviation caused by the data, which is more conducive to discovering the essence of the network architectures._ 

## **4 Experiments and evaluations** 

We evaluate MeCo from three aspects: (i) The correlation with the ground truth and comparison with the baseline proxies; (ii) The dependency on data/labels; (iii) Performance of MeCo-based NAS scheme. We first describe the experimental configurations and then give the detailed performance. 

## **4.1 Experimental configurations** 

**Hardware.** We fully implement our MeCo in Python. We evaluate MeCo on a desktop with an Inter Core i7-12700F CPU and GeForce RTX 3090. 

**Benchmarks.** We use several popular benchmarks in NAS. (i) _NATS-Bench-TSS_ [41, 42] contains 15,625 CNN architectures with different topologies. We demonstrate the performance of the zero-cost proxied on NATS-Bench-TSS with CIFAR-10 [43], CIFAR-100 [43], ImageNet16-120[44], and three extra datasets [45] in Appendix E.1. (ii) _NATS-Bench-SSS_ [42] contains 32,768 CNN architectures with the same topology but with different numbers of channels. We show the results on NATS-BenchSSS with CIFAR-10, CIFAR-100, and ImageNet16-120. (iii) _NAS-Bench-301_ [46] is a surrogate NAS benchmark for the DARTS search space. We illustrate the results with CIFAR-10 and the extra three datasets in Appendix E.1. (iv) _TransBench-101_ [47] is a benchmark composed of two subsets, i.e., TransBench-101-Micro and TransBench-101-Macro. The former is evaluated with 10 datasets and the latter with 7 datasets. We present these results in Appendix E.3. (v) _NAS-Bench-101_ [48] includes 423,624 cell-based CNN architectures, which are trained and evaluated on the CIFAR-10 dataset and shown in Appendix E.2. (vi) _AutoFormer_ [49] and _MobileNet OFA_ [50]. The description and results are shown in Appendix E.4. 

**Baselines.** The baselines for zero-cost proxies include the number of parameters #Param, grasp [21], fisher [23], snip [20], synflow [22], jacov [7], grad_norm [8], NTK [24], zen [27], NASWOT [7], KNAS [51], NASI [52], GradSign [53], and ZiCo [9]. Specific forms of these proxies are provided in Appendix B. We adopt the Spearman rank correlation coefficient _ρ_ between zero-cost proxies and the dataset ground truth to measure their performance. 

## **4.2 Performance of our** MeCo **and comparisons** 

We evaluate _ρ_ between MeCo and the ground truth, and compare it with the baseline zero-cost proxies on various benchmarks. We present the results on NATS-Bench-TSS and NATS-Bench-SSS in Table 1, and the results on NAS-Bench-301 are provided in Table 2. We run MeCo for ten times with different inputs and present the mean and standard deviation. 

**NATS-Bench-TSS.** We run our MeCo and all baseline proxies on 15,625 networks of NATS-BenchTSS with three public datasets. As we can see in Table 1, MeCo achieves 0.894 _±_ 0.003, 0.883 _±_ 0.005, 

6 

Table 1: Spearman correlation coefficients _ρ_ of proxies on NATS-Bench-TSS and NATS-Bench-SSS 

|Approach|NATS-Bench-TSS|NATS-Bench-SSS<br>CIFAT-10<br>CIFAR-100<br>ImageNet16|
|---|---|---|
||CIFAT-10<br>CIFAR-100<br>ImageNet16||
|grasp<br>fsher<br>grad_norm<br>snip<br>synfow<br>#Param<br>NWOT<br>jacov<br>NTK<br>zen<br>KNAS<br>NASI<br>GradSign<br>ZiCo|0.39<br>0.46<br>0.45<br>0.40<br>0.46<br>0.42<br>0.42<br>0.49<br>0.47<br>0.43<br>0.49<br>0.48<br>0.74<br>0.76<br>0.75<br>0.72<br>0.73<br>0.69<br>0.77<br>0.80<br>0.77<br>0.73<br>0.70<br>0.70<br>0.76<br>0.75<br>0.72<br>0.38<br>0.36<br>0.40<br>0.20<br>0.35<br>0.42<br>0.44<br>0.43<br>0.63<br>0.77<br>0.79<br>0.78<br>0.80<br>0.81<br>0.79|-0.13<br>0.01<br>0.42<br>0.44<br>0.55<br>0.47<br>0.51<br>0.49<br>0.67<br>0.59<br>0.62<br>0.76<br>**0.81**<br>0.80<br>0.57<br>0.72<br>0.73<br>0.84<br>0.45<br>0.43<br>0.42<br>0.30<br>0.13<br>0.30<br>0.34<br>0.29<br>0.28<br>0.69<br>0.71<br>0.87<br>0.25<br>0.12<br>0.32<br>0.17<br>0.04<br>0.20<br>0.21<br>0.16<br>0.04<br>0.73<br>0.75<br>**0.88**|
|MeCo**(Ours)**<br>MeCoopt**(Ours)**|**0.894**_±_**0.003**<br>**0.883**_±_**0.005**<br>**0.845**_±_**0.004**<br>**0.901**_±_**0.002**<br>**0.890**_±_**0.003**<br>**0.850**_±_**0.003**|-0.79_±_0.01<br>**-0.87**_±_**0.01**<br>-0.86_±_0.02<br>**0.89**_±_**0.002**<br>**0.83**_±_**0.004**<br>**0.89**_±_**0.003**|



and 0.845 _±_ 0.004 correlation on CIFAR-10, CIFAR-100, and ImageNet16-120, respectively. The variation shows that our method is stable and independent of the input samples. The Spearman correlation coefficients are significantly higher than the baselines, e.g., the SOTA proxy ZiCo [9] only achieves 0.80, 0.81, 0.79 on three datasets, which is 0.06 to 0.09 lower than ours. Note that ZiCo is the first proxy that is proved to be consistently better than the naive proxy #Param and requires a batch of input data. On the other hand, we surpass all the previous works and only use one data sample. The high correlation coefficients demonstrate the effectiveness of MeCo, which is consistent with the theoretical analysis. To make our results more intuitive, we sample 2,000 networks randomly and present the relationship between MeCo and the ground truth on three public datasets. The results are shown in Figure 4(a) in the Appendix E, where the trend in the scatter charts effectively proves that our zero-cost proxy has a highly strong correlation with the architecture quality. 

**NATS-Bench-SSS.** The results of MeCo on 32,768 networks of NATS-Bench-SSS are summarized in Table 1. Among all the datasets, MeCo shows the highest correlation with the ground truth on CIFAR100, i.e., _ρ_ = _−_ 0 _._ 87 _±_ 0 _._ 01, which is 0.07 higher than the best baseline synflow. MeCo also achieves stable and competitive results on CIFAR-10 and ImageNet16-120 with one data as input. We find that MeCo is highly _negatively_ correlated with the test accuracy, which is somehow counterintuitive. In fact, MeCo is sensitive to the number of channels (#channels) that cause this phenomenon. Specifically, the channels are viewed as the data samples as described in Section 3.2.1, This leads to the fact that the more channels contained, the larger the Pearson matrix, which will lead to **smaller** _p_ 0. On the other hand, larger #channels will normally lead to higher accuracy of the architectures (yet lower 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0007-04.png)


**----- Start of picture text -----**<br>
Channels v.s. Eigenvalue<br>1<br>0 . 8<br>0 . 6<br>0 . 4<br>0 . 2<br>0 200 400<br>Number of Channels<br>Eigenvalue<br>Minimum<br>**----- End of picture text -----**<br>


Figure 2: The influence of different channel numbers of data on the minimum eigenvalue. 

convergence rate). Thus, the variation of #channels in NATS-Bench-SSS(chosen from 8 to 64) leads to a negative correlation on this benchmark. In NATS-Bench-TSS, however, all the networks share the same #channels at each stage, thus MeCo does not show a negative correlation. To verify this, we generate random data with different numbers of channels and calculated _p_ 0 respectively, and demonstrate the results in Figure 2. We also sample 2,000 networks randomly and present the correlation of MeCo v.s. test accuracy in Figure 4(b) in the Appendix E. 

7 

**NATS-Bench-301** . We demonstrate the results of MeCo and the baseline proxies on NATS-Bench301 with CIFAR-10 in Table 2. It can be shown from the results that MeCo and MeCoopt also achieve the highest correlation with the testing accuracy on NATS-Bench-301, which is 0.04-0.05 higher than the SOTA approach ZiCo. More experimental results of MeCo can be found in Appendix E 

Table 2: _ρ_ between zero-cost proxies and test accuracy on NAS-Bench-301 with CIFAR-10 

|Table 2: _ρ_between zer|o-cost proxies and test accuracy on NAS-Bench-301 with CIFAR-10|
|---|---|
|Dataset<br>MeCo<br>MeCoopt|Baselines|
||grasp<br>fsher<br>grad_norm<br>snip<br>synfow<br>l2_norm<br>#Param<br>zen<br>jacov<br>nwot<br>ZiCo|
|CIFAR-10<br>**0.7**_±_**0.01**<br>**0.71**_±_**0.01**|0.34<br>-0.28<br>-0.04<br>-0.05<br>0.18<br>0.45<br>0.46<br>0.43<br>-0.04<br>0.47<br>0.66|



## **4.3 Dependency on data** 

MeCo is designed to be fully data-independent due to the characteristics of Pearson correlation matrix (Theorem 2). To further illustrate this attribute, we generate a random dataset and use it as the input of MeCo and NTK [24], which is also independent of the data and labels. We compute the Spearman correlation coefficients between MeCo and the test accuracy on three public datasets, and compare them with the baseline approach. All the results for MeCo is ran ten times with random inputs. 

More concretely, the random data is generated following a Gaussian distribution and fed into the initialized network. Note that the baseline method requires a batch of random data as input, while MeCo only uses one. We present the results for NATS-Bench-TSS and NATS-Bench-SSS in Table 3. 

The results show that even if we use random data as the inputs, MeCo is still able to maintain a high correlation with the corresponding ground truth. For example, _ρ_ = 0 _._ 87 on NATS-BenchTSS with CIFAR-10 and _ρ_ = _−_ 0 _._ 93 on NATSBench-SSS with ImageNet16-120, which are significantly higher than the data-independent proxy NTK. Note that most existing proxies rely on the gradients and backpropagation. Hence they cannot utilize the random input data for architecture evaluation, such as ZiCo and jacov. 

Table 3: Spearman correlation coefficients of MeCo and NTK with random data on NATSBench-TSS and NATS-Bench-SSS 

|Dataset|NTAS-Bench-TSS|NATS-Bench-SSS|
|---|---|---|
||MeCo<br>NTK[24]|MeCo<br>NTK[24|
|**CIFAR-10**<br>**CIFAR-100**<br>**ImageNet16-120**|0.87_±_0.001<br>0.78<br>0.85_±_0.001<br>0.79<br>0.83_±_0.001<br>0.76|-0.87_±_0.000<br>0.23<br>-0.88_±_0.001<br>0.16<br>-0.94_±_0.000<br>0.20|



## **4.4 NAS with** MeCo 

We integrate our MeCo with the existing search space and generation strategy to comprise a complete NAS method. Specifically, we use NATSBench-TSS, DARTS-CNN [6], and MobileNet V3 [54] as the search spaces to demonstrate the performance. We use the Zero-Cost-PT [55] as the generation strategy. The algorithm of MeCobased NAS and the results on MobileNet V3 are presented in Appendix C and F.3, respectively. 

**NATS-Bench-TSS.** We directly evaluate all networks in NATS-Bench-TSS using MeCo and baseline proxies. The average test accuracy of the Top-10 networks on three datasets is summarized in Table 12 in Appendix F.1. As shown in the table, the average test accuracy on CIFAR-10 of MeCo is 0 _._ 08% higher than the best baseline 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0008-12.png)


**----- Start of picture text -----**<br>
NATS-Bench-TSS NATS-Bench-SSS<br>0 . 89 − 0 . 5<br>0 . 88 − 0 . 6<br>0 . 87<br>0 . 86 − 0 . 7<br>0 . 85 − 0 . 8<br>0 . 84 − 0 . 9<br>0 . 83<br>16 32 64 128 256 16 32 64 128 256<br>data size data size<br>CIFAR-10; CIFAR-100; ImageNet16-120<br>ρ ρ<br>**----- End of picture text -----**<br>


Figure 3: Spearman correlation coefficients of MeCo with different data size on NATS-BenchTSS and NATS-Bench-SSS, respectively. 

ZiCo. Moreover, we further search the networks using Zero-Cost-PT with different proxies and run the search algorithm three times. The networks we searched have the highest average test accuracy on three public datasets, that is, 93.76%, 71.11%, and 41.44% on CIFAR-10, CIFAR-100, and ImageNet16-120, respectively. The results are provided in Table 13 in Appendix F.1. 

**DARTS-CNN.** We compare our MeCo-based NAS with multi-shot, one-shot, and zero-shot NAS schemes in DARTS-CNN spaces. All the NAS schemes are executed on CIFAR-10 and CIFAR-100 

8 

datasets, respectively, to compare the Top-1 errors and the search cost. For zero-shot NAS, we use Zero-Cost-PT as the generation strategy due to its effectiveness. The results on CIFAR-10 are summarized in Table 4, where we denote multi-shot, one-shot, and zero-shot as MS, OS, and ZS, respectively. We present the CIFAR-100 results in Appendix F.2. The results demonstrate that our MeCo-based NAS outperforms the existing baseline methods. More concretely, for zero-shot methods, MeCo outperforms all the baselines for 0.04% to 0.43% on Top-1 error with a similar level of search cost under the same generation strategy. For training-based methods, MeCo-based NAS achieves competitive accuracy but with much less search cost. For instance, our approach requires only 0.08 GPU days and obtains 97.31% accuracy, while DARTS-PT costs 10 _×_ GPU days than ours with 0.08% higher accuracy. 

Table 4: Comparisons of MeCo-based NAS with baselines using DARTS-CNN and CIFAR-10 

|Approach|Test Error<br>(%)|Search Cost<br>(GPU Days)|Params<br>(M)|Method|
|---|---|---|---|---|
|AmoebaNet-A [56]|3_._34_±_0_._06|3150|3.2|MS|
|PNAS [57]|3_._41_±_0_._09|225|3.2|MS|
|DARTS (1st) [6]|3_._00_±_0_._14|1_._5|3.3|OS|
|DARTS-PT[58]|2_._61_±_0_._08|0.8|3.0|OS|
|SDARTS-RS [59]|2_._67_±_0_._03|0.4|3.4|OS|
|SGAS (Cri.1)[60]|2_._66_±_0_._24|0_._8|3.7|OS|
|Eigen-NAS[26]|7_._4|-|-|ZS|
|TE-NAS [24]|2_._63_±_0_._064|0_._05|3.8|ZS|
|Zero-Cost-PTsynfow[22]|2_._96_±_0_._11|0.03|5.1|ZS|
|Zero-Cost-PTfsher[23]|3_._12_±_0_._16|0.05|2.5|ZS|
|Zero-Cost-PTgrasp [21]|2_._73_±_0_._10|0.1|3.3|ZS|
|Zero-Cost-PTjacov[7]|2_._88_±_0_._15|0.04|3.5|ZS|
|Zero-Cost-PTsnip[20]|2_._90_±_0_._03|0.04|4.0|ZS|
|Zero-Cost-PTNTK[24]|2_._89_±_0_._09|0.21|4.1|ZS|
|Zero-Cost-PTZiCo[9]|2_._80_±_0_._03|0.04|5.1|ZS|
|**Zero-Cost-PT**MeCo**(Ours)**|**2.69**_±_**0.05**|**0.08**|4.2|ZS|



We remark that the experimental results in this section demonstrate the following properties of MeCo: (i) **High correlation with the ground truth;** (ii) **Requies only one data without labels for a single forward pass.** That is because MeCo is established upon the observations over the multi-channel CNN layers. Moreover, the minimum eigenvalue of the Pearson correlation matrix is independent of the data labels, and thus we exert MeCo on the feature maps instead of the gradients. 

## **5 Discussion and Optimization** 

## **5.1 Optimization method** MeCoopt 

In the previous sections, we illustrated the theoretical basis of MeCo and evaluated our proposed proxy on various benchmarks. However, as described in Section 4.2, the negative correlation on NATS-Bench-SSS and some other tasks (e.g., Transbench-101 in the appendix) reflect that MeCo is highly sensitive to the number of channels. Although this phenomenon does not contradict the theoretical results or undermine the effectiveness of MeCo, it might be problematic when _p_ 0 = 0. To this end, we present an optimization method on top of MeCo to alleviate the channel-sensitive trait. 

Specifically, for convolutional layers, _p_ 0 is strictly greater than zero if for _∀i_ = _j_ , _xi_ ∦ _xj_ , and _c < w × h_ , where _c_ is the number of channels, _w, h_ are the size of the inputs. Thus, in real-world applications, MeCo might lose efficacy during the down-sampling procedure. To address this issue, instead of flattening all the channels of the feature map as described in Section 3.2.1, we randomly sample a _fixed_ number of channels and flatten them as matrix _P[′]_ . We then compute the final result by multiplying the minimum eigenvalue of **P** _[′]_ with a channel weight, such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0009-08.png)


9 

where _c_[(] _[i]_[)] is the number of channels in the _i_ -th layer, and _n_ is the fixed sampling numbers. The high-level idea of this optimization is to limit the dimension of the Pearson correlation matrix by constraining #channels. Instead of computing upon a large matrix, we calculate the minimum eigenvalue upon a fixed-sized matrix and then enlarge it with corresponding constants. Note that the bonus of this optimization is that the time cost can be controlled because the matrix dimension is significantly reduced compared to the original one. 

To show the effectiveness of MeCoopt, we calculate the Spearman correlation coefficients of MeCoopt on NATS-Bench-TSS, NATS-Bench-SSS (Table 1), NATS-Bench-301 (Table 2 and Table 8), and Transbench-101 (Table 10 and Table 11). All the experiments are conducted under _n_ = 8. The results demonstrate that MeCoopt effectively solves the negative correlation issue on multiple benchmarks, e.g., NATS-Bench-SSS and TransBench-101. Meanwhile, the majority of the positive correlations are promoted up to 0.23, and the remaining results are not impacted. 

## **5.2 Ablation study** 

**Sample size** . We consider the effect of the sample size on the performance of MeCo. We select the size of data from _{_ 16 _,_ 32 _,_ 64 _,_ 128 _,_ 256 _}_ , respectively, and calculate the Spearman rank correlation coefficient _ρ_ between MeCo and the test accuracy of the networks on three public datasets. The results are summarized in Figure 3. For NATS-Bench-TSS, _ρ_ increases rapidly before size 64 and grows slowly after that. The maximum value is acquired at size 128, which is slightly higher than size 64. On the other hand, the overall fluctuation is relatively large on NATS-Bench-SSS. The Spearman correlation coefficients are low when the size is smaller than 32, and reach higher at 32. Then _ρ_ reduces as the sample size grows larger. The ablation results show that the most appropriate size of random data is around 32 and 64, which is also adopted in our settings. 

**Sample numbers** . We explore the performance of MeCo regarding the sample numbers that are used as the inputs. We randomly choose a different number of samples to evaluate the architectures in the entire NATS-Bench-TSS and NATS-Bench-SSS, respectively. The results are shown in Figure 5. It can be demonstrated from the figure that MeCo is relatively stable when varying the number of samples. The performance of MeCo fluctuates around 0.01 throughout the experiments. These results support that MeCo only requires one data sample. 

**Fixed channel numbers** _n_ . We conduct an ablation study on _n_ v.s. MeCoopt with NATS-BenchTSS and NATS-Bench-SSS to illustrate the best choice of _n_ . The results are shown in Table 5. 

Table 5: Spearman correlation coefficients _ρ_ of proxies on NATS-Bench-TSS and NATS-Bench-SSS 

|**Approach**|NATS-Bench-TSS|NATS-Bench-SSS<br>CIFAT-10<br>CIFAR-100<br>ImageNet16|
|---|---|---|
||CIFAT-10<br>CIFAR-100<br>ImageNet16||
|_n_= 4<br>_n_= 6<br>_n_= 8|0.87_±_0.003<br>0.88_±_0.002<br>0.84_±_0.005<br>0.88_±_0.004<br>0.88_±_0.001<br>0.84_±_0.004<br>0.90_±_0.002<br>0.89_±_0.003<br>0.85_±_0.003|0.88_±_0.005<br>0.82_±_0.003<br>0.84_±_0.003<br>0.88_±_0.007<br>0.83_±_0.002<br>0.85_±_0.006<br>0.89_±_0.002<br>0.83_±_0.004<br>0.89_±_0.003|



The results show that with _n_ growing larger, MeCoopt increases by 0.1 to 0.5. We set _n_ = 8 in our experiments to obtain the best performance. 

## **6 Conclusion** 

In this work, we propose a novel feature-based zero-cost proxy called MeCo and its optimization method MeCoopt. Unlike the existing methods, our zero-cost proxies require only one data for a single forward pass. Specifically, we theoretically approximate a multi-channel convolution layer to an over-parameterized NN layer. We then harness the relationship between the theoretical properties and the minimum eigenvalue of the Pearson correlation matrix to craft our new proxies. We rigorously implement our proxies and extensively design the experiments. The experimental results show that MeCo and MeCoopt significantly outperforms the SOTA zero-cost proxies, which is also independent of the data and labels. Moreover, MeCo can be integrated into a complete NAS and enables us to efficiently find the architecture with the highest accuracy. 

10 

## **Acknowledgment** 

We sincerely thank all the reviewers of NeurIPS’23 for their constructive suggestions which significantly improve the quality of our paper. The work was supported by National Key R&D Program of China [Grant No. 2022ZD0115901], in part by the National Natural Science Foundation of China (Grant No. 62177007), and the China-Central Eastern European Countries High Education Joint Education Project (Grant No. 202012). 

## **References** 

- [1] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in _International Conference on Learning Representations_ , 2015. 

- [2] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 2016, pp. 770–778. 

- [3] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” in _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 2016, pp. 779–788. 

- [4] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” _Neural computation_ , vol. 9, no. 8, pp. 1735–1780, 1997. 

- [5] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” _Advances in neural information processing systems_ , vol. 30, 2017. 

- [6] H. Liu, K. Simonyan, and Y. Yang, “DARTS: Differentiable architecture search,” in _International Conference on Learning Representations_ , 2019. 

- [7] J. Mellor, J. Turner, A. Storkey, and E. J. Crowley, “Neural architecture search without training,” in _International Conference on Machine Learning_ . PMLR, 2021, pp. 7588–7598. 

- [8] M. S. Abdelfattah, A. Mehrotra, Ł. Dudziak, and N. D. Lane, “Zero-cost proxies for lightweight nas,” in _International Conference on Learning Representations_ , 2021. 

- [9] G. Li, Y. Yang, K. Bhardwaj, and R. Marculescu, “Zico: Zero-shot NAS via inverse coefficient of variation on gradients,” in _The Eleventh International Conference on Learning Representations_ , 2023. 

- [10] B. Zoph and Q. Le, “Neural architecture search with reinforcement learning,” in _International Conference on Learning Representations_ , 2017. 

- [11] L. Xie and A. Yuille, “Genetic cnn,” in _2017 IEEE International Conference on Computer Vision (ICCV)_ , 2017, pp. 1388–1397. 

- [12] B. Zoph, V. Vasudevan, J. Shlens, and Q. V. Le, “Learning transferable architectures for scalable image recognition,” in _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 2018, pp. 8697–8710. 

- [13] E. Real, S. Moore, A. Selle, S. Saxena, Y. L. Suematsu, J. Tan, Q. V. Le, and A. Kurakin, “Large-scale evolution of image classifiers,” in _International Conference on Machine Learning_ . PMLR, 2017, pp. 2902–2911. 

- [14] C. Thornton, F. Hutter, H. H. Hoos, and K. Leyton-Brown, “Auto-weka: Combined selection and hyperparameter optimization of classification algorithms,” in _Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining_ , 2013, pp. 847–855. 

- [15] Y. Geifman and R. El-Yaniv, “Deep active learning with a neural architecture search,” _Advances in Neural Information Processing Systems_ , vol. 32, 2019. 

11 

- [16] T. Elsken, J. H. Metzen, and F. Hutter, “Efficient multi-objective neural architecture search via lamarckian evolution,” in _International Conference on Learning Representations_ , 2019. 

- [17] M. Javaheripi, G. de Rosa, S. Mukherjee, S. Shah, T. Religa, C. C. Teodoro Mendes, S. Bubeck, F. Koushanfar, and D. Dey, “Litetransformersearch: Training-free neural architecture search for efficient language models,” _Advances in Neural Information Processing Systems_ , vol. 35, pp. 24 254–24 267, 2022. 

- [18] I. Fedorov, R. Matas, H. Tann, C. Zhou, M. Mattina, and P. Whatmough, “Udc: Unified dnas for compressible tinyml models for neural processing units,” _Advances in Neural Information Processing Systems_ , vol. 35, pp. 18 456–18 471, 2022. 

- [19] X. Wang, W. Guo, J. Su, X. Yang, and J. Yan, “Zarts: On zero-order optimization for neural architecture search,” in _Advances in Neural Information Processing Systems_ , S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 12 868–12 880. 

- [20] N. Lee, T. Ajanthan, and P. Torr, “Snip: Single-shot network pruning based on connection sensitivity,” in _International Conference on Learning Representations_ , 2019. 

- [21] C. Wang, G. Zhang, and R. Grosse, “Picking winning tickets before training by preserving gradient flow,” in _International Conference on Learning Representations_ , 2020. 

- [22] H. Tanaka, D. Kunin, D. L. Yamins, and S. Ganguli, “Pruning neural networks without any data by iteratively conserving synaptic flow,” _Advances in Neural Information Processing Systems_ , vol. 33, pp. 6377–6389, 2020. 

- [23] J. Turner, E. J. Crowley, M. O’Boyle, A. Storkey, and G. Gray, “Blockswap: Fisher-guided block substitution for network compression on a budget,” in _International Conference on Learning Representations_ , 2020. 

- [24] W. Chen, X. Gong, and Z. Wang, “Neural architecture search on imagenet in four gpu hours: A theoretically inspired perspective,” in _International Conference on Learning Representations (ICLR)_ , 2021. 

- [25] Y. Shu, Z. Dai, Z. Wu, and B. K. H. Low, “Unifying and boosting gradient-based training-free neural architecture search,” in _Advances in Neural Information Processing Systems_ , A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, Eds., 2022. 

- [26] Z. Zhu, F. Liu, G. Chrysos, and V. Cevher, “Generalization properties of nas under activation and skip connection search,” _Advances in Neural Information Processing Systems_ , vol. 35, pp. 23 551–23 565, 2022. 

- [27] M. Lin, P. Wang, Z. Sun, H. Chen, X. Sun, Q. Qian, H. Li, and R. Jin, “Zen-nas: A zero-shot nas for high-performance image recognition,” in _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 2021, pp. 347–356. 

- [28] A. Jacot, F. Gabriel, and C. Hongler, “Neural tangent kernel: Convergence and generalization in neural networks,” _Advances in neural information processing systems_ , vol. 31, 2018. 

- [29] S. Arora, S. S. Du, W. Hu, Z. Li, R. R. Salakhutdinov, and R. Wang, “On exact computation with an infinitely wide neural net,” _Advances in neural information processing systems_ , vol. 32, 2019. 

- [30] J. Lee, L. Xiao, S. Schoenholz, Y. Bahri, R. Novak, J. Sohl-Dickstein, and J. Pennington, “Wide neural networks of any depth evolve as linear models under gradient descent,” _Advances in neural information processing systems_ , vol. 32, 2019. 

- [31] S. S. Du, X. Zhai, B. Poczos, and A. Singh, “Gradient descent provably optimizes overparameterized neural networks,” in _International Conference on Learning Representations_ , 2019. 

- [32] B. Hanin and M. Nica, “Finite depth and width corrections to the neural tangent kernel,” in _International Conference on Learning Representations_ , 2020. 

12 

- [33] S. Du, J. Lee, H. Li, L. Wang, and X. Zhai, “Gradient descent finds global minima of deep neural networks,” in _International conference on machine learning_ . PMLR, 2019, pp. 1675–1685. 

- [34] D. Zou and Q. Gu, “An improved analysis of training over-parameterized deep neural networks,” _Advances in neural information processing systems_ , vol. 32, 2019. 

- [35] Q. N. Nguyen and M. Mondelli, “Global convergence of deep networks with one wide layer followed by pyramidal topology,” _Advances in Neural Information Processing Systems_ , vol. 33, pp. 11 961–11 972, 2020. 

- [36] N. Lee, T. Ajanthan, S. Gould, and P. H. S. Torr, “A signal propagation perspective for pruning neural networks at initialization,” in _International Conference on Learning Representations_ , 2020. [Online]. Available: https://openreview.net/forum?id=HJeTo2VFwH 

- [37] K. Bhardwaj, G. Li, and R. Marculescu, “How does topology influence gradient propagation and model performance of deep networks with densenet-type skip connections?” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2021, pp. 13 498– 13 507. 

- [38] H. Li, Z. Xu, G. Taylor, C. Studer, and T. Goldstein, “Visualizing the loss landscape of neural nets,” _Advances in neural information processing systems_ , vol. 31, 2018. 

- [39] R. A. Horn and C. R. Johnson, _Matrix analysis_ . Cambridge university press, 2012. 

- [40] Y. Cao and Q. Gu, “Generalization bounds of stochastic gradient descent for wide and deep neural networks,” _Advances in neural information processing systems_ , vol. 32, 2019. 

- [41] X. Dong and Y. Yang, “Nas-bench-201: Extending the scope of reproducible neural architecture search,” in _International Conference on Learning Representations_ , 2020. 

- [42] X. Dong, L. Liu, K. Musial, and B. Gabrys, “Nats-bench: Benchmarking nas algorithms for architecture topology and size,” _IEEE transactions on pattern analysis and machine intelligence_ , 2021. 

- [43] A. Krizhevsky, G. Hinton _et al._ , “Learning multiple layers of features from tiny images.” Toronto, ON, Canada, 2009. 

- [44] P. Chrabaszcz, I. Loshchilov, and F. Hutter, “A downsampled variant of imagenet as an alternative to the cifar datasets,” _arXiv preprint arXiv:1707.08819_ , 2017. 

- [45] A. R. Zamir, A. Sax, W. Shen, L. J. Guibas, J. Malik, and S. Savarese, “Taskonomy: Disentangling task transfer learning,” in _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 2018, pp. 3712–3722. 

- [46] A. Zela, J. N. Siems, L. Zimmer, J. Lukasik, M. Keuper, and F. Hutter, “Surrogate nas benchmarks: Going beyond the limited search spaces of tabular nas benchmarks,” in _International Conference on Learning Representations_ , 2022. 

- [47] Y. Duan, X. Chen, H. Xu, Z. Chen, X. Liang, T. Zhang, and Z. Li, “Transnas-bench-101: Improving transferability and generalizability of cross-task neural architecture search,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2021, pp. 5251–5260. 

- [48] C. Ying, A. Klein, E. Christiansen, E. Real, K. Murphy, and F. Hutter, “Nas-bench-101: Towards reproducible neural architecture search,” in _International Conference on Machine Learning_ . PMLR, 2019, pp. 7105–7114. 

- [49] M. Chen, H. Peng, J. Fu, and H. Ling, “Autoformer: Searching transformers for visual recognition,” in _Proceedings of the IEEE/CVF international conference on computer vision_ , 2021, pp. 12 270–12 280. 

- [50] H. Cai, C. Gan, T. Wang, Z. Zhang, and S. Han, “Once-for-all: Train one network and specialize it for efficient deployment,” in _International Conference on Learning Representations_ , 2020. [Online]. Available: https://openreview.net/forum?id=HylxE1HKwS 

13 

- [51] J. Xu, L. Zhao, J. Lin, R. Gao, X. Sun, and H. Yang, “Knas: green neural architecture search,” in _International Conference on Machine Learning_ . PMLR, 2021, pp. 11 613–11 625. 

- [52] Y. Shu, S. Cai, Z. Dai, B. C. Ooi, and B. K. H. Low, “NASI: Label- and data-agnostic neural architecture search at initialization,” in _International Conference on Learning Representations_ , 2022. [Online]. Available: https://openreview.net/forum?id=v-v1cpNNK_v 

- [53] Z. Zhang and Z. Jia, “Gradsign: Model performance inference with theoretical insights,” in _International Conference on Learning Representations_ , 2022. [Online]. Available: https://openreview.net/forum?id=HObMhrCeAAF 

- [54] A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan _et al._ , “Searching for mobilenetv3,” in _Proceedings of the IEEE/CVF international conference on computer vision_ , 2019, pp. 1314–1324. 

- [55] L. Xiang, Ł. Dudziak, M. S. Abdelfattah, T. Chau, N. D. Lane, and H. Wen, “Zero-cost proxies meet differentiable architecture search,” _arXiv preprint arXiv:2106.06799_ , 2021. 

- [56] E. Real, A. Aggarwal, Y. Huang, and Q. V. Le, “Regularized evolution for image classifier architecture search,” in _Proceedings of the aaai conference on artificial intelligence_ , vol. 33, no. 01, 2019, pp. 4780–4789. 

- [57] C. Liu, B. Zoph, M. Neumann, J. Shlens, W. Hua, L.-J. Li, L. Fei-Fei, A. Yuille, J. Huang, and K. Murphy, “Progressive neural architecture search,” in _Proceedings of the European conference on computer vision (ECCV)_ , 2018, pp. 19–34. 

- [58] R. Wang, M. Cheng, X. Chen, X. Tang, and C.-J. Hsieh, “Rethinking architecture selection in differentiable nas,” in _International Conference on Learning Representation_ , 2021. 

- [59] X. Chen and C.-J. Hsieh, “Stabilizing differentiable architecture search via perturbation-based regularization,” in _International conference on machine learning_ . PMLR, 2020, pp. 1554–1565. 

- [60] G. Li, G. Qian, I. C. Delgadillo, M. Muller, A. Thabet, and B. Ghanem, “Sgas: Sequential greedy architecture search,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , June 2020. 

- [61] W. Hoeffding, “Probability inequalities for sums of bounded random variables,” _The collected works of Wassily Hoeffding_ , pp. 409–426, 1994. 

- [62] G. H. Golub and C. F. Van Loan, _Matrix computations_ . JHU press, 2013. 

- [63] Y. Xu, L. Xie, X. Zhang, X. Chen, G.-J. Qi, Q. Tian, and H. Xiong, “Pc-darts: Partial channel connections for memory-efficient architecture search,” in _International Conference on Learning Representations_ , 2020. 

- [64] X. Chen, L. Xie, J. Wu, and Q. Tian, “Progressive differentiable architecture search: Bridging the depth gap between search and evaluation,” in _Proceedings of the IEEE/CVF international conference on computer vision_ , 2019, pp. 1294–1303. 

- [65] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” in _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 2017, pp. 4700–4708. 

- [66] S. Xie, H. Zheng, C. Liu, and L. Lin, “SNAS: stochastic neural architecture search,” in _International Conference on Learning Representations_ , 2019. 

- [67] A. Zela, T. Elsken, T. Saikia, Y. Marrakchi, T. Brox, and F. Hutter, “Understanding and robustifying differentiable architecture search,” in _International Conference on Learning Representations_ , 2020. 

- [68] P. Ye, B. Li, Y. Li, T. Chen, J. Fan, and W. Ouyang, “ _β_ -darts: Beta-decay regularization for differentiable architecture search,” in _2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ . IEEE, 2022, pp. 10 864–10 873. 

14 

## **A Detailed proofs** 

## **A.1 Proof of Theorem 2** 

In this section, we present the proof for our main theorem (Theorem 2 in Section 3.4). We first introduce the following definitions and lemmas to better illustrate our proof. 

**Lemma 1** (Lemma 3.1 in [31]) **.** _If m_ = Ω � _nλ_[2] 0[2][log] � _nδ_ �[�] _, then with probability at least_ 1 _− δ, ∥_ **H** (0) _−_ **H** _[∞] ∥_ 2 _≤[λ]_ 4[0] _[and]_[3] 4 _[λ]_[0] _[≤][λ]_[min][(] **[H]**[(0))] _[ ≤]_[5] 4 _[λ]_[0] _[.]_ **Definition 4.** _We denote the empirical indicator matrix as_ **W** (0) _with entry_ [ **W** (0)] _ij such that_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0015-04.png)


_and indicator matrix as_ **W**[¯] _with entry_ [ **W**[¯] ] _ij_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0015-06.png)


The indicator matrix **W**[¯] has the following property. **Proposition 1.** _If_ **x** _i_ ∦ **x** _j, then_ [ **W**[¯] ] _ij >_ 0 _._ 

**Proof.** If Proposition 1 does not hold, then _∃i_ 0 _, j_ 0 _∈_ [ _n_ ] _×_ [ _n_ ], satisfying 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0015-09.png)


Thus **x** _i_ 0 = _−_ **x** _j_ 0, which contradicts the hypothesis **x** _i_ 0 ∦ **x** _j_ 0. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0015-11.png)


In real-world datasets, the possibility of two different data being parallel is slight. Thus, proposition 1 holds in general. We further introduce Weyl inequality as follows: 

**Lemma 2** (Weyl inequality [39]) **.** _Let_ **A** _,_ **B** _∈_ R _[n][×][n] be Hermitian matrices, and let the eigenvalues of_ **A** _,_ **B** _, and_ **A** + **B** _be {λi_ ( **A** ) _}[n] i_ =1 _[,][ {][λ][i]_[(] **[B]**[)] _[}][n] i_ =1 _[and][ {][λ][i]_[(] **[A]**[+] **[B]**[)] _[}][n] i_ =1 _[, respectively. The eigenvalues] of each matrix are arranged in ascending order. Then we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0015-14.png)


_for each i_ = 1 _, . . . , n, with equality for some pair i, j if and only if there is a nonzero vector_ **x** _such that_ **Ax** = _λi_ + _j_ ( **A** ) **x** _,_ **Bx** = _λn−j_ ( **B** ) **x** _, and_ ( **A** + **B** ) **x** = _λi_ ( **A** + **B** ) **x** _. Also,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0015-16.png)


_for each i_ = 1 _, . . . , n, with equality for some pair i, j if and only if there is a nonzero vector_ **x** _such that_ **Ax** = _λi−j_ +1( **A** ) **x** _,_ **Bx** = _λj_ ( **B** ) **x** _, and_ ( **A** + **B** ) **x** = _λi_ ( **A** + **B** ) **x** _. If_ **A** _and_ **B** _have no common eigenvector, then inequality (14) and (15) are strict inequality._ 

Now, we provide the full proof of Theorem 2. 

**Theorem 2.** Suppose _f_ is an NN with a single hidden layer and ReLU activation function. Assume **X** _∈_ R _[d][×][n]_ , **w** (0) _∼ N_ ( **0** _,_ **I** ), _P_ ( **X** ) _≻_ 0, _p_ 0 := _λ_ min( _P_ ( **X** )), and hidden nodes _m_ = Ω � _nλ_[6][4] 0 _d[δ]_[3][2] �, then the following formula holds with probability at least 1 _− δ_ over the initialization 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0015-20.png)


where _c_ is a constant depending on _m_ and _d_ . 

15 

_Proof._ To simplify our proof, we assume _µ_ **x** = 0, _σ_ **x** = 1, and _∥_ **x** _∥_ 2 _≤ C_ for all **x** _∈_ **X** . Recall that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0016-01.png)


Due to I _{_ **w** _r[T]_[(0)] **[x]** _[i][≥]_[0] _[,]_ **[ w]** _r[T]_[(0)] **[x]** _[j][≥]_[0] _[}]_[ is an independent random variable between 0 and 1, then by] Hoeffding’s inequality [61], the following inequality holds with probability 1 _− δ_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0016-03.png)


> [(][1] _[/][δ]_[)] Let _µ_ 0 = min( _i,j_ ) _∈_ [ _n_ ] _×_ [ _n_ ][ **W**[¯] ] _ij_ , and choose _m >_[16 lo] _µ_[g][2] 0 , then we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0016-05.png)


Define the matrix **M** with entry [ **M** ] _ij_ as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0016-07.png)


where _c_ ( _m, d_ ) = _c_ ( _m_ )( _d −_ 1). We claim that if _m_ is large enough, then **M** is positive definite. To clarify this statement, we consider the gap between **H** (0) and **M** : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0016-09.png)


If we choose _m >_[64 lo][g][(][1] _[/][δ]_[)(] _λ_[2] 0 _[d][−]_[1][)][2] _[n]_[4] _[C]_[4] , we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0016-11.png)


Then the following formula holds by matrix perturbation theory (Corollary 6.3.8 in [39]) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0016-13.png)


which indicates that **M** is positive definite. We next exert Lemma 2 on **M** and get 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0016-15.png)


That means 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0016-17.png)


Therefore, combined with Theorem 1, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0016-19.png)


where _c_ ( _m, d_ ) is a constant depending on _m_ and _d_ . This completes the proof of **Theorem 2** . ■ 

16 

## **A.2 Proof of Theorem 3** 

**Proof.** By Corollary 3.10 and Remark 3.11 in [40], we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0017-02.png)


According to the courant minimax principle[62], D.2 in [26], and inequality 16, we get 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0017-04.png)


Thus, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0017-06.png)


## **B Zero-cost proxies** 

In this section, we provide the details of the baseline zero-cost proxies. Suppose _L_ is the loss function and _θ_ is the parameters of an initialized network. We denote _◦_ as the Hadamard product. The concrete formulations of the existing zero-cost proxies are as follows. 

- snip. Lee et al. [20] use the changes in loss caused by the parameter perturbations to measure the importance of the parameters in an initialized network, such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0017-10.png)


- grasp. Wang et al. [21] replace the loss change in snip with the change of the gradient norm to establish the proxy, such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0017-12.png)


where _H_ is the Hessian. 

- synflow. To avoid layer collapse, Tanaka et al. [22] utilize the product of all parameters in the network during the parameter perturbation to represent the loss, such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0017-15.png)


- grad_norm. Abdelfattah et al. [8] adopt the _l_ 2 norm of the gradients in an initialized work as a proxy, such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0017-17.png)


- jacov _/_ NWOT. Mellor et al. [7] use the correlation of activations within a network as a proxy to evaluate the performance of the network, such that 

17 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0018-00.png)


where _NA_ is the number of rectified linear units, _dH_ ( **c** _i,_ **c** _j_ ) represents the Hamming distance between two binary codes **c** _i_ and **c** _j_ . 

- NTK. Chen et al. [24] propose to use the condition number of NTK to measure the trainability of the networks, such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0018-03.png)


where Θ[ˆ] _train_ stands for NTK of the networks. In our paper, we calculate the Spearman correlation coefficient between 1 _/κN_ and the test accuracy of the networks. 

- zen. Lin et al. [27] propose Zen-Score, in which they design an efficient zero-cost proxy with Gaussian random inputs, such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0018-06.png)


where _σi,j_ is the mini-batch standard deviation statistic of the _j_ -th channel in BN. 

- NASI. Shu et al. [52] propose NASI to evaluate the networks by approximating the trace of the NTK, such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0018-09.png)


where _Xj_ is a mini-batch of data with size _|Xj|_ = _b_ . 

- KNAS. Xu et al. [51] propose to use gradient kernel to evaluate the networks, such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0018-12.png)


where _y[L]_ is the output of _L_ -th layer. 

- GradSign. Zhang and Jia [53] propose GradSign, in which they analyze the sample-wise optimization landscape of the networks, such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0018-15.png)


- ZiCo. Li et al. [9] explore the effect of gradient properties on network performance. They use absolute mean and standard deviation values of gradients to evaluate network performance: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0018-17.png)


where _N_ and _D_ represent the number of batches and network layers, respectively. _θl_ represents the parameters of the _l_ -th layers. 

18 

## **C Algorithm of** MeCo **-based NAS** 

We slightly abuse the notation and denote MeCo as our proxy function for the network. We adopt the Zero-Cost-PT [55] to integrate MeCo to a zero-shot NAS. The algorithm of our MeCo-based NAS is summarized in Algorithm 1. We denote _A_ 0 as an untrained supernet, _et_ represents the _t_ -th edge, which stands for mixed operation in the search cells, and _et,k_ is the _k_ -th operation of _t_ -th edge. We denote _E_ , _N_ , and _O_ as the set of edges, nodes, and candidate operations in the search cells, respectively. For any node _n ∈N_ , we use _E_ ( _n_ ) as the set of its input edges, and _e[k] n_[is the] _[ k]_[-th] element of _E_ ( _n_ ). Note that we can use other zero-cost proxies as described in section B to replace MeCo in Algorithm 1. 

## **Algorithm 1** MeCo-based NAS 

**Require:** _A_ 0: An untrained supernet; _E_ : The set of edges in search cells; _N_ : The set of nodes in search cells; _O_ : The set of candidate operations; _N_ : the number of candidate networks. **Ensure:** The best network _Abest_ . 

1: // Stage 1: Architecture Proposal 2: _C_ = _∅_ ; 3: **for** _i_ = 1; _i ≤ N_ ; _i_ + + **do** 4: **for** _j_ = 1; _j ≤|E|_ ; _j_ + + **do** 5: Randomly choose an un-discretized edge _et_ 6: Choose the best edge from the supernet, s.t. 

_et,best_ = arg min MeCo( _A_ 0 _/et,k_ ) 1 _≤k≤|O|_ 

7: Use operation _et,best_ to substitute _et_ 8: **end for** 9: _A|E|_ consists of _{et,best|_ 1 _≤ t ≤|E|}_ 10: **for** _j_ = 1; _j < |N|_ , _j_ + + **do** _▷_ prune the edges of the obtained architecture _A|E|_ 11: Randomly select an unselected node _n ∈N_ 12: **for** _k_ = 1; _k < |E_ ( _n_ ) _|_ ; _k_ + + **do** 13: Calculate MeCo of the architecture _A|E|/e[k] n_ 14: **end for** 15: Retain edges _e_[1] _n_[,] _[ e]_[2] _n_[with the 1st and 2nd best][ MeCo][ value, and remove the other edges] 16: **end for** 17: Get the candidate networks _Ai_ that consist of _{et,best|_ 1 _≤ t ≤E}_ , and append it to the set _C_ 18: **end for** 

19: // Stage 2: Architecture Validation 

20: Get the best network: 

_Abest_ = arg max MeCo( _Ai_ ) _, s.t. Ai ∈ C_ 1 _≤i≤N_ 

## **D Experimental configurations** 

We use the same settings for the experiments as in [55]. We summarized the configurations of the searching phase and training phase on CIFAR-10 and CIFAR-100 in Table 6 and Table 7, respectively. For ZiCo, we use two mini-batch of data, which has a size of 64. For the other baseline proxies, we use one mini-batch of data. On the other hand, our MeCo only uses one random data **x** _∈_ R[1] _[×]_[3] _[×]_[32] _[×]_[32] . 

## **E More experimental results of** _ρ_ 

## **E.1** MeCo **on NASBench-201 and NASBench-301 with three extra datasets** 

We evaluate MeCo and MeCoopt on NASBench-201 and NASBench-301 with Spherical-CIFAR-100, NinaPro, and SVHN, respectively. We only use one random data _x ∈_ R[3] _[×]_[32] _[×]_[32] for MeCo and MeCoopt. The results are summarized in Table 8. Our MeCo achieves competitive results on most benchmarks and achieves the best results on NASBench-201-SVHN, where the Spearman correlation coefficient is _ρ_ = 0 _._ 88. 

19 

Table 6: The settings of Zero-Cost-PT with all proxies in DARTS-CNN for CIFAR-10 

|Settings|Searching phase|Searching phase|Training phase|Training phase|
|---|---|---|---|---|
||||||
||Baselines|MeCo(Ours)|Baselines|MeCo(Ours)|
||||||
|batch size<br>cutout<br>cutout length<br>learning rate<br>learning rate min<br>momentum<br>weight decay<br>grad clip<br>init channels<br>layers<br>drop path prob|64<br>True<br>16<br>0.025<br>0.001<br>0.9<br>3e-4<br>5<br>16<br>8<br>-|1<br>False<br>-<br>0.025<br>0.001<br>0.9<br>3e-4<br>5<br>16<br>8<br>-|96<br>True<br>16<br>0.025<br>-<br>0.9<br>3e-4<br>5<br>36<br>20<br>0.2|96<br>True<br>16<br>0.025<br>-<br>0.9<br>3e-4<br>5<br>36<br>20<br>0.2|



Table 7: The settings of Zero-Cost-PT with all proxies in DARTS-CNN for CIFAR-100 

|Settings|Searching phase|Searching phase|Training phase|Training phase|
|---|---|---|---|---|
||||||
||Baselines|MeCo(Ours)|Baselines|MeCo(Ours)|
||||||
|batch size<br>cutout<br>cutout length<br>learning rate<br>learning rate min<br>momentum<br>weight decay<br>grad clip<br>init channels<br>layers<br>drop path prob|64<br>True<br>16<br>0.025<br>0.001<br>0.9<br>3e-4<br>5<br>16<br>8<br>-|1<br>False<br>-<br>0.025<br>0.001<br>0.9<br>3e-4<br>5<br>16<br>8<br>-|96<br>True<br>16<br>0.025<br>-<br>0.9<br>3e-4<br>5<br>16<br>20<br>0.2|96<br>True<br>16<br>0.025<br>-<br>0.9<br>3e-4<br>5<br>16<br>20<br>0.2|



Table 8: Comparisons of _ρ_ with baselines using NASBench-301 and NASBench-201 on three extra datasets 

|Table 8: Compariso<br>datasets|ns of_ρ_with baselines using NASBench|-301 and NASBench-201 on three extra|
|---|---|---|
|**Method**|**NASBench-301**<br>**Sph-Cifar100**<br>**NinaPro**<br>**SVHN**|**NASBench-201**<br>**Sph-Cifar100**<br>**NinaPro**<br>**SVHN**|
|grasp<br>fsher<br>grad_norm<br>snip<br>synfow<br>l2_norm<br>#params<br>zen<br>jacov<br>nwot|0.13<br>0.04<br>0.18<br>0.00<br>-0.11<br>0.05<br>-0.00<br>**-0.20**<br>0.42<br>-0.01<br>-0.10<br>0.38<br>0.05<br>-0.07<br>0.50<br>**0.12**<br>-0.07<br>**0.70**<br>0.07<br>-0.07<br>**0.70**<br>0.07<br>-0.09<br>0.68<br>0.08<br>0.13<br>-0.36<br>0.05<br>0.02<br>0.64|-0.01<br>-0.01<br>0.62<br>0.07<br>**-0.38**<br>0.71<br>-0.08<br>-0.23<br>0.77<br>-0.09<br>-0.28<br>0.76<br>0.13<br>0.02<br>0.71<br>-0.00<br>0.02<br>0.67<br>-0.14<br>-0.11<br>0.72<br>0.23<br>0.15<br>0.18<br>**-0.41**<br>0.29<br>0.67<br>-0.02<br>0.06<br>0.76|
|MeCo**(Ours)**<br>MeCoopt**(Ours)**|**-0.05**<br>**-0.11**<br>**0.68**<br>**0.03**<br>**0.12**<br>**0.68**|**-0.23**<br>**0.02**<br>**0.88**<br>-<br>-<br>-|



20 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0021-00.png)


**----- Start of picture text -----**<br>
CIFAR-10 (0.89) CIFAR-100 (0.88) ImageNet16-120 (0.85)<br>80<br>80 60 40<br>60<br>40<br>40 20<br>20<br>20<br>0 0<br>0 20 40 60 80 0 20 40 60 80 0 20 40<br>MeCo MeCo MeCo<br>(a) Performance on NATS-Bench-TSS<br>CIFAR-10 (-0.80) CIFAR-100 (-0.89) ImageNet16-120 (-0.88)<br>50<br>90 70<br>40<br>60<br>85 30<br>50<br>20<br>40<br>10 20 30 0 10 20 10 20 30<br>MeCo MeCo MeCo<br>(b) Performance on NATS-Bench-SSS<br>acc acc acc<br>Test Test Test<br>acc acc acc<br>Test Test Test<br>**----- End of picture text -----**<br>


Figure 4: Relationship between our zero-cost proxies and test accuracy for NATS-Bench-TSS and NATS-Bench-SSS on three datasets. 

## **E.2** MeCo **on NAS-Bench-101** 

We present the Spearman correlation coefficient between all zero-cost proxies and the test accuracy of the networks in NAS-Bench-101. Our MeCo uses one data of CIFAR-10, i.e., **x** _∈_ R[1] _[×]_[3] _[×]_[32] _[×]_[32] while the baselines adopt a batch of samples as input. The results are summarized in Table 9. 

Table 9: Spearman correlation coefficients _ρ_ between zero-cost proxies and test accuracy on NASBench-101 

|Table 9: Spearma<br>Bench-101|n correlation coeffcients_ρ_between zero-cost proxies and test accuracy on NAS|
|---|---|
|Dataset<br>MeCo|Baselines<br>grasp[21]<br>fsher[23]<br>grad_norm[8]<br>snip[20]<br>synfow[22]<br>jacov[7]<br>zen[27]<br>ZiCo[9]|
|CIFAR-10<br>0.44|-0.33<br>-0.27<br>-0.32<br>-0.25<br>0.36<br>-0.35<br>**0.63**<br>**0.63**|



MeCo achieves competitive results on NAS-Bench-101 with CIFAR-10, i.e., the Spearman correlation coefficient obtains 0.44, which is higher than the majority of the baselines. However, zen and ZiCo perform better than MeCo. 

## **E.3** MeCo **on TransBench-101** 

We further evaluate our MeCo on diverse tasks. Specifically, we compare MeCo and the other proxies on Transbench-101-Micro and Transbench-101-Macro [47], respectively. 

**TransBench-101-Micro.** We evaluate MeCo and MeCoopt on TransBench-101-Micro with ten tasks. We calculate the Spearman correlation between our proxies and the accuracy of the networks. The results are summarized in Table 10. MeCo achieves the best performance on the three tasks, i.e., Class Objection, Spherical-Cifar100, and NinaPro, which are 0.58, 0.85, and 0.47, respectively. It can be seen that MeCoopt effectively improves the performance of MeCo. For example, MeCoopt improves MeCo from 0.62 to 0.77 on segmentation tasks. 

**TransBench-101-Macro.** We compare MeCo, MeCoopt and the baselines on TransBench-101-Macro with seven tasks. The results are summarized in Table 11. Our proxies achieve the best performance on Autoencoding, Jigsaw, and Surface Normal, which are 0.74, 0.48, and 0.80, respectively. Our proxies also achieve competitive results on the remaining tasks. Experimental results show that 

21 

our proxies can be used for diverse tasks. It can be demonstrated that MeCo and MeCoopt become ineffective on a few tasks, such as Room Layout. We would like to note that the existing proxies do not achieve a high correlation on all tasks consistently. 

Table 10: Comparisons of _ρ_ with baselines using Transbench-101-Micro on Ten Tasks 

|Table|10: Compar|isons of|_ρ_with base|lines u|sing Tr|ansbench-1|01-Mi|cro on Te|n Tasks||
|---|---|---|---|---|---|---|---|---|---|---|
|**Approach**|**Autoencoding**|**Class**<br>**Objection**|**Scene**<br>**Classifcation**|**Jigsaw**|**Surface**<br>**Normal**|**Segmantation**|**Room**<br>**Layout**|**Spherical**<br>**-Cifar100**|**NinaPro**|**SVHN**|
|grasp<br>fsher|-0.12<br>**-0.58**|-0.22<br>0.44|-0.43<br>-0.13|-0.12<br>0.30|0.01<br>0.16|0.00<br>0.12|-0.29<br>0.30|-0.03<br>0.72|-0.20<br>0.42|-0.24<br>0.81|
|grad_norm|-0.32|0.39|-0.33|0.36|0.36|0.60|0.25|0.72|0.40|0.78|
|snip<br>synfow|-0.27<br>0.00|0.45<br>0.48|-0.14<br>0.27|0.41<br>0.47|0.49<br>0.00|0.68<br>0.00|0.32<br>0.30|0.76<br>0.79|0.42<br>0.45|0.83<br>**0.92**|
|l2_norm|0.04|0.32|0.28|0.35|0.50|0.48|0.18|0.53|0.36|0.52|
|#params|-0.01|0.45|0.32|0.44|0.62|0.68|0.30|0.79|0.36|0.76|
|zen|0.14|0.54|0.27|0.51|0.71|0.67|0.38|0.67|0.42|0.74|
|jacov|0.18|0.51|0.19|**0.56**|**0.75**|**0.80**|**0.40**|0.71|0.40|0.77|
|nwot|0.03|0.39|**0.89**|0.42|0.57|0.53|0.25|0.64|0.38|0.63|
|zico|0.35|-|0.71|0.52|0.68|-|-|-|-|-|
|MeCo**(Ours)**|**0.03**|**0.58**|**0.62**|**0.45**|**0.65**|**0.62**|**-0.25**|**0.85**|**0.47**|**0.88**|
|MeCoopt**(Ours)**|**0.03**|**0.59**|**0.64**|**0.47**|**0.67**|**0.77**|**0.26**|**0.85**|**0.47**|**0.88**|



Table 11: Comparisons of _ρ_ with baselines using Transbench-101-Macro on Seven Tasks 

|Table 11: C|omparisons of|_ρ_with base|lines using Tr|ansbench|-101-Ma|cro on Seven T|asks|
|---|---|---|---|---|---|---|---|
|**Approach**|**Autoencoding**|**Class**<br>**Objection**|**Scene**<br>**Classifcation**|**Jigsaw**|**Surface**<br>**Normal**|**Segmantation**|**Room**<br>**Layout**|
|grasp<br>fsher|-0.02<br>-0.19|-0.64<br>-0.30|-0.43<br>-0.13|-0.26<br>-0.26|-0.05<br>0.15|-0.02<br>0.03|-0.26<br>-0.26|
|grad_norm|0.31|-0.56|-0.33|-0.27|0.35|0.21|-0.27|
|snip<br>synfow|0.20<br>0.00|-0.38<br>0.12|-0.14<br>0.27|-0.19<br>0.34|0.45<br>0.00|0.27<br>0.00|-0.19<br>0.34|
|l2_norm|-0.20|0.08|0.28|0.15|0.30|0.18|0.15|
|#params|-0.18|0.16|0.32|0.15|0.30|0.06|0.15|
|zen|-0.01|0.10|0.27|0.24|0.38|0.27|0.24|
|jacov|0.45|0.07|0.19|0.19|0.50|0.57|0.19|
|nwot|**0.67**|**0.83**|**0.89**|**0.48**|0.78|**0.80**|**0.76**|
|MeCo**(Ours)**|**0.51**|**0.59**|**0.81**|**0.17**|**0.80**|**0.62**|**0.23**|
|MeCoopt**(Ours)**|**0.74**|**0.73**|**0.76**|**0.48**|**0.76**|**0.63**|**0.33**|



## **E.4** MeCo **on AutoFormer and MobileNet OFA** 

**AutoFormer.** Chen et al. [49] proposed a novel one-shot architecture search framework for transformer-based models. We load the trained supernets and regenerate the candidate subnets. We then re-evaluate the subnets to obtain the accuracy on ImageNet and compute MeCo. The correlation of MeCo and model accuracy on AutoFormer is 0.45. 

**OFA.** To solve the problem of efficient inference across devices and resource constraints, Cai et al. [50] proposed to train a once-for-all (OFA) network, which supports diverse architectural settings. We randomly sample 1,000 subnets from the OFA network and use the accuracy of the predictor predictions as the test accuracy. The Spearman correlation between MeCo and test accuracy is 0.86. 

## **F More experimental results of NAS with** MeCo 

In this section, we provide more results and comparisons of our zero-shot NAS on NATS-Bench-TSS and DARTS-CNN. In the following descriptions, we denote multi-shot, one-shot, and zero-shot as MS, OS, and ZS, respectively. 

## **F.1 Results on NATS-Bench-TSS** 

In the NATS-Bench-TSS search space, we evaluate all networks using zero-cost proxies and choose the Top-10 networks with the highest scores. Then we calculate the average test accuracy and standard deviation. The results are summarized in Table 12. As shown in the table, the networks searched by MeCo have the highest average precision on CIFAR-10, which is 0.08 higher than the best baseline proxy ZiCo. For CIFAR-100 and ImageNet16-120, MeCo achieves competitive results, 

22 

e.g., 70 _._ 86% _±_ 0 _._ 96% with CIFAR-100 and 42 _._ 59% _±_ 1 _._ 77% with ImageNet16-120. In all, the results demonstrate that our MeCo has a great advantage in evaluating network performance, considering MeCo only requires one data sample as input. 

We further combine MeCo with Zero-Cost-PT [55] and search for the best architecture three times with different seeds. The accuracy of the selected architectures as well as the comparisons with the baseline methods are presented in Table 13. It can be shown from the results that our MeCo-based NAS achieves competitive results with the SOTA baselines, e.g., synflow, zen, and ZiCo. However, MeCo-based NAS invokes one data for a single forward pass, thus being more resource-saving. 

Table 12: The average test accuracy of Top-10 architectures obtained by various zero-cost proxies on NATS-Bench-TSS using CIAFR-10, CIAFR-100, and ImageNet16-120, respectively 

|Dataset<br>MeCo|Baselines<br>grasp[21]<br>fsher[23]<br>grad_norm[8]<br>snip[20]<br>synfow[22]<br>jacov[7]<br>NTK[24]<br>zen[27]<br>ZiCo[9]|
|---|---|
|**CIFAR-10**<br>**93**_._**64**<br>_±_**0**_._**31**|89_._34<br>89_._27<br>89_._27<br>89_._27<br>93_._27<br>91_._23<br>92_._67<br>58_._99<br>93_._56<br>_±_2_._16<br>_±_2_._10<br>_±_2_._10<br>_±_2_._10<br>_±_0_._74<br>_±_1_._01<br>_±_0_._46<br>_±_3_._44<br>_±_0_._23|
|**CIFAR-100**<br>70_._86<br>_±_0_._96|60_._89<br>61_._06<br>60_._89<br>60_._89<br>**71**_._**12**<br>68_._50<br>69_._31<br>12_._73<br>70_._64<br>_±_3_._88<br>_±_4_._02<br>_±_3_._88<br>_±_3_._88<br>_±_**1**_._**59**<br>_±_1_._21<br>_±_1_._17<br>_±_1_._33<br>_±_0_._28|
|**ImageNet16-120**<br>42_._59<br>_±_1_._77|22_._99<br>24_._10<br>23_._35<br>23_._35<br>42_._65<br>40_._59<br>39_._98<br>15_._10<br>**42**_._**74**<br>_±_11_._01<br>_±_11_._36<br>_±_11_._44<br>_±_11_._44<br>_±_3_._59<br>_±_1_._86<br>_±_1_._73<br>_±_0_._51<br>_±_**1**_._**78**|



Table 13: The test accuracy of optimal architectures obtained by Zero-Cost-PT with various zero-cost proxies on NATS-Bench-TSS using CIAFR-10, CIFAR-100, and ImageNet16-120, respectively 

|Dataset<br>MeCo|Baselines<br>grasp[21]<br>fsher[23]<br>grad_norm[8]<br>snip[20]<br>synfow[22]<br>jacov[7]<br>NTK[24]<br>zen[27]<br>ZiCo[9]|
|---|---|
|**CIFAR-10**<br>**93**_._**76**<br>_±_**0**|92_._59<br>88_._39<br>91_._64<br>90_._11<br>**93**_._**76**<br>91_._92<br>92_._61<br>**93**_._**76**<br>**93**_._**76**<br>_±_1_._12<br>_±_2_._55<br>_±_0_._68<br>_±_2_._85<br>_±_**0**<br>_±_2_._40<br>_±_0_._65<br>_±_**0**<br>_±_**0**|
|**CIFAR-100**<br>**71**_._**11**<br>_±_**0**|68_._98<br>65_._77<br>65_._20<br>65_._29<br>**71**_._**11**<br>69_._67<br>68_._27<br>**71**_._**11**<br>**71**_._**11**<br>_±_2_._69<br>_±_0_._93<br>_±_0_._56<br>_±_0_._96<br>_±_**0**<br>_±_2_._39<br>_±_2_._34<br>_±_**0**<br>_±_**0**|
|**ImageNet16-120**<br>**41**_._**44**<br>_±_**0**|35_._29<br>28_._91<br>35_._82<br>37_._38<br>**41**_._**44**<br>40_._35<br>41_._25<br>**41**_._**44**<br>**41**_._**44**<br>_±_8_._03<br>_±_5_._18<br>_±_3_._99<br>_±_4_._41<br>_±_**0**<br>_±_6_._56<br>_±_2_._37<br>_±_**0**<br>_±_**0**|



## **F.2 Results on DARTS-CNN** 

For DARTS-CNN search space, we search the architectures by Zero-Cost-PT with different zero-cost proxies on CIFAR-100. Each searched network is trained five times, and the results are summarized in Table 14. There are two settings in our experiments on DARTS-CNN: networks initialized with 16 channels and trained as in Table 7, and networks initialized with 36 channels and trained as in [63]. It can be shown from Table 14 that MeCo achieves competitive results compared with MS, OS, and ZS baselines. More concretely, MeCo-based NAS obtains 19 _._ 33% test error with 0.08 GPU days, which outperforms all the ZS methods under the same settings. On the other hand, compared with the manual, MS, and OS methods, MeCo is also competitive. For example, MeCo-based NAS achieves 83.14% accuracy, which is only 0.34% lower than the baseline method _β_ -DARTS [64], but five times more efficient in computation. 

We further visualize the networks searched by Zero-Cost-PT with different proxies on DARTS-CNN. The results on CIFAR-10 and CIFAR-100 are presented in Figure 6 and Figure 7, respectively. 

## **F.3 Results on MobileNet V3** 

We search the architectures on MobileNet space using Algorithm 1. We retrain the searched network using ImageNet-1K for 480 epochs with a batch size of 256 and input resolution 224 _×_ 224. The results are summarized in Table 15. MeCo achieves 77 _._ 8% Top-1 accuracy, which is 0 _._ 7% higher than the SOTA baseline method ZiCo. Moreover, our method only takes 0.04 GPU days, which outperforms the SOTA methods. 

## **G Limitations and future works** 

23 

Table 14: Comparison of our method with SOTA NAS methods using DARTS-CNN and CIFAR-100. 

|Approach|Test Error<br>(%)|Search Cost<br>(GPU Days)|Params<br>(MB)|Method|
|---|---|---|---|---|
|DenseNet-BC[65]|17_._18|-|25.6|Manual|
|NASNet-A[12]|16_._82|2000|3.3|MS|
|DARTS(1st)[6]|17_._76|1.5|3.3|OS|
|SNAS [66]|17_._55|1.5|2.8|OS|
|P-DARTS[64]|15_._92_±_0_._18|0.4|3.7|OS|
|R-DARTS[67]|18_._01_±_0_._26|-|-|OS|
|PC-DARTS[63]|16_._9|0.1|3.6|OS|
|_β_-DARTS[68]|16_._52_±_0_._03|0.4|3_._83_±_0_._08|OS|
|Zero-Cost-PT_†_<br>synfow [22]<br>Zero-Cost-PT_†_<br>fsher[23]<br>Zero-Cost-PT_†_<br>grasp [21]<br>Zero-Cost-PT_†_<br>jacov[7]<br>Zero-Cost-PT_†_<br>snip[20]<br>Zero-Cost-PT_†_<br>NTK[24]|19_._82_±_0_._35<br>21_._14_±_0_._24<br>22_._65_±_0_._30<br>22_._90_±_0_._35<br>19_._95_±_0_._28<br>20_._30_±_0_._33|0.04<br>0.06<br>0.13<br>0.04<br>0.04<br>0.19|1.2<br>0.7<br>0.7<br>0.6<br>0.8<br>0.9|ZS<br>ZS<br>ZS<br>ZS<br>ZS<br>ZS|
|Zero-Cost-PTZiCo_†_[9]|19_._54_±_0_._28|0.06|1.1|ZS|
|**Zero-Cost-PT**_†_<br>MeCo|**19**_._**33**_±_**0**_._**25**|0.08|0.8|ZS|
|**Zero-Cost-PT**_‡_<br>MeCo|**16**_._**86**_±_**0**_._**30**|0.08|3.7|ZS|



_†_ : networks initialized with 16 channels and trained as settings in 7. 

_‡_ : networks initialized with 36 channels and trained as settings in [63]. 

Table 15: Comparison of our method with SOTA NAS methods using MobileNet and ImageNet-1K. 

|**Approach**|**Top-1**<br>(**%**)|**Search Cost**<br>(**GPU Days**)|**Params**<br>(**M**)|**Method**|
|---|---|---|---|---|
|MobileNet-V3(1.0)|75.2|288|5.3|MS|
|PNAS|74.2|224|5.1|MS|
|DARTS|73.3|4|4.7|OS|
|PC-DARTS|75.8|3.8|-|OS|
|SPOS|74.7|8.3|-|OS|
|GreedyNAS|74.9|7.6|3.8|OS|
|TE-NAS|75.5|0.17|5.4|ZS|
|ZiCo|77.1|0.4|-|ZS|
|Zero-Cost-PT|76.4|0.04|8.0|ZS|
|Zen-score|76.1|0.5|-|ZS|
|**MeCo (Ours)**|**77.8**|**0.08**|7.9|ZS|



24 

Finally, we demonstrate the limitations of this work and the possible directions of our future work. In Section 5, We propose an optimization method to alleviate the channel-sensitive issue of MeCo. This weight-sampling approach can be further improved in future work. Moreover, we would like to point out that the evaluation of a zero-cost proxy is often tied to the availability of benchmarks. Hench though the proxies are “zero cost”, the evaluation of the proxies is strongly dependent on a benchmark, which in the first place is very expensive to create. Finally, although MeCo achieves the highest correlation with the test accuracy on multiple benchmarks, it shows near zero correlations on a few tasks (e.g., TransBench-101-Micro with Autoencoding). We will leave these issues as our future work. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0025-01.png)


**----- Start of picture text -----**<br>
NATS-Bench-TSS NATS-Bench-SSS<br>0 . 9 − 0 . 78<br>0 . 89 − 0 . 8<br>0 . 88 − 0 . 82<br>0 . 87 − 0 . 84<br>0 . 86 − 0 . 86<br>00 .. 8584 − 0 . 88<br>0 . 83 − 0 . 9<br>1 2 3 4 5 1 2 3 4 5<br># Samples # Samples<br>CIFAR-10; CIFAR-100; ImageNet16-120<br>⇢<br>**----- End of picture text -----**<br>


Figure 5: Spearman correlation coefficients of MeCo with different number of samples on NATSBench-TSS and NATS-Bench-SSS, respectively. 

25 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0026-00.png)


**----- Start of picture text -----**<br>
sep_conv_5x5<br>sep_conv_5x5<br>sep_conv_5x5 0<br>c_{k-2} sep_conv_5x5 3<br>sep_conv_5x5sep_conv_5x5sep_conv_5x5 21 c_{k}<br>c_{k-1} sep_conv_5x5sep_conv_5x5 1<br>sep_conv_5x5<br>sep_conv_5x5<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0026-01.png)


**----- Start of picture text -----**<br>
sep_conv_5x5 sep_conv_5x5<br>c_{k-2} sep_conv_5x5 sep_conv_5x5 0<br>sep_conv_5x5 3 c_{k-2} sep_conv_5x5 3<br>c_{k-1} sep_conv_5x5sep_conv_5x5sep_conv_5x5 sep_conv_5x520 sep_conv_5x5 1 c_{k} c_{k-1} sep_conv_5x5sep_conv_5x5sep_conv_5x5 21 c_{k}<br>sep_conv_5x5<br>Normal Cell<br>Reduction Cell<br>(a) synflow<br>c_{k-1}c_{k-2} avg_pool_3x3avg_pool_3x3avg_pool_3x3dil_conv_3x3 sep_conv_5x5avg_pool_3x301 dil_conv_3x3skip_connect 23 c_{k} c_{k-1}c_{k-2} sep_conv_5x5sep_conv_3x3 max_pool_3x3sep_conv_5x50 max_pool_3x3sep_conv_3x3sep_conv_3x3 13 dil_conv_5x5 2 c_{k}<br>Normal Cell Reduction Cell<br>(b) fisher<br>dil_conv_3x3<br>c_{k-1} sep_conv_5x5avg_pool_3x3dil_conv_5x5 dil_conv_3x31 sep_conv_3x3sep_conv_5x5sep_conv_3x3 23 c_{k} c_{k-2} sep_conv_3x3sep_conv_5x5avg_pool_3x3skip_connect 210 sep_conv_3x3 3 c_{k}<br>c_{k-2} sep_conv_3x3 0 c_{k-1} sep_conv_5x5<br>sep_conv_5x5<br>Normal Cell<br>Reduction Cell<br>(c) snip<br>sep_conv_5x5<br>c_{k-2}c_{k-1} sep_conv_5x5avg_pool_3x3avg_pool_3x3skip_connect dil_conv_3x310 sep_conv_3x3sep_conv_5x5sep_conv_3x3 32 c_{k} c_{k-2}c_{k-1} sep_conv_3x3avg_pool_3x3skip_connectskip_connectskip_connect max_pool_3x3201 sep_conv_3x3 3 c_{k}<br>Normal Cell<br>Reduction Cell<br>(d) grasp<br>c_{k-1}c_{k-2} max_pool_3x3dil_conv_3x3 dil_conv_3x30 sep_conv_3x3sep_conv_5x5dil_conv_3x3 1 sep_conv_3x3dil_conv_3x3 23 c_{k} c_{k-2}c_{k-1} max_pool_3x3max_pool_3x3sep_conv_3x3avg_pool_3x3 sep_conv_3x301 sep_conv_5x5dil_conv_3x3 2 sep_conv_3x3 3 c_{k}<br>Normal Cell Reduction Cell<br>(e) jacov<br>sep_conv_3x3<br>c_{k-2} dil_conv_5x5dil_conv_5x5dil_conv_3x3 03 dil_conv_5x5 2 c_{k} c_{k-1} sep_conv_3x3dil_conv_3x3 sep_conv_5x50 sep_conv_5x5 dil_conv_5x51 sep_conv_5x5 2 c_{k}<br>sep_conv_3x3 1 sep_conv_3x3 3<br>c_{k-1} sep_conv_5x5 c_{k-2} sep_conv_5x5<br>sep_conv_5x5<br>Reduction Cell<br>Normal Cell<br>(f) NTK<br>sep_conv_3x3<br>sep_conv_5x5 skip_connect<br>c_{k-2} sep_conv_5x5 sep_conv_5x5 0<br>sep_conv_5x5 3 c_{k-2} sep_conv_5x5 3<br>c_{k-1} sep_conv_5x5sep_conv_5x5sep_conv_3x3 sep_conv_5x520 sep_conv_5x5 1 c_{k} c_{k-1} sep_conv_5x5sep_conv_5x5sep_conv_5x5 21 c_{k}<br>sep_conv_5x5<br>Normal Cell<br>Reduction Cell<br>(g) ZiCo<br>**----- End of picture text -----**<br>


26 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0027-00.png)


**----- Start of picture text -----**<br>
sep_conv_5x5 dil_conv_5x5<br>c_{k-2} dil_conv_5x5skip_connect 1 sep_conv_5x5 c_{k-2} sep_conv_5x5sep_conv_5x5 0<br>dil_conv_5x5 0 sep_conv_3x3 3 c_{k} sep_conv_5x5 1 c_{k}<br>c_{k-1} sep_conv_5x5sep_conv_3x3 2 c_{k-1} dil_conv_5x5dil_conv_5x5 dil_conv_5x52 sep_conv_3x3 3<br>Normal Cell Reduction Cell<br>(h) MeCo<br>**----- End of picture text -----**<br>


Figure 6: Cells found by Zero-Cost-PT with all zero-cost proxies on the DARTS-CNN search space using CIFAR-10 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0027-02.png)


**----- Start of picture text -----**<br>
sep_conv_5x5<br>sep_conv_5x5 sep_conv_5x5<br>c_{k-2} sep_conv_5x5 sep_conv_5x5 0<br>sep_conv_5x5 3 c_{k-2} sep_conv_5x5 3<br>c_{k-1} sep_conv_5x5sep_conv_5x5sep_conv_5x5 sep_conv_5x520 sep_conv_5x5 1 c_{k} c_{k-1} sep_conv_5x5sep_conv_5x5sep_conv_5x5 21 c_{k}<br>sep_conv_5x5<br>Normal Cell<br>Reduction Cell<br>(a) synflow<br>dil_conv_5x5<br>c_{k-2}c_{k-1} max_pool_3x3max_pool_3x3sep_conv_3x3sep_conv_5x5skip_connect dil_conv_3x3301 skip_connect 2 c_{k} c_{k-2}c_{k-1} sep_conv_5x5sep_conv_3x3sep_conv_5x5sep_conv_3x3 max_pool_3x3sep_conv_3x302 sep_conv_3x3dil_conv_3x3 13 c_{k}<br>Reduction Cell<br>Normal Cell<br>(b) fisher<br>c_{k-1} skip_connect sep_conv_3x3 dil_conv_3x3 1 max_pool_3x3 3 c_{k-2} sep_conv_3x3skip_connect skip_connect 1<br>c_{k-2} sep_conv_5x5 skip_connect0 sep_conv_5x5sep_conv_3x3 2 c_{k} c_{k-1} sep_conv_5x5dil_conv_3x3 dil_conv_5x50 sep_conv_5x5skip_connect 23 c_{k}<br>Normal Cel Reduction Cell<br>(c) snip<br>avg_pool_3x3 dil_conv_3x3<br>c_{k-2}c_{k-1} max_pool_3x3sep_conv_3x3avg_pool_3x3avg_pool_3x3 10 sep_conv_3x3sep_conv_5x5 2 sep_conv_3x3 3 c_{k} c_{k-2}c_{k-1} max_pool_3x3sep_conv_3x3dil_conv_3x3skip_connect dil_conv_3x302 sep_conv_3x3dil_conv_3x3 13 c_{k}<br>Normal Cell Reduction Cell<br>(d) grasp<br>sep_conv_3x3<br>max_pool_3x3 0 c_{k-1} dil_conv_5x5<br>c_{k-2} avg_pool_3x3dil_conv_3x3 avg_pool_3x31 avg_pool_3x3 2 dil_conv_5x5 3 c_{k} sep_conv_5x5sep_conv_5x5dil_conv_3x3 23 c_{k}<br>c_{k-1} dil_conv_5x5 dil_conv_5x5 c_{k-2} skip_connect avg_pool_3x30 skip_connect 1<br>Normal Cell<br>Reduction Cell<br>(e) jacov<br>**----- End of picture text -----**<br>


27 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation_images/nips_2023_MeCo_Zero-Shot_NAS_with_One_Data_and_Single_Forward_Pass_via_Minimum_Eigenvalue_of_Correlation.pdf-0028-00.png)


**----- Start of picture text -----**<br>
dil_conv_5x5<br>c_{k-1} sep_conv_5x5dil_conv_5x5 0 c_{k-2} sep_conv_3x3<br>c_{k-2} dil_conv_5x5dil_conv_5x5 sep_conv_3x3dil_conv_3x31 sep_conv_5x5dil_conv_5x5 23 c_{k} c_{k-1} sep_conv_5x5sep_conv_5x5sep_conv_5x5sep_conv_5x5 sep_conv_3x3310 sep_conv_3x3 2 c_{k}<br>Normal Cell<br>Reduction Cell<br>(f) NTK<br>sep_conv_3x3 sep_conv_5x5<br>c_{k-2} sep_conv_3x3sep_conv_5x5sep_conv_3x3 12 sep_conv_3x3 3 c_{k} c_{k-2} sep_conv_5x5sep_conv_3x3sep_conv_3x3 03 sep_conv_3x3 2 c_{k}<br>sep_conv_5x5 0 sep_conv_5x5 1<br>c_{k-1} sep_conv_5x5 c_{k-1} sep_conv_3x3<br>sep_conv_5x5 dil_conv_5x5<br>Normal Cell Reduction Cell<br>(g) ZiCo<br>c_{k-2}c_{k-1} sep_conv_3x3dil_conv_3x3 sep_conv_5x5dil_conv_5x50 sep_conv_3x3dil_conv_5x5 sep_conv_5x513 sep_conv_3x3 2 c_{k} c_{k-2}c_{k-1} sep_conv_3x3sep_conv_3x3avg_pool_3x3dil_conv_3x3 sep_conv_5x530 sep_conv_3x3dil_conv_5x5 1 sep_conv_3x3 2 c_{k}<br>Normal Cell Reduction Cell<br>(h) MeCo<br>**----- End of picture text -----**<br>


Figure 7: Cells found by Zero-Cost-PT with all zero-cost proxies on the DARTS-CNN search space using CIFAR-100 

28 

