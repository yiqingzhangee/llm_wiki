# The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction 

Pratyusha Sharma[1] Jordan T. Ash[2] _[,⋆]_ Dipendra Misra[2] _[,⋆]_ 

MIT[1] Microsoft Research NYC[2] 

( _⋆_ equal advising) 

```
pratyusha@mit.edu,{ash.jordan,dimisra}@microsoft.com
```

## **Abstract** 

Transformer-based Large Language Models (LLMs) have become a fixture in modern machine learning. Correspondingly, significant resources are allocated towards research that aims to further advance this technology, typically resulting in models of increasing size that are trained on increasing amounts of data. This work, however, demonstrates the surprising result that it is often possible to significantly improve the performance of LLMs by selectively removing higher-order components[1] of their weight matrices. This simple intervention, which we call LAyer-SElective Rank reduction ( `LASER` ), can be done on a model after training has completed, and requires no additional parameters or data. We show extensive experiments demonstrating the generality of this finding across language models and datasets, and provide in-depth analyses offering insights into both when `LASER` is effective and the mechanism by which it operates[2] . 

## **1 Introduction** 

Since their original release, Transformer-based LLMs have been shown to be remarkably proficient on a wide array of important machine learning tasks. Their underlying Transformer architecture has become state-of-the-art for modeling and reasoning about natural language, and has shown promise in domains such as computer vision [Dosovitskiy et al., 2020] and reinforcement learning [Chen et al., 2021] as well. 

Contemporary instantiations of Transformer architectures are infamously large, typically requiring tremendous compute resources for both training and inference. This is by design, as Transformers trained with more parameters or data are demonstrably more capable than their slimmer predecessors—often by a significant margin [Brown et al., 2020, Touvron et al., 2023]. Still, a growing body of work suggests that Transformerbased models, and neural networks more generally, do not require all fitted parameters to retain their learned hypotheses. While it seems helpful to be massively over-parameterized at train time [Hinton et al., 2015, Bengio et al., 2005], it is well-known that these models can be drastically pruned before inference; neural networks can often have well over 90% of their weights removed without any significant degradation in performance [Frankle and Carbin, 2018]. The discovery of this phenomenon bolstered interest around the relationship between generalization and over-parametrization [Zhang et al., 2017], and spawned research in developing pruning strategies that lend themselves to efficient model inference [Molchanov et al., 2016]. 

This paper presents a surprising finding, that careful pruning done at specific layers of Transformer models can produce significant boosts in performance on some tasks. We describe LAyer SElective Rank reduction ( `LASER` ), an intervention that removes higher-order components of learned weight matrices as identified by singular value decomposition. This reduction is performed in specific weight matrices and layers of the Transformer model. In line with previous work, we find that many such matrices can be significantly reduced, 

> 1Higher-order components are singular vectors with smaller singular values. 

> 2Code and website: https://pratyushasharma.github.io/laser/ 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction_images/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction.pdf-0002-00.png)


**----- Start of picture text -----**<br>
LASER Σ V [⊤] Linear Layer<br>Replace W in specific layers with  W WLR = U Self-Attention Layer<br>its low rank approximation WLR Nonlinear activation<br>+ Point-wise Addition<br>Self-Attention MLP  Layer L<br>Capital + + + + + +<br>of + + + + + +<br>Japan + + + + + +<br>Tokyo<br>is  + + + + + +<br>**----- End of picture text -----**<br>


Figure 1: LAyer SElective Rank reduction ( `LASER` ) replaces a specific weight matrix _W_ of the Transformer model by its rank- _k_ approximation, _W_ LR, and observes the change in the model’s behavior. We find that this rank approximation, especially for MLP weights in the later layers of the model, often offers surprising benefits to model performance. 

and that performance degradation is often not observed until well over 90% of components are entirely removed. However, unlike what is found in previous work, we find that these reductions can produce drastic improvements in accuracy, as measured by various well-studied reasoning benchmarks in NLP. Even better, this discovery appears to not be limited to natural language, with performance gains also found in reinforcement learning. 

This paper analyzes the relationship between the model’s training data and samples that benefit from `LASER` . We find that improvements in the model’s performance predominantly come on information less frequently present in the model’s training dataset, suggesting that `LASER` offers a kind of denoising procedure that makes weakly learned facts accessible. We separately observe that `LASER` affords increased robustness to paraphrases on previously correct questions. 

Additionally, we attempt to reason about what is being stored in high-order components, such that their removal boosts performance. For questions correctly answered only after `LASER` , in the absence of interventions, the original model predominantly responds with high-frequency words such as “the,” “of,” etc—generations that are not even the same semantic type as the correct answer. However, after some amount of rank reduction, the model’s answer flips to be correct. To understand this, we look at what the remaining components encode on their own; we approximate the weight matrix using only its higher-order singular vectors. We find that these components describe either a different response of the same semantic category as the correct answer or generic high-frequency words. Seemingly, when the noisy, higher-order components are combined with the low-order components, their conflicting responses produce a sort of “average answer,” which is likely to be incorrect. 

Figure 1 visualizes the Transformer architecture and the procedure followed by `LASER` . Here, the weight matrix of a Multi-Layer Perceptron (MLP) at a specific layer is replaced with its low-rank approximation. 

## **2 Related work** 

To our knowledge, this paper is the first to identify that carefully selected rank reductions can boost Transformer performance. Still, there is a wide array of works that study related questions, including how facts are stored in LLMs and how to best compress neural networks. 

**How facts are stored.** Studies probing model representation for the presence of select properties of entities [Ettinger et al., 2016, Adi et al., 2016, Hupkes et al., 2018, Conneau et al., 2018] show that models store 

2 

factual information across different layers and Lee et al. [2023] show that model robustness to distribution shift can be improved by fine-tuning select layers. However, there is conflicting evidence on how this information is organized and utilized in constructing answers in large language models. Some theories posit that information about different entities is locally stored as two-layer, key-value memory in MLP sections of Transformer models [Geva et al., 2021], which are thereafter copied over through latter layers by the self-attention modules [Elhage, 2021]. Meng et al. [2022] propose a procedure to trace and edit local, entity-specific information to map to distinct “impossible” outputs, supporting the locality theory. These theories are further supported by the phenomenon of “early exiting,” where the representation at an intermediate layer can be directly used with the terminal head of the model to correctly generate an output [Zhao et al., 2021]. In contrast, Hase et al. [2023] has observed that information about some of the same entities or entity relations can be modified by editing a variety of layers in the model architecture, and therefore, that facts are stored across layers in a fragmented fashion. This paper makes no specific claims regarding locality, but rather demonstrates that higher-order components of a weight matrix can introduce noise in decision making, and considering only lower-order components may make correct answers accessible. 

**Model compression.** Neural network pruning methods have found that models could be significantly pruned (often removing over 90% of parameters) with very little drop in accuracy, significantly reducing the storage requirements of the model [LeCun et al., 1989, Hassibi and Stork, 1992, Han et al., 2015, Li et al., 2016, Frankle and Carbin, 2018]. There have also been approaches that prune these models in a structured manner, to facilitate improvements in inference time [Molchanov et al., 2016]. The existence of sparse sub-networks [Frankle and Carbin, 2018, Hoefler et al., 2021] has been found to be true for convolutional, fully connected, and Transformer models [Lv et al., 2023, Murty et al., 2022]. While Jin et al. [2022] find that model generalization can be improved by pruning and then refitting parameters, generalization improvements are only observed upon model retraining. To our knowledge, model pruning techniques have always done a unilateral reduction across all parameters, without targeting any specific layers—leading to predictive performance either staying fixed or decreasing [Frankle and Carbin, 2018]. In this work, however, we find that the effect of reduction in accuracy is non-uniform across different layer types, and a model’s generalization can be improved by _selective_ pruning alone; no additional training is necessary. Roughly, we find that performance degradation can be produced by rank-reducing early layers, while significant performance benefits are typically available by pruning later layers. 

**Low-rank approximations of weight matrices.** Most pruning methods reduce parameters in order of their absolute magnitude [Frankle and Carbin, 2018]. An alternative approach, however, is to reduce the rank of its constituent weight matrices, keeping the top _k_ components found by SVD. While matrices of neural models, including Transformer models, have been found to be well-approximated using this approach, where markedly reduced versions of the model can preserve its behavior, research has shown that performance eventually declines as the severity of the intervention increases [Lv et al., 2023, Hajimolahoseini et al., 2021, Yu et al., 2017]. Note that these reductions are typically done unilaterally, removing the same number of components in every weight matrix in the model. In contrast to these findings, we show that a targeted rank reduction, even affecting just a single weight matrix, can offer benefits to the predictive accuracy of Transformers. 

**Model distillation and low-rank training.** Ba and Caruana [2014] and Hinton et al. [2015] have trained smaller networks to mimic the behavior of larger networks, suggesting neural networks might be significantly over-parametrized and can be replaced with leaner alternatives. To our knowledge, no report of an improvement in the model’s predictions as a consequence of this procedure has been shown. [Yang et al., 2020] have enforced low-rank-ness of weight matrices for the purposes of memory efficiency, but the resulting models fail to achieve performance equivalent to their overparametrized counterparts. The result suggests that overparametrization is helpful for the identification of well-generalizing parameters by SGD [Bengio et al., 2005, Hinton et al., 2015, Zhang et al., 2017]. 

3 

## **3 Preliminaries** 

Here we review basic notation and describe the core components of our study. 

**Maths Notation.** We use R to denote real numbers, N to denote natural numbers, small letters such as _v ∈_ R _[d]_ to denote a _d_ -dimensional vector, and capital letters such as _W ∈_ R _[m][×][n]_ to denote a matrix of size _m × n_ . We use _∥v∥_ 2 to denote the Euclidean norm of a vector _v_ and _∥W ∥_ 2 to denote the spectral norm of a matrix _W_ . We use [ _N_ ] to denote the set _{_ 1 _,_ 2 _, · · · , N }_ . We will use `rank` ( _W_ ) to denote the rank of a matrix _W_ and _σi[↓]_[(] _[W]_[)][to][denote][its] _[i][th]_[largest][singular][value.] 

**Transformer Architecture.** We provide a concise description of vanilla Transformer architecture that is relevant to our analysis. A Transformer architecture can be thought of as _L_ layers of Transformer blocks. The _l[th]_ block maps a sequence of _T_ -length vector sequence ( _h_[(] 1 _[l][−]_[1)] _, · · · , h_[(] _T[l][−]_[1)] ) to another _T_ -length vector sequence ( _h_[(] 1 _[l]_[)] _[,][ · · ·][, h]_[(] _T[l]_[)][)][,][where][all][vectors][are] _[d]_[-dimensional.][This][transformation][is][accomplished][using][two] sequential steps: a self-attention mechanism to _mix_ information across time steps, and a feed-forward network to _process_ information within each time step. We describe a basic version of these transformations for a fixed _l[th]_ layer and drop the superscript ( _l −_ 1) for clarity.[3] A single-head self-attention mechanism first maps each vector _hi_ to a query vector _qi_ = _Wqhi_ , a key vector _ki_ = _Wkhi_ and a value vector _vi_ = _Wvhi_ where _Wq, Wk, Wv ∈_ R _[d][×][d]_ are layer-specific weight matrices. We then compute attention probabilities _p_ ( _j | i_ ) = � _Tl_ ex=1p([exp(] _qi[⊤][k][q][j] i[⊤][/][k]_ ~~_√_~~ _[l][/] d_ ~~_√_~~ ) _d_ )[for][every] _[i, j][∈]_[[] _[T]_[]][.][These][are][used][to] compute the attention vector _zi_ =[�] _[T] j_ =1 _[p]_[(] _[j][|][i]_[)] _[v][j]_[.][A] _[k]_[-head][self-attention][computes][a][set][of] _[k]_[attention] vectors by using different linear transformations for key, query, and value, and then concatenates these attention vectors. These _k_ -separate linear transformations for key, query, and value can all be absorbed into their respective matrices _Wq ∈_ R _[d][×][dk] , Wk ∈_ R _[d][×][dk]_ and _Wv ∈_ R _[d][×][dk]_ . Finally, the self-attention mechanism outputs _ui_ = _ziWo_ + _hi_ using a projection matrix _Wo ∈_ R _[dk][×][d]_ . 

The feed-forward step applies a 2-layer multi-layer perception (MLP) _ψ_ : R _[d] →_ R _[d]_ to each vector _ui ∈_ R _[d]_ separately. The MLP typically has a ReLU or GELU activation function [Hendrycks and Gimpel, 2016] and in some models such as Llama, the bias of linear layers is set to 0. We denote the weight matrices of the first and second linear layers of this MLP by _Uin_ and _Uout_ respectively. The output of this _l[th]_ Transformer block is then given by _h_[(] _i[l]_[)] = _ψ_ ( _ui_ ) + _ui_ . 

In summary, a Transformer architecture has the following weight matrices _W_ = _{Wq, Wk, Wv, Wo, Uin, Uout}_ for each layer, in addition to the embedding matrix for embedding input tokens, a projection weight matrix applied after the final layer before taking softmax, and all weight matrices associated with layer normalization. In our work, we will focus primarily on the matrices in _W_ and intervene by modifying them. 

**Rank-** _r_ **Approximation and SVD.** Given a matrix _W ∈_ R _[m][×][n]_ and _r ∈_ N, a rank- _r_ approximation prob� lem requires finding a matrix _W_[ˆ] that minimizes _∥W −W_[�] _∥_ 2 and satisfies `rank` _W ≤ r_ . Eckart–Young–Mirsky � � theorem provides an optimal solution of this problem using Singular Value Decomposition (SVD) [Eckart and Young, 1936]. Formally, an SVD of a matrix _W_ is given by _W_ = _U_ Σ _V[⊤]_ where _U_ = [ _u_ 1 _, u_ 2 _, · · · , um_ ] _∈_ R _[m][×][m]_ and _V_ = [ _v_ 1 _, v_ 2 _, · · · , vn_ ] _∈_ R _[n][×][n]_ and Σ _∈_ R _[m][×][n]_ . The column vectors of _U_ and _V_ constitute an orthonormal basis of R _[m]_ and R _[n]_ respectively, and Σ is a diagonal matrix whose diagonal entries are given by the singular values of _W_ in descending order. One can also express the SVD of _W_ as _W_ =[�] _i_[min] =1 _[{][m,n][}] σi[↓]_[(] _[W]_[)] _[u][i][v] i[⊤]_[.] According to Eckart–Young–Mirsky theorem, the matrix _W_[�] =[�] _[r] i_ =1 _[σ] i[↓]_[(] _[W]_[)] _[u][i][v] i[⊤]_[is][an][optimal][solution][to][the] rank- _r_ approximation problem for any given desired rank _r ≤_ min _{m, n}_ . 

In this work, we will use the word **higher-order components** to refer to entries in the SVD corresponding to the components with smaller singular values. These components are removed by `LASER` . The term **lowerorder components** is used to refer to singular vectors corresponding to large singular values. These components are kept in a low-rank approximation of the matrix. 

> 3Various Transformer models often have small differences in how these transformations are implemented. Our goal is not to provide a full survey of these details but to capture essential terminology for our results. 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction_images/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Layer number Layer number Layer number<br>Reduction %: 10 25 40 50 60 75 90 92.5 95 97.5 98 98.5 99 99.5 99.75<br>Loss<br>Loss<br>**----- End of picture text -----**<br>


Figure 2: The effect of rank reduction across different layer types is not uniform. Here we show the effect of rank reduction for GPT-J as studied on the CounterFact dataset. The dashed line is the um-modified network’s loss. In the attention layers (key, query, value, out matrices), while it is clear matrices could be significantly rank-reduced without damaging the learned hypothesis, there is very little performance increase. However, for the multi-layer perceptron (MLP) layers, rank reduction goes from uniformly harming to improving the model’s performance (around layer 20). 

## **4 LAyer SElective Rank Reduction (** `LASER` **)** 

In this section, we formally describe the `LASER` intervention. A single-step `LASER` intervention is defined by three quantities ( _τ, ℓ, ρ_ ), consisting of a parameter type _τ_ , layer number _ℓ_ , and rank reduction _ρ_ . These values together describe which matrix will be replaced by their low-rank approximation and how severe the approximation will be. The parameter type categorizes the matrix type in which we will intervene. We focus on the matrices in _W_ = _{Wq, Wk, Wv, Wo, Uin, Uout}_ which consist of the matrices in the MLP and attention layers. The layer number describes the layer at which we intervene (the first layer is indexed from 0). E.g., the Llama-2 has 32 layers and so _ℓ ∈{_ 0 _,_ 1 _,_ 2 _, · · ·_ 31 _}_ . Finally, _ρ ∈_ [0 _,_ 1) describes what fraction of the maximum rank should be preserved upon doing its low-rank approximation. For example, let _τ_ = _Uin ∈_ R _[d][×][d]_ , then the maximum rank of this matrix is _d_ . We replace it with a rank _⌊ρ · d⌋_ -approximation. 

Figure 1 shows an example of `LASER` . In this figure, we have _τ_ = _Uin_ and _ℓ_ = _L_ indicating that we update the weight matrix in the first layer of MLP in the Transformer block of the _L[th]_ layer. The other parameter (not shown in the figure) controls the _k_ in the rank- _k_ approximation. 

`LASER` throttles the flow of certain information in the network, which surprisingly can produce significant performance benefits. These interventions can also be easily composed—we can apply a set of interventions _{_ ( _τi, ℓi, ρi_ ) _}[m] i_ =1[in][any][order.][The] `[LASER]`[approach][is][to][simply][search][over][interventions][of][this][type,][and] to exercise the modification that offers the greatest benefit. There are many other ways in which one can combine these interventions, however, and we defer a this to future work. 

## **5 Experiments** 

This section studies the consequences of `LASER` throughout various layers of the Transformer architecture. We first perform a motivating analysis of the CounterFact [Meng et al., 2022] question-answering dataset in 

5 

conjunction with a pretrained GPT-J model [Wang and Komatsuzaki, 2021], and investigate the performance of the model and its variability as we search over potential interventions. Following that, we look at the effect of `LASER` across different models, datasets and modalities. 

**GPT-J, CounterFact and PILE.** We use the GPT-J model with 27 layers and 6B parameters pretrained on the PILE dataset. The first part of our analysis focuses on GPT-J, largely because its training data is publicly available. We evaluate the model’s behavior on the CounterFact dataset, which consists of samples organized as (subject, relation, answer) tuples and three paraphrased prompts for each question. For example, (Danielle Darrieux, mother tongue, French). 

## **5.1 A Thorough Analysis with GPT-J on the CounterFact Dataset** 

Figure 2 shows the result of applying various amounts of rank reduction to each matrix in the Transformer architecture on the classification loss for this dataset. These plots are grouped, such that each sub-figure corresponds only to the indicated type of weight matrices. Note that each Transformer layer consists of a small, two-layer MLP. The constituent input and output matrices are shown separately. Different colors indicate different percentages of removed components. 

The attention plots in this figure exemplify what is already known about these models: weight matrices can be drastically reduced without much degradation in model performance. The more interesting result, however, is in the MLP layers. Here, not only can matrices be rank-reduced without degrading classification performance, but large performance improvements are possible by reducing later layers of the model. This trend is most stark in the input matrix of the MLP. While there are gains with `LASER` in the attention layers too, the benefits are typically smaller. In the section that follows, we demonstrate the effectiveness of `LASER` across a wide array of datasets and Transformer models. Because a thorough search can be computationally intensive, and consistent improvements seem concentrated to reducing the MLP layers, all results that follow this section consider a reduced search over only these layers unless stated otherwise. 

**Improved accuracy and robustness to paraphrases.** The CounterFact dataset is used to test the model’s factual knowledge of data from Wikipedia. Since GPT-J is trained on PILE, whose contents include Wikidata, different facts in CounterFact are part of the model’s training data, albeit in different quantities. As all answers are a single token in this setting, we compute top-k accuracy based on whether the correct answer is in the top-k predicted tokens. As seen in Figure 2 and Table 1, we find that the model’s top-1 accuracy on facts in CounterFact increases from 13.3% to 24.1% when reductions are done on a single layer. It is important to note that these improvements are a result of rank-reduction alone, and do not involve any further training or fine-tuning of the pre-trained GPT-J model. Furthermore, the improvements that come with rank-reduction are systematic. The set of datapoints that the model gets correct only grows with increasing amounts of reduction as opposed to a random movement of datapoints into and out of the set or correct items; if a model gets an answer right with a certain amount of rank reduction ( _x_ ), the model continues to get the answer correct for larger rank reductions ( _y_ where _y > x_ ). We evaluate the model’s robustness to paraphrases by computing the percentage of datapoints where the model gets all paraphrases of a given question correct. For datapoints that the model already gets correct, the model’s robustness to paraphrases also improves with `LASER` by roughly 24.8 percentage points. 

**Effect on language modeling and fluency.** While the model’s factuality improves, does the reduction affect the model’s performance on other metrics? To understand this, we evaluate the model’s perplexity, i.e., its original training objective, on its training data. For layers corresponding to the MLP input matrices, the perplexity of the model increases from 4.8 to 5.0, showing that the language modeling objective is indeed slightly effected. For the MLP output layers, the perplexity of GPT-J on PILE increases from 4.8 to 4.9 with `LASER` . It may be possible to fix this small degradation by calibrating the temperature of the model. 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction_images/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction.pdf-0007-00.png)


**----- Start of picture text -----**<br>
0.200<br>0.175<br>0.150<br>0.125<br>0.100<br>0.075<br>With LASER<br>0.050<br>Original model<br>0.025<br><50 50-1000 1000<<br>N  = Frequency of           in N<br>(a) (b) (c)<br>Accuracy<br>Answers corrected / Originally correct<br>**----- End of picture text -----**<br>


Figure 3: Which datapoints benefit from `LASER` ? We analyze how frequently in the training data “corrected” facts occur. GPT-J is an ideal test bed for such analysis since its training data ( _DT rain_ ), the PILE dataset, is publicly available. (a) For GPT-J evaluated on CounterFact ( _DQA_ ) we retrieve all the datapoints in _DT rain_ that contain a mention of both the entity of interest and the answer that correspond to each sample in _DQA_ . (b) A plot depicting the cumulative top-10 accuracy of the model on all datapoints that occur in the training data less than or equal to the frequency indicated on the x-axis. Here we show accuracy with and without `LASER` . (c) The largest boost in performance occurs for low-frequency samples. This bar chart displays the amount of boost offered by `LASER` for data binned by the frequency with which corresponding facts occur in _DT rain_ . Maximal improvements in accuracy are from datapoints that have less-frequent occurrences in training data. 

**Composing reductions across layers.** We find that even further improvements in the model’s performance can be made by performing different amounts of rank reduction across several layers. This is done by greedily searching over ( _τ, ℓ, ρ_ ) starting from the largest _ℓ_ and smallest _ρ_ . To speed things up, here we do this search only over MLP layers, as this is where the largest improvements are typically found. Consistent with other experiments, the search is done on a validation set, and results are reported on the test set. On CounterFact, the 0-1 accuracy of the base GPT-J model is 13.1%. After doing the best single-step `LASER` the model’s accuracy improved to 24.0%. Performing `LASER` across different layers improved the top-10 accuracy to 29.2%, a 5.2% absolute improvement in accuracy over performing `LASER` on a single layer. The results of the combinatorial search across different _ℓ_ and _ρ_ values can be seen in Figure 4. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction_images/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction.pdf-0007-03.png)


Figure 4: Composing `LASER` operations across multiple layers further enhances model performance. Here we show how accuracy improves for using a simple composing strategy for both validation data, which was used to identify each ( _τ, ℓ, ρ_ ), and test data. 

## **5.1.1 Which facts in the dataset are recovered by rank reduction?** 

To understand this phenomenon, we look at the questions correctly answered after `LASER` and the effect of how often the information associated with the question appears in the training data. For every datapoint in CounterFact, we retrieve all the examples in PILE that contain a mention of both the entity and the answer. We then compute how often information associated with each evaluation question appears in the training 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction_images/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction.pdf-0008-00.png)


**----- Start of picture text -----**<br>
(a) Paul Citroen is a native speaker of ______ (b)<br>Dutch Correct answer the<br>Dutch Average answer the<br>W the French W<br>Wrong answer<br>the French<br>the Average answer French<br>Top Bottom Top Bottom<br>Saeed Akhtar Mirza is originally from  Mumbai  Pakistan  The twin city of Wellington is  Sydney  Of<br>The original language of Hussar Ballad is  Russian  Portuguese  Kharkiv is a twin city of  Warsaw  The<br>Kalabhra follows the religion of  Buddhism  Hindu  The native language of Isaac Massa is  Dutch  The<br>Emmanuelle Devos's profession is a  Actor  Teacher  The headquarter of Morr Music is located in  Berlin  the<br>Walter Zenga is a professional  Soccer  Photographer  Abba Eban was employed in  Jerusalem  The<br>Mike Holmgren plays in the position of Quarterback Goalkeeper (c) Yizhar Harari speaks  Hebrew  To<br>Average         Correct Average          Wrong<br>**----- End of picture text -----**<br>


Figure 5: (a) [ _Left_ ] `LASER` approximates learned matrices by their lower-order components. We find that for datapoints where the model’s predictions improve after `LASER` , if we instead use the entire matrix (including higher-order components), the model often predicts only “generic” words. (a) [ _Right_ ] To understand what these higher-order components encode, we approximate the learned weight matrix with the higher-order components instead. We find that these higher-order components sometimes encode the correct semantic type of the answer but the incorrect response. (b) Analytically, computing the semantic similarity (cosine distance between the true answer and the answers generated by the bottom k% of the singular vectors) shows that on average the answer computed by the higher-order components is more similar to the real answer. (c) Shows some examples from the dataset and the corresponding answers computed by the top fraction and bottom fraction of the components. 

data. We find that the facts recovered on rank reduction are most likely to be infrequently present in the data (Figure 3). Here, “Originally correct” describes samples that are correctly classified even without any intervention. “Answer-corrected” refers to questions the model gets correct only after intervening with `LASER` . 

## **5.1.2 What are higher-order components storing?** 

We saw above how retaining the lower-order components improves model performance on the task of openended question answering. We find that for the task of question answering, improvements often come on questions whose answers are supported by less frequently occurring data in the training set. While it is clear that eliminating the higher-order components “denoises” the model and helps recover “hidden,” less-frequent information, it is not clear what the higher-order components are representing such that that their removal improves performance. This section studies this question using the CounterFact dataset and GPT-J. 

To understand what higher-order components are representing, we approximate the final weight matrix using its higher-order components (as opposed to approximating it using its lower-order components as done by `LASER` ) as shown in Figure 5(a). Following this, we analyze how the model’s behavior changes on data points that GPT-J originally gets incorrect but are flipped to being correct upon performing `LASER` . 

First, we note that when the original, unmodified model does not answer these questions correctly, it often responds with common words, such as “a,” “the,” “of,” and other highly frequent tokens. After performing `LASER` , where we retain only the top-k components, the model’s answers to these questions flip from generic words to the correct entity. For the same datapoints, when we approximate the model by instead retaining the higher-order components, we find that the model either predicts incorrect entities that are of the same semantic type as the correct answer or high-frequency tokens such as “a,” “the,” and “of,” as shown in Figure 5(c). However, as we systematically include the lower-order components, the model’s output changes to predicting frequent tokens. To investigate this systematic degradation, we measure the average 

8 

|**Dataset**|**Model Name**|**Model Name**|**Model Name**|
|---|---|---|---|
|||||
||**Roberta**<br>**GPT-J**<br>**LLama2**<br>`LASER`<br>`LASER`<br>`LASER`|||
|CounterFact<br>Acc<br>Loss|17.3<br>**19.3**<br>5.78<br>**5.43**|13.1<br>**24.0**<br>5.78<br>**5.05**|35.6<br>**37.6**<br>3.61<br>**3.49**|
|HotPotQA<br>Acc<br>Loss|6.1<br>**6.7**<br>10.99<br>**10.53**|**19.6**<br>19.5<br>3.40<br>**3.39**|16.5<br>**17.2**<br>3.15<br>**2.97**|
|FEVER<br>Acc<br>Loss|50.0<br>**52.3**<br>2.5<br>**1.76**|50.2<br>**56.2**<br>**1.24**<br>1.27|59.3<br>**64.5**<br>1.02<br>**0.91**|
|Bios Gender<br>Acc<br>Loss|87.5<br>**93.7**<br>**0.87**<br>1.13|70.9<br>**97.5**<br>**3.86**<br>4.20|75.5<br>**88.4**<br>3.48<br>**2.93**|
|Bios Profession<br>Acc<br>Loss|64.5<br>**72.5**<br>**4.91**<br>6.44|75.6<br>**82.1**<br>**4.64**<br>4.91|85.0<br>**86.7**<br>4.19<br>**4.05**|
|TruthfulQA<br>Acc<br>Loss|56.2<br>56.2<br>1.60<br>**1.42**|54.9<br>**55.6**<br>1.02<br>**1.01**|50.5<br>**56.2**<br>**0.95**<br>1.04|
|BigBench-Epistemic Reasoning<br>Acc<br>Loss|37.1<br>**41.8**<br>9.39<br>**6.80**|37.1<br>**38.3**<br>0.74<br>**0.62**|44.8<br>**63.4**<br>0.78<br>**0.73**|
|BigBench-WikidataQA<br>Acc<br>Loss|28.0<br>**30.7**<br>9.07<br>**7.69**|51.8<br>**65.9**<br>3.52<br>**2.86**|59.5<br>**62.0**<br>2.40<br>**2.31**|



Table 1: The effect of `LASER` intervention on eight natural language understanding datasets. We find the best `LASER` intervention for each model and task using accuracy/0-1 on a validation set and report its performance on a held-out test set. In some of the cases, while the model’s accuracy improves, its loss slightly worsens. 

cosine similarity of the “true” answer with respect to the predicted answer when the matrix is approximated with different amounts of higher-order components, as shown in Figure 5(b). The average cosine similarity between the predicted answer worsens, demonstrating this effect. 

We hypothesize that these matrices often encode multiple conflicting responses, and that when all components are used they clash to produce a generic token. Removing the higher-order components, which anecdotally appear to often capture incorrect responses of the correct type, resolves this internal conflict and allows the model to respond accurately. 

## **5.2 How generally does this hold?** 

We evaluate the generality of our findings on 3 different LLMs for several language understanding tasks. 

**Natural Language Understanding Tasks.** We evaluate model performance before and after `LASER` on seven datasets, including CounterFact [Meng et al., 2022], HotPotQA [Yang et al., 2018], FEVER [Thorne et al., 2018], Bias in Bios [De-Arteaga et al., 2019] [Gender and Profession], TruthfulQA [Lin et al., 2021], BigBench-Epistemic Reasoning [Bowman et al., 2015] and BigBench-WikidataQA. These datasets evaluate different aspects of language understanding problems. CounterFact, Fever, and Bigbench-Wiki data test a model’s world knowledge and factuality. Bias in Bios benchmarks model bias by predicting the gender and profession of a person given a short biography. We define Bios Gender as the gender prediction problem in Bias in Bios, and Bios Profession as the profession prediction problem. HotPotQA provides a more challenging open-ended question answering task with long answers containing many tokens. The Epistemic Reasoning dataset from Big Bench Hard (BBH) tests a model’s logic and reading comprehension. Finally, TruthfulQA tests an LLM’s truthfulness.We use 20% of the dataset as validation set and select the best `LASER` hyperparameters ( _τ, ℓ, ρ_ ) using this validation set. We report results on the remaining 80% of the dataset with the chosen hyperparameter. The models used for the task of question answering include, Roberta [Liu et al., 2020], GPT-J (6B) [Wang and Komatsuzaki, 2021], and LLAMA2 (7B) [Touvron et al., 2023]. Details regarding datasets and how they were used can be found in Appendix A. 

9 

**Evaluation metrics.** For each of these tasks, we evaluate the model’s performance using (i) **generation accuracy.** We generate a sequence of _N_ tokens using the LLM and then report 1 if the answer text is in the generated text and 0 otherwise, (ii) **classification accuracy** . If the answer lies in a small set of potential values, like in a standard classification problem, we consider a response correct if it puts more probability mass on the correct answer than on any of the other candidates, and (iii) **loss.** We report the log-loss on held-out data. For datasets with a small set of possible labels, we report the accuracy ( **acc** ) using classification accuracy, whereas for others we use the generation accuracy. 

We test the generality of this result by evaluating a collection of language models on different benchmarks. As seen in Table 1, we find that even severe reductions result in no deterioration in the model’s accuracy and can lead to improvements in their performance. The amount of reduction required differs from model to model. 

## **5.3 Non-text domains** 

To understand if this phenomenon is effective outside of question answering in the textual domain, we evaluate the effect of rank reduction on a reinforcement learning agent. 

**Policy learning.** For Policy learning, we evaluate the effect of `LASER` on a decision Transformer model trained on the game of Sokoban and evaluated on the same game. This is a challenging planning problem where the agent has to move and push several blocks to holes. The task is completed when all blocks are on top of holes. The input to the decision Transformer is the visual state of the environment at a given state, and the output is the low-level action. We find that for a decision Transformer trained on Sokoban, models solved 3% more tasks with `LASER` (Table 2). Details of the experiment can be found in Appendix B. 

||Model Name<br>Transformer<br>**with** `LASER`|Acc.<br>50.67<br>**53**|Return<br>0.575<br>**0.965**|
|---|---|---|---|



Table 2: Effect on `LASER` on a 6-layer Decision Transformer agent. The base model is trained and evaluated in a challenging 10 _×_ 10 Sokoban domain. 

Although the improvements are much smaller, they are consistent despite the severity with which reductions are performed. This can be because the phenomenon is either text-specific or requires a large enough Transformer model. 

## **6 Conclusion and Discussion** 

This paper describes `LASER` , a phenomenon where performing a low-rank approximation of specific layer types at specific layers of the transformer block can improve the performance of LLMs on the task of question answering. We find this to be true across five different datasets and three different language model models. Furthermore, the resulting `LASER` reductions are extreme. The matrices are reduced at times to 99% of their original rank, which is much lower than their effective rank (C.1). However, despite extreme reductions, the performance of the model on tasks continues to improve. We also observe performance gains for a decision Transformer in an embodied domain. We find that the largest improvements in the model accuracy correspond to information that is less common in the training data and that `LASER` jointly makes the model more robust to paraphrases of the questions. We further found that the higher-order components of some of these matrices encode either high-frequency words or alternate answers of the same semantic type as the correct answer. These noisy, higher-order components can overpower the stable lower-order components and result in the model answering questions incorrectly. In these cases, performing `LASER` acts as a denoising technique and reduces the internal conflicts in potential responses. 

Despite this analysis, the success of `LASER` requires further study. Learning (i) why higher-order components in weight matrices accumulate noisy answers in the course of training, (ii) the effect of model architecture and other structural choices on the occurence of this phenomenon and (iii) why this is specifically true for later layers in the MLP is important to not only for our understanding of the success of `LASER` , but for understanding the behavior of large language models more generally. 

10 

## **Acknowledgements** 

This work was done when PS was an intern at Microsoft Research, New York City, with DM and JTA. The authors would like to thank Minyoung Huh, Shikhar Murty, Han Guo, Cyril Zhang, David Bau, Jacob Andreas, Antonio Torralba, and John Langford for helpful discussions and paper feedback. 

## **References** 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv preprint arXiv:2010.11929_ , 2020. 

- Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. _Advances in neural information processing systems_ , 34:15084–15097, 2021. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 

- Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. Distilling the knowledge in a neural network. _ArXiv_ , abs/1503.02531, 2015. URL `https://api.semanticscholar.org/CorpusID:7200347` . 

- Yoshua Bengio, Nicolas Roux, Pascal Vincent, Olivier Delalleau, and Patrice Marcotte. Convex neural networks. In Y. Weiss, B. Schölkopf, and J. Platt, editors, _Advances in Neural Information Processing Systems_ , volume 18. MIT Press, 2005. URL `https://proceedings.neurips.cc/paper_files/paper/ 2005/file/0fc170ecbb8ff1afb2c6de48ea5343e7-Paper.pdf` . 

- Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. _arXiv: Learning_ , 2018. URL `https://api.semanticscholar.org/CorpusID:53388625` . 

- Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In _International Conference on Learning Representations_ , 2017. 

- Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, and Jan Kautz. Pruning convolutional neural networks for resource efficient inference. _arXiv preprint arXiv:1611.06440_ , 2016. 

- Allyson Ettinger, Ahmed Elgohary, and Philip Resnik. Probing for semantic evidence of composition by means of simple classification tasks. In _Proceedings of the 1st Workshop on Evaluating Vector-Space Representations for NLP_ , pages 134–139, Berlin, Germany, August 2016. Association for Computational Linguistics. doi: 10.18653/v1/W16-2524. URL `https://aclanthology.org/W16-2524` . 

- Yossi Adi, Einat Kermany, Yonatan Belinkov, Ofer Lavi, and Yoav Goldberg. Fine-grained analysis of sentence embeddings using auxiliary prediction tasks. _ICLR_ , abs/1608.04207, 2016. 

- Dieuwke Hupkes, Sara Veldhoen, and Willem Zuidema. Visualisation and ‘diagnostic classifiers’ reveal how recurrent and recursive neural networks process hierarchical structure. _J. Artif. Intell. Res._ , 61(1):907–926, January 2018. 

- Alexis Conneau, German Kruszewski, Guillaume Lample, Loïc Barrault, and Marco Baroni. What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , 

11 

pages 2126–2136, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-1198. URL `https://aclanthology.org/P18-1198` . 

- Yoonho Lee, Annie S Chen, Fahim Tajwar, Ananya Kumar, Huaxiu Yao, Percy Liang, and Chelsea Finn. Surgical fine-tuning improves adaptation to distribution shifts. In _The Eleventh International Conference on Learning Representations_ , 2023. URL `https://openreview.net/forum?id=APuPRxjHvZ` . 

- Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer Feed-Forward layers are KeyValue memories. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 5484–5495, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. 

- N. Elhage. A mathematical framework for transformer circuits. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ . https: //transformer-circuits.pub/2021/framework/index.html, 2021. 

- Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. _Advances in Neural Information Processing Systems_ , 36, 2022. 

- Sumu Zhao, Damian Pascual, Gino Brunner, and Roger Wattenhofer. Of Non-Linearity and Commutativity in BERT. In _International Joint Conference on Neural Networks (IJCNN), Virtual-only_ , July 2021. 

- Peter Hase, Mohit Bansal, Been Kim, and Asma Ghandeharioun. Does localization inform editing? surprising differences in causality-based localization vs. knowledge editing in language models. _ArXiv_ , abs/2301.04213, 2023. URL `https://api.semanticscholar.org/CorpusID:255595518` . 

- Yann LeCun, John Denker, and Sara Solla. Optimal brain damage. In D. Touretzky, editor, _Advances in Neural Information Processing Systems_ , volume 2. Morgan-Kaufmann, 1989. URL `https://proceedings. neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf` . 

- Babak Hassibi and David Stork. Second order derivatives for network pruning: Optimal brain surgeon. In S. Hanson, J. Cowan, and C. Giles, editors, _Advances in Neural Information Processing Systems_ , volume 5. Morgan-Kaufmann, 1992. URL `https://proceedings.neurips.cc/paper_files/paper/1992/file/ 303ed4c69846ab36c2904d3ba8573050-Paper.pdf` . 

- Song Han, Jeff Pool, John Tran, and William J. Dally. Learning both weights and connections for efficient neural network. In _NIPS_ , 2015. URL `https://api.semanticscholar.org/CorpusID:2238772` . 

- Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. _ArXiv_ , abs/1608.08710, 2016. URL `https://api.semanticscholar.org/CorpusID:14089312` . 

- Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste. Sparsity in deep learning: pruning and growth for efficient inference and training in neural networks. _J. Mach. Learn. Res._ , 22(1): 10882–11005, January 2021. 

- Xiuqing Lv, Peng Zhang, Sunzhu Li, Guobing Gan, and Yueheng Sun. LightFormer: Light-weight transformer using SVD-based weight transfer and parameter sharing. In _Findings of the Association for Computational Linguistics: ACL 2023_ , Toronto, Canada, July 2023. Association for Computational Linguistics. 

- Shikhar Murty, Pratyusha Sharma, Jacob Andreas, and Christopher D. Manning. Characterizing intrinsic compositionality in transformers with tree projections, 2022. 

- Tian Jin, Michael Carbin, Daniel M. Roy, Jonathan Frankle, and Gintare Karolina Dziugaite. Pruning’s effect on generalization through the lens of training and regularization. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, _Advances in Neural Information Processing Systems_ , 2022. URL `https://openreview.net/forum?id=OrcLKV9sKWp` . 

12 

- Habib Hajimolahoseini, Mehdi Rezagholizadeh, Vahid Partovinia, Marzieh Tahaei, Omar Mohamed Awad, and Yang Liu. Compressing pre-trained language models using progressive low rank decomposition. _Advances in Neural Information Processing Systems_ , 2021. 

- Xiyu Yu, Tongliang Liu, Xinchao Wang, and Dacheng Tao. On compressing deep models by low rank and sparse decomposition. In _2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 67–76, July 2017. 

- Jimmy Ba and Rich Caruana. Do deep nets really need to be deep? In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K.Q. Weinberger, editors, _Advances in Neural Information Processing Systems_ , volume 27. Curran Associates, Inc., 2014. URL `https://proceedings.neurips.cc/paper_files/paper/ 2014/file/ea8fcd92d59581717e06eb187f10666d-Paper.pdf` . 

- Huanrui Yang, Minxue Tang, Wei Wen, Feng Yan, Daniel Hu, Ang Li, Hai Li, and Yiran Chen. Learning lowrank deep neural networks via singular vector orthogonality regularization and singular value sparsification, 2020. 

- Dan Hendrycks and Kevin Gimpel. Gaussian error linear units. _arXiv preprint arXiv:1606.08415_ , 2016. 

- Carl Eckart and Gale Young. The approximation of one matrix by another of lower rank. _Psychometrika_ , 1 (3):211–218, 1936. 

- Ben Wang and Aran Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. `https://github.com/kingoflolz/mesh-transformer-jax` , May 2021. 

- Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In _Conference on Empirical Methods in Natural Language Processing_ , 2018. URL `https: //api.semanticscholar.org/CorpusID:52822214` . 

- James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. FEVER: a large-scale dataset for fact extraction and VERification. In _NAACL-HLT_ , 2018. 

- Maria De-Arteaga, Alexey Romanov, Hanna Wallach, Jennifer Chayes, Christian Borgs, Alexandra Chouldechova, Sahin Geyik, Krishnaram Kenthapadi, and Adam Tauman Kalai. Bias in bios. In _Proceedings of the Conference on Fairness, Accountability, and Transparency_ . ACM, jan 2019. doi: 10.1145/3287560.3287572. URL `https://doi.org/10.1145%2F3287560.3287572` . 

- Stephanie C. Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. In _Annual Meeting of the Association for Computational Linguistics_ , 2021. URL `https: //api.semanticscholar.org/CorpusID:237532606` . 

- Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. A large annotated corpus for learning natural language inference. In Lluís Màrquez, Chris Callison-Burch, and Jian Su, editors, _Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing_ , pages 632–642, Lisbon, Portugal, September 2015. Association for Computational Linguistics. doi: 10.18653/v1/D15-1075. URL `https://aclanthology.org/D15-1075` . 

- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Ro{bert}a: A robustly optimized {bert} pretraining approach, 2020. URL `https://openreview.net/forum?id=SyxS0T4tvS` . 

- Yanai Elazar, Nora Kassner, Shauli Ravfogel, Abhilasha Ravichander, Ed Hovy, Hinrich Schutze, and Yoav Goldberg. Measuring and improving consistency in pretrained language models. _ArXiv_ , abs/2102.01017, 2021. 

- Max-Philipp B. Schrader. gym-sokoban. `https://github.com/mpSchrader/gym-sokoban` , 2018. 

13 

- Olivier Roy and Martin Vetterli. The effective rank: A measure of effective dimensionality. In _2007 15th European Signal Processing Conference_ , pages 606–610, 2007. 

14 

## **Appendix** 

## **A Dataset Details** 

**CounterFact.** The CounterFact dataset is derived from the PARAREL dataset [Elazar et al., 2021] and contains knowledge tuples of the kind _t[c]_ = ( _s, r, o[c]_ ), where _s_ is the subject, _r_ is the relation and _o_ is the object. These tuples are constructed using entities listed in Wikidata. The datapoints are accompanied by handwritten prompt templates for each category. The CounterFact dataset also contains suggested edits to the true facts represented in the dataset. For this study, the set of counterfactual edits are not used. 

**PILE.** The PILE dataset is an approximately 1TB language modeling dataset that was used to pre-train GPT-J. It contains text from 22 smaller datasets, including Wikipedia, OpenWebText2, and StackExchange, to name a few. The PILE dataset was used to study the effect of `LASER` on the behavior of the model on the original training data distribution. For the study on quantifying the occurrences of entities in the training data, the training data split of PILE was used. However, the measure of change in perplexity of the model after `LASER` was measured on the validation split of the dataset. 

**HotpotQA.** We use the HotPotQA dataset available on HuggingFace. An example question is “ _What are the names of the current members of American heavy metal band who wrote the music for Hurt Locker The Musical?_ " and the answer is “ _Hetfield and Ulrich, longtime lead guitarist Kirk Hammett, and bassist Robert Trujillo_ ". We use the validation split of this dataset for identifying `LASER` parameters, containing 7,100 datapoints, and the train split, containing 90k datapoints, for evaluation. 

**FEVER.** The FEVER (Fact Extraction and Verification) dataset consists of claims and label characterizing each as either true or false. The goal is to predict the label given the claim. It consists of 185,445 such claims and is constructed using data from Wikipedia. An example claim is “ _Roman Atwood is a content creator._ ” and the label is “ _true_ .” 

**Bias in Bios.** Bias in Bios is a dataset that consists of descriptions of people, and the task is to correctly identify the gender and occupation of the person. An example datapoint is “ _Nancy Lee graduated from Lehigh University, with honours in 1998. Nancy has years of experience in weight loss surgery, patient support, education, and diabetes_ ,” the gender label is “ _Female_ ” and the profession is “ _Nurse_ .” **BigBench-Epistemic Reasoning:** The epistemic reasoning dataset tests a model’s ability to make entailment decisions from a pair of sentences. Specifically, epistemic reasoning assesses a given model’s understanding of “theory of mind,” which involves the capacity to analyze the cognitive states of various agents. A sample consists of a text containing a premise and hypothesis and a label which is either “entailment” or “non-entailment.” An example datapoint is “Premise: Emma knows that James thinks that there is milk in the fridge. Hypothesis: James thinks that there is milk in the fridge.” The label here is “entailment.” 

**TruthfulQA:** TruthfulQA evaluates the extent to which a model replicates human falsehoods. The benchmark contains 817 questions that span 38 categories, including health, law, finance and politics. The dataset emphasizes that solely increasing the scale of models may not be as effective in enhancing truthfulness as fine-tuning with training objectives that go beyond imitating text from the web. An example datapoint is “What happens to you if you eat watermelon seeds? [‘Nothing happens,’ ‘You eat watermelon seeds,’ ‘The watermelon seeds pass through your digestive system,’ ‘You will not digest the watermelon seeds,’ ‘The watermelon seeds will be excreted,’ ‘You grow watermelons in your stomach,’ ‘You get sick,’ ‘You have bad dreams,’ ‘You die,’ ‘You get indigestion,’ ‘You fall unconscious,’ ‘You digest the watermelon seeds’]”. 

15 

**QA Wiki Data:** The task tests a model’s ability to answer open domain, cloze-style question answering on randomly sampled Wikipedia triplets. This evaluates world knowledge on a large collection of facts and information from a knowledge graph extracted from Wikipedia data. An example datapoint from this dataset ” is “Gabon shares a border with Cameroon. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction_images/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction.pdf-0016-01.png)


Figure 6: An example of a Sokoban task. The game requires an agent to move the orange boxes to their desired locations (red squares) in a complex warehouse-like environment without getting trapping themselves. Playing successfully requires the agent to reason over long time horizons effectively. 

## **B Details of the Decision Transformer Domain** 

**Sokoban Details.** We show an image of the Sokoban task in Figure 6. Sokoban is a warehouse-keeping transportation game that requires long-horizon reasoning and planning over multiple time steps. The task is to move all boxes to their target locations without getting trapped. We use the Gym Sokoban environment Schrader [2018], and train a 5-layer decision transformer model using 10[6] optimal episodes of the game. In our setting, the maximum return of the game is set to 10. 

## **C Extended Analysis** 

## **C.1 Are weight matrices already low-rank?** 

As seen in Figure 7, we find that `LASER` approximated matrices with their low-rank approximations much beyond their effective rank as computed by [Roy and Vetterli, 2007]. To study this, we computed the effective rank of the MLP matrices for which `LASER` helps for GPT-J model using the method described by Roy and Vetterli [2007]. The plot shows that although matrices of the later layer have a lower effective rank than the earlier layers, the computed effective rank is significantly larger than the reduction % until which `LASER` helps. 

## **C.2 How much reduction is too much?** 

We see that for many of the matrices in cases where reduction helps, with increasing amounts of rank-reduction, the model first monotonically improves before it starts to worsen, as seen in Figure 8. The point up to which it improves varies depending on the layer type and layer number. However, the monotonic improvement and worsening are observed consistently. 

What is the effect of removing the layer completely? We find that, removing the layer completely can be better than retaining its matrix with its full rank, however it is observed to be worse than the model with the low-rank approximation of the matrix. 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction_images/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction.pdf-0017-00.png)


**----- Start of picture text -----**<br>
Layer Type<br>Layer Number<br>Effective Rank<br>**----- End of picture text -----**<br>


Figure 7: Effective rank of the matrices computed as described by Roy and Vetterli [2007] 

## **C.3 Does** `LASER` **select the same layer for different tasks?** 

We find that the maximum improvements on different tasks come from `LASER` on different layers of the model. Figure 9 shows that for GPT-J on different tasks, the best-performing models across tasks have reduced matrices in different layers. 

## **C.4 Measuring Perplexity on PILE.** 

To measure the effect of the interventions on language modelling, we compute the perplexity of the reduced model on the evaluation set of PILE. The perplexity of the fixed-length GPT-J model is evaluated using a sliding window strategy over the sequence of tokens with a stride of 512 tokens. While there is an improvement in the task at hand, the model’s perplexity worsens slightly after applying `LASER` . We do not yet fully understand what the worsening in perplexity of the model corresponds to and leave this for future study. 

## **C.5 Final** `LASER` **search results** 

Table 3 shows the final search results of `LASER` for models and datasets from Table 1. These values are obtained by reporting the optimal LASER parameters that maximize the validation accuracy. The results show that the optimal improvements in the models typically come from later layers in the transformer model, typically from reducing the MLP Input matrix. Note that _ℓ_ = 0 denotes that the intervention is done on the first layer. For reference, recall that Llama2 has 32 layers, GPT-J has 28 layers, and Roberta has 12 layers. The magnitudes of reduction are also quite large, with the rank at times being reduced to 1% of the original matrix’s rank. 

## **D Alternate Pruning Methods** 

Instead of approximating weight matrices with their rank-k approximations, we tried _Absolute Weight Pruning_ [Frankle and Carbin, 2018]. Here, we zero out the bottom x% of the weights of the matrix by their absolute magnitude. The results for GPT-J on CounterFact can be seen in Figure 10. In this case too, we find that 

17 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction_images/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction.pdf-0018-00.png)


**----- Start of picture text -----**<br>
% Reduction<br>Top-10 accuracy<br>**----- End of picture text -----**<br>


Figure 8: While the performance of the models continues to improve with large amounts of reduction, after a point it starts to worsen. The plot shows the top-10 accuracy of GPT-J on CounterFact. A dip in performance is observed at 99.95% reduction. 

|**Dataset**|**Model**|
|---|---|
|CounterFact<br>HotPotQA<br>FEVER<br>Bios Gender<br>Bios Prof.<br>BigBench-Epistemic Reasoning<br>TruthfulQA<br>BigBench-WikidataQA|**Roberta**<br>**GPT-J**<br>**Llama2 7B**<br>[_τ, ℓ, ρ_]<br>[_τ, ℓ, ρ_]<br>[_τ, ℓ, ρ_]|
||[_Uin,_8_,_0_._8]<br>[_Uin,_27_,_0_._01]<br>[_Uin,_28_,_0_._05]<br>[_Uout,_2_,_0_._4]<br>[_Uin,_27_,_0_._1]<br>[_Uin,_27_,_0_._2]<br>[_Uin,_3_,_0_._4]<br>[_Uin,_24_,_0_._01]<br>[_Uin,_30_,_0_._2]<br>[_Uin,_9_,_0_._9]<br>[_Uin,_14_,_0_._01]<br>[_Uin,_24_,_0_._01]<br>[_Uin,_3_,_0_._9]<br>[_Uin,_18_,_0_._01]<br>[_Uout,_30_,_0_._4]<br>[_Uout,_1_,_0_._4]<br>[_Uin,_26_,_0_._01]<br>[_Uout,_28_,_0_._01]<br>[_Uin,_0_,_0_._01]<br>[_Uin,_7_,_0_._8]<br>[_Uin,_30_,_0_._05]<br>[_Uin,_7_,_0_._4]<br>[_Uin,_27_,_0_._01]<br>[_Uin,_27_,_0_._01]|



Table 3: Final search results of `LASER` : In top-performing models, significant benefits from rank reduction are typically observed in later layers. The amount of reduction is severe, for example, in GPT-J on CounterFact, the rank of the MLP matrix is reduced from 4096 to rank 4. This is about 99% of the matrix’s original rank. 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction_images/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction.pdf-0019-00.png)


**----- Start of picture text -----**<br>
MLP Output Matrix<br>MLP Input Matrix<br>CounterFact FEVER HotPot Bias in Bios<br>**----- End of picture text -----**<br>


Figure 9: For GPT-J across different datasets, the largest benefit of `LASER` comes from reductions on different layer numbers. Even though the largest benefits are typically from the MLP layers in the later layers of the model, the layer number differs for different dataset-model pairs. 

the accuracy of the model increases with pruning later layers of the MLP. We leave further study of this phenomenon for future work. 

## **E Implementation Details** 

## **E.1 Dataset Processing Details** 

We process each dataset described in Table 1 separately. In each case, we use 20% of the processed dataset as the validation set which we use to select the best `LASER` hyperparameters ( _τ, ρ, ℓ_ ). This validation set can be different from the validation set of the original unprocessed dataset. We use accuracy on the validation set for selecting the best hyperparameters for each LLM and a given dataset. Table 4 summarizes the size of these filtered dataset. We describe the dataset-specific processing below: 

**CounterFact.** We use the original dataset which consists of roughly 20,000 examples and 3 paraphrase for each example. This gives us 65,757 examples for the entire dataset. The set of possible labels in this QA task is open-ended. For Roberta and GPT-J LLMs, the labels are always exactly one token, while for Llama2 the labels can be multiple tokens long. 

**Hotpot.** We combine the included validation and training sets of Hotpot to increase the dataset size. We then filter out all examples where the answers are more than 15 tokens long according to the Llama2 tokenizer. We convert the original question to the prompt _“<question> The answer is”_ (if the question ends with ? or .) or _“<question>? The answer is”_ where the prompt variable _<question>_ is replaced by the original question. This gives us a dataset of size 14,618. The set of possible labels in this QA task is open-ended and are multi-token long for all three LLMs that we consider. 

**Fever.** We merge the dev and test split of the original Fever dataset. We then filter out samples where with duplicate claims (inputs) but different labels (outputs). This results in a dataset of 13,086 samples, including 

19 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction_images/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction.pdf-0020-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction_images/iclr_2024_The_Truth_is_in_There_Improving_Reasoning_in_Language_Models_with_Layer-Selective_Rank_Reduction.pdf-0020-01.png)


**----- Start of picture text -----**<br>
Reduction %: 10 25 40 50 60 75 90 92.5 95 97.5 98 98.5 99 99.5 99.75<br>**----- End of picture text -----**<br>


Figure 10: Similar to `LASER` , we perform layer-selective absolute weight pruning, where a fraction of the weights with smaller absolute magnitudes are set to zero. For GPT-J on CounterFact, we find a similar improvement in model performance with this intervention as we do with `LASER` . A thorough analysis of how absolute weight pruning might help improve model performance on different natural language understanding tasks and its connection to `LASER` is left for future work. 

6,510 from the original dev set. Here there are only two possible labels: true and false. We convert each question into the prompt _“Consider the following claim: <question>. Is this claim true or false. The claim is”_ . The prompt variable _<question>_ is replaced by the original question. 

**Bios Gender.** We use only the dev split of the original Bias in Bios dataset. This gives us a dataset of size 39,642. The only possible labels in this QA task are two: male and female. We convert each input bio into the prompt _“Consider the following text: <bio>. Is the person in this text male or female? The person is”_ . The prompt variable _<bio>_ is replaced by the original bio. 

**Bios Profession.** We use only the dev split of the original Bias in Bios dataset. The goal here is to predict the profession for a given bio. We only keep datapoints which contains profession with a few tokens, namely, journalist, poet, composer, model, teacher, architect, painter, and professor. This gives us a dataset of size 19,223. The aforementioned professions compose the list of possible labels. We convert each input bio into the prompt _“Consider the following text: <bio>. What is the profession of the person in this text? The profession of this person is”_ . The prompt variable _<bio>_ is replaced by the original bio. **BigBench Epistemic Reasoning.** We merge the validation and train split of the Big Bench epistemic reasoning dataset. This gives us a dataset of size 2000. The set of possible labels here are: _entailment_ and _non-entailment_ which are multi-token long for all LLMs. We do not process the text. 

**Truthful QA.** We use the validation split of the Truthful QA dataset. The truthful QA dataset consists of multiple choice questions. We convert the dataset into separately checking the correctness of each answer independent of other answers. Specifically, a sample with 4 multiple choice answers gets converted into 4 separate samples, each with a true or false answer. We convert each question and answer pair into the 

20 

|**Dataset Name**|**Dataset Size**|
|---|---|
|CounterFact<br>HotpotQA<br>FEVER<br>Bios Gender<br>Bios Profession<br>TruthfulQA<br>BigBench-Epsitemic Reasoning<br>BigBench-WikidataQA|65757<br>14618<br>13086<br>39642<br>19223<br>5882<br>2000<br>20321|



Table 4: Size of the filtered dataset used for evaluating `LASER` . We use 20% of the dataset for selecting `LASER` hyperparameters ( _τ, ℓ, ρ_ ) and the evaluate the best model on the remaining. 

prompt _“<question> <answer>. Is this statement true or false. This statement is”_ if the answer does not end with period (.), otherwise, we convert it into _“<question> <answer> Is this statement true or false. This statement is”_ . The prompt variables _<question>_ and _<answer>_ are replaced by the original question and answer respectively. The processed dataset consists of 5,882 samples. 

**BigBench Wikidata QA.** We merge the validation and train split of the Big Bench Wikidata QA dataset. We filter out examples where the number of target labels are more than 1. This gives us a dataset of size 20,321. This QA task has an open-ended set of labels. 

## **E.2 Details for Computing Accuracy and Log Loss** 

The procedure used to compute accuracy and log loss varies across the different datasets. Typically, for QA datasets with open-ended labels, we generate the predicted answer by doing greedy sampling using the LLM, i.e., with temperature set to 0. We report the prediction as correct if and only if the answer is in the generated text. We lower case and strip whitespaces before comparing text. We call this the _generation accuracy_ . In contrast, for datasets with a small set of possible label choice, we predict the label with the highest probability under the LLM and report the prediction as correct if and only if the predicted label is the correct label. We call this the _classification accuracy_ . 

As Roberta is a masked language model, we do generation by creating a prompt with <mask> tokens, and predicting these masked tokens. When generating the response, we use a fixed number of masked tokens which may not correspond to the number of tokens in the answer. However, when computing the log-loss of the answer, we add as many masked tokens as the number of tokens in the answer, and compute the log probabilities of the answer tokens under the model corresponding to these masked tokens. 

The procedure for computing log-loss of the gold answer given context is the same across all dataset. We describe the dataset specific details for computing accuracies below. 

**CounterFact.** We use generation accuracy to evaluate success. For GPT-J and Roberta, we generate a single token as all labels are single token long, whereas for Llama we generate up to 10 tokens. 

**HotPotQA.** We use generation accuracy to evaluate success. For GPT-J and Llama2, we generate up to 15 tokens. For Roberta we only use 5 tokens since Roberta struggles to fill-in more than a few masked tokens, this is understandable as Roberta is trained by masking out a small number of tokens (typically 15% tokens), and using these models to predict a large number of masked tokens then suffers from distributional shift. 

**Fever.** We use classification accuracy to measure success. We predict the label from { _true_ , _false_ } that has the highest probability under the model. 

21 

|`LASER` **hyperparameter**|**Search Space**|
|---|---|
|_τ_<br>_ℓ_<br>_ρ_|MLP weight matrices _Uin_ and _Uout_<br>all layers in the model<br>{0.9, 0.8, 0.6, 0.2, 0.1, 0.05, 0.01}|



Table 5: `LASER` hyperparameters 

**Bios Gender.** We use classification accuracy to measure success. We predict the label from { _male_ , _female_ } that has the highest probability under the model. 

**Bios Profession.** We use classification accuracy to measure success. We predict the label from the list of possible professions that has the highest probability under the model. 

**BigBench Epistemic Reasoning.** We use classification accuracy to measure success. We predict the label from { _entailment_ , _non-entailment_ } that has the highest probability under the model. 

**TruthfulQA.** We use classification accuracy to measure success. We predict the label from { _true_ , _false_ } that has the highest probability under the model. 

**BigBench WikidataQA.** As the set of labels are open-ended, we compute the accuracy using generation similar to CounterFact. For GPT-J and Llama2 we generate up to 10 tokens, whereas for Roberta we generate 5 tokens. 

## **E.3 Code** 

We use PyTorch for all experiments. We use the HuggingFace implementation for all three large language models. We use Llama2 7GB weights provided by Meta. We use the SVD implementation available in PyTorch for experiments. The code can be found at: https://github.com/pratyushasharma/laser 

## **E.4 Compute Details** 

We ran each experiment on a cluster with V100 and A2600 GPUs. Each experiment took about 1-3hrs to finish. For all settings, we search over hyperparameters listed in Table 5. For the GPT-J+CounterFact setting, depending on the experiment and plots, we run a much more fine-grained search over each hyperparameter. 

22 

