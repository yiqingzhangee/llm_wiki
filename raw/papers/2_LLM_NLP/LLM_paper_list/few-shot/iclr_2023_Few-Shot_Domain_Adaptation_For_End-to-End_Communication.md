Published as a conference paper at ICLR 2023 

## - FEW-SHOT DOMAIN ADAPTATION FOR END TO-END COMMUNICATION 

Jayaram Raghuram[1] , Yijing Zeng[1] , Dolores García Martí[2] , Rafael Ruiz Ortiz[2] , Somesh Jha[1,3] , Joerg Widmer[2] , Suman Banerjee[1] 

> 1 University of Wisconsin - Madison 2 IMDEA Networks Institute, Madrid 3 XaiPient {jayaramr, yijingzeng, jha, suman}@cs.wisc.edu {dolores.garcia, rafael.ruiz, joerg.widmer}@imdea.org 

## ABSTRACT 

The problem of end-to-end learning of a communication system using an autoencoder – consisting of an encoder, channel, and decoder modeled using neural networks – has recently been shown to be an effective approach. A challenge faced in the practical adoption of this learning approach is that under changing channel conditions (e.g. a wireless link), it requires frequent retraining of the autoencoder in order to maintain a low decoding error rate. Since retraining is both time consuming and requires a large number of samples, it becomes impractical when the channel distribution is changing quickly. We propose to address this problem using a fast and sample-efficient (few-shot) domain adaptation method that does not change the encoder and decoder networks. Different from conventional training-time unsupervised or semi-supervised domain adaptation, here we have a trained autoencoder from a source distribution that we want to adapt (at test time) to a target distribution using only a small labeled dataset, and no unlabeled data. We focus on a generative channel model based on the Gaussian mixture density network (MDN), and propose a regularized, parameter-efficient adaptation of the MDN using a set of affine transformations. The learned affine transformations are then used to design an optimal transformation at the decoder input to compensate for the distribution shift, and effectively present to the decoder inputs close to the source distribution. Experiments on many simulated distribution changes common to the wireless setting, and a real mmWave FPGA testbed demonstrate the effectiveness of our method at adaptation using very few target domain samples[1] . 

## 1 INTRODUCTION 

End-to-end (e2e) learning of a communication system using an autoencoder has been recently shown to be a promising approach for designing the next generation of wireless networks (O’Shea & Hoydis, 2017; Dörner et al., 2018; Aoudia & Hoydis, 2019; O’Shea et al., 2019; Ye et al., 2018; Wang et al., 2017). This new paradigm is a viable alternative for optimizing communication in diverse applications, hardware, and environments (Hoydis et al., 2021). It is particularly promising for dense deployments of low-cost transceivers, where there is interference between the devices and hardware imperfections that are difficult to model analytically. The key idea of e2e learning for a communication system is to use an autoencoder architecture to model and learn the transmitter and receiver jointly using neural networks in order to minimize the e2e symbol error rate (SER). 

The channel ( _i.e._ , propagation medium and transceiver imperfections) can be represented as a stochastic transfer function that transforms its input **z** _∈_ R _[d]_ to an output **x** _∈_ R _[d]_ . It can be regarded as a black-box that is typically non-linear and non-differentiable due to hardware imperfections ( _e.g._ , quantization and amplifiers). Since autoencoders are trained using stochastic gradient descent (SGD)-based optimization (O’Shea & Hoydis, 2017), it is challenging to work with a black-box channel that is not differentiable. One approach to address this problem is to use a known mathemat- 

> 1Code for our work: https://github.com/jayaram-r/domain-adaptation-autoencoder 

1 

Published as a conference paper at ICLR 2023 

ical model of the channel ( _e.g._ , additive Gaussian noise), which would enable the computation of gradients with respect to the autoencoder parameters via backpropagation. However, such standard channel models do not capture well the realistic channel effects as shown in Aoudia & Hoydis (2018). Alternatively, recent works have proposed to learn the channel using deep generative models that approximate _p_ ( **x** _|_ **z** ), the conditional probability density of the channel, using Generative Adversarial Networks (GANs) (O’Shea et al., 2019; Ye et al., 2018), Mixture Density Networks (MDNs) (García Martí et al., 2020), and conditional Variational Autoencoders (VAEs) (Xia et al., 2020). The use of a differentiable generative model of the channel enables SGD-based training of the autoencoder, while also capturing realistic channel effects better than standard models. 

Although this e2e optimization with a generative channel model learned from data can improve the physical-layer design for communication systems, in reality, channels often change, requiring collection of a large number of samples and frequent retraining of the channel model and autoencoder. For this reason, _adapting the generative channel model and the autoencoder as often as possible, using only a small number of samples_ is required for good communication performance. Prior works have (to be best of our knowledge) not addressed the adaptation problem for autoencoder-based e2e learning, which is crucial for real-time deployment of such a system under frequently-changing channel conditions. In this paper, we study the problem of domain adaptation (DA) of autoencoders using an MDN as the channel model. In contrast to conventional DA, where the target domain has a large unlabeled dataset and sometimes also a small labeled dataset (semi-supervised DA) (Ben-David et al., 2006), here we consider a _few-shot DA_ setting where the target domain has _only_ a small labeled dataset, and no unlabeled data. This setting applies to our problem since we only get to collect a small number of labeled samples at a time from the changing target domain (here the channel)[2] . 

Towards addressing this important practical problem, we make the following contributions: 

- We propose a parameter- and sample-efficient method for adapting a generative MDN (used for modeling the channel) based on the properties of Gaussian mixtures (§ 3.1 and § 3.2). 

- Based on the MDN adaptation, we propose an optimal input-transformation method at the decoder that compensates for changes in the channel distribution, and decreases or maintains the error rate of the autoencoder without any modification to the encoder and decoder networks (§ 3.3). 

- Experiments on a mmWave FPGA platform and a number of simulated distribution changes show strong performance improvements for our method. For instance, in the FPGA experiment, our method improves the SER by 69% with only 10 samples per class from the target distribution (§ 4). 

**Related Work.** Recent approaches for DA such as DANN (Ganin et al., 2016), based on adversarial learning of a shared representation between the source and target domains (Ganin & Lempitsky, 2015; Ganin et al., 2016; Long et al., 2018; Saito et al., 2018; Zhao et al., 2019; Johansson et al., 2019), have achieved much success on computer vision and natural language processing. Their high-level idea is to adversarially learn a shared feature representation for which inputs from the source and target distributions are nearly indistinguishable to a _domain discriminator_ DNN, such that a _label predictor_ DNN using this representation and trained using labeled data from only the source domain also generalizes well to the target domain. Adversarial DA methods are not suitable for our problem, which requires fast and frequent test-time DA, because of their high computational and sample complexity and the imbalance in the number of source and target domain samples. 

Related frameworks such as transfer learning (Long et al., 2015; 2016), model-agnostic metalearning (Finn et al., 2017), domain-adaptive few-shot learning (Zhao et al., 2021; Sun et al., 2019), and supervised DA (Motiian et al., 2017a;b) also deal with the problem of adaptation using a small number of samples. Most of them are not applicable to our problem because they primarily address novel classes (with potentially different distributions), and knowledge transfer from existing to novel tasks. Motiian et al. (2017a) is closely related since they also deal with a target domain that only has a small labeled dataset and has the same label space. The _key difference_ is that Motiian et al. (2017a) address the training-time few-shot DA problem, while we focus on test-time few-shot DA. Specifically, their adversarial DA method requires both the source and target domain datasets at training time, and can be computationally expensive to retrain for every new batch of target domain data (a key motivation for this work is to avoid frequent retraining). 

> 2In our problem, labels correspond to the transmitted messages and are essentially obtained for free (see § 3). 

2 

Published as a conference paper at ICLR 2023 

## 2 PRIMER ON AUTOENCODER-BASED END-TO-END COMMUNICATION 

**Notations.** We denote vectors and matrices with boldface symbols. We define the indicator function 1 ( _c_ ) that takes value 1 (0) when the condition _c_ is true (false). For any integer _n ≥_ 1, we define [ _n_ ] = _{_ 1 _, · · · , n}_ . We denote the one-hot-coded vector with 1 at index _i_ and the rest zeros by **1** _i_ . The probability density of a multivariate Gaussian with mean _**µ**_ and covariance matrix **Σ** is denoted by _N_ ( **x** _|_ _**µ** ,_ **Σ** ). We use the superscripts s and t to denote quantities corresponding to the source and target domain respectively. Table 2 in the Appendix provides a quick reference for the notations. 

Following (O’Shea & Hoydis, 2017; Dörner et al., 2018), consider a singleinput, single-output (SISO) communimessage Input Wireless channel Predicted message cation system shown in Fig. 1, consisting of a transmitter (or encoder), channel, and receiver (or decoder). The Encoder / Transmitter channel modelGenerative Decoder / Receiver encoder **E** _**θ** e_ ( _·_ ) is a multi-layer feedneural network neural network forward neural network (NN) with parameters _**θ** e_ , that maps an input message _y ∈Y_ := _{_ 1 _, · · · , m}_ into an Figure 1: Autoencoder-based end-to-end communication system encoded symbol **z** _∈_ R _[d]_ . The input with a generative channel model. 

message _y_ is mapped into a one-hot-coded vector **1** _y_ prior to being processed by the encoder[3] . The message _y_ is equivalent to a class label in machine learning terms, and the encoded symbol **z** = **E** _**θ** e_ ( **1** _y_ ) is like a representative vector for the class _y_ . We note that the dimension of the encoding _d_ is small (less than 10), and _d_ = 2 is typically used to coincide with traditional modulation techniques (O’Shea & Hoydis, 2017; Goldsmith, 2005). The set of distinct encoded symbols _Z_ = _{_ **E** _**θ** e_ ( **1** 1) _, · · · ,_ **E** _**θ** e_ ( **1** _m_ ) _}_ is referred to as the _constellation of the autoencoder_ . 

The symbol **z** is transmitted (via the custom modulation learned by the encoder) over a communication channel, represented by an unknown conditional probability density _p_ ( **x** _|_ **z** ), and is received at the output of the channel as a noisy, distorted symbol **x** _∈_ R _[d]_ . The decoder **D** _**θ** d_ ( _·_ ) is also a multilayer, feed-forward NN with parameters _**θ** d_ that predicts the class-posterior probabilities over the _m_ messages based on the distorted channel output **x** . The decoder is essentially a classifier whose input-output mapping is defined by **D** _**θ** d_ ( **x** ) := [ _P_ _**θ** d_ (1 _|_ **x** ) _, · · · , P_ _**θ** d_ ( _m |_ **x** )], where _P_ _**θ** d_ ( _y |_ **x** ) is the predicted probability of class _y_ given **x** . The class with the highest predicted probability is the � decoded message _y_ ( **x** ) = argmax _y∈Y P_ _**θ** d_ ( _y |_ **x** ). As in standard classification, the performance� metric of the autoencoder is the _symbol error rate_ (SER), defined as E( **x** _,y_ )[ 1 ( _y_ ( **x** ) _̸_ = _y_ )]. 

**Generative Channel Model.** In order to learn the encoder and decoder networks using SGDbased optimization, it is necessary to have a differentiable backward path from the decoder to the encoder through the channel. We address this by learning a parametric generative model of the channel _P_ _**θ** c_ ( **x** _|_ **z** ) (with parameters _**θ** c_ ) that closely approximates the true channel conditional density _p_ ( **x** _|_ **z** ). There exists a stochastic data generation or sampling function **x** = **h** _**θ** c_ ( **z** _,_ **u** ) corresponding to the generative model, where **u** captures the random aspects of the channel ( _e.g._ , noise and phase offsets; details in Appendix E). In this work, we model the conditional density of the channel using a _set of m Gaussian mixtures_ , one per input message (or class) _y ∈Y_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0003-07.png)


Here, _k_ is the number of components, _**µ** i_ ( **z** ) _∈_ R _[d]_ is the mean vector, **Σ** _i_ ( **z** ) _∈_ R _[d][×][d]_ is the (symmetric, positive-definite) covariance matrix, and _πi_ ( **z** ) _∈_ [0 _,_ 1] is the prior probability of component _i_ . It is convenient to express the component prior probability in terms of the softmax function as _πi_ ( **z** ) = _e[α][i]_[(] **[z]**[)] _/_[�] _[k] j_ =1 _[e][α][j]_[(] **[z]**[)] _[,][∀][i][∈]_[[] _[k]_[]][,][where] _[α][i]_[(] **[z]**[)] _[∈]_[R][are][the][component][prior] logits. We define the parameter vector of component _i_ as _**ϕ** i_ ( **z** ) _[T]_ = [ _αi_ ( **z** ) _,_ _**µ** i_ ( **z** ) _[T] ,_ vec( **Σ** _i_ ( **z** )) _[T]_ ], where vec( _·_ ) is the vector representation of the unique entries of the covariance matrix. We also define the combined parameter vector from all components by _**ϕ**_ ( **z** ) _[T]_ = [ _**ϕ**_ 1( **z** ) _[T] , · · · ,_ _**ϕ** k_ ( **z** ) _[T]_ ]. 

An MDN can model complex conditional distributions by combining a feed-forward network with a parametric mixture density (Bishop, 1994; 2007). We use the MDN to predict the parameters of the 

> 3The encoder has a normalization layer that constrains the average power of the symbols (see Appendix D). 

3 

Published as a conference paper at ICLR 2023 

Gaussian mixtures _**ϕ**_ ( **z** ) as a function of its input symbol **z** , _i.e._ , _**ϕ**_ ( **z** ) = **M** _**θ** c_ ( **z** ), where _**θ** c_ are the parameters of the MDN network. The MDN output with all the mixture parameters has dimension _p_ = _k_ ( _d_ ( _d_ + 1) _/_ 2 + _d_ + 1). While there are competing methods for generative modeling of the channel such as conditional GANs (Ye et al., 2018) and VAEs (Xia et al., 2020), we choose the Gaussian MDN based on i) the strong approximation properties of Gaussian mixtures (Kostantinos, 2000) for learning probability distributions; and ii) the analytical and computational tractability it lends to our domain adaptation formulation. The effectiveness of a Gaussian MDN for wireless channel modeling has also been shown in García Martí et al. (2020). 

The input-output function of the autoencoder is given by **f** _**θ**_ ( **1** _y_ ) = **D** _**θ** d_ ( **h** _**θ** c_ ( **E** _**θ** e_ ( **1** _y_ ) _,_ **u** )), and the goal of autoencoder learning is to minimize the symbol error rate. Since the sampling function **h** _**θ** c_ of a Gaussian mixture channel is not directly differentiable, we apply the Gumbel-Softmax reparametrization (Jang et al., 2017) to obtain a differentiable sampling function (details in Appendix E). More background, including the training algorithm of the autoencoder, is in Appendix D. 

## 3 PROPOSED METHOD 

**Problem Setup.** Let **x** _, y,_ **z** denote a realization of the channel output, message (class label), and channel input (symbol) distributed according to the joint distribution _p_ ( **x** _, y,_ **z** ). We first establish the following result about the joint distribution. 

**Proposition 1.** _The joint distributions p_ ( **x** _, y,_ **z** ) _and p_ ( **x** _, y_ ) _can be expressed in the following form:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0004-06.png)


_where δ_ ( _·_ ) _is the Dirac delta (or Impulse) function, and we define p_ ( **x** _| y_ ) := _p_ ( **x** _|_ **E** _**θ** e_ ( **1** _y_ )) _as the conditional distribution of_ **x** _given the class y._ 

The proof is simple and given in Appendix A. Let _D[s]_ = _{_ ( **x** _[s] i[, y] i[s][,]_ **[ z]** _[s] i_[)] _[,][i]_[ = 1] _[,][ · · ·][, N][ s][}]_[ be a large] dataset from a source distribution _p[s]_ ( **x** _, y,_ **z** ) = _p[s]_ ( **x** _| y_ ) _p[s]_ ( _y_ ) _δ_ ( **z** _−_ **E** _**θ** e_ ( **1** _y_ )). The data collection involves sending multiple copies of each of the _m_ messages through the channel ( _e.g._ , over the air from the transmitter to receiver) by using a standard modulation technique (encoding) for **z** ( _e.g._ , M-QAM (Goldsmith, 2005)), and observing the corresponding channel output **x** . Different from conventional machine learning, where class labeling is expensive, in this setting the class label is simply the message transmitted, which is _obtained for free_ while collecting the data. The MDN channel model and autoencoder are trained on _D[s]_ according to Algorithm 1 (see Appendix D.3). 

Due to changes in the channel condition and environmental factors ( _e.g._ , moving obstacles), suppose the data distribution changes to _p[t]_ ( **x** _, y,_ **z** ) = _p[t]_ ( **x** _| y_ ) _p[t]_ ( _y_ ) _δ_ ( **z** _−_ **E** _**θ** e_ ( **1** _y_ )). While the distribution change may cause a drop in the autoencoder’s performance, we assume that it is gradual enough that domain adaptation is possible (David et al., 2010) (by domain, here 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0004-10.png)


**----- Start of picture text -----**<br>
Mixture Adaptation Adapted mixture<br>MDN channel model parameters Layer parameters<br>**----- End of picture text -----**<br>


Figure 2: Proposed MDN adaptation method. 

we mean the state of the communication channel during the time period when the MDN and autoencoder are trained). As discussed in § 1, the main challenge in this setting is to collect a sufficiently large dataset to retrain the MDN and autoencoder under the distribution shift. Therefore, suppose we collect a small dataset from the target distribution _D[t]_ = _{_ ( **x** _[t] i[, y] i[t][,]_ **[ z]** _[t] i_[)] _[,][i]_[ = 1] _[,][ · · ·][, N][ t][}]_[, where] _N[t] ≪ N[s]_ . _Our goal is to design a few-shot domain adaptation for the MDN and autoencoder in order to maintain or improve the symbol error rate._ 

**Distribution Change.** Referring to the joint distribution Eq. (2), the class prior _p_ ( _y_ ) is the prior probability of a message _y_ transmitted through the system. In this work, we make a reasonable practical assumption that this prior probability does not change, _i.e._ , _p[t]_ ( _y_ ) _≈ p[s]_ ( _y_ ) _, ∀y ∈Y_ . However, the class-conditional distribution of channel output _p_ ( **x** _| y_ ) changes, and therefore the class-posterior distribution _p_ ( _y |_ **x** ) also changes. This is commonly referred to as the _conditional shift_ assumption (Zhang et al., 2013) (different from covariate shift (Sugiyama et al., 2007)). 

**Overview of the Proposed Method.** Recall from Eqn. (1) that we model the channel distribution _p_ ( **x** _|_ **z** ) as a Gaussian mixture _P_ _**θ** c_ ( **x** _|_ **z** ), whose parameters are predicted by the MDN, _i.e._ , _**ϕ**_ ( **z** ) = 

4 

Published as a conference paper at ICLR 2023 

**M** _**θ** c_ ( **z** ). From Proposition 1, the _m class-conditional distributions_ of **x** are given by _p_ ( **x** _| y_ ) = _p_ ( **x** _|_ **E** _**θ** e_ ( **1** _y_ )) _, ∀y ∈Y_ . Therefore, in our setting, _adaptation of the class-conditional distributions is equivalent to adaptation of the m Gaussian mixtures_ in Eqn. (1. Adaptation of the Gaussian mixtures can be directly accomplished by adapting the MDN ( _i.e._ , the parameters _**θ** c_ ) using the small target-domain dataset _D[t]_ . Our proposed adaptation of the autoencoder consists of two _key steps_ : 

1. A light-weight, parameter-efficient adaptation of the MDN using the small target dataset _D[t]_ . 

2. An _efficient feature transformation_ at the input of the decoder (based on the MDN adaptation) that compensates for changes in the class-conditional distributions. 

Our method requires adaptation of _only the MDN (channel model)_ , while the encoder and decoder networks ( _**θ** e_ and _**θ** d_ ) remain unchanged, making it amenable to fast and frequent adaptation that requires collecting only a small target dataset each time (few-shot setting). 

## 3.1 MDN CHANNEL MODEL ADAPTATION 

Our goal is to adapt the _m_ Gaussian mixtures in Eqn (1) that model the source class-conditional distributions. Suppose the _m_ adapted Gaussian mixtures corresponding to the (unknown) target class-conditional distributions are 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0005-07.png)


where _**θ**_[�] _c_ are parameters of the adapted (target) MDN, and the component means, covariances, and prior probabilities with a hat notation are defined as in § 2. The adapted MDN predicts all the parameters of the target Gaussian mixture as _**ϕ**_[�] ( **z** ) = **M** _**θ**_ � _c_ ( **z** ) as shown in Fig. 2, where _**ϕ**_[�] ( **z** ) is defined in the same way as _**ϕ**_ ( **z** ). Instead of naively fine-tuning all the MDN parameters _**θ** c_ , or even just the final fully-connected layer[4] , we propose a parameter-efficient adaptation of the MDN based on the affine-transformation property of the Gaussian distribution, _i.e._ , one can transform between any two multivariate Gaussians through a general affine transformation. First, we state some _basic assumptions_ required to make the proposed adaptation tractable. 

- **A1)** The source and target Gaussian mixtures per class have the same number of components _k_ . **A2)** The source and target Gaussian mixtures (from each class) have a one-to-one correspondence between their components. 

Assumption A1 is made in order to not have to change the architecture of the MDN during adaptation due to adding or removing of components. Both assumptions A1 and A2[5] make it tractable to find the closed-form expression for a simplified KL-divergence between the source and target Gaussian mixtures per class (see Proposition 2). 

**Parameter Transformations.** As shown in Appendix B.2, the transformations between the source and target Gaussian mixture parameters, for any symbol **z** _∈Z_ and component _i ∈_ [ _k_ ], are given by 

� � � _**µ** i_ ( **z** ) = **A** _i_ _**µ** i_ ( **z** ) + **b** _i,_ **Σ** _i_ ( **z** ) = **C** _i_ **Σ** _i_ ( **z** ) **C** _[T] i[,]_[and] _αi_ ( **z** ) = _βi αi_ ( **z** ) + _γi._ (4) 

The affine transformation parameters **A** _i ∈_ R _[d][×][d]_ and **b** _i ∈_ R _[d]_ transform the means, **C** _i ∈_ R _[d][×][d]_ transforms the covariance matrix, and _βi, γi ∈_ R transform the prior logits. The vector of all _adaptation parameters_ to be optimized is defined by _**ψ**[T]_ = [ _**ψ**_ 1 _[T][,][ · · ·][,]_ _**[ ψ]** k[T]_[]][, where] _**[ψ]**[i]_[ contains all the] affine-transformation parameters from component _i_ . The number of adaptation parameters is given by _k_ (2 _d_[2] + _d_ + 2). This is typically much smaller than the number of MDN parameters (weights and biases from all layers), even if we consider only the final fully-connected layer for fine-tuning (see Table 1). In Fig. 2, the adaptation layer mapping _**ϕ**_ ( **z** ) to _**ϕ**_[�] ( **z** ) basically implements the parameter transformations defined in Eqn. (4). We observe that the affine-transformation parameters are not dependent on the symbol **z** (or the class), which is a constraint we impose in order to keep the number of adaptation parameters small. This is also consistent with the MDN parameters _**θ** c_ being independent of the symbol **z** . Allowing the affine transformations to depend of **z** would provide more flexibility, but at the same time require more target domain data for successful adaptation. 

> 4We show in our experiments that both the fine-tuning approaches fail to adapt well. 

> 5We perform ablation experiments (Appendix C.4) that evaluate our method under random Gaussian mixtures with mismatched components. We find that our method is robust even when these assumptions are violated. 

5 

Published as a conference paper at ICLR 2023 

**Proposition 2.** _Given m Gaussian mixtures from the source domain and m Gaussian mixtures from the target domain (one each per class), which satisfy Assumptions A1 and A2, the KL-divergence between P_ _**θ** c_ ( **x** _, K |_ **z** ) _and P_ _**θ**_ � _c_ ( **x** _, K |_ **z** ) _can be computed in closed-form, and is given by:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0006-02.png)


_where K is the mixture component random variable. The first term is the KL-divergence between the component prior probabilities, which simplifies into a function of the parameters_ [ _β_ 1 _, γ_ 1 _, · · · , βk, γk_ ] _. The second term involves the KL-divergence between two multivariate Gaussians (a standard result), which also simplifies into a function of_ _**ψ** ._ 

The proof and the final expression for the KL-divergence as a function of _**ψ**_ are given in Appendix A.1. The symbol priors _{p_ ( **z** ) _,_ **z** _∈Z}_ are estimated using the class proportions from the source dataset _D[s]_ . We note that this result is different from the KL-divergence between two arbitrary Gaussian mixtures, for which there is no closed-form expression (Hershey & Olsen, 2007). 

## 3.2 REGULARIZED ADAPTATION OBJECTIVE 

From the above analysis, we can formulate the MDN adaptation as the equivalent problem of finding the optimal set of affine transformations (one per-component) mapping the source to the target Gaussian mixtures. To reduce the possibility of the adaptation finding bad solutions due to the small-sample setting, we introduce a regularization term based on the KL-divergence (defined earlier), which constrains the distribution shift produced by the affine transformations. We consider two scenarios for adaptation: 1) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0006-07.png)


**----- Start of picture text -----**<br>
Feature Predicted message<br>Wireless channel  transformation<br>Decoder / Receiver<br>neural network<br>Dense layer  +<br>Multiple dense layers softmax activation<br>**----- End of picture text -----**<br>


Figure 3: Proposed decoder adaptation using feature transformations. 

_Generative adaptation_ of the MDN in isolation and 2) _Discriminative adaptation_ of the MDN as part of the autoencoder. In the first case, the goal of adaptation is to find a good generative model for the target channel distribution, while in the second case the goal is to improve the classification accuracy of the autoencoder on the target distribution. We focus on the discriminative adaptation here, and present the very similar generative adaptation in Appendix B.3. 

Since the goal of adaptation is to improving the decoder’s accuracy in recovering the transmitted symbol **z** from the channel output **x** , we use the (negative) symbol posterior log-likelihood (PLL) as the first data-dependent term of the adaptation objective. The second term is the simplified KLdivergence between the source and target Gaussian mixtures, which does _not_ depend on the data. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0006-11.png)


The symbol posterior _P_ _**θ**_ � _c_ ( **z** _|_ **x** ) is computed from the conditional _P_ _**θ**_ � _c_ ( **x** _|_ **z** ) and the symbol priors _{p_ ( **z** ) _,_ **z** _∈Z}_ using Bayes rule. We observe that the adaptation objective is a smooth and nonconvex function of _**ψ**_ . Also, computation of the objective and its gradient (w.r.t _**ψ**_ ) are inexpensive operations since **i)** they _do not require forward and back-propagation_ through the layers of the MDN and **ii)** both _N[t]_ and the dimension of _**ψ**_ are small. Therefore, we use the BFGS Quasi-Newton method (Nocedal & Wright, 2006) for minimization, instead of SGD-based large-scale optimization ( _e.g._ , Adam). The regularization constant _λ_ is a hyper-parameter of the proposed method, and we _propose a validation metric_ in Appendix B.4) to set its value automatically. 

## 3.3 DECODER ADAPTATION USING FEATURE TRANSFORMATIONS 

We propose a computationally-efficient feature transformation **g** _[−]_[1] : R _[d] �→_ R _[d]_ at the decoder such � that the transformed inputs **x** _[s]_ = **g** _[−]_[1] ( **x** _[t]_ ) are closely aligned to the source distribution on which 

6 

Published as a conference paper at ICLR 2023 

the decoder was trained (see Fig. 3). This is based on the optimal affine-transformations _**ψ**_ of the MDN found by minimizing Eqn. (6). This method does not require any change to the trained encoder and decoder networks, making it well suited for the few-shot DA setting. 

Consider a test input **x** _[t]_ at the decoder from the target-domain marginal distribution _p[t]_ ( **x** ) = � **z** _∈Z[p]_[(] **[z]**[)][�] _[k] i_ =1 _[π]_[�] _[i]_[(] **[z]**[)] _[ N]_ � **x** _|_ � _**µ** i_ ( **z** ) _,_ **Σ**[�] _i_ ( **z** )�. As shown in Appendix B.2, conditioned on a given symbol **z** _∈Z_ and component _i ∈_ [ _k_ ], the affine transformation that maps from the target Gaussian distribution **x** _[t] |_ **z** _, i ∼ N_ ( **x** _|_ � _**µ** i_ ( **z** ) _,_ **Σ**[�] _i_ ( **z** )) to the source Gaussian distribution **x** _[s] |_ **z** _, i ∼ N_ ( **x** _|_ _**µ** i_ ( **z** ) _,_ **Σ** _i_ ( **z** )) is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0007-03.png)


However, this transformation requires knowledge of both the transmitted symbol **z** and the mixture component _i_ , which are not observed at the decoder (the decoder only observes the channel output **x** _[t]_ ). We address this by taking the expected affine transformation from target to source, where the expectation is with respect to the joint posterior over the symbol **z** and component _i_ , given the channel output **x** _[t]_ . This posterior distribution based on the target Gaussian mixture is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0007-05.png)


The expected inverse-affine feature transformation at the decoder is then defined as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0007-07.png)


We show that this conditional expectation is the _optimal transformation_ from the standpoint of meansquared-error estimation (Kay, 1993) in Appendix A.2. The adapted decoder based on this feature transformation is illustrated in Fig. 3 and defined as **D**[�] _**θ** d_ ( **x** _[t]_ ; _**ψ**_ ) := **D** _**θ** d_ ( **g** _[−]_[1] ( **x** _[t]_ )). For small to moderate number of symbols _m_ and number of components _k_ , this transformation is computationally efficient and easy to implement at the receiver of a communication system. A discussion of the computational complexity of the proposed method is given in Appendix B.5. 

## 4 EXPERIMENTS 

We perform experiments to evaluate the proposed adaptation method for the MDN and autoencoder. Our main findings are summarized as follows: **1)** the proposed method adapts well to changes in the channel distribution using only a few samples per class, often leading to strong improvement over the baselines; **2)** our method performs well under multiple simulated distribution changes, and notably on our mmWave FPGA experiments; **3)** Extensive ablation studies show that the proposed KL-divergence based regularization and the validation metric for setting _λ_ are effective. 

**Setup.** We implemented the MDN, autoencoder networks, and the adaptation methods in Python using TensorFlow (Abadi et al., 2015) and TensorFlow Probability. We used the following setting in our experiments. The size of the message set _m_ is fixed to 16, corresponding to 4 bits. The dimension of the encoding (output of the encoder) _d_ is set to 2, and the number of mixture components _k_ is set to 5. More details on the experimental setup, neural network architecture, and the hyper-parameters are given in Appendix C.1. 

**Baseline Methods.** We compare the performance of our method with the following baselines: **1) No adaptation** , which is the MDN and autoencoder from the source domain without adaptation. **2) Retrained MDN and autoencoder** , which is like an “oracle method” that has access to a large dataset from the target domain. **3) Finetune** - where the method optimizes all the MDN parameters for 200 epochs and optimizes the decoder for 20 epochs[6] . **4) Finetune last** - which follows the same approach as “Finetune”, but only optimizes the last layer of MDN (all the layers of the decoder are however optimized). We note that traditional domain adaptation methods are not suitable for this problem because it requires adaptation of both the MDN (generative model) and the decoder. 

**Datasets.** The simulated channel variations are based on models commonly used for wireless communication, specifically: i) Additive white Gaussian noise (AWGN), ii) Ricean fading, and iii) 

> 6We found no significant gains with larger number of epochs in this case. 

7 

Published as a conference paper at ICLR 2023 

Uniform or flat fading (Goldsmith, 2005). Details on these channel models and calculation of the their signal-to-noise ratio (SNR) are provided in Appendix F. We also created simulated distribution changes using random, class-conditional Gaussian mixtures for both the source and target channels (we also include random phase shifts). The parameters of the source and target Gaussian mixtures are generated in a random but controlled manner as detailed in Appendix C.3. We also evaluate the performance of the adaptation methods on real over-the-air wireless experiments. We use a recent high-performance mmWave testbed (Lacruz et al., 2021), featuring a high-end FPGA board with 2 GHz bandwidth per-channel and 60 GHz SIVERS antennas (SIVERSIMA, 2020). We introduce distribution changes via (In-phase and Quadrature-phase) IQ imbalance-based distortions to the symbol constellation, and gradually increase the level of imbalance to the system[7] . More details on the FPGA experimental setup are given in Appendix C.2. 

**Evaluation Protocol.** Due to the space limit, we provide details of the evaluation protocol such as train, adaptation, and test sample sizes, and the number of random trials used to get averaged performance in Appendix C.1. We report the symbol error rate (SER) on a large held-out test dataset (from the target domain) as a function of the number of target-domain samples per class. The only hyper-parameter _λ_ of our method is set automatically using the validation metric proposed in B.4. 

## 4.1 AUTOENCODER ADAPTATION ON SIMULATED DISTRIBUTION CHANGES 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0008-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0008-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0008-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0008-07.png)


**----- Start of picture text -----**<br>
(a) AWGN to Uniform fading. (b) Ricean fading to Uniform fading. (c) AWGN to Ricean fading.<br>**----- End of picture text -----**<br>


Figure 4: Autoencoder adaptation on distribution shifts based on standard channel models. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0008-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0008-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0008-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0008-12.png)


**----- Start of picture text -----**<br>
(a) Uniform fading to Ricean fading. (b) Random Gaussian mixtures. (c) Random Gaussian mixtures with<br>random phase shifts.<br>**----- End of picture text -----**<br>


Figure 5: Autoencoder adaptation on distribution shifts based on standard channel models and random Gaussian mixtures. In figure (c), the target domain additionally includes random phase shifts. 

The adaptation results under simulated distributions changes are given in Figs. 4 and 5, with the symbol error rates plotted as a function of the number of target samples per class. In Fig. 4, we consider standard channel distributions such as AWGN, Ricean fading, and Uniform fading. In Fig. 5, we consider random Gaussian mixtures for both the source and the target distributions. We observe that the proposed adaptation leads to a strong improvement in SER in all cases, except in the case of AWGN to Ricean fading (Fig. 4. c). We provide some insights on the failure of our method in this case in Appendix C.5. Note that the methods “No adapt” and “Retrained autoenc” have the same SER for all target sample sizes ( _i.e._ , a horizontal line). We find both the finetuning baselines to 

> 7IQ imbalance is a common issue in RF communication that introduces distortions to the final constellation. 

8 

Published as a conference paper at ICLR 2023 

have very similar SER in all cases, and there is not much improvement compared to no adaptation. This suggests that our approach of constraining the number of adaptation parameters and using the KL-divergence regularization are effective in the few-shot DA setting (see Table 1). 

## 4.2 AUTOENCODER ADAPTATION ON FPGA EXPERIMENTS 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0009-03.png)


**----- Start of picture text -----**<br>
No adaptAffine adapt FinetuneFinetune Last Retrained autoenc No adaptAffine adapt FinetuneFinetune Last Retrained autoenc No adaptAffine adapt FinetuneFinetune Last Retrained autoenc<br>0.1 0.2 0.2<br>0.05 0.1<br>0.1<br>0.05<br>0.02 0.05<br>0.01 0.02<br>0.005 0.01 0.02<br>0.01<br>10 20 30 40 50 10 20 30 40 50 10 20 30 40 50<br>#target samples per-class #target samples per-class #target samples per-class<br>(a) 20% IQ imbalance (b) 25% IQ imbalance (c) 30% IQ imbalance<br>Symbol error Rate Symbol error Rate Symbol error Rate<br>**----- End of picture text -----**<br>


Figure 6: Autoencoder adaptation on the FPGA experiments with distribution change based on IQ imbalance. A higher IQ imbalance results in more distribution change. 

For this experiment, different levels of distribution change are introduced by varying the IQ imbalance over 20%, 25%, and 30% (higher IQ imbalance corresponds to larger distribution change). From Fig. 6, we observe that the proposed method achieves significant reduction in error rate compared to the (non-oracle) baselines. The relative improvement in SER over the baselines is more pronounced under higher IQ imbalance. For instance, at 30% IQ imbalance, our method achieves a relative SER improvement of around 69% over the fine-tuning baselines using only 10 samples per-class. 

## 4.3 ADDITIONAL EXPERIMENTS 

We have performed a number of additional experiments including ablation studies, which are reported in Appendix C.4 through C.6. They include: **1)** evaluating the proposed validation metric for automatically setting the hyper-parameter _λ_ ; **2)** evaluating the importance of the KLdivergence regularization in the adaptation objective; **3)** performance of our method when the source and target Gaussian mixtures have a mismatch in the components (addressing As- 

Table 1: Number of parameters being optimized by the MDN adaptation methods. 

||Adaptation<br>method<br>Finetune|# parameters<br>_nh_(_nh_+_d_+ 2)<br>+_k_(2_d_+ 1) (_nh_+ 1)|# parameters<br>(specifc)<br>12925|
|---|---|---|---|
||Finetune-last-layer<br>Proposed|_k_(2_d_+ 1) (_nh_+ 1)<br>_k_(_d_2 + 2_d_+ 2)|2525<br>50|



sumptions A1 and A2); **4)** performance of our method when there is no distribution shift; and **5)** performance of the generative adaptation of the MDN channel. To summarize the observations, we found the validation metric to be effective at setting the value of _λ_ , and that our method has good performance even when Assumptions A1 and A2 are violated, or when there is no distribution shift. The generative MDN adaptation leads to increased log-likelihoods with as low as 2 samples per class. 

## 5 CONCLUSIONS 

In this work, we explore one of the first approaches for domain adaptation of autoencoder based e2e communication in the few-shot setting. We first propose a light-weight and parameter-efficient method for adapting a Gaussian MDN with a very small number of samples from the target distribution. Based on the MDN adaptation, we propose an optimal input transformation method at the decoder that attempts to closely align the target domain inputs to the source domain. We demonstrate the effectiveness of the proposed methods through extensive experiments on both simulated channels and a mmWave FPGA testbed. A discussion of limitations and future directions is given in Appendix B.6. 

9 

Published as a conference paper at ICLR 2023 

## ACKNOWLEDGMENTS 

Banerjee, Raghuram, and Zeng were supported in part through the following grants — US National Science Foundation’s CNS-2112562, CNS-2107060, CNS-2003129, CNS-1838733, and CNS-1647152, and the US Department of Commerce’s 70NANB21H043. Somesh Jha was partially supported by the DARPA-GARD problem under agreement number 885000. The authors from IMDEA Networks were sponsored by the Spanish Ministry of Economic Affairs and Digital Transformation under European Union NextGeneration-EU projects TSI-063000-2021-59 RISC-6G and TSI-063000-2021-63 MAP-6G, and by the Regional Government of Madrid and the European Union through the European Regional Development Fund (ERDF) project REACT-CONTACT-CM-23479. 

## REFERENCES 

- Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dandelion Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. URL https://www.tensorflow. org/. Software available from tensorflow.org. 

- Fayçal Ait Aoudia and Jakob Hoydis. End-to-end learning of communications systems without a channel model. In _52nd Asilomar Conference on Signals, Systems, and Computers (ACSSC)_ , pp. 298–303. IEEE, 2018. doi: 10.1109/ACSSC.2018.8645416. URL https://doi.org/10.1109/ACSSC.2018.8645416. 

- Fayçal Ait Aoudia and Jakob Hoydis. Model-free training of end-to-end communication systems. _IEEE Journal on Selected Areas in Communication_ , 37(11):2503–2516, 2019. doi: 10.1109/JSAC.2019.2933891. URL https://doi.org/10.1109/JSAC.2019.2933891. 

- Shai Ben-David, John Blitzer, Koby Crammer, and Fernando Pereira. Analysis of representations for domain adaptation. In _Advances in Neural Information Processing Systems 19, Proceedings of the Twentieth Annual Conference on Neural Information Processing Systems_ , pp. 137– 144. MIT Press, 2006. URL https://proceedings.neurips.cc/paper/2006/hash/ b1b0432ceafb0ce714426e9114852ac7-Abstract.html. 

- Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman Vaughan. A theory of learning from different domains. _Machine Learning_ , 79(1-2):151–175, 2010. doi: 10.1007/ s10994-009-5152-4. URL https://doi.org/10.1007/s10994-009-5152-4. 

- Christopher M. Bishop. Mixture density networks. Technical report, Aston University, 1994. http:// publications.aston.ac.uk/id/eprint/373/. 

- Christopher M. Bishop. _Pattern recognition and machine learning, 5th Edition_ , chapter 5, pp. 272–277. Information Science and Statistics. Springer, 2007. ISBN 9780387310732. URL https://www.worldcat. org/oclc/71008143. 

- John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman. Learning bounds for domain adaptation. In _Advances in Neural Information Processing Systems 20, Proceedings of the Twenty-First Annual Conference on Neural Information Processing Systems_ , pp. 129–136. Curran Associates, Inc., 2007. URL https://proceedings.neurips.cc/paper/2007/hash/ 42e77b63637ab381e8be5f8318cc28a2-Abstract.html. 

- Shai Ben David, Tyler Lu, Teresa Luu, and Dávid Pál. Impossibility theorems for domain adaptation. In _Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS)_ , pp. 129–136. JMLR Workshop and Conference Proceedings, 2010. 

- Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using Real NVP. In _5th International Conference on Learning Representations (ICLR), Conference Track Proceedings_ . OpenReview.net, 2017. URL https://openreview.net/forum?id=HkpbnH9lx. 

- Sebastian Dörner, Sebastian Cammerer, Jakob Hoydis, and Stephan ten Brink. Deep learning based communication over the air. _IEEE Journal of Selected Topics in Signal Processing_ , 12(1):132–143, 2018. doi: 10.1109/JSTSP.2017.2784180. URL https://doi.org/10.1109/JSTSP.2017.2784180. 

- Marthinus Christoffel Du Plessis and Masashi Sugiyama. Semi-supervised learning of class balance under class-prior change by distribution matching. _Neural Networks_ , 50:110–119, 2014. 

10 

Published as a conference paper at ICLR 2023 

- Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In Doina Precup and Yee Whye Teh (eds.), _Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017_ , volume 70 of _Proceedings of Machine Learning Research_ , pp. 1126–1135. PMLR, 2017. 

- Yaroslav Ganin and Victor S. Lempitsky. Unsupervised domain adaptation by backpropagation. In _Proceedings of the 32nd International Conference on Machine Learning (ICML)_ , volume 37 of _JMLR Workshop and Conference Proceedings_ , pp. 1180–1189. JMLR.org, 2015. URL http://proceedings.mlr.press/ v37/ganin15.html. 

- Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor S. Lempitsky. Domain-adversarial training of neural networks. _Journal of Machine Learning Research_ , 17:59:1–59:35, 2016. URL http://jmlr.org/papers/v17/15-239.html. 

- Dolores García Martí, Joan Palacios Beltrán, Jesús Omar Lacruz, and Joerg Widmer. A mixture density channel model for deep learning-based wireless physical layer design. In _Proceedings of the 23rd International ACM Conference on Modeling, Analysis and Simulation of Wireless and Mobile Systems_ , pp. 53–62, 2020. 

- Andrea Goldsmith. _Wireless Communications_ . Cambridge University Press, 2005. 

- John R. Hershey and Peder A. Olsen. Approximating the Kullback-Leibler divergence between Gaussian mixture models. In _Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)_ , pp. 317–320. IEEE, 2007. doi: 10.1109/ICASSP.2007.366913. URL https://doi.org/10. 1109/ICASSP.2007.366913. 

- Jakob Hoydis, Fayçal Ait Aoudia, Alvaro Valcarce, and Harish Viswanathan. Toward a 6g ai-native air interface. _IEEE Communications Magazine_ , 59(5):76–81, 2021. 

- Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with Gumbel-Softmax. In _5th International Conference on Learning Representations (ICLR), Conference Track Proceedings_ . OpenReview.net, 2017. URL https://openreview.net/forum?id=rkE3y85ee. 

- Fredrik D. Johansson, David A. Sontag, and Rajesh Ranganath. Support and invertibility in domain-invariant representations. In _The 22nd International Conference on Artificial Intelligence and Statistics (AISTATS)_ , volume 89 of _Proceedings of Machine Learning Research_ , pp. 527–536. PMLR, 2019. URL http:// proceedings.mlr.press/v89/johansson19a.html. 

- Steven M Kay. _Fundamentals of statistical signal processing: Estimation theory_ . Prentice-Hall, Inc., 1993. 

- Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _3rd International Conference on Learning Representations (ICLR), Conference Track Proceedings_ , 2015. URL http:// arxiv.org/abs/1412.6980. 

- N Kostantinos. Gaussian mixtures and their applications to signal processing. _Advanced signal processing handbook: Theory and implementation for radar, sonar, and medical imaging real time systems_ , pp. 3–1, 2000. 

- Jesus O. Lacruz, Rafael Ruiz, and Joerg Widmer. A real-time experimentation platform for sub-6 ghz and millimeter-wave MIMO systems. In _ACM MobiSys’21_ , 2021. doi: 10.1145/3458864.3466868. 

- Jean Paul Linnartz. Ricean fading. http://www.wirelesscommunication.nl/reference/ chaptr03/ricepdf/rice.htm, 2001. 

- Mingsheng Long, Yue Cao, Jianmin Wang, and Michael I. Jordan. Learning transferable features with deep adaptation networks. In _Proceedings of the 32nd International Conference on Machine Learning (ICML)_ , volume 37 of _JMLR Workshop and Conference Proceedings_ , pp. 97–105. JMLR.org, 2015. URL http: //proceedings.mlr.press/v37/long15.html. 

- Mingsheng Long, Han Zhu, Jianmin Wang, and Michael I. Jordan. Unsupervised domain adaptation with residual transfer networks. In _Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems_ , pp. 136–144, 2016. URL https://proceedings.neurips. cc/paper/2016/hash/ac627ab1ccbdb62ec96e702f07f6425b-Abstract.html. 

- Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Michael I. Jordan. Conditional adversarial domain adaptation. In _Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems_ , pp. 1647–1657, 2018. URL https://proceedings.neurips.cc/ paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html. 

11 

Published as a conference paper at ICLR 2023 

- Saeid Motiian, Quinn Jones, Seyed Mehdi Iranmanesh, and Gianfranco Doretto. Few-shot adversarial domain adaptation. In _Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems_ , pp. 6670–6680, 2017a. URL https://proceedings.neurips.cc/ paper/2017/hash/21c5bba1dd6aed9ab48c2b34c1a0adde-Abstract.html. 

- Saeid Motiian, Marco Piccirilli, Donald A. Adjeroh, and Gianfranco Doretto. Unified deep supervised domain adaptation and generalization. In _IEEE International Conference on Computer Vision ICCV_ , pp. 5716–5726. IEEE Computer Society, 2017b. doi: 10.1109/ICCV.2017.609. URL https://doi.org/10.1109/ ICCV.2017.609. 

- Jorge Nocedal and Stephen Wright. _Numerical Optimization_ , chapter 5–7. Springer Science & Business Media, 2006. 

- Timothy J. O’Shea and Jakob Hoydis. An introduction to deep learning for the physical layer. _IEEE Transactions on Cognitive Communications and Networking_ , 3(4):563–575, 2017. doi: 10.1109/TCCN.2017.2758370. URL https://doi.org/10.1109/TCCN.2017.2758370. 

- Timothy J. O’Shea, Tamoghna Roy, and Nathan West. Approximating the void: Learning stochastic channel models from observation with variational generative adversarial networks. In _International Conference on Computing, Networking and Communications (ICNC)_ , pp. 681–686. IEEE, 2019. doi: 10.1109/ICCNC.2019. 8685573. URL https://doi.org/10.1109/ICCNC.2019.8685573. 

- Marco Saerens, Patrice Latinne, and Christine Decaestecker. Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure. _Neural Computation_ , 14(1):21–41, 2002. doi: 10.1162/ 089976602753284446. URL https://doi.org/10.1162/089976602753284446. 

- Kuniaki Saito, Kohei Watanabe, Yoshitaka Ushiku, and Tatsuya Harada. Maximum classifier discrepancy for unsupervised domain adaptation. In _IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 3723–3732. IEEE Computer Society, 2018. doi: 10.1109/CVPR. 2018.00392. URL http://openaccess.thecvf.com/content_cvpr_2018/html/Saito_ Maximum_Classifier_Discrepancy_CVPR_2018_paper.html. 

- SIVERSIMA. _EVK06002 Development Kit_ , 2020. https://www.siversima.com/product/ evk-06002-00/. 

- Masashi Sugiyama and Motoaki Kawanabe. _Machine Learning in Non-stationary Environments: Introduction to Covariate Shift Adaptation_ . MIT Press, 2012. 

- Masashi Sugiyama, Matthias Krauledat, and Klaus-Robert Müller. Covariate shift adaptation by importance weighted cross validation. _Journal of Machine Learning Research_ , 8:985–1005, 2007. URL http://dl. acm.org/citation.cfm?id=1390324. 

- Qianru Sun, Yaoyao Liu, Tat-Seng Chua, and Bernt Schiele. Meta-transfer learning for few-shot learning. In _IEEE Conference on Computer Vision and Pattern Recognition CVPR_ , pp. 403–412. Computer Vision Foundation / IEEE, 2019. doi: 10.1109/CVPR.2019.00049. 

- Tianqi Wang, Chao-Kai Wen, Hanqing Wang, Feifei Gao, Tao Jiang, and Shi Jin. Deep learning for wireless physical layer: Opportunities and challenges. _CoRR_ , abs/1710.05312, 2017. URL http://arxiv.org/ abs/1710.05312. 

- William Xia, Sundeep Rangan, Marco Mezzavilla, Angel Lozano, Giovanni Geraci, Vasilii Semkin, and Giuseppe Loianno. Millimeter wave channel modeling via generative neural networks. In _IEEE Globecom Workshops (GC Wkshps)_ , pp. 1–6. IEEE, 2020. doi: 10.1109/GCWkshps50303.2020.9367420. URL https://doi.org/10.1109/GCWkshps50303.2020.9367420. 

- Hao Ye, Geoffrey Ye Li, Biing-Hwang Fred Juang, and Kathiravetpillai Sivanesan. Channel agnostic end-to-end learning based communication systems with conditional GAN. In _IEEE Globecom Workshops (GC Wkshps)_ , pp. 1–5. IEEE, 2018. doi: 10.1109/GLOCOMW.2018.8644250. URL https://doi.org/10.1109/ GLOCOMW.2018.8644250. 

- Kun Zhang, Bernhard Schölkopf, Krikamol Muandet, and Zhikun Wang. Domain adaptation under target and conditional shift. In _Proceedings of the 30th International Conference on Machine Learning (ICML)_ , volume 28 of _JMLR Workshop and Conference Proceedings_ , pp. 819–827. JMLR.org, 2013. URL http: //proceedings.mlr.press/v28/zhang13d.html. 

- An Zhao, Mingyu Ding, Zhiwu Lu, Tao Xiang, Yulei Niu, Jiechao Guan, and Ji-Rong Wen. Domain-adaptive few-shot learning. In _IEEE Winter Conference on Applications of Computer Vision, WACV_ , pp. 1389–1398. IEEE, 2021. doi: 10.1109/WACV48630.2021.00143. URL https://doi.org/10.1109/WACV48630. 2021.00143. 

12 

Published as a conference paper at ICLR 2023 

Han Zhao, Remi Tachet des Combes, Kun Zhang, and Geoffrey J. Gordon. On learning invariant representations for domain adaptation. In _Proceedings of the 36th International Conference on Machine Learning (ICML)_ , volume 97 of _Proceedings of Machine Learning Research_ , pp. 7523–7532. PMLR, 2019. URL http: //proceedings.mlr.press/v97/zhao19a.html. 

13 

Published as a conference paper at ICLR 2023 

## **Appendix** 

Table 2: Commonly used notations 

|**Notation**||**Description**|
|---|---|---|
|_y ∈Y_ :=_{_1_, · · · , m}_||Input message or class label. Usually_m_= 2_b_, where_b_is the number of bits.|
|**1**_y, y ∈Y_||One-hot-coded representation of a label (message)_y_, with1at position_y_and zeros elsewhere.|
|**z**_∈Z ⊂_R_d_ with|_|Z|_=_m_|Encoded representation or symbol vector corresponding to an input message.|
|**x**_∈_R_d_||Channel output that is the feature vector to be classifed by the decoder.|
|**E****_θ_**_e_(**1**_y_)||Encoder NN with parameters**_θ_**_e_ mapping a one-hot-coded message to a symbol vector inR_d_.|
|**D****_θ_**_d_(**x**) = [_P_**_θ_**_d_(1_|_**x**)_, · · · , P_**_θ_**_d_(_m |_**x**)]||Decoder NN with parameters**_θ_**_d_mapping the channel output into probabilities over the message set.|
|�_y_(**x**) = argmax_y∈Y P_**_θ_**_d_(_y |_**x**)||Class (message) prediction of the decoder.|
|_P_**_θ_**_c_(**x**_|_**z**)||Conditional density (generative) model of the channel with parameters**_θ_**_c_.|
|**_ϕ_**(**z**) = **M****_θ_**_c_(**z**)||Mixture density network that predicts the parameters of a Gaussian mixture.|
|**x** = **h****_θ_**_c_(**z**_,_**u**)||Transfer or sampling function corresponding to the channel conditional density.|
|**f****_θ_**(**1**_y_) = **D****_θ_**_d_(**h****_θ_**_c_(**E****_θ_**_e_(**1**_y_)_,_**u**))||Input-output mapping of the autoencoder with combined parameter vector **_θ_**_T_ = [**_θ_**_T_<br>_e ,_**_θ_**_T_<br>_c ,_**_θ_**_T_<br>_d_ ].|
|**_ψ_**_T_ = [**_ψ_**_T_<br>1 _, · · · ,_**_ψ_**_T_<br>_k_ ]||Affne transformation (adaptation) parameters per component used to adapt the MDN.|
|**gz**_i_ and **g**_−_1<br>**z**_i , i ∈_[_k_]_,_**z**_∈Z_||Affne transformations between the components of the source-to-target Gaussian mixtures and vice-verse.|
|_D_KL(_p, q_)||Kullback-Leibler divergence between the distributions_p_and_q_.|
|_N_(_· |_**_µ_**_,_**Σ**)||Multivariate Gaussian density with mean vector**_µ_**and covariance matrix**Σ**.|
|_δ_(**x**_−_**x**0)||Dirac delta or impulse function centered at**x**0.|
|Cat(_p_1_, · · · , pk_)||Categorical distribution with _pi ≥_0 and �<br>_i pi_ = 1.|
|1(_c_)||Indicator function mapping a predicate_c_to1if true and0if false.|
|_∥_**x**_∥p_||_ℓp_norm of a vector**x**.|



The appendices are organized as follows: 

- Appendix A discusses the theoretical results from the main paper. 

- Appendix B provides additional details on the proposed method including: 

- Discussion on class labels and labeled data in the communication setting (Appendix B.1). 

- Feature and parameter transformation between multivariate Gaussians (Appendix B.2). 

- Generative adaptation of the MDN channel (Appendix B.3). 

- The validation metric used for setting the hyper-parameter _λ_ (Appendix B.4). 

- Computational complexity analysis of the proposed method (Appendix B.5). 

- Limitations and future work (Appendix B.6). 

- Appendix C provides additional details on the experiments and additional results, including ablation studies of the proposed method. 

- Appendix D provides additional background on the following topics: 1) components of an endto-end autoencoder-based communication system, 2) generative modeling using mixture density networks, 3) training algorithm of the autoencoder, and 4) a primer on domain adaptation. 

- Appendix E provides details on the MDN training and differentiable sampling using the Gumbelsoftmax reparametrization. 

- Appendix F provides details on the simulated channel distributions used in our experiments. 

## A THEORETICAL RESULTS 

**Propostion 1 (restatement).** The joint distributions _p_ ( **x** _, y,_ **z** ) and _p_ ( **x** _, y_ ) can be expressed in the following form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0014-19.png)


where _δ_ ( _·_ ) is the Dirac delta (or Impulse) function, and we define _p_ ( **x** _| y_ ) := _p_ ( **x** _|_ **E** _**θ** e_ ( **1** _y_ )) as the conditional distribution of **x** given the class _y_ . 

**Proof.** It follows from the dependence _y →_ **z** _→_ **x** defined by our generative model that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0014-22.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0014-23.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0014-24.png)


In the second step, the conditional _p_ ( **z** _| y_ ) reduces to the Dirac delta since the symbol **z** can only take one of the _m_ values from the constellation _Z_ = _{_ **E** _**θ** e_ ( **1** 1) _, · · · ,_ **E** _**θ** e_ ( **1** _m_ ) _}_ (for a fixed encoder mapping). The distribution _p_ ( **x** _, y_ ) in Eq. (9) is obtained from the third step by integrating _p_ ( **x** _, y,_ **z** ) over all **z** , and using the integration property of the Dirac delta. 

14 

Published as a conference paper at ICLR 2023 

## A.1 KL-DIVERGENCE BETWEEN THE SOURCE AND TARGET GAUSSIAN MIXTURES 

**Propostion 2 (restatement).** Given _m_ Gaussian mixtures from the source domain and _m_ Gaussian mixtures from the target domain (one each per class), which satisfy Assumptions A1 and A2, the KL-divergence between _P_ _**θ** c_ ( **x** _, K |_ **z** ) and _P_ _**θ**_ � _c_ ( **x** _, K |_ **z** ) can be computed in closed-form, and is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0015-03.png)


where _K_ is the mixture component random variable. The first term is the KL-divergence between the component prior probabilities, which simplifies into a function of the parameters [ _β_ 1 _, γ_ 1 _, · · · , βk, γk_ ] . The second term involves the KL-divergence between two multivariate Gaussians (a standard result), which also simplifies into a function of _**ψ**_ . 

**Proof.** Referring to § 3.1, we derive the closed-form KL-divergence between the source and target Gaussian mixtures under Assumptions 1 and 2, _i.e._ , the source and target Gaussian mixtures have the same number of components that have a one-to-one association. Recall that _**θ** c_ and _**θ**_[�] _c_ are the parameters of the original (source) and the adapted (target) MDN respectively. Let _K ∈{_ 1 _, · · · , k}_ denote the latent component random variable. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0015-06.png)


The second term in the final expression involves the KL-divergence between two multivariate Gaussians (a standard result) given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0015-08.png)


For clarity, we further simplify Eq. (11) for the case of diagonal covariances by applying the above result. Recall that the Gaussian mixture parameters of the source and target domains are related by the parameter transformations in Eq. (4). The second term in Eq. (11) involving the KL-divergence between multivariate Gaussians, simplifies to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0015-10.png)


15 

Published as a conference paper at ICLR 2023 

The first term in Eq. (11) involving the KL-divergence between the component prior probabilties can be expressed as a function of the adaptation parameters [ _β_ 1 _, γ_ 1 _, · · · , βk, γk_ ] as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0016-02.png)


where _q_ ( **z** ) =[�] _[k] j_ =1 _[e][α][j]_[(] **[z]**[)][and] _[q]_[�][(] **[x]**[)][=][�] _[k] j_ =1 _[e][β][j][α][j]_[(] **[z]**[) +] _[ γ][j]_[are the normalization terms in the] softmax function. Substituting Eqs. (12) and (13) into the last step of Eq. (11) gives the KL-divergence between the source and target Gaussian mixtures as a function of the adaptation parameters _**ψ**_ . 

## A.2 OPTIMALITY OF THE FEATURE TRANSFORMATION 

We show that the proposed feature transformation at the decoder in § 3.3 is optimal in the mimimum mean-squared error sense. The problem setting is that, at the decoder, we observe an input **x** _[t]_ from the target domain marginal distribution, _i.e._ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0016-06.png)


where _Z_ = _{_ **E** _**θ** e_ ( **1** 1) _, · · · ,_ **E** _**θ** e_ ( **1** _m_ ) _}_ is the encoder’s constellation. Suppose we knew the symbol **z** = **E** _**θ** e_ ( **1** _y_ ) that was transmitted and the mixture component _i ∈_ [ _k_ ], then the transformation **gz** _[−] i_[1][(] **[x]** _[t]_[)][ in Eq.][(7) can map] **[ x]** _[t]_[ to the corresponding Gaussian component of the source distribution.] However, since **z** and _i_ are not observed at the decoder, we propose to find the transformation **g** _[−]_[1] : R _[d] �→_ R _[d]_ (independent of **z** and _i_ ) that minimizes the following expected squared error: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0016-08.png)


This is the conditional expectation over ( **z** _, i_ ) given **x** _[t]_ with respect to the posterior distribution _P_ _**θ**_ � _c_ ( **z** _, i |_ **x** ). Since **x** _[t]_ is fixed, the above objective is a function of the vector **w** := **g** _[−]_[1] ( **x** _[t]_ ) _∈_ R _[d]_ , and it can be simplified as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0016-10.png)


Note that **w** comes outside the expectation since it does not depend on **z** or _i_ . The minimum of this simple quadratic function can be found by setting the gradient of _J_ with respect to **w** to **0** , giving 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0016-12.png)


This is the feature transformation at the decoder proposed in § 3.3. 

## B ADDITIONAL DETAILS ON THE PROPOSED METHOD 

In this section we provide additional details on the proposed method that could not be discussed in § 3 of the main paper. 

## B.1 CLASS LABELS AND LABELED DATA 

We would like to clarify that the statement “class labels are available for free” is made in Section 3 in order to highlight the fact that class labels are easy to obtain in this end-to-end communication 

16 

Published as a conference paper at ICLR 2023 

setting, unlike other domains (e.g. computer vision) where labeling data could be expensive. Since the transmitted message is also the class label, it is always available without additional effort during the data collection (from the packet preambles). However, note that it is still challenging / expensive to collect a large number of samples for domain adaptation, as discussed in Section 1. In contrast, it may be easy to obtain plenty of unlabeled data in other domains such as computer vision, where labeling is expensive. 

In communication protocols, preambles are attached to the front of the packets for synchronization, carrier frequency offset correction, and other tasks. The preambles consist of sequences of known symbols (which have a one-to-one mapping to the messages). Therefore, these sequences can be used as the labeled dataset since the receiver obtains the distorted symbol and knows the ground truth. The proposed MDN adaptation and input transformation at the decoder do not incur any modifications to the encoder (transmitter side). The constellation learned by the autoencoder is kept fixed during adaptation. Therefore, using the preambles from a small number of packets, our method performs adaptation at the receiver side and maintains the symbol error rate performance without communicating any information back to the encoder. 

## B.2 TRANSFORMATION BETWEEN MULTIVARIATE GAUSSIANS 

We discuss the feature and parameter transformations between any two multivariate Gaussians. This result was applied to formulate the MDN adaptation in Eqs. (4) and (7). Consider first the standard � transformation from **x** _∼ N_ ( _· |_ _**µ** ,_ **Σ** ) to **x** _∼ N_ ( _· |_ � _**µ** ,_ **Σ**[�] ) given by the two-step process: 

- Apply a whitening transformation **z** = **D** _[−]_[1] _[/]_[2] **U** _[T]_ ( **x** _−_ _**µ**_ ) such that **z** _∼ N_ ( _· |_ **0** _,_ **I** ). 

- Transform **z** into the new Gaussian density using **x** � = **U**[�] **D**[�][1] _[/]_[2] **z** + _**µ**_ �. 

We have denoted the eigen-decomposition of the covariance matrices by **U** � **D** � **U** � _[T]_ , where **U** and **U** � are the orthonormal eigenvector matrices, and **DΣ** and= **UDUD** � are the diagonal _[T]_ and **Σ**[�] = eigenvalue matrices. Combining the two steps, the overall transformation from **x** to � **x** is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0017-08.png)


Suppose we define the matrix **C** = **U**[�] **D**[�][1] _[/]_[2] **D** _[−]_[1] _[/]_[2] **U** _[T]_ , then it is easily verified that the covariance matrices are related by **Σ**[�] = **C Σ C** _[T]_ . In general, the mean vector and covariance matrix of any two Gaussians can be related by the following parameter transformations: 

� � _**µ**_ = **A** _**µ**_ + **b** and **Σ** = **C Σ C** _[T] ,_ (16) 

with parameters **A** _∈_ R _[d][×][d]_ , **b** _∈_ R _[d]_ , and **C** _∈_ R _[d][×][d]_ . Substituting the above parameter transformations into the feature transformation in Eq. (15), we get 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0017-12.png)


� From the above, we can also define the inverse feature transformation from **x** _∼ N_ ( _· |_ � _**µ** ,_ **Σ**[�] ) to **x** _∼ N_ ( _· |_ _**µ** ,_ **Σ** ) : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0017-14.png)


## B.3 GENERATIVE ADAPTATION OF THE MDN 

In § 3.2, we discussed the discriminative adaptation objective for the MDN, which is used when the MDN is adapted as part of the autoencoder in order to improve the end-to-end error rate. This adaptation approach was used for the experiments in § 4. On the other hand, we may be interested in adapting the MDN in isolation with the goal of improving its performance as a generative model of the channel. For this scenario, the adaptation objective Eq. 6 is modified as follows. The first (data-dependent) term is replaced with the negative conditional log-likelihood (CLL) of the target dataset, while the second KL-divergence term remains the same: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0017-17.png)


where _**µ**_ � _i_ ( **z** ) _,_ **Σ**[�] _i_ ( **z** ) and _α_ � _i_ ( **z** ) as a function of _**ψ**_ are given by Eq. (4). The parameters of the original Gaussian mixture _αi_ ( **z** ) _,_ _**µ** i_ ( **z** ) _,_ **Σ** _i_ ( **z** ) _, ∀i_ are constants since they have no dependence on 

17 

Published as a conference paper at ICLR 2023 

_**ψ**_ . The regularization constant _λ ≥_ 0 controls the allowed KL-divergence between the source and target Gaussian mixtures. Small values of _λ_ weight the CLL term more, allowing more exploration in the adaptation, while large values of _λ_ impose a strong regularization to constrain the space of target distributions. We evaluate the performance of this generative MDN adaptation in Appendix C.6. 

## B.4 VALIDATION METRIC FOR AUTOMATICALLY SETTING _λ_ 

The choice of _λ_ in the adaptation objectives Eqs. (6) and 17 is crucial as it sets the right level of regularization suitable for the target domain distribution. Since the target domain dataset is very small, it is difficult to apply cross-validation type of methods to select _λ_ . We propose a validation metric _V_ ( _**ψ**_ ; _D[t]_ ) that utilizes the feature-transformed target domain dataset to evaluate the quality of the adapted solutions for different _λ_ values. 

Let _**ψ**_ denote the adaptation parameters found by minimizing the objective Eq. (6) for a specific _λ ≥_ 0. The feature transformation (from target to source domain) at the decoder **g** _[−]_[1] ( **x** ) based on the adaptation parameters _**ψ**_ is given by Eq. (8). Recall that the target domain dataset is _D[t]_ = _{_ ( **x** _[t] n[, y] n[t][,]_ **[ z]** _n[t]_[)] _[,][n]_[ = 1] _[,][ · · ·][, N][ t][}]_[.][We define the feature-transformed target domain dataset as:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0018-05.png)


Suppose _**ψ**_ is a good adaptation solution, then we expect the decoder (trained on the source domain dataset) to have good classification performance on _D_ trans _[t]_[.][For a given feature-transformed] target domain sample, the decoder predicts the class posterior probabilities: **D** _**θ** d_ ( **g** _[−]_[1] ( **x** _[t] n_[))][=] [ _P_ _**θ** d_ �1 _|_ **g** _[−]_[1] ( **x** _[t] n_[)] � _, · · · , P_ _**θ** d_ � _m |_ **g** _[−]_[1] ( **x** _[t] n_[)] �]. We define the validation metric as the _negative posterior log-likelihood_ of the decoder on _D_ trans _[t]_[, given by] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0018-07.png)


We expect smaller values of _V_ ( _**ψ**_ ; _D[t]_ ) to correspond to better adaptation solutions. The adaptation objective is minimized with _λ_ varied over a range of values, and in each case the adapted solution _**ψ**_ is evaluated using the validation metric. The pair of _λ_ and _**ψ**_ resulting in the smallest validation metric is chosen as the final adapted solution. The search set of _λ_ used in our experiments was _{_ 10 _[−]_[5] _,_ 10 _[−]_[4] _,_ 10 _[−]_[3] _,_ 10 _[−]_[2] _,_ 0 _._ 1 _,_ 1 _,_ 10 _,_ 100 _}_ . See Appendix C.4 for an ablation study on the choice of hyper-parameter _λ_ using this validation metric. 

**Generative MDN Adaptation.** The validation metric proposed above depends on the decoder, and cannot be used when the MDN is adapted as a generative model in isolation (Appendix B.3). For this setting, we modify the validation metric based on the following idea. Suppose the adaptation finds a good solution, then we expect _D_ trans _[t]_[to have a high conditional log-likelihood under the (original)] source domain MDN. The validation metric is therefore given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0018-10.png)


where _P_ _**θ** c_ is the Gaussian mixture given by Eq. 1. 

## B.5 COMPLEXITY ANALYSIS 

We provide an analysis of the computational complexity of the proposed adaptation methods. 

## **MDN Adaptation.** 

The number of free parameters being optimized in the adaptation objective (Eqs. 6 or 17) is given by _|_ _**ψ** |_ = _k_ (2 _d_[2] + _d_ + 2). This is much smaller than the number of parameters in a typical MDN, even considering only the final fully-connected layer (see Table 1 for a comparison). Each step of the BFGS optimization involves computing the objective function, its gradient, and an estimate of its inverse Hessian. The cost of one step of BFGS can thus be expressed as _O_ ( _N[t] k d_[2] _|_ _**ψ** |_[2] ). Suppose BFGS runs for a maximum of _T_ iterations and the optimization is repeated for _L_ values of _λ_ , then the overall cost of adaptation is given by _O_ ( _L T N[t] k d_[2] _|_ _**ψ** |_[2] ). Note that the optimization for different _λ_ values can be easily solved in parallel. 

18 

Published as a conference paper at ICLR 2023 

## **Test-time Adaptation at the Decoder.** 

We analyze the computational cost of the feature transformation-based adaptation at the decoder proposed in § 3.3. Consider a single test input **x** _[t]_ at the decoder. The feature transformation method first computes the posterior distribution _P_ _**θ**_ � _c_ ( **z** _, i |_ **x** _[t]_ ) over the set of symbols-component pairs of size _k m_ . Computation of each exponent factor in the posterior distribution requires _O_ ( _d_[3] ) operations for the full-covariance case, and _O_ ( _d_ ) operations for the diagonal covariance case. This corresponds to calculation of the log of the Gaussian density. Therefore, computation of the posterior distribution for a single ( **z** _, i_ ) pair requires _O_ ( _k m d_[3] ) operations for the full-covariance case (similarly for the diagonal case). Computation of the affine transformation **gz** _[−] i_[1][(] **[x]** _[t]_[)][for][a] single ( **z** _, i_ ) pair requires _O_ ( _d_[2] ) operations (the matrix **C** _i_ only needs to be inverted once prior to test-time adaptation). Since calculation of the posterior term dominates the computation, the overall cost of computing the transformation in Eq (8) over the _k m_ symbol-component pairs will be _O_ ( _k m k m d_[3] ) = _O_ ( _k_[2] _m_[2] _d_[3] ). 

We note that in practical communication systems _d_ is small (typically _d_ = 2). The number of symbols or messages _m_ can vary from 4 to 1024 in powers of 2. The number of mixture components _k_ can be any positive integer, but is usually not more than a few tens to keep the size of the MDN practical. Therefore, the computational cost of test-time adaptation at the decoder based on the feature transformation method is relatively small, making our proposed adaptation very computationally efficient to implement at the receiver side of a communication system. 

## B.6 LIMITATIONS AND FUTURE WORK 

The proposed work focuses mainly on a mixture density network (MDN) as the generative channel model, which allows us to exploit some of their useful properties in our formulation. Generalizing the proposed few-shot domain adaptation to other types of generative channel models such as conditional GANs, VAEs, and normalizing flows (Dinh et al., 2017) could be an interesting direction. These generative models can handle more high-dimensional structured inputs. 

The proposed work does not adapt the encoder network, _i.e._ , the autoencoder constellation is not adapted to changes in the channel distribution. Adapting the encoder, decoder, and channel networks jointly would allow for more flexibility, but would likely be slower and require more data from the target distribution. 

We focused on memoryless channels, where inter-symbol-interference (ISI) is not a problem. In practice, communication channels can have memory and ISI would have to be addressed by the training and adaptation methods. Under changing channels, one would have to also adapt an Equalizer model (algorithm) in order to mitigate ISI. 

## C ADDITIONAL EXPERIMENTS 

We provide additional details on the experiments in § 4 and report additional results, including ablation studies on the proposed method. 

## C.1 EXPERIMENTAL SETUP 

We implemented the mixture density network and communication autoencoder models using TensorFlow (Abadi et al., 2015) and TensorFlow Probability. We used the BFGS optimizer implementation available in TensorFlow Probability. The code base for our work has been submitted as a supplementary material. All the experiments were run on a Macbook Pro with 16 GB memory and 8 CPU cores. Table 3 summarizes the architecture of the encoder, MDN (channel model), and decoder neural networks. Note that the output layer of the MDN is a concatenation (denoted by _⊕_ ) of three fully-connected layers predicting the means, variances, and mixing prior logit parameters of the Gaussian mixture. The following setting is used in all our experiments. The size of the message set _m_ (also the number of classes) was fixed to 16, corresponding to 4 bits. The dimension of the encoding _d_ was set to 2, and the number of mixture components _k_ was set to 5. The size of the hidden layers _nh_ was set to 100. 

19 

Published as a conference paper at ICLR 2023 

Table 3: Architecture of the Encoder, MDN channel, and Decoder neural networks. FC - fully connected (dense) layer; _⊕_ denotes layer concatenation; ELU - exponential linear unit; _m_ - number of messages; _d_ - encoding dimension; _k_ - number of mixture components; _nh_ - size of a hidden layer. 

|Network|Layer|Activation|
|---|---|---|
||FC,_m × nh_|ReLU|
|Encoder|FC,_nh × d_|Linear|
||Normalization (avg. power)|None|
||FC,_d × nh_|ReLU|
|MDN|FC,_nh × nh_|ReLU|
||FC,_nh × kd_ (means)|Linear|
||_⊕_FC,_nh × kd_ (variances)|ELU +1 +_ϵ_|
||_⊕_FC,_nh × k_ (prior logits)|Linear|
|Decoder|FC,_d × nh_<br>FC,_nh × m_|ReLU<br>Softmax|



The parameters _**ψ**_ of the proposed adaptation method are initialized as follows for each component _i_ : 

**A** _i_ = **I** _d,_ **b** _i_ = **0** _,_ **C** _i_ = **I** _d, βi_ = 1 _, γi_ = 0 _,_ 

where **I** _d_ is the _d × d_ identity matrix. This initialization ensures that the target Gaussian mixtures (per class) are always initially equal to the source Gaussian mixtures. The regularization constant _λ_ in the adaptation objective was varied over 8 equally-spaced values on the log-scale with range 10 _[−]_[5] to 100, specifically _{_ 10 _[−]_[5] _,_ 10 _[−]_[4] _,_ 10 _[−]_[3] _,_ 10 _[−]_[2] _,_ 0 _._ 1 _,_ 1 _,_ 10 _,_ 100 _}_ . The _λ_ value and _**ψ**_ corresponding to the smallest validation metric are selected as the final solution. 

We used the Adam optimizer (Kingma & Ba, 2015) with a fixed learning rate of 0 _._ 001, batch size of 128, and 100 epochs for training the MDN. For adaptation of the MDN using the baseline methods _Finetune_ and _Finetune last_ , we used Adam with the same learning rate for 200 epochs. The batch size is set as _b_ = max _{_ 10 _,_ 0 _._ 1 _N[t] }_ , where _N[t]_ is number of adaptation samples in the target dataset. For training the autoencoder using Algorithm 1, we found that stochastic gradient descent (SGD) with Nesterov momentum (constant 0 _._ 9), and an exponential learning rate schedule between 0 _._ 1 and 0 _._ 005 works better than Adam. 

**Finetuning Baselines.** We provide additional details on the baselines _Finetune_ and _Finetune last_ . Both the methods first initialize the target domain MDN, encoder, and decoder networks with the corresponding parameters from the source domain. The method _Finetune_ first finetunes all the MDN parameters to minimize the conditional log-likelihood of the target dataset using the Adam optimizer. After the MDN is finetuned, we freeze the parameters of the MDN and encoder, and train only the decoder using data generated from the updated MDN channel. The method _Finetune last_ differs from _Finetune_ in that it optimizes only the weights of the final MDN layer. 

From the results in Figures 4, 5, and 6, we observe that the baselines _Finetune_ and _Finetune last_ have very similar performance compared to the case of no adaptation. We have investigated this carefully and verified that this is not due to a bug or insufficient optimization ( _e.g._ , by checking if the final weights of the MDN and decoder are different for both methods). For both methods, we tried a range of learning rates for Adam and increased the number of epochs to a large number (beyond 200 was not helpful). We have reported the best-case results for these methods, which suggests that they are not effective at adaptation using small target domain datasets. As mentioned in Section 4.1, we hypothesize that using the KL-divergence based regularization and constraining the number of adaptation parameters leads to more effective performance of our method. 

**Uncertainty Estimation.** Since there is inherent randomness in our experiments, especially with the small sample sizes of the target dataset, we always report average results from multiple trials. For the experiments on standard simulated channel variations ( _e.g._ , AWGN to Ricean fading), we report the results from 10 trials. For the random Gaussian mixtures experiment, we report the average and standard error over 50 random source/target dataset pairs. For the FPGA experiments, we report the results from 20 random trials. The average metrics (symbol error rate and log-likelihood) are reported in the plots. 

20 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0021-01.png)


Figure 7: Example data plot for the case where the source and target domain class-conditional distributions are random Gaussian mixtures. The training data is from the source distribution, while the adaptation data (with 10 samples per class) and test data are from the target distribution. 

**Evaluation Protocol.** We create a random class-stratified 50-50 train-test split (each of size 300,000) for data from both the source and target domains. Performance on both domains is always evaluated on the held-out test split. The train split from the target domain dataset is sub-sampled to create adaptation datasets of different sizes, specifically with 5, 10, 20, 30, 40, and 50 samples per class (symbol). For the generative adaptation experiments on the MDN (Appendix C.6), the number of adaptation samples from the target domain is reduced even further. We varied it from 2 samples perclass to 20 samples per-class in order to highlight the improvements obtained by the proposed method. The oracle baseline method, which retrains the autoencoder and MDN on the target distribution, uses the entire training dataset from the target domain. 

**Choice of SNR.** For the experiments on simulated channel distributions such as AWGN, Ricean fading, and Uniform fading, we set the signal-to-noise ratio (SNR) to 14 dB for the source distribution and 20 dB for the target distribution. The connection between the SNR and the distribution parameters is given in Appendix F. We have experimented with other combinations of SNR for the source and target channels and found a similar trend in the adaptation performance. 

In the simulated experiments, we focused on the SNR range of 14 dB to 20 dB. Our process for selecting this SNR range was by first evaluating the symbol error rate (SER) vs. SNR curve of the autoencoder for the different simulated channel distributions. We found that going below 14 dB SNR results in a degradation of the autoencoder’s performance (except for the AWGN channel, which we do not use as a target distribution). Also, going above 20 dB SNR did not lead to a significant decrease in the SER. For the channels such as Ricean fading and Uniform fading, we found that even a retrained autoencoder has a relatively high error rate for lower SNRs. 

## C.2 DETAILS ON THE FPGA EXPERIMENT 

Referring to the experiment in § 4.2, for the real and over-the-air traces we used the platform from Lacruz et al. (2021). This ultra-wide-band mm-wave transceiver baseband memory-based design is developed on top of an ZCU111 RFSoC FPGA. This evaluation board features a Zynq Ultrascale + ZCU28DR. This FPGA is equipped with 8 _×_ 8 AD/DA converters with Giga-sampling capabilities, which make it ideal for RF system development; the 4 GB DDR4 memories contain RF-ADCs with up to 4 GSPS of sampling rate, and RF-DACs with up to 6.544 GSPS. This board also includes a quad-core ARM Cortex-A53 and a dual-core ARM Cortex-R5 real-time processor. 

For the radio frequency, we used 60 GHz RF front-end antennas. These kits include a 16 + 16 TRX patch array antenna plus the RF module with up/down conversion from baseband to I/Q channels, 

21 

Published as a conference paper at ICLR 2023 

and TX/RX local oscillator (LO) frequency control. The antennas use 57 _−_ 71 GHz, a range of frequencies that cover the unlicensed 60 GHz band for mm-wave channels, and are managed from a PC Host via USB. 

We implemented a hardware on the loop training. For the experimentation on real traces, we use Matlab as a central axis. The PC host running Matlab is connected to the platform via Ethernet. The FPGA can transmit different custom waveforms like 16-QAM frames from the 802.11ad and 802.11ay standards, with 2 GHz of bandwidth. The frames are sent over-the-air via 60 GHz radio frequency kits, and the samples are stored at the FPGA DDR memory. We decode the received data from the transmission, removing the preamble and header fields and extracting the symbols to train the MDN. We add a preamble to the generated constellation from the MDN for packet detection purposes, and we transmit again the new waveforms over-the-air. Finally, the adaptation is performed offline with the decoded symbols from the custom autoencoder-learned constellation. 

## **Source and Target Domains.** 

For the experiment in § 4.2, we introduced distribution changes via IQ imbalance-based distortions to the symbol constellation, and evaluated the adaptation performance as a function of the level of imbalance. The source domain would be the original channel, the over-the-air link between the transmitter and receiver on which the training data is collected. This source domain data is used for training the MDN and the autoencoder. The target domain would be a modification of the source domain where the symbols used by the transmitter are distorted by modifying the in-phase and quadrature-phase (IQ) components of the RF signal. This causes a change in the distribution observed by the receiver (decoder), leading to a drop in performance without any adaptation. 

## C.3 DETAILS ON THE RANDOM GAUSSIAN MIXTURE DATASETS 

We created a simulated distribution shift setting where data from both the source and target domains are generated from class-conditional Gaussian mixtures whose parameters are modified between the two domains ( _e.g._ , see Fig. 7). The parameters for the source and target Gaussian mixtures are generated as follows: 

**Source Domain.** The source domain data is generated with a standard 16-QAM constellation _Z_ QAM, which has 16 classes (messages). Let _ks_ be the number of components in the source Gaussian mixture. 

For each **z** _∈Z_ QAM: 

- Calculate _d_ min, the minimum distance from **z** to the remaining symbols in _Z_ QAM. Let _σs_ = _d_ min _/_ 4 be a constant standard deviation for this symbol. 

- Component priors: generate _πi_ ( **z** ) _∼_ Unif(0 _._ 05 _,_ 0 _._ 95) _, ∀i ∈_ [ _ks_ ]. Normalize the priors to sum to 1. 

- Component means: generate _**µ** i_ ( **z** ) _∼ N_ ( _· |_ **z** _, σs_[[2]] 

   - _s_[[2]] **[I]**[)] _[,][∀][i][ ∈]_[[] _[k][s]_[]][.] 

- iid 

- • Component covariances: generate _s_ 1 _, · · · , sd ∼_ Unif(0 _._ 2 _σs, σs_ ) and let **Σ** _i_ ( **z** ) = diag( _s_[2] 1 _[,][ · · ·][, s]_[2] _d_[)] _[,][∀][i][ ∈]_[[] _[k][s]_[]][ (the covariances are diagonal).] 

- Generate _N[s] / m_ samples corresponding to symbol **z** from the Gaussian mixture: **x** _[s] n[∼]_ � _ki_ =1 _s[π][i]_[(] **[z]**[)] _[ N]_[(] **[x]** _[ |]_ _**[ µ]**[i]_[(] **[z]**[)] _[,]_ **[ Σ]** _[i]_[(] **[z]**[))][.] 

**Target Domain.** The parameters of the target Gaussian mixture are generated in a very similar way. The MDN and autoencoder are trained on the source domain dataset. Let _Z_ = _{_ **E** _**θ** e_ ( **1** 1) _, · · · ,_ **E** _**θ** e_ ( **1** _m_ ) _}_ be the constellation learned by the autoencoder. Let _kt_ be the number of components in the target Gaussian mixture. 

For each **z** _∈Z_ : 

- Calculate _d_ min, the minimum distance from **z** to the remaining symbols in _Z_ . Let _σt_ = _d_ min _/_ 4 be a constant standard deviation for this symbol. 

- Component priors: generate _π_ � _i_ ( **z** ) _∼_ Unif(0 _._ 05 _,_ 0 _._ 95) _, ∀i ∈_ [ _kt_ ]. Normalize the priors to sum to 1. 

- Component means: generate _**µ**_ � _i_ ( **z** ) _∼ N_ ( _· |_ **z** _, σt_[2] **[I]**[)] _[,][∀][i][ ∈]_[[] _[k][t]_[]][.] 

22 

Published as a conference paper at ICLR 2023 

- iid � 

- • Component covariances: generate _s_ 1 _, · · · , sd ∼_ Unif(0 _._ 2 _σt, σt_ ) and let **Σ** _i_ ( **z** ) = diag( _s_[2] 1 _[,][ · · ·][, s]_[2] _d_[)] _[,][∀][i][ ∈]_[[] _[k][t]_[]][ (the covariances are diagonal).] 

- Generate _N[t] / m_ samples corresponding to symbol **z** from the Gaussian mixture: **x** _[t] n[∼]_ � _ki_ =1 _t[π]_[�] _[i]_[(] **[z]**[)] _[ N]_[(] **[x]** _[ |]_[ �] _**[µ]**[i]_[(] **[z]**[)] _[,]_ **[Σ]**[�] _[i]_[(] **[z]**[))][.] 

We set _ks_ = _kt_ = 3, except for the experiment where the source and target Gaussian mixtures are mismatched. In this case, _ks_ and _kt_ are randomly selected for each dataset from the range _{_ 3 _,_ 4 _,_ 5 _,_ 6 _}_ . 

**Random Phase Shift.** We allow the channel output **x** to be randomly phase shifted on top of other distribution changes. This is done by matrix multiplication of **x** with a rotation matrix, where the rotation angle for each sample is uniformly selected from [ _−ϕ, ϕ_ ]. We set _ϕ_ to _π/_ 18 or 10 degrees. Results on a dataset with random phase shift applied on top of random Gaussian mixture distribution shift can be found in Fig. 5c. 

## C.4 ABLATION EXPERIMENTS 

We perform ablation experiments to understand: **1)** the choice of the hyper-parameter _λ_ , **2)** the importance of the KL-divergence regularization in the adaptation objective, **3)** performance of our method when the source and target Gaussian mixtures have mismatched components, and **4)** the performance of our method when there is no distribution change. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0023-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0023-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0023-09.png)


**----- Start of picture text -----**<br>
(a) AWGN to Uniform fading. (b) AWGN to Ricean fading.<br>(c) Ricean fading to Uniform Fading. (d) Random Gaussian Mixtures.<br>**----- End of picture text -----**<br>


Figure 8: Evaluation of the validation metric for selection of the hyper-parameter _λ_ under different distribution shifts. The setting _λ_ = auto corresponds to automatic choice using the validation metric. 

**Automatic Selection of Hyper-parameter** _λ_ **.** We evaluate the proposed validation metric for automatically selecting the hyper-parameter _λ_ and report the results in Fig. 8. We run the proposed method for different fixed values of _λ_ as well as the automatically-selected _λ_ , and compare their 

23 

Published as a conference paper at ICLR 2023 

performance on the target domain test set. We consider both simulated channel variations and the random Gaussian mixture datasets. From the figure, we observe that in most cases performance based on the automatically set value of _λ_ is better than other fixed choices of _λ_ . The case of adaptation from AWGN to Ricean fading is an exception, where our method does not learn a good adaptation solution (see Fig. 4c). In this case, we observe from Fig. 8b that the setting _λ_ = 0 _._ 0001 has the best symbol error rate. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0024-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0024-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0024-04.png)


**----- Start of picture text -----**<br>
(a) AWGN to Uniform fading. (b) AWGN to Ricean fading.<br>(c) Ricean fading to Uniform Fading. (d) Random Gaussian Mixtures.<br>**----- End of picture text -----**<br>


Figure 9: Evaluating the importance of the KL-divergence term in the adaptation objective. The setting _λ_ = auto corresponds to automatic choice using the validation metric. 

**Performance Under Component Mismatch.** We evaluate the symbol error rate performance of all the methods in the setting where the number of components in the source and target Gaussian mixtures is mismatched. The number of components in the source and target Gaussian mixtures is randomly selected from the range 3 to 6. From Fig. 11, we observe that the proposed method has strong performance improvements even in this mismatched setting, suggesting that our method can perform well even when Assumptions A1 and A2 are not satisfied. 

**Importance of the KL-divergence Regularization.** Recall that the adaptation objectives Eqs. (6) and (17) include the KL-divergence term scaled by _λ_ in order to avoid large distribution changes when there is not enough support from the small target-domain dataset. A natural question to ask is whether this term is useful and helps improve the adaptation solution when _λ >_ 0. To answer this, we compare the performance of our method with _λ_ = 0 with that our our method with _λ_ set automatically using the validation metric. The results of this comparison are given in Fig. 9 on four simulated channel variations. The results are averaged over multiple trials as before. It is clear that setting _λ_ = 0 for our method leads to much higher symbol error rates compared to setting _λ_ to a non-zero value using the validation metric, establishing the importance of the KL-divergence term. 

24 

Published as a conference paper at ICLR 2023 

**Performance Under No Distribution Change.** We evaluate the symbol error rate performance of all the methods in the setting where there is no distribution change. In this setting, the performance of the MDN and autoencoder should not change, and we expect the proposed adaptation method to maintain a similar performance (not lead to increased symbol error rate). In Fig. 10, we report the results of this experiment when both the source and target channel distributions are either Ricean fading or Uniform fading. We consider a medium SNR value of 14 dB and a high SNR value of 20 dB. We observe that our method is relatively stable even when there is no distribution change, and there is only a small increase in error rate. For instance, in Fig. 10c, the error rate of our method increases from 0.015 to 0.018 for 5 samples per class. 

We expect that a practical system that frequently adapts to changes in the channel distribution should first have a distribution change-detection algorithm that takes a batch of new samples from the channel and detects whether there is any change in the distribution. The actual domain adaptation algorithm is then applied only when a distribution change is detected. In this way, any potential drop in the autoencoder’s performance when there is no distribution change can be made less likely. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0025-03.png)


**----- Start of picture text -----**<br>
No adapt Finetune Retrained autoenc No adapt Finetune Retrained autoenc<br>Affine adapt (ours) Finetune last Affine adapt (ours) Finetune last<br>0.150<br>0.27<br>0.145<br>0.26<br>0.140<br>0.25<br>0.135<br>0.24<br>0.130<br>0.23<br>0.125<br>0.22<br>0.120<br>10 20 30 40 50 10 20 30 40 50<br>#target samples per-class #target samples per-class<br>(a) Uniform Fading, SNR = 14 dB. (b) Uniform Fading, SNR = 20 dB.<br>No adapt Finetune Retrained autoenc No adapt Finetune Retrained autoenc<br>Affine adapt (ours) Finetune last Affine adapt (ours) Finetune last<br>1e 6<br>5<br>0.019<br>4<br>0.018<br>0.017 3<br>0.016<br>2<br>0.015<br>1<br>0.014<br>10 20 30 40 50 10 20 30 40 50<br>#target samples per-class #target samples per-class<br>(c) Ricean Fading, SNR = 14 dB. (d) Ricean Fading, SNR = 20 dB.<br>Symbol error rate Symbol error rate<br>Symbol error rate Symbol error rate<br>**----- End of picture text -----**<br>


Figure 10: Performance under no distribution change, _i.e._ , the source and target distributions are the same. The type of channel distribution and the corresponding SNR common to both the source and target domain is indicated. In Fig. (d) note that the y-axis is scaled by 10[6] . 

## C.5 ANALYSIS OF THE FAILURE ON AWGN TO RICEAN FADING 

Referring to Fig. 4. c in the main paper, we observe that our method has a worse symbol error rate compared to no adaptation and the other baselines for the adaptation setting from an AWGN channel at 14dB SNR to a Ricean fading channel at 20dB SNR. In order to get an intuition for this, we look into the data distribution plots from the source and target channels (see Fig. 12). From the figure, we observe that the target distribution is actually very class-separable, but the distribution shift between 

25 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0026-01.png)


Figure 11: Performance under distribution shift where the source and target distributions are random Gaussian mixtures with mismatched components. 

the AWGN and Ricean fading channels is quite large, particularly because of the high SNR of the Ricean fading channel. In this case, the classifier from the source domain is able to classify the target domain data reasonably well without any adaptation. The fine-tuning baselines (“Finetune” and “Finetune last”) do not lead to much improvement and have similar SER as the case of no adaptation. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0026-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0026-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0026-06.png)


**----- Start of picture text -----**<br>
(a) AWGN data at 14dB SNR. (b) Ricean fading data at 20dB SNR.<br>**----- End of picture text -----**<br>


Figure 12: Data plots of the source domain (AWGN at 14dB SNR) and the target domain (Ricean fading at 20dB SNR). The black points are the channel inputs ( _i.e._ , the symbols). A 16-QAM constellation is used in the source domain (left), while the learned autoencoder constellation is used in the target domain (right). 

On the other hand, the proposed method leads to worse SER on the target domain after adaptation. We hypothesize that it fails in this case because the set of affine transformation are unable to capture the large distribution change well. In our current formulation, the affine transformations per component are shared by all the classes. From Eq. 4, the affine transformation parameters are **A** _i,_ **b** _i,_ **C** _i, βi,_ and _γi, i ∈_ [ _k_ ]. Based on the data plots, we think that allowing the affine transformations to be class-specific can handle this adaptation setting better. While allowing the affine transformations to be class-specific provides more flexibility, it also introduces more parameters which need to be optimized using a small dataset from the target distribution. 

## C.6 EXPERIMENTS ON GENERATIVE ADAPTATION OF THE MDN 

We evaluate the generative adaptation method of the MDN discussed in Appendix B.3, where the goal is to achieve a high conditional log-likelihood (CLL) under the target data distribution. As shown in Fig. 13, we consider the standard simulated distribution changes including random Gaussian mixtures, and vary the number of target samples per-class from 2 to 20. For each pair of source-target distributions and each target sample size, the experiment is repeated for 50 trials, and the average CLL 

26 

Published as a conference paper at ICLR 2023 

is reported. The CLL is calculated on a held-out test dataset of size 25,000 from the target distribution. For the simulated channels, the SNR of the source and target distributions are each randomly chosen from the range 10 dB to 20 dB. From Fig. 13, we observe that the proposed MDN adaptation has significantly higher CLL compared to no adaptation and the finetuning baseline methods, especially for very small number of target samples per class. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0027-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0027-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0027-04.png)


**----- Start of picture text -----**<br>
(a) AWGN to Uniform fading. (b) Ricean fading to Uniform fading.<br>(c) AWGN to Ricean fading. (d) Random Gaussian mixtures.<br>**----- End of picture text -----**<br>


Figure 13: Generative adaptation of the MDN channel to different types of distribution shifts. Larger values of conditional log-likelihood correspond to better adaptation solutions. 

## C.7 EVALUATION WITH VERY FEW TARGET DOMAIN SAMPLES 

To evaluate how the proposed method performs with fewer than 10 samples per class, we ran an experiment on some simulated distribution changes with 1 _,_ 2 _,_ 4 _,_ 6 _,_ and 8 samples per class. We report these results in Fig. 14 for the following distribution changes: 1) Random Gaussian mixtures (Fig. 14a), and 2) Ricean fading to Uniform fading (Fig. 14b). We follow the same protocol as in the main paper and report the average results from multiple random trials. From Fig. 14a, corresponding to random Gaussian mixtures, we observe that the proposed method is able to improve performance (decrease SER) starting from 1 sample per class. 

However, in Fig. 14a (Ricean fading to Uniform fading), the proposed method increases the SER (worse) compared to no adaptation for 1 and 2 samples per class. The SER starts to decrease significantly from 4 samples per class. We hypothesize that the reason for failure of the proposed method with 1 and 2 samples per class could be the strong distribution change and complexity of the target uniform fading channel. While this may seem concerning, please note that we chose these strong simulated distribution changes in order to demonstrate the potential improvements of our method. In practical wireless channels, the distribution changes are likely to be more gradual, so that the proposed method can usually adapt well with only a few samples per class. 

27 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0028-01.png)


**----- Start of picture text -----**<br>
No adapt Finetune Retrained autoenc No adapt Finetune Retrained autoenc<br>Affine adapt (ours) Finetune last Affine adapt (ours) Finetune last<br>0.14<br>0.12 0.30<br>0.10<br>0.25<br>0.08<br>0.06 0.20<br>0.04<br>0.15<br>0.02<br>1 5 10 20 30 40 50 1 5 10 20 30 40 50<br>#target samples per-class #target samples per-class<br>(a) Random Gaussian mixtures. (b) Ricean fading (14dB) to Uniform fading (20dB).<br>Symbol error rate Symbol error rate<br>**----- End of picture text -----**<br>


Figure 14: Results of autoencoder adaptation on simulated distribution changes with very few target domain samples (starting from 1 sample per class). For the case where both the source and target domain data are from random Gaussian mixtures, our method is able to improve the SER even with 1 sample per class. In the case of adaptation from Ricean fading to Uniform fading, our method increases the SER (compared to no adaptation) for 1 and 2 samples per class, but starts to decrease the SER after that. 

## C.8 GAP BETWEEN THE PROPOSED METHOD AND THE RETRAINED AUTOENCODER 

In this section, we address the following question: Does the performance of the proposed method approach that of the oracle retrained method as the size of the adaptation dataset increases? We note that the oracle method “Retrained autoenc” retrains the channel model, encoder and decoder networks jointly using Algorithm 1 in Appendix D.3. In our experiments, we used 20 _,_ 000 samples from the target (domain) distribution for training the MDN channel in between each epoch of the encoder/decoder update, and used 300 _,_ 000 samples for training the encoder/decoder networks with the MDN parameters fixed. The retrained autoencoder therefore learns a custom _optimal_ constellation (encoding) for the target distribution. 

Table 4: Performance of the proposed adaptation method with a large number of target domain samples, and comparison with an oracle retrained autoencoder for the target domain. 

|Source and target domains|Proposed method<br>(50 samples/class)|Proposed method<br>(1000 samples/class)|Retrained autoencoder|
|---|---|---|---|
|Random Gaussian mixtures|0.0766|0.0695|0.0154|
|AWGN to Uniform fading|0.2370|0.2138|0.1345|
|Ricean fading to Uniform fading|0.1945|0.1815|0.1345|
|AWGN to Ricean fading|6.381e-4|2.531e-5|3.125e-6|
|Uniform fading to Ricean fading|6.484e-4|5.503e-4|3.125e-6|



Our adaptation method, on the other hand, does not modify the constellation learned by the autoencoder from the source domain data, which could be sub-optimal for the target domain. This restriction of not modifying the constellation is practically advantageous since there is no need to communicate the new symbols back to the transmitter. Also, there is no need to change the transmitter side encoding frequently. Since the proposed method does not have the full flexibility of the retrained autoencoder, we do not expect its performance (SER) to converge to that of the latter under significant distribution change. For instance, Fig. 12b shows the learned autoencoder constellation for a Ricean fading channel of 20 dB SNR. This is quite different from the optimal constellation for an AWGN channel of 14 dB SNR (which is closer to a 16-QAM constellation). 

To understand this better, we ran some experiments to evaluate our method with a large (target) adaptation dataset, and compared it to the oracle retrained autoencoder. In Table 4, we report the SER of our method with 1000 samples per class (so 16 _,_ 000 samples overall) on a subset of simulated 

28 

Published as a conference paper at ICLR 2023 

distribution changes. We follow the same protocol as the main paper, and report the average SER from multiple random trials on a large test set from the target distribution. 

We also report the performance of our method with 50 samples/class for comparison. From the table, we observe that there is a gap in the SER of our method (with 1000 samples/class) and that of the retrained autoencoder. We believe that allowing the encoder symbols to be optimized might be required in order to bridge this gap in SER. In conclusion, when it is possible to collect sufficient (large) data from the target distribution, it might be better to allow the encoder network to also be optimized via retraining. 

## D DETAILED BACKGROUND 

Expanding on § 2, we provide additional background on the following topics: 1) components of an end-to-end autoencoder-based communication system, 2) generative modeling using mixture density networks, 3) training algorithm of the autoencoder, and 4) a primer on domain adaptation. 

## D.1 AUTOENCODER-BASED END-TO-END LEARNING 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0029-06.png)


**----- Start of picture text -----**<br>
Input<br>message  Wireless channel  Predicted message<br>Generative<br>Encoder / Transmitter channel model Decoder / Receiver<br>neural network neural network<br>One-hot encoding Multiple dense layers Normalization layer Multiple dense layers Dense layer  + softmax activation<br>**----- End of picture text -----**<br>


Figure 15: An end-to-end autoencoder-based communication system with a generative channel model. 

Consider a single-input, single-output wireless communication system as shown in Fig. 1, where the transmitter encodes and transmits messages _y_ from the set _Y_ = _{_ 1 _,_ 2 _, · · · , m}_ to the receiver through _d ≥_ 2 discrete uses of the wireless channel. The receiver attempts to accurately decode the transmitted message from the distorted and noisy channel output **x** . We discuss the end-to-end learning of such a system using the concept of autoencoders (O’Shea & Hoydis, 2017; Dörner et al., 2018). 

**Transmitter / Encoder Neural Network.** The transmitter or encoder part of the autoencoder is modeled as a multi-layer, feed-forward neural network (NN) that takes as input the one-hot-coded representation **1** _y_ of a message _y ∈Y_ , and produces an encoded symbol vector **z** = **E** _**θ** e_ ( **1** _y_ ) _∈_ R _[d]_ . Here, _**θ** e_ are the parameters (weights and biases) of the encoder NN and _d_ is the encoding dimension. Due to hardware constraints present at the transmitter, a normalization layer is used as the final layer of the encoder network in order to constrain the average power and/or the amplitude of the symbol vectors. The average power constraint is defined as E[ _∥_ **z** _∥_ 2[2][] =][ E] _[y]_[[] _[∥]_ **[E]** _**[θ]** e_[(] **[1]** _[y]_[)] _[∥]_ 2[2][]] _[ ≤][c]_[, where] the expectation is over the prior distribution of the input messages, and _c_ is typically set to 1. The amplitude constraint is defined as _|zi| ≤_ 1 _, ∀i ∈_ [ _d_ ]. The size of the message set is usually chosen to be a power of 2, _i.e._ , _m_ = 2 _[b]_ representing _b_ bits of information. Following O’Shea & Hoydis (2017), the communication rate of this system is the number of bits transmitted per channel use, which in this case is _R_ = _b / d_ . An autoencoder transmitting _b_ bits over _d_ uses of the channel is referred to as a ( _d, b_ ) autoencoder. For example, a (2 _,_ 4) autoencoder uses a message set of size 16 and an encoding dimension of 2, with a communication rate _R_ = 2 bits/channel use. 

**Receiver / Decoder Neural Network.** The receiver or decoder component is also a multilayer, feedforward NN that takes the channel output **x** _∈_ R _[d]_ as its input and outputs a probability distribution over the _m_ messages. The input-output mapping of the decoder NN can be expressed as **D** _**θ** d_ ( **x** ) := [ _P_ _**θ** d_ (1 _|_ **x** ) _, · · · , P_ _**θ** d_ ( _m |_ **x** )] _[T]_ , where _**θ** d_ are the parameters of the decoder NN. The softmax activation function is used at the final layer to ensure that the outputs are valid probabilities. The message corresponding to the highest output probability is predicted as the decoded message, _i.e._ , � _y_ ( **x** ) = argmax _y∈Y P_ _**θ** d_ ( _y |_ **x** ). The decoder NN is essentially a _discriminative classifier_ that 

29 

Published as a conference paper at ICLR 2023 

learns to accurately categorize the received (distorted) symbol vector into one of the _m_ message classes. This is in contrast to conventional autoencoders, where the decoder learns to accurately reconstruct a high-dimensional tensor input from its low-dimensional representation learned by the encoder. In the case of communication autoencoders, the _symbol or block error rate (BLER)_ , defined � as E( **x** _,y_ )[ 1 ( _y_ ( **x** ) _̸_ = _y_ )], is used as the end-to-end performance metric. 

**Channel Model.** As discussed in § 1, the wireless channel can be represented by a _conditional probability density_ of the channel output given its input _p_ ( **x** _|_ **z** ). The channel can be equivalently characterized by a _stochastic transfer function_ **x** = **h** ( **z** _,_ **u** ) that transforms the encoded symbol vector into the channel output, where **u** captures the stochastic components of the channel ( _e.g._ , random noise, phase offsets). For example, an additive white Gaussian noise (AWGN) channel is represented by **x** = **h** ( **z** _,_ **u** ) = **z** + **u** , with **u** _∼N_ ( _· |_ **0** _, σ_[2] **I** _d_ ) and _p_ ( **x** _|_ **z** ) = _N_ ( **x** _|_ **z** _, σ_[2] **I** _d_ ). For realistic wireless channels, the transfer function and conditional probability density are usually unknown and hard to approximate well with standard mathematical models. Recently, a number of works have applied generative models such as conditional generative adversarial networks (GANs) (O’Shea et al., 2019; Ye et al., 2018), MDNs (García Martí et al., 2020), and conditional variational autoencoders (VAEs) (Xia et al., 2020) for modeling the wireless channel. To model a wireless channel, generative methods learn a parametric model _P_ _**θ** c_ ( **x** _|_ **z** ) (possibly a neural network) that closely approximates the true conditional density of the channel from a dataset of channel input/output observations. Learning a generative model of the channel comes with _important advantages_ . 1) Once the parameters of the channel model are learned from data, the model can be used to generate _any number of representative samples_ from the channel distribution. 2) A channel model with a differentiable transfer function makes it possible to backpropagate gradients of the autoencoder loss through the channel and train the autoencoder using stochastic gradient descent (SGD)-based optimization. 3) It allows for continuous adaptation of the generative channel model to variations in the channel conditions, and thereby maintain a low error rate of the autoencoder. 

## D.2 GENERATIVE CHANNEL MODEL USING A MIXTURE DENSITY NETWORK 

As discussed in § 2, we use an MDN (Bishop, 1994; 2007) with Gaussian components to model the conditional density of the channel. MDNs can model complex conditional densities by combining a (feed-forward) neural network with a standard parametric mixture model ( _e.g._ , mixture of Gaussians). The MDN learns to predict the parameters of the mixture model _**ϕ**_ ( **z** ) as a function of the channel input **z** . This can be expressed as _**ϕ**_ ( **z** ) = **M** _**θ** c_ ( **z** ) , where _**θ** c_ are the parameters of the neural network. The parameters of the mixture model defined by the MDN are a concatenation of the parameters from the _k_ density components, _i.e._ , _**ϕ**_ ( **z** ) _[T]_ = [ _**ϕ**_ 1( **z** ) _[T] , · · · ,_ _**ϕ** k_ ( **z** ) _[T]_ ], where _**ϕ** i_ ( **z** ) is the parameter vector of component _i_ . Focusing on a Gaussian mixture, the _channel conditional density_ given each symbol **z** _∈Z_ is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0030-05.png)


where _**µ** i_ ( **z** ) _∈_ R _[d]_ is the mean vector, **Σ** _i_ ( **z** ) _∈_ R _[d][×][d]_ is the (symmetric, positive-definite) covariance matrix, and _πi_ ( **z** ) _∈_ [0 _,_ 1] is the weight (prior probability) of component _i_ . Also, _K_ is the latent random variable denoting the mixture component of origin. The weights of the mixture are parameterized using the softmax function as _πi_ ( **z** ) = _e[α][i]_[(] **[z]**[)] _/_[�] _[k] j_ =1 _[e][α][j]_[(] **[z]**[)] _[,][∀][i]_[in order to satisfy the probability] constraint. The MDN simply predicts the un-normalized weights _αi_ ( **z** ) _∈_ R (also known as the _prior logits_ ). We define the parameter vector of component _i_ as _**ϕ** i_ ( **z** ) _[T]_ = [ _αi_ ( **z** ) _,_ _**µ** i_ ( **z** ) _[T] ,_ vec( **Σ** _i_ ( **z** )) _[T]_ ], where vec( _·_ ) is the vector representation of the unique entries of the covariance matrix. Details on the _conditional log-likelihood (CLL)_ training objective and the _transfer function_ of the MDN can be found in Appendix E. 

## D.3 TRAINING OF THE AUTOENCODER 

In this section, we provide a formal discussion of the end-to-end training of the autoencoder. First, we define the input-output mapping of the autoencoder as **f** _**θ**_ ( **1** _y_ ) = **D** _**θ** d_ ( **h** _**θ** c_ ( **E** _**θ** e_ ( **1** _y_ ) _,_ **u** )) = ( **D** _**θ** d ◦_ **h** _**θ** c_ ( _·,_ **u** ) _◦_ **E** _**θ** e_ )( **1** _y_ ), where _**θ**[T]_ = [ _**θ** e[T][,]_ _**[ θ]** c[T][,]_ _**[ θ]** d[T]_[]][is the combined vector of parameters from] the encoder, channel, and decoder. Given an input message _y ∈Y_ , the autoencoder maps the one-hotcoded representation of _y_ into an output probability vector over the message set. Note that, while the 

30 

Published as a conference paper at ICLR 2023 

encoder and decoder neural networks are deterministic, a forward pass through the autoencoder is stochastic due to the channel transfer function **h** _**θ** c_ . The learning objective of the autoencoder is to accurately recover the input message at the decoder with a high probability. The cross-entropy (CE) loss, which is commonly used for training classifiers, is also suitable for end-to-end training of the autoencoder. For an input _y_ with encoded representation **z** = **E** _**θ** e_ ( **1** _y_ ), channel output **x** = **h** _**θ** c_ ( **z** _,_ **u** ), and decoded output **D** _**θ** d_ ( **x** ) = [ _P_ _**θ** d_ (1 _|_ **x** ) _, · · · , P_ _**θ** d_ ( _m |_ **x** )] _[T]_ , the CE loss is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0031-02.png)


which is always non-negative and takes the minimum value 0 when the correct message is decoded with probability 1. The autoencoder aims to minimize the following expected CE loss over the input message set and the channel output: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0031-04.png)


Here _{p_ ( _y_ ) _, ∀y ∈Y}_ is the prior probability over the input messages, which is usually taken to be uniform in the absence of prior knowledge. In practice, the autoencoder minimizes an empirical estimate of the expected CE loss function by generating a large set of samples from the channel conditional density given each message. Let _X[y]_ = _{_ **x** _[y] n_[=] **[h]** _**[θ]** c_[(] **[E]** _**[θ]** e_[(] **[1]** _[y]_[)] _[,]_ **[ u]** _[n]_[)] _[,][n]_[=][1] _[,][ · · ·][, N][}]_ denote a set of independent and identically distributed (iid) samples from _P_ _**θ** c_ ( **x** _|_ **E** _**θ** e_ ( **1** _y_ )), the channel conditional density given message _y_ . Also, let _X_ = _∪yX[y]_ denote the combined set of samples from all messages. The empirical expectation of the autoencoder CE loss (22) is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0031-06.png)


It is clear from the above equation that the channel transfer function **h** _**θ** c_ should be differentiable in order to be able to backpropagate gradients through the channel to the encoder network. The transfer function defining sample generation for a Gaussian MDN channel is discussed in Appendix E. 

**Algorithm 1** End-to-end training of the autoencoder with an MDN channel 

- 1: **Inputs:** Message size _m_ ; Encoding dimension _d_ ; Initial constellation _{_ **E** 0( **1** _y_ ) _, ∀y ∈Y}_ ; Number of optimization epochs for the autoencoder _Nae_ and channel _Nce_ . 

- 2: **Output:** Trained network parameters _**θ** e,_ _**θ** c,_ _**θ** d_ . 

- 3: Initialize the encoder, channel, and decoder network parameters. 

- 4: Sample training data _Dc_[(0)] from the channel using the initial constellation. 

- 5: Train the channel model for _Nce_ epochs to minimize _L_ ch( _**θ** c_ ; _Dc_[(0)][)][ (Eq.][24).] 

- 6: **for** epoch _t_ = 1 _, · · · , Nae_ : 

- 7: Freeze the channel model parameters _**θ** c_ . 

- 8: Perform a round of mini-batch SGD updates of _**θ** e_ and _**θ** d_ with respect to _Lauto_ ( _**θ**_ ; _X_ ). 9: Sample data _Dc_[(] _[t]_[)] from the channel using the updated constellation _{_ **E** _**θ** e_ ( **1** _y_ ) _, ∀y ∈Y}_ . 

- 10: Train the channel model for _Nce_ epochs to minimize _L_ ch( _**θ** c_ ; _Dc_[(] _[t]_[)][)][ (Eq.][24).] 

## 11: **end for** 

## 12: **Return** _**θ** e,_ _**θ** c,_ _**θ** d_ . 

The training algorithm for jointly learning the autoencoder and channel model ( _e.g._ , García Martí et al. (2020)) is given in Algorithm 1. It is an alternating (cyclic) optimization of the channel parameters and the autoencoder (encoder and decoder) parameters. The reason this type of alternating optimization is required is because the empirical expectation of the CE loss Eq. (23) is valid _only when_ the channel conditional density ( _i.e._ , _**θ** c_ ) is fixed. The training algorithm can be summarized as follows. First, the channel model is trained for _Nce_ epochs using data sampled from the channel with an initial encoder constellation ( _e.g._ , M-QAM). With the channel model parameters fixed, the parameters of the encoder and decoder networks are optimized for one epoch of mini-batch SGD updates (using any adaptive learning rate algorithm _e.g._ , Adam (Kingma & Ba, 2015)). Since the channel model is no longer 

31 

Published as a conference paper at ICLR 2023 

optimal for the updated encoder constellation, it is retrained for _Nce_ epochs using data sampled from the channel with the updated constellation. This alternate training of the encoder/decoder and the channel networks is repeated for _Nae_ epochs or until convergence. 

Finally, we observe some interesting nuances of the communication autoencoder learning task that is not common to other domains such as images. 1) The size of the input space is finite, equal to the number of distinct messages _m_ . Because of the stochastic nature of the channel transfer function, the same input message results in a different autoencoder output each time. 2) There is theoretically no limit on the number of samples that can be generated for training and validating the autoencoder. These factors make the autoencoder learning less susceptible to overfitting, unlike in other domains. 

## D.4 A PRIMER ON DOMAIN ADAPTATION 

We provide a brief review of domain adaptation (DA) problem and literature. In the traditional learning setting, training and test data are assumed to be sampled independently from the same distribution _P_ ( **x** _, y_ ), where **x** and _y_ are the input vector and target respectively[8] . In many real world settings, it can be hard or impractical to collect a large labeled dataset _Dt[ℓ]_[for a] _[ target domain]_ where the machine learning model ( _e.g._ , a DNN classifier) is to be deployed. On the other hand, it is common to have access to a large unlabeled dataset _Dt[u]_[from the target domain, and a large labeled] dataset _Ds[ℓ]_[from a different but related] _[ source domain]_[9][.][Both] _[ D] s[ℓ]_[and] _[ D] t[u]_[are much larger than] _[ D] t[ℓ]_[,] and in most cases there is no labeled data from the target domain (referred to as unsupervised DA). For the target domain, the unlabeled dataset (and labeled dataset if any) are sampled from an unknown target distribution, _i.e._ , **x** _∈Dt[u][∼][P][t]_[(] **[x]**[)][ and][(] **[x]** _[, y]_[)] _[ ∈D] t[ℓ][∼][P][t]_[(] **[x]** _[, y]_[)][ .][For the source domain, the] labeled dataset is sampled from an unknown source distribution, _i.e._ , ( **x** _, y_ ) _∈Ds[ℓ][∼][P][s]_[(] **[x]** _[, y]_[)][ .][The] goal of unsupervised DA is to leverage the available labeled and unlabeled datasets from the two ˆ domains to learn a predictor, denoted by the parametric function _y_ = _f_ _**θ**_ ( **x** ), such that the following risk function w.r.t the target distribution is minimized: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0032-05.png)


where _ℓ_ (ˆ _y, y_ ) is a loss function that penalizes the prediction _y_ ˆ for deviating from the true value _y_ ( _e.g._ , cross-entropy or hinge loss). In a similar way, we can define the risk function w.r.t the source distribution _Rs_ [ _f_ _**θ**_ ]. A number of seminal works in DA theory (Ben-David et al., 2006; Blitzer et al., 2007; Ben-David et al., 2010) have studied this learning setting and provide bounds on _Rt_ [ _f_ _**θ**_ ] in terms of _Rs_ [ _f_ _**θ**_ ] and the divergence between source and target domain distributions. Motivated by this foundational theory, a number of recent works (Ganin & Lempitsky, 2015; Ganin et al., 2016; Long et al., 2018; Saito et al., 2018; Zhao et al., 2019; Johansson et al., 2019) have proposed using DNNs for adversarially learning a shared representation across the source and target domains such that a predictor using this representation and trained using labeled data from only the source domain also generalizes well to the target domain. An influential work in this line of DA is the domain adversarial neural network (DANN) proposed by Ganin & Lempitsky (2015) and later by Ganin et al. (2016). The key idea behind the DANN approach is to adversarially train a label predictor NN and a domain discriminator NN in order to learn a feature representation for which i) the source and target inputs are nearly indistinguishable to the domain discriminator, and ii) the label predictor has good generalization performance on the source domain inputs. 

**Special Cases of DA.** While the general DA problem addresses the scenario where _Ps_ ( **x** _, y_ ) and _Pt_ ( **x** _, y_ ) are different, certain special cases of DA have also been explored. One such special case is _covariate shift_ (Sugiyama et al., 2007; Sugiyama & Kawanabe, 2012), where only the marginal distribution of the inputs changes ( _i.e._ , _Pt_ ( **x** ) _̸_ = _Ps_ ( **x** )), but the conditional distribution of the target given the input does not change ( _i.e._ , _Pt_ ( _y |_ **x** ) _≈ Ps_ ( _y |_ **x** )). Another special case is the so-called _label shift_ or class-prior mismatch (Saerens et al., 2002; Du Plessis & Sugiyama, 2014), where only the marginal distribution of the label changes ( _i.e._ , _Pt_ ( _y_ ) _̸_ = _Ps_ ( _y_ )), but the conditional distribution of the input given the target does not change ( _i.e._ , _Pt_ ( **x** _| y_ ) _≈ Ps_ ( **x** _| y_ )). Prior works have proposed targeted theory and methods for these special cases of DA. 

> 8The notation used in this section is different from the rest of the paper, but consistent with the statistical learning literature. 

> 9One could have multiple source domains in practice; we consider the single source domain setting. 

32 

Published as a conference paper at ICLR 2023 

## E MDN TRAINING AND SAMPLE GENERATION 

In § 2, we briefly discussed how a Gaussian mixture density network (MDN) can be used to learn a generative model for the channel. Here, we provide details on the training algorithm for the MDN, followed by a discussion on the sampling function **h** _**θ** c_ of an MDN, and how to make the sampling function differentiable to enable SGD-based training of the autoencoder. Given a dataset of input-output pairs sampled from the channel _Dc_ = _{_ ( **z** _n,_ **x** _n_ ) _, n_ = 1 _, · · · , Nc}_ , the MDN is trained to minimize the negative conditional log-likelihood (CLL) of the data given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0033-03.png)


where the Gaussian mixture _P_ _**θ** c_ ( **x** _n |_ **z** _n_ ) is given by Eq. (1). With a large _Nc_ , the MDN can learn a sufficiently-complex parametric density model of the channel. The negative CLL objective can be interpreted as the sample estimate of the Kullback-Leibler divergence between the true (unknown) conditional density _P_ ( **x** _|_ **z** ) and the conditional density modeled by the MDN _P_ _**θ** c_ ( **x** _|_ **z** ). Therefore, minimizing the negative CLL finds the MDN parameters _**θ** c_ that lead to a close approximation of the true conditional density. Standard SGD-based optimization methods such as Adam (Kingma & Ba, 2015) can be applied to find the MDN parameters _**θ** c_ that (locally) minimize the negative CLL. 

After the MDN is trained, new simulated samples from the channel distribution can be generated from the Gaussian mixture using the following stochastic sampling method. We focus on the diagonal covariance case for simplicity, where _**σ** i_[2][(] **[z]**[)] _[ ∈]_[R] _[d]_ +[are the diagonal elements of the covariance matrix] **Σ** _i_ ( **z** ) for component _i_ . 

1. Randomly select an encoded symbol **z** from the constellation according to the prior distribution _{p_ ( **z** ) _,_ **z** _∈Z}_[10] . 

2. Randomly select a component _K_ = _i_ according to the mixture weights _{π_ 1( **z** ) _, · · · , πk_ ( **z** ) _}_ . 

3. Randomly sample **u** from the isotropic _d_ -dimensional Gaussian density **u** _∼ N_ ( _· |_ **0** _,_ **I** _d_ ). 

4. Generate the channel output as **x** = _**σ** i_[2][(] **[z]**[)] _[ ⊙]_ **[u]**[+] _**[µ]**[i]_[(] **[z]**[)][.] 

Here _⊙_ refers to the element-wise (Hadamard) product of two vectors. The channel transfer or sampling function for a Gaussian MDN can thus be expressed as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0033-11.png)


where _K ∼_ Cat( _π_ 1( **z** ) _, · · · , πk_ ( **z** )) and **u** _∼ N_ ( _· |_ **0** _,_ **I** _d_ ). Note that this transfer function is not differentiable w.r.t parameters _{πi_ ( **z** ) _}_ and the MDN weights predicting it, because of the indicator function. As such, it is not directly suitable for SGD (backpropagation) based end-to-end training of the autoencoder (see Algorithm 1). We next propose a differentiable approximation of the MDN transfer function based on the Gumbel softmax reparametrization (Jang et al., 2017), which is used in our autoencoder implementation. 

## E.1 DIFFERENTIABLE MDN TRANSFER FUNCTION 

Consider the transfer function of the MDN in Eq. (25). We would like to replace sampling from the categorical mixture prior Cat( _π_ 1( **z** ) _, · · · , πk_ ( **z** )) with a differentiable function that closely approximates it. We apply the _Gumbel-Softmax_ reparametrization (Jang et al., 2017) which solves this exact problem. Recall that the component prior probabilities can be written in terms of the prior logits as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0033-15.png)


iid Consider _k_ iid standard Gumbel random variables _G_ 1 _, · · · , Gk ∼_ Gumbel(0 _,_ 1) . It can be shown that, for any **z** _∈Z_ , the random variable 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0033-17.png)


> 10This is the same as the prior distribution over the input messages. 

33 

Published as a conference paper at ICLR 2023 

follows the categorical distribution Cat( _π_ 1( **z** ) _, · · · , πk_ ( **z** )). This standard result is known as the Gumbel-max transformation. While Eq. (26) can be directly used inside the indicator function in Eq. (25), the argmax will still result in the transfer function being non-differentiable. Therefore, we use the following _temperature-scaled softmax_ function as a smooth approximation of the argmax 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0034-02.png)


where _τ >_ 0 is a temperature constant. For small values of _τ_ , the temperature-scaled softmax will closely approximate the argmax, and the vector [ _S_[�] 1( **z** ; _τ_ ) _, · · · , S_[�] _k_ ( **z** ; _τ_ )] will closely approximate the one-hot vector [ 1 ( _S_ ( **z** ) = 1) _, · · · ,_ 1 ( _S_ ( **z** ) = _k_ )]. 

Applying this Gumbel softmax reparametrization in Eq. (25), we define a modified differentiable transfer function for the Gaussian MDN as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0034-05.png)


With this transfer function, it is straightforward to compute gradients with respect to the prior logits _αi_ ( **z** ) _, ∀i_ . Another neat outcome of this approach is that the stochastic components (Gumbel random variables _Gi_ ) are fully decoupled from the deterministic parameters _αi_ ( **z** ) in the gradient calculations with respect to _S_[�] _i_ ( **z** ; _τ_ ). In our experiments, we used this Gumbel-softmax based smooth transfer function while training the autoencoder, but during prediction (inference), we use the exact argmax based transfer function. We found _τ_ = 0 _._ 01 to be a good choice for all the experiments. 

## F SIMULATED CHANNEL VARIATION MODELS 

We provide details of the mathematical models used to create the simulated channel variations in our experiments. These models are frequently used in the study of wireless channels (Goldsmith, 2005). 

## F.1 ADDITIVE WHITE GAUSSIAN NOISE (AWGN) MODEL 

This is the simplest type of channel model where the channel output **x** is obtained by adding random Gaussian noise **n** to the channel input **z** , _i.e._ , **x** = **z** + **n** . It is assumed that the noise **n** _∼N_ ( _· |_ **0** _, σ_ 0[2] **[I]** _[d]_[)][is independent of] **[ z]**[.][We find the signal-to-noise ratio (SNR) of this channel, and] specify how to set the noise variance _σ_ 0[2][in order to achieve a target SNR value.] 

The average power in the signal component of **x** is given by E[ _∥_ **z** _∥_ 2[2][]][=] _[p]_[avg][.][The noise power in this] case is given by E[ _∥_ **n** _∥_ 2[2][]][=] _[σ]_ 0[2][.][The signal-to-noise ratio (SNR) for the AWGN model is therefore] given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0034-12.png)


where _R_ is the communication rate of the system in bits/channel use (O’Shea & Hoydis, 2017). To simulate an AWGN channel with a target SNR of _Eb/N_ 0, we select the noise variance as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0034-14.png)


## F.2 UNIFORM FADING MODEL 

The channel output **x** _∈_ R _[d]_ for this model as a function of the channel input (symbol vector) **z** _∈_ R _[d]_ is given by 

**x** = _A_ **z** + **n** _,_ 

where _A ∼_ Unif[0 _, a_ ] is a uniformly-distributed scale factor, and **n** _∼N_ ( _· |_ **0** _, σ_ 0[2] **[I]** _[d]_[)][ is an additive] Gaussian noise vector. Both _A_ and **n** are assumed to be independent of each other and **z** . The average power in the signal component of **x** is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0034-19.png)


34 

Published as a conference paper at ICLR 2023 

where _p_ avg denotes the average power in the channel input **z** . The noise power in this case is given by E[ _∥_ **n** _∥_ 2[2][]][=] _[σ]_ 0[2][.][The signal-to-noise ratio (SNR) for this model is therefore given by] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0035-02.png)


where _R_ is the communication rate of the system in bits/channel use. We select the fading factor _a_ such that the channel output has a target SNR value using the following equation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0035-04.png)


## F.3 RICEAN AND RAYLEIGH FADING MODELS 

The channel output for the Ricean fading model is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0035-07.png)


where **A** is a diagonal matrix with the diagonal elements _a_ 1 _, · · · , ad ∼_ iid Rice( _· | ν, σa_ 2[)][following a] Rice distribution, and **n** _∼N_ ( _· |_ **0** _, σ_ 0[2] **[I]** _[d]_[)][ is an additive Gaussian noise vector.][It is assumed that] **[ n]** and **A** are independent of each other and of **z** . Note that Rayleigh fading is a special case of Ricean fading when the parameter _ν_ = 0. For this model, the average power in the signal component of **x** is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0035-09.png)


where _p_ avg denotes the average power in the channel input **z** . We used the fact that the second moment of the Rice distribution is given by E[ _a_[2] _i_[]][=][2] _[ σ] a_[2][+] _[ ν]_[2][.][It is useful to consider the derived] parameters _K_ = _ν_[2] _/_ 2 _σa_[2][which corresponds to the ratio of power along the line-of-sight (LoS)] path to the power along the remaining paths, and Ω= 2 _σa_[2][+] _[ ν]_[2][which corresponds to the total] power received along all the paths. The SNR for this model is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0035-11.png)


For a given input average power and target SNR, the parameters of the Rice distribution can be set using the equation 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0035-13.png)


To create channel variations of different SNR, we fix the variance _σa_[2][and vary the power of the LoS] component _ν_[2] . Suppose the smallest SNR value considered is _S_ min, we set _σa_[2][using] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0035-15.png)


and set _ν_ to achieve a target SNR _Eb/N_ 0 using 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0035-17.png)


For this choice of parameters, the power ratio of LoS to non-LoS components is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication_images/iclr_2023_Few-Shot_Domain_Adaptation_For_End-to-End_Communication.pdf-0035-19.png)


The _K_ -factor for Rician fading in indoor channel environments with an unobstructed line-of-sight is typically in the range 4 dB to 14 dB (Linnartz, 2001). Rayleigh fading is obtained for _K_ = 0 (or _ν_ = 0). 

Finally, note that the vector **z** is composed of one or more pairs of in-phase and quadrature (IQ) components of the encoded signal ( _i.e._ , dimension _d_ = 2 _p_ ). Since each IQ component is transmitted as a single RF signal, the Ricean amplitude scale is kept the same for successive pairs of IQ components in **z** . In other words, the amplitude scales are chosen to be _a_ 1 _, a_ 1 _, · · · , ap, ap_ . This does not change any of the above results. 

35 

