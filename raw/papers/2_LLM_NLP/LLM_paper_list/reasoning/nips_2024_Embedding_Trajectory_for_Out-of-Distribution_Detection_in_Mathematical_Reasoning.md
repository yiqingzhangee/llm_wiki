# **Embedding Trajectory for Out-of-Distribution Detection in Mathematical Reasoning** 

**Yiming Wang** _[α]_ **Pei Zhang** _[β,γ]_ **Baosong Yang** _[β,]_[�] **Derek F. Wong** _[γ]_ **Zhuosheng Zhang** _[α]_ **Rui Wang** _[α,]_[�] 

> _α_ Shanghai Jiao Tong University _β_ Tongyi Lab _γ_ NLP2CT Lab, University of Macau 

� : `Corresponding Author` 

`Email:` _α_ `{yiming.wang, wangrui12}@sjtu.edu.cn` _β_ `yangbaosong.ybs@alibaba-inc.com` 

## **Abstract** 

Real-world data deviating from the independent and identically distributed ( _i.i.d._ ) assumption of in-distribution training data poses security threats to deep networks, thus advancing out-of-distribution (OOD) detection algorithms. Detection methods in generative language models (GLMs) mainly focus on uncertainty estimation and embedding distance measurement, with the latter proven to be most effective in traditional linguistic tasks like summarization and translation. However, another complex generative scenario mathematical reasoning poses significant challenges to embedding-based methods due to its high-density feature of output spaces, but this feature causes larger discrepancies in the embedding shift trajectory between different samples in latent spaces. Hence, we propose a trajectory-based method TV Score, which uses trajectory volatility for OOD detection in mathematical reasoning. Experiments show that our method outperforms all traditional algorithms on GLMs under mathematical reasoning scenarios and can be extended to more applications with high-density features in output spaces, such as multiple-choice questions. 

� `https://github.com/Alsace08/OOD-Math-Reasoning` 

## **1 Introduction** 

The rapid development of generative language models (GLMs) [40, 41, 5, 2, 50] has empowered them to fit diverse and challenging datasets, showing strong generalization over in-distribution (ID) test data satisfying the independent and identically distributed ( _i.i.d._ ) assumption. However, unconstrained inputs in real-world settings frequently trigger distributional drifts, called out-ofdistribution (OOD) data. In such scenarios, model performance often deteriorates unexpectedly, yielding harmful outcomes. Thus, OOD detection [38, 4] is critical in safeguarding model security. 

A highly scalable detection method must not rely on specific OOD data distributions, so simulating scenarios where OOD data is unavailable is more promising. Most existing research mainly focuses on vision and text classification tasks [12, 30, 55, 26, 48]. In contrast, studies addressing OOD detection on GLMs remain relatively niche, despite the more severe risks associated with OOD perils in GLMs due to the potential for error propagation in autoregressive generated sequences [43]. Existing methods on GLMs only focus on traditional text generation scenarios like summarization and translation, and these methods did not step outside the research framework of uncertainty estimation [33, 31] and embedding distance measure [43]. Among these, [43] has demonstrated that embeddingbased methods are currently the only optimal solution for text generation scenarios, they determine whether a new sample is ID or OOD by calculating the _Mahalanobis Distance_ [22] between the new sample embedding and ID embedding distribution in the static input or output space. 

Recently, mathematical reasoning has emerged as a challenging generative task and a crucial benchmark for evaluating model abilities. However, it presents unique phenomena in both input and output spaces that render embedding-based methods inapplicable, as shown in Figure 1: 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Input Output Input Output<br>Algebra Bible<br>Geometry WMT News<br>Number Theory TED Talk<br>Precalculus Health<br>Algebra WMT News<br>Output: 1Input: Suppose that  𝑓𝑓 is a function and  𝑓𝑓 [−1] is the inverse of  𝑓𝑓 .  If  𝑓𝑓(1) = 2 ,  𝑓𝑓(2) = 6 , and  𝑓𝑓(3) = 5 , then what is  𝑓𝑓 [−1] (𝑓𝑓 [−1] (6)) ? Input: Supporters say the tunnels would benefit the environment and offer Californians a more secure water supply.Output: 支持者们表示，这两条隧道将使环境受益并帮助加利福尼亚州确保供水更加安全。<br>Input: The product of the proper positive integer factors of  Number Theory 𝑛𝑛 can be written as  𝑛𝑛𝑎 𝑐𝑐+𝑏𝑏 , where  𝑥𝑥 is the number of positive divisors  𝑛𝑛 Health<br>has,  𝑐𝑐 is a positive integer, and the greatest common factor of the three integers  𝑎𝑎 ,  𝑏𝑏 , and  𝑐𝑐 is  1 . What is  𝑎𝑎+ 𝑏𝑏+ 𝑐𝑐 ? Input: The White House Coronavirus Task Force was established on January 29.<br>Output: 1 Output: 白宫冠状病毒工作组是成立于 2020 年 1 月 29 日的以应对美国 COVID-19 疫情的工作组。<br>(a)  Mathematical Reasoning (b)  Text Generation (Translation)<br>**----- End of picture text -----**<br>


Figure 1: Embedding projection and cases of input and output spaces under mathematical reasoning and text generation scenarios. We select MATH [6] dataset for mathematical reasoning and OPUS [49] for text generation, each with four diverse domains. Different colors represent different domains, with lighter and darker shades indicating input and output. We use SimCSE [9] for sentence embeddings and UMAP [34] for dimensionality reduction. Appendix B shows detailed settings and examples. 

- _**Input space** of mathematical reasoning exhibits vague clustering features across various domains, in contrast to the more defined clusters observed in text generation._ This indicates that embedding may struggle to capture the complexity of mathematical questions. 

- _**Output space** of mathematical reasoning exhibits high-density characteristics with significant overlap between different domains_ . we call this phenomenon **Pattern Collapse** . Since the output is mathematically symbolic [53, 37], it compresses the search space, increasing the likelihood of overlap between questions from disparate domains (As cases of Figure 1a). More importantly, the sequence tokenization used in GLMs allows for substantial token sharing among mathematically distinct expressions, as these mathematical tokens are primarily drawn from digits 0-9 and finite special symbols such as decimal points and square roots, rather than diverse linguistic elements. These scalar outputs lack distinctive features associated with specific domain distributions. 

More discussion of the two phenomena, especially the “ _pattern collapse_ ”, will be presented in Section 6. Given the limitations of traditional methods, we aim to explore innovative OOD detection solutions in mathematical reasoning scenarios. We transform our focus from static embedding space to the dynamic embedding trajectory, this motivation stems from a theoretical insight, as shown in Figure 2: _“pattern collapse” causes the convergence of the trajectory endpoints of different samples, leading to significant trajectory differences across samples_ . In Section 2.1, we model and prove this hypothesis to elucidate the intuition of using trajectories as a measure. Subsequently, in Section 2.2, we perform empirical experiments to investigate the underlying causes of trajectory differences between ID and OOD samples. Our findings reveal a phenomenon we term **Early Stabilization** , wherein _GLMs achieve primary reasoning in later stages for ID samples, a pattern not observed in OOD samples_ . This observation provides direct evidence for the rationale behind using trajectory as a measure. 

Based on these analyses, in Section 3, we propose the **T** rajectory **V** olatility detection algorithm ( **TV Score** ) for mathematical reasoning. Then in Section 4 and 5, we conduct extensive OOD detection experiments with diverse datasets and GLMs to validate the effectiveness of our method. We also tackle the challenging scenario of OOD quality estimation, which raises higher precision demands on the OOD scores. Results indicate that our method surpasses all traditional algorithms for GLMs under the mathematical reasoning scenario. Additionally, we demonstrate the extension of our method to a broader range of tasks with high-density features in output spaces, such as multiple-choice questions. 

**Problem Statement: OOD Detection on GLMs.** We start by formalizing GLMs. Let _X_ and _Y_ be the input and output spaces with _PX_ and _PY_ be the marginal distributions for respective space, and _PX ,Y_ is the joint data distribution defined over _X × Y_ . GLMs are trained given input sequence _**x**_ = _x_ 1 _x_ 2 _...xt ∼ PX_ of length _t_ to autoregressively generate the next token in the corresponding output sequence _**y**_ = _y_ 1 _y_ 2 _...yn ∼ PY_ of length _n_ over the likelihood model _p_ _**θ**_ ( _**y** |_ _**x**_ ) =[�] _[n] i_ =1 _[p]_ _**[θ]**_[(] _[y][i][|]_ _**[y]**[≺][i][,]_ _**[ x]**_[)][,] where each _xi_ and _yi_ are taken from vocabulary _V_ and _**θ**_ is sampled from the parameter space Θ. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Mathematical Reasoning Text Generation (e.g. Translation)<br>𝜙𝜙𝑙𝑙 ⋅:  Embedding in  𝑙𝑙 -th layer Output Space Output Space<br>𝑠𝑠𝑖𝑖, 𝑠𝑠𝑗𝑗 : samples in the dataset<br>Input Space 𝜙𝜙𝐿𝐿/2 𝑠𝑠𝑗𝑗 𝜙𝜙𝐿𝐿 𝑠𝑠𝑖𝑖 − 𝜙𝜙𝐿𝐿 𝑠𝑠𝑗𝑗 2 [≈0] Input Space 𝜙𝜙𝐿𝐿/2 𝑠𝑠𝑗𝑗<br>𝜙𝜙𝐿𝐿 𝑠𝑠𝑖𝑖 − 𝜙𝜙𝐿𝐿 𝑠𝑠𝑗𝑗 2 [≫0]<br>𝜙𝜙0 𝑠𝑠𝑖𝑖 − 𝜙𝜙0 𝑠𝑠𝑗𝑗 2 [≫0] 𝜙𝜙𝐿𝐿/2 𝑠𝑠𝑖𝑖 𝜙𝜙0 𝑠𝑠𝑖𝑖 − 𝜙𝜙0 𝑠𝑠𝑗𝑗 2 [≫0] 𝜙𝜙𝐿𝐿/2 𝑠𝑠𝑖𝑖<br>**----- End of picture text -----**<br>


Figure 2: The “pattern collapse” phenomenon only exists in mathematical reasoning scenarios, where two samples initially distant in distance will converge approximately at the endpoint after undergoing embedding shifts, and does not occur in text generation scenarios. This produces a greater likelihood of trajectory variation under different samples in mathematical reasoning. 

Assume that _P_[�] _X ,Y_ denote a distribution sufficiently different from _PX ,Y_ , the goal of OOD detection in GLMs is to find a score function _f_ ( _**x** ,_ _**y** ,_ _**θ**_ ) for each sample and a threshold _ϵ_ , which may rely on the features of _X_ , _Y_ , and Θ, to achieve a high discrimination accuracy goal: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0003-03.png)


## **2 Dynamic Embedding Trajectory Distinguishes ID and OOD Samples in Mathematical Reasoning** 

**We begin by defining the embedding trajectory and its volatility features.** Given a sample _s_ , we put its input to the language model _pθ_ with _L_ layers, and the output sequence consists of _T_ tokens. For the _t_ -th token, we denote its output embedding at layer _l_ as _**h**[t] l_[, with each embedding] a _d_ -dimensional vector. Following the definitions in [43, 51], we define the average embedding _T_ _**y** l_ = _T_[1] � _t_ =1 _**[h]** l[t]_[as the sentence embedding at layer] _[ l]_[.][Then, the embedding trajectory is formed] as a progressive chain of these embeddings: _**y**_ 0 _→_ _**y**_ 1 _→· · · →_ _**y** l →· · · →_ _**y** L−_ 1 _→_ _**y** L_ . To measure the change magnitudes of the embedding trajectory in the latent space, we define two types of trajectory volatilities: 

- _Dimension-independent volatility_ _**V** I_ ( _s_ ) _∈_ R _[d]_ : For sample _s_ , we first obtain the embedding difference vector _|_ _**y** l −_ _**y** l−_ 1 _|_ between each adjacent-layer pair _l −_ 1 and _l_ . _**V** I_ is the average of all differences across layers, it captures the local trajectory changes across individual dimensions: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0003-07.png)


- _Dimension-joint volatility VJ_ ( _s_ ) _∈_ R: For sample _s_ , we first obtain the L2-norm of the embedding difference _||_ _**y** l −_ _**y** l−_ 1 _||_ 2 between each adjacent-layer pair _l −_ 1 and _l_ . _VJ_ is the average of all such differences across layers, it captures the global changes in the trajectory: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0003-09.png)


In this section, we will clarify our motivation: _Why Dynamic Embedding Trajectory As The Measure?_ This stems from the phenomenon of “ _pattern collapse_ ” in the output space, where the endpoints of different sample trajectories converge to a high-density region. This constraint potentially increases the likelihood of trajectory differences across samples. We will model and prove this theoretical intuition in Section 2.1 through the _Dimension-independent volatility_ . After gaining insight into trajectory differences, we specifically explore the differences between ID and OOD sample trajectories. We will empirically investigate this in Section 2.2 through the _Dimension-joint volatility_ . 

## **2.1 Theoretical Intuition: Trajectory Differences with Higher Likelihood** 

Figure 1 has illustrated the “ _pattern collapse_ ” phenomenon in the output space in mathematical reasoning scenarios. We abstract this phenomenon in Figure 2, which compares the trajectory 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0004-00.png)


**----- Start of picture text -----**<br>
5040 OOD data: GSM8KIn-Distribution (ID)Out-of-Distribution (OOD) 5040 OOD data: AddSub In-Distribution (ID) Out-of-Distribution (OOD) 10080 OOD data: SVAMPInOut-of-Distribution (OOD)-Distribution (ID) 5040 OOD data: SingleEqIn-Distribution (ID)Out-of-Distribution (OOD) 5040 OOD data: SingleOpIn-Distribution (ID)Out-of-Distribution (OOD)<br>30 30 60 30 30<br>20 20 40 20 20<br>10 10 20 10 10<br>0 0 0 0 0<br>0 5 10 15 20 25 30 0 5 10 15 20 25 30 0 5 10 15 20 25 30 0 5 10 15 20 25 30 0 5 10 15 20 25 30<br>Layer Number Layer Number Layer Number Layer Number Layer Number<br>8070 OOD data: MATH In-Distribution (ID) Out-of-Distribution (OOD)-algebra 7060 OOD data: MATHIn-Distribution (ID)Out-of-Distribution (OOD)-geometry 60 OOD data: MATH-counting_and_probabilityIn-Distribution (ID) Out-of-Distribution (OOD) 100 OOD data: MATHIn-Distribution (ID) Out-of-Distribution (OOD) -number_theory 120 OOD data: MATHIn-Distribution (ID) Out-of-Distribution (OOD) -precalculus<br>60 50 50 80 100<br>50403020 403020 403020 6040 806040<br>10 10 10 20 20<br>0 0 0 0 0<br>0 5 10 15 20 25 30 0 5 10 15 20 25 30 0 5 10 15 20 25 30 0 5 10 15 20 25 30 0 5 10 15 20 25 30<br>Layer Number Layer Number Layer Number Layer Number Layer Number<br>2-norm value 2-norm value 2-norm value 2-norm value 2-norm value<br>2-norm value 2-norm value 2-norm value 2-norm value 2-norm value<br>**----- End of picture text -----**<br>


Figure 3: Trajectory volatility curve comparisons between one ID data and ten OOD data from diverse mathematical domains. Each trajectory represents the average of all samples from the corresponding datasets, with color shading being the sample standard deviation. Llama2-7B is used for the backbone. 

trend between different samples in the mathematical reasoning and text generation scenarios: In mathematical reasoning, when initial points of two trajectories are separated by any distance in the input space, they typically converge to a significantly closer distance in the output space after undergoing an embedding shift. However, in text generation, outputs from different samples may not exhibit this same convergence. This finding inspires the following theoretical intuition: **Hard constraints on trajectory endpoints in mathematical reasoning allow for a higher probability of trajectory differences under different samples** , as expressed by the key hypothesis: 

**Hypothesis 1** _In scenarios characterized by pattern collapse in the output space, the probability of trajectory volatility differences across samples increases._ 

We now formalize this hypothesis. Assume that the output space is a Gaussian distribution _N_ ( _**c** ,_ Σ[2] ), so the output embedding _**y** L ∼N_ ( _**c** ,_ Σ[2] ). For two different samples _si_ and _sj_ , their dimensionindependent volatilities are _**V** I_ ( _si_ ) and _**V** I_ ( _sj_ ), respectively. We want to show that when pattern collapse exists, the probability of _**V** I_ ( _si_ ) _−_ _**V** I_ ( _sj_ ) = **0** will increase. According to the pattern collapse features under different scenarios, we can constraint the _**y** L_ as follows: 

- For mathematical reasoning with pattern collapse, Σ _→_ _**O**_ , so we approximate that _**y** L ≡_ _**c**_ ; 

- For text generation without pattern collapse, Σ = diag( _δ_ 1 _, δ_ 2 _, · · · , δD_ ) _̸_ = _**O**_ , so _**y** L ̸≡_ _**c**_ . 

With such formalized constraints, we model our Hypothesis 1 as a main theorem: 

**Theorem 2.1 (Main Theorem)** _We assume that {_ _**y** l}[L] l_ =1 _[are][all][independent][variables][sampling] from vector space_ R _[d] . For different samples si and sj, their embedding sets are {_ [ _**y** i_ ] _l}[L] l_ =1 _[and] {_ [ _**y** j_ ] _l}[L] l_ =1 _[, respectively.][The likelihood of trajectory volatility differences between][ s][i][ and][ s][j][under] mathematical reasoning scenarios is higher than that under text generation scenarios, which means:_ 

E _{_ [ _**y** i_ ] _l}Ll_ =1 _[,][{]_[[] _**[y]**[j]_[]] _[l][}][L] l_ =1 _[∼U]_[(][R] _[d]_[)] _[ {]_ _**[V]**[I]_[(] _[s][i]_[)] _[ −]_ _**[V]**[I]_[(] _[s][j]_[)] _[ ̸]_[=] **[ 0]** _[|]_[[] _**[y]**[i]_[]] _[L][,]_[ [] _**[y]**[j]_[]] _[L][≡]_ _**[c]**[}] >_ E _{_ [ _**y** i_ ] _l}Ll_ =1 _[,][{]_[[] _**[y]**[j]_[]] _[l][}][L] l_ =1 _[∼U]_[(][R] _[d]_[)] � _**V** I_ ( _si_ ) _−_ _**V** I_ ( _sj_ ) _̸_ = **0** _|_ [ _**y** i_ ] _L,_ [ _**y** j_ ] _L ∼N_ ( _**c** ,_ Σ[2] )� Due to space limits, we present the **Complete Proof** in Appendix C. This demonstrates that when pattern collapse occurs, the probability of trajectory volatility differences across samples increases. 

## **2.2 Empirical Analysis: Early Stabilization of ID samples** 

The theoretical analysis in Section 2.1 provides an intuition for using embedding trajectory as the measure for distinguishing between different samples. However, the specific trajectory differences between ID and OOD samples remain unclear. Thus, we investigate this empirically in this section. 

We select eleven mathematical reasoning datasets: one for ID data and ten for OOD data. The MultiArith dataset serves as the ID data, while the OOD datasets include GSM8K, SVAMP, AddSub, SingleEq, SingleOp, and MATH. The latter encompasses five tasks across various mathematical domains: Algebra, Geometry, Counting and Probability, Number Theory, and Precalculus. This selection includes varying task types and levels of difficulty. We first train a base Llama2-7B (32 layers) using the ID training set, then inference on the ID test set and all OOD test sets. Details regarding the datasets, model, and implementation can be found in Section 4.1. 

4 

We measure the dimension-joint volatility for all samples from each ID and OOD test set. Specifically, we compute each adjacent-layer change magnitude _||_ _**y** l −_ _**y** l−_ 1 _||_ 2, and connect all values as a change curve, with higher value means the higher volatility. Figure 3 shows ten curve comparisons, with each sub-figure including the ID data and one OOD data. We find that the change magnitude slightly until the 20th layer. After 20 layers, for ID data, the change magnitude is again suppressed after a few layers of inference, while for OOD data, the magnitude is maintained at a relatively high level. 

We term this phenomenon “ **Early Stabilization** ”: For ID data, GLMs largely complete their reasoning in the mid-to-late stages, and simple adjustments are sufficient after that. However, for OOD data, GLMs can still not complete accurate reasoning at a later stage. They thus can only randomly switch to a specific output pattern, _i.e._ , the scalar mathematical expression pattern. This provides strong evidence that using embedding trajectory for OOD detection may be effective. 

## **3 TV Score: Trajectory Volatility Score for OOD Detection** 

In Section 2.2, we have identified a significant difference in embedding trajectory volatilities between ID and OOD samples in mathematical reasoning. We now aim to leverage this phenomenon to develop a lightweight OOD detection solution tailored for mathematical reasoning scenarios. 

Inspired by static embedding methods that use the ID sample embedding cluster as the reference to measure the _Mahalanobis Distance_ (MaDis) [22], we similarly aim to use the ID sample trajectory cluster as the reference to measure the difference between them and a new sample trajectory. While measuring the difference between an embedding and an embedding cluster only requires a simple MaDis calculation, quantifying the difference between a trajectory and a trajectory cluster is less intuitive. Thus, we seek a transition idea starting from the static space Gaussian assumption. 

First, we obey the assumption under static embeddings [3, 43] to fit all ID embeddings at each layer _l_ to a Gaussian distribution _Gl_ = _N_ ( _**µ** l,_ **Σ** _l_ ). Next, for a new sample with _**y** l_ be its embedding at layer _l_ , we map it to its MaDis _f_ ( _**y** l_ ) : R _[d] →_ R with _Gl_ as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0005-06.png)


Finally, we treat the MaDis differences _|f_ ( _**y** l_ ) _− f_ ( _**y** l−_ 1) _|_ between adjacent layer-pairs _l −_ 1 and _l_ as the adjacent-layer volatility of the new sample, and average all adjacent-layer volatilities as the final trajectory volatility score (TV Score _S_ ) of this sample: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0005-08.png)


The anticipated trend is that when the two adjacent embeddings after MaDis mapping exhibit a greater difference, the embedding change between the two layers is more volatile compared to ID data. This enables us to identify new samples with trajectory volatility features that significantly deviate from those of the ID samples, thus increasing the likelihood that they are OOD samples. 

Furthermore, as a trajectory, outliers in the trajectory may significantly impact feature extraction [21, 28]. To mitigate this, we explore higher-order differential smoothing techniques to enhance trajectory smoothness. We first define the _k_ -order embedding _∇[k]_ _**y** l_ and Gaussian distribution _∇[k] Gl_ = _∇[k] N_ ( _**µ** l,_ **Σ** _l_ ) for all _l ≤ L − k_ based on the backward difference: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0005-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0005-12.png)


Following the definition of TV Score, we define the trajectory volatility score after differential smoothing (TV Score w/ DiSmo _∇[k] S_ ) as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0005-14.png)


The **Algorithmic Process** and **Computational Complexity** are detailed in Appendix D. 

5 

## **4 Experiments** 

## **4.1 Setup** 

**Dataset Selection.** For the ID dataset, we use the MultiArith [44], which consists of Math Word Problems on arithmetic reasoning. For the OOD datasets, we intuitively introduce two types of detection scenarios following [43]: (i) _Far-shift OOD_ scenario, we select the MATH [11] dataset with five domains of algebra, geometry, counting and probability, number theory, and precalculus; (ii) _Near-shift OOD_ scenario, we select five independent datasets: GSM8K [6], SVAMP [39], AddSub [13], SingleEq [18], and SingleOp [18]. We consider the ID data negative(-) and the OOD data positive(+). Refer to Appendix E.1 for basic information and OOD features of these datasets. 

**Data Split and Sampling.** Given the limited data size of MultiArith, totaling only 600 samples and lacking a standard division, we allocate 360 samples for training and 240 for testing. However, with such a small test set, randomness in evaluation becomes a concern. To mitigate this, we conduct test sampling and set the sampling size as 1000. Specifically, we denote ID dataset as _D_ in and OOD dataset as _D_ out. For each sampling, the collection is _{D_ in _, D_[�] out _}_ where _D_[�] out _⊂D_ out and _|D_ in _|_ = _|D_[�] out _|_ , this guarantees positive and negative sample balance. We report both the mean and standard variance of the results to enhance the reliability of evaluations. Refer to Appendix E.2 for the ID dataset split. 

**Implementation.** _To measure the application value of our method used in cutting-edge GLMs_ , we use Llama2-7B [50] and GPT2-XL (1.5B) [5] as our backbones for ID dataset training. Refer to Appendix E.3 for training details. However, there exists uncertainty about the data used in the pre-training phase, especially for Llama2 because its data is closed-source. Some research [57, 59] have confirmed the absence of data leakage in Llama2 for the MATH and GSM8K datasets, we still conduct pre-experiments to examine the rationality of the OOD data selection rigorously. Our criterion is the claim that a dataset can be categorized as OOD if it exceeds the capabilities of the base model, as proposed in prior studies [47, 27]. Results of the pre-experiments are shown in Appendix E.4, and they can confirm that these datasets can be considered as OOD data for the two GLMs. 

**Baseline.** We compare our method with five training-free baselines where OOD training data are unavailable. We refer to the latest survey [20] to select them for the scarcity of OOD detection methods on GLMs: (1) Maximum Softmax Probability (Prob.) [12]; (2) Monte-Carlo Dropout [8]; (3) Sequence Perplexity [3]; (4) Input Embedding [43]; (5) Output Embedding [43]. Refer to Appendix E.5 for details. Additionally, we set the smoothing order _k_ ranges from 1 to 5 for TV Score w/ DiSmo and report the highest among them when with smoothing. 

**Evaluation.** We divide the OOD detection evaluation into two scenarios: (1) **Offline Detection** , which classifies whether samples from a given list belong to OOD. For each collection _{D_ in _, D_[�] out _}_ , we report the AUROC [35] and FPR95 metrics. The former represents the area under the ROC curve and the latter represents the value of FPR at 95% TPR. (2) **Online Detection** , which utilizes the offline detection results to calculate an optimal classification threshold for a direct determination of whether new samples belong to OOD. We introduce the metrics in the corresponding result sub-sections. 

## **4.2 Main Results** 

**Offline Detection.** Table 1 presents the results of the offline detection scenarios. 

- _Performance Analysis_ : In the far-shift OOD setting, **our average performance surpasses 98 (Llama2-7B) and 96 (GPT2-XL) under the AUROC metric** , surpassing the optimal baseline by 10+ points. Moreover, our performance stands at **an impressive 5.21 (Llama2-7B) and 9.89 (GPT2-XL) under the FPR95 metric, representing a remarkable 80%+ reduction** compared to the optimal baseline, far surpassing all baseline methods. In the near-shift OOD setting, the robustness of our method is even more impressive. All of the baseline methods show significant performance degradation, especially in Llama2-7B, with the AUROC metric decreasing below 60 and the FPR95 elevating above 80. However, our method maintains excellent performances, with AUROC scores surpassing 90 and FPR95 below 30. This indicates that for more fine-grained OOD detection scenarios, our method demonstrates greater adaptability. 

- _Model Analysis_ : Comparing performances of Llama2-7B and GPT2-XL, we find two phenomena: (i) Results on GPT2-XL are more stable, performance differences between GPT2-XL on farand near-drift settings are not significant, while Llama2-7B shows a significant performance 

6 

Table 1: AUROC and FPR95 results of the **Offline Detection** scenario. Underline and **bold** denote SOTA among all baselines and all methods, respectively. We report the _average_ results under each setting in the main text, results of each dataset are shown in Table 11 and 12 (Appendix F). 

|_Model_|_Model_|||**Llama2-7**|**B**[50]||||**GPT2-X**|**L**[5]||
|---|---|---|---|---|---|---|---|---|---|---|---|
|_Method_<br>_Metric_|||Far-shift|OOD|Near-shif|t OOD||Far-shift|OOD|Near-shif|t OOD<br>FPR95_↓_|
||||AUROC_↑_|FPR95_↓_|AUROC_↑_|FPR95_↓_||AUROC_↑_|FPR95_↓_|AUROC_↑_||
|Max Softmax Prob. [12]<br>Monte-Carlo Dropout[8]<br>Perplexity[3]<br>Input Embedding[43]<br>Output Embedding[43]|||78.66_±_1.38<br>68.63_±_2.21<br>85.64_±_1.46<br>75.89_±_1.03<br>74.86_±_1.39|81.44_±_3.56<br>87.04_±_4.88<br>53.06_±_4.36<br>67.87_±_3.69<br>75.21_±_2.16|60.14_±_1.54<br>52.33_±_2.21<br>59.35_±_1.89<br>60.33_±_1.37<br>44.50_±_1.06|88.91_±_2.41<br>91.92_±_1.89<br>86.09_±_1.89<br>84.65_±_2.53<br>86.46_±_1.59||70.54_±_1.42<br>66.18_±_1.87<br>80.82_±_1.04<br>86.26_±_0.84<br>77.95_±_1.16|78.29_±_2.02<br>84.69_±_1.65<br>64.53_±_2.10<br>49.33_±_2.10<br>65.64_±_3.42|67.12_±_1.20<br>63.54_±_1.72<br>73.74_±_1.12<br>83.22_±_0.88<br>79.28_±_1.24|76.27_±_2.66<br>78.08_±_2.50<br>72.39_±_1.27<br>52.90_±_3.16<br>64.70_±_2.72|
|TV Score|(Ours)||**98.76**_±_**0.11**|**5.21**_±_**0.98**|**92.64**_±_**0.39**|**28.39**_±_**1.38**||93.47_±_0.08|24.10_±_0.95|**94.86**_±_**0.23**|13.82_±_0.36|
|w/ DiSm|o (Ours)||93.25_±_0.76|41.82_±_4.69|56.99_±_1.41|88.01_±_1.71||**96.54**_±_**0.11**|**9.89**_±_**0.61**|94.19_±_0.25|**13.66**_±_**0.69**|
|∆(**bold**-|underline<br>)||+13.12|-47.85|+32.31|-56.26||+10.28|-39.44|+11.64|-39.24|



Table 2: Accuracy and Robustness results of the **Online Detection** scenario. We mainly compare our method with embedding-based methods, and **bold** denotes the best among these methods. 

||Far-shift OOD Setting||Near-shift OOD Setting|
|---|---|---|---|
|_Dataset_|**Accuracy**_↑_<br>**Robustness**_↓_<br>_Dataset_<br>I-Emb. / O-Emb. / TV (ours)||**Accuracy**_↑_<br>**Robustness**_↓_<br>I-Emb. / O-Emb. / TV (ours)|
|Algebra<br>Geometry<br>Cnt.&Prob<br>Num.Theory<br>Precalculus|76.43 / 45.42 /**93.88**<br>5.27 / 6.94 /**0.97**<br>74.32 / 54.79 /**94.47**<br>2.44 / 2.43 /**1.65**<br>50.31 / 27.55 /**93.74**<br>9.99 /**2.34**/ 2.36<br>85.80 / 54.38 /**92.08**<br>3.31 / 11.45 /**2.34**<br>80.33 / 88.50 /**99.28**<br>6.13 / 1.38 /**0.67**|GSM8K<br>SVAMP<br>AddSub<br>SingleEq<br>SingleOp|81.49 / 75.32 /**93.39**<br>10.08 / 3.36 /**2.05**<br>68.66 / 63.33 /**94.88**<br>5.26 / 3.54 /**2.13**<br>**79.16**/ 78.09 / 74.11<br>3.21 / 6.98 /**2.77**<br>59.83 / 72.56 /**93.15**<br>11.57 /**3.14**/ 3.17<br>69.38 / 62.20 /**95.75**<br>4.00 /**2.37**/ 2.45|
|_Average_|73.44 / 54.13 /**94.69**<br>5.43 / 4.91 /**1.60**|_Average_|71.70 / 70.30 /**90.26**<br>6.82 / 3.88 /**2.51**|



degradation (mainly for baselines) on near-shift setting; (ii) the DiSmo technique is more effective on GPT2-XL, which suggests that there are more anomalous learning tendencies in latent spaces of small models, and the smoothing helps to minimize these anomalies. 

In addition, we conduct significant tests (Details are shown in Table 11 - 14). We find that our methods almost pass all significance tests, while the embedding-based methods have the lowest pass rate among baselines, suggesting that their results are more susceptible to sampling error. We also find that the performance of differential smoothing fluctuates greatly in different settings. Therefore, we conduct the ablation of the smoothing order _k_ . Results and analyses are shown in Appendix F.1. 

**Online Detection.** In this part, we utilize the TV score for online OOD discrimination. For each collection _{D_ in _, D_[�] out _}_ , we obtain a detector and computer the optimal cut-off _τi_ of Youden Index, which is at the point in the AUROC curve where TPR _−_ FPR is maximum. Then for all OOD samples _s ∈D_ out _− D_[�] out, we donate _t_ as the sampling size and computer the discrimination accuracy: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0007-07.png)


In addition, the discrimination accuracy should vary less under different data collections, reflecting the discriminator’s robustness. Therefore, we denote the Robustness metric as sampling variance. 

Table 2 presents the results in Llama2-7B. Compared to the embedding-based methods, **our TV score obtains about an average of 20-point accuracy improvement in both far-shift OOD and near-shift OOD settings** , and on some datasets, such as Cnt.&Prob, our TV score achieves more than 40 points of improvement. These all imply that TV Score can perform online discrimination of OOD samples more accurately. In addition, our TV score also possesses stronger robustness, which means that in real scenarios, we can find the optimal threshold more consistently in the face of different accessible ID and OOD data, reducing the potential riskiness due to uncontrollable data acquisition. 

7 

## **5 Generalizability Exploration** 

## **5.1 Beyond Detection: OOD Quality Estimation** 

In this part, we utilize the TV score for generative quality estimation (QE). For text generation, the QE performance is usually measured by calculating the correlation coefficient between automatic scores and human ratings. However, QE in mathematical reasoning scenarios is not a well-defined problem. For one mathematical question, its answer is either right or wrong, the intermediate state does not exist. For example, when the correct answer is 12 _._ 5, it is difficult to judge which is better between generated answers of 1 _._ 25 and 13 _._ 6. The human approach may be to judge by comparing the difference value or similarity like Rouge [24] and BertScore [60] between the generated answer and the correct answer, which is unfair to the machine because there is a lot of randomness in the intermediary process of computation, and the solution pattern of machines is case-based [14], so it is not suitable to judge the machine-generated results with customized mathematical rules. 

Therefore, we use the binary direct matching[1] to compare the model-generated answers with the correct answers. Considering the open-ended output of the GLMs, we give a loose matching condition, _i.e._ , as long as the correct answer is included in the generated answer by the model, the generated answer is recognized as correct and the matching score is 1, otherwise the matching score is 0. We compute the Kendall rank correlation coefficient _τ_ [45] and Spearman rank correlation coefficient [36] between each OOD score and the matching score. 

Table 3: **OOD Quality Estimation** : Kendall’s _τ_ and Spearman correlation between various OOD scores and benchmark quality metric binary matching. Each column shows the correlation when ID and OOD samples are merged. Underline denotes the SOTA among all baselines, and **bold** denotes the SOTA among our methods. We report the _average_ results under each setting in the main text, results of each dataset are shown in Table 13 and 14 (Appendix F). 

|_Model_|_Model_|||**Llama2-**|**7B**[50]||||**GPT2-X**|**L**[5]||
|---|---|---|---|---|---|---|---|---|---|---|---|
|_Method_<br>_Metric_|||Far-shift|OOD|Near-shif|t OOD||Far-shift|OOD|Near-shif|t OOD<br>Spearman_↑_|
||||Kendall_↑_|Spearman_↑_|Kendall_↑_|Spearman_↑_||Kendall_↑_|Spearman_↑_|Kendall_↑_||
|Max Softmax Prob.<br>Perplexity<br>Input Embedding<br>Output Embedding|||0.024_±_0.020<br>0.050_±_0.015<br>0.078_±_0.016<br>0.058_±_0.018|0.038_±_0.020<br>0.045_±_0.016<br>0.102_±_0.017<br>0.025_±_0.017|0.038_±_0.018<br>0.074_±_0.017<br>0.036_±_0.018<br>0.038_±_0.015|0.026_±_0.018<br>0.050_±_0.018<br>0.115_±_0.017<br>0.012_±_0.017||0.066_±_0.015<br>0.036_±_0.014<br>0.059_±_0.012<br>0.050_±_0.012|0.044_±_0.016<br>0.038_±_0.017<br>0.098_±_0.016<br>0.016_±_0.017|0.057_±_0.018<br>0.035_±_0.018<br>0.012_±_0.018<br>0.036_±_0.017|0.057_±_0.022<br>0.058_±_0.019<br>0.068_±_0.016<br>0.029_±_0.021|
|TV Score|(Ours)||**0.161**_±_**0.012**|0.147_±_0.015|**0.159**_±_**0.017**|**0.158**_±_**0.017**||0.138_±_0.010|0.123_±_0.013|**0.131**_±_**0.015**|0.146_±_0.015|
|w/ DiSm|o (Ours)||0.111_±_0.016|**0.152**_±_**0.015**|0.113_±_0.018|0.134_±_0.017||**0.139**_±_**0.009**|**0.141**_±_**0.014**|0.123_±_0.014|**0.154**_±_**0.016**|
|∆(**bold**-|underline<br>)||+0.083|+0.050|+0.085|+0.043||+0.073|+0.043|+0.074|+0.086|



Table 3 presents the results. For Llama2-7B, when compared with Kendall correlation, the correlation improvement of TV scores **over SOTA baselines reaches up to 100% under both far-shift and near-shift OOD settings.** Compared with Spearman correlation, TV scores demonstrate a correlation enhancement **over SOTA baselines by up to 100% under far-shift OOD setting and 30% under near-shift OOD setting** . GPT2-XL also demonstrates excellent performance. These findings indicate that our TV scores not only facilitate the binary discrimination of ID and OOD samples but also substantially reflect the quality and precision of generated mathematical reasoning. 

## **5.2 Beyond Mathematical Reasoning** 

Apart from mathematical reasoning, our method also has a wider range of potential applications that **can be extended to any task where the output space exhibits the pattern collapse property** . An example would be multiple-choice questions, which is a popular evaluation tool in the era of large language models and also display the pattern collapse property due to the limited output space being confined to the “ABCD” four options. To verify the generalizability of our method, we conduct experiments using the multiple-choice dataset MMLU [10], and our method also outperforms all traditional algorithms in this setting. Results and analyses are shown in Appendix F.4. 

> 1Although direct matching is the most accurate solution, it suffers from two issues: (i) Generated answers may include much noisy content, increasing the matching difficulty; (ii) Performances on GLMs of mathematical reasoning is poor, which unbalances the positive and negative samples and increases the randomness. 

8 

## **6 Rethink Inapplicability of Static Embedding Methods** 

In Figure 1, we find that the static embedding space for both input and output fails to capture distinct features across different domains. This renders traditional embedding-based methods unsuitable for mathematical reasoning tasks. Experimental results in Section 4 also verify this disadvantage. In this section, we conduct a detailed analysis of these phenomena, and reveal the root cause of the poor performance of static embedding methods in mathematical reasoning scenarios. 

## **6.1 Input Space: Representation Dilemma on Mathematical Expressions** 

The input space in mathematical reasoning exhibits vague clustering features compared to text generation scenarios as shown in Figure 1. We speculate that as semantic representations, embeddings cannot accurately measure mathematical expressions in the mathematical sense. 

We begin by considering a counter-intuitive toy example. We construct a benchmark expression “A: `2+3+4=?` ” and three diverse expressions: “X: `8-0+9=?` ”, “Y: `xˆ2*yˆ3*xˆ4=?` ”, and “Z: `234*2+345*4+456*4-243*3=?` ”. From the mathematical sense, X is ID for A, whereas Y and Z are OOD for A since the difficulty and knowledge used to solve Y and Z are largely different from A compared to X. Table 4 presents the cosine 

Table 4: A toy example about cosine similarities between different mathematical expressions. 

||Mathematical Expression|Cosine with Benchmark|
|---|---|---|
||Benchmark: `2+3+4=?`||
||`8-0+9=?`|0.65|
||`xˆ2*yˆ3*xˆ4=?`|0.83|
||`234*2+345*4+456*4-243*3=?`|0.78|



similarity between each of them and A. Notably, X exhibits the lowest similarity to A, primarily due to having fewer token sharing with A. This is inconsistent with human perception of mathematics. 

Table 5: AUROC score matrix produced after alternating the MATH dataset’s five domains as ID and OOD data measured by **(a) Input Embedding Mahalanobis Distance** and **(b) Output Embedding (w/ CoT) Mahalanobis Distance** . Darker colors represent better performances. 

|**(w/ CoT) M**|**ahalanobis Distance**. Darker colors represent better performances.|**ahalanobis Distance**. Darker colors represent better performances.|
|---|---|---|
||OOD Domain<br>**(a) Input Embedding MaDis**<br>**(b) Output Embedding (w/ CoT) MaDis**<br>algebra<br>geometry<br>cnt.&prob<br>num.theory<br>precalculus<br>algebra<br>geometry<br>cnt.&prob<br>num.theory<br>precalculus||
||**(a) Input Embedding MaDis**||
||algebra<br>geometry<br>cnt.&prob<br>num.theory<br>precalculus||
|ID Domain<br>algebra<br>geometry<br>cnt.&prob<br>num.theory<br>precalculus|-<br>63.77<br>80.01<br>58.79<br>50.80|-<br>71.37<br>84.24<br>65.06<br>56.71|
||85.68<br>-<br>88.14<br>86.55<br>69.03|72.12<br>-<br>92.42<br>86.08<br>70.89|
||49.95<br>44.94<br>-<br>38.02<br>51.76|46.19<br>53.61<br>-<br>40.15<br>51.63|
||66.35<br>78.87<br>50.00<br>-<br>71.93|63.02<br>78.76<br>69.85<br>-<br>67.96|
||85.14<br>79.31<br>86.18<br>89.34<br>-|79.84<br>85.65<br>86.77<br>90.35<br>-|



To demonstrate this phenomenon, we use one of five domains from the MATH dataset as the ID and the remaining four domains as OOD. We rotate this setting across five iterations. Under each ID-OOD pair, we allocate half of the ID samples to compute the ID embedding distribution, while the other half and all OOD samples constitute the test samples. The Mahalanobis distance is then calculated between embeddings and the ID distributions for each sample input, yielding the OOD score. We employ SimCSE [9] to generate sentence embeddings. Table 5(a) presents the AUROC score matrix, in nearly half of the settings, the AUROC value hovers around 50, _i.e._ , completely random. This suggests that embeddings can not distinguish between domains in a mathematical sense. 

## **6.2 Output Space: High-density “** _**Pattern Collapse**_ **”** 

The output space in mathematical reasoning exhibits a unique high-density characteristic, which we call “ _**pattern collapse**_ ”. As shown in Figure 1, the distribution of output embeddings across various domains in the mathematical reasoning scenarios is significantly concentrated and indistinguishable. 

Two reasons cause this phenomenon: **(1) Expression level** : Output space of mathematical reasoning is scalar [37], resulting in a compressed search space with a higher probability of overlap in two widely 3 divergent questions. For example, “1 + 3 =” and “�1 _[x]_[d] _[x]_[=][” are both “][4][”;] **[ (2) Token level]**[:][We] usually categorize different mathematical reasoning domains from a mathematical sense. However, _GLMs model real numbers or mathematical expressions not in a mathematical sense, but based on the average embedding of all discrete token embeddings after sequence tokenization_ . Through 

9 

tokenization, two expressions that are very different in the mathematical sense ( _e.g._ , Two numbers that are far apart on the real number line) may share many tokens because regular mathematical expressions are only taken from 0-9 number tokens and a limited number of special symbols, such as decimal points, slashes, and root signs. Thus, the collapse occurs at the token level during the autoregressive prediction of each token. 

We conduct the following statistical analysis to demonstrate the universality of “ _pattern collapse_ ” across various mathematical reasoning tasks: We categorize the mathematical tasks into various types across different domains and difficulties. In each task, we count the token number _N_ and the token type number _NT_ in the dataset, then compute the token duplication rate _D_ and the vocab coverage _C_ as follows: We use Llama-2 tokenizator [50], whose token type number in vocab _NV_ is 32000. The computation metric is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0010-02.png)


We also test translation and summarization tasks by taking samples with the same token size as the mathematical reasoning dataset for a clear comparison. Table 6 presents the statistics data, we find that: (i) The average token duplication rate was 98.9% on all math tasks, and even a staggering 99.9% on some easy arithmetic tasks; In contrast, the token duplication rate on the text 

Table 6: Statistics about output tokens on mathematical reasoning (seven different task types) and text generation (translation and summarization). 

|Task Type<br>_N_<br>_NT_<br>_D_<br>_C_|Task Type<br>_N_<br>_NT_<br>_D_<br>_C_|
|---|---|
|Mathematical Reasoning||
|||
|Arithmetic(easy)<br>Arithmetic(hard)<br>Algebra<br>Geometry<br>Cnt.&Prob.<br>Num.Theory<br>Precalculus|16136<br>14<br>99.9%<br>0.04%<br>5663<br>16<br>99.7%<br>0.05%<br>5234<br>107<br>98.0%<br>0.33%<br>2615<br>75<br>97.1%<br>0.23%<br>2524<br>43<br>98.3%<br>0.13%<br>2395<br>71<br>97.1%<br>0.22%<br>3388<br>84<br>97.5%<br>0.26%|
|||
|_Average_|_5422_<br>_58_<br>_98.9%_<br>_0.18%_|
|Text Generation||
|||
|Translation|2500<br>1065<br>57.4%<br>3.32%<br>5000<br>1832<br>63.3%<br>5.10%<br>10000<br>2980<br>70.2%<br>9.31%|
|||
|_Average_|_5833_<br>_1959_<br>_66.4%_<br>_6.12%_|
|||
|Summarization|2500<br>1265<br>49.4%<br>4.01%<br>5000<br>1970<br>60.6%<br>6.16%<br>10000<br>3192<br>68.0%<br>9.98%|
|||
|_Average_|_5833_<br>_2142_<br>_63.2%_<br>_6.69%_|



generation task is only about 60%, with about 2000 token types, and still increasing as the total number of tokens increases. These data and comparisons demonstrate that pattern collapse occurs on mathematical reasoning and not on text generation. (ii) Token repetition rate exceeded 97% on all seven math tasks of different difficulties and types. From these conclusions, we can demonstrate that the “ _pattern collapse_ ” occurs on generally all types of mathematical reasoning tasks. 

## **6.3 Can Chain-of-Thought Address “** _**pattern collapse**_ **”?** 

A straightforward approach to addressing “ _pattern collapse_ ” in the output space is to leverage chainof-thought (CoT) techniques [54, 17, 62, 52, 61] to **expand the output space size** . Likewise, we adopt the solution steps associated with each sample in the MATH dataset as the output and employ SimCSE to derive embedding representations. The experimental setup aligns with Section 4.1, and the results are shown in Table 5(b). We note a similar phenomenon as in Table 5(a), _i.e._ , the detection accuracy under different ID-OOD pairs varies greatly, and thus the detection randomness is more pronounced. This suggests that despite that CoT expands the output space size, the output answer is still essentially related to the difficulty and digit of mathematical reasoning, and the semantic embedding representation cannot reflect these features accurately. 

## **7 Conclusion** 

We propose the TV Score, a lightweight OOD detection method for mathematical reasoning, which distinguishes between ID and OOD samples by the embedding trajectory volatility in the latent space. We identify bottlenecks in OOD detection for mathematical reasoning and prove them empirically and theoretically. Experiments show that our method substantially outperforms all traditional algorithms, and can be extended to more application scenarios beyond mathematical reasoning. 

10 

## **Limitations** 

Our limitation mainly lies in the relatively small sizes of datasets used in our experiments. Due to the difficulty of collecting and labeling mathematical reasoning data, dataset sizes in this field are generally small, mostly in the hundreds or thousands, making it difficult to obtain millions of data for training and reasoning as in translation or summarization tasks. To address this, we adopt test sampling to reduce the randomness under small-scale testing and mitigate the data imbalance. 

## **Ethics Statement** 

The data and models used in this work are sourced from the official version of the original paper, and we strictly adhere to the provided usage protocol. Regarding the data, no modifications have been made to the original dataset. Regarding the models, supervised fine-tuning and OOD data inference are involved. To mitigate the risk of uncontrollable outputs, all generated outputs in the experiments have been reviewed to ensure their safety. Furthermore, as our focus is solely on mathematical reasoning and does not involve sensitive content, we would not cause any potential societal impact. 

## **Acknowledgment** 

This work is supported by the Alibaba Innovative Research Program. Yiming and Rui are partially supported by the General Program of the National Natural Science Foundation of China (62176153), the Shanghai Municipal Science and Technology Major Project (2021SHZDZX0102), and SMPZhipu.AI Large Model Cross-Disciplinary Fund, the Science and Technology Development Fund. Zhuosheng is partially supported by the National Natural Science Foundation of China (62406188). This work is also partially supported by the Science and Technology Development Fund, Macau SAR (Grant Nos. FDCT/060/2022/AFJ, FDCT/0070/2022/AMJ), and the Multi-year Research Grant from the University of Macau (Grant No. MYRG-GRG2023-00006-FST-UMDF). 

## **References** 

- [1] Vahdat Abdelzad, Krzysztof Czarnecki, Rick Salay, Taylor Denounden, Sachin Vernekar, and Buu Phan. Detecting out-of-distribution inputs in deep neural networks using an early-layer output. _arXiv preprint arXiv:1910.10307_ , 2019. 

- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- [3] Udit Arora, William Huang, and He He. Types of out-of-distribution texts and how to detect them. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 10687–10701, 2021. 

- [4] Arsenii Ashukha, Alexander Lyzhov, Dmitry Molchanov, and Dmitry Vetrov. Pitfalls of indomain uncertainty estimation and ensembling in deep learning. In _International Conference on Learning Representations_ , 2019. 

- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- [6] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- [7] Taylor Denouden, Rick Salay, Krzysztof Czarnecki, Vahdat Abdelzad, Buu Phan, and Sachin Vernekar. Improving reconstruction autoencoder out-of-distribution detection with mahalanobis distance. _arXiv preprint arXiv:1812.02765_ , 2018. 

11 

- [8] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In _international conference on machine learning_ , pages 1050–1059. PMLR, 2016. 

- [9] Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence embeddings. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 6894–6910, 2021. 

- [10] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In _International Conference on Learning Representations_ , 2020. 

- [11] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In _Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)_ , 2021. 

- [12] Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In _International Conference on Learning Representations_ , 2016. 

- [13] Mohammad Javad Hosseini, Hannaneh Hajishirzi, Oren Etzioni, and Nate Kushman. Learning to solve arithmetic word problems with verb categorization. In _Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 523–533, 2014. 

- [14] Yi Hu, Xiaojuan Tang, Haotong Yang, and Muhan Zhang. Case-based or rule-based: How do transformers do the math? _arXiv preprint arXiv:2402.17709_ , 2024. 

- [15] Rui Huang, Andrew Geng, and Yixuan Li. On the importance of gradients for detecting distributional shifts in the wild. _Advances in Neural Information Processing Systems_ , 34:677– 689, 2021. 

- [16] Wenyu Jiang, Yuxin Ge, Hao Cheng, Mingcai Chen, Shuai Feng, and Chongjun Wang. Read: Aggregating reconstruction error into out-of-distribution detection. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 37, pages 14910–14918, 2023. 

- [17] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. _Advances in neural information processing systems_ , 35:22199–22213, 2022. 

- [18] Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. Mawps: A math word problem repository. In _Proceedings of the 2016 conference of the north american chapter of the association for computational linguistics: human language technologies_ , pages 1152–1157, 2016. 

- [19] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. _Advances in neural information processing systems_ , 30, 2017. 

- [20] Hao Lang, Yinhe Zheng, Yixuan Li, SUN Jian, Fei Huang, and Yongbin Li. A survey on out-of-distribution detection in nlp. _Transactions on Machine Learning Research_ , 2023. 

- [21] Rikard Laxhammar and Göran Falkman. Online learning and sequential anomaly detection in trajectories. _IEEE transactions on pattern analysis and machine intelligence_ , 36(6):1158–1173, 2013. 

- [22] Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. _Advances in neural information processing systems_ , 31, 2018. 

- [23] Shiyu Liang, Yixuan Li, and R Srikant. Enhancing the reliability of out-of-distribution image detection in neural networks. In _International Conference on Learning Representations_ , 2018. 

- [24] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In _Text summarization branches out_ , pages 74–81, 2004. 

12 

- [25] Zi Lin, Jeremiah Zhe Liu, and Jingbo Shang. Towards collaborative neural-symbolic graph semantic parsing via uncertainty. _Findings of the Association for Computational Linguistics: ACL 2022_ , 2022. 

- [26] Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. Energy-based out-of-distribution detection. _Advances in neural information processing systems_ , 33:21464–21475, 2020. 

- [27] Yang Liu, Yuanshun Yao, Jean-Francois Ton, Xiaoying Zhang, Ruocheng Guo, Hao Cheng, Yegor Klochkov, Muhammad Faaiz Taufiq, and Hang Li. Trustworthy llms: a survey and guideline for evaluating large language models’ alignment. In _Socially Responsible Language Modelling Research_ , 2023. 

- [28] Yiding Liu, Kaiqi Zhao, Gao Cong, and Zhifeng Bao. Online anomalous trajectory detection with deep generative sequence modeling. In _2020 IEEE 36th International Conference on Data Engineering (ICDE)_ , pages 949–960. IEEE, 2020. 

- [29] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _International Conference on Learning Representations_ , 2018. 

- [30] Christos Louizos and Max Welling. Multiplicative normalizing flows for variational bayesian neural networks. In _International Conference on Machine Learning_ , pages 2218–2227. PMLR, 2017. 

- [31] Denis Lukovnikov, Sina Daubener, and Asja Fischer. Detecting compositionally out-ofdistribution examples in semantic parsing. In _Findings of the Association for Computational Linguistics: EMNLP 2021_ , pages 591–598, 2021. 

- [32] Andrey Malinin, Neil Band, Yarin Gal, Mark Gales, Alexander Ganshin, German Chesnokov, Alexey Noskov, Andrey Ploskonosov, Liudmila Prokhorenkova, Ivan Provilkov, et al. Shifts: A dataset of real distributional shift across multiple large-scale tasks. In _Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)_ , 2021. 

- [33] Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction. In _International Conference on Learning Representations_ , 2020. 

- [34] Leland McInnes, John Healy, Nathaniel Saul, and Lukas Großberger. Umap: Uniform manifold approximation and projection. _Journal of Open Source Software_ , 3(29):861, 2018. 

- [35] Charles E Metz. Basic principles of roc analysis. In _Seminars in nuclear medicine_ , volume 8, pages 283–298. Elsevier, 1978. 

- [36] Leann Myers and Maria J Sirois. Spearman correlation coefficients, differences between. _Encyclopedia of statistical sciences_ , 12, 2004. 

- [37] Inderjeet Nair and Lu Wang. Midgard: Self-consistency using minimum description length for structured commonsense reasoning, 2024. 

- [38] Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model’s uncertainty? evaluating predictive uncertainty under dataset shift. _Advances in neural information processing systems_ , 32, 2019. 

- [39] Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 2080–2094, 2021. 

- [40] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018. 

- [41] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9, 2019. 

13 

- [42] Jie Ren, Stanislav Fort, Jeremiah Liu, Abhijit Guha Roy, Shreyas Padhy, and Balaji Lakshminarayanan. A simple fix to mahalanobis distance for improving near-ood detection. _arXiv preprint arXiv:2106.09022_ , 2021. 

- [43] Jie Ren, Jiaming Luo, Yao Zhao, Kundan Krishna, Mohammad Saleh, Balaji Lakshminarayanan, and Peter J Liu. Out-of-distribution detection and selective generation for conditional language models. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- [44] Subhro Roy and Dan Roth. Solving general arithmetic word problems. In _Proceedings of the Conference on Empirical Methods in Natural Language Processing_ , pages 1743–1752, 2015. 

- [45] Pranab Kumar Sen. Estimates of the regression coefficient based on kendall’s tau. _Journal of the American statistical association_ , 63(324):1379–1389, 1968. 

- [46] Rupesh K Srivastava, Klaus Greff, and Jürgen Schmidhuber. Training very deep networks. _Advances in neural information processing systems_ , 28, 2015. 

- [47] Lichao Sun, Yue Huang, Haoran Wang, Siyuan Wu, Qihui Zhang, Chujie Gao, Yixin Huang, Wenhan Lyu, Yixuan Zhang, Xiner Li, et al. Trustllm: Trustworthiness in large language models. _arXiv preprint arXiv:2401.05561_ , 2024. 

- [48] Yiyou Sun, Yifei Ming, Xiaojin Zhu, and Yixuan Li. Out-of-distribution detection with deep nearest neighbors. In _International Conference on Machine Learning_ , pages 20827–20840. PMLR, 2022. 

- [49] Jörg Tiedemann. Parallel data, tools and interfaces in opus. In _Proceedings of the Eighth International Conference on Language Resources and Evaluation_ , pages 2214–2218, 2012. 

- [50] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023. 

- [51] Yiming Wang, Pei Zhang, Baosong Yang, Derek F Wong, and Rui Wang. Latent space chain-ofembedding enables output-free llm self-evaluation. _arXiv preprint arXiv:2410.13640_ , 2024. 

- [52] Yiming Wang, Zhuosheng Zhang, and Rui Wang. Element-aware summarization with large language models: Expert-aligned evaluation and chain-of-thought method. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 8640–8665, 2023. 

- [53] Yiming Wang, Zhuosheng Zhang, Pei Zhang, Baosong Yang, and Rui Wang. Meta-reasoning: Semantics-symbol deconstruction for large language models. In _Findings of the Association for Computational Linguistics ACL 2024_ , pages 622–643, 2024. 

- [54] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- [55] Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, and Roger Grosse. Flipout: Efficient pseudoindependent weight perturbations on mini-batches. In _International Conference on Learning Representations_ , 2018. 

- [56] Tim Z Xiao, Aidan Gomez, and Yarin Gal. Wat zei je? detecting out-of-distribution translations with variational transformers. 2020. 

- [57] Ruijie Xu, Zengzhi Wang, Run-Ze Fan, and Pengfei Liu. Benchmarking benchmark leakage in large language models. _arXiv preprint arXiv:2404.18824_ , 2024. 

- [58] Jingkang Yang, Kaiyang Zhou, Yixuan Li, and Ziwei Liu. Generalized out-of-distribution detection: A survey. _arXiv preprint arXiv:2110.11334_ , 2021. 

- [59] Hugh Zhang, Jeff Da, Dean Lee, Vaughn Robinson, Catherine Wu, Will Song, Tiffany Zhao, Pranav Raja, Dylan Slack, Qin Lyu, et al. A careful examination of large language model performance on grade school arithmetic. _arXiv preprint arXiv:2405.00332_ , 2024. 

14 

- [60] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. In _International Conference on Learning Representations_ , 2019. 

- [61] Zhuosheng Zhang, Yao Yao, Aston Zhang, Xiangru Tang, Xinbei Ma, Zhiwei He, Yiming Wang, Mark Gerstein, Rui Wang, Gongshen Liu, et al. Igniting language intelligence: The hitchhiker’s guide from chain-of-thought reasoning to language agents. _arXiv preprint arXiv:2311.11797_ , 2023. 

- [62] Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. _arXiv preprint arXiv:2210.03493_ , 2022. 

15 

## **Appendix** 

## **A Related Work** 

**Data-unavailable OOD Detection: Generic Methods.** In scenarios where OOD data is unavailable. Detection methods are categorized into three main types: (1) Output-based methods assess confidence using predicted probabilities. (2) Ensemble-based methods assess the uncertainty of a collection of supporting models, classical techniques are Monte-Carlo dropout [46, 8] based on Bayesian inference and deep ensemble [19, 31]; (3) Feature-based methods assess the Mahalanobis Distance between OOD samples and the distribution of the ID data feature space, usually considering the input and output spaces [22, 42, 48], occasionally extending to specific hidden layer spaces [1]. Besides, there are some fragmented methods, such as gradient methods [23, 15] and autoencoder reconstruction methods [7, 16]. Still, these methods suffer from optimization and computational complexity with serious performance bottlenecks [58] and thus are not mainstream detection methods. 

**OOD Detection in GLMs.** Relatively few studies have explored OOD detection in GLMs, mainly in semantic parsing [31, 25], speech recognition [33], machine translation [56, 32], summarization [43], and they do not jump out of the frameworks of uncertainty estimation, ensemble-based methods, and embedding-based methods. To our knowledge, **we are the first to study OOD detection on mathematical reasoning, and we have found the failure of traditional algorithms in this scenario.** Mathematical reasoning is an important and difficult research topic in the era of LLMs, and this research is valuable for the scenario expansion of OOD detection algorithms on language models. 

## **B Motivation Visualization Details of Figure 1** 

## **B.1 Projection Setting** 

We use UMAP [34] for projection visualization, which is a nonlinear dimensionality reduction technique for mapping high-dimensional data into low-dimensional spaces. It uses optimization of the local and global structure of the original data to produce a high-quality mapping that can preserve the original data’s local and global characteristics. 

The UMAP algorithm consists of two steps: (i) Calculate the local similarity between each data point and its immediate neighbors to construct a local similarity map. (ii) Map the high-dimensional data to a low-dimensional space by optimizing an objective function that maintains the local and global structure. The hyperparameter “ `n_neighbors` ” in the first step is key, we set it as 10 in our paper. 

## **B.2 More Examples** 

We select the MATH [6] dataset for mathematical reasoning and the OPUS [49] for text generation (translation). In the MATH, we choose four domains: algebra, geometry, number theory, and precalculus; In the OPUS, we also choose four domains: Bible, news, TED, and health. We present examples of inputs and outputs under different domains for mathematical reasoning and translation scenarios in Table 7 and Table 8, respectively. They correspond to the case projection of Figure 1. Obviously, under mathematical reasoning, the outputs under different domains may appear exactly the same, while it is impossible under translation. 

## **C Theoretical Analysis of Section 2.1** 

In this section, we theoretically analyze _Why pattern collapse in the output space leads to a greater likelihood of volatility differences in trajectories under different samples,_ which corresponds to Hypothesis 1 in the main paper. 

## **C.1 Problem Setup** 

**Latent Space Embedding.** For a given sample _si_ , [ _**y** i_ ] _l ∈_ R _[d]_ (1 _≤ l ≤ L_ ) denote the hidden embedding of the _l_ -th layer. We define the embedding coordinate in the latent space as _{_ ( _l,_ [ _**y** i_ ] _l_ ) _}[L] l_ =1[.] 

16 

Table 7: Examples of inputs/outputs from different domains in the mathematical reasoning scenario. 

|Table 7: Examples of inputs/outputs from different domains in the mathematical reasoning scenario.|Table 7: Examples of inputs/outputs from different domains in the mathematical reasoning scenario.|Table 7: Examples of inputs/outputs from different domains in the mathematical reasoning scenario.|
|---|---|---|
|**Input**<br>**Output**|||
|_Domain 1: Algebra_|||
|Suppose that_f_ is a function and_f −_1 is the inverse of_f_. If_f_(1) = 2, _f_(2) = 6, and<br>_f_(3) = 5, then what is_f −_1(_f −_1(6))?<br>1|||
|_Domain 2: Geometry_|||
|Two chords,_AB_and_CD,_meet inside a circle at_P._If_AP_ =_CP_ = 7_,_then what is _BP_<br>_DP_ ?<br>1|||
|_Domain 3: Number Theory_|||
|The product of the proper positive integer factors of_n_can be written as_n_(_ax_+_b_)_/c_, where<br>_x_is the number of positive divisors_n_has,_c_is a positive integer, and the greatest common<br>factor of the three integers_a_,_b_, and_c_is1. What is_a_+_b_+_c_?<br>1|||
|_Domain 4: Precalculus_|||
|Simplify|1<br>1_−_tan2 _x_ +|1<br>1_−_cot2 _x._<br>1|



Table 8: Examples of inputs/outputs from different domains in the translation scenario. 

|**Input**|**Output**|
|---|---|
|_Domain 1: Bible_||
|The earth yielded grass, herbs yielding seed after their|`于是地发生了青草，和结种子的菜蔬，各从其`|
|kind, and trees bearing fruit, with its seed in it, after|`类；并结果子的树木，各从其类；果子都包着`|
|their kind; and God saw that it was good.|`核。神看着是好的。`|
|_Domain 2: News_||
|Supporters say the tunnels would beneft the|`支持者们表示，这两条隧道将使环境受益并帮助`|
|environment and offer Californians a more secure|`加利福尼亚州确保供水更加安全。`|
|water supply.||
|_Domain 3: TED_||
|(Applause) June Cohen: Frank, that was beautiful, so|`（鼓掌）主持：`Frank`刚刚真是太美丽、太感人`|
|touching.|`啦！`|
|_Domain 4: Health_||
|The White House Coronavirus Task Force was|`白宫冠状病毒工作组（`White House Coronavirus|
|established on January 29.|Task Force`）是成立于`2020`年`1`月`29`日的以应对美`|
||`国`COVID-19`疫情的工作组。`|



**Embedding Interpolation.** We assume the _**F** i_ ( _x_ ) = [ _Fi_ 1( _x_ ) _, Fi_ 2( _x_ ) _, ..., Fid_ ( _x_ )] _[⊤]_ : R _→_ R _[d]_ with _d_ independent components fits the _L_ coordinates, representing a continuous learning trajectory for _si_ , so [ _**y** i_ ] _l_ = _**F** i_ ( _l_ ). _**F** i_ ( _x_ ) is taken from the _functional space_ _**X**_ , so _{_ _**F** i_ ( _l_ ) _}[L] l_ =1[are all independent] variables. We constrain each component function of this _**F** i_ ( _x_ ) to satisfy the _m_ -order ( _m ≥_ 3) derivability property. Under this setting, the definition of dimension-independent trajectory volatility (Eq. 2) equates to _**V**_ ( _si_ ) = [ _V_ 1( _si_ ) _, V_ 2( _si_ ) _, ..., VL_ ( _si_ )] _[⊤] L−_ 1 (9) = _L_[1] � [ _|Fi_ 1( _l_ ) _− Fi_ 1( _l −_ 1) _|, |Fi_ 2( _l_ ) _− Fi_ 2( _l −_ 1) _|, ..., |Fid_ ( _l_ ) _− Fid_ ( _l −_ 1) _|_ ] _[⊤] l_ =1 

**Modeling.** We observe the “ _pattern collapse_ ” phenomenon in the output space in Figure 1. We abstract this phenomenon in Figure 2, which compares the trajectory trend between different samples in the mathematical reasoning and text generation scenarios. 

17 

We specify that [ _**y** i_ ] _L_ = _**F** i_ ( _L_ ) _∼N_ ( _**c** ,_ Σ[2] ), where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0018-01.png)


According to the pattern collapse property under different tasks, we can constraint the endpoint embedding _**F** i_ ( _L_ ) in the output space: 

- For mathematical reasoning with pattern collapse, Σ _→_ _**O**_ , so we approximate that _**F** i_ ( _L_ ) _≡_ _**c**_ ; 

- For text generation without pattern collapse, Σ _̸_ = _**O**_ , so _**F** i_ ( _L_ ) _̸≡_ _**c**_ . 

With such constraints, we model the main theorem: 

**Theorem C.1 (Main Theorem)** _For different samples si and sj, the likelihood of variations in trajectory volatility under mathematical reasoning scenarios is higher than that under text generation scenarios, which means:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0018-07.png)


_where U_ (R _[d]_ ) _denotes the uniform distribution defined in d-dimensional real number space._ 

## **C.2 Prelinimary** 

Next, we move on to the formal proofs. We begin with some propositions and lemmas that will be useful in the main theorem. 

**Proposition C.1 (Lagrange Remainder Term)** _In the Taylor Expansion expression_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0018-12.png)


_Proof._ We consider the case of _a < x_ with _a > x_ identically. We use the Fundamental Theorem of Calculus (FTC) for the most basic expansion of _f_ ( _x_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0018-14.png)


Continue to use the FTC to expand the derivatives in integrals: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0018-16.png)


Therefore, the generalized remainder is known as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0018-18.png)


18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0019-00.png)


According to the Lagrange’s Mean Value Theorem, there must exist a number _c ∈_ [ _a, x_ ] with 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0019-02.png)


this gives us 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0019-04.png)


and the equation on the left side of the proposition is proved completely. The inequality on the right-hand side is clearly established by the Lagrange’s Mean Value Theorem. □ 

**Lemma C.1 (Error Bound for the Midpoint Rule)** _Suppose that f_ ( _x_ ) _is a m-th_ ( _m ≥_ 2) _order differentiable function on the interval_ ( _−∞,_ + _∞_ ) _, and K_ = sup ���� dd2 _xf_[2] ��� : _x ∈_ [ _a, b_ ]� _∈_ R _, then_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0019-07.png)


_Proof._ We do the first order Taylor Expansion for _f_ ( _x_ ) at the midpoint _x_ = _[a]_[+] 2 _[b]_ of the interval [ _a, b_ ]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0019-09.png)


Following Theorem C.1, we do the inequality scaling as below: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0019-11.png)


**Lemma C.2 (Differential-Integral Error Order Estimation)** _Suppose that f_ ( _x_ ) _is a m-th_ ( _m ≥_ 3) _order differentiable function on the interval_ ( _−∞,_ + _∞_ ) _and Ki_ = sup ���� dd _ixf[i]_ ��� : _x ∈_ ( _−∞,_ + _∞_ )� _<_ + _∞_ ( _i_ = 1 _,_ 2 _, · · · , m_ ) _, then on the closed interval_ [ _a, b_ ] _, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0019-13.png)


19 

_Proof._ For each sub-interval � _a_ + _[b][−] L[a]_[(] _[i][ −]_[1)] _[, a]_[ +] _[b][−] L[a][i]_ �, we perform a Taylor Expansion at the midpoint _x_ = _a_ + _[b][−] L[a]_[(] _[i][ −]_[1] 2[)][ to obtain the following expression:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0020-01.png)


Following Lemma C.1, we do the inequality scaling for the first-order derivative summation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0020-03.png)


## **C.3 Main Theorem Proof** 

Finally, we formally prove our main theorem C.1 in this sub-section, thereby verifying the Hypotheses 1 in the main paper. Since the _d_ components of _**V**_ ( _si_ ) _−_ _**V**_ ( _sj_ ) are independent of each other, we only consider the _d_ -th dimensional component and the rest of the components _Vd_ ( _si_ ) _− Vd_ ( _sj_ ) can be proved in the same way. 

_Proof._ According to Lemma C.2, we can approximate _Vd_ ( _si_ ) _− Vd_ ( _sj_ ) as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0020-07.png)


Now we want to remove the absolute value of the integrated function. Due to the continuity of the first-order derivative, there must be several zero points that are not extreme points, dividing the domain of function definition into several open intervals with alternating constant positive and negative function values. We first define such a set of zeros on the domain of definition _D_ = [1 _, L_ ]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0020-09.png)


20 

For all zero points _it ∈Xi_ or _jt ∈Xj_ , they must satisfy the following properties to ensure that they are not extreme points: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0021-01.png)


and _jt_ is the same as _it_ . Therefore, in the sub-interval (1 _, i_ 1) _,_ ( _i_ 1 _, i_ 2) _, · · · ,_ ( _ip, L_ ), _Fid_ ( _x_ ) is alternately constant positive and constant negative, and the same as _Fjd_ ( _x_ ). 

We assume that the first interval (1 _, i_ 1) and (1 _, j_ 1) are constant positive intervals (the same can be proven for constant negative intervals). In this setting, we can continue to simplify Eq.22: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0021-04.png)


Since _Fid_ ( _x_ ) and _Fjd_ ( _x_ ) are taken from the functional space, _{Fid_ (1) _, {Fid_ ( _ik_ ) _}[p] k_ =1 _[}]_[and] _{Fjd_ (1) _, {Fjd_ ( _jk_ ) _}[q] k_ =1 _[}]_[ can be seen as independent variables.][We let] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0021-06.png)


then we can rewrite Eq.25 to the matrix form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0021-08.png)


where _**A**_ is the coefficient matrix, and _**x**_ is the unknown variable vector: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0021-10.png)


We let _ip ≡ jq_ (mod 2), and return to the condition in the mathematical expectation on both sides of the inequality in the main theorem C.1 for the following categorical discussion: 

- If _Fid_ ( _L_ ) = _Fjd_ ( _L_ ) = _cd_ , then _**c**_ = **0** , so _Vd_ ( _si_ ) _− Vd_ ( _sj_ ) = 0 = _⇒_ _**Ax**_ = **0** ; 

- If _Fid_ ( _L_ ) _, Fjd_ ( _L_ ) _∼N_ ( _cd, δd_[2][)][, then] _**[ c]**[ ̸≡]_ **[0]**[, so] _[ V][d]_[(] _[s][i]_[)] _[ −][V][d]_[(] _[s][j]_[)] _[ ̸]_[= 0 =] _[⇒]_ _**[Ax]**_[ =] _[ −]_ _**[c]**_[.] 

We denote _N_ ( _A_ ) as the solution space of _{_ _**x** |_ _**Ax**_ = **0** _}_ and _P_ ( _A_ ) as the solution space of _{_ _**x** |_ _**Ax**_ = _−_ _**c** }_ . For _x ∈ P_ ( _A_ ), _x_ = _xp_ or _x_ = _xp_ + _xn_ where _xp_ is the special solution and _xn_ is the solution in zero space, _i.e._ , _xn ∈ N_ ( _A_ ). Since rank( _A_ ) = 1 _< p_ + _q_ + 2, the special solution _xp_ exists, so the size of the solution space _P_ ( _A_ ) is larger than _N_ ( _A_ ). In this case, when the variable _**x**_ is sampled from the real number space, its probability of being in _N_ ( _A_ ) is smaller, namely its probability of not being in _N_ ( _A_ ) is larger. This is exactly equivalent to the form of the proof of Main Theorem C.1, so the proof is complete. □ 

21 

## **C.4 Extended Conclusion** 

During the proof of the main theorem, we can unlock a hidden conclusion due to the embedding interpolation: _GLMs with a larger number of hidden layers may achieve more stable detection performance,_ i.e. _, discrepancies in embedding volatility of different samples will be more obvious._ 

In Lemma C.2, we have proved the upper bound of differential-integral error order estimation is the equivalent infinitesimals of 1 _/L_ . Actually, when _L →_ + _∞_ , we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0022-03.png)


which is clear according to the definition of Riemann Sum. This means that when the value of _L_ increases, the differential-integral approximation error will be reduced, so the conclusion of the main theorem will be more accurate. In our experiments, we use GPT2-XL (48 layers) and Llama2-7B (32 layers) as training backbones and find that the average performance of GPT2-XL is higher and more stable than that of Llama2-7B, while the number of layers of GPT2-XL is 1.5 times that of Llama2-7B. This further validates the correctness of our extended conclusions. 

## **D Algorithm: TV Score Computation** 

The pseudo-code of our TV Score computation pipeline is shown in Algorithm 1. 

**Algorithm 1** Trajectory Volatility (TV) Score Computation 

**Input:** _L_ : The number of hidden layers _N_ : The size of ID dataset _k_ : the smoothing differential order of TV score _{_ _**y** l}_ 1 _≤l≤L_ : the average hidden embedding of the OOD sample in each layer _{_ [ ˆ _**y** l_ ] _i}_ 1 _≤l≤L,_ 1 _≤i≤N_ : the average hidden embedding of all _N_ ID samples in each layer 1: **for** _l ←_ 1 **to** _L_ **do** 2: Fitting ID samples _{_ [ ˆ _**y** l_ ] _i}_ 1 _≤l≤L,_ 1 _≤i≤N_ to Gaussian distribution _Gl_ = _N_ ( _**µ** l,_ **Σ** _l_ ) 3: _f_ ( _**y** l_ ) _←_ ( _**y** l −_ _**µ** l_ ) _[⊤]_ (Σ _l_ ) _[−]_[1] ( _**y** l −_ _**µ** l_ ) 4: **end for** 5: **for** _i →_ 1 **to** _k_ **do** 6: **for** _l ←_ 1 **to** _L_ **do** 7: ∆[(] _[i]_[)] _Gl_ = _N_ ( _**µ** l_[(] _[i]_[)] _[,]_ **[ Σ]**[(] _l[i]_[)][)] _[ ←N]_[(][�] _[i] t_ =0[(] _[−]_[1)] _[i]_[+] _[t]_[C] _i[t]_ _**[µ]**[l]_[+] _[i][,]_[�] _[i] t_ =0[C] _i[t]_ **[Σ]** _[l]_[+] _[i]_[)] 8: **end for** _−_ 1 9: _f_[(] _[i]_[)] ( _**y** l_ ) _←_ � _**y** l_[(] _[i]_[)] _−_ _**µ** l_[(] _[i]_[)][�] _[⊤]_[�] Σ[(] _l[i]_[)] � � _**y** l_[(] _[i]_[)] _−_ _**µ** l_[(] _[i]_[)][�] 10: **end for** 11: **for** _i →_ 1 **to** _n_ **do** 12: TV _i ←_ `average` � _f_[(] _[i]_[)] ( _**y** l_ )�1 _≤l≤L_ 13: **end for Output:** _{_ TV _i}_ 1 _≤i≤k_ 

As for the computational complexity, once we have all _**yl**_ , our TV score requires only scalar addition and multiplication. During the ID distribution fitting phase, both operations are _O_ ( _Ldn_ ), where _n_ is the dataset size. During the score computation phase, both operations are _O_ ( _Ldk_ ). In practice, the computation time is measured in milliseconds. 

## **E Experimental Setting Details** 

## **E.1 Basic Information of Dataset** 

For the ID dataset, we use the MultiArith [44], which consists of Math Word Problems on arithmetic reasoning. For the OOD datasets, we intuitively introduce two types of detection scenarios following [43]: **(i) Far-shift OOD** setting, we select the MATH [11] as the OOD data, which spans across 

22 

distinct mathematical domains encompassing algebra, geometry, counting and probability, number theory, and precalculus. It contains college difficulty level math problems, whereas MultiArith has only elementary school difficulty, and thus can be considered as sourced from far-different distributions; (ii) **Near-shift OOD** setting, we select five arithmetic reasoning datasets as the OOD data: GSM8K [6], SVAMP [39], AddSub [13], SingleEq [18], and SingleOp [18], they all consist of Math Word Problems like the MultiArith but require different reasoning hops and knowledge points for solving the problems, and thus can be considered as sourced from near-different distributions. In addition, we present the data sizes and examples of all ID and OOD datasets, as shown in Table 9. 

## **E.2 ID Dataset Split** 

For the ID dataset MultiArith, we find that every 100 consecutive samples show a similar quadratic operation pattern ( _e.g._ , samples with id 0-100 are a mixture of addition and subtraction, and id 100-200 are a mixture of addition and multiplication). Therefore, we divide it into 6 subsets (6*100) in order. In each subset, we take the first 60 as training samples and the last 40 as test samples. 

## **E.3 Training Implementation** 

We train Llama2-7B [50] and GPT2-XL [5] models on the training split of MultiArith. Llama2-7B is trained with AdamW optimizer [29] for 10K steps and 8 batch sizes in 4-card RTX 3090 (2 per card). The learning rate is set to 1e-5, the warmup step to 10, and the maximum gradient normalization to 0.3. GPT2-XL is trained for 3K steps and 128 batch sizes in a single RTX 3090, and other configurations are the same as Llama2-7B. 

## **E.4 OOD Dataset Selection Rationality** 

In this part, we examine the rationality of the OOD data selection, ensuring that the OOD data distribution significantly differs from the pre-trained data distribution and has not been fully learned during the pre-training phase. Some research [57, 59] have confirmed the absence of data leakage in Llama2 for MATH and GSM8K datasets, we still conduct experiments and analyses to ensure this. 

For GPT2-XL, We can determine this from the time dimension. GPT-2 was released in 2019, but the MATH and GSM8k datasets were released in 2021, so they are unlikely to appear in the pre-training data. However, for Llama2-7B, due to the closed-source data, we cannot confirm which data the model used in the pre-training phase, so we cannot fully determine whether the selected dataset was OOD for the model from a data perspective. 

Therefore, we argue that a dataset can be considered OOD when it is beyond the capability of the base model as claimed by prior work [47, 27]. We test all ten datasets we select as the OOD data in the pre-trained Llama2-7B and GPT2-XL model, and Table 10 shows the results. We find that GPT2-XL cannot handle any of the ten mathematical reasoning tasks, and Llama2-7B performs with very low accuracy. Therefore, from a capability standpoint, we can ensure that these datasets are OOD for these two GLMs. 

## **E.5 Baseline** 

Let _**x**_ represent the input sequence and _**y**_ the output sequence, with lengths denoted as _nx_ and _ny_ respectively. In addition, we assume that _p_ ( _·_ ; _**θ**_ ) represents the GLM parameterized by _**θ**_ that has been trained in ID dataset _D_ , outputting a sequence of softmax probabilities. We compare some training-free baseline methods as below: 

- Maximum Softmax Probability [12]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0023-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0023-13.png)


where _q_ ( _**θ**_ ) _≈ p_ ( _**θ** |D_ ). 

23 

Table 9: ID and OOD datasets used in this paper. 

|**In-Distribution Dataset**||
|---|---|
|**MultiArith (Data Size: 600)**||
|Q: Kaleb was collecting cans for recycling. On Saturday he flled 5 bags up and on Sunday he flled 5 more||
|bags. If each bag had 4 cans in it, how many cans did he pick up total?||
|A: 40.0||
|**Far Shift Out-of-Distribution Dataset**|**Near Shift Out-of-Distribution Dataset**|
|**MATH-Algebra (Data Size: 1187)**|**GSM8K (Data Size: 1318)**|
|Q: How many real numbers are not in the domain of|Q: Judy teaches 5 dance classes, every day, on the|
|the function|weekdays and 8 classes on Saturday. If each class has|
|_f_(_x_) =<br>1<br>_x −_64 +<br>1<br>_x_2 _−_64 +<br>1<br>_x_3 _−_64 ?|15 students and she charges $15.00 per student, how<br>much money does she make in 1 week?|
|A: 4|A: 7425|
|**MATH-Geometry (Data Size: 479)**|**SVAMP (Data Size: 1000)**|
|Q: Suppose we are given seven points that are equally|Q: A mailman has to give 38 pieces of junk mail to|
|spaced around a circle. If_P_,_Q_, and_R_are chosen to|each of the 78 blocks. If there are 19 houses on a|
|be any three of these points, then how many different|block. How many pieces of junk mail should he give|
|possible values are there for_m_∠_PQR_?|each house?|
|A: 5|A: 2.0|
|**MATH-Counting and Probability (Data Size: 474)**|**AddSub (Data Size: 395)**|
|Q: Amy’s grandmother gave her 3 identical chocolate|Q: While taking inventory at her pastry shop, Kelly|
|chip cookies and 4 identical sugar cookies. In how|realizes that she had 0.4 box of baking powder|
|many different orders can Amy eat the cookies such|yesterday, but the supply is now down to 0.3 box. How|
|that either she eats a chocolate chip cookie frst, she|much more baking powder did Kelly have yesterday?|
|eats a chocolate chip cookie last, or both?||
|A: 25|A: 0.1|
|**MATH-Number Theory (Data Size: 540)**|**SingleEq (Data Size: 508)**|
|Q: Notice that|Q: Fred had 7 dimes in his bank. His sister borrowed|
|31_·_37 = 1147_._|3 of his dimes. How many dimes does Fred have now?|
|Find some integer_n_with0_≤n <_2293such that||
|31_n ≡_3<br>(mod 2293)_._||
|A: 222|A: 4.0|
|**MATH-Precalculus (Data Size: 546)**|**SingleOp (Data Size: 562)**|
|Q: Let **a** =<br><br><br>1<br>3<br>0<br><br>_,_ **b** =<br><br><br>_−_11<br>5<br>2<br><br>_,_ and **c** =|Q: Pamela starts with 30 bottle caps. Jean takes 26<br>away. How many bottle caps does Pamela end with?|
|<br><br>1 +<br>~~_√_~~<br>5<br>4<br>_−_5<br><br>_._Find_k_if the vectors**a**+**b**+**c**and||
|3(**b**_×_**c**)_−_8(**c**_×_**a**) +_k_(**a**_×_**b**)|A: 4.0|
|are orthogonal.||
|A: 5||



24 

Table 10: Accuracies of all datasets we select as the OOD data in pre-trained GLMs. 

|Far-shift OOD Setting|Far-shift OOD Setting|Near-shift OOD Setting|Near-shift OOD Setting|
|---|---|---|---|
|||||
|_Dataset_|**Llama2-7B**<br>**GPT2-XL**<br>_Dataset_<br>Accuracy of pre-trained model||**Llama2-7B**<br>**GPT2-XL**<br>Accuracy of pre-trained model|
|||||
|Algebra<br>Geometry<br>Cnt.&Prob<br>Num.Theory<br>Precalculus|6 / 1187<br>0 / 1187<br>2 / 479<br>0 / 479<br>4 / 474<br>0 / 474<br>0 / 540<br>0 / 540<br>1 / 546<br>0 / 540|GSM8K<br>SVAMP<br>AddSub<br>SingleEq<br>SingleOp|0 / 1318<br>0 / 1318<br>8 / 1000<br>0 / 1000<br>13 / 395<br>0 / 1000<br>5 / 395<br>0 / 508<br>7 / 508<br>0 / 508|



- Sequence Perplexity [43]: 

- Input Embedding [43]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0025-04.png)


- where _**µ** x_ and **Σ** _x_ represent the mean and variance of the Gaussian distribution associated with the first-layer hidden state. 

- Output Embedding [43]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0025-07.png)


where _**µ** y_ and **Σ** _y_ represent the mean and variance of the Gaussian distribution associated with the final-layer hidden state. 

## **F Supplementary Experimental Results** 

## **F.1 Hyperparameter Ablation: Smoothing Order k** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0025-11.png)


**----- Start of picture text -----**<br>
Algebra Cnt.&Prob Precalculus GSM8K AddSub SingleOp<br>Geometry Num.Theory SVAMP SingleEq<br>100 100<br>80 80<br>60 60<br>40 40<br>20 20<br>0 0<br>0 1 2 3 4 5 0 1 2 3 4 5<br>100 100<br>80 80<br>60 60<br>40 40<br>20 20<br>0 0<br>0 1 2 3 4 5 0 1 2 3 4 5<br>Smoothing Order k<br>AUROC<br>FPR95<br>**----- End of picture text -----**<br>


Figure 4: **Smoothing order** _k_ **analysis** : _k_ ranges from 0 _−_ 5 ( _k_ = 0 corresponds to the original TV Score). The upper part is for the OOD detection scenario and the lower part is for the OOD quality estimation scenario; the left part is for the far-shift OOD datasets and the right part is for the near-shift OOD datasets. 

In the main experiments, we found that differential smoothing is not as effective as the basic TV score, with excellent results occurring on only a few datasets. Figure 4 visualizes the results for the 

25 

OOD detection scenario with no smoothing ( _k_ = 0) and smoothing order _k_ 1-5. We find that there is a significant effect of differential smoothing in two cases: (1) very good performance without smoothing, _e.g._ , the precalculus dataset, where the AUROC results are almost close to 100%; and (2) significantly poor performance without smoothing, _e.g._ , the AddSub dataset, where the FPR95 results on all other near-shift OOD datasets are 20 below, the results on this dataset are more than 80, when differential smoothing helps to eliminate some of the noise features and helps better detection. 

In addition, for the case of _k >_ 0, the peak performance basically occurs in the case of _k_ = 1 or 2, which indicates that when _k_ is too large, the phenomenon of over-smoothing tends to occur. Too much useful feature information is lost, leading to a decrease in detection accuracy. 

## **F.2 OOD Detection** 

Results of each dataset are shown in Table 11 (Llama2-7B) and Table 12 (GPT2-XL). 

Table 11: AUROC and FPR95 results in **Llama2-7B** of the **Offline Detection scenario** ( _p_ -value > 0 _._ 05 are grayed out). Underline and **bold** denote SOTA among all baselines and all methods. 

|||||||**Far-shift OO**|**D Setting**||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|_Method_<br>_Dataset_|Alg|ebra|Geo|metry<br>|Counting an|d Probability|Numbe|r Theory|Preca|lculus|_Ave_|_rage_|
|||||||**AUROC**_↑_/|**FPR95**_↓_||||||
|Max Softmax Prob. [12]<br>Monte-Carlo Dropout[8]<br>Perplexity[3]<br>Input Embedding[43]<br>Output Embedding[43]|79.97_±_1.60 <br>72.12_±_1.68 <br>84.17_±_1.43<br><br>81.51_±_1.09 <br>76.95_±_1.54|/ 80.97_±_4.90<br>/ 85.43_±_6.60<br>/52.75_±_5.13<br>/ 62.84_±_4.56<br>/ 74.92_±_3.02|82.60_±_1.25 <br>74.23_±_1.74 <br>88.67_±_1.36<br>|/ 82.57_±_4.62<br>/ 81.17_±_6.28<br>/53.32_±_5.38<br>/ 69.66_±_2.07<br>/ 80.97_±_2.23|63.87_±_1.64 <br>55.64_±_2.87 <br>77.42_±_1.97<br>|/ 96.19_±_1.57<br>/ 97.15_±_1.34<br>/71.28_±_3.90<br>/ 88.35_±_1.72<br>/ 88.45_±_1.58|76.20_±_1.46 <br>67.53_±_2.08 <br>83.88_±_2.13 <br>84.42_±_0.85<br>|/ 85.20_±_2.69<br>/ 89.09_±_5.41<br>/ 68.90_±_4.22<br>/54.80_±_7.25<br>/ 80.18_±_1.71|90.68_±_0.94 <br>73.61_±_2.66 <br>94.05_±_0.43<br>|/ 62.27_±_4.03<br>/ 82.36_±_4.78<br>/19.03_±_3.17<br>/ 63.71_±_2.87<br>/ 51.51_±_2.27|78.66_±_1.38 <br>68.63_±_2.21 <br>85.64_±_1.46<br><br>75.89_±_1.03 <br>74.86_±_1.39|/ 81.44_±_3.56<br>/ 87.04_±_4.88<br>/53.06_±_4.36<br>/ 67.87_±_3.69<br>/ 75.21_±_2.16|
||||75.41_±_0.97 <br>78.72_±_1.31||62.53_±_1.30 <br>61.43_±_1.40||||75.59_±_0.92 <br>86.97_±_1.32||||
||||||||70.23_±_1.36||||||
|TV score (Ours)|**98.87**_±_**0.16**|/**4.67**_±_**1.41**|**99.03**_±_**0.09**|/**3.70**_±_**0.42**|**97.70**_±_**0.15**|/**8.83**_±_**1.33**|**98.43**_±_**0.13**|/**7.37**_±_**1.46**|99.78_±_0.02|/ 1.47_±_0.20|**98.76**_±_**0.11**|/**5.21**_±_**0.98**|
|w/ DiSmo (Ours)|94.71_±_0.93|/ 39.65_±_7.28|94.08_±_0.80|/ 50.52_±_6.14|83.08_±_1.28|/ 80.07_±_1.57|94.57_±_0.75|/ 37.74_±_7.58|**99.82**_±_**0.02**|/**1.11**_±_**0.19**|93.25_±_0.76|/ 41.82_±_4.69|
|∆(**bold**-underline<br>)|+14.70|/ -48.08|+10.36|/ -49.62|+20.28|/ -62.45|+14.01|/ -47.43|+5.77 /|-17.92|+13.12|/ -47.85|
|||||||**Near-shift OO**|**D Setting**||||||
|_Method_<br>_Dataset_|GS|M8K|SVA|MP|Add|Sub|Sing|leEq|Sing|leOp|_Ave_|_rage_|
|||||||**AUROC**_↑_/|**FPR95**_↓_||||||
|Max Softmax Prob. [12]<br>Monte-Carlo Dropout[8]<br>Perplexity[3]<br>Input Embedding[43]<br>Output Embedding[43]|53.08_±_1.67<br> <br>48.87_±_2.43 <br>52.24_±_2.57 <br>45.68_±_1.50 <br>35.39_±_1.24|/ 94.07_±_1.86<br>/ 96.78_±_1.21<br>/ 95.56_±_1.21<br>/ 95.05_±_1.17<br>/91.22_±_1.27|56.56_±_1.53 <br>44.90_±_2.76 <br>55.12_±_1.80 <br>60.92_±_1.34<br>|/ 90.06_±_2.39<br>/ 92.33_±_2.09<br>/ 89.24_±_2.09<br>/86.97_±_4.13<br>/ 90.41_±_1.61|63.31_±_1.88 <br>57.34_±_1.57 <br>62.88_±_1.76 <br>66.28_±_0.92<br>|/ 87.36_±_2.42<br>/ 89.15_±_2.34<br>/ 80.96_±_2.34<br>/76.09_±_2.93<br>/ 77.12_±_3.11|66.68_±_1.32 <br>54.09_±_2.42 <br>67.14_±_1.34<br>|/ 86.15_±_2.61<br>/ 90.07_±_1.96<br>/81.38_±_1.96<br>/ 87.36_±_1.89<br>/ 86.69_±_0.91|61.07_±_1.30 <br>56.46_±_1.85 <br>59.39_±_1.98 <br>67.60_±_1.20<br>|/ 86.91_±_2.76<br>/ 91.21_±_1.83<br>/ 83.30_±_1.83<br>/77.78_±_2.53<br>/ 86.87_±_1.04|60.14_±_1.54 <br>52.33_±_2.21 <br>59.35_±_1.89 <br>60.33_±_1.37<br><br>44.50_±_1.06|/ 88.91_±_2.41<br>/ 91.92_±_1.89<br>/ 86.09_±_1.89<br>/84.65_±_2.53<br>/ 86.46_±_1.59|
||||||||61.18_±_1.22 <br>43.70_±_1.02||||||
||||36.77_±_1.14||63.08_±_0.92||||43.55_±_0.99||||
|TV score (Ours)|**94.88**_±_**0.25**|/**14.22**_±_**1.64**|**94.51**_±_**0.20**|/**12.89**_±_**1.11**|85.84_±_1.06|/ 82.63_±_2.22|**93.97**_±_**0.24**|/**17.39**_±_**1.09**|**94.00**_±_**0.20**|/**14.61**_±_**0.85**|**92.64**_±_**0.39  **|**/ 28.39**_±_**1.38**|
|w/ DiSmo (Ours)|55.21_±_1.91|/ 95.71_±_0.88|38.24_±_1.53|/ 87.06_±_0.94|**87.06**_±_**0.94**|/**71.02**_±_**4.33**|56.66_±_1.34|/ 93.76_±_1.28|47.46_±_1.32|/ 92.48_±_1.13|56.99_±_1.41|/ 88.01_±_1.71|
|∆(**bold**-underline<br>)|+41.80|/ -77.00|+33.59|/ -74.08|+20.78|/ -5.07|+26.83|/ -63.99|+26.40|/ -63.17|+32.31|/ -56.26|



Table 12: AUROC and FPR95 results in **GPT2-XL** of the **Offline Detection scenario** ( _p_ -value > 0 _._ 05 are grayed out). Underline and **bold** denote SOTA among all baselines and all methods. 

|||||||**Far-shift OO**|**D Setting**||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|_Method_<br>_Dataset_|Alg|ebra|Geo|metry<br>|Counting an|d Probability|Number|Theory|Precal|culus|_Ave_|_rage_|
|||||||**AUROC**_↑_/|**FPR95**_↓_||||||
|Max Softmax Prob. [12]<br>Monte-Carlo Dropout[8]<br>Perplexity[3]<br>Input Embedding[43]<br>Output Embedding[43]|72.13_±_1.42 <br>62.41_±_2.10 <br>84.24_±_1.01 <br>86.23_±_0.78<br><br>78.13_±_1.14|/78.35_±_2.35<br>/ 88.02_±_1.79<br>/ 63.78_±_2.38<br>/46.12_±_1.45<br>/ 64.46_±_2.78|64.09_±_1.78/<br>66.63_±_1.82/<br>82.17_±_0.89/<br>83.20_±_0.94<br>/|87.42_±_1.24<br>82.49_±_1.12<br>67.90_±_2.45<br>53.29_±_2.06<br>68.82_±_3.26|68.45_±_1.56/<br>59.32_±_2.03/<br>79.12_±_1.20/<br>79.58_±_1.43<br>/|82.35_±_1.57<br>92.34_±_1.54<br>72.03_±_1.23<br>60.45_±_2.95<br> 84.63_±_4.24|69.37_±_1.43/<br>63.88_±_1.92/<br>72.40_±_1.37/<br>89.44_±_0.64<br>/|82.09_±_1.90<br>86.71_±_1.95<br>71.65_±_1.45<br> 52.49_±_1.87<br>56.23_±_4.05|78.67_±_0.89/<br>70.19_±_1.47/<br>86.19_±_0.75/<br>92.86_±_0.39<br>/|61.23_±_3.02<br>73.90_±_1.87<br>47.27_±_2.98<br>34.32_±_2.16<br>54.08_±_2.79|70.54_±_1.42 <br>66.18_±_1.87 <br>80.82_±_1.04 <br>86.26_±_0.84<br>/<br>77.95_±_1.16|/ 78.29_±_2.02<br>/ 84.69_±_1.65<br>/ 64.53_±_2.10<br>49.33_±_2.10<br>/ 65.64_±_3.42|
||||77.98_±_1.17/||71.07_±_1.67/||80.25_±_0.97/||82.34_±_0.86/||||
|TV score (Ours)|**98.35**_±_**0.07**|/**6.24**_±_**0.24**|**97.27**_±_**0.11**|/**9.23**_±_**0.35**|85.52_±_0.12/|52.51_±_1.78|92.96_±_0.06/|29.86_±_1.26|93.24_±_0.04/|22.68_±_1.14|93.47_±_0.08|/ 24.10_±_0.95|
|w/ DiSmo (Ours)|93.68_±_0.19|/ 23.24_±_1.45|94.39_±_0.15/|12.17_±_0.79|**95.83**_±_**0.13**|/**9.86**_±_**0.46**|**99.17**_±_**0.04**|/**2.42**_±_**0.22**|**99.62**_±_**0.02**|/**1.75**_±_**0.13**|**96.54**_±_**0.11**|/**9.89**_±_**0.61**|
|∆(**bold**-underline<br>)|+12.12|/ -39.88|+14.07|/ -44.06|+16.25|/ -50.59|+9.27 /|-50.07|+6.24 /|-32.57|+10.28|/ -39.44|
|||||||**Near-shift OO**|**D Setting**||||||
|_Method_<br>_Dataset_|GS|M8K|SVA|MP|Add|Sub|Sing|leEq|Sing|leOp|_Ave_|_rage_|
|||||||**AUROC**_↑_/|**FPR95**_↓_||||||
|Max Softmax Prob. [12]<br>Monte-Carlo Dropout[8]<br>Perplexity[3]<br>Input Embedding[43]<br>Output Embedding[43]|54.06_±_1.58 <br>63.44_±_2.08 <br>72.89_±_1.56 <br>87.58_±_1.14<br><br>82.05_±_1.36|/ 92.45_±_2.37<br>/ 79.85_±_3.35<br>/ 74.63_±_1.52<br>/48.45_±_2.06<br>/ 60.44_±_3.96|65.50_±_1.43/<br>62.13_±_1.65/<br>70.79_±_1.38/<br>84.81_±_1.20<br>/|73.62_±_4.18<br>77.28_±_3.21<br>79.65_±_1.21<br>62.75_±_5.68<br> 68.43_±_4.38|70.60_±_0.92/<br>67.97_±_1.62/<br>67.50_±_1.04/<br>80.34_±_0.64<br>/|77.56_±_2.76<br>71.42_±_2.88<br>87.76_±_0.92<br>46.62_±_2.17<br>66.72_±_1.77|79.96_±_0.87/<br>71.29_±_1.43/<br>87.14_±_0.57<br>/|57.38_±_1.95<br>69.22_±_2.41<br> 43.09_±_1.23<br>49.05_±_3.24<br>70.23_±_1.87|65.47_±_1.18/<br>52.89_±_1.80/<br>70.40_±_1.06/<br>81.91_±_0.93<br>/|80.37_±_2.02<br>92.78_±_0.67<br>76.80_±_1.45<br>57.65_±_2.67<br>57.68_±_1.60|67.12_±_1.20 <br>63.54_±_1.72 <br>73.74_±_1.12 <br>83.22_±_0.88<br><br>79.28_±_1.24|/ 76.27_±_2.66<br>/ 78.08_±_2.50<br>/ 72.39_±_1.27<br>/52.90_±_3.16<br>/ 64.70_±_2.72|
||||||||81.44_±_0.49/<br>80.00_±_1.02/||||||
||||81.42_±_2.02/||72.98_±_0.85/||||79.94_±_0.94/||||
|TV score (Ours)|**98.26**_±_**0.06**|/**1.78**_±_**0.04**|**98.99**_±_**0.02**|/**1.13**_±_**0.03**|80.56_±_0.82/|52.23_±_1.29|97.35_±_0.23/|13.91_±_0.45|99.12_±_0.02|/ 0.07_±_0.01|**94.86**_±_**0.23**|/ 13.82_±_0.36|
|w/ DiSmo (Ours)|97.62_±_0.21|/ 3.58_±_0.16|92.19_±_0.44/|13.07_±_1.68|**83.35**_±_**0.57**/|**41.70**_±_**1.29**|**97.83**_±_**0.03**|/**9.94**_±_**0.32**|**99.94**_±_**0.02**|/**0.03**_±_**0.01**|94.19_±_0.25|/**13.66**_±_**0.69**|
|∆(**bold**-underline<br>)|+10.68|/ -46.67|+14.18|/ -61.62|+3.01|/ -4.92|+10.69|/ -33.15|+18.03|/ -57.62|+11.64|/ -39.24|



26 

## **F.3 Beyond Detection: OOD Quality Estimation** 

Results of each dataset are shown in Table 13 (Llama2-7B) and Table 14 (GPT2-XL). 

Table 13: **OOD Quality Estimation (Llama2-7B)** : Kendall’s _τ_ and Spearman correlation between various OOD scores and benchmark quality metric binary matching. Each column shows the correlation when ID and OOD samples are merged. Underline denotes the SOTA among all baselines, and **bold** denotes the SOTA among our methods. 

|_Method_<br>_Dataset_|||**ID + Far-s**|**hift OOD**|||||**ID + Near-s**|**hift OOD**|||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||Algebra|Geometry|Cnt.&Prob.<br>|Num.Theory|Precalculus|_Average_|GSM8K|SVAMP|AddSub|SingleEq|SingleOp|_Average_|
|||||**Kendall**|**Rank Corre**|**lation Coeffc**|**ient**||||||
|Max Softmax Prob. [12]<br>Perplexity[3]<br>Input Embedding[43]<br>Output Embedding[43]|0.035_±_0.025<br>-0.027_±_0.020<br>0.074_±_0.020<br>0.050_±_0.023|0.024_±_0.020<br>0.071_±_0.013<br>0.119_±_0.015<br>0.078_±_0.019|0.084_±_0.022<br>0.112_±_0.015<br>0.054_±_0.018<br>0.064_±_0.020|-0.043_±_0.021<br>0.035_±_0.016<br>0.111_±_0.016<br>0.065_±_0.020|0.021_±_0.014<br>0.036_±_0.011<br>0.034_±_0.013<br>0.034_±_0.009|0.024_±_0.020<br>0.050_±_0.015<br>0.078_±_0.016<br>0.058_±_0.018|0.064_±_0.022<br>0.073_±_0.019<br>0.042_±_0.020<br>0.042_±_0.018|0.067_±_0.015<br>0.027_±_0.017<br>-0.025_±_0.016<br>-0.001_±_0.013|0.076_±_0.016<br>0.082_±_0.015<br>0.052_±_0.014<br>0.056_±_0.015|0.057_±_0.019<br>0.091_±_0.019<br>0.055_±_0.020<br>0.032_±_0.015|-0.075_±_0.017<br>0.079_±_0.017<br>0.058_±_0.018<br>0.061_±_0.015|0.038_±_0.018<br>0.074_±_0.017<br>0.036_±_0.018<br>0.038_±_0.015|
|TV score (Ours)|0.182_±_0.016|**0.116**_±_**0.011**|**0.191**_±_**0.011**|**0.174**_±_**0.014**|0.142_±_0.009|**0.161**_±_**0.012**|**0.146**_±_**0.010**|**0.209**_±_**0.019**|0.195_±_0.018|**0.145**_±_**0.017**|**0.101**_±_**0.020**|**0.159**_±_**0.017**|
|w/ DiSmo (Ours)|**0.095**_±_**0.020**|0.059_±_0.014|0.123_±_0.020|0.131_±_0.016|**0.148**_±_**0.009**|0.111_±_0.016|0.178_±_0.025|0.113_±_0.015|**0.121**_±_**0.016**|0.079_±_0.017|0.076_±_0.019|0.113_±_0.018|
|∆(**bold**-underline<br>)|+0.021|-0.003|+0.079|+0.109|+0.112|+0.083|+0.073|+0.142|+0.039|+0.055|+0.022|+0.085|
|||||**Spearma**|**n Rank Corr**|**elation Coeff**|**cient**||||||
|Max Softmax Prob. [12]<br>Perplexity[3]<br>Input Embedding[43]<br>Output Embedding[43]|0.027_±_0.021<br>0.039_±_0.014<br>0.150_±_0.021<br>0.002_±_0.018|0.057_±_0.019<br>0.018_±_0.018<br>0.136_±_0.018<br>0.001_±_0.020|0.039_±_0.021<br>0.073_±_0.017<br>0.061_±_0.015<br>0.015_±_0.015|0.067_±_0.021<br>0.051_±_0.015<br>0.038_±_0.016<br>0.033_±_0.016|0.004_±_0.017<br>0.045_±_0.015<br>0.127_±_0.016<br>0.012_±_0.018|0.038_±_0.020<br>0.045_±_0.016<br>0.102_±_0.017<br>0.025_±_0.017|0.071_±_0.021<br>0.086_±_0.019<br>0.042_±_0.022<br>-0.040_±_0.017|-0.086_±_0.018<br>0.090_±_0.019<br>0.128_±_0.016<br>-0.019_±_0.016|0.063_±_0.019<br>0.016_±_0.019<br>0.154_±_0.016<br>0.045_±_0.019|0.081_±_0.015<br>0.039_±_0.015<br>0.110_±_0.017<br>0.101_±_0.018|0.002_±_0.019<br>0.017_±_0.016<br>0.140_±_0.015<br>-0.027_±_0.017|0.026_±_0.018<br>0.050_±_0.018<br>0.115_±_0.017<br>0.012_±_0.017|
|TV score (Ours)|0.122_±_0.012|**0.169**_±_**0.015**|0.102_±_0.016|0.126_±_0.016|**0.216**_±_**0.014**|0.147_±_0.015|**0.124**_±_**0.016**|**0.188**_±_**0.017**|0.127_±_0.017|**0.165**_±_**0.017**|**0.188**_±_**0.019**|**0.158**_±_**0.017**|
|w/ DiSmo (Ours)|**0.139**_±_**0.014**|0.148_±_0.013|**0.176**_±_**0.017**|**0.178**_±_**0.017**|0.121_±_0.016|**0.152**_±_**0.015**|0.071_±_0.018|0.150_±_0.020|**0.167**_±_**0.014**|0.131_±_0.017|0.153_±_0.017|0.134_±_0.017|
|∆(**bold**-underline<br>)|-0.011|+0.033|+0.103|+0.127|+0.089|+0.050|+0.038|+0.060|+0.013|+0.055|+0.048|+0.043|



Table 14: **OOD Quality Estimation (GPT2-XL)** : Kendall’s _τ_ and Spearman correlation between various OOD scores and benchmark quality metric binary matching. Each column shows the correlation when ID and OOD samples are merged. Underline denotes the SOTA among all baselines, and **bold** denotes the SOTA among our methods. 

|_Method_<br>_Dataset_|||**ID + Far-s**|**hift OOD**|||||**ID + Near-**|**shift OOD**|||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||Algebra|Geometry|Cnt.&Prob.|Num.Theory|Precalculus|_Average_|GSM8K|SVAMP|AddSub|SingleEq|SingleOp|_Average_|
|||||**Kendall**|**Rank Corre**|**lation Coeffc**|**ient**||||||
|Max Softmax Prob. [12]<br>Perplexity[3]<br>Input Embedding[43]<br>Output Embedding[43]|0.078_±_0.018<br>0.023_±_0.014<br>0.084_±_0.012<br>0.089_±_0.012|0.032_±_0.015<br>0.086_±_0.015<br>0.052_±_0.011<br>0.023_±_0.012|0.093_±_0.014<br>0.043_±_0.016<br>0.089_±_0.013<br>0.058_±_0.013|0.059_±_0.017<br>-0.009_±_0.015<br>0.062_±_0.011<br>0.014_±_0.012|0.068_±_0.012<br>0.036_±_0.012<br>0.007_±_0.010<br>0.067_±_0.011|0.066_±_0.015<br>0.036_±_0.014<br>0.059_±_0.012<br>0.050_±_0.012|0.032_±_0.024<br>0.068_±_0.022<br>0.041_±_0.019<br>0.068_±_0.018|0.061_±_0.020<br>0.008_±_0.022<br>0.010_±_0.020<br>0.036_±_0.020|0.088_±_0.018<br>0.035_±_0.018<br>-0.008_±_0.016<br>0.033_±_0.018|0.023_±_0.017<br>0.001_±_0.015<br>0.019_±_0.015<br>0.022_±_0.014|0.081_±_0.019<br>0.064_±_0.015<br>0.000_±_0.018<br>0.021_±_0.017|0.057_±_0.018<br>0.035_±_0.018<br>0.012_±_0.018<br>0.036_±_0.017|
|TV score (Ours)|0.127_±_0.010|**0.142**_±_**0.010**|**0.158**_±_**0.009**|0.098_±_0.011|**0.167**_±_**0.009**|0.138_±_0.010|0.087_±_0.013|**0.166**_±_**0.019**|**0.110**_±_**0.015**|**0.152**_±_**0.013**|0.141_±_0.016|**0.131**_±_**0.015**|
|w/ DiSmo (Ours)|**0.134**_±_**0.010**|0.112_±_0.009|0.132_±_0.010|**0.161**_±_**0.009**|0.158_±_0.009|**0.139**_±_**0.009**|**0.102**_±_**0.011**|0.099_±_0.021|0.094_±_0.015|0.126_±_0.013|**0.193**_±_**0.012**|0.123_±_0.014|
|∆(**bold**-underline<br>)|+0.045|+0.056|+0.065|+0.099|+0.099|+0.080|+0.034|+0.105|+0.022|+0.129|+0.112|+0.074|
|||||**Spearma**|**n Rank Corr**|**elation Coeff**|**cient**||||||
|Max Softmax Prob. [12]<br>Perplexity[3]<br>Input Embedding[43]<br>Output Embedding[43]|0.001_±_0.015<br>0.056_±_0.018<br>0.141_±_0.016<br>0.012_±_0.018|0.035_±_0.017<br>0.076_±_0.016<br>0.135_±_0.016<br>-0.098_±_0.016|0.054_±_0.016<br>0.024_±_0.016<br>0.087_±_0.014<br>0.076_±_0.016|0.076_±_0.016<br>0.035_±_0.017<br>0.097_±_0.017<br>0.054_±_0.017|0.056_±_0.018<br>-0.002_±_0.016<br>0.029_±_0.016<br>0.035_±_0.017|0.044_±_0.016<br>0.038_±_0.017<br>0.098_±_0.016<br>0.016_±_0.017|0.032_±_0.023<br>0.023_±_0.021<br>0.043_±_0.018<br>0.010_±_0.019|0.066_±_0.024<br>0.094_±_0.022<br>0.084_±_0.020<br>0.059_±_0.025|0.031_±_0.021<br>0.065_±_0.016<br>0.098_±_0.016<br>-0.077_±_0.019|0.067_±_0.021<br>0.088_±_0.019<br>0.024_±_0.016<br>0.084_±_0.019|0.087_±_0.020<br>0.019_±_0.019<br>0.090_±_0.017<br>0.067_±_0.023|0.057_±_0.022<br>0.058_±_0.019<br>0.068_±_0.016<br>0.029_±_0.021|
|TV score (Ours)|0.165_±_0.013|0.110_±_0.012|**0.141**_±_**0.012**|0.086_±_0.015|**0.115**_±_**0.014**|0.123_±_0.013|0.112_±_0.012|**0.164**_±_**0.012**|0.125_±_0.018|**0.168**_±_**0.018**|**0.161**_±_**0.014**|0.146_±_0.015|
|w/ DiSmo (Ours)|**0.173**_±_**0.011**|**0.208**_±_**0.010**|0.072_±_0.015|**0.145**_±_**0.019**|0.109_±_0.016|**0.141**_±_**0.014**|**0.134**_±_**0.015**|0.127_±_0.016|**0.189**_±_**0.017**|0.167_±_0.017|0.155_±_0.016|**0.154**_±_**0.016**|
|∆(**bold**-underline<br>)|+0.032|+0.073|+0.054|+0.048|+0.059|+0.043|+0.091|+0.070|+0.091|+0.080|+0.071|+0.086|



## **F.4 Beyond Mathematical Reasoning** 

Our method has a wider range of application scenarios beyond mathematical reasoning. To verify generalizability, we choose the multiple-choice quizzing task, which has the same “pattern collapse” property as mathematical reasoning, since the output space is limited to the “ABCD” four options. 

We select the MMLU dataset [10] and choose eight domains among it: high school mathematics, high school physics, high school chemistry, high school biology, high school geography, high school government and politics, high school psychology, high school statistics. We test eight rounds, each using one of the domains as the ID dataset and the remaining seven domains as the OOD datasets. 

We use Llama2-7B as the training backbone, each model is trained for 3K steps and 8 batch sizes in 4-card NVIDIA Tesla V100 (2 per card). The AUROC score matrices are shown in Figure 5(a)-(e), 

27 

presenting the results for TV score, input embedding, output embedding, and perplexity, respectively. In the figures, the Roman numeral I - VIII are represented as: 

- I = high school mathematics 

- II = high school physics 

- III = high school chemistry 

- IV = high school biology 

- V = high school geography 

- VI = high school government and politics 

- VII = high school psychology 

- VIII = high school statistics 

We find that MS-Prob and PPL nearly fail on the multiple-choice task and the output embedding is not as excellent as expected, which is caused by the pattern collapse phenomenon. 

Our method is comparable to the input embedding method and has very good absolute performances. For some far-shift OOD scenarios, _e.g._ , mathematics-psychology (I-VII), physics-politics (II-VI), etc., performances of the input embedding method and our method are basically the same, and there exists a reasonable range of competing phenomena, _e.g._ , our method performs more advantageously under physics-biology (II-IV), and the input embedding method is better under physics-geography (II-V). For some near-shift OOD scenarios, _e.g._ , mathematics-statistics (I-VIII), where both domains essentially belong to the category of math, our method will be more well-performed, indicating that the embedding-based method produces performance degradation in fine-grained scenarios, while our method possesses stronger robustness. 

Overall, our method is scalable and has greater advantages in fine-grained detection scenarios. 

28 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0029-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0029-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning_images/nips_2024_Embedding_Trajectory_for_Out-of-Distribution_Detection_in_Mathematical_Reasoning.pdf-0029-02.png)


**----- Start of picture text -----**<br>
(a)  Maximum Probability (b)  Sequence Perplexity<br>(c)  Input Embedding (d)  Output Embedding<br>(e)  TV Score (Ours)<br>**----- End of picture text -----**<br>


Figure 5: AUROC score matrix in MMLU dataset of different OOD scores. Rows represent ID data, and columns represent OOD data. 

29 

## **NeurIPS Paper Checklist** 

## 1. **Claims** 

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? 

Answer: [Yes] 

Justification: [Abstract, Section 1] Guidelines: 

   - The answer NA means that the abstract and introduction do not include the claims made in the paper. 

   - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. 

   - The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. 

   - It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper. 

2. **Limitations** 

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] 

Justification: [Limitations] Guidelines: 

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper. 

- The authors are encouraged to create a separate "Limitations" section in their paper. 

- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be. 

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. 

- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon. 

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size. 

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness. 

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations. 

## 3. **Theory Assumptions and Proofs** 

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? 

Answer: [Yes] 

30 

Justification: [Section 2, Appendix C] Guidelines: 

- The answer NA means that the paper does not include theoretical results. 

- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced. 

- All assumptions should be clearly stated or referenced in the statement of any theorems. 

- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. 

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. 

- Theorems and Lemmas that the proof relies upon should be properly referenced. 

## 4. **Experimental Result Reproducibility** 

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? 

Answer: [Yes] 

Justification: [Section 4, Appendix E] 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not. 

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. 

- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. 

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example 

- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. 

- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. 

- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). 

- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 

## 5. **Open access to data and code** 

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? 

31 

Answer: [Yes] 

Justification: [Section 4, Appendix E] Guidelines: 

- The answer NA means that paper does not include experiments requiring code. 

- Please see the NeurIPS code and data submission guidelines ( `https://nips.cc/ public/guides/CodeSubmissionPolicy` ) for more details. 

- While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark). 

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( `https: //nips.cc/public/guides/CodeSubmissionPolicy` ) for more details. 

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc. 

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why. 

- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable). 

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 

## 6. **Experimental Setting/Details** 

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? 

Answer: [Yes] 

Justification: [Section 4, Appendix E] 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. 

- The full details can be provided either with the code, in appendix, or as supplemental material. 

## 7. **Experiment Statistical Significance** 

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? 

Answer: [Yes] 

Justification: [Section 4, Appendix F] Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. 

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions). 

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) 

- The assumptions made should be given (e.g., Normally distributed errors). 

- It should be clear whether the error bar is the standard deviation or the standard error of the mean. 

32 

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified. 

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates). 

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 

## 8. **Experiments Compute Resources** 

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? 

Answer: [Yes] 

Justification: [Section 4, Appendix E] 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage. 

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute. 

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper). 

## 9. **Code Of Ethics** 

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics `https://neurips.cc/public/EthicsGuidelines` ? 

Answer: [Yes] 

Justification: [Ethics Statement] Guidelines: 

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. 

- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics. 

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 

## 10. **Broader Impacts** 

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? 

Answer: [Yes] 

Justification: [Ethics Statement] 

Guidelines: 

- The answer NA means that there is no societal impact of the work performed. 

- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact. 

- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations. 

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to 

33 

generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster. 

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. 

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 

## 11. **Safeguards** 

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? 

Answer: [Yes] 

Justification: [Ethics Statement] 

Guidelines: 

- The answer NA means that the paper poses no such risks. 

- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters. 

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images. 

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 

## 12. **Licenses for existing assets** 

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? 

Answer: [Yes] 

Justification: [Section 4, Appendix E] 

Guidelines: 

- The answer NA means that the paper does not use existing assets. 

- The authors should cite the original paper that produced the code package or dataset. 

- The authors should state which version of the asset is used and, if possible, include a URL. 

- The name of the license (e.g., CC-BY 4.0) should be included for each asset. 

- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided. 

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, `paperswithcode.com/datasets` has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset. 

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided. 

- If this information is not available online, the authors are encouraged to reach out to the asset’s creators. 

## 13. **New Assets** 

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? 

34 

Answer: [NA] . 

Justification: [Our paper does not introduce new assets.] Guidelines: 

- The answer NA means that the paper does not release new assets. 

- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc. 

- The paper should discuss whether and how consent was obtained from people whose asset is used. 

- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 

## 14. **Crowdsourcing and Research with Human Subjects** 

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? 

Answer: [NA] . 

Justification: [Our paper does not involve crowdsourcing experiments and research.] Guidelines: 

   - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. 

   - Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. 

   - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 

15. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects** 

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? 

Answer: [NA] . 

Justification: [NA] 

Guidelines: 

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. 

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. 

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. 

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 

35 

