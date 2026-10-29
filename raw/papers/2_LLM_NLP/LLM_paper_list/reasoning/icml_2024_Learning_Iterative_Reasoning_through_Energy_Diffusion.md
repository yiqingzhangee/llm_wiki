## **Learning Iterative Reasoning through Energy Diffusion** 

**Yilun Du**[1 *] **Jiayuan Mao**[1 *] **Joshua Tenenbaum**[1] 

## **Abstract** 

We introduce _iterative reasoning through energy diffusion_ (IRED), a novel framework for learning to reason for a variety of tasks by formulating reasoning and decision-making problems with energy-based optimization. IRED learns energy functions to represent the constraints between input conditions and desired outputs. After training, IRED adapts the number of optimization steps during inference based on problem difficulty, enabling it to solve problems outside its training distribution — such as more complex Sudoku puzzles, matrix completion with large value magnitudes, and path finding in larger graphs. Key to our method’s success is two novel techniques: learning a sequence of annealed energy landscapes for easier inference and a combination of score function and energy landscape supervision for faster and more stable training. Our experiments show that IRED outperforms existing methods in continuous-space reasoning, discrete-space reasoning, and planning tasks, particularly in more challenging scenarios. Code and visualizations are at https://energy-based-model. github.io/ired. 

## **1. Introduction** 

Being able to solve complex reasoning tasks such as logic inference, mathematical proofs, and decision-making is one of the hallmarks of artificial intelligence. Researchers in various fields have been working on domain-specific algorithms for solving these tasks, typically utilizing various forms of search or optimization in iterative manners ( _e.g_ ., dynamic programming and gradient descent). These domain-specific algorithms are usually highly efficient and effective, but they usually can not directly handle sensory data and typically 

> *Equal contribution 1MIT. Correspondence to: Yilun Du _<_ yilundu@mit.edu _>_ , Jiayuan Mao _<_ jiayuanm@mit.edu _>_ . 

> _Proceedings of the 41[st] International Conference on Machine Learning_ , Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion_images/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion.pdf-0001-09.png)


**----- Start of picture text -----**<br>
Diverse Training Tasks<br>? ⋯ 1.0<br>0.3 ⋯ ?<br>⋮ ⋱ ⋮<br>0.82 ⋯ ?<br>Matrix Completion Sudoku Path Finding (Red-Green)<br>**----- End of picture text -----**<br>


**Learned Annealed Energy Functions** 𝐸!" (𝑥, 𝑦) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion_images/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion.pdf-0001-11.png)


**----- Start of picture text -----**<br>
1.5 ⋯ −1.3<br>3.1 ⋯ ?<br>⋮ ⋱ ⋮<br>? ⋯ ?<br>(Large Magnitude) (Few Given Numbers) (Larger Graphs)<br>Generalization to harder instances<br>**----- End of picture text -----**<br>


_Figure 1._ **Reasoning as Energy Diffusion** – IRED formulates reasoning problem with inputs _**x**_ and output _**y**_ , as an energy minimization problem over a learned energy function. It can be trained stably for a wide variety of reasoning tasks and achieves strong generalization to harder problem instances, through adaptive computation in the optimization process. 

require users or experts to encode rules in domain-specific languages (such as axioms used in mathematical provers or domain theories in planning). Furthermore, it is usually hard for these systems to learn from experience to improve their performance on familiar tasks. A large body of work has been trying to address these limitations by incorporating machine learning in order to handle sensory inputs and learn to formulate and solve problems. Typical ideas include utilizing these domain-specific solvers as a submodule in a deep neural network (e.g., SAT solvers; Wang et al., 2019) or building structured neural networks that can realize algorithms (e.g., dynamic programming; Xu et al., 2019). 

Illustrated in Figure 1, we take a different approach to address the aforementioned challenges by formulating various kinds of reasoning and decision-making problems as an optimization problem. In particular, we consider the learning-to-reason problem as learning an energy function _Eθ_ ( _**x** ,_ _**y**_ ) over input conditions _**x**_ and desired output _**y**_ . For example, logical deduction can be cast as finding possible assignments to variables that satisfy all logical constraints; theorem proving can be cast as finding a sequence of valid 

1 

**Iterative Reasoning through Energy Diffusion** 

deduction steps that entails the goal; planning can be cast as finding a sequence of actions that respect the transition model of the environment and achieve the goal. This formulation directly allows us to learn the underlying constraints for a given task automatically from input-output data, without additional task-specific knowledge. Therefore, we can solve a wide variety of tasks across different domains using the same underlying training and inference paradigm, by only swapping out the neural network encoder for different data formats of _**x**_ and _**y**_ . Another important feature of this optimization-based formulation is that during inference time, we can choose to apply a different amount of computation depending on the hardness of the problem by inspecting the value of the function _Eθ_ ( _**x** ,_ _**y**_ ). 

In particular, in this paper, we propose _iterative reasoning through energy diffusion_ (IRED), a general framework for learning to reason. IRED is trained on a dataset of paired ( _**x** ,_ _**y**_ ) data, and can recover the underlying energy function describing the objective function and constraints. During inference, because we are explicitly solving an optimization problem of finding the _**y**[∗]_ that maximizes the energy function _Eθ_ , we can run an adaptive number of optimization steps depending on the hardness of the problem. This enables us to solve problems that are beyond the training distribution, for example, Sudoku puzzles with a harder difficulty level, matrix manipulation under worse condition numbers, and sorting arrays with a larger size. 

Our paper is not the first one to propose the use of energybased models (EBMs) as a general framework for learning and reasoning (see, for example, Du et al., 2022). Although being a general framework for learning and reasoning, existing work falls short in its training speed, stability, and inference-time optimization hardness. These issues are critical and fundamentally hard because the learning of EBMs typically involves back-propagation through the entire iterative optimization process, and in general, the function landscape of _Eθ_ can be complex with a large number of local optima. In this paper, we propose two important techniques to address these two challenges. Drawing inspiration from diffusion models and their relations to energy-based models (Ho et al., 2020; Du et al., 2023), instead of learning a single energy landscape, we instead learn a sequence of annealed energy landscapes, where smoother landscapes are being first optimized before optimizing for sharper ones afterward. Furthermore, in contrast to earlier work on EBM learning, IRED uses a combination of denoising supervision and direct supervision through negative sample mining. Both techniques can be implemented without the need to backpropagate through the optimization process, thereby making our learning algorithm both stable and fast. 

We show the effectiveness of IRED on three groups of tasks: continuous-space reasoning ( _e.g_ ., matrix completion, in- 

version), discrete-space reasoning ( _e.g_ ., Sodoku solving, graph connectivity prediction), and planning ( _e.g_ ., finding paths on graphs). Compared with various domain-specific and domain-independent learning-to-reason baselines, including recurrent adaptive computation (Palm et al., 2018), EBM (Du et al., 2022) and diffusion-based models (Ho et al., 2020), IRED outperforms all of them, especially on test instances that are of higher difficulty levels, such as on matrices with larger value magnitudes, sudoku of higher difficulty levels, and larger graphs. Ablation studies show that the proposed optimization paradigm enables stable training and better generalization. 

## **2. Related Work** 

**Learning to reason with optimization.** A wide variety of reasoning problems can be formulated as an optimization problem, including constraint satisfaction problems (CSPs), mathematical programs, discrete-space (Kautz et al., 2006) and continuous-space ( _i.e_ ., trajectory optimization, see Bryson, 2018, for a survey) optimization problems, and even algorithmic reasoning tasks (Brockett, 1991). The highlevel idea is to cast these inference and decision-making problems as finding a set of variables that minimizes an objective function subject to constraints. Recently, there has been a growing interest in learning the objective and constraint functions instead of manually specifying them, which would be useful for domains where people do not have expert knowledge or simply the functions are too hard to be specified ( _e.g_ ., over high-dimensional sensory inputs). 

Along this line, the first group of papers has explored using domain-specific optimization solvers as a computation block in neural networks. For example, Amos & Kolter (2017); Donti et al. (2017) integrates quadratic program solvers, Djolonga & Krause (2017); Wilder et al. (2019) studies submodular programs solvers, Wang et al. (2019) uses differentiable Max-SAT solvers, Yang et al. (2020) considers answer-set programming (ASP) solvers, Manhaeve et al. (2018) considers probabilistic logic programming solvers, and Rocktaschel & Riedel¨ (2017) integrates symbolic theorem-proving solvers. However, due to the dependence on a particular problem formulation language, these frameworks are usually limited to solving problems of a particular kind. 

The second group of papers has explored using a generic optimization framework as the underlying formulation. For example, Bai et al. (2019); Anil et al. (2022) utilizes equilibrium energy minimization inside a neural network to save memory, Rubanova et al. (2022); Comas et al. (2023) utilizes energy minimization to simulate physical dynamics by using neural networks to parameterize an energy function. Our paper falls into this group as well. Similar to our work, Du et al. (2022) uses energy-based models for learn- 

2 

**Iterative Reasoning through Energy Diffusion** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion_images/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion.pdf-0003-01.png)


**----- Start of picture text -----**<br>
(a) Learned Energy Landscapes (b) Score Function Supervision<br>ℒ= ∇&𝐸!$ 𝒙, 𝒚+ 𝝐−𝝐<br>...... (c) Energy Landscape Supervision<br>𝑒 [%'][!]<br>𝐸!"(𝒙, 𝒚) 𝐸!#(𝒙, 𝒚) 𝐸!$%"(𝒙, 𝒚) 𝐸!$(𝒙, 𝒚) ℒ= −log(𝑒 [%'][!] + 𝑒 [%']["] )<br>**----- End of picture text -----**<br>


_Figure 2._ **IRED Learns a Sequence of Energy Landscapes.** During inference time, we optimize for _**y**[∗]_ that minimizes the energy function, and we gradually increase the complexity of the energy optimization problem. The energy functions are trained with a combination of score function supervision and energy landscape supervision. 

ing to reason. In contrast to it, this paper proposes to use a combination of denoising diffusion and supervised energy landscape training. This gives us stable training and strong performance. A concurrent work from Sun & Yang (2023) also considers using diffusion models for solving combinatorial optimization problems. However, their formulation is developed specifically for combinatorial optimization problems on graphs, but our diffusion formulation with energy parameterizations, landscape supervision, and substep optimizations can be generally applied to many optimization and decision-making domains. 

**Learning to reason with iterative neural computation.** Another popular line of research studies using neural networks with iterative computation for reasoning. They draw motivation from the fact that many domain-specific constraint solvers are indeed iterative optimization algorithms ( _e.g_ ., gradient descent). At a high level, there are two groups of work: leveraging explicit program representations (Graves et al., 2014; Neelakantan et al., 2015; Reed & De Freitas, 2016; Cai et al., 2017; Chen et al., 2020b; Banino et al., 2021, ; typically with external memories) and using recurrent neural networks (Graves, 2016; Kaiser & Sutskever, 2016; Chung et al., 2017; Bolukbasi et al., 2017; Yang et al., 2017; Dong et al., 2019; Dehghani et al., 2019; Schwarzschild et al., 2021; Yang et al., 2023a). One of the key challenges in both types of approaches is when to halt the computation. Researchers have been tackling this problem through reinforcement learning (Chen et al., 2020a; Chung et al., 2017), leveraging hierarchical decomposition of programs (Cai et al., 2017), heuristic policies (Bolukbasi et al., 2017), and variational inference (Banino et al., 2021). However, these approaches are usually unstable, and many of them require manual hyper-parameter tuning (Banino et al., 2021) or additional human annotations (Cai et al., 2017). In this paper, we focus on an orthogonal approach by solving a broad set of reasoning problems by casting them as an optimization on learned energy landscapes. During optimization, the energy function of the landscapes naturally acts as a termination criterion. 

**Energy-based models and diffusion models.** Our work is related to past work formulating prediction using EnergyBased Models (EBMs) (LeCun et al., 2006). Most recent 

EBMs have focused on learning probabilistic models over data (Xie et al., 2016; 2018; Du & Mordatch, 2019; Grathwohl et al., 2020; Du et al., 2021; Arbel et al., 2021; Xiao et al., 2020) but most similar to our work (Du et al., 2022) focuses on using energy minimization to solve reasoning tasks. Our work leverages the connection of energy based models and diffusion models (Du et al., 2023) to more effectively learn energy landscapes for solving reasoning problems. 

An important difference between our proposed approach and standard diffusion models is that diffusion models usually focus on learning a particular sampling path transitioning from Gaussian noise to a target solution, where individual transition kernels across timesteps are learned. However, when obtaining a solution using these transition kernels, errors often accumulate across sampling timesteps, preventing a diffusion model from obtaining a precise answer to a reasoning problem. By contrast, we formulate predicting solutions as optimizing an annealed sequence of energy landscapes. In this setting, multiple steps of optimization are run at each energy landscape to ensure that we are at an energy minima at every landscape. These multiple steps of optimization prevent the accumulation of errors from using transition kernels in diffusion models, as they project the sample to an energy minima, which is likely “in distribution” to what has been seen during training. 

## **3. Learning Iterative Reasoning through Energy Optimization** 

Let _D_ = _{X, Y }_ be a dataset for a reasoning task consisting of inputs _**x** ∈_ R _[O]_ and corresponding solutions _**y** ∈_ R _[M]_ . We aim to learn a neural network-based prediction model NN _θ_ ( _·_ ) which can generalize execution NN _θ_ ( _**x**[′]_ ) to a test distribution _**x**[′] ∈_ R _[O][′]_ , where _**x**[′]_ can be significantly larger and more challenging than the training data _**x** ∈ X_ ( _e.g_ ., of higher dimensions, or with larger number magnitudes), by leveraging a possibly increased computational budget. 

We formulate this adaptive model as an iterative energy optimization in Section 3.1. Our overall framework is illustrated in Figure 2. In particular, we construct an annealed sequence of energy functions to improve optimization. To involve 

3 

**Iterative Reasoning through Energy Diffusion** 

training stability, speed, and performance, we propose to shape the energy landscape to correctly assign minimal energy to ground truth solutions. We provide full pseudocode for training our approach in Section 3.4 with training following Algorithm 1 and inference following Algorithm 2. 

## **3.1. Reasoning as Annealed Energy Minimization** 

A wide variety of reasoning and decision making problems can be formulated as an optimization problem. Traditionally, researchers have been focused on designing various domainspecific algorithms for solving different problems, typically with search, gradient-based optimization, or other forms of iterative computation, and also integrating machine learning to help. In this work, we take a different approach of formulating various kinds of reasoning and decision-making problems as an optimization process over a learned energybased model (EBM): _Eθ_ ( _**x** ,_ _**y**_ ) : R _[O] ×_ R _[M] →_ R. Under this formulation, the final prediction problem can be cast as finding the solution _**y**_ according to: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion_images/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion.pdf-0004-04.png)


One can use gradient descent to find such solutions: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion_images/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion.pdf-0004-06.png)


where _λ_ is the step size for optimization and the initial prediction _**y**_[0] is initialized from a fixed noise distribution ( _i.e_ ., Gaussian throughout the paper). The final output of _**y**[T]_ is obtained after _T_ steps of optimization. 

In earlier work using a similar formulation Du et al. (2022), such EBM _Eθ_ ( _**x** ,_ _**y**_ ) is trained by differentiating through the _T_ steps of optimization and minimizing the MSE with the ground truth label _**y** L_ Opt( _θ_ ) = _∥_ _**y** i[T][−]_ _**[y]**[i][∥]_[2] _[.]_[ This ap-] proach requires the forward and backward computation of K steps of optimization at training, which makes it slow and unstable. Furthermore, because the EBM _Eθ_ may have a complex optimization landscape[1] , robustly finding solutions to Equation 2 is fundamentally hard. 

As a general solution to stable training and better test-time optimization, instead of directly learning _Eθ_ ( _**x** ,_ _**y**_ ), at a high-level, we propose to learn a sequence of annealed energy functions _Eθ[k]_[(] _[k]_[= 0] _[,]_[ 1] _[,][ · · ·][, K]_[), and supervise the] EBM learning with the gradient of the energy function: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion_images/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion.pdf-0004-10.png)


During training time, we obtain the ground truth for _**y**_ by generating a noise-corrupted label _**y**_ + _ϵ_ , following a schedule of noise corruptions. By supervising on the gradient, our approach is substantially faster and more stable than earlier works using plain EBMs (Du et al., 2022) as it only supervises training of a single step of the optimization. 

1For example, the 3-SAT problem exhibits steep energy minima surrounded by flat energy landscapes. 

## **3.2. Learning Sequence of Annealed Energy Landscapes** 

Our key idea to mitigate the hard optimization problem of Equation 2 is to use simulated annealing — where smoother energy landscapes are first optimized before optimizing sharper ones afterward, as illustrated in Figure 2. 

Similar to diffusion models (Sohl-Dickstein et al., 2015; Ho et al., 2020), we propose to optimize and learn an annealed sequence of energy landscapes, with earlier energy landscapes being smoother to optimize and the latter ones more difficult. Given a ground truth label _**y**_ , we learn a sequence of _K_ energy functions[2] _Eθ[k]_[(] _**[x]**[,]_ _**[ y]**_[)][ over the ground] truth label distribution _p_ ( _**y**[∗] |x_ ), where each energy function is learned to represent an EBM distribution 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion_images/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion.pdf-0004-16.png)


over a sequence of noise scales _σk_ . Here, _N_ ( _·|µ, σ_ ) is the Gaussian density function. Larger values of _σk_ correspond to smoother energy landscapes while smaller values lead to sharper landscapes, with the energy minima of landscape _k_ corresponding to �1 _− σk_[2] _**[y]**[∗]_[(which][can][be][scaled][by] 1 ~~_√_~~ 1 _−σk_[2][to obtain the ground truth prediction] _**[ y]**[∗]_[).] 

We can directly learn each energy landscape by supervising the gradient of energy function to denoise the corrupted ground truth label _**y**[∗]_ from the dataset 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion_images/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion.pdf-0004-19.png)


where _ϵ ∼N_ (0 _,_ 1). Given a set of _K_ different learned energy landscapes, we can initialize a data sample from Gaussian noise and sequentially run _T_ steps of optimization following Equation 2 over each energy landscape _k_ (starting with high noise levels and progressing to lower noise levels). The optimization result in the previous energy landscape is used to initialize optimization in the next landscape, after _√_ 1 _−σk_[2] scaled by the appropriate scaling factor ~~_√_~~ 1 _−σk_[2] _−_ 1[.] 

## **3.3. Shaping the Energy Landscape** 

In the denoising training objective Equation 5, while the gradient of the energy landscape is locally trained to restore the ground truth label _**y**_ , it is not necessarily the case that the overall global energy minima arg min _**y** Eθ_ ( _**x** ,_ _**y** , k_ ) corresponds to the ground truth label ~~�~~ 1 _− σk_[2] _**[y]**[∗]_[.] 

To enforce that the global energy minima of each of the _k_ energy landscapes corresponds to the ground truth energy minima, we further propose a contrastive loss, where 

> 2Empirically, in our experiments, we found that setting _K_ = 10 was sufficient across all the domains we considered. With a total of 10 energy landscapes, we can smoothly transition from a Gaussian-like landscape (with _K_ = 10) to a sharp and discontinuous landscape (with _K_ = 1). 

4 

**Iterative Reasoning through Energy Diffusion** 

## **Algorithm 1** IRED Training 

## **Algorithm 2** IRED prediction algorithm 

|**Input:** Input task **_x_**_i_, Step Sizes _λk_, Number of Land-|
|---|
|scapes_K_, EBM_Eθ_(_·_), Optimization Steps_T_.|
|˜**_y_**_i ∼N_(0_,_1)|
|**for**each landscape_k_ = 1to_K_ **do**|
|**for**run T steps of optimization_t_= 1to_T_ **do**|
|_▷Optimize candidate solution_ ˜**_y_**_i with gradient:_|
|˜**_y_**_′_<br>_i ←_˜**_y_**_i −λk∇_**_y_**_Eθ_(**_x_**_i,_˜**_y_**_i, k_)|
|_▷Check if the gradient descent step decreases energy:_|
|**if**_Eθ_(**_x_**_i,_˜**_y_**_i, k_)_> Eθ_(**_x_**_i,_˜**_y_**_′_<br>_i, k_)**then**|
|**_y_**_i ←_**_y_**_′_<br>_i_|
|**end if**|
|**end for**<br>_▷Scale optimized candidate solution:_|
|˜**_y_**_i ←_<br>_√_<br>1_−σ_2<br>_k_<br>~~_√_~~<br>1_−σ_2<br>_k−_1 ˜**_y_**_i_<br>**end for**|
|**return****_y_** = ˜**_y_**_i_|



**Input:** Problem Dist _pD_ ( _**x** ,_ _**y**_ ), EBM _Eθ_ ( _·_ ), Noise Schedules _{σk}_ , Corruption Function _c_ ( _·_ ), Landscapes _k_ . **while** not converged **do** _▷ Supervise the Energy Landscape through Denoising:_ _**x** i,_ _**y** i ∼ pD, ϵ ∼N_ (0 _,_ 1) _, k ∼{_ 1 _, . . . , K}_ _**y**_ ˜ _i ←_ ~~�~~ 1 _− σk_[2] _**[y]**[i]_[ +] _[ σ][k][ϵ] L_ MSE _←∥∇yEθ_ ( _**x** i,_ _**y**_ ˜ _i, k_ ) _− ϵ∥_[2] _▷ Shape the Energy Landscape Contrastively:_ _**y** i[−][←][c]_[(] _**[y]**[i]_[)] _**y**_ ˜ _i− ←_ �1 _− σk_[2] _**[y]** i[−]_[+] _[ σ][k][ϵ] Ei_[+] _[←][E][θ]_[(] _**[x]**[i][,]_ _**[y]**_[˜] _[i][, k]_[)][;] _[ E] i[−][←][E][θ]_[(] _**[x]**[i][,]_[ ˜] _**[y]** i[−][, k]_[)] _e[−][E] i_[+] _L_ Contrast _←−_ log � _e[−][E] i_[+] + _e[−][E] i[−]_ � _▷ Optimize objective LMSE_ + _LContrast wrt θ:_ ∆ _θ ←∇θ_ ( _L_ MSE + _L_ Contrast) Update _θ_ based on ∆ _θ_ using Adam optimizer **end while** 

we construct a set of negative label _**y**[−]_ (formed by noise corrupting the ground truth label _**y**[∗]_ ). Given an energy _E_[+] = _Eθ_ ( _**x** ,_ �1 _− σk_[2] _**[y]**[∗]_[+] _[ σ][k][ϵ]_[;] _[ k]_[)][ of the ][g][round truth la-] bel _**y**[∗]_ and an energy _E−_ = _Eθ_ ( _**x** ,_ �1 _− σk_[2] _**[y]**[−]_[+] _[σ][k][ϵ]_[;] _[ k]_[)][ of] _e[−][E]_[+] the negative label _**y**[−]_ , _L_ Contrast( _θ_ ) = _−_ log � _e[−][E]_[+] + _e[−][E][−]_ �. To reduce the variance of the contrastive loss, we use the same sampled noise value _ϵ_ for both _**y**_ and _**y**[−]_ . 

## **3.4. Combined Training and Inference Paradigms** 

We provide the overall pseudocode for training IRED in Algorithm 1 and executing algorithmic reasoning with IRED in Algorithm 2. We use a cosine beta schedule to train annealed energy landscapes and use a total of 10 energy landscapes (we empirically found that more landscapes did not lead to improved performance). At inference time, we can vary the number of optimization steps _T_ for each energy landscape to make trade-offs between performances and inference speed. 

In principle, when the solution is not well-defined, it is possible to use IRED to model multi-modal distributions, similar to how diffusion models have been proven effective in modeling multi-modal image distributions. Depending on the particular use case, one may also add additional inference-time constraints ( _e.g_ ., by composing the learned IRED energy function with other energy functions) to select favorable solutions. 

## **4. Experiments** 

We compare IRED with both domain-specific and domainindependent baselines on three domains: continuous algo- 

rithmic reasoning, discrete-space reasoning, and planning. As we will break down in the following sections, the main advantages of IRED are twofold. First, compared with energy-based models (IREM), it is faster to train since it does not require backpropagation through the optimization process. Second, in terms of task performance, our focus is on generalization to “harder” problems, particularly leveraging the contrastive energy supervision and runtime iterative refinements. The idea is that after learning a correct energy landscape, the model can adaptively use more computation at test time to directly generalize to harder problems: we will focus on evaluating this generalization across all domains. 

## **4.1. Continuous Algorithmic Reasoning** 

**Setup.** We first evaluate IRED on a set of continuous algorithmic reasoning tasks from Du et al. (2022). We consider three matrix operations on 20 _×_ 20 matrices, which are encoded 400-dimensional vectors: 

1. _Addition_ : We first evaluate neural networks in their ability to add matrices together (element-wise). We also evaluate neural network on harder variants of the addition problems at test time by feeding input vectors with larger magnitudes. 

2. _Matrix Completion_ : Next, we evaluate neural networks on their ability to do low-rank matrix completion. We mask out 50% of the entries of a low-rank input matrix constructed two separate rank 10 matrices _U_ and _V_ , and train networks to reconstruct the original input matrix. We construct harder variants of the matrix completion problem at test time by increasing the magnitude of values in _U_ and _V_ . 

5 

## **Iterative Reasoning through Energy Diffusion** 

|||**Same**|**Harder**|
|---|---|---|---|
|**Task**|**Method**|**Diffculty**|**Diffculty**|
||Feedforward|0.0448|0.7029|
||Recurrent|0.3610|2.6133|
|**Addition**|Programmatic|0.0111|0.3446|
||Diffusion|0.0071|0.5931|
||IREM<br>IRED (ours)|0.0003<br>**0.0002**|0.0021<br>**0.0020**|
||Feedforward|0.0203|0.2720|
|**Matrix**<br>**Completion**|Recurrent<br>Programmatic<br>Diffusion|0.0266<br>0.0203<br>0.0219|0.3285<br>0.2637<br>0.2142|
||IREM|0.0183|0.2074|
||IRED (Ours)|**0.0174**|**0.2054**|
|**Matrix**<br>**Inverse**|Feedforward<br>Recurrent<br>Programmatic<br>Diffusion<br>IREM<br>IRED (Ours)|0.0112<br>0.0109<br>0.0124<br>0.0115<br>0.0108<br>**0.0095**|0.2150<br>0.2123<br>0.2209<br>0.2132<br>0.2083<br>**0.2063**|



_Table 1._ **Continuous Algorithmic Reasoning.** Test evaluation performance on continuous algorithmic tasks. Inputs and outputs are 20 by 20 matrices. Error is reported using elementwise mean square error. Models are evaluated on test problems drawn from the training distribution (same difficulty) and a harder test distribution (harder difficulty). IRED outperforms comparisons. 

3. _Matrix Inverse_ : Finally, we evaluate neural networks on their ability to compute matrix inverses. We construct harder matrix inverse problems by considering less well-conditioned input matrices. 

We report the underlying mean-squared error (MSE) between the predictions and the associated ground truth outputs on test problem instances. To more effectively generate negative samples for IRED in this domain, we first noise-corrupt ground truth labels and then run two steps of gradient optimization on the energy landscape to form negative samples. Details can be found in Appendix A. 

**Baselines.** We compare our approach to a set of iterative reasoning baselines found in (Du et al., 2022): _(Feedforward):_ an iterative reasoning approach where the same MLP is repeatedly applied, _(Recurrent):_ an iterative reasoning approach where the recurrent network is repeatedly applied, and _(Programmatic):_ an iterative reasoning approach which repeatedly applies a learned programmatic module (Banino et al., 2021). We further compare with the IREM method (Du et al., 2022), as well as using a denoising diffusion model directly to solve continuous tasks. All methods use identical architectures (with small differences due to recurrent layers or timestep conditioning). 

**Quantitative Results.** We compare IRED with baselines across settings in Table 1. Similar to IREM, IRED is able to nearly perfectly solve the task of the addition, as well as generalize to larger addition matrices. On other tasks, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion_images/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion.pdf-0006-07.png)


**----- Start of picture text -----**<br>
Error Map Error Map Error Map Error Map<br>(Landscape 2) (Landscape 3) (Landscape 5) (Landscape 10)<br>Figure 3.  Optimized Solutions Across Landscapes  – Error maps<br>of intermediate optimized solutions. Optimized solutions at earlier<br>landscapes are less accurate than later ones.<br>Energy Landscape 1 Energy Landscape 5 Energy Landscape 10<br>**----- End of picture text -----**<br>


_Figure 3._ **Optimized Solutions Across Landscapes** – Error maps of intermediate optimized solutions. Optimized solutions at earlier landscapes are less accurate than later ones. 

_Figure 4._ **Energy Landscape** – Predicted energy values for _**y**_ and the corresponding MSE distance of _**y**_ from the problem solution across different landscapes on the matrix inverse task. The earlier energy landscapes are smoother than the later ones. 

IRED outperforms IREM and also generalizes better to harder problems. Furthermore, our approach substantially outperforms directly using a diffusion process to predict solutions, which lacks the iterative energy minimization procedure that explicitly learns the task constraints. 

**Qualitative Visualization.** We provide a qualitative visualization of the error map of the optimized solution on the matrix inverse task at each different learned energy landscape in Figure 3. The error of optimized solutions at different energy landscapes decreases over time. 

**Energy Landscape.** We visualize the learned energy landscape in Figure 4 as a function of the distance of an input label from the ground truth label. In early energy landscapes, the difference between energy values of solutions close and far from the ground truth solution is low, and therefore the energy landscape is relatively flat. At later landscapes, the energy value increases substantially as the input solution deviates from the ground truth solution. 

**Performance with Increased Computation.** We analyze the performance on the matrix inverse task as a factor of an increased number of computational steps in Table 2. We find that running additional steps of optimization slightly improves performance on in-distribution tasks and substantially improves performance on harder problems. 

**Ablation.** We ablate each component of IRED in Table 3. In the first two rows of Table 3, we compare our gradientdescent-based optimization with a noisy optimization procedure corresponding to the diffusion reverse process for each energy landscape. In the third row, we compare the difference between running multiple steps of optimization as opposed to a single energy optimization step. Finally, We then consider the effect of contrastively shaping the energy landscape. All components lead to improved performance. 

6 

**Iterative Reasoning through Energy Diffusion** 

|**Opt. Steps**<br>**Same Diffculty**<br>**Harder Diffculty**|**Task**|**Method**|**Test**|**Harder**|
|---|---|---|---|---|
|10<br>0.0096<br>0.2110<br>20<br>0.0096<br>0.2100||IREM|**Dataset**<br>93.5%|**Dataset**<br>24.6%|
|30<br>0.0096<br>0.2090<br>40<br>0.0095<br>0.2063|**Sudoku**|Diffusion<br>SAT-Net|66.1%<br>98.3%|10.3%<br>3.2%|
|||RRN|**99.8%**|28.6%|
|_ble 2._**Continuous-Space Reasoning Performance vs Reason-**||IRED (ours)|99.4%|**62.1%**|
|**g Steps.** More reasoning steps in IRED at inference time sub-<br>antially improve generalization to harder diffculty tasks on the<br>atrix inverse task. IRED is trained with 10 energy landscapes.|**Visual Sudoku**|SAT-Net<br>RRN<br>IRED (ours)|63.2%<br>**99.8%**<br>98.3%|0.0%<br>28.6%<br>**46.6%**|
|**Gradient Optimization Contrastive**<br>**Same**<br>**Harder**||IREM|94.3%|89.8%|
|**Descent**<br>**Refnement**<br>**Shaping**<br>**Diffculty Diffculty**|**Connectivity**|Diffusion|61.6%|61.3%|
|N<br>N<br>N<br>00158<br>02223||IRED (ours)|**99.1%**|**93.8%**|



_Table 2._ **Continuous-Space Reasoning Performance vs Reasoning Steps.** More reasoning steps in IRED at inference time substantially improve generalization to harder difficulty tasks on the matrix inverse task. IRED is trained with 10 energy landscapes. 

|**Gradient **|**Optimization **|**Contrastive**|**Same**|**Harder**|
|---|---|---|---|---|
|**Descent**|**Refnement**|**Shaping**|**Diffculty **|**Diffculty**|
|No|No|No|0.0158|0.2223|
|Yes|No|No|0.0097|0.2135|
|Yes|Yes|No|0.0097|0.2113|
|Yes|Yes|Yes|0.0095|0.2063|



_Table 4._ **Discrete Reasoning Performance.** We evaluate models on the Sudoku task and the connectivity task. Sudoku:” the harder dataset has between 17 to 34 entries given, while models are trained with 31 to 42 entries given. Connectivity: the harder graphs have at most 18 nodes while the training graphs have only 12. 

_Table 3._ **Continuous Ablations –** Ablations of proposed components of IRED on performance on the matrix inverse task. Leveraging gradient descent to optimize energy functions, using multiple steps of optimization at each energy landscape and contrastively shaping the energy landscape with ground truth labels all improve the performance on the Inverse task. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion_images/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion.pdf-0007-06.png)


**----- Start of picture text -----**<br>
Energy Minima Energy Minima Energy Minima<br>(Landscape 1) (Landscape 3) (Landscape 10)<br>**----- End of picture text -----**<br>


## **4.2. Discrete-Space Reasoning** 

**Setup.** The second group of tasks evaluates IRED on its reasoning in discrete spaces ( _i.e_ ., values are all binary or one-hot categorical). We run evaluations on two tasks: Sudoku solving and graph connectivity reasoning. 

_Figure 5._ **Optimized Boards Across Landscapes** – Plot of the minimal energy board across energy landscapes, given the same initial board. Later energy landscapes lead to more accurate boards. We highlight inconsistent entries in red. 

1. _Sudoku_ : In the Sudoku game, the model is given a partially filled Sudoku board, with 0’s filled-in entries that are currently unknown. The task is to predict a valid solution that jointly satisfies the Sodoku rules and that is consistent with the given numbers. We use the dataset from SAT-Net (Wang et al., 2019) as the training and standard test dataset. In SAT-Net, the number of given numbers is within the range of [31 _,_ 42]. Our harder dataset is from RRN (Palm et al., 2018), which is a different Sudoku dataset where the number of given numbers is within [17 _,_ 34]. We will show that our system, being trained on simpler Sudoku games with fewer blank entries, generalizes to harder instances. 

- we do not limit the path lengths between nodes (in contrast to Dong et al. (2019)), on the training set, the maximum distance between two connected nodes is 9, and in the harder test set, the maximum distance is 16. 

**Quantitative Results.** We compare IRED with both IREM and diffusion baselines. On the Sudoku task, we further compare with the domain-specific SAT-Net method. In Table 4, we find that our approach substantially outperforms all baselines across both evaluated discrete-space reasoning settings. In Sudoku, our approach generalizes substantially better than the SAT-Net model and the RRN model to the harder dataset consisting of fewer given Sudoku elements and is capable of obtaining an accuracy of roughly 62.1% compared to an accuracy of 3.2% obtained by SAT-Net and 28.6% by RRN. For cases in which our approach fails, we found that our approach sometimes erroneously assigns low energy to partially accurate answers. For example, there can be a Sudoku board that is not fully valid, but the model assigns lower energy to it than to the ground truth board. 

2. _Connectivity_ : In the graph connectivity task, the model is given the adjacency matrix of a graph (1 if there is an edge directly connecting two nodes). The task is to predict the connectivity matrix of the graph (1 if there is a path connecting two nodes). In literature (Dong et al., 2019), this task is an example that requires a dynamic number of reasoning “steps” (depending on the treewidth of the graph). Therefore, prior papers primarily focus on computing connectivity between nodes within _k_ -steps away. Our training and standard test sets contain graphs with at most 12 nodes and our harder dataset contains graphs with 18 nodes. Since 

**Qualitative Results.** We qualitatively visualize intermediate optimized samples across energy landscapes on Sudoku in Figure 6. Optimized boards are increasingly accurate in later landscapes. 

7 

**Iterative Reasoning through Energy Diffusion** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion_images/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion.pdf-0008-01.png)


**----- Start of picture text -----**<br>
Sudoku Performance vs Optimization Steps<br>100<br>90<br>80<br>70<br>60<br>50 Test Dataset<br>Harder Dataset<br>40<br>5 10 15 20<br>Optimization Steps Per Landscape<br>Sudoku Performance<br>**----- End of picture text -----**<br>


_Figure 6._ **Sudoku Performance with Optimization Steps** – Generalization to harder Sudoku problems significantly improves with a larger number of optimization steps in Sudoku. 

|**Gradient **|**Optimization **|**Contrastive**|**Same**|**Harder**|
|---|---|---|---|---|
|**Descent**|**Refnement**|**Shaping**|**Diffculty **|**Diffculty**|
|No|No|No|97.0%|35.0%|
|Yes|No|No|98.8%|45.1%|
|Yes|Yes|No|99.3%|59.7%|
|Yes|Yes|Yes|99.4%|62.1%|



_Table 5._ **Discrete Reasoning Ablations –** Ablations of proposed components of IRED on performance on the Sudoku task. Leveraging gradient descent to optimize energy functions, using multiple steps of optimization at each energy landscape and contrastively shaping the energy landscape with ground truth labels all improve the performance. 

**Performance with Computation.** In Figure 6, we assess the impact of the number of optimization steps at each energy landscape on the performance in Sudoku on both the test and harder datasets. We find that performance substantially improves on the harder dataset with an increased number of optimization steps with more modest improvement on the test datasets. By formulating reasoning as energy optimization, we can adaptively change the number of optimization steps dependent on difficulty of task, enabling us to generalize substantially better on harder tasks. 

**Extension to Visual Sudoku.** IRED can also be extended to deal with other input formats, such as images. To illustrate this, we conducte a new experiment on the Visual Sudoku dataset (Wang et al., 2019), where the board is not represented by one-hot vectors but now consists of MNIST digits written on a grid. We use a CNN to encode the image and fuse the image embeddings with the noisy answer to predict energy values. Shown in Table 4, we observed a similar performance advantage of our model compared to the baseline. 

**Ablation.** In Table 5, we ablate each component of IRED on the Sudoku task. Similar to the continuous setting, we find that each component of IRED, gradient based optimization, multiple steps of optimization, and contrastive energy shaping all lead to improved performance. While performance is modestly improved on the test dataset, it is 

||**Task**|**Method**|**Test**|**Harder**|
|---|---|---|---|---|
||||**Dataset**|**Dataset**|
||**Shortest Path**|IREM<br>Diffusion<br>IRED (ours)|90.4%<br>45.2%<br>**92.6%**|88.4%<br>46.9%<br>**91.9%**|



_Table 6._ **Planning Performance.** Test evaluation performance on the shortest path task. The harder tasks consists of graphs of size 25 while models are trained on graphs of size 15. 

||**Gradient **|**Optimization **|**Contrastive**|**Same**|**Harder**|
|---|---|---|---|---|---|
||**Descent**|**Refnement**|**Shaping**|**Diffculty **|**Diffculty**|
||No|No|No|80.8%|80.4%|
||Yes<br>Yes|No<br>Yes|No<br>No|80.7%<br>88.6%|79.1%<br>87.9%|
||Yes|Yes|Yes|**92.6%**|**91.9%**|



_Table 7._ **Path Planning Ablations –** Ablations of proposed components of IRED on performance on the path planning task. Using multiple steps of optimization at each energy landscape and contrastively shaping the energy landscape with ground truth labels both improve path planning performance. 

substantially improved on the harder generalization dataset with each added component. 

## **4.3. Planning** 

**Setup.** In this section, we evaluate IRED on a basic decision-making problem of finding the shortest path in a graph. In this task, the input to the model is the adjacency matrix of a directed graph, together with two additional node embeddings indicating the start and the goal node of the path-finding problem. The task is to predict a sequence of actions corresponding to the plan. Concretely, the output is a matrix of size [ _T, N_ ], where _T_ is the number of planning steps and _N_ is the number of nodes in the graph. Each entry ( _t, i_ ) has a value of 1 if the _t_ -th step of the shortest path is at node _i_ and has a value of 0 otherwise. For all models, including ours, we use a spatial-temporal graph convolution network (STGCN; Yan et al., 2018) to encode the adjacency matrix and the prediction and predict energy function values or score functions. In short, the STGCN has multiple layers. At each layer, for each node, it fuses all features of nodes that are connected to it in the graph and from the current timestep or adjacent timesteps. Just like standard graph convolutional networks, it uses a sum-pooling mechanism to aggregate all embeddings from adjacent nodes. Since, in practice, such planning models are generally evaluated with closed-loop execution, here we only evaluate the success rate that the first action produced by the model shortens the distance between the current node and the goal node. This is the same planning and execution strategy as DiffusionPolicy (Chi et al., 2023). 

**Quantitative Results.** We compare IRED with all baselines across settings in Table 6. IRED outperforms both 

8 

## **Iterative Reasoning through Energy Diffusion** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion_images/icml_2024_Learning_Iterative_Reasoning_through_Energy_Diffusion.pdf-0009-01.png)


**----- Start of picture text -----**<br>
Energy Minima Energy Minima Energy Minima<br>(Landscape 1) (Landscape 3) (Landscape 10)<br>**----- End of picture text -----**<br>


_Figure 7._ **Optimized Plans Across Landscapes** – Plot of next action prediction in plans across energy landscapes. In each visualization, the green/red nodes indicate start/goal nodes with connections between nodes indicated with arrows. The darkness of a node indicates the score for selecting the corresponding node as the next node to move to in the predicted plan. As landscapes are sequentially optimized, the correct next action is selected. 

baselines, especially the diffusion model, by a large margin. Both methods based on energy based formulation (IREM and IRED) perform well on this task, validating the hypothesis that learning energy functions is an effective method for encoding planning problems such as the edge constraints in the path-finding task. Finally, since all methods uses the same STGCN encoder, there is no significant performance drop on generalization to the harder dataset. 

**Qualitative Results.** Similarly to other tasks, we can also visualize the generated solutions by the model across different landscapes. Figure 7 visualizes the prediction of the first node to move to by the planning model. We normalize the prediction scores to 0 to 1. The darker the color, the higher the score. As can be seen in the figure, our IRED model is capable of gradually finding the immediate next action to take: at step 3 it excluded node 7, and the score gradually concentrates on node 8. 

**Ablations.** We ablate each component of IRED in Table 7. We found that running multiple steps of optimization and shaping the energy landscape both lead to improved performance on both the same-difficulty test cases and harder difficulty ones. 

## **5. Conclusion and Discussions** 

In this paper, we present IRED, a new approach to solving complex reasoning tasks by formulating it as an energy minimization process on a sequence of energy landscapes. We illustrate, on both continuous, discrete, and planning domains, how iterative computation utilizing IRED enables better algorithmic performance and generalization to more complex instances of problems. We further illustrate how the underlying algorithmic computation learned by IRED may be nested to implement more complex algorithmic computations. 

Our current reasoning approach with IRED has several limitations. First, the inference time optimization procedure in 

IRED can still be improved because currently, it requires many steps of gradient descent to find an energy minima. For tasks with known specifications (e.g., shortest path), IRED will conceivably run slower than the algorithms designed specifically for them (e.g., polynomial algorithms for pathfinding), although it is worth noting that IRED is a general machine learning algorithm that does not assume a given task specification and can automatically learn the underlying task constraints from data. On the other hand, it would be interesting to explore how an amortized neural network generator for generating initial solutions or guided optimizers can speed up this procedure. Second, our sequence of annealed energy landscapes is defined through a sequence of added Gaussian noise increments — it would be further interesting to learn the sequence of energy landscapes to enable adaptive optimization. Another current limitation of IRED is that out of the box, IRED in its current form does not leverage any additional memory. Therefore, for tasks that would benefit from explicitly using additional memory to store intermediate results (analogous to chainof-thought reasoning tasks in language or visual reasoning), IRED might not be as effective as other approaches. 

So far we have been applying IRED in continuous and discrete reasoning tasks, and planning on discrete spaces (graphs). Other potential applications of IRED include general mathematical reasoning (Amini et al., 2019; Lu et al., 2021) and decision-making in hybrid discrete-continuous spaces (Garrett et al., 2021; Yang et al., 2023b; Fang et al., 2023). For example, IRED can serve directly as a policy model. In our experiments, we have supervised the energy functions using IID samples from a fixed dataset (as positive examples). However, the energy function can also be supervised with reward signals by modeling a distribution of actions that favor actions yielding high discounted returns (similar to REINFORCE). 

## **Acknowledgements** 

We gratefully acknowledge support from NSF grant 2214177; from AFOSR grant FA9550-22-1-0249; from ONR MURI grant N00014-22-1-2740; and from ARO grant W911NF-23-1-0034; from MIT Quest for Intelligence; from the MIT-IBM Watson AI Lab; from ONR Science of AI; and from Simons Center for the Social Brain. Yilun Du is supported by a NSF Graduate Fellowship. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of our sponsors. 

## **Impact Statement** 

No immediate negative social impacts can be derived from the proposed approach in its current form as our evaluation is carried out on relatively standard simple datasets. 

9 

**Iterative Reasoning through Energy Diffusion** 

## **References** 

- Amini, A., Gabriel, S., Lin, S., Koncel-Kedziorski, R., Choi, Y., and Hajishirzi, H. MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms. In _NAACL_ , 2019. 

- Amos, B. and Kolter, J. Z. Optnet: Differentiable Optimization as a Layer in Neural Networks. In _ICML_ , 2017. 

- Anil, C., Pokle, A., Liang, K., Treutlein, J., Wu, Y., Bai, S., Kolter, J. Z., and Grosse, R. B. Path Independent Equilibrium Models Can Better Exploit Test-Time Computation. In _NeurIPS_ , 2022. 

- Arbel, M., Zhou, L., and Gretton, A. Generalized Energy Based Models. In _ICLR_ , 2021. 

- Bai, S., Kolter, J. Z., and Koltun, V. Deep Equilibrium Models. In _NeurIPS_ , 2019. 

- Banino, A., Balaguer, J., and Blundell, C. Pondernet: Learning to Ponder. In _ICML Workshop on Automated Machine Learning_ , 2021. 

- Bolukbasi, T., Wang, J., Dekel, O., and Saligrama, V. Adaptive Neural Networks for Efficient Inference. In _ICML_ , 2017. 

- Brockett, R. W. Dynamical Systems That Sort Lists, Diagonalize Matrices, and Solve Linear Programming Problems. _Linear Algebra and Its Applications_ , 146:79–91, 1991. 

- Bryson, A. E. _Applied Optimal Control: Optimization, Estimation and Control_ . Routledge, 2018. 

- Cai, J., Shin, R., and Song, D. Making Neural Programming Architectures Generalize via Recursion. In _ICLR_ , 2017. 

- Chen, X., Dai, H., Li, Y., Gao, X., and Song, L. Learning to Stop while Learning to Predict. In _ICML_ , 2020a. 

- Chen, X., Liang, C., Yu, A. W., Song, D., and Zhou, D. Compositional Generalization via Neural-Symbolic Stack Machines. In _NeurIPS_ , 2020b. 

- Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., and Song, S. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. In _RSS_ , 2023. 

- Chung, J., Ahn, S., and Bengio, Y. Hierarchical Multiscale Recurrent Neural Networks. In _ICLR_ , 2017. 

- Comas, A., Du, Y., Lopez, C. F., Ghimire, S., Sznaier, M., Tenenbaum, J. B., and Camps, O. Inferring Relational Potentials in Interacting Systems. In _ICML_ , 2023. 

- Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., and Kaiser, Ł. Universal Transformers. In _ICLR_ , 2019. 

- Djolonga, J. and Krause, A. Differentiable Learning of Submodular Models. In _NeurIPS_ , 2017. 

- Dong, H., Mao, J., Lin, T., Wang, C., Li, L., and Zhou, D. Neural Logic Machines. In _ICLR_ , 2019. 

- Donti, P. L., Amos, B., and Kolter, J. Z. Task-based Endto-End Model Learning in Stochastic Optimization. In _NeurIPS_ , 2017. 

- Du, Y. and Mordatch, I. Implicit generation and modeling with energy based models. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- Du, Y., Li, S., Tenenbaum, B. J., and Mordatch, I. Improved Contrastive Divergence Training of Energy Based Models. In _ICML_ , 2021. 

- Du, Y., Li, S., Tenenbaum, J., and Mordatch, I. Learning Iterative Reasoning through Energy Minimization. In _ICML_ , 2022. 

- Du, Y., Durkan, C., Strudel, R., Tenenbaum, J. B., Dieleman, S., Fergus, R., Sohl-Dickstein, J., Doucet, A., and Grathwohl, W. S. Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC. In _ICML_ , 2023. 

- Fang, X., Garrett, C. R., Eppner, C., Lozano-Perez, T., Kael-´ bling, L. P., and Fox, D. DiMSam: Diffusion Models as Samplers for Task and Motion Planning under Partial Observability. _arXiv:2306.13196_ , 2023. 

- Garrett, C. R., Chitnis, R., Holladay, R., Kim, B., Silver, T., Kaelbling, L. P., and Lozano-Perez,´ T. Integrated Task and Motion Planning. _Annual Review of Control, Robotics, and Autonomous Systems_ , 4(1):265–293, 2021. 

- Grathwohl, W., Wang, K.-C., Jacobsen, J.-H., Duvenaud, D., and Zemel, R. Cutting Out the Middle-Man: Training and Evaluating Energy-Based Models Without Sampling. In _ICML_ , 2020. 

- Graves, A. Adaptive Computation Time for Recurrent Neural Networks. _arXiv:1603.08983_ , 2016. 

- Graves, A., Wayne, G., and Danihelka, I. Neural Turing Machines. _arXiv:1410.5401_ , 2014. 

- Graves, A., Wayne, G., Reynolds, M., Harley, T., Danihelka, I., Grabska-Barwinska,´ A., Colmenarejo, S. G., Grefenstette, E., Ramalho, T., Agapiou, J., et al. Hybrid Computing Using a Neural Network with Dynamic External Memory. _Nature_ , 538(7626):471–476, 2016. 

- He, K., Zhang, X., Ren, S., and Sun, J. Deep Residual Learning for Image Recognition. In _CVPR_ , 2016. 

10 

**Iterative Reasoning through Energy Diffusion** 

- Ho, J., Jain, A., and Abbeel, P. Denoising Diffusion Probabilistic Models. In _NeurIPS_ , 2020. 

- Kaiser, Ł. and Sutskever, I. Neural GPUs Learn Algorithms. In _ICLR_ , 2016. 

- Kautz, H., Selman, B., and Hoffmann, J. SATPlan: Planning as Satisfiability. In _IPC_ , 2006. 

- LeCun, Y., Chopra, S., and Hadsell, R. A Tutorial on Energybased Learning, 2006. 

- Lu, P., Gong, R., Jiang, S., Qiu, L., Huang, S., Liang, X., and Zhu, S.-C. Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning. In _ACL_ , 2021. 

- Manhaeve, R., Dumancic, S., Kimmig, A., Demeester, T., and De Raedt, L. Deepproblog: Neural Probabilistic Logic Programming. In _NeurIPS_ , 2018. 

- Neelakantan, A., Le, Q. V., and Sutskever, I. Neural Programmer: Inducing Latent Programs with Gradient Descent. In _ICLR_ , 2015. 

- Palm, R., Paquet, U., and Winther, O. Recurrent Relational Networks. In _NeurIPS_ , 2018. 

- Reed, S. and De Freitas, N. Neural Programmer-Interpreters. In _ICLR_ , 2016. 

- Rocktaschel, T. and Riedel, S.¨ End-to-End Differentiable Proving. In _NeurIPS_ , 2017. 

   - Xiao, Z., Kreis, K., Kautz, J., and Vahdat, A. VAEBM: A Symbiosis Between Variational Autoencoders and Energy-based Models. In _ICLR_ , 2020. 

   - Xie, J., Lu, Y., Zhu, S.-C., and Wu, Y. A Theory of Generative Convnet. In _ICML_ . PMLR, 2016. 

   - Xie, J., Lu, Y., Gao, R., Zhu, S.-C., and Wu, Y. N. Cooperative training of descriptor and generator networks. _T-PAMI_ , 42(1):27–45, 2018. 

   - Xu, K., Li, J., Zhang, M., Du, S. S., Kawarabayashi, K.i., and Jegelka, S. What Can Neural Networks Reason About? In _ICLR_ , 2019. 

   - Yan, S., Xiong, Y., and Lin, D. Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. In _AAAI_ , 2018. 

   - Yang, F., Yang, Z., and Cohen, W. W. Differentiable Learning of Logical Rules for Knowledge Base Reasoning. In _NeurIPS_ , 2017. 

   - Yang, Z., Ishay, A., and Lee, J. NeurASP: Embracing Neural Networks into Answer Set Programming. In _IJCAI_ , 2020. 

   - Yang, Z., Ishay, A., and Lee, J. Learning to Solve Constraint Satisfaction Problems with Recurrent Transformer. In _ICLR_ , 2023a. 

   - Yang, Z., Mao, J., Du, Y., Wu, J., Tenenbaum, J. B., LozanoPerez, T., and Kaelbling, L. P.´ Compositional DiffusionBased Continuous Constraint Solvers. In _CoRL_ , 2023b. 

- Rubanova, Y., Sanchez-Gonzalez, A., Pfaff, T., and Battaglia, P. Constraint-Based Graph Network Simulator. In _ICML_ , 2022. 

- Schwarzschild, A., Borgnia, E., Gupta, A., Huang, F., Vishkin, U., Goldblum, M., and Goldstein, T. Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks. In _NeurIPS_ , 2021. 

- Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., and Ganguli, S. Deep Unsupervised Learning Using Nonequilibrium Thermodynamics. In _ICML_ , 2015. 

- Sun, Z. and Yang, Y. Difusco: Graph-Based Diffusion Solvers for Combinatorial Optimization. In _NeurIPS_ , 2023. 

- Wang, P.-W., Donti, P., Wilder, B., and Kolter, Z. SATNet: Bridging Deep Learning and Logical Reasoning using a Differentiable Satisfiability Solver. In _ICML_ , 2019. 

- Wilder, B., Dilkina, B., and Tambe, M. Melding the DataDecisions Pipeline: Decision-Focused Learning for Combinatorial Optimization. In _AAAI_ , 2019. 

11 

**Iterative Reasoning through Energy Diffusion** 

## **Supplementary Material for Iterative Reasoning through Energy Diffusion** 

In this appendix we provide additional details on IRED. We first provide experimental details on our evaluated tasks in Appendix A. Next, we discuss individual model architectures used in Appendix B. 

## **A. Experimental Details** 

**Continuous Tasks** We use dataset setups from (Du et al., 2022) for continuous tasks. Models were trained in approximately 2 hours on a single Nvidia RTX 2080 using a training batch size of 2048 and the Adam optimizer with learning rate 1e-4. Models was trained for approximately 50,000 iterations and evaluated on 20000 test problems. 

**Discrete Tasks** For Sudoku, we train models for 50000 iterations using a single Nvidia RTX 2080 using a training batch size of 64 with the Adam optimizer with learning rate 1e-4 and are evaluated on the full test datasets provided in (Wang et al., 2019; Palm et al., 2018). 

For Connectivity tasks, we generate random graphs using algorithms from Graves et al. (2016). Essentially, it first generates a set of random points on a 2D plane uniformly inside a unit square. Next, it samples the out-degree _k_ (the number of out-going edges) for each node based on a uniform distribution. Finally, it connects each node to its _k_ closest neighbors. The advantage of this generation process is that the generated graph will be close to a planar graph so that it can be easily visualized, and more importantly, it allows for fine-grained control of the connectivity of the graph by setting the out-degree range. In practice, we uniformly sample the out-degrees from [1 _, n/_ 2], where _n_ is the number of nodes in the graph. Based on the sampled adjacency matrix, we use the floyd-warshall algorithm to compute its connectivity matrix. Note that there are no distances associated with the edges ( _i.e_ ., all edges are of unit length). We train models for 100000 iterations using a single Nvidia RTX 2080 with batch size 512 with the Adam optimizer. 

## **B. Model Architectures** 

**Continuous Task.** For continuous tasks, we use the architecture in Table 8 to train both IRED and the IREM baseline. We use the architecture in Table 9 to parameterize the diffusion model baseline. 

**Discrete Task.** For Sudoku, we use the architecture in Table 10. It encodes the Sudoku board with a convolutional neural network with the residual connection design, borrowed from He et al. (2016). 

For Connectivity, we use the architecture adapted from Dong et al. (2019), as detailed in Table 11. It uses a relational neural network to fuse the connectivity information from neighboring nodes. 

**Planning Task.** For planning tasks, we use the architecture the same architecture as the connectivity task, as detailed in Table 12. To encode temporal information, at each layer, we stack the node embedding from the previous time step, the current step, and the next step to implement temporal convolution. This is equivalent to the spatial-temporal graph convolution networks (STGCN; Yan et al., 2018). 

We have also attached the code to reproduce all experiments in the supplementary material. 

**Planning Task** For planning, we use the same procedure as in the connectivity tasks to generate graphs. To create graphs with a large enough treewidth, we set the out-degree sample range to be [1 _, n/_ 5], where _n_ is the number of nodes. All edges are of unit length. We use the floyd-warshall algorithm to compute the pairwise shortest distance. This enables us to evaluate whether the first action predicted by the model shortens the distance between the current node and the goal. We train models for 100000 iterations using a single Nvidia RTX 2080 with batch size 512 with the Adam optimizer. 

12 

**Iterative Reasoning through Energy Diffusion** 

|Linear 512<br>Linear 512<br>Linear 512<br>Linear_→_1|NLM Arity=2, Hideen=64|
|---|---|
||NLM Arity=2, Hideen=64|
||Max-Pooling over All Node Features|
||Linear, 1|



_Table 8._ The model architecture for IRED on continuous-space algorithmic reasoning tasks 

_Table 12._ The model architecture for IRED and diffusion baselines on the shortest-path task. For the diffusion baseline, we simply remove the pooling layer and apply the same linear layer on all node embedding to predict the noise value for each entry. 

Linear 512 Linear 512 Linear 512 Linear _→_ Output Dim 

_Table 9._ The model architecture for diffusion baselines on continuous-space algorithmic reasoning tasks. 

3x3 Conv2D, 384 Resblock 384 Resblock 384 Resblock 384 Resblock 384 Resblock 384 Resblock 384 3x3 Conv2D, 9 

_Table 10._ The model architecture for IRED and diffusion baselines on the Sudoku task. The energy value is computed using the L2 norm of the final predicted output similar to (Du et al., 2023), while the output is directly used as noise prediction for the diffusion baseline. 

NLM Arity=3, Hidden=64 NLM Arity=3, Hidden=64 Max-Pooling over All Edge Features Linear, 1 

_Table 11._ The model architecture for IRED and diffusion baselines on the connectivity task. For the diffusion baseline, we simply remove the pooling layer and apply the same linear layer on all edge embeddings to predict the noise value for each entry. 

13 

