# **Offline Imitation Learning with Variational Counterfactual Reasoning** 

**Zexu Sun**[§] **, Bowei He**[†] **, Jinxin Liu**[‡] **, Xu Chen**[§][∗] **, Chen Ma**[†] **, Shuai Zhang**[¶] 

§Gaoling School of Artificial Intelligence, Renmin University of China 

†Department of Computer Science, City University of Hong Kong 

‡School of Engineering, Westlake University ¶DiDi Chuxing `{sunzexu21, xu.chen}@ruc.edu.cn` , `boweihe2-c@my.cityu.edu.hk liujinxin@westlake.edu.cn` , `chenma@cityu.edu.hk` , `shuai.zhang@tju.edu.cn` 

## **Abstract** 

In offline imitation learning (IL), an agent aims to learn an optimal expert behavior policy without additional online environment interactions. However, in many real-world scenarios, such as robotics manipulation, the offline dataset is collected from suboptimal behaviors without rewards. Due to the scarce expert data, the agents usually suffer from simply memorizing poor trajectories and are vulnerable to the variations in the environments, lacking the capability of generalizing to new environments. To automatically generate high-quality expert data and improve the generalization ability of the agent, we propose a framework named Offline Imitation Learning with Counterfactual data Augmentation (OILCA) by doing counterfactual inference. In particular, we leverage identifiable variational autoencoder to generate _counterfactual_ samples for expert data augmentation. We theoretically analyze the influence of the generated expert data and the improvement of generalization. Moreover, we conduct extensive experiments to demonstrate that our approach significantly outperforms various baselines on both DEEPMIND CONTROL SUITE benchmark for in-distribution performance and CAUSALWORLD benchmark for out-of-distribution generalization. Our code is available at `https://github. com/ZexuSun/OILCA-NeurIPS23` . 

## **1 Introduction** 

By utilizing the pre-collected expert data, imitation learning (IL) allows us to circumvent the difficulty in designing proper rewards for decision-making tasks and learning an expert policy. Theoretically, as long as adequate expert data are accessible, we can easily learn an imitator policy that maintains a sufficient capacity to approximate the expert behaviors [25, 30, 29, 31]. However, in practice, several challenging issues hinder its applicability to practical tasks. In particular, expert data are often limited, and due to the typical requisite for online interaction with the environment, performing such online IL may be costly or unsafe in real-world scenarios such as self-driving or industrial robotics [11, 34, 37]. Alternatively, in such settings, we might instead have access to large amounts of pre-collected unlabeled data, which are of unknown quality and may consist of both good-performing and poor-performing trajectories. For example, in self-driving tasks, a number of human driving behaviors may be available; in industrial robotics domains, one may have access to large amounts of robot data. The question then arises: can we perform offline IL with only limited expert data and the previously collected unlabeled data, thus relaxing the costly online IL requirements? 

Traditional behavior cloning (BC) [5] directly mimics historical behaviors logged in offline data (both expert data and unlabeled data) via supervised learning. However, in our above setting, BC 

∗Corresponding author 

37th Conference on Neural Information Processing Systems (NeurIPS 2023). 

suffers from unstable training, as it relies on sufficient high-quality offline data, which is unrealistic. Besides, utilizing unlabeled data indiscriminately will lead to severe catastrophes: Bad trajectories mislead policy learning, and good trajectories fail to provide strong enough guidance signals for policy learning. In order to better distinguish the effect of different quality trajectories on policy learning, various solutions are proposed correspondingly. For example, ORIL [39] learns a reward model to relabel previously collected data by contrasting expert and unlabeled trajectories from a fixed dataset; DWBC [36] introduces a discriminator-weighted task to assist the policy learning. However, such discriminator-based methods are prone to overfitting and suffer from poor generalization ability, especially when only very limited expert data are provided. 

Indeed, the expert data play an important role in offline IL and indicate the well-performing agent’s intention. Thus, how to fully understand the expert’s behaviors or preferences via limited available expert data becomes pretty crucial. In this paper, we propose to investigate counterfactual techniques for interpreting such expert behaviors. Specifically, we leverage the _variational counterfactual reasoning_ [23, 4] to augment the expert data, as typical imitation learning data augmentation 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0002-02.png)


**----- Start of picture text -----**<br>
What action might the  Test Envirionments<br>expert take if a different  In-distribution<br>Expert Data state is observed?<br>Generate Counterfactual Expert data<br>Unlabeled Data via Variational Counterfactual  Out-of-distribution<br>Reasoning<br>Exogenous Variable<br>**----- End of picture text -----**<br>


Figure 1: Agent is trained with the collected dataset containing limited expert data and large amounts of unlabeled data, and tested on both in-distribution and out-of-distribution environments. 

methods [3, 7] easily generate noisy data and inevitably bias the learning agent. Here, we conduct the counterfactual reasoning of the expert by answering the following question: 

## _“What action might the expert take if a different state is observed?”_ 

Throughout this paper, we consider the structure causal model (SCM) underlying the offline training data and introduce an exogenous variable that influences the states and yet is unobserved. This variable is utilized as the minimal edits to existing expert training data so that we can generate counterfactual states that cause an agent to change its behavior. Intuitively, this exogenous variable captures variations of features in the environment. By introducing additional variations in the states during training, we encourage the model to rely less on the idiosyncrasies of a given environment. In detail, we leverage an identifiable generative model to generate the counterfactual expert data, thus enhancing the agent’s capabilities of generalization in test environments (Figure 1). 

The main contributions of this paper are summarized as follows: 

- We propose a novel learning framework OILCA for offline IL. Using both training data and the augmentation model, we can generate counterfactual expert data and improve the generalization of the learned agent. 

- We theoretically analyze the disentanglement identifiability of the constructed exogenous variable and the influence of augmented counterfactual expert data via a sampler policy. We also guarantee the improvement of generalization ability from the perspective of error bound. 

- We conduct extensive experiments and provide related analysis. The empirical results about in-distribution performance on DEEPMIND CONTROL SUITE benchmark and out-of-distribution generalization on CAUSALWORLD benchmark both demonstrate the effectiveness of our method. 

## **2 Related Works** 

**Offline IL** A significant inspiration for this work grows from the offline imitation learning technique on how to learn policies from the demonstrations. Most of these methods take the idea of behavior cloning (BC) [5] that utilizes supervised learning to learn to act. However, due to the presence of suboptimal demonstrations, the performance of BC is limited to mediocre levels on many datasets. To address this issue, ORIL [39] learns a reward function and uses it to relabel offline trajectories. 

2 

However, it suffers from high computational costs and the difficulty of performing offline RL under distributional shifts. Trained on all data, BCND [26] reuses another policy learned by BC as the weight of the original BC objective, but its performance can even be worse if the suboptimal data occupies the major part of the offline dataset. LobsDICE [12] learns to imitate the expert policy via optimization in the space of stationary distributions. It solves a single convex minimization problem, which minimizes the divergence between the two-state transition distributions induced by the expert and the agent policy. CEIL [16] explicitly learns a hindsight embedding function together with a contextual policy. To achieve the expert matching objective for IL, CEIL advocates for optimizing a contextual variable such that it biases the contextual policy towards mimicking expert behaviors. DWBC [36] introduces an additional discriminator to distinguish expert and unlabeled demonstrations, and the outputs of the discriminator serve as the weights of the BC loss. CLUE [17] proposes to learn an intrinsic reward that is consistent with the expert intention via enforcing the embeddings of expert data to a calibrated contextual representation. OILCA aims to augment the the scarce expert data to improve the performance of the learned policy. 

**Causal Dynamics RL** Adopting this formalism allows one to cast several important problems within RL as questions of causal inference, such as off-policy evaluation [6, 22], learning baselines for model-free RL [20], and policy transfer [10, 15]. CTRL [19] applies SCM dynamics to the data augmentation in continuous sample spaces and discusses the conditions under which the generated transitions are uniquely identifiable counterfactual samples. This approach models state and action variables as unstructured vectors, emphasizing benefits in modeling action interventions for scenarios such as clinical healthcare where exploratory policies cannot be directly deployed. MOCODA [24] applies a learned locally factored dynamics model to an augmented distribution of states and actions to generate counterfactual transitions for RL. FOCUS [38] can reconstruct the causal structure accurately and illustrate the feasibility of learning causal structure in offline RL. OILCA uses the identifiable generative model to infer the distribution of the exogenous variable in the causal MDP, then performs the counterfactual data augmentation to augment the scarce expert data in offline IL. 

## **3 Preliminaries** 

## **3.1 Problem Definition** 

We consider the causal Markov Decision Process (MDP) [19] with an additive noise. In our problem setting, we have an offline static dataset consisting of _i.i.d_ tuples Dall = { _s[i] t[,a][i] t[,s][i] t_ +1[}] _ni_ =all1[s.t.][(] _[s][t][,a][t]_[) ∼] _ρ_ ( _st,at_ ) _,st_ +1 ∼ _fε_ ( _st,at,ut_ +1), where _ρ_ ( _st,at_ ) is an offline state-action distribution resulting from some behavior policies, _fε_ ( _st,at,ut_ +1) represents the causal transition mechanism, _ut_ +1 is the sample of the exogenous variable _u_ , which is unobserved, and _ε_ is a small permutation. Let D _E_ and D _U_ be the sets of expert and unlabeled demonstrations respectively, our goal is to only leverage the offline batch data Dall ∶= D _E_ ∪D _U_ to learn an optimal policy _π_ without any online interaction. 

## **3.2 Counterfactual Reasoning** 

We provide a brief background on counterfactual reasoning. Further details can be found in [23]. 

**Definition 1** (Structural Causal Model (SCM)) **.** _A structural causal model_ M _over variables_ **X** = { _X_ 1 _,...,Xn_ } _consists of a set of independent exogenous variables_ **U** = { **u** 1 _,...,_ **u** _n_ } _with prior distributions P_ ( **u** _i_ ) _and a set of functions f_ 1 _,...,fn such that Xi_ = _fi_ ( **PA** _i,_ **u** _i_ ) _, where_ **PA** _i_ ⊂ **X** _are parents of Xi. Therefore, the distribution of the SCM, which is denoted P_[M] _, is determined by the functions and the prior distributions of exogenous variables._ 

Inferring the exogenous random variables based on the observations, we can intervene in the observations and inspect the consequences. 

**Definition 2** _replacing some functions_ ( _do_ -intervention in SCM) _fi_ ( **PA** _i,_ **u** _i_ ) _with_ **.** _An intervention fi_ ( **PA**[˜] _i,_ **u** _i_ ) _I, where_ = _do_ ( _X_ **PA** ˜ _i_ ∶= _i fis the intervened parents ofi_ ( **PA**[˜] _i,_ **u** _i_ )) _is defined as Xi. The intervened SCM is indicated as_ M _[I] , and, consequently, its distribution is denoted as P_[M][;] _[I] ._ 

The counterfactual inference with which we can answer the _what if_ questions will be obtained in the following process: 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Exogenous variable Latent representation<br>Observations Counterfactuals<br>sample<br>u u<br>do -Intervention<br>(a) SCM of causal MDP (b) SCM of causal MDP with  do -intervention<br>**----- End of picture text -----**<br>


Figure 2: SCM of causal Markov Decision Process (MDP). We incorporate an exogenous variable in the SCM that is learned and utilized for counterfactual reasoning about _do_ -intervention. 

1. Infer the posterior distribution of exogenous variable _P_ ( **u** _i_ ∣ **X** = **x** ), where **x** is a set of observations. Replace the prior distribution _P_ ( **u** _i_ ) with the posterior distribution _P_ ( **u** _i_ ∣ **X** = **x** ) in the SCM. 

2. We denote the resulted SCM as M **x** and its distribution as _P_[M] **[x]** , perform an intervention _I_ on M **x** to reach _P_[M] **[x]**[;] _[I]_ . 

3. Return the output of _P_[M] **[x]**[;] _[I]_ as the counterfactual inference. 

**SCM representation of causal MDP** We encode causal MDP (under a policy _π_ ) into an SCM M. The SCM shown in Figure 2 consists of an exogenous variable _u_ and a set of functions that transmit the state _st_ , action _at_ and latent representation _ut_ +1 of _u_ to the next state _st_ +1 _e.g. st_ +1 ∼ _**f ε**_ ( _st,at,ut_ +1), where _**ε**_ is a small perturbation, and subsequently, to the next action _at_ +1, _e.g. at_ +1 ∼ _π_ (⋅∣ _st_ +1). To variableperform counterfactual inference on the SCM(the counterfactual next action _s_ ˜ _t,_ ˜ _at_ ). _u_ Specifically, we sample the latent representation, and then generate the counterfactual next state ˜ _at_ +1, _i.e. a_ ˜ _t_ +1 ∼ M _π_ ˆ(⋅∣, we intervene in the original parents pair _s_ ˜ _t_ +1), where ˜ _stu_ ˜+ _t_ 1+,1 _i.e._ from the distribution of exogenous _π_ ˆ is the sampler policy. _s_ ˜ _t_ +1 ∼ _**f ε**_ ( _s_ ˜ _t,_ ˜ _at,_ ˜ _ut_ +1 (), and also _st,at_ ) to 

## **3.3 Variational Autoencoder and Identifiability** 

We briefly introduce the Variational Autoencoder (VAE) and present its identifiability results. The VAE can be conceptualized as the amalgamation of a generative latent variable model and an associated inference model, both of which are parameterized by neural networks [13]. Specifically, the VAE learns the joint distribution _p_ _**θ**_ ( _**x** ,_ _**z**_ ) = _p_ _**θ**_ ( _**x**_ ∣ _**z**_ ) _p_ _**θ**_ ( _**z**_ ), where _p_ _**θ**_ ( _**x**_ ∣ _**z**_ ) represents the conditional distribution of observing _**x**_ given _**z**_ . Here, _**θ**_ denotes the set of generative parameters, and _p_ _**θ**_ ( _**z**_ ) = Π _[I] i_ =1 _[p]_ _**[θ]**_[ (] _[z][i]_[)][represents][the][factorized][prior][distribution][of][the][latent][variables.][By] incorporating an inference model _qϕ_ ( _**z**_ ∣ _**x**_ ), the parameters _**ϕ**_ and _**θ**_ can be jointly optimized by maximizing the evidence lower bound (ELBO) on the marginal likelihood _pθ_ ( _**x**_ ). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0004-08.png)


where _D_ KL denotes the KL-divergence between the approximation and the true posterior, and L is a lower bound of the marginal likelihood _p_ _**θ**_ ( _**x**_ ) because of the non-negativity of the KLdivergence. Recently, it has been shown that VAEs with unconditional prior distributions _p_ _**θ**_ ( _**z**_ ) are not identifiable [18], but the latent factors _**z**_ can be identified with a conditionally factorized prior distribution _p_ _**θ**_ ( _**x**_ ∣ _**z**_ ) over the latent variables to break the symmetry [9]. 

## **4 Offline Imitation Learning with Counterfactual Data Augmentation** 

At a high level, OILCA consists of following steps: (1) data augmentation via variational counterfactual reasoning, and (2) cooperatively learning a discriminator and a policy by using D _U_ and augmented D _E_ . In this section, we detail these two steps and present our method’s theoretical guarantees. 

4 

## **4.1 Counterfactual Data Augmentation** 

Our intuition is that the counterfactual expert data can model deeper relations between states and actions, which can help the learned policy be generalized better. Hence, in this section, we aim to generate counterfactual expert data by answering the _what if_ question in Section 1. Consequently, counterfactual expert data augmentation is especially suitable for some real-world settings, where executing policies during learning can be too costly or slow. 

As presented in Section 3.2, the counterfactual data can be generated by using the posterior of the exogenous variable obtained from the observations. Thus, we can briefly introduce the conditional VAE [13, 28] to build the latent representation of the exogenous variable _u_ . Moreover, Considering the identifiability of unsupervised disentangled representation learning [18], an additionally observed variable _c_ is needed, where _c_ could be, for example, the time index or previous data points in a time series, some kind of (possibly noisy) class label, or another concurrently observed variable [9]. Formally, let _**θ**_ = ( _**f** ,_ _**T** ,_ _**λ**_ ) be the parameters of the following conditional generative model: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0005-03.png)


where we first define: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0005-05.png)


Equation (2) describes the generative mechanism of _st_ +1 given the underlying exogenous variable _u_ , along with ( _st,at_ ). Equation (3) implies that the observed representation _st_ +1 is an additive noise function, _i.e._ , _st_ +1 = _**f** st,at_ ( _u_ ) + _**ε**_ where _**ε**_ is independent of _**f** st,at_ or _u_ . Moreover, this formulation also can be found in the SCM representation of causal MDP in Section 3.2, where we treat the function _f_ as the parameter _**f**_ of the model. 

Following standard conditional VAE [28] derivation, the evidence lower bound (ELBO) for each sample for _st_ +1 of the above generative model can be written as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0005-08.png)


Note that the ELBO in Equation (4) contains an additional term of the log _p_ ( _c_ ) that does not affect identifiability but improves the estimation for the conditional prior [21], we present the theoretical guarantees of disentanglement identifiability in Section 4.3. Moreover, the encoder _q_ _**ϕ**_ contains all the observable variables. The reason is that, as shown in Figure 2(a), from a causal perspective, _st,at_ and _u_ form a collider at _st_ +1, which means that when given _st_ +1, _st_ and _at_ are related to _u_ . 

We seek to augment the expert data in D _E_ . As shown in Figure 2(b), once the posterior of exogenous variablein the parentsin D _U_ . Moreover, we sample the latent representation _q_ _**ϕ**_ ( _u_ ( ∣ _sstt,a,att_ ) _,s_ of _t_ + _s_ 1 _,ct_ +1) in is obtained, we can perform D _E_ by re-sampling a different pair _u_ ˜ _t_ + _do_ 1 from the learned posterior distribution of-intervention. In particular, we intervene ( _s_ ˜ _t,_ ˜ _at_ ) from the collected data exogenous variable _u_ . Then utilizing ˜ _st_ , ˜ _at_ , and _u_ ˜ _t_ +1, we can generate the counterfactual next state according to Equation (3), which is denoted as ˜ _st_ +1. Subsequently, we pre-train a sampler policy _π_ ˆ _E_ with the originalgenerated tuples ( D _s_ ˜ _tE_ +1to sample the counterfactual next action _,_ ˜ _at_ +1), we constitute them in D _E_ . ˜ _at_ +1 = _π_ ˆ _E_ (⋅∣ _s_ ˜ _t_ +1). Thus, for all the 

## **4.2 Offline Agent Imitation Learning** 

In general, the counterfactual data augmentation of our method can enhance many offline IL methods. It is worth noting that as described in [40], discriminator-based methods are easy to be over-fitted, which can more directly show the importance and effectiveness of the counterfactual augmented data. Thus, in this section, also to best leverage the unlabeled data, we use Discriminator-Weighted Behavioral Cloning (DWBC) [36], a state-of-the-art discriminator-based offline IL method. This method introduces a unified framework to learn the policy and discriminator cooperatively. The discriminator training gets information from the policy _π_ _**ω**_ as additional input, yielding a new discriminating task whose learning objective is as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0005-13.png)


5 

**Algorithm 1** Training procedure of OILCA. 

**Input:** Dataset D _E_ , D _U_ , Dall, pre-trained sampler policy _π_ ˆ _E_ , hyperparameters _η_ , _α_ , initial variational couterfactual parameters _**θ** ,_ _**ϕ**_ , discriminator parameters _**ψ**_ , policy parameters _**ω**_ , data augmentation batch number _B_ . **Output:** Learned policy parameters _**ω**_ . 

1: **while** counterfactual training **do** ▷ Variational counterfactual reasoning 2: Sample ( _st,at,st_ +1 _,c_ ) ∼Dall to form a training batch 3: Update _**θ**_ and _**ϕ**_ according to Equation (4) 4: **end while** 

5: **for** _b_ = 1 to _B_ **do** ▷ Expert data augmentation 7:6:8: Generate the counterfactualSampleD _E_ ∪( _s_ ˜ ( _t_ + _s_ 1 _t,a,_ ˜ _att,s_ +1 _t_ )+1 _,c_ ) ∼D _E_ ˜ _s_ , _t_ (+1 _s_ ˜ according to Equation (2), then predict _t,_ ˜ _at_ ) ∼D _U_ to form an augmentation batch ˜ _at_ +1 = ˆ _πE_ (⋅∣ _s_ ˜ _t_ +1) 9: **end for** 

10: **while** agent training **do** ▷ Learning the discriminator and policy cooperatively 11: Sample ( _st,at_ ) ∼D _E_ and ( _s_[′] _t[,a]_[′] _t_[) ∼D] _[U]_[to form a training batch] 12: Update _**ψ**_ according Equation (5) every 100 training steps ▷ Discriminator learning 13: Update _**ω**_ according to Equation (6) every 1 training step ▷ Policy learning 

14: **end while** 

where _D_ _**ψ**_ is the discriminator, _η_ is called the class prior. In the previous works [35, 39], _η_ is a fixed hyperparameter often assigned as 0.5. 

Notice that now _π_ _**ω**_ appears in the input of _D_ _**ψ**_ , which means that imitation information from log _π_ _**ω**_ will affect L _**ψ**_ , and further impact the learning of _D_ _**ψ**_ . Thus, inspired by the idea of adversarial training, DWBC [36] introduces a new learning objective for BC Task: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0006-08.png)


where _d_ represents _D_ _**ψ**_ ( _st,at,_ log _π_ _**ω**_ ) for simplicity, _α_ is the weight factor ( _α_ > 1). The detailed training procedure of OILCA is shown in Algorithm 1. Moreover, we also present more possible combinations with other offline IL methods and the related results in Appendix F. 

## **4.3 Theoretical Analysis** 

In this section, we theoretically analyze our method, which mainly contains three aspects: (1) disentanglement identifiability, (2) the influence of the augmented data, and (3) the generalization ability of the learned policy. 

Our disentanglement identifiability extends the theory of iVAE [9]. To begin with, some assumptions are needed. Considering the SCM in Figure 2(a), we assume the data generation process in Assumption 1. 

**Assumption 1.** _(a) The distribution of the exogenous variable u is independent of time but dependent on the auxiliary variable c. (b) The prior distributions of exogenous variable u are different across auxiliary variable c. (c) The trasition mechanism p_ ( _st_ +1 ∣ _st,at,u_ ) _are invariant across different auxiliary variable c. (d) Given the exogenous variable u, the next state st_ +1 _is independent of the auxiliary variable c. i.e. st_ +1 � _c_ ∣ _u._ 

Part of the above assumption is also used in [19]; considering the identifiability of iVAE, we also add some necessary assumptions, which are also practical. In the following discussion, we will show that when the underlying data-generating mechanism satisfies Assumption 1, the exogenous variable _u_ can be identified up to permutation and affine transformations if the conditional prior distribution _p_ ( _u_ ∣ _c_ ) belongs to a general exponential family distribution. 

**Assumption 2.** _The prior distribution of the exogenous variable p_ ( _u_ ∣ _c_ ) _follows a general exponential family with its parameter specified by an arbitrary function_ _**λ**_ ( _c_ ) _and sufficient statistics_ _**T**_ ( _u_ ) = [ _**T d**_ ( _u_ ) _,_ _**T** NN_ ( _u_ )] _, here_ _**T d**_ ( _u_ ) _is defined by the concatenation of_ _**T d**_ ( _u_ ) = 

6 

[ _**T**_ 1( _u_[1] ) _[T] ,_ ⋯ _,_ _**T** d_ ( _u[d]_ ) _[T]_ ] _[T] from a factorized exponential family and the outputs of a neural network_ _**T** NN_ ( _u_ ) _with universal approximation power. The probability density can be written as:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0007-01.png)


Under the Assumption 2, and leveraging the ELBO in Equation (4), we can obtain the following identifiability of the parameters in the model. For convenience, we omit the subscript of _**f** st,at_ as _**f**_ . 

**Theorem 1.** _Assume that we observe data sampled from a generative model defined according to Equation_ (2) _-_ (3) _and Equation_ (7) _with parameters_ ( _**f** ,_ _**T** ,_ _**λ**_ ) _, the following holds:_ 

- _(i) The set_ { _st_ +1 ∈S ∶ _φ_ _**ε**_ ( _st_ +1 = 0)} _has measure zero, where φ_ _**ε** is the characteristic function of the density p_ _**ε** defined in Equation_ (3) _._ 

- _(ii) The function_ _**f** is injective and all of its second-order cross partial derivatives exist._ 

- _(iii) The sufficient statistics_ _**T d** are twice differentiable._ 

- _(iv) There exist k_ + 1 _distinct points c_[0] _,...,c[k] such that the matrix_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0007-08.png)


_of size k_ × _k is invertible._ 

_Then, the parameters_ _**θ**_ = ( _**f** ,_ _**T** ,_ _**λ**_ ) _are identifiable up to an equivalence class induced by permutation and component-wise transformations._ 

Theorem 1 guarantees the identifiability of Equation (2). We present its proof in Appendix A. 

Moreover, Theorem 1 further implies a consistent result on the conditional VAE. If the variational distribution of encoder _q_ _**ϕ**_ is a broad parametric family that includes the true posterior, we have the following results. 

**Theorem 2.** _Assume the following holds:_ 

- _(i) There exists the_ ( _**θ** ,_ _**ϕ**_ ) _such that the family of distributions q_ _**ϕ**_ ( _u_ ∣ _st,at,st_ +1 _,c_ ) _contains p_ _**θ**_ ( _u_ ∣ _st,at,st_ +1 _,c_ ) _._ 

- _(ii) We maximize_ L( _**θ** ,_ _**ϕ**_ ) _with respect to both_ _**θ** and_ _**ϕ** ._ 

_Then, given infinite data, OILCA can learn the true parameters_ _**θ**_[∗] ∶= ( _**f**_[∗] _,_ _**T**_[∗] _,_ _**λ**_[∗] ) _._ 

We present the corresponding proof in Appendix B. Theorem 2 is proved by assuming our conditional VAE is flexible enough to ensure the ELBO is tight for some parameters and the optimization algorithm can achieve the global maximum of ELBO. 

In our framework, the current generated expert sample pairs ( _s_ ˜ _t_ +1 _,_ ˜ _at_ +1) are estimated based on the sampler policy _π_ ˆ _E_ . However, _π_ ˆ _E_ may be not perfect, and its predicted results may contain noise. Thus, we would like to answer: “given the noise level of the sampler policy, how many samples one need to achieve sufficiently well performance?”. Using _κ_ ∈(0 _,_ 0 _._ 5) indicates the noise level of ˆ _πE_ . If ˆ _πE_ can exactly recover the true action _at_ +1 (i.e., _κ_ = 0 ), then the generated sequences are perfect ˆ without any noise. On the contrary, _κ_ = 0 _._ 5 means that _πE_ can only produce random results, and the generated sequences are fully noisy. Then we have the following theorem: 

**Theorem 3.** _Given a hypothesis class_ H _, for any ϵ,δ_ ∈(0 _,_ 1) _and κ_ ∈(0 _,_ 0 _._ 5) _, if π_ ˆ _E_ ∈H _is the pretrained policy model learned based on the empirical risk minimization (ERM), and the sample_ 2 log([2] _δ_[H][)] _complexity (i.e., number of samples) is larger than ϵ_[2] (1−2 _κ_ )[2] _[,][then][the][error][between][the][model] estimated and true results is smaller than ϵ with probability larger than_ 1 − _δ._ 

The related proof details are presented in Appendix C. From Theorem 3, we can see: in order to guarantee the same performance with a given probability (i.e., _ϵ_ and _δ_ are fixed), one needs to generate 2 log([2] _δ_[H][)] more than _ϵ_[2] (1−2 _κ_ )[2][sequences.][If the noise level] _[ κ]_[ is larger, more samples have to be generated.] Extremely, if the pre-trained policy can only produce fully noisy information (i.e., _κ_ = 0 _._ 5), then infinity number of samples are required, which is impossible in reality. 

7 

For the generalization ability, [8] explains the efficacy of counterfactual augmented data by the empirical evidence. In the context of offline IL, a straightforward yet very relevant conclusion from the analysis of generalization ability is the strong dependence on the number of expert data [25]. We work with finite state and action spaces (∣S∣ _,_ ∣A∣< ∞), and for the learned policy _π_ _**ω**_ , we can analyze the generalization ability from the perspective of error upper bound with the optimal expert policy _π_[∗] . 

**Theorem 4.** _Let_ ∣D _E_ ∣ _be the number of empirical expert data used to train the policy and ϵ be the expected upper bound of generalization error. There exists constant h such that, if_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0008-02.png)


_and each state st is sampled uniformly, then, with probability at least_ 1 − _δ, we have:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0008-04.png)


_which shows that increasing_ ∣D _E_ ∣ _drastically improves the generalization guarantee._ 

Note that we provide the proof details for Theorem 4 in Appendix D. 

## **5 Experiments** 

In this section, we evaluate the performance of OILCA, aiming to answer the following questions. **Q1:** With the synthetic toy environment of the causal MDP, can OILCA disentangle the latent representations of the exogenous variable? **Q2:** For an in-distribution test environment, can OILCA improve the performance? **Q3:** For an out-of-distribution test environment, can OILCA improve the generalization? In addition, we use five baseline methods: BC-exp (BC on the expert data D _E_ ), BC-all (BC on all demonstrations Dall), ORIL [39], BCND [26], LobsDICE [12], DWBC [36]. More details about these methods are presented in Appendix E. Furthermore, the hyper-parameters of our method and baselines are all detail-tuned for better performance. 

## **5.1 Simulations on Toy Environment (Q1)** 

In real-world situations, we can hardly get the actual distributions of the exogenous variable. To evaluate the disentanglement identifiability in variational counterfactual reasoning, we build a toy environment to show that our method can indeed get the disentangled distributions of the exogenous variable. 

**Toy environment** For the toy environment, we consider a simple 2D navigation task. The agent can move a fixed distance in each of the four directions. States are continuous and are considered as the 2D position of the agent. The goal is to navigate to a specific target state within a bounding box. The reward is the negative distance between the agent’s state and the target state. For the initialization of the environment, we consider _C_ kinds of Gaussian distributions (Equation (7)) for the exogenous variable, where _C_ = 3, and the conditional variable _c_ is the class label. We design the transition function by using a multi-layer perceptron (MLP) with invertible activations. We present the details about data collection and model learning in Appendix E. 

**Results** We visualize the identifiability in such a 2D case in Figure 3, where we plot the sources, the posterior distributions learned by OILCA, and the posterior distributions learned by a vanilla conditional VAE, respectively. Our method recovers the original sources to trivial indeterminacies (rotation and sign flip), whereas the vanilla conditional VAE fails to separate the latent variables well. To show the effectiveness of our method in the constructed toy environment, we also conduct repeated experiments for 1K episodes (500 steps per episode) to compare OILCA with all baseline methods. The results are presented in Figure 4, showing that OILCA achieves the highest average return. To analyze the influence of the number of augmented samples (Theorem 4), we also conduct the experiments with varying numbers of counterfactual expert data; the result is shown in Figure 5. The X-axis represents the percentage of ∣D _E_ ∣/∣D _U_ ∣, where D _E_ is the augmented expert data. We can observe that within a certain interval, the generalization ability of the learned policy does improve with the increase of ∣D _E_ ∣. 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0009-00.png)


**----- Start of picture text -----**<br>
14 13.0<br>12 12.5<br>12.0<br>10<br>11.5<br>8 11.0<br>6 10.5<br>4 10.0<br>2 9.5<br>(a) (b) (c) 0 BC-exp BC-all ORIL BCND LobsDICE DWBC OILCA 9.0 10% 30% Percentage50% 70% 90%<br>Average Return Average Return<br>**----- End of picture text -----**<br>


Figure 3: Visualization of both observation and latent spaces of the exogenous variable. (a) Samples from the true distribution of the sources _p_ _**θ**_[∗] ( _u_ ∣ _c_ ). (b) Samples from the posterior _q_ _**ϕ**_ ( _u_ ∣ _st,at,st_ +1 _,c_ ). (c) Samples from the posterior _q_ _**ϕ**_ **[′]** ( _u_ ∣ _st,at,st_ +1) without class label. 

Figure 4: Performance Figure 5: Performance of of OILCA and baselines OILCA with the growing in the toy environment. percentage of ∣D _E_ ∣/∣D _U_ ∣. We plot the mean and the We plot the average restandard errors of the avturn’s mean and standard eraged return over five errors over five random random seeds. seeds. 

Table 1: Results for in-distribution performance on DEEPMIND CONTROL SUITE. We report the average return of episodes (with the length of 1K steps) over five random seeds. The best results and second best results are **bold** and underlined, respectively. 

|`Task Name`|BC-exp<br>BC-all<br>ORIL<br>BCND<br>LobsDICE|BC-exp<br>BC-all<br>ORIL<br>BCND<br>LobsDICE|DWBC<br>**OILCA**|
|---|---|---|---|
|CARTPOLESWINGUP<br>CHEETAHRUN<br>FINGERTURNHARD<br>FISHSWIM<br>HUMANOIDRUN<br>MANIPULATORINSERTBALL<br>MANIPULATORINSERTPEG<br>WALKERSTAND<br>WALKERWALK|195.44<br>±7.39<br>269.03<br>±7.06<br>221.24<br>±14.49<br>243.52<br>±11.33<br>66.59<br>±11.09<br>90.01<br>±31.74<br>45.08<br>±9.88<br>96.06<br>±16.15<br>129.20<br>±4.51<br>104.56<br>±8.32<br>185.57<br>±26.75<br>204.67<br>±13.18<br>74.59<br>±11.73<br>68.87<br>±11.93<br>84.90<br>±1.96<br>153.28<br>±19.29<br>77.59<br>±8.63<br>138.93<br>±9.14<br>96.88<br>±10.76<br>257.01<br>±11.21<br>91.61<br>±7.49<br>98.59<br>±1.62<br>98.25<br>±2.68<br>141.71<br>±15.30<br>92.02<br>±15.04<br>119.63<br>±6.37<br>105.86<br>±5.10<br>220.66<br>±15.14<br>169.14<br>±8.00<br>192.14<br>±37.91<br>181.23<br>±10.31<br>279.66<br>±12.69<br>29.44<br>±2.18<br>75.79<br>±6.34<br>41.43<br>±4.05<br>157.44<br>±9.31|292.96<br>±11.05<br>74.53<br>±7.75<br>190.93<br>±12.19<br>188.84<br>±11.28<br>120.87<br>±10.66<br>197.79<br>±1.98<br>299.19<br>±3.40<br>252.34<br>±7.73<br>102.14<br>±5.94|382.55<br>±8.95<br>**608.38**<br>±35.54<br>66.87<br>±4.60<br>**116.05**<br>±14.65<br>243.47<br>±17.12<br>**298.73**<br>±5.11<br>212.39<br>±7.62<br>**290.28**<br>±10.07<br>302.33<br>±14.53<br>**460.96**<br>±17.55<br>107.86<br>±15.01<br>**296.83**<br>±3.43<br>238.39<br>±19.76<br>**305.66**<br>±4.91<br>280.07<br>±5.79<br>**395.51**<br>±8.05<br>166.95<br>±10.68<br>**377.19**<br>±15.87|



## **5.2 In-distribution Performance (Q2)** 

We use DEEPMIND CONTROL SUITE [32] to evaluate the performance of in-distribution performance. Similar to [39], we also define an episode as positive if its episodic reward is among the top 20% episodes for the task. For the auxiliary variable _c_ , we add three kinds of different Gaussian noise distributions into the environment (encoded as _c_ = {0 _,_ 1 _,_ 2}). Moreover, we present the detailed data collection and statistics in Appendix E.1). We report the results in Table 1. We can observe that OILCA outperforms baselines on all tasks; this shows the effectiveness of the augmented counterfactual expert data. We also find that the performance of ORIL and DWBC tends to decrease in testing for some tasks (ORIL: CHEETAH RUN, WALKER WALK; DWBC: CHEETAH RUN, MANIPULATOR INSERT BALL); this “overfitting” phenomenon also occurs in experiments of previous offline RL works [14, 33]. This is perhaps due to limited data size and model generalization bottleneck. DWBC gets better results on most tasks, but for some tasks, such as Manipulator Insert Ball and the Walker Walk, OILCA achieves more than twice the average return than DWBC. 

## **5.3 Out-of-distribution Generalization (Q3)** 

To evaluate the out-of-distribution generalization of OILCA, we use a benchmark named CAUSALWORLD [2], which provides a combinatorial family of tasks with a common causal structure and 

9 

Table 2: Results for out-of-distribution generalization on CAUSALWORLD. We report the average return of episodes (length varies for different tasks) over five random seeds. All the models are trained on _space_ **A** and tested on _space_ **B** to show the out-of-distribution performance [2]. 

|`Task Name`||BC-exp|BC-all|ORIL<br>BCND<br>LobsDICE|DWBC<br>**OILCA**|
|---|---|---|---|---|---|
|REACHING<br>PUSHING<br>PICKING<br>PICK ANDPLACE<br>STACKING2<br>TOWERS<br>STACKEDBLOCKS<br>CREATIVESTACKEDBLOCKS<br>GENERAL||281.18<br>±16.45<br>256.64<br>±12.70<br>270.01<br>±13.13<br>294.06<br>±7.34<br>496.63<br>±7.68<br>667.81<br>±9.27<br>581.91<br>±26.92<br>496.32<br>±26.92<br>492.78<br>±17.64|176.54<br>±9.75<br>235.58<br>±10.23<br>258.54<br>±16.53<br>225.42<br>±12.44<br>394.91<br>±16.98<br>784.88<br>±17.17<br>452.88<br>±18.78<br>529.82<br>±31.01<br>547.32<br>±8.49|339.40<br>±12.98<br>228.33<br>±7.14<br>243.29<br>±9.84<br>283.91<br>±19.72<br>191.23<br>±12.64<br>206.44<br>±15.35<br>388.15<br>±19.21<br>221.89<br>±7.68<br>337.78<br>±12.09<br>270.75<br>±14.87<br>259.12<br>±8.01<br>266.09<br>±10.31<br>388.55<br>±10.93<br>339.18<br>±9.46<br>362.47<br>±17.05<br>655.96<br>±15.14<br>139.30<br>±18.22<br>535.74<br>±13.76<br>702.15<br>±15.30<br>341.97<br>±33.69<br>250.44<br>±14.08<br>882.27<br>±46.79<br>288.55<br>±19.63<br>317.95<br>±32.03<br>647.95<br>±24.39<br>195.06<br>±9.80<br>458.27<br>±18.69|479.92<br>±18.75<br>**976.60**<br>±20.13<br>298.09<br>±14.94<br>**405.08**<br>±24.03<br>366.26<br>±8.77<br>**491.09**<br>±6.44<br>349.66<br>±7.39<br>**490.24**<br>±11.69<br>481.07<br>±10.11<br>**831.82**<br>±11.78<br>768.68<br>±24.77<br>**994.82**<br>±5.76<br>1596.96<br>±81.84<br>**2617.71**<br>±88.07<br>700.23<br>±13.71<br>**1348.49**<br>±55.05<br>585.98<br>±19.25<br>**891.14**<br>±23.12|



underlying factors. Furthermore, the auxiliary variable _c_ represents the different _do_ -interventions on the task environments [2]. We collect the offline data by using three different _do_ -interventions on environment features ( `stage_color` , `stage_friction` . `floor_friction` ) to generate offline datasets, while other features are set as the defaults. More data collection and statistics details are presented in Appendix E.1. We show the comparative results in Table 2. It is evident that OILCA also achieves the best performance on all the tasks; ORIL and DWBC perform the second-best results on most tasks. BCND performs poorly compared to other methods. The reason may be that the collected data in space **A** can be regarded as low-quality data when evaluating on space **B** , which may lead to the poor performance of a single BC and cause cumulative errors for the BC ensembles of BCND. LobsDICE even performs poorer than BC-exp on most tasks. This is because the KL regularization, which regularizes the learned policy to stay close to D _U_ is too conservative, resulting in a suboptimal policy, especially when D _U_ contains a large collection of noisy data. This indeed hurts the performance of LobsDICE for out-of-distribution generalization. 

## **6 Conclusion** 

In this paper, we propose an effective and generalizable offline imitation learning framework OILCA, which can learn a policy from the expert and unlabeled demonstrations. We apply a novel identifiable counterfactual expert data augmentation approach to facilitate the offline imitation learning. We also analyze the influence of generated data and the generalization ability theoretically. The experimental results demonstrate that our method achieves better performance in simulated and public tasks. 

## **Acknowledgement** 

This work is supported in part by National Key R&D Program of China (2022ZD0120103), National Natural Science Foundation of China (No. 62102420), Beijing Outstanding Young Scientist Program NO. BJJWZYJH012019100020098, Intelligent Social Governance Platform, Major Innovation & Planning Interdisciplinary Platform for the “Double-First Class” Initiative, Renmin University of China, Public Computing Cloud, Renmin University of China, fund for building world-class universities (disciplines) of Renmin University of China, Intelligent Social Governance Platform. 

## **References** 

> [1] Alekh Agarwal, Nan Jiang, Sham M. Kakade, and Wen Sun. Reinforcement learning: Theory and algorithms, 2021. 

10 

- [2] Ossama Ahmed, Frederik Träuble, Anirudh Goyal, Alexander Neitz, Yoshua Bengio, Bernhard Schölkopf, Manuel Wüthrich, and Stefan Bauer. Causalworld: A robotic manipulation benchmark for causal structure and transfer learning. _arXiv preprint arXiv:2010.04296_ , 2020. 

- [3] Dafni Antotsiou, Carlo Ciliberto, and Tae-Kyun Kim. Adversarial imitation learning with trajectorial augmentation and correction. In _2021 IEEE International Conference on Robotics and Automation (ICRA)_ , pages 4724–4730. IEEE, 2021. 

- [4] Léon Bottou, Jonas Peters, Joaquin Quiñonero-Candela, Denis X Charles, D Max Chickering, Elon Portugaly, Dipankar Ray, Patrice Simard, and Ed Snelson. Counterfactual reasoning and learning systems: The example of computational advertising. _Journal of Machine Learning Research_ , 14(11), 2013. 

- [5] Ivan Bratko, Tanja Urbanˇciˇc, and Claude Sammut. Behavioural cloning: phenomena, results and problems. _IFAC Proceedings Volumes_ , 28(21):143–149, 1995. 

- [6] Lars Buesing, Theophane Weber, Yori Zwols, Sebastien Racaniere, Arthur Guez, Jean-Baptiste Lespiau, and Nicolas Heess. Woulda, coulda, shoulda: Counterfactually-guided policy search. _arXiv preprint arXiv:1811.06272_ , 2018. 

- [7] Satoshi Hoshino, Tomoki Hisada, and Ryota Oikawa. Imitation learning based on data augmentation for robotic reaching. In _2021 60th Annual Conference of the Society of Instrument and Control Engineers of Japan (SICE)_ , pages 417–424. IEEE, 2021. 

- [8] Divyansh Kaushik, Amrith Setlur, Eduard Hovy, and Zachary C Lipton. Explaining the efficacy of counterfactually augmented data. _arXiv preprint arXiv:2010.02114_ , 2020. 

- [9] Ilyes Khemakhem, Diederik Kingma, Ricardo Monti, and Aapo Hyvarinen. Variational autoencoders and nonlinear ica: A unifying framework. In _International Conference on Artificial Intelligence and Statistics_ , pages 2207–2217. PMLR, 2020. 

- [10] Taylor W Killian, Marzyeh Ghassemi, and Shalmali Joshi. Counterfactually guided policy transfer in clinical settings. In _Conference on Health, Inference, and Learning_ , pages 5–31. PMLR, 2022. 

- [11] Beomjoon Kim and Joelle Pineau. Socially adaptive path planning in human environments using inverse reinforcement learning. _International Journal of Social Robotics_ , 8:51–66, 2016. 

- [12] Geon-Hyeong Kim, Jongmin Lee, Youngsoo Jang, Hongseok Yang, and Kee-Eung Kim. Lobsdice: offline imitation learning from observation via stationary distribution correction estimation. _arXiv preprint arXiv:2202.13536_ , 2022. 

- [13] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. _arXiv preprint arXiv:1312.6114_ , 2013. 

- [14] Aviral Kumar, Justin Fu, Matthew Soh, George Tucker, and Sergey Levine. Stabilizing offpolicy q-learning via bootstrapping error reduction. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- [15] Yao Lai, Jinxin Liu, Zhentao Tang, Bin Wang, Jianye Hao, and Ping Luo. Chipformer: Transferable chip placement via offline decision transformer. _arXiv preprint arXiv:2306.14744_ , 2023. 

- [16] Jinxin Liu, Li He, Yachen Kang, Zifeng Zhuang, Donglin Wang, and Huazhe Xu. Ceil: Generalized contextual imitation learning. _arXiv preprint arXiv:2306.14534_ , 2023. 

- [17] Jinxin Liu, Lipeng Zu, Li He, and Donglin Wang. Clue: Calibrated latent guidance for offline reinforcement learning. _arXiv preprint arXiv:2306.13412_ , 2023. 

- [18] Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Schölkopf, and Olivier Bachem. Challenging common assumptions in the unsupervised learning of disentangled representations. In _international conference on machine learning_ , pages 4114– 4124. PMLR, 2019. 

11 

- [19] Chaochao Lu, Biwei Huang, Ke Wang, José Miguel Hernández-Lobato, Kun Zhang, and Bernhard Schölkopf. Sample-efficient reinforcement learning via counterfactual-based data augmentation. _arXiv preprint arXiv:2012.09092_ , 2020. 

- [20] Thomas Mesnard, Théophane Weber, Fabio Viola, Shantanu Thakoor, Alaa Saade, Anna Harutyunyan, Will Dabney, Tom Stepleton, Nicolas Heess, Arthur Guez, et al. Counterfactual credit assignment in model-free reinforcement learning. _arXiv preprint arXiv:2011.09464_ , 2020. 

- [21] Graziano Mita, Maurizio Filippone, and Pietro Michiardi. An identifiable double vae for disentangled representations. In _International Conference on Machine Learning_ , pages 7769– 7779. PMLR, 2021. 

- [22] Michael Oberst and David Sontag. Counterfactual off-policy evaluation with gumbel-max structural causal models. In _International Conference on Machine Learning_ , pages 4881–4890. PMLR, 2019. 

- [23] Judea Pearl. Models, reasoning and inference. _Cambridge, UK: CambridgeUniversityPress_ , 19 (2), 2000. 

- [24] Silviu Pitis, Elliot Creager, Ajay Mandlekar, and Animesh Garg. Mocoda: Model-based counterfactual data augmentation. _arXiv preprint arXiv:2210.11287_ , 2022. 

- [25] Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In _Proceedings of the fourteenth international conference on artificial intelligence and statistics_ , pages 627–635. JMLR Workshop and Conference Proceedings, 2011. 

- [26] Fumihiro Sasaki and Ryota Yamashina. Behavioral cloning from noisy demonstrations. In _International Conference on Learning Representations_ , 2021. 

- [27] Shai Shalev-Shwartz and Shai Ben-David. _Understanding machine learning: From theory to algorithms_ . Cambridge university press, 2014. 

- [28] Kihyuk Sohn, Honglak Lee, and Xinchen Yan. Learning structured output representation using deep conditional generative models. _Advances in neural information processing systems_ , 28, 2015. 

- [29] Jonathan Spencer, Sanjiban Choudhury, Arun Venkatraman, Brian Ziebart, and J Andrew Bagnell. Feedback in imitation learning: The three regimes of covariate shift. _arXiv preprint arXiv:2102.02872_ , 2021. 

- [30] Wen Sun, Anirudh Vemula, Byron Boots, and Drew Bagnell. Provably efficient imitation learning from observation alone. In _International conference on machine learning_ , pages 6036–6045. PMLR, 2019. 

- [31] Gokul Swamy, Sanjiban Choudhury, J Andrew Bagnell, and Steven Wu. Of moments and matching: A game-theoretic framework for closing the imitation gap. In _International Conference on Machine Learning_ , pages 10022–10032. PMLR, 2021. 

- [32] Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al. Deepmind control suite. _arXiv preprint arXiv:1801.00690_ , 2018. 

- [33] Yifan Wu, George Tucker, and Ofir Nachum. Behavior regularized offline reinforcement learning. _arXiv preprint arXiv:1911.11361_ , 2019. 

- [34] Zheng Wu, Liting Sun, Wei Zhan, Chenyu Yang, and Masayoshi Tomizuka. Efficient samplingbased maximum entropy inverse reinforcement learning with application to autonomous driving. _IEEE Robotics and Automation Letters_ , 5(4):5355–5362, 2020. 

- [35] Danfei Xu and Misha Denil. Positive-unlabeled reward learning. In _Conference on Robot Learning_ , pages 205–219. PMLR, 2021. 

12 

- [36] Haoran Xu, Xianyuan Zhan, Honglei Yin, and Huiling Qin. Discriminator-weighted offline imitation learning from suboptimal demonstrations. In _International Conference on Machine Learning_ , pages 24725–24742. PMLR, 2022. 

- [37] Zhenting Zhao, Bowei He, Wenhao Luo, and Rui Liu. Collective conditioned reflex: A bioinspired fast emergency reaction mechanism for designing safe multi-robot systems. _IEEE Robotics and Automation Letters_ , 7(4):10985–10990, 2022. 

- [38] Zheng-Mao Zhu, Xiong-Hui Chen, Hong-Long Tian, Kun Zhang, and Yang Yu. Offline reinforcement learning with causal structured world models. _arXiv preprint arXiv:2206.01474_ , 2022. 

- [39] Konrad Zolna, Alexander Novikov, Ksenia Konyushkova, Caglar Gulcehre, Ziyu Wang, Yusuf Aytar, Misha Denil, Nando de Freitas, and Scott Reed. Offline learning from demonstrations and unlabeled experience. _arXiv preprint arXiv:2011.13885_ , 2020. 

- [40] Konrad Zolna, Scott Reed, Alexander Novikov, Sergio Gomez Colmenarejo, David Budden, Serkan Cabi, Misha Denil, Nando de Freitas, and Ziyu Wang. Task-relevant adversarial imitation learning. In _Conference on Robot Learning_ , pages 247–263. PMLR, 2021. 

13 

## **A Proof of Theorem 1** 

In this section, we provide proof for the disentanglement identifiability of the inferred exogenous variable. Our proof consists of three main components. It is worth noting that we also use _**f**_ to replace _**f** st,at_ for simplicity. 

## _Proof._ The following are the three steps: 

**Step I** We use the first assumption in Theorem 1 to demonstrate that the observed data distributions are equivalent to the noiseless distributions. Specifically, suppose that we have two sets of parameters ( _**f** ,_ _**T** ,_ _**λ**_ ) and ( _**f**_[˜] _,_ _**T**_[˜] _,_ _**λ**_[˜] ), such that for all pairs ( _st_ +1 _,c_ ) (( _s,c_ ) for simplicity), we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0014-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0014-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0014-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0014-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0014-08.png)


where: 

- in Equation (14), _J_ denotes the Jacobian, and we make the change of variable _s_ = _**f**_ ( _u_ ) on the left-hand side, and _s_ = _**f**_[˜] ( _u_ ) on the right-hand side. 

- in Equation (15), we introduce 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0014-12.png)


- in Equation (16), ∗ denotes the convolution operator. 

- in Equation (17), _F_ denotes the Fourier transformation and _φ_ _**ε**_ = _F_ [ _p_ _**ε**_ ]. 

- in Equation (18), _φ_ _**ε**_ ( _**w**_ ) is dropped because it is non-zero almost everywhere according to the first assumption of Theorem 1. 

Equation (19) is valid for all ( _s,c_ ). What is basically says is that for the distributions to be the same after adding the noise, the noise-free distributions have to be the same. Note that _s_ here is a general variable, and we are actually dealing with the noise-free probability densities. 

**Step II** Using Equation (20) to substitute Equation (19), we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0014-18.png)


Then, we can apply logarithm on the above equation and substitute _p_ _**T** ,_ _**λ**_ with its definition in Equation (3), and obtain 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0014-20.png)


14 

Let _c_[0] _,_ ⋯ _,c[k]_ be the _k_ + 1 points defined in the fourth assumption of Theorem 1, we can obtain _k_ + 1 equation. By subtracting the first equation from the remaining _k_ equations, we then obtain: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0015-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0015-02.png)


Finally, we multiply both side by _L_[−] _[T]_ and obtain 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0015-04.png)


where _A_ = _L_[−] _[T] L_ and _**n**_ = _L_[−] _[T]_ _**m**_ . 

**Step III** Now recall the definition of _**T**_ and the third assumption. We start by evaluating Equation (25) at _k_ + 1 points of _u[l] ,s[l]_ and obtain _k_ + 1 equations. Then, we subtract the first equation from the remaining _k_ + 1 equations: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0015-07.png)


Next, we only need to show that for _u_ 0, there exist _k_ points _u_[1] _,_ ⋯ _,u[k]_ such that the columns are linear independent, which can be proven by contradiction. Suppose that there exists no such _u[l]_ ∈{ _u_[0] _,_ ⋯ _,u[k]_ }, then ⟨ _**T**_ ( _u[l]_ ) − _**T**_ ( _u_[0] ) _,_ _**λ**_ ⟩= 0 and thus _**T**_ ( _u[l]_ ) = _**T**_ ( _u_[0] ) = const. This contradicts with the assumption that the prior distribution is strongly exponential. Therefore, there must exist _k_ + 1 points such that the transformation is invertible. Then we have ( _**f** ,_ _**T** ,_ _**λ**_ ) ∼( _**f**_[˜] _,_ _**T**_[˜] _,_ _**λ**_[˜] ). 

## **B Proof of Theorem 2** 

According to Equation (4), if the family _q_ _**ϕ**_ ( _u_ ∣ _st,at,st_ +1 _,c_ ) is large enough to include _p_ _**θ**_ ( _u_ ∣ _st_ +1 _,st,at,c_ ), then by optimizing the loss over its parameter _**ϕ**_ , we will minimize the KL term, eventually reaching zero, and the loss will be equal to the log-likelihood. 

The conditional VAE, in this case, inherits all the properties of maximum likelihood estimation. In this particular case, since our identifiability is guaranteed up to equivalence classes, the consistency of MLE means that we converge to the equivalence class (Theorem 1) of true parameter _**θ**_[∗] _i.e_ . Under the condition of infinite data. 

## **C Proof of Theorem 3** 

Suppose the prediction error of _π_ ˆ _E_ is _e_ ( _i.e_ ., ∑ I( _a[π] t_[ˆ] _[E]_ ≠ _at_ ) = _e_ ) _,at_ is the true action that an expert take, then the mismatching probability between the observed and predicted results comes from two parts: (1) The observed result is true, but the prediction is wrong, that is, _e_ (1 − _κ_ ). (2) The observed result is wrong, but the prediction is right, that is (1 − _e_ ) _κ_ . Thus, the total mismatching probability is _κ_ + _e_ (1 − 2 _κ_ ). 

The following proof is based on the reduction to absurdity. We first propose an assumption and then derive contradicts to invalidate the assumption. 

**Assumption.** Suppose the prediction error of _π_ ˆ _E_ (i.e., _e_ ) is larger than _ϵ_ . Then, at least one of the following statements hold: 

(1) The empirical mis-matching rate of _π_ ˆ _E_ is smaller than _κ_ + _[ϵ]_[(][1][−] 2[2] _[κ]_[)] . 

15 

- (2) The empirical mis-matching rate of the optimal _h_[∗] ∈H ( _i.e._ , the prediction error of _h_[∗] is 0) is larger than _κ_ + _[ϵ]_[(][1][−] 2[2] _[κ]_[)] . 

These statements are easy to understand, since if both of them do not hold, we can conclude that the empirical loss of _π_ ˆ _E_ is larger than that of _h_[∗] , which does not agree with the ERM definition. 

**Contradicts.** To begin with, we review the uniform convergence properties [27] by the following lemma: 

**Lemma 1.** _Let_ H _be a hypothesis class, then for any ϵ_ ∈(0 _,_ 1) _and h_ ∈H _, if the number of training samples is m, the following formula holds:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0016-04.png)


_where R and R_[ˆ] _are the expectation and empirical losses, respectively._ 

For statement (1), since the prediction error of ˆ _πE_ is larger than _ϵ_ , the expectation loss _R_ ( _π_ ˆ _E_ ) is larger than _κ_ + _ϵ_ (1 − 2 _κ_ ). If the empirical loss _R_[ˆ] ( _π_ ˆ _E_ ) is smaller than _κ_ + _[ϵ]_[(][1][−] 2[2] _[κ]_[)] , then ∣ _R_ ( _π_ ˆ _E_ ) − _R_[ˆ] ( _π_ ˆ _E_ )∣ should be larger than _[ϵ]_[(][1][−] 2[2] _[κ]_[)] . At the same time, according to Lemma 1, when the sample number _m_ 2 log([2][∣] _δ_[H][∣] ) is larger than _ϵ_[2] (1−2 _κ_ )[2][, we have][ P][(∣] _[R]_[(] _[π]_[ˆ] _[E]_[) −] _[R]_[ˆ][(] _[π]_[ˆ] _[E]_[)∣>] _[ϵ]_[(][1][−] 2[2] _[κ]_[)] ) < _δ_ . 

For statement (2), the expectation loss of _h_[∗] is _κ_ , i.e., _R_ ( _h_[∗] ) = _κ_ . If the empirical loss _R_[ˆ] ( _h_[∗] ) is larger than _κ_ + _[ϵ]_[(][1][−] 2[2] _[κ]_[)] , then ∣ _R_ ( _h_[∗] )− _R_[ˆ] ( _h_[∗] ) ∣ should be larger than _[ϵ]_[(][1][−] 2[2] _[κ]_[)] . According to Lemma 2 log([2][∣] _δ_[H][∣] ) 1, when the sample number _m_ is larger than _ϵ_[2] (1−2 _κ_ )[2][, we have][ P][(∣] _[R]_[ (] _[h]_[∗][) −] _[R]_[ˆ][ (] _[h]_[∗][)∣>] _[ϵ]_[(][1][−] 2[2] _[κ]_[)] ) < _δ_ . 

As a result, both of the above statements hold with the probability smaller than _δ_ , which implies that the prediction error of _π_ ˆ _E_ is smaller than _ϵ_ with the probability larger than 1 − _δ_ . 

## **D Proof of Theorem 4** 

**Lemma 2.** _(Proposition A.8 of Agarwal et al. [1]). Let z be a discrete random variable that takes_ ⃗ _values in_ {1 _,...,d_ } _, distributed according to q. We write q as a vector where q_ = [Pr( _z_ = _j_ )] _[d] j_ =1 _[.] Assume we have n i.i.d. samples, and that our empirical estimate of q_ ⃗ _is_ [⃗ _q_ ] _j_ = ∑ _[n] i_ =1 **[1]**[[] _[z][i]_[=] _[ j]_[]/] _[n][.] We have that_ ∀ _ϵ_ > 0 _:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0016-11.png)


_which implies that:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0016-13.png)


_Proof._ Applying Lemma 2, we have that for considering a fixed _st_ , wp. at least 1 − _δ_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0016-15.png)


where _n_ is the number of expert data used to estimate _π_ _**ω**_ (⋅∣ _st_ ). Then we apply the union bound across all states and actions to get that wp. at least 1 − _δ_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0016-17.png)


The result follows by rearranging _n_ and relabeling _h_ . 

**Remark 1.** _How much counterfactual expert data can we generate using our OILCA framework? Supposing we have n independent state action tuples in the expert data, we run the data augmentation module for m times, which means that we can augment each state to m counterfactual states and subsequently to m corresponding counterfactual actions. Thus, in total, we can obtain n[m] counterfactual tuples–an exponential increase for the previously given expert data. Back to Theorem 4, this demonstrates that our OILCA can effectively enhance the policy’s generalization ability._ 

16 

## **E Training Details** 

## **E.1 Data Generation and Statistics** 

**Toy Environment** The dimensions of state and action are both 2. For the exogenous variable, we generate the non-stationary 2D Gaussian data as follows: _u_[∗] ∣ _c_ ∼N ( _µ_ ( _c_ ) _,_ diag ( _σ_[2] ( _c_ ))), where _c_ is the class label. _µ_ 1( _c_ ) = 0 for all _c_ and _µ_ 2( _c_ ) = _αγ_ ( _c_ ), where _α_ ∈ R and _γ_ is a permutation. The variance _σ_[2] ( _c_ ) is generated randomly and independently across the classes. For the transition function, we use an MLP to generate the next state _st_ +1, such that _st_ +1 = MLP( _st,at,ut_ +1), where _ut_ +1 is the sample of _u_ at timestep _t_ + 1. For each class of exogenous variables, we generate 1K episodes for the data collection (500 steps per episode). Similar to DEEPMIND CONTROL SUITE, we also define a positive episode if its reward is among the top 20% episodes, and each of these positives is randomly chosen to constitute D _E_ with 101[chance.][As a result, we choose 75 episodes in][ D] _[E]_[and] 925 episodes in D _U_ . For the online testing, we can evaluate all the methods on the toy environment with any kind of distribution of the exogenous variable. 

**DEEPMIND CONTROL SUITE** DEEPMIND CONTROL SUITE (Figure 6) contains a variety of continuous control tasks involving locomotion and simple manipulation. States consist of joint angles and velocities, and action spaces vary depending on the task. The episodes are 1000 steps long, and the environment reward is continuous, with a maximum value of 1 per step. During the collection of offline data, we apply random Gaussian perturbation to the action outputted by the policy. This perturbation is specified in the XML configuration file as an integral part of the environment. Additionally, the distribution of the perturbation differs across different environment initialization (auxiliary variable _c_ ) due to their initialization seeds. In particular, different seeds correspond to different mean and variance of the Gaussian distribution perturbation via the random number generator. This approach is employed to introduce uncertainty into the environment [], thereby aligning with our problem setting. We define an episode as positive if its episodic return is among the top 20% episodes; each of these positives is randomly chosen to constitute D _E_ with 101 chance. We present the details in Table 3. 

|**Task**<br>Cartpole swingup|**Total**<br>40|**D****_E_**<br>2|
|---|---|---|
|Cheetah run|300|3|
|Finger turn hard|500|9|
|Fish swim|200|1|
|Humanoid run|3000|53|
|Manipulator insert ball<br>Manipulator insert peg|1500<br>1500|30<br>23|
|Walker stand|200|4|
|Walker walk|200|6|
|Reaching|600|12|
|Pushing<br>Picking|600<br>600|13<br>15|
|Pick and Place|600|12|
|Stacking2|600|11|
|Towers|600|13|
|Stacked Blocks|600|13|
|Creative Stacked Block|600|14|
|General|600|12|



Table 3: **Datasets statistics.** The total number of episodes and corresponding number of expert demonstrations (D _E_ ) per task. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0017-06.png)


**----- Start of picture text -----**<br>
Humanoid run Manipulator  Cartpole swing up<br>insert ball<br>Manipulator Cheetah run Finger turn hard<br>insert peg<br>Fish swim Walker stand Walker walk<br>**----- End of picture text -----**<br>


Figure 6: DEEPMIND CONTROL SUITE is a set of popular continuous control environments with tasks of varying difficulties, including locomotion and simple object manipulation. 

**CAUSALWORLD** CAUSALWORLD provides a combinatorial family of such tasks with common causal structure and underlying factors (including, e.g., robot and object masses, colors, sizes) (Figure 

17 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0018-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0018-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0018-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0018-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0018-04.png)


**----- Start of picture text -----**<br>
(a) Pushing (b) Picking (c) Pick and Place (d) Stacking2<br>(e) Stacked Blocks (f) General (g) CreativeStackedBlocks (h) Towers<br>**----- End of picture text -----**<br>


Figure 7: Example tasks from the task generators provided in the CAUSALWORLD. The goal shape is visualized in opaque red, and the blocks are visualized in blue. 

7). We conduct the offline dataset collection process by using various online behavior policies. We collect the mixed dataset by using three kinds of do-interventions (Figure 8) on different environment features. And we divide the offline dataset into D _E_ and D _U_ , similar to the DEEPMIND CONTROL SUITE. The detailed statistics about the dataset are presented in Table 3. 

## **E.2 Detailed Descriptions of Baselines** 

- **BC-exp:** Behavioral cloning on expert data D _E_ . D _E_ owns higher quality data but fewer quantities and thus causes serious compounding error problems to the resulting policy. 

- **BC-all:** Behavioral cloning on all data Dall. BC-all can generalize better than BC-exp due to access to a much larger dataset, but its performance may be negatively impacted by the low-quality data in Dall. 

- **ORIL** [39] **:** ORIL learns a reward function and uses it to solve an offline RL problem. It suffers from high computational costs and the difficulty of performing offline RL under distributional shifts. 

- **BCND** [26] **:** BCND is trained on all data, and it reuses another policy learned by BC as the weight of the original BC objective. Its performance will be worse if the suboptimal data occupies the major part of the offline dataset. 

- **LobsDICE** [12] **:** LobsDICE optimizes in the space of state-action stationary distributions and state-transition stationary distributions rather than in the space of policies. 

- **DWBC** [36] **:** DWBC is trained on all data. It mainly designs a new IL algorithm, where the discriminator outputs serve as the weights of the BC loss. 

## **F Additional Results** 

## **F.1 In-distribution Experiments on CAUSALWORLD** 

To further show the in-distribution performance, we supplement the experiments on CAUSALWORLD, in which both training and testing are conducted on space **A** . The results are shown in Table 4. In most tasks, our OILCA still achieves the highest average episode return, demonstrating our method’s effectiveness across different scenarios. Especially comparing the results in Table 2 and Table 4, we can notice that the advantage of OILCA for out-of-distribution generalization is more obvious. This proves the strong generalization ability of the counterfactual data augmentation module, which makes the offline imitation learning policy more robust to the data distribution shift. This point is especially significant in out-of-distribution scenarios, where the data distribution shifts more intensely. 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0019-00.png)


**----- Start of picture text -----**<br>
do(floor_color=’white’,<br>      block_size=0.065,<br>      …etc)<br>**----- End of picture text -----**<br>


Figure 8: Example of _do_ -interventions on exposed variables in CAUSALWORLD. 

Table 4: Results for in-distribution performance on CAUSALWORLD. We report the average return of episodes (length varies for different tasks) over five random seeds. All the models are trained on _space_ **A** and tested on _space_ **A** to show the in-distribution performance [2]. The best results and second best results are **bold** and underlined, respectively. 

|`Task Name`||BC-exp|BC-all<br>ORIL|BCND<br>LobsDICE|DWBC<br>**OILCA**|DWBC<br>**OILCA**|
|---|---|---|---|---|---|---|
|REACHING<br>PUSHING<br>PICKING<br>PICK ANDPLACE<br>STACKING2<br>TOWERS<br>STACKEDBLOCKS<br>CREATIVESTACKEDBLOCKS<br>GENERAL||353.98<br>±11.48<br>331.32<br>±6.36<br>394.63<br>±12.98<br>453.59<br>±7.58<br>596.14<br>±15.76<br>723.49<br>±15.82<br>1320.97<br>±19.83<br>684.52<br>±16.69<br>626.15<br>±20.57|247.60<br>±15.99<br>372.39<br>±9.58<br>310.62<br>±9.21<br>364.37<br>±8.36<br>360.28<br>±8.98<br>427.39<br>±13.69<br>355.83<br>±8.47<br>348.34<br>±11.63<br>435.12<br>±12.81<br>467.11<br>±13.19<br>947.96<br>±17.56<br>679.93<br>±8.68<br>947.96<br>±25.45<br>1520.62<br>±31.62<br>593.41<br>±26.86<br>758.04<br>±12.70<br>691.37<br>±17.22<br>**1072.05**<br>±47.26|358.36<br>±15.45<br>323.43<br>±10.13<br>335.87<br>±9.02<br>275.38<br>±9.93<br>381.45<br>±8.63<br>326.97<br>±12.31<br>376.34<br>±9.87<br>287.81<br>±10.06<br>476.33<br>±5.21<br>378.3<br>±7.65<br>680.61<br>±8.57<br>735.79<br>±12.23<br>1247.96<br>±29.14<br>958.64<br>±26.56<br>933.88<br>±16.57<br>601.18<br>±19.42<br>572.70<br>±11.28<br>549.89<br>±15.31|530.96<br>±8.70<br>436.22<br>±5.39<br>479.05<br>±8.57<br>448.89<br>±12.49<br>631.75<br>±8.54<br>915.26<br>±17.97<br>2116.51<br>±32.97<br>870.29<br>±24.56<br>786.44<br>±18.52|**986.19**<br>±10.27<br>**579.55**<br>±12.64<br>**648.34**<br>±8.51<br>**588.87**<br>±9.29<br>**920.18**<br>±7.36<br>**1263.94**<br>±8.98<br>**3210.23**<br>±43.63<br>**1476.41**<br>±25.94<br>964.32<br>±17.08|



## **F.2 Combinations with Other Base Offline IL Methods** 

To validate that the effectiveness of our method is not restricted by the base offline IL methods, we combine the Counterfactual data Augmentation (CA) part with ORIL and BCND, which are represented as ORIL+CA and BCND+CA, respectively. Also, we conduct corresponding experiments on the benchmarks in this paper, and the results are shown in Table 5. From the table, we can observe that the CA module can always help improve policy performance regardless of the base policy choice, which demonstrates its wide applicability. Besides, referring to the results in Table 5, Table 1, and Table 2, we can find that ORIL+CA and BCND+CA outperform all methods without CA’s assistance in most tasks, which implies that the simple counterfactual data augmentation may even work better than the complicated learning method designs. 

## **F.3 Performance of changing the auxiliary variable** _c_ 

To show the influence of the different choices of the auxiliary variable _c_ , we conduct additional experiments on the CAUSALWORLD benchmark. Specially, for the change of _c_ ’s choice, we apply the similar _do_ -interventions to more features ( _i.e._ block color, block mass) and fewer features. The performance of our OILCA under different intervened features (different choices of _c_ ) is shown in Table 6. Specially, _C_ = 1 represents feature set `stage_friction` , _C_ = 2 represents feature set ( `stage_friction` , `floor_friction` ), _C_ = 3 represents feature set ( `stage_color` , 

19 

Table 5: Results for in-distribution performance on part of tasks in DEEPMIND CONTROL SUITE and out-of-distribution generalization on part of tasks in CAUSALWORLD. We report the average return of episodes (length varies for different tasks) over five random seeds. The training and testing procedures follow those introduced in Section 5. All the results obtained by CA-assisted methods are **bold** to highlight the effect of the counterfactual data augmentation module. 

|`Benchmark`|`Task Name`|ORIL<br>ORIL+CA<br>BCND<br>BCND+CA<br>DWBC<br>OILCA|
|---|---|---|
|DEEPMIND<br>CONTROL<br>SUITE|CARTPOLESWINGUP<br>CHEETAHRUN<br>FINGERTURNHARD<br>FISHSWIM|221.24<br>±14.49<br>**426.79**<br>±12.09<br>243.52<br>±11.33<br>**452.68**<br>±12.86<br>382.55<br>±8.95<br>**608.38**<br>±35.54<br>45.08<br>±9.88<br>**78.44**<br>±6.95<br>96.06<br>±16.15<br>**158.62**<br>±8.85<br>66.87<br>±4.60<br>**116.05**<br>±14.65<br>185.57<br>±26.75<br>**227.94**<br>±15.47<br>204.67<br>±13.18<br>**284.29**<br>±12.03<br>243.47<br>±17.12<br>**298.73**<br>±5.11<br>84.90<br>±1.96<br>**156.92**<br>±8.18<br>153.28<br>±19.29<br>**268.56**<br>±6.03<br>212.39<br>±7.62<br>**290.29**<br>±10.07|
|CAUSAL<br>WORLD|REACHING<br>PUSHING<br>PICKING<br>PICK ANDPLACE|339.40<br>±12.98<br>**652.21**<br>±7.05<br>228.33<br>±7.14<br>**582.44**<br>±9.07<br>479.92<br>±18.75<br>**976.60**<br>±20.13<br>283.91<br>±19.72<br>**367.46**<br>±6.31<br>191.23<br>±12.64<br>**320.94**<br>±10.37<br>298.09<br>±14.94<br>**405.08**<br>±24.03<br>388.15<br>±19.21<br>**458.03**<br>±13.95<br>221.89<br>±7.68<br>**486.32**<br>±8.03<br>366.26<br>±8.77<br>**491.09**<br>±6.44<br>270.75<br>±14.87<br>**372.18**<br>±10.74<br>259.12<br>±8.01<br>**393.59**<br>±7.81<br>349.66<br>±7.39<br>**490.24**<br>±11.69|



Table 6: Results for under different choice of _c_ on the CAUSALWORLD benchmark (out-ofdistribution). We report the average return of episodes (length varies for different tasks) over five random seeds. 

|distribution). We report <br>fve random seeds.|the average return of episodes (length varies for different tasks) over|
|---|---|
|`Task Name`|_C_ =1<br>_C_ =2<br>_C_ =3<br>_C_ =4<br>_C_ =5|
|REACHING<br>PUSHING<br>PICKING<br>PICK ANDPLACE<br>STACKING2<br>Towers<br>STACKEDBLOCKS<br>CREATIVESTACKEDBLOCKS<br>GENERAL|928.62±22.38<br>957.54±18.39<br>976.60±20.13<br>985.25±17.26<br>1037.12±19.15<br>389.16±9.43<br>396.52±17.29<br>405.08±24.03<br>426.60±15.37<br>429.42±12.28<br>462.54±9.08<br>484.21±11.37<br>491.09±6.44<br>522.96±13.27<br>525.20±12.28<br>464.68±10.27<br>486.74±8.52<br>490.24±11.69<br>511.76±9.05<br>523.46±15.42<br>794.81±16.50<br>803.27±13.26<br>831.82±11.78<br>867.43±9.82<br>871.43±18.19<br>972.34±12.36<br>979.23±8.72<br>994.82±5.76<br>1027.16±17.25<br>1029.37±8.06<br>2317.48±74.32<br>2558.35±42.17<br>2617.71±88.07<br>2682.76±69.25<br>2754.39±82.16<br>1226.72±62.18<br>1297.20±39.42<br>1348.49±55.05<br>1468.65±27.63<br>1486.51±41.29<br>868.62±7.65<br>875.55±19.28<br>891.14±23.12<br>926.19±17.34<br>934.74±16.20|



`stage_friction` , `floor_friction` ), _C_ = 4 represents feature set ( `stage_color` , `stage_friction` , `floor_friction` , `block_mass` ), _C_ = 5 represents feature set ( `stage_color` , `stage_friction` , `floor_friction` , `block_color` , `block_mass` ). 

From the above Table 6, we can find that our OILCA can achieve a consistent performance improvement over different baselines under different choices of _c_ . This demonstrates that the empirical performance of our method is relatively robust to the selection of this variable _c_ . In fact, when increasing the number of intervened features (the number of _c_ choices), we can observe our model can achieve better performance. This is because the policy can learn to adapt to more diverse/uncertain environment configurations during the training phase. 

## **F.4 Influence of the augmented data** 

In order to prove that the performance will not decay when further improving the _DE_ / _DU_ , we further increase _DE_ / _DU_ (larger than 1) and conduct the experiments with three tasks in DEEPMIND CONTROL SUITE of our method OILCA. Moreover, to show the quality of augmented data, we show the performance gap when increasing expert data proportion using two kinds of augmented data: 1) sampling with the policy in the online environment for more true expert data (Expert Data), 2) our counterfactual data augmentation method OILCA (Augmented Data). The experimental results are shown in Table 7. 

20 

Table 7: Results for the Influence of the augmented data with improving the proportion of augmented data and comparison to the true expert data in DEEPMIND CONTROL SUITE Benchmark. 

|`Task Name`|CARTPOLESWINGUP|CHEETAHRUN|CARTPOLESWINGUP|
|---|---|---|---|
|**Proportion**|Augmented Data<br>Expert Data|Augmented Data<br>Expert Data|Augmented Data<br>Expert Data|
|10%<br>30%<br>50%<br>70%<br>90%<br>100%<br>200%<br>300%<br>500%<br>1000%|430.21±13.20<br>441.36±12.01<br>463.78±21.95<br>472.92±7.62<br>502.81±20.76<br>520.15±15.43<br>557.90±16.62<br>562.89±20.47<br>589.01±38.29<br>593.37±16.81<br>608.38±35.54<br>621.80±9.26<br>596.52±28.37<br>634.12±18.29<br>612.30±41.25<br>635.93±25.15<br>601.47±27.82<br>627.47±22.86<br>605.81±31.63<br>629.94±23.28|71.85±8.26<br>74.56±3.29<br>86.44±13.62<br>82.06±9.36<br>92.60±16.51<br>89.21±12.98<br>105.57±11.29<br>111.27±11.56<br>113.12±9.25<br>118.32±15.27<br>116.05±14.65<br>128.07±8.31<br>106.39±10.08<br>132.64±14.24<br>118.51±15.72<br>125.18±8.73<br>109.96±9.84<br>129.72±12.34<br>117.08±7.69<br>124.80±9.46|261.77±14.68<br>255.62±18.29<br>269.85±13.39<br>272.18±12.25<br>276.12±9.82<br>285.48±8.36<br>283.69±12.71<br>295.83±13.48<br>288.27±7.09<br>306.26±10.81<br>298.73±5.11<br>303.51±11.67<br>303.64±12.91<br>311.70±9.74<br>301.57±8.30<br>305.42±14.53<br>289.15±15.27<br>302.15±12.16<br>295.48±7.84<br>304.93±11.19|



From Table 7, we can find that the performance will converge when the proportion is close to 100%, and further improving it indeed will not improve the performance obviously. This can be explained by the results that augmenting too much data can hardly bring additional effective information gain to the learned policy. Moreover, our augmented counterfactual data behaves slightly worse than augmentation with true expert data under most proportions, though achieving obvious improvement over other IL baselines. This shows that the augmented data through our method is high-quality enough. 

## **F.5 Learning Curves of OILCA** 

We provide the learning curves of OILCA in Figure 9. In detail, we deploy our trained policy to the online environment at each epoch and then collect 100 episodes for computing the average episode return. From the figures, we can observe that the policies can converge after 200 epochs in most tasks. The fluctuation of the curves mainly comes from the instability of the base offline IL method. 

## **G Limitation Analysis** 

We simply analyze the limitations of this work in this section. In this paper, we only provide the theoretical guarantee to the generalization ability of learned policy from the perspective of the counterfactual samples’ number. Actually, why the samples generated by the counterfactual augmentation module are more meaningful and can help the learned policy generalize better than samples obtained by other augmentation methods is also worth exploring theoretically. 

21 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning_images/nips_2023_Offline_Imitation_Learning_with_Variational_Counterfactual_Reasoning.pdf-0022-00.png)


**----- Start of picture text -----**<br>
Cartpole Swingup Cheetah Run Finger Turn Hard<br>650 130 300<br>120<br>600 110 280<br>550 100 260<br>90<br>500<br>80 240<br>450 70 220<br>40 60 80 100 120 140 160 180 200 40 60 80 100 120 140 160 180 200 60 80 100 120 140 160 180 200<br>Training Epoch Training Epoch Training Epoch<br>(a) Cartpole Swingup (b) Cheetah Run (c) Finger Turn Hard<br>Fish Swim Humanoid Run Manipulator Insert Ball<br>310 310<br>300 450 300<br>290 400 290<br>280 350 280<br>270<br>270 300<br>260<br>260 250 250<br>250 200 240<br>240 230<br>60 80 100 120 140 160 180 200 25 50 75 100 125 150 175 200 80 100 120 140 160 180 200<br>Training Epoch Training Epoch Training Epoch<br>(d) Fish Swim (e) Humanoid Run (f) Manipulator Insert Ball<br>Manipulator Insert Peg Walker Stand Walker Walk<br>400<br>300 400 350<br>300<br>280 380 250<br>260 200<br>360<br>240 150<br>340 100<br>220 50<br>320 0<br>80 100 120 140 160 180 200 25 50 75 100 125 150 175 200 25 50 75 100 125 150 175 200<br>Training Epoch Training Epoch Training Epoch<br>(g) Manipulator Insert Peg (h) Walker Stand (i) Walker Walk<br>Average Episode Return Average Episode Return Average Episode Return<br>Average Episode Return Average Episode Return Average Episode Return<br>Average Episode Return Average Episode Return Average Episode Return<br>**----- End of picture text -----**<br>


Figure 9: Learning curves of OILCA on 9 tasks of DEEPMIND CONTROL SUITE. 

22 

