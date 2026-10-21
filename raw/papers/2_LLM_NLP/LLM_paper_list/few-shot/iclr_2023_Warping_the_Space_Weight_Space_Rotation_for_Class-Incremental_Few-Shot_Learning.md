Published as a conference paper at ICLR 2023 

## WARPING THE SPACE: WEIGHT SPACE ROTATION FOR CLASS-INCREMENTAL FEW-SHOT LEARNING 

**Do-Yeon Kim**[1] **, Dong-Jun Han**[2] _[∗]_ **, Jun Seo**[3] **, Jaekyun Moon**[1] 

1 Korea Advanced Institute of Science & Technology, 2 Purdue Univirsity, 3 LG AI Research dy.kim@kaist.ac.kr, han762@purdue.edu, jun.seo@lgresearch.ai, jmoon@kaist.edu 

## ABSTRACT 

Class-incremental few-shot learning, where new sets of classes are provided sequentially with only a few training samples, presents a great challenge due to catastrophic forgetting of old knowledge and overfitting caused by lack of data. During finetuning on new classes, the performance on previous classes deteriorates quickly even when only a small fraction of parameters are updated, since the previous knowledge is broadly associated with most of the model parameters in the original parameter space. In this paper, we introduce WaRP, the _weight space rotation process_ , which transforms the original parameter space into a new space so that we can push most of the previous knowledge compactly into only a few important parameters. By properly identifying and freezing these key parameters in the new weight space, we can finetune the remaining parameters without affecting the knowledge of previous classes. As a result, WaRP provides an additional room for the model to effectively learn new classes in future incremental sessions. Experimental results confirm the effectiveness of our solution and show the improved performance over the state-of-the-art methods. 

## 1 INTRODUCTION 

Humans can easily acquire new concepts while preserving old experiences continually over the course of their life span. With a growing desire to imitate such ability, incremental or continual learning has been brought into the spotlight in the AI community recently (Hung et al., 2019; Wortsman et al., 2020; Saha et al., 2021). Here, due to storage and privacy constraints, it is impractical to save all the training samples of previous tasks during the training process (Desai et al., 2021). The most challenging issue in this setup is to preserve the knowledge of previous tasks against catastrophic forgetting (Serra et al., 2018). More recently, an increasing need for such learning capability when dealing with rare data (e.g., military image, medical data, photos of rare animals) has encouraged many researchers to focus on a more challenging setup, known as class-incremental few-shot learning (CIFSL). In CIFSL, each task consists of only a few training samples, making the problem much harder as we must additionally handle the severe overfitting issue caused by lack of training data. 

Prior works on CIFSL typically take the following two steps: (i) pretraining the model on the first task (base classes) and (ii) adapting the pretrained model to new classes (novel classes) in each training session (e.g., via finetuning), assuming that the first task contains a sufficiently large number of training samples. One recent work, F2M (Shi et al., 2021), tries to find the flat local minima during the pretraining stage and then finetunes the model within this flat area for learning novel classes so that the forgetting issue could be resolved. Another line of work, named FSLL (Mazumder et al., 2021), mitigates the forgetting issue by keeping some important parameters frozen and finetunes only the remaining trainable parameters during the incremental learning sessions. Several other works adopt different strategies to learn new classes well during incremental sessions without forgetting (Tao et al., 2020; Cheraghian et al., 2021a; Zhang et al., 2021; Kukleva et al., 2021; Chen & Lee, 2021; Aky¨urek et al., 2022). 

**Motivation.** One key strategy shared among some of these prior works is: model update is performed in the _original parameter space_ , which is defined here with the _standard basis_ . To illustrate the 

> _∗_ Corresponding author. 

1 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning_images/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning.pdf-0002-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning_images/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning.pdf-0002-02.png)


Figure 1: Visualization of loss landscape for base classes after pretraining. **Left:** red dashed arrow lines refer to directions of standard basis. Finetuning along either of these directions at each incremental session can adversely affect the loss. In this original space, it is generally challenging to capture the flat directions directly. **Right:** weight space rotation with desired basis (blue dashed lines) provides us with flat direction _K_ 11; following this direction, we can finetune the model without performance loss on the base classes. The performance on the novel classes are also preserved by accumulating and freezing the important parameters obtained at each session. 

concept of standard basis in this context, consider an example with 2 _×_ 2 weight matrix _W_ for a specific layer. By defining _wij_ as the ( _i, j_ )-th parameter of _W_ , we can unroll _W_ as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning_images/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning.pdf-0002-05.png)


Here, we denote **E** = _{_ **e** 11 _,_ **e** 12 _,_ **e** 21 _,_ **e** 22 _}_ as standard basis, upon which the weight space is parameterized by **w** = _{w_ 11 _, w_ 12 _, w_ 21 _, w_ 22 _}_ . Neural networks are typically implemented in the weight space spanned by **E** , and thus the learning processes (e.g. gradient computation, parameter update) are performed with respect to the parameters in **w** as shown in Figure 1. However, our empirical observation in Figure 2, which showcases the effect of simple finetuning in two different spaces, suggests that the accuracies of previous classes are extremely vulnerable to model finetuning in the original space, even when we finetune only a small fraction of 80 _mini_ ImageNet parameters; finetuning only 3% of model parameters (freeze 97%) throughout the incremental sessions significantly degrades the 60 performance on base classes. This implies that the previous knowl40 edge is more broadly associated with, to a certain extent, most New Space (97% freeze) of the model parameters in the original space. Motivated by this, 20 New Space (3% freeze)Original Space (97% freeze) we pose the following question: _Can we find another basis, i.e., a_ 1 3 Original S5 pace (73% freeze)9 _new weight parameter space, such that we can push most of the_ Session _previous knowledge compactly into only a few key parameters?_ Figure 2: Original vs. New space. 

**Contributions.** In this paper, we introduce the concept of **W** eight sp **a** ce **R** otation **P** rocess (WaRP) that provides a solution to the above question. By viewing the weight matrix of a neural network from a different perspective, WaRP transforms the original parameter space into a new space. For any given orthonormal set of matrices, we can reparameterize the weight in (1) as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning_images/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning.pdf-0002-08.png)


> whereand _w_ � _Bij_ is the weight parameter that is reparameterized according to= _{K_ 11 _, K_ 12 _, K_ 21 _, K_ 22 _}_ is a newly constructed basis that consists of orthonormal matrices _Kij_ . As shown in Figure 1� , we can always rotate the axes by which the weight is represented for an arbitrary basis _B_ . Here, if _w_ 12 in (2) plays a key role in determining the output of the layer, then _{K_ 11 _, K_ 21 _, K_ 22 _}_ can be viewed as 

> flat directions.parameter _w_ �12Thus, finetuning can be performedin the new space, which effectively _along the flat directions_ preserves the previous by freezing the importantknowledge (Figure 2). Figure 1 provides high-level descriptions of WaRP and our finetuning process along the flat direction. 

By introducing the concept of WaRP, we propose a new strategy to construct an appropriate basis _B_ using the low-rankness property of activation, and redefine the model parameters in this new space so that we can push most of the knowledge of base classes compactly into only a few parameters. Once we construct a new weight space, we identify and freeze the important parameters for keeping the knowledge of previous classes at the end of each training session, and finetune only the remaining 

2 

Published as a conference paper at ICLR 2023 

parameters in the next session without performance degradation on previous knowledge. With a careful construction of new basis and a proper criterion for identifying important parameters, we can preserve the knowledge of base classes after finetuning the model on novel classes at each incremental session. The knowledge of novel classes can be also preserved by accumulating the important parameters and keeping them frozen. It turns out that WaRP is easy to implement and the proposed score criterion is efficient to compute. Experimental results on various benchmarks show the improved performance of our method over the state-of-the-art baselines. 

## 2 WEIGHT SPACE ROTATION 

Before describing our solution for CIFSL, we first introduce the general concept of WaRP proposed in this paper. To begin with, we view the weight matrix _W ∈_ R _[m][×][n]_ as a linear combination of orthonormal matrices _{_ **e** _ij}i,j_ , where **e** _ij ∈_ R _[m][×][n]_ is defined as a matrix having one for the ( _i, j_ )-th element and zeros for others. We can rewrite _W_ as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning_images/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning.pdf-0003-04.png)


where _wij_ denotes the ( _i, j_ )-th element of _W_ which is the coefficient of **e** _ij_ . Since neural networks are parameterized by coefficients **w** := _{wij}i,j_ in practical deep learning implementations, typical learning operations such as gradient descent are conducted on the weight space spanned by the standard basis **E** := _{_ **e** _ij}i,j_ . However, the learned knowledge of previous tasks is likely to be associated with most of the parameters in **w** (i.e., most directions in standard basis **E** ); updating a small fraction of selected parameters could give rise to significant performance degradation on previous classes as shown in Figure 2, suggesting that the previous performance is highly vulnerable to finetuning in the original space. 

In this paper, we extend the above linear algebraic interpretation of the weight matrix (based on standard basis **E** ) using an arbitrary basis _B_ = _{Kij ∈_ R _[m][×][n] |i ∈_ [ _m_ ] _, j ∈_ [ _n_ ] _}_ that spans R _[m][×][n]_ . Specifically, let _B_ be an arbitrary set of orthonormal matrices satisfying: (i) _⟨K, K[′] ⟩_ = 0 if _K ̸_ = _K[′]_ for _K, K[′] ∈B_ , and (ii) _∥K∥_ = � _⟨K, K⟩_ = 1 for all _K ∈B_ , where _∥· ∥_ and _⟨·, ·⟩_ are the norm and inner product properly defined in R _[m][×][n]_ , respectively. Since a vector space _H_ := R _[m][×][n]_ equipped with the inner product operation _⟨A, B⟩_ := trace( _A[⊤] B_ ) (for _A, B ∈H_ ) is Hilbert space, one can always rewrite _W ∈H_ as a linear combination of orthonormal matrices in _B_ as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning_images/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning.pdf-0003-07.png)


Note that this equation generalizes (3) to the case with an arbitrary basis _B_ ; the special case with _KB_ spanning _ij_ = **e** _ij_ (i.e., R _[m] B[×][n]_ =, we can always reparameterize the layer of the model with **E** ) reduces to the result in equation (3). Thus for any orthonormal set of matrices **w** � := _{⟨W, K⟩}K∈B_ in place of **w** . In the next section, we describe how WaRP can address forgetting in CIFSL. 

## 3 PROPOSED ALGORITHM FOR CIFSL 

**Problem Setup and Notations.** The main goal of CIFSL is to incrementally train the model on a new set of classes at each training session, while avoiding forgetting of old classes. Let _Tk_ be the task given in _k_ -th session with tuples of training sample and its class label, i.e., _Tk_ = _{_ ( _x[k] t[, y] t[k]_[)] _[}][N] t_ =1 _[k]_[,] where _Nk_ denotes the number of samples in _Tk_ . For simplicity, we omit super/subscripts of _x[k] t_[and] _yt[k]_[if these can be clearly referred from the context.][By defining] _[ n][s]_[as the total number of training] sessions, we assume that the corresponding sets of categories in each task, denoted by _C_ 1 _, . . . , Cns_ , are completely disjoint. As in previous works (Tao et al., 2020; Shi et al., 2021), we consider a setup where a sufficiently large amount training data are available in the first task _T_ 1. In contrast, the subsequent tasks _T_ 2 _. . . , Tns_ consist of only a small number of samples per class, e.g., 5 samples per class. We consider a practical setup where all training samples of previous tasks cannot be accessed during training due to privacy and storage constraints. We use the term ‘task’ and ‘session’ interchangeably throughout the paper. We let _Wl ∈_ R _[m][l][×][n][l]_ be the weight matrix of _l_ -th layer in the neural network. We discard the layer index _l_ and use _W ∈_ R _[m][×][n]_ for weight whenever the context obviates it. _In_ denotes the _n × n_ identity matrix, [ _n_ ] denotes the set _{_ 1 _, . . . n}_ , and _{ai}i_ or _{ai}i∈I_ represents _{ai|i ∈I}_ for some index set _I_ . 

3 

Published as a conference paper at ICLR 2023 

**Overview of Approach.** Given the model trained on the first task _T_ 1, our first step is to construct a new basis via WaRP on which most directions are flat. Once new basis is constructed in the first session, this new basis is fixed throughout the whole remaining (incremental) sessions. At the end of each session, we identify the important parameters based on the scoring criterion and freeze them. In the next session, we finetune only the remaining parameters, namely, the directions of flat axes, and the same process is repeated for the future tasks. Throughout the learning process, the important parameters are accumulated and kept frozen to preserve the knowledge of both novel and base classes. 

In the following, we start by describing how to construct an appropriate basis via WaRP for CIFSL and how to implement it in practice (Sec. 3.1). Then we describe our criterion for identifying the important parameters in each incremental session (Sec. 3.2). Finally, we describe the overall procedure of our algorithm for CIFSL with details (Sec. 3.3). 

## 3.1 CONSTRUCTING NEW BASIS FOR WEIGHT SPACE ROTATION 

To find an appropriate basis to reparameterize _W ∈_ R _[m][×][n]_ , we start by expressing the basis by using a pair of unitary matrices. Given any two unitary matrices _V ∈_ R _[m][×][m]_ , _U ∈_ R _[n][×][n]_ , it can be seen that the set _B_ = _{Kij_ := **v** _i_ **u** _[⊤] j[∈]_[R] _[m][×][n][|][i][ ∈]_[[] _[m]_[]] _[, j][∈]_[[] _[n]_[]] _[}]_[ is orthonormal, where] **[ v]** _[i]_[ and] **[ u]** _[j]_[denote] the _i_ -th and _j_ -th column vectors of _V_ and _U_ , respectively. Thus from (4), we can rewrite _W_ as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning_images/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning.pdf-0004-05.png)


where _w_ � _ij_ := _⟨W,_ **v** _i_ **u** _[⊤] j[⟩]_[.][In][other][words,] _[W]_[is][reparameterized][by] _[{][w]_[ �] _[ij][}][i,j]_[using][the][basis] _{_ **v** _i_ **u** _[⊤] j[}][i,j]_[constructed with arbitrary two unitary matrices] _[ V]_[and] _[ U]_[.][Now the question is, how to] design appropriate _V_ and _U_ to make most of the elements in the basis to be flat directions? 

For a specific layer _l_ , we define a matrix Φ _l_ = [ _ϕl_ ( _x_ 1) _, ϕl_ ( _x_ 2) _, . . . , ϕl_ ( _xN_ 1)] where _ϕl_ ( _xt_ ) denotes the activation of the _l_ -th layer for the _t_ -th sample in the first task _T_ 1. Then the singular value decomposition of covariance of activations is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning_images/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning.pdf-0004-08.png)


where Σ _l_ is a diagonal matrix having singular values on its diagonal and _Ul_ is the corresponding unitary matrix. Here, note that the activations of deep neural networks are known to have low-rankness property as shown in (Suzuki et al., 2020b; Wang et al., 2021), i.e., singular values decrease rapidly and most of the singular values are close to 0. Thanks to this low-rankness property, by adopting� _U_ = _Ul_ in (5) for the _l_ -th layer _Wl_ and defining _wij[l]_[=] _[ ⟨][W][ l][,]_ **[ v]** _[i]_ **[u]** _[⊤] j[⟩]_[, we have] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning_images/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning.pdf-0004-10.png)


for any unitary matrix _V_ and any sample _x_ in _T_ 1. Namely, the effect of activation _ϕl_ ( _x_ ) is highly likely to be negligible in most directions in _B_ = _{_ **v** _i_ **u** _[⊤] j[}][i,j]_[during forward propagation.][In other] words, most directions in _B_ are flat and thus only a few directions and their corresponding parameters are meaningful. So if we can properly identify the important parameters among _{w_ � _ij}i,j_ and freeze them, finetuning along the remaining directions (for which **v** _i_ **u** _[⊤] j[ϕ][l]_[(] _[x]_[)] _[≈]_ **[0]**[)][does][rarely][change] the output _Wlϕl_ ( _x_ ) for _x_ in _T_ 1. This implies that the feature embedding of previous classes can be preserved by freezing only a small fraction of parameters in this new weight space. 

**Implementation.** Note that the actual implementation of WaRP in (5) turns out to be simple in practice. By using the fact that _V V[⊤]_ = _Im_ and _UU[⊤]_ = _In_ , we have _W_ = _V V[⊤] WUU[⊤]_ = _V WU_[�] _[⊤]_ where _W_[�] := _V[⊤] WU_ . From the definition above, _w_ � _ij_ is ( _i, j_ )-th element of _W_[�] . Thus to apply WaRP, we just multiply unitary matrices to _W_ and reparameterize the layer with _W_[�] while _V_ , _U_ remain fixed. Throughout the algorithm, we set _V_ = _Im_ which we found to be a valid choice. 

**Convolutional Case.** Implementing WaRP for convolutional layers is almost the same with the case of fully-connected layers except for flattening the feature maps and reshaping the weight/unitary matrices, to make the pipelines workable as in fully-connected layer. The detailed implementation for convolutional layer can be found in Appendix A.2. 

4 

Published as a conference paper at ICLR 2023 

## 3.2 IDENTIFYING IMPORTANT PARAMETERS 

In the new weight space constructed in Sec. 3.1, our scheme identifies the important parameters at the end of each training session and keep freezing them throughout the training process. The remaining trainable parameters (except the accumulated important parameters) are finetuned in the next session. In this subsection, we describe our strategy for selecting the important parameters in each session. 

To identify the trainable parameters (i.e., flat directions), at the end of each _k_ -th session, we take the score criterion that is well compatible with our new space constructed by WaRP. By the chain rule,taking the derivative of the loss function, denoted by _L_ , with respect to the parameter _w_ � _ij[l]_[yields:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning_images/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning.pdf-0005-04.png)


where _dzl_ denotes the gradient of the loss with respect to the output of the layer _zl_ = _Wlϕl_ ( _x_ ) for any _x_ in _Tk_ . Interestingly, it turns out that the term ( _a_ ) in equation (8) also appears in (7). In addition, as shown in (8), the trend of the magnitude of _dL/dw_ � _ij[l]_[follows the tendency of][ (] _[a]_[)][.][From these] observations, it can be seen that the gradient can indirectly capture the influence of finetuning within each direction **v** _i_ **u** _[⊤] j_[on] _[ T][k]_[.][Motivated by this, we consider the following score criterion:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning_images/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning.pdf-0005-06.png)


where _Lk_ ( _b_ ) is the loss computed with batch _b_ in _Tk_ , and _Dk[n][b]_[is a set that consists of] _[ N][k][/n][b]_[batches] with size _nb_ in _Tk_ . In words, the score is computed by summing up the magnitudes of the gradients using the samples insimulates the gradient with respect to _Tk_ . Here, if _s[l] ij_[is small, then] _w_ � _ij_ . Thus, finetuning with respect to this[ ∆] _[w]_[�] _[ij]_ **[v]** _[i]_ **[u]** _[⊤] j[ϕ][l]_[(] _[x]_[)][ is likely to be small where] _w_ � _ij_ (following the[ ∆] _[w]_[�] _[ij]_ direction of **v** _i_ **u** _[⊤] j_[) on new classes may not distort the feature embedding of] _[ all previous classes]_[.] **Remarks.** We wish to highlight two points here. First, WaRP in Sec. 3.1 and the score criterion in Sec. 3.2 work in a highly complementary manner; if we use the standard basis, the flat directions are limited and thus the performance could be restricted regardless of the scoring criterion. If we use a different score criterion (e.g. weight magnitude as done in (Mazumder et al., 2021)) other than (9) in the new weight space, we cannot fully exploit the low-rankness as in (8) and thus cannot properly identify flat directions. These results are confirmed via experiments in Sec. 5.3. Secondly, due to the simplicity of implementing WaRP, we can rely on auto-grad computation of gradient with respect to the weight matrix; we can take the derivative with respect to _W_[�] rather than with respect to _w_ � _ij_ one by one, to obtain the scores at once. Thus it is efficient to compute. 

## 3.3 OVERALL PROCEDURE FOR CIFSL 

Based on WaRP and the proposed score criterion, we describe the overall procedure of our algorithm. **First Session (** _k_ = 1 **).** We pretrain the embedding network and the linear classifier _W_ cls by minimizing the cross-entropy loss _L_ CE( _Tk_ ) = _− N_[1] _k_ � _Nt_ =1 _k_ � _c∈Ck_[1] _{yt[k]_[=] _[c][}]_[ log] � � _c[′] ∈∪_ ex _[k] i_ =1p( _[C] γ[i] c_[exp(] _,t_ ) _[γ][c][′][,t]_[)] � with _k_ = 1, where 1 _{·}_ is an indicator function and _γc,t_ denotes the output of classifier corresponding to class _c_ . When pretraining is done, we replace the weight vector of linear classifier _W_ cls corresponding to class _c_ with the prototype (average of feature embeddings corresponding to class _c_ ), which we found is a good option for balancing the model performance on base classes and future novel classes. 

Now we construct a new basis for each layer according to Sec. 3 _._ 1 and apply WaRP. Here, to alleviate the overfitting issue on novel classes during the remaining incremental sessions, we restrict the learnable space by applying WaRP only to the last few layers while keeping other layers fixed throughout the whole training process. Note that once we construct a new basis in the first session, this basis is fixed throughout the remaining sessions and thus model updates are performed in the new/fixed space. See Appendix A.4 for discussions on validity of the new space for novel classes. 

At the end of the first session, we identify the important/trainable parameters based on the proposed importance score (9) in Sec. 3.2. We select the parameters with the scores up to top _α ×_ 100% as important parameters and let the remaining ones as trainable parameters, where _α_ is the parameter 

5 

Published as a conference paper at ICLR 2023 

keeping ratio, a hyperparameter defined in range [0 _,_ 1]. To indicate whether each element of _W_[�] _l_ is important or not, we introduce a 0-1 mask **M** _l ∈_ R _[m][l][×][n][l]_ as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning_images/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning.pdf-0006-02.png)


where _sα_ is the score value corresponding to top _α ×_ 100% of the scores of all parameters in all selected layers, i.e., in _{s[l] ij[|][i][ ∈]_[[] _[m][l]_[]] _[, j][∈]_[[] _[n][l]_[]] _[,]_[selected layers] _[ l][}]_[.][If the][ (] _[i, j]_[)][-th element of] **[ M]** _[l]_[is][ 1][,] the ( _i, j_ )-th element of _W_[�] _l_ is regarded as an important parameter. 

**Incremental Sessions (** _k ≥_ 2 **).** In the beginning of each incremental session _k ≥_ 2, we compute the prototypes corresponding to new classes in _Ck_ and utilize them as a classifier along with the prototypes of the base classes. To preserve the knowledge of previous classes during finetuning, we only update the unimportant parameters using the importance mask **M** _l_ obtained from the previous session. We block the gradient computation with respect to important parameters at each layer _l_ as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning_images/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning.pdf-0006-05.png)


and let _Wl ← V W_[�] _lU[⊤]_ , where _stop_ ~~_g_~~ _rad_ ( _·_ ) is an operation that detaches the input from the computational graph during loss computation, and _⊙_ denotes element-wise multiplication. Thus, gradient descent with respect to _W_[�] _l_ in (11) leads to model update of the unimportant parameters only. In each incremental session _k_ , we finetune the model _W_[�] _l_ using the cross-entropy loss _L_ CE( _Tk_ ) for each selected layer _l_ , while keeping the weights of the classifier fixed during finetuning. 

After finetuning is finished on task _Tk_ , we also identify the important parameters for _Tk_ to be considered for the next session; we obtain the importance mask _Ml,k_ using criterion in (10) for a given _α_ . To preserve not only the knowledge of _Tk_ but also of classes in previous sessions, we accumulate important parameters identified so far: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning_images/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning.pdf-0006-08.png)


All the knowledge encountered so far can be properly preserved by this accumulated mask. 

Finally, when each _k_ -th session is finished, the prediction is made on the test samples from all encountered classes, i.e., for a given input, the prediction _y_ ˆtest is given by _y_ ˆtest = arg max _c′∈∪ki_ =1 _[C][i][ γ][c][′][,t]_[.][The] whole process described above is repeated until the last incremental session, i.e., for _k_ = 2 _,_ 3 _, . . . ns_ . 

## 4 RELATED WORKS 

**Class-Incremental Few-shot Learning.** There have been many attempts to solve class-incremental few-shot learning where only a few labeled samples are available in each task (Cheraghian et al., 2021b; Zhu et al., 2021; Zhou et al., 2022b; Cheraghian et al., 2021a). To address both catastrophic forgetting and overfitting issues in CIFSL, some prior works take meta-learning approaches (Gidaris & Komodakis, 2018; Ren et al., 2019; Yoon et al., 2020). They typically mimic the inference stage during training so that the model can rapidly adapt to new classes. However, these works assume that only one set of classes is given during the incremental session and do not consider a scenario in which “more than one” sets of new classes are consecutively provided. TOPIC (Tao et al., 2020) is the first work that considered more than one incoming sets of classes, providing a cornerstone for future emerging studies (Akyurek et al.¨ , 2022; Chen & Lee, 2021; Zhang et al., 2021; Mazumder et al., 2021; Shi et al., 2021; Zhou et al., 2022a). The authors of TOPIC adopted neural gas network to maintain the topology of feature space, preventing forgetting issue. FSLL (Mazumder et al., 2021) addressed the overfitting issue by choosing a few trainable parameters to be updated for finetuning and prevented forgetting issue by imposing additional regularization to minimize the deviation from the pretrained model parameters. However, this additional regularization may limit the learnability of the model on new classes. Unlike many existing works, (Shi et al., 2021; Zhou et al., 2022a) turned their focus on preparing a desired pretrained model in advance suitable for future incremental sessions. Specifically, the authors of F2M (Shi et al., 2021) inject noise to model parameters during pretraining to find the flat local minima, and finetune the model within this flat area during incremental sessions. However, pretraining with noise injection can lead to a longer training time for the model to converge and induce a little compromise on the performance of the pretrained model in return for the flatness of local minima. Compared to existing works where the proposed algorithms are performed in the ‘original parameter space’ (e.g. freezing/updating the parameters in standard basis as done in FSLL), 

6 

Published as a conference paper at ICLR 2023 

Table 1: Accuracy on _mini_ ImageNet dataset under 5-way 5-shot incremental few-shot learning setup. 

|Method|Session|
|---|---|
||1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9|
|Finetuning<br>iCaRL (Rebuff et al.,2017)<br>Rebalance (Hou et al.,2019)<br>GPM (Saha et al.,2021)<br>EEIL (Castro et al.,2018)<br>TOPIC (Tao et al.,2020)<br>FSLL (Mazumder et al.,2021)<br>FSLL+SS (Mazumder et al.,2021)<br>CEC (Zhang et al.,2021)<br>F2M (Shi et al.,2021)|**72.99**<br>67.79<br>63.71<br>60.17<br>56.82<br>53.18<br>48.36<br>43.43<br>39.60<br>71.77<br>61.85<br>58.12<br>54.60<br>51.49<br>48.47<br>45.90<br>44.19<br>42.71<br>72.30<br>66.37<br>61.00<br>56.93<br>53.31<br>49.93<br>46.47<br>44.13<br>42.19<br>**72.99**<br>68.04<br>64.18<br>60.96<br>58.32<br>55.66<br>52.90<br>51.10<br>49.96<br>61.31<br>46.58<br>44.00<br>37.29<br>33.14<br>27.12<br>24.10<br>21.57<br>19.58<br>61.31<br>50.09<br>45.17<br>41.16<br>37.48<br>35.52<br>32.19<br>29.46<br>24.42<br>66.48<br>61.75<br>58.16<br>54.16<br>51.10<br>48.53<br>46.54<br>44.20<br>42.28<br>68.85<br>63.14<br>59.24<br>55.23<br>52.24<br>49.65<br>47.74<br>45.23<br>43.92<br>72.00<br>66.83<br>62.97<br>59.43<br>56.70<br>53.73<br>51.19<br>49.24<br>47.63<br>72.05<br>67.47<br>63.16<br>59.70<br>56.71<br>53.77<br>51.11<br>49.21<br>47.84|
|WaRP (Ours)|**72.99**<br>**68.10**<br>**64.31**<br>**61.30**<br>**58.64**<br>**56.08**<br>**53.40**<br>**51.72**<br>**50.65**|



our method explores the ‘new parameter space’ where the axes are aligned with many flat directions along which the model is stably updated, by reparameterzing the weight as in equation (4). With a careful construction of the new weight space, we can finetune along the _flat directions_ by properly identifying and freezing the important parameters at each session, enjoying both advantages of F2M (Shi et al., 2021) and FSLL (Mazumder et al., 2021), without any dedicated process to find the flat region in advance or additional regularization on model parameters. Although not directly targeting CIFSL, there is another work (Liu et al., 2018) that reparameterizes the weight using a rotation matrix for tackling continual learning. Its focus is mainly on a good diagonalization of the Fisher information matrix (FIM) that shows up in the objective function of (Kirkpatrick et al., 2017). In our work, however, the role of rotation is different in that we rotate the axes of loss landscape for identifying/utilizing the flat directions aligned with new basis by further exploiting the low-rankness property. 

**Utilizing Low-Rankness Property of Activation.** There are a few works that exploit the lowrankness property of activation in deep neural network (Suzuki et al., 2020b;a; Saha et al., 2021; Wang et al., 2021). This suggests that the inputs are highly concentrated on the subspace spanned by only a few number of certain orthogonal vectors. (Suzuki et al., 2020b; Wang et al., 2021) confirm this property by showing the empirical distribution of singular values at some layers. The authors of (Suzuki et al., 2020a) proposed a network compression scheme by exploiting this property which brings on small number of degrees of freedom. In (Suzuki et al., 2020b), the authors derived the tight generalization bound of deep neural network with the help of high compressibility of the model induced by this property. Another line of works (Saha et al., 2021; Wang et al., 2021) try to solve continual learning problem by utilizing the low-rankness property. At each learning session, they update the model parameters using the gradient orthogonal to the subspace which the previous tasks are concentrated on. Note that our work turns out to generalize the update process of (Saha et al., 2021; Wang et al., 2021) in that if we freeze the parameters of the weight matrix in a column-wise manner, the gradient update of our method can be reduced to that of these works as special cases. 

## 5 EXPERIMENTS 

## 5.1 EXPERIMENTAL SETUP & DETAILS 

**Datasets.** We evaluate our method on three benchmark datasets, CIFAR100 (Krizhevsky et al., 2019), _mini_ ImageNet (Vinyals et al., 2016) and CUB200 (Wah et al., 2011), in the CIFSL setting. For both CIFAR100 and _mini_ ImageNet, we split the total of 100 classes into 60 classes (for base classes) and 40 classes (for novel classes). The 40 novel classes are divided into 8 different sets having 5 categories each, and allocated to each incremental session following the 5-way 5-shot setting. For CUB200, we split the total of 200 classes into 100 for the base classes and the remaining 100 for the novel classes, where the 100 novel classes are divided into 10 different sets having 10 categories each, following the 10-way 5-shot setting in each incremental session. We follow the same split configuration proposed by TOPIC (Tao et al., 2020) in all datasets. 

**Model Architecture.** For the embedding network, following the standard setup in (Tao et al., 2020; Mazumder et al., 2021; Shi et al., 2021), we use ResNet20 for CIFAR100 and use ResNet18 (He et al., 2016) for the others. Additionally, to see how a wider model affects the ability of our method for making additional room for learning new classes, we also conduct experiments with ResNet18[1] 

> 1The ResNet18 model architecture for CIFAR100 dataset is adopted from https://github.com/ kuangliu/pytorch-cifar/blob/master/models/resnet.py 

7 

Published as a conference paper at ICLR 2023 

Table 2: Accuracy on CUB200 dataset under 10-way 5-shot incremental few-shot learning setup. 

|Method|Session|
|---|---|
||1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11|
|Finetuning<br>iCaRL (Rebuff et al.,2017)<br>Rebalance (Hou et al.,2019)<br>GPM (Saha et al.,2021)<br>EEIL (Castro et al.,2018)<br>TOPIC (Tao et al.,2020)<br>FSLL (Mazumder et al.,2021)<br>FSLL+SS (Mazumder et al.,2021)<br>CEC (Zhang et al.,2021)<br>F2M (Shi et al.,2021)|**77.74**<br>73.49<br>68.29<br>60.60<br>51.62<br>34.37<br>25.84<br>23.36<br>19.01<br>13.05<br>12.24<br>75.95<br>60.90<br>57.65<br>54.51<br>50.83<br>48.21<br>46.95<br>45.74<br>43.21<br>43.01<br>41.27<br>77.44<br>58.10<br>50.15<br>44.80<br>39.12<br>34.44<br>31.73<br>29.75<br>27.56<br>26.93<br>25.30<br>**77.74**<br>73.92<br>70.49<br>66.38<br>64.51<br>62.03<br>60.76<br>59.10<br>56.55<br>56.02<br>54.82<br>68.68<br>53.63<br>47.91<br>44.20<br>36.30<br>27.46<br>25.93<br>24.70<br>23.95<br>24.13<br>22.11<br>68.68<br>62.49<br>54.81<br>49.99<br>45.25<br>41.40<br>38.35<br>35.36<br>32.22<br>28.31<br>26.28<br>72.77<br>69.33<br>65.51<br>62.66<br>61.10<br>58.65<br>57.78<br>57.26<br>55.59<br>55.39<br>54.21<br>75.63<br>71.81<br>68.16<br>64.32<br>62.61<br>60.10<br>58.82<br>58.70<br>56.45<br>56.41<br>55.82<br>75.85<br>71.94<br>68.50<br>63.50<br>62.43<br>58.27<br>57.73<br>55.81<br>54.83<br>53.52<br>52.28<br>77.13<br>73.92<br>70.27<br>66.37<br>64.34<br>61.69<br>60.52<br>59.38<br>57.15<br>56.94<br>55.89|
|WaRP (Ours)|**77.74**<br>**74.15**<br>**70.82**<br>**66.90**<br>**65.01**<br>**62.64**<br>**61.40**<br>**59.86**<br>**57.95**<br>**57.77**<br>**57.01**|



Table 3: Accuracy on CIFAR100 dataset under 5-way 5-shot incremental few-shot learning setup. 

|Method|Session|
|---|---|
||1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9|
|Finetuning<br>GPM (Saha et al.,2021)<br>FSLL (Mazumder et al.,2021)<br>FSLL+SS (Mazumder et al.,2021)<br>CEC (Zhang et al.,2021)<br>F2M (Shi et al.,2021)|80.31<br>75.43<br>70.82<br>66.26<br>62.21<br>59.08<br>56.43<br>53.58<br>51.05<br>80.31<br>75.54<br>71.27<br>66.97<br>63.63<br>60.48<br>57.92<br>55.57<br>53.18<br>80.24<br>74.32<br>69.03<br>64.75<br>61.37<br>58.56<br>56.48<br>54.47<br>52.27<br>**81.42**<br>74.79<br>69.33<br>64.97<br>61.60<br>58.92<br>57.06<br>55.28<br>53.12<br>79.69<br>74.77<br>70.24<br>66.13<br>62.93<br>59.94<br>57.67<br>55.63<br>53.44<br>79.78<br>74.93<br>70.68<br>66.58<br>63.45<br>60.51<br>58.30<br>56.20<br>53.99|
|WaRP (Ours)|80.31<br>**75.86**<br>**71.87**<br>**67.58**<br>**64.39**<br>**61.34**<br>**59.15**<br>**57.10**<br>**54.74**|



on CIFAR100, which is wider than ResNet20. We randomly initialize the network before training on base classes for both CIFAR100 and _mini_ ImageNet. For CUB200, we deploy the Pytorch’s built-in pretrained model for initialization as existing works have done. 

**Evaluation & Design Parameter.** At each session, we evaluate our method by measuring the accuracy on test samples from all encountered classes so far. We conduct 5 simulations under different random seeds and report average values. Regarding the hyperparameters, the parameter keeping ratio is set to be _α_ = 0 _._ 1 for all experiments. We apply WaRP to the last 1 or 2 resnet blocks while the remaining parameters including batchnorm layers are fixed after the pretraining stage in the first session. For computing the output of the classifier, we use the standard dot product in the beginning, and replace it with cosine similarity (Gidaris & Komodakis, 2018) after adopting the prototypes as a classifier. More detailed hyperparameter settings can be found in Appendix A.1. 

**Baselines.** We consider the following recent methods as baselines: iCaRL (Rebuffi et al., 2017), Rebalance (Hou et al., 2019), GPM (Saha et al., 2021), EEIL (Castro et al., 2018), TOPIC (Tao et al., 2020), FSLL (Mazumder et al., 2021), CEC (Zhang et al., 2021) and F2M (Shi et al., 2021). The performance of these baselines are mostly taken from (Zhang et al., 2021; Shi et al., 2021) for fair comparisons. We also consider simple finetuning, which is equivalent to our method with _α_ = 0 using a smaller learning rate. 

## 5.2 MAIN EXPERIMENTAL RESULTS 

We first report the evaluated accuracy of each scheme on _mini_ ImageNet, CUB200 and CIFAR100 for ResNet18, in Tables 1, 2 and 3, respectively. Due to space limitation, the result on CIFAR100 for ResNet20 is left to Appendix A.3. As can be seen from the results in Tables 1 and 2, WaRP consistently outperforms all the baselines in whole incremental sessions. More specifically, iCaRL, Rebalance and EEIL are far below WaRP, since all these methods are not originally developed for handling the lack of data in CIFSL. The naive finetuning method shows poor performance in all results as the model is highly overfitted to new classes and forgets previous ones rapidly due to scarcity of data. It is worth noting that the performance gain of WaRP against one of SOTA methods, F2M, is 2.68% for _mini_ ImageNet and 1.12% for CUB200. For CIFAR100 dataset, since there are no prior results for ResNet18, we reproduced the results of the following SOTA methods: GPM, FSLL, CEC and F2M. In Table 3, the trend is consistent with the results on _mini_ ImageNet and CUB200, where the performance gains over the F2M and CEC are 0.75% and 1.30%, respectively. 

To see the effect of finetuning in WaRP clearly, we compare our method with a simple and strong baseline which takes the prototype of each class as classifier for both base and novel classes. Since this baseline is equivalent to the case of our method with no finetuning, we can directly see the pure gains of finetuning in WaRP by comparing with this baseline. As can seen from the results on CUB200 in Table 4, this prototype-based baseline already outperforms or is comparable to SOTA methods, which is consistent with the observation from F2M (Shi et al., 2021). We can see performance gains 

8 

Published as a conference paper at ICLR 2023 

Table 4: Performance improvement of WaRP over prototype-based baseline on CUB200 dataset. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning_images/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning.pdf-0009-02.png)


**----- Start of picture text -----**<br>
Session<br>Method<br>1 2 3 4 5 6 7 8 9 10 11<br>Baseline (Prototype) 77.74 73.88 70.40 66.45 64.40 61.88 60.46 58.89 56.93 56.48 55.46<br>WaRP (Ours) 77.74 74.15 70.82 66.90 65.01 62.64 61.40 59.86 57.95 57.77 57.01<br>Improvement +0.27 +0.42 +0.45 +0.61 +0.76 +0.94 +0.97 +1.02 +1.29 +1.55<br>80 CUB200 80  mini ImageNet  mini ImageNet<br>53<br>75 70<br>70 52<br>60<br>6560 Base: WaRPBase: FSLL 50 51<br>5550 Base: PrototypeNovel: WaRPNovel: FSLL 4030 New + Gs (WaRP)New + WsStandard + GsStandard + Ws 5049<br>45 Novel: Prototype<br>40 20 48<br>35 10 47<br>30 0<br>2 3 4 5 6 7 8 9 10 11 1 2 3 4 5 6 7 8 9 6 7 8 9<br>Session Session Session<br>Accuracy (%) Accuracy (%) Accuracy (%)<br>**----- End of picture text -----**<br>


(a) Accuracy on base/novel classes. (b) Effect of basis and score. (c) Performance with varying _α_ . 

Figure 3: Further studies on WaRP using CUB200 and _mini_ ImageNet datasets. 

of WaRP over this strong baseline coherently in all incremental sessions, where the gain increases as the session progresses. 

## 5.3 FURTHER STUDIES ON WARP 

**Performance on Base and Novel Classes.** In Figure 3a, we report the accuracies on base classes and novel classes separately. We compare our WaRP with FSLL (Mazumder et al., 2021), which also finetunes the model by selecting trainable parameters as ours, but in the original parameter space defined with the standard basis. We also report the performance of the prototype-based baseline considered in Table 4. We have the following key observations in Figure 3a. First, for all sessions, the base class performance of WaRP sacrifices little compared to the prototype baseline (0.61% drop in the last session) while achieving better result than FSLL (1.40% gain in the last session). This indicates that WaRP can effectively keep the knowledge of base classes throughout the learning process. Regarding the novel classes, WaRP has considerable gains compared to both FSLL (1.2%2.8% gain) and the prototype-based baseline (3%-5% gain), indicating that WaRP effectively provides additional room for learning new classes in the new space, compared to FSLL which may limit the learnability on new classes due to additional regularization. The overall results confirm the ability of WaRP for preserving the knowledge of previous classes and learning new classes. 

**Ablation on Choices of Basis/Score.** To see how the proposed new basis and score criterion work in a highly complementary manner, in Figure 3b, we compare the performance of 4 different (basis, score) combinations: we consider standard basis and new basis for the basis candidates. For the score criterion, we consider weight score (Ws) and gradient score (Gs), which are the magnitudes of weight and gradient, respectively. As we can see, if we do not apply our new space and score criterion simultaneously, the performance extremely degrades as the session progresses while our method shows much better performance than other combinations. 

**Ablation on Varying** _**α**_ **.** In Figure 3c, we consider ablation on varying _α_ ranging from 0.01 to 0.50 to see the robustness of WaRP against the perturbing hyperparameter. For an extreme case with _α_ = 0 _._ 01, the accuracy is low since most of model parameters are updated and thus the previous knowledge cannot be preserved. However, if _α_ is larger than a specific small threshold (0.05 in this case), WaRP consistently provides additional gain against the baseline, suggesting that the performance gain is robust to perturbing _α_ . 

## 6 CONCLUSION 

In this paper, we introduced the concept of weight space rotation process (WaRP), a general framework for reparameterizing the weight matrix of the neural network. By taking advantage of WaRP, we proposed a novel strategy for CIFSL that effectively learns new classes while preserving previous knowledge, alleviating forgetting/overfitting issues. Experimental results on various benchmarks confirmed the advantage of our method over the state-of-the-arts. Although we focused on CIFSL in this paper, the general concept of WaRP can be utilized in any deep learning applications where the knowledge of the model should be kept in a few parameters, which we leave for a future work. 

9 

Published as a conference paper at ICLR 2023 

## REPRODUCIBILITY STATEMENT 

The detailed experimental setups including datasets, model architectures, evaluation metrics and hyperparameters are described in Sec. 5.1 and Appendix A.1. We also provide the source code in Supplementary Material. We run the simulations 5 times with different random seeds using NVIDIA GeForce RTX 3090 GPU machine and report average values. The code is available at https://github.com/EdwinKim3069/WaRP-CIFSL. 

## ACKNOWLEDGMENTS 

This work was supported by Samsung Electronics Co., Ltd. and by IITP funds from MSIT of Korea (No. 2020-0-00626). 

## REFERENCES 

- Afra Feyza Akyurek, Ekin Aky¨ urek, Derry Wijaya, and Jacob Andreas.¨ Subspace regularizers for few-shot class incremental learning. In _International Conference on Learning Representations_ , 2022. 

- Francisco M Castro, Manuel J Mar´ın-Jimenez, Nicol´ as Guil, Cordelia Schmid, and Karteek Alahari.´ End-to-end incremental learning. In _Proceedings of the European conference on computer vision (ECCV)_ , pp. 233–248, 2018. 

- Kuilin Chen and Chi-Guhn Lee. Incremental few-shot learning via vector quantization in deep embedded space. In _International Conference on Learning Representations_ , 2021. 

- Ali Cheraghian, Shafin Rahman, Pengfei Fang, Soumava Kumar Roy, Lars Petersson, and Mehrtash Harandi. Semantic-aware knowledge distillation for few-shot class-incremental learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 2534–2543, 2021a. 

- Ali Cheraghian, Shafin Rahman, Sameera Ramasinghe, Pengfei Fang, Christian Simon, Lars Petersson, and Mehrtash Harandi. Synthesized feature based few-shot class-incremental learning on a mixture of subspaces. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 8661–8670, 2021b. 

- Zhixiang Chi, Li Gu, Huan Liu, Yang Wang, Yuanhao Yu, and Jin Tang. Metafscil: A meta-learning approach for few-shot class incremental learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 14166–14175, 2022. 

- Pradnya Desai, Phung Lai, NhatHai Phan, and My T Thai. Continual learning with differential privacy. In _International Conference on Neural Information Processing_ , pp. 334–343. Springer, 2021. 

- Spyros Gidaris and Nikos Komodakis. Dynamic few-shot visual learning without forgetting. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 4367–4375, 2018. 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 770–778, 2016. 

- Michael Hersche, Geethan Karunaratne, Giovanni Cherubini, Luca Benini, Abu Sebastian, and Abbas Rahimi. Constrained few-shot class-incremental learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 9057–9067, 2022. 

- Saihui Hou, Xinyu Pan, Chen Change Loy, Zilei Wang, and Dahua Lin. Learning a unified classifier incrementally via rebalancing. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 831–839, 2019. 

- Ching-Yi Hung, Cheng-Hao Tu, Cheng-En Wu, Chien-Hung Chen, Yi-Ming Chan, and Chu-Song Chen. Compacting, picking and growing for unforgetting continual learning. _Advances in Neural Information Processing Systems_ , 32, 2019. 

10 

Published as a conference paper at ICLR 2023 

- James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. _Proceedings of the national academy of sciences_ , 114 (13):3521–3526, 2017. 

- Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Cifar-10, cifar-100 (canadian institute for advanced research). _URL: http://www. cs. toronto. edu/kriz/cifar. html (Last accessed_ , 2019. 

- Anna Kukleva, Hilde Kuehne, and Bernt Schiele. Generalized and incremental few-shot learning by explicit learning and calibration without forgetting. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 9020–9029, 2021. 

- Namhoon Lee, Thalaiyasingam Ajanthan, and Philip H. S. Torr. Snip: single-shot network pruning based on connection sensitivity. In _International Conference on Learning Representations_ , 2019. 

- Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. In _International Conference on Learning Representations_ , 2017. 

- Huan Liu, Li Gu, Zhixiang Chi, Yang Wang, Yuanhao Yu, Jun Chen, and Jin Tang. Few-shot class-incremental learning via entropy-regularized data-free replay. In _European Conference on Computer Vision_ , pp. 146–162. Springer, 2022. 

- Xialei Liu, Marc Masana, Luis Herranz, Joost Van de Weijer, Antonio M Lopez, and Andrew D Bagdanov. Rotate your networks: Better weight consolidation and less catastrophic forgetting. In _2018 24th International Conference on Pattern Recognition (ICPR)_ , pp. 2262–2268. IEEE, 2018. 

- Yuchen Liu, Zhixin Shu, Yijun Li, Zhe Lin, Federico Perazzi, and Sun-Yuan Kung. Content-aware gan compression. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 12156–12166, 2021. 

- Pratik Mazumder, Pravendra Singh, and Piyush Rai. Few-shot lifelong learning. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pp. 2337–2345, 2021. 

- Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In _Proceedings of the IEEE conference on Computer Vision and Pattern Recognition_ , pp. 2001–2010, 2017. 

- Mengye Ren, Renjie Liao, Ethan Fetaya, and Richard Zemel. Incremental few-shot learning with attention attractor networks. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- Gobinda Saha, Isha Garg, and Kaushik Roy. Gradient projection memory for continual learning. In _International Conference on Learning Representations_ , 2021. 

- Joan Serra, Didac Suris, Marius Miron, and Alexandros Karatzoglou. Overcoming catastrophic forgetting with hard attention to the task. In _International Conference on Machine Learning_ , pp. 4548–4557. PMLR, 2018. 

- Guangyuan Shi, Jiaxin Chen, Wenlong Zhang, Li-Ming Zhan, and Xiao-Ming Wu. Overcoming catastrophic forgetting in incremental few-shot learning by finding flat minima. _Advances in Neural Information Processing Systems_ , 34:6747–6761, 2021. 

- Taiji Suzuki, Hiroshi Abe, Tomoya Murata, Shingo Horiuchi, Kotaro Ito, Tokuma Wachi, So Hirai, Masatoshi Yukishima, and Tomoaki Nishimura. Spectral pruning: Compressing deep neural networks via spectral analysis and its generalization error. In _IJCAI_ , 2020a. 

- Taiji Suzuki, Hiroshi Abe, and Tomoaki Nishimura. Compression based bound for non-compressed network: unified generalization error analysis of large compressible deep neural network. In _International Conference on Learning Representations_ , 2020b. 

- Xiaoyu Tao, Xiaopeng Hong, Xinyuan Chang, Songlin Dong, Xing Wei, and Yihong Gong. Few-shot class-incremental learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 12183–12192, 2020. 

11 

Published as a conference paper at ICLR 2023 

- Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. _Advances in neural information processing systems_ , 29, 2016. 

- Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011. 

- Shipeng Wang, Xiaorong Li, Jian Sun, and Zongben Xu. Training networks in null space of feature covariance for continual learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 184–193, 2021. 

- Mitchell Wortsman, Vivek Ramanujan, Rosanne Liu, Aniruddha Kembhavi, Mohammad Rastegari, Jason Yosinski, and Ali Farhadi. Supermasks in superposition. _Advances in Neural Information Processing Systems_ , 33:15173–15184, 2020. 

- Sung Whan Yoon, Do-Yeon Kim, Jun Seo, and Jaekyun Moon. Xtarnet: Learning to extract taskadaptive representation for incremental few-shot learning. In _International Conference on Machine Learning_ , pp. 10852–10860. PMLR, 2020. 

- Chi Zhang, Nan Song, Guosheng Lin, Yun Zheng, Pan Pan, and Yinghui Xu. Few-shot incremental learning with continually evolved classifiers. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 12455–12464, 2021. 

- Da-Wei Zhou, Fu-Yun Wang, Han-Jia Ye, Liang Ma, Shiliang Pu, and De-Chuan Zhan. Forward compatible few-shot class-incremental learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 9046–9056, 2022a. 

- Da-Wei Zhou, Han-Jia Ye, Liang Ma, Di Xie, Shiliang Pu, and De-Chuan Zhan. Few-shot classincremental learning by sampling multi-phase tasks. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2022b. 

- Kai Zhu, Yang Cao, Wei Zhai, Jie Cheng, and Zheng-Jun Zha. Self-promoted prototype refinement for few-shot class-incremental learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 6801–6810, 2021. 

12 

Published as a conference paper at ICLR 2023 

## A APPENDIX 

## A.1 HYPERPARAMETERS 

**Optimizer.** We use SGD optimizer with a momentum of 0 _._ 9 throughout the whole sessions for all our simulations. 

**First Session (** _k_ = 1 **).** We pretrain the model with the batch size of 128 in the first session for all datasets. The number of pretraining epochs is 210 for ResNet18 on both CIFAR100 and _mini_ ImageNet and 300 for ResNet20 on CIFAR100. Regarding ResNet18 on CUB200, as we adopt Pytorch built-in pretrained model as done in most of previous works (Tao et al., 2020; Zhang et al., 2021), we pretrain the model using a smaller learning rate of 0.01 following (Chen & Lee, 2021), and the total number of epochs is 120. The learning rate decays by 1/10 at last 90, 60, 30 epoch to the end of training (for example, the learning rate decays at 210, 240, 270 if the total training epochs is 300). 

**Incremental Sessions (** _k ≥_ 2 **).** Following (Mazumder et al., 2021), we finetune the model with training epochs of 30 for all datasets/architectures in WaRP experiments. We use the learning rate of 0.01 for ResNet18 and 0.001 for ResNet20. When searching the learning rate, we randomly split trainset of base classes into train/validation so that the validation set has a ratio of about 10% in total. We also randomly sample the data from trainset of novel classes (different from the 5-shots that we use for training) and use them as validation set for novel classes. We search the learning rate within the grids of [1e-1, 1e-2, 1e-3] on CUB200 for ResNet18 and on CIFAR100 for ResNet20. As for the _mini_ ImageNet and CIFAR100 for ResNet18, we follow the setting of CUB200. 

**Overall Settings for WaRP Algorithm.** Throughout the whole datasets/architectures, we randomly sample 20 batches (20 _×_ 128 = 2 _,_ 560 samples) from base classes and use them for computing SVD in the first session. For computing score criterion, we set the batch size in equation (9) to be _nb_ = 60 for base classes and _nb_ = 1 for new classes. As for the layers to be applied weight space rotation, we apply WaRP to the last two resnet blocks (out of four blocks in total) for ResNet18 model and apply to the last one block (out of three blocks in total) for ResNet20 model. We do not apply WaRP to the batch-normalization layer. Note that we freeze all the layers, to which WaRP is not applied, after completing the first session. The hyperparameter of keeping ratio _α_ at each session is set to be 0.1. The temperature for cosine similarity (Gidaris & Komodakis, 2018) in classifier is set to be a constant value of 16, following (Zhang et al., 2021). 

## A.2 IMPLEMENTATION OF WARP FOR CONVOLUTIONAL LAYER CASE 

We first denote the filter (kernel) of convolutional layer by _W ∈_ R _[m][×][n][×][k][×][k]_ where _k_ is size of kernel, _m_ is the target number of output channels and _n_ is the number of input channels, and let the activation (input of the layer) be denoted by Φ _∈_ R _[n][×][h][×][w]_ where _h_ and _w_ are height and width of activation feature map respectively. For computing SVD of activation map, all the patches in the feature map, that the kernel slides across, are flattened as a vector form and all these flattened vectors are stacked to be 2-dimensional matrix. For example, if the kernel size _k_ is 3 and the stride and padding are 1 respectively, the shape of stacked version of activation matrix is (3 _×_ 3 _× n, h × w_ ) and the resulting unitary matrix would be _U_ 2-dim _∈_ R[9] _[n][×]_[9] _[n]_ . Then after we reshape the filter _W_ into 2-dimensional weight matrix as _W_ 2-dim _∈_ R _[m][×][nk]_[2] , the remaining process is the same with fully-connected layer case; we reparameterize the layer with _W_[�] 2-dim = _V_ 2-dim _[⊤][W]_[2-dim] _[U]_[2-dim][where] _[V]_[2-dim] _[∈]_[R] _[m][×][m]_[and] _U_ 2-dim _∈_ R _[nk]_[2] _[×][nk]_[2] are unitary matrices. 

During computing the loss, there are two options we can choose to implement _W ← V WU_[�] _[⊤]_ . The first one is just to compute _W_ 2-dim = _V_ 2-dim _W_[�] 2-dim _U_ 2-dim _[⊤]_[and][reshape] _[W]_[2-dim][to][be][the][same][with] _W ∈_ R _[m][×][n][×][k][×][k]_ at every loss computation. This approach is easy to implement but might be computationally inefficient. To pursuit a slight computational efficiency, we take the second approach: configure this layer as three convolutions with filters of _V_ conv _∈_ R _[m][×][m][×]_[1] _[×]_[1] _, W_[�] conv _∈_ R _[m][×][nk]_[2] _[×]_[1] _[×]_[1] and _U_ conv _∈_ R _[nk]_[2] _[×][n][×][k][×][k]_ where _V_ conv _, W_[�] conv and _U_ conv are reshaped version of _V_ 2-dim _, W_[�] 2-dim and _U_ 2-dim respectively. 

13 

Published as a conference paper at ICLR 2023 

## A.3 ADDITIONAL EXPERIMENTAL RESULTS 

We report additional results on CIFAR100 for ResNet20 as shown in Table 5. As ResNet20 is too narrow (e.g., the number of channels in the last layer is 64) to make additional room that can be reaped by WaRP for learning new classes, the gain is rather minimal compared to the results for wider network of ResNet18 (the number of channels in the last layer is 512) as shown in Table 1, 2 and 3. However, as can be seen in Table 6, WaRP still consistently provides gains in all sessions against prototype-based baseline which we considered in Sec. 5.2. Note that the prototype-based baseline is strong in that this baseline already outperforms or is comparable to state-of-the-art methods, which is also confirmed in F2M (Shi et al., 2021). Moreover, the effectiveness of WaRP is not restricted by the dataset; result on the same dataset, CIFAR100, for ResNet18 achieves state-of-the-art as can be seen in Table 3 and reads appreciable gains compared to ResNet20 case as shown in Table 7. Thus, the limited gain in small network can be easily handled by replacing the layers of the model with wider layers in practical deployment of our method. 

Table 8 and Figure 4 refers to the results on repeated experiments of Table 4 and Figure 3c on different datasets respectively. As shown in Figure 4, WaRP consistently provides the gains against strong baseline (prototype) once the _α_ is larger than a certain threshold value, which suggests that the WaRP is fairly robust to perturbing hyperparameter _α_ value. 

Table 5: Accuracy on CIFAR100 dataset under 5-way 5-shot incremental few-shot learning setup for ResNet20. 

|Method|Session|
|---|---|
||1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9|
|Finetuning<br>iCaRL (Rebuff et al.,2017)<br>Rebalance (Hou et al.,2019)<br>GPM (Saha et al.,2021)<br>EEIL (Castro et al.,2018)<br>TOPIC (Tao et al.,2020)<br>FSLL (Mazumder et al.,2021)<br>FSLL+SS (Mazumder et al.,2021)<br>CEC (Zhang et al.,2021)<br>F2M (Shi et al.,2021)|74.21<br>69.81<br>65.61<br>61.59<br>58.53<br>55.59<br>53.29<br>51.20<br>49.06<br>72.05<br>65.35<br>61.55<br>57.83<br>54.61<br>51.74<br>49.71<br>47.49<br>45.03<br>**74.45**<br>67.74<br>62.72<br>57.14<br>52.78<br>48.62<br>45.56<br>42.43<br>39.22<br>74.21<br>69.84<br>65.66<br>61.69<br>58.53<br>55.56<br>53.29<br>51.27<br>49.16<br>64.10<br>53.11<br>43.71<br>35.15<br>28.96<br>24.98<br>21.01<br>17.26<br>15.85<br>64.10<br>55.88<br>47.07<br>45.16<br>40.11<br>36.38<br>33.96<br>31.55<br>29.37<br>64.10<br>55.85<br>51.71<br>48.59<br>45.34<br>43.25<br>41.52<br>39.81<br>38.16<br>66.76<br>55.52<br>52.20<br>49.17<br>46.23<br>44.64<br>43.07<br>41.20<br>39.57<br>73.07<br>68.88<br>65.26<br>61.19<br>58.09<br>55.57<br>53.22<br>51.34<br>49.14<br>71.45<br>68.10<br>64.43<br>60.80<br>57.76<br>55.26<br>**53.53**<br>**51.57**<br>**49.35**|
|WaRP (Ours)|74.21<br>**69.96**<br>**65.86**<br>**61.92**<br>**58.74**<br>**55.79**<br>53.50<br>51.51<br>49.33|



Table 6: Performance improvement of WaRP over prototype-based baseline on CIFAR100 dataset for ResNet20. 

|Method|Session|
|---|---|
||1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9|
|Baseline (Prototype)<br>WaRP (Ours)|74.21<br>69.83<br>65.67<br>61.72<br>58.55<br>55.62<br>53.30<br>51.27<br>49.11<br>74.21<br>69.96<br>65.86<br>61.92<br>58.74<br>55.79<br>53.50<br>51.51<br>49.33|
|Improvement|**+0.13**<br>**+0.19**<br>**+0.20**<br>**+0.19**<br>**+0.17**<br>**+0.20**<br>**+0.24**<br>**+0.22**|



Table 7: Performance improvement of WaRP over prototype-based baseline on CIFAR100 dataset for ResNet18. 

|Method|Session|
|---|---|
||1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9|
|Baseline (Prototype)<br>WaRP (Ours)|80.31<br>75.54<br>71.22<br>67.02<br>63.73<br>60.65<br>58.30<br>56.11<br>53.84<br>80.31<br>75.86<br>71.87<br>67.58<br>64.39<br>61.34<br>59.15<br>57.10<br>54.74|
|Improvement|**+0.32**<br>**+0.65**<br>**+0.56**<br>**+0.66**<br>**+0.69**<br>**+0.85**<br>**+0.99**<br>**+0.90**|



Additionally, we further compare our method with more recent works (Zhou et al., 2022b; Chi et al., 2022; Hersche et al., 2022; Liu et al., 2022) on _mini_ ImageNet dataset. Note that as (Hersche et al., 2022) adopted ResNet12 network for evaluating their method, we reproduced the results of (Hersche et al., 2022) by adopting ResNet18 to match the architecture. As shown in the table 9, we confirm our method outperforms all the baselines, except for (Zhou et al., 2022b), but still shows better performance in most sessions. Moreover, we found that (Akyurek¨ et al., 2022) provided the results with ResNet12 architecture as done in (Hersche et al., 2022), thus we also evaluate our 

14 

Published as a conference paper at ICLR 2023 

Table 8: Performance improvement of WaRP over prototype-based baseline on _mini_ ImageNet dataset. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning_images/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning.pdf-0015-02.png)


**----- Start of picture text -----**<br>
Session<br>Method<br>1 2 3 4 5 6 7 8 9<br>Baseline (Prototype) 72.99 68.07 64.09 60.83 58.06 55.38 52.68 50.88 49.70<br>WaRP (Ours) 72.99 68.10 64.31 61.30 58.64 56.08 53.40 51.72 50.65<br>Improvement +0.03 +0.22 +0.47 +0.58 +0.70 +0.72 +0.84 +0.95<br>CIFAR100 for ResNet18 CIFAR100 for ResNet20 CUB200<br>56 49.5 60<br>49.4 58<br>55 49.3 56<br>49.2 54<br>54 49.1 52<br>49 50<br>7 8 9 9 8 9 10 11<br>Session Session Session<br>(a) Performance with varying  α (b) Performance with varying  α (c) Performance with varying  α<br>on CIFAR100 for ResNet18. on CIFAR100 for ResNet20. on CUB200.<br>Accuracy (%) Accuracy (%) Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 4: Performance with varying _α_ on different datasets/architectures. 

method for ResNet12 architecture and compare with (Akyurek et al.¨ , 2022; Hersche et al., 2022) on _mini_ ImageNet. As can be seen in the Table 10, our method shows better performance than the baselines which is consistent with the case for ResNet18. 

Table 9: Accuracy on _mini_ ImageNet dataset under 5-way 5-shot incremental few-shot setup for ResNet18. 

|Method|Session|
|---|---|
||1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9|
|LIMIT (Zhou et al.,2022b)<br>MetaFSCIL (Chi et al.,2022)<br>C-FSCIL Mode 1 (Hersche et al.,2022)<br>C-FSCIL Mode 2 (Hersche et al.,2022)<br>C-FSCIL Mode 3 (Hersche et al.,2022)<br>Entropy-reg (Liu et al.,2022)|72.32<br>**68.47**<br>64.30<br>60.78<br>57.95<br>55.07<br>52.70<br>50.72<br>49.19<br>72.04<br>67.94<br>63.77<br>60.29<br>57.58<br>55.16<br>52.90<br>50.79<br>49.19<br>64.72<br>59.63<br>55.43<br>51.95<br>48.99<br>46.44<br>43.86<br>41.76<br>39.78<br>64.73<br>59.68<br>55.41<br>52.04<br>49.31<br>46.65<br>44.11<br>42.09<br>40.15<br>64.73<br>59.75<br>55.23<br>51.97<br>49.05<br>46.36<br>43.62<br>41.43<br>39.69<br>71.84<br>67.12<br>63.21<br>59.77<br>57.01<br>53.95<br>51.55<br>49.52<br>48.21|
|WaRP (Ours)|**72.99**<br>68.10<br>**64.31**<br>**61.30**<br>**58.64**<br>**56.08**<br>**53.40**<br>**51.72**<br>**50.65**|



Lastly, we compare with (Hersche et al., 2022) on CIFAR100 dataset for ResNet12 (we further evaluate our method on CIFAR100 for ResNet12 instead of ResNet18), as (Hersche et al., 2022) also provided the results on CIFAR100 for ResNet12. Table 11 indicates that our method also shows the improved performance compared to (Hersche et al., 2022) on CIFAR100, which is consistent with the results on _mini_ ImageNet. 

## A.4 VALIDITY OF THE NEW BASIS ON NOVEL CLASSES 

As aforementioned, once the new basis is constructed in the first session, the new basis is fixed during the whole remaining sessions. The future incoming tasks are not used to reconstruct or modify the basis to avoid unexpected behavior or statistical unreliability due to their lack of training samples. Here one may argue that the new basis is suitable only for preserving the knowledge of the base classes. Recall that we apply WaRP only to the last few layers while remaining the other layers fixed to resolve the overfitting issue. Here, since the activation of each layer depends not only on the task itself but also on preceding layers, the important parameters identified in the new basis are shared among the base and novel classes, i.e., the novel classes are also aligned with the new basis to some extent. To figure out this, we measure the keep ratio of important parameters that is accumulated as the session progressed. As shown in Figure 5, although we set keep ratio _α_ as 0.1 for all sessions, the measured accumulated ratio does not grow by multiples of 0.1. Rather, the 

15 

Published as a conference paper at ICLR 2023 

Table 10: Accuracy on _mini_ ImageNet dataset under 5-way 5-shot incremental few-shot setup for ResNet12. 

|Method|Session|
|---|---|
||1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9|
|Subspace-reg (Aky¨urek et al.,2022)<br>C-FSCIL Mode 1 (Hersche et al.,2022)<br>C-FSCIL Mode 2 (Hersche et al.,2022)<br>C-FSCIL Mode 3 (Hersche et al.,2022)|80.37<br>73.92<br>69.00<br>65.10<br>61.73<br>58.12<br>54.98<br>52.21<br>49.65<br>76.37<br>70.94<br>66.36<br>62.64<br>59.31<br>56.02<br>53.14<br>51.04<br>48.87<br>76.45<br>71.23<br>66.71<br>63.01<br>60.09<br>56.73<br>53.94<br>52.01<br>50.08<br>76.40<br>71.14<br>66.46<br>63.29<br>60.42<br>57.46<br>54.78<br>53.11<br>51.41|
|WaRP (Ours)|**82.05**<br>**77.06**<br>**73.17**<br>**70.19**<br>**67.92**<br>**65.14**<br>**62.36**<br>**60.57**<br>**59.49**|



Table 11: Accuracy on CIFAR100 dataset under 5-way 5-shot incremental few-shot setup for ResNet12. 

|Method|Session|
|---|---|
||1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9|
|C-FSCIL Mode 1 (Hersche et al.,2022)<br>C-FSCIL Mode 2 (Hersche et al.,2022)<br>C-FSCIL Mode 3 (Hersche et al.,2022)|77.47<br>72.20<br>67.53<br>63.23<br>59.58<br>56.67<br>53.94<br>51.55<br>49.36<br>77.50<br>72.45<br>67.94<br>63.80<br>60.24<br>57.34<br>54.61<br>52.41<br>50.23<br>77.47<br>72.40<br>67.47<br>63.25<br>59.84<br>56.95<br>54.42<br>52.47<br>50.47|
|WaRP (Ours)|**79.30**<br>**75.27**<br>**71.49**<br>**67.60**<br>**64.55**<br>**61.90**<br>**59.89**<br>**57.82**<br>**55.49**|



accumulated ratio increases by some value smaller than _α_ . As can be seen from the observation, the important parameters at each session somewhat overlap since the activation also depends on the preceding layers. This suggests that the new basis constructed in the first session would be aligned with the future tasks to a certain extent. Hence the new basis is also eligible for preserving the knowledge of the novel classes with our score criterion in (9). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning_images/iclr_2023_Warping_the_Space_Weight_Space_Rotation_for_Class-Incremental_Few-Shot_Learning.pdf-0016-06.png)


**----- Start of picture text -----**<br>
CIFAR100 for ResNet18 CIFAR100 for ResNet20<br>0.26 0.2<br>0.18<br>0.22<br>0.16<br>0.18<br>0.14<br>0.14 0.12<br>0.1<br>0.1<br>1 3 5 7 9 1 3 5 7 9<br>Session Session<br> mini ImageNet CUB200<br>0.26 0.2<br>0.18<br>0.22<br>0.16<br>0.18<br>0.14<br>0.14 0.12<br>0.1<br>0.1<br>1 3 5 7 9 1 3 5 7 9 11<br>Session Session<br>Ratio of Important Params Ratio of Important Params<br>Ratio of Important Params Ratio of Important Params<br>**----- End of picture text -----**<br>


Figure 5: Accumulated keep ratios as the session progresses on various datasets. 

Along with this observation, we further verify the ability to preserve the novel tasks by showing how the mode accuracy on novel classes varies as session grows. The table 12 and 13 show the accuracy on _T_ 2 and _T_ 3 respectively, i.e., the first and second incremental novel task. We evaluated on _mini_ ImageNet dataset. 

Moreover, to single out the effectiveness of WaRP in preserving the previous knowledge, we further evaluate the accuracies on a certain novel task when the model knows task ID. In other words, the prediction is made when the output logits corresponding to this novel task are only given. By doing so, we can exclude the effect of forgetting induced by the interference from the output logits 

16 

Published as a conference paper at ICLR 2023 

Table 12: Performance on the first novel task _T_ 2 at each incremental session on _mini_ ImageNet dataset. 

|Method|Session|
|---|---|
||2<br>3<br>4<br>5<br>6<br>7<br>8<br>9|
|Prototype<br>WaRP (Ours)|21.52<br>20.68<br>19.88<br>19.28<br>18.92<br>18.24<br>17.84<br>17.04<br>**27.16**<br>**25.48**<br>**24.32**<br>**23.92**<br>**23.52**<br>**22.84**<br>**22.12**<br>**21.04**|



Table 13: Performance on the first novel task _T_ 3 at each incremental session on _mini_ ImageNet dataset. 

|Method|Session|
|---|---|
||3<br>4<br>5<br>6<br>7<br>8<br>9|
|Prototype<br>WaRP (Ours)|22.32<br>21.48<br>21.00<br>19.68<br>18.48<br>17.96<br>17.88<br>**27.76**<br>**26.76**<br>**25.84**<br>**23.64**<br>**22.00**<br>**21.60**<br>**21.52**|



corresponding to other tasks. As can be seen in the table 14 and 15, the new basis constructed using base classes is also suitable for preserving the knowledge of the novel classes. 

Table 14: Performance on the first novel task _T_ 2 on _mini_ ImageNet dataset when the task ID is known. 

|Method|Session|
|---|---|
||2<br>3<br>4<br>5<br>6<br>7<br>8<br>9|
|Prototype<br>WaRP (Ours)|58.92<br>58.92<br>58.92<br>58.92<br>58.92<br>58.92<br>58.92<br>58.92<br>**60.60**<br>**60.76**<br>**60.68**<br>**60.56**<br>**60.36**<br>**60.40**<br>**60.40**<br>**60.48**|



## A.5 COMPARISON WITH EXISTING NETWORK PRUNING SCHEMES 

As our method identifies the important parameters, we can compare our method against the network pruning schemes. We consider three network pruning methods: two for structured (channel/filter) pruning (Li et al., 2017; Liu et al., 2021) and one for unstructured pruning, SNIP (Lee et al., 2019). We adopted the methods of identifying parameters, including the score criterion and the structure of identification (e.g. whether identifying channel-wise or element-wise manner) proposed in these works. Although we set the ratio of important parameters _α_ as 0 _._ 1 in WaRP experiments, during the reproducing the results on the methods from (Li et al., 2017; Liu et al., 2021; Lee et al., 2019), we tuned the parameter of this ratio _α_ to get their best performances. We found that _α_ = 0 _._ 9 _∼_ 0 _._ 95 is suitable for (Li et al., 2017; Liu et al., 2021; Lee et al., 2019) as all of them still consider the original parameter space which exhibits the fundamental limitation as shown in the Figure 2. The results are evaluated on _mini_ ImageNet dataset. As can be seen in the table 16, our method outperforms all of score criterion suggested in pruning schemes we considered, which indicates that the new space and its compatible score criterion are very effective in solving CIFSL. 

## A.6 COMPUTATIONAL COST 

We first separate the computational aspects of our scheme into two parts and discuss them separately: (a) constructing new basis in the first session, (b) finetuning/inference in the incremental sessions. 

(a) After pretraining the model in the first session, we construct new basis by using singular value decomposition (SVD). The computational complexity of SVD is known as _O_ ( _n_[3] ) where the covariance of activation is in the shape of _n × n_ . Although constructing the new basis appears to have somewhat large complexity due to the SVD, we only need to construct it _just once_ in the first session. Moreover, the real time taken to construct the basis is not that significant compared to that of pretraining in the first session in practice. The table 17 shows the elapsed times for constructing the basis and pretraining with only 5 epochs on miniImageNet dataset. We tested on NVIDIA GeForce RTX 3090 GPU. 

(b) During finetuning in the new basis, the complexity induced in the layer, to which the WaRP is applied, increases as much as when training with 3 layers due to _V_ and _U_ multiplied on both sides of 

17 

Published as a conference paper at ICLR 2023 

Table 15: Performance on the first novel task _T_ 3 on _mini_ ImageNet dataset when the task ID is known. 

|Method|Session|
|---|---|
||3<br>4<br>5<br>6<br>7<br>8<br>9|
|Prototype<br>WaRP (Ours)|69.40<br>69.40<br>69.40<br>69.40<br>69.40<br>69.40<br>69.40<br>**71.04**<br>**70.72**<br>**70.80**<br>**70.64**<br>**70.64**<br>**70.52**<br>**70.44**|



Table 16: Comparison against various network pruning schemes on _mini_ ImageNet dataset. 

|Method|Session|
|---|---|
||1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9|
|Filter prune (Li et al.,2017) (ratio=0.90)<br>Filter prune (Li et al.,2017) (ratio=0.95)<br>Channel prune (Liu et al.,2021) (ratio=0.90)<br>Channel prune (Liu et al.,2021) (ratio=0.95)<br>SNIP (Lee et al.,2019) (ratio=0.90)<br>SNIP (Lee et al.,2019) (ratio=0.95)|72.99<br>66.57<br>56.81<br>41.19<br>29.98<br>24.24<br>19.44<br>17.52<br>13.15<br>72.99<br>67.15<br>62.07<br>55.77<br>49.75<br>44.88<br>38.39<br>36.17<br>33.14<br>72.99<br>66.77<br>61.19<br>56.78<br>53.05<br>49.34<br>45.39<br>42.75<br>40.37<br>72.99<br>67.81<br>63.75<br>60.58<br>57.86<br>55.26<br>52.50<br>50.71<br>49.55<br>72.99<br>66.98<br>62.42<br>59.16<br>56.52<br>54.11<br>51.40<br>49.63<br>48.44<br>72.99<br>67.65<br>62.41<br>58.38<br>55.48<br>53.04<br>50.38<br>48.59<br>47.48|
|WaRP (Ours)|72.99<br>**68.10**<br>**64.31**<br>**61.30**<br>**58.64**<br>**56.08**<br>**53.40**<br>**51.72**<br>**50.65**|



_W_ in practical implementation. However, this difference does not that significantly slow down the finetuning as the model is finetuned only with a small iterations using only a few labeled samples in the actual incremental session. The table 18 shows the elapsed times for finetuning in both new and original space on miniImageNet dataset (we report averaged time per each incremental session). We tested on NVIDIA GeForce RTX 3090 GPU. 

Moreover, we would like to highlight that, after completing the finetuning, we recover this layer to its original one, i.e., _W_ = _V WU_[�] _[⊤]_ so that the inference time complexity does not increase. 

## A.7 NOTES ON THE SPACE OF MATRIX 

In equation (4), we deal with the “space of matrix” and generalize the linear algebraic concepts (which we are familiar with in space R _[k]_ , for example) to this matrix space. Here, we define the inner product as _⟨A, B⟩_ = trace( _A[⊤] B_ ) in Sec. 2. In fact, this inner product actually can be seen as a typical dot product in the vector space R _[k]_ for any _k_ (i.e. **a** _·_ **b** =[�] _[k] i_ =1 _[a][i][b][i]_[ for any vectors] **[ a]** _[,]_ **[ b]** _[ ∈]_[R] _[k]_ where _ai_ and _bi_ are _i_ -th element of **a** and **b** respectively), if _A, B ∈_ R _[m][×][n]_ are flattened as a vector form. In other words, it can be easily seen that _⟨A, B⟩_ = _A_ flatten _· B_ flatten where _A_ flatten _, B_ flatten _∈_ R _[nm]_ is flattened version of _A_ and _B_ respectively. Thus, once _W_ and _Kij_ are flattened, we can easily interpret the equation (4) as the flattened version of _W_ and _Kij_ are in the vector space R _[nm]_ and the linear algebraic concepts we are familiar with in R _[k]_ such as linear combination, orthonormal vectors are directly adopted to this equation. 

18 

Published as a conference paper at ICLR 2023 

Table 17: Elapsed time for constructing new basis and pretraining the model with 5 epochs. 

||Construct basis|Pretrain with|5|epochs|
|---|---|---|---|---|
|Elapsed time (sec)|136.73|123.49|||



Table 18: Elapsed time for finetuning in new and original space. 

||New space|Original space|
|---|---|---|
|Elapsed time (sec)|1.2661|0.8472|



19 

