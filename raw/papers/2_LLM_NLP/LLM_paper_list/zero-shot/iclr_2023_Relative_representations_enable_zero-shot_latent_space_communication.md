# RELATIVE REPRESENTATIONS ENABLE - ZERO SHOT LATENT SPACE COMMUNICATION 

**Luca Moschella**[1] _[,][∗]_ **Valentino Maiorca**[1] _[,][∗]_ 

**Marco Fumero**[1] **Antonio Norelli**[1] **Francesco Locatello**[2] _[,][†]_ **Emanuele Rodol`a**[1] 

1Sapienza University of Rome 2Amazon Web Services 

## ABSTRACT 

Neural networks embed the geometric structure of a data manifold lying in a high-dimensional space into latent representations. Ideally, the distribution of the data points in the latent space should depend only on the task, the data, the loss, and other architecture-specific constraints. However, factors such as the random weights initialization, training hyperparameters, or other sources of randomness in the training phase may induce incoherent latent spaces that hinder any form of reuse. Nevertheless, we empirically observe that, under the same data and modeling choices, distinct latent spaces typically differ by an unknown quasi-isometric transformation: that is, in each space, the distances between the encodings do not change. In this work, we propose to adopt pairwise similarities as an alternative data representation, that can be used to enforce the desired invariance without any additional training. We show how neural architectures can leverage these relative representations to guarantee, in practice, latent isometry invariance, effectively enabling latent space communication: from zero-shot model stitching to latent space comparison between diverse settings. We extensively validate the generalization capability of our approach on different datasets, spanning various modalities (images, text, graphs), tasks (e.g., classification, reconstruction) and architectures (e.g., CNNs, GCNs, transformers). 

## 1 INTRODUCTION 

Neural Networks (NN) learn to transform high dimensional data into meaningful representations that are helpful for solving downstream tasks. Typically, these representations are seen as elements of a vector space, denoted as latent space, which corresponds to the constrained output (explicitly or implicitly) of a key component of the NN, e.g., the bottleneck in an Autoencoder (AE), or the word embedding space in an NLP task. The underlying assumption is that the learned latent spaces should be the best encoding given the data distribution, the downstream task, and the network constraints. 

In practice, however, the learned latent spaces are subject to changes even when the above assumptions remain fixed. We illustrate this phenomenon in Figure 1, where we show the latent spaces produced by an AE with a two-dimensional bottleneck, trained on the MNIST dataset several times from scratch. These spaces differ from one another, breaking the fundamental assumptions made above. The distribution of the latent embeddings is affected by several factors, such as the random initialization of the network weights, the data shuffling, hyperparameters, and other stochastic processes in the training phase. Although different, the learned representations in Figure 1 are _intrinsically_ similar: the distances between the embedded representations are approximately the same across all spaces, even if their absolute coordinates differ. Indeed, the learned latent spaces are the same up to a nearly isometric transformation.[1] 

This symmetry is a consequence of the implicit biases underlying the optimization process (Soudry et al., 2018) forcing the model to generalize and, therefore, to give similar representations to similar samples with respect to the task. There exist infinitely many spatial arrangements complying with these similarity constraints, each associated with a different isometry. But while the resulting mod- 

> _∗_ Equal contribution. _†_ Work done outside of Amazon. 

> 1To the best of our knowledge, the first to acknowledge this behavior was Olah (2015) in a blogpost. 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication_images/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Train 1 Train 2 Train 3 Train 4<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication_images/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication.pdf-0002-01.png)


Figure 1: Latent spaces learned by distinct trainings of the same AE on the MNIST dataset. The bottleneck has size 2, thus there is no dimensionality reduction involved in the visualization of the latent space. The stochasticity in the training phase induces the same representations up to isometry. As we show in Figure 5, this property holds even for high-dimensional latent spaces. 

els will be equally good in terms of the task, one still encounters several practical problems. For example, it is notoriously challenging to compare latent spaces across different trainings or across different NNs; perhaps more importantly, re-using neural components trained on different embeddings of the same data becomes impossible, since they are incompatible. To overcome this, we propose adopting a local coordinate system defined by the data itself. Each data point becomes a set of coefficients that encode the point as a function of other data samples, instead of an independent point in R _[d]_ . The proposed _relative_ representation directly encodes the intrinsic information underlying the data, and can be made fully invariant to isometries by construction. Remarkably, this enables a form of compositionality between learning models; it allows, for instance, to stitch together an encoder trained on ImageNet with a decoder trained on CIFAR, as we showcase in our experiments. 

Our main contributions can be summarized as follows: 

- We show that the representations learned by NNs are subject to change due to several factors in the training process, and model these changes via latent space isometries. 

- We introduce a novel relative representation for latent embeddings, that is invariant by construction to the isometries induced by stochastic factors in the training process. 

- For the first time, we successfully demonstrate _zero-shot stitching_ of neural components produced by distinct training regimens, e.g., due to different seeds or different neural architectures; we validate our findings on different data modalities (e.g. images, text). 

- Our framework also provides a _quantitative_ measure of performance while training neural models, which is differentiable, does not need any labeled data, and is correlated with standard performance measures such as accuracy. 

## 2 RELATED WORK 

**Representation similarity.** Recently, there has been growing agreement that good networks learn similar representations across a variety of architectures, tasks and domains (Morcos et al., 2018; Li et al., 2016; Kornblith et al., 2019; Bonheme & Grzes, 2022; Tsitsulin et al., 2020; Barannikov et al., 2022; Vuli´c et al., 2020; Lample et al., 2018; Lenc & Vedaldi, 2015), although this is still debated (Wang et al., 2018) and missing strong theoretical justifications. Supported by the empirical evidence widely reported in these works, our method assumes that well-performing neural networks trained on similar tasks and data produce similar latent spaces, which allows us to define a representation that unifies all these spaces. 

**Model stitching.** Lenc & Vedaldi (2015) introduced _trainable_ stitching layers that allow swapping parts of different networks, while Bansal et al. (2021); Csisz´arik et al. (2021) employed stitching to quantitatively verify statements such as “good networks learn similar representations” and “more data, width or time is better”. Other works, such as Gygli et al. (2021), tried to directly produce compatible and reusable network components without stitching layers; more in general, stitching has been adopted in the literature to analyze neural networks. In our work, we sidestep the need for trainable stitching layers and propose _zero-shot_ model stitching to effectively reuse models. 

2 

**Relative information.** The attention mechanism (Vaswani et al., 2017) and its variants (Kossen et al., 2021) exploit the relationship between features to extract meaningful representations. Prototypical Networks (Snell et al., 2017) learn a metric space where the classification can be performed by measuring the distances to prototype representations. Shalam & Korman (2022) proposed the Self Optimal Transport feature transform to enrich the sample representations with higher order relations between the instance features, while Alvarez-Melis et al. (2019) proposed a general formulation of the optimal transport that accounts for global invariances in the underlying feature spaces. Mathematically, our method bears resemblance to a kernel method (Hofmann et al., 2008) as it employs inner products of embedded features as a core ingredient. However, differently from kernel methods, we do not introduce learnable parameters and, crucially, we compute the representations explicitly without resorting to a kernel trick. 

## 3 METHOD 

Given a training set X, standard NNs learn an embedding function _Eθ_ : X _→_ R _[d]_ , parametrized by _θ_ , which maps each sample _**x**_[(] _[i]_[)] _∈_ X to its latent representation, or **absolute representation** , _**ex**_ ( _i_ ) = _Eθ_ ( _**x**_[(] _[i]_[)] ). This representation is then exploited to solve downstream tasks, such as classification, reconstruction or generation, optimizing over some objective function of the general form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication_images/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication.pdf-0003-03.png)


Here, EX denotes the expectation over the training distribution, and _Reg_ ( _θ_ ) encodes additional constraints on the weights _θ_ . As previously discussed, we argue that the learned weights _θ[∗]_ are not only a function of X and of the specific loss appearing in Equation 1, but in practice they are also affected by the optimization process used to train the network due to weight initialization, data shuffling, hyperparameters, and other stochastic factors. We denote these factors collectively by _φ_ . In particular, as shown in Figure 1, changing these factors induces a transformation _T_ over the latent space, i.e., _φ → φ[′]_ implies _Eθ_ ( _**x**_[(] _[i]_[)] ) _→ TEθ_ ( _**x**_[(] _[i]_[)] ). We make the core assumption that _T_ is an _isometric_ transformation, preserving the distances between elements of the latent space, namely _d_ ( _**ex**_ ( _i_ ) _,_ _**ex**_ ( _j_ )) = _d_ ( _T_ _**ex**_ ( _i_ ) _, T_ _**ex**_ ( _j_ )) for every ( _**x**_[(] _[i]_[)] _,_ _**x**_[(] _[j]_[)] ) _∈_ X. While the isometry assumption might seem too restrictive, in practice it arises in several real scenarios as we show in the sequel. 

## 3.1 RELATIVE REPRESENTATIONS 

To build our representation, we start by selecting a subset A of the training data X, which we denote as **anchor** samples. Every sample in the training distribution will be represented with respect to the embedded anchors _**ea**_ ( _j_ ) = _E_ ( _**a**_[(] _[j]_[)] ) with _**a**_[(] _[j]_[)] _∈_ A. As a measure capturing the relation between the anchors and the other samples, we consider a generic similarity function _sim_ : R _[d] ×_ R _[d] →_ R, yielding a scalar score _r_ between two absolute representations _r_ = _sim_ ( _**ex**_ ( _i_ ) _,_ _**ex**_ ( _j_ )). Given the anchors A in an arbitrary ordering _a_[(1)] _, . . . , a_[(] _[|]_[A] _[|]_[)] , we define the **relative representation** of _**x**_[(] _[i]_[)] _∈_ X as: 

_**rx**_ ( _i_ ) = ( _sim_ ( _**ex**_ ( _i_ ) _,_ _**ea**_ (1)) _, sim_ ( _**ex**_ ( _i_ ) _,_ _**ea**_ (2)) _, . . . , sim_ ( _**ex**_ ( _i_ ) _,_ _**ea**_ ( _|_ A _|_ ))) _._ (2) Figure 2 illustrates the key differences between absolute and relative representations. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication_images/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication.pdf-0003-08.png)


**----- Start of picture text -----**<br>
⇒ ⇒<br>**----- End of picture text -----**<br>


Figure 2: _Left_ : Three anchors (colored stars) are selected on the data manifold; given a point on the manifold (blue dot), we compute its similarity w.r.t. the three anchors, yielding a vector of dimensionality 3 (middle). _Right_ : Each dimension is treated as coefficients in a coordinate system defined by the anchors. Anchors are orthogonal in this example only for visualization purposes. 

**Choice of the anchors.** Anchors directly affect the expressivity of the relative representation space, and are related to the task at hand. For example, in a classification task, we should sample anchors from each class in the training set, in order to well represent each data sample in X. 

3 

One case of interest arises when the data comes from different domains X, Y, and we are given a partial correspondence Γ : _P_ X _�→ P_ Y mapping from a subset of X to a subset of Y. In this case, we can sample anchors AX _⊆ P_ X and obtain corresponding anchors on the other domain directly as Γ(A). We refer to these as _parallel anchors_ . We show an example of parallel anchors in Section 5.2, where X and Y represent Amazon reviews written in two different languages. 

The choice of the anchors is not restricted to elements in the training distribution. Given an encoder pre-trained on a fixed training distribution, we can pick elements from a set A[˜] that is out-of-domain w.r.t. X, and build the relative representations on top of A[˜] . We refer to these as _OOD anchors_ and exploit them, e.g., to solve domain adaptation tasks where we do not have access to a correspondence, and have scarce data labels. We refer again to the Sections 5.2 and 5.3 for real-world examples. 

**Isometry invariance.** In this work, we choose the cosine similarity as the similarity function due to the properties it induces on the relative representation. The cosine similarity _SC_ is the dot product of unit vectors, corresponding to the cosine of the angle _θ_ between the two: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication_images/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication.pdf-0004-03.png)


Importantly, cos _θ_ does not change if we apply the same orthogonal transformation _**T**_ to two vectors _**a**_ and _**b**_ , i.e., the cosine similarity is invariant to rotations, reflections, and rescaling. While this is not true for translations, NNs commonly employ normalization techniques (e.g., InstanceNorm (Ulyanov et al., 2016)) to center the latent spaces around zero. Under this assumption, cosine similarity guarantees a relative representation _**rx**_ ( _i_ ) invariant to isometric transformations. 

This means we have the freedom to change the embedding function _Eθ_ with any other function _E_[˜] that produces different but isometric representations, i.e.: 

[ _SC_ ( _**ex**_ ( _i_ ) _,_ _**ea**_ (1)) _, . . . , SC_ ( _**ex**_ ( _i_ ) _,_ _**ea**_ ( _|_ A _|_ )] = [ _SC_ (˜ _**ex**_ ( _i_ ) _,_ ˜ _**ea**_ (1)) _, . . . , SC_ (˜ _**ex**_ ( _i_ ) _,_ ˜ _**ea**_ ( _|_ A _|_ ))] _,_ (4) ˜ where _**ex**_ ( _i_ ) = _E_[˜] ( _**x**_[(] _[i]_[)] ) = _**T** E_ ( _**x**_[(] _[i]_[)] ) and _**T**_ is an arbitrary isometric transformation. A practical application of this invariance is the possibility of comparing latent spaces across multiple trainings, and re-using models as demonstrated in Sections 4 and 5. 

**Quasi-isometry invariance.** We remark that other choices of similarity function can be made to enforce different invariances into the representation. For example, to relax the isometry assumption, one may impose invariance to non-isometric deformations with bounded distortion. We did not find this to be a necessity in our experiments, as typically NNs that generalize sufficiently well can handle small perturbations of the input. Nevertheless, this invariance can be enforced by design by using vector quantization algorithms. We preliminarily explore this approach in Figure 7, leaving further exploration to future work. 

## 4 LATENT SPACE COMMUNICATION 

In this section, we demonstrate how our relative representations can effectively be used to produce latent spaces that are stable under a variety of factors. Our main question is the following: Given two different learning models that are trained separately on different data, can we compare their latent embeddings? In asking this, we assume that the two models are trained on a similar phenomenon, e.g., on two different samplings of the English language or on two different modalities. 

We answer in the positive, showing the gained invariance enables effective communication between different, but semantically equivalent latent spaces. In particular, we analyze how different word embedding spaces, once projected onto relative representations, are intrinsically the same (Section 4.1); we then show how the similarity between the relative counterparts of two or more embedding spaces is a surprisingly good predictor of model performance (Section 4.2); finally, we confirm that relative representations in the training phase are not detrimental to performance (Section 4.3). 

## 4.1 WORD EMBEDDINGS 

**Experimental setting.** We select two different word embeddings on the English language, namely FastText (Bojanowski et al., 2017) and Word2Vec (Mikolov et al., 2013). Both models are pretrained on different data, but partly share a vocabulary from which we extract _≈_ 20K words. For each 

4 

embedding space, we convert their absolute embeddings to corresponding relative representations, using 300 randomly drawn parallel anchors. In Table 1 (left) we show the original embeddings and the resulting relative ones. We measure the degree of similarity in two ways: (i) _Jaccard_ : for each word vector in the source space, we compute the Jaccard similarity between its top-k source neighbors and its top-k target neighbors; (ii) _Mean Reciprocal Rank_ : for each embedded word in the source space, we compute its (reciprocal) ranking among the top-k neighbors in the target. 

|Absolute<br>Relative|FastText|Word2Vec||Source<br>Target<br>Jaccard↑<br>MRR↑|
|---|---|---|---|---|
||||||
|||||Absolute<br>FastText<br>FastText<br>1_._00<br>1_._00<br>FastText<br>Word2Vec<br>0_._00<br>0_._00<br>Word2Vec<br>FastText<br>0_._00<br>0_._00<br>Word2Vec<br>Word2Vec<br>1_._00<br>1_._00<br>Relative<br>FastText<br>FastText<br>1_._00<br>1_._00<br>FastText<br>Word2Vec<br>**0**_._**34**<br>**0**_._**94**<br>Word2Vec<br>FastText<br>**0**_._**39**<br>**0**_._**98**<br>Word2Vec<br>Word2Vec<br>1_._00<br>1_._00|
||||||
||||||



Table 1: Qualitative _(left)_ and quantitative _(right)_ comparison of English word embeddings using absolute and relative representations. All metrics are calculated with _K_ = 10. 

**Result analysis.** Table 1 _(left)_ highlights clusters of semantically similar words and shows that the absolute representations are incoherent across the two latent spaces, while the relative embeddings are highly similar. The average Jaccard distance reported in Table 1 _(right)_ , says that the neighborhoods of the relative representations are matched exactly 34% of the time in one direction, and 39% of the time in the other one (the missing 61% is due to semantic differences, that are not taken into account by the discrete nature of the Jaccard metric). By contrast, the absolute embeddings are never matched exactly (Jaccard score equal to zero); for a match to happen, it would mean that the FastText and Word2Vec embeddings of a given English word are almost the same, which is highly unlikely. The performance gap is even more evident in terms of MRR, which is close to a perfect score for the relative representations. 

Overall, these results show that relative representations are preserved across different word embedding models, validating our isometry assumptions. 

## 4.2 LATENT DISTANCE AS A PERFORMANCE PROXY 

**Experimental setting.** In this experiment, we consider a node classification task on the Cora graph dataset (Sen et al., 2008). We first train a _reference_ model that achieves good accuracy on a validation set. Then, we train _≈_ 2000 models with various combinations of seed, number of epochs, number of layers, dropout probability, activation functions, optimizer type, learning rate or type of graph embedder (Table 8). All the models are classically trained using absolute representations, which are converted to relative post-training by projecting the embeddings onto 300 randomly drawn but fixed anchors. For each model, we measure its classification accuracy and compute the similarity of its space with the reference one. This similarity is computed as the average cosine similarity between the node embeddings produced by a given model and the corresponding embeddings in the reference. 

**Result analysis.** The scatter plot in Figure 3 _(left)_ shows that better-performing models tend to be the ones with the latent spaces most similar to the reference. The performance-similarity correlation also holds over time, as shown in Figure 3 _(right)_ . Additional correlation examples are in Figure 8. Interestingly, this metric is differentiable, enabling an explicit supervision signal on the latent space, which does not require labeled data and could be readily exploited in a teacher-student framework. 

Overall, these results suggest that the similarity between the relative representations of latent spaces is a remarkably good proxy to evaluate model performance. 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication_images/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication.pdf-0006-00.png)


**----- Start of picture text -----**<br>
Similarity Epochs<br>Similarity<br>Performance Performance<br>**----- End of picture text -----**<br>


Figure 3: Graph node classification task on Cora. _Left:_ Correlation between the performance of _≈_ 2000 models and the similarity of their latent spaces with respect to a well-performing reference model. _Right:_ The same correlation plotted over time. The mean Pearson correlation over all models is 0 _._ 955, after filtering out the models having best validation accuracy below 0 _._ 5. 

## 4.3 TRAINING WITH ABSOLUTE VS. RELATIVE REPRESENTATIONS 

**Experimental setting.** Finally, we compare architectures that do or do not employ the relative representation while training. In these experiments, the models vary slightly according to the dataset; however, the relative and absolute versions are always comparable in terms of architecture, number of learnable parameters and hyperparameters. We refer to the supplementary material and the open-source code for further details on their implementation. In this section we consider classification tasks on several datasets, spanning the image domain (Lecun et al., 1998; Xiao et al., 2017; Krizhevsky, 2009) and the graph domain (Yang et al., 2016). 

Table 2: Performance comparison between relative and absolute representations on several image and graph datasets. The metric is the classification weighted F1 score ( _±_ std), over 6 seeds. 

||**Image Classifcation**<br>**MNIST**<br>**F-MNIST**<br>**CIFAR-10**<br>**CIFAR-100**|**Graph Node Classifcation**<br>**Cora**<br>**CiteSeer**<br>**PubMed**|
|---|---|---|
|**Relative**<br>**Absolute**|97_._91_±_0_._07<br>90_._19_±_0_._27<br>87_._70_±_0_._09<br>66_._72_±_0_._35<br>97_._95_±_0_._10<br>90_._32_±_0_._21<br>87_._85_±_0_._06<br>68_._88_±_0_._14|0_._89_±_0_._02<br>0_._77_±_0_._03<br>0_._91_±_0_._01<br>0_._90_±_0_._01<br>0_._78_±_0_._03<br>0_._91_±_0_._01|



**Result analysis.** The results, reported in Table 2, show that relative representations, when used at training time, are not detrimental to performance in general. This is further shown in Tables 3 to 6 and 13 to 16, where a subset of the results compares the absolute and relative representations on a variety of domains, datasets and tasks. 

Overall, these results show that relative representations are effective when involved in end-to-end training, without significant performance drops. 

## 5 ZERO-SHOT MODEL STITCHING 

In this section, we illustrate how the latent space communication demonstrated in Section 4 enables zero-shot interoperability of pre-trained neural components. In previous works, such as Lenc & Vedaldi (2015); Bansal et al. (2021), stitching layers are _trainable_ linear projections that allow swapping parts of different networks. Instead, relative representations unlock the possibility of _zeroshot stitching_ different neural components, treating them as frozen black-box modules. 

We define a generic _stitched model_ as the composition of an encoder, that embeds data, plus a decoder specialized in a downstream task (classification, reconstruction). The stitching operation is always performed without training or fine-tuning, in a zero-shot fashion. Hereafter, we showcase stitching capabilities across combinations of different stochasticity sources (Figure 4 and table 3), neural architectures (Tables 4 and 5) or datasets (Table 6). Finally, we present strong real-world applications in NLP (Section 5.2) and CV (Section 5.3), e.g. zero-shot predictions on novel languages. Additional implementation details are given in the supplementary materials. 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication_images/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication.pdf-0007-00.png)


**----- Start of picture text -----**<br>
S<br>Abs.<br>AE<br>Rel.<br>Abs.<br>VAE<br>Rel.<br>**----- End of picture text -----**<br>


Figure 4: Reconstruction examples. Each column is a different image, row pairs are different architectures. In each pair, we first report the non-stitched reconstructions, then the stitched ones. 

## 5.1 IMAGE RECONSTRUCTION 

**Experimental setting.** We perform zero-shot stitching with AEs and VAEs trained end-to-end on several image datasets. For each combination of model and dataset, we perform five trainings, identical but with different seeds. Thus, we zero-shot stitch together the resulting encoders and decoders. The stitching operation simply consists of composing an encoder with a decoder that is trained on a different seed. 

**Result analysis.** In Figure 4 the stitched models that employ absolute representations ( _Abs._ ) produce erroneous predictions, since the latent spaces obtained from distinct trainings are incompatible. Interestingly, although the absolute VAE does not produce compatible latent spaces, it is regularized, thus all embeddings produced by the encoders correspond to wrong but semantically meaningful reconstructions. Relative representations ( _Rel._ ) exhibit almost indistinguishable reconstructions between the models trained end-to-end and the stitched ones. Quantitative results are in Table 3. 

Overall, these results prove that relative representations are invariant to training stochasticity. 

Table 3: Stitching performance. The MSE ( _±_ std) between the ground truth X and the reconstructions is computed over 5 different seeds. Stitching with our relative representations yields an error up to two orders of magnitude less than the absolute counterpart. 

||||**MNIST**|**F-MNIST**|**CIFAR-10**|**CIFAR-100**|**MSE ↓**|
|---|---|---|---|---|---|---|---|
|**AE**|_Abs._<br>_Rel._|Non-Stitch.<br>Stitch.<br>Non-Stitch.<br>Stitch.|0_._66_±_0_._02<br>97_._79_±_2_._48<br>1_._18_±_0_._02<br>2_._83_±_0_._20|1_._57_±_0_._03<br>120_._54_±_6_._81<br>3_._59_±_0_._04<br>6_._37_±_0_._29|1_._94_±_0_._08<br>86_._74_±_4_._37<br>2_._83_±_0_._13<br>5_._39_±_1_._18|2_._13_±_0_._08<br>97_._17_±_3_._50<br>3_._50_±_0_._08<br>18_._03_±_12_._46|1_._58_±_0_._05<br>100_._56_±_4_._29<br>2_._78_±_0_._07<br>8_._16_±_3_._53|
|**VAE**|_Abs._<br>_Rel._|Non-Stitch.<br>Stitch.<br>Non-Stitch.<br>Stitch.|1_._31_±_0_._04<br>98_._51_±_1_._49<br>2_._97_±_0_._14<br>13_._43_±_6_._79|4_._38_±_0_._03<br>118_._96_±_2_._96<br>6_._81_±_0_._06<br>24_._03_±_13_._15|2_._68_±_0_._06<br>69_._02_±_1_._54<br>5_._18_±_0_._22<br>11_._20_±_3_._15|3_._00_±_0_._03<br>78_._57_±_1_._88<br>5_._93_±_0_._14<br>11_._23_±_2_._38|2_._84_±_0_._04<br>91_._27_±_1_._97<br>5_._22_±_0_._14<br>14_._97_±_6_._37|



## 5.2 TEXT CLASSIFICATION 

In this Section, we show practical examples of the use of parallel anchors (Sec 3.1). 

**Experimental setting.** We consider two different text classification settings. 

_Cross-lingual_ : given a review predict the associated star rating, done on multi-lingual data from the Amazon Reviews dataset (Keung et al., 2020). Following the original paper, we work on 

7 

Table 4: Cross-lingual stitching performance comparison. The table reports the mean weighted F1 ( _±_ std) and MAE on Amazon Reviews coarse-grained, across 5 seeds. 

|**Decoder**<br>**Encoder**|Absolute<br>FScore<br>MAE|Relat|ive|
|---|---|---|---|
|||Translated<br>FScore<br>MAE|Wikipedia|
||||FScore<br>MAE|
|en<br>en<br>es<br>fr<br>ja|91_._54_±_0_._58<br>0_._08_±_0_._01<br>43_._67_±_1_._09<br>0_._56_±_0_._01<br>54_._41_±_1_._61<br>0_._45_±_0_._02<br>48_._72_±_0_._90<br>0_._51_±_0_._01|90_._06_±_0_._60<br>0_._10_±_0_._01<br>82_._78_±_0_._81<br>0_._17_±_0_._01<br>78_._49_±_0_._66<br>0_._21_±_0_._01<br>65_._72_±_0_._55<br>0_._34_±_0_._01|90_._45_±_0_._52<br>0_._10_±_0_._01<br>78_._53_±_0_._30<br>0_._21_±_0_._00<br>70_._41_±_0_._57<br>0_._29_±_0_._01<br>66_._31_±_0_._80<br>0_._34_±_0_._01|



Table 5: Cross-architecture stitching performance comparison. The table reports the mean weighted F1 ( _±_ std) for each dataset, across 5 different seeds. 

|TREC<br>DBpedia|Amazon Reviews|
|---|---|
||_Coarse_<br>_Fine_|
|Abs.<br>Non-Stitch<br>91_._70_±_1_._39<br>98_._62_±_0_._58<br>Stitch<br>21_._49_±_3_._64<br>6_._96_±_1_._46|87_._81_±_1_._58<br>55_._35_±_3_._19<br>49_._58_±_2_._95<br>19_._01_±_2_._04|
|Rel.<br>Non-Stitch<br>88_._08_±_1_._37<br>97_._42_±_2_._05<br>Stitch<br>75_._89_±_5_._38<br>80_._47_±_21_._14|85_._08_±_1_._93<br>48_._92_±_3_._57<br>72_._37_±_7_._32<br>33_._24_±_7_._21|



a binarized version of the task, with FScore and MAE as metrics. In the supplementary material, we report results on the fine-grained formulation. We adopt four different pre-trained languagespecific RoBERTa transformers (Liu et al., 2019) and evaluate their zero-shot stitching performance on languages never seen by the classifier. We use parallel anchors in two modalities: i) _Translated_ : consider English reviews translated[2] into the other languages; ii) _Wikipedia_ : adopt an external corpus, WikiMatrix (Schwenk et al., 2021), providing parallel sentences extracted from Wikipedia. 

_Cross-architecture_ : assessed on three different datasets: TREC (coarse) (Hovy et al., 2001), DBpedia (Zhang et al., 2015), Amazon Reviews (English split). We adopt two different pretrained BERT (Devlin et al., 2019) transformers (cased and uncased version), ELECTRA (Clark et al., 2020) and RoBERTa. 

**Result analysis.** With these results, we show for the first time that it is possible to learn to solve a downstream task on a specific language or transformer, and perform predictions on another. 

Tables 4 and 5 show how the relative representations allow reusing a trained classification head across different languages or encoder architectures. Models stitched with absolute representations show performance that is comparable to random guessing across the board, proving that relative representations are a key element for the success of this kind of zero-shot stitching. Indeed, one can only stitch together models with absolute representations if they happen to have the same latent dimensionality. Moreover, the anchor selection results in Table 4 highlight the robustness that relative representations have on the choice of anchors, even when they are noisy ( _Translated_ case), or their distribution differs from the one of the downstream task ( _Wikipedia_ case), as long as their encoding can be handled correctly by the encoder. In our case, the encoder is pre-trained to represent a variety of texts in a specific language, thus, even if WikiMatrix has a completely different domain from Amazon Reviews, the transformer still computes a meaningful and comparable representation with those of the reviews. We report in Tables 13 and 14 complete results on all languages combination, and in Table 15 the performance obtained by a multi-lingual transformer. To the best of our knowledge, it is the only alternative to obtaining compatible representations across languages. 

According to these results, relative representations show invariance to different architectures and data distribution shifts (e.g., different train languages). 

> 2We used the =GOOGLETRANSLATE function available in Google Sheets. 

8 

## 5.3 IMAGE CLASSIFICATION 

In this Section, we show practical examples of the use of OOD anchors (Sec 3.1). 

Table 6: Stitching performance comparison with different encoding techniques. The table reports the mean weighted F1 ( _±_ std) on CIFAR-100 coarse-grained and ImageNet1k, across 5 seeds. 

|**Decoder**<br>**Encoder**|CIFAR-100<br>Absolute<br>Relative|ImageNet1k<br>Absolute<br>Relative|
|---|---|---|
|rexnet-100<br>rexnet-100<br>vit-base-patch16-224<br>vit-base-resnet50-384<br>vit-small-patch16-224|82_._06_±_0_._15<br>80_._22_±_0_._28<br>-<br>54_._98_±_0_._44<br>-<br>53_._33_±_0_._37<br>-<br>59_._82_±_0_._32|73_._78_±_0_._29<br>72_._61_±_0_._16<br>-<br>37_._39_±_0_._36<br>-<br>42_._36_±_0_._36<br>-<br>43_._75_±_0_._27|
|vit-base-patch16-224<br>rexnet-100<br>vit-base-patch16-224<br>vit-base-resnet50-384<br>vit-small-patch16-224|-<br>76_._81_±_0_._49<br>93_._15_±_0_._05<br>91_._94_±_0_._10<br>6_._21_±_0_._33<br>81_._42_±_0_._38<br>-<br>84_._29_±_0_._86|-<br>30_._78_±_0_._81<br>80_._91_±_0_._29<br>78_._86_±_0_._33<br>0_._07_±_0_._05<br>44_._72_±_0_._57<br>-<br>48_._31_±_0_._72|
|vit-base-resnet50-384<br>rexnet-100<br>vit-base-patch16-224<br>vit-base-resnet50-384<br>vit-small-patch16-224|-<br>79_._79_±_0_._43<br>4_._69_±_0_._07<br>84_._46_±_0_._19<br>91_._41_±_0_._09<br>90_._77_±_0_._16<br>-<br>84_._66_±_0_._16|-<br>53_._46_±_0_._68<br>0_._08_±_0_._04<br>62_._21_±_0_._54<br>82_._55_±_0_._30<br>81_._88_±_0_._16<br>-<br>61_._32_±_0_._36|
|vit-small-patch16-224<br>rexnet-100<br>vit-base-patch16-224<br>vit-base-resnet50-384<br>vit-small-patch16-224|-<br>75_._35_±_0_._41<br>-<br>81_._23_±_0_._31<br>-<br>78_._35_±_0_._69<br>90_._07_±_0_._19<br>88_._85_±_0_._44|-<br>37_._58_±_0_._44<br>-<br>50_._08_±_0_._63<br>-<br>45_._45_±_1_._41<br>77_._73_±_0_._41<br>76_._36_±_0_._40|



**Experimental setting.** We consider a classification task on ImageNet1k and CIFAR-100 with coarse labels (20), and 4 different pre-trained image encoders: three variants of the ViT transformer (Dosovitskiy et al., 2020) and RexNet (Han et al., 2020). 

**Result analysis.** The results in Table 6 highlight how the relative representations allow to stitch modules with different encoding dimensionality, since the encodings are converted to a representation with size equal to the number of anchors. Further, the results demonstrate the ability to generalize and perform zero-shot stitching on CIFAR-100, although that data was never seen by the encoder since it is a frozen transformer trained on ImageNet1k. Interestingly, rexnet-100 is the only transformer whose latent dimensionality is higher than the number of anchors, and the biggest drop in stitching performance happens when the decoder is trained on it. This suggests the number of anchors is an important hyperparameter; we refer to Figure 6 for a deeper analysis. 

Overall, these results prove that relative representations can bridge general-purpose encoders and pre-trained task-specific decoders. 

## 6 CONCLUSION 

In this work, we introduced the concept of relative representations to enable zero-shot latent space communication, with several practical consequences as showcased in our discussion and experiments. Our work proves that a latent semantic correspondence between data domains, when present, can be exploited through a simple shift of representation, without resorting to sophisticated processing or heavy training. 

**Limitations and future work.** Our work is open to several follow-up directions. While in this paper we considered the cosine similarity, different functions can enforce additional invariances in the relative representation. The study of invariant latent spaces as a general direction has the potential to lead to further impact; in Figure 7 we showed preliminary results of this possibility, obtaining representations that are invariant to near-isometries with _guaranteed_ bounded distortion via vector quantization. Another interesting line of research to improve the representation expressivity would be to estimate _geodesic_ distances over the data manifold instead of adopting Euclidean approximations. Similarly, we believe that the connections between the composition of the anchors set A and the expressivity of relative representations demands additional research. For example, the training cost is directly affected by the number and update frequency of the anchors. Finally, the stitching procedure may be extended to multiple layers, promoting reusable network components. 

9 

## REPRODUCIBILITY STATEMENT 

We describe in detail the relative representation computation in Section 3.1. We describe the experimental settings for the various scenarios, and refer to the supplementary material for further implementation details (Appendix A.5). Moreover, we release a well-documented and modular codebase, with the relative representation layer being implemented as a stand-alone PyTorch module. All the checkpoints used in the experiments are versioned with DVC (Kuprieiev et al., 2022) to easily reproduce all the figures and tables. The stand-alone module allows the integration of the relative representations in any existing neural network effortlessly. 

## ACKNOWLEDGMENTS 

This work is supported by the ERC Starting Grant No. 802554 (SPECGEO). 

## REFERENCES 

- David Alvarez-Melis, Stefanie Jegelka, and Tommi S. Jaakkola. Towards optimal transport with global invariances. In Kamalika Chaudhuri and Masashi Sugiyama (eds.), _The 22nd International Conference on Artificial Intelligence and Statistics, AISTATS 2019, 16-18 April 2019, Naha, Okinawa, Japan_ , volume 89 of _Proceedings of Machine Learning Research_ , pp. 1870–1879. PMLR, 2019. URL http://proceedings.mlr.press/v89/alvarez-melis19a.html. 

- Yamini Bansal, Preetum Nakkiran, and Boaz Barak. Revisiting model stitching to compare neural representations. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan (eds.), _Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual_ , pp. 225–236, 2021. URL https://proceedings.neurips.cc/ paper/2021/hash/01ded4259d101feb739b06c399e9cd9c-Abstract.html. 

- Serguei Barannikov, Ilya Trofimov, Nikita Balabin, and Evgeny Burnaev. Representation topology divergence: A method for comparing neural network representations. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), _Proceedings of the 39th International Conference on Machine Learning_ , volume 162 of _Proceedings of Machine Learning Research_ , pp. 1607–1626. PMLR, 17–23 Jul 2022. URL https://proceedings. mlr.press/v162/barannikov22a.html. 

- Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. Enriching word vectors with subword information. _Transactions of the Association for Computational Linguistics_ , 5:135–146, 2017. doi: 10.1162/tacl ~~a 0~~ 0051. URL https://aclanthology.org/Q17-1010. 

- Lisa Bonheme and Marek Grzes. How do variational autoencoders learn? insights from representational similarity, 2022. URL https://arxiv.org/abs/2205.08399. 

- Kevin Clark, Minh-Thang Luong, Quoc V. Le, and Christopher D. Manning. ELECTRA: pretraining text encoders as discriminators rather than generators. In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net, 2020. URL https://openreview.net/forum?id=r1xMH1BtvB. 

- Adri´an Csisz´arik, P´eter K˝or¨osi-Szab´o, Akos[´] K. Matszangosz, Gergely Papp, and D´aniel Varga. Similarity and matching of neural network representations, 2021. URL https://arxiv. org/abs/2110.14633. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pp. 4171–4186, Minneapolis, Minnesota, 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://aclanthology.org/N19-1423. 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition 

10 

at scale. _ArXiv preprint_ , abs/2010.11929, 2020. URL https://arxiv.org/abs/2010. 11929. 

- Michael Gygli, Jasper Uijlings, and Vittorio Ferrari. Towards reusable network components by learning compatible representations. _AAAI_ , 35(9):7620–7629, 2021. 

- Dongyoon Han, Sangdoo Yun, Byeongho Heo, and YoungJoon Yoo. Rethinking channel dimensions for efficient model design. _ArXiv preprint_ , abs/2007.00992, 2020. URL https://arxiv. org/abs/2007.00992. 

- Thomas Hofmann, Bernhard Sch¨olkopf, and Alexander J Smola. Kernel methods in machine learning. _The annals of statistics_ , 36(3):1171–1220, 2008. 

- Eduard Hovy, Laurie Gerber, Ulf Hermjakob, Chin-Yew Lin, and Deepak Ravichandran. Toward semantics-based answer pinpointing. In _Proceedings of the First International Conference on Human Language Technology Research_ , 2001. URL https://www.aclweb.org/ anthology/H01-1069. 

- Phillip Keung, Yichao Lu, Gy¨orgy Szarvas, and Noah A. Smith. The multilingual Amazon reviews corpus. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 4563–4568, Online, 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.369. URL https://aclanthology.org/2020. emnlp-main.369. 

- Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey E. Hinton. Similarity of neural network representations revisited. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), _Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9- 15 June 2019, Long Beach, California, USA_ , volume 97 of _Proceedings of Machine Learning Research_ , pp. 3519–3529. PMLR, 2019. URL http://proceedings.mlr.press/v97/ kornblith19a.html. 

- Jannik Kossen, Neil Band, Clare Lyle, Aidan N. Gomez, Tom Rainforth, and Yarin Gal. Selfattention between datapoints: Going beyond individual input-output pairs in deep learning, 2021. URL https://arxiv.org/abs/2106.02584. 

- Alex Krizhevsky. Learning multiple layers of features from tiny images. pp. 32–33, 2009. URL https://www.cs.toronto.edu/˜kriz/learning-features-2009-TR.pdf. 

- Ruslan Kuprieiev, skshetry, Dmitry Petrov, Paweł Redzy´nski, Peter Rowlands, Casper da CostaLuis, Alexander Schepanovski, Ivan Shcheklein, Batuhan Taskaya, Gao, Jorge Orpinel, David de la Iglesia Castro, F´abio Santos, Aman Sharma, Dave Berenbaum, Zhanibek, Dani Hodovic, daniele, Nikita Kodenko, Andrew Grigorev, Earl, Nabanita Dash, George Vyshnya, Ronan Lamy, maykulkarni, Max Hora, Vera, and Sanidhya Mangal. Dvc: Data version control - git for data & models, 2022. URL https://doi.org/10.5281/zenodo.7083378. 

- Guillaume Lample, Alexis Conneau, Marc’Aurelio Ranzato, Ludovic Denoyer, and Herv´e J´egou. Word translation without parallel data. In _6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings_ . OpenReview.net, 2018. URL https://openreview.net/forum?id=H196sainb. 

- Y Lecun, L Bottou, Y Bengio, and P Haffner. Gradient-based learning applied to document recognition. _Proc. IEEE_ , 86(11):2278–2324, 1998. 

- Karel Lenc and Andrea Vedaldi. Understanding image representations by measuring their equivariance and equivalence. In _IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015_ , pp. 991–999. IEEE Computer Society, 2015. doi: 10. 1109/CVPR.2015.7298701. URL https://doi.org/10.1109/CVPR.2015.7298701. 

- Yixuan Li, Jason Yosinski, Jeff Clune, Hod Lipson, and John E. Hopcroft. Convergent learning: Do different neural networks learn the same representations? In Yoshua Bengio and Yann LeCun (eds.), _4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings_ , 2016. URL http://arxiv.org/abs/ 1511.07543. 

11 

- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. _ArXiv preprint_ , abs/1907.11692, 2019. URL https://arxiv.org/abs/1907. 11692. 

- Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. _ArXiv preprint_ , abs/1301.3781, 2013. URL https://arxiv.org/ abs/1301.3781. 

- Ari S. Morcos, Maithra Raghu, and Samy Bengio. Insights on representational similarity in neural networks with canonical correlation. In Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen Grauman, Nicol`o Cesa-Bianchi, and Roman Garnett (eds.), _Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montr´eal, Canada_ , pp. 5732–5741, 2018. URL https://proceedings.neurips.cc/paper/2018/hash/ a7a3d70c6d17a73140918996d03c014f-Abstract.html. 

- Christopher Olah. Visualizing representations: Deep learning and human beings. http: //colah.github.io/posts/2015-01-Visualizing-Representations/, 2015. Accessed: 2022-8-2. 

- Holger Schwenk, Vishrav Chaudhary, Shuo Sun, Hongyu Gong, and Francisco Guzm´an. WikiMatrix: Mining 135M parallel sentences in 1620 language pairs from Wikipedia. In _Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume_ , pp. 1351–1361, Online, 2021. Association for Computational Linguistics. URL https://aclanthology.org/2021.eacl-main.115. 

- Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina EliassiRad. Collective Classification in Network Data. _AIMag._ , 29(3):93, 2008. ISSN 2371-9621. doi: 10.1609/aimag.v29i3.2157. 

- Daniel Shalam and Simon Korman. The self-optimal-transport feature transform. _ArXiv preprint_ , abs/2204.03065, 2022. URL https://arxiv.org/abs/2204.03065. 

- Jake Snell, Kevin Swersky, and Richard S. Zemel. Prototypical networks for few-shot learning. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (eds.), _Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA_ , pp. 4077–4087, 2017. URL https://proceedings.neurips.cc/ paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html. 

- Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, and Nathan Srebro. The implicit bias of gradient descent on separable data. In _6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings_ . OpenReview.net, 2018. URL https://openreview.net/forum?id=r1q7n9gAb. 

- Anton Tsitsulin, Marina Munkhoeva, Davide Mottin, Panagiotis Karras, Alexander M. Bronstein, Ivan V. Oseledets, and Emmanuel M¨uller. The shape of data: Intrinsic distance for data distributions. In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net, 2020. URL https://openreview.net/ forum?id=HyebplHYwB. 

- Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Instance normalization: The missing ingredient for fast stylization. _ArXiv preprint_ , abs/1607.08022, 2016. URL https://arxiv. org/abs/1607.08022. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (eds.), _Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA_ , pp. 5998–6008, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/ 3f5ee243547dee91fbd053c1c4a845aa-Abstract.html. 

12 

- Ivan Vuli´c, Sebastian Ruder, and Anders Søgaard. Are all good word vector spaces isomorphic? In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 3178–3192, Online, 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.257. URL https://aclanthology.org/2020. emnlp-main.257. 

- Liwei Wang, Lunjia Hu, Jiayuan Gu, Zhiqiang Hu, Yue Wu, Kun He, and John E. Hopcroft. Towards understanding learning representations: To what extent do different neural networks learn the same representation. In Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen Grauman, Nicol`o Cesa-Bianchi, and Roman Garnett (eds.), _Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montr´eal, Canada_ , pp. 9607–9616, 2018. URL https://proceedings.neurips.cc/paper/2018/hash/ 5fc34ed307aac159a30d81181c99847e-Abstract.html. 

- Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, 2017. URL http://arxiv.org/ abs/1708.07747. cite arxiv:1708.07747Comment: Dataset is freely available at https://github.com/zalandoresearch/fashion-mnist Benchmark is available at http://fashionmnist.s3-website.eu-central-1.amazonaws.com/. 

- Zhilin Yang, William W. Cohen, and Ruslan Salakhutdinov. Revisiting semi-supervised learning with graph embeddings. In Maria-Florina Balcan and Kilian Q. Weinberger (eds.), _Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016_ , volume 48 of _JMLR Workshop and Conference Proceedings_ , pp. 40–48. JMLR.org, 2016. URL http://proceedings.mlr.press/v48/yanga16.html. 

- Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett (eds.), _Advances in Neural Information Processing Systems_ , volume 28. Curran Associates, Inc., 2015. URL https://proceedings.neurips.cc/paper/2015/file/ 250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf. 

## A APPENDIX 

## A.1 LATENT SPACE ISOMETRIES 

In Figure 1, multiple trainings of the same two-dimensional AE produce isometric latent spaces; in Figure 5 we show this property also holds on AEs with a high-dimensional bottleneck. In the first row, PCA is fitted indipendently in each column, and since the PCA transformation produces the same output everywhere the latent spaces are isometric. In the second row, PCA is fitted only on the first latent space; since in this case the PCA transformations produces different outputs, the latent spaces although isometric are extrinsically different. 

## A.2 ANCHORS ANALYSIS 

The cardinality of the anchors set A and the choice of specific anchors is crucial to the quality of the relative representations. At the extreme, selecting one single anchor or the same repeated data points for all anchors, will produce collapsed relative representations. We believe that additional research is required to obtain a better understanding on the optimal choice for A. Questions like “Are anchors set composed only by stopwords worse than the ones composed by meaningful and diverse words?” require empirical evidence and could help revealing the semantics of the latent space. Indeed, each anchor is associated with a dimension in a relative representation; one could inspect the anchor data point to get a sense of the meaning of that latent dimension. 

Below, we report a preliminary study on the performance sensitivity against the cardinality of the anchors set. In Figure 6 we report the performance on the node classification task on Cora, with a model trained end-to-end adopting the relative representations while training, and on image classification tasks on CIFAR-100, with a frozen encoder. The performance improves monotonically 

13 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication_images/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication.pdf-0014-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication_images/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication.pdf-0014-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication_images/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication.pdf-0014-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication_images/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication.pdf-0014-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication_images/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication.pdf-0014-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication_images/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication.pdf-0014-05.png)


Figure 5: Latent spaces learned by distinct trainings of the same high-dimensional AE on the MNIST dataset. Each column is the latent space obtained by the AE with a different seed. On the first row, the dimensionality reduction is performed through PCAs fitted independently on each latent space, meanwhile, on the second row PCA is fitted on the leftmost latent space and then applied to all of them. 

as the number of anchors increase when the absolute representations are frozen _(right)_ . Differently, training models end-to-end proves to be more susceptible to model collapse and instabilities, as increasing the number of anchors does not always improve the performance _(left)_ . Further research on the relation between the absolute latent space dimensionality and the relative representation dimensionality (i.e., the number of anchors) is needed to clarify how the two quantities impact the performance, when training end-to-end or not. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication_images/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication.pdf-0014-08.png)


**----- Start of picture text -----**<br>
Number of anchors Number of anchors<br>Performance<br>**----- End of picture text -----**<br>


Figure 6: Accuracy vs Number of anchors. Each point is a trained model. _Left_ : Trained embedder on Cora, node classification. _Right_ : Frozen transformer on Cifar100 coarse-grained, image classification. Left is less stable because the absolute embeddings are trained, and we are working on a domain that is less stable (graphs). Some collapsed examples are not visualized. 

## A.3 QUASI-ISOMETRY INVARIANCE WITH GUARANTEED BOUNDS 

In this section, we explore a slightly modified version of the similarity function adopted in the main paper. The experimental setting is the same as in Section 4.1. We want to measure the similarity between pairs of absolute embeddings and their relative counterparts. To get some kind of quantitative measure, we add a similarity score calculated as the pairwise cosine distance between the two embedding types, averaged. Therefore, a lower score indicates the spaces are more similar. On top of the standard relative representations, the ones computed with _sim_ = _SC_ , here we try to improve the similarity measure with guaranteed robustness to quasi-isometries up to a desired distortion bound. 

14 

In Figure 7 we report preliminary results that adopt this technique: a vector-quantized similarity function produces relative representations which are more similar (they have a lower score). The vector-quantization is done through agglomerative clustering on the absolute embeddings at various thresholds _t_ . We leave to future works the study of the trade-off between a guaranteed bound on the near-isometries invariance and the expressiveness of the resulting representations. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication_images/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication.pdf-0015-01.png)


**----- Start of picture text -----**<br>
Abs. Rel. Rel. Qnt. t  = 1 Rel. Qnt. t  = 1 . 5 Rel. Qnt. t  = 2<br>Score ↓ 3 . 30 1 . 24 1 . 19 1 . 09 1 . 02<br>FastText<br>Word2Vec<br>**----- End of picture text -----**<br>


Figure 7: The FastText and Word2Vec embeddings of a subset of the English dictionary. The score is the pairwise distance average between the two embedding types, thus a lower score indicates the spaces are more similar. The absolute representations appear very dissimilar meanwhile the relative representations yield almost identical spaces. Quantizing the absolute representations by performing agglomerative clustering with distance threshold _t_ produces even more similar spaces. 

## A.4 DATASET INFORMATION 

In Table 7 we summarize the datasets utilized in our work, and for each one, we specify the number of classes, to give an idea about the classification difficulty. 

Table 7: All the datasets utilized in our work with their number of classes. 

||**Dataset**|**Number of Classes**|
|---|---|---|
||MNIST|10|
|**Image**|Fashion MNIST<br>CIFAR-10<br>CIFAR-100|10<br>10<br>20 (coarse) — 100 (fne)|
||ImageNet1k|1000|
|**Graph**|Cora<br>CiteSeer<br>PubMed|7<br>6<br>3|
|**Text**|TREC<br>DBpedia<br>Amazon Reviews|6 (coarse) — 50 (fne)<br>14<br>2 (coarse) — 5 (fne)|



- A.5 IMPLEMENTATION DETAILS 

In this Section, following the corresponing sections in the main paper, we report implementation details for all the experimental settings considered. 

**Tools & Technologies** In all the experiments presented in this work, the following tools were used: 

- _PyTorch Lightning_ , to ensure reproducible results while also getting a clean and modular codebase; 

15 

- _Weights and Biases_ , to log experiments and compare runs across huge sweeps; 

- _Transformers by HuggingFace_ , to get ready-to-use transformers for both text and images; 

- _Datasets by HuggingFace_ , to access most of the NLP datasets and ImageNet for CV; 

- _DVC_ , for data versioning; 

- _PyTorch Geometric_ , to handle graph datasets and get ready-to-use GNN architectures. 

## A.5.1 WORD EMBEDDINGS 

For both the Figure and the Table in Section 4.1, the number of anchors is set to 300 for a fair comparison with the dimensionality of the original spaces. For visualization purposes, we needed the figure to both show an easy clusterable and restricted set of word embeddings. They are obtained by subsampling the shared vocabulary with the following procedure: we select 4 random pivot words, and for each of them we consider the top-200 words in their neighborhood. This results in a total of 800 points divided in 4 clusters, the ones used only for the visualization part. For the quantitative part (table results), we select 20K random words from the shared vocabulary with a fixed seed for reproducibility purposes. 

## A.5.2 LATENT DISTANCE AS A PERFORMANCE PROXY 

The hyperperameters used in Section 4.2 are summarized in Table 8. 

Table 8: The reference model and exhaustive hyperparameter combinations pertaining Section 4.2. 

|Hyperparameter|Reference Model|Sweep|
|---|---|---|
|Seed|1|0,1,2,3,4|
|Epochs|500|10,30,50|
|Number of layers|32|32,64|
|Dropout Probability|0.5|0.1,0.5|
|Hidden Activations|ReLU|ReLU,Tanh|
|Convolution Activation|ReLU|ReLU,Tanh|
|Optimizer|Adam|Adam,SGD|
|Learning Rate|0.02|0.01,0.02|
|Graph Embedder|GCNConv|GCNConv,GINConv|



## A.5.3 TRAINING WITH ABSOLUTE VS. RELATIVE REPRESENTATIONS 

**Image Classification** The architecture is a standard deep CNN. We run a sweep for each dataset where we vary only the random seed (over 10 possible in total). We then aggregate by dataset and encoding type to obtain the final results with their standard deviation. 

**Graph Classification** We run a sweep identical to the one in Table 8 for the reference model, except that we sweep on the “Number of layers” with two values: 32 and 64. Each configuration is repeated with 10 different seeds, then we aggregate by dataset and encoding type to obtain the final results with their standard deviation. 

## A.5.4 IMAGE RECONSTRUCTION 

The relative and absolute models appearing in Figure 4 are vanilla AEs and VAEs, the same for all the datasets, and have a comparable number of trainable parameters. Their architecture is composed by simple convolutions, deconvolutions and mean squared error as reconstruction loss. The number of anchors is 500 and the latent dimensionality of the absolute representations is 500. 

## A.5.5 TEXT CLASSIFICATION 

We report in Tables 9 to 11 details on the transformers and anchors adopted in Section 5.2. 

16 

Table 9: The HuggingFace transformers employed in Section 5.2 to tackle the _Cross-lingual_ setting. 

|Language|HuggingFace transformers name|Encoding Dim|
|---|---|---|
|English|roberta-base|768|
|Spanish|PlanTL-GOB-ES/roberta-base-bne|768|
|French|ClassCat/roberta-base-french|768|
|Japanese|nlp-waseda/roberta-base-japanese|768|



Table 10: The HuggingFace transformers employed in Section 5.2 to tackle the _Cross-architecture_ setting. 

|HuggingFace transformers name|Encoding Dim|
|---|---|
|bert-base-cased|768|
|bert-base-uncased|768|
|google/electra-base-discriminator|768|
|roberta-base|768|



**Preprocessing** Following the original work in which the Amazon Reviews dataset was proposed (Keung et al., 2020), we utilize both the _title_ and _body_ of each review. We differ in not using the category and in how we merge them; namely, we add the title as prefix for the body and add a full stop as separator when needed (avoiding duplicates). To obtain a single latent encoding for each sample, with fixed shape, we take the last hidden state and select the representation corresponding to the _[CLS]_ token. 

**Wikipedia anchors** We use WikiMatrix, a corpus of sentences extracted from Wikipedia. The sentences are parallel between pairs of languages (i.e., same sentences translated in two languages), and since we are looking for a collection of parallel anchors between all 4 languages, we decided to use the English language as a pivot to compute the intersection. To get the final results, we considered only the sentences with margin score _≥_ 1 _._ 06, getting high-quality sentence alignments. In Table 11 we show the total number of parallel sentences when computing the intersections. We randomly selected 768 samples to use as anchors. 

## A.5.6 IMAGE CLASSIFICATION 

The details of the transformers used in Section 5.3 are summarized in Table 12. 

## A.6 ADDITIONAL RESULTS 

In this section we report additional results on the correlation between latent similarity and performance in Figure 8, results on the multilingual stitching both with Amazon coarse-grained in Table 13 and fine-grained in Table 14, results on the image classification stitching on CIFAR-100 

Table 11: WikiMatrix analysis. Each row shows the number of parallel sentences having a translation available in all the languages of that row. Since we consider all four languages, we have 3338 parallel sentences available. 

|Languages|Number of Sentences|
|---|---|
|en, es|2302527|
|en, ja|264259|
|en, fr|1682477|
|en, es, fr|23200|
|en, es, ja|147665|
|en, fr, ja|20990|
|en, es, fr, ja|**3338**|



17 

Table 12: Timm transformers used in Section 5.3. 

|Version|Timm model name|Encoding Dim|Training data|
|---|---|---|---|
|ViT|vit<br>~~b~~ase<br>~~p~~atch16<br>~~2~~24|768|JFT-300M, ImageNet|
|ViT|vit<br>~~s~~mall<br>~~p~~atch16<br>~~2~~24|384|ImageNet|
|ViT|vit<br>~~b~~ase<br>~~r~~esnet50<br>~~3~~84|768|ImageNet|
|RexNet|rexnet<br>~~1~~00|1280|ImageNet|



fine-grained in Table 16. Moreover, we evaluate the stitching performance of a multilingual transformer in Table 15. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication_images/iclr_2023_Relative_representations_enable_zero-shot_latent_space_communication.pdf-0018-03.png)


Figure 8: Correlation plot between performance and latent similarity with the reference model for multiple different models, over time. 

18 

Table 13: Stitching performance comparison with different encodings techniques. The table reports the mean weighted F1 (± std) and MAE classification performance on Amazon Reviews coarsegrained, across 5 different seeds. All the language pairs are shown. 

|**Decoder**<br>**Encoder**|Absolute<br>FScore<br>MAE|Relat|ive|
|---|---|---|---|
|||Translated<br>FScore<br>MAE|Wikipedia|
||||FScore<br>MAE|
|en<br>en<br>es<br>fr<br>ja<br>es<br>en<br>es<br>fr<br>ja<br>fr<br>en<br>es<br>fr<br>ja<br>ja<br>en<br>es<br>fr<br>ja|91_._54_±_0_._58<br>0_._08_±_0_._01<br>43_._67_±_1_._09<br>0_._56_±_0_._01<br>54_._41_±_1_._61<br>0_._45_±_0_._02<br>48_._72_±_0_._90<br>0_._51_±_0_._01<br>33_._23_±_1_._00<br>0_._66_±_0_._01<br>91_._64_±_1_._02<br>0_._08_±_0_._01<br>47_._66_±_0_._70<br>0_._52_±_0_._01<br>53_._10_±_2_._27<br>0_._46_±_0_._02<br>51_._00_±_2_._63<br>0_._49_±_0_._03<br>51_._96_±_2_._81<br>0_._48_±_0_._03<br>88_._22_±_0_._75<br>0_._12_±_0_._01<br>50_._32_±_4_._16<br>0_._50_±_0_._04<br>53_._82_±_2_._62<br>0_._46_±_0_._03<br>44_._91_±_2_._21<br>0_._55_±_0_._02<br>66_._46_±_1_._30<br>0_._34_±_0_._01<br>83_._30_±_0_._67<br>0_._17_±_0_._01|90_._06_±_0_._60<br>0_._10_±_0_._01<br>82_._78_±_0_._81<br>0_._17_±_0_._01<br>78_._49_±_0_._66<br>0_._21_±_0_._01<br>65_._72_±_0_._55<br>0_._34_±_0_._01<br>78_._68_±_2_._74<br>0_._21_±_0_._03<br>89_._96_±_0_._77<br>0_._10_±_0_._01<br>78_._57_±_1_._80<br>0_._21_±_0_._02<br>67_._69_±_0_._24<br>0_._32_±_0_._00<br>83_._32_±_1_._80<br>0_._17_±_0_._02<br>82_._50_±_0_._83<br>0_._17_±_0_._01<br>85_._68_±_1_._37<br>0_._14_±_0_._01<br>69_._38_±_0_._73<br>0_._31_±_0_._01<br>68_._66_±_3_._62<br>0_._31_±_0_._04<br>70_._37_±_6_._94<br>0_._29_±_0_._06<br>76_._49_±_1_._13<br>0_._23_±_0_._01<br>81_._04_±_0_._82<br>0_._19_±_0_._01|90_._45_±_0_._52<br>0_._10_±_0_._01<br>78_._53_±_0_._30<br>0_._21_±_0_._00<br>70_._41_±_0_._57<br>0_._29_±_0_._01<br>66_._31_±_0_._80<br>0_._34_±_0_._01<br>76_._65_±_3_._23<br>0_._23_±_0_._03<br>89_._62_±_0_._94<br>0_._10_±_0_._01<br>75_._25_±_0_._76<br>0_._25_±_0_._01<br>61_._84_±_0_._61<br>0_._38_±_0_._01<br>75_._55_±_0_._37<br>0_._24_±_0_._00<br>77_._12_±_0_._88<br>0_._23_±_0_._01<br>86_._45_±_0_._96<br>0_._13_±_0_._01<br>62_._79_±_0_._27<br>0_._37_±_0_._00<br>70_._26_±_3_._16<br>0_._29_±_0_._03<br>58_._54_±_1_._21<br>0_._41_±_0_._01<br>63_._94_±_2_._70<br>0_._36_±_0_._02<br>80_._80_±_1_._25<br>0_._19_±_0_._01|



Table 14: Stitching performance comparison with different encodings techniques. The table reports the mean weighted F1 (± std) and MAE classification performance on Amazon Reviews finegrained, across 5 different seeds. All the language pairs are shown. 

|**Decoder**<br>**Encoder**|Absolute<br>FScore<br>MAE|Relat|ive|
|---|---|---|---|
|||Translated<br>FScore<br>MAE|Wikipedia|
||||FScore<br>MAE|
|en<br>en<br>es<br>fr<br>ja<br>es<br>en<br>es<br>fr<br>ja<br>fr<br>en<br>es<br>fr<br>ja<br>ja<br>en<br>es<br>fr<br>ja|65_._46_±_2_._89<br>0_._38_±_0_._02<br>22_._70_±_0_._41<br>1_._39_±_0_._03<br>30_._75_±_0_._67<br>1_._19_±_0_._02<br>24_._85_±_0_._91<br>1_._37_±_0_._07<br>21_._24_±_0_._81<br>1_._43_±_0_._07<br>61_._29_±_3_._04<br>0_._43_±_0_._02<br>29_._02_±_0_._85<br>1_._26_±_0_._05<br>29_._23_±_1_._32<br>1_._22_±_0_._02<br>27_._39_±_1_._22<br>1_._23_±_0_._06<br>29_._47_±_3_._68<br>1_._18_±_0_._07<br>56_._40_±_1_._89<br>0_._51_±_0_._01<br>25_._92_±_1_._31<br>1_._25_±_0_._05<br>29_._36_±_0_._59<br>1_._17_±_0_._04<br>25_._64_±_1_._77<br>1_._28_±_0_._04<br>31_._79_±_1_._91<br>1_._06_±_0_._02<br>54_._09_±_1_._35<br>0_._60_±_0_._02|61_._18_±_1_._92<br>0_._44_±_0_._02<br>51_._67_±_1_._20<br>0_._62_±_0_._01<br>49_._18_±_0_._83<br>0_._69_±_0_._02<br>37_._34_±_1_._49<br>0_._99_±_0_._02<br>51_._02_±_2_._54<br>0_._68_±_0_._05<br>57_._89_±_3_._80<br>0_._48_±_0_._03<br>48_._40_±_1_._02<br>0_._71_±_0_._02<br>37_._22_±_1_._56<br>1_._03_±_0_._04<br>45_._55_±_3_._55<br>0_._76_±_0_._09<br>40_._29_±_1_._72<br>0_._90_±_0_._04<br>53_._58_±_0_._70<br>0_._57_±_0_._01<br>38_._60_±_1_._03<br>0_._96_±_0_._02<br>38_._19_±_2_._28<br>0_._88_±_0_._03<br>34_._23_±_2_._62<br>1_._00_±_0_._05<br>38_._50_±_2_._46<br>0_._89_±_0_._02<br>50_._89_±_1_._70<br>0_._65_±_0_._02|62_._36_±_2_._23<br>0_._43_±_0_._02<br>45_._40_±_0_._68<br>0_._76_±_0_._01<br>40_._29_±_0_._90<br>0_._91_±_0_._02<br>37_._73_±_0_._70<br>1_._01_±_0_._02<br>47_._70_±_5_._08<br>0_._73_±_0_._10<br>57_._96_±_4_._40<br>0_._48_±_0_._03<br>44_._92_±_1_._83<br>0_._77_±_0_._01<br>34_._56_±_0_._87<br>1_._08_±_0_._04<br>39_._01_±_1_._25<br>0_._88_±_0_._06<br>41_._29_±_2_._01<br>0_._83_±_0_._04<br>54_._23_±_0_._95<br>0_._56_±_0_._01<br>35_._22_±_0_._56<br>1_._08_±_0_._02<br>36_._57_±_1_._72<br>0_._98_±_0_._02<br>33_._16_±_2_._28<br>1_._06_±_0_._03<br>36_._68_±_3_._14<br>1_._00_±_0_._05<br>51_._64_±_1_._47<br>0_._65_±_0_._02|



19 

Table 15: Stitching performance comparison on XLM-R, a multilingual model by design. The table reports the mean weighted F1 ( _±_ std) and MAE classification performance on Amazon Reviews fine-grained, across 5 different seeds. 

|||Absolute|Absolute|Relative|Relative|
|---|---|---|---|---|---|
|**Decoder**|**Encoder**|FScore|MAE|FScore|MAE|
||en|65_._27_±_0_._94|0_._41_±_0_._01|58_._24_±_1_._92|0_._51_±_0_._03|
||es|59_._55_±_0_._76|0_._48_±_0_._01|52_._81_±_1_._57|0_._62_±_0_._02|
|en|fr|58_._58_±_1_._04|0_._49_±_0_._01|54_._01_±_1_._34|0_._59_±_0_._02|
||ja|57_._98_±_0_._77|0_._52_±_0_._01|48_._47_±_2_._67|0_._71_±_0_._04|
||en|60_._32_±_1_._50|0_._47_±_0_._01|45_._69_±_2_._19|0_._87_±_0_._07|
||es|61_._25_±_1_._74|0_._44_±_0_._01|57_._61_±_0_._73|0_._51_±_0_._01|
|es|fr|59_._50_±_1_._41|0_._47_±_0_._01|45_._16_±_3_._30|0_._83_±_0_._09|
||ja|58_._24_±_1_._31|0_._51_±_0_._02|41_._14_±_1_._76|0_._99_±_0_._05|
||en|58_._00_±_4_._21|0_._49_±_0_._03|52_._37_±_1_._66|0_._66_±_0_._03|
|fr|es<br>fr|56_._87_±_3_._79<br>57_._99_±_3_._88|0_._49_±_0_._03<br>0_._47_±_0_._02|54_._99_±_0_._46<br>57_._00_±_0_._90|0_._57_±_0_._01<br>0_._52_±_0_._01|
||ja|55_._83_±_3_._32|0_._53_±_0_._03|39_._15_±_1_._21|1_._02_±_0_._03|
||en|59_._53_±_1_._73|0_._48_±_0_._01|39_._46_±_2_._34|1_._04_±_0_._07|
|ja|es<br>fr|57_._02_±_1_._36<br>57_._48_±_1_._06|0_._51_±_0_._00<br>0_._51_±_0_._01|40_._74_±_2_._75<br>43_._36_±_3_._70|0_._97_±_0_._09<br>0_._89_±_0_._11|
||ja|61_._43_±_0_._97|0_._45_±_0_._01|57_._67_±_1_._17|0_._51_±_0_._01|



Table 16: Stitching performance comparison with different encodings techniques. The table reports the mean weighted F1 ( _±_ std) classification performance on CIFAR-100 fine-grained, across 5 different seeds. 

|**Decoder**|**Encoder**|Absolute|Relative|
|---|---|---|---|
||rexnet-100|72_._77_±_0_._19|71_._39_±_0_._18|
|rexnet-100|vit-base-patch16-224<br>vit-base-resnet50-384|-<br>-|40_._68_±_0_._50<br>38_._18_±_0_._24|
||vit-small-patch16-224|-|44_._11_±_0_._84|
||rexnet-100|-|57_._81_±_0_._39|
|vit-base-patch16-224|vit-base-patch16-224<br>vit-base-resnet50-384|88_._69_±_0_._14<br>1_._08_±_0_._19|87_._05_±_0_._34<br>66_._65_±_1_._79|
||vit-small-patch16-224|-|73_._73_±_0_._60|
||rexnet-100|-|66_._91_±_0_._79|
|vit-base-resnet50-384|vit-base-patch16-224<br>vit-base-resnet50-384|1_._10_±_0_._09<br>85_._85_±_0_._18|75_._70_±_0_._68<br>85_._04_±_0_._38|
||vit-small-patch16-224|-|75_._52_±_0_._36|
||rexnet-100|-|56_._60_±_0_._39|
|vit-small-patch16-224|vit-base-patch16-224<br>vit-base-resnet50-384|-<br>-|70_._14_±_0_._46<br>62_._85_±_1_._22|
||vit-small-patch16-224|84_._11_±_0_._14|83_._24_±_0_._13|



20 

