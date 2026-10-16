# **Alignment with human representations supports robust few-shot learning** 

**Ilia Sucholutsky** Department of Computer Science Princeton University `is2961@princeton.edu` 

**Thomas L. Griffiths** Department of Psychology Department of Computer Science Princeton University 

## **Abstract** 

Should we care whether AI systems have representations of the world that are similar to those of humans? We provide an information-theoretic analysis that suggests that there should be a U-shaped relationship between the degree of representational alignment with humans and performance on few-shot learning tasks. We confirm this prediction empirically, finding such a relationship in an analysis of the performance of 491 computer vision models. We also show that highly-aligned models are more robust to both adversarial attacks and domain shifts. Our results suggest that human alignment is often a sufficient, but not necessary, condition for models to make effective use of limited data, be robust, and generalize well. 

## **1 Introduction** 

As AI systems are increasingly deployed in settings that involve interactions with humans, exploring the extent to which these systems are aligned with humans becomes more significant. While this exploration has largely focused on the alignment of the _values_ of AI systems with humans [6, 15], the alignment of their _representations_ is also important. Representing the world in the same way is a precursor to being able to express common values and to comprehensible generalization. To the extent that humans have accurate representations of the world, representational alignment is also an effective source of inductive bias that might make it possible to learn from limited data. 

As a motivating example, imagine a meeting between a 16th century alchemist and a 21st century chemist. They live in the same physical world and are intimately familiar with the materials that comprise it, but they would have significant difficulty expressing their values and generalizing the results of an experiment they observe together. The alchemist would likely learn poorly from examples of a reaction demonstrated by the chemist, not having the right inductive biases for the way the world actually works. The alchemist and the chemist lack representational alignment – they represent the world in fundamentally different ways – and this impedes generalization and learning. 

In this paper, we provide a theoretical and empirical investigation of the consequences of representational alignment with humans for AI systems, using popular computer vision tasks as a way to study this phenomenon. We define representational alignment as the degree to which the latent representations of a model match the latent representations of humans for the same set of stimuli, and refer to models that are representationally aligned with humans as being “human-aligned.” Several recent papers have proposed ways to measure [21, 22], explain [24, 18], and even improve [27, 5] the representational alignment of models. However, many models that score low on these alignment metrics still have high performance on downstream tasks like image classification [18, 24, 5]. 

So, are there any real advantages (or disadvantages) to using human-aligned models? To answer this question, we develop an information-theoretic framework for analyzing representational alignment. This framework enables us to make predictions about emergent behavior in human-aligned models. 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2023_Alignment_with_human_representations_supports_robust_few-shot_learning_images/nips_2023_Alignment_with_human_representations_supports_robust_few-shot_learning.pdf-0002-00.png)


Figure 1: Schematic of representational alignment between two agents, Alice and Bob. **A** : Shared data ( _x_ ) is shown to both agents (images are from the animal subset of the similarity datasets from Peterson et al. [27]). **B** : Both agents form representations ( _fA_ ( _x_ ) and _fB_ ( _x_ )) of the objects they observe. **C** : Agents are asked to produce pairwise similarity matrices corresponding to their representations. The similarity judgments can then be compared to measure alignment between the agents. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2023_Alignment_with_human_representations_supports_robust_few-shot_learning_images/nips_2023_Alignment_with_human_representations_supports_robust_few-shot_learning.pdf-0002-02.png)


Figure 2: Schematic of triplet-based supervised learning where one agent is a teacher and the other a student. **A** : A new object is shown only to the Teacher. **B** : The Teacher forms a representation of the object and sends the Student a triplet relating the new object to two objects previously observed by both agents. **C** : The Student interprets the triplet in their own representation space and eliminates the half-plane where the new object cannot be located (shaded in red) according to the triplet. 

In particular, our framework predicts that few-shot transfer learning performance should have a U-shaped relationship with alignment. To verify the predictions of our framework and to probe for additional properties that arise as a consequence of alignment, we conduct a series of experiments in which we assess the downstream properties of human-aligned models compared to their non-aligned counterparts. From our experiments comparing 491 large computer-vision models to over 425,000 human judgments (across 1200 participants), we identify three properties: 

- Models with either high or low alignment with humans are better at few-shot learning (FSL) than models with medium alignment, even correcting for pre-training performance. 

- Human-aligned models are more robust to adversarial examples, even when correcting for classification performance on the pre-training dataset. 

- Human-aligned models are more robust to domain shift, even when correcting for classification performance on the pre-training dataset. 

The U-shaped relationship between alignment and few-shot learning helps to explain why previous results have not consistently observed benefits of representational alignment. Overall, our results suggest that representational alignment can provide real and significant benefits for downstream tasks, but that it may be a sufficient rather than necessary condition for these benefits to emerge. 

2 

## **2 Related Work** 

With AI systems entering mainstream usage, aligning these models with human values and understanding is increasingly important [6, 15]. This concept, referred to as AI alignment, is an important but still largely open problem [41], in part due to the difficulty of formalizing it [32], or even reaching consensus on a definition [15]. In this paper, we focus on formalizing a specific aspect of AI alignment: representational alignment. 

Representational alignment is a measure of agreement between the representations of two learning agents (one of whom is typically a human). There are numerous names, definitions, measures, and uses of this form of alignment across various fields, including cognitive science, neuroscience, and machine learning. Some of the other names include latent space alignment [36], concept(ual) alignment [33, 24], system alignment [7, 28, 1], representational similarity analysis (RSA) [16], and model alignment [22]. Shepard [30] proposed that human representations can be recovered by using behavioral data to measure the similarity of a set of stimuli and then finding embeddings that satisfy those similarity associations using methods like multi-dimensional scaling (MDS). Similarly, in neuroscience, Representational Similarity Analysis (RSA) is a popular technique for relating neural, behavioral, and computational representations of the same set of stimuli via similarity analysis [16]. While similarity analysis has clearly proven itself to be a powerful method, exhaustively collecting pairwise similarity judgments is expensive ( _O_ ( _N_[2] ) judgments for _N_ stimuli) and there have been numerous proposals aiming to develop more efficient methods of recovering human representations. 

Jamieson and Nowak [14] proposed an active learning scheme for querying for human judgments using triplets of the form “is _a_ closer to _b_ than to _c_ ?” and derived bounds on the number of required queries for lossless completion of the full similarity matrix using such queries. When an approximate solution is acceptable, Peterson et al. [27] showed that pre-trained computer vision models can be used to approximate human perceptual similarity judgments over images. Marjieh et al. [21] showed that human perceptual similarity can be more accurately, but still efficiently, approximated from natural language descriptions of the stimuli of interest (for example by using large language models to estimate similarity over pairs of these descriptions). Marjieh et al. [22] extended this result to more domains (vision, audio, and video) and measured alignment for hundreds of pre-trained models. 

Several recent studies have also attempted to identify what design choices lead to improved representational alignment in models [18, 24, 5], although Moschella et al. [23] found that even with variation in design choices, many models trained on the same dataset end up learning similar ‘relative representations’ (embeddings projected into a relational form like a similarity matrix), or in other words, converge to the same representational space. Tucker et al. [36] showed that representational alignment emerges not only in static settings like image classification, but also dynamic reinforcement learning tasks involving human-robot interaction. Several studies have also focused on studying alignment specifically in humans, both between different people and for a single person but across multiple tasks and domains [7, 28]. 

Although several recent papers have proposed ways to measure [21, 22], explain [24, 18], and even improve [27, 5] the representational alignment of models, few have focused on studying the downstream impact of a model being representationally aligned with humans, and many studies simply rely on the intuition that better alignment leads to better performance to justify pursuing increased alignment. While there is recent evidence to suggest that alignment may help humans learn across domains and perform zero-shot generalization [1], there is also evidence to suggest that alignment may not always be beneficial for models, with models scoring low on alignment metrics achieving higher performance on downstream tasks like image classification [18, 24, 5]. Our goal in this paper is to conduct an in-depth study into the downstream effects of representational alignment. Our theoretical and empirical results both validate and explain the apparently conflicting results seen in previous literature as special cases of the (more complex than previously suspected) effects of representational alignment. 

## **3 Theory** 

If we want to measure representational alignment across different architectures with potentially mismatched embedding coordinate spaces and dimensionalities, then we need a definition of representation spaces that enables comparisons of such spaces. Inspired by the cognitive science literature on 

3 

using non-metric similarity triplets to recover hidden psychological representations, and building on rigorous computational theory that analyzes such triplets [14], we propose a triplet-based definition of representation learning. We summarize representational alignment under our framework in the schematic shown in Figure 1. 

**Definition 3.1.** As proposed by Jamieson and Nowak [14], for an ordered set of _n_ objects in _d_ dimensions represented as a vector _x ∈_ R _[nd]_ , a **similarity triplet** corresponding to the question ‘is _xi_ closer to _xj_ than to _xk_ ?’ is a membership query of the form 1 _x∈rijk_ where _rijk_ = _{x ∈_ R _[nd]_ : _|xi − xj| < |xi − xk|}_ . 

**Definition 3.2.** For an ordered set of objects _x ∈_ R _[nd]_ and model _M_ with embeddings _fM_ : R _[d] →_ R _[d][M]_ , the **triplet-based representation space** of _M_ is the set of all triplets _SM_ ( _x_ ) = _{_ ( _i, j, k_ ) _∈ {_ 1 _, ..., n}_[3] : 1 _x∈rijkM[, i]_[=] _[j, j]_[=] _[k, k]_[=] _[i][}]_[where] _[r] ijk[M]_[=] _[{][x][∈]_[R] _[nd]_[:] _[|][f][M]_[(] _[x][i]_[)] _[ −][f][M]_[(] _[x][j]_[)] _[|][<] |fM_ ( _xi_ ) _− fM_ ( _xk_ ) _|}_ . _Remark_ 3.3 _._ Since switching the order of _j_ and _k_ in a triplet deterministically flips the query between 0 and 1, it is easy to see that, for a set of _n_ objects, a representation space is determined by a set of just _[n]_[(] _[n][−]_[1] 2[)(] _[n][−]_[2][)] unique triplets. Thus, for a set of _n_ objects, _SM_ ( _x_ ) can be represented as a binary vector of length _[n]_[(] _[n][−]_[1] 2[)(] _[n][−]_[2][)] and we use this interpretation in the remainder of this section. 

**Definition 3.4.** Consider a training set _x ∈_ R _[nd]_ and a corresponding target representation space _ST_ ( _x_ ). We define **representation learning** with a parametrized model _Mθ_ as optimizing the objective min _θ ℓ_ ( _ST_ ( _x_ ) _, SMθ_ ( _x_ )) where _ℓ_ is a loss function penalizing divergence between the target and learned representation spaces. 

Sucholutsky et al. [34] showed that many ML objectives and associated supervision signals, ranging from similarity judgments for contrastive learning to labels for classification, can be converted into similarity triplets making them compatible with this definition of representation learning. We also immediately get an intuitive definition of alignment between two agents (e.g. a model and a person) as the rate of agreement on matched triplets (also known as simple match coefficient). 

**Definition 3.5.** Consider a dataset _x ∈_ R _[nd]_ and two agents, _A_ and _B_ , with corresponding representation spaces _SA_ ( _x_ ) and _SB_ ( _x_ ). We define **representational misalignment** between _A_ and _B_ as _DR_ ( _A, B_ ; _x_ ) = _[||] n[S]_ ( _[A] n_[(] _−[x]_ 1)([)] _[−][S] n[B] −_[(] 2) _[x]_[)] _/[||]_ 2[1][.] **[Representational alignment]**[ is then simply][ 1] _[ −][D][R]_[(] _[A, B]_[;] _[ x]_[)][.] 

**Definition 3.6.** Consider a single set of three points _t_ = ( _xi, xj, xk_ ) _∈ X ⊆_ R[3] _[d]_ sampled uniformly at random, and two agents, _A_ and _B_ . We define **stochastic representational misalignment** between _A_ and _B_ as _DP_ ( _A, B_ ; _X_ ) = _P_ ( 1 _x∈rijkA_[=][ 1] _[x][∈][r] ijk[B]_[)][.] **[Stochastic representational alignment]**[ is then] simply 1 _− DP_ ( _A, B_ ; _X_ ). 

Since each triplet corresponds to one bit of information [14], and we have a probabilistic definition of alignment, we can now use information theory to define what it means for one agent to supervise another (potentially misaligned) agent. We visualize supervised learning under our framework in Figure 2. For perfectly aligned models, Jamieson and Nowak [14] derive the following result. **Lemma 3.7.** _Consider input space X ⊆_ R _[nd] , shared data x ∼ X, new object c ∈_ R _[d] , and models A and B with DP_ ( _A, B_ ; _X_ ) = 0 _._ Ω( _d_ log( _n_ )) _triplet queries are required for B to identify the location of c relative to all objects in x._ 

**Proposition 3.8.** _For input space X ⊆_ R _[nd] , shared data x ∼ X, new object c ∈_ R _[d] , and models d_ log( _n_ ) _A, B with DP_ ( _A, B_ ; _X_ ) = _ϵ,_ Ω( 1+ _ϵ_ log( _ϵ_ )+(1 _−ϵ_ ) log(1 _−ϵ_ )[)] _[ triplets are required for][ B][ to learn][ c][.]_ 

_Proof._ Consider a communication game involving Alice and Bob, a dataset _x_ that they both have their own representations for (we call this ‘shared data’), and a new object _c_ that only Alice can see. Alice can only communicate with Bob by giving binary responses to triplet queries based on Alice’s own representations of the objects. The goal is for Bob to learn the location of _c_ with respect to the other objects in _x_ . _DP_ ( _A, B_ ; _X_ ) = _ϵ_ is the probability for any triplet drawn from _x ∪ c_ to be flipped in Bob’s representation space relative to Alice’s. This is equivalent to Alice and Bob communicating over a binary symmetric channel where the probability of a bit flip is _ϵ_ . The capacity of this channel is _Cϵ_ = 1 _− H_ ( _ϵ,_ 1 _− ϵ_ ) = 1 + _ϵ_ log( _ϵ_ ) + (1 _− ϵ_ ) log(1 _− ϵ_ ). For error-free communication over this channel, the highest achievable rate is _R < Cϵ_ , meaning at least _C_ 1 _ϵ_[bits are required to communicate] each useful bit over this channel. By Lemma 3.7, Ω( _d_ log( _n_ )) useful bits are required for Bob to learn the location of _c_ over a channel with no error, and thus a total of Ω( _[d]_[ lo] _C_[g(] _ϵ[n]_[)] ) bits are required. 

4 

**Theorem 3.9** ( **Few-shot learning** ) **.** _Consider input space X ⊆_ R _[nd] , shared data x ∼ X, new objects c ∈_ R _[md] and three models A, B_ 1 _, and B_ 2 _with DP_ ( _A, B_ 1; _X_ ) = _ϵB_ 1 _and DP_ ( _A, B_ 2; _X_ ) = _ϵB_ 2 _. If |_ 0 _._ 5 _− ϵB_ 1 _| < |_ 0 _._ 5 _− ϵB_ 2 _|, then B_ 2 _requires fewer queries to learn c than B_ 1 _does._ 

_Proof._ From Proposition 3.8, it immediately follows that _B_ 1 requires Ω( _[md] C_[ lo] _ϵB_[g(] 1 _[n]_[)] ) queries and _B_ 2 requires Ω( _[md] C_[ lo] _ϵB_[g(] 2 _[n]_[)] ) queries. _Cϵ_ : [0 _,_ 1] _→_ [0 _,_ 1] is a symmetric convex function with a minimum of 0 at _ϵ_ = 0 _._ 5 and maxima of 1 at _ϵ_ = 0 _,_ 1. Thus, if f _|_ 0 _._ 5 _− ϵB_ 1 _| < |_ 0 _._ 5 _− ϵB_ 2 _|_ , then Ω( _[md] C_[ lo] _ϵB_[g(] 1 _[n]_[)] ) _>_ Ω( _[md] C_[ lo] _ϵB_[g(] 2 _[n]_[)] ) . 

Sucholutsky et al. [34] showed that commonly used supervision signals, like hard and soft classification labels, can be converted into sets of triplets. Reducing the number of required triplets to learn the location of a new object is equivalent to reducing the number of required labels. **It follows from Theorem 3.9 that high or low alignment leads to high few-shot learning performance on downstream tasks like classification** (where learning a new class can be formulated as learning the location of the centroid or boundaries of that class using a small number of labels), **while medium alignment leads to low few-shot learning performance** . Intuitively, this comes as a result of mutual information between the teacher and student representations being minimized when alignment is at 0 _._ 5. In other words, if Bob knows that most bits coming from Alice are flipped (i.e. _DP_ ( _A, B_ ) = _ϵ >_ 0 _._ 5), then going forward, Bob can flip all the bits coming from Alice to achieve a lower error rate (i.e. _DP_ ( _A, B_[¯] ) = 1 _− ϵ <_ 0 _._ 5). 

We note that generalizing under domain-shift can be considered a form of zero-shot transfer learning and selecting adversarial examples can be seen as selecting objects that maximize representational disagreement between a model and a human. Under these interpretations, our framework also predicts that highly-aligned models are robust to both domain shift and adversarial examples though the framing in these cases is less intuitive than for the few-shot learning case. We share some preliminary theoretical analyses of robustness properties in the Supplement. 

## **4 Experiments** 

Our theoretical analysis shows how representational alignment can serve as a source of inductive bias that reduces the number of bits that need to be acquired from data. In particular, Theorem 3.9 predicts an unexpected U-shaped relationship between alignment and few-shot learning performance. We now present a series of experiments designed to test this prediction and examine whether it extends to robustness to adversarial examples and domain shift. 

**Models.** The pre-trained models evaluated in this paper are taken from the PyTorch Image Models package [39]. The full list of 491 models used in our experiments can be found in the Supplement. 

**Data.** All of the models used in this paper were pre-trained on ImageNet-1k [29] and had their performance evaluated on the ImageNet-1k validation set. Adversarial robustness was measured on the ImageNet-A dataset of adversarial examples that have been found to foil most ImageNet models [10]. Zero-shot domain shift robustness was measured on the ImageNet-R dataset of renditions (e.g., paintings, toys, statues, etc.) of ImageNet classes [9] and the ImageNet-Sketch dataset of blackand-white sketches of the ImageNet classes [38]. All ImageNet and ImageNet variant results come from the PyTorch Image Models package [39]. Few-shot transfer learning performance was evaluated on the CIFAR100 [17] test set with _n ∈{_ 1 _,_ 5 _,_ 10 _,_ 20 _,_ 40 _,_ 80 _}_ examples per class used for few-shot learning and the remaining examples used for evaluation. We measure alignment by computing the three metrics described below on the six image datasets from Peterson et al. [27] and their respective sets of similarity judgments consisting of over 425,000 judgments across 1200 participants, as described by Marjieh et al. [21]. We average each alignment metric over the six datasets. 

**Alignment metrics.** The representational alignment metrics we consider are correlation over pairwise similarity judgments and agreement between similarity triplets (i.e., the proportion of triplets that have the same response for the matched sets of representations). Similarity triplets provide non-metric information since in each query we discard the actual pairwise distances and only retain which one is larger. Analogously, in the pairwise case, if we use Spearman rank correlation then we are comparing the relative magnitudes of pairwise distances (of which the triplets form a subset) and discarding the 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2023_Alignment_with_human_representations_supports_robust_few-shot_learning_images/nips_2023_Alignment_with_human_representations_supports_robust_few-shot_learning.pdf-0006-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2023_Alignment_with_human_representations_supports_robust_few-shot_learning_images/nips_2023_Alignment_with_human_representations_supports_robust_few-shot_learning.pdf-0006-01.png)


Figure 3: **Left** : Average _n_ -shot transfer learning performance using linear probing, a 1-layer classification head, and a 2-layer classification head on CIFAR100 of the five models with highest, lowest, and closest-to-the-mean levels for each of Pearson ( _ρP_ ), Spearman ( _ρS_ ), and triplet alignment. **Right** : Pearson correlation (with 95% confidence intervals) between _n_ -shot transfer learning performance using each classification head on CIFAR100 and each _z_[2] alignment metric. 

magnitudes. On the other hand, Pearson correlation over pairwise similarities takes into account the magnitudes of the pairwise similarity judgments which, if accurate, can provide more information about fine-grained differences that affect alignment. We use all three metrics in our analyses and refer to them as Spearman pairwise alignment, Pearson pairwise alignment, and triplet alignment. 

**Few-shot learning.** We measure few-shot transfer learning performance using three methods: linear probing and two classifier heads. For linear probing, we take the embeddings coming from the penultimate layer of a model and fit a logistic regression using ‘scikit-learn’ [26]. For the classifier heads, we use a one- and two-hidden layer neural network implemented in PyTorch [25], both with an input dropout rate of 0.8 and the latter with a ReLU hidden layer. 

**Correcting correlation for ImageNet performance.** For each of our experiments, ImageNet-1k performance can be a confounding variable when trying to understand the relationship between alignment and a downstream property. To account for this, when measuring correlation between alignment and other properties, we compute partial correlation with ImageNet-1k Top-1 validation accuracy as a covariate using the ‘Pingouin’ Python package [37]. 

## **4.1 Results** 

**Alignment predicts few-shot learning performance.** For each alignment metric, we compare the _n_ -shot transfer learning performance of the five models with the highest alignment, the five models with the lowest alignment, and the five models with the nearest-to-the-mean alignment as shown in Figure 3. We find that according to all three alignment metrics, the most aligned models have far better _n_ -shot learning performance at all levels of _n_ . 

**Alignment predicts adversarial robustness.** For each alignment metric, we also compare the performance on adversarial examples of the five models with the highest alignment, the five models with the lowest alignment, and the five models with the nearest-to-the-mean alignment as seen in Table 1. We again find that according to all three alignment metrics, the most aligned models perform far better on both Top-1 and Top-5 predictions for ImageNet-A. 

**Alignment predicts domain-shift robustness.** Finally, for each alignment metric, we also compare the domain-shift robustness of the five models with the highest alignment, the five models with the lowest alignment, and the five models with the nearest-to-the-mean alignment as seen in Table 1. We find that for all three alignment metrics, the most aligned models perform far better on both Top-1 and Top-5 predictions for both ImageNet-R and ImageNet-S. 

6 

Table 1: Average top-1 and top-5 accuracy on ImageNet-A, ImageNet-S, and ImageNet-R of the five models with highest, lowest, and closest-to-the-mean levels of Pearson ( _ρP_ ), Spearman ( _ρS_ ), and triplet alignment. 

||**Alignment metrics**<br>_ρ_**p**<br>_ρ_**s**<br>Triplet|
|---|---|
|**IN-A**<br>**(Top 1)**<br>**IN-A**<br>**(Top 5)**|High 5<br>59.93<br>46.83<br>44.56<br>Mid 5<br>10.26<br>12.30<br>18.35<br>Low 5<br>30.89<br>21.80<br>29.99|
||High 5<br>84.92<br>73.86<br>72.20<br>Mid 5<br>36.54<br>38.14<br>46.63<br>Low 5<br>59.10<br>49.36<br>57.70|
|**IN-R**<br>**(Top 1)**<br>**IN-R**<br>**(Top 5)**|High 5<br>61.57<br>53.78<br>54.27<br>Mid 5<br>39.16<br>40.01<br>41.80<br>Low 5<br>51.33<br>46.76<br>50.16|
||High 5<br>75.92<br>68.61<br>68.82<br>Mid 5<br>55.08<br>56.81<br>57.67<br>Low 5<br>66.55<br>61.84<br>65.22|
|**IN-S**<br>**(Top 1)**<br>**IN-S**<br>**(Top 5)**|High 5<br>48.06<br>40.37<br>40.31<br>Mid 5<br>26.90<br>28.02<br>29.85<br>Low 5<br>38.11<br>33.90<br>37.04|
||High 5<br>71.10<br>62.45<br>62.24<br>Mid 5<br>44.90<br>46.67<br>48.58<br>Low 5<br>58.91<br>53.43<br>57.57|



Table 2: Pearson correlation between Top1 and Top-5 accuracy on ImageNet-A, ImageNet-R, and ImageNet-Sketch (-S) and Pearson ( _ρP_ ), Spearman ( _ρS_ ), and triplet _z_[2] alignment metrics. 

||**Alignment metrics**<br>_ρ_**p**<br>_ρ_**s**<br>Triplet|
|---|---|
|**IN-A**<br>**(Top 1)**<br>**IN-A**<br>**(Top 5)**|0.239<br>(p=0.000)<br>0.151<br>(p=0.001)<br>0.157<br>(p=0.000)|
||0.210<br>(p=0.000)<br>0.112<br>(p=0.013)<br>0.121<br>(p=0.007)|
|**IN-R**<br>**(Top 1)**<br>**IN-R**<br>**(Top 5)**|0.166<br>(p=0.000)<br>0.164<br>(p=0.000)<br>0.182<br>(p=0.000)|
||0.173<br>(p=0.000)<br>0.170<br>(p=0.000)<br>0.194<br>(p=0.000)|
|**IN-S**<br>**(Top 1)**<br>**IN-S**<br>**(Top 5)**|0.191<br>(p=0.000)<br>0.164<br>(p=0.000)<br>0.179<br>(p=0.000)|
||0.211<br>(p=0.000)<br>0.197<br>(p=0.000)<br>0.218<br>(p=0.000)|



**Alignment has non-linear relationships with downstream properties.** Based on the results in Table 1 and Figure 3, while models with low alignment underperform models with high alignment, they seem to consistently outperform models with medium alignment. Furthermore, when looking at the entire set of models in Figure 4 (Left), there appears to be a non-linear, potentially quadratic relationship between alignment and few-shot learning performance. This matches the U-shaped behavior predicted by Theorem 3.9. To test this relationship, we measure the correlations between the downstream properties of interest and the _z_[2] transformations of each alignment metric where _z_[2] ( _xi_ ) = ( _[x][i] σ[−][µ]_ )[2] , _µ_ is the mean, and _σ_ is the standard deviation. We find statistically significant, positive correlations between the _z_[2] alignment metrics and few-shot transfer learning performance as seen in Figure 3 (Right). Using _z_[2] alignment metrics also results in slightly stronger correlations for domain-shift robustness but weaker correlations for adversarial robustness as seen in Table 2. 

**Alignment is a domain-specific metric.** Since CIFAR100 is made up of naturalistic object images and so were the pre-training dataset and alignment evaluation datasets, we now test few-shot transfer learning performance on the MNIST [19] and FMNIST [40] datasets with _n ∈{_ 1 _,_ 5 _,_ 10 _,_ 20 _,_ 40 _,_ 80 _}_ examples per class to evaluate how alignment generalizes to datasets that look very different than the ones on which it was measured. FMNIST is a dataset of grayscale clothing item images. While fairly different, it is still made up of everyday objects as are the datasets on which we measured alignment. Meanwhile, MNIST is a dataset of grayscale handwritten digit images and is completely different from the datasets on which measured alignment. Thus, CIFAR100 is most similar to the datasets on which alignment was measured, followed by FMNIST and then MNIST. As shown in Figure 4 (Right), for all three datasets, we find statistically significant correlations between our Pearson alignment _z_[2] metric and few-shot transfer learning performance. However, we find that the effect size decreases for FMNIST and even more so for MNIST. This suggests that like any other metric, there are limits to how well representational alignment can generalize, and that it has the strongest predictive power when measured on datasets similar to those on which downstream properties are being evaluated. 

**Interactions of architecture and alignment.** The results above suggest that when alignment is measured on near-distribution datasets, it can predict about 20% of the variance in one-shot learning performance. This is impressive given the large number of factors we know have an impact on 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2023_Alignment_with_human_representations_supports_robust_few-shot_learning_images/nips_2023_Alignment_with_human_representations_supports_robust_few-shot_learning.pdf-0008-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2023_Alignment_with_human_representations_supports_robust_few-shot_learning_images/nips_2023_Alignment_with_human_representations_supports_robust_few-shot_learning.pdf-0008-01.png)


Figure 4: **Left** : Comparing 1-shot learning performance on CIFAR100 to Pearson pairwise alignment for all 491 models. **Right** : Pearson correlation (with 95% confidence intervals) between _n_ -shot transfer learning performance using linear probing and the Pearson ( _ρP_ ) _z_[2] alignment metric for CIFAR100, FMNIST, and MNIST. 

downstream performance (architecture, optimizer, random seed, initialization, early termination, hardware effects on stochasticity, and many other hyperparameters). We also investigated whether some of these factors can predict alignment or interact with it to affect its relationship with downstream performance. Going through the results of our 491 models (see the Supplemental Information for the full results), we find the following insights ranked from most supporting evidence to least: 1) knowledge distillation[11] seems to increase both alignment and few-shot learning performance; 2) increasing depth/size tends to increase at least one of alignment and FSL performance (increasing ResNet[8] depth improves alignment but not FSL performance increasing depth for Resnet-RS[2] increases few-shot learning performance and not alignment; increasing Swin Transformer [20] size increases both); 3) pre-training on ImageNet21k/22k before fine-tuning on ImageNet1k increases both alignment and few-shot learning performance; 4) batch-normalization [13] for VGG [31] leads to lower alignment but slightly higher FSL performance; 5) pruning EfficientNet [35] seems to slightly decrease alignment but maintain FSL performance. We also find that for many architectures, the models cluster together (i.e. almost all MobileNets [12] have low-to-medium alignment and low FSL performance, almost all NFNets [3] have high alignment and medium-to-high FSL performance). These findings suggest that architecture and training decisions can impact alignment as well as its effect on downstream properties. Thus researchers may be able to directly optimize for alignment when selecting, designing, or optimizing model architectures. 

## **5 Discussion** 

Our experimental results confirm our theoretical predictions – not only do highly-aligned models exhibit better few-shot learning and robustness properties than models with lower alignment, but we also observe the U-shaped relationship with alignment predicted in Theorem 3.9. We now dissect the theoretical and empirical results via three key questions. 

**Which alignment metric should we be using?** We find that Spearman pairwise alignment is almost perfectly correlated with triplet alignment ( _ρ_ = 0 _._ 992). This suggests that triplets, though only a small fraction of the full set of quadruplets, already capture the majority of the non-metric information contained in the entire set of quadruplets. Pearson pairwise alignment has a strong, though not quite as strong, correlation with triplet alignment ( _ρ_ = 0 _._ 824). While all three metrics have statistically significant correlations with the downstream properties we care about, Pearson pairwise alignment seems to have the strongest correlations. This suggests that there is recoverable metric information about representations encoded in the magnitudes of human similarity judgments, even when these judgments are potentially noisy due to being elicited without anchor stimuli that would ground the scale. The information-theoretic representation learning framework would need to be extended in future work to quantify this additional information. 

8 

**Which positive downstream properties do human-aligned models exhibit?** As predicted by our information-theoretic representation learning framework, our experiments suggest that very humanaligned models are better at few-shot transfer learning, more robust to adversarial examples, and more robust to test-time domain shift than models with lower degrees of alignment. The correlations between alignment and each downstream property were positive and statistically significant and, in every experiment we conducted, the models with the highest level of alignment outperformed the other models. These results seem to confirm the intuition that human alignment is useful in tasks where we want to use human supervision to elicit human-like behavior. While the models with the highest level of alignment clearly exhibit the best downstream performance across all three sets of tasks, our results suggest an additional unexpected insight: there is a U-shaped relationship between alignment and two of the properties we test: robustness to domain shift and few-shot transfer learning performance. Thus, while a high degree of alignment may be sufficient for eliciting desirable properties in models, it does not appear to be necessary. In fact, in cases where achieving high alignment is impractical (e.g., due to limitations on human-labeled data), it is possible that better results may be achieved by avoiding alignment altogether. 

**Are there downsides and limitations to representational alignment?** We have shown that alignment is not a domain-agnostic metric and that its downstream effects are strongest when evaluated on in-distribution datasets. It is also clear that increasing alignment can damage performance across multiple criteria when that increase moves the alignment level into the medium range. But are there downsides to increasing alignment of models that are already at or past that range? Throughout this study, one of our key assumptions was that the task being solved is designed and specified by humans, or at least easily solvable by humans. However, there are numerous domains where humans have poor performance or where our representations of the problem or stimuli are not helpful for solving the task and a different set of inductive biases are required. For example, many domains targeted by deep learning – such as protein folding, drug design, and social network analysis – require geometric inductive biases [4]. In these cases, the goal should be to achieve alignment with the underlying laws governing the system of interest (e.g., physical forces or mathematical laws), rather than with humans. Finally, we note that as in all cases where models are trained with human data, social biases found in the human data may be reflected or even amplified in representationally-aligned models. 

## **6 Conclusion** 

Our findings confirm the intuition that representational alignment with humans elicits desirable humanlike traits in models, including the ability to generalize from small data and lower susceptibility to adversarial attacks and domain shift. However, as both our theory and experiments suggest, increasing alignment is not always desirable, first due to a U-shaped relationship between alignment and desirable traits, and second, because there are domains where human representations simply are not useful. Notably, our discovery of the U-shaped relationship serves to resolve the tension between previously conflicting findings regarding whether alignment improves performance. We hope that our framework and results motivate further study into both the positive and negative consequences of aligning models with humans across diverse domains. We believe that representational alignment is a quantifiable and tangible route for making progress on the general AI alignment problem by allowing us to measure agreement between models and humans even over abstract domains. 

**Limitations and broader impact.** We discussed the limitations and potential negative impacts of _alignment_ in Section 5, but we also want to note the potential limitations and impacts of our _study of alignment_ . First, we note that our approach assumes access to internal representations of models, but many new models today are being released via closed-source APIs that provide access only to model outputs. We hope that this and future studies on representational alignment will encourage developers to provide more open access to model internals. Second, we note that our experiments only evaluated alignment with the aggregate representational space of a large group of people which may not actually be individually representative of any of those people. Our proposed theory allows for alignment with either individuals or groups, but it is difficult to collect a large set of similarity judgments from individual participants. To ensure that we can understand whose voices, biases, and beliefs our models are actually aligned with, we believe that finding a way to account for individual differences is an important future direction. 

9 

## **Acknowledgements** 

We would like to thank Lukas Muttenthaler for excellent discussions that helped shape some of the ideas explored in this paper. This work was supported by an ONR grant (N00014-18-1-2873) to TLG and an NSERC fellowship (567554-2022) to IS. 

## **References** 

- [1] K. Aho, B. D. Roads, and B. C. Love. System alignment supports cross-domain learning and zero-shot generalisation. _Cognition_ , 227:105200, 2022. ISSN 0010-0277. doi: https: //doi.org/10.1016/j.cognition.2022.105200. 

- [2] I. Bello, W. Fedus, X. Du, E. D. Cubuk, A. Srinivas, T.-Y. Lin, J. Shlens, and B. Zoph. Revisiting resnets: Improved training and scaling strategies. _Advances in Neural Information Processing Systems_ , 34:22614–22627, 2021. 

- [3] A. Brock, S. De, S. L. Smith, and K. Simonyan. High-performance large-scale image recognition without normalization. In _International Conference on Machine Learning_ , pages 1059–1071. PMLR, 2021. 

- [4] M. M. Bronstein, J. Bruna, T. Cohen, and P. Veliˇckovi´c. Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. _arXiv preprint arXiv:2104.13478_ , 2021. 

- [5] T. Fel, I. Felipe, D. Linsley, and T. Serre. Harmonizing the object recognition strategies of deep neural networks with humans. _arXiv preprint arXiv:2211.04533_ , 2022. 

- [6] I. Gabriel. Artificial intelligence, values, and alignment. _Minds and machines_ , 30(3):411–437, 2020. 

- [7] R. L. Goldstone and B. J. Rogosky. Using relations within conceptual systems to translate across conceptual systems. _Cognition_ , 84(3):295–320, 2002. ISSN 0010-0277. doi: https: //doi.org/10.1016/S0010-0277(02)00053-7. 

- [8] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 770– 778, 2016. 

- [9] D. Hendrycks, S. Basart, N. Mu, S. Kadavath, F. Wang, E. Dorundo, R. Desai, T. Zhu, S. Parajuli, M. Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 8340–8349, 2021. 

- [10] D. Hendrycks, K. Zhao, S. Basart, J. Steinhardt, and D. Song. Natural adversarial examples. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15262–15271, 2021. 

- [11] G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network. _arXiv preprint arXiv:1503.02531_ , 2015. 

- [12] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. _arXiv preprint arXiv:1704.04861_ , 2017. 

- [13] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In _International conference on machine learning_ , pages 448–456. pmlr, 2015. 

- [14] K. G. Jamieson and R. D. Nowak. Low-dimensional embedding using adaptively selected ordinal data. _2011 49th Annual Allerton Conference on Communication, Control, and Computing (Allerton)_ , pages 1077–1084, 2011. 

- [15] J. H. Kirchner, L. Smith, J. Thibodeau, K. McDonell, and L. Reynolds. Researching alignment research: Unsupervised analysis. _arXiv preprint arXiv:2206.02841_ , 2022. 

- [16] N. Kriegeskorte, M. Mur, and P. Bandettini. Representational similarity analysis - connecting the branches of systems neuroscience. _Frontiers in Systems Neuroscience_ , 2, 2008. ISSN 1662-5137. doi: 10.3389/neuro.06.004.2008. 

- [17] A. Krizhevsky, G. Hinton, et al. Learning multiple layers of features from tiny images. 2009. 

10 

- [18] M. Kumar, N. Houlsby, N. Kalchbrenner, and E. D. Cubuk. Do better ImageNet classifiers assess perceptual similarity better? _Transactions of Machine Learning Research_ , 2022. 

- [19] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. _Proceedings of the IEEE_ , 86(11):2278–2324, 1998. doi: 10.1109/5.726791. 

- [20] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 10012–10022, 2021. 

- [21] R. Marjieh, I. Sucholutsky, T. R. Sumers, N. Jacoby, and T. L. Griffiths. Predicting human similarity judgments using large language models. _arXiv preprint arXiv:2202.04728_ , 2022. 

- [22] R. Marjieh, P. van Rijn, I. Sucholutsky, T. R. Sumers, H. Lee, T. L. Griffiths, and N. Jacoby. Words are all you need? Capturing human sensory similarity with textual descriptors. _arXiv preprint arXiv:2206.04105_ , 2022. 

- [23] L. Moschella, V. Maiorca, M. Fumero, A. Norelli, F. Locatello, and E. Rodolà. Relative representations enable zero-shot latent space communication. _arXiv preprint arXiv:2209.15430_ , 2022. 

- [24] L. Muttenthaler, J. Dippel, L. Linhardt, R. A. Vandermeulen, and S. Kornblith. Human alignment of neural network representations. _arXiv preprint arXiv:2211.01201_ , 2022. 

- [25] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala. PyTorch: An imperative style, high-performance deep learning library. In _Advances in Neural Information Processing Systems 32_ , pages 8024–8035. Curran Associates, Inc., 2019. 

- [26] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. _Journal of Machine Learning Research_ , 12:2825–2830, 2011. 

- [27] J. C. Peterson, J. T. Abbott, and T. L. Griffiths. Evaluating (and improving) the correspondence between deep neural networks and human representations. _Cognitive science_ , 42(8):2648–2669, 2018. 

- [28] B. D. Roads and B. C. Love. Learning as the unsupervised alignment of conceptual systems. _Nature Machine Intelligence_ , 2(1):76–82, 2020. 

- [29] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. ImageNet large scale visual recognition challenge. _International journal of computer vision_ , 115(3):211–252, 2015. 

- [30] R. N. Shepard. Multidimensional scaling, tree-fitting, and clustering. _Science_ , 210(4468): 390–398, 1980. 

- [31] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. _arXiv preprint arXiv:1409.1556_ , 2014. 

- [32] N. Soares and B. Fallenstein. Aligning superintelligence with human interests: A technical research agenda. _Machine Intelligence Research Institute (MIRI) technical report_ , 8, 2014. 

- [33] A. Stolk, L. Verhagen, and I. Toni. Conceptual alignment: How brains achieve mutual understanding. _Trends in Cognitive Sciences_ , 20(3):180–191, 2016. ISSN 1364-6613. doi: https://doi.org/10.1016/j.tics.2015.11.007. 

- [34] I. Sucholutsky, R. Marjieh, N. Jacoby, and T. L. Griffiths. On the informativeness of supervision signals. _arXiv preprint arXiv:2211.01407_ , 2022. 

- [35] M. Tan and Q. Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In _International conference on machine learning_ , pages 6105–6114. PMLR, 2019. 

- [36] M. Tucker, Y. Zhou, and J. A. Shah. Latent space alignment using adversarially guided self-play. _International Journal of Human–Computer Interaction_ , 38(18-20):1753–1771, 2022. 

- [37] R. Vallat. Pingouin: Statistics in Python. _Journal of Open Source Software_ , 3(31):1026, 2018. doi: 10.21105/joss.01026. 

- [38] H. Wang, S. Ge, Z. Lipton, and E. P. Xing. Learning robust global representations by penalizing local predictive power. _Advances in Neural Information Processing Systems_ , 32, 2019. 

11 

- [39] R. Wightman. PyTorch Image Models. `https://github.com/rwightman/ pytorch-image-models` , 2019. 

- [40] H. Xiao, K. Rasul, and R. Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. _arXiv preprint arXiv:1708.07747_ , 2017. 

- [41] E. Yudkowsky. The AI alignment problem: why it is hard, and where to start. _Symbolic Systems Distinguished Speaker_ , 2016. 

12 

## **A Additional theoretical results** 

**Definition A.1.** Consider model _A_ with input space _X ⊆_ R _[nd]_ , previously observed data _x ∼ X_ , and _k_ class centroids _c ∈_ R _[kd]_ learned by _A_ . We define **domain shift** as an update to the class centroids _c → c[∗] ∈_ R _[kd]_ . **Domain shift sensitivity** is then the proportion of triplets flipped as a result of this update. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2023_Alignment_with_human_representations_supports_robust_few-shot_learning_images/nips_2023_Alignment_with_human_representations_supports_robust_few-shot_learning.pdf-0013-02.png)


From this definition and Theorem 3.9, it immediately follows that sensitivity to domain shift should have the same U-shaped relationship with alignment that few-shot learning does in cases where the teacher model is robust to domain shift. 

**Corollary A.2.** _**(Alignment and domain-shift robustness).** Consider input space X ⊆_ R _[nd] , shared data x ∼ X, and three models, A, B_ 1 _, and B_ 2 _with DP_ ( _A, B_ 1; _X_ ) = _ϵB_ 1 _and DP_ ( _A, B_ 2; _X_ ) = _ϵB_ 2 _. Let c ∈_ R _[kd] be k class centroids learned by A, B_ 1 _and B_ 2 _. If σA_ ( _c, c[∗]_ ) = 0 _and |_ 0 _._ 5 _− ϵB_ 1 _| < |_ 0 _._ 5 _− ϵB_ 2 _|, then σB_ 1( _c, c[∗]_ ) _< σB_ 2( _c, c[∗]_ ) _._ 

We can also use this framework to define robustness to adversarial examples. We assume that an adversarial example is an object that maximizes perceptual (i.e. representational) disagreement between the teacher and the student. 

**Definition A.3.** Consider input space _X ⊆_ R _[nd]_ , shared data _x ∼ X_ , and two models, _A_ and _B_ , with _DP_ ( _A, B_ 1; _X_ ) = _ϵB_ . An **adversarial example** is an object _e ∈_ R _[d]_ that maximizes disagreement between _A_ and _B_ on _S_ ( _x_ ; _e_ ), the subset of _n_ ( _n −_ 1) _/_ 2 triplets relating the objects in _x_ to _e_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2023_Alignment_with_human_representations_supports_robust_few-shot_learning_images/nips_2023_Alignment_with_human_representations_supports_robust_few-shot_learning.pdf-0013-07.png)


Using Definition 3.6 we immediately get the following result. 

**Lemma A.4.** _Consider an input space X ⊆_ R _nd, and two agents, A and B. DP_ ( _A, B_ ; _X_ ) = _E_ [ _[||][S] n[A]_ ( _n_[(] _[X] −_ 1)([)] _[−][S] n[B] −_[(] 2) _[X] /_[)] 2 _[||]_[1][]] _[.]_ 

We can now show that a model that is more aligned with the teacher will, on average, also be more robust to adversarial examples. 

**Theorem A.5.** _**(Alignment and adversarial robustness).** Consider input space X ⊆_ R _[nd] , shared data x ∼ X, and three models, A, B_ 1 _, and B_ 2 _with DP_ ( _A, B_ 1; _X_ ) = _ϵB_ 1 _and DP_ ( _A, B_ 2; _X_ ) = _ϵB_ 2 _. If ϵB_ 1 _< ϵB_ 2 _, then E_ [max _e∈x ||SA_ ( _x_ ; _e_ ) _− SB_ 1( _x_ ; _e_ ) _||_ 1] _< E_ [max _e∈X ||SA_ ( _x_ ; _e_ ) _− SB_ 2( _x_ ; _e_ ) _||_ 1] _._ 

_Proof._ Note that for a set of _k_ binomial random variables _Xi ∼ Bin_ ( _n, p_ ), the expectation of the _k_ -th order statistic is _E_ [ _X_ ( _k_ )] =[�] _[n] x_ =0[(1] _[ −][F]_[(] _[x]_[;] _[ n, p]_[)] _[k]_[)][ where] _[ F]_[(] _[x]_[;] _[ n, p]_[)][=] _[P]_[(] _[X][i][≤][x]_[)][.][In the] case of adversarial examples, let _Xi_ be a random variable corresponding to the set of objects sampled uniformly from the input space _X ⊆_ R _[nd]_ then _U_ = _||SA_ ( _X_ ; _e_ ) _− SB_ 1( _X_ ; _e_ ) _||_ 1 _, Y ∼ Bin_ ( _n_ ( _n −_ 1) _/_ 2 _, ϵB_ 1) and similarly _V_ = _||SA_ ( _X_ ; _e_ ) _− SB_ 2( _X_ ; _e_ ) _||_ 1 _, V ∼ Bin_ ( _n_ ( _n −_ 1) _/_ 2 _, ϵB_ 1). In that case, the expected disagreement of _A_ and _B_ 1 on an adversarial example is _E_ [ _U_ ( _n_ )] =[�] _[n] x_ =0[(] _[n][−]_[1)] _[/]_[2] (1 _− F_ ( _x_ ; _n_ ( _n −_ 1) _/_ 2 _, ϵB_ 1) _[n]_ ) and for _A_ and _B_ 2 it is _E_ [ _V_ ( _n_ )] =[�] _[n] x_ =0[(] _[n][−]_[1)] _[/]_[2] (1 _− F_ ( _x_ ; _n_ ( _n −_ 1) _/_ 2 _, ϵB_ 2) _[n]_ ). If _ϵB_ 1 _< ϵB_ 2 , then _F_ ( _x_ ; _n_ ( _n−_ 1) _/_ 2 _, ϵB_ 1) _> F_ ( _x_ ; _n_ ( _n−_ 1) _/_ 2 _, ϵB_ 2) and thus _E_ [ _U_ ( _n_ )] _< E_ [ _V_ ( _n_ )]. 

_Remark_ A.6 _._ While this theorem shows that increased alignment generally leads to increased adversarial robustness, this relies on a representational metric of adversarial examples. However, in practice, adversarial robustness is often measured using hard classification error as a simple proxy. This proxy does not capture the fine-grained degree of misalignment between humans and a model on each example. As a result, when measuring adversarial robustness using this proxy, the effect of alignment may be dampened by the U-shaped effect seen in other classification settings as mentioned above. 

## **B List of 491 models used in experiments** 

adv_inception_v3, bat_resnext26ts, beit_base_patch16_224, beit_base_patch16_384, beit_large_patch16_224, beit_large_patch16_384, botnet26t_256, cait_s24_224, cait_s24_384, 

13 

cait_s36_384, cait_xs24_384, cait_xxs24_224, cait_xxs24_384, cait_xxs36_224, cait_xxs36_384, coat_lite_mini, coat_lite_small, coat_lite_tiny, coat_mini, coat_tiny, convit_base, convit_small, convit_tiny, convmixer_1024_20_ks9_p14, convmixer_1536_20, convmixer_768_32, convnext_base, convnext_base_384_in22ft1k, convnext_base_in22ft1k, convnext_large, convnext_large_384_in22ft1k, convnext_large_in22ft1k, convnext_small, convnext_tiny, cspdarknet53, cspresnet50, cspresnext50, deit_base_patch16_224, deit_base_patch16_384, deit_small_patch16_224, deit_tiny_patch16_224, densenet121, densenet161, densenet169, densenet201, densenetblur121d, dla102, dla102x, dla102x2, dla169, dla34, dla46_c, dla46x_c, dla60, dla60_res2net, dla60_res2next, dla60x, dla60x_c, dm_nfnet_f0, dm_nfnet_f1, dm_nfnet_f2, dpn107, dpn131, dpn68, dpn68b, dpn92, dpn98, eca_botnext26ts_256, eca_halonext26ts, eca_nfnet_l0, eca_nfnet_l1, eca_nfnet_l2, eca_resnet33ts, eca_resnext26ts, ecaresnet101d, ecaresnet101d_pruned, ecaresnet269d, ecaresnet26t, ecaresnet50d, ecaresnet50d_pruned, ecaresnet50t, ecaresnetlight, efficientnet_b0, efficientnet_b1, efficientnet_b1_pruned, efficientnet_b2, efficientnet_b2_pruned, efficientnet_b3, efficientnet_b3_pruned, efficientnet_b4, efficientnet_el, efficientnet_el_pruned, efficientnet_em, efficientnet_es, efficientnet_es_pruned, efficientnet_lite0, efficientnetv2_rw_m, efficientnetv2_rw_s, efficientnetv2_rw_t, ens_adv_inception_resnet_v2, ese_vovnet19b_dw, ese_vovnet39b, fbnetc_100, fbnetv3_b, fbnetv3_d, fbnetv3_g, gc_efficientnetv2_rw_t, gcresnet33ts, gcresnet50t, gcresnext26ts, gcresnext50ts, gernet_l, gernet_m, gernet_s, ghostnet_100, gluon_inception_v3, gluon_resnet101_v1b, gluon_resnet101_v1c, gluon_resnet101_v1d, gluon_resnet101_v1s, gluon_resnet152_v1b, gluon_resnet152_v1c, gluon_resnet152_v1d, gluon_resnet152_v1s, gluon_resnet18_v1b, gluon_resnet34_v1b, gluon_resnet50_v1b, gluon_resnet50_v1c, gluon_resnet50_v1d, gluon_resnet50_v1s, gluon_resnext101_32x4d, gluon_resnext101_64x4d, gluon_resnext50_32x4d, gluon_senet154, gluon_seresnext101_32x4d, gluon_seresnext101_64x4d, gluon_seresnext50_32x4d, gluon_xception65, gmixer_24_224, gmlp_s16_224, halo2botnet50ts_256, halonet26t, halonet50ts, haloregnetz_b, hardcorenas_a, hardcorenas_b, hardcorenas_c, hardcorenas_d, hardcorenas_e, hardcorenas_f, hrnet_w18, hrnet_w18_small, hrnet_w18_small_v2, hrnet_w30, hrnet_w32, hrnet_w40, hrnet_w44, hrnet_w48, hrnet_w64, ig_resnext101_32x16d, ig_resnext101_32x8d, inception_resnet_v2, inception_v3, inception_v4, jx_nest_base, jx_nest_small, jx_nest_tiny, lambda_resnet26rpt_256, lambda_resnet26t, lambda_resnet50ts, lamhalobotnet50ts_256, lcnet_050, lcnet_075, lcnet_100, legacy_senet154, legacy_seresnet101, legacy_seresnet152, legacy_seresnet18, legacy_seresnet34, legacy_seresnet50, legacy_seresnext101_32x4d, legacy_seresnext26_32x4d, legacy_seresnext50_32x4d, mixer_b16_224, mixer_b16_224_miil, mixnet_l, mixnet_m, mixnet_s, mixnet_xl, mnasnet_100, mnasnet_small, mobilenetv2_050, mobilenetv2_100, mobilenetv2_110d, mobilenetv2_120d, mobilenetv2_140, mobilenetv3_large_100, mobilenetv3_large_100_miil, mobilenetv3_rw, nasnetalarge, nf_regnet_b1, nf_resnet50, nfnet_l0, pit_b_224, pit_s_224, pit_ti_224, pit_xs_224, pnasnet5large, regnetx_002, regnetx_004, regnetx_006, regnetx_008, regnetx_016, regnetx_032, regnetx_040, regnetx_064, regnetx_080, regnetx_120, regnetx_160, regnetx_320, regnety_002, regnety_004, regnety_006, regnety_008, regnety_016, regnety_032, regnety_040, regnety_064, regnety_080, regnety_120, regnety_160, regnety_320, regnetz_b16, regnetz_c16, regnetz_d32, regnetz_d8, regnetz_e8, repvgg_a2, repvgg_b0, repvgg_b1, repvgg_b1g4, repvgg_b2, repvgg_b2g4, repvgg_b3, repvgg_b3g4, res2net101_26w_4s, res2net50_14w_8s, res2net50_26w_4s, res2net50_26w_6s, res2net50_26w_8s, res2net50_48w_2s, res2next50, resmlp_12_224, resmlp_12_distilled_224, resmlp_24_224, resmlp_24_distilled_224, resmlp_36_224, resmlp_36_distilled_224, resmlp_big_24_224, resmlp_big_24_224_in22ft1k, resmlp_big_24_distilled_224, resnest101e, resnest14d, resnest200e, resnest269e, resnest26d, resnest50d, resnest50d_1s4x24d, resnest50d_4s2x40d, resnet101, resnet101d, resnet152, resnet152d, resnet18, resnet18d, resnet200d, resnet26, resnet26d, resnet26t, resnet32ts, resnet33ts, resnet34, resnet34d, resnet50, resnet50_gn, resnet50d, resnet51q, resnet61q, resnetblur50, resnetrs101, resnetrs152, resnetrs200, resnetrs270, resnetrs350, resnetrs420, resnetrs50, resnetv2_101, resnetv2_101x1_bitm, resnetv2_50, resnetv2_50x1_bit_distilled, resnetv2_50x1_bitm, resnext101_32x8d, resnext26ts, resnext50_32x4d, resnext50d_32x4d, rexnet_100, rexnet_130, rexnet_150, rexnet_200, sebotnet33ts_256, sehalonet33ts, selecsls42b, selecsls60, selecsls60b, semnasnet_075, semnasnet_100, seresnet152d, seresnet33ts, seresnet50, seresnext26d_32x4d, seresnext26t_32x4d, seresnext26ts, seresnext50_32x4d, skresnet18, skresnet34, skresnext50_32x4d, spnasnet_100, ssl_resnet18, ssl_resnet50, ssl_resnext101_32x16d, ssl_resnext101_32x4d, ssl_resnext101_32x8d, ssl_resnext50_32x4d, swin_base_patch4_window12_384, swin_base_patch4_window7_224, swin_large_patch4_window12_384, swin_large_patch4_window7_224, 

14 

swin_small_patch4_window7_224, swin_tiny_patch4_window7_224, swsl_resnet18, swsl_resnet50, swsl_resnext101_32x16d, swsl_resnext101_32x4d, swsl_resnext101_32x8d, swsl_resnext50_32x4d, tf_efficientnet_b0, tf_efficientnet_b0_ap, tf_efficientnet_b0_ns, tf_efficientnet_b1, tf_efficientnet_b1_ap, tf_efficientnet_b1_ns, tf_efficientnet_b2, tf_efficientnet_b2_ap, tf_efficientnet_b2_ns, tf_efficientnet_b3, tf_efficientnet_b3_ap, tf_efficientnet_b3_ns, tf_efficientnet_b4, tf_efficientnet_b4_ap, tf_efficientnet_b4_ns, tf_efficientnet_b5, tf_efficientnet_b5_ap, tf_efficientnet_b5_ns, tf_efficientnet_b6, tf_efficientnet_b6_ap, tf_efficientnet_b6_ns, tf_efficientnet_b7, tf_efficientnet_b7_ap, tf_efficientnet_b7_ns, tf_efficientnet_cc_b0_4e, tf_efficientnet_cc_b0_8e, tf_efficientnet_cc_b1_8e, tf_efficientnet_el, tf_efficientnet_em, tf_efficientnet_es, tf_efficientnet_lite0, tf_efficientnet_lite1, tf_efficientnet_lite2, tf_efficientnet_lite3, tf_efficientnet_lite4, tf_efficientnetv2_b0, tf_efficientnetv2_b1, tf_efficientnetv2_b2, tf_efficientnetv2_b3, tf_efficientnetv2_l, tf_efficientnetv2_l_in21ft1k, tf_efficientnetv2_m, tf_efficientnetv2_m_in21ft1k, tf_efficientnetv2_s, tf_efficientnetv2_s_in21ft1k, tf_inception_v3, tf_mixnet_l, tf_mixnet_m, tf_mixnet_s, tf_mobilenetv3_large_075, tf_mobilenetv3_large_100, tf_mobilenetv3_large_minimal_100, tf_mobilenetv3_small_075, tf_mobilenetv3_small_100, tf_mobilenetv3_small_minimal_100, tinynet_a, tinynet_b, tinynet_c, tinynet_d, tinynet_e, tnt_s_patch16_224, tv_densenet121, tv_resnet101, tv_resnet152, tv_resnet34, tv_resnet50, tv_resnext50_32x4d, twins_pcpvt_base, twins_pcpvt_large, twins_pcpvt_small, twins_svt_base, twins_svt_large, twins_svt_small, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn, visformer_small, vit_base_patch16_224, vit_base_patch16_224_miil, vit_base_patch16_384, vit_base_patch32_224, vit_base_patch32_384, vit_base_patch8_224, vit_base_r50_s16_384, vit_small_patch16_224, vit_small_patch16_384, vit_small_patch32_224, vit_small_patch32_384, vit_small_r26_s32_224, vit_small_r26_s32_384, vit_tiny_patch16_224, vit_tiny_patch16_384, vit_tiny_r_s16_p8_224, vit_tiny_r_s16_p8_384, wide_resnet101_2, wide_resnet50_2, xception, xception41, xception65, xception71, xcit_large_24_p16_224, xcit_large_24_p16_224_dist, xcit_large_24_p16_384_dist, xcit_large_24_p8_224, xcit_large_24_p8_224_dist, xcit_large_24_p8_384_dist, xcit_medium_24_p16_224, xcit_medium_24_p16_224_dist, xcit_medium_24_p16_384_dist, xcit_medium_24_p8_224, xcit_medium_24_p8_224_dist, xcit_nano_12_p16_224, xcit_nano_12_p16_224_dist, xcit_nano_12_p16_384_dist, xcit_nano_12_p8_224, xcit_nano_12_p8_224_dist, xcit_nano_12_p8_384_dist, xcit_small_12_p16_224, xcit_small_12_p16_224_dist, xcit_small_12_p16_384_dist, xcit_small_12_p8_224, xcit_small_12_p8_224_dist, xcit_small_12_p8_384_dist, xcit_small_24_p16_224, xcit_small_24_p16_224_dist, xcit_small_24_p16_384_dist, xcit_small_24_p8_224, xcit_small_24_p8_224_dist, xcit_small_24_p8_384_dist, xcit_tiny_12_p16_224, xcit_tiny_12_p16_224_dist, xcit_tiny_12_p16_384_dist, xcit_tiny_12_p8_224, xcit_tiny_12_p8_224_dist, xcit_tiny_12_p8_384_dist, xcit_tiny_24_p16_224, xcit_tiny_24_p16_224_dist, xcit_tiny_24_p16_384_dist, xcit_tiny_24_p8_224, xcit_tiny_24_p8_224_dist, xcit_tiny_24_p8_384_dist 

## **C Reproducibility: Code and results data** 

All code and full results data are provided as part of the supplemental information. We will share them publicly after the anonymity period is over. All experiments were conducted on an AWS “x1.16xlarge” instance (no GPUs). 

15 

