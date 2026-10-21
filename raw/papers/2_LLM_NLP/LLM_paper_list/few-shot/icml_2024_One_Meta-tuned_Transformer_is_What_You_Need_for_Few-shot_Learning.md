## **One Meta-tuned Transformer is What You Need for Few-shot Learning** 

## **Xu Yang**[1] **Huaxiu Yao**[2] **Ying Wei**[3] 

## **Abstract** 

Pre-trained vision transformers have revolutionized few-shot image classification, and it has been recently demonstrated that the previous common practice of meta-learning in synergy with these pre-trained transformers still holds significance. In this work, we design a new framework centered exclusively on attention mechanisms, called MetaFormer, which extends the vision transformers beyond patch token interactions to encompass relationships between samples and tasks simultaneously for further advancing their downstream task performance. Leveraging the intrinsical property of ViTs in handling local patch relationships, we propose Masked Sample Attention (MSA) to efficiently embed the sample relationships into the network, where an adaptive mask is attached for enhancing task-specific feature consistency and providing flexibility in switching between fewshot learning setups. To encapsulate task relationships while filtering out background noise, Patchgrained Task Attention (PTA) is designed to maintain a dynamic knowledge pool consolidating diverse patterns from historical tasks. MetaFormer demonstrates coherence and compatibility with off-the-shelf pre-trained vision transformers and shows significant improvements in both inductive and transductive few-shot learning scenarios, outperforming state-of-the-art methods by up to 8 _._ 77% and 6 _._ 25% on 12 in-domain and 10 crossdomain datasets, respectively. 

## **1. Introduction** 

There has been a sustained focus on few-shot learning (Vinyals et al., 2016b; Snell et al., 2017), a paradigm where only a few labeled samples ( _a.k.a._ support samples) 

1City University of Hong Kong 2University of North Carolina at Chapel Hill[3] Nanyang Technological University. Correspondence to: Xu Yang _<_ xyang337-c@my.cityu.edu.hk _>_ , Ying Wei _<_ ying.wei@ntu.edu.sg _>_ . 

_Proceedings of the 41[st] International Conference on Machine Learning_ , Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s). 

are provided to predict unlabelled samples ( _a.k.a._ query samples). The overarching objective is to emulate human-level intelligence capable of swiftly assimilating new concepts. Meta-learning (Thrun & Pratt, 2012) has been a de-facto approach in dealing with few-shot learning, via leveraging the knowledge learned from previous tasks (Finn et al., 2017; Raghu et al., 2019). State-of-the-art meta-learning practices have evolved from modelling the relationships between samples (Vinyals et al., 2016b; Snell et al., 2017; Oreshkin et al., 2018; Hou et al., 2019; Ye et al., 2020; Chen et al., 2021a) to exploring fine-grained relationships among the building blocks of samples (Zhang et al., 2020a; Doersch et al., 2020; Kang et al., 2021; Afrasiyabi et al., 2022) and, more recently, to conditional meta-learning where the transferable knowledge shared among only closely related tasks improves generalization (Rusu et al., 2018; Yao et al., 2019; Requeima et al., 2019; Bateni et al., 2020; Zhou et al., 2021a; Jiang et al., 2022). 

The majority of the aforementioned research advancements have predominantly unfolded within the realm of Convolutional Neural Networks (CNNs) (He et al., 2016; Zagoruyko & Komodakis, 2016). However, recent strides in pre-trained Vision Transformers (ViTs) (Caron et al., 2021; Touvron et al., 2021; Zhou et al., 2022a) have notably surpassed CNNs across a diverse array of vision tasks (Dosovitskiy et al., 2020; Liu et al., 2021; Zhu et al., 2020; Ranftl et al., 2021; Strudel et al., 2021). Consequently, one anticipates that the various levels of relationships delineated earlier can collaborate with ViTs to propel the performance of fewshot learning to new heights. While we acknowledge the preliminary efforts (Hiller et al., 2022; Hu et al., 2022), demonstrating that meta-learning can effectively synergize with pre-trained ViTs to further enhance their few-shot learning performance, it is noteworthy that, to the best of our knowledge, there has been no exploration into employing self-attention as the exclusive building block to model the aforementioned three levels of relationships. 

In this work, we pose the question of whether it is plausible to construct a high-performing transformer that simultaneously encapsulates the three distinct relationships. We argue that such a design holds the potential to surmount two inherent limitations in prior CNN-based meta-learning models. First, constructing the correspondence between building blocks, such as optimal matching (Zhang et al., 2020a), in 

1 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

CNNs remains computationally costly. In contrast, ViTs, leveraging patches as tokens, intrinsically and efficiently establish the relationship between patches as building blocks. Second, the relationships between tasks in existing CNNbased models hinge on task embeddings (Rusu et al., 2018; Achille et al., 2019; Yao et al., 2019), often derived from image-level embeddings. This approach introduces irrelevant noise such as background and compromises local structures. Henceforth, tasks with similar embeddings modulate their parameters (e.g., through a FiLM layer (Perez et al., 2018; Requeima et al., 2019; Triantafillou et al., 2021)) to stay close, while the high dimensionality of deep neural networks renders such modulation less effective to push away dissimilar tasks that inherently need to be distinctly separated. Conversely, ViTs present an alternative approach by modeling task relationships explicitly through self-attention. 

Motivated by these observations, we propose a few-shot learning architecture centered exclusively on attention mechanisms. Our framework, named MetaFormer (derived from Meta-tuned Transformer), extends the ViT by augmenting the self-attention mechanism beyond patch token interactions to encompass relationships between samples and tasks. In the context of an _N_ -way _K_ -shot task with _n_ patches per sample, and _N_ support and _M_ query samples, the computation of similarities between all samples incurs a potential time complexity of _O_ (( _n_ ( _N_ + _M_ ) _K_ )[2] ). This computational complexity is considerable, particularly in light of the often substantial value of _M_ and the large number of patches ( _n_ ). To address this challenge, we propose Masked Sample Attention (MSA), which implements sample relationships separately after each attention layer of the original ViT in a decoupled manner. Simultaneously, it involves an adaptive mask to not only enforce consistency in taskspecific discriminative features across all samples within a task but also facilitate the switch between few-shot learning setups. For addressing task relationships at the building block level, we propose the Patch-grained Task Attention (PTA). PTA learns a probe vector for each task summarizing discriminative patch-level patterns and maintains a dynamic knowledge pool consolidating diverse patterns from historical tasks. Subsequently, PTA conditions on attention-based mechanisms for knowledge retrieval and aggregation. 

The contributions of this integrated transformer are outlined as follows. (1) _Coherence and compatibility_ : we provide a coherent few-shot learning framework that exclusively leverages attention mechanisms, thus ensuring compatibility with off-the-shelf ViT-backed pre-trained transformers (including CLIP) and enhancing their few-shot learning performances. (2) _Flexibility_ : the adaptive mask surprisingly supports both inductive and transductive few-shot learning protocols. (3) _Practical efficacy_ : we conduct our experiments on 12 in-domain and 10 cross-domain few-shot generalization datasets, on which MetaFormer outperforms both 

inductive and transductive state-of-the-art methods with improvements of up to 8 _._ 77% and 6 _._ 25%. 

## **2. Related work** 

**Meta-Learning in Few-Shot Classification.** Meta-learning serves as a fundamental framework for few-shot learning with the aim of distilling and transferring prior knowledge for quickly adapting to new unseen tasks. Most related to our work are metric-based and feedforward-adaptation meta-learning methods. Metric-based methods (Vinyals et al., 2016b; Snell et al., 2017; Sung et al., 2018; Oreshkin et al., 2018; Lee et al., 2019; Chen et al., 2021a; Ma et al., 2021c; Simon et al., 2020; Lai et al., 2022) seek to embed samples into global universal feature representations and aim at improving distance metrics for better measuring sample similarity in the embedding space. Recent works establish fine-grained local feature matching by leveraging Earth Mover’s Distance (Zhang et al., 2020a), spatial prototypes (Doersch et al., 2020) or set-to-set metrics (Afrasiyabi et al., 2022). However, fixed embedding is not very robust and sufficient to accommodate tasks with significant shifts due to cluttered backgrounds and intricate scenes. Several feedforward approaches are proposed to perform task adaptation. One line of work generates task-conditioned classifiers (Rusu et al., 2018; Xu et al., 2020), convolution kernels (Ma et al., 2021b; Zhmoginov et al., 2022), or batch normalization parameters (Requeima et al., 2019; Bateni et al., 2020). Another line directly adapts the feature embedding to new tasks utilizing within-support (Ye et al., 2020) and support-query (Hou et al., 2019; Kang et al., 2021) sample relationships. Our approach follows the second vein and the main difference is that we embed intra-task interactions into the network at various scales to fully leverage coarse-to-fine multi-scale information for learning a richer task-specific feature embedding. To handle tasks with different distributions, a handful of works built upon the gradient-based methods try to extract the underlying task structure for customizing initialization (Yao et al., 2019; 2020; Zhou et al., 2021a; Jiang et al., 2022). However, these algorithms rely on time-consuming clustering and the discriminative task representations are difficult to learn (Jiang et al., 2022). Recent works (Wang et al., 2022a; Douillard et al., 2022; Wang et al., 2022b; Smith et al., 2023) propose inter-task attention to prevent catastrophic forgetting in the continual learning setting. Unlike them retaining a single vector (Wang et al., 2022a), our method maintains a knowledge pool to organize structured meta-knowledge (i.e., key feature patterns), which is then tailored for reuse in the current task through attention-based aggregation. 

**Vision Transformers for Few-Shot Learning.** Large-scale pre-trained vision transformers (Dosovitskiy et al., 2020; Liu et al., 2021; Tu et al., 2022), utilizing the data-driven 

2 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

prior and self-attention mechanism to encode long-range dependency in the data, generalize better than CNNs (Zhang et al., 2022). Combined with self-supervised pre-training techniques, recent works have shown the potential of fewshot vision transformers via designing self-distillation training to mitigate the overfitting issue (He et al., 2022b; Dong et al., 2022; Lin et al., 2023). For example, HcT (He et al., 2022b) utilize the DINO-based (Caron et al., 2021) teacherstudent framework to distill the global class token and train three cascaded transformers with two pooling layers in between. To further supervise the patch tokens, SUN (Dong et al., 2022) adopts the patch-level pseudo labels generated by the teacher network and SMKD (Lin et al., 2023) introduces the patch reconstruction loss in Masked Image Modeling (MIM) (He et al., 2022a; Bao et al., 2022; Zhou et al., 2022a). Beyond self-supervised ViTs, the initial investigations conducted by Hu et al. (Hu et al., 2022) and Hiller et al. (Hiller et al., 2022) underscore the advantageous impact of integrating meta-learning techniques to further augment the few-shot learning capabilities of ViTs. FewTURE (Hiller et al., 2022), specifically, uses a supportaware patch importance mask learned in the inner loop to address supervision collapse. However, its utility is mostly confined to the top classifier, restricting the network’s ability to refine features for tasks with significant distribution shifts. We also ground our proposed method on pre-trained vision transformers and introduce a novel meta-learning approach that embeds sample and task relationships into ViTs, enabling the learning of more discriminative features for each specific task. Our contribution is orthogonal to SKMD and we empirically show that MetaFormer can further improve the joint performance. 

## **3. MetaFormer for Few-shot Classification** 

We present our approach in this section. The overall architecture of our MetaFormer is illustrated in Figure 1. We start by briefly introducing the meta-tuning practice and the self-attention of vision transformer in Section 3.1 and then elaborate on our proposed Masked Sample Attention (MSA) and Patch-grained Task Attention (PTA) in Section 3.2 and Section 3.3, respectively. Finally, using MSA and PTA as core building blocks, we present a new vision transformer with holistic attention for meta-learning in Section 3.4. 

## **3.1. Preliminaries** 

**Meta-tuning Practice** . Meta-learning aims to learn a model capable of adapting quickly to recognize new classes with only a few labeled examples. Following recent literature (Hu et al., 2022; Hiller et al., 2022), we initially pre-train a model on the base training dataset and then meta-tune the MetaFormer in the episodic training manner (Vinyals et al., 2016b). In a classical _N_ -way _K_ -shot setting, each episode 

randomly selects _N_ classes to form the support set _S_ = _{_ ( _x[c] i[, y] i[c]_[)] _[}][N] i_ =1 _[×][K]_ containing _K_ samples in each class and the _M_ query set _Q_ = �� _x[t] j[, y] j[t]_ �� _j_ =1[with] _[ M]_[samples.] 

**Self-attention in ViTs** . Given a _N_ -way _K_ -shot task with _NK_ + _M_ images _x ∈_ R _[Height][×][W idth][×]_[3] as input, ViTs (Dosovitskiy et al., 2020) first divide individual images into _n_ non-overlapping patches and then map them into _d_ -dimension tokens through a linear projection layer. After that, a trainable class token is prepended as the final input token sequence **X** _∈_ R[(] _[NK]_[+] _[M]_[)] _[×][L][×][d]_ ( _L_ = _n_ +1), taken by several Multi-head Self-Attention (MHSA) layers and MLP layers for feature extraction. Consider a MHSA layer with _H_ heads, and query, key, and value embeddings of the input **X** are given as **Q** = **W** _[Q]_ **X** , **K** = **W** _[K]_ **X** , **V** = **W** _[V]_ **X** , respectively. The output of MHSA is given as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning_images/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning.pdf-0003-08.png)


where **W** _[O]_ is the output projection matrix, _dk_ = _d/H_ denotes the head dimension, and _σ_ ( _·_ ) signifies the softmax activation function. The matrix **A** _i ∈_ R _[L][×][L]_ serves as the attention matrix, measuring the pairwise token affinities across different patch locations. Following the MHSA as defined in (1), each token representing a building block within the image gains awareness of its relational context with all other tokens. 

## **3.2. Masked Sample Attention** 

For computational efficiency, the very few existing methods capture sample relationships by attaching one or several extra attention layers on top of the fixed feature extractor (Ye et al., 2020; Doersch et al., 2020; Hiller et al., 2022). However, it is demonstrated that different layers of the backbone yield different semantic levels of feature embedding and thus different types of knowledge (Raghu et al., 2021). Motivated by this, we propose to leverage coarse-to-fine multi-scale information across layers to capture discriminative sample interactions at the patch token level. 

A straightforward and intuitive approach is to perform selfattention over both patch and sample dimensions simultaneously. Given the task input **X** , the core computation of one MHSA layer primarily revolves around calculating the attention matrix **A** _J ∈_ R[(] _[NK]_[+] _[M]_[)] _[L][×]_[(] _[NK]_[+] _[M]_[)] _[L]_ in (1). Therefore, the complexity of the joint space-sample attention is _O_ (( _NK_ + _M_ )[2] _L_[2] ). Such joint space-sample interaction empowers the vision transformer to capture sample relationships for task-specific embedding, but it comes at a high computational cost and incurs heavy memory footprints. 

Inspired by the divided space-time attention in video transformers (Bertasius et al., 2021), we introduce a more ef- 

3 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning_images/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning.pdf-0004-01.png)


**----- Start of picture text -----**<br>
(c) Masked Sample Attention<br>Full Sample Interaction<br>inductive<br>Class<br>Support Labels Embedding<br>Add & Norm … Q Sample Causal Mask Masked Sample Interaction<br>Layer L …<br>Sample Attention Support Tokens C! R K M S<br>Layer 2 Add & Norm …… V R<br>Query Tokens<br>Layer l<br>Patch Attention<br>…… …… Add & Norm (d) Patch-grained Task Attention Task 1 … Task n 𝐶" Concatenation Along Batch DimensionMasked Interaction<br>Patch Embedding Task Attention Update Element-wise Addition<br>Task Probe Vector Knowledge Pool R Reshape<br>… Matrix Multiplication<br>(b) Holistic Attention Q<br>Retrieval M Mask Operation<br>K, V cos K, V S Softmax<br>Support  Query cos Cosine Similarity<br>Support Q<br>(a) Overall Architecture<br>cat dogbird rabbit<br>Support<br>Query<br>…<br>Fusion<br>Aggregation<br>**----- End of picture text -----**<br>


_Figure 1._ **Overview –** (a) The architecture of the MetaFormer with holistic attention, which extracts feature representations of support and query samples with only one feedforward while following the inductive protocol; (b) Three modules build holistic attention integrated with intra- and inter-task interaction in a sequential mode; (c) Schematic illustration of the proposed Masked Sample Attention (MSA) with sample causal attention mask to exploit the within-support and support-query relations for task-specific embeddings. (d) Schematic illustration of the proposed Patch-grained Task Attention (PTA) where foreground region is further concentrated by previous relevant semantic knowledge with the task-specific probe vector. 

ficient architecture designed to decouple patch attention and sample attention, illustrated in Figure 1(b). In the case of decoupled patch-sample attention, within each layer, our approach initially computes patch-only attention as (1) to obtain features isolating backgrounds and emphasizing underlying objects. Subsequently, we reshape the token sequence to R _[L][×]_[(] _[NK]_[+] _[M]_[)] _[×][d]_ that is fed to MHSA with sample attention matrix **A** _S ∈_ R[(] _[NK]_[+] _[M]_[)] _[×]_[(] _[NK]_[+] _[M]_[)] , incorporating sample interactions across all patches at the same patch location to capture the similarities and variances among samples, which is essential for the feature extraction in a given task to discern task-specific discriminative regions. As such, the computation complexity is reduced to _O_ ( _L_ ( _NK_ + _M_ )[2] + ( _NK_ + _M_ ) _L_[2] ). See Figure 5 in Appendix J for an illustration. Though this decoupling shares the spirit with video transformers (Ho et al., 2019; Bertasius et al., 2021), it is crucial to highlight that our consideration of the sample-to-sample relationship in few-shot learning presents a unique challenge distinct from the frame-to-frame relationship in videos, i.e., query samples have to be differentiated from support ones. This challenge motivates the following introduction of sample causal masks. 

As shown in Figure 1(c), we introduce our Masked Sample Attention (MSA) with label infusion and the designed causal masking mechanism to further enforce consistency in the attended features across samples within a task. We first get the embedded support category information **W** _[c] y ∈_ R[1] _[×][d]_ via the linear projection matrix **W** _[c]_ , which is infused to support 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning_images/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning.pdf-0004-05.png)


**----- Start of picture text -----**<br>
(a) Inductive Sample Causal Mask (b) Autoregressive Sample Causal Mask<br>Activated Sample Interaction Masked Sample Interaction<br>Support Support<br>Query Query<br>**----- End of picture text -----**<br>


_Figure 2._ Sample Causal Attenntion Mask. An example with _N_ = 4 and _K_ = 1. (a) Inductive sample causal mask for withinsupport and support-query sample correspondence learning. (b) Autoregressive sample causal mask for extra query-query sample correspondence learning. 

tokens through the elementwise addition. For the obtained sample attention **A** _S_ , we maintain the sample causal mask **H** _∈_ R[(] _[NK]_[+] _[M]_[)] _[×]_[(] _[NK]_[+] _[M]_[)] to restrict the sample interaction patterns as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning_images/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning.pdf-0004-08.png)


where _⊙_ is the element-wise product. 

Through the constraint, support samples can attend themselves to strengthen intra- and inter-class discriminative clues, which query samples utilize for task-specific feature consistency learning. Note that this mask mechanism also makes our method comply with the inductive protocols. In the autoregressive scenario, we also extend our MSA with the autoregressive causally-masked sample attention 

4 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

to embed the query-query interactions into the vision transformer. Figure 2(b) shows an example mask with _N_ = 4 and _K_ = 1. Query samples attend to support samples and earlier predicted queries in an autoregressive fashion, which thus serves to implicitly expand the support set for subsequent query predictions. 

## **3.3. Patch-grained Task Attention** 

In this section, we introduce details of the proposed Patchgrained Task Attention (PTA), as illustrated in Figure 1(d). PTA aims to transfer previous task knowledge for regularizing the adaptation in new tasks. To this end, PTA learns a task-specific probe vector summarizing discriminative patch-level patterns and maintains a knowledge pool consolidated during meta-training to organize learned knowledge. When a new task comes, it leverages attention-based mechanisms to tap into and aggregate relevant knowledge from the knowledge pool. We elaborate on four key components as follows: task probe vector aggregation, knowledge retrieval, pool consolidation, and knowledge fusion. 

**Task Probe Vector Aggregation.** Given a task consisting of support and query sets, we first gather the task information with learnable task probe vectors **g** _∈_ R _[T][ ×][d]_ , which are computed along with support patch tokens **X** _C ∈_ R _[NK][×][L][×][d]_ to aggregate the key parts of samples and the whole task representations. Specifically, we perform the task aggregation using attention as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning_images/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning.pdf-0005-05.png)


where **Qg** is query embedding of task probe vectors; **KX** _C_ and **VX** _C_ are key and value embeddings of support patch tokens, respectively. This allows task probe vectors to focus on relevant task-specific feature patterns and ignore irrelevant semantics of each sample. 

**Knowledge Retrieval.** After gathering the task information, we retrieve relevant knowledge using a simple weighted summation strategy from the knowledge pool **p** _∈_ R _[Z][×][d]_ with _Z_ components (which will be introduced below). The retrieval is formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning_images/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning.pdf-0005-08.png)


where _γ_ is the score function based on cosine similarity between task probe vectors and pool components. **r** _∈_ R _[T][ ×][d]_ is the retrieved historical knowledge that can be thought of as key feature semantics (e.g., ears and eyes of the dog) related to the current task samples. 

**Pool Consolidation.** During meta-training, we maintain a knowledge pool **p** updated by every sequentially coming task. To consolidate the learned knowledge in the pool, we select relevant components from the pool and integrate them 

with new information brought by the current task as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning_images/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning.pdf-0005-12.png)


where _s_ = argmax _γ_ � **g** _j[′][,]_ **[ p]** � representing the indices of the component most similar to _j_ -th task probe vector. This method also allows us to control the pool size and the memory consumption. 

**Knowledge Fusion.** To regularize the adaptation of new tasks with historical knowledge, we deliver the union of original task vectors **g** and retrieved knowledge **r** to enhance the support patch token representations via the attention mechanism as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning_images/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning.pdf-0005-15.png)


where **QX** _C_ is query embedding of support patch tokens and **K** [ **g** ; **r** ] and **V** [ **g** ; **r** ] are key and value embeddings of regularized task-specific semantics, respectively. The intuition is to leverage well-learned feature semantics in previous similar tasks to strengthen discriminative regions in new tasks. 

## **3.4. Meta-training and Inference with MetaFormer** 

**Meta-training.** Using MSA and PTA as basic building blocks working in conjunction with original ViT modules, we propose a new vision transformer _fθ_ with holistic attention, named MetaFormer, encapsulating three distinct relationships between patch tokens, samples and tasks at different semantic levels, to extract rich task-specific feature representations. For the inductive protocol (Vinyals et al., 2016b), MetaFormer configures the sample causal mask as the inductive variant, which, coupled with the inherent Layer Normalization in vision transformers, ensures independent predictions for each query sample. Built upon feature embeddings extracted by MetaFormer, we estimate the class patch prototypes by averaging support patch tokens per class _pk_ = _|S_ 1 _[k] |_ � _x∈S[k][ f][θ]_[ (] _[x]_[)][.][Query samples are] predicted based on patch-wise cosine similarity with prototypes (Lai et al., 2022; Hiller et al., 2022). The probability of _k[th]_ category is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning_images/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning.pdf-0005-19.png)


where _d_ indicates the cosine distance and _τ_ is scaling temperature. The cross-entropy loss function with the few-shot label _yt_ is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning_images/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning.pdf-0005-21.png)


We also introduce the autoregressive setting from regression tasks (Nguyen & Grover, 2022; Bruinsma et al., 2023) to classification scenarios, seamlessly switching with an 

5 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

autoregressive sample causal mask to allow interactions between subsequent query samples and those predicted earlier in a single pass. The support prototypes are then enriched by feeding previously predicted queries as the auxiliary support set _S_[˜] with predicted probability belonging to class _k_ . We employ _P_ (ˆ _yt_ = _k | xt_ ) as the sample weights to compute auxiliary prototypes through a weighted average _p_ types are updated by the mean ofˆ _k_ = � _x∈S_[˜] _[k]_ 1 _[P]_[ (] _[k][|][x]_[)] � _x∈S_[˜] _[k][ P]_[(] _[k] p[|] k[x]_[)] and _[f][θ]_[(] _[x] p_ ˆ[)] _k_[.] . Given mod[New][proto-] eling dependencies between all _M_ query samples requires _M_ prototype updates, we propose managing updates with a sampling size of _r_ queries at a time for faster and more consistent prototype refinement. 

**Inference.** MetaFormer extracts support and query feature embeddings in a single feedforward pass during inference, configuring the inductive and autoregressive sample causal masks for inductive and autoregressive settings, respectively. For autoregressive inference, we adopt an inference-time augmentation similar to prior works (Bendou et al., 2022; Zhu & Koniusz, 2023) while differing by shuffling the order of the samples in a meta-testing task _s_ times and then computing the most confident logits as the final prediction. 

settings, respectively, showcasing the remarkable effectiveness of our holistic attention mechanism in harnessing the full potential of transformers for meta-learning. Additionally, MetaFormer works synergistically with self-distillation training methods (He et al., 2022b; Lin et al., 2023) to enhance task-specific feature embedding and overall performance. Results on CIFAR-FS and FC100, presented in Table 2, further validate the superiority of MetaFormer-I. As illustrated in Table 3, MetaFormer demonstrates computational efficiency over FewTURE and SMKD, primarily by eliminating the need for inference-time tuning and reducing the large resolution requirements. It is noteworthy that MetaFormer facilitates full sample interactions, as opposed to FewTURE, which focuses solely on contextual relationships. Detailed evaluations of computational cost are provided in Appendix H. Seamless transitioning from the inductive setting with the autoregressive sample causal mask, MetaFormer-A outperforms protoLP (Zhu & Koniusz, 2023) in the majority of cases by a clear margin and stands out as the first pure transformer-backed method for transductive few-shot image classification. For more comprehensive comparison results and discussions, refer to Appendix A.3. 

## **4.2. Broader Study of Few-Shot Learning** 

## **4. Experiments** 

## **4.1. Standard Few-Shot Learning** 

**Datasets.** We train and evaluate our MetaFormer on the four standard few-shot benchmarks: _mini_ ImageNet (Vinyals et al., 2016b), _tiered_ ImageNet (Ren et al., 2018b), CIFARFS (Bertinetto et al., 2019) and FC-100 (Oreshkin et al., 2018). In all experiments, we follow the standard data usage specifications same as Hiller et al. (2022), splitting data into the meta-training set, meta-validation set, and meta-test set, and classes in each set are mutually exclusive. The details of each dataset are described in Appendix A.1. 

**Implementation Details.** We train our method in two stages following Hiller et al. (2022): self-supvervised pretraining and meta-tuning. We first pre-train our vision transformer backbone (Dosovitskiy et al., 2020; Liu et al., 2021) utilizing a self-supervised training objective (Zhou et al., 2022a). Subsequently, we integrate our proposed MSA and PTA into the vision transformer for meta-learning. We denote our framework as MetaFormer-I in the inductive setting and MetaFormer-A in the autoregressive setting. See Appendix A.2 for more training and evaluation details. 

**Comparison with the State-of-the-art.** The comparison results with related or recent state-of-the-art (SOTA) methods on _mini_ ImageNet and _tiered_ ImageNet is shown in Table 1. Our method significantly outperforms previous SOTA meta-learning approaches. For instance, on _mini_ ImageNet, MetaFormer-I exceeds FewTURE (Hiller et al., 2022) by **7** _._ **76** % and **5** _._ **51** % in 1-shot and 5-shot 

**Cross-Domain and Multi-Domain Few-shot Classifica-** 

**tion.** To further investigate the fast adaptation ability of our method, we evaluate the MetaFormer in more challenging cross-domain (Chen et al., 2019; Oh et al., 2022) and multi-domain (Triantafillou et al., 2020) scenarios, containing dual category and domain shifts. Details on benchmarks and implementation are available in Appendix B and Appendix C. We evaluate MetaFomer, meta-trained on _mini_ ImageNet, on cross-domain few-shot classification benchmarks in Table 4 (See more results and discussions in Appendix B.3). MetaFormer demonstrates remarkable task adaptability, surpassing previous in-domain SOTA meta-learning methods FewTURE (Hiller et al., 2022) and PMF (Hu et al., 2022) with a significant improvement of up to **8** _._ **77** % and **16** _._ **86** %, highlighting its effectiveness in bridging domain gaps. MetaFormer also improves the transfer learning method SMKD (Lin et al., 2023) by up to **2** _._ **57** %. Furthermore, in Table 14 (Appendix C.3), we assess MetaFormer on the large-scale and challenging MetaDataset (Triantafillou et al., 2020), where it outperforms PMF (Hu et al., 2022) in handling tasks with diverse distributions. We attribute such impressive improvement to our proposed holistic attention mechanism, which not only facilitates sample correspondence learning but also enables knowledge reuse, thus aiding task adaptation to obtain more discriminative feature representations for each task. 

**Compatibility with Other Backbones and Foundation** 

**Models.** Different from isotropic vision transformers, hierarchical models (Liu et al., 2021) bring greater efficiency 

6 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

_Table 1._ **Average classification accuracy (%) for** 5 **-way** 1 **-shot and** 5 **-way** 5 **-shot scenarios.** Reported are the mean and 95% confidence interval on the unseen test sets of _mini_ ImageNet (Vinyals et al., 2016a) and _tiered_ ImageNet (Ren et al., 2018a). * denotes results reported by us. More comprehensive results are shown in the appendix. 

|**Method**|**Setting**|**Backbone**<br>**# Params**|**_mini_ImageNet**<br>**1-shot**<br>**5-shot**|**_tiered_ImageNet**<br>**1-shot**<br>**5-shot**|
|---|---|---|---|---|
|FEAT (Ye et al.,2020)<br>Inductive<br>_ResNet-12_<br>14.1 M<br>66_._78_±_0_._20<br>82_._05_±_0_._14<br>70_._80_±_0_._23<br>84_._79_±_0_._16<br>DeepEMD (Zhang et al.,2020a)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>65_._91_±_0_._82<br>82_._41_±_0_._56<br>71_._16_±_0_._87<br>86_._03_±_0_._58<br>RENet (Kang et al.,2021)<br>Inductive<br>_ResNet-12_<br>12.6 M<br>67_._60_±_0_._44<br>82_._58_±_0_._30<br>71_._61_±_0_._51<br>85_._28_±_0_._35<br>COSOC(Luo et al.,2021)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>69_._28_±_0_._49<br>85_._16_±_0_._42<br>73_._57_±_0_._43<br>87_._57_±_0_._10|||||
|LEO (Rusu et al.,2018)<br>MetaQDA(Zhanget al.,2021c)|Inductive<br>Inductive|_WRN-28-10_<br>36.8 M<br>_WRN-28-10_<br>36.5 M|61_._76_±_0_._08<br>77_._59_±_0_._12<br>67_._83_±_0_._64<br>84_._28_±_0_._69|66_._33_±_0_._05<br>81_._44_±_0_._09<br>74_._33_±_0_._65<br>89_._56_±_0_._79|
|SUN (Dong et al.,2022)<br>FewTURE (Hiller et al.,2022)<br>PMF_∗_(Hu et al.,2022)|Inductive<br>Inductive<br>Inductive|_ViT_<br>12.5 M<br>_ViT-Small_<br>22 M<br>_ViT-Small_<br>21 M|67_._80_±_0_._45<br>83_._25_±_0_._30<br>68_._02_±_0_._88<br>84_._51_±_0_._53<br>71_._01_±_0_._81<br>86_._03_±_0_._53|72_._99_±_0_._50<br>86_._74_±_0_._33<br>72_._96_±_0_._92<br>86_._43_±_0_._67<br>76_._61_±_0_._89<br>89_._66_±_0_._54|
|MetaFormer-I(Ours)|Inductive|_ViT-Small_<br>24.5 M|**75**_._**78**_±_**0**_._**71**<br>**90**_._**02**_±_**0**_._**44**|**79**_._**05**_±_**0**_._**81**<br>**90**_._**40**_±_**0**_._**53**|
|HCTransformers (He et al.,2022b)<br>SMKD (Lin et al.,2023)|Inductive<br>Inductive|3_×ViT-Small_<br>63 M<br>_ViT-Small_<br>21 M|74_._74_±_0_._17<br>89_._19_±_0_._13<br>74_._28_±_0_._18<br>88_._89_±_0_._09|79_._67_±_0_._20<br>91_._72_±_0_._11<br>78_._83_±_0_._20<br>91_._21_±_0_._11|
|SMKD + MetaFormer-I(Ours)|Inductive|_ViT-Small_<br>24.5 M|**76**_._**54**_±_**0**_._**73**<br>**90**_._**76**_±_**0**_._**41**|**80**_._**57**_±_**0**_._**82**<br>**92**_._**42**_±_**0**_._**49**|
|EASY (Bendou et al.,2022)<br>protoLP (Zhu & Koniusz,2023)|Transductive<br>Transductive|_3×ResNet-12_<br>37.2 M<br>_WRN-28-10_<br>36.5 M|84_._04_±_0_._23<br>89_._14_±_0_._11<br>84_._32_±_0_._21<br>90_._02_±_0_._12|84_._29_±_0_._24<br>89_._76_±_0_._14<br>**89**_._**65**_±_**0**_._**22**<br>93_._21_±_0_._13|
|MetaFormer-A(Ours)|Transductive|_ViT-Small_<br>24.5 M|**84**_._**78**_±_**0**_._**79**<br>**91**_._**39**_±_**0**_._**42**|88_._38_±_0_._78<br>**93**_._**37**_±_**0**_._**45**|



_Table 2._ **Average classification accuracy (%) for** 5 **-way** 1 **-shot and** 5 **-way** 5 **-shot scenarios.** Reported are the mean and 95% confidence interval on the unseen test sets of CIFAR-FS (Bertinetto et al., 2019) and FC100 (Oreshkin et al., 2018). * denotes results reported by us. More comprehensive results are shown in the appendix. 

|**Method**|**Setting**|**Backbone**<br>**# Params**|**CIFAR-FS**<br>**1-shot**<br>**5-shot**|**FC100**<br>**1-shot**<br>**5-shot**|
|---|---|---|---|---|
|MetaOpt (Lee et al.,2019)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>72_._00_±_0_._70<br>84_._20_±_0_._50<br>41_._10_±_0_._60<br>55_._50_±_0_._60<br>RFS (Tian et al.,2020)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>73_._90_±_0_._80<br>86_._90_±_0_._50<br>44_._60_±_0_._70<br>60_._90_±_0_._60<br>BML (Zhou et al.,2021b)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>73_._45_±_0_._47<br>88_._04_±_0_._33<br>45_._00_±_0_._41<br>63_._03_±_0_._41<br>TPMN(Wu et al.,2021)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>75_._50_±_0_._90<br>87_._20_±_0_._60<br>46_._93_±_0_._71<br>63_._26_±_0_._74|||||
|PSST (Chen et al.,2021b)<br>Meta-QDA(Zhanget al.,2021c)|Inductive<br>Inductive|_WRN-28-10_<br>36.5 M<br>_WRN-28-10_<br>36.5 M|77_._02_±_0_._38<br>88_._45_±_0_._35<br>75_._83_±_0_._88<br>88_._79_±_0_._75|-<br>-<br>-<br>-|
|SUN (Dong et al.,2022)<br>FewTURE (Hiller et al.,2022)<br>PMF_∗_(Hu et al.,2022)|Inductive<br>Inductive<br>Inductive|_ViT_<br>12.5M<br>_ViT-Small_<br>22 M<br>_ViT-Small_<br>21 M|78_._37_±_0_._46<br>88_._84_±_0_._32<br>76_._10_±_0_._88<br>86_._14_±_0_._64<br>76_._70_±_0_._90<br>87_._61_±_0_._60|-<br>-<br>46_._20_±_0_._79<br>63_._14_±_0_._73<br>45_._79_±_0_._73<br>63_._64_±_0_._72|
|MetaFormer-I(Ours)|Inductive|_ViT-Small_<br>24.5 M|**80**_._**16**_±_**0**_._**76**<br>**90**_._**57**_±_**0**_._**55**|**51**_._**14**_±_**0**_._**71**<br>**68**_._**33**_±_**0**_._**74**|
|HCTransformers (He et al.,2022b)<br>SMKD (Lin et al.,2023)|Inductive<br>Inductive|3_×ViT-Small_<br>63 M<br>_ViT-Small_<br>21M|78_._89_±_0_._18<br>90_._50_±_0_._09<br>80_._08_±_0_._18<br>90_._91_±_0_._13|48_._27_±_0_._15<br>66_._42_±_0_._16<br>50_._38_±_0_._16<br>68_._50_±_0_._16|
|SMKD + MetaFormer-I(Ours)|Inductive|_ViT-Small_<br>24.5 M|**81**_._**49**_±_**0**_._**74**<br>**91**_._**91**_±_**0**_._**54**|**52**_._**18**_±_**0**_._**78**<br>**71**_._**29**_±_**0**_._**73**|
|EASY (Bendou et al.,2022)<br>protoLP (Zhu & Koniusz,2023)|Transductive<br>Transductive|_3×ResNet-12_<br>37.2 M<br>_WRN-28-10_<br>36.5 M|87_._16_±_0_._21<br>90_._47_±_0_._15<br>87_._69_±_0_._23<br>90_._82_±_0_._15|54_._13_±_0_._24<br>66_._86_±_0_._19<br>_−_<br>_−_|
|MetaFormer-A(Ours)|Transductive|_ViT-Small_<br>24.5 M|**88**_._**34**_±_**0**_._**76**<br>**92**_._**21**_±_**0**_._**59**|**58**_._**04**_±_**0**_._**99**<br>**70**_._**80**_±_**0**_._**76**|



_Table 3._ Inference efficiency comparison with existing methods on the _mini_ -ImageNet. 

|_mini_-ImageNet.|||
|---|---|---|
|**Method**|**GLOPs**|**Inference time**[ms]<br>**1-shot**<br>**5-shot**|
|FewTURE<br>5.01<br>77_._35_±_0_._47<br>111_._22_±_1_._27<br>SMKD<br>12.58<br>137_._58_±_0_._66<br>171_._37_±_0_._78|||
|MetaFormer-I|4.88|**67**_._**65**_±_**0**_._**78**<br>**105**_._**72**_±_**1**_._**06**|



and capture multi-scale information. Our evaluation of the MetaFormer on Swin-Transformer (Liu et al., 2021) (see Table15 in Appendix E) reveals consistent improvements and underscores its broad applicability. Pre-trained vision foundation models demonstrate impressive zero-shot image classification performance (Radford et al., 2021). Recent 

works have shown that CLIP’s performance on downstream tasks can be further enhanced by utilizing few-shot data and techniques (Zhang et al., 2021b; Zhou et al., 2022b; Zhu et al., 2023). These approaches have shown promising improvements over frozen models like Zero-shot CLIP. To further investigate the adaptation ability of our proposed method, we adapt our method to CLIP model with ViT-B/16 for advancing its performance in downstream tasks. Implementation details are available in Appendix D. As shown in Table 5, CLIP pre-trained on large-scale web-crawled image-text pairs struggles with downstream datasets exhibiting a large domain gap, such as the medical dataset of ISIC. Adapting CLIP with few labeled samples is pivotal to guarantee better performance, though naively increasing the 

7 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

_Table 4._ **Broader study of cross-domain few-shot learning.** Average classification accuracy (%) for 5-way 1-shot scenario when meta-learning on _mini_ ImageNet (Vinyals et al., 2016a) but meta-testing on cross-domain few-shot benchmarks. 

|**Method**|**CUB**<br>**Cars**<br>**Places**<br>**Plantae**<br>**CropDisease**<br>**EuroSAT**<br>**ISIC**<br>**ChestX**|
|---|---|
|PMF (Hu et al.,2022)<br>40_._12_±_0_._74<br>33_._91_±_0_._64<br>60_._84_±_0_._88<br>43_._78_±_0_._77<br>72_._62_±_0_._86<br>64_._24_±_0_._80<br>31_._00_±_0_._58<br>22_._40_±_0_._44<br>FewTURE (Hiller et al.,2022)<br>48_._21_±_0_._83<br>33_._97_±_0_._63<br>58_._74_±_0_._91<br>43_._31_±_0_._76<br>68_._22_±_0_._88<br>61_._77_±_0_._81<br>28_._67_±_0_._56<br>22_._60_±_0_._44||
|MetaFormer-I(Ours)|**56**_._**98**_±_**0**_._**93**<br>**37**_._**32**_±_**0**_._**66**<br>**61**_._**90**_±_**0**_._**89**<br>**49**_._**30**_±_**0**_._**82**<br>**74**_._**48**_±_**0**_._**82**<br>**68**_._**23**_±_**0**_._**79**<br>**33**_._**22**_±_**0**_._**57**<br>**23**_._**02**_±_**0**_._**42**|
|SMKD (Lin et al.,2023)|54_._64_±_0_._84<br>34_._30_±_0_._64<br>62_._75_±_0_._92<br>45_._57_±_0_._81<br>75_._99_±_0_._82<br>68_._58_±_0_._77<br>33_._92_±_0_._62<br>22_._59_±_0_._41|
|SMKD+MetaFormer-I(Ours)|**57**_._**21**_±_**0**_._**88**<br>**35**_._**38**_±_**0**_._**64**<br>**62**_._**89**_±_**0**_._**85**<br>**46**_._**13**_±_**0**_._**80**<br>**76**_._**06**_±_**0**_._**79**<br>**70**_._**04**_±_**0**_._**82**<br>**34**_._**69**_±_**0**_._**62**<br>**22**_._**65**_±_**0**_._**40**|
|protoLP (Zhu & Koniusz,2023)|**69**_._**94**_±_**1**_._**23**<br>35_._50_±_0_._93<br>67_._44_±_1_._31<br>44_._26_±_1_._15<br>87_._13_±_1_._10<br>77_._89_±_1_._20<br>33_._00_±_0_._78<br>21_._70_±_0_._44|
|MetaFormer-A (Ours)|67_._39_±_0_._98<br>**35**_._**68**_±_**0**_._**74**<br>**70**_._**78**_±_**1**_._**05**<br>**50**_._**51**_±_**0**_._**98**<br>**87**_._**36**_±_**0**_._**84**<br>**79**_._**44**_±_**0**_._**87**<br>**36**_._**38**_±_**0**_._**72**<br>**22**_._**91**_±_**0**_._**41**|



number of parameters to adapt even incurs overfitting. Our method significantly enhances Zero-shot CLIP on EuroSAT by **42** _._ **76** % and ISIC by **43** _._ **81** %, and its adaptation ability also surpasses Tip-Adapter by a large margin. 

_Table 5._ **Classification results with foundation model for 1-shot EuroSAT and ISIC** . Reported are the mean and 95% confidence interval on the test set. ViT-B/16 with the patch size 16 _×_ 16 is adopted for the vision branch in all methods. 

|**Method**|**EuroSAT**|**ISIC**|
|---|---|---|
|Zero-shot CLIP<br>48_._73_±_0_._98<br>21_._07_±_0_._76<br>Tip-Adapter<br>69_._85_±_0_._75<br>28_._70_±_0_._97<br>Tip-Adapter-F<br>72_._01_±_0_._97<br>32_._27_±_1_._11<br>Tip-Adapter-F with more layers<br>51_._95_±_0_._86<br>16_._17_±_0_._78<br>Tip-Adapter-F+MetaFormer-I (Ours)<br>**88**_._**83**_±_**0**_._**78**<br>**45**_._**96**_±_**1**_._**44**|||



## **4.3. Ablation study** 

**Component Analysis.** In this section, we investigate the individual contributions of key components in MetaFormer. The impact on performance, along with the associated increase in the number of additional learnable parameters, are detailed in Table 6, demonstrating that both components bolster performance with only a modest increase in computational overhead. We address the potential concern that observed performance gains could be attributed solely to an increase in parameters in Appendix H.3 Table 19, where we present a detailed parameter comparison to underscore that the remarkable enhancements stem from the components’ inherent design. The effectiveness of the proposed adaptive sample causal masks is further evaluated in Appendix F Table 16, revealing that a lack of effective constraints between support and query samples (see Appendix H.3 Figure 4 for details of the ablated masks of within-support and supportquery) results in suboptimal performance. We demonstrate in Appendix G Table 17 that PTA exhibits superior design and performance. 

_Table 6._ Component ablation studies and the number of additional learnable parameters on _mini_ ImageNet. 

|**MSA**<br>**PTA**|**Add. Params.**|**_mini_ImageNet**<br>**1-shot**<br>**5-shot**|
|---|---|---|
|✓<br>✓<br>+3_._57M<br>75_._78_±_0_._71<br>90_._02_±_0_._44|||
|✓<br>✗<br>✗<br>✓|+2_._01M<br>+1_._56M|74_._64_±_0_._76<br>87_._53_±_0_._47<br>73_._63_±_0_._75<br>87_._76_±_0_._52|



**Decoupled Mechanism** . The self-attention in traditional ViTs allows each patch token to interact with all other patch locations, especially in deeper layers. Thus, each patch token processes and stores non-local information, which is the key for vision transformers to handle potential spatial misalignment between different images. These intuitions are supported by the results we have obtained in Table 7. We here resize the image resolution to 192 _×_ 192 due to the higher memory footprint of the joint attention. The results demonstrate our decoupled approximation achieves comparable performance while offering a significant reduction in computational costs. This finding supports our hypothesis that the decoupled attention mechanism can effectively leverage cross-sample information, even in the presence of misalignment between images. We conclude that our decoupled patch-sample attention achieves a trade-off between accuracy and efficiency for capturing task-specific sample interactions. 

_Table 7._ Comparison results of joint and decoupled sample-patch attention mechanisms on _mini_ ImageNet. 

|tention mechanisms on_mini_ImageNet.||
|---|---|
|**Method**|**mini 1-shot**|
|||
|Joint Sample-Patch Attention<br>Decoupled Sample-Patch Attention|73_._35_±_0_._75<br>73_._21_±_0_._76|



**Distance Metrics** . We extend our analysis by incorporating a comparative study with the regularized Mahalanobis distance (Bateni et al., 2022) as a representative of Euclideanbased metrics, alongside our initial approach using cosine similarity. In Table 8, we observe that temperature scaling plays a pivotal role in modulating the interaction with the softmax, and scaled cosine similarity performs at par with the Euclidean distance, which aligns with the effects of metric scaling (Oreshkin et al., 2018). Furthermore, our analysis shows the resilience of cosine similarity against the intervention of noisy patch information for computing local pairwise distances. 

**Class Embeddings** . Unlike previous meta-learning methods using class embeddings for better modeling the taskspecific distribution (Rusu et al., 2018; Xu et al., 2020; Zhmoginov et al., 2022), our approach incorporates sample label information into the feature extraction process to capture nuanced distinctions between classes. The results 

8 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

_Table 8._ Comparison results of different metrics on _mini_ ImageNet. Temp.: Temperature Scaling, Patch.: Patch-wise distance. 

|**Method**|**Temp.**<br>**Patch.**|**mini 1-shot**|
|---|---|---|
|Cosine Distance<br>✗<br>✗<br>64_._22_±_0_._83<br>✓<br>✗<br>74_._14_±_0_._73<br>✓<br>✓<br>75_._78_±_0_._71|||
|Mahalanobis Distance|✗<br>✗<br>✓<br>✗<br>✓<br>✓|72_._93_±_0_._74<br>73_._09_±_0_._75<br>71_._97_±_0_._76|



in Table 9 underscore that introducing class embeddings via either concatenation or summation facilitates differentiation between classes and further strengthens task-specific discriminative clues with the guidance of a sample causal mask. 

_Table 9._ Different label infusion methods with ViT-Small on _mini_ ImageNet. 

|ageNet.|||
|---|---|---|
|**Method**|**Backbone**|**mini 1-shot**|
||||
|w/o. label<br>concatenation<br>summation|_ViT-Small_<br>_ViT-Small_<br>_ViT-Small_|73.70<br>74.59<br>74.64|



**Inference-time Augmentation Strategies** . In transductive few-shot classification tasks, our model MetaFormer-A incorporates an inference-time augmentation strategy by reshuffling the sample order, while prior state-of-the-art methods involve evaluating multiple random crops per sample and subsequently averaging the features for the final prediction (Bendou et al., 2022; Qi et al., 2021; Zhu & Koniusz, 2023). We evaluate the impact of different augmentation strategies in Table 10. The results showcase the unique and synergistic role of reshuffling in enhancing the autoregressive and sequential nature of our MetaFormer-A. 

_Table 10._ Comparison of different inference-time augmentation strategies. We report 1-shot accuracy on CIFAR-FS. Crop.: Random Cropping 

|dom Cropping||||
|---|---|---|---|
|**Method**|**Backbone**|**Crop.**<br>**Reshuffing**|**1-shot**|
|protoLP<br>_WRN-28-10_<br>✓<br>✗<br>87_._69_±_0_._23<br>protoLP<br>_WRN-28-10_<br>✓<br>✓<br>87_._69_±_0_._23<br>MetaFormer-A<br>_ViT-Small_<br>✗<br>✗<br>85_._76_±_0_._77<br>MetaFormer-A<br>_ViT-Small_<br>✗<br>✓<br>88_._34_±_0_._76||||



## **4.4. Qualitative Analysis** 

Figure 3 presents visualizations of our holistic attention mechanism, with columns depicting the attention maps of three distinct attention modules. The results reveal that the sample correspondence learning guided by spatial and sample attention modules suppresses irrelevant regions by exploiting pattern relations within and across samples, leading to the extraction of more discriminative, task-specific features. Furthermore, the task attention module adeptly transfers semantic knowledge from previous tasks to new 

ones, with a particular focus on the critical components of foreground objects. When integrated with intra- and intertask attention, holistic attention yields a more precise and comprehensive response map, predominantly concentrated on the foreground region. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning_images/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning.pdf-0009-12.png)


**----- Start of picture text -----**<br>
Input Images Patch Attention Sample Attention Task Attention Holistic Attention<br>Malamute<br>Boxer<br>Golden Retriever<br>**----- End of picture text -----**<br>


_Figure 3._ Response visualization for MetaFormer with holistic attention. 

## **5. Conclusions** 

This paper proposes MetaFormer, a novel Vit-backed metalearning approach for few-shot classification. MetaFormer capitalizes on the transformer architecture to orchestrate holistic attention, integrating two lightweight modules to capture intra-task and inter-task interactions. Through Masked Sample Attention (MSA), MetaFormer promotes sample consistency for adapting task-specific discriminative feature representations. Meanwhile, Patch-grained Task Attention (PTA) leverages a dynamic knowledge pool to infuse relevant historical knowledge into current task adaptation. Configuring different adaptive masks, MetaFormer supports both inductive and transductive few-shot learning protocols. Extensive experiments demonstrate the superiority of MetaFormer across standard in-domain, cross-domain, and multi-domain benchmarks. 

## **Impact Statement** 

**Broader Impact.** Our proposed meta-learning method integrates seamlessly with recent state-of-the-art pre-trained vision transformers and foundation models, significantly enhancing their few-shot classification capabilities, and hence has the potential to advance the field of machine learning. The deployment of such advanced machine learning models in real-world scenarios must be governed by careful consideration of privacy, fairness, and transparency to ensure responsible use. 

**Limitations.** The current study primarily focuses on fewshot image classification, leaving the exploration of our method’s applicability to detection and segmentation tasks for future work. 

9 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

## **References** 

- Achille, A., Lam, M., Tewari, R., Ravichandran, A., Maji, S., Fowlkes, C. C., Soatto, S., and Perona, P. Task2vec: Task embedding for meta-learning. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pp. 6430–6439, 2019. 

- Afrasiyabi, A., Lalonde, J.-F., and Gagne, C.´ Mixture-based feature space learning for few-shot image classification. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 9041–9051, 2021. 

- Afrasiyabi, A., Larochelle, H., Lalonde, J.-F., and Gagne, C.´ Matching feature sets for few-shot image classification. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 9014–9024, June 2022. 

- Bao, H., Dong, L., Piao, S., and Wei, F. BEit: BERT pre-training of image transformers. In _International Conference on Learning Representations_ , 2022. 

- Bateni, P., Goyal, R., Masrani, V., Wood, F., and Sigal, L. Improved few-shot visual classification. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 14493–14502, 2020. 

- Bateni, P., Barber, J., Goyal, R., Masrani, V., van de Meent, J.-W., Sigal, L., and Wood, F. Beyond simple metalearning: Multi-purpose models for multi-domain, active and continual few-shot learning. _arXiv preprint arXiv:2201.05151_ , 2022. 

- Bendou, Y., Hu, Y., Lafargue, R., Lioi, G., Pasdeloup, B., Pateux, S., and Gripon, V. Easy—ensemble augmentedshot-y-shaped learning: State-of-the-art few-shot classification with simple components. _Journal of Imaging_ , 8 (7):179, 2022. 

- Bertasius, G., Wang, H., and Torresani, L. Is space-time attention all you need for video understanding? In _ICML_ , volume 2, pp. 4, 2021. 

- Bertinetto, L., Henriques, J. F., Torr, P., and Vedaldi, A. Meta-learning with differentiable closed-form solvers. In _International Conference on Learning Representations_ , 2019. 

- Bruinsma, W. P., Markou, S., Requeima, J., Foong, A. Y. K., Andersson, T. R., Vaughan, A., Buonomo, A., Hosking, J. S., and Turner, R. E. Autoregressive conditional neural processes. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ , 2023. 

- Caron, M., Touvron, H., Misra, I., Jegou,´ H., Mairal, J., Bojanowski, P., and Joulin, A. Emerging properties in 

- self-supervised vision transformers. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 9650–9660, 2021. 

- Chen, W.-Y., Liu, Y.-C., Kira, Z., Wang, Y.-C., and Huang, J.-B. A closer look at few-shot classification. In _International Conference on Learning Representations_ , 2019. 

- Chen, Y., Liu, Z., Xu, H., Darrell, T., and Wang, X. Metabaseline: Exploring simple meta-learning for few-shot learning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 9062–9071, 2021a. 

- Chen, Z., Ge, J., Zhan, H., Huang, S., and Wang, D. Pareto self-supervised training for few-shot learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 13663–13672, 2021b. 

- Doersch, C., Gupta, A., and Zisserman, A. Crosstransformers: spatially-aware few-shot transfer. _Advances in Neural Information Processing Systems_ , 33:21981–21993, 2020. 

- Dong, B., Zhou, P., Yan, S., and Zuo, W. Self-promoted supervision for few-shot transformer. In _European Conference on Computer Vision_ , pp. 329–347, 2022. 

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv preprint arXiv:2010.11929_ , 2020. 

- Douillard, A., Rame, A., Couairon, G., and Cord, M.´ Dytox: Transformers for continual learning with dynamic token expansion. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 9285– 9295, 2022. 

- Fei, N., Lu, Z., Xiang, T., and Huang, S. Melr: Metalearning via modeling episode-level relationships for fewshot learning. In _International Conference on Learning Representations_ , 2020. 

- Finn, C., Abbeel, P., and Levine, S. Model-agnostic metalearning for fast adaptation of deep networks. In _International Conference on Machine Learning_ , pp. 1126–1135, 2017. 

- Gao, Z., Wu, Y., Jia, Y., and Harandi, M. Curvature generation in curved spaces for few-shot learning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 8691–8700, 2021. 

- Gidaris, S., Bursuc, A., Komodakis, N., Perez,´ P., and Cord, M. Boosting few-shot visual learning with selfsupervision. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 8059–8068, 2019. 

10 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

- Guo, Y., Codella, N. C., Karlinsky, L., Codella, J. V., Smith, J. R., Saenko, K., Rosing, T., and Feris, R. A broader study of cross-domain few-shot learning. In _European Conference on Computer Vision_ , pp. 124–141, 2020. 

- He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 770–778, 2016. 

- He, K., Chen, X., Xie, S., Li, Y., Dollar, P., and Girshick,´ R. Masked autoencoders are scalable vision learners. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 16000–16009, 2022a. 

- He, Y., Liang, W., Zhao, D., Zhou, H.-Y., Ge, W., Yu, Y., and Zhang, W. Attribute surrogates learning and spectral tokens pooling in transformers for few-shot learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 9119–9129, 2022b. 

- Helber, P., Bischke, B., Dengel, A., and Borth, D. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. _IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing_ , 12(7):2217–2226, 2019. 

- Hiller, M., Ma, R., Harandi, M., and Drummond, T. Rethinking generalization in few-shot classification. _arXiv preprint arXiv:2206.07267_ , 2022. 

- Ho, J., Kalchbrenner, N., Weissenborn, D., and Salimans, T. Axial attention in multidimensional transformers. _arXiv preprint arXiv:1912.12180_ , 2019. 

- Hou, R., Chang, H., Ma, B., Shan, S., and Chen, X. Cross attention network for few-shot classification. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- Hu, S. X., Li, D., Stuhmer,¨ J., Kim, M., and Hospedales, T. M. Pushing the limits of simple pipelines for few-shot learning: External data and fine-tuning make a difference. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 9068–9077, 2022. 

- Hu, Y., Gripon, V., and Pateux, S. Leveraging the feature distribution in transfer-based few-shot learning. In _International Conference on Artificial Neural Networks_ , pp. 487–499, 2021. 

- Jiang, W., Kwok, J., and Zhang, Y. Subspace learning for effective meta-learning. In _International Conference on Machine Learning_ , pp. 10177–10194, 2022. 

- Kang, D., Kwon, H., Min, J., and Cho, M. Relational embedding for few-shot classification. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 2021. 

- Kim, J., Kim, H., and Kim, G. Model-agnostic boundaryadversarial sampling for test-time generalization in fewshot learning. In _European Conference on Computer Vision_ , pp. 599–617, 2020. 

- Krizhevsky, A. and Hinton, G. Learning multiple layers of features from tiny images. _Master’s thesis, Department of Computer Science, University of Toronto_ , 2009. 

- Lai, J., Yang, S., Liu, W., Zeng, Y., Huang, Z., Wu, W., Liu, J., Gao, B.-B., and Wang, C. tsf: Transformerbased semantic filter for few-shot learning. In _European Conference on Computer Vision_ , pp. 1–19, 2022. 

- Lazarou, M., Stathaki, T., and Avrithis, Y. Iterative label cleaning for transductive and semi-supervised few-shot learning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 8751–8760, 2021. 

- Lee, K., Maji, S., Ravichandran, A., and Soatto, S. Metalearning with differentiable convex optimization. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 10657–10665, 2019. 

- Lin, H., Han, G., Ma, J., Huang, S., Lin, X., and Chang, S.-F. Supervised masked knowledge distillation for few-shot transformers. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 19649–19659, 2023. 

- Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. Swin transformer: Hierarchical vision transformer using shifted windows. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 10012–10022, 2021. 

- Luo, X., Wei, L., Wen, L., Yang, J., Xie, L., Xu, Z., and Tian, Q. Rectifying the shortcut learning of background for few-shot learning. _Advances in Neural Information Processing Systems_ , 34, 2021. 

- Ma, J., Xie, H., Han, G., Chang, S.-F., Galstyan, A., and Abd-Almageed, W. Partner-assisted learning for few-shot image classification. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 10573– 10582, 2021a. 

- Ma, R., Fang, P., Avraham, G., Zuo, Y., Drummond, T., and Harandi, M. Learning instance and task-aware dynamic kernels for few shot learning. _arXiv preprint arXiv:2112.03494_ , 2021b. 

- Ma, R., Fang, P., Drummond, T., and Harandi, M. Adaptive poincare point to set distance for few-shot classification.´ _arXiv preprint arXiv:2112.01719_ , 2021c. 

- Nguyen, T. and Grover, A. Transformer neural processes: Uncertainty-aware meta learning via sequence modeling. _arXiv preprint arXiv:2207.04179_ , 2022. 

11 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

- Oh, J., Kim, S., Ho, N., Kim, J.-H., Song, H., and Yun, S.-Y. Understanding cross-domain few-shot learning: An experimental study. _arXiv preprint arXiv:2202.01339_ , 2022. 

- Oreshkin, B., Rodr´ıguez Lopez, P., and Lacoste, A.´ Tadam: Task dependent adaptive metric for improved few-shot learning. _Advances in Neural Information Processing Systems_ , 31, 2018. 

- Perez, E., Strub, F., De Vries, H., Dumoulin, V., and Courville, A. Film: Visual reasoning with a general conditioning layer. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 32, 2018. 

- Qi, G., Yu, H., Lu, Z., and Li, S. Transductive few-shot classification on the oblique manifold. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 8412–8422, 2021. 

- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In _International Conference on Machine Learning_ , pp. 8748–8763, 2021. 

- Raghu, A., Raghu, M., Bengio, S., and Vinyals, O. Rapid learning or feature reuse? towards understanding the effectiveness of maml. _arXiv preprint arXiv:1909.09157_ , 2019. 

- Raghu, M., Unterthiner, T., Kornblith, S., Zhang, C., and Dosovitskiy, A. Do vision transformers see like convolutional neural networks? _Advances in Neural Information Processing Systems_ , 34:12116–12128, 2021. 

- Ranftl, R., Bochkovskiy, A., and Koltun, V. Vision transformers for dense prediction. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pp. 12179–12188, 2021. 

- Ren, M., Triantafillou, E., Ravi, S., Snell, J., Swersky, K., Tenenbaum, J. B., Larochelle, H., and Zemel, R. S. Meta-learning for semi-supervised few-shot classification. _arXiv preprint arXiv:1803.00676_ , 2018a. 

- Ren, M., Triantafillou, E., Ravi, S., Snell, J., Swersky, K., Tenenbaum, J. B., Larochelle, H., and Zemel, R. S. Meta-learning for semi-supervised few-shot classification. _arXiv preprint arXiv:1803.00676_ , 2018b. 

- Requeima, J., Gordon, J., Bronskill, J., Nowozin, S., and Turner, R. E. Fast and flexible multi-task classification using conditional neural adaptive processes. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- Rusu, A. A., Rao, D., Sygnowski, J., Vinyals, O., Pascanu, R., Osindero, S., and Hadsell, R. Meta-learning with 

latent embedding optimization. In _International Conference on Learning Representations_ , 2018. 

- Simon, C., Koniusz, P., Nock, R., and Harandi, M. Adaptive subspaces for few-shot learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 4136–4145, 2020. 

- Smith, J. S., Karlinsky, L., Gutta, V., Cascante-Bonilla, P., Kim, D., Arbelle, A., Panda, R., Feris, R., and Kira, Z. Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 11909–11919, 2023. 

- Snell, J., Swersky, K., and Zemel, R. Prototypical networks for few-shot learning. _Advances in Neural Information Processing Systems_ , 30, 2017. 

- Strudel, R., Garcia, R., Laptev, I., and Schmid, C. Segmenter: Transformer for semantic segmentation. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pp. 7262–7272, 2021. 

- Sung, F., Yang, Y., Zhang, L., Xiang, T., Torr, P. H., and Hospedales, T. M. Learning to compare: Relation network for few-shot learning. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 1199–1208, 2018. 

- Thrun, S. and Pratt, L. _Learning to learn_ . Springer Science & Business Media, 2012. 

- Tian, Y., Wang, Y., Krishnan, D., Tenenbaum, J. B., and Isola, P. Rethinking few-shot image classification: a good embedding is all you need? In _European Conference on Computer Vision_ , pp. 266–282, 2020. 

- Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., and Jegou, H. Training data-efficient image transform-´ ers & distillation through attention. In _International Conference on Machine Learning_ , pp. 10347–10357, 2021. 

- Triantafillou, E., Zhu, T., Dumoulin, V., Lamblin, P., Evci, U., Xu, K., Goroshin, R., Gelada, C., Swersky, K., Manzagol, P.-A., et al. Meta-dataset: A dataset of datasets for learning to learn from few examples. In _International Conference on Learning Representations_ , 2020. 

- Triantafillou, E., Larochelle, H., Zemel, R., and Dumoulin, V. Learning a universal template for few-shot dataset generalization. In _International Conference on Machine Learning_ , pp. 10424–10433, 2021. 

- Tschandl, P., Rosendahl, C., and Kittler, H. The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. _Scientific data_ , 5(1):1–9, 2018. 

12 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

- Tseng, H.-Y., Lee, H.-Y., Huang, J.-B., and Yang, M.-H. Cross-domain few-shot classification via learned featurewise transformation. _arXiv preprint arXiv:2001.08735_ , 2020. 

- Tu, Z., Talebi, H., Zhang, H., Yang, F., Milanfar, P., Bovik, A., and Li, Y. Maxvit: Multi-axis vision transformer. _arXiv preprint arXiv:2204.01697_ , 2022. 

- Vinyals, O., Blundell, C., Lillicrap, T., Wierstra, D., et al. Matching networks for one shot learning. _Advances in Neural Information Processing Systems_ , 29:3630–3638, 2016a. 

- Vinyals, O., Blundell, C., Lillicrap, T., Wierstra, D., et al. Matching networks for one shot learning. _Advances in Neural Information Processing Systems_ , 29, 2016b. 

- Wang, Z., Liu, L., Duan, Y., Kong, Y., and Tao, D. Continual learning with lifelong vision transformer. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 171–181, 2022a. 

- Wang, Z., Zhang, Z., Lee, C.-Y., Zhang, H., Sun, R., Ren, X., Su, G., Perot, V., Dy, J., and Pfister, T. Learning to prompt for continual learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 139–149, 2022b. 

- Wertheimer, D., Tang, L., and Hariharan, B. Few-shot classification with feature map reconstruction networks. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 8012–8021, 2021. 

- Wu, J., Zhang, T., Zhang, Y., and Wu, F. Task-aware part mining network for few-shot learning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 8433–8442, 2021. 

- Xie, J., Long, F., Lv, J., Wang, Q., and Li, P. Joint distribution matters: Deep brownian distance covariance for few-shot classification. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 7972–7981, 2022. 

- Xu, C., Fu, Y., Liu, C., Wang, C., Li, J., Huang, F., Zhang, L., and Xue, X. Learning dynamic alignment via meta-filter for few-shot learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 5182–5191, 2021. 

- Xu, J., Ton, J.-F., Kim, H., Kosiorek, A., and Teh, Y. W. Metafun: Meta-learning with iterative functional updates. In _International Conference on Machine Learning_ , pp. 10617–10627, 2020. 

- Yao, H., Wei, Y., Huang, J., and Li, Z. Hierarchically structured meta-learning. In _International Conference on Machine Learning_ , pp. 7045–7054, 2019. 

- Yao, H., Wu, X., Tao, Z., Li, Y., Ding, B., Li, R., and Li, Z. Automated relational meta-learning. _arXiv preprint arXiv:2001.00745_ , 2020. 

- Ye, H.-J., Hu, H., Zhan, D.-C., and Sha, F. Few-shot learning via embedding adaptation with set-to-set functions. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 8808–8817, 2020. 

- Zagoruyko, S. and Komodakis, N. Wide residual networks. _arXiv preprint arXiv:1605.07146_ , 2016. 

- Zhang, C., Cai, Y., Lin, G., and Shen, C. Deepemd: Fewshot image classification with differentiable earth mover’s distance and structured classifiers. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 12203–12213, 2020a. 

- Zhang, C., Ding, H., Lin, G., Li, R., Wang, C., and Shen, C. Meta navigator: Search for a good adaptation policy for few-shot learning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 9435– 9444, 2021a. 

- Zhang, C., Zhang, M., Zhang, S., Jin, D., Zhou, Q., Cai, Z., Zhao, H., Liu, X., and Liu, Z. Delving deep into the generalization of vision transformers under distribution shifts. In _Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition_ , pp. 7277–7286, 2022. 

- Zhang, M., Zhang, J., Lu, Z., Xiang, T., Ding, M., and Huang, S. Iept: Instance-level and episode-level pretext tasks for few-shot learning. In _International Conference on Learning Representations_ , 2020b. 

- Zhang, R., Fang, R., Zhang, W., Gao, P., Li, K., Dai, J., Qiao, Y., and Li, H. Tip-adapter: Training-free clip-adapter for better vision-language modeling. _arXiv preprint arXiv:2111.03930_ , 2021b. 

- Zhang, X., Meng, D., Gouk, H., and Hospedales, T. M. Shallow bayesian meta learning for real-world few-shot recognition. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 651–660, 2021c. 

- Zhao, J., Yang, Y., Lin, X., Yang, J., and He, L. Looking wider for better adaptive representation in few-shot learning. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pp. 10981–10989, 2021. 

- Zhmoginov, A., Sandler, M., and Vladymyrov, M. Hypertransformer: Model generation for supervised and semi-supervised few-shot learning. In _International Conference on Machine Learning_ , pp. 27075–27098, 2022. 

- Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., and Kong, T. ibot: Image bert pre-training with online tokenizer. _International Conference on Learning Representations (ICLR)_ , 2022a. 

13 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

- Zhou, K., Yang, J., Loy, C. C., and Liu, Z. Conditional prompt learning for vision-language models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 16816–16825, 2022b. 

- Zhou, P., Zou, Y., Yuan, X.-T., Feng, J., Xiong, C., and Hoi, S. Task similarity aware meta learning: Theory-inspired improvement on maml. In _Uncertainty in artificial intelligence_ , pp. 23–33, 2021a. 

- Zhou, Z., Qiu, X., Xie, J., Wu, J., and Zhang, C. Binocular mutual learning for improving few-shot classification. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 8402–8411, 2021b. 

- Zhu, H. and Koniusz, P. Transductive few-shot learning with prototype-based label propagation by iterative graph refinement. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 23996– 24006, 2023. 

- Zhu, X., Su, W., Lu, L., Li, B., Wang, X., and Dai, J. Deformable detr: Deformable transformers for end-to-end object detection. _arXiv preprint arXiv:2010.04159_ , 2020. 

- Zhu, X., Zhang, R., He, B., Zhou, A., Wang, D., Zhao, B., and Gao, P. Not all features matter: Enhancing fewshot clip with adaptive prior refinement. _arXiv preprint arXiv:2304.01195_ , 2023. 

14 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

## One Meta-tuned Transformer is What You Need for Few-shot Learning - _Supplementary Material-_ 

## **A. Setup for In-Domain Few-Shot Evaluation** 

## **A.1. Datasets Used for Benchmarks** 

For standard few-shot image classification evaluation with only class shift, we train and evaluate our MetaFormer presented in the main paper on the following few-shot benchmarks: _**mini**_ **ImageNet.** (Vinyals et al., 2016b) is a subset of the ImageNet1K, consisting of 100 classes and 600 images in each category. The classes are divided into 64, 16, and 20 for training, validation, and test, respectively. _**tiered**_ **ImageNet.** (Ren et al., 2018b) is another larger and more challenging subset of ImageNet-1K. It contains 34 higher-level nodes near the root of ImageNet, which are 608 classes in total. The dataset is split into 20, 6, and 8 higher-level nodes and corresponding 351, 97, and 160 classes as the training, validation, and testing set, respectively. **CIFAR-FS** (Bertinetto et al., 2019) contains 100 classes and 600 images from the CIFAR100 dataset (Krizhevsky & Hinton, 2009). The classes are split into 64 for training, 16 for validation, and 20 for testing. **FC100** (Oreshkin et al., 2018) is built from the CIFAR100 (Krizhevsky & Hinton, 2009) employing a splitting strategy analogous to that of the _tiered_ ImageNet dataset to enhance difficulty, giving rise to 60 training, 20 validation, and 20 test classes. 

## **A.2. Additional Implementation Details** 

**Pretraining.** For MetaFormer, we adhere to the strategy delineated by Hiller et al. (2022) for pretraining our vision transformer backbones on the meta-training split of each dataset, maintaining most of the training hyperparameter configurations reported in their study. Concretely, we employ default two global crops and ten local crops with respective crop scales of (0.4, 1.0) and (0.05, 0.4). We use the image resolution of 224 _×_ 224 and the output is projected to 8192 dimensions. A patch size of 16 and window size of 7 are used for aligning standard settings in ViT-small (Dosovitskiy et al., 2020; Touvron et al., 2021) and Swin-tiny (Liu et al., 2021), respectively. A batch size of 512 and a cosine-decaying learning rate schedule are used. For SMKD-MetaFormer, we follow Lin et al. (2023) to train the vision transformer backbones with the image resolution of 480 _×_ 480. 

**Meta-tuning.** We integrate our proposed MSA and PTA into the original vision transformer in every other layer, starting from the 6th layer, to construct holistic attention for meta-learning. Here, MSA and PTA are randomly initialized. The number of task probe vectors _T_ is configured to one for ViT and eight for Swin, and the pool size is set to _Z_ = 50. As training progresses, we consolidate via Eqn. (5) for a new task which is sufficiently similar to previous components, where we evaluate the similarity via cosine similarity and set a similarity threshold of 0.5; otherwise, we directly add the task-specific _g[′]_ to the pool if there is available capacity. During meta-tuning, we follow most of the training techniques used in FewTURE (Hiller et al., 2022). We employ the SGD optimizer, utilizing a cosine-decaying learning rate initiated at 2 _×_ 10 _[−]_[4] , a momentum value of 0.9, and a weight decay of 5 _×_ 10 _[−]_[4] across all datasets. The input image size is set to 224 _×_ 224 for MetaFormer and 360 _×_ 360 for SMKD-MetaFormer. Typically, training is conducted for a maximum of 200 epochs. To mitigate the risk of overfitting, we adopt the early stopping strategy coupled with freezing parameters of the first three layers. For a fair comparison with the state-of-the-art transductive methods (Lazarou et al., 2021; Zhu & Koniusz, 2023), we adopt the same feature pre-processing as Hu et al. (2021) for MetaFormer-A. We set the sampling size _r_ = 15 and shuffle the order of samples _s_ = 30 times for autoregressive inference. All additional hyperparameters are selected on 600 randomly sampled episodes from the respective validation sets to ascertain the optimal parameter configuration. For the evaluation of few-shot learning, we conduct a random sampling of 600 episodes from the test set to evaluate our model. 

## **A.3. Results** 

We present a more comprehensive few-shot evaluation in Table 11 and Table 12, comparing various methods on the four benchmark datasets. Based on ViT-Small, our MetaFormer consistently outperforms established meta-learning methods PMF (Hu et al., 2022) and FewTURE (Hiller et al., 2022) across all evaluations. Notably, this remarkable performance is achieved without the need for inference-time fine-tuning, underscoring its robustness and effectiveness. We find that our MetaFormer effectively synergizes with transfer learning method SMKD (Lin et al., 2023) to further enhance its performance. When compared to the transductive protoLP (Zhu & Koniusz, 2023), MetaFormer exhibits superior performance in most scenarios with a smaller number of parameters. 

15 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

_Table 11._ **More comprehensive results on** _**mini**_ **ImageNet and** _**tiered**_ **ImageNet for** 5 **-way** 1 **-shot and** 5 **-way** 5 **-shot scenarios.** Reported are the mean and 95% confidence interval of average classification accuracy (%) on the unseen meta-test set. * denotes results reported by us. 

|us.|||||
|---|---|---|---|---|
|**Method**|**Setting**|**Backbone**<br>**# Params**|**_mini_ImageNet**<br>**1-shot**<br>**5-shot**|**_tiered_ImageNet**<br>**1-shot**<br>**5-shot**|
|MatchNet (Vinyals et al.,2016b)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>61_._24_±_0_._29<br>73_._93_±_0_._23<br>71_._01_±_0_._33<br>83_._12_±_0_._24<br>ProtoNet (Snell et al.,2017)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>62_._29_±_0_._33<br>79_._46_±_0_._48<br>68_._25_±_0_._23<br>84_._01_±_0_._56<br>FEAT (Ye et al.,2020)<br>Inductive<br>_ResNet-12_<br>14.1 M<br>66_._78_±_0_._20<br>82_._05_±_0_._14<br>70_._80_±_0_._23<br>84_._79_±_0_._16<br>DeepEMD (Zhang et al.,2020a)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>65_._91_±_0_._82<br>82_._41_±_0_._56<br>71_._16_±_0_._87<br>86_._03_±_0_._58<br>IEPT (Zhang et al.,2020b)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>67_._05_±_0_._44<br>82_._90_±_0_._30<br>72_._24_±_0_._50<br>86_._73_±_0_._34<br>MELR (Fei et al.,2020)<br>Inductive<br>_ResNet-12_<br>14.1 M<br>67_._40_±_0_._43<br>83_._40_±_0_._28<br>72_._14_±_0_._51<br>87_._01_±_0_._35<br>FRN (Wertheimer et al.,2021)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>66_._45_±_0_._19<br>82_._83_±_0_._13<br>72_._06_±_0_._22<br>86_._89_±_0_._14<br>CG (Zhao et al.,2021)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>67_._02_±_0_._20<br>82_._32_±_0_._14<br>71_._66_±_0_._23<br>85_._50_±_0_._15<br>DMF (Xu et al.,2021)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>67_._76_±_0_._46<br>82_._71_±_0_._31<br>71_._89_±_0_._52<br>85_._96_±_0_._35<br>BML (Zhou et al.,2021b)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>67_._04_±_0_._63<br>83_._63_±_0_._29<br>68_._99_±_0_._50<br>85_._49_±_0_._34<br>CNL (Zhao et al.,2021)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>67_._96_±_0_._98<br>83_._36_±_0_._51<br>73_._42_±_0_._95<br>87_._72_±_0_._75<br>Meta-NVG (Zhang et al.,2021a)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>67_._14_±_0_._80<br>83_._82_±_0_._51<br>74_._58_±_0_._88<br>86_._73_±_0_._61<br>RENet (Kang et al.,2021)<br>Inductive<br>_ResNet-12_<br>12.6 M<br>67_._60_±_0_._44<br>82_._58_±_0_._30<br>71_._61_±_0_._51<br>85_._28_±_0_._35<br>PAL (Ma et al.,2021a)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>69_._37_±_0_._64<br>84_._40_±_0_._44<br>72_._25_±_0_._72<br>86_._95_±_0_._47<br>COSOC (Luo et al.,2021)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>69_._28_±_0_._49<br>85_._16_±_0_._42<br>73_._57_±_0_._43<br>87_._57_±_0_._10<br>Meta DeepBDC(Xie et al.,2022)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>67_._34_±_0_._43<br>84_._46_±_0_._28<br>72_._34_±_0_._49<br>87_._31_±_0_._32|||||
|LEO (Rusu et al.,2018)<br>MetaFun (Xu et al.,2020)<br>CC+rot (Gidaris et al.,2019)<br>FEAT (Ye et al.,2020)<br>MetaQDA (Zhang et al.,2021c)<br>OM(Qi et al.,2021)|Inductive<br>Inductive<br>Inductive<br>Inductive<br>Inductive<br>Inductive|_WRN-28-10_<br>36.8 M<br>_WRN-28-10_<br>37.7 M<br>_WRN-28-10_<br>36.5 M<br>_WRN-28-10_<br>38.1 M<br>_WRN-28-10_<br>36.5 M<br>_WRN-28-10_<br>36.5 M|61_._76_±_0_._08<br>77_._59_±_0_._12<br>62_._12_±_0_._30<br>78_._20_±_0_._16<br>62_._93_±_0_._45<br>79_._87_±_0_._33<br>65_._10_±_0_._20<br>81_._11_±_0_._14<br>67_._83_±_0_._64<br>84_._28_±_0_._69<br>66_._78_±_0_._30<br>85_._29_±_0_._41|66_._33_±_0_._05<br>81_._44_±_0_._09<br>67_._72_±_0_._14<br>83_._28_±_0_._12<br>70_._53_±_0_._51<br>84_._98_±_0_._36<br>70_._41_±_0_._23<br>84_._38_±_0_._16<br>74_._33_±_0_._65<br>89_._56_±_0_._79<br>71_._54_±_0_._29<br>87_._79_±_0_._46|
|SUN (Dong et al.,2022)<br>FewTURE (Hiller et al.,2022)<br>FewTURE (Hiller et al.,2022)<br>PMF_∗_(Hu et al.,2022)|Inductive<br>Inductive<br>Inductive<br>Inductive|_ViT_<br>12.5 M<br>_ViT-Small_<br>22 M<br>_Swin-Tiny_<br>29 M<br>_ViT-Small_<br>21 M|67_._80_±_0_._45<br>83_._25_±_0_._30<br>68_._02_±_0_._88<br>84_._51_±_0_._53<br>72_._40_±_0_._78<br>86_._38_±_0_._49<br>71_._01_±_0_._81<br>86_._03_±_0_._53|72_._99_±_0_._50<br>86_._74_±_0_._33<br>72_._96_±_0_._92<br>86_._43_±_0_._67<br>76_._32_±_0_._87<br>89_._96_±_0_._55<br>76_._61_±_0_._89<br>89_._66_±_0_._54|
|MetaFormer-I(Ours)|Inductive|_ViT-Small_<br>24.5 M|**75**_._**78**_±_**0**_._**71**<br>**90**_._**02**_±_**0**_._**44**|**79**_._**05**_±_**0**_._**81**<br>**90**_._**40**_±_**0**_._**53**|
|HCTransformers (He et al.,2022b)<br>SMKD (Lin et al.,2023)|Inductive<br>Inductive|3_×ViT-Small_<br>63 M<br>_ViT-Small_<br>21 M|74_._74_±_0_._17<br>89_._19_±_0_._13<br>74_._28_±_0_._18<br>88_._89_±_0_._09|79_._67_±_0_._20<br>91_._72_±_0_._11<br>78_._83_±_0_._20<br>91_._21_±_0_._11|
|SMKD + MetaFormer-I(Ours)|Inductive|_ViT-Small_<br>24.5 M|**76**_._**54**_±_**0**_._**73**<br>**90**_._**76**_±_**0**_._**41**|**80**_._**57**_±_**0**_._**82**<br>**92**_._**42**_±_**0**_._**49**|
|PT+MAP (Hu et al.,2021)<br>iLPC (Lazarou et al.,2021)<br>EASY (Bendou et al.,2022)<br>protoLP (Zhu & Koniusz,2023)|Transductive<br>Transductive<br>Transductive<br>Transductive|_WRN-28-10_<br>36.5 M<br>_WRN-28-10_<br>36.5 M<br>_3×ResNet-12_<br>37.2 M<br>_WRN-28-10_<br>36.5 M|82_._92_±_0_._26<br>88_._82_±_0_._13<br>83_._05_±_0_._79<br>88_._82_±_0_._42<br>84_._04_±_0_._23<br>89_._14_±_0_._11<br>84_._32_±_0_._21<br>90_._02_±_0_._12|_−_<br>_−_<br>88_._50_±_0_._75<br>92_._46_±_0_._42<br>84_._29_±_0_._24<br>89_._76_±_0_._14<br>**89**_._**65**_±_**0**_._**22**<br>93_._21_±_0_._13|
|MetaFormer-A(Ours)|Transductive|_ViT-Small_<br>24.5 M|**84**_._**78**_±_**0**_._**79**<br>**91**_._**39**_±_**0**_._**42**|88_._38_±_0_._78<br>**93**_._**37**_±_**0**_._**45**|



## **B. Setup for Cross-Domain Few-Shot Evaluation** 

## **B.1. Datasets Used for Benchmarks** 

We use _mini_ ImageNet as the source dataset for meta-training and perform the cross-domain few-shot evaluation on eight datasets with varying domain similarity, following Oh et al. (2022). The datasets can be separated into two groups: **BSCD-FSL benchmark** (Guo et al., 2020) and **nonBSCD-FSL** . For BSCD-FSL benchmark (CropDisease, EuroSAT, ISIC, ChestX), we follow Guo et al. (2020) for the dataset split. And for nonBSCD-FSL benchmark (CUB, Car, Plantaem Places), we follow Tseng et al. (2020) for the splitting procedure. We refer to Oh et al. (2022) for a more detailed description of each dataset. 

## **B.2. Implementation Details** 

In our cross-domain experiments, we meta-train our MetaFormer-I and MetaFormer-A on the _mini_ ImageNet dataset as in Section 4.1 in the main paper and then freeze all parameters during evaluation on cross-domain benchmarks. For a fair comparison, we adhere to the standard meta-testing procedure to assess the performance of baseline models trained on _mini_ ImageNet, including FewTURE (Hiller et al., 2022), PMF (Hu et al., 2022), SMKD (Lin et al., 2023) and protoLP (Zhu & Koniusz, 2023). 

16 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

_Table 12._ **More comprehensive results on CIFAR-FS and FC100 for** 5 **-way** 1 **-shot and** 5 **-way** 5 **-shot scenarios.** Reported are the mean and 95% confidence interval of average classification accuracy (%) on the unseen meta-test set. * denotes results reported by us. 

|**Method**|**Setting**|**Backbone**<br>**# Params**|**CIFAR-FS**<br>**1-shot**<br>**5-shot**|**FC100**<br>**1-shot**<br>**5-shot**|
|---|---|---|---|---|
|ProtoNet (Snell et al.,2017)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>-<br>-<br>41_._54_±_0_._76<br>57_._08_±_0_._76<br>MetaOpt (Lee et al.,2019)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>72_._00_±_0_._70<br>84_._20_±_0_._50<br>41_._10_±_0_._60<br>55_._50_±_0_._60<br>MABAS (Kim et al.,2020)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>73_._51_±_0_._92<br>85_._65_±_0_._65<br>42_._31_±_0_._75<br>58_._16_±_0_._78<br>RFS (Tian et al.,2020)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>73_._90_±_0_._80<br>86_._90_±_0_._50<br>44_._60_±_0_._70<br>60_._90_±_0_._60<br>BML (Zhou et al.,2021b)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>73_._45_±_0_._47<br>88_._04_±_0_._33<br>45_._00_±_0_._41<br>63_._03_±_0_._41<br>CG (Gao et al.,2021)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>73_._00_±_0_._70<br>85_._80_±_0_._50<br>-<br>-<br>Meta-NVG (Zhang et al.,2021a)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>74_._63_±_0_._91<br>86_._45_±_0_._59<br>46_._40_±_0_._81<br>61_._33_±_0_._71<br>RENet (Kang et al.,2021)<br>Inductive<br>_ResNet-12_<br>12.6 M<br>74_._51_±_0_._46<br>86_._60_±_0_._32<br>-<br>-<br>TPMN (Wu et al.,2021)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>75_._50_±_0_._90<br>87_._20_±_0_._60<br>46_._93_±_0_._71<br>63_._26_±_0_._74<br>MixFSL(Afrasiyabi et al.,2021)<br>Inductive<br>_ResNet-12_<br>12.4 M<br>-<br>-<br>44_._89_±_0_._63<br>60_._70_±_0_._60|||||
|CC+rot (Gidaris et al.,2019)<br>PSST (Chen et al.,2021b)<br>Meta-QDA(Zhanget al.,2021c)|Inductive<br>Inductive<br>Inductive|WRN-28-10<br>73_._62_±_0_._31<br>_WRN-28-10_<br>36.5 M<br>_WRN-28-10_<br>36.5 M|86_._05_±_0_._22<br>-<br>77_._02_±_0_._38<br>88_._45_±_0_._35<br>75_._83_±_0_._88<br>88_._79_±_0_._75|-<br>-<br>-<br>-<br>-|
|SUN (Dong et al.,2022)<br>FewTURE (Hiller et al.,2022)<br>FewTURE (Hiller et al.,2022)<br>PMF_∗_(Hu et al.,2022)|Inductive<br>Inductive<br>Inductive<br>Inductive|_ViT_<br>12.5M<br>_ViT-Small_<br>22 M<br>_Swin-Tiny_<br>29 M<br>_ViT-Small_<br>21 M|78_._37_±_0_._46<br>88_._84_±_0_._32<br>76_._10_±_0_._88<br>86_._14_±_0_._64<br>77_._76_±_0_._81<br>88_._90_±_0_._59<br>76_._70_±_0_._90<br>87_._61_±_0_._60|-<br>-<br>46_._20_±_0_._79<br>63_._14_±_0_._73<br>47_._68_±_0_._78<br>63_._81_±_0_._75<br>45_._79_±_0_._73<br>63_._64_±_0_._72|
|MetaFormer-I(Ours)|Inductive|_ViT-Small_<br>24.5 M|**80**_._**16**_±_**0**_._**76**<br>**90**_._**57**_±_**0**_._**55**|**51**_._**14**_±_**0**_._**71**<br>**68**_._**33**_±_**0**_._**74**|
|HCTransformers (He et al.,2022b)<br>SMKD (Lin et al.,2023)|Inductive<br>Inductive|3_×ViT-Small_<br>63 M<br>_ViT-Small_<br>21M|78_._89_±_0_._18<br>90_._50_±_0_._09<br>80_._08_±_0_._18<br>90_._91_±_0_._13|48_._27_±_0_._15<br>66_._42_±_0_._16<br>50_._38_±_0_._16<br>68_._50_±_0_._16|
|SMKD + MetaFormer-I(Ours)|Inductive|_ViT-Small_<br>24.5 M|**81**_._**49**_±_**0**_._**74**<br>**91**_._**91**_±_**0**_._**54**|**52**_._**18**_±_**0**_._**78**<br>**71**_._**29**_±_**0**_._**73**|
|PT+MAP (Hu et al.,2021)<br>iLPC (Lazarou et al.,2021)<br>EASY (Bendou et al.,2022)<br>protoLP (Zhu & Koniusz,2023)|Transductive<br>Transductive<br>Transductive<br>Transductive|_WRN-28-10_<br>36.5 M<br>_WRN-28-10_<br>36.5 M<br>_3×ResNet-12_<br>37.2 M<br>_WRN-28-10_<br>36.5 M|87_._69_±_0_._23<br>90_._68_±_0_._15<br>86_._51_±_0_._75<br>90_._60_±_0_._48<br>87_._16_±_0_._21<br>90_._47_±_0_._15<br>87_._69_±_0_._23<br>90_._82_±_0_._15|_−_<br>_−_<br>_−±−_<br>_−±−_<br>54_._13_±_0_._24<br>66_._86_±_0_._19<br>_−_<br>_−_|
|MetaFormer-A(Ours)|Transductive|_ViT-Small_<br>24.5 M|**88**_._**34**_±_**0**_._**76**<br>**92**_._**21**_±_**0**_._**59**|**58**_._**04**_±_**0**_._**99**<br>**70**_._**80**_±_**0**_._**76**|



## **B.3. Results.** 

In Table 13, we present a comparative evaluation of MetaFormer against other previous state-of-the-art methods in the 5-way 5-shot cross-domain scenario. MetaFormer exhibits superior performance across various domains, surpassing its meta-learning counterparts in both inductive and transductive settings by significant margins of up to 8 _._ 20% and 22 _._ 60%, respectively. Notably, MetaFormer outperforms PMF (Hu et al., 2022) in most cases by a clear margin, achieving this without test-time full fine-tuning. Our results also show that MetaFormer improves the transfer learning method SMKD (Lin et al., 2023) by up to 3 _._ 70%. These findings underscore the robustness and effectiveness of MetaFormer in task adaptation. 

_Table 13._ **Broader study of cross-domain few-shot learning.** Average classification accuracy (%) for 5-way 5-shot scenario when meta-learning on _mini_ ImageNet (Vinyals et al., 2016a) but meta-testing on cross-domain few-shot benchmarks. _†_ means test-time finetuning is employed. 

|fnetuning is employed.||
|---|---|
|**Method**|**CUB**<br>**Cars**<br>**Places**<br>**Plantae**<br>**CropDisease**<br>**EuroSAT**<br>**ISIC**<br>**ChestX**|
|PMF_†_ (Hu et al.,2022)<br>63_._42_±_0_._74<br>**49**_._**23**_±_**0**_._**75**<br>77_._52_±_0_._69<br>63_._89_±_0_._71<br>89_._17_±_0_._55<br>84_._02_±_0_._55<br>44_._76_±_0_._56<br>25_._84_±_0_._43<br>FewTURE_†_ (Hiller et al.,2022)<br>67_._70_±_0_._77<br>46_._54_±_0_._73<br>74_._70_±_0_._69<br>61_._72_±_0_._71<br>86_._41_±_0_._56<br>77_._88_±_0_._57<br>38_._53_±_0_._54<br>25_._54_±_0_._43||
|MetaFormer-I(Ours)|**75**_._**90**_±_**0**_._**77**<br>48_._62_±_0_._79<br>**78**_._**42**_±_**0**_._**65**<br>**65**_._**40**_±_**0**_._**74**<br>**89**_._**72**_±_**0**_._**58**<br>**85**_._**01**_±_**0**_._**51**<br>**45**_._**90**_±_**0**_._**57**<br>**26**_._**47**_±_**0**_._**45**|
|SMKD (Lin et al.,2023)|77_._17_±_0_._69<br>50_._72_±_0_._71<br>80_._52_±_0_._64<br>63_._98_±_0_._72<br>92_._11_±_0_._45<br>85_._28_±_0_._54<br>47_._58_±_0_._62<br>25_._75_±_0_._43|
|SMKD+MetaFormer-I(Ours)|**80**_._**87**_±_**0**_._**68**<br>**52**_._**68**_±_**0**_._**74**<br>**80**_._**96**_±_**0**_._**67**<br>**65**_._**60**_±_**0**_._**72**<br>**92**_._**91**_±_**0**_._**45**<br>**86**_._**60**_±_**0**_._**50**<br>**49**_._**80**_±_**0**_._**60**<br>**26**_._**25**_±_**0**_._**45**|
|protoLP (Zhu & Koniusz,2023)|76_._95_±_1_._10<br>44_._49_±_0_._93<br>74_._81_±_1_._11<br>54_._67_±_1_._07<br>82_._96_±_1_._57<br>65_._99_±_1_._86<br>37_._74_±_0_._83<br>23_._18_±_0_._45|
|MetaFormer-A (Ours)|**82**_._**92**_±_**0**_._**68**<br>**52**_._**92**_±_**0**_._**81**<br>**83**_._**18**_±_**0**_._**62**<br>**67**_._**11**_±_**0**_._**74**<br>**94**_._**66**_±_**0**_._**44**<br>**88**_._**59**_±_**0**_._**48**<br>**49**_._**55**_±_**0**_._**62**<br>**25**_._**86**_±_**0**_._**45**|



## **C. Setup for Multi-Domain Few-Shot Evaluation** 

## **C.1. Datasets used for Benchmarks** 

**Meta-Dataset** (Triantafillou et al., 2020) is a more challenging and realistic large-scale benchmark consisting of ten image datasets including ImageNet-1k, Omniglot, Aircraft, CUB, Textures, QuickDraw, Fungi, VGG Flower, Traffic Signs, and 

17 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

MSCOCO, each with specified train, val and test splits. We follow Hu et al. (2022) to utilize the train and val splits of the initial eight datasets (in-domain) for meta-training and validation, while employing the test splits of all datasets for meta-testing. We refer to Triantafillou et al. (2020) for an in-depth exploration of Meta-Dataset. 

## **C.2. Implementation Details** 

We meta-train both PMF (Hu et al., 2022) and our MetaFormer build upon the same pre-trained vision transformer (Caron et al., 2021) in a 5-way 1-shot setting, adhering to most of the unchanged training hyperparameters reported in PMF. The meta-tuning phase includes a 10-epoch warm-up, followed by a total of 100 epochs of training. We use the SGD optimizer with a momentum of 0.9 and a cosine-decaying learning rate scheduler is employed, initialized at 5 _×_ 10 _[−]_[4] . 

## **C.3. Results.** 

We evaluate the effectiveness of MetaFormer on the large-scale and challenging Meta-Dataset in multi-domain scenarios. Table 14 presents the test accuracy measured on each dataset meta-test set. MetaFormer achieves comparable or superior performance compared to previous state-of-the-art meta-learning method, PMF (Hu et al., 2022), in adapting to most domains. The superior overall performance of MetaFormer, especially in settings with scarce samples (e.g., one sample per category), underscores the efficacy of our proposed approach for fast adaptation in each domain. Additionally, the computational efficiency of MetaFormer-I facilitates quicker deployment, making our method particularly suitable for real-world applications in resource-constrained scenarios. 

_Table 14._ **Broader study of multi-domain few-shot learning.** Average classification accuracy (%) for 5-way 1-shot scenario. 

|**Model**|**FT**|**In-domain**|**Out-of-domain**|**Avg**|
|---|---|---|---|---|
|||INet<br>Omglot<br>Acraft<br>CUB<br>DTD<br>QDraw<br>Fungi<br>Flower|Sign<br>COCO||
|PMF (Hu et al.,2022)<br>Y||56.35<br>94.22<br>88.00<br>84.63<br>**52.90**<br>75.18<br>84.13<br>75.20<br>55.02<br>49.69<br>71.53|||
|MetaFormer-I (Ours)|-|**63.41**<br>**94.57**<br>87.93<br>**89.17**<br>51.33<br>75.10<br>81.97<br>**85.06**|**57.33**<br>**53.64**|**73.95**|
|MetaFormer-I (Ours)|Y|**64.25**<br>**94.64**<br>87.93<br>**89.11**<br>56.52<br>75.10<br>81.97<br>**85.31**|**62.86**<br>**53.82**|**75.15**|
|MetaFormer-A(Ours)|-|**66.03**<br>**96.47**<br>**89.75**<br>**91.95**<br>52.20<br>**79.17**<br>**84.44**<br>**88.88**|**58.89**<br>**58.12**|**76.59**|



## **D. MetaFormer on Large-Scale Foundation Models** 

We evaluate different methods on 1-shot EuroSAT (Helber et al., 2019) and ISIC (Tschandl et al., 2018) datasets. For both training and testing phases, we employ the episodic approach as described in Vinyals et al. (2016b). Note that we strictly follow the TiP-Adapter-F (Zhang et al., 2021b) pipeline to sample support set from the train set and query set from the test set to build the task for evaluation since there are no new classes in the test set. For instance, in the EuroSAT dataset with 10 classes, we construct the 10-way 1-shot task, where the support and query sets are drawn from the train and test split of the EuroSAT dataset, respectively. We also integrate the cache model from TiP-Adapter as the auxiliary classifier head. We only fine-tune introduced MSA and keep frozen the visual encoder and textual encoder of CLIP. We train our method for 20 epochs on both datasets and we employ the SGD optimizer with a cosine-decaying learning rate initiated at 2 _×_ 10 _[−]_[4] , a momentum value of 0.9, and a weight decay of 5 _×_ 10 _[−]_[4] . We test using the pre-trained word embeddings of a single prompt, “a photo of a [CLASS].” for all methods. 

## **E. MetaFormer on Hierarchical Transformers.** 

We extend MetaFormer to integrate with Swin (Liu et al., 2021), which employs shifted local window for performing self-attention within each window and merges patch embeddings to build hierarchical structures, thereby aggregating multi-scale information. As shown in Table 15, the experiments are conducted on the same pre-trained Swin-Tiny model and the results consistently demonstrate that MetaFormer outperforms FewTURE in both settings. These results highlight the versatility and general applicability of our method. 

## **F. Abaltion of Other Design Strategies** 

In Table 16, we explore various design choices for our approach. The variability of feature semantics across different layers in the backbone leads us to investigate which layer optimally facilitates sample interaction. The results are shown in Table 16a, indicating that starting to build intra-task interaction from stage 6 is moderate. Notably, the integration of 

18 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

_Table 15._ **Comparison results with the Swin-Transformer backbone on** _**mini**_ **Imagenet** . 

|_._ **Comparison results with the**|**Swin-Transf**|**ormer backbone on****_mini_I**|
|---|---|---|
|**Method**|**Backbone**|**1-shot**<br>**5-shot**|
|FewTURE (Hiller et al.,2022)<br>_Swin-Tiny_<br>72_._40_±_0_._78<br>86_._38_±_0_._49<br>MetaFormer-I (Ours)<br>_Swin-Tiny_<br>**74**_._**17**_±_**0**_._**73**<br>**89**_._**17**_±_**0**_._**45**|||



_Table 16._ Comparison results of different architecture design strategies. 

|(a) Different MSA locations in ViT-Samll.|(a) Different MSA locations in ViT-Samll.|(b) Different MSA|variants with ViT-Small.|(c) The number of probe vectors with ViT-|(c) The number of probe vectors with ViT-|
|---|---|---|---|---|---|
|||||Small.||
|Location|mini-1s (%)|Method|mini-1s (%)|||
|[5_,_7_,_9]<br>[6_,_8_,_10]|74.36<br>74.64|within-support<br>support-query|73.30<br>73.28|Number<br>1|mini-1s (%)<br>75.78|
|[11]|72.34|global features|71.49|4|75.28|
|||||8|75.44|



|(d) The number of probe vectors with Swin-|(d) The number of probe vectors with Swin-|(e) Different consolidation strategies with|(e) Different consolidation strategies with|(f) The size of knowledge pool with ViT-|(f) The size of knowledge pool with ViT-|
|---|---|---|---|---|---|
|Tiny.||ViT-Samll.||Samll.||
|Number|mini-1s (%)|Method|mini-1s (%)|Size|mini-1s (%)|
|1|73.89|with averaging|75.77|10|75.50|
|4|74.01|without averaging|75.57|50|75.78|
|8|74.17|max pooling|75.63|100|75.74|



multi-scale semantic information accounts for an improvement of 2.3%. Figure 4 shows the two alternative sample causal masks for our MetaFormer-I, where we encode the sample relationship separately. As shown in Table 16b, the ablated masks of within-support and support-query manifest sub-optimal performance, further validating that MSA with the inductive mask works not because of the introduction of extra parameters. This also underscores the complementary benefits of employing both interactions in our design on enhanced task-specific representations. Additionally, we observe that relying solely on global image features incurs a significant loss of critical information necessary for capturing discriminative relationships among samples. The results in Table 16c and Table 16d demonstrate that our model is not very sensitive to the number of task probe vectors. Also, as shown in Table 16e and Table 16f, the performance difference between various consolidation strategies is relatively marginal due to the nature of the cosine similarity-based score function employed during knowledge retrieval, and a sufficiently diverse but compact knowledge pool leads to improvements in performance. An excessively large pool risks performance degradation due to less effective consolidation. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning_images/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning.pdf-0019-07.png)


**----- Start of picture text -----**<br>
(a) Within-Support Sample Causal Mask (b) Support-Query Sample Causal Mask<br>Activated Sample Interaction Masked Sample Interaction<br>Support Support<br>Query Query<br>**----- End of picture text -----**<br>


_Figure 4._ The alternative sample casual masks for MetaFormer-I. 

19 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

## **G. Comparison with Other Inter-task Attention Methods** 

In Table 17, we compare our Patch-grained Task Attention (PTA) with other inter-task attention designs. Compared with IT-att presented in Wang et al. (2022a), we note that problem settings and underlying motivations of these modules are distinct. PTA is rooted in the domain of few-shot learning, where the paramount concern is facilitating knowledge transfer between tasks. Conversely, IT-att (Wang et al., 2022a) is grounded in continual learning, where the primary focus lies in mitigating catastrophic forgetting. While both PTA and IT-att seemingly adopt a learnable embedding for each task, their approaches to knowledge representation and utilization diverge significantly. PTA maintains a knowledge pool and captures task-specific information at the patch level. On the other hand, IT-att keeps a record of a single key and a single bias to store all past knowledge, relying on importance-based regularization to enforce task proximity and combat forgetting. Our analysis reveals that the use of a global task vector leads to a performance decrease of 0 _._ 97%. PTA demonstrates a 1 _._ 3% improvement over IT-att, which can be attributed to greater flexibility and expressiveness, aligning more closely with the objective of knowledge transfer in few-shot learning. 

_Table 17._ Comparison results of different inter-task attention architecture design strategies. We report 5-shot accuracy on _mini_ ImageNet for different choices. 

|**Method**|**BackBone**|**Acc. (%)**|
|---|---|---|
|IT-att (Wang et al.,2022a)<br>_ViT-Small_<br>88_._70_±_0_._50<br>Global task vector<br>_ViT-Small_<br>89_._05_±_0_._48<br>PTA<br>_ViT-Small_<br>**90**_._**02**_±_**0**_._**44**|||



## **H. Computational Analysis** 

## **H.1. Inference Time Comparison** 

We have conducted a detailed comparative analysis of the computational efficiency between our MetaFormer and other state-of-the-art inductive methods for 5-way 1-shot and 5-way 5-shot scenarios on the miniImageNet, as presented in Table 3. The evaluation of inference latency is conducted on an NVIDIA RTX A6000 GPU. For FewTURE, we set the optimal inner loop steps to 15 for 5-way 1-shot and to 20 for 5-way 5-shot. Our analysis reveals that MetaFomer achieves superior computational efficiency compared to the methods evaluated. Note that previous works (Bendou et al., 2022; Qi et al., 2021; Zhu & Koniusz, 2023) utilize an inference-time augmentation technique, which involves performing inference 30 times for each randomly cropped augmented sample and then averaging the features for the final prediction. To ensure a fair comparison of inference-time latency with these state-of-the-art methods, we pre-extract features for both protoLP (Zhu & Koniusz, 2023) and MetaFormer and calculate the inference time in the 5-way 5-shot scenario. The results exhibit the superior computational efficiency of MetaFormer-A, with an inference time of 34.44 ms compared to protoLP’s 40.61 ms, which is hindered by its reliance on time-consuming label propagation. 

## **H.2. Training time Comparison** 

We further compare the training clock times between our MetaFormer-I and FewTURE on _mini_ ImageNet in the 5-way 1-shot configuration. The results, presented in Table 18, indicate that our approach incurs a slightly higher training time. Since MetaFormer-I enhances adaptability to diverse tasks without the need for compute-intensive inference-time fine-tuning, this capability potentially leads to long-term savings in computational resources. 

_Table 18._ Training time comparison on _mini_ ImageNet. 

|**Method**|**Training Clock Time**|
|---|---|
|FewTURE|6.0 hours (200 epochs)|
|MetaFormer-I|8.7 hours (200 epochs)|



## **H.3. The Number of Parameters Comparison** 

In Table 19, we present a selection of representative and state-of-the-art methods, detailing their number of backbone parameters and the total number of parameters. Simply increasing more parameters by changing the backbone architecture 

20 

## **One Meta-tuned Transformer is What You Need for Few-shot Learning** 

does not necessarily lead to better performance. We observe that FewTURE (Hiller et al., 2022) and HCTransformers (He et al., 2022b), despite possessing a larger number of parameters, markedly lag behind the proposed MetaFormer. We also conduct a comparative analysis with an ablated version, achieved by naively augmenting the number of layers in ViT-Small to make it comparable with the proposed MetaFormer. The results presented substantiate that merely increasing parameters cannot fully address the challenges inherent in few-shot learning. In fact, such augmentation may even elevate the risk of overfitting. It’s crucial to demonstrate that our enhancements are not merely due to an increased parameter count. MetaFormer is cost-effective, achieving remarkable performance gains over the previous meta-learning SOTA FewTURE (Hiller et al., 2022) with only a modest increment in parameter count. Also, note that the conversion from inductive to autoregressive version leads to no extra parameters, further emphasizing its efficiency. 

_Table 19._ Comparsion of state-of-the-art methods with the number of parameters. 

|**Method**|**Backbone**|_≈_**# Params**|**# Total Params**|**_mini_ImageNet**<br>**1-shot**<br>**5-shot**|
|---|---|---|---|---|
|FEAT (Ye et al.,2020)<br>_ResNet-12_<br>12.4 M<br>14.1 M<br>66_._78_±_0_._20<br>82_._05_±_0_._14<br>FEAT (Ye et al.,2020)<br>_WRN-28-10_<br>36.5 M<br>38.1 M<br>65_._10_±_0_._20<br>81_._11_±_0_._14<br>FewTURE (Hiller et al.,2022)<br>_ViT-Small_<br>21 M<br>22 M<br>68_._02_±_0_._88<br>84_._51_±_0_._53<br>FewTURE (Hiller et al.,2022)<br>_Swin-Tiny_<br>28 M<br>29 M<br>72_._40_±_0_._78<br>86_._38_±_0_._49<br>HCTransformers (He et al.,2022b)<br>3_×ViT-Small_<br>63 M<br>63 M<br>74_._74_±_0_._17<br>89_._19_±_0_._13<br>SMKD(Lin et al.,2023)<br>_ViT-Small_<br>21 M<br>21 M<br>74_._28_±_0_._18<br>88_._89_±_0_._09|||||
|ViT with more layers|_ViT-Small_|21 M|25.2 M|69_._75_±_0_._71<br>84_._12_±_0_._56|
|MetaFormer-I (Ours)|_ViT-Small_|21 M|24.5 M|**75**_._**78**_±_**0**_._**71**<br>**90**_._**02**_±_**0**_._**44**|
|MetaFormer-A(Ours)|_ViT-Small_|21 M|24.5 M|**84**_._**78**_±_**0**_._**79**<br>**91**_._**39**_±_**0**_._**42**|



## **I. Comparison with CNN-based Meta-Learning Methods** 

In this section, we give more in-depth discussions concerning prior research in the realm of meta-learning that incorporates vision transformers as their foundation architectures. We posit that the challenge of architectural inconsistency partially accounts for the limited research in the realm of meta-learning grounded on ViT. In the Table 20, we adapt FiLM, a technique commonly employed in CNN-based meta-learning for task adaptation through conditioned batch normalization (Requeima et al., 2019; Oreshkin et al., 2018), into layer normalization layers of ViT for task conditioning. As shown in the table, our experiments reveal a performance drop when ViT was applied with FiLM. Another key challenge is the increased parameter requirement of ViT. FewTURE (Hiller et al., 2022), as expounded in the Related Work section, is the pioneering work that tailors to ViT via inner-loop token importance reweighting, and addresses the second challenge via self-supervised pre-training on the meta-training dataset. Our approach, empowering sample-to-sample and task-to-task interaction, further improves the accuracy substantially. 

_Table 20._ Comparison results with different meta-learning approaches for Vision Transformer on the _mini_ Imagenet. 

|**Method**|**BackBone**|**5-way 1-shot**|
|---|---|---|
|Vanilla ViT<br>_ViT-Small_<br>69_._03_±_0_._71<br>ViT+FiLM<br>_ViT-Small_<br>58_._75_±_0_._73<br>MetaFormer-I<br>_ViT-Small_<br>**75**_._**78**_±_**0**_._**71**<br>MetaFormer-A<br>_ViT-Small_<br>**84**_._**78**_±_**0**_._**79**|||



## **J. Illustration of Decoupled Patch-Sample Attention** 

Figure 5 compares the complexity between the joint patch-sample attention and our decoupled patch-sample attention. 

## **K. Additional Analysis of Task Probe Vector** 

As shown in Figure 6, we analyze the task probe vectors on _mini_ ImageNet across different tasks sampled from meta-train and meta-test sets. The visualization effectively underscores the efficacy of the learned task probe vectors in capturing task relationships. For example, we observe a higher similarity in task features among tasks involving car tires, dogs, and long-legged animals. This demonstrates MetaFormer’s capability in discerning and utilizing task dynamics. 

21 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning_images/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning.pdf-0022-01.png)


**----- Start of picture text -----**<br>
Sample Sample<br>(a) Joint Patch-Sample Attention (b) Decoupled Patch-Sample attention<br>𝑂(((𝑁𝐾+ 𝑀)𝐿) [!] ) 𝑂((𝑁𝐾+ 𝑀) · 𝐿 [!] +(𝑁𝐾+ 𝑀) [!] · L)<br>Patch Patch<br>**----- End of picture text -----**<br>


_Figure 5._ **Complexity comparison with joint patch-sample attention approaches.** For a _N_ -way _K_ -shot task with _M_ queries, our method decouples the patch-sample attention by first performing self-attention between _L_ patches within each image to aggregate spatial information and then computing sample interactions across all patches at the same patch location to capture the similarities and variances among samples. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning_images/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning.pdf-0022-03.png)


**----- Start of picture text -----**<br>
(a) Task probe vector similarity (b) Task visualization<br>0<br>1<br>2<br>Task ID Task ID 3<br>4<br>5<br>**----- End of picture text -----**<br>


_Figure 6._ Interpretation of task probe vector. Each task is randomly selected from _mini_ ImageNet. (a) We show the similarity heatmap between task probe vectors, where deeper color means higher similarity. (b) We show the visualization of the corresponding tasks. 

22 

**One Meta-tuned Transformer is What You Need for Few-shot Learning** 

## **L. More Qualitative Results** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning_images/icml_2024_One_Meta-tuned_Transformer_is_What_You_Need_for_Few-shot_Learning.pdf-0023-02.png)


**----- Start of picture text -----**<br>
Input Images Patch Attention Sample Attention Task Attention Holistic Attention<br>Bookshop<br>Bulldog<br>Trifle<br>House Finch<br>Walker Hound<br>Boxer<br>Electric Guitar<br>**----- End of picture text -----**<br>


_Figure 7._ More visualization of our holistic attention mechanism. Three attention modules collaboratively focusing on task-specific foreground regions. 

23 

