## **Molecule Generation with Fragment Retrieval Augmentation** 

**Seul Lee**[1] _[∗]_ **Karsten Kreis**[2] **Srimukh Prasad Veccham**[2] **Meng Liu**[2] **Danny Reidenbach**[2] **Saee Paliwal**[2] **Arash Vahdat**[2] _[†]_ **Weili Nie**[2] _[†]_ 1KAIST 2NVIDIA `seul.lee@kaist.ac.kr` 

```
{kkreis,sveccham,menliu,dreidenbach,saeep,avahdat,wnie}@nvidia.com
```

## **Abstract** 

Fragment-based drug discovery, in which molecular fragments are assembled into new molecules with desirable biochemical properties, has achieved great success. However, many fragment-based molecule generation methods show limited exploration beyond the existing fragments in the database as they only reassemble or slightly modify the given ones. To tackle this problem, we propose a new fragmentbased molecule generation framework with retrieval augmentation, namely _Fragment Retrieval-Augmented Generation_ ( _f_ -RAG). _f_ -RAG is based on a pre-trained molecular generative model that proposes additional fragments from input fragments to complete and generate a new molecule. Given a fragment vocabulary, _f_ -RAG retrieves two types of fragments: (1) _hard fragments_ , which serve as building blocks that will be explicitly included in the newly generated molecule, and (2) _soft fragments_ , which serve as reference to guide the generation of new fragments through a trainable _fragment injection module_ . To extrapolate beyond the existing fragments, _f_ -RAG updates the fragment vocabulary with generated fragments via an iterative refinement process which is further enhanced with post-hoc genetic fragment modification. _f_ -RAG can achieve an improved exploration-exploitation trade-off by maintaining a pool of fragments and expanding it with novel and high-quality fragments through a strong generative prior. 

## **1 Introduction** 

The goal of small molecule drug discovery is to discover molecules with specific biochemical target properties, such as synthesizability [9], non-toxicity [40], solubility [31] and binding affinity, in the vast chemical space. Fragment-based drug discovery (FBDD) has been considered as an effective approach to explore the chemical space and has resulted in many successful marketed drugs [27]. Contrary to high-throughput screening (HTS) [41] that searches from a library of drug-like molecules, FBDD constructs a library of molecular fragments to synthesize new molecules beyond the existing molecule library, leading to a better chemical coverage [5]. 

Recently, generative models have been adopted in the field of FBDD to accelerate the process of searching for drug 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation_images/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation.pdf-0001-08.png)


**----- Start of picture text -----**<br>
f -RAG (ours)<br>Optimization Genetic GFN<br>performance Mol GA<br>REINVENT<br>Novelty<br>Diversity<br>Synthesizability<br>**----- End of picture text -----**<br>


Figure 1: **A radar plot of target properties.** _f_ -RAG strikes better balance among optimization performance, diversity, novelty, and synthesizability than the state-of-the-art techniques on the PMO benchmark [10]. 

- _∗_ Work done during an internship at NVIDIA. _†_ Equal advising. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation_images/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Fragment update<br>SAFE-GPT with retrieval augmentation<br>Hard fragments Input<br>* embedding New molecule<br>Augmented<br>Fragment * Early layers  Soft fragment Fragment  embedding Later layers<br>vocabulary of SAFE-GPT embeddings injection of SAFE-GPT<br>* Soft fragments<br>*<br>*<br>* * * Molecule<br>Fragment Genetic fragment modification update<br>update<br>* * Decompose Mutation Crossover populationMolecule<br>*<br>* New molecule Parent molecules<br>Molecule update<br>**----- End of picture text -----**<br>


Figure 2: **The overall framework of** _f_ **-RAG.** After an initial fragment vocabulary is constructed from an existing molecule library, two types of fragments are retrieved during generation. Hard fragments are explicitly included in the newly generated molecules, while soft fragments implicitly guide the generation of new fragments. SAFE-GPT generates a molecule using hard fragments as input, while the fragment injection module in the middle of the SAFE-GPT layers injects the embeddings of soft fragments into the input embedding. After the generation, the molecule population and fragment vocabulary are updated with the newly generated molecule and its fragments, respectively. The exploration is further enhanced with genetic fragment modification, which also updates the fragment vocabulary and molecule population. 

candidates [18, 45, 46, 30, 20, 11, 25]. These methods reassemble or slightly modify the fragments to generate new molecules with the knowledge of the generative model. As the methods are allowed to exploit chemical knowledge in the form of fragments to narrow the search space, they have shown meaningful success in generating optimized molecules. However, their performance is largely limited by the existing library of molecular fragments, where discovering new fragments is either impossible or highly limited. Some of the methods [14, 38, 25] suggested making modifications to existing fragments, but the modifications are only applied to small and local substructures and still heavily dependent on the existing fragments. The limited exploration beyond known fragments greatly hinders the possibility of generating diverse and novel drug candidates that may exhibit better target properties. Therefore, it is a challenge to improve the ability of discovering novel high-quality fragments, while exploiting existing chemical space effectively. 

To this end, we propose a fragment-based molecule generation framework leveraging retrievalaugmented generation (RAG) [26], namely _Fragment Retrieval-Augmented Generation_ ( _f_ -RAG). As shown in Figure 2, _f_ -RAG augments the pre-trained molecular language model SAFE-GPT [34] with two types of retrieved fragments: _hard fragments_ and _soft fragments_ for a better explorationexploitation trade-off. Specifically, we first construct a fragment vocabulary by decomposing known molecules from the existing library into fragments and scoring the fragments by measuring their contribution to target properties. From the fragment vocabulary, _f_ -RAG first retrieves fragments that will be explicitly included in the new molecule (i.e., hard fragments). Hard fragments serve as the input context to the molecular language model that predicts the remaining fragments. In addition to retrieval of hard fragments, _f_ -RAG retrieves fragments that will not be part of the generated molecule but provide informative guidance on predicting novel, diverse molecules (i.e., soft fragments). Concretely, the embeddings of soft fragments are fused with the embeddings of hard fragments (or input embeddings) through a lightweight _fragment injection module_ in the middle of SAFE-GPT. The fragment injection module allows SAFE-GPT to generate new fragments by referring to the information conveyed by soft fragments. 

During training, we only update the fragment injection module while keeping SAFE-GPT frozen. Inspired by Wang et al. [42], we train _f_ -RAG to learn how to leverage the retrieved fragments for molecule generation, using a self-supervised loss that predicts the most similar one in the set of soft fragments. At inference, _f_ -RAG dynamically updates the fragment vocabulary with newly generated fragments via an iterative refinement process. To further enhance exploration, we propose to modify the generated fragments from SAFE-GPT with a post-hoc genetic fragment modification process. The proposed _f_ -RAG takes the advantages of both hard and soft fragment retrieval to achieve an improved exploration-exploitation trade-off. We verify _f_ -RAG on various molecular optimization tasks, by examining the optimization performance, along with diversity, novelty, and 

2 

synthesizability. As shown in Figure 1, _f_ -RAG exhibits the best balance across these essential considerations, demonstrating its applicability as a promising tool for drug discovery. 

We summarize our contributions as follows: 

- We introduce _f_ -RAG, a novel molecular generative framework that combines FBDD and RAG. 

- We propose a retrieval augmentation strategy that operates at the fragment level with two types of fragments, allowing fine-grained guidance to achieve an improved exploration-exploitation trade-off and generate high-quality drug candidates. 

- Through extensive experiments, we demonstrate the effectiveness of _f_ -RAG in various drug discovery tasks that simulate real-world scenarios. 

## **2 Related Work** 

**Fragment-based molecule generation.** Fragment-based molecular generative models refer to a class of methods that reassemble existing molecular substructures (i.e., fragments) to generate new molecules. Jin et al. [18] proposed to find big molecular substructures that satisfy the given chemical properties, and learn to complete the substructures into final molecules by adding small branches. Xie et al. [45] proposed to progressively add or delete fragments using Markov chain Monte Carlo (MCMC) sampling. Yang et al. [46] proposed to use reinforcement learning (RL) to assemble fragments, while Maziarz et al. [30], Kong et al. [20], and Geng et al. [11] used VAE-based techniques. Lee et al. [25] proposed to take the target chemical properties into account in the fragment vocabulary construction and used a combination of RL and a genetic algorithm (GA) to assemble and modify the fragments. On the other hand, graph-based GAs [14, 38] decompose parent molecules into fragments that are combined to generate an offspring molecule, and are also mutated with a small probability. Since fragment-based strategies are limited by generating molecules outside of the possible combinations of existing fragments, they suffer from limited exploration in the chemical space. Some of the methods [14, 38, 25] suggest making modifications to existing fragments to overcome this problem, but this is not a fundamental solution because the modifications are only local, still being based on the existing fragments. 

**Retrieval-augmented molecule generation.** Retrieval-augmented generation (RAG) [26] refers to a technique that retrieves context from external data databases to guide the generation of a generative model. Recently, RAG has gained attention as a means to enhance accuracy and reliability of large language models (LLMs) [6, 36, 3]. In the field of molecule optimization, Liu et al. [28] developed a text-based drug editing framework that repurposes a conversational language model for solving molecular tasks, where the domain knowledge is injected by a retrieval and feedback module. More related to us, Wang et al. [42] proposed to use a pre-trained molecular language model to generate molecules, while augmenting the generation with retrieved molecules that have high target properties. Contrary to this method that retrieves molecules, our proposed _f_ -RAG employs a fine-grained retrieval augmentation scheme that operates at the fragment level to achieve an improved exploration-exploitation trade-off. 

## **3 Method** 

In this section, we introduce our Fragment Retrieval-Augmented Generation ( _f_ -RAG) framework. _f_ -RAG aims to generate optimized molecules by leveraging existing chemical knowledge through RAG, while exploring beyond the known chemical space under the fragment-based drug discovery (FBDD) paradigm. We first introduce the hard fragment retrieval in Sec. 3.1. Then, we present the soft fragment retrieval in Sec. 3.2. Lastly, we describe the genetic fragment modification in Sec. 3.3. 

## **3.1 Hard Fragment Retrieval** 

Given a set of _N_ molecules _xi_ and their corresponding properties _yi ∈_ [0 _,_ 1], denoted as _D_ = _{_ ( _xi, yi_ ) _}[N] i_ =1[, we first construct a fragment vocabulary.][We adopt an arm-linker-arm slicing algorithm] provided by Noutahi et al. [34] which decomposes a molecule _x_ into three fragments: two arms _F_ arm (i.e., fragments that have one attachment point) and one linker _F_ linker (i.e., a fragment that has two attachment points). Decomposing molecules into arms and linkers (or scaffolds) is a popular approach 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation_images/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Arm Linker design<br>vocab * SAFE-GPT<br>vocabArm *<br>[*]c1cccc(Br)c1.[*]C(=O)C#CCN1CCCCC1 c13cccc(Br)c1.C2(=O)C#CCN1CCCCC1.N2S3(=O)=O<br>Arm * * Motif extension<br>vocab *<br>SAFE-GPT<br>*<br>Linker<br>vocab<br>[*]C1C(C(=O)OCCCC)=C(C)Nc2ncnn21 C13C(C(=O)OCCCC)=C(C)Nc2ncnn21.C13=NCCS1<br>**----- End of picture text -----**<br>


Figure 3: **Hard fragment retrieval** of _f_ -RAG. With a probability of 50%, _f_ -RAG either retrieves two arms as hard fragments for linker design **(top)** or one arm and one linker as hard fragments for motif extension **(bottom)** . 

utilized in various drug discovery strategies, such as scaffold decoration or scaffold hopping [37]. We ignored the molecules in the training set that cannot be decomposed, and a set of arms _F_ arm = _{F_ arm _,j}_[2] _j_ =1 _[N]_[and a set of linkers] _[ F]_[linker][=] _[{][F]_[linker] _[,j][}][N] j_ =1[are obtained after the algorithm is applied] to the molecules _{xi}[N] i_ =1[.][Subsequently, we calculate the score of each fragment] _[ F][j][∈F]_[arm] _[ ∪F]_[linker] using the average property of all molecules containing _Fj_ as their substructure as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation_images/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation.pdf-0004-03.png)


where score( _Fj_ ) _∈_ [0 _,_ 1], and _S_ ( _Fj_ ) = _{_ ( _x, y_ ) _∈D_ : _Fj_ is a fragment of _x}_ . Intuitively, the fragment score evaluates the contribution of a given fragment to the target property of the whole molecule of which it is a part. From _F_ arm and _F_ linker, we choose the top- _N_ frag fragments based on the score to construct the arm fragment vocabulary _V_ arm _⊂F_ arm and the linker fragment vocabulary _V_ linker _⊂F_ linker, respectively. 

Given the fragment vocabularies _V_ arm and _V_ linker consisting of high-property fragments, two hard fragments are randomly retrieved from the vocabularies. The hard fragments together form a partial molecular sequence that serves as input to a pre-trained molecular language model. In this work, we employ Sequential Attachment-based Fragment Embedding (SAFE) [34] as the molecular representation and SAFE-GPT [34] as the backbone generative model of _f_ -RAG. SAFE is a noncanonical version of simplified molecular-input line-entry system (SMILES) [43] that represents molecules as a sequence of dot-connected fragments. Importantly, the order of fragments in a SAFE string does not affect the molecular identity. Using the SAFE representation, _f_ -RAG forces the hard fragments to be included in a newly generated molecule by providing them as an input sequence to the language model to complete the rest of the sequence. 

During generation, with a probability of 50%, _f_ -RAG either (1) retrieves two hard fragments from _V_ arm or (2) retrieves one from _V_ arm and one from _V_ linker. In the former case, _f_ -RAG performs linker design, which generates a new fragment that links the input fragments. In the latter case, _f_ -RAG first randomly selects an attachment point in the retrieved linker and combines it with the retrieved arm to form a single fragment, and then performs motif extension, which generates a new fragment that completes the molecule (Figure 3). 

## **3.2 Soft Fragment Retrieval** 

Given two hard fragments as input, the molecular language model generates one new fragment to complete a molecule. Instead of relying solely on the model to generate the new fragment, we propose to augment the generation with the information of _K_ retrieved fragments, which we refer to as soft fragments, to guide the generation. Specifically, if the two hard fragments are all arms, _f_ -RAG randomly retrieves soft fragments from _V_ linker. If one of the hard fragments is an arm and another is a linker, _f_ -RAG randomly retrieves soft fragments from _V_ arm. Using up to the _L_ -th layer of the lauguage model LM[0:] _[L]_ , the embeddings of the input sequence _x_ input and the soft fragments _{F_ soft _,k}[K] k_ =1[are obtained as follows:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation_images/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation.pdf-0004-09.png)


4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation_images/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation.pdf-0005-00.png)


**----- Start of picture text -----**<br>
SAFE-GPT with retrieval augmentation<br>Input<br>Input sequence embedding Target sequence<br>𝐹!.𝐹" Augmented  𝐹!"##<br>Training molecule𝑥= 𝐹".𝐹$.𝐹! Soft fragments𝐹#𝐹"$$# of SAFE-GPTEarly layers  Soft fragmentembeddings Fragment injection embedding of SAFE-GPTLater layers  !𝐹!"## Cross entropy<br>𝐹##$$<br>⋯<br>𝐹#%$$<br>**----- End of picture text -----**<br>


Figure 4: **The self-supervised training process of the fragment injection module of** _f_ **-RAG.** _F[k]_[NN] denotes the _k_ -th most similar fragment to _F_ . Using _F_ 1 and _F_ 2 as hard fragments, while using _F_ 3 and its neighbors _{F_ 3 _[kNN] }k[K]_ =2[as soft fragments, the training objective is to predict] _[ F]_[ 1] 3 _[NN]_ . 

Subsequently, _f_ -RAG injects the embeddings of soft fragments through a trainable _fragment injection module_ . Following Wang et al. [42], the fragment injection module uses cross-attention to fuse the embeddings of the input sequence and soft fragments as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation_images/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation.pdf-0005-03.png)


where FI is the fragment injection module, Query, Key, and Value are multi-layer perceptrons (MLPs), and _d_ Key is the output dimension of Key. Next, a molecule is generated by decoding the augmented embedding _**h**_ using the later layers of the language model as _x_ new = LM _[L]_[+1:] _[L][T]_ ( _**h**_ ), where _LT_ is the total number of layers of the model. With the fragment injection module, _f_ -RAG can utilize information of soft fragments to generate novel fragments which are also likely to contribute to the high target properties. During generation, the fragment vocabulary is dynamically updated through an iterative process that scores newly generated fragments based on Eq. (1) and replaces fragments in the fragment vocabulary to the top- _N_ frag fragments. 

Next, we need to train _f_ -RAG to learn how to augment the retrieved soft fragments into the molecule generation. To retain the high generation quality of SAFE-GPT and make the training process efficient, we keep the backbone language model frozen and only train the lightweight fragment injection module. Inspired by Wang et al. [42], we propose a new self-supervised objective that predicts the most similar fragment to the input fragments. Specifically, each molecular sequence _x_ in the training set is first decomposed into fragment sequences ( _F_ 1 _, F_ 2 _, F_ 3) with a random permutation between the fragments, using the same slicing algorithm used in the vocabulary construction. Importantly, using the SAFE representation, _x_ can be simply represented by connecting its fragments with dots as _F_ 1 _.F_ 2 _.F_ 3. We consider the first two fragments as hard fragments. Given the remaining fragment _F_ 3, we retrieve its _K_ most similar fragments _{F_ 3 _[kNN] }[K] k_ =1[from the training fragment pool.][Here, we use the pairwise] Tanimoto similarity using Morgan fingerprints of radius 2 and 1024 bits. Using the hard fragments as the input sequence as _F_ 1 _.F_ 2, the objective is to predict the most similar fragment _F_ 3[1] _[NN]_ utilizing the original fragment and the next _K −_ 1 most similar fragments _{F_ 3 _[kNN] }[K] k_ =2[as the soft fragments.] The training process is illustrated in Figure 4, and the details are provided in Section D.1. 

Note that the training of the fragment injection module is target property-agnostic, as the fragments used for training are independent of the target property. In contrast, the fragment vocabularies used for generation are target property-specific, as it is constructed using the scoring function in Eq. (1). This allows _f_ -RAG to effectively generate optimized molecules across different target properties without any retraining. 

## **3.3 Genetic Fragment Modification** 

To further enhance exploration in the chemical space, we propose to modify the generated fragments with a post-hoc genetic algorithm (GA). Specifically, we adopt the operations of Graph GA [14]. We first initialize the population _P_ with the top- _N_ mol molecules generated by our fragment retrievalaugmented SAFE-GPT based on the target property _y_ . Parent molecules are then randomly selected from the population and offspring molecules are generated by the crossover and mutation operations (see Jensen [14] for more details). The offspring molecules can have new fragments not contained in the initial fragment vocabulary, and the fragment vocabularies _V_ arm and _V_ linker are again updated by the top- _N_ frag fragments based on the scores of Eq. (1). In the subsequent generation, the population _P_ is updated with the generated molecules so far by both SAFE-GPT and GA. 

5 

As shown in Figure 2, _f_ -RAG generates desirible molecules through multiple cycles of (1) the SAFE-GPT generation augmented with the hard fragment retrieval (Section 3.1) and the soft fragment retrieval (Section 3.2), and (2) the GA generation (Section 3.3). Through this interplay of hard fragment retrieval, soft fragment retrieval, and the genetic fragment modification, _f_ -RAG can exploit existing chemical knowledge through the form of fragments both explicitly and implicitly, while exploring beyond initial fragments by the dynamic vocabulary update. We summarize the generation process of _f_ -RAG in Algorithm 1 in Section C. 

## **4 Experiments** 

We validate _f_ -RAG on molecule generation tasks that simulate various real-world drug discovery problems. We first conduct experiments on the practical molecular optimization (PMO) benchmark [10] in Section 4.1. We then conduct experiments to generate novel molecules that have high binding affinity, drug-likeness, and synthesizability in Section 4.2. We further perform analyses in Section 4.3. 

## **4.1 Experiments on PMO Benchmark** 

**Setup.** We demonstrate the efficacy of _f_ -RAG on the 23 tasks from the PMO benchmark. Following the standard setting of the benchmark, we set the maximum number of oracle calls to 10,000 and evaluate **optimization performance** with the area under the curve (AUC) of the average property scores of the top-10 molecules versus oracle calls. In addition, we evaluate **diversity** , **novelty** , and **synthesizability** of generated molecules, other essential considerations in drug discovery. We use the Therapeutics Data Commons (TDC) library [12] to calculate diversity. Following previous works [18, 45, 24, 25], novelty is defined as the fraction of molecules that have the maximum Tanimoto similarity less than 0 _._ 4 with the training molecules. We use the synthetic accessibility (SA) [8] score of the TDC library to measure synthesizability. These values are measured using the top-100 generated molecules, following the setting of the benchmark. Further explanation on the evaluation metrics and experimental setup are provided in Section D.3. 

**Baselines.** We employ the top-7 methods reported by the PMO benchmark and two recent state-ofthe-art methods as our baselines. **Graph GA** [14] is a GA with a fragment-level crossover operation, and **Mol GA** [38] is a more hyperparameter-tuned version of Graph GA. **Genetic GFN** [19] is a method that uses Graph GA to guide generation of a GFlowNet. **REINVENT** [35] is a SMILESbased RL model and **SELFIES-REINVENT** is a modified REINVENT that uses the self-referencing embedded strings (SELFIES) [21] representation. **GP BO** [39] is a method that optimizes the Gaussian process acquisition function with Graph GA. **STONED** [33] is a GA-based model that operates on SELFIES strings. **LSTM HC** [7] is a SMILES-based LSTM model. **SMILES GA** [47] is a GA whose genetic operations are based on the SMILES context-free grammar. 

**Results.** The results of optimization performance are shown in Table 1. _f_ -RAG outperforms the previous methods in terms of the sum of the AUC top-10 values and achieves the highest AUC top-10 values in 12 out of 23 tasks, demonstrating that the proposed combination of hard fragment retrieval, soft fragment retrieval, and genetic fragment modification is highly effective in discovering optimized drug candidates that have high target properties. On the other hand, the average scores of diversity, novelty, and synthesizability of the generated molecules are summarized in Table 2, and the full results are presented in Tables 4, 5, and 6. As shown in these tables, _f_ -RAG achieves the best diversity and synthesizability, and the second best novelty. Notably, _f_ -RAG shows the highest diversity in 12 out of 23 tasks, and the highest synthesizability in 19 out of 23 tasks. Note that the high novelty value of Mol GA comes at the cost of other important factors, i.e., optimization performance, diversity, and synthesizability. The essential considerations in drug discovery often conflict with each other, making the drug discovery problem challenging. These trade-offs are also visualized in Figure 1 and Figure 7, and _f_ -RAG effectively improves the trade-offs by utilizing existing fragments while dynamically updating the fragment vocabulary with newly proposed fragments. 

## **4.2 Optimization of Docking Score under QED, SA, and Novelty Constraints** 

**Setup.** Following Lee et al. [24] and Lee et al. [25], we validate _f_ -RAG in a set of tasks that aim to optimize the binding affinity against a target protein while also maintaining high drug-likeness, 

6 

Table 1: **PMO AUC top-10 results.** The results are the mean and standard deviation of 3 independent runs. The results for Genetic GFN [19] and Mol GA [38] are taken from the respective original papers and the results for other baselines are taken from Gao et al. [10]. The best results are highlighted in bold. 

|Oracle|_f_-RAG (ours)|Genetic GFN|Mol GA|REINVENT|Graph GA|
|---|---|---|---|---|---|
|albuterol_similarity|**0.977**_±_0.002|0.949_±_0.010|0.896_±_0.035|0.882_±_0.006|0.838_±_0.016|
|amlodipine_mpo|0.749_±_0.019|**0.761**_±_0.019|0.688_±_0.039|0.635_±_0.035|0.661_±_0.020|
|celecoxib_rediscovery|0.778_±_0.007|**0.802**_±_0.029|0.567_±_0.083|0.713_±_0.067|0.630_±_0.097|
|deco_hop|**0.936**_±_0.011|0.733_±_0.109|0.649_±_0.025|0.666_±_0.044|0.619_±_0.004|
|drd2|**0.992**_±_0.000|0.974_±_0.006|0.936_±_0.016|0.945_±_0.007|0.964_±_0.012|
|fexofenadine_mpo|**0.856**_±_0.016|**0.856**_±_0.039|0.825_±_0.019|0.784_±_0.006|0.760_±_0.011|
|gsk3b|**0.969**_±_0.003|0.881_±_0.042|0.843_±_0.039|0.865_±_0.043|0.788_±_0.070|
|isomers_c7h8n2o2|0.955_±_0.008|**0.969**_±_0.003|0.878_±_0.026|0.852_±_0.036|0.862_±_0.065|
|isomers_c9h10n2o2pf2cl|0.850_±_0.005|**0.897**_±_0.007|0.865_±_0.012|0.642_±_0.054|0.719_±_0.047|
|jnk3|**0.904**_±_0.004|0.764_±_0.069|0.702_±_0.123|0.783_±_0.023|0.553_±_0.136|
|median1|0.340_±_0.007|**0.379**_±_0.010|0.257_±_0.009|0.356_±_0.009|0.294_±_0.021|
|median2|**0.323**_±_0.005|0.294_±_0.007|0.301_±_0.021|0.276_±_0.008|0.273_±_0.009|
|mestranol_similarity|0.671_±_0.021|**0.708**_±_0.057|0.591_±_0.053|0.618_±_0.048|0.579_±_0.022|
|osimertinib_mpo|**0.866**_±_0.009|0.860_±_0.008|0.844_±_0.015|0.837_±_0.009|0.831_±_0.005|
|perindopril_mpo|**0.681**_±_0.017|0.595_±_0.014|0.547_±_0.022|0.537_±_0.016|0.538_±_0.009|
|qed|0.939_±_0.001|**0.942**_±_0.000|0.941_±_0.001|0.941_±_0.000|0.940_±_0.000|
|ranolazine_mpo|**0.820**_±_0.016|0.819_±_0.018|0.804_±_0.011|0.760_±_0.009|0.728_±_0.012|
|scaffold_hop|0.576_±_0.014|**0.615**_±_0.100|0.527_±_0.025|0.560_±_0.019|0.517_±_0.007|
|sitagliptin_mpo|0.601_±_0.011|**0.634**_±_0.039|0.582_±_0.040|0.021_±_0.003|0.433_±_0.075|
|thiothixene_rediscovery|**0.584**_±_0.009|0.583_±_0.034|0.519_±_0.041|0.534_±_0.013|0.479_±_0.025|
|troglitazone_rediscovery|0.448_±_0.017|**0.511**_±_0.054|0.427_±_0.031|0.441_±_0.032|0.390_±_0.016|
|valsartan_smarts|**0.627**_±_0.058|0.135_±_0.271|0.000_±_0.000|0.178_±_0.358|0.000_±_0.000|
|zaleplon_mpo|0.486_±_0.004|**0.552**_±_0.033|0.519_±_0.029|0.358_±_0.062|0.346_±_0.032|
|Sum|**16.928**|16.213|14.708|14.196|13.751|
|||||||
|Oracle|SELFIES-<br>REINVENT|GP BO|STONED|LSTM HC|SMILES GA|
|albuterol_similarity|0.826_±_0.030|0.898_±_0.014|0.745_±_0.076|0.719_±_0.018|0.661_±_0.066|
|amlodipine_mpo|0.607_±_0.014|0.583_±_0.044|0.608_±_0.046|0.593_±_0.016|0.549_±_0.009|
|celecoxib_rediscovery|0.573_±_0.043|0.723_±_0.053|0.382_±_0.041|0.539_±_0.018|0.344_±_0.027|
|deco_hop|0.631_±_0.012|0.629_±_0.018|0.611_±_0.008|0.826_±_0.017|0.611_±_0.006|
|drd2|0.943_±_0.005|0.923_±_0.017|0.913_±_0.020|0.919_±_0.015|0.908_±_0.019|
|fexofenadine_mpo|0.741_±_0.002|0.722_±_0.005|0.797_±_0.016|0.725_±_0.003|0.721_±_0.015|
|gsk3b|0.780_±_0.037|0.851_±_0.041|0.668_±_0.049|0.839_±_0.015|0.629_±_0.044|
|isomers_c7h8n2o2|0.849_±_0.034|0.680_±_0.117|0.899_±_0.011|0.485_±_0.045|0.913_±_0.021|
|isomers_c9h10n2o2pf2cl|0.733_±_0.029|0.469_±_0.180|0.805_±_0.031|0.342_±_0.027|0.860_±_0.065|
|jnk3|0.631_±_0.064|0.564_±_0.155|0.523_±_0.092|0.661_±_0.039|0.316_±_0.022|
|median1|0.355_±_0.011|0.301_±_0.014|0.266_±_0.016|0.255_±_0.010|0.192_±_0.012|
|median2|0.255_±_0.005|0.297_±_0.009|0.245_±_0.032|0.248_±_0.008|0.198_±_0.005|
|mestranol_similarity|0.620_±_0.029|0.627_±_0.089|0.609_±_0.101|0.526_±_0.032|0.469_±_0.029|
|osimertinib_mpo|0.820_±_0.003|0.787_±_0.006|0.822_±_0.012|0.796_±_0.002|0.817_±_0.011|
|perindopril_mpo|0.517_±_0.021|0.493_±_0.011|0.488_±_0.011|0.489_±_0.007|0.447_±_0.013|
|qed|0.940_±_0.000|0.937_±_0.000|0.941_±_0.000|0.939_±_0.000|0.940_±_0.000|
|ranolazine_mpo|0.748_±_0.018|0.735_±_0.013|0.765_±_0.029|0.714_±_0.008|0.699_±_0.026|
|scaffold_hop|0.525_±_0.013|0.548_±_0.019|0.521_±_0.034|0.533_±_0.012|0.494_±_0.011|
|sitagliptin_mpo|0.194_±_0.121|0.186_±_0.055|0.393_±_0.083|0.066_±_0.019|0.363_±_0.057|
|thiothixene_rediscovery|0.495_±_0.040|0.559_±_0.027|0.367_±_0.027|0.438_±_0.008|0.315_±_0.017|
|troglitazone_rediscovery|0.348_±_0.012|0.410_±_0.015|0.320_±_0.018|0.354_±_0.016|0.263_±_0.024|
|valsartan_smarts|0.000_±_0.000|0.000_±_0.000|0.000_±_0.000|0.000_±_0.000|0.000_±_0.000|
|zaleplon_mpo|0.333_±_0.026|0.221_±_0.072|0.325_±_0.027|0.206_±_0.006|0.334_±_0.041|
|Sum|13.471|13.156|13.024|12.223|12.054|



Table 2: **Top-100 diversity, top-100 novelty, and top-100 SA score results.** The results are the average values of all 23 tasks. The best results are highlighted in bold. 

|Metric|_f_-RAG (ours)|Genetic GFN|Mol GA|REINVENT|
|---|---|---|---|---|
|Average diversity_↑_|**0.532**|0.443|0.491|0.468|
|Average novelty_↑_|0.800|0.724|**0.845**|0.540|
|Average SA score_↓_|**2.026**|3.770|4.605|3.207|



7 

Table 3: **Novel top 5% docking score (kcal/mol) results.** The results are the means and standard deviations of 3 independent runs. The results for RationaleRL, PS-VAE, RetMol, and GEAM are taken from Lee et al. [25]. Other baseline results except for Genetic GFN are taken from Lee et al. [24]. Lower is better, and the best results are highlighted in bold. 

|Method|Targetprotein|
|---|---|
||parp1<br>fa7<br>5ht1b<br>braf<br>jak2|
|JT-VAE [16]<br>REINVENT [35]<br>Graph GA [14]<br>MORLD [15]<br>HierVAE [17]<br>GA+D [32]<br>MARS [45]<br>GEGL [1]<br>RationaleRL [18]<br>FREED [46]<br>PS-VAE [20]<br>MOOD [24]<br>RetMol [42]<br>GEAM [25]<br>Genetic GFN [19]|-9.482_±_0.132<br>-7.683_±_0.048<br>-9.382_±_0.332<br>-9.079_±_0.069<br>-8.885_±_0.026<br>-8.702_±_0.523<br>-7.205_±_0.264<br>-8.770_±_0.316<br>-8.392_±_0.400<br>-8.165_±_0.277<br>-10.949_±_0.532<br>-7.365_±_0.326<br>-10.422_±_0.670<br>-10.789_±_0.341<br>-10.167_±_0.576<br>-7.532_±_0.260<br>-6.263_±_0.165<br>-7.869_±_0.650<br>-8.040_±_0.337<br>-7.816_±_0.133<br>-9.487_±_0.278<br>-6.812_±_0.274<br>-8.081_±_0.252<br>-8.978_±_0.525<br>-8.285_±_0.370<br>-8.365_±_0.201<br>-6.539_±_0.297<br>-8.567_±_0.177<br>-9.371_±_0.728<br>-8.610_±_0.104<br>-9.716_±_0.082<br>-7.839_±_0.018<br>-9.804_±_0.073<br>-9.569_±_0.078<br>-9.150_±_0.114<br>-9.329_±_0.170<br>-7.470_±_0.013<br>-9.086_±_0.067<br>-9.073_±_0.047<br>-8.601_±_0.038<br>-10.663_±_0.086<br>-8.129_±_0.048<br>-9.005_±_0.155<br>_No hit found_<br>-9.398_±_0.076<br>-10.579_±_0.104<br>-8.378_±_0.044<br>-10.714_±_0.183<br>-10.561_±_0.080<br>-9.735_±_0.022<br>-9.978_±_0.091<br>-8.028_±_0.050<br>-9.887_±_0.115<br>-9.637_±_0.049<br>-9.464_±_0.129<br>-10.865_±_0.113<br>-8.160_±_0.071<br>-11.145_±_0.042<br>-11.063_±_0.034<br>-10.147_±_0.060<br>-8.590_±_0.475<br>-5.448_±_0.688<br>-6.980_±_0.740<br>-8.811_±_0.574<br>-7.133_±_0.242<br>-12.891_±_0.158<br>-9.890_±_0.116<br>-12.374_±_0.036<br>-12.342_±_0.095<br>-11.816_±_0.067<br>-9.227_±_0.644<br>-7.288_±_0.433<br>-8.973_±_0.804<br>-8.719_±_0.190<br>-8.539_±_0.592|
|_f_-RAG (ours)|**-12.945**_±_0.053<br>**-9.899**_±_0.205<br>**-12.670**_±_0.144<br>**-12.390**_±_0.046<br>**-11.842**_±_0.316|



synthesizability, and novelty. Following the previous works, we use docking score calculated by QuickVina 2 [2] with five protein targets, parp1, fa7, 5ht1b, braf, and jak2, to measure binding affinity. We use quantitative estimates of drug-likeness (QED) [4] and SA [8] to measure drug-likeness and synthesizability, respectively. Following the previous works, we set the target property _y_ as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation_images/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation.pdf-0008-03.png)


where DS[�] and SA[�] are the normalized DS and SA according to Eq. (7) in Section D.4. We generate 3,000 molecules and evaluate them using **novel top 5% DS (kcal/mol)** , which indicates the mean DS of the top 5% unique and novel hits. Here, novel hits are defined as the molecules that satisfy all the following criteria: (the maximum similarity with the training molecules) _<_ 0 _._ 4, DS _<_ (the median DS of known active molecules), QED _>_ 0 _._ 5, and SA _<_ 5. Further details are provided in Section D.4. 

**Baselines. JT-VAE** [16], **HierVAE** [17], **MARS** [45], **RationaleRL** [18], **FREED** [46], **PSVAE** [20], and **GEAM** [25] are the methods that first construct a fragment vocabulary using a molecular dataset, then learn to assemble those fragments to generate new molecules. On the other hand, **Graph GA** [14], **GEGL** [1], and **Genetic GFN** [19] are GA-based methods that utilize the information of fragments by adopting the fragment-based crossover operation. **GA+D** [32] is a discriminator-enhanced GA method that operates on the SELFIES representation. **RetMol** [42] is a retrieval-based method that uses retrieved example molecules to augment the generation of a pre-trained molecular language model. **REINVENT** [35] and **MORLD** [15] are RL models that operate on SMILES and graph molecular representations, respectively. **MOOD** [24] is a diffusion model that employs an out-of-distribution control to improve novelty. 

**Results.** The results are shown in Table 3. In all five tasks, _f_ -RAG outperforms all the baselines. Note that the evaluation metric, novel top 5% DS, is designed to reflect the nature of the drug discovery problem, which optimizes multiple target properties by finding a good balance between exploration and exploitation. The results demonstrate the superiority of _f_ -RAG in generating drug-like, synthesizable, and novel drug candidates that have high binding affinity to the target protein with the improved exploration-exploitation trade-off. We additionally visualize the generated molecules in Figure 8. 

**Additional comparison with GEAM.** To further justify the advantage of _f_ -RAG over GEAM, we additionally include the results of seven multi-property optimization (MPO) tasks in the PMO benchmark (Section 4.1) in Table 7. As we can see, _f_ -RAG significantly outperforms GEAM in all the tasks, validating its applicability to a wide range of drug discovery problems. 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation_images/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation.pdf-0009-00.png)


**----- Start of picture text -----**<br>
f -RAG (hard+soft+GA) f -RAG (hard+soft+GA) f -RAG<br>f -RAG (soft+GA) f -RAG (soft+GA) f -RAG (𝛿=0.8)<br>f -RAG (hard+GA) performanceOptimization f -RAG (hard+GA) performanceOptimization f -RAG (𝛿=0.6)<br>f -RAG (hard+soft) f -RAG (hard+soft) f -RAG (𝛿=0.4)<br>(a) (b) f -RAG (GA) (c)<br>Number of oracle calls Novelty Novelty<br>Avg. Top-10 Diversity Diversity<br>Synthesizability Synthesizability<br>**----- End of picture text -----**<br>


Figure 5: **(a) The optimization curves** in the deco_hop task of the PMO benchmark of the ablated _f_ -RAGs. Solid lines denote the mean and shaded areas denote the standard deviation of 3 independent runs. **(b) Overall results** of the ablated _f_ -RAGs. **(c) Results with different values of** _δ_ of the similarity-based fragment filter. 

## **4.3 Analysis** 

**Ablation study.** To verify the effect of each component of _f_ -RAG, we conduct experiments with various combinations of the components, i.e., hard fragment retrieval (hard), soft fragment retrieval (soft), and genetic fragment modification (GA), in Figure 5(a) and Figure 5(b). The detailed results are shown in Table 8 and Table 9. For _f_ -RAG (soft+GA) that does not utilize the hard fragment retrieval, we randomly initialized the fragment vocabularies instead of selecting the top- _N_ frag fragments based on Eq. (1) and let the model use random hard fragments. Note that even though _f_ -RAG (hard+soft), _f_ -RAG that does not utilize the genetic fragment modification, shows very high diversity, the value is not meaningful as the generated molecules are not highly optimized, not novel, and not synthesizable. The same applies to the high novelty of _f_ -RAG (soft+GA) and _f_ -RAG (GA). On the contrary, the full _f_ -RAG achieves the best optimization performance and synthesizability while showing reasonably high diversity and novelty. Importantly, _f_ -RAG outperforms _f_ -RAG (hard+GA) in all the metrics, especially for diversity and novelty, indicating the soft fragment retrieval aids the model in generating more diverse and novel drug candidates while maintaining their high target properties. 

**Controlling optimization performance-diversity trade-off.** It is well-known that molecular diversity is important in drug discovery, especially when the oracle has noise. However, optimization performance against the target property and molecular diversity are conflicting factors [10, 19]. To control this trade-off between optimization performance and diversity, we additionally introduce a _similarity-based fragment filtering_ strategy, which excludes similar fragments from the fragment vocabulary. Specifically, when updating the vocabulary, new fragments that have a higher Tanimoto similarity than _δ_ to fragments in the vocabulary are filtered out. Morgan fingerprints of radius 2 and 1024 bits are used to calculate the Tanimoto similarity. The results of the similarity-based fragment filter with different values of _δ_ are shown in Figure 5(c), verifying _f_ -RAG can increase diversity at the expense of optimization performance by controlling _δ_ . 

**Effect of the fragment vocabulary update.** To analyze the effect of the dynamic update of the fragment vocabulary during generation, we visualize the distribution of the docking scores of molecules generated by _f_ -RAG with or with the fragment vocabulary update in Figure 6. The dynamic update allows _f_ -RAG to generate more optimized molecules in terms of the target property. Notably, with the dynamic update, _f_ -RAG can discover molecules that have higher binding affinity to the target protein than the top molecule in the training set, and we visualize an example of such molecules in the figure. Note that the molecule also has low similarity (0 _._ 321) with the training molecules. This result demonstrates the explorability of _f_ -RAG to discover drug candidates that lie beyond the training distribution. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation_images/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation.pdf-0009-06.png)


**----- Start of picture text -----**<br>
With fragment update<br>Without fragment update<br>Best in<br>ZINC250k<br>Docking score (kcal/mol)<br>Density<br>**----- End of picture text -----**<br>


Figure 6: **DS distribution** of molecules generated by _f_ -RAG with or without the fragment vocabulary update in the jak2 task. 

9 

## **5 Conclusion** 

We proposed _f_ -RAG, a new fragment-based molecular generative framework that utilizes hard fragment retrieval, soft fragment retrieval, and genetic fragment modification. With hard fragment retrieval, _f_ -RAG can explicitly exploit the information contained in existing fragments that contribute to the target properties. With soft fragment retrieval, _f_ -RAG can balance between exploitation of existing chemical knowledge and exploration beyond the existing fragment vocabulary. Soft fragment retrieval with SAFE-GPT allows _f_ -RAG to propose new fragments that are likely to contribute to the target chemical properties. The proposed novel fragments are then dynamically incorporated in the fragment vocabulary throughout generation. This exploration is further enhanced with genetic fragment modification, which modifies fragments with genetic operations and updates the vocabulary. Through extensive experiments, we demonstrated the efficacy of _f_ -RAG to improve the explorationexploitation trade-off. _f_ -RAG outperforms previous methods, achieving state-of-the-art performance to synthesize diverse, novel, and synthesizable drug candidates with high target properties. 

## **Acknowledgements** 

We would like to thank the NVIDIA GenAIR team for their comments during the development of the framework. 

## **References** 

- [1] Sungsoo Ahn, Junsu Kim, Hankook Lee, and Jinwoo Shin. Guiding deep molecular optimization with genetic exploration. _Advances in neural information processing systems_ , 33:12008–12021, 2020. 

- [2] Amr Alhossary, Stephanus Daniel Handoko, Yuguang Mu, and Chee-Keong Kwoh. Fast, accurate, and reliable molecular docking with quickvina 2. _Bioinformatics_ , 31(13):2214–2216, 2015. 

- [3] Akari Asai, Sewon Min, Zexuan Zhong, and Danqi Chen. Retrieval-based language models and applications. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 6: Tutorial Abstracts)_ , pages 41–46, 2023. 

- [4] G Richard Bickerton, Gaia V Paolini, Jérémy Besnard, Sorel Muresan, and Andrew L Hopkins. Quantifying the chemical beauty of drugs. _Nature chemistry_ , 4(2):90–98, 2012. 

- [5] Marta Bon, Alan Bilsland, Justin Bower, and Kirsten McAulay. Fragment-based drug discovery—the importance of high-quality molecule libraries. _Molecular Oncology_ , 16(21): 3761–3777, 2022. 

- [6] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. Improving language models by retrieving from trillions of tokens. In _International conference on machine learning_ , pages 2206–2240. PMLR, 2022. 

- [7] Nathan Brown, Marco Fiscato, Marwin HS Segler, and Alain C Vaucher. Guacamol: benchmarking models for de novo molecular design. _Journal of chemical information and modeling_ , 59(3):1096–1108, 2019. 

- [8] Peter Ertl and Ansgar Schuffenhauer. Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. _Journal of cheminformatics_ , 1:1–11, 2009. 

- [9] Wenhao Gao and Connor W Coley. The synthesizability of molecules proposed by generative models. _Journal of chemical information and modeling_ , 60(12):5714–5723, 2020. 

- [10] Wenhao Gao, Tianfan Fu, Jimeng Sun, and Connor Coley. Sample efficiency matters: a benchmark for practical molecular optimization. _Advances in Neural Information Processing Systems_ , 35:21342–21357, 2022. 

10 

- [11] Zijie Geng, Shufang Xie, Yingce Xia, Lijun Wu, Tao Qin, Jie Wang, Yongdong Zhang, Feng Wu, and Tie-Yan Liu. De novo molecular generation via connection-aware motif mining. In _International Conference on Learning Representations_ , 2023. 

- [12] Kexin Huang, Tianfan Fu, Wenhao Gao, Yue Zhao, Yusuf H Roohani, Jure Leskovec, Connor W Coley, Cao Xiao, Jimeng Sun, and Marinka Zitnik. Therapeutics data commons: Machine learning datasets and tasks for drug discovery and development. In _NeurIPS Track Datasets and Benchmarks_ , 2021. 

- [13] John J Irwin, Teague Sterling, Michael M Mysinger, Erin S Bolstad, and Ryan G Coleman. Zinc: a free tool to discover chemistry for biology. _Journal of chemical information and modeling_ , 52 (7):1757–1768, 2012. 

- [14] Jan H Jensen. A graph-based genetic algorithm and generative model/monte carlo tree search for the exploration of chemical space. _Chemical science_ , 10(12):3567–3572, 2019. 

- [15] Woosung Jeon and Dongsup Kim. Autonomous molecule generation using reinforcement learning and docking to develop potential novel inhibitors. _Scientific Reports_ , 10, 12 2020. 

- [16] Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation. In _International Conference on Machine Learning_ , pages 2323–2332. PMLR, 2018. 

- [17] Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Hierarchical generation of molecular graphs using structural motifs. In _International Conference on Machine Learning_ , pages 4839–4848. PMLR, 2020. 

- [18] Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Multi-objective molecule generation using interpretable substructures. In _International conference on machine learning_ , pages 4849–4859. PMLR, 2020. 

- [19] Hyeonah Kim, Minsu Kim, Sanghyeok Choi, and Jinkyoo Park. Genetic-guided gflownets: Advancing in practical molecular optimization benchmark. _arXiv preprint arXiv:2402.05961_ , 2024. 

- [20] Xiangzhe Kong, Wenbing Huang, Zhixing Tan, and Yang Liu. Molecule generation by principal subgraph mining and assembling. _Advances in Neural Information Processing Systems_ , 35: 2550–2563, 2022. 

- [21] Mario Krenn, Florian Häse, AkshatKumar Nigam, Pascal Friederich, and Alan Aspuru-Guzik. Self-referencing embedded strings (selfies): A 100% robust molecular string representation. _Machine Learning: Science and Technology_ , 1(4):045024, 2020. 

- [22] Matt J Kusner, Brooks Paige, and José Miguel Hernández-Lobato. Grammar variational autoencoder. In _International Conference on Machine Learning_ , pages 1945–1954. PMLR, 2017. 

- [23] Greg Landrum et al. RDKit: Open-source cheminformatics software, 2016. _URL http://www. rdkit. org/, https://github. com/rdkit/rdkit_ , 2016. 

- [24] Seul Lee, Jaehyeong Jo, and Sung Ju Hwang. Exploring chemical space with score-based out-ofdistribution generation. In _International Conference on Machine Learning_ , pages 18872–18892. PMLR, 2023. 

- [25] Seul Lee, Seanie Lee, Kenji Kawaguchi, and Sung Ju Hwang. Drug discovery with dynamic goal-aware fragments. _International Conference on Machine Learning_ , 2024. 

- [26] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. _Advances in Neural Information Processing Systems_ , 33:9459–9474, 2020. 

- [27] Qingxin Li. Application of fragment-based drug discovery to versatile targets. _Frontiers in molecular biosciences_ , 7:180, 2020. 

11 

- [28] Shengchao Liu, Jiongxiao Wang, Yijin Yang, Chengpeng Wang, Ling Liu, Hongyu Guo, and Chaowei Xiao. Conversational drug editing using retrieval and domain feedback. In _The Twelfth International Conference on Learning Representations_ , 2023. 

- [29] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _International Conference on Learning Representations_ , 2019. 

- [30] Krzysztof Maziarz, Henry Richard Jackson-Flux, Pashmina Cameron, Finton Sirockin, Nadine Schneider, Nikolaus Stiefl, Marwin Segler, and Marc Brockschmidt. Learning to extend molecular scaffolds with structural motifs. In _International Conference on Learning Representations_ , 2021. 

- [31] Nicholas A Meanwell. Improving drug candidates by design: a focus on physicochemical properties as a means of improving compound disposition and safety. _Chemical research in toxicology_ , 24(9):1420–1456, 2011. 

- [32] AkshatKumar Nigam, Pascal Friederich, Mario Krenn, and Alan Aspuru-Guzik. Augmenting genetic algorithms with deep neural networks for exploring the chemical space. In _International Conference on Learning Representations_ , 2020. 

- [33] AkshatKumar Nigam, Robert Pollice, Mario Krenn, Gabriel dos Passos Gomes, and Alan Aspuru-Guzik. Beyond generative models: superfast traversal, optimization, novelty, exploration and discovery (stoned) algorithm for molecules using selfies. _Chemical science_ , 12(20):7079– 7090, 2021. 

- [34] Emmanuel Noutahi, Cristian Gabellini, Michael Craig, Jonathan SC Lim, and Prudencio Tossou. Gotta be safe: a new framework for molecular design. _Digital Discovery_ , 3(4):796–804, 2024. 

- [35] Marcus Olivecrona, Thomas Blaschke, Ola Engkvist, and Hongming Chen. Molecular de-novo design through deep reinforcement learning. _Journal of cheminformatics_ , 9(1):1–14, 2017. 

- [36] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-context retrieval-augmented language models. _Transactions of the Association for Computational Linguistics_ , 11:1316–1331, 2023. 

- [37] Gisbert Schneider, Werner Neidhart, Thomas Giller, and Gerard Schmid. “scaffold-hopping” by topological pharmacophore search: a contribution to virtual screening. _Angewandte Chemie International Edition_ , 38(19):2894–2896, 1999. 

- [38] Austin Tripp and José Miguel Hernández-Lobato. Genetic algorithms are strong baselines for molecule generation. _arXiv preprint arXiv:2310.09267_ , 2023. 

- [39] Austin Tripp, Gregor NC Simm, and José Miguel Hernández-Lobato. A fresh look at de novo molecular design benchmarks. In _NeurIPS 2021 AI for Science Workshop_ , 2021. 

- [40] Fabio Urbina, Filippa Lentzos, Cédric Invernizzi, and Sean Ekins. Dual use of artificialintelligence-powered drug discovery. _Nature Machine Intelligence_ , 4(3):189–191, 2022. 

- [41] W Patrick Walters, Matthew T Stahl, and Mark A Murcko. Virtual screening—an overview. _Drug discovery today_ , 3(4):160–178, 1998. 

- [42] Zichao Wang, Weili Nie, Zhuoran Qiao, Chaowei Xiao, Richard Baraniuk, and Anima Anandkumar. Retrieval-based controllable molecule generation. In _International Conference on Learning Representations_ , 2023. 

- [43] David Weininger. Smiles, a chemical language and information system. 1. introduction to methodology and encoding rules. _Journal of chemical information and computer sciences_ , 28 (1):31–36, 1988. 

- [44] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. Transformers: Stateof-the-art natural language processing. In _Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations_ , pages 38–45, 2020. 

12 

- [45] Yutong Xie, Chence Shi, Hao Zhou, Yuwei Yang, Weinan Zhang, Yong Yu, and Lei Li. Mars: Markov molecular sampling for multi-objective drug discovery. In _International Conference on Learning Representations_ , 2020. 

- [46] Soojung Yang, Doyeong Hwang, Seul Lee, Seongok Ryu, and Sung Ju Hwang. Hit and lead discovery with explorative rl and fragment-based molecule generation. _Advances in Neural Information Processing Systems_ , 34:7924–7936, 2021. 

- [47] Naruki Yoshikawa, Kei Terayama, Masato Sumita, Teruki Homma, Kenta Oono, and Koji Tsuda. Population-based de novo molecule generation, using grammatical evolution. _Chemistry Letters_ , 47(11):1431–1434, 2018. 

13 

## **A Limitations** 

Since our proposed _f_ -RAG is built on a pre-trained backbone molecular language model, it relies heavily on the generation performance of the backbone model. However, this also means that our method delegates the difficult task of molecule generation to a large model and lets the lightweight fragment injection module take care of the relatively easy task of fragment retrieval augmentation. This strategy enables very efficient and fast training (Section D.5) and makes _f_ -RAG a simple but powerful method to solve various drug discovery tasks. 

## **B Broader Impacts** 

Through our paper, we demonstrated that _f_ -RAG can achieve improved trade-offs between various considerations in drug discovery, showing its strong applicability to real-world drug discovery tasks. However, given its effectiveness, _f_ -RAG has the possibility to be used maliciously to generate harmful molecules. This can be prevented by setting the target properties to comprehensively consider toxicity and other side effects, or filtering out toxic fragments from the fragment vocabulary during generation. 

## **C Generation Process of** _f_ **-RAG** 

**Algorithm 1** Generation Process of _f_ -RAG 

**Input:** Dataset _D_ , fragment vocabulary size _N_ frag, molecule population size _N_ mol, number of soft fragments _K_ , number of total generations _G_ , number of SAFE-GPT generations per cycle _G_ SAFE-GPT, number of GA generations per cycle _G_ GA Set _V_ arm _←_ top- _N_ frag arms obtained from _D_ (Eq. (1)) Set _V_ linker _←_ top- _N_ frag linkers obtained from _D_ (Eq. (1)) Set _P ←∅_ Set _M ←∅_ **while** _|M| < G_ **do** _▷ Fragment retrieval-augmented SAFE-GPT generation_ **for** _i_ = 1 _,_ 2 _, . . . , N_ SAFE-GPT **do** Randomly retrieve two hard fragments _F_ hard _,_ 1 _, F_ hard _,_ 2 from _V_ arm _∪V_ linker Randomly retrieve _K_ soft fragments _{F_ soft _}[K] k_ =1[from] _[ V]_[arm] _[ ∪V]_[linker] Using _F_ hard _,_ 1 _.F_ hard _,_ 2 as input and _{F_ soft _}[K] k_ =1[as soft fragments, run SAFE-GPT to generate a] molecule _x_ Update _M ←M ∪{x}_ Decompose _x_ into _F_ arm _,_ 1 _, F_ linker, and _F_ arm _,_ 2 Update _V_ arm _←_ top- _N_ frag arms from _V_ arm _∪{F_ arm _,_ 1 _, F_ arm _,_ 2 _}_ Update Update _V_ linker _←_ top- _N_ frag linkers from _V_ linker _∪ F_ linker Update _P ←_ top- _N_ mol from _P ∪{x}_ **end for** _▷ GA generation_ **for** _i_ = 1 _,_ 2 _, . . . , N_ GA **do** Select parent molecules from _P_ Perform crossover and mutation to generate a molecule _x_ Update _M ←M ∪{x}_ Decompose _x_ into _F_ arm _,_ 1 _, F_ linker, and _F_ arm _,_ 2 Update _V_ arm _←_ top- _N_ frag arms from _V_ arm _∪{F_ arm _,_ 1 _, F_ arm _,_ 2 _}_ Update Update _V_ linker _←_ top- _N_ frag linkers from _V_ linker _∪ F_ linker Update _P ←_ top- _N_ mol from _P ∪{x}_ **end for end while Output:** Generated molecules _M_ 

14 

## **D Experimental Details** 

## **D.1 Training of Fragment Injection Module** 

In this section, we describe the details for training the fragment injection module, the only part of the _f_ -RAG framework that is trained. _f_ -RAG has 2,362,368 trainable parameters, coming from the fragment injection module. These correspond to only 2.64% of the total parameters of 89,648,640, indicating that the fragment injection module is very lightweight compared to the backbone molecular language model. 

Throughout the paper, we used the official codebase including the pre-trained SAFE-GPT[3] of Noutahi et al. [34]. Following the codebase, we used the HuggingFace Transformer library [44] (Apache-2.0 license) to train the fragment injection module. We set the number of retrieved soft fragments to _K_ = 10. For the layer of the backbone language model that the fragment injection module will be inserted behind, we conducted experiments with the search space _L ∈{_ 1 _,_ 6 _}_ , and found the _L_ = 1-st layer works well. The results showing this comparison are provided in Table 11. The fragment injection module was trained to 8 epochs with a learning rate of 1 _×_ 10 _[−]_[4] using the AdamW optimizer [29]. We performed searches with the search spaces (epoch) _∈{_ 5 _,_ 8 _,_ 10 _}_ and (learning rate) _∈{_ 1 _×_ 10 _[−]_[4] _,_ 5 _×_ 10 _[−]_[3] _}_ , respectively. 

## **D.2 Genetic Operations for Fragment Modification** 

As described in Section 3.3, we adopt the operations of Graph GA [14] to modify the generated fragments to further enhance exploration of _f_ -RAG. Specifically, in the crossover operation, parents are cut at random positions at ring or non-ring positions with a probability of 50%, and random fragments from the cut are combined to generate offspring. In the mutation operation, bond insertion/deletion, atom insertion/deletion, bond order swapping, or atom changes are performed on the offspring molecule with a predefined probability. 

## **D.3 Experiments on PMO Benchmark** 

In this section, we describe the experimental details used in the experiments of Section 4.1. 

**Implementation details.** We used the official codebase[4] of Gao et al. [10] for implementing the experiments. We used ZINC250k [13] with the same train/test split used by Kusner et al. [22] to train the fragment injection module and construct the initial fragment vocabulary. We set the size of the fragment vocabulary to _N_ frag = 50 and the size of the molecule population to _N_ mol = 50. When constructing the fragment vocabulary, fragments were filtered according to the number of atoms. We searched the size range in the search space [(min. number of atoms), (max. number of atoms)] _∈ {_ [5 _,_ 12] _,_ [10 _,_ 16] _,_ [10 _,_ 30] _}_ . As a result, we used the range [5 _,_ 12] for the albuterol_similarity, isomers_c7h8n2o2m, isomers_c9h10n2o2pf2cl, median1, qed, sitagliptin_mpo, zaleplon_mpo tasks, [10 _,_ 30] for the gsk3b and jnk3 tasks, and [10 _,_ 16] for the rest of the tasks. We also confined the size of the generate molecules. We used the range [10 _,_ 30] for the albuterol_similarity, isomers_c7h8n2o2m, isomers_c9h10n2o2pf2cl, median1, qed, sitagliptin_mpo, zaleplon_mpo tasks, [30 _,_ 80] for the gsk3b and jnk3 tasks, and [20 _,_ 50] for the rest of the tasks. We set the mutation rate of the GA to 0 _._ 1. We set the number of SAFE-GPT generation and number of GA generation in one cycle to _G_ SAFE-GPT = 10 and _G_ GA = 10, respectively. 

**Measuring novelty and synthesizability.** Following previous works [18, 45, 24, 25], novelty of the generated molecules _X_ is measured by the fraction of molecules that have a similarity less than 0.4 compared to its nearest neighbor _x_ SNN in the training set as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation_images/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation.pdf-0015-10.png)


where _N_ = _|X|_ and sim( _x, x[′]_ ) is the pairwise Tanimoto similarity of molecules _x_ and _x[′]_ . The similarity is calculated using Morgan fingerprints of radius 2 and 1024 bits obtained by the RDKit [23] 

> 3 `https://github.com/datamol-io/safe` (Apache-2.0 license) 

> 4 `https://github.com/wenhao-gao/mol_opt` (MIT license) 

15 

library. On the other hand, since synthetic accessibility (SA) [8] scores range from 1 to 10 with higher scores indicating more difficult synthesis, we measure synthesizability using the normalized SA score as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation_images/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation.pdf-0016-01.png)


**Visualizing the Radar Plots.** To draw Figure 1 and Figure 5, sum AUC top-10, average top-100 diversity, average top-100 novelty, and normalized average top-100 SA score on the PMO benchmark in Table 1 and Table 2 were used to indicate optimization performance, diversity, novelty, and synthesizability, respectively. After the SA score is normalized according to Eq. (6), all the values are min-max normalized to [0 _,_ 1]. 

## **D.4 Optimization of Docking Score under QED, SA, and Novelty Constraints** 

In this section, we describe the experimental details used in the experiments of Section 4.2. 

**Implementation details.** We used the official codebase[5] of Lee et al. [24] for implementing the experiments. Following Lee et al. [24] and Lee et al. [25], we used ZINC250k [13] with the same train/test split used by Kusner et al. [22] to train the fragment injection module and construct the initial fragment vocabulary. As in Section D.3, when constructing the fragment vocabulary, fragments were filtered based on their number of atoms. We set the range to [5 _,_ 12]. To confinee the size of the generate molecules, we used the range [20 _,_ 40]. The mutation rate of the GA was set to 0 _._ 1. We set the number of SAFE-GPT generation and number of GA generation in one cycle to _G_ SAFE-GPT = 1 and _G_ GA = 3, respectively. 

**Measuring docking score, QED, SA, and novelty.** We strictly followed the setting used in previous works [24, 25] to measure these properties. Specifically, we used QuickVina 2 [2] to calculate docking scores (DSs). We used the RDKit [23] library to calculate QED and SA. Following previous works, we compute the normalized DS (DS) and the normalized SA ([�] SA) as follows:[�] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation_images/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation.pdf-0016-07.png)


where clip is the function that clips the value in the range [ _−_ 20 _,_ 0]. As in Section D.3, novelty is determined as 1 _{_ sim( _x, x_ SNN) _<_ 0 _._ 4 _}_ . 

## **D.5 Computing resources** 

We trained the fragment injection module using one GeForce RTX 3090 GPU. The training took less than 4 hours. We generated molecules using one Titan XP (12GB), GeForce RTX 2080 Ti (11GB), or GeForce RTX 3090 GPU (24GB). The experiments on each task in Section 4.1 took less than 2 hours and the experiments on each task in Section 4.2 took approximately 2 hours. 

> 5 `https://github.com/SeulLee05/MOOD` (MIT license) 

16 

## **E Additional Experimental Results** 

In this section, we provide additional experimental results. 

**Diversity, novelty, and synthesizability results on PMO benchmark.** We provide the full results of Table 2 in Table 4, Table 5, and Table 6. As shown in the tables, _f_ -RAG achieves the best diversity and synthesizability, while showing the second best novelty. We also provide the pairwise trade-off plots in Figure 7. 

**Comparison with GEAM on PMO benchmark.** To provide additional comparisons of _f_ -RAG to GEAM [25], we followed the setup of Lee et al. [25] to conduct experiments on the seven MPO tasks in the PMO benchmark. As shown in Table 7, _f_ -RAG largely outperforms the baselines including GEAM in all of the tasks. 

**Ablation study.** We provide the full results of Figure 5(b) in Table 8 and Table 9. As shown in the tables, Figure 5(a), and Figure 5(b), the three components of _f_ -RAG, hard fragment retrieval, soft fragment retrieval, and genetic fragment modification, are essential to the superior performance of _f_ -RAG and improved balance between various drug discovery considerations. 

**Controlling optimization performance-diversity trade-off.** We provide the full results of Figure 5(c) in Table 10. As shown in the table and Figure 5(c), _f_ -RAG can effectively control the performance-diversity trade-off with the similarity-based fragment filtering. 

**Effect of location of the fragment injection module.** We provide the results of _f_ -RAG with _L_ = 1 and _L_ = 6 in Table 11. As shown in the table, _f_ -RAG with _L_ = 1 works better than _f_ -RAG with _L_ = 6, probably because injecting information from soft fragments in an earlier layer can carry more information into the final augmented embedding. As a result of this experiment, _f_ -RAG in the main experiments used _L_ = 1. 

**Visualization of generated molecules.** We visualize examples of the generated molecules by _f_ -RAG in Figure 8. The examples are drawn randomly from the molecules used to calculate the novel top 5% DS values in Table 3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation_images/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation.pdf-0017-08.png)


**----- Start of picture text -----**<br>
f -RAG Mol GA f -RAG f -RAG<br>Mol GA REINVENT<br>Genetic GFN<br>REINVENT Genetic GFN Mol GA Genetic GFN<br>REINVENT<br>Optimization performance Optimization performance Optimization performance<br>Better Better Better<br>Diversity Novelty<br>Synthesizability<br>**----- End of picture text -----**<br>


Figure 7: **Various trade-offs between the basic considerations in drug discovery.** Sum AUC top-10, average top-100 diversity, average top-100 novelty, and normalized average top-100 SA score on the PMO benchmark are used to measure optimization performance, diversity, novelty, and synthesizability, respectively. 

17 

Table 4: **Top-100 diversity results.** The results are the mean of 3 independent runs. The best results are highlighted in bold. 

|Oracle|_f_-RAG (ours)|Genetic GFN|Mol GA|REINVENT|
|---|---|---|---|---|
|albuterol_similarity|0.505|0.375|**0.596**|0.418|
|amlodipine_mpo|**0.394**|0.280|0.371|0.334|
|celecoxib_rediscovery|**0.360**|0.283|0.318|0.306|
|deco_hop|0.482|**0.536**|0.422|0.445|
|drd2|0.491|0.521|**0.548**|0.496|
|fexofenadine_mpo|**0.526**|0.424|0.508|0.403|
|gsk3b|0.329|0.159|0.085|**0.402**|
|isomers_c7h8n2o2|**0.824**|0.786|0.820|0.765|
|isomers_c9h10n2o2pf2cl|**0.797**|0.621|0.789|0.699|
|jnk3|0.318|**0.354**|0.045|0.205|
|median1|**0.614**|0.430|0.543|0.436|
|median2|0.424|**0.470**|0.424|0.382|
|mestranol_similarity|**0.572**|0.214|0.443|0.322|
|osimertinib_mpo|0.480|0.450|**0.505**|0.444|
|perindopril_mpo|0.375|0.394|**0.454**|0.346|
|qed|**0.816**|0.642|0.601|0.699|
|ranolazine_mpo|**0.570**|0.383|0.420|0.318|
|scaffold_hop|**0.479**|0.465|0.451|0.456|
|sitagliptin_mpo|0.647|0.165|0.632|**0.673**|
|thiothixene_rediscovery|0.428|0.388|**0.438**|0.387|
|troglitazone_rediscovery|**0.445**|0.391|0.407|0.410|
|valsartan_smarts|0.600|0.881|**0.902**|0.882|
|zaleplon_mpo|**0.751**|0.581|0.563|0.530|
|Average|**0.532**|0.443|0.491|0.468|



Table 5: **Top-100 novelty results.** The results are the mean of 3 independent runs. The best results are highlighted in bold. 

|Oracle|_f_-RAG (ours)|Genetic GFN|Mol GA|REINVENT|
|---|---|---|---|---|
|albuterol_similarity|0.543|0.667|**0.957**|0.553|
|amlodipine_mpo|0.000|0.037|**0.113**|0.000|
|celecoxib_rediscovery|0.930|**0.980**|0.885|0.943|
|deco_hop|0.650|0.923|**1.000**|0.893|
|drd2|0.570|0.780|**0.987**|0.357|
|fexofenadine_mpo|**1.000**|0.940|**1.000**|0.547|
|gsk3b|**1.000**|**1.000**|**1.000**|0.757|
|isomers_c7h8n2o2|**0.953**|0.647|0.947|0.373|
|isomers_c9h10n2o2pf2cl|0.993|0.747|**1.000**|0.273|
|jnk3|**1.000**|0.933|**1.000**|**1.000**|
|median1|0.437|0.810|**0.980**|0.433|
|median2|0.880|0.127|**0.943**|0.010|
|mestranol_similarity|0.850|0.893|**0.970**|0.923|
|osimertinib_mpo|0.917|0.737|**1.000**|0.617|
|perindopril_mpo|0.973|0.850|**1.000**|0.930|
|qed|**0.757**|0.423|0.053|0.063|
|ranolazine_mpo|**1.000**|0.453|**1.000**|0.290|
|scaffold_hop|0.810|0.993|**1.000**|0.930|
|sitagliptin_mpo|0.920|**1.000**|**1.000**|0.303|
|thiothixene_rediscovery|0.927|**0.980**|0.960|0.920|
|troglitazone_rediscovery|0.683|**0.917**|0.683|0.833|
|valsartan_smarts|**0.820**|0.200|0.323|0.123|
|zaleplon_mpo|**0.793**|0.610|0.750|0.343|
|Average|0.800|0.724|**0.845**|0.540|



18 

Table 6: **Top-100 SA score results.** The results are the mean of 3 independent runs. Lower is better and the best results are highlighted in bold. 

|Oracle|_f_-RAG (ours)|Genetic GFN|Mol GA|REINVENT|
|---|---|---|---|---|
|albuterol_similarity|**1.451**|3.241|3.822|3.311|
|amlodipine_mpo|**2.218**|3.675|3.806|3.645|
|celecoxib_rediscovery|**1.584**|2.735|2.388|2.667|
|deco_hop|**2.601**|4.333|5.066|3.262|
|drd2|**1.079**|2.605|3.051|2.514|
|fexofenadine_mpo|**1.837**|4.437|4.721|4.193|
|gsk3b|**1.253**|4.785|10.000|2.964|
|isomers_c7h8n2o2|**1.997**|3.354|4.721|3.323|
|isomers_c9h10n2o2pf2cl|**1.125**|4.108|5.314|2.828|
|jnk3|**1.746**|4.679|10.000|4.202|
|median1|**2.036**|4.098|4.516|4.058|
|median2|3.408|3.022|3.861|**2.957**|
|mestranol_similarity|**1.987**|4.241|4.780|4.131|
|osimertinib_mpo|**3.053**|3.734|4.293|3.257|
|perindopril_mpo|**2.153**|4.674|4.648|4.146|
|qed|2.968|3.340|2.501|**2.232**|
|ranolazine_mpo|**2.055**|3.489|4.422|3.096|
|scaffold_hop|3.691|3.513|5.146|**3.030**|
|sitagliptin_mpo|**1.627**|5.652|5.479|2.912|
|thiothixene_rediscovery|2.866|2.621|2.872|**2.444**|
|troglitazone_rediscovery|**1.631**|4.541|3.672|3.255|
|valsartan_smarts|**2.867**|3.127|3.530|3.100|
|zaleplon_mpo|**1.574**|2.717|3.296|2.236|
|Average|**2.026**|3.770|4.605|3.207|



Table 7: **PMO MPO AUC top-100 results.** The results are the means of 3 runs. The results for REINVENT, Graph GA, and SELFIES-REINVENT are taken from Gao et al. [10], and the results for GEAM are taken from Lee et al. [25]. The best results are highlighted in bold. 

|Method|MPO Benchmark|
|---|---|
||Amlodipine<br>Fexofenadine<br>Osimertinib<br>Perindopril<br>Ranolazine<br>Sitagliptin<br>Zaleplon|
|REINVENT [35]<br>Graph GA [14]<br>SELFIES-REINVENT<br>GEAM [25]|0.608<br>0.752<br>0.806<br>0.511<br>0.719<br>0.006<br>0.325<br>0.622<br>0.731<br>0.799<br>0.503<br>0.670<br>0.330<br>0.305<br>0.574<br>0.705<br>0.791<br>0.487<br>0.695<br>0.118<br>0.257<br>0.626<br>0.799<br>0.831<br>0.514<br>0.714<br>0.417<br>0.402|
|_f_-RAG (ours)|**0.704**_±_0.018<br>**0.827**_±_0.010<br>**0.850**_±_0.007<br>**0.640**_±_0.017<br>**0.779**_±_0.016<br>**0.458**_±_0.024<br>**0.423**_±_0.001|



19 

Table 8: **PMO AUC top-10 results of the ablated versions of** _f_ **-RAG.** The results are the mean of 3 independent runs. The best results are highlighted in bold. 

|Hard retrieval<br>Soft retrieval<br>GA|_f_-RAG|
|---|---|
||✓<br>x<br>✓<br>✓<br>x<br>✓<br>✓<br>x<br>✓<br>x<br>✓<br>✓<br>✓<br>x<br>✓|
|albuterol_similarity<br>amlodipine_mpo<br>celecoxib_rediscovery<br>deco_hop<br>drd2<br>fexofenadine_mpo<br>gsk3b<br>isomers_c7h8n2o2<br>isomers_c9h10n2o2pf2cl<br>jnk3<br>median1<br>median2<br>mestranol_similarity<br>osimertinib_mpo<br>perindopril_mpo<br>qed<br>ranolazine_mpo<br>scaffold_hop<br>sitagliptin_mpo<br>thiothixene_rediscovery<br>troglitazone_rediscovery<br>valsartan_smarts<br>zaleplon_mpo|**0.977**<br>0.907<br>0.974<br>0.875<br>0.929<br>**0.749**<br>0.620<br>0.747<br>0.629<br>0.666<br>**0.778**<br>0.524<br>0.748<br>0.505<br>0.695<br>**0.936**<br>0.621<br>0.908<br>0.811<br>0.645<br>0.992<br>0.987<br>**0.993**<br>**0.993**<br>0.976<br>**0.856**<br>0.823<br>0.838<br>0.756<br>0.830<br>**0.969**<br>0.825<br>0.967<br>0.876<br>0.822<br>**0.955**<br>0.940<br>0.931<br>0.891<br>0.968<br>0.850<br>0.795<br>**0.880**<br>0.727<br>0.896<br>**0.904**<br>0.834<br>0.903<br>0.800<br>0.612<br>**0.340**<br>0.293<br>0.339<br>0.290<br>0.285<br>**0.323**<br>0.250<br>0.315<br>0.255<br>0.284<br>0.671<br>0.650<br>**0.683**<br>0.566<br>0.550<br>0.866<br>0.835<br>**0.867**<br>0.821<br>0.838<br>**0.681**<br>0.530<br>0.667<br>0.585<br>0.564<br>0.939<br>0.937<br>0.941<br>0.937<br>**0.942**<br>**0.820**<br>0.800<br>0.809<br>0.726<br>0.808<br>0.576<br>0.556<br>**0.589**<br>0.509<br>0.522<br>0.601<br>0.502<br>**0.615**<br>0.304<br>0.576<br>**0.584**<br>0.458<br>0.579<br>0.432<br>0.545<br>0.448<br>0.397<br>**0.459**<br>0.313<br>0.396<br>**0.627**<br>0.685<br>0.624<br>0.000<br>0.540<br>0.486<br>0.462<br>**0.510**<br>0.447<br>0.506|
|Sum|**16.928**<br>15.231<br>16.890<br>14.048<br>15.395|



Table 9: **Top-100 diversity, top-100 novelty, and top-100 SA score results of the ablated versions of** _f_ **-RAG.** The results are the average values of all 23 tasks. The best results are highlighted in bold. 

|Hard retrieval<br>Soft retrieval<br>GA|_f_-RAG<br>✓<br>x<br>✓<br>✓<br>x<br>✓<br>✓<br>x<br>✓<br>x<br>✓<br>✓<br>✓<br>x<br>✓|
|---|---|
|Average diversity_↑_<br>Average novelty_↑_<br>Average SA score_↓_|0.532<br>0.544<br>0.497<br>**0.744**<br>0.489<br>0.800<br>**0.924**<br>0.720<br>0.438<br>0.860<br>**2.026**<br>4.113<br>2.231<br>3.443<br>4.235|



Table 10: **Top-100 diversity, top-100 novelty, and top-100 SA score results with different values of** _δ_ **of the similarity-based fragment filter.** The results are the average values of all 23 tasks. 

|Metric|_f_-RAG|_f_-RAG (_δ_ = 0_._8)|_f_-RAG (_δ_ = 0_._6)|_f_-RAG (_δ_ = 0_._4)|
|---|---|---|---|---|
|Sum AUC_↑_|16.928|16.648|16.262|15.765|
|Average diversity_↑_|0.532|0.606|0.681|0.724|
|Average novelty_↑_|0.800|0.778|0.751|0.796|
|Average SA score_↓_|2.026|3.836|3.825|3.852|



20 

Table 11: **Effect of location of the fragment injection module** _L_ **.** The results are the mean of 3 independent runs. _f_ -RAG in the main experiments used _L_ = 1. The best results are highlighted in bold. 

|Oracle|_f_-RAG (_L_= 1)|_f_-RAG (_L_= 6)|
|---|---|---|
|albuterol_similarity|**0.977**|0.971|
|amlodipine_mpo|**0.749**|0.721|
|celecoxib_rediscovery|**0.778**|0.764|
|deco_hop|**0.936**|0.935|
|drd2|**0.992**|0.991|
|fexofenadine_mpo|**0.856**|0.847|
|gsk3b|0.969|**0.974**|
|isomers_c7h8n2o2|**0.955**|0.950|
|isomers_c9h10n2o2pf2cl|**0.850**|0.848|
|jnk3|**0.904**|0.901|
|median1|0.340|0.351|
|median2|**0.323**|0.322|
|mestranol_similarity|0.671|**0.685**|
|osimertinib_mpo|**0.866**|0.865|
|perindopril_mpo|0.681|**0.698**|
|qed|**0.939**|**0.939**|
|ranolazine_mpo|**0.820**|0.798|
|scaffold_hop|0.576|**0.586**|
|sitagliptin_mpo|**0.601**|0.547|
|thiothixene_rediscovery|**0.584**|0.583|
|troglitazone_rediscovery|**0.448**|0.447|
|valsartan_smarts|0.627|**0.685**|
|zaleplon_mpo|**0.486**|**0.486**|
|Sum|**16.928**|16.894|



21 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation_images/nips_2024_Molecule_Generation_with_Fragment_Retrieval_Augmentation.pdf-0022-00.png)


**----- Start of picture text -----**<br>
-12.8, 0.561, 3.28, 0.338 -14.1, 0.516, 3.15, 0.373 -12.8, 0.676, 3.10, 0.308 -12.6, 0.677, 3.61, 0.385 -12.5, 0.697, 3.47, 0.375<br>parp1<br>-12.7, 0.821, 3.27, 0.358 -13.1, 0.751, 3.42, 0.329 -12.9, 0.790, 3.64, 0.294 -13.3, 0.684, 4.13, 0.355 -12.7, 0.546, 3.33, 0.250<br>-9.6, 0.748, 4.18, 0.260 -10.0, 0.709, 3.73, 0.333 -10.1, 0.665, 4.11, 0.318 -9.5, 0.753, 3.22, 0.296 -9.9, 0.650, 3.07, 0.381<br>fa7<br>-10.1, 0.701, 3.09, 0.394 -9.8, 0.799, 2.92, 0.358 -9.9, 0.601, 3.24, 0.342 -9.4, 0.728, 3.09, 0.352 -10.7, 0.563, 3.05, 0.329<br>-12.7, 0.640, 3.14, 0.313 -12.4, 0.629, 2.88, 0.380 -13.5, 0.590, 2.98, 0.391 -13.1, 0.548, 3.23, 0.315 -12.4, 0.533, 3.39, 0.346<br>5ht1b<br>-12.4, 0.874, 3.39, 0.315 -12.2, 0.545, 3.20, 0.278 -13.1, 0.569, 3.30, 0.351 -12.5, 0.888, 2.99, 0.367 -13.4, 0.781, 4.44, 0.314<br>-12.6, 0.545, 2.68, 0.385 -11.9, 0.501, 2.65, 0.371 -13.1, 0.711, 3.85, 0.329 -13.0, 0.547, 3.40, 0.319 -12.1, 0.662, 3.42, 0.307<br>braf<br>-12.1, 0.810, 3.00, 0.329 -12.4, 0.518, 3.00, 0.350 -12.8, 0.551, 2.82, 0.329 -12.0, 0.539, 3.62, 0.276 -11.9, 0.679, 2.65, 0.375<br>-11.6, 0.895, 3.37, 0.324 -11.5, 0.549, 3.05, 0.267 -13.3, 0.538, 3.16, 0.279 -11.9, 0.618, 3.72, 0.313 -11.3, 0.743, 3.13, 0.367<br>jak2<br>-11.9, 0.883, 2.74, 0.333 -12.1, 0.705, 4.16, 0.303 -11.4, 0.575, 2.91, 0.366 -11.5, 0.702, 3.46, 0.284 -12.5, 0.560, 2.88, 0.343<br>**----- End of picture text -----**<br>


Figure 8: **The generated molecules by** _f_ **-RAG.** Random top 5% novel hits generated in the experiments of Section 4.2 are visualized. The docking score (kcal/mol), QED, SA, and the maximum similarity with the molecules in the training set are at the bottom of each molecule. 

22 

## **NeurIPS Paper Checklist** 

## 1. **Claims** 

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? 

Answer: [Yes] 

Justification: The main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope. Guidelines: 

   - The answer NA means that the abstract and introduction do not include the claims made in the paper. 

   - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. 

   - The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. 

   - It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper. 

2. **Limitations** 

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] 

Justification: We discussed the limitations in Section A. 

Guidelines: 

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

Answer: [NA] 

23 

Justification: This paper does not include theoretical results. Guidelines: 

- The answer NA means that the paper does not include theoretical results. 

- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced. 

- All assumptions should be clearly stated or referenced in the statement of any theorems. 

- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. 

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. 

- Theorems and Lemmas that the proof relies upon should be properly referenced. 

## 4. **Experimental Result Reproducibility** 

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] 

Justification: The code to reproduce the results in our paper is provided as the supplementary material. 

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

24 

Answer: [Yes] 

Justification: The code to reproduce the results in our paper is provided as the supplementary material. 

Guidelines: 

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

Justification: We provided the training details in Section D.1, and the evaluation details in Section D.3 and Section D.4. 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. 

- The full details can be provided either with the code, in appendix, or as supplemental material. 

## 7. **Experiment Statistical Significance** 

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? 

Answer: [Yes] 

Justification: We reported the standard deviation of the main results in Table 1 and Table 3. We provided the standard deviation of the ablation study in Figure 5(a). 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. 

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions). 

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) 

25 

- The assumptions made should be given (e.g., Normally distributed errors). 

- It should be clear whether the error bar is the standard deviation or the standard error of the mean. 

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified. 

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates). 

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 

## 8. **Experiments Compute Resources** 

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? 

Answer: [Yes] 

Justification: We provided the details on computing resources in Section D.5. 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage. 

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute. 

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper). 

## 9. **Code Of Ethics** 

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics `https://neurips.cc/public/EthicsGuidelines` ? 

Answer: [Yes] 

Justification: We reviewed the NeurIPS Code of Ethics. Guidelines: 

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. 

- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics. 

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 

## 10. **Broader Impacts** 

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? 

Answer: [Yes] 

Justification: We discussed potential positive and negative societal impacts in Section B. Guidelines: 

- The answer NA means that there is no societal impact of the work performed. 

- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact. 

- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations. 

26 

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster. 

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. 

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 

## 11. **Safeguards** 

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? 

Answer: [NA] 

Justification: The paper poses no such risks. 

Guidelines: 

- The answer NA means that the paper poses no such risks. 

- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters. 

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images. 

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 

## 12. **Licenses for existing assets** 

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? 

Answer: [Yes] 

Justification: For existing assets used in the paper, we cited their original paper, repository URL, and license name in Section D.1, Section D.3, and Section D.4. 

Guidelines: 

- The answer NA means that the paper does not use existing assets. 

- The authors should cite the original paper that produced the code package or dataset. 

- The authors should state which version of the asset is used and, if possible, include a URL. 

- The name of the license (e.g., CC-BY 4.0) should be included for each asset. 

- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided. 

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, `paperswithcode.com/datasets` has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset. 

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided. 

27 

- If this information is not available online, the authors are encouraged to reach out to the asset’s creators. 

## 13. **New Assets** 

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? 

Answer: [Yes] 

Justification: The code of our work, including a description to run the code and the license, is provided as the supplementary material. 

Guidelines: 

- The answer NA means that the paper does not release new assets. 

- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc. 

- The paper should discuss whether and how consent was obtained from people whose asset is used. 

- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 

## 14. **Crowdsourcing and Research with Human Subjects** 

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? 

Answer: [NA] 

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: 

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. 

- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. 

- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 

## 15. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects** 

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? 

Answer: [NA] 

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: 

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. 

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. 

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. 

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 

28 

