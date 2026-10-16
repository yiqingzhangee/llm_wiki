Published as a conference paper at ICLR 2025 

## RAG-SR: RETRIEVAL-AUGMENTED GENERATION FOR NEURAL SYMBOLIC REGRESSION 

**Hengzhe Zhang**[1] _[,]_[3] **, Qi Chen**[1] _[,]_[3] **, Bing Xue**[1] _[,]_[3] **, Wolfgang Banzhaf**[2] **, Mengjie Zhang**[1] _[,]_[3] _[∗]_ 1School of Engineering and Computer Science, Victoria University of Wellington, New Zealand 2Department of Computer Science and Engineering, Michigan State University, USA 3Centre for Data Science and Artificial Intelligence, Victoria University of Wellington, New Zealand 

## ABSTRACT 

Symbolic regression is a key task in machine learning, aiming to discover mathematical expressions that best describe a dataset. While deep learning has increased interest in using neural networks for symbolic regression, many existing approaches rely on pre-trained models. These models require significant computational resources and struggle with regression tasks involving unseen functions and variables. A pre-training-free paradigm is needed to better integrate with searchbased symbolic regression algorithms. To address these limitations, we propose a novel framework for symbolic regression that integrates evolutionary feature construction with a neural network, without the need for pre-training. Our approach adaptively generates symbolic trees that align with the desired semantics in realtime using a language model trained via online supervised learning, providing effective building blocks for feature construction. To mitigate hallucinations from the language model, we design a retrieval-augmented generation mechanism that explicitly leverages searched symbolic expressions. Additionally, we introduce a scale-invariant data augmentation technique that further improves the robustness and generalization of the model. Experimental results demonstrate that our framework achieves state-of-the-art accuracy across 25 regression algorithms and 120 regression tasks[1] . 

## 1 INTRODUCTION 

Symbolic regression (SR) is a machine learning technique that searches the space of symbolic expressions to identify models that best fit a dataset (Sun et al., 2023; Fong et al., 2023). Unlike traditional regression methods, which assume a fixed model structure, SR automatically determines both the structure and parameters of the model. This flexibility allows SR to achieve both high accuracy and interpretability, making it especially valuable in fields such as physics (Udrescu et al., 2020), biology (Brunton et al., 2016), and finance (Liu & Guo, 2023), where uncovering transparent, understandable models is crucial for scientific discovery and informed decision-making. 

In this paper, we focus on an automated feature construction approach to SR (Cava et al., 2019). The key idea is to generate a set of symbolic trees/features, Φ = _{ϕ_ 1 _, . . . , ϕm}_ , from a dataset ( _X, Y_ ) to enhance the performance of an interpretable model _M_ , such as linear regression (Cava et al., 2019; Zhang et al., 2023a). The objective is to minimize the loss function _L_ (Φ; **X** _, y_ ), defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0001-08.png)


where _N_ represents the number of instances, and _Xi_ and _yi_ represent the features and label for the _i_ -th instance in the training data. By decomposing the SR task into the discovery of feature sets, this approach reduces the complexity of the problem. Even if each feature _ϕ_ is weakly correlated with the target _Y_ , the model can still perform well as long as the features collectively complement each other in predicting the target. This symbolic regression paradigm is particularly effective for 

> _∗_ Emails: _{_ hengzhe.zhang, qi.chen, bing.xue, mengjie.zhang _}_ @ecs.vuw.ac.nz, banzhafw@msu.edu. 

> 1Source Code: https://github.com/hengzhe-zhang/RAG-SR 

1 

Published as a conference paper at ICLR 2025 

complex real-world problems, where the complexity of the underlying system cannot be captured by a simple equation. 

Traditional SR methods, predominantly based on genetic programming (GP) (Banzhaf et al., 1998), perform gradient-free searches within the symbolic space (Jiang & Xue, 2024). While effective at exploration, these methods often lack search effectiveness due to limited guidance from accumulated knowledge during the evolutionary process. Recent advances in deep learning for SR (Biggio et al., 2021; Kamienny et al., 2022) aim to address these inefficiencies by leveraging knowledge more effectively. 

Deep learning-based SR typically follows three primary paradigms: pre-trained language models (Biggio et al., 2021; Kamienny et al., 2022), reinforcement learning (Landajuela et al., 2021), and sparse supervised learning (Sahoo et al., 2018). Sparse supervised learning does not generate symbolic models directly; instead, it relies on heuristic pruning and neural architecture search to sparsify the network so that it can be converted into symbolic expressions (Li et al., 2024). In contrast, pre-trained language models and reinforcement learning can generate symbolic models directly. However, pre-trained language models require prior assumptions about the problem space, limiting their generalizability to novel tasks involving unseen functions and features. Additionally, identifying an optimal set of features for modeling complex real-world systems is time-consuming, making it impractical to generate many pairs of symbolic models and their outputs for pre-training. While reinforcement learning with a language model offers task adaptability (Landajuela et al., 2022), its low sample efficiency remains a significant drawback. Therefore, it is desirable to explore supervised learning methods that do not rely on pretraining for SR to overcome these challenges. 

To develop an effective and efficient neural network for SR, we propose a novel neural networkbased symbolic regression framework inspired by geometric semantic genetic programming (GSGP) (Moraglio et al., 2012). As illustrated in Figure 1, the core idea is to use a neural network to dynamically predict the best feature _ϕ_ to replace an existing feature in the current set Φ = _{ϕ_ 1 _, . . . , ϕm}_ , with the goal of filling the gap in the residual _R_ , referred to as the desired semantics in this paper. Throughout the evolutionary process, the relationship between the semantics/outputs of each symbolic tree _ϕ_ ( _X_ ) and its symbolic representation _ϕ_ is captured and stored in a neural semantic library, which is continuously updated in an online fashion. 

One challenge with neural semantic libraries is that language models may generate features _ϕ_ that are grammatically correct but irrelevant to the desired semantics _R_ . In the language model domain, this is known as hallucination (Sun et al., 2024). To mitigate this, we propose a retrieval-augmented generation technique to reduce hallucination and generate symbolic trees that better align with the desired semantics. In summary, the key contributions of this paper are as follows: 

- We propose a semantic descent algorithm to optimize symbolic models using a neural network with online supervised learning. The neural network continuously learns to generate symbolic trees that precisely capture the desired semantics, pushing the boundaries of deep symbolic regression to handle complex problems. 

- To reduce hallucination in language models, we develop a retrieval-augmented generation mechanism. This technique makes the generated symbolic models are not only grammatically correct but also better aligned with the desired semantics, resulting in more accurate predictions. 

- To better capture the relationship between desired semantics and retrieved symbolic expressions, we propose a masked contrastive loss, which more accurately generates symbolic trees by aligning the embeddings of desired semantics with those of the retrieved expressions. 

- We propose a data augmentation and double query strategy to fully exploit the scaleinvariant properties of feature construction-based symbolic regression, further improving the effectiveness of generated symbolic expressions. 

## 2 RELATED WORK 

In the domain of neural symbolic regression, a key advantage of pre-trained models is that, once pre-trained (Biggio et al., 2021; Kamienny et al., 2022), models can be reused for similar tasks 

2 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0003-01.png)


**----- Start of picture text -----**<br>
Old Solution (ξ,ζ,γ)=β1ξ+β2ζ+β3Γ New Solution (ξ,ζ,δ)=β̂1ξ+β̂2ζ+β̂3δ<br>Old Solution<br>Tree B Tree B<br>ζX1 ζX2 ζX3 ζX1 ζX2 ζX3<br>+ + ÷ ξX1 Tree A ξX2 ξX3 SemanticsOld ξX1 Tree AξX2 ξX3 SemanticsCurrent<br>X [1] X [1] X [2] X [1] X [1] X [1] Target<br>γX1 γX2 γX3 Target δX1 δX2 δX3<br>Tree A Tree B Tree C Tree C   Y1 Y2 Y3 Tree σ<br>Delete  Replace a Tree with a New Tree  Add<br>Replace Subtrees  Tree C  Generated by a Neural Network Tree σ<br>Mutation with Randomly<br>Generated Ones Tree B Query + Prompt<br>ζX1 ζX2 ζX3 X [1] X [1]<br>X [1] + X [2] X [1] + X [3] X [2] ÷ X [1] ξX1 Tree AξX2 ξTemporaryX3Semantics RX1ResidualRX2 RX3 Tree σ̂ with semanticsαX1SemanticLibraryαX2 αX3<br>Tree A [*] Tree B [*] Tree C [*] γX1 γX2 γX3<br>New Solution (Pending Replacement)Tree C  Target Query NetworkNeural<br>Neural Network with Retrieval<br>Computing Residual Augmentation to Generate a New Tree<br>Evolutionary Algorithm  Retrieval Augmented Neural Generation<br>(Syntactic/Symbolic Space) (Semantic/Output Space)<br>**----- End of picture text -----**<br>


Figure 1: Comparison of the evolutionary algorithm and retrieval-augmented neural semantic library for feature construction-based symbolic regression. 

without further optimization. These models are designed to solve a distribution of tasks through mechanisms such as invariance encoding (Holt et al., 2023), contrastive learning (Li et al., 2022), or conditional constraints (Bendinelli et al., 2023) to capture relationships among different SR tasks within a problem space. However, these methods may struggle with tasks beyond the scope of the pre-training data, particularly when encountering different function sets or more variables than those seen during training (Shojaee et al., 2024a; Meidani et al., 2024). Fine-tuning could alleviate the misalignment between training and target tasks, through approaches like reinforcement learning (Holt et al., 2023) or using imitation learning to learn successful mutations (Kamienny et al., 2023). However, fine-tuning large pre-trained language models can be challenging. Thus, exploring how online learning techniques can be applied exclusively to enhance SR remains a promising and underexplored direction. 

Reinforcement learning (RL), on the other hand, learns the probability distribution of promising symbolic models (Landajuela et al., 2021; Xu et al., 2024) by interacting with the environment, allowing it to adapt to different function sets for various tasks. However, deep symbolic optimization via RL often suffers from low sample efficiency, requiring integration with GP (Mundhenk et al., 2021) or Monte Carlo Tree Search (MCTS) (Xu et al., 2024) techniques to improve performance. Furthermore, RL typically simplifies feedback to a scalar reward, such as mean squared error (Landajuela et al., 2021), which limits the richness of information provided during the search process. A more effective approach would involve using a loss vector rather than a scalar loss to provide richer feedback and enhance overall search effectiveness. 

Sparse supervised learning methods, such as deep equation learners (Sahoo et al., 2018) and efficient symbolic policy learning (Guo et al., 2024), aim to derive interpretable symbolic models by regularizing neural networks (Zhang et al., 2023c). However, since the _L_ 0 norm is non-differentiable, these techniques often rely on heuristic pruning approaches to convert neural networks into interpretable expressions. Additionally, they typically require neural architecture search methods to identify suitable architectures before gradient-based training (Li et al., 2024). 

Evolutionary symbolic regression is primarily based on the GP framework, which automatically discovers symbolic models without predefined structures to fit the training data (Fong et al., 2023). Recently, semantic GP has gained substantial attention (Moraglio et al., 2012; Zhang et al., 2023b). Unlike traditional GP, which operates in the syntactic/symbolic space, semantic GP works in the semantic/output space. By focusing on semantic space, solution generation operators can ensure that the newly generated solutions have more predictable behavior, such as guaranteed loss reduction—something conventional GP operators often lack. A key challenge in semantic GP is generating GP trees that satisfy the desired semantics (Moraglio et al., 2012). A common strategy is to build a semantic library that stores evaluated GP trees (Pawlak et al., 2014). In semantic mutation, this library is searched for trees that closely match the target semantics, and the best-matching tree 

3 

Published as a conference paper at ICLR 2025 

## **Algorithm 1** Semantic Descent 

|**Alg**|**orithm 1**Semantic Descent|**orithm 1**Semantic Descent||
|---|---|---|---|
|1:|**Input:** Features Φ = _{ϕ_1_, . . . , ϕm}_, semantics library _L_, neural network model||_N_, current semantics|
||Φ(_X_), target_Y_, neural generation probability_P_neural|||
|2:|**Output:** Updated featuresΦ|||
|3:|_O ←_Random permutation of_{_1_,_2_, . . . , m}_||_▷_Shuffe tree indices|
|4:|**for**each_i ∈O_**do**|||
|5:|˜_ϕi_(_X_)_←ϕi_(_X_)_−µi_<br>_σi_||_▷_Normalized feature|
|6:|Φ(_X_)temp _←_Φ(_X_)_−βi_ ˜_ϕi_(_X_)|||
|7:|**R**_←Y −_Φ(_X_)temp||_▷_Compute residual_R_|
|8:|**if**rand()_< P_neural **then**|||
|9:|_ϕi ←N_(**R**_, L_)|_▷_Generate new tree using neural model||
|10:|Φ(_X_)_←_Φ(_X_)temp|||
|11:|**continue**||_▷_Proceed to next tree|
|12:|**end if**|||
|13:|_ϕ_new _←_ExactRetrieval(**R**_, ϕi, L_)|||
|14:|_ϕi,_Φ(_X_)_←_ExactReplacement(_ϕ_new_, ϕ_new(_X_)_,_Φ(_X_)_,_Φ(_X_)temp_,_**R**_, Y_)|||
|15:|**end for**|||



is selected. However, this approach relies solely on existing building blocks without leveraging historical knowledge to create new ones. To address this issue, it is crucial to incorporate deep learning techniques to learn from the evolutionary learning process and generate better symbolic models that align with the desired semantics. 

## 3 ALGORITHM 

The proposed method is based on an evolutionary algorithm framework, encompassing solution initialization, generation, evaluation, selection, and archive maintenance. This work primarily focuses on the solution generation phase, introducing a neural semantic library for solution generation, designed to explicitly retain and apply knowledge throughout the evolutionary process. Solution generation consists of two primary components: semantic descent and retrieval-augmented generation. 

## 3.1 SEMANTIC DESCENT 

In this work, we propose Semantic Descent (SD), an iterative optimization procedure designed to improve model performance by selectively replacing suboptimal features. Unlike methods such as geometric semantic GP (Moraglio et al., 2012) or gradient boosting (Feng et al., 2018), which incrementally add new features to minimize error, SD focuses on replacing existing trees in the model with more informative ones. This approach helps maintain a compact model structure while continuously improving accuracy. 

At each iteration, a tree _ϕi_ is randomly selected from the set of trees _{ϕ_ 1 _, . . . , ϕm}_ that define the semantics/outputs of the model Φ( _X_ ) = _β_ 1 _ϕ_ 1( _X_ ) + _· · ·_ + _βmϕm_ ( _X_ ) + _α_ , where _β_ represents the coefficients and _α_ is the intercept. The contribution of _ϕi_ is temporarily removed, resulting in temporary semantics Φ[temp] ( _X_ ) = Φ( _X_ ) _− βiϕi_ ( _X_ ). The residual _R_ = _Y −_ Φ[temp] ( _X_ ) of the model is then computed, where _Y_ is the target output. The residual _R_ represents the difference between the prediction and the target. 

As shown in Algorithm 1, the core idea of SD is to fill the gap in the residual _R_ by replacing the current tree _ϕi_ with a better alternative, either generated by a neural model _N_ (line 9) or retrieved from a semantic library _L_ (line 13). The semantic library _L_ stores all previously evaluated symbolic trees and subtrees _ψ_ along with their semantics/outputs _ψ_ ( _X_ ). The neural model _N_ learns the mapping between the semantics _ψ_ ( _X_ ) and the corresponding symbolic tree _ψ_ . This enables the neural network _N_ to construct a new feature _ϕ_ new using _R_ as input, thereby generating a new feature to reduce the model’s error. 

The probability of generating new trees using the neural network is _P_ neural, detailed in Section 3.2. The probability of retrieving a tree from the semantic library is 1 _− P_ neural. The key idea of exact retrieval is to search the library for the tree that most closely matches the desired semantics, i.e., the residual _R_ , as detailed in Appendix C. Since the linear regression model automatically adjusts 

4 

Published as a conference paper at ICLR 2025 

feature magnitudes and intercepts, the residual _R_ is normalized using the _L_ 2 norm before being used as input for retrieval or neural generation, i.e., _R ← ||RR−||R_[¯] 2[.][The][replacement][process][is][re-] peated iteratively until all trees _ϕ_ within the solution Φ have been traversed. By focusing on feature replacement instead of addition, SD enables efficient model refinement while maintaining a fixed feature set, allowing for both interpretability and performance improvements. 

## 3.2 RETRIEVAL-AUGMENTED GENERATION 

To learn the mapping between symbolic trees, _ϕ_ , and their corresponding semantics, _ϕ_ ( _X_ ), the process involves three steps: First, the trees and semantics are collected from the evolutionary process (Section 3.2.2). Next, they are converted into training data using specially designed encoding rules (Section 3.2.1). Finally, a neural network is trained on the collected data (Section 3.2.3), using cross-entropy loss and masked contrastive loss (Section 3.2.4). 

## 3.2.1 DATA COLLECTION AND NETWORK TRAINING 

The semantic library _L_ is dynamically constructed during the evolutionary process. During solution evaluation, each subtree _ψ_ and its corresponding semantics _ψ_ ( _X_ ) are stored in a first-in-first-out queue _Q_ with an upper limit of 10,000 entries for training the neural network and future retrieval. To facilitate efficient retrieval, a k-dimensional tree (k-d tree) is constructed using the semantics stored in _Q_ at the end of each generation in the evolutionary process, reducing query complexity to _O_ (log( _N_ )), where _N_ is the number of stored trees. The neural network is also trained at the end of each generation. To prevent unnecessary training, an internal validation set monitors performance degradation. If the validation loss does not increase, network training is skipped for that generation to save computational resources. Nevertheless, the retrieval library is updated even when network training is bypassed, ensuring that knowledge base is continuously updated throughout the evolutionary process. 

## 3.2.2 ENCODING AND DECODING RULES FOR SYMBOLIC TREES 

To generate valid expressions without requiring an explicit end token, we designed a specialized encoding and decoding scheme. Symbolic trees are encoded using a level-order traversal method, specifically breadth-first search, to convert the tree into a linear sequence. To maintain interpretability, the number of functions in the symbolic tree is capped at _nF_ . Given this limit and the maximum number of children any function can have, _α_ max, the maximum number of terminal nodes is: 

_nT_ = 1 + ( _nF ×_ ( _α_ max _−_ 1)) _._ (2) 

Given _nF_ and _nT_ , the neural network outputs a fixed-length sequence with the first _nF_ tokens decoded as functions or terminals,while the remaining _nT_ tokens are restricted to terminals by setting function probabilities to zero during decoding. This formulation reframes the symbolic tree generation task as a multi-class classification problem. Detailed pseudocode for the encoding and decoding processes is provided in Appendix E. 

## 3.2.3 OVERALL ARCHITECTURE FOR RETRIEVAL-AUGMENTED GENERATION 

As shown in Figure 2, the neural architecture consists of two main components: a Multilayer Perceptron (MLP) and a Transformer model. Their relationship is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0005-12.png)


Each of the two components plays a distinct and complementary role in generating a symbolic tree _ϕ_ based on the desired semantics _R_ . The MLP transforms raw semantics into a meaningful feature representation that can guide the generation of the symbolic tree. On the other hand, the Transformer encoder processes the nearest symbolic tree _ϕ_[ˆ] , retrieved from the semantics library _L_ , which serves as a prompt to reduce hallucination. The outputs of the MLP and Transformer are concatenated and then passed through a Transformer decoder to generate a sequence of _L_ tokens. **W** _∈_ R _[S][×][D]_ is a linear layer that projects the output of the Transformer decoder **H** Decoder _∈_ R _[B][×][L][×][D]_ into the symbol space **O** _∈_ R _[B][×][L][×][S]_ , where _S_ is the number of unique symbols. These tokens are subsequently decoded into a valid symbolic tree. 

5 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0006-01.png)


**----- Start of picture text -----**<br>
Semantic Transformer Transformer Softmax 1 2 3<br>Library Encoder Decoder Linear<br>Add & Norm Add & Norm + + +<br>Feed<br>Feed Forward<br>+ Forward Add & Norm - - -<br>Query Prompt Add & Norm Multi-HeadAttention<br>αX1 X1 αX2 X1 αX3 Multi-HeadAttention Add & NormMulti-HeadMasked X1 X1 X1<br>Tree σ̂ with semantics Positional  + Attention<br>Encoding Input Positional Encoding + X2 X2 X2<br>Embedding Output Predictions<br>Embedding<br>RX1 RX2 RX3 Inputs Concatenate Outputs<br>Desired  (shifted right)<br>1 2 3<br>Semantics<br>1 + + X1 X2<br>Level Order<br>2 X1 X2 3 Traversal<br>MLP Tree<br>**----- End of picture text -----**<br>


Figure 2: Neural network architecture for symbolic tree generation. 

**Intention Encoding:** The desired semantics _R ∈_ R _[B][×][N]_ is processed through an MLP to produce a feature matrix **F** MLP _∈_ R _[B][×][K]_ , where _K_ is the dimensionality of the hidden layer. The MLP consists of _NL_ layers, and at each layer _i_ , the transformation is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0006-04.png)


where **x** _i ∈_ R _[B][×][K]_ is the input to the _i_ -th layer, **W** _i ∈_ R _[K][×][K]_ is the weight matrix, **b** _i ∈_ R _[K]_ is the bias vector, BN _i_ denotes the batch normalization layer, SiLU _i_ is the Sigmoid Linear Unit activation function (Elfwing et al., 2018), and Dropout _i_ is the dropout layer with a specified dropout rate. This MLP layer results in a feature matrix **F** MLP _∈_ R _[B][×][K]_ , which is then passed through a linear layer to match the dimensionality from _K_ to _D_ , yielding **F**[mapped] MLP _∈_ R _[B][×][D]_ , where _D_ is the dimensionality of the Transformer-encoded representation. 

**Retrieval-Augmented Encoding:** For the desired semantics _R_ , a KD-Tree is used to retrieve the nearest symbolic tree _ϕ_[ˆ] from the semantic library _L_ , based on Euclidean distance and subject to the constraint that the tree contains no more than _nF_ nodes. The retrieved tree _ϕ_[ˆ] is then processed through an embedding layer to generate **V** ˆ _ϕ ∈_ R _[B][×][L][×][E]_ , where _L_ is the sequence length of the tree encoding and _E_ is the dimensionality of the embedding space. The embedding layer consists of an embedding matrix **E** _∈_ R _[S][×][E]_ . The embedded representation **V** ˆ _ϕ_ is then encoded using the Transformer model to produce a symbolic model embedding **H** Transformer _∈_ R _[B][×][L][×][D]_ . The Transformer encoder applies self-attention and feedforward layers with residual connections as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0006-07.png)


**Decoding:** The combined feature representation **H** Combined = **F**[mapped] MLP _⊕_ **H** Transformer _∈_ R _[B][×]_[(] _[L]_[+1)] _[×][D]_ is fed into a Transformer decoder to generate the contextual embeddings **H** Decoder _∈_ R _[B][×][L][×][D]_ . The decoding process is performed auto-regressively, utilizing a greedy decoding strategy. 

## 3.2.4 LOSS FUNCTION 

**Masked Contrastive Loss:** The intention encoding should ideally learn useful knowledge not only from target expressions _ϕ_ but also from the retrieved symbolic expressions _ϕ_[ˆ] . In parallel, the retrieval-augmented encoding should be aware of the semantics of the nearest symbolic expressions. To fulfill these objectives, we propose a contrastive loss that aligns the embeddings from both the intention encoding and the retrieval-augmented encoding components. 

Given the nearest semantics _ϕ_[ˆ] ( _X_ ) _∈_ R _[B][×][N]_ , it is processed through a MLP to generate a feature matrix of nearest semantics **F** nearest _∈_ R _[B][×][K]_ . Simultaneously, the embedding of the symbolic model 

6 

Published as a conference paper at ICLR 2025 

**H** Transformer _∈_ R _[B][×][L][×][D]_ is averaged along the sequence length dimension to produce the averaged embedding **H** avg _∈_ R _[B][×][D]_ . Then, the InfoNCE loss (Oord et al., 2018), a popular objective in contrastive learning, is employed to maximize the similarity between the nearest semantics feature matrix **F** nearest and the averaged symbolic embeddings **H** avg, while minimizing similarity with negative samples from the same batch. To alleviate false negatives, i.e., when two samples in a batch are semantically similar, the InfoNCE loss is masked by a mask matrix _mask_ . The mask matrix is designed such that non-diagonal elements with an absolute cosine similarity greater than 0.99 are marked as false (indicating false negatives), while all other entries are marked as true. The masked InfoNCE loss is formally defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0007-02.png)


where sim( _·, ·_ ) denotes cosine similarity, and _τ_ is a temperature parameter controlling the sharpness of the softmax function. This contrastive loss ensures that the nearest semantics are closely aligned with their corresponding symbolic representations in the embedding space, while differentiating them from unrelated samples. 

**Cross-Entropy Loss:** The model is also trained using cross-entropy loss over the sequence of _L_ symbols. Let **o** _[i]_ true _[∈]_[R] _[S]_[denote the one-hot encoded ground truth for the] _[ i]_[-th position, and] **[ o]** _[i]_ pred _[∈]_ R _[S]_ denote the predicted probability distribution at that position. Formally, the cross-entropy loss for each sequence is defined as _L_ cross-entropy = _−_[�] _[L] i_ =1 **[o]** true _[i][·]_[ log(] **[o]** _[i]_ pred[)][.][The final loss] _[ L]_[ is a weighted] sum of the cross-entropy loss and the contrastive loss: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0007-05.png)


where _λ_ is a hyperparameter that balances the contributions of the two losses. 

## 3.3 DATA AUGMENTATION AND DOUBLE QUERY 

In linear regression, the sign of coefficients is automatically adjusted, so the sign of the semantics is not crucial. However, the training data may only include one side of a training pair ( _ψ, ψ_ ( _X_ )), without considering its opposite, ( _ψ, −ψ_ ( _X_ )). Consequently, when the desired semantics is _−ψ_ ( _X_ ), the model may fail to generate the correct symbolic tree _ψ_ . To address this issue, we augment the training data by including both ( _ψ, ψ_ ( _X_ )) and ( _ψ, −ψ_ ( _X_ )) pairs: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0007-09.png)


During decoding, both _R_ and _−R_ are used to query the neural network, generating candidate trees _ϕ_ and _ϕ[′]_ . The tree with the highest probability is selected as the final symbolic model. This technique, referred to as double query (DQ), allows the model to generate symbolic trees with sign-insensitive semantics, thereby improving the effectiveness of neural generation. 

## 4 EXPERIMENTS 

This section is divided into two parts. The first part evaluates the effectiveness of the proposed components in improving the prediction accuracy of the neural semantic library. The second part investigates the performance of integrating the SR method with the retrieval-augmented neural semantic library. It compares this integrated approach to state-of-the-art SR methods. 

## 4.1 EXPERIMENTAL RESULTS OF NEURAL SEMANTIC LIBRARY 

**Experimental Settings:** To evaluate the effectiveness of the proposed techniques in enhancing the learning capabilities of the neural semantic library, we conduct the first experiment on synthetic data. The objective is to evaluate how various components contribute to the learning effectiveness of the neural semantic library. In this experiment, 10 variables and 50 training instances are randomly drawn from a Gaussian distribution _N_ (0 _,_ 100). Then, a total of 10000 symbolic expressions with random heights _h ∈_ [0 _,_ 5] are generated using the grow method (Banzhaf et al., 1998) from GP and evaluated on the randomly generated data. The maximum number of functions _nF_ is set to 5, and 

7 

Published as a conference paper at ICLR 2025 

expressions exceeding this limit are filtered out. To avoid redundancy, only one semantically equivalent GP tree is retained, ensuring no symbolic expressions overlap between training and test sets. This setup ensures that the final metric reflects the ability of the neural network to learn patterns and generalize to unseen data, rather than simply fitting to previously seen examples. A total of 80% of the symbolic models are used for training, while the remaining 20% are reserved for testing. The evaluation metric is the edit distance (Matsubara et al., 2022; Bertschinger et al., 2023) between the generated symbolic tree and the ground truth, where a smaller distance indicates that the neural semantic library generates more effective building blocks, significantly aiding the evolutionary algorithm in finding optimal solutions. Each experiment is run 5 times to ensure stable and reliable results. 

**Parameter Settings:** For the neural network, the dropout rate is set to 0.1. The MLP consists of 3 layers, while both the encoder and decoder Transformers have 1 layer each. The hidden layer size is set to 64 neurons. A learning rate of 0.01 and a batch size of 64 are used. Early stopping with a patience of 5 epochs is employed to prevent overfitting. The weight of contrastive loss _λ_ is set to 0.05. 

**Experimental Results (Edit Distance):** The experimental results for edit distance on the test set are presented in Figure 3. First, comparing neural generation with simple retrieval from the library (W/O NN), neural generation performs better by a large margin, indicating the effectiveness of using a neural network for symbolic tree generation. As for the ablation results of components, the results show that including all components achieves the lowest median edit distance, indicating that the combination of all proposed techniques provides the best overall performance. Among the components, the RAG technique has the most significant impact, highlighting that external knowledge from the semantic library significantly improves the neural network’s ability to generate relevant symbolic trees. Data augmentation (DA) also plays a crucial role, ranking as the second most important component. Without DA, the model struggles to handle the scale-invariant nature of feature construction, leading to worse performance. The compact boxplots reflect the consistency and reliability of these components. Dropout has a moderate positive effect, indicating that overfitting control techniques are helpful for training the neural semantic library. Similarly, contrastive learning (CL) shows a moderate impact, confirming the effectiveness of using contrastive loss to align the intention encoding with retrieval augmentation encoding components. Finally, DQ also improves effectiveness, showing that even simply generating multiple solutions during inference can lead to better solutions, which aligns with findings from large language models (Wang et al., 2023). The impact of DQ becomes more pronounced in the absence of DA, suggesting that DA partially compensates for the lack of DQ. 

**Experimental Results (Running Time):** The running time comparisons in Figure 4 demonstrate that RAG moderately increases the overall running time. However, one advantage of incorporating RAG into the component is that new trees can be seamlessly added to the retrieval library to improve accuracy without requiring model fine-tuning, making the algorithm efficient for application in an online learning setting. For DA and DQ, removing these components reduces the running time from 44 seconds to 35 and 29 seconds, respectively, indicating that they do introduce some computational overhead. However, given the accuracy improvements they provide, the increase in computational time is acceptable. Although removing both DA and DQ significantly reduces computational cost, the substantial loss of edit distance from 3.82 to 4.35 outweighs the benefit of faster execution. 

**Examples of Generated Trees:** Table 1 provides examples of symbolic trees generated by the neural network with and without retrieval augmentation, along with the retrieved trees. The results demonstrate that the retrieved trees share certain similarities with the ground truth, such as variable usage. These results validate that providing the retrieval tree as a prompt helps the neural network generate more relevant trees, reducing hallucination compared to relying solely on the desired semantics. 

## 4.2 EXPERIMENTS OF RAG-SR 

**Datasets:** In this study, we primarily focus on 120 black-box datasets from the PMLB benchmark (Olson et al., 2017), which are particularly challenging for pre-training methods (Kamienny et al., 2022) due to the potential absence of simple symbolic expressions to model these datasets. The results on the 119 Feynman and 14 Strogatz datasets are presented in Appendix L.2. 

8 

Published as a conference paper at ICLR 2025 

Table 1: Examples of symbolic trees generated by the retrieval-augmented neural network, simple neural network, retrieval library, and ground truth. 

|RAG-NN Generated Tree (Distance)|Simple NN Generated Tree (Distance)|
|---|---|
|sin(sin(ARG3)) (0)|cos(cos(cos(ARG9))) (4)|
|aq(ARG7, ARG8) (0)|abs(maximum(ARG7, ARG7)) (3)|
|maximum(ARG1, ARG8) (0)|subtract(ARG1, ARG1) (2)|
|sqrt(sqrt(ARG2)) (0)|abs(abs(ARG2)) (2)|
|subtract(ARG6, ARG7) (0)|maximum(ARG7, ARG7) (2)|
|Retrieval Tree (Distance)|Ground Truth Tree|
|sin(ARG3) (1)|sin(sin(ARG3))|
|abs(negative(maximum(aq(ARG8, ARG0), aq(ARG7, ARG8)))) (6)|aq(ARG7, ARG8)|
|maximum(add(absolute(sin(cos(ARG6))), ARG8), ARG1) (6)|maximum(ARG1, ARG8)|
|square(abs(ARG2)) (2)|sqrt(sqrt(ARG2))|
|subtract(ARG7, ARG6) (2)|subtract(ARG6, ARG7)|




![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0009-03.png)


**----- Start of picture text -----**<br>
Distribution of Edit Distance by Configuration 4 . 35 Median Running Time by Configuration<br>4.44.2 4.16 4.18 4.18 4.29 40 44.41 29.30 41.59 34.20 34.75 28.95<br>23.48<br>4.0 3.82 3.85 3.91 20<br>3.8<br>0<br>Figure 3: Ablation study of components based on Figure 4: Ablation study of components with re-<br>edit distance on the test set. spect to running time (training and inference).<br>All W/O DQ W/O CLW/O Dropout W/O DA W/O RAG W/O NNW/O DA+DQ All W/O DQ W/O CLW/O Dropout W/O DA W/O RAGW/O DA+DQ<br>(s)<br>Edit Distance<br>Median Running Time<br>**----- End of picture text -----**<br>


**Evaluation Protocol:** The evaluation follows the established procedures of state-of-the-art symbolic regression benchmarks (La Cava et al., 2021). Specifically, each dataset is split into training and testing sets with a 75:25 ratio, and experiments are repeated 10 times for robustness. The _R_[2] score on the test set is used as the evaluation metric. To better handle categorical variables, we use a target encoder (Micci-Barreca, 2001). Furthermore, to prevent any single feature from disproportionately influencing the semantics, all input features are normalized using min-max scaling (Raymond et al., 2020). 

**Parameter Settings:** For GP, we follow conventional parameter settings: a population size of 200 and a maximum of 100 generations. Each solution consists of 10 trees, representing 10 features. The probability of using neural generation, _P_ neural, is set to 0.1. 

**Experimental Results (Accuracy):** The experimental results on SRBench are presented in Figure 5. The proposed method, RAG-SR, outperforms all state-of-the-art symbolic regression and machine learning techniques in terms of _R_[2] scores. Notably, it surpasses the TPSR method (Shojaee et al., 2024a), which combines MCTS with a pre-trained end-to-end Transformer (Kamienny et al., 2022). The improvement is statistically significant, as confirmed by the Wilcoxon signed-rank test with Benjamini-Hochberg correction, shown in Figure 6. This indicates the effectiveness of using a purely online training language model for learning symbolic expressions. Compared to SBP-GP (Pawlak et al., 2014), which is a purely retrieval-based geometric semantic GP that does not use a neural network, the significant advantage of RAG-SR demonstrates the effectiveness of using a neural network to dynamically generate symbolic models. 

**Experimental Results (Complexity):** The model complexity of RAG-SR follows the definition of SRBench, where the final model is converted into a SymPy-compatible expression, and the number of nodes in the symbolic tree is counted as a measure of complexity. As shown in Figure 5, RAG-SR produces models that are an order of magnitude smaller in size compared to PS-Tree (Zhang et al., 2022), which is a piecewise SR method that ranks second in _R_[2] scores in Figure 5. The Pareto front of test _R_[2] scores and model size rank is shown in Figure 7, where RAG-SR appears on the first Pareto front, indicating that RAG-SR achieves a good balance between accuracy and model complexity. 

**Experimental Results (Training Time):** The training time of RAG-SR is comparable to that of FEAT, a standard feature-construction-based SR method (Cava et al., 2019), suggesting that the 

9 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0010-01.png)


**----- Start of picture text -----**<br>
R 2 ��"+, �(!"%��$." �* $'$'#��$&"��+�<br>�������<br>����*""<br>�����<br>��)"*('<br>�������<br>�����<br>�����<br>������<br>���<br>����<br>���������<br>�! �((+,<br>� '!(&�(*"+,<br>�����<br>�������<br>����<br>����<br>�"*'"%�$!#"<br>�#)%" *'<br>����<br>�����<br>���<br>�$'" *<br>����<br>� ���"-'& '<br>����� ���� ���� ���� ���� ���� �� � �� � �� � �� � �� � �� � �� � �� �<br>**----- End of picture text -----**<br>


Figure 5: _R_[2] scores, model sizes, and training time of 25 algorithms on 120 regression problems. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0010-03.png)


**----- Start of picture text -----**<br>
�"$�','&�*"!&���)�&#�+�*+�������*+��α = ��������<br>����-&%�&����''*+������������ p < ����⋅α �� ��� ��!�" �"$�%&����""%& �����<br>��������� ���� ���<br>����������� p < ����⋅α ���� ����'! �!�<br>��)&�$�"�!����� �� ���$��� ������� ��$!�������<br>���������"&��)��� p < ����⋅α ������������ �����<br>��&�'%�')�*+���������)���(�)'& p < α �� �#�$"!������ ������� ����<br>������<br>����������� &'�*"!&" "��&�� � ��������� ���� �������<br>!($��)&<br>�#���$!� ����<br>���� ��!��$<br>�<br>� � �� �� ��<br>R 2 ���%&���!�<br>Figure 6: Pairwise statistical comparisons of Figure 7: Pareto front of the rank of test R [2]<br>test  R [2] scores on regression problems. scores and model size for different algorithms.<br>�"������(����!�<br>��� ������ ����-&%�& ����''*+ ��� ��� ����� ���� ��� �������� ���� ��)&�$�"�!� ���� �"&��) ��� ���� �(�)'& ���)�� ������ ��&�'%�')�*+ ������ ���� ���� ���<br>**----- End of picture text -----**<br>


computational cost of learning a neural semantic library is within an acceptable range. However, compared to TPSR, which directly leverages a pre-trained model to guide SR without requiring fine-tuning, RAG-SR is an order of magnitude slower. This discrepancy is partly due to the fact that, in the current implementation, all neural networks in RAG-SR are trained on a CPU due to limited computational resources. Training the neural networks on a GPU could potentially reduce the computational time of RAG-SR. 

## 5 CONCLUSIONS 

In this paper, we propose a novel feature construction-based SR method with a retrieval-augmented neural semantic library. Ablation studies confirm that the retrieval augmentation mechanism effectively mitigates the issue of hallucination, enabling the generation of more accurate symbolic trees that align with the desired symbolic trees. Furthermore, data augmentation and double query techniques effectively improve the neural network’s ability to generate symbolic trees that account for the scale-invariant characteristics of feature construction-based SR. Experimental results on large-scale symbolic regression benchmarks demonstrate that RAG-SR significantly outperforms state-of-theart SR techniques, including those guided by pre-trained language models. For future directions, introducing constraints on model complexity may help reduce the risk of overfitting, particularly with datasets that contain noise or limited samples, presenting a promising direction for future research. 

10 

Published as a conference paper at ICLR 2025 

## REFERENCES 

- Ignacio Arnaldo, Krzysztof Krawiec, and Una-May O’Reilly. Multiple regression genetic programming. In _Proceedings of the 2014 Annual Conference on Genetic and Evolutionary Computation_ , pp. 879–886, 2014. 

- Wolfgang Banzhaf, Peter Nordin, Robert E Keller, and Frank D Francone. _Genetic programming: an introduction: on the automatic evolution of computer programs and its applications_ . Morgan Kaufmann Publishers Inc., 1998. 

- Tommaso Bendinelli, Luca Biggio, and Pierre-Alexandre Kamienny. Controllable neural symbolic regression. In _International Conference on Machine Learning_ , pp. 2063–2077. PMLR, 2023. 

- Amanda Bertschinger, Q Tyrell Davis, James Bagrow, and Joshua Bongard. The metric is the message: Benchmarking challenges for neural symbolic regression. In _Joint European Conference on Machine Learning and Knowledge Discovery in Databases_ , pp. 161–177. Springer, 2023. 

- Manish Bhattarai, Javier E Santos, Shawn Jones, Ayan Biswas, Boian Alexandrov, and Daniel O’Malley. Enhancing code translation in language models with few-shot learning via retrievalaugmented generation. _arXiv preprint arXiv:2407.19619_ , 2024. 

- Luca Biggio, Tommaso Bendinelli, Alexander Neitz, Aurelien Lucchi, and Giambattista Parascandolo. Neural symbolic regression that scales. In _International Conference on Machine Learning_ , pp. 936–945. Pmlr, 2021. 

- Steven L Brunton, Joshua L Proctor, and J Nathan Kutz. Discovering governing equations from data by sparse identification of nonlinear dynamical systems. _Proceedings of the National Academy of Sciences_ , 113(15):3932–3937, 2016. 

- William La Cava, Tilak Raj Singh, James Taggart, Srinivas Suri, and Jason Moore. Learning concise representations for regression by evolving networks of trees. In _International Conference on Learning Representations_ , 2019. 

- Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. _Neural networks_ , 107:3–11, 2018. 

- Ji Feng, Yang Yu, and Zhi-Hua Zhou. Multi-layered gradient boosting decision trees. _Advances in Neural Information Processing Systems_ , 31, 2018. 

- Kei Sen Fong, Shelvia Wongso, and Mehul Motani. Rethinking symbolic regression: Morphology and adaptability in the context of evolutionary algorithms. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- Arya Grayeli, Atharva Sehgal, Omar Costilla-Reyes, Miles Cranmer, and Swarat Chaudhuri. Symbolic regression with a learned concept library. _arXiv preprint arXiv:2409.09359_ , 2024. 

- Jiaming Guo, Rui Zhang, Shaohui Peng, Qi Yi, Xing Hu, Ruizhi Chen, Zidong Du, Ling Li, Qi Guo, Yunji Chen, et al. Efficient symbolic policy learning with differentiable symbolic expression. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In _Proceedings of the IEEE International Conference on Computer Vision_ , pp. 1026–1034, 2015. 

- Thomas Helmuth, Lee Spector, and James Matheson. Solving uncompromising problems with lexicase selection. _IEEE Transactions on Evolutionary Computation_ , 19(5):630–643, 2014. 

- Samuel Holt, Zhaozhi Qian, and Mihaela van der Schaar. Deep generative symbolic regression. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https: //openreview.net/forum?id=o7koEEMA1bR. 

- Joey Hong, David Dohan, Rishabh Singh, Charles Sutton, and Manzil Zaheer. Latent programmer: Discrete latent codes for program synthesis. In _International Conference on Machine Learning_ , pp. 4308–4318. PMLR, 2021. 

11 

Published as a conference paper at ICLR 2025 

- Nan Jiang and Yexiang Xue. Racing control variable genetic programming for symbolic regression. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pp. 12901–12909, 2024. 

- Pierre-Alexandre Kamienny, St´ephane d’Ascoli, Guillaume Lample, and Franc¸ois Charton. End-toend symbolic regression with transformers. _Advances in Neural Information Processing Systems_ , 35:10269–10281, 2022. 

- Pierre-Alexandre Kamienny, Guillaume Lample, Sylvain Lamprier, and Marco Virgolin. Deep generative symbolic regression with monte-carlo-tree-search. In _International Conference on Machine Learning_ , pp. 15655–15668. PMLR, 2023. 

- William La Cava, Thomas Helmuth, Lee Spector, and Jason H Moore. A probabilistic and multiobjective analysis of lexicase selection and _ε_ -lexicase selection. _Evolutionary Computation_ , 27 (3):377–402, 2019. 

- William La Cava, Bogdan Burlacu, Marco Virgolin, Michael Kommenda, Patryk Orzechowski, Fabr´ıcio Olivetti de Franc¸a, Ying Jin, and Jason H Moore. Contemporary symbolic regression methods and their relative performance. _Advances in neural information processing systems_ , 2021(DB1):1, 2021. 

- Mikel Landajuela, Brenden K Petersen, Sookyung Kim, Claudio P Santiago, Ruben Glatt, Nathan Mundhenk, Jacob F Pettit, and Daniel Faissol. Discovering symbolic policies with deep reinforcement learning. In _International Conference on Machine Learning_ , pp. 5979–5989. PMLR, 2021. 

- Mikel Landajuela, Chak Shing Lee, Jiachen Yang, Ruben Glatt, Claudio P Santiago, Ignacio Aravena, Terrell Mundhenk, Garrett Mulcahy, and Brenden K Petersen. A unified framework for deep symbolic regression. _Advances in Neural Information Processing Systems_ , 35:33985–33998, 2022. 

- Wenqiang Li, Weijun Li, Linjun Sun, Min Wu, Lina Yu, Jingyi Liu, Yanjie Li, and Songsong Tian. Transformer-based model for symbolic regression via joint supervised learning. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- Wenqiang Li, Weijun Li, Lina Yu, Min Wu, Linjun Sun, Jingyi Liu, Yanjie Li, Shu Wei, Deng Yusong, and Meilan Hao. A neural-guided dynamic symbolic network for exploring mathematical expressions from data. In _Forty-first International Conference on Machine Learning_ , 2024. URL https://openreview.net/forum?id=IejxxE9DO2. 

- Jiacheng Liu and Siqi Guo. Symbolic regressions in non-physical systems, 2023. URL https: //openreview.net/forum?id=RuCQRXk7a7G. 

- Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In _International Conference on Learning Representations_ , 2017. URL https://openreview. net/forum?id=Skq89Scxx. 

- Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _International Conference on Learning Representations_ , 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7. 

- Yoshitomo Matsubara, Naoya Chiba, Ryo Igarashi, and Yoshitaka Ushiku. Rethinking symbolic regression datasets and benchmarks for scientific discovery. _Journal of Data-centric Machine Learning Research_ , 2022. 

- Kazem Meidani, Parshin Shojaee, Chandan K. Reddy, and Amir Barati Farimani. SNIP: Bridging mathematical symbolic and numeric realms with unified pre-training. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum? id=KZSEgJGPxu. 

- Daniele Micci-Barreca. A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. _ACM SIGKDD explorations newsletter_ , 3(1):27–32, 2001. 

12 

Published as a conference paper at ICLR 2025 

- Alberto Moraglio, Krzysztof Krawiec, and Colin G Johnson. Geometric semantic genetic programming. In _Parallel Problem Solving from Nature-PPSN XII: 12th International Conference, Taormina, Italy, September 1-5, 2012, Proceedings, Part I 12_ , pp. 21–31. Springer, 2012. 

- Terrell Mundhenk, Mikel Landajuela, Ruben Glatt, Claudio P Santiago, Brenden K Petersen, et al. Symbolic regression via deep reinforcement learning enhanced genetic programming seeding. _Advances in Neural Information Processing Systems_ , 34:24912–24923, 2021. 

- Ji Ni, Russ H Drieberg, and Peter I Rockett. The use of an analytic quotient operator in genetic programming. _IEEE Transactions on Evolutionary Computation_ , 17(1):146–152, 2012. 

- Randal S Olson, William La Cava, Patryk Orzechowski, Ryan J Urbanowicz, and Jason H Moore. Pmlb: a large benchmark suite for machine learning evaluation and comparison. _BioData mining_ , 10:1–13, 2017. 

- Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. _arXiv preprint arXiv:1807.03748_ , 2018. 

- Md Rizwan Parvez, Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. Retrieval augmented code generation and summarization. _arXiv preprint arXiv:2108.11601_ , 2021. 

- Tomasz P Pawlak, Bartosz Wieloch, and Krzysztof Krawiec. Semantic backpropagation for designing search operators in genetic programming. _IEEE Transactions on Evolutionary Computation_ , 19(3):326–340, 2014. 

- Christian Raymond, Qi Chen, Bing Xue, and Mengjie Zhang. Adaptive weighted splines: A new representation to genetic programming for symbolic regression. In _Proceedings of the 2020 Genetic and Evolutionary Computation Conference_ , pp. 1003–1011, 2020. 

- Subham Sahoo, Christoph Lampert, and Georg Martius. Learning equations for extrapolation and control. In _International Conference on Machine Learning_ , pp. 4442–4450. Pmlr, 2018. 

- Parshin Shojaee, Kazem Meidani, Amir Barati Farimani, and Chandan Reddy. Transformer-based planning for symbolic regression. _Advances in Neural Information Processing Systems_ , 36, 2024a. 

- Parshin Shojaee, Kazem Meidani, Shashank Gupta, Amir Barati Farimani, and Chandan K Reddy. Llm-sr: Scientific equation discovery via programming with large language models. _arXiv preprint arXiv:2404.18400_ , 2024b. 

- Fangzheng Sun, Yang Liu, Jian-Xun Wang, and Hao Sun. Symbolic physics learner: Discovering governing equations via monte carlo tree search. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum?id= ZTK3SefE8_Z. 

- YuHong Sun, Zhangyue Yin, Qipeng Guo, Jiawen Wu, Xipeng Qiu, and Hui Zhao. Benchmarking hallucination in large language models based on unanswerable math word problem. In Nicoletta Calzolari, Min-Yen Kan, Veronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue (eds.), _Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)_ , pp. 2178–2188, Torino, Italia, May 2024. ELRA and ICCL. URL https://aclanthology.org/2024.lrec-main.196. 

- Silviu-Marian Udrescu, Andrew Tan, Jiahai Feng, Orisvaldo Neto, Tailin Wu, and Max Tegmark. Ai feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity. _Advances in Neural Information Processing Systems_ , 33:4860–4871, 2020. 

- Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. _Advances in neural information processing systems_ , 30, 2017. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum?id=1PL1NIMMrw. 

13 

Published as a conference paper at ICLR 2025 

- Zora Zhiruo Wang, Akari Asai, Xinyan Velocity Yu, Frank F Xu, Yiqing Xie, Graham Neubig, and Daniel Fried. Coderag-bench: Can retrieval augment code generation? _arXiv preprint arXiv:2406.14497_ , 2024. 

- Yilong Xu, Yang Liu, and Hao Sun. Reinforcement symbolic regression machine. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview. net/forum?id=PJVUWpPnZC. 

- Hengzhe Zhang, Aimin Zhou, Hong Qian, and Hu Zhang. PS-Tree: A piecewise symbolic regression tree. _Swarm and Evolutionary Computation_ , 71:101061, 2022. 

- Hengzhe Zhang, Qi Chen, Bing Xue, Wolfgang Banzhaf, and Mengjie Zhang. Modular multi-tree genetic programming for evolutionary feature construction for regression. _IEEE Transactions on Evolutionary Computation_ , 2023a. 

- Hengzhe Zhang, Qi Chen, Bing Xue, Wolfgang Banzhaf, and Mengjie Zhang. A semantic-based hoist mutation operator for evolutionary feature construction in regression. _IEEE Transactions on Evolutionary Computation_ , 2023b. 

- Michael Zhang, Samuel Kim, Peter Y Lu, and Marin Soljaˇci´c. Deep learning and symbolic regression for discovering parametric equations. _IEEE Transactions on Neural Networks and Learning Systems_ , 2023c. 

14 

Published as a conference paper at ICLR 2025 

## A ALGORITHM WORKFLOW 

The algorithm workflow, shown in Figure 8, includes the following steps: 

- **Solution Initialization:** The population is initialized using random symbolic trees generated through ramped half-and-half methods (Banzhaf et al., 1998). In this approach, half of the GP trees are initialized with random depths, while the other half are initialized at the maximum initial depth. Each solution contains _m_ symbolic trees, with each tree representing a unique feature. 

- **Solution Evaluation:** Features _{ϕ_ 1( _X_ ) _, . . . , ϕm_ ( _X_ ) _}_ are constructed from the symbolic trees Φ = _{ϕ_ 1 _, . . . , ϕm}_ and the training data _X_ . These features are evaluated via ridge regression using efficient leave-one-out cross-validation to compute the loss _L_ (Φ( _X_ )). 

- **Semantic Library Construction:** To reduce the dimensionality of inputs to the neural network and optimize memory usage, a subset of hard instances is selected every 10 generations. Specifically, all training instances _Xi ∈ X_ are ranked by the median loss across all solutions, i.e., Median( _L_ (Φ( _Xi_ )) _|_ Φ _∈ P_ ), and the top _k_ = 50 instances, ranked in descending order, are chosen. The instances are further sorted by their corresponding training labels _Yi_ . Consequently, the positions of some hard instances remain unchanged after instance selection, which may facilitate the continuous training of the neural network. The semantics of these selected instances are stored in a retrieval library, which is used for training the neural semantic library and for exact retrieval. It is important to note that instance selection is solely for semantic library construction and semantic descent. Solution evaluation is conducted on the entire training dataset. 

   - **Retrieval Library:** Symbolic trees and their corresponding semantics are stored in a first-in-first-out queue _Q_ , which can hold up to 10,000 symbolic trees. A KD-Tree is used for efficient retrieval, reducing the search complexity from _O_ ( _N_ ) to _O_ (log( _N_ )). For neural network training, a subset of symbolic trees from _Q_ , where function nodes are fewer than or equal to five, is used to construct a separate KD-Tree for rapid retrieval of eligible trees. 

   - **Neural Semantic Library:** The neural network is trained using pairs of symbolic trees and their corresponding semantics from the retrieval library. The architecture, described in Section 3.2, combines a multi-layer perceptron with a Transformer to process desired semantics with retrieved symbolic trees. 

- **Solution Selection:** Lexicase selection (Helmuth et al., 2014) is used to identify promising solutions based on their objective values. Solutions are iteratively eliminated by applying randomly constructed criteria. In each round, a random instance _i ∈_ [0 _, N_ ] is selected, and solutions with losses greater than the median absolute deviation are eliminated (La Cava et al., 2019). This elimination process continues until only one solution remains, which is then selected as the promising solution. For a population of _P_ solutions, _P_ solutions are selected by iteratively applying these steps. 

- **Solution Generation:** New solutions are generated based on the selected promising solutions using a memetic algorithm framework. First, semantic descent performs a local search to improve a solution, followed by evolutionary search to explore the search space and mitigate the risk of local optima: 

   - **Semantic Descent:** As detailed in Section 3.1, the residual _R_ is used to query the neural semantic library or the retrieval library to either generate or retrieve GP trees. The neural generation process is described in Section 3.2. For retrieving GP trees, Appendix C explains the strategy to retrieve relevant GP trees from the library based on desired semantics. Since the retrieved features have already been evaluated, the loss reduction can be easily calculated after replacement, as explained in Appendix D. If replacing the tree does not reduce the loss, the current round of replacement is skipped. 

   - **Evolutionary Search:** New symbolic trees are generated using genetic programming operators such as random subtree crossover and mutation (Banzhaf et al., 1998). For _m_ symbolic trees, the crossover and mutation operators are applied _m_ times with their respective probabilities to ensure sufficient exploration of the search space. 

15 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0016-01.png)


**----- Start of picture text -----**<br>
End<br>Termination?<br>Population Initialization<br>Archive Maintenance<br>Solution Evaluation<br>Neural Network Training<br>Retrieval Library Construction<br>Retrieval Library Construction<br>Neural Network Training<br>Solution Evaluation<br>Parent Selection<br>No Exact<br>Retrieval Evolutionary<br>Neural Generation? Generation<br>Neural<br>Yes<br>Generation<br>**----- End of picture text -----**<br>


Figure 8: Workflow of neural semantic library-based symbolic regression. 

- **Archive Maintenance:** The historically best-performing solution is stored in an archive for making predictions on unseen data. 

The processes of solution evaluation, library construction, solution selection, solution generation and archive maintenance are repeated until the stopping criteria are met. 

## B LIBRARY UPDATE 

The library is updated at the end of each generation during the evolutionary process. As detailed in Algorithm 2, the library is managed using two first-in-first-out queues, _T_ and _S_ , which store the most recent 10,000 trees and their corresponding semantics, respectively. The semantics _ψ_ ( _X_ ) are first normalized to _ψ_ norm( _X_ ) to ensure that retrieval is insensitive to scale. This normalization is necessary because the linear regression algorithm can automatically determine the scaling coefficient _β_ for each constructed feature. Subsequently, all GP trees are appended to the queue _T_ , while their corresponding semantics are appended to the queue _S_ and stored in a hash set _H_ seen. 

To control memory usage, the hash set prevents duplicate GP trees from being added to the library. When the queues _T_ and _S_ reach their maximum limit _T_ max, the oldest entries are removed, as illustrated in Line 21. 

For computational efficiency, the semantics of each subtree _ψ_ ( _X_ ) are precomputed during the solution evaluation phase. Consequently, during the library update phase, it is unnecessary to recompute the semantics of each tree _ψ_ based on the data _X_ . This precomputation ensures that the library update process ensures computationally efficient. 

Additionally, for efficient querying in exact retrieval, a KD-Tree is constructed at the end of the library update. To efficiently retrieve symbolic expressions that meet the function node limits of neural generation, another KD-Tree is constructed based on symbolic expressions with a maximum of _nF_ function nodes. More details on the construction of this KD-Tree for neural generation are provided in Appendix F. Together, the two queues, _T_ and _S_ , along with the two KD-Trees, form the semantic library _L_ . 

## C EXACT RETRIEVAL OF THE SYMBOLIC TREE 

Exact retrieval is directly using symbolic trees from the retrieval library that align with the desired semantics _R_ , complementing neural generation. This occurs with a probability of 1 _− P_ neural, as described in Algorithm 1. In this process, the desired semantics _R_ are used to retrieve the top _κ_ = 10 symbolic trees with the smallest distances to _R_ , as shown in Algorithm 3. These _κ_ trees 

16 

Published as a conference paper at ICLR 2025 

## **Algorithm 2** Library Update 

1: **Input:** Population of individuals _P_ , tree queue _T_ , semantics queue _S_ , hash set of seen semantics _H_ seen, maximum tree limit _T_ max 

2: **Output:** Updated tree queue _T_ and semantics queue _S_ 

3: **for** each individual Φ _∈ P_ **do** 4: **for** each pair ( _ψ_ ( _X_ ) _, ψ_ ), where _ψ_ is a GP tree/subtree in Φ and _ψ_ ( _X_ ) is its corresponding semantics **do** 5: **if** _∥ψ_ ( _X_ ) _∥_ = 0 **then** 6: **Continue** _▷_ Skip semantics with zero norm 7: **end if** _ψ_ ( _X_ ) 8: _ψ_ norm( _X_ ) _← ∥ψ_ ( _X_ ) _∥ ▷_ Normalize semantics 9: **if** _ψ_ norm( _X_ ) contains NaN or _∞_ **then** 10: **Continue** _▷_ Skip invalid trees 11: **end if** 12: **if** _ψ_ norm( _X_ ) _∈H_ seen **then** 13: **Continue** _▷_ Skip already seen semantics 14: **end if** 15: _T ←T ∪{ψ} ▷_ Add tree to queue 16: _S ←S ∪{ψ_ norm( _X_ ) _} ▷_ Add semantics to queue 17: _H_ seen _←H_ seen _∪{ψ_ norm( _X_ ) _} ▷_ Update hash set of seen semantics 18: **end for** 19: **end for** 20: **if** _|T | > T_ max **then** _▷_ Retain only the most recent _T_ max entries 21: _T ←T_ [ _−T_ max :] 22: _S ←S_ [ _−T_ max :] 23: **end if** 

**Algorithm 3** Exact Retrieval 

1: **Input:** Desired semantics _R_ , current tree _ϕi_ , semantic library _L_ , maximum retrieval count _κ_ = 10 2: **Output:** Proposed tree _ϕ_ new 3: _Lκ ←_ Top- _κ_ � _L,_ distance( _L, R_ )� _▷_ Retrieve top _κ_ trees with smallest distances to _R_ 4: _Lκ ←_ Sort( _Lκ,_ distance( _Lκ, R_ )) _▷_ Sort by increasing distance to _R_ 5: _ϕ_ new _←_ None _▷_ Initialize proposed tree 6: **for** each tree _ϕ ∈Lκ_ **do** 7: **if** size( _ϕ_ ) _≤_ size( _ϕi_ ) **then** 8: _ϕ_ new _← ϕ ▷_ Select the first valid tree 9: **Break** _▷_ Stop further iteration 10: **end if** 11: **end for** 12: **if** _ϕ_ new = None **then** 13: **Return Skip** _▷_ Skip generation if no valid tree is found 14: **else** 15: **Return** _ϕ_ new _▷_ Return the proposed tree 16: **end if** 

are then ranked based on their distance to the desired semantics _R_ , and the first tree with the same or fewer nodes than the current tree _ϕi_ is selected as the proposed tree _ϕ_ new. This size constraint encourages the generation of simpler, more interpretable symbolic expressions. If no such tree is found, the current round of retrieval-augmented generation is skipped. 

## D EXACT REPLACEMENT 

Exact replacement is a mechanism that replaces features in a solution with more effective features stored in the retrieval library. Since both the features _ϕ_ and their associated semantics _ϕ_ ( _X_ ) are stored in the retrieval library _L_ , it is straightforward to determine the optimal scaling factor during 

17 

Published as a conference paper at ICLR 2025 

## **Algorithm 4** Exact Replacement 

1: **Input:** New proposed tree _ϕ_ new, current semantics Φ( _X_ ), proposed updated semantics _ϕ_ new( _X_ ), temporary semantics Φ( _X_ )[temp] , residual **R** , target values **y** 

2: **Output:** Updated tree _ϕi_ , updated semantics Φ( _X_ ) 

3: **if** _ϕ_ new = _∅_ **then** 4: **continue** _▷_ No suitable replacement found 5: **end if** 6: **if** _∥_ Φ(Φ( _XX_ )) _∥_ 2[=] _∥ϕϕ_ newnew(( _XX_ )) _∥_ 2 **[then]** 7: **continue** _▷_ Identical semantics; skip replacement 8: **end if** 9: _β ←_[(] _[ϕ]_[new] ( _ϕ_[(] new _[X]_[)] ( _[−] X[ϕ]_[¯] )[new] _−ϕ_[¯][(] new _[X]_[))] ( _X[·]_[(] **[R]** ))[2] _[−]_ **R**[¯] ) _▷_ Compute scaling factor 10: _α ←_ **R**[¯] _− β · ϕ_[¯] new( _X_ ) _▷_ Compute bias term 11: Φ( _X_ )[trial] _←_ Φ( _X_ )[temp] + _β · ϕ_ new( _X_ ) + _α ▷_ Propose updated semantics 12: MSEtrial _← |X_ 1 _|_ � _j∈X_[(Φ(] _[X][j]_[)][trial] _[ −]_ **[y]** _[j]_[)][2] 1 13: MSEcurrent _← |X|_ � _j∈X_[(Φ(] _[X][j]_[)] _[ −]_ **[y]** _[j]_[)][2] 14: **if** MSEtrial _≤_ MSEcurrent **then** 15: _ϕi ← ϕ_ new _▷_ Accept replacement 16: Φ( _X_ ) _←_ Φ( _X_ )[trial] 17: **end if** 

the replacement process and calculate the new residual after the replacement. The detailed logic is presented in Algorithm 4. Specifically, Lines 9 and 10 describe how to compute the scaling coefficient _β_ and bias term _α_ , respectively, based on the semantics of the newly proposed feature, _ϕ_ new( _X_ ), and the desired semantics, **R** . The algorithm then calculates and compares the error before and after the replacement, denoted by MSEcurrent and MSEtrial, respectively. As shown in Line 15, if the error decreases after the replacement, the new feature is accepted, and the semantics are updated. Otherwise, the original feature is retained. 

## E TRAVERSAL ALGORITHM 

**Prediction to Symbolic Tree** The output format of the neural network consists of a sequence of _nF_ + _nT_ elements, where the first _nF_ elements represent functions or terminals, and the remaining _nT_ elements represent terminals. For example, the output could be [ _mul, abs, add, x_ 1 _, x_ 2 _, x_ 3], which corresponds to the mathematical expression _mul_ ( _abs_ ( _x_ 1) _, add_ ( _x_ 2 _, x_ 3)), a pre-order traversal of a symbolic tree. Thus, to decode the output, a conversion from level-order traversal to pre-order traversal is required, as described in Algorithm 6. Since the neural network does not use an explicit end token, some terminal elements may remain unused after decoding and are discarded. As shown in line 5, a tracking variable _τ_ is used to monitor whether a full symbolic tree has been formed. Once a complete tree is generated, the decoding process stops, and any remaining tokens are discarded. Additionally, during decoding, if a constant placeholder is predicted (line 18), a random constant is sampled from the uniform distribution _U_ [ _−_ 1 _,_ 1]. This delegates the task of constant tuning to the evolutionary algorithm rather than relying on the neural network to predict the constant accurately. 

**Symbolic Tree to Training Target** For training data, symbolic expressions such as _mul_ ( _abs_ ( _x_ 1) _, add_ ( _x_ 2 _, x_ 3)) are represented as [ _mul, abs, x_ 1 _, add, x_ 2 _, x_ 3]. However, to serve as proper training labels, this representation must be reordered to reflect a level-order traversal. The algorithm for converting a list of pre-order traversals into level-order representations is detailed in Algorithm 6. The key idea is to collect the required number of child nodes at each level (line 9) and then recursively process these child nodes to obtain subtrees in pre-order traversal (line 13). Finally, the subtrees are assembled into an integral tree in pre-order traversal (line 17). 

## F NEURAL NETWORK TRAINING DETAILS 

## F.1 PARAMETER SETTINGS 

To encourage the neural network to escape local optima, we employ cosine annealing with warm restarts (Loshchilov & Hutter, 2017). Additionally, the Adam optimizer with decoupled weight 

18 

Published as a conference paper at ICLR 2025 

**Algorithm 5** Preorder Traversal to Level Order Traversal 

1: **procedure** LEVELORDERTRAVERSAL( _N_ ) _▷ N_ is the list of node names 2: _τ ←_ 0 _▷_ Initialize tracking variable _τ_ 3: _S ←∅ ▷_ Initialize an empty list _S_ 4: **for** each _ni ∈N_ **do** _▷_ Iterate over each node _ni_ in _N_ 5: **if** _τ_ = 0 **then** _▷_ Exit loop if a complete tree has been formed 6: **break** 7: **end if** 8: **if** _ni ∈F_ **then** _▷_ Check if _ni_ is a function in _F_ 9: _f ←F_ [ _ni_ ] 10: _α ←_ arity( _f_ ) _▷_ Get the arity _α_ of function _f_ 11: _τ ← τ −_ 1 12: _τ ← τ_ + _α_ 13: _S ←S ∪{f } ▷_ Append _f_ to list _S_ 14: **else if** _ni ∈T_ **then** _▷_ Check if _ni_ is a terminal in _T_ 15: _τ ← τ −_ 1 16: _t ←T_ [ _ni_ ] 17: **if** _t_ is a constant **then** 18: _t ←U_ [ _−_ 1 _,_ 1] 19: **end if** 20: _S ←S ∪{t} ▷_ Append _t_ to list _S_ 21: **end if** 22: **end for** 23: **return** _S ▷_ Return the final list _S_ representing the tree 24: **end procedure** 

**Algorithm 6** Level Order Traversal to Preorder Traversal 

1: **procedure** PREORDERTRAVERSAL( _N_ ) 2: **if** _N_ = _∅_ **then** _▷_ Return an empty list if no nodes are left 3: **return** _∅_ 4: **end if** 5: _n_ root _←_ pop( _N ,_ 0) _▷_ Get the first node 6: **if** _n_ root is a **Primitive then** _▷_ Check if the root is a function 7: _A ←∅ ▷_ Initialize argument list 8: _R ←∅ ▷_ Initialize root argument list 9: **for** _i ←_ 1 to arity( _n_ root) **do** _▷_ Collect the function’s arguments 10: _ni ←_ pop( _N ,_ 0) 11: _R ←R ∪{ni}_ 12: **end for** 13: **for** each _ni ∈R_ **do** _▷_ Reorder arguments recursively 14: _Ni ←_ PreorderTraversal( _{ni} ∪N_ ) 15: _A ←A ∪Ni_ 16: **end for** 17: **return** _{n_ root _} ∪A ▷_ Return the function followed by its subtrees 18: **else** 19: **return** _{n_ root _} ▷_ Return the terminal as is 20: **end if** 21: **end procedure** 

decay (Loshchilov & Hutter, 2019) is used to mitigate overfitting, with a weight decay parameter of 10 _[−]_[4] . For the MLP initialization, we apply Kaiming initialization (He et al., 2015) to ensure appropriate scaling of the weights. The number of training epochs is set to 1000. 

19 

Published as a conference paper at ICLR 2025 

Table 2: Functions and Corresponding Expressions 

|**Primitive**|**Expression**|**Expression**|
|---|---|---|
||||
|Add<br>Sub<br>Mul<br>AQ<br>Sqrt<br>AbsLog<br>Abs<br>Square<br>Max<br>Min<br>Neg<br>Sin_π_<br>Cos_π_|_a_+_b_<br>_a −b_<br>_a · b_<br>_a_<br>~~_√_~~<br>1+_b_2<br>_√_<br>_a_<br>log(_|a|_+ 1))<br>_|a|_<br>_a_2<br>max(_a, b_)<br>min(_a, b_)<br>_−a_<br>sin(_π × a_)<br>cos(_π × a_)||



## F.2 TRAINING PROCEDURE 

The pseudocode for training the neural network is provided in Algorithm 7. First, the data is split into training and validation sets, as shown in Line 2. The data consists of pairs of semantics and GP trees from the library _L_ , excluding GP trees that exceed the limit on the number of function nodes. **X** train represents the semantics of GP trees, and **y** train represents the nodes of GP trees corresponding to the semantics in **X** train. Similarly, **X** val and **y** val are the corresponding validation data with analogous meanings. Next, the data is augmented using the data augmentation strategy introduced in Section 3.3. Subsequently, retrieval trees are prepared for each instance in both the training and validation data. 

To avoid trivializing the training task as a simple retrieval task, the KD-Tree is initially constructed using only the training set, as shown in Line 4. When preparing the training data in Line 4, the second-nearest tree is selected as the nearest GP tree to form the sets of retrieved trees, **N** train, since the nearest tree corresponds to the GP tree itself. For the validation set, as the KD-Tree includes only the training set in Line 5, the nearest GP tree is used as the retrieved tree to form the set **N** val. After retrieving the required samples, the KD-Tree is reconstructed using a combination of the training and validation sets, as shown in Line 7. 

To save computational resources during training of the neural semantics library, we employ three strategies. First, the neural network is continuously trained during the evolution process. Second, at the beginning of training, as shown in Line 9, the validation loss is checked to determine if the current validation set loss, _L_[current] val , is smaller than the previous validation set loss, _L_[prev] val[.][If][the] validation loss does not degrade, training is skipped to save computational resources. Notably, the construction of the KD-Tree precedes this step, ensuring that the retrieval mechanism can always access the latest information without requiring updates to the neural network. This is a key advantage of RAG-based neural generation. Lastly, the neural network tracks the validation loss throughout the training process. If the validation loss does not improve over _τ_ iterations, the algorithm performs early stopping to conserve computational resources. These three mechanisms collectively ensure that RAG-SR is an efficient SR algorithm. 

## G DETAILS OF GENETIC PROGRAMMING PARAMETERS 

The crossover and mutation rates for GP are set to 0.9 and 0.1, respectively. The depth limit of GP trees is set to 10. The function set used in GP is listed in Table 2. To prevent division by zero errors, the division operator is replaced with the analytical quotient operator (Ni et al., 2012). Random constants are drawn from a uniform distribution, _U_ [ _−_ 1 _,_ 1]. 

20 

Published as a conference paper at ICLR 2025 

## **Algorithm 7** Training NN 

1: **Input:** Training data _D_ train, batch size _B_ , epochs _T_ , learning rate _η_ , validation split _α_ , patience _τ_ , last validation loss _L_[prev] val 2: ( **X** train _,_ **X** val _,_ **y** train _,_ **y** val) _←_ SplitData( _D_ train _, α_ ) 3: ( **X** train _,_ **X** val _,_ **y** train _,_ **y** val) _←_ AugmentData( **X** train _,_ **X** val _,_ **y** train _,_ **y** val) 4: ( **N** train _,_ **X** train _,_ **y** train) _←_ RetrieveTrees( **X** train _,_ **y** train) 5: _K ←_ BuildKDTree( **X** train _,_ **y** train) _▷ K_ is the KD-Tree 6: ( **N** val _,_ **X** val _,_ **y** val) _←_ PrepareValidationSet( _K,_ **X** val _,_ **y** val) 7: _K ←_ BuildKDTree( **X** train _∪_ **X** val _,_ **y** train _∪_ **y** val) _▷_ Reconstruct KD-Tree using full data 8: _L_[current] val _←_ CalculateValidationLoss( **X** val _,_ **y** val _,_ **N** val) 9: **if** _L_[current] val _< L_[prev] val **[then]** 10: **return** _▷_ Skip training if validation loss improves 11: **end if** 12: _L_[best] val _[←∞] ▷_ Initialize best validation loss 13: _C_ patience _←_ 0 _▷_ Initialize patience counter 14: **for** _t_ = 1 to _T_ **do** _▷_ Train using augmented data and retrieved data 15: TrainBatch( **X** train _,_ **y** train _,_ **N** train) 16: _L_ val _←_ CalculateValidationLoss( **X** val _,_ **y** val _,_ **N** val) 17: **if** _L_ val _< L_[best] val **[then]** 18: _L_[best] val _[←][L]_[val] _▷_ Update best validation loss 19: _C_ patience _←_ 0 _▷_ Reset patience counter 20: **else** 21: _C_ patience _← C_ patience + 1 _▷_ Increment patience counter 22: **end if** 23: **if** _C_ patience _> τ_ **then** 24: **Break** _▷_ Early stopping 25: **end if** 26: **end for** 27: **return** 

## H EFFECT OF CONTRASTIVE LEARNING 

The contrastive learning mechanism is designed to capture relationships between samples within each batch. To evaluate its effectiveness in enhancing representation learning, we present the pairwise correlation between the encodings of MLP in Figure 9. The first two figures show the correlation of the encoded desired semantics **F**[mapped] MLP for 16 randomly selected test instances, with and without contrastive learning, respectively. The third figure shows the raw correlation between desired semantics, while the fourth figure presents the absolute correlation, which is a more appropriate metric for feature construction-based symbolic regression techniques. The results indicate that contrastive learning successfully captures two desired semantics with opposite signs that are highly correlated and represent the same symbolic expression. In contrast, without contrastive learning, the neural network lacks this knowledge and may generate significantly different trees when queried with an opposite sign, as it fails to recognize their equivalence. These results highlight the benefit of using contrastive learning to jointly train the intention encoding and retrieval augmentation encoding components. 

## I SENSITIVITY OF THE WEIGHT OF CONTRASTIVE LOSS 

In this section, we analyze the sensitivity of the contrastive loss weight _λ_ in RAG-NN. Specifically, we evaluate five different _λ_ values: 0.01, 0.025, 0.05, 0.1, and 0.2. The impact of varying _λ_ on the edit distance of the test set is shown in Figure 10. The results demonstrate that the weight of the contrastive loss significantly influences accuracy. While the contrastive loss can improve performance, the weight _λ_ must be carefully tuned using cross-validation. Otherwise, the contrastive loss may fail to achieve optimal performance. Regarding training time, the effect of different _λ_ values on runtime is presented in Figure 11. Overall, the runtime remains stable across different _λ_ values. Therefore, when tuning _λ_ , accuracy should be the primary consideration. 

21 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0022-01.png)


**----- Start of picture text -----**<br>
With Contrastive Learning Without Contrastive Learning<br>R2: 0.7506 R2: 0.2735<br>1.00 0.22 0.35 0.24 0.16 0.68 0.10 0.43 0.20 0.24 1.00 0.20 0.52 0.16 0.22 0.31 0.52 0.35 0.07 0.15<br>0.22 1.00 0.50 0.17 0.10 0.14 0.64 0.59 0.17 0.15 0.20 1.00 0.26 0.57 0.07 0.21 0.41 0.15 0.05 0.20<br>0.35 0.50 1.00 0.20 0.64 0.05 0.35 0.79 0.20 0.54 0.52 0.26 1.00 0.28 0.23 0.38 0.21 0.11 0.29 0.08<br>0.24 0.17 0.20 1.00 0.06 0.35 0.15 0.20 0.92 0.22 0.16 0.57 0.28 1.00 0.00 0.35 0.01 0.15 0.35 0.07<br>0.16 0.10 0.64 0.06 1.00 0.16 0.10 0.44 0.10 0.59 0.22 0.07 0.23 0.00 1.00 0.69 0.14 0.68 0.40 0.27<br>0.68 0.14 0.05 0.35 0.16 1.00 0.04 0.26 0.29 0.07 0.31 0.21 0.38 0.35 0.69 1.00 0.08 0.59 0.72 0.19<br>0.10 0.64 0.35 0.15 0.10 0.04 1.00 0.27 0.23 0.20 0.52 0.41 0.21 0.01 0.14 0.08 1.00 0.23 0.34 0.02<br>0.43 0.59 0.79 0.20 0.44 0.26 0.27 1.00 0.12 0.29 0.35 0.15 0.11 0.15 0.68 0.59 0.23 1.00 0.22 0.35<br>0.20 0.17 0.20 0.92 0.10 0.29 0.23 0.12 1.00 0.19 0.07 0.05 0.29 0.35 0.40 0.72 0.34 0.22 1.00 0.12<br>0.24 0.15 0.54 0.22 0.59 0.07 0.20 0.29 0.19 1.00 0.15 0.20 0.08 0.07 0.27 0.19 0.02 0.35 0.12 1.00<br>0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9<br>Embedding Index Embedding Index<br>Raw Correlation Absolute Correlation<br>1.00 -0.11 0.31 0.04 -0.23 -1.00 0.11 -0.31 -0.04 0.23 1.00 0.11 0.31 0.04 0.23 1.00 0.11 0.31 0.04 0.23<br>-0.11 1.00 -0.11 0.28 -0.05 0.11 -1.00 0.11 -0.28 0.05 0.11 1.00 0.11 0.28 0.05 0.11 1.00 0.11 0.28 0.05<br>0.31 -0.11 1.00 0.06 -0.52 -0.31 0.11 -1.00 -0.06 0.52 0.31 0.11 1.00 0.06 0.52 0.31 0.11 1.00 0.06 0.52<br>0.04 0.28 0.06 1.00 -0.05 -0.04 -0.28 -0.06 -1.00 0.05 0.04 0.28 0.06 1.00 0.05 0.04 0.28 0.06 1.00 0.05<br>-0.23 -0.05 -0.52 -0.05 1.00 0.23 0.05 0.52 0.05 -1.00 0.23 0.05 0.52 0.05 1.00 0.23 0.05 0.52 0.05 1.00<br>-1.00 0.11 -0.31 -0.04 0.23 1.00 -0.11 0.31 0.04 -0.23 1.00 0.11 0.31 0.04 0.23 1.00 0.11 0.31 0.04 0.23<br>0.11 -1.00 0.11 -0.28 0.05 -0.11 1.00 -0.11 0.28 -0.05 0.11 1.00 0.11 0.28 0.05 0.11 1.00 0.11 0.28 0.05<br>-0.31 0.11 -1.00 -0.06 0.52 0.31 -0.11 1.00 0.06 -0.52 0.31 0.11 1.00 0.06 0.52 0.31 0.11 1.00 0.06 0.52<br>-0.04 -0.28 -0.06 -1.00 0.05 0.04 0.28 0.06 1.00 -0.05 0.04 0.28 0.06 1.00 0.05 0.04 0.28 0.06 1.00 0.05<br>0.23 0.05 0.52 0.05 -1.00 -0.23 -0.05 -0.52 -0.05 1.00 0.23 0.05 0.52 0.05 1.00 0.23 0.05 0.52 0.05 1.00<br>0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9<br>Embedding Index Embedding Index<br>0 0<br>1 1<br>2 2<br>3 3<br>4 4<br>5 5<br>Embedding Index Embedding Index<br>6 6<br>7 7<br>8 8<br>9 9<br>0 0<br>1 1<br>2 2<br>3 3<br>4 4<br>5 5<br>Embedding Index Embedding Index<br>6 6<br>7 7<br>8 8<br>9 9<br>**----- End of picture text -----**<br>


Figure 9: Effect of contrastive learning on pairwise correlation of MLP encodings. 

## J DIFFERENT NUMBER OF VARIABLES 

The effect of the proposed components in the neural semantic library may vary with the number of variables. In this paper, we study the impact of the number of variables on the results. The experiments in Section 4.1 are conducted with a variable count of 10. For comparison, additional experiments with variable counts of _{_ 15, 20, 25, 30, 50 _}_ are also performed. Since more variables mean more possible symbolic expressions, the number of symbolic expressions sampled is scaled by the ratio between the number of variables and 10. The results show that the conclusions of the ablation studies in Section 4.1 remain consistent across different numbers of variables. Specifically, both RAG and DA are important contributors to the effectiveness of the neural semantic library. Interestingly, the impact of RAG becomes more significant as the number of variables increases. A possible reason for this trend is that the data might be imbalanced, with some variables being 

22 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0023-01.png)


**----- Start of picture text -----**<br>
Distribution of Edit Distance by Configuration Median Running Time by Configuration<br>4.05 60<br>3.97<br>4.00 3.92 3.87 3.91 44.41 43.86 46.17<br>40 39.30 36.61<br>3.95 3.82<br>3.90 20<br>3.85<br>0<br>3.80<br>index<br>Figure 10: Impact of varying λ on edit distance Figure 11: Impact of varying  λ  on running time.<br>of the test set.<br>Edit Distance by Number of Variables Training Time by Number of Variables<br>3 . 82 281 . 9 6<br>3.90 3.83<br>300<br>3.85<br>3.76<br>3.75<br>3.80 3 . 76 200<br>3.72 134.69<br>3.75 103.50 104.60<br>100 67 . 54<br>44.41<br>3.70<br>10 15 20 25 30 50 10 15 20 25 30 50<br>Number of Variables Number of Variables<br>Figure 12: Impact of varying the number of vari- Figure 13: Impact of varying the number of vari-<br>ables on edit distance on the test set. ables on running time.<br>Distribution of Edit Distance by Configuration Median Running Time by Configuration<br>4 . 17 100<br>4.15 87.15 83.19<br>4.15 77.23<br>4.2 75 67.54<br>3.94 62.38<br>51.15<br>4.0 3.77 3.93 50 44.61<br>3.8 3.72 3.76 25<br>0<br>Figure 14: Ablation study of components based Figure 15: Ablation study of components with<br>on edit distance on the test set (15 variables). respect to running time (15 variables).<br>0.01 0.025 0.05 0.1 0.2 0.01 0.025 0.05 0.1 0.2<br>All W/O DQ W/O CLW/O Dropout W/O DA W/O NNW/O DA+DQW/O RAG All W/O DQ W/O CLW/O Dropout W/O DAW/O DA+DQ W/O RAG<br> Running Time (s)<br>Edit Distance<br>Median<br>Edit Distance<br>Running Time (s)<br>)<br>Edit Distance<br>Median Running Time (s<br>**----- End of picture text -----**<br>


Figure 10: Impact of varying _λ_ on edit distance of the test set. 

underrepresented. In such cases, the retrieval process can extract the relevant variables, even if they represent only a small portion of the data. Therefore, the RAG technique becomes increasingly important as the number of variables increases. More analysis of why RAG could be more effective in high-dimensional scenarios is provided in Appendix K. 

To better visualize how the number of variables influences the effectiveness of the neural semantic library, we present the change in edit distance with respect to the number of variables in Figure 12 and the running time with respect to the number of variables in Figure 13. The results indicate that the model can maintain a stable level of prediction accuracy with different number of variables. This indicates that the proposed method demonstrates good scalability for tasks with varying numbers of variables. In comparison, the end-to-end Transformer for symbolic regression (Kamienny et al., 2022) can only be used to train tasks with up to 10 variables. Regarding training time, an increase in the number of variables does lead to a longer training time; however, the increase is modest. These results demonstrate that the neural semantic library is also efficient for tasks with different numbers of variables. 

23 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0024-01.png)


**----- Start of picture text -----**<br>
Distribution of Edit Distance by Configuration Median Running Time by Configuration<br>4 . 29 4 . 35<br>4.4 103.50 107.82<br>100 92.88 95.55<br>4.2 3.87 4.03 4.18<br>52.91 53.93<br>4.0 50 43.67<br>3.76 3.79 3.81<br>3.8<br>0<br>Figure 16: Ablation study of components based Figure 17: Ablation study of components with<br>on edit distance on the test set (20 variables). respect to running time (20 variables).<br>Distribution of Edit Distance by Configuration Median Running Time by Configuration<br>4 . 48<br>4.28<br>4.50 111.26 104.60 109.75<br>4.07 100 96.34 86.37 92.97<br>4.25 4.13<br>59.68<br>4.00 3.74 3.76 3.80 3.90 50<br>3.75<br>0<br>Figure 18: Ablation study of components based Figure 19: Ablation study of components with<br>on edit distance on the test set (25 variables). respect to running time (25 variables).<br>Distribution of Edit Distance by Configuration Median Running Time by Configuration<br>4 . 36<br>4.50 4.09 150 134.69 128.30 129.21 121.96 127.40<br>4.25 4.12<br>3.81 3.89 100 92.60 92.72<br>4.00 3.75 3.76 3.78<br>50<br>3.75<br>0<br>Figure 20: Ablation study of components based Figure 21: Ablation study of components with<br>on edit distance on the test set (30 variables). respect to running time (30 variables).<br>Distribution of Edit Distance by Configuration Median Running Time by Configuration<br>4 . 64<br>4.50<br>5.0 400<br>4.08<br>300 281.96 273.63<br>4.5 3.83 3.86 3.88 4.06 4.18 200 242.66 221.22 212.43 137.28 246.85<br>4.0 100<br>0<br>All W/O DQW/O Dropout W/O CL W/O DA W/O NN W/O RAGW/O DA+DQ All W/O DQW/O Dropout W/O CL W/O DA W/O RAGW/O DA+DQ<br>W/O CL All W/O DQW/O Dropout W/O DA W/O NNW/O DA+DQW/O RAG W/O CL All W/O DQW/O Dropout W/O DAW/O DA+DQ W/O RAG<br>All W/O CL W/O DQW/O Dropout W/O DAW/O DA+DQ W/O NN W/O RAG All W/O CL W/O DQW/O Dropout W/O DAW/O DA+DQ W/O RAG<br>All W/O CL W/O DQW/O Dropout W/O DA W/O NNW/O DA+DQW/O RAG All W/O CL W/O DQW/O Dropout W/O DAW/O DA+DQ W/O RAG<br>)<br>Edit Distance<br>Median Running Time (s<br>)<br>Edit Distance<br>Median Running Time (s<br>)<br>Edit Distance<br>Median Running Time (s<br>)<br>Edit Distance<br>Median Running Time (s<br>**----- End of picture text -----**<br>


Figure 22: Ablation study of components based on edit distance on the test set (50 variables). 

Figure 23: Ablation study of components with respect to running time (50 variables). 

24 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0025-01.png)


**----- Start of picture text -----**<br>
Distribution of Terminals Distribution of Terminals<br>130 129 124 129 124<br>100 113 109 108 105 104 103 94 100 109 105 104<br>50 50 38 35 30 27 19<br>0 0<br>Figure 24: Distribution of terminals with 10 vari- Figure 25: Distribution of terminals with 10 vari-<br>ables. ables and an imbalance ratio of 0.25.<br>Edit Distance without RAG Edit Distance by Imbalance Ratio<br>4 . 37 0 . 45<br>4.39 4.37 0.5<br>4.4 0.36 0.29<br>0.4<br>4.3 0.29<br>0.3<br>4.2 4 . 15<br>0.2<br>4.1<br>0.25 0.5 0.75 No Imbalance 0.25 0.5 0.75 No Imbalance<br>Imbalance Ratio Imbalance Ratio<br>Figure 26: Edit distance across different imbal- Figure 27: Performance improvement using<br>ance ratios without retrieval-augmented genera- retrieval-augmented generation across different<br>tion. imbalance ratios.<br>ARG9 ARG3 ARG5 ARG2 ARG4 ARG7 ARG8 ARG1 ARG6 ARG0 ARG3 ARG5 ARG4 ARG8 ARG1 ARG9 ARG6 ARG7 ARG0 ARG2<br>Counts Counts<br>Edit Distance<br>Improvement on Edit Distance<br>**----- End of picture text -----**<br>


## K RETRIEVAL AUGMENTATION FOR IMBALANCED DATA 

In Appendix J, RAG has been shown to be particularly beneficial when handling a large number of variables. One potential explanation is the increased data imbalance as the number of variables grows. Specifically, the frequency distribution of variables at position 3 (starting from 0) for 10 and 50 variables is depicted in Figure 24 and Figure 28, respectively. With 10 variables, the most frequent variable appears 49 times and the least frequent appears 37 times, showing a relatively balanced distribution. However, for 50 variables, the most frequent variable appears 26 times while the least frequent variable appears only 7 times, indicating a significant imbalance. This confirms that larger variable sets lead to more imbalanced data distributions. 

To validate the effectiveness of the RAG mechanism in handling imbalanced data, we randomly subsample half of the terminals with varying ratios to create imbalanced data for the case of 10 variables. For training data with selected terminals at position 3, subsampling retained the data with a probability equal to the subsampling ratio _s_ . The subsampling ratios are [0 _._ 25 _,_ 0 _._ 5 _,_ 0 _._ 75]. Figure 25 shows the distribution after subsampling with a ratio of 0.25. The experimental results are shown in Figure 26 and Figure 27. As shown in Figure 26, the edit distance worsens as the subsampling ratio increases, confirming that imbalanced training data negatively impacts performance. The results in Figure 27 demonstrate that the performance improvement from using RAG becomes more pronounced as the subsampling ratio increases. This indicates that RAG effectively mitigates the issues associated with imbalanced data, explaining why its impact is more significant when dealing with larger numbers of variables. 

## L FURTHER ANALYSIS ON THE SYMBOLIC REGRESSION BENCHMARK 

## L.1 COMPARISON ON LOW-DIMENSIONAL BLACK-BOX DATASETS 

In the domain of deep symbolic regression, low-dimensional subsets of black-box datasets are widely used as evaluation criteria. Specifically, we select all datasets from the PMLB benchmark with 10 or fewer variables and no categorical features (Kamienny et al., 2022; Shojaee et al., 2024a; Meidani et al., 2024). Figure 29 shows the results under these criteria. Compared to Figure 5, the rel- 

25 

Published as a conference paper at ICLR 2025 

Distribution of Terminals ~~232323~~ 20 2121212020191919191918181818181818171717171716161616161515151514141414141414131312121211111010 10 0 

Figure 28: Distribution of functions and terminals in the case of 50 variables. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0026-03.png)


**----- Start of picture text -----**<br>
R 2 ��"+, �(!"%��$." �* $'$'#��$&"��+�<br>�������<br>����*""<br>�����<br>��)"*('<br>�����<br>�������<br>�����<br>������<br>���<br>���������<br>����<br>� '!(&�(*"+,<br>�����<br>�! �((+,<br>�"*'"%�$!#"<br>�������<br>����<br>����<br>�#)%" *'<br>���<br>����<br>�����<br>�$'" *<br>����<br>� ���"-'& '<br>����� ���� ���� ���� ���� ���� �� � �� � �� � �� � �� � �� � �� � �� �<br>**----- End of picture text -----**<br>


Figure 29: _R_[2] scores, model sizes, and training time of 25 algorithms on low-dimensional black-box regression problems. 

ative rankings of the algorithms remain largely unchanged. Therefore, the conclusions in Section 4.2 remain valid in this case. 

## L.2 COMPARISON ON FEYNMAN AND STROGATZ DATASETS 

**Experimental Settings** The Feynman and Strogatz datasets are widely used synthetic benchmarks for symbolic regression. AI-Feynman consists of 119 datasets, while Strogatz contains 14 datasets. For RAG-SR, we employ a function set relevant to the physics domain, including the operations + _, −, ∗, /,_ sin _,_ cos. The division operator is implemented as a protected division to handle edge cases, capping extremely small positive and negative denominators at 10 _[−]_[10] and _−_ 10 _[−]_[10] , respectively. The training protocol follows the SRBench standard. All baseline algorithms use 10,000 subsampled training instances from the AI-Feynman datasets (La Cava et al., 2021). However, our method subsamples only 500 instances from the Feynman datasets to reduce computational costs. No noise is added to the labels. For the Strogatz datasets, we use all available training instances. Following the definition in SRBench (La Cava et al., 2021), accuracy is defined as 1 if the _R_[2] value exceeds 0.999, and 0 otherwise. 

**Experimental Results** The experimental results on the Feynman datasets are shown in Figure 30. Algorithms are ranked based on their test _R_[2] scores on the Strogatz datasets. As shown, RAGSR achieves the highest test _R_[2] scores on the Strogatz datasets, demonstrating the strong learning capability of the proposed method. On the Feynman datasets, RAG-SR ranks second, trailing only MRGP (Arnaldo et al., 2014). However, the model size of MRGP is more than an order of magnitude larger than RAG-SR, while RAG-SR achieves a good balance. In comparison to deep learning-based symbolic regression methods such as TPSR (Shojaee et al., 2024a) and SNIP (Meidani et al., 2024), 

26 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0027-01.png)


**----- Start of picture text -----**<br>
���+(��,��&#+*"&%���* ���� �&� #��"-  �(�"%"%!��"$ ��)�<br>������<br>�' (&%<br>������<br>����<br>��������<br>����<br>����<br>��� ,%$�% � � * �� � (&+'<br>������ � ,%$�%<br>���� �*(&!�*-<br>���<br>�����<br>���<br>����<br>���<br>!'# �(%<br>���<br>� �� �� �� �� ��� �� � �� � �� � �� � �� � �� � �� �<br>**----- End of picture text -----**<br>


Figure 30: _R_[2] scores, model sizes, and training time of 17 algorithms on 119 Feynman and 14 Strogatz datasets. 

RAG-SR outperforms them by a large margin, further validating the effectiveness of leveraging a neural semantic library in symbolic regression. 

## M PARAMETER SENSITIVITY ANALYSIS 

## M.1 INFLUENCE OF _P_ NEURAL 

_P_ neural controls the rate at which the neural semantic library is used to generate new individuals. In this section, we analyze the influence of _P_ neural after evaluating 2000 solutions. Each probability value of _P_ neural was tested on 120 black-box datasets, with each experiment repeated 30 times to ensure stability. The results, shown in Figure 31, demonstrate that the test _R_[2] scores gradually improve as the probability of using the neural semantic library increases. Since neural generation is more time-consuming than retrieval, the default value of 0.1, which corresponds to applying neural generation to one out of ten GP trees in each solution during replacement, is a reasonable choice that provides a good balance between accuracy and efficiency. Nonetheless, in cases where a GPU is available to provide faster inference, a higher value of _P_ neural can yield better results. 

## M.2 INFLUENCE OF NUMBER OF TREES 

The number of trees determines the number of constructed features in each solution. RAG-SR follows conventions in the evolutionary feature construction literature, using 10 trees as the default parameter. In this section, we compare three configurations for the number of trees in each solution: 5, 10, and 15. To ensure stability, each configuration was tested on 120 black-box datasets, with each experiment repeated 30 times. The results after evaluating 2000 solutions are shown in Figure 32. The results indicate a substantial performance gap between using 10 trees and using 5 trees. Several factors may contribute to this improvement. First, a larger number of GP trees in each solution provides more candidates to be stored in the library, allowing neural generation and exact retrieval to identify better GP trees. Additionally, having more trees in each solution enables the base learner to more flexibly select appropriate features for making predictions, which may further enhance performance. However, increasing the number of trees also leads to an increase in model size and evaluation time for each solution, as shown in Figure 33 and Figure 34. Therefore, setting the number of trees to 10 strikes a good balance between accuracy, model size, and training time. 

27 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG_SR_Retrieval_Augmente_images/iclr_2025_RAG_SR_Retrieval_Augmente.pdf-0028-01.png)


**----- Start of picture text -----**<br>
Test R 2  by Pneural Test R 2  by Number of Trees<br>0 . 927 0 . 927 0 . 937<br>0.93<br>0.920 0.925 0.920<br>0.917<br>0.92 0.900<br>0.907<br>0.91 0.875<br>0.847<br>0.850<br>0.90<br>0.0 0.05 0.1 0.2 0.5 5 10 15<br>Pneural Number of Trees<br>Figure 31: Sensitivity of test  R [2] score to  P neural. Figure 32: Sensitivity of test  R [[2]] score to number<br>of trees.<br>Model Size by Number of Trees Training Time by Number of Trees<br>119 . 75 0 525 . 15 8<br>120<br>500<br>100<br>403.342<br>81.117 400<br>80<br>60 300<br>44.617 258.325<br>5 10 15 5 10 15<br>Number of Trees Number of Trees<br>Figure 33: Sensitivity of model size to number of Figure 34: Sensitivity of training time to number<br>trees. of trees.<br>2R 2R<br>Test  Test<br>Model Size<br>raining Time (s)<br>T<br>**----- End of picture text -----**<br>


Figure 32: Sensitivity of test _R_[[2]] score to number of trees. 

## N RELATED WORK 

## N.1 RAG FOR CODE GENERATION 

The Retrieval-Augmented Generation (RAG) technique has been widely used in the domain of code generation to enhance accuracy by retrieving code from open-source databases (Parvez et al., 2021; Wang et al., 2024). Similarly, RAG has also been applied to code transformation tasks, demonstrating improved performance compared to zero-shot code translation (Bhattarai et al., 2024). These RAG-based techniques typically rely on existing repositories (Parvez et al., 2021). In this paper, we demonstrate that RAG is not restricted to pre-existing code repositories but can also be applied to code generated by evolutionary algorithms. Furthermore, we show that data augmentation can improve prediction accuracy when using RAG in conjunction with neural networks for symbolic expression generation. This suggests that, for code generation, employing synonym replacement or paraphrasing models to augment code summaries could further enhance the efficiency of RAG-based systems. 

## N.2 VQ-VAE FOR CODE GENERATION 

In the program synthesis domain, Vector Quantized-Variational Autoencoder (VQ-VAE) (Van Den Oord et al., 2017) has been employed to learn discrete representations from input-output pairs, enabling the capture of high-level concepts (Hong et al., 2021). Both VQ-VAE and RAG leverage continuous semantics to retrieve discrete information for code generation. However, VQ-VAE utilizes an autoencoder to learn high-level conceptual representations from data, whereas RAG directly retrieves relevant information from a database. Compared to VQ-VAE, the RAG technique offers greater interpretability for end-users by allowing inspection of the retrieved information. Moreover, RAG provides an efficient mechanism to update knowledge without requiring retraining of the neural network. Although there are many differences between VQ-VAE and RAG, these techniques are not mutually exclusive. Integrating them in the context of neural symbolic regression presents a promising avenue for future research. 

28 

Published as a conference paper at ICLR 2025 

Table 3: Examples of symbolic trees generated by the retrieval-augmented neural network when the ground truth exceeds the generation limit. 

|RAG-NN Generated Tree (Distance)|RAG-NN Generated Tree (Distance)||Ground Truth Tree|
|---|---|---|---|
|abs|~~l~~og(abs<br>~~l~~og(abs<br>~~l~~og(ARG1))) (3)|absolute(abs<br>~~l~~og(negative(absolute(abs<br>~~l~~og(abs<br>~~l~~og(ARG1))))))||
|cos<br>~~p~~i(sin<br>~~p~~i(cos<br>~~p~~i(ARG5))) (3)|||abs<br>~~l~~og(absolute(cos<br>~~p~~i(sin<br>~~p~~i(sqrt(cos<br>pi(ARG5))))))|
|abs<br>~~l~~og(absolute(abs<br>~~l~~og(ARG7))) (3)|||abs<br>~~l~~og(absolute(sqrt(absolute(sqrt(abs<br>log(ARG7))))))|
|sin<br>~~p~~i(square(sin<br>~~p~~i(ARG8))) (3)||negative(cos<br>~~p~~i(absolute(sin<br>pi(square(sin<br>~~p~~i(ARG8))))))||
||abs<br>~~l~~og(cos<br>~~p~~i(ARG8)) (4)||cos<br>~~p~~i(abs<br>~~l~~og(sin<br>~~p~~i(cos<br>~~p~~i(abs<br>~~l~~og(cos<br>~~p~~i(ARG8))))))|
||sqrt(sqrt(ARG4)) (4)||sin<br>~~p~~i(abs<br>~~l~~og(abs<br>~~l~~og(sqrt(abs<br>~~l~~og(sqrt(ARG4))))))|
||abs<br>~~l~~og(sin<br>pi(ARG1)) (4)||abs<br>~~l~~og(negative(cos<br>~~p~~i(sqrt(sqrt(sin<br>~~p~~i(ARG1))))))|
|abs|~~l~~og(cos<br>~~p~~i(abs<br>~~l~~og(ARG5))) (4)||cos<br>~~p~~i(abs<br>log(negative(sqrt(cos<br>~~p~~i(negative(ARG5))))))|
|abs|~~l~~og(square(abs<br>~~l~~og(ARG5))) (4)|abs|~~l~~og(square(abs<br>~~l~~og(negative(abs<br>~~l~~og(aq(ARG7, ARG5))))))|
||sqrt(sqrt(ARG2)) (4)||sqrt(negative(absolute(sqrt(sqrt(square(ARG2))))))|



## N.3 LARGE LANGUAGE MODELS FOR SYMBOLIC REGRESSION 

Recently, large language models (LLMs) have been increasingly applied to symbolic regression tasks. These approaches often involve leveraging historical programs (Shojaee et al., 2024b) or extracting patterns from historical programs (Grayeli et al., 2024) to guide LLM-based symbolic regression through prompt engineering. By instructing LLMs to learn from high-quality historical programs (Shojaee et al., 2024b) while avoiding low-quality ones (Grayeli et al., 2024), these methods aim to leverage LLMs to implement effective SR systems. However, using the MSE score as the criterion for crafting prompts can be misleading. Some expressions with higher MSE scores may still serve as valuable building blocks. From the perspective of the semantic space, the most useful expression for a candidate solution is not necessarily the one with the lowest MSE but the one that effectively fills the residual gap. This insight highlights the need for a fine-grained retrieval augmentation strategy that focuses on the semantics of each promising solution when crafting prompts. Thus, this paper builds a library of GP trees on semantics and retrieves from this library to guide the language model in generating useful expressions for each promising solution. 

## O OUT-OF-LIMIT GENERATION 

The neural retrieval library is trained on symbolic expressions with heights ranging between [0 _,_ 5]. However, in an SR task, the ground truth expression may exceed the generation limit of the neural model. In this section, we evaluate the performance of the neural retrieval library when the ground truth surpasses this limit. Specifically, we generate GP trees with a height of 6 and compare the outputs of RAG-NN with the ground truth, as shown in Table 3. The results indicate that even when the ground truth exceeds the neural generation limit, the neural model can still generate meaningful sub-expressions that assist GP in discovering better GP trees. 

29 

