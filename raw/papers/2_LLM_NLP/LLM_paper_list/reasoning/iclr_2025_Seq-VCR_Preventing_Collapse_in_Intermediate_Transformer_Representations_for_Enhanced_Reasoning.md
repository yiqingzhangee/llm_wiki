Preprint. Under review. 

# SEQ-VCR:PREVENTING COLLAPSE IN INTERMEDIATE TRANSFORMER REPRESENTATIONS FOR ENHANCED REASONING 

**Md Rifat Arefin**[1] _[,]_[2] _[,]_[3] _[∗]_ **Gopeshh Subbaraj**[1] _[,]_[2] **, Nicolas Gontier**[3] **, Yann LeCun**[5] _[,]_[6] **, Irina Rish**[1] _[,]_[2] **, Ravid Shwartz-Ziv**[6] **, Christopher Pal**[3] _[,]_[4] 

1Universit´e de Montr´eal, 2Mila, 3ServiceNow, 4Polytechnique Montreal, 5Meta FAIR, 6New York University 

## ABSTRACT 

Decoder-only Transformers often struggle with complex reasoning tasks, particularly arithmetic reasoning requiring multiple sequential operations. In this work, we identify _representation collapse_ in the model’s intermediate layers as a key factor limiting their reasoning capabilities. To address this, we propose **Sequential Variance-Covariance Regularization (Seq-VCR**[1] **)** , which enhances the entropy of intermediate representations and prevents collapse. Combined with dummy pause tokens as substitutes for chain-of-thought (CoT) tokens, our method significantly improves performance in arithmetic reasoning problems. In the challenging 5 _×_ 5 integer multiplication task, our approach achieves 99 _._ 5% exact match accuracy, outperforming models of the same size (which yield 0% accuracy) and GPT-4 with five-shot CoT prompting (44%). We also demonstrate superior results on arithmetic expression and longest increasing subsequence (LIS) datasets. Our findings highlight the importance of preventing intermediate layer representation collapse to enhance the reasoning capabilities of Transformers and show that Seq-VCR offers an effective solution without requiring explicit CoT supervision. 

## 1 INTRODUCTION 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0001-08.png)


**----- Start of picture text -----**<br>
6 7 8 9 0<br>1<br>1 2 3 4 5<br>2 6 7 8 9 0<br>x x x x x x<br>3 x x x x x x<br>x x x x x x<br>4 x x x x x x<br>x x x x x x<br>5 # # # # # # # # # #<br>Position 0 1 2 3 4 5 6 7 8 9<br>Mult 1 2 3 4 5 5 4 3 2 1<br>Sum 0 1 2 3 4 4 3 2 1 0<br>Carry 1 2 3 4 5 4 3 2 1 0<br>Total 2 5 8 11 14 13 10 7 4 1<br>**----- End of picture text -----**<br>


Figure 1: Position-wise number of operations needed for 5x5 digits integer multiplication task. Middle tokens in the output sequence need more operations than the peripheral ones, making their prediction much harder (as shown in Figure 6). Example of 12345 x 67890 is shown here. 

Large Language Models (LLMs) based on Transformer architectures have achieved remarkable success across a wide range of tasks, positioning them as foundational models in artificial intelligence (Bommasani et al., 2021). Despite their impressive capabilities, LLMs often struggle with tasks 

> _∗_ rifat.arefin@mila.quebec, Work was completed during internship at ServiceNow. 1 https://github.com/rarefin/seq ~~v~~ cr 

1 

Preprint. Under review. 

requiring complex reasoning, particularly arithmetic reasoning that necessitates multiple sequential operations (Bubeck et al., 2023). These challenges are attributed to the models’ limitations in handling tasks that involve deep cognitive abilities and multi-step reasoning processes. 

One of the key obstacles is the _representation collapse_ in the intermediate layers of Transformer models. Representation collapse occurs when internal representation diversity diminishes, leading to less informative features and hindering the model’s ability to solve complex tasks (Jing et al., 2021). For arithmetic reasoning, transformers struggle with successive carryovers and storing intermediate results Qiu et al. (2024), which are essential for solving complex sub-tasks, requiring more computations (Figure 1). We hypothesize that representation collapse prevents the model from effectively performing sub-tasks by calculating successive carryovers and storing intermediate results, which are essential for accurate prediction. 

To address this limitation, we introduce **Sequential Variance-Covariance Regularization (SeqVCR)** , a regularization technique designed to enhance the entropy of intermediate representations and prevent representation collapse. By increasing the diversity of representations within the model’s layers, Seq-VCR enables the Transformer to maintain richer and more informative features throughout the computation process. 

Furthermore, we incorporate dummy pause tokens as substitutes for chain-of-thought (CoT) tokens. While CoT prompting has been shown to improve reasoning by breaking down tasks into intermediate steps (Wei et al., 2022), it often requires explicit supervision and can be computationally expensive. Our approach leverages pause tokens to simulate the effect of CoT without the need for explicit intermediate reasoning steps. 

We validate our method on challenging arithmetic reasoning tasks. Notably, on the 5 _×_ 5 integer multiplication task, our approach achieves 99 _._ 5% exact match accuracy, surpassing models of the same size (which yield 0% accuracy) and GPT-4 with five-shot CoT prompting (44%). We also demonstrate significant improvements on arithmetic expression and longest increasing subsequence (LIS) datasets. 

Our contributions are summarized as follows: 

- We identify representation collapse in intermediate layers as a key limitation affecting the reasoning capabilities of Transformer models. 

- We propose Seq-VCR, a regularization technique that enhances the diversity of intermediate representations and prevents collapse. 

- We demonstrate that combining Seq-VCR with pause tokens enables models to solve complex arithmetic reasoning tasks without explicit CoT supervision. 

- We provide extensive experimental results showing significant improvements over baseline models and state-of-the-art LLMs like GPT-4. 

## 2 LITERATURE REVIEW 

**Transformer Reasoning** Research in large language models (LLMs) has advanced significantly. Models like BERT (Devlin et al., 2019) and GPT (Radford & Narasimhan, 2018) demonstrated improvements in natural language understanding. However, challenges remain in tasks that require deep cognitive abilities. Integrating external knowledge has shown promise in improving reasoning capabilities (Bosselut et al., 2019). Recent advances include techniques such as chain-of-thought (CoT) prompting, which allow models such as GPT-3 (Brown et al., 2020) to perform complex reasoning by breaking tasks into intermediate steps (Wei et al., 2022) with human supervision, called process supervision (Lightman et al., 2023). Wang et al. (2022) shows that even incorrect but coherent intermediate steps can improve reasoning performance. Jin et al. (2024) found increasing the length of the reasoning steps in the prompts, even without adding new information to the prompt, improves the reasoning abilities of LLMs. 

Wang et al. (2024), investigated the addition of dummy tokens such as periods or hash symbols to inputs at inference time. They found that this simple modification could improve performance in arithmetic reasoning tasks. Some works show that these dummy tokens, commonly referred to as filler tokens, do not extend transformers’ abilities beyond TC[0] circuit complexity, but still 

2 

Preprint. Under review. 

significantly enhance problem-solving within this class (Merrill & Sabharwal, 2023; Strobl et al., 2023). Building on this idea, Goyal et al. (2023a) introduced a more comprehensive framework called ”pause-training.” Their approach involves incorporating learnable _< pause >_ tokens during both pre-training and fine-tuning of language models. While there are differences between the pause tokens and filler tokens approach, they share the goal of providing additional computation time for transformers which extends the expressive power of transformers within the TC[0] class without changing the fundamental limitations of the model architecture. Chain-of-thought prompting can potentially elevate transformers beyond the TC[0] complexity class; However, it may not be necessary for all problems. Furthermore, filler tokens have been demonstrated to enhance the expressivity of transformers (Pfau et al., 2024), even in cases where traditional transformers, lacking chain-ofthought mechanisms, exhibit insufficient expressive power (Sanford et al., 2024). 

**Representation Collapse** Representation collapse in representation learning degrades model performance by making features indistinguishable. It often occurs in unsupervised and self-supervised learning tasks due to the lack of labeled data, leading to trivial solutions and a low-rank feature space. Research by Jing et al. (2021) highlights that representation collapse is due to poor optimization, loss function design, and improper regularization. Recent studies (Grill et al., 2020) suggest the use of contrastive learning, such as SimCLR, to mitigate this by contrasting positive and negative samples. Additionally, Zbontar et al. (2021) introduced Barlow Twins to minimize redundancy between learned representations. A further advancement is VICReg (Bardes et al., 2021; Zhu et al., 2023) which employs variance, invariance, and covariance regularization to prevent collapse, ensuring diverse and non-trivial representations. While these researches focuse on images, it is less explored in language models. Recently, Barbero et al. (2024) studied last-layer representations of the next tokens and found representation collapse in pre-trained models. 

## 3 METHODOLOGY 

## 3.1 PRELIMINARIES 

Consider a finite set of token vocabulary denoted by _V_ = (1 _. . . v_ ), while _x, y ∈ V_ constitutes a sequence of these tokens representing the input **x** = [ _x_ 1 _, . . . , xK_ ] and the output **y** = [ _y_ 1 _, . . . , yT_ ], where _K_ and _T_ indicate the respective sequence lengths of the input and output. 

We covert the input **x** and output sequence **y** in an embedded sequence: 

_Eemb_ = [˜ _x_ 1 _, . . . ,_ ˜ _xK,_ ˜ _y_ 1 _, . . . ,_ ˜ _yT_ ] _∈_ R _[T]_[ +] _[K][×][d]_ and _Epos_ = [ _p_ 1 _, p_ 2 _, . . ._ ] _∈_ R _[T]_[ +] _[K][×][d]_ of dimension _d_ . We can get the language model layer-wise as: _f_ = ( _fcls ◦ fL ◦ fL−_ 1 _. . . ◦ f_ 0) 

where _f_ 0 = [˜ _x_ 1 + _p_ 1 _, . . . ,_ ˜ _xK_ + _pK,_ ˜ _y_ 1 + _pK_ +1 _, . . . ,_ ˜ _yT_ + _pK_ + _T_ ] and _fcls ∈_ R _[|][V][ |×][d]_ is the final linear classification layer. _fl_ is the self-attention layer with MLP for layer _l_ (Vaswani et al., 2017). 

## 3.2 NEXT-TOKEN PREDICTION LOSS 

Large language models like _GPT_ (Radford et al., 2018) (Generative Pre-trained Transformer) are trained to auto-regressively predict the next token in a sequence. This minimizes the difference between predicted and actual tokens for coherent and contextually appropriate text. A loss function like cross-entropy for next-token prediction is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0003-11.png)


where _yt_ is the one-hot encoded true token at time step _t_ and input tokens _x_ , modeled by a language model, _f_ . 

## 3.3 REPRESENTATION COLLAPSE IN TRANSFORMERS 

Representation collapse in Transformers manifests as a reduction in the diversity of internal representations across layers, particularly in tasks requiring complex reasoning. This collapse hinders the 

3 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Vanilla Pause Seq-VCR Seq-VCR+Pause Pretrained<br>0.8<br>0.5<br>0.7 0.4<br>0.6 0.3<br>0.5 0.2<br>0.1<br>0.4<br>0.0<br>0 1 2 3 4 5 0 2 4 6 8 10 12<br>Layer Layer<br>(a) Training minGPT from scratch on  Arithmetic Ex- (b) Fine-tuning GPT-2 Small on 5  ×  5  digit Mul-<br>pression tiplication<br>Average Entropy Average Entropy<br>**----- End of picture text -----**<br>


Figure 2: **Representation collapse** across layers during (a) training (or fine-tuning(b)) for two datasets. The x-axis represents the layer ID, while the y-axis shows the degree of collapse as measured by representation Matrix-Entropy. The results highlight how intermediate layers (shown by the decline in Entropy) experience representation collapse during **Vanilla** training or fine-tuning for both datasets, indicating potential bottlenecks in information flow or feature learning. Tools like **Pause** Goyal et al. (2023b) token-based tuning can’t fix it, but our proposed regularization **SeqVCR** can improve collapse. 

model’s ability to capture and process intricate patterns necessary for tasks like multi-digit multiplication. 

To illustrate this, we analyze the layer-wise representation entropy of a GPT-2 model fine-tuned on the 5 _×_ 5 digit multiplication task. As shown in Figure 2(b), there is a significant drop in entropy in the intermediate layers, indicating a collapse in representation diversity. We hypothesize that this representation collapse is the reason why it is very hard for the model to calculate multiplication, carry, and store them, thus resulting in poorer performance, as shown in Figure 6 for the output positions, which require more computation (Figure 1). 

For computing the **representation collapse** of a token sequence, we utilise the matrix entropy of the representation vectors in different layers of the transformer model. We investigate _α_ -order matrixbased entropy (Giraldo et al., 2014; Skean et al., 2023) as a practical alternative to R´enyi’s _α_ -order entropy (R´enyi, 1961). This metric uses a similarity kernel _κ_ on layer-wise representations of token sequences from a distribution, without assuming the true nature of the distribution. The kernel selection _κ_ is flexible, including Gaussian, linear, or Laplacian kernels. We choose the linear kernel _κ_ ( _a, b_ ) = _ab[T]_ due to the linear representation hypothesis (Park et al., 2024), suggesting that language model representations encode high-level concepts in linearly separable manifolds. 

Aussuming _Z_[(] _[l]_[)] = _fl ∈_ R _[T][ ×][d]_ is the _l[th]_ representations of the _T_ tokens in the sequence of dimension _d_ , we can first construct the Gram matrix **K** = _κ_ � _Z_[(] _[l]_[)] _, Z_[(] _[l]_[)][�] = _Z_[(] _[l]_[)] _Z_[(] _[l]_[)] _[T] ∈_ R[T] _[×]_[T] consisting of all pairwise evaluations of the _T_ token representations in _Z_[(] _[l]_[)] . Then the matrix-based entropy of order _α >_ 0 is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0004-07.png)


Equation 2 is the _α_ -order R´enyi entropy of the Gram matrix eigenvalues[2] . Each eigenvalue is divided by tr( **K** ) and raised to the _α_ power so that the eigenvalues of **K** sum to 1, as tr( _·_ ) =[�] _[n] i_ =1 _[λ][i]_[(] _[·]_[)][), al-] lowing them to be treated as a probability distribution. Each eigenvalue of **K** represents the variance 

> 2The non-zero eigenvalues of the Gram matrix _Z_ ( _l_ ) _Z_ ( _l_ ) _[T]_ are equivalent to those of the covariance matrix _Z_[(] _[l]_[)] _[T] Z_[(] _[l]_[)] . Using the covariance matrix instead of the Gram matrix in Equation 2 makes no difference and is more computationally efficient if _d < T_ . 

4 

Preprint. Under review. 

of the sequence token representations in a specific direction of the principal component Scholkopf & Smola (2018). Low entropy indicates a heavy-tailed distribution in which few components dominate the variance in _Z_[(] _[l]_[)] . At maximum entropy, the eigenvalues form a uniform distribution, spreading token representations equally in all directions. Depending on _α_ , various special cases of matrixbased entropy can be recovered. Specifically, when lim _α→_ 1 it equals Shannon entropy, also known as Von Neumann entropy in quantum information theory Bach (2022); Boes et al. (2019)). 

## 3.4 SEQUENTIAL VARIANCE COVARIANCE REGULARIZATION (SEQ-VCR) 

To prevent representation collapse, we propose Seq-VCR, a regularization technique that encourages the model to maintain high variance and low covariance in its intermediate representations. SeqVCR is inspired by techniques used in self-supervised learning for image representations (Bardes et al., 2021) and is adapted here for sequential data in Transformer models. 

Seq-VCR is applied to the final output of the model ( _X_ = _fcls_ ). Since it requires a calculation of the covariance matrix and a pre-trained model such as GPT-2, the dimension can be _fcls_ R[5000] _[×][d]_ , such dimensional calculation is often difficult. So we use a linear projection layer of the representation layer _fproj_ which projects the representation of the layer _l_ into a smaller embedding space such as _X_ = _fproj_ ( _fl_ ) _R[T][ ×][d]_ , _fproj ∈_ R[2048] _[×][d]_ . 

Now we can assume that we have _N_ samples in the batch, so _N_ number of _X_ . We can calculate the covariance matrix across the batch dimension, _C ∈ R[T][ ×][d][×][d]_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0005-06.png)


where _λ_ 1 and _λ_ 2 are the coefficients of the variance and covariance regularization terms. _η_ is a small constant, used for numerical stability. 

The Variance Term encourages unit variance in each dimension, while the Covariance Term penalizes covariance between different dimensions, promoting decorrelation and diversity in representations. 

## 3.5 INCORPORATING PAUSE TOKENS 

We further enhance the model’s reasoning capacity by introducing dummy pause tokens, which act as placeholders for intermediate computation steps. Unlike explicit CoT tokens, pause tokens do not require supervision and serve to allocate additional computational resources within the model (Goyal et al., 2023b). 

## 3.6 TRAINING OBJECTIVE 

The overall training objective combines the standard next-token prediction loss with the Seq-VCR regularization. Our final loss **L** is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0005-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0005-14.png)


This encourages the model to not only predict the next token accurately but also maintain diverse and informative intermediate representations. 

## 4 EXPERIMENTS 

## 4.1 DATA 

We conduct experiments on three tasks: we first consider the **multi-digit multiplication** task from the BIG-bench benchmark (Srivastava et al., 2022), which is the most challenging among arithmetic 

5 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0006-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0006-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0006-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0006-04.png)


**----- Start of picture text -----**<br>
(a) Arithmetic Expression (b) LIS (c) 5x5 Multiplication<br>**----- End of picture text -----**<br>


Figure 3: Illustrations of Input, Output and CoT on the Arithmetic, LIS and mutliplication datasets 

tasks (Yang et al., 2023). In particular, we use the four-digit (4 × 4) and five-digit (5 × 5) multiplication problems, since these two tasks prove very challenging to solve under no CoT, utilizing the training data generated by Deng et al. (2023). 

Next, we focus on the **Arithmetic Expressions** Feng et al. (2024) dataset. This task focuses on evaluating arithmetic expressions. The input sequence for this task is a sequence consisting of numbers, mathematical operators such as addition (+), subtraction (-), multiplication ( _×_ ), division ( _÷_ ) and brackets, followed by the equal sign. The goal of this task is to calculate the input arithmetic expression and generate the correct result. This task is naturally well-suited for a Chain-of-Thought (CoT) method, where each step does part of the calculation, slowly solving one operation at a time while keeping the other parts the same. 

Finally, we also conduct experiments in a more general setting outside arithmetic tasks, called Dynamic Programming (DP). Dynamic Programming is a framework for solving decision-making problems. The core idea in this setting is to break down a complex task into a series of smaller subtasks that can be solved sequentially. We chose a very popular problem in this setting called the **Longest Increasing Sub-sequence (LIS)** as described in the Introduction to Algorithms book (Cormen et al., 2022). For this task, the goal is to find the length of the longest increasing subsequence of a given integer sequence. We generate datasets with different input sequence lengths ranging from _{_ 50, 80, 100 _}_ as shown in Feng et al. (2024). Moreover, all input sequences, and answers in LIS are bounded-range integers and can therefore be tokenized (similar to the multi-digit multiplication task). The CoT tokens for this task consist of the dynamic programming array plus the final answer. Figure 3 shows a working example for a sample problem in all the datasets considered. 

Moreover, Table 3 in the Appendix provides an overview of different tasks across the datasets i.e., Multiplication, Arithmetic Expression and LIS, along with their respective input, chain-of-thought (CoT), and output token details. 

## 4.2 MODELS AND TRAINING CONFIGURATIONS 

We conduct experiments using two models: 

**GPT-2 Small Fine-Tuning** We fine-tune a pre-trained GPT-2 Small model on the multi-digit multiplication tasks. Fine-tuning is performed for 40 epochs with a learning rate of 5 _×_ 10 _[−]_[4] and a batch size of 32. 

**minGPT Training from Scratch** We train a minGPT model from scratch on the Arithmetic Expressions and LIS datasets. Training is conducted for 100 epochs with a learning rate of 1 _×_ 10 _[−]_[4] and a batch size of 128. 

We compare five configurations: 

- **Vanilla** : Standard training without CoT or pause tokens. 

- **With CoT** : Training with explicit CoT supervision. 

- **Pause** : Inserting pause tokens in the input sequence. 

- **Seq-VCR** : Applying Seq-VCR regularisation. 

- **Seq-VCR + Pause** : Combining Seq-VCR with pause tokens. 

6 

Preprint. Under review. 

## 4.3 RESULTS 

## 4.3.1 REPRESENTATION DIVERSITY 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0007-03.png)


**----- Start of picture text -----**<br>
Vanilla Pause Seq-VCR Seq-VCR+Pause<br>layer0 layer1 layer2<br>100 100 100<br>50<br>0 0 0<br>0.50 0.52 0.54 0.56 0.58 0.60 0.62 0.64 0.2 0.3 0.4 0.5 0.6 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65<br>layer3 layer4 layer5<br>100 100 100<br>50 50 50<br>0 0 0<br>0.1 0.2 0.3 0.4 0.5 0.6 0.1 0.2 0.3 0.4 0.5 0.6 0.1 0.2 0.3 0.4 0.5 0.6<br>layer6 layer7 layer8<br>100 100 100<br>50 50 50<br>0 0 0<br>0.1 0.2 0.3 0.4 0.5 0.6 0.1 0.2 0.3 0.4 0.5 0.6 0.2 0.3 0.4 0.5 0.6<br>layer9 layer10 layer11<br>100 100 100<br>50 50 50<br>0 0 0<br>0.2 0.3 0.4 0.5 0.6 0.2 0.3 0.4 0.5 0.6 0.2 0.3 0.4 0.5 0.6 0.7<br>layer12<br>100<br>50<br>0<br>0.3 0.4 0.5 0.6 0.7<br>Entropy<br>Frequency<br>**----- End of picture text -----**<br>


Figure 4: Layer-wise entropy distributions for different configurations on the 5 _×_ 5 multiplication task. _Seq-VCR_ and _Seq-VCR+Pause_ maintain higher entropy across layers, indicating greater representation diversity. 

Seq-VCR significantly enhances representation diversity, as evidenced by the layer-wise distribution of representation entropy across different configurations. In Figure 4, we observe that configurations employing our regularization technique, particularly _Seq-VCR_ and _Seq-VCR+Pause_ , exhibit higher entropy values in intermediate layers compared to the other configurations ( _Vanilla_ , and _Pause_ ). This increased diversity in representation entropy suggests that the model is able to capture a wider range of features and maintain distinct representations. The histograms also reveal that while other configurations show peaks at lower entropy values, indicating potential representation collapse and redundancy in the intermediate representations, our method fosters a more uniform distribution across the layers. This uniformity suggests a richer feature space being utilized by the model, enabling it to generalize better to unseen data, thus improving performance on downstream tasks. 

## 4.4 LEARNING DYNAMICS 

Our regularization method, _Seq-VCR_ or _Seq-VCR+Pause_ cause a significant phase transition (improving the next token prediction loss) in the performance of the model across datasets compared to other configurations such as _Vanilla_ , and _Pause_ , which do not induce this phase transition as shown in Figure 5. Specifically, this phase transition marks a distinct shift in the model’s ability to capture and generalize representations during training or finetuning. 

Without regularization, learning is difficult and training loss saturates. Seq-VCR improves the learning curve. Adding Seq-VCR with 2 Pause tokens causes a sharp phase transition (Figure 5 (a)) and solves the 5 _×_ 5 digit multiplication task. Figure 5 (b) shows similar improvements in arithmetic reasoning task that is learned from scratch. 

## 4.5 RESULTS ON 4 _×_ 4 AND 5 _×_ 5 DIGITS MULTIPLICATION TASKS 

Table 1 shows the results on 4 _×_ 4 and 5 _×_ 5 digits using a fine-tuned GPT-2 Small. Compared to no CoT configurations like ( _Vanilla_ , _Pause_ ), our method enables solving tasks previously not solvable without explicit CoT. We can see the effectiveness of the _Seq-VCR_ on 4 _×_ 4 digit multiplication task where without pause token it improves performance from 25% to 52% and with 2 pause tokens _SeqVCR + Pause_ achieves accuracy 99 _._ 2%, which is close to _With CoT_ performance and significantly better than other configuration without CoT. It even outperforms the 5-shot prompt(with or without CoT) performance of GPT-3.5 and GPT-4.0. 

7 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0008-01.png)


**----- Start of picture text -----**<br>
Seq-VCR+Pause Vanilla Pause Seq-VCR<br>1.2 3.0<br>0.9 2.4<br>1.8<br>0.6<br>1.2<br>0.3<br>0.6<br>0.0<br>0 10 20 30 40 0 20 40 60 80 100<br>Epochs Epochs<br>Next Token Prediction Loss Next Token Prediction Loss<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0008-02.png)


**----- Start of picture text -----**<br>
(a) Fine-tuning GPT-2 Small on 5  ×  5 digit Multipli- (b) Training minGPT from scratch on Arithmetic<br>cation Expression<br>**----- End of picture text -----**<br>


Figure 5: Learning dynamics (curves for next token prediction loss) illustrating the phase transition observed across datasets when applying our regularization methods. The x-axis represents training epochs, and the y-axis denotes the model’s loss. The phase transition is characterized by a sharp reduction in loss, marking a distinct shift in the learning regime when using _Seq-VCR_ and _Seq-VCR + Pause_ , compared to the gradual decline or saturating curves in other configurations. 

|Model|Confguration|4x4 Mult|5x5 Mult|
|---|---|---|---|
|GPT-3.5|With CoT<br>No CoT|0.43<br>0.02|0.05<br>0.00|
|GPT-4|With CoT<br>No CoT|0.77<br>0.04|0.44<br>0.00|
|GPT-2 Small|With CoT<br>Vanilla<br>Pause<br>Seq-VCR<br>Seq-VCR + Pause|1.0<br>0.25<br>0.28<br>0.52<br>0.992|1.0<br>0.0<br>0.0<br>0.0<br>0.995|



Table 1: Accuracy (exact match) on 4 _×_ 4 and 5 _×_ 5 digits Multiplication Tasks. GPT-3.5 and GPT-4 results are taken from Deng et al. (2024) which are produced by 5-shot prompt 

We also see from Table 1 that GPT-4 with CoT 5- shot prompts achieves 44 _._ 0% for 5 _×_ 5 digit multiplication, while _Vanilla_ , _Seq-VCR_ , and _Pause_ achieve 0% without CoT. Because, all configurations struggle with predicting middle tokens in the output sequence (shown in Figure 6). This happens because those middle tokens require more computations to calculate successive carry and store them, since these tokens require more computation (as shown in Figure 1), thus making them difficult sub-task than predicting the peripheral tokens which requires less computes, thus easier to solve. However, _Seq-VCR+Pause_ with 2 Pause tokens solves the 5 _×_ 5 digit multiplication task, previously unsolved without CoT tokens. Figure 2(b) shows our regularization _Seq-VCR_ increases intermediate layer entropy, allowing more exploration in the representation space. We hypothesize that along with better representation by _Seq-VCR_ , 2 Pause tokens increase the computational ability, allowing the model to carry the necessary multiplication values and solve the task. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0008-07.png)


**----- Start of picture text -----**<br>
Vanilla Pause Seq-VCR Seq-VCR+Pause<br>100<br>80<br>60<br>40<br>20<br>0 1 2 3 4 5 6 7 8 9<br>Index<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 6: Position-wise accuracy of Multiplication of 5x5 digits with different configuration of GPT-2 Small. We see models fail on the middle tokens they require more compute. 

8 

Preprint. Under review. 

- 4.6 RESULTS ON ARITHMETIC EXPRESSION AND LONGEST INCREASING SUB-SEQUENCE (LIS) DATASETS 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0009-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0009-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0009-04.png)


**----- Start of picture text -----**<br>
(a) Test accuracy on Arithmetic Expressions Dataset (b) Test accuracy on LIS Dataset<br>**----- End of picture text -----**<br>


Figure 7: Performance of each method across tasks of varying difficulty 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0009-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0009-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0009-08.png)


**----- Start of picture text -----**<br>
(a) Test accuracy on 6 operator Arithmetic Ex- (b) Test accuracy on LIS Dataset with 100 Input<br>pression Sequence Length<br>**----- End of picture text -----**<br>


Figure 8: Performance of each method on the complex task w/ varying number of layers 

Figure 7 illustrates how the accuracy varies across different methods (represented by different bar colors in the plot) and with changing levels of task complexity, such as the number of operators in case of the Arithmetic Expressions Dataset or input sequence length in case of the LIS Dataset. 

We can observe that as the task complexity is low i.e in case of 4 operators in the arithmetic expression and input sequence length of 50 in the LIS dataset, all the methods perform quite well. But, as the number of operators or sequence length increases, the tasks become more complex which shows the need for models other than the vanilla baselines. 

Seq-VCR demonstrates substantial improvement over vanilla models and achieved results comparable to chain-of-thought (CoT) prompting method as shown in Feng et al. (2024). While CoT outperforms the other methods, there is an increase in the serial computation due to its step-by-step reasoning approach. On the other hand, Pause Tokens and regularization primarily boost parallel computation, enhancing the model’s expressivity for these types of problems without significantly increasing serial computation. The results presented in Figure 7 are based on models with a 5-layer configuration, optimized using the best-performing number of pause tokens. We find that the number of pause-tokens that works best for a particular model and layer config vary a bit. For example, the best working Seq-VCR + Pause model uses 7 pause tokens in case of the 5 operator task in Arithmetic Expression while 4 pause tokens works the best for the LIS task with 80 as input sequence length. Through Figure 8 we show how the different models perform by itself in the most complex 

9 

Preprint. Under review. 

task in each dataset. We also clearly see that more depth of the model helps every method in solving the task better. Another conclusion we can draw from figures 7 8 is that even though the combination of Seq-VCR with Pause Tokens works the best (almost close to CoT method), Seq-VCR gets similar test accuracy without the use of Pause Tokens in both datasets. All results in figure 78 are over 3 seed runs. Detailed ablation studies, exploring variations in the number of pause tokens and layers, can be found in the appendix. 

## 5 CONCLUSION 

Despite the remarkable success of large transformer based language models, they exhibit deficiencies in multi-step reasoning tasks. One proposed remedy is the application of Chain-of-Thought supervision, which, although effective, entails considerable cost. Conversely, there has been limited exploration into enhancing the capacity of the pre-trained model representation itself without explicit supervision. Our analysis indicates that intermediate layer representation collapse detrimentally affects the computation of intermediate information essential for solving arithmetic reasoning tasks. Our proposed regularization technique enhances the model’s representation capacity, thereby improving its ability to solve these tasks. We think that further research is required to enhance the capability of pre-training as well, where improved representation learning will enhance the model with greater capabilities. 

## ACKNOWLEDGMENTS 

We acknowledge the support of the Canada CIFAR AI Chair Program and IVADO. We thank Mila and Compute Canada for providing computational resources. 

## REFERENCES 

- Francis Bach. Information theory with kernel methods. _IEEE Transactions on Information Theory_ , 69(2):752–775, 2022. 

- Federico Barbero, Andrea Banino, Steven Kapturowski, Dharshan Kumaran, Jo˜ao GM Ara´ujo, Alex Vitvitskyi, Razvan Pascanu, and Petar Veliˇckovi´c. Transformers need glasses! information oversquashing in language tasks. _arXiv preprint arXiv:2406.04267_ , 2024. 

- Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. _arXiv preprint arXiv:2105.04906_ , 2021. 

- Paul Boes, Jens Eisert, Rodrigo Gallego, Markus P M¨uller, and Henrik Wilming. Von neumann entropy from unitarity. _Physical review letters_ , 122(21):210402, 2019. 

- Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. _arXiv preprint arXiv:2108.07258_ , 2021. 

- Antoine Bosselut, Hannah Rashkin, Maarten Sap, Chaitanya Malaviya, Asli Celikyilmaz, and Yejin Choi. Comet: Commonsense transformers for automatic knowledge graph construction. In _ACL_ , 2019. 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. _ArXiv_ , abs/2005.14165, 2020. URL https://api.semanticscholar.org/ CorpusID:218971783. 

- S´ebastien Bubeck, Varun Chandrasekaran, Ronen Eldan, John A. Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuan-Fang Li, Scott M. Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, and Yi Zhang. Sparks of artificial general intelligence: Early experiments 

10 

Preprint. Under review. 

with gpt-4. _ArXiv_ , abs/2303.12712, 2023. URL https://api.semanticscholar.org/ CorpusID:257663729. 

Thomas H Cormen, Charles E Leiserson, Ronald L Rivest, and Clifford Stein. _Introduction to algorithms_ . MIT press, 2022. 

Yuntian Deng, Kiran Prasad, Roland Fernandez, Paul Smolensky, Vishrav Chaudhary, and Stuart Shieber. Implicit chain of thought reasoning via knowledge distillation. _arXiv preprint arXiv:2311.01460_ , 2023. 

Yuntian Deng, Yejin Choi, and Stuart Shieber. From explicit cot to implicit cot: Learning to internalize cot step by step. _arXiv preprint arXiv:2405.14838_ , 2024. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In _North American Chapter of the Association for Computational Linguistics_ , 2019. URL https://api.semanticscholar.org/ CorpusID:52967399. 

- Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing the mystery behind chain of thought: a theoretical perspective. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Luis Gonzalo Sanchez Giraldo, Murali Rao, and Jose C Principe. Measures of entropy from data using infinitely divisible kernels. _IEEE Transactions on Information Theory_ , 61(1):535–548, 2014. 

- Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. Think before you speak: Training language models with pause tokens. _ArXiv_ , abs/2310.02226, 2023a. URL https://api.semanticscholar.org/CorpusID: 263608983. 

- Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. Think before you speak: Training language models with pause tokens. _arXiv preprint arXiv:2310.02226_ , 2023b. 

- Jean-Bastien Grill, Florian Strub, Florent Altch´e, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. _Advances in neural information processing systems_ , 33:21271–21284, 2020. 

- Mingyu Jin, Qinkai Yu, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, Mengnan Du, et al. The impact of reasoning step length on large language models. _arXiv preprint arXiv:2401.04925_ , 2024. 

- Li Jing, Pascal Vincent, Yann LeCun, and Yuandong Tian. Understanding dimensional collapse in contrastive self-supervised learning. _arXiv preprint arXiv:2110.09348_ , 2021. 

- Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. _arXiv preprint arXiv:2305.20050_ , 2023. 

- William Merrill and Ashish Sabharwal. The parallelism tradeoff: Limitations of log-precision transformers. _Transactions of the Association for Computational Linguistics_ , 11:531–545, 2023. 

- Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. In _Forty-first International Conference on Machine Learning_ , 2024. 

- Jacob Pfau, William Merrill, and Samuel R Bowman. Let’s think dot by dot: Hidden computation in transformer language models. _arXiv preprint arXiv:2404.15758_ , 2024. 

- Luyu Qiu, Jianing Li, Chi Su, Chen Jason Zhang, and Lei Chen. Dissecting multiplication in transformers: Insights into llms. _arXiv preprint arXiv:2407.15360_ , 2024. 

11 

Preprint. Under review. 

- Alec Radford and Karthik Narasimhan. Improving language understanding by generative pretraining. 2018. URL https://api.semanticscholar.org/CorpusID:49313245. 

- Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018. 

- Alfr´ed R´enyi. On measures of entropy and information. In _Proceedings of the fourth Berkeley symposium on mathematical statistics and probability, volume 1: contributions to the theory of statistics_ , volume 4, pp. 547–562. University of California Press, 1961. 

- Clayton Sanford, Daniel J Hsu, and Matus Telgarsky. Representational strengths and limitations of transformers. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Bernhard Scholkopf and Alexander J Smola. _Learning with kernels: support vector machines, regularization, optimization, and beyond_ . MIT press, 2018. 

- Oscar Skean, Jhoan Keider Hoyos Osorio, Austin J Brockmeier, and Luis Gonzalo Sanchez Giraldo. Dime: Maximizing mutual information by a difference of matrix-based entropies. _arXiv preprint arXiv:2301.08164_ , 2023. 

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adri`a Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat Agarwal, Alethea Power, Alex Ray, Alex Warstadt, Alexander W. Kocurek, Ali Safaya, Ali Tazarv, Alice Xiang, Alicia Parrish, Allen Nie, Aman Hussain, Amanda Askell, Amanda Dsouza, Ambrose Slone, Ameet Annasaheb Rahane, Anantharaman S. Iyer, Anders Andreassen, Andrea Madotto, Andrea Santilli, Andreas Stuhlmuller, Andrew M. Dai, Andrew La, Andrew Kyle Lampinen, Andy Zou, Angela Jiang, Angelica Chen, Anh Vuong, Animesh Gupta, Anna Gottardi, Antonio Norelli, Anu Venkatesh, Arash Gholamidavoodi, Arfa Tabassum, Arul Menezes, Arun Kirubarajan, Asher Mullokandov, Ashish Sabharwal, Austin Herrick, Avia Efrat, Aykut Erdem, Ayla Karakacs, B. Ryan Roberts, Bao Sheng Loe, Barret Zoph, Bartlomiej Bojanowski, Batuhan Ozyurt, Behnam Hedayatnia, Behnam Neyshabur, Benjamin Inden, Benno Stein, Berk Ekmekci, Bill Yuchen Lin, Blake Stephen Howald, Bryan Orinion, Cameron Diao, Cameron Dour, Catherine Stinson, Cedrick Argueta, C’esar Ferri Ram’irez, Chandan Singh, Charles Rathkopf, Chenlin Meng, Chitta Baral, Chiyu Wu, Chris Callison-Burch, Chris Waites, Christian Voigt, Christopher D. Manning, Christopher Potts, Cindy Ramirez, Clara Rivera, Clemencia Siro, Colin Raffel, Courtney Ashcraft, Cristina Garbacea, Damien Sileo, Daniel H Garrette, Dan Hendrycks, Dan Kilman, Dan Roth, Daniel Freeman, Daniel Khashabi, Daniel Levy, Daniel Mosegu’i Gonz’alez, Danielle R. Perszyk, Danny Hernandez, Danqi Chen, Daphne Ippolito, Dar Gilboa, David Dohan, David Drakard, David Jurgens, Debajyoti Datta, Deep Ganguli, Denis Emelin, Denis Kleyko, Deniz Yuret, Derek Chen, Derek Tam, Dieuwke Hupkes, Diganta Misra, Dilyar Buzan, Dimitri Coelho Mollo, Diyi Yang, Dong-Ho Lee, Dylan Schrader, Ekaterina Shutova, Ekin Dogus Cubuk, Elad Segal, Eleanor Hagerman, Elizabeth Barnes, Elizabeth P. Donoway, Ellie Pavlick, Emanuele Rodol`a, Emma Lam, Eric Chu, Eric Tang, Erkut Erdem, Ernie Chang, Ethan A. Chi, Ethan Dyer, Ethan J. Jerzak, Ethan Kim, Eunice Engefu Manyasi, Evgenii Zheltonozhskii, Fanyue Xia, Fatemeh Siar, Fernando Mart’inez-Plumed, Francesca Happ’e, Franc¸ois Chollet, Frieda Rong, Gaurav Mishra, Genta Indra Winata, Gerard de Melo, Germ´an Kruszewski, Giambattista Parascandolo, Giorgio Mariani, Gloria Wang, Gonzalo Jaimovitch-L’opez, Gregor Betz, Guy Gur-Ari, Hana Galijasevic, Hannah Kim, Hannah Rashkin, Hannaneh Hajishirzi, Harsh Mehta, Hayden Bogar, Henry Shevlin, Hinrich Schutze, Hiromu Yakura, Hongming Zhang, Hugh Mee Wong, Ian Ng, Isaac Noble, Jaap Jumelet, Jack Geissinger, John Kernion, Jacob Hilton, Jaehoon Lee, Jaime Fern´andez Fisac, James B. Simon, James Koppel, James Zheng, James Zou, Jan Koco’n, Jana Thompson, Janelle Wingfield, Jared Kaplan, Jarema Radom, Jascha Narain Sohl-Dickstein, Jason Phang, Jason Wei, Jason Yosinski, Jekaterina Novikova, Jelle Bosscher, Jennifer Marsh, Jeremy Kim, Jeroen Taal, Jesse Engel, Jesujoba Oluwadara Alabi, Jiacheng Xu, Jiaming Song, Jillian Tang, Jane W Waweru, John Burden, John Miller, John U. Balis, Jonathan Batchelder, Jonathan Berant, Jorg Frohberg, Jos Rozen, Jos´e Hern´andez-Orallo, Joseph Boudeman, Joseph Guerr, Joseph Jones, Joshua Tenenbaum, Joshua S. Rule, Joyce Chua, Kamil Kanclerz, Karen Livescu, Karl Krauth, Karthik Gopalakrishnan, Katerina Ignatyeva, Katja Markert, Kaustubh D. Dhole, Kevin Gimpel, Kevin Omondi, Kory Wallace Mathewson, Kristen Chiafullo, Ksenia Shkaruta, Kumar Shridhar, Kyle McDonell, Kyle Richardson, Laria Reynolds, Leo Gao, Li Zhang, Liam Dugan, Lianhui Qin, 

12 

Preprint. Under review. 

Lidia Contreras-Ochando, Louis-Philippe Morency, Luca Moschella, Luca Lam, Lucy Noble, Ludwig Schmidt, Luheng He, Luis Oliveros Col’on, Luke Metz, Lutfi Kerem cSenel, Maarten Bosma, Maarten Sap, Maartje ter Hoeve, Maheen Farooqi, Manaal Faruqui, Mantas Mazeika, Marco Baturan, Marco Marelli, Marco Maru, Maria Jose Ram’irez Quintana, Marie Tolkiehn, Mario Giulianelli, Martha Lewis, Martin Potthast, Matthew L. Leavitt, Matthias Hagen, M’aty’as Schubert, Medina Baitemirova, Melody Arnaud, Melvin Andrew McElrath, Michael A. Yee, Michael Cohen, Michael Gu, Michael Igorevich Ivanitskiy, Michael Starritt, Michael Strube, Michal Swkedrowski, Michele Bevilacqua, Michihiro Yasunaga, Mihir Kale, Mike Cain, Mimee Xu, Mirac Suzgun, Mitch Walker, Monica Tiwari, Mohit Bansal, Moin Aminnaseri, Mor Geva, Mozhdeh Gheini, T MukundVarma, Nanyun Peng, Nathan A. Chi, Nayeon Lee, Neta Gur-Ari Krakover, Nicholas Cameron, Nicholas Roberts, Nick Doiron, Nicole Martinez, Nikita Nangia, Niklas Deckers, Niklas Muennighoff, Nitish Shirish Keskar, Niveditha Iyer, Noah Constant, Noah Fiedel, Nuan Wen, Oliver Zhang, Omar Agha, Omar Elbaghdadi, Omer Levy, Owain Evans, Pablo Antonio Moreno Casares, Parth Doshi, Pascale Fung, Paul Pu Liang, Paul Vicol, Pegah Alipoormolabashi, Peiyuan Liao, Percy Liang, Peter Chang, Peter Eckersley, Phu Mon Htut, PiBei Hwang, P. Milkowski, Piyush S. Patil, Pouya Pezeshkpour, Priti Oli, Qiaozhu Mei, Qing Lyu, Qinlang Chen, Rabin Banjade, Rachel Etta Rudolph, Raefer Gabriel, Rahel Habacker, Ramon Risco, Raphael Milliere, Rhythm Garg, Richard Barnes, Rif A. Saurous, Riku Arakawa, Robbe Raymaekers, Robert Frank, Rohan Sikand, Roman Novak, Roman Sitelew, Ronan Lebras, Rosanne Liu, Rowan Jacobs, Rui Zhang, Ruslan Salakhutdinov, Ryan Chi, Ryan Lee, Ryan Stovall, Ryan Teehan, Rylan Yang, Sahib Singh, Saif M. Mohammad, Sajant Anand, Sam Dillavou, Sam Shleifer, Sam Wiseman, Samuel Gruetter, Samuel R. Bowman, Samuel S. Schoenholz, Sanghyun Han, Sanjeev Kwatra, Sarah A. Rous, Sarik Ghazarian, Sayan Ghosh, Sean Casey, Sebastian Bischoff, Sebastian Gehrmann, Sebastian Schuster, Sepideh Sadeghi, Shadi S. Hamdan, Sharon Zhou, Shashank Srivastava, Sherry Shi, Shikhar Singh, Shima Asaadi, Shixiang Shane Gu, Shubh Pachchigar, Shubham Toshniwal, Shyam Upadhyay, Shyamolima Debnath, Siamak Shakeri, Simon Thormeyer, Simone Melzi, Siva Reddy, Sneha Priscilla Makini, Soo-Hwan Lee, Spencer Torene, Sriharsha Hatwar, Stanislas Dehaene, Stefan Divic, Stefano Ermon, Stella Biderman, Stephanie Lin, Stephen Prasad, Steven T Piantadosi, Stuart M. Shieber, Summer Misherghi, Svetlana Kiritchenko, Swaroop Mishra, Tal Linzen, Tal Schuster, Tao Li, Tao Yu, Tariq Ali, Tatsunori Hashimoto, Te-Lin Wu, Theo Desbordes, Theodore Rothschild, Thomas Phan, Tianle Wang, Tiberius Nkinyili, Timo Schick, Timofei Kornev, Titus Tunduny, Tobias Gerstenberg, Trenton Chang, Trishala Neeraj, Tushar Khot, Tyler Shultz, Uri Shaham, Vedant Misra, Vera Demberg, Victoria Nyamai, Vikas Raunak, Vinay Venkatesh Ramasesh, Vinay Uday Prabhu, Vishakh Padmakumar, Vivek Srikumar, William Fedus, William Saunders, William Zhang, Wout Vossen, Xiang Ren, Xiaoyu Tong, Xinran Zhao, Xinyi Wu, Xudong Shen, Yadollah Yaghoobzadeh, Yair Lakretz, Yangqiu Song, Yasaman Bahri, Yejin Choi, Yichi Yang, Yiding Hao, Yifu Chen, Yonatan Belinkov, Yu Hou, Yu Hou, Yuntao Bai, Zachary Seid, Zhuoye Zhao, Zijian Wang, Zijie J. Wang, Zirui Wang, and Ziyi Wu. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _ArXiv_ , abs/2206.04615, 2022. URL https://api.semanticscholar.org/CorpusID:263625818. 

- Lena Strobl, William Merrill, Gail Weiss, David Chiang, and Dana Angluin. Transformers as recognizers of formal languages: A survey on expressivity. _arXiv preprint arXiv:2311.00208_ , 2023. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei. Augmenting language models with long-term memory. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. _arXiv preprint arXiv:2203.11171_ , 2022. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Huai hsin Chi, F. Xia, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. _ArXiv_ , abs/2201.11903, 2022. URL https://api.semanticscholar.org/ CorpusID:246411621. 

13 

Preprint. Under review. 

- Zhen Yang, Ming Ding, Qingsong Lv, Zhihuan Jiang, Zehai He, Yuyi Guo, Jinfeng Bai, and Jie Tang. Gpt can solve mathematical problems without a calculator. _arXiv preprint arXiv:2309.03241_ , 2023. 

- Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stephane Deny. Barlow twins: Selfsupervised learning via redundancy reduction. In Marina Meila and Tong Zhang (eds.), _Proceedings of the 38th International Conference on Machine Learning_ , volume 139 of _Proceedings of Machine Learning Research_ , pp. 12310–12320. PMLR, 18–24 Jul 2021. URL https: //proceedings.mlr.press/v139/zbontar21a.html. 

- Jiachen Zhu, Katrina Evtimova, Yubei Chen, Ravid Shwartz-Ziv, and Yann LeCun. Variancecovariance regularization improves representation learning. _arXiv preprint arXiv:2306.13292_ , 2023. 

## A APPENDIX 

- A.1 EXPERIMENTS 

|**Parameter**|**Value/Details**|
|---|---|
|**Learning Rate**|0.0001|
|**Batch Size**|128|
|**Optimizer**|AdamW|
|**Dropout**|0.1|
|**Attn. Heads**|4|
|**Epochs**|100|
|**Number of Layers**|4,5,6|
|**Input Sequence Length**|50,80,100|
|**# of Operators**|4,5,6|
|**Total Compute Resources**|1 32GB GPU,6CPU,32 GB RAM|
|**Total Job Time**|max 24 Hrs|



Table 2: Summary of Hyper-parameters, Compute Resources, and Time for Experiments. Each experiment was run on same configurations of GPUs, with the total time dependent on the number of layers and input parameters such as Dataset, task complexity. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0014-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0014-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0014-10.png)


**----- Start of picture text -----**<br>
(a) Test accuracy on Arithmetic Expressions Dataset (b) Test accuracy on LIS Dataset<br>**----- End of picture text -----**<br>


Figure 9: Seq-VCR + Pause Accuracy Vs Number of Layers for tasks of different complexity in Arithmetic Expressions and LIS 

Both plots in Figure9 demonstrate the trade-offs between model complexity (in terms of layers) and the complexity of the input (number of operators or sequence length). The model performs best when handling simpler configurations, such as fewer operators or shorter sequences, and tends 

14 

Preprint. Under review. 

to struggle with more complex setups as the number of operators or sequence length increases. Moreover, Increasing the number of layers seems to mitigate performance drops to some extent, especially in simpler cases, but it cannot fully offset the challenges posed by more complex inputs. All results in figure 9 are over 3 seed runs. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning_images/iclr_2025_Seq-VCR_Preventing_Collapse_in_Intermediate_Transformer_Representations_for_Enhanced_Reasoning.pdf-0015-02.png)


Figure 10: Ablation of Varying Pause Tokens and Comparing Vanilla Vs Seq-VCR + Pause Tokens 

Figure10 shows that Seq-VCR + Pause model consistently achieves high accuracy (around 0.8) across all numbers of pause tokens, indicating that the addition of pause tokens does not significantly affect its performance. The Vanilla model, on the other hand, shows much lower accuracy across all settings, with a slight decrease in accuracy as the number of pause tokens increases (especially noticeable at 4, 6, and 8 pause tokens). 

|Dataset|Task|Size|# Input<br>Tokens|# CoT<br>Tokens|# Output<br>Tokens|
|---|---|---|---|---|---|
|Multiplication|4 x 4 Mult<br>5 x 5 Mult|808k<br>808k|9<br>11|47<br>75|8<br>10|
|Arithmetic<br>Expression|4 Operators<br>5 Operators<br>6 Operators|1M<br>1M<br>1M|19<br>23<br>27|24<br>40<br>60|1<br>1<br>1|
|LIS|Seq Length 50<br>Seq Length 80<br>Seq Length 100|1M<br>1M<br>1M|50<br>80<br>100|50<br>80<br>100|1<br>1<br>1|



Table 3: Dataset statistics. Size refers to the training set. The number of input, output, and intermediate chain of thought tokens are median values on the validation set. The number of tokens are based on the GPT-2 tokenizer, and a special ending symbol is counted for both intermediate tokens and output tokens. 

15 

