Under review as a conference paper at ICLR 2025 

# MIXTURE OF PARROTS : EXPERTS IMPROVE MEMORIZATION MORE THAN REASONING 

**Samy Jelassi** _[∗]_ Harvard University 

**Clara Mohri** 

Harvard University 

**David Brandfonbrener Alex Gu** Harvard University MIT Kempner Institute 

**Nikhil Vyas** Harvard University 

## **Nikhil Anand** 

Harvard University Kempner Institute 

## **David Alvarez-Melis** 

Harvard University Kempner Institute 

## **Yuanzhi Li** 

Microsoft Research 

## **Sham M. Kakade** 

Harvard University Kempner Institute 

## **Eran Malach** 

Harvard University Kempner Institute 

## ABSTRACT 

The Mixture-of-Experts (MoE) architecture enables a significant increase in the total number of model parameters with minimal computational overhead. However, it is not clear what performance tradeoffs, if any, exist between MoEs and standard dense transformers. In this paper, we show that as we increase the number of experts (while fixing the number of active parameters), the memorization performance consistently increases while the reasoning capabilities saturate. 

We begin by analyzing the theoretical limitations of MoEs at reasoning. We prove that there exist graph problems that cannot be solved by any number of experts of a certain width; however, the same task can be easily solved by a dense model with a slightly larger width. On the other hand, we find that on memory-intensive tasks, MoEs can effectively leverage a small number of active parameters with a large number of experts to memorize the data. We empirically validate these findings on synthetic graph problems and memory-intensive closed book retrieval tasks. Lastly, we pre-train a series of MoEs and dense transformers and evaluate them on commonly used benchmarks in math and natural language. We find that increasing the number of experts helps solve knowledge-intensive tasks, but fails to yield the same benefits for reasoning tasks. 

## 1 INTRODUCTION 

The explosion in capabilities of large language models in recent years has largely been enabled by scaling their size, as measured by the number of parameters in the model. In the standard Transformer architecture, scaling the number of parameters entails a proportional increase in computational cost, e.g. doubling the number of parameters requires doubling the number of floating-point operations (FLOPs), making training and inference more computational intensive. Mixture-of-Experts (MoE) were introduced as a solution for this problem (Shazeer et al., 2017; Lepikhin et al., 2020; Fedus et al., 2022). MoEs replace the single MLP in each Transformer block with multiple MLPs (called experts), where each token is routed to a few experts based on a linear routing function. The number of parameters in the MoE layer therefore increases with the total number of experts, while the compute increases only with the number of “active” experts (i.e., the number of experts to which the token is routed to). This offers a promising option for scaling models: increase the number of experts instead of the model dimension or its depth. For this reason, MoEs have become very popular, and many frontier models today are based on the MoE architecture (Achiam et al., 2023; Databricks, 2023; Anil et al., 2023; Dai et al., 2024; Jiang et al., 2024; Yang et al., 2024). 

> _∗_ Correspondence to: Samy Jelassi _<_ sjelassi@fas.harvard.edu _>_ 

1 

Under review as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0002-01.png)


**----- Start of picture text -----**<br>
NLP (world knowledge) NLP (commonsense) MATH<br>35 53 44<br>26 50 32<br>20 46<br>18<br>14<br>42<br>86 39 72<br>16M 49M 159M 569M 2.1B 16M 49M 159M 569M 2.1B 16M 49M 159M 569M 2.1B<br>Number of total parameters Number of total parameters Number of total parameters<br>Dense transformer MoE (18M active parameters) MoE (58M active parameters) MoE (200M active parameters)<br>)<br>%(Accuracy )%( )%(<br>Accuracy Accuracy<br>F1<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0002-02.png)


**----- Start of picture text -----**<br>
(a) Evaluation: world knowledge (b) Evaluation: commonsense (c) Evaluation: math<br>**----- End of picture text -----**<br>


Figure 1: **(a) Evaluation: world knowledge.** We train a series of dense transformers and MoEs on 65B tokens from a corpus essentially made of Fineweb-edu, Cosmopedia and Wikipedia (see Section 5 for details). We then evaluate the models on several world knowledge benchmarks (e.g., TriviaQA (Joshi et al., 2017), Natural Questions (Kwiatkowski et al., 2019)) and report the average F1 accuracy. Surprisingly, at a fixed number of total parameters, MoEs with substantially fewer active parameters approximately match the performance of dense models. This highlights the importance of experts in tasks that require memorization. **(b) Evaluation: commonsense.** Here we evaluate the aforementioned pre-trained models on natural language commonsense benchmarks (e.g., HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2021)). On these reasoning tasks, we observe that MoEs perform worse than dense models and more significant benefits are obtained by increasing the number of active parameters. **(c) Evaluation: math.** Here we train a series of dense transformers and MoEs on 65B tokens from a corpus essentially made of Proof-Pile2 (Azerbayev et al., 2023) (see Section 5 for details). The results are consistent with the ones in (b): MoEs perform worse than dense models at equal number of total parameters. 

In this work we study whether MoE indeed offers a “free-lunch”, enabling gains in performance with no computational cost. Interestingly, we find that the benefit from MoEs greatly depends on the task at hand. We show that for reasoning-based tasks, such as graph problems and mathematical reasoning, MoEs offer limited performance gains, and increasing the number of experts cannot compete with scaling the dimension (width) of the model. On the other hand, for memory-intensive tasks, we show that scaling the number of experts is competitive with scaling standard “dense” MLPs. 

To demonstrate these claims, we begin with a theoretical analysis of MoEs and dense models. We use communication-complexity lower bounds to show that a single-layer MoE requires a critical dimension to solve a simple graph connectivity problem, implying that MoEs offer no benefit for solving this problem and only consume unnecessary memory. On the other hand, we show that for a pure memorization task, where the model only needs to “remember” an arbitrary set of examples, scaling the number of experts is equivalent to scaling the number of parameters in dense transformers, implying a significant computational gain when fixing the number of active parameters (Section 3). We continue by experimentally validating these results, comparing MoEs against dense models on synthetic training data. We train these models on finding the shortest path in random graphs, where we show that MoE accuracy does not improve as we increase the number of experts, but that accuracy consistently increases with width for a dense transformer (Figure 4b). Following this, we train different models on the task of memorizing a large phone-book. We demonstrate that MoEs excel in memorization, matching the performance of dense transformers with the same number of total parameters but with substantially less computational cost (Figure 4a). 

Finally, we train dense transformers and MoEs on real datasets of mathematical reasoning and natural language, and perform intensive benchmarking of these models on a wide variety of downstream tasks. For memory-intensive tasks, MoEs surprisingly have a great advantage, where increasing the number of experts can match the performance of large dense models (Figure 1a). However, we show that for tasks that rely on reasoning, scaling the number of experts cannot compete with increasing the model dimension (Figures 1b-1c). Moreover, MoEs exhibit some memorization behaviors when trained on math problems (Figure 5). Taken together, our results show that the gains from using MoEs depend greatly on the nature of the training data and downstream task, and that while MoEs can improve performance in certain cases, sometimes increasing the effective size (width) of the model is unavoidable. 

2 

Under review as a conference paper at ICLR 2025 

## 2 RELATED WORK 

**Mixture of Experts.** Mixture-of-Experts (MoE) date back to the work of Jacobs et al. (1991); Jordan & Jacobs (1994). Shazeer et al. (2017); Fedus et al. (2022) were the first to scale this idea to deep learning and obtain state-of-the-art models in machine translation. Since then, several works have improved their routing algorithms (Lepikhin et al., 2020; Lewis et al., 2021; Roller et al., 2021; Clark et al., 2022; Zhou et al., 2022; Antoniak et al., 2023; Zhong et al., 2024), have improved their downstream performance after finetuning (Du et al., 2022; Zoph et al., 2022) or made their training and inference more efficient (Rajbhandari et al., 2022; Gale et al., 2023; Pan et al., 2024; Tan et al., 2024). However, only a few papers have studied the science of MoEs and their comparison with dense transformers. Clark et al. (2022); Krajewski et al. (2024) establish scaling laws for MoEs. Chen et al. (2022) design a specific classification problem where a model with multiple experts provably outperforms one with only one expert. Shazeer et al. (2017); Lepikhin et al. (2020); Artetxe et al. (2021); Lewis et al. (2021); Fedus et al. (2022); Du et al. (2022) show that given a fixed FLOP budget, MoEs are always better. However, these papers claim that on a per parameter basis, MoEs always seem comparatively worse than dense models. In this paper, we temper this claim by showing that it depends on the _nature of the task_ at hand: on reasoning tasks, we validate this claim but on memory-intensive tasks, equally-sized MoEs perform as well as dense transformers. 

**Language models and memorization.** Large language models (LLMs) store a considerable amount of knowledge in their parameters (Petroni et al., 2019; Heinzerling & Inui, 2020). They memorize useful knowledge such as facts and commonsense (Zhao et al., 2024). Many works studied how memorization occurs in LLMs by developing tools to locate the knowledge in the model (Meng et al., 2022; Allen-Zhu & Li, 2023; Liu et al., 2024) or by tracking the training dynamics (Tirumala et al., 2022; Speicher et al., 2024). We draw inspiration from Allen-Zhu & Li (2023) and evaluate the memorization of our models by pre-training them on a mixture of datasets that includes Wikipedia, and at test time, evaluate them on world knowledge benchmarks, which are essentially question answering tasks on Wikipedia facts. With respect to theoretical findings, Kim et al. (2023); Mahdavi et al. (2023); Madden et al. (2024) provide upper bounds on the number of parameters needed for dense transformers to perform memorization tasks under various conditions. 

**Language models and reasoning.** In recent years, transformer-based language models have displayed remarkable effectiveness in solving a broad range of reasoning tasks. Specifically, the reasoning capabilities of transformers have been studied in the context of arithmetic problems (Jelassi et al., 2023; Cho et al., 2024; Hou et al., 2024; Zhou et al., 2024; McLeish et al., 2024; Lee et al., 2023), mathematical reasoning (Zhang et al., 2022; Imani et al., 2023; Wei et al., 2022) graph problems (Sanford et al., 2024; Fatemi et al., 2023; Jin et al., 2023; Wang et al., 2024) and code challenges (Shi et al., 2024; Zhu et al., 2024). Recently, state-of-the-art language models were used for solving complex math olympiad problems (DeepMind, 2024; NuminaMath, 2024; OpenAI, 2024). With respect to theoretical findings, various works study the reasoning capabilities of transformers, relating their expressive power to other complexity classes and formal languages (Weiss et al., 2021; Zhou et al., 2023; Strobl et al., 2024). Other works study how chain-of-thought can improve the reasoning capabilities of language models in terms of expressive power and learnability (Abbe et al., 2024; Merrill & Sabharwal, 2023; Malach, 2024). However, the reasoning capabilities of MoE language models compared to their dense counterparts have received comparatively less attention. 

## 3 THEORY: REPRESENTATIONAL CAPACITY 

In this section, we analyze the capability of MoE transformers compared to standard (dense) models. We begin by studying a simple graph problem that requires scaling the hidden dimension of the transformer, showing that MoEs with small hidden dimension cannot solve this problem, regardless of the number of experts used. Then, we show that MoEs can effectively memorize random inputs, requiring significantly less computational resources (active parameters) compared to dense models. 

## 3.1 SETTING 

Consider a one-layer transformer _f ∈_ Transformer _[N] m,H,_ 1[which takes as input a sequence of length] _N_ and has logarithmic bit-precision. _f_ embeds the input into dimension _m_ via the function _ϕ_ . _f_ has 

3 

Under review as a conference paper at ICLR 2025 

_H ≥_ 1 attention heads, whose outputs are combined via concatenation before we apply point-wise function _ψ_[1] . 

Define the parameters as _Qh, Vh, Kh ∈_ R _[m][×][m]_ , _ϕ_ : _X →_ R _[m]_ , _ψ_ : R _[m] →_ R. The output of _f_ is: 

_f_ ( _**x**_ 1 _, . . . ,_ _**x** N_ ) = _ψ_ ��softmax� _ϕ_ ( _xN_ ) _[⊤] QhKh[⊤][ϕ]_[(] _[X]_[)] � _ϕ_ ( _X_ ) _Vh_ � _h∈_ [ _H_ ]� _._ 

_f_ is a _dense_ transformer, if _ψ_ is an MLP, i.e. function of the form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0004-05.png)


where _σ_ is the ReLU activation function. _f ∈_ Transformer _[N] m,H,_ 1 _,K_[is a] _[ sparse]_[ (MoE) transformer] with _K_ experts if _ψ_ is a function of the form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0004-07.png)


where _**W**_ 1 _, . . . ,_ _**W** k ∈_ R _[m][′][×][m]_ , _**b**_ 1 _, . . . ,_ _**b** k ∈_ R _[m][′]_ , **u** 1 _, . . . ,_ **u** _k ∈_ R _[m][′]_ are the parameters of each expert and _r_ 1 _, . . . , rk_ define the routing function (we use top-1 routing). 

## 3.2 MOES REQUIRE A CRITICAL HIDDEN SIZE TO SOLVE GRAPH REASONING TASKS 

In this section, we analyze the graph reasoning capabilities of dense and sparse transformers. We define the length-2 path problem on a graph, and use it as a means to understand other graph reasoning tasks such as graph connectivity, shortest path, and cycle detection. 

**Definition 3.1** (Length-2 Path Problem) **.** The input is a graph _G_ = ( _V, E_ ). The source _s ∈ V_ and a destination _d ∈ V_ are fixed for all tasks as the 0 and _|V |_ vertex. The length-2 path problem asks whether there is a path of length 2 from _s_ to _d_ . 

Graph connectivity, shortest path, and cycle detection are all graph reasoning tasks which reduce to the length-2 path problem due to (Sanford et al., 2024) and Lemma C.2. We provide a lower-bound on the width required for a sparse transformer to solve the length-2 path problem, and an upper-bound on the width required for a dense transformer to solve the problem. Further, we show a separation between dense and sparse transformers with the same number of parameters: for a sufficiently large amount of experts in the sparse model, it cannot solve the same problem that a dense model can solve with the same amount of _total_ parameters. 

**Lower bound on width of depth-1 MoE for reasoning.** We begin by showing a lower-bound on the width for a depth-1 mixture of expert model for the length-2 path problem. This lower bound implies a lower bound for search and retrieval tasks such as graph connectivity, shortest path, and cycle detection. 

**Theorem 3.2** (Length-2 path lower-bound on sparse transformers) **.** _For some input sequence G_ = ( _V, E_ ) _, fix two disjoint subsets A, B ⊂_ [ _N −_ 1] _, and consider a single-layer transformer f ∈ Transformer[N] m,H,_ 1 _,K[with][ O]_[(log] _[ N]_[)] _[-bit precision that solves length-2 path for any input][ X][where] XA is a function of edges with the source s, XB is a function of edges with the destination d. Then, f has width satisfying mH_ = Ω( _|V |/_ log _N_ ) _._ 

The proof follows almost identically from the proof in (Sanford et al., 2024) for the class Transformer _[N] m,H,_ 1[.][The original proof does not place constraints on the function] _[ ψ]_[ and is based on] a communication-complexity argument. As such we may design _ψ_ so that it first routes and then chooses which expert to apply. We give a complete proof in Appendix C. As such, the result of (Sanford et al., 2024) can also be extended to the class Transformer _[N] m,H,_ 1 _,K_[.] 

**Upper bound on width of depth-1 dense transformer for reasoning.** In this section we give an upper bound for the width required for a dense model to solve the length-2 path problem. 

1 In multi-layer Transformers, each layer outputs a vector of size _m_ . However, since our focus in this section will be on binary classification problems, we will let the transformer output a single scalar, and we interpret the output of the final token as the prediction for the classification task. 

4 

Under review as a conference paper at ICLR 2025 

**Theorem 3.3** (Length-2 path width upper bound for transformer) **.** _There exists a transformer of width |V |, H_ = 1 _, and O_ (log _N_ ) _-bit precision that solves length-2 path problem for any input._ 

The proof relies on an encoding of the inputs where the output values only exceed a certain threshold when _u_ and _v_ , the source and destination vertices, have edges with a common vertex. We defer the proof to Appendix C. 

**Parameter-matched comparison of dense and sparse depth-1 transformers.** Using the lowerbound on width required for a sparse transformer (Theorem 3.2) and the upper-bound on width required for a dense transformer (Theorem 3.3), we compare dense and sparse transformers when they have the same number of total parameters. We find that when the number of experts exceeds (log _N_ )[2] , the sparse model is unable to solve the same task as the dense model. 

**Corollary 3.4.** _Consider a sparse transformer (with K experts) and a dense transformer with the same number of parameters. There exists a number of experts K so that the the sparse model is not able to solve the reasoning task, but the dense transformer solves the task._ 

_Proof._ Suppose we have two depth-1 transformers, where one is a dense model and the other is a mixture of experts with _K_ experts. Let the width of the dense model be _md_ , and the width of the sparse model be _ms_ . The number of parameters in the dense model is _O_ ( _m_[2] _d_[)][ and the number of] parameters in the sparse model is _O_ ( _Km_[2] _s_[)][.][In order to match the number of parameters, it must be] the case that _ms_ = ~~_√_~~ _[m] K[d]_[.][Suppose we let] _[ m][d]_[=] _[ |][V][ |]_[, as this is sufficient to solve the above problems.] For any _K ≥_ Ω�(log _N_ )[2][�] , the sparse model is not sufficiently wide to solve the problem. 

## 3.3 MOES USE THEIR EXPERTS TO SOLVE MEMORY-INTENSIVE TASKS 

In this section, we provide an upper-bound on the number of parameters necessary for a sparse transformer to solve memorization tasks, followed by a lower-bound on the number of parameters needed for a dense transformer to solve the same task. We use these results to compare the memorization capabilities of dense and sparse transformers with the same number of active parameters. We find that with enough experts, the sparse transformer is able to solve memorization tasks with less active parameters than the dense transformer. In both bounds we assume that transformer has logarithmic number of bits to encode each parameter. 

We consider sequences _{_ ( _X[i] , yi_ ) _}[n] i_ =1[where] _[X][i][∈]_[R] _[N][×][m]_[are][input][sequences][of][length] _[N]_[in] dimension _m_ such that _X[i]_ [ _j_ ] is sampled from a Gaussian distribution _N_ (0 _, Im_ ). We assume _y_ 1 _, . . . , yN ∈{±_ **1** _}_ are arbitrary labels for the _n_ sequences. The objective is for a transformer to memorize these sequences, i.e. map each input _X[i]_ to a label _yi_ . The classification is determined by the sign of the last token output. 

**Upper-bound on MoE for memorization.** We begin by showing that, with high probability over the choice of the inputs, the MoE architecture can memorize (i.e., arbitrarily label the examples), with a small number of active parameters. **Theorem 3.5.** _With probability at least_ 0 _._ 99 _, there exists a one-layer MoE transformer with K experts, using O_[˜] _n active parameters and O_[˜] ( _n_ + _Km_ ) _total parameters stored in O_[˜] (1) � _K_[+] _[ Km]_ � _bits that, when applied to each sequence X[i] , outputs at the last token a value whose sign matches yi, i.e.,_ sign( _f_ ( _Xi_ )) = _yi for all i_ = 1 _, . . . , n._[2] 

Specifically, if we choose _K_ = ~~�~~ _n/m_ we get that an MoE architecture can solve the memorization problem with _O_[˜] ( _[√] nm_ ) parameters. To prove this, we show that for a particular routing function, the number of samples routed to each expert is approximately˜ _n/K_ . Then, we show that an expert with _O_ ( _n/mK_ ) neurons can memorize a sample of size _O_ ( _n/K_ ). We present the proof in Appendix C.2. 

**Lower bound on memorization with dense Transformer.** Next, we give a lower-bound on the number of parameters for a dense transformer to perform memorization. 

> 2We use _O_ ˜ and ˜Ω to hide logarithmic factors. 

5 

Under review as a conference paper at ICLR 2025 

**Theorem 3.6** (Lower bound for dense model) **.** _Given the same task as above, a dense Transformer requires_ Ω([˜] _n_ ) _parameters to solve the memorization task._ 

This bound follows from the fact that there are 2 _[n]_ possible labels for any fixed set of _n_ inputs, and at most 2 _[cW]_ functions with _W_ parameters and _c_ bit per parameters. The proof is in Appendix C.2. 

**Separation between MoEs and Dense Models.** Observe that the previous results on memorization imply a separation between MoEs and dense models in terms of the number of active parameters. Namely, we show that an MoE with _O_[˜] ( _[√] nm_ ) active parameters can memorize, while a dense model requires Ω([˜] _n_ ) parameters. So, for _n ≫ m_ , MoEs are significantly more efficient. Comparing the number of total parameters, MoEs require _O_[˜] ( _n_ + _Km_ ) parameters, so both MoE and dense models have linear dependence on _n_ in the total parameter count. 

## 4 SYNTHETIC EXPERIMENTS 

In the previous section, we proved that there exist graph connectivity problems that cannot be solved by any number of experts of a certain width but the same task can be solved by a dense model with a slightly larger width. Our goal in this section is to verify that our theoretical analysis bears out experimentally when training models from scratch on synthetic data, before moving on to study pre-trained models in Section 5. We mainly focus on two tasks: the _shortest path_ problem (Figure 2), which we use as a synthetic task to represent reasoning problems, and the _phone-book_ task (Figure 3), to measure the memorization ability of our models. Our experiments in this section highlight that adding experts yields greater performance improvements on memorization tasks than reasoning tasks. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0006-06.png)


Figure 2: Illustration of the shortest path task. We feed the model with a sequence that lists all the edges in the input graph and ends with the query (in green) which asks the model to find a shortest path between two vertices (from vertex 1 to vertex 4 in the figure). The model then autoregressively returns the shortest path (in purple). 

## 4.1 EXPERIMENTAL SETUP 

**Architecture.** We opt for the Mistral (Jiang et al., 2023) and Mixtral (Jiang et al., 2024) architectures as the backbones of our Transformer and MoE models, respectively. The two architectures are identical and only differ by the addition of a gating module and multiple experts in Mixtral. For both model types, we fix the number of layers to _L_ = 12. For the dense transformers, we vary model size by sweeping the width _d ∈{_ 256 _,_ 512 _,_ 1024 _}_ . For MoEs, we sweep over widths _d ∈{_ 256 _,_ 512 _}_ and the number of experts _E ∈{_ 8 _,_ 16 _,_ 32 _,_ 64 _}_ . To be consistent with our experiments in Section 5, we set the intermediate dimension in the FFN block to be equal to _d_ (and not 4 _d_ ). We use token-choice routing, do not apply any token dropping and each token is routed to the top-2 experts. Lastly, in both this section and Section 5, we report for each model the number of non-embedding parameters which we refer to as the total number of parameters. 

**Shortest path task.** For a graph with _n_ vertices, our token space _V_ is of size _n_ + 6 with tokens encoding the vertices and some special tokens: _V_ = _{_ 1 _, . . . , n, ⟨_ EDGE _⟩ , ⟨_ BOS _⟩ , ⟨_ EOS _⟩ , ⟨_ PAD _⟩ , ⟨_ SEP _⟩ , /}_ where _⟨_ BOS _⟩_ is the beginning of sentence token, _⟨_ EOS _⟩_ the end of sentence token, _⟨_ PAD _⟩_ the padding token, _⟨_ EDGE _⟩_ is the token indicating an edge between two vertices and, _⟨_ SEP _⟩_ and “/” are separator tokens. Each sequences describes the graph by a list of all the edges followed by two randomly sampled vertices and the shortest path between these latter (see Figure 2). All the graphs are directed and sampled according to the Erdos-R¨ enyi model, with´ _n_ vertices and probability _p_ for each edge to exist. We vary _n ∈{_ 25 _,_ 30 _,_ 50 _,_ 40 _,_ 45 _,_ 50 _,_ 55 _}_ and set _p_ such that the average length of the shortest path is 3.5. Each train/test pair corresponds to _one_ value of ( _n, p_ ), we do not mix graph sizes. 

6 

Under review as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0007-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0007-02.png)


Figure 3: Illustration of the phone-book task for closed-book retrieval. The model is first trained to memorize a phone-book (illustrated on the right). Then, we randomly select a name in the phone-book (in green) and ask the model to return their phone number (in purple) without access to the phone-book. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0007-04.png)


**----- Start of picture text -----**<br>
PHONE-BOOK SHORTEST PATH (n=50)<br>5.5M 80<br>3M<br>75<br>1M<br>500k<br>68<br>100k<br>50k 63<br>5M 22M 88M 352M 700M 5M 22M 88M 352M 700M<br>Number of total parameters Number of total parameters<br>Dense transformer MoE (10M active parameters) MoE (42M active parameters)<br>(a) Phone-book memorization (b) Shortest path (50-node graphs)<br>)<br>size %(<br>Phonebook Accuracy<br>**----- End of picture text -----**<br>


Figure 4: **(a) Phone-book memorization** : We train a series of dense transformers and MoEs on phone-books of varying sizes and then evaluate their memorization capacity. We report the maximal phone-book size where the model obtains more than 90% accuracy. The maximal phone-book size correlates with the total (and not active) number of parameters. **(b) Shortest path (total parameters)** : We train models to find the shortest path in 50-node graphs and report the test accuracy. Here, increasing the number of experts provides limited improvements and the performance rather correlates with the number of active parameters. 

**Phone-book task.** Our token space _V_ is of size 39 and made of the alphabet letters, digits and special tokens: _V_ = _{a, . . . , z,_ 0 _, . . . ,_ 9 _, ⟨_ BOS _⟩ , ⟨_ EOS _⟩ , ⟨_ SEP _⟩}_ . We generate phone-books where the names consist of 5 letters and the phone numbers of 8 digits (see Figure 3). We ensure that both the names and numbers are unique. 

**Datasets.** For the graph experiments, the training set size is 1e6 and the test set consists of 1e3 held-out examples that are sampled from the _same_ distribution as the training examples. For the phone-book experiments, we vary the training set size over _{_ 1e5 _,_ 5e5 _,_ 1e6 _,_ 1 _._ 5e6 _,_ 2e6 _,_ 2 _._ 5e6 _,_ 3e6 _}_ and the test set consists of 1e3 queries from the training set. 

**Optimization.** We use the AdamW optimizer (Loshchilov et al., 2017) with a weight decay equal to 0.1. We sweep the learning rate over _{_ 5e _−_ 5 _,_ 1e _−_ 4 _,_ 5e _−_ 4 _,_ 1e _−_ 3 _}_ , the number of epochs over _{_ 2 _,_ 5 _,_ 10 _,_ 15 _}_ , and set the maximal possible batch size among _{_ 8 _,_ 16 _,_ 32 _}_ . We use a warmup during the 20% first training steps and a linear decay scheduler. All models are trained by next-token prediction. In the graph task, we apply a mask on the input instance so that we only penalize the model whenever it makes a mistake on the labels (and not on the inputs and labels jointly). In the phone-book experiment, we do not apply any masking. 

**Evaluation.** For each task we compute the exact accuracy, i.e. we count the generation as correct only if it fully matches the ground truth. For the phone-book task, we report the size of the maximal phone-book where we observe at least 90% exact accuracy. 

## 4.2 MEMORIZATION: TOTAL PARAMETERS PREDICT PERFORMANCE 

We train dense transformers and MoEs on phone-books of different sizes and at test time, evaluate whether they memorize the phone number of some names. Figure 4a reports the maximal phone-book size where the model manages to get an accuracy greater than 90%. This gives us an estimate of the memorization capacity of the model. 

7 

Under review as a conference paper at ICLR 2025 

The findings are clear: no matter the number of active parameters, MoEs match the performance of dense transformers with the same number of total parameters. This suggests that MoEs are able to effectively leverage the extra parameters in additional experts by routing tokens to the experts that contain the necessary information from the training corpus. This scaling is remarkable in this case since it even holds when we are only routing to 2 out of 64 experts. For instance, we find that an MoE model with only 42M active parameters outperforms a dense model with 10x as many parameters. This type of impressively efficient memorization capacity may be a major reason behind the success of MoE architectures. 

## 4.3 REASONING: TOTAL PARAMETERS DO NOT PREDICT PERFORMANCE 

We train dense transformers and MoEs on the shortest path task and then query the models to find the shortest paths in novel, held-out graphs. Figure 4b reports the performance on graphs with 50 nodes with respect to their number of total parameters. Contrary to the phone-book experiment, increasing the number of experts does not consistently improve the performance of MoEs. Essentially, we find that active parameters rather than total parameters is a better predictor of performance for these reasoning tasks. 

To connect back to the theory from Section 3, note that active parameters is directly determined by the width of the network since we always route to exactly 2 experts and fix the depth. Thus, these results corroborate the theory by showing that width (i.e. active parameters) determines the performance on these graph reasoning problems and that increasing the number of experts is not helpful. In Section 5, we will further corroborate this idea through evaluation of pre-trained models on commonsense and math reasoning benchmarks. 

## 5 PRE-TRAINED MODELS 

In this section, we pre-train dense transformers and MoEs and compare their performance on standard math and natural language benchmarks. We break the downstream tasks into those that require more memorization and those that require more reasoning. The memorization-intensive tasks test for “world knowledge” and consist of benchmarks like TriviaQA (Joshi et al., 2017). We break the reasoning-intensive tasks into two subcategories: one for natural language reasoning tasks like WinoGrande (Sakaguchi et al., 2021) and another for mathematical reasoning tasks like HendrycksMATH (Hendrycks et al., 2021). These tasks may be seen as real-world analogs of the stylized phone-book and shortest path tasks studied in Section 4. 

We observe that performance on world-knowledge tasks is governed by the total number of parameters while performance on reasoning tasks depends more on the number of active parameters (Figure 1). Additionally, we conduct an experiment that indicates memorization from MoEs may be harming reasoning performance since there is a larger gap between train and test accuracy for MoEs than dense models at fixed total parameters (Figure 5). Finally, we conduct an ablation where we compare models at fixed validation perplexity rather than model size. We find that MoEs perform better on world knowledge tasks and similarly on reasoning tasks compared to dense models (Figure 6). 

## 5.1 SETUP 

**Architecture.** We train dense transformers and MoEs using the OLMoE codebase (Muennighoff et al., 2024). We set the number of layers _L_ = 20 and vary the width _d ∈ {_ 256 _,_ 512 _,_ 1024 _,_ 2048 _,_ 4096 _}_ for dense transformers and _d ∈{_ 256 _,_ 512 _,_ 1024 _}_ for MoEs. Similarly to Muennighoff et al. (2024), we consistently set the intermediate dimension in the FFN/MoE blocks to _d_ (and not 4 _d_ ). For MoEs, we vary the number of experts _E ∈{_ 8 _,_ 16 _,_ 32 _,_ 64 _}_ . For the specific case of width 256, we also train a MoE with 256 experts because its parameter count approximately matches the one of a width-2048 dense model and thus, we can compare the downstream performance of the two models. We use top-2 token-choice routing, without token dropping which is implemented in the dMoE function from the Megablocks package (Gale et al., 2023). 

**Training hyperparameters.** We use the AdamW optimizer (Loshchilov et al., 2017) with a weight decay equal to 0.1. We set the learning rate to 0 _._ 001, train on 63B tokens (60k steps) with batch size 

8 

Under review as a conference paper at ICLR 2025 

512 and sequence length of 2048. We use warmup during the 20% first training steps and a linear decay scheduler. We train our models using FSDP (Zhao et al., 2023). 

**Pre-training datasets.** We train two collections of models, one collection on natural language and another one on math. The natural language dataset is a mixture constituted of FineWeb-edu (Penedo et al., 2024), Cosmopedia (Ben Allal et al., 2024), Wikipedia and the training sets of the downstream tasks we evaluate on. The math dataset is a mixture made of Proof-Pile 2 (Azerbayev et al., 2023) and instruction datasets such as OpenMathInstruct (Toshniwal et al., 2024) and MetaMathQA (Yu et al., 2023). Each of the two training mixture approximately totals 65B tokens. A precise description of the training mixtures can be found in Appendix A. 

**Evaluation.** We measure the validation perplexity on 5,000 held-out sequences sampled from the training distribution. And we evaluate our models on a series of natural language and math benchmarks. Explicitly, we divide them into three categories: 

- World-knowledge tasks: TriviaQA (Joshi et al., 2017), Natural Questions (Kwiatkowski et al., 2019), HotpotQA (Yang et al., 2018), WebQuestions (Berant et al., 2013), ComplexWebQuestions (Talmor & Berant, 2018). 

- Commonsense tasks: ARC-C and ARC-E (Clark et al., 2018), CommonsenseQA (Talmor et al., 2018), HellaSwag (Zellers et al., 2019), OpenbookQA (Mihaylov et al., 2018), PIQA (Bisk et al., 2020), SciQ (Welbl et al., 2017), SIQA (Sap et al., 2019), WinoGrande (Sakaguchi et al., 2021). 

- Math benchmarks: SVAMP (Patel et al., 2021), GSM8k (Cobbe et al., 2021), GSM-Hard (Gao et al., 2023), Hendrycks-MATH (Hendrycks et al., 2021) and Minerva-MATH (Lewkowycz et al., 2022). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0009-07.png)


**----- Start of picture text -----**<br>
GSM8k<br>30<br>25<br>18<br>14<br>11<br>7<br>3<br>16M 49M 159M 569M 2.1B<br>Number of total parameters<br>(a)<br>HENDRYCKS-MATH<br>45<br>37<br>31<br>20<br>8<br>2<br>16M 49M 159M 569M 2.1B<br>Number of total parameters<br>(b)<br>Dense transformer<br>MoE (18M active parameters)<br>MoE (58M active parameters)<br>MoE (200M active parameters)<br>gap<br>Generalization<br>gap<br>Generalization<br>**----- End of picture text -----**<br>


In all our experiments, we plot the average accuracy for each of these three categories. We report the corresponding per-task performance in Appendix B. 

- 5.2 RESULTS 

**Experts improve memorization more than reasoning.** We observe that the conclusions from our theoretical results and synthetic experiments also hold when pre-training and evaluating language models on natural language and math. In Figure 1a, we report the accuracy of our models with respect to the number of _total_ parameters. All the lines in the plot approximately coincide which implies that regardless of the number of active parameters, MoEs can effectively use their routing to leverage all of their parameters to solve memory-intensive tasks. On the other hand, on commonsense and math benchmarks (Figures 1b,1c) we find that MoEs do not reach the performance of dense models with the same number of total parameters. This indicates that for these reasoning tasks, increasing the dense model width is more effective that adding experts. 

## **On math tasks, MoEs display a higher train-test gap** 

Figure 5: Generalization gap i.e., difference between the training and test accuracies, when the test set is GSM8k (a) and Hendrycks-MATH (b). 

**than dense models, suggestive of memorization.** We provide additional evidence that memorization occurs in pre-trained MoEs by considering the generalization gap. In Figure 5 we select 6,319 random problems from the 

OpenMathInstruct dataset, which is part of the training mixture data. More precisely, we pick 5,000 Hendrycks-MATH like examples and 1,319 GSM8k-like examples to ensure that the number of training examples matches with the corresponding number of examples in GSM8k and HendrycksMATH test sets. We then report the _generalization gap_ , which is the gap between the accuracy on training examples and test examples. While both dense transformers and MoEs make a _single_ pass on the OpenMathInstruct dataset, Figure 5 shows that at scales beyond 159M parameters, MoEs suffer 

9 

Under review as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0010-01.png)


**----- Start of picture text -----**<br>
NLP (world knowledge) NLP (commonsense) MATH<br>26 53 44<br>50<br>32<br>20<br>46<br>14 18<br>42<br>8 7<br>6 39 2<br>6.1 7.3 9.2 12.2 17.6 6.1 7.3 9.2 12.2 17.6 3.2 3.5 3.9 4.5 5.5<br>Validation perplexity Validation perplexity Validation perplexity<br>Dense transformer MoE (18M active parameters) MoE (58M active parameters) MoE (200M active parameters)<br>(a) (b) (c)<br>)<br>%(Accuracy )%( )%(<br>Accuracy Accuracy<br>F1<br>**----- End of picture text -----**<br>


Figure 6: (a) On world knowledge benchmarks, MoEs consistently outperform dense transformers in downstream performance when fixing the validation perplexity. (b-c) In reasoning benchmarks, dense transformers perform about the same as MoEs at a fixed validation perplexity. MoEs can achieve these perplexities with less active parameters, but may require substantially more total parameters. 

from a more significant generalization gap than dense transformers. This is suggestive that MoEs are more prone to memorize training data than dense models. 

**MoE models excel at world knowledge tasks but match dense models in reasoning when perplexity is fixed.** Finally, we focus on the relationship between validation perplexity and downstream performance in Figure 6. Rather than comparing models by their parameter count, we can compare them based on how well they fit the training distribution as measured by validation perplexity. Even though two models may have the same perplexity, they will have learned different functions. The question is then if we can see any high level patterns in which types of functions a particular model class is more likely to learn. Figure 6a shows that at a fixed perplexity, the MoE models outperform the dense models on world knowledge tasks. This suggests that MoEs do have a bias towards learning functions that memorize training data. On the other hand, Figures 6b and 6c show that MoEs and dense models perform about the same on the reasoning tasks at fixed validation perplexity. We can square this with the results from Figure 1 by noting that at equal total number of parameters an MoE has worse validation perplexity than the corresponding dense model. This suggests that while MoEs do not change the relationship between perplexity and downstream accuracy on reasoning tasks relative to dense models, they may struggle to learn the reasoning parts of the training distribution as well. 

Overall, our main findings in Figure 1 and supplementary experiments in Figures 5 and 6 corroborate the hypothesis that MoEs can effectively use more experts to increase their memory capacity, but not necessarily their capability to reason. 

## 6 DISCUSSION 

In recent years, scaling up the number of parameters in Transformers has been the dominant approach for improving performance on language modeling. A standard Transformer of dimension _d_ and sequence length _L_ has number of parameters which scales with _O_ ( _d_[2] ), and run-time that scales with _O_ ( _d_[2] _L_[2] ). Improving the efficiency can either attempt to reduce the dependence on _L_ or _d_ . Sub-quadratic attention variants attempt to improve dependence on _L_ (Katharopoulos et al., 2020; Peng et al., 2023; Fu et al., 2022; Gu & Dao, 2023), while MoEs attempt to improve dependence on _d_ by scaling the number of parameters without scaling the dimension of the model. 

This paper illuminates the costs and benefits of this reduced dependence on _d_ . We show that for some reasoning-intensive tasks increasing the dimension _d_ is inevitable, and scaling the computation with _O_ ( _d_[2] ) seems unavoidable. This remains true regardless of the different design choices in the MoE architecture and is backed up empirically. We note that there is increasing interest in developing non-MoE models with sub-quadratic dependence on _d_ , using some structural assumptions on the weight layers (Kamalakara et al., 2022; Dao et al., 2021; 2022; Fu et al., 2024), which could provide an alternative. 

10 

Under review as a conference paper at ICLR 2025 

On the other hand, we find that MoEs are highly effective at knowledge intensive tasks. They are able to much more efficiently memorize facts than dense models with a similar number of active parameters, even matching the performance of dense models with the same number of total parameters. This suggests that MoEs are valuable as memorization machines and perhaps this particular capability can be leveraged while relying on other architectures for more reasoning-intensive tasks. 

**Limitations and future work.** While we provide substantial experiments on pre-trained models with up to _≤_ 2 _._ 1B parameters, we recognize that large scale MoEs like Mixtral (Jiang et al., 2024), DeepSeek-V2 (Dai et al., 2024), and others have orders of magnitude more parameters. We hypothesize that our results would still be meaningful at larger scales due to the strong theoretical underpinning, but it is not guaranteed. Moreover, as suggested above, it would be an interesting direction for future work to propose new architectures with reduced _d_ dependence that can get the best of both worlds and solve reasoning and memorization tasks. 

## ACKNOWLEDGEMENTS 

We thank Cyril Zhang for helpful discussions and Max Shad for his support while running the experiments. Kempner Institute computing resources enabled this work. Samy Jelassi acknowledges funding supported by the Center of Mathematical Sciences and Applications. Alex Gu is supported by the National Science Foundation (NSF) Graduate Research Fellowship under Grant No. 2141064. David Alvarez-Melis acknowledges support from the Kempner Institute, the Aramont Fellowship Fund, and the FAS Dean’s Competitive Fund for Promising Scholarship. This work has been made possible in part by a gift from the Chan Zuckerberg Initiative Foundation to establish the Kempner Institute for the Study of Natural and Artificial Intelligence; support from the Office of Naval Research under award N00014-22-1-2377, and the National Science Foundation Grant under award #IIS 2229881. 

## REFERENCES 

- Emmanuel Abbe, Samy Bengio, Aryo Lotfi, Colin Sandon, and Omid Saremi. How far can transformers reason? the locality barrier and inductive scratchpad, 2024. URL https: //arxiv.org/abs/2406.06467. 

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.1, knowledge storage and extraction. _arXiv preprint arXiv:2309.14316_ , 2023. 

- Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: A family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 1, 2023. 

- Szymon Antoniak, Sebastian Jaszczur, Michał Krutul, Maciej Pioro, Jakub Krajewski, Jan Ludziejew-´ ski, Tomasz Odrzygo´zd´ z,´ and Marek Cygan. Mixture of tokens: Efficient llms through crossexample aggregation. _arXiv preprint arXiv:2310.15961_ , 2023. 

- Mikel Artetxe, Shruti Bhosale, Naman Goyal, Todor Mihaylov, Myle Ott, Sam Shleifer, Xi Victoria Lin, Jingfei Du, Srinivasan Iyer, Ramakanth Pasunuru, et al. Efficient large scale language modeling with mixtures of experts. _arXiv preprint arXiv:2112.10684_ , 2021. 

- Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics, 2023. 

- Loubna Ben Allal, Anton Lozhkov, Guilherme Penedo, Thomas Wolf, and Leandro von Werra. Cosmopedia, 2024. URL https://huggingface.co/datasets/HuggingFaceTB/ cosmopedia. 

11 

Under review as a conference paper at ICLR 2025 

- Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on freebase from question-answer pairs. In _Proceedings of the 2013 conference on empirical methods in natural language processing_ , pp. 1533–1544, 2013. 

- Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 34, pp. 7432–7439, 2020. 

- Zixiang Chen, Yihe Deng, Yue Wu, Quanquan Gu, and Yuanzhi Li. Towards understanding mixture of experts in deep learning. _arXiv preprint arXiv:2208.02813_ , 2022. 

- Hanseul Cho, Jaeyoung Cha, Pranjal Awasthi, Srinadh Bhojanapalli, Anupam Gupta, and Chulhee Yun. Position coupling: Leveraging task structure for improved length generalization of transformers. _arXiv preprint arXiv:2405.20671_ , 2024. 

- Aidan Clark, Diego de Las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hoffmann, Bogdan Damoc, Blake Hechtman, Trevor Cai, Sebastian Borgeaud, et al. Unified scaling laws for routed language models. In _International conference on machine learning_ , pp. 4057–4086. PMLR, 2022. 

- Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. _arXiv preprint arXiv:1803.05457_ , 2018. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixtureof-experts language models. _arXiv preprint arXiv:2401.06066_ , 2024. 

- Amit Daniely. Memorizing gaussians with no over-parameterizaion via gradient decent on neural networks. _arXiv preprint arXiv:2003.12895_ , 2020. 

- Tri Dao, Beidi Chen, Kaizhao Liang, Jiaming Yang, Zhao Song, Atri Rudra, and Christopher Re. Pixelated butterfly: Simple and efficient sparse training for neural network models. _arXiv preprint arXiv:2112.00029_ , 2021. 

- Tri Dao, Beidi Chen, Nimit S Sohoni, Arjun Desai, Michael Poli, Jessica Grogan, Alexander Liu, Aniruddh Rao, Atri Rudra, and Christopher Re.´ Monarch: Expressive structured matrices for efficient and accurate training. In _International Conference on Machine Learning_ , pp. 4690–4721. PMLR, 2022. 

Databricks. Introducing dbrx: A new state-of-the-art open llm. _Databricks Blog_ , 2023. URL https://www.databricks.com/blog/ introducing-dbrx-new-state-art-open-llm. Accessed: 2023-10-12. 

DeepMind. Ai achieves silver-medal standard solving international mathematical olympiad problems. https://deepmind.google/discover/blog/ ai-solves-imo-problems-at-silver-medal-level/, 2024. 

- Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. Glam: Efficient scaling of language models with mixture-of-experts. In _International Conference on Machine Learning_ , pp. 5547–5569. PMLR, 2022. 

- Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. Talk like a graph: Encoding graphs for large language models. _arXiv preprint arXiv:2310.04560_ , 2023. 

- William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. _Journal of Machine Learning Research_ , 23(120):1–39, 2022. 

12 

Under review as a conference paper at ICLR 2025 

Dan Fu, Simran Arora, Jessica Grogan, Isys Johnson, Evan Sabri Eyuboglu, Armin Thomas, Benjamin Spector, Michael Poli, Atri Rudra, and Christopher Re.´ Monarch mixer: A simple sub-quadratic gemm-based architecture. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Daniel Y Fu, Tri Dao, Khaled K Saab, Armin W Thomas, Atri Rudra, and Christopher Re.´ Hungry hungry hippos: Towards language modeling with state space models. _arXiv preprint arXiv:2212.14052_ , 2022. 

- Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia. Megablocks: Efficient sparse training with mixture-of-experts. _Proceedings of Machine Learning and Systems_ , 5:288–304, 2023. 

- Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In _International Conference on Machine Learning_ , pp. 10764–10799. PMLR, 2023. 

- Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. _arXiv preprint arXiv:2312.00752_ , 2023. 

Benjamin Heinzerling and Kentaro Inui. Language models as knowledge bases: On entity representations, storage capacity, and paraphrased queries. _arXiv preprint arXiv:2008.09036_ , 2020. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _arXiv preprint arXiv:2103.03874_ , 2021. 

- Kaiying Hou, David Brandfonbrener, Sham Kakade, Samy Jelassi, and Eran Malach. Universal length generalization with turing programs. _arXiv preprint arXiv:2407.03310_ , 2024. 

- Shima Imani, Liang Du, and Harsh Shrivastava. Mathprompter: Mathematical reasoning using large language models. _arXiv preprint arXiv:2303.05398_ , 2023. 

- Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. _Neural computation_ , 3(1):79–87, 1991. 

- Samy Jelassi, Stephane d’Ascoli, Carles Domingo-Enrich, Yuhuai Wu, Yuanzhi Li, and Fran´ c¸ois Charton. Length generalization in arithmetic transformers. _arXiv preprint arXiv:2306.15400_ , 2023. 

- Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. _arXiv preprint arXiv:2310.06825_ , 2023. 

- Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. _arXiv preprint arXiv:2401.04088_ , 2024. 

- Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, and Jiawei Han. Large language models on graphs: A comprehensive survey. _arXiv preprint arXiv:2312.02783_ , 2023. 

- Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm. _Neural computation_ , 6(2):181–214, 1994. 

- Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. _arXiv preprint arXiv:1705.03551_ , 2017. 

- Siddhartha Rao Kamalakara, Acyr Locatelli, Bharat Venkitesh, Jimmy Ba, Yarin Gal, and Aidan N Gomez. Exploring low rank training of deep neural networks. _arXiv preprint arXiv:2209.13569_ , 2022. 

- Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In _International conference on machine learning_ , pp. 5156–5165. PMLR, 2020. 

- Junghwan Kim, Michelle Kim, and Barzan Mozafari. Provable memorization capacity of transformers. In _The Eleventh International Conference on Learning Representations_ , 2023. 

13 

Under review as a conference paper at ICLR 2025 

Jakub Krajewski, Jan Ludziejewski, Kamil Adamczewski, Maciej Pioro, Michał Krutul, Szymon´ Antoniak, Kamil Ciebiera, Krystian Krol, Tomasz Odrzyg´ o´zd´ z, Piotr Sankowski, et al.´ Scaling laws for fine-grained mixture of experts. _arXiv preprint arXiv:2402.07871_ , 2024. 

- Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. _Transactions of the Association for Computational Linguistics_ , 7:453–466, 2019. 

- B. Laurent and P. Massart. Adaptive estimation of a quadratic functional by model selection. _The Annals of Statistics_ , 28(5):1302–1338, 2000. ISSN 00905364, 21688966. URL http: //www.jstor.org/stable/2674095. 

- Nayoung Lee, Kartik Sreenivasan, Jason D Lee, Kangwook Lee, and Dimitris Papailiopoulos. Teaching arithmetic to small transformers. _arXiv preprint arXiv:2307.03381_ , 2023. 

- Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. _arXiv preprint arXiv:2006.16668_ , 2020. 

- Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, and Luke Zettlemoyer. Base layers: Simplifying training of large, sparse models. In _International Conference on Machine Learning_ , pp. 6265–6274. PMLR, 2021. 

- Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. _Advances in Neural Information Processing Systems_ , 35:3843–3857, 2022. 

- Bingbin Liu, Sebastien Bubeck, Ronen Eldan, Janardhan Kulkarni, Yuanzhi Li, Anh Nguyen, Rachel Ward, and Yi Zhang. Tinygsm: achieving¿ 80% on gsm8k with small language models. _arXiv preprint arXiv:2312.09241_ , 2023. 

- Yan Liu, Yu Liu, Xiaokang Chen, Pin-Yu Chen, Daoguang Zan, Min-Yen Kan, and Tsung-Yi Ho. The devil is in the neurons: Interpreting and mitigating social biases in language models. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Ilya Loshchilov, Frank Hutter, et al. Fixing weight decay regularization in adam. _arXiv preprint arXiv:1711.05101_ , 5, 2017. 

- Liam Madden, Curtis Fox, and Christos Thrampoulidis. Upper and lower memory capacity bounds of transformers for next-token prediction. _arXiv preprint arXiv:2405.13718_ , 2024. 

- Sadegh Mahdavi, Renjie Liao, and Christos Thrampoulidis. Memorization capacity of multi-head attention in transformers. _arXiv preprint arXiv:2306.02010_ , 2023. 

- Eran Malach. Auto-regressive next-token predictors are universal learners. In _Proceedings of the 41st International Conference on Machine Learning_ , volume 235 of _Proceedings of Machine Learning Research_ , pp. 34417–34431. PMLR, 21–27 Jul 2024. 

- Sean McLeish, Arpit Bansal, Alex Stein, Neel Jain, John Kirchenbauer, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Jonas Geiping, Avi Schwarzschild, and Tom Goldstein. Transformers can do arithmetic with the right embeddings, 2024. URL https://arxiv.org/abs/2405. 17399. 

- Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt. _Advances in Neural Information Processing Systems_ , 35:17359–17372, 2022. 

- William Merrill and Ashish Sabharwal. The expresssive power of transformers with chain of thought. _arXiv preprint arXiv:2310.07923_ , 2023. 

- Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. _arXiv preprint arXiv:1809.02789_ , 2018. 

14 

Under review as a conference paper at ICLR 2025 

- Arindam Mitra, Hamed Khanpour, Corby Rosset, and Ahmed Awadallah. Orca-math: Unlocking the potential of slms in grade school math, 2024. 

- Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, Ali Farhadi, Noah A. Smith, Pang Wei Koh, Amanpreet Singh, and Hannaneh Hajishirzi. Olmoe: Open mixture-of-experts language models, 2024. URL https://arxiv.org/abs/2409.02060. 

NuminaMath. How numinamath won the 1st aimo progress prize. https://huggingface.co/ blog/winning-aimo-progress-prize, 2024. 

OpenAI. Introducing openai o1. https://openai.com/o1/, 2024. 

- Bowen Pan, Yikang Shen, Haokun Liu, Mayank Mishra, Gaoyuan Zhang, Aude Oliva, Colin Raffel, and Rameswar Panda. Dense training, sparse inference: Rethinking training of mixture-of-experts language models. _arXiv preprint arXiv:2404.05567_ , 2024. 

- Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. Openwebmath: An open dataset of high-quality mathematical web text. _arXiv preprint arXiv:2310.06786_ , 2023. 

- Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? _arXiv preprint arXiv:2103.07191_ , 2021. 

- Guilherme Penedo, Hynek Kydl´ıcek,ˇ Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf, et al. The fineweb datasets: Decanting the web for the finest text data at scale. _arXiv preprint arXiv:2406.17557_ , 2024. 

- Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, et al. Rwkv: Reinventing rnns for the transformer era. _arXiv preprint arXiv:2305.13048_ , 2023. 

- Fabio Petroni, Tim Rocktaschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H Miller,¨ and Sebastian Riedel. Language models as knowledge bases? _arXiv preprint arXiv:1909.01066_ , 2019. 

- Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale. In _International conference on machine learning_ , pp. 18332–18346. PMLR, 2022. 

- Stephen Roller, Sainbayar Sukhbaatar, Jason Weston, et al. Hash layers for large sparse models. _Advances in Neural Information Processing Systems_ , 34:17555–17566, 2021. 

- Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. _Communications of the ACM_ , 64(9):99–106, 2021. 

- Clayton Sanford, Bahare Fatemi, Ethan Hall, Anton Tsitsulin, Mehran Kazemi, Jonathan Halcrow, Bryan Perozzi, and Vahab Mirrokni. Understanding transformer reasoning capabilities via graph algorithms. _arXiv preprint arXiv:2405.18512_ , 2024. 

- Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. _arXiv preprint arXiv:1904.09728_ , 2019. 

- David Saxton, Edward Grefenstette, Felix Hill, and Pushmeet Kohli. Analysing mathematical reasoning abilities of neural models. _arXiv preprint arXiv:1904.01557_ , 2019. 

- Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. _arXiv preprint arXiv:1701.06538_ , 2017. 

- Quan Shi, Michael Tang, Karthik Narasimhan, and Shunyu Yao. Can language models solve olympiad programming? _arXiv preprint arXiv:2404.10952_ , 2024. 

15 

Under review as a conference paper at ICLR 2025 

- Till Speicher, Aflah Mohammad Khan, Qinyuan Wu, Vedant Nanda, Soumi Das, Bishwamittra Ghosh, Krishna P Gummadi, and Evimaria Terzi. Understanding the mechanics and dynamics of memorisation in large language models: A case study with random strings. 2024. 

- Lena Strobl, William Merrill, Gail Weiss, David Chiang, and Dana Angluin. What formal languages can transformers express? a survey. _Transactions of the Association for Computational Linguistics_ , 12:543–561, 2024. 

- Alon Talmor and Jonathan Berant. The web as a knowledge-base for answering complex questions. _arXiv preprint arXiv:1803.06643_ , 2018. 

- Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. _arXiv preprint arXiv:1811.00937_ , 2018. 

- Shawn Tan, Yikang Shen, Rameswar Panda, and Aaron Courville. Scattered mixture-of-experts implementation. _arXiv preprint arXiv:2403.08245_ , 2024. 

- Kushal Tirumala, Aram Markosyan, Luke Zettlemoyer, and Armen Aghajanyan. Memorization without overfitting: Analyzing the training dynamics of large language models. _Advances in Neural Information Processing Systems_ , 35:38274–38290, 2022. 

- Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, and Igor Gitman. Openmathinstruct-1: A 1.8 million math instruction tuning dataset. _arXiv preprint arXiv:2402.10176_ , 2024. 

- Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, and Yulia Tsvetkov. Can language models solve graph problems in natural language? _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- Gail Weiss, Yoav Goldberg, and Eran Yahav. Thinking like transformers. In Marina Meila and Tong Zhang (eds.), _Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event_ , volume 139 of _Proceedings of Machine Learning Research_ , pp. 11080–11090. PMLR, 2021. URL http://proceedings.mlr.press/v139/ weiss21a.html. 

- Johannes Welbl, Nelson F Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. _arXiv preprint arXiv:1707.06209_ , 2017. 

- An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. _arXiv preprint arXiv:2407.10671_ , 2024. 

- Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. _arXiv preprint arXiv:1809.09600_ , 2018. 

- Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. _arXiv preprint arXiv:2309.12284_ , 2023. 

- Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen. Mammoth2: Scaling instructions from the web. _arXiv preprint arXiv:2405.03548_ , 2024. 

- Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? _arXiv preprint arXiv:1905.07830_ , 2019. 

- Yi Zhang, Arturs Backurs, Sebastien´ Bubeck, Ronen Eldan, Suriya Gunasekar, and Tal Wagner. Unveiling transformers with lego: a synthetic reasoning task. _arXiv preprint arXiv:2206.04301_ , 2022. 

16 

Under review as a conference paper at ICLR 2025 

- Yifan Zhang. Stackmathqa: A curated collection of 2 million mathematical questions and answers sourced from stack exchange, 2024. URL https://huggingface.co/datasets/ math-ai/StackMathQA. 

- Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. _arXiv preprint arXiv:2304.11277_ , 2023. 

- Zirui Zhao, Wee Sun Lee, and David Hsu. Large language models as commonsense knowledge for large-scale task planning. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Zexuan Zhong, Mengzhou Xia, Danqi Chen, and Mike Lewis. Lory: Fully differentiable mixtureof-experts for autoregressive language model pre-training. _arXiv preprint arXiv:2405.03133_ , 2024. 

- Hattie Zhou, Arwen Bradley, Etai Littwin, Noam Razin, Omid Saremi, Josh Susskind, Samy Bengio, and Preetum Nakkiran. What algorithms can transformers learn? a study in length generalization, 2023. URL https://arxiv.org/abs/2310.16028. 

- Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M Dai, Quoc V Le, James Laudon, et al. Mixture-of-experts with expert choice routing. _Advances in Neural Information Processing Systems_ , 35:7103–7114, 2022. 

- Yongchao Zhou, Uri Alon, Xinyun Chen, Xuezhi Wang, Rishabh Agarwal, and Denny Zhou. Transformers can achieve length generalization but not robustly. _arXiv preprint arXiv:2402.09371_ , 2024. 

- Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y Wu, Yukun Li, Huazuo Gao, Shirong Ma, et al. Deepseek-coder-v2: Breaking the barrier of closed-source models in code intelligence. _arXiv preprint arXiv:2406.11931_ , 2024. 

- Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. St-moe: Designing stable and transferable sparse expert models. _arXiv preprint arXiv:2202.08906_ , 2022. 

17 

Under review as a conference paper at ICLR 2025 

## A DETAILS ON THE PRE-TRAINING DATASETS 

In Section 5, we pretrain two collections of models, one on “natural language” and the other on “math”. Here, we give a precise breakdown of our training mixtures. We start with the “natural language” training mixture that totals 64B tokens: 

- 37B tokens from Fineweb-edu dedup (Penedo et al., 2024). 

- 14B tokens from Cosmopedia (Ben Allal et al., 2024). 

- 12B tokens from Wikipedia (we loop over Wikipedia 3 times). 

- 1B tokens from the training set of the downstream tasks we test on. We create 3 copies of each of these to increase their presence in the mixture. The presence of these datasets is pretty important as argued in Allen-Zhu & Li (2023) so that the model is familiar with the downstream tasks at test time. 

   - _∗_ ComplexWebQuestions training set (Talmor & Berant, 2018) 

   - _∗_ HotPotQA training set (Yang et al., 2018) 

   - _∗_ Natural Questions training set (Kwiatkowski et al., 2019) 

   - _∗_ TriviaQA training set (Joshi et al., 2017) 

   - _∗_ WebQuestions training set (Berant et al., 2013) 

   - _∗_ ARC-Easy and ARC-Challenge training sets (Clark et al., 2018) 

   - _∗_ Hellaswag training set (Zellers et al., 2019) 

   - _∗_ OpenBookQA training set (Mihaylov et al., 2018) 

   - _∗_ PIQA training set (Bisk et al., 2020) 

   - _∗_ SciQ training set (Welbl et al., 2017) 

   - _∗_ SIQA training set (Sap et al., 2019) 

   - _∗_ Winogrande training set (Sakaguchi et al., 2021) 

Our “math” training mixture that totals 66B tokens gathers: 

- 55B tokens from Proof-Pile 2 (Azerbayev et al., 2023) that contain AlgebraicStack (11B), OpenWebMath (Paster et al., 2023) and ArXiv (29B). 

- 2B tokens from OpenMathInstruct-1: we select the instances with a correct answer from the training set (Toshniwal et al., 2024) 

- 7B tokens from DeepMind math (Saxton et al., 2019) 

- 2B tokens from the following instruction-like datasets: 

   - _∗_ Math-Orca (Mitra et al., 2024) 

   - _∗_ TinyGSM (Liu et al., 2023) (we only select 1 million examples from there). 

   - _∗_ StackMathQA (Zhang, 2024) 

   - _∗_ MAmmoTH2 (Yue et al., 2024) (we only select the mathstackexchange subset). 

   - _∗_ NuminaMath-CoT (NuminaMath, 2024) (duplicated 3 times) 

   - _∗_ MetaMathQA (Yu et al., 2023) (duplicated 3 times) 

## B ADDITIONAL EXPERIMENTS 

In all our experiments in Section 5, we report the average accuracy performance obtained by our pre-trained models on respectively world knowledge, commonsense and math benchmarks. Here, we provide the results per task. In Subsection B.1, we display for each task, the downstream performance on a per parameter basis (similar to Figure 1) and in Subsection B.2, we plot for each task, the downstream performance on a per validation perplexity basis (similar to Figure 6). 

- B.1 DOWNSTREAM PERFORMANCE ON A PER PARAMETER BASIS 

18 

Under review as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0019-01.png)


**----- Start of picture text -----**<br>
COMPLEXWEBQUESTIONS HOTPOT-QA NQ<br>27 21 16<br>18<br>12<br>19 15<br>14 11 7<br>9<br>7 7 32<br>16M 49M 159M 569M 2.1B 16M 49M 159M 569M 2.1B 16M 49M 159M 569M 2.1B<br>Number of total parameters Number of total parameters Number of total parameters<br>TRIVIA-QA WEBQUESTIONS<br>38 30<br>25<br>27<br>16<br>18<br>119 7<br>16M 49M 159M 569M 2.1B 16M 49M 159M 569M 2.1B<br>Number of total parameters Number of total parameters<br>Dense transformer MoE (18M active parameters) MoE (58M active parameters) MoE (200M active parameters)<br>Figure 7: Downstream performance on the world knowledge tasks with respect to the total number of<br>parameters of the models.<br>ARCC ARCE COMMONSENSEQA<br>37 64 48<br>34<br>57 44<br>49 37<br>25<br>40 30<br>22<br>20 34 26<br>16M 49M 159M 569M 2.1B 16M 49M 159M 569M 2.1B 16M 49M 159M 569M 2.1B<br>Number of total parameters Number of total parameters Number of total parameters<br>HELLASWAG OPENBOOKQA PIQA<br>51 38 65<br>64<br>47<br>33<br>43 59<br>30 57<br>35 28<br>34 53<br>16M 49M 159M 569M 2.1B 16M 49M 159M 569M 2.1B 16M 49M 159M 569M 2.1B<br>Number of total parameters Number of total parameters Number of total parameters<br>SCIQ SIQA WINOGRANDE<br>89 48 55<br>87<br>85 45<br>54<br>78 42 52<br>39 52<br>69 37 51<br>16M 49M 159M 569M 2.1B 16M 49M 159M 569M 2.1B 16M 49M 159M 569M 2.1B<br>Number of total parameters Number of total parameters Number of total parameters<br>Dense transformer MoE (18M active parameters) MoE (58M active parameters) MoE (200M active parameters)<br>)%( )%( )%(<br>Accuracy Accuracy Accuracy<br>F1 F1 F1<br>)%( )%(<br>Accuracy Accuracy<br>F1 F1<br>)%( )%( )%(<br>Accuracy Accuracy Accuracy<br>)%( )%( )%(<br>Accuracy Accuracy Accuracy<br>)%( )%( )%(<br>Accuracy Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 7: Downstream performance on the world knowledge tasks with respect to the total number of parameters of the models. 

Figure 8: Downstream performance on the commonsense tasks with respect to the total number of parameters of the models. 

- B.2 DOWNSTREAM PERFORMANCE ON A PER VAL PERPLEXITY BASIS 

19 

Under review as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0020-01.png)


**----- Start of picture text -----**<br>
GSM-HARD GSM8K HENDRYCKS-MATH<br>45 49 30<br>34 35<br>19<br>16 19 10<br>6 5 5<br>2 1 2<br>16M 49M 159M 569M 2.1B 16M 49M 159M 569M 2.1B 16M 49M 159M 569M 2.1B<br>Number of total parameters Number of total parameters Number of total parameters<br>MINERVA SVAMP<br>28 67<br>54<br>18<br>34<br>9<br>5 14<br>2 4<br>16M 49M 159M 569M 2.1B 16M 49M 159M 569M 2.1B<br>Number of total parameters Number of total parameters<br>Dense transformer MoE (18M active parameters) MoE (58M active parameters) MoE (200M active parameters)<br>Figure 9: Downstream performance on the math benchmarks with respect to the total number of<br>parameters of the models.<br>COMPLEXWEBQUESTIONS HOTPOT-QA NQ<br>27 21 16<br>18<br>12<br>19 15<br>14 11 7<br>9<br>7 7 32<br>6.1 7.3 9.2 12.2 17.6 6.1 7.3 9.2 12.2 17.6 6.1 7.3 9.2 12.2 17.6<br>Validation perplexity Validation perplexity Validation perplexity<br>TRIVIA-QA WEBQUESTIONS<br>38 30<br>25<br>27<br>16<br>18<br>119 7<br>6.1 7.3 9.2 12.2 17.6 6.1 7.3 9.2 12.2 17.6<br>Validation perplexity Validation perplexity<br>Dense transformer MoE (18M active parameters) MoE (58M active parameters) MoE (200M active parameters)<br>)%( )%( )%(<br>Accuracy Accuracy Accuracy<br>)%( )%(<br>Accuracy Accuracy<br>)%( )%( )%(<br>Accuracy Accuracy Accuracy<br>F1 F1 F1<br>)%( )%(<br>Accuracy Accuracy<br>F1 F1<br>**----- End of picture text -----**<br>


Figure 9: Downstream performance on the math benchmarks with respect to the total number of parameters of the models. 

Figure 10: Downstream performance on the world knowledge tasks with respect to the validation perplexity. 

20 

Under review as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0021-01.png)


**----- Start of picture text -----**<br>
ARCC ARCE COMMONSENSEQA<br>37 64 48<br>34<br>57 44<br>49 37<br>25<br>40 30<br>22<br>20 34 26<br>6.1 7.3 9.2 12.2 17.6 6.1 7.3 9.2 12.2 17.6 6.1 7.3 9.2 12.2 17.6<br>Validation perplexity Validation perplexity Validation perplexity<br>HELLASWAG OPENBOOKQA PIQA<br>51 38 65<br>64<br>47<br>33<br>43 59<br>30 57<br>35 28<br>34 53<br>6.1 7.3 9.2 12.2 17.6 6.1 7.3 9.2 12.2 17.6 6.1 7.3 9.2 12.2 17.6<br>Validation perplexity Validation perplexity Validation perplexity<br>SCIQ SIQA WINOGRANDE<br>89 48 55<br>87<br>85 45<br>54<br>78 42 52<br>39 52<br>69 37 51<br>6.1 7.3 9.2 12.2 17.6 6.1 7.3 9.2 12.2 17.6 6.1 7.3 9.2 12.2 17.6<br>Validation perplexity Validation perplexity Validation perplexity<br>Dense transformer MoE (18M active parameters) MoE (58M active parameters) MoE (200M active parameters)<br>)%( )%( )%(<br>Accuracy Accuracy Accuracy<br>)%( )%( )%(<br>Accuracy Accuracy Accuracy<br>)%( )%( )%(<br>Accuracy Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 11: Performance on the commonsense tasks with respect to the validation perplexity. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0021-03.png)


**----- Start of picture text -----**<br>
GSM-HARD GSM8K MATH<br>45 49 30<br>34 35<br>19<br>16 19 10<br>6 5 5<br>2 1 2<br>3.2 3.5 3.9 4.5 5.5 3.2 3.5 3.9 4.5 5.5 3.2 3.5 3.9 4.5 5.5<br>Validation perplexity Validation perplexity Validation perplexity<br>MINERVA SVAMP<br>28 67<br>54<br>18<br>34<br>9<br>5 14<br>2 4<br>3.2 3.5 3.9 4.5 5.5 3.2 3.5 3.9 4.5 5.5<br>Validation perplexity Validation perplexity<br>Dense transformer MoE (18M active parameters) MoE (58M active parameters) MoE (200M active parameters)<br>)%( )%( )%(<br>Accuracy Accuracy Accuracy<br>)%( )%(<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 12: Downstream performance on the math benchmarks with respect to the validation perplexity. 

## C PROOFS 

- C.1 REASONING PROOFS 

**Definition C.1** (Set-disjointness task) **.** Set disjointness is the following task: given two inputs _A, B ∈{_ 0 _,_ 1 _}[r]_ for some _r ∈_ N, compute max _i AiBi_ . 

Set-disjointness can be thought of as follows: Alice and Bob are given sets _A_ and _B_ respectively. Their objective is to determine whether they have any overlapping items in their sets. 

21 

Under review as a conference paper at ICLR 2025 

**Lemma C.2** (Equivalence of set-disjointness and length-2 path) **.** _The set-disjointness task is equivalent to the length-2 path task._ 

_Proof._ ( = _⇒_ ): Given an instance of set-disjointness, we can encode it into a length-2 path problem. Denote every item _i_ as a vertex. Denote two extra vertices as _A_ , _B_ , corresponding to Alice and Bob. For every element _i_ that Alice has, draw an edge between _A_ and _i_ . For every element _i_ that Bob has, draw an edge between _B_ to _i_ . If and only if there are any overlapping elements, then there is a length-2 path from _A_ to _B_ . The number of elements because the number of vertices that do not belong to Alice or Bob. 

( _⇐_ = ): Consider an instance _G_ = ( _V, E_ ) _, s, d_ of length-2 path, where _s_ is the source vertex and _d_ is the sink vertex. For all vertices with an edge with _s_ , put this element into Alice’s set of elements. For all vertices with an edge with _d_ , put this element into Bobs’s set of elements. If and only if there is a length-2 path, then Alice and Bob’s sets are overlapping. Then, _r_ is the number of vertices. 

**Lemma C.3** (Communication complexity lower-bound on concatenated outputs) **.** _For some sequence length, fix two disjoint subsets A, B ⊂_ [ _N −_ 1] _, and consider a single-layer transformer f ∈ Transformer[N] m,H,_ 1 _[with][ O]_[(log] _[ N]_[)] _[-bit precision that solves set disjointness for any input][ X][where] XA is a function of Alice’s input a ∈{_ 0 _,_ 1 _}[r] , XB is a function of Bob’s input b ∈{_ 0 _,_ 1 _}[r] , and X_ [ _N_ ] _\_ ( _A∪B_ ) _is fixed regardless of a, b. Then, f has width satisfying mH_ = Ω( _r/_ log _N_ ) _._ 

_Proof._ By re-writing the following, the remainder of the proof from Sanford et al. (2024) still holds. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0022-06.png)


This is because we may still use the same definition for _Zh,S, Lh,S_ as in the proof. Hence, this concludes the proof. 

## C.1.1 PROOF OF THEOREM 3.2 

We restate the corollary. 

**Theorem C.4** (Theorem 3.2) **.** _For some input sequence G_ = ( _V, E_ ) _, fix two disjoint subsets A, B ⊂_ [ _N −_ 1] _, and consider a single-layer transformer f ∈ Transformer[N] m,H,_ 1 _,K[with][O]_[(log] _[ N]_[)] _[-bit] precision that solves length-2 path for any input X where XA is a function of edges with the source s, XB is a function of edges with the destination d. Then, f has width satisfying mH_ = Ω( _|V |/_ log _N_ ) _._ 

## _Proof._ The proof outline is as follows: 

1. Adapt Lemma 39 (Sanford et al., 2024) to support concatenation instead of addition from different attention heads. 

2. The lower bound with concatenation holds for length-2 path because set-disjointness and length-2 path are equivalent. 

3. Extend the result to sparse transformers. 

We complete the first step with Lemma C.3. We complete the second set due to Lemma C.2. It remains to show that a router function also yields the same lower bound. We show that Lemma 39 of Sanford et al. (2024) can be generalized to the case in which _ψ_ is applied according to a routing function. Specifically, consider a top-1 routing function _r_ : R _[m] →_ [ _K_ ], and _K_ element-wise functions _ψ_ 1 _, . . . , ψK_ : R _[m] →_ R. For shorthand, define: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0022-16.png)


which is the output of the attention head prior to applying the element-wise transformation. Next, we define _f_ ( _XN_ ) as the output when the router function _r_ is used to select _ψi_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0022-18.png)


Because the lower bound does not place any restrictions on the function _ψ_ and rather argues a communication-complexity lower bound due to information from _Y_ ( _XN_ ), the lower bound also holds for a routing function. 

22 

Under review as a conference paper at ICLR 2025 

## C.1.2 PROOF OF THEOREM 3.3 

We re-state Theorem 3.3 and give its proof. 

**Theorem C.5** (Theorem 3.3) **.** _For sequence length N , there exists f ∈ Transformer[N] m,_ 1 _,_ 1 _[with] O_ (log _N_ ) _-bit precision and width |V | that solves length-2 path for any input X._ 

_Proof._ Tokens are elements in _V_ = _V ∪{_ 0 _} × V ∪{_ 0 _}_ . The input is as follows: for vertex _i_ , if the source shares an edge with that vertex, then the _i_ ’th input value is ( _s, i_ ). Otherwise, it is ( _s,_ 0). The first _|V |_ tokens we see correspond to edges possibly shared with the source vertex. Then, the last _|V |_ input tokens correspond to edges possibly shared with the destination vertex and share the same format as the first _r_ tokens. In between, we can have arbitrary edges ( _u, v_ ). We define an embedding function where **e** _i_ is the _i_ ’th standard basis vector in dimension _r_ . 

_ϕ_ : _V →_ R _[|][V][ |]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0023-06.png)


Next, we define _Vh ∈_ R _[|][V][ |×|][V][ |]_ to be the identity matrix, and _Qh, Vh ∈_ R _[|][V][ |×|][V][ |]_ both to have 0 everywhere. Consequently, the attention matrix is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0023-08.png)


For any entry that exceeds _|V_ 1 _|_[, the correct answer is there is a length-2 path.][Hence, any thresholding] functionis representedwhichwithachieves _O_ (logthis _N_ )separationis sufficientsuffices.to separateProvidedbetweenthat _c_ ˜anyandrounding2˜ _c_ , then _O|V_ 1(log _|_[to] _N[c]_[˜] )[so] bits[that] are[it] sufficient. 

## C.2 MEMORIZATION PROOFS 

Assume that _K_ is a power of 2, and let _m_ = _m_ 0 + log( _K_ ). We associate each expert _i_ with a vector **v** _i ∈{±_ 1 _}_[log(] _[K]_[)] and choose **r** _i_ = ( _v_ 1 _, . . . , v_ log( _K_ ) _,_ 0 _, . . . ,_ 0) _∈_ R _[m]_ . **Lemma C.6.** _For some c >_ 0 _, and some input x ∼N_ (0 _, cIm_ ) _, the probability of routing to expert i is_ 1 _/K for all i._ 

_Proof._ By construction, we choose the expert as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0023-13.png)


The solution above must be the one whose signs match those of _x_ . Because the entries of _x_ are 0-mean, this implies that the probability of routing to any particular expert is 1 _/K_ . 

**Lemma C.7.** _Fix δ ∈_ (0 _,_ 1) _and some expert j. With probability at least_ 1 _− δ, the number of examples routed to j is at most_ 2 _N/K, given N ≥[K]_[2][ ln] 2[(][1] _[/][δ]_[)] _examples from N_ (0 _, cIm_ ) _._ 

_Proof._ The proof relies on a Hoeffding bound: for bounded independent random variables _X_ 1 _, . . . , XN_ such that _a ≤ Xi ≤ b ∀i_ , and _SN_ =[�] _[N] i_ =1 _[X][i]_[, we have:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0023-17.png)


In our case, let _Xi_ = **1** _{_ sample _i_ is routed to expert _j}_ , for some fixed _j ∈_ [ _K_ ]. Note that P[ _Xi_ = 1] = 1 _/K_ due to Lemma C.6 and as such _E_ [ _SN_ ] = _N/K_ . Further, _Xi ∈_ [0 _,_ 1]. Hence we obtain: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0023-19.png)


23 

Under review as a conference paper at ICLR 2025 

In order to upper-bound this probability with _δ_ , we obtain: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0024-02.png)


**Lemma C.8.** _Fix δ ∈_ (0 _,_ 1) _, K ∈_ N _and the routing function as described in Lemma C.6. Given n ≥[K]_[2][ ln] 2[(][1] _[/][δ]_[)] _samples with embedding dimension m. For every expert i, with probability at least_ 1 _− δ, there exists an MLP of width O_[˜] ( _n/mK_ ) _that correctly classifies the examples routed to this expert._ 

> _Proof._ We begin with a result from Daniely (2020). Consider _h_ , a depth-2 network with _q_ neurons, an activation function _ϕ_ : R _→_ R, which is _O_ (1)-Lipschitz, piecewise twice-differentiable, and satisfies E _X∼N_ (0 _,_ 1)[ _ϕ[′]_ ( _X_ )] = 0, weights **a** _i ∈{±_ 1 _}_ such that[�] _[q] i_ =1 _[a][i]_[=] _[ O]_[(] _[√] q_ ). Such a network, defined by _W_ and _a_ , computes the function: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0024-05.png)


For our proof we consider _ϕ_ to be the absolute value function, and later show how to obtain an MLP with ReLU activations. 

Consider the set of samples (( _x_ 1 _, y_ 1) _, . . . ,_ ( _xN , yN_ )), where _xi ∼N_ (0 _, IM_ ) and _yi ∈{±_ 1 _}_ is a Rademacher random variable. The objective is to find a _W[∗] , a[∗]_ such that _yihW ∗,a∗_ ( _xi_ ) _>_ 0 _∀i_ . Daniely (2020) gives the following result: for _N ≤_ log _M_[4] ( _qM_ )[and] _[q][≥]_[log][4][(] _[M]_[)][,][with][probability] 1 _− o_ (1) there exists _h_ such that, for every _i ∈_ [ _N_ ], _yih_ ( _x_ ) = Ω(ln( _M_ )). 

We build on this result as follows: first, we assume that the attention matrix performs averaging of the input sequence, such that _xi_ = _N_ 1 � _Nj_ =1[(] _[X][i]_[)] _[j]_[where][where] _[X][i]_[is][the] _[i]_[-th][input][sequence][to] memorize. Assuming the inputs are also Gaussian, then so is the average. Hence, _xi ∈_ R _[m]_ and _xi ∼N_ (0 _, cIm_ ) for _c_ = _L_[1][.] 

Next, we fix the number of experts to be _K_ . We fix a router function _r_ : R[log] _[ K] →_ [ _K_ ] such that, on average, each expert must memorize _n/K_ sequences, and with high probability, each expert must memorize at most 2 _n/K_ sequences. We use the construction from Lemma C.6 where the router only acts on the first log _K_ entries of each vector. Hence, we have _K_ MLPs, each tasked with memorizing at most[2] _[n]_[We use the network construction from][ Daniely][ (][2020][) to only the] _K_[input-output values.] last _m −_ log _K ≥ m/_ 2 coordinates, which are not used by the router, so that their distribution is independent and Gaussian. 

_K_ ( _m−_ log _K_ ) _q K mq_ Hence, with _n ≤_ 2 log[4] ( _m−_ log _K_ )) _[≤]_ 4 log[4] ( _m−_ log _K_ ))[and] _[q][≥]_[log][4][(] _[m][−]_[log] _[ K]_[)][,][with][high] probability, _yihj_ ( _xi_ ) = Ω(ln( _m_ )), where _hj_ is the expert to which _xi_ is routed. It follows that width 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0024-11.png)


is sufficient for each individual expert, and we assume that _mK_ 4 _n[≥]_[1][.] 

In order to obtain a MLP with a ReLU activation defined as _σ_ ( _x_ ) = max _{_ 0 _, x}_ , we use the fact that _|x|_ = _σ_ ( _x_ ) + _σ_ ( _−x_ ). This is because absolute value is a valid activation function in Daniely (2020), but ReLU is not. However, doubling the width of each MLP is sufficient to obtain MLPs with ReLU activations instead, i.e.: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0024-14.png)


24 

Under review as a conference paper at ICLR 2025 

This is done by taking a solution which is of form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0025-02.png)


and converting it to the form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0025-04.png)


The router has _K_ vectors each of dimension _m_ . Consequently, we need _O_ ( _Km_ ) parameters to store the router. Each expert has width[8] _[n]_[ lo][g][4] _mK_[(] _[m][−]_[lo][g] _[ K]_[)] = _O_[˜] � _mKn_ � and _O_[˜] � _Kn_ � parameters. 

**Corollary C.9.** _Let δ ∈_ (0 _,_ 1) _, and fix K >_ 1 _. For n ≥[K]_[2][ ln] 2[(] _[K][/][δ]_[)] _, with probability at least_ 1 _− δ, there exists a sparse transformer s with K experts such that yis_ ( _xi_ ) = Ω(ln _m_ ) _. It has O_[˜] ( _n_ + _Km_ ) _parameters and O_[˜] ( _n/K_ + _Km_ ) _active parameters._ 

_Proof._ For each expert, we apply the result from Lemma C.8 with _δ[′]_ = _δ/K_ . Hence, for every expert, with probability at most _δ/K_ , there does not exist an MLP of width _O_[˜] ( _n/mK_ ) which memorizes its samples. By a union bound, this implies that with probability at most _δ_ , at least one of the experts is not able to memorize its samples. Hence, with probability at least 1 _− δ_ , all of the experts are able to perform memorization on their given samples. In total, this implies we will use _O_[˜] ( _n_ + _K_ ) parameters to store the entire mixture of expert network. The number of active parameters is _O_[˜] ( _n/K_ + _K_ ). 

**Lemma C.10** (Bound on _ℓ_ 2-norm of vector from _N_ (0 _, cIm_ ) ) **.** _Let x ∼N_ (0 _, cIm_ ) _for some c >_ 0 _, and let δ ∈_ (0 _,_ 1) _. Then, with probability at least_ 1 _− δ, there exists a constant cδ,m >_ 0 _, such that_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0025-09.png)


> _Proof._ Each component _xi_ of _x_ is distributed as _xi ∼N_ (0 _, c_ ). We can express _xi_ as _xi_ = _[√] c zi_ , where _zi ∼N_ (0 _,_ 1). Then, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0025-11.png)


The sum[�] _[m] i_ =1 _[z] i_[2][follows a chi-squared distribution with] _[ m]_[ degrees of freedom.][By the Laurent-] Massart theorem Laurent & Massart (2000), for all _t >_ 0, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0025-13.png)


Multiplying both sides inside the probability by _c_ , we obtain 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0025-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0025-16.png)


25 

Under review as a conference paper at ICLR 2025 

**Lemma C.11** (Bounded bit complexity required) **.** _With high probability, O_[˜] (1) _bits are required to store each weight in the network,O_ ˜� _Kn_[+] _[ K]_ � _active bits are required. and O_[˜] ( _n_ + _K_ ) _bits are required to store the entire network._ 

_Proof._ We show that the learning result from Daniely (2020) gives a network with bounded bit complexity. Hence, it suffices to bound the bit complexity of the initial weights and the bit complexity of the gradient update. First, we begin with _h_ ( _x_ ) = ~~�~~ 1 _q_ � _qi_ =1 _[a][i][σ]_[(] _[⟨][w][i][, x][⟩]_[)][.][The objective is to show] 

> there existscreating bins for the weights, and replace eachthat _∥wi∥∞ h_[˜] _≤_ ( _xM_ ) with bounded bit complexity which satisfiesfor some _M >_ 0. We then obtain that _wi_ with _w_ ˜ _∥i w_ where _i − yiwh_[˜] _∥_ ˜( _iwx∥_ 2 _ii_ ) _−≤ >w[√]_ 0˜ _i ∀m∥∞iε_ ..Suppose we begin by _≤_ In addition, we state _ε_ . Further, assume that with probability at least 1 _− δ_ , _∥x∥_ 2 _≤ cδ,m_ due to Lemma C.10. Using this, we have that: 

˜ ˜ _|σ_ ( _⟨wi, x⟩_ ) _− σ_ ( _⟨wi, x⟩_ ) _| ≤|⟨wi − wi, x⟩|_ ( _σ_ is 1-Lipschitz) _≤∥wi − w_ ˜ _i∥_ 2 _∥x∥_ 2 (Cauchy-Schwarz) _≤[√] mεcδ,m_ w.p. _≥_ 1 _− δ_ (Lemma C.10) _._ 

We apply this result to obtain that with probability at least 1 _− δ_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0026-06.png)


Given that _h_ ( _x_ ) _≥ O_ (1), then for some arbitrarily small constant (we use[1] 4[)][we][require:] 14 _[≤] √q[√] mεcδ,m_ , or equivalently, _ε ≤_ 4 _cδ,m_ 1 ~~_[√]_~~ _qm_[.][Because] _[ h]_[(] _[x]_[)] _[≥][O]_[(1)][, we use this separation of][1] 4 to show that _h_[˜] ( _x_ ) remains the same sign as _h_ ( _x_ ) (however this constant can be replaced with an arbitrarily small constant so as to satisfy the requirement). 

Consider _wi_[(0)] to be the initialization of _wi_ prior to the gradient step. Then, because, _M_ = _∥wi∥∞_ , we have that _M_ = max _i_ � _wi_[(0)] _− η[∂h] ∂w_[(] _[x] i_[)] � = max _i_ � _wi_[(0)] _−[n]_[ lo] _m_[g] _[ m] ∂h∂w_ ( _xi_ ) �. Assuming that _wi_[(0)] is initialized with bounded _ℓ∞_ norm of 1, then we obtain that _M ≤_ 1 + _[n] m_[ lo] ~~_[√]_~~[g] _[ m] q[c][δ,m]_[.][This is due to] Lemma C.10 and that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0026-09.png)


Using this, with high probability, we require log� 2 _Mε_ � = log�8 _Mcδ[√] qm_ � bits (by replacing _ε_ ). Hence, with high probability, we require the number of bits to be at most: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning_images/iclr_2025_Mixture_of_Parrots_Experts_improve_memorization_more_than_reasoning.pdf-0026-11.png)


We restate Theorem 3.6. 

**Theorem C.12** (Lower bound for dense model) **.** _Given the same task as above, a dense Transformer requires_ Ω([˜] _n_ ) _parameters to solve the memorization task._ 

_Proof of Theorem 3.6._ Let _c_ be the number of bits used for encoding each parameters (and we assume that _c_ is logarithmic in the problem parameters). Denote by _H_ the class of all transformers with _W_ parameters and _c_ bits per parameters. Since _H_ is a finite class, where each function in the class can be encoded with _cW_ bits, we have _|H| ≤_ 2 _[cW]_ . Let _X_[1] _, . . . , X[N] ∈_ R _[n][×][d]_ be the _N_ input points. Assume a _H_ can solve the memorization task. Then, for every choice of _y_ 1 _, . . . , yN ∈{±_ 1 _}_ , there exists a transformer _f ∈H_ s.t. _f_ ( _Xi_ ) = _yi_ for all _i ∈_ [ _N_ ]. There are 2 _[N]_ possible assignments for _y_ 1 _, . . . yN_ and therefore there are at least 2 _[N]_ different functions in _H_ . So, we get 2 _[N] ≤|H| ≤_ 2 _[cW]_ and therefore _W ≥ N/c_ . 

26 

Under review as a conference paper at ICLR 2025 

**Lemma C.13** (Active parameter comparison between dense and sparse transformers) **.** _There exist cases in which the amount of active parameters required to perform memorization is less for a sparse transformer than a dense transformer._ 

_Proof._ As shown, a dense transformer requires Ω([˜] _n_ ) parameters (and active parameters) to perform memorization of _N_ sequences. In contrast, for _n_ sufficiently large and fixed _K >_ 1, it holds that _[n][<][n]_[, which shows that the number of active parameters required is less for a sparse] _K_[+] _[ K]_ transformer. 

27 

