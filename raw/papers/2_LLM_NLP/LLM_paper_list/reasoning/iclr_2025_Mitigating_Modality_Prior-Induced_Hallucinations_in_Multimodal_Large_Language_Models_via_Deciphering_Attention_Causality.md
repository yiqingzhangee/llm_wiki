Published as a conference paper at ICLR 2025 

# - MITIGATING MODALITY PRIOR-INDUCED HALLUCI - NATIONS IN MULTIMODAL LARGE LANGUAGE MOD ELS VIA DECIPHERING ATTENTION CAUSALITY 

**Guanyu Zhou[1] Yibo Yan[1,2] Xin Zou[1] Kun Wang[3] Aiwei Liu[1,4] Xuming Hu[1,2,]** _[∗]_ 

1The Hong Kong University of Science and Technology (Guangzhou) 

2The Hong Kong University of Science and Technology 

3Nanyang Technological University, 4Tsinghua University guanyuzhou.ai@gmail.com, xuminghu@hkust-gz.edu.cn 

## ABSTRACT 

Multimodal Large Language Models (MLLMs) have emerged as a central focus in both industry and academia, but often suffer from biases introduced by visual and language priors, which can lead to multimodal hallucination. These biases arise from the visual encoder and the Large Language Model (LLM) backbone, affecting the attention mechanism responsible for aligning multimodal inputs. Existing decoding-based mitigation methods focus on statistical correlations and _overlook the causal relationships between attention mechanisms and model output_ , limiting their effectiveness in addressing these biases. To tackle this issue, we propose a causal inference framework termed **CAUSALMM** that applies structural **causal** modeling to **M** LL **M** s, treating modality priors as a confounder between attention mechanisms and output. Specifically, by employing back-door adjustment and counterfactual reasoning at both the visual and language attention levels, our method mitigates the negative effects of modality priors and enhances the alignment of MLLM’s inputs and outputs, with a maximum score improvement of **65.3%** on 6 VLind-Bench indicators and **164** points on MME Benchmark compared to conventional methods. Extensive experiments validate the effectiveness of our approach while being a plug-and-play solution. Our code is available at: https://github.com/The-Martyr/CausalMM. 

## 1 INTRODUCTION 

Recent research on Multimodal Large Language Models (MLLMs) has achieved great progress in diverse applications (Yin et al., 2023; Jin et al., 2024; Yan et al., 2024; Zou et al., 2024b), particularly due to their reliance on Transformer models (Vaswani, 2017), where performance is driven by the attention mechanism (Hassanin et al., 2024). In particular, such a mechanism enables the model to assign weights to input information, such as images and text, guiding the generation of outputs. However, the inherent bias in the initial parameters of the model, namely the **modality priors** , can negatively impact output quality via the attention mechanism (Tong et al., 2024a; Zhao et al., 2024; Lee et al., 2024; Chen et al., 2024). In widely used MLLM architectures, attention that most significantly influences output can be divided into two components: visual en- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0001-11.png)


**----- Start of picture text -----**<br>
Is there a<br>person in  logits(𝑦|𝑥, 𝑣)<br>the image?<br>(1 + α)logits(𝑦|𝑥, 𝑣)<br>−α logits(𝑦|𝑥, 𝑣′)<br>LLM<br>logits(𝑦|𝑥, 𝑣′)<br>(a) Visual Contrastive Decoding<br>Causal<br>Inference<br>Deciphering<br>Attention<br>Causality<br>(b) CausalMM (Ours)<br>**----- End of picture text -----**<br>


**Figure 1:** The comparison of conventional hallucination mitigation paradigm ( _e.g._ , VCD) and our proposed CAUSALMM. 

> _∗_ Corresponding author. 

1 

Published as a conference paper at ICLR 2025 

coder attention and Large Language Model (LLM) backbone attention (Liu et al., 2024c). The parametric knowledge of the visual encoder ( _i.e._ , **visual priors** ) affects the alignment of multimodal information by affecting the visual encoder’s attention (Tong et al., 2024a;b). Similarly, the knowledge embedded in the LLM’s parameters, referred to as **language priors** , may compromise the model’s fidelity to multimodal inputs through attention (Lee et al., 2024). These biases, stemming from the visual encoder and the MLLM’s over-reliance on language priors, may lead to issues such as multimodal hallucinations, ultimately degrading model performance (Yang et al., 2023). Several approaches have been proposed to enhance model output without modifying the model weights (Leng et al., 2024; Huang et al., 2024; Zou et al., 2024a). However, as illustrated in Figure 1 (a), existing decoding strategies primarily rely on statistical correlations and predetermined conclusions from posterior analysis to optimize outputs, _without systematically studying the causal relationship between visual attention, language attention, modality priors, and model output_ . In this context, the attention mechanism adjusts weights solely based on parameter knowledge, which limits the model’s ability to comprehend underlying dependencies in the reasoning process, exacerbates bias, leading to problems such as multimodal hallucinations. 

Modality priors are one of the confounding factors in the causal path of MLLM. We introduce a causal reasoning framework **CAUSALMM** , which can help us better capture the causal impact of effective attention on MLLM output in the presence of these confounding factors, thereby improving the performance of multimodal tasks, as shown in Figure 1 (b). Specifically, we construct a structural causal model (Pearl, 2009) for MLLM, and use intervention and counterfactual reasoning methods under the back-door adjustment paradigm to derive the causal effects of visual and language attention on the model output despite the confounding effect of modal priors. The CAUSALMM method is based on counterfactual reasoning at the visual and language attention levels, which ensures that the model output is more consistent with the multimodal input, thereby mitigating the negative impact of modal priors on performance. Experimental results show that CAUSALMM significantly reduces modal prior bias and improves performance on different tasks, improving **143.7** points on 6 indicators of VLind-Bench, **164** points on the MME Benchmark, and an average improvement of **5.37%** on the three benchmarks of POPE. 

Our key contributions can be summarized as follows: ❶ We have constructed a structural causal framework called **CAUSALMM** flexible for any MLLM, exploring the issues of visual and language priors within the framework. ❷ We apply counterfactual reasoning at the levels of visual and language attention, making the output more aligned with multimodal inputs. ❸ Through comprehensive experiments, we have demonstrated the superior performance of our method in alleviating MLLM hallucinations. In addition, our framework is plug-and-play, and can be integrated with other training-free methods for further improvement. 

## 2 RELATED WORKS 

**Multimodal Large Language Models.** In recent years, MLLMs have seen significant advancements (Yin et al., 2023; Jin et al., 2024; Huo et al., 2024; Yan & Lee, 2024). Notable works include VITA (Fu et al., 2024b), the first open-source MLLM capable of processing video, image, text, and audio, demonstrating robust performance across various benchmarks. Cambrian-1 (Tong et al., 2024a) is a family of MLLMs designed with a vision-centric approach, achieving state-of-theart performance and providing comprehensive resources for instruction-tuned MLLMs. Additionally, research on training-free reasoning stage improvements, such as VCD (Leng et al., 2024) and OPERA (Huang et al., 2024), has focused on leveraging human experience to enhance model performance without additional training (Li et al., 2023b; Zheng et al., 2024). In this work, we manage to apply causal reasoning (Pearl, 2009) to make the MLLM automatically optimize the output. 

**Causal Inference in Multimodal Learning.** The field of causal inference has seen significant advancements (Pearl, 2009; Xu et al., 2020; Cheng et al., 2023; Gong et al., 2022; Fang & Liang, 2024; Wu et al., 2022), particularly in the context of LLMs and vision systems (Zhang et al., 2023a; Rao et al., 2021). Researchers have explored the integration of causal reasoning to enhance the interpretability and robustness of these models (Xu et al., 2020; Zou et al., 2023). For instance, LLMs have been shown to generate accurate causal arguments across various tasks, surpassing traditional methods (Kıcıman et al., 2023). A comprehensive survey has highlighted the potential of causal inference frameworks to improve reasoning capacity, fairness, and multimodality in LLMs (Liu et al., 2024d). Additionally, recent work showcased the use of LLM-guided discovery to significantly improve causal ordering accuracy (Vashishtha et al., 2023). Different from previous attempts, we tend to use causal reasoning to balance the visual priors and language priors of the model output. 

2 

Published as a conference paper at ICLR 2025 

**Modality Priors.** Research on modality priors in MLLMs has seen significant advancements (Tong et al., 2024a; Peng et al., 2023; Lukics & Luk´acs, 2022; Gema et al., 2024). Studies focused on overcoming language priors by integrating visual modules, enhancing the impact of visual content on model outputs. For instance, (Zhao et al., 2022) proposed a method to improve visual content in Visual Question Answering (VQA) tasks, which proved effective across multiple datasets. Additionally, benchmarks like VLind-Bench (Lee et al., 2024) have been developed to measure language priors in MLLMs, revealing a strong reliance on textual patterns. On the other hand, visual priors have been addressed by augmenting off-the-shelf LLMs to support multimodal inputs and outputs through cost-effective training strategies (Zhang et al., 2024). 

## 3 METHODOLOGY 

In this section, we construct a structural causal model of MLLM and generate different counterfactual attentions through intervention for counterfactual reasoning based on the back-door criterion. 

## 3.1 STRUCTURAL CAUSAL MODEL 

We construct a structural causal model (SCM) to describe the relationships among various components of a MLLM (Yang et al., 2021; Pawlowski et al., 2020). In particular, our SCM captures the interactions between the visual and language modalities by modeling causal dependencies among input image ( _I_ ), visual attention ( _Ai_ ), visual token embeddings ( _Ti_ ), language token embeddings ( _Tt_ ), language priors ( _Pl_ ), visual priors ( _Pv_ ), MLLM attention ( _At_ ), and model output ( _O_ ). 

The causal graph is formulated as follows: 

- _I → Ai_ : The image input _I_ influences the visual attention layer _Ai_ . 

- _I → Ti_ : The image input _I_ directly affects the visual token embeddings _Ti_ . 

- _Pv → Ai_ : Visual priors _Pv_ contribute to the attention in the visual attention module. 

- _Pv → Ti_ : Visual priors _Pv_ also influence the formation of visual token embeddings _Ti_ . 

- _Ai → Ti_ : Visual attention _Ai_ impacts the encoding of visual tokens. 

- _Ti → O_ : Visual tokens _Ti_ contribute directly to the model’s output. 

- _Tt → At_ : Language token embeddings _Tt_ influence the MLLM’s attention _At_ . 

- _Tt → O_ : Language token embeddings _Tt_ directly impact the final output. 

- _Pl → At_ : Language priors _Pl_ inform the MLLM’s attention mechanism _At_ . 

- _Pl → O_ : Language priors _Pl_ directly affect the model output _O_ . 

- _At → O_ : LLM attention _At_ shapes the final output _O_ . 

In this causal graph, both visual priors ( _Pv_ ) and language priors ( _Pl_ ) serve as confounding factors, influencing the attention layers and embedding representations in both modalities. These priors are mixed into the model and can lead to biased outputs. Our goal is to quantify the causal effect of visual attention ( _Ai_ ) and language attention ( _At_ ) on the model output ( _O_ ), while accounting for these confounding effects through intervention and counterfactual reasoning. 

## 3.2 INTERVENTION ON MULTIMODAL ATTENTIONS 

We perform specific interventions on the attention layers of both the visual and language components to investigate their causal effects on the model’s output. These interventions modify the attention weights to generate counterfactual outputs, allowing us to isolate the impact of each modality. 

For visual attention, we intervene by replacing the original attention map _Ai_ with a counterfactual state _A[∗] i_[,][expressed as] _[ do]_[(] _[A][i]_[=] _[A][∗] i_[)][.][The counterfactual state] _[ A][∗] i_[can take various forms,][such as] random attention weights, uniform distributions, reversed scores, or shuffled attention maps (Rao et al., 2021). Each configuration reveals different aspects of how visual attention influences the output, independent of other factors like the image _I_ and visual processing _Pv_ . 

Similarly, we intervene in the language attention by applying _do_ ( _At_ = _A[∗] t_[)][,][where] _[A][∗] t_[represents] alternative attention states that allow us to explore the impact of the language attention module on the final output, free from the influences of _Tt_ , _Ti_ , and _Pl_ . 

The counterfactual attention states are specified as follows: 

3 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Tt Tt Tt I Visual Input<br>O Output<br>Pl Pl Pl<br>Pv Visual Prior<br>At O At O At O Pl Language Prior<br>I Ai I Ai I Ai Ai Visual Attention<br>Pv Pv Pv At LLM Attention<br>Ti Visual Token Embedding<br>Ti Ti Ti Tt Text Token Embedding<br>(I) Vision Only (II) Language Only (III) Multimodal Collaboration<br>Causal Path Truncated Affected Path Truncated + Path<br>**----- End of picture text -----**<br>


**Figure 2: Causal diagram of counterfactual reasoning.** ❶ In vision-only counterfactual reasoning, we only intervene in visual attention ( _i.e._ , the attention of the visual encoder). ❷ In language-only counterfactual reasoning, we only intervene in the multi-head self-attention of LLM. ❸ In multimodal collaborative counterfactual reasoning, we intervene in both visual and language attention at the same time and obtain the sum of their collaborative causal effects. 

1. _**Random Attention**_ : Replace the original attention scores with random values drawn from a uniform distribution. For the visual encoder, attention scores _Ai_ ( _h, w_ ) at spatial locations ( _h, w_ ) are replaced as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0004-04.png)


where _U_ (0 _,_ 1) is a random variable drawn from a uniform distribution, _σ_ represents the scaling factor for attention, and _αv_ denotes the normalization parameter. Similarly, for the language model, the random attention values _At_ ( _n_ ) over tokens _n_ are given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0004-06.png)


where _β_ is the language attention scaling factor and _αl_ is the language normalization term. 

2. _**Uniform Attention**_ : Assign a constant value to all attention scores. For the visual encoder, the attention at location ( _h, w_ ) is replaced by the average value: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0004-09.png)


where _H_ and _W_ represent the height and width of attention map, and _ϵ_ is a small perturbation added to avoid exact uniformity. For the language model, the attention over _N_ tokens is distributed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0004-11.png)


where _δ_ is a small constant ensuring numerical stability. 

3. _**Reversed Attention**_ : Invert the attention map by subtracting each attention score from the maximum value of the map. For the visual encoder: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0004-14.png)


where _λ_ is an offset parameter to control the inversion. For the language model: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0004-16.png)


where _ζ_ is the inversion factor for language attention. 

4. _**Shuffled Attention**_ : Randomly permute the attention scores across spatial locations for the visual encoder. The new attention map _A[′] i_[is created by permuting the original scores] _[ A][i]_[:] 

_A[′] i_[(] _[h, w]_[) =] _[ A][i]_[(] _[π]_[(] _[h]_[)] _[, π]_[(] _[w]_[))] _[,]_ (7) 

where _π_ ( _h_ ) and _π_ ( _w_ ) are random permutations of the height and width indices. This intervention is specific to the visual encoder and does not apply to the language model, as token order is significant in language processing. 

By conducting these interventions, we can observe the independent contributions of both visual and language attention to the model’s output, controlling for confounding factors such as the image _I_ , the tokens _Tt_ , and the model’s intermediate representations _Pv_ and _Pl_ . 

4 

Published as a conference paper at ICLR 2025 

## 3.3 COUNTERFACTUAL REASONING 

To formalize the impact of counterfactual interventions on the model output, we perform counterfactual reasoning based on the back-door adjustment principle (Pearl, 2009; Li et al., 2023a; Adib et al., 2020; Zhang et al., 2023b). The back-door criterion ensures that we properly account for confounding factors ( _I_ , _Pv_ , _Pl_ ) when estimating the causal effect of attention mechanisms. Under the framework of back-door adjustment, we are able to effectively obtain the causal effects of other variables under the influence of the confounding factor of modal priors. The specific proof can be found in Sec. A.1. To measure the causal effect of the attention mechanism, we use counterfactual reasoning to simulate the case of attention failure. For the visual attention ( _Ai_ ): 

_Peffect_ ~~_V_~~[=] _[ E] Ai∼A_ ˜ _i_[[] _[P]_[(] _[O][|][A][i]_[=] **[ A]** _[i][, I]_[=] **[ I]** _[, P][v]_[=] **[ P]** _[v]_[)] _[ −][P]_[(] _[O][|]_[do][(] _[A][i]_[=] **[ a]** _[i]_[)] _[, I]_[=] **[ I]** _[, P][v]_[=] **[ P]** _[v]_[)]] _[ .]_ Here, _Peffect V_[represents the causal effect of the visual attention mechanism on the model output] _O_ . The term **A** _i_ denotes the observed visual attention, whereas **a** _i_ represents the intervention applied to the visual attention. For vision-only: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0005-04.png)


In this equation, _tnext,_ v indicates the index of the next token chosen based solely on visual attention. The variable _ℓi_ stands for the original logits of the _i_ -th token, and _ℓ_ cf ~~v~~ _,i_[is the counterfactual logit] derived from the visual modality. _γ_ represents the degree of confidence in the treatment effect. ”j” iterates over all tokens in the denominator (to compute the softmax normalization). For the LLM attention ( _At_ ): 

_Peffect_ ~~_L_~~[=] _[ E] At∼A_ ˜ _t_[[] _[P]_[(] _[O][|][A][t]_[=] **[ A]** _[t][, T][t]_[=] **[ T]** _[t][, P][l]_[=] **[ P]** _[l]_[)] _[ −][P]_[(] _[O][|]_[do][(] _[A][t]_[=] **[ a]** _[t]_[)] _[, T][t]_[=] **[ T]** _[t][, P][l]_[=] **[ P]** _[l]_[)]] _[ ,]_ Where _Peffect L_[denotes the causal effect of the language model attention on the output] _[ O]_[.][The no-] tation **A** _t_ is the observed language model attention, and **a** _t_ is the intervention applied to the language model attention. For language-only: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0005-07.png)


This equation describes the selection of the next token _tnext,_ l based purely on language attention. Here, _ℓi_ is the original logits of the _i_ -th token, and _ℓ_ cf ~~l~~ _,i_[is the counterfactual logit derived from the] language modality. In a multimodal setting, the combined causal effect is given by: 

_Peffect_ ~~_M_~~[=] _[ E] Ai,At∼A_ ˜ _i,A_[˜] _t_[[] _[P]_[(] _[O][|][A][i]_[=] **[ A]** _[i][, A][t]_[=] **[ A]** _[t][, I]_[=] **[ I]** _[, T][t]_[=] **[ T]** _[t][, P][v]_[=] **[ P]** _[v][, P][l]_[=] **[ P]** _[l]_[)]] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0005-10.png)


Where _Peffect M_[represents the combined causal effect of both visual and language attention mecha-] nisms on the output _O_ . When integrating visual and language modalities enhanced by counterfactual reasoning, the final token selection is determined by: 

_e_[max(] _[ℓ][i]_[+] _[γ]_[((] _[ℓ][i][−][ℓ]_[cf] v _,i_[)+(] _[ℓ] i[−][ℓ]_ cf ~~l~~ _,i_[))] _[−]_[log(] _[ϵ]_[)] _[−]_[max] _j[ℓ] j[,][−∞]_[)] _tnext_ = arg max _. i_ � � _j[e]_[max(] _[ℓ][j]_[+] _[γ]_[((] _[ℓ][j][−][ℓ]_[cf] v _,j_[)+(] _[ℓ] j[−][ℓ]_ cf l _,j_[))] _[−]_[log(] _[ϵ]_[)] _[−]_[max] _k[ℓ] k[,][−∞]_[)] � 

This equation defines the final token selection _tnext_ by integrating the effects of both visual and language attention mechanisms, thereby mitigating the negative influence of priors in both modalities and enabling more robust decoding strategies. In all cases we use direct sampling. 

## 4 EXPERIMENTS 

In this section, we verify the effectiveness of the CAUSALMM on different benchmarks and implement ablation for different categories of counterfactual attention and number of intervention layers. The case study and gpt-aided-evaluation are in 4.4. 

## 4.1 EXPERIMENTAL SETUP 

## 4.1.1 BENCHMARKS 

**VLind-Bench.** VLind-Bench (Lee et al., 2024) is a benchmark designed to measure language priors in MLLMs. It disentangles language priors from commonsense knowledge (CK), visual perception (VP), and commonsense biases (CB). There is significant reliance on language priors across models, and the Pipeline Score (SLP) offers insights beyond task-level evaluation. 

5 

Published as a conference paper at ICLR 2025 

**POPE.** POPE (Polling-based Object Probing Evaluation) (Li et al., 2023c) is a benchmark for evaluating MLLMs in accurately determining the presence or absence of specific objects in images, assessing object-level hallucination. The framework utilizes Y/N questions derived from object annotations. Evaluation metrics include standard binary classification measures — accuracy, precision, recall, and F1 score — offering a clear quantitative assessment of MLLM performance in distinguishing real from hallucinated objects. 

**MME.** MME (Multimodal Large Language Model Evaluation) benchmark (Fu et al., 2024a) quantitatively assesses MLLMs across ten perception-related and four cognition-focused subtasks. To measure object-level hallucination, it uses subsets focused on object existence and count, while attribute-level hallucinations are assessed through subsets concerning object position and color. 

## 4.1.2 BASELINES 

**Regular setting.** We use two baseline MLLMs LLaVa-1.5 (Li et al., 2023c; Liu et al., 2024b) and Qwen2-VL (Wang et al., 2024) for our baseline setting. 

**VCD.** Visual Contrastive Decoding (Leng et al., 2024) is a training-free technique that mitigates object hallucinations in MLLMs. By contrasting output distributions from original and distorted visual inputs, VCD reduces the model’s over-reliance on statistical biases and unimodal priors. 

**OPERA.** Over-trust Penalty and Retrospection-Allocation (Huang et al., 2024) is an decoding-based method that mitigates hallucinations in MLLMs. It introduces a penalty term during beam search to address over-trust issues, and incorporates a rollback strategy for token selection. 

## 4.2 MAIN RESULTS 

**Results on VLind-Bench.** As shown in the figure 3, the experimental results on the VLind-Bench benchmark (Lee et al., 2024) are particularly interesting. On the LLaVA1.5 model, other methods failed to achieve significant performance improvements in balancing modality priors, while the performance under the multimodal collaborative setting has made a significant leap, indicating that the visual priors and language priors of LLaVA1.5 are balanced. The visual priors of the Qwen2-VL model 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0006-09.png)


**Figure 3: Scores of different methods on VLind-Bench.** CAUSALMM method significantly improves the model’s score on VLind-Bench. 

has been improved, so that the language setting and the multimodal collaborative setting have achieved similar optimal performance. 

This observation can be attributed to the nature of VLind-Bench, which comprises a suite of evaluation frameworks designed to elucidate the influence of various factors and to quantify the reliance on language priors. Such an evaluation paradigm imposes stringent requirements on the equilibrium of the model’s multimodal prior knowledge. Our multimodal collaborative method has notably enhanced the baseline model’s performance across all metrics, effectively achieving a balance in the model’s modal priors. Compared with other methods that follow human priors, the CAUSALMM method’s automatic capture of the causal effect of attention enables it to balance the bias of different modalities simultaneously. This outcome robustly substantiates the efficacy of our methodology (Liu et al., 2024d). 

**Results on POPE.** The experimental analysis conducted on the POPE benchmark (see Table 1), as delineated in prior studies (Li et al., 2023c; Lin et al., 2014; Schwenk et al., 2022; Hudson & Manning, 2019), reveals that our proposed CAUSALMM demonstrates superior performance in mitigating object-level hallucinations across random, popular, and adversarial settings. CAUSALMM consistently outperforms existing baselines on the most evaluation metrics, indicating a robust enhancement in performance, with an average metric improvement of **5.37%** . 

6 

Published as a conference paper at ICLR 2025 

**Table 1: Main results on POPE tasks.** We evaluate the POPE task accuracy of various MLLMs on the MSCOCO, A-OKVQA, and GQA datasets with LLaVa-1.5 under different decoding settings. **Regular** refers to the scenario where direct sampling is applied. **Vision** , **Language** and **Multimodal** refer to vision-only, language-only, and multimodal collaboration variants of CAUSALMM. The **bold** and the underlined refer to the highest and second highest metrics under each setting, respectively. Each value is followed by the difference relative to regular setting. 

||**Dataset**|**Setting**<br>**Method**|**Accuracy**|**Precision**<br>**Recall**<br>**F1 Score**|
|---|---|---|---|---|
||MSCOCO|Random<br>Regular<br>VCD<br>OPERA<br>**Vision**<br>**Language**<br>**Multimodal**|83.53 (0.00)<br>86.40 (2.87)<br>**89.20**(5.67)<br>86.46 (2.93)<br>88.00 (4.47)<br>88.93<br>(5.40)|92.12 (0.00)<br>73.33 (0.00)<br>81.66 (0.00)<br>94.68 (2.56)<br>77.13 (3.80)<br>85.01 (3.35)<br>92.68 (0.56)<br>**85.26**(11.9)<br>**88.81**(7.15)<br>**96.27**(4.15)<br>75.86 (2.53)<br>84.86 (3.20)<br>95.96<br>(3.84)<br>79.33 (6.00)<br>86.86 (5.20)<br>95.20 (3.08)<br>82.00<br>(8.67)<br>88.10<br>(6.44)<br>87.89 (0.00)<br>72.13 (0.00)<br>79.23 (0.00)<br>89.29 (1.40)<br>76.20 (4.07)<br>82.23 (3.00)<br>88.24 (0.35)<br>85.26 (13.1)<br>86.62 (7.39)<br>91.57<br>(3.68)<br>76.13 (3.00)<br>83.14 (3.91)<br>**91.80**(3.91)<br>88.13<br>(16.0)<br>87.17<br>(7.94)<br>86.35 (1.46)<br>**88.20**(16.0)<br>**87.26**(8.03)<br>82.96 (0.00)<br>72.06 (0.00)<br>77.13 (0.00)<br>84.47 (1.51)<br>76.20 (4.14)<br>80.12 (3.99)<br>78.79 (4.17)<br>**85.20**(13.1)<br>81.87<br>(4.74)<br>86.64<br>(3.68)<br>76.13 (4.07)<br>81.05 (3.92)<br>86.28 (3.32)<br>75.46 (3.40)<br>80.51 (3.38)<br>**87.69**(4.73)<br>78.40<br>(6.34)<br>**82.78**(5.65)<br>87.67 (0.00)<br>79.20 (0.00)<br>83.22 (0.00)<br>88.27 (0.60)<br>82.80 (3.60)<br>85.44 (2.22)<br>86.13 (1.54)<br>**91.13**(11.9)<br>84.59 (1.37)<br>90.24<br>(2.57)<br>84.46 (5.26)<br>87.25<br>(4.03)<br>89.75 (2.08)<br>81.20 (2.00)<br>85.26 (2.04)<br>**91.89**(4.22)<br>85.40<br>(6.20)<br>**88.52**(5.30)<br>80.87 (0.00)<br>79.20 (0.00)<br>80.02 (0.00)<br>81.44 (0.57)<br>82.80 (3.60)<br>82.11 (2.09)<br>78.92 (2.05)<br>**91.13**(11.9)<br>84.59<br>(4.57)<br>83.74 (2.87)<br>84.46<br>(5.26)<br>84.10 (4.08)<br>89.75<br>(8.88)<br>81.20 (2.00)<br>**85.26**(5.24)<br>**92.60**(11.7)<br>77.60 (1.60)<br>84.43 (4.41)<br>72.33 (0.00)<br>78.60 (0.00)<br>75.33 (0.00)<br>72.90 (0.57)<br>83.06 (4.46)<br>77.65 (2.32)<br>67.77 (4.56)<br>**91.13**(12.5)<br>**84.59**(9.26)<br>73.43 (1.10)<br>84.20 (5.60)<br>78.44 (3.11)<br>**74.98**(2.65)<br>82.33 (3.73)<br>78.48 (3.15)<br>74.41<br>(2.08)<br>84.93<br>(6.33)<br>79.32<br>(3.99)<br>87.11 (0.00)<br>78.86 (0.00)<br>82.78 (0.00)<br>88.21 (1.10)<br>82.80 (3.94)<br>85.41 (2.63)<br>85.45 (1.66)<br>**92.80**(13.9)<br>**88.90 (6.12)**<br>90.53<br>(3.42)<br>83.53 (4.67)<br>86.89 (4.11)<br>90.18 (3.07)<br>82.06 (3.20)<br>85.93 (3.15)<br>**90.81**(3.70)<br>85.66<br>(6.80)<br>88.16<br>(5.38)<br>77.32 (0.00)<br>78.86 (0.00)<br>78.08 (0.00)<br>77.04 (0.28)<br>82.80 (3.94)<br>79.82 (1.74)<br>73.65 (3.67)<br>**92.80**(13.9)<br>82.12<br>(4.04)<br>79.20<br>(1.88)<br>83.53 (4.67)<br>81.31 (3.23)<br>78.70 (1.38)<br>82.06 (3.20)<br>80.35 (2.27)<br>**80.36**(2.04)<br>85.66<br>(6.80)<br>**82.92**(4.84)<br>73.31 (0.00)<br>79.13 (0.00)<br>76.61 (0.00)<br>73.23 (0.08)<br>83.00 (3.87)<br>77.81 (1.20)<br>68.43 (4.88)<br>**92.80**(13.6)<br>78.77<br>(2.16)<br>73.43 (0.12)<br>84.20 (5.07)<br>78.44 (1.83)<br>74.21<br>(0.90)<br>81.53 (2.40)<br>77.70 (1.09)<br>**76.49**(3.18)<br>85.26<br>(6.13)<br>**80.64**(3.03)|
|||Popular<br>Regular<br>VCD<br>OPERA<br>**Vision**<br>**Language**<br>**Multimodal**|81.10 (0.00)<br>83.53 (2.43)<br>86.83 (5.73)<br>84.56 (3.46)<br>87.03<br>(5.93)<br>**87.13**(6.03)||
|||Adversarial<br>Regular<br>VCD<br>OPERA<br>**Vision**<br>**Language**<br>**Multimodal**|78.63 (0.00)<br>81.10 (2.47)<br>81.13 (2.50)<br>82.20<br>(3.57)<br>81.73 (3.10)<br>**83.70**(5.07)||
||A-OKVQA|Random<br>Regular<br>VCD<br>OPERA<br>**Vision**<br>**Language**<br>**Multimodal**|84.03 (0.00)<br>85.90 (1.87)<br>88.23<br>(4.20)<br>87.66 (3.63)<br>85.96 (1.93)<br>**88.93**(4.90)||
|||Popular<br>Regular<br>VCD<br>OPERA<br>**Vision**<br>**Language**<br>**Multimodal**|80.23 (0.00)<br>81.96 (1.73)<br>83.40 (3.17)<br>84.03 (3.80)<br>**85.96**(5.73)<br>85.70<br>(5.47)||
|||Adversarial<br>Regular<br>VCD<br>OPERA<br>**Vision**<br>**Language**<br>**Multimodal**|74.26 (0.00)<br>76.10 (1.84)<br>73.90 (0.36)<br>76.86 (2.60)<br>77.43<br>(3.17)<br>**77.86**(3.60)||
||GQA|Random<br>Regular<br>VCD<br>OPERA<br>**Vision**<br>**Language**<br>**Multimodal**|83.60 (0.00)<br>85.86 (2.26)<br>88.50(5.90)<br>87.40 (3.80)<br>86.56 (2.96)<br>**88.50**(5.90)||
|||Popular<br>Regular<br>VCD<br>OPERA<br>**Vision**<br>**Language**<br>**Multimodal**|77.86 (0.00)<br>79.06 (1.20)<br>79.80 (1.94)<br>80.80<br>(2.94)<br>79.93 (2.07)<br>**82.36**(4.50)||
|||Adversarial<br>Regular<br>VCD<br>OPERA<br>**Vision**<br>**Language**<br>**Multimodal**|75.16 (0.00)<br>76.33 (1.17)<br>75.00 (0.16)<br>76.80<br>(1.64)<br>76.60 (1.44)<br>**79.53**(4.37)||



7 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0008-01.png)


**Figure 4: Result comparison of different categories on MME Benchmark across different methods.** In most tasks, the scores obtained by CAUSALMM are higher than baselines, which verifies its effectiveness. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0008-03.png)


**Figure 5: Result comparison of perception and cognition views on MME Benchmark across different methods.** In both perception and cognition dimensions, variants of CAUSALMM outperform the others. 

Notably, both the vision-only and language-only variants of CAUSALMM exhibit significant improvements in effectiveness. Furthermore, the multimodal collaborative approach within our model achieves the highest accuracy, underscoring the synergistic benefits of integrating multiple modalities. Despite the observed performance decline in various baselines when subjected to popular and adversarial settings, our model maintains remarkable stability. This observation suggests that our CAUSALMM method is instrumental in enhancing stability. Moreover, the equilibrium of multimodal parameter priors is deemed crucial, as it can, to a certain extent, amplify the advantages conferred by the balanced priors of distinct modalities. This equilibrium is pivotal in effectively curtailing multimodal hallucinations. 

**Results on MME.** The empirical investigations conducted on the MME benchmark (Fu et al., 2024a) offer a thorough assessment of both object-level and attribute-level hallucinations. It has been discerned that while models such as LLaVA-1.5 (Liu et al., 2024c;b) and Qwen2VL (Wang et al., 2024) exhibit commendable performance in evaluating the presence of objects, they encounter challenges when dealing with more intricate queries, notably those involving counting. As indicated in Figure 4 and Figure 5, our CAUSALMM has been instrumental in significantly enhancing the performance of these models, yielding substantial improvements. 

In the domain of attribute-level evalua- **Table 2: Evaluation on the subset of MME perception.** tion, it has been observed that models While most of the data are similar, the CAUSALMM method are more prone to hallucinations concernhelps Qwen2-VL improve the performance of multiple indiing attributes like color. Our proposed cators in MME Benchmark. CAUSALMM, once again, demonstrates Method OCR celebrity landmark count significant improvements in this area. The CAUSALMM methods have demonstrated Regular 147.50 147.64 182.05 160.00 **Vision** 162.50 150.29 182.75 165.00 robust performance across various met- **Language** 170.00 168.23 182.50 160.00 rics, particularly excelling in numerical **Multimodal 170.00 168.23 182.75 165.00** computations and counting, which also 

translates into an advantage in the overall score. Although the performance on tasks such as Position remains relatively consistent, the overall enhancements in the perception and cognitive categories underscore the effectiveness of these methods in reducing hallucinations. 

In the context of poster and scene tasks, the language-only method has achieved the highest performance, which serves as a compelling validation of the impact of language priors on model performance. The MME fullset evaluation corroborates that our CAUSALMM method consistently maintains superior performance across a diverse array of tasks and models, thereby further substantiating its practical utility in enhancing the precision and reliability of MLLMs. 

8 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0009-01.png)


**Figure 6: Ablation on different counterfactual attentions.** The specific value is obtained by taking the average of all the results. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0009-03.png)


**Figure 7: Ablation on intervention cross layers.** We explored the relationship between the number of layers of intervention in the LLM and the causal effect. 

## 4.3 ABLATION STUDY 

**Ablation on different counterfactual attention.** To explore the generation of generalized counterfactual attention through interventions (Pearl, 2009), we evaluated four distinct types of counterfactual attention. Ablation experiments were conducted to systematically assess the impact of each type on model performance, as presented in Figure 6. The results demonstrate that using random attention as the anchor for the causal effect leads to the most substantial improvement in model performance. This improvement arises because perturbed attention, when aligned with average attention, can be more clearly distinguished from the original attention. This alignment aligns with the principles of the average causal effect. 

The reason for this finding is that perturbed attention, when close to the average attention level, better reflects a generalizable attention distribution pattern. Such generalizability enables a more accurate estimation of the causal effect, as it reduces the influence of outlier attention patterns that may not be representative of the overall dataset. Therefore, this approach more effectively meets the criteria for estimating the average causal effect, contributing to the observed performance improvement. 

**Ablation on intervention cross layers.** Beyond the categorization of counterfactuals, the effectiveness of counterfactual attention depends on its application across different layers of a large language model. To investigate the influence of language priors at various depths, interventions were meticulously conducted in the early, middle, and late layers of the model. This multi-layered approach is based on the hypothesis that language priors exert varying levels of influence at different stages of language processing. 

By intervening at different layers, we aimed to determine whether counterfactual attention could effectively modulate these priors. Based on the experimental results in Figure 7, interventions between shallow and middle layers proved to be the most effective. We hypothesize that these layers represent the initial stages where language priors significantly impact processing. Interventions in this range can effectively establish anchor points that are influenced by language priors, thereby improving model output to a certain extent. 

## 4.4 CASE STUDY 

**Case Study on LLaVA-Bench.** To provide a more vivid illustration of the impact of our CAUSALMM method, a case study was conducted on the LLaVABench dataset (Liu et al., 2024c). This study employed specific visual questions and the corresponding model responses to elucidate the enhancement in model output quality and the mitigation of adverse effects, such as hallucinations, attributable to the CAUSALMM method. A representative example is depicted in Figure 8. Objects like _boat_ , 

**Table 3: GPT-4o-aided-evaluation.** The evaluation results of gpt4-o as an expert. The four indicators represent the overall quality, conversational, detailedness and complexity. 

||Method<br>Regular<br>**Vision**<br>**Language**<br>**Multimodal**|All<br>84.7<br>84.8<br>84.7<br>**85.0**|Conv<br>87.7<br>88.8<br>88.8<br>**88.8**|Detail<br>89.3<br>86.7<br>88.0<br>**89.3**|Cplx<br>80.4<br>**81.4**<br>80.4<br>80.0|
|---|---|---|---|---|---|



which frequently co-occur with the potential ground truth object _ocean_ , are prone to being hallucinated. However, the application of our CAUSALMM method notably diminishes these hallucinatory tendencies. It enables the model to discern the city situated at the base of the volcano while maintaining a coherent and informative output text. This outcome underscores the efficacy of CAUSALMM in refining the output and curtailing the emergence of spurious associations. 

9 

Published as a conference paper at ICLR 2025 

**GPT-4o-aided-evaluation.** Supplementing the standard benchmark assessments, we have employed the GPT-4o[*] as an evaluative referee to quantitatively measure the efficacy of our CAUSALMM method. The evaluation was conducted using a 10-point scoring system, with the results compiled in Table 3. The results indicate that CAUSALMM is more adept at generating responses that align with the sophisticated evaluative standards set by GPT-4o. 

## **POSITIVE CASE** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0010-03.png)


**Regular:** The image showcases a picturesque scene of a mountainous island with a large hill or mountain in the middle. The island is surrounded by a beautiful blue ocean, which adds to the serene atmosphere. The mountain appears to be a volcano, giving it a striking appearance. In the background, there are several small boats scattered across the ocean, likely enjoying the peaceful waters. The combination of the island, mountain, and ocean creates a breathtaking view, making it an ideal location for relaxation and exploration. 

**CAUSALMM:** The image features a breathtaking view of a large, rocky island with a lush green hillside. The island is situated in the middle of a body of water, which could be an ocean or a lake. The island’s terrain is a mix of rocky cliffs and green vegetation, creating a stunning contrast against the water. In the background, a city can be seen, possibly on the mainland or another island. The city’s buildings are visible against the backdrop of the large island, adding to the overall beauty of the scene. 

**Figure 8:** Positive case of CausalMM. 

## **NEGATIVE CASE** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0010-08.png)


**----- Start of picture text -----**<br>
Prompt:  Is there any strawberry-flavored<br>yogurt in the fridge?<br>Regular: Yes, there is a container<br>of strawberry-flavored yogurt in the<br>fridge.<br>CAUSALMM: Yes, there is a con-<br>tainer of strawberry-flavored yogurt in<br>the fridge.<br>Figure 9: Unsolved case of CausalMM.<br>**----- End of picture text -----**<br>


## 5 CONCLUSION 

Though promising, MLLMs are prone to biases from visual and language priors, which can degrade performance and cause multimodal hallucinations. These biases stem from the influence of the visual encoder and LLM backbone on the attention mechanism, hindering the model’s ability to align multimodal inputs effectively. To overcome this, we introduced a causal reasoning framework termed CAUSALMM that applies structural causal modeling to MLLMs, treating modality priors as a confounding factor. By leveraging back-door adjustment and counterfactual reasoning at both visual and language attention levels, CAUSALMM demonstrates significant reductions in language priors bias and offers a plug-and-play solution compatible with other training-free approaches, providing a insightful path forward for trustyworthy multimodal intelligence. 

> *https://platform.openai.com/docs/models/gpt-4o 

10 

Published as a conference paper at ICLR 2025 

## 6 ACKNOWLEDGMENTS 

This work was supported by CAAI-Ant Group Research Fund; Guangdong Provincial Department of Education Project (Grant No.2024KQNCX028); Scientific Research Projects for the Highereducational Institutions (Grant No.2024312096), Education Bureau of Guangzhou Municipality; Guangzhou-HKUST(GZ) Joint Funding Program (Grant No.2025A03J3957), Education Bureau of Guangzhou Municipality. 

## REFERENCES 

Riddhiman Adib, Paul Griffin, Sheikh Iqbal Ahamed, and Mohammad Adibuzzaman. A causally formulated hazard ratio estimation through backdoor adjustment on structural causal model. In _Machine Learning for Healthcare Conference_ , pp. 376–396. PMLR, 2020. 

- Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. Hallucination of multimodal large language models: A survey. _arXiv preprint arXiv:2404.18930_ , 2024. 

- Meiqi Chen, Yixin Cao, Yan Zhang, and Chaochao Lu. Quantifying and mitigating unimodal biases in multimodal large language models: A causal perspective. _arXiv preprint arXiv:2403.18346_ , 2024. 

- Yuxiao Cheng, Runzhao Yang, Tingxiong Xiao, Zongren Li, Jinli Suo, Kunlun He, and Qionghai Dai. Cuts: Neural causal discovery from irregular time-series data. _arXiv preprint arXiv:2302.07458_ , 2023. 

- Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. Dola: Decoding by contrasting layers improves factuality in large language models. _arXiv preprint arXiv:2309.03883_ , 2023. 

- Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023. 

- Yaxin Fang and Faming Liang. Causal-stonet: Causal inference for high-dimensional complex data. _arXiv preprint arXiv:2403.18994_ , 2024. 

- Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models, 2024a. 

- Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Xiong Wang, Di Yin, Long Ma, Xiawu Zheng, et al. Vita: Towards open-source interactive omni multimodal llm. _arXiv preprint arXiv:2408.05211_ , 2024b. 

- Aryo Pradipta Gema, Chen Jin, Ahmed Abdulaal, Tom Diethe, Philip Teare, Beatrice Alex, Pasquale Minervini, and Amrutha Saseendran. Decore: Decoding by contrasting retrieval heads to mitigate hallucinations. _arXiv preprint arXiv:2410.18860_ , 2024. 

Wenbo Gong, Joel Jennings, Cheng Zhang, and Nick Pawlowski. Rhino: Deep causal temporal relationship learning with history-dependent noise. _arXiv preprint arXiv:2210.14706_ , 2022. 

- Mohammed Hassanin, Saeed Anwar, Ibrahim Radwan, Fahad Shahbaz Khan, and Ajmal Mian. Visual attention methods in deep learning: An in-depth survey. _Information Fusion_ , 108:102417, 2024. 

- Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 13418–13427, 2024. 

11 

Published as a conference paper at ICLR 2025 

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 6700–6709, 2019. 

Jiahao Huo, Yibo Yan, Boren Hu, Yutao Yue, and Xuming Hu. Mmneuron: Discovering neuron-level domain-specific interpretation in multimodal large language model. _arXiv preprint arXiv:2406.11193_ , 2024. 

- Yizhang Jin, Jian Li, Yexin Liu, Tianjun Gu, Kai Wu, Zhengkai Jiang, Muyang He, Bo Zhao, Xin Tan, Zhenye Gan, et al. Efficient multimodal large language models: A survey. _arXiv preprint arXiv:2405.10739_ , 2024. 

- Emre Kıcıman, Robert Ness, Amit Sharma, and Chenhao Tan. Causal reasoning and large language models: Opening a new frontier for causality. _arXiv preprint arXiv:2305.00050_ , 2023. 

- Kang-il Lee, Minbeom Kim, Seunghyun Yoon, Minsung Kim, Dongryeol Lee, Hyukhun Koh, and Kyomin Jung. Vlind-bench: Measuring language priors in large vision-language models. _arXiv preprint arXiv:2406.08702_ , 2024. 

- Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 13872–13882, 2024. 

- Wenhui Li, Xinqi Su, Dan Song, Lanjun Wang, Kun Zhang, and An-An Liu. Towards deconfounded image-text matching with causal inference. In _Proceedings of the 31st ACM International Conference on Multimedia_ , pp. 6264–6273, 2023a. 

- Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori B Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive decoding: Open-ended text generation as optimization. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 12286–12312, 2023b. 

- Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 292–305, 2023c. 

- Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13_ , pp. 740–755. Springer, 2014. 

- Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. A survey on hallucination in large vision-language models. _arXiv preprint arXiv:2402.00253_ , 2024a. 

- Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 26296–26306, 2024b. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _Advances in neural information processing systems_ , 36, 2024c. 

- Xiaoyu Liu, Paiheng Xu, Junda Wu, Jiaxin Yuan, Yifan Yang, Yuhang Zhou, Fuxiao Liu, Tianrui Guan, Haoliang Wang, Tong Yu, et al. Large language models and causal inference in collaboration: A comprehensive survey. _arXiv preprint arXiv:2403.09606_ , 2024d. 

- Krisztina S´ara Lukics and Agnes[´] Luk´acs. Modality, presentation, domain and training effects in statistical learning. _Scientific Reports_ , 12(1):20878, 2022. 

- Nick Pawlowski, Daniel Coelho de Castro, and Ben Glocker. Deep structural causal models for tractable counterfactual inference. _Advances in neural information processing systems_ , 33:857– 869, 2020. 

12 

Published as a conference paper at ICLR 2025 

Judea Pearl. _Causality_ . Cambridge university press, 2009. 

Daowan Peng, Wei Wei, Xian-Ling Mao, Yuanyuan Fu, and Dangyang Chen. An empirical study on the language modal in visual question answering. _arXiv preprint arXiv:2305.10143_ , 2023. 

- Yongming Rao, Guangyi Chen, Jiwen Lu, and Jie Zhou. Counterfactual attention learning for finegrained visual categorization and re-identification. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pp. 1025–1034, 2021. 

- Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In _European conference on computer vision_ , pp. 146–162. Springer, 2022. 

- Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. _arXiv preprint arXiv:2405.09818_ , 2024. 

- Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. _arXiv preprint arXiv:2406.16860_ , 2024a. 

- Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 9568–9578, June 2024b. 

- Aniket Vashishtha, Abbavaram Gowtham Reddy, Abhinav Kumar, Saketh Bachu, Vineeth N Balasubramanian, and Amit Sharma. Causal inference using llm-guided discovery. _arXiv preprint arXiv:2310.15117_ , 2023. 

- A Vaswani. Attention is all you need. _Advances in Neural Information Processing Systems_ , 2017. 

- Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. _arXiv preprint arXiv:2409.12191_ , 2024. 

- Yulun Wu, Robert A Barton, Zichen Wang, Vassilis N Ioannidis, Carlo De Donno, Layne C Price, Luis F Voloch, and George Karypis. Predicting cellular responses with variational causal inference and refined relational information. _arXiv preprint arXiv:2210.00116_ , 2022. 

- Guandong Xu, Tri Dung Duong, Qian Li, Shaowu Liu, and Xianzhi Wang. Causality learning: A new perspective for interpretable machine learning. _arXiv:2006.16789_ , 2020. 

- Yibo Yan and Joey Lee. Georeasoner: Reasoning on geospatially grounded context for natural language understanding. _arXiv preprint arXiv:2408.11366_ , 2024. 

- Yibo Yan, Haomin Wen, Siru Zhong, Wei Chen, Haodong Chen, Qingsong Wen, Roger Zimmermann, and Yuxuan Liang. Urbanclip: Learning text-enhanced urban region profiling with contrastive language-image pretraining from the web. In _Proceedings of the ACM on Web Conference 2024_ , pp. 4006–4017, 2024. 

- Mengyue Yang, Furui Liu, Zhitang Chen, Xinwei Shen, Jianye Hao, and Jun Wang. Causalvae: Disentangled representation learning via neural structural causal models. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 9593–9602, 2021. 

- Yue Yang, Artemis Panagopoulou, Shenghao Zhou, Daniel Jin, Chris Callison-Burch, and Mark Yatskar. Language in a bottle: Language model guided concept bottlenecks for interpretable image classification. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 19187–19197, 2023. 

- Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, and Fei Huang. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 13040–13051, 2024. 

13 

Published as a conference paper at ICLR 2025 

Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. _arXiv preprint arXiv:2306.13549_ , 2023. 

- Duzhen Zhang, Yahan Yu, Chenxing Li, Jiahua Dong, Dan Su, Chenhui Chu, and Dong Yu. Mmllms: Recent advances in multimodal large language models. _arXiv preprint arXiv:2401.13601_ , 2024. 

- Kexuan Zhang, Qiyu Sun, Chaoqiang Zhao, and Yang Tang. Causal reasoning in typical computer vision tasks. _arXiv:2307.13992_ , 2023a. 

- Zaixi Zhang, Qi Liu, Zhicai Wang, Zepu Lu, and Qingyong Hu. Backdoor defense via deconfounded representation learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 12228–12238, 2023b. 

- Jia Zhao, Xuesong Zhang, Xuefeng Wang, Ying Yang, and Gang Sun. Overcoming language priors in vqa via adding visual module. _Neural Computing and Applications_ , 34(11):9015–9023, 2022. 

- Zheng Zhao, Emilio Monti, Jens Lehmann, and Haytham Assem. Enhancing contextual understanding in large language models through contrastive decoding. In _Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico City, Mexico, June 16-21, 2024_ , 2024. 

- Kening Zheng, Junkai Chen, Yibo Yan, Xin Zou, and Xuming Hu. Reefknot: A comprehensive benchmark for relation hallucination evaluation, analysis and mitigation in multimodal large language models. _arXiv preprint arXiv:2408.09429_ , 2024. 

- Xin Zou, Chang Tang, Xiao Zheng, Zhenglai Li, Xiao He, Shan An, and Xinwang Liu. Dpnet: Dynamic poly-attention network for trustworthy multi-modal classification. In _Proceedings of the 31st ACM International Conference on Multimedia_ , pp. 3550–3559, 2023. 

- Xin Zou, Yizhou Wang, Yibo Yan, Sirui Huang, Kening Zheng, Junkai Chen, Chang Tang, and Xuming Hu. Look twice before you answer: Memory-space visual retracing for hallucination mitigation in multimodal large language models. _arXiv preprint arXiv:2410.03577_ , 2024a. 

- Xin gchen Zou, Yibo Yan, Xixuan Hao, Yuehong Hu, Haomin Wen, Erdong Liu, Junbo Zhang, Yong Li, Tianrui Li, Yu Zheng, et al. Deep learning for cross-domain data fusion in urban computing: Taxonomy, advances, and outlook. _Information Fusion_ , 113:102606, 2024b. 

14 

Published as a conference paper at ICLR 2025 

## A APPENDIX 

## A.1 FURTHER DEMONSTRATION 

## STRUCTURAL CAUSAL MODEL (SCM): 

Take the three core variables mentioned in the article as an example. 

## VARIABLES AND THEIR ROLES: 

   - _A_ (attention): This represents the model’s attention mechanism that we aim to evaluate or manipulate. 

   - _M_ (modality priors): Modality priors influence both the model’s attention ( _A_ ) and the output ( _O_ ), thus creating confounding. 

   - _O_ (model output): The outcome variable, which is affected both directly by _A_ and indirectly through _M_ . 

- CAUSAL STRUCTURE AND BACK-DOOR PATHS: 

   - The back-door path in this SCM is _A ← M → O_ , which starts with an arrow pointing into _A_ and creates a confounding junction structure. 

   - To isolate the causal effect of _A_ on _O_ , the confounding influence of _M_ must be blocked. 

## BACK-DOOR CRITERION: 

To apply back-door adjustment, the adjustment set _M_ must satisfy the following criteria: 

1. _M_ blocks all back-door paths from _A_ to _O_ . 

2. _M_ does not include any descendants of _A_ (i.e., variables causally influenced by _A_ ). 

By intervening on _A_ and adjusting for _M_ , we can isolate the causal effect of _A_ on _O_ . 

BACK-DOOR ADJUSTMENT FORMULA: 

Given a sufficient adjustment set _M_ , the causal effect _P_ ( _o | do_ ( _a_ )) is identified as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0015-19.png)


DERIVATION: 

1. **Starting with the interventional distribution:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0015-22.png)


2. **Using the property of the intervention** _do_ ( _a_ ) **:** Under the intervention _do_ ( _a_ ), the variable _A_ is no longer influenced by _M_ . Thus: 

_P_ ( _m | do_ ( _a_ )) = _P_ ( _m_ ) 

3. **Replacing** _P_ ( _o | do_ ( _a_ ) _, m_ ) **with the observational counterpart:** Due to the back-door criterion, _M_ blocks all confounding paths, allowing: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0015-26.png)


4. **Combining these results:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0015-28.png)


15 

Published as a conference paper at ICLR 2025 

APPLICATION TO ATTENTION-OUTPUT FRAMEWORK: 

In the context of our framework: 

1. **Back-door path:** The back-door path _A ← M → O_ reflects the confounding effect of modality priors ( _M_ ) on the attention mechanism ( _A_ ) and the model’s output ( _O_ ). 

2. **Intervention:** By intervening on _A_ , we ensure that the causal effect of attention on the output is isolated, free from the influence of modality priors. 

3. **Adjustment:** To block the back-door path, we adjust for _M_ , computing the summation over all possible values of _M_ to account for its confounding effect. 

## FULL FORMULA FOR THE FRAMEWORK: 

In our framework, the causal effect of attention ( _A_ ) on the model output ( _O_ ) can be computed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0016-08.png)


- _P_ ( _o | a, m_ ): The conditional probability of the output given attention _A_ and modality priors _M_ . 

- _P_ ( _m_ ): The marginal probability of modality priors _M_ . 

By applying the back-door adjustment formula, we mitigate the influence of confounding modality priors, ensuring that the attention mechanism’s causal contribution to the output is properly estimated. 

16 

Published as a conference paper at ICLR 2025 

## A.2 ADDITIONAL EXPERIMENTAL RESULTS 

To demonstrate the effectiveness of our approach on large multimodal language models of different architectures, we added experimental data from the Q-former-based InstructBLIP model and the embedding-autoregressive-based Chameleon model to the original experimental data from the vision encoder-mlp-llm paradigm. See tab. 4 and tab. 5 for specific data. Comparisons with more baseline methods can be found in tab. 6. 

**Table 4: Additional Experimental Results on POPE tasks: Chameleon.** We evaluate the POPE task accuracy of various MLLMs on the MSCOCO, A-OKVQA, and GQA datasets with Chameleon (Team, 2024) under different decoding settings. **Regular** refers to the scenario where direct sampling is applied. **Language** refer to language-only. 

||**Dataset**|**Setting**<br>**Method**<br>**Accuracy**<br>**Precision**<br>**Recall**<br>**F1 Score**|
|---|---|---|
||MSCOCO|Random<br>Regular<br>61.90<br>57.46<br>91.67<br>70.64<br>**Language**<br>69.23<br>63.17<br>92.27<br>74.99<br>Popular<br>Regular<br>65.10<br>59.86<br>91.67<br>72.43<br>**Language**<br>69.43<br>63.34<br>92.27<br>75.12<br>Adversarial<br>Regular<br>60.20<br>56.28<br>91.40<br>69.66<br>**Language**<br>64.00<br>58.94<br>92.33<br>71.95<br>Random<br>Regular<br>60.37<br>56.26<br>93.20<br>70.16<br>**Language**<br>65.70<br>60.14<br>93.13<br>73.08<br>Popular<br>Regular<br>57.30<br>54.25<br>93.20<br>68.58<br>**Language**<br>63.07<br>58.16<br>93.13<br>71.60<br>Adversarial<br>Regular<br>53.57<br>51.99<br>93.20<br>66.75<br>**Language**<br>56.83<br>53.96<br>93.13<br>68.33<br>Random<br>Regular<br>60.37<br>56.26<br>93.20<br>70.16<br>**Language**<br>68.43<br>62.18<br>94.13<br>74.89<br>Popular<br>Regular<br>59.37<br>55.76<br>90.67<br>69.05<br>**Language**<br>66.73<br>60.81<br>94.13<br>73.89<br>Adversarial<br>Regular<br>52.73<br>51.55<br>90.67<br>65.73<br>**Language**<br>57.77<br>54.50<br>94.13<br>69.03|
||A-OKVQA||
||GQA||



17 

Published as a conference paper at ICLR 2025 

**Table 5: Additional Experimental Results on POPE tasks: InstructBLIP.** We evaluate the POPE task accuracy of various MLLMs on the MSCOCO, A-OKVQA, and GQA datasets with InstructBLIP (Dai et al., 2023) under different decoding settings. **Regular** refers to the scenario where direct sampling is applied. **Vision** , **Language** and **Multimodal** refer to vision-only, language-only, and multimodal collaboration variants of CAUSALMM. 

||**Dataset**|**Setting**<br>**Method**<br>**Accuracy**<br>**Precision**<br>**Recall**<br>**F1 Score**|
|---|---|---|
||MSCOCO|Random<br>Regular<br>80.71<br>81.67<br>79.19<br>80.41<br>VCD<br>84.53<br>88.55<br>79.32<br>83.68<br>**Vision**<br>87.17<br>92.72<br>80.67<br>86.27<br>**Language**<br>86.90<br>94.89<br>78.00<br>85.62<br>**Multimodal**<br>87.90<br>94.59<br>80.40<br>86.92<br>Popular<br>Regular<br>78.22<br>77.87<br>78.85<br>78.36<br>VCD<br>81.47<br>82.89<br>79.32<br>81.07<br>**Vision**<br>83.97<br>86.37<br>80.67<br>83.42<br>**Language**<br>83.53<br>87.71<br>78.00<br>82.57<br>**Multimodal**<br>84.90<br>88.35<br>80.40<br>84.19<br>Adversarial<br>Regular<br>75.84<br>74.30<br>79.03<br>76.59<br>VCD<br>79.56<br>79.67<br>79.39<br>79.52<br>**Vision**<br>81.47<br>81.89<br>80.80<br>81.34<br>**Language**<br>82.00<br>84.73<br>78.07<br>81.26<br>**Multimodal**<br>82.43<br>83.71<br>80.53<br>82.09<br>Random<br>Regular<br>80.91<br>77.97<br>86.16<br>81.86<br>VCD<br>84.11<br>82.21<br>87.05<br>84.56<br>**Vision**<br>87.33<br>85.94<br>89.27<br>87.57<br>**Language**<br>87.87<br>87.72<br>88.07<br>87.89<br>**Multimodal**<br>88.47<br>87.86<br>89.27<br>88.56<br>Popular<br>Regular<br>76.19<br>72.16<br>85.28<br>78.17<br>VCD<br>79.78<br>76.00<br>87.05<br>81.15<br>**Vision**<br>81.07<br>76.69<br>89.27<br>82.50<br>**Language**<br>82.33<br>79.01<br>88.07<br>83.29<br>**Multimodal**<br>82.13<br>78.45<br>88.60<br>83.22<br>Adversarial<br>Regular<br>70.71<br>65.91<br>85.83<br>75.56<br>VCD<br>74.33<br>69.46<br>86.87<br>77.19<br>**Vision**<br>74.83<br>69.11<br>89.80<br>78.11<br>**Language**<br>76.27<br>71.07<br>88.60<br>78.87<br>**Multimodal**<br>75.97<br>70.51<br>89.27<br>78.79<br>Random<br>Regular<br>79.65<br>77.14<br>84.29<br>80.56<br>VCD<br>83.69<br>81.84<br>86.61<br>84.16<br>**Vision**<br>86.10<br>84.56<br>88.33<br>86.40<br>**Language**<br>86.67<br>86.86<br>86.40<br>86.63<br>**Multimodal**<br>87.23<br>86.67<br>88.00<br>87.33<br>Popular<br>Regular<br>73.87<br>69.63<br>84.69<br>76.42<br>VCD<br>78.57<br>74.62<br>86.61<br>80.17<br>**Vision**<br>77.77<br>72.92<br>88.33<br>79.89<br>**Language**<br>79.17<br>75.48<br>86.40<br>80.57<br>**Multimodal**<br>78.97<br>74.99<br>86.93<br>80.52<br>Adversarial<br>Regular<br>70.56<br>66.12<br>84.33<br>74.12<br>VCD<br>75.08<br>70.59<br>85.99<br>77.53<br>**Vision**<br>74.50<br>69.33<br>87.87<br>77.51<br>**Language**<br>76.30<br>71.81<br>86.60<br>78.51<br>**Multimodal**<br>75.83<br>71.19<br>86.80<br>78.22|
||A-OKVQA||
||GQA||



18 

Published as a conference paper at ICLR 2025 

**Table 6: More results on POPE tasks.** We evaluate the POPE task accuracy of various MLLMs on the POPE benchmark with LLaVa-1.5 and InstructBLIP under different decoding settings. In the table, the values taken are the averages of the three parts of the POPE benchmark (MSCOCO, A-OKVQA, GQA). **Regular** refers to the scenario where direct sampling is applied. **Vision** , **Language** and **Multimodal** refer to vision-only, language-only, and multimodal collaboration variants of CAUSALMM. DOLA stands for DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models(Chuang et al., 2023). 

||**Dataset**|**Setting**<br>**Method**<br>**Accuracy**<br>**Precision**<br>**Recall**<br>**F1 Score**|
|---|---|---|
||InstructBLIP|Random<br>Regular<br>80.42<br>78.93<br>83.21<br>80.94<br>DOLA<br>83.00<br>83.06<br>83.13<br>83.00<br>VCD<br>84.11<br>84.20<br>84.33<br>84.13<br>OPERA<br>85.07<br>88.39<br>80.73<br>84.39<br>AGLA<br>87.30<br>88.83<br>85.68<br>87.07<br>Vision<br>86.87<br>87.74<br>86.09<br>86.75<br>Language<br>87.15<br>89.82<br>84.16<br>86.71<br>Multimodal<br>87.87<br>89.71<br>85.89<br>87.60<br>Popular<br>Regular<br>76.09<br>73.22<br>82.94<br>77.65<br>DOLA<br>78.99<br>77.12<br>83.13<br>79.85<br>VCD<br>79.94<br>77.84<br>84.33<br>80.80<br>OPERA<br>78.33<br>73.85<br>87.73<br>80.20<br>AGLA<br>81.86<br>80.17<br>85.68<br>82.58<br>Vision<br>80.94<br>78.66<br>86.09<br>81.94<br>Language<br>81.68<br>80.73<br>84.16<br>82.14<br>Multimodal<br>82.00<br>80.60<br>85.31<br>82.64<br>Adversarial<br>Regular<br>72.37<br>68.78<br>83.06<br>75.42<br>DOLA<br>74.67<br>71.53<br>83.11<br>76.68<br>VCD<br>76.32<br>73.24<br>84.08<br>78.08<br>OPERA<br>75.50<br>70.49<br>87.73<br>78.17<br>AGLA<br>77.29<br>74.09<br>85.67<br>79.16<br>Vision<br>76.93<br>73.44<br>86.16<br>78.99<br>Language<br>78.19<br>75.87<br>84.42<br>79.55<br>Multimodal<br>78.08<br>75.14<br>85.53<br>79.70<br>Random<br>Regular<br>83.72<br>89.30<br>77.13<br>82.55<br>DOLA<br>84.78<br>87.59<br>81.27<br>84.19<br>VCD<br>86.05<br>90.39<br>80.91<br>85.29<br>OPERA<br>88.64<br>88.09<br>89.73<br>87.43<br>AGLA<br>88.54<br>94.41<br>82.08<br>87.71<br>Vision<br>87.17<br>92.35<br>81.28<br>86.33<br>Language<br>86.84<br>91.96<br>80.86<br>85.68<br>Multimodal<br>88.79<br>92.63<br>84.35<br>88.26<br>Popular<br>Regular<br>79.73<br>82.03<br>76.73<br>79.11<br>DOLA<br>79.75<br>84.11<br>76.22<br>80.61<br>VCD<br>81.52<br>82.59<br>80.60<br>81.39<br>OPERA<br>83.34<br>80.27<br>89.73<br>84.44<br>AGLA<br>85.14<br>87.88<br>82.08<br>84.68<br>Vision<br>83.13<br>84.84<br>81.37<br>82.85<br>Language<br>84.31<br>86.75<br>83.80<br>84.26<br>Multimodal<br>85.06<br>86.44<br>83.82<br>84.87<br>Adversarial<br>Regular<br>76.02<br>76.20<br>76.60<br>76.36<br>DOLA<br>76.32<br>77.27<br>75.47<br>76.16<br>VCD<br>77.84<br>76.87<br>80.75<br>78.53<br>OPERA<br>76.68<br>71.66<br>89.71<br>79.46<br>AGLA<br>81.13<br>81.20<br>82.10<br>81.36<br>Vision<br>78.62<br>77.83<br>81.51<br>79.31<br>Language<br>78.59<br>78.49<br>79.77<br>78.90<br>Multimodal<br>80.36<br>79.53<br>82.86<br>80.91|
||LLaVA-1.5||



19 

Published as a conference paper at ICLR 2025 

## A.3 VISUALIZATION OF COUNTERFACTUAL ATTENTIONS 

## A.3.1 VISION ATTENTION 

In this work, we used four commonly used counterfactual visual attentions (Rao et al., 2021): random, reverse, uniform, and shuffle. They represent taking random values for global attention, reversing global attention, using consistent attention values, and disrupting the original attention distribution. They can all effectively provide anchor points for obtaining causal effects, thereby helping the model improve potential modal priors. Among them, the settings of random and uniform are closest to the average value in value distribution, so they can provide the largest positive average causal effect. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0020-04.png)


**Figure 10: Normal vision attention of vision encoder.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0020-06.png)


**Figure 11: Shuffled vision attention of vision encoder.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0020-08.png)


**Figure 13: Reversed vision attention of vision encoder.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0020-10.png)


**Figure 12: Random vision attention of vision encoder.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0020-12.png)


**Figure 14: Uniform vision attention of vision encoder.** 

20 

Published as a conference paper at ICLR 2025 

## A.3.2 LANGUAGE ATTENTION 

We visualize four similar counterfactual attentions: they represent taking random values for global attention, negating global attention, using consistent attention values, and disrupting the original attention distribution. We take three of them for visualization. Similarly, they can effectively provide anchors for obtaining causal effects, thereby helping the model improve the potential modal prior. Compared with visual attention, large language models with large parameters are not as sensitive to changes in attention as visual encoders. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0021-03.png)


**Figure 15: Visualization of normal LLM attention.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0021-05.png)


**Figure 17: Visualization of reversed LLM attention.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0021-07.png)


**Figure 16: Visualization of random LLM attention.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0021-09.png)


**Figure 18: Visualization of uniform LLM attention.** 

21 

Published as a conference paper at ICLR 2025 

## A.4 CASE STUDY 

We have selected some typical cases to demonstrate the effect of our method. The CausalMM method balances different modal priors to weaken the bias that may be caused by the model’s own parameter knowledge from the perspective of vision and language, so that the model’s output can be more aligned with multimodal input. This improvement is reflected in the model’s perception and cognitive ability of specific things, and the potential hallucinations of the original model have been effectively improved. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0022-03.png)


**Prompt:** How many uncut fruits are in the image? **Regular:** There are four uncut fruits in the image. **Our Method:** There are three uncut fruits in the image. 

**Figure 19: Case of counting task.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0022-06.png)


## **Prompt:** 

Describe this photo in detail. 

**Regular:** The image features a painting of a dog wearing a Renaissance-style dress, giving it a human-like appearance. The dog is sitting on a chair, with its head turned to the side, and appears to be looking at the viewer. The painting captures the dog’s unique and artistic expression, making it an interesting and eye-catching piece. 

**Our Method:** The image features a painting of a dog wearing a Renaissance-style dress and a hat. The dog appears to be sitting down, possibly posing for the portrait. The dog’s dress is long, covering its body from the neck down. The painting captures the dog’s unique appearance, making it an interesting and creative piece of art. 

**Figure 20: Case of description task.** 

22 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0023-01.png)


## **Prompt:** 

What material appears to make up the creature? 

## **Regular:** 

The creature appears to be made of a combination of plastic and glass, as it has a plastic body and glass eyes. 

## **Our Method:** 

The creature appears to be made of plastic. 

**Figure 21: Case of perception task.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0023-09.png)


## **Prompt:** 

Describe this photo in detail. 

## **Regular:** 

The image showcases a tall skyscraper towering over a cityscape. The tower is prominently visible in the middle of the scene, reaching high into the sky. The cityscape around the tower includes numerous buildings of various sizes and shapes, creating a bustling urban atmosphere. There are also several cars scattered throughout the scene, indicating that the area is busy with traffic. Additionally, there are three people visible in the image, likely going about their daily activities or enjoying the view of the towering skyscraper. 

## **Our Method:** 

The image features a tall, skinny building with a large antenna on top of it. The building is located in a city, surrounded by other buildings, and it appears to be a prominent landmark. The antenna is positioned towards the top of the building, making it stand out. In the background, there are several other buildings visible, showcasing the urban landscape. The scene captures the essence of a bustling cityscape with a mix of architectural styles. 

**Figure 22: Case of description task.** 

23 

Published as a conference paper at ICLR 2025 

## **Limitation of CAUSALMM** 

We further evaluated the effect of the CAUSALMM method based on a case study to explore the limitations of the method. The specific example is in fig. 23. We found that even after correcting some of the hallucinations caused by visual and language priors, our method still did not significantly improve the acquisition of high-level semantics. We believe that the bottleneck of our method is the performance bottleneck of the vision encoder and the LLM backbone. In future work, we will explore how to maximize the positive impact of balanced modal priors when the backbone model is fixed. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality_images/iclr_2025_Mitigating_Modality_Prior-Induced_Hallucinations_in_Multimodal_Large_Language_Models_via_Deciphering_Attention_Causality.pdf-0024-03.png)


## **Prompt:** 

What is funny about this image? Describe it panel by panel. 

## **Regular:** 

The image is a collage of three panels, each showing a different aspect of the same scene. The first panel features a cell phone plugged into a charger, while the second panel shows a person holding a box of earphones. The third panel displays a close-up of the person’s hand holding the earphones. The funny aspect of this image is that the person’s hand is holding the earphones in the same position as the cell phone is plugged into the charger in the first panel. This creates a humorous visual connection between the two objects, as if the person’s hand is plugging the earphones into the charger. 

## **Our Method:** 

The image is a collage of three panels, each showing a different aspect of a cell phone being charged. The first panel features a close-up of the cell phone’s charging port, which is plugged into a charger. The second panel shows a person holding a charger, likely preparing to plug it into the cell phone. The third panel displays a person holding a USB cable, which is also being used to charge the cell phone. The funny aspect of this image is that the person in the third panel is holding the USB cable upside down, which is not the correct way to plug it into the cell phone. This adds a humorous element to the otherwise straightforward image of charging a cell phone. 

**Figure 23: Case of description task.** 

24 

Published as a conference paper at ICLR 2025 

## A.5 GPT-AIED-EVALUATION TEMPLATE 

For gpt-aided-evaluation, we have designed a variety of prompt templates to try to achieve a fairer evaluation. The following is a more effective template for reference. 

## **GPT-aied-evaluation Template** 

1. Image Description Evaluation: You will be provided with a set of image descriptions and a list of comments about the image. Your task is to evaluate each comment for hallucinations, which are inaccuracies or inconsistencies with the factual descriptions. 

2. Hallucination Identification: Pay special attention to comments that claim the existence of something not present in the descriptions, describe objects or attributes incorrectly, or make unrelated statements. 

3. Judgment and Revision: For each comment, provide a judgment (hallucination, correct, or cannot judge) and, if necessary, rewrite the comment to accurately reflect the image content. Ensure that the revised comments are detailed, coherent, and free of hallucinations. 

4. Scoring Criteria: Rate the performance of the AI on a scale of 1 to 10 for each of the following criteria: 

Accuracy: How well the response aligns with the factual image content. Detailedness: The richness of the response in necessary details, excluding hallucinated parts. 

5. Output Format: 

Judgment: List each comment with its judgment (hallucination, correct, or cannot judge) and reason. 

Revised Sentences: Provide revised comments where necessary. 

Scores: Output the scores for accuracy and detailedness, with reasons. 

Example: 

Region Descriptions of the Image: 

[10, 20, 50, 60]: A red apple on a white plate. 

- [70, 30, 120, 80]: A blue cup on a wooden table. 

Comments for Evaluation: 

1. The apple is green. 

2. There is a spoon next to the cup. 

3. The atmosphere in the room is cozy. 

Your Output: 

Judgement: 

1. hallucination: The description states the apple is red, not green. 

2. cannot judge: The region descriptions do not mention a spoon. 

3. correct: The comment does not contradict the provided descriptions. 

Revised Sentences: 

1. The apple is red. Scores: 

Accuracy: 7 8 

Reason: Assistant 1 had one hallucination, Assistant 2’s response is consistent with the descriptions. Detailedness: 6 8 

Reason: Assistant 1’s response lacks necessary details due to the hallucination, Assistant 2 provides a richer description without hallucinations. 

25 

