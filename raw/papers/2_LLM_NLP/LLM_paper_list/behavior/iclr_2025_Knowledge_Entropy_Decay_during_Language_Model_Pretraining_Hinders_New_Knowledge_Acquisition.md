Preprint. Under review. 

# KNOWLEDGE ENTROPY DECAY DURING LANGUAGE MODEL PRETRAINING HINDERS NEW KNOWLEDGE ACQUISITION 

**Jiyeon Kim**[1] _[∗]_ 

**Hyunji Lee**[1] _[∗]_ **Hyowon Cho**[1] 

**Joel Jang**[2] 

**Hyeonbin Hwang**[1] **Seungpil Won**[3] **Youbin Ahn**[3] **Dohaeng Lee**[3] **Minjoon Seo**[1] 

1 KAIST AI 2 University of Washington 3 LG AI Research 

{jiyeon.kim, hyunji.amy.lee, minjoon}@kaist.ac.kr 

ABSTRACT 

In this work, we investigate how a model’s tendency to broadly integrate its parametric knowledge evolves throughout pretraining and how this behavior affects overall performance, particularly in terms of knowledge acquisition and forgetting. We introduce the concept of _knowledge entropy_ , which quantifies the range of memory sources the model engages with; high knowledge entropy indicates that the model utilizes a wide range of memory sources, while low knowledge entropy suggests reliance on specific sources with greater certainty. Our analysis reveals a consistent decline in knowledge entropy as pretraining advances. We also find that the decline is closely associated with a reduction in the model’s ability to acquire and retain knowledge, leading us to conclude that diminishing knowledge entropy (smaller number of active memory sources) impairs the model’s knowledge acquisition and retention capabilities. We find further support for this by demonstrating that increasing the activity of inactive memory sources enhances the model’s capacity for knowledge acquisition and retention[1] . 

## 1 INTRODUCTION 

Recent studies have analyzed how language models store world knowledge in their parameters and utilize this knowledge to generate responses during inference time (Geva et al., 2021; Dai et al., 2022a;b; Meng et al., 2022; Yao et al., 2024). However, little is known about how their behavior of integrating various factual knowledge embedded in their parameters changes throughout the pretraining stage. In this work, we perform a deep analysis of how a model’s property of broadly integrating diverse parametric knowledge evolves throughout pretraining and how these shifts affect overall performance, particularly in terms of knowledge acquisition and forgetting in a continual learning setup. We hypothesize that this varying level of integration may explain why models in the later stages of pretraining encounter challenges in acquiring new knowledge (Dohare et al., 2024; Jang et al., 2022; Chang et al., 2024). 

We introduce _knowledge entropy_ , which reflects how a language model integrates various knowledge sources, to investigate how this behavior evolves throughout pretraining. Recent studies have shown that feed-forward layers (FFNs) serve as a key-value memory (Geva et al., 2022; 2021; Dai et al., 2022a). Building on this research, as shown in Figure 1, we view the second projection matrix _V_ as a memory, composed of memory 

> _∗_ Denotes equal contribution 

> 1Code in https://github.com/kaistAI/Knowledge-Entropy.git 

1 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0002-01.png)


**----- Start of picture text -----**<br>
... ... ... [Continual Learning Performance]<br>Memory
Vectors ... Initial Mid Final<br>Knowledge<br>Entropy Forgetting<br>CoefficientsMemory  ...<br>Acquisition<br>... Pretraining<br>Pretraining 
 Initial Mid Final steps<br>Initial Mid Final steps<br>**----- End of picture text -----**<br>


Figure 1: Illustration of our findings: distribution of memory coefficients _C_[¯] in feed-forward layers become sparser throughout pretraining, as indicated by a decrease in knowledge entropy _H_ ( _θ_ ). This sparsity deteriorates the model’s knowledge acquisition _A_ ( _θ_ ) and increases forgetting _F_ ( _θ_ ) when conducting continual knowledge learning with models from different pretraining stages. Thereby, as denoted by the star, when we artificially increase the knowledge entropy of the final stage model, both knowledge acquisition and retention increase. 

vectors, which store the model’s parametric knowledge, and view the first projection matrix _K_ as generating coefficients _C_[¯] that determine how these memory vectors are combined. Knowledge entropy measures how sparsely these memory coefficients are distributed; high knowledge entropy indicates that the model tends to integrate a broad range of memory vectors whereas the model with low knowledge entropy relies on specific memory vectors with high certainty. We analyze models at different stages of pretraining to investigate how knowledge entropy changes throughout pretraining. Our findings show that models in the later stages of pretraining tend to exhibit lower knowledge entropy, suggesting a shift from utilizing a larger set of active memory vectors to a smaller, more focused set as pretraining progresses. 

We hypothesize that changes in knowledge entropy would influence the model’s behavior when encountering new knowledge. To test this, we conduct a thorough analysis of the model’s ability to acquire new knowledge and retain existing knowledge in a continual knowledge learning[2] scenario on a target corpus (Jang et al., 2022; Wu et al., 2023), starting from different stages of pretraining. This involves further training the pretrained model on new-domain corpora using a language modeling objective to integrate new knowledge. Our results reveal a strong correlation between knowledge entropy and the model’s ability to acquire and retain knowledge: both knowledge entropy and knowledge acquisition and retention decrease as the pretraining progresses. 

We assume that this correlation arises because lower knowledge entropy indicates a smaller set of active memory vectors, leading to frequent overwriting of these memory vectors to store new knowledge. To test this assumption, we conduct experiments where we artificially increase the activity of previously inactive memory vectors, allowing the model to store new knowledge across a broader range of memory vectors. Surprisingly, we observe that these modified models demonstrate improved knowledge acquisition and reduced forgetting compared to the original models when undergoing continual knowledge learning; though not to the same extent as the original pretrained model with equivalent knowledge entropy. Such a result bolsters our hypothesis that having a limited number of active memory vectors (low knowledge entropy) plays a critical role in explaining the degradation of the model’s ability to acquire and retain knowledge as pretraining advances. 

> 2In this work, we use the term “continual knowledge learning” and “continual learning” interchangeably. 

2 

Preprint. Under review. 

Overall, our findings reveal that **as pretraining progresses, models exhibit a narrower integration of memory vectors, reflected by decreasing knowledge entropy, which hinders both knowledge acquisition and retention** . Models in the later stages of pretraining[3] , show low knowledge entropy, leading to poor knowledge acquisition and higher forgetting rates despite being trained on larger datasets. In contrast, earlystage models display high knowledge entropy, enabling better knowledge acquisition and retention; however, their performance is often limited by weaker language modeling capabilities. Thus, mid-stage models strike a balance, showing strong knowledge acquisition and retention along with overall performance, making them a practical choice for further training to incorporate new knowledge. To the best of our knowledge, this is the first work to analyze how a model’s behavior in integrating various memory vectors changes across pretraining stages and the subsequent effects on performance when acquiring new knowledge in a continual knowledge learning setup. 

## 2 RELATED WORK 

**Dynamics of Knowledge in Language Models** Recent studies have shown that language models embed world knowledge within their parameters and integrate this knowledge to generate responses (Yang, 2024; Petroni et al., 2019; Wang et al., 2021). Thereby, various research efforts aim to understand these dynamics of knowledge in language models (how they learn, store, and engage their parametric knowledge) during inference and training phases. Several studies have focused on investigating the inference process: Geva et al. (2023) analyzes the role of different layers in language models. Allen-Zhu & Li (2024b) demonstrates that model parameters have a limited knowledge capacity. Some studies suggest key-value memory (Geva et al., 2021; Meng et al., 2022; Dai et al., 2022a). Other research focuses on the pretraining phase. Liu et al. (2021) studies the sequence that language models learn various types of knowledge. Allen-Zhu & Li (2024a) examines strategies to enhance knowledge storage and extraction. Teehan et al. analyzes internal structural changes. Sun & Dredze (2024) investigates the interaction between pretraining and finetuning. Chang et al. (2024) analyzes patterns of knowledge acquisition specifically during the pretraining process, addressing the question of how language models acquire knowledge during pretraining. While their study shares similarities with ours in investigating knowledge acquisition behavior during LLM training, our work takes a different focus. We aim to understand _why_ LLMs encounter increasing difficulty in acquiring new knowledge as pretraining progresses, exploring the underlying reasons behind the challenges faced by later-stage models in learning new knowledge. To the best of our knowledge, our work is the first to explore how the behavior of language models in integrating their knowledge evolves throughout the pretraining phase, and to analyze how these changes affect model performance in terms of knowledge acquisition and forgetting in continual knowledge learning. 

**Entropy in Natural Language Processing** In information theory, entropy quantifies the value of information, where predictable (certain) events have low entropy and unpredictable (uncertain) events have high entropy (Lairez, 2022; Majenz, 2018). In natural language processing (NLP), entropy is used in various ways to measure the certainty of language models. Yang (2024) analyzes the entropy of model outputs based on input prompts. Araujo et al. (2022) calculates the entropy of outputs at each layer to determine weight adjustments in a continual learning setup. Other papers focus on token probability entropy to understand the information required to predict the next word in a sequence (Vazhentsev et al.; Geng et al., 2024; Malinin & Gales, 2021). Lower entropy in a model’s predictions may indicate that the model has become more certain about its predictions based on training data. Additionally, Kumar & Sarawagi (2019) measures entropy over the cross-attention layer to assess the uncertainty in the attention layer of encoder-decoder models. The entropy proposed in our paper, _knowledge entropy_ , differs from previous work in that it focuses on the 

3We define the final stage as the last stage of the pre-determined learning rate schedule, with the most decayed learning rate. The mid-stage refers to around 50% of the schedule, while the initial stage refers to around 20%. 

3 

Preprint. Under review. 

entropy of a model’s parametric knowledge, assessing the uncertainty or variability in utilizing the knowledge encoded within the language model. 

## 3 KNOWLEDGE ENTROPY 

In this section, we introduce knowledge entropy (Section 3.1) to examine how broadly the model integrates its parametric knowledge and describe the experimental setup used to measure it (Section 3.2). Next, in Section 3.3, we measure knowledge entropy at various pretraining stages to analyze how the model’s knowledge integration behavior evolves over pretraining. In Section 3.4, we extend our investigation by exploring alternative definitions of entropy. 

## 3.1 DEFINITION 

In this work, we introduce a new concept, _knowledge entropy_ , to analyze the scope of a model’s access patterns to its parametric knowledge. Low knowledge entropy suggests that the model relies on a narrower set of specific knowledge sources with high certainty whereas high knowledge entropy indicates that the model integrates with a diverse range of knowledge sources. Inspired by prior research that considers feed-forward layers (FFNs) as _key-value memory_ containing a model’s parametric memory (Geva et al., 2021; Dai et al., 2022a; Meng et al., 2022; Dong et al., 2022), we consider the knowledge source to be the _memory vectors_ , which is the second projection matrix of FFN. We measure how broadly the model integrates these memory vectors with _memory coefficients_ , which are calculated by the first projection matrix and the activation function. 

Geva et al. (2021) propose the concept of _key-value memory_ , demonstrating that FFNs function similarly to the key-value neural memories (Sukhbaatar et al., 2015). The feed-forward layer consists of two projection layers and activation in the middle: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0004-07.png)


where **x** _∈_ R _[d]_ . The first projection matrix ( _K ∈_ R _[m][×][d]_ ) corresponds to the keys, and the second projection matrix ( _V ∈_ R _[m][×][d]_ ) represents the values, or the _memories_ comprised of _memory vectors_ . The output, _FFN_ ( **x** ), is a linear combination of the memory vectors **v** _i_ =1 _,··· ,m ∈_ R _[d]_ which are the rows of _V_ , where the _coefficients_[4] _C_ are determined by _f_ ( **x** _· K[T]_ ), with _f_ being a non-linear activation function such as ReLU. Previous studies have shown that various types of factual and linguistic knowledge are encoded within these memories (Dai et al., 2022a; Geva et al., 2022; Meng et al., 2022; Dong et al., 2022). Thus, the final output is generated by combining the contributions of these memory vectors, where the memory coefficients determine the combination. 

Thereby knowledge entropy, _H_ ( _θ_ ), is calculated by the sum of layer-wise entropy _H_ ( _θ[l]_ ), which is based on the average coefficient _C_[¯] _[l] ∈_ R _[m]_ averaged across all tokens in dataset _D_ , as described in Equation 2. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0004-10.png)


4We use the terms “coefficient” and “memory coefficient” interchangeably. 

4 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0005-01.png)


**----- Start of picture text -----**<br>
70 Attention Entropy 2.8<br>1B NTP Entropy<br>144.0 7B 297.6<br>2.7<br>60<br>143.9<br>297.4 2.6<br>143.8<br>50<br>143.7 297.2 2.5<br>143.6 297.0 40 2.4<br>0 20 40 60 80 100 0 200 400 600<br>Rate of Pre-trained Step (%) Pre-trained steps (k)<br>NTP Entropy<br>Entropy(1B) Entropy(7B)<br>Attention Entropy<br>**----- End of picture text -----**<br>


Figure 2: Entropy (y-axis) across different model states (x-axis) for OLMo 1B and 7B. The x-axis represents the rate of the current step relative to the last step (738k for 1B and 557k for 7B). 

Figure 3: Entropy (y-axis) defined with Attention weight and next token prediction probability across different model states (x-axis) for OLMo 1B. 

_Cn,j_[(] _[l]_[)][represents the coefficient of the] _[ j]_[-th token position of the] _[ n]_[-th instance at layer] _[ l]_[,][ ¯] _[c] i[l]_[indicates the] _[ i]_[-th] element from _C_[¯] _[l]_ , _Tn_ is the sequence length of the _n_ -th instance in the dataset _D_ , _m_ is the inner dimension of feed-forward layer, and _L_ denotes the number of layers in the model. 

## 3.2 EXPERIMENT SETUP 

To conduct the experiment, we use the OLMo (Groeneveld et al., 2024) models (1B and 7B), which are open-source large language models with intermediate pretraining checkpoints released, trained on the Dolma dataset (Soldaini et al., 2024)[5] . To measure knowledge entropy, we use a subset of Dolma, 2k instances that appear in the first batch within the official pretraining data order to ensure that all models we are using have seen the corpus during pretraining step. Please note that the trend persists across other corpora as well (Figure 7 in Appendix A.2); however, since we are analyzing the model’s behavior throughout training, we define knowledge entropy based on calculations using the training dataset. 

In the case of OLMo, the memory coefficient _Cn,j_[(] _[l]_[)][is calculated as] _[ C] n,j_[(] _[l]_[)][=][abs][(][SwiGLU][(] **[x][j]**[))][ where] **[ x][j]**[is] the j-th token of input **x** and SwiGLU (Shazeer, 2020) is the activation function. We apply the absolute value since the SwiGLU allows negative values and the magnitude determines the contribution of the corresponding memory vector in the linear combination. Then, the absolute values are converted into a probability distribution. We also show experimentally that the trend persists with different choices of activation functions. Further details regarding knowledge entropy can be found in Appendix A.2. 

## 3.3 FINAL MODELS TEND TO EXHIBIT LOWER KNOWLEDGE ENTROPY 

Figure 2 illustrates how knowledge entropy (y-axis) changes across different stages of pretraining (x-axis). The results show a **consistent decrease in knowledge entropy as pretraining progresses** in both 1B and 7B models. This trend suggests that models in the later stages of pretraining tend to engage with a narrower range of memories, relying more heavily on specific memory vectors rather than accessing and integrating knowledge from a broader range of memories. A consistent reduction in knowledge entropy is observed across all layers, with the most significant reduction occurring in the last layer, which closely resembles the output distribution right before the token prediction (Figure 8 in Appendix). 

> 5We used OLMo and Dolma from the official repository 

5 

Preprint. Under review. 

## 3.4 SIMILAR TRENDS ARE OBSERVED BY DIFFERENT DEFINITIONS OF ENTROPY 

While our work defines knowledge entropy focusing on the feed-forward layer, previous studies have examined entropy in different contexts, such as the entropy of attention (Kumar & Sarawagi, 2019) and the entropy of next-token prediction over the vocabulary space (Vazhentsev et al.; Malinin & Gales, 2021). To gain a more comprehensive understanding of the model’s overall behavior, we extend our analysis by exploring the entropy trends in both attention mechanisms and next-token prediction. The formula and details are provided in Appendix A.3) 

**Entropy of Attention Layers** Following Kumar & Sarawagi (2019), we measure _attention entropy_ to capture the degree of uncertainty in attention weights. It is calculated as the sum of layer-wise entropy, where the layer-wise entropy measures the sparsity of the attention weights in each attention head. Thus, _attention entropy_ reflects how much weight the model assigns to specific tokens with confidence when generating the next token given the input based on token relationships. Figure 3 shows that attention entropy consistently decreases during pretraining, with a sharp decline in the early stages followed by a more gradual reduction. This trend suggests that the model learns to focus on contextually important tokens within the attention layer. 

**Entropy of Next Token Prediction** Entropy can also be measured based on the probability distribution of next-token predictions over the vocabulary space (Vazhentsev et al.; Geng et al., 2024; Malinin & Gales, 2021). Figure 3 shows that the entropy of the next token prediction also consistently decreases throughout pretraining, reflecting the model’s increasing certainty in its next token prediction. 

## 4 KNOWLEDGE ACQUISITION AND FORGETTING 

We hypothesize that the reduction of knowledge entropy as pretraining progresses impacts the model’s knowledge acquisition and forgetting as low knowledge entropy indicates sparse activation of memory vectors, thus the vectors are likely to be consistently overwritten when new knowledge is introduced. To test this hypothesis, we measure knowledge acquisition and forgetting using checkpoints from different stages of pretraining in a continual knowledge learning setup (Jang et al., 2022; Wu et al., 2023), where further training is performed on new-domain corpora by next token prediction to inject new knowledge into the pretrained models. Section 4.1 details the experimental setup and the metrics used. In Section 4.2, we present the results of knowledge acquisition and forgetting across various pretraining stages. Finally, Section 4.3 further explores whether a relationship exists between the two behaviors: activating the inactive memory vectors increases the knowledge acquisition ability. 

## 4.1 EXPERIMENT SETUP 

**Model & Hyperparameters** We experiment using intermediate checkpoints from OLMo[6] . Hyperparameters are chosen following previous research on continual knowledge learning (Jang et al., 2022; Kim et al., 2023) and we test various combinations to assess their generalizability. For batch size, we test 128 and 2048; for learning rate, we experiment with 1e-4, 4e-4, and 1e-3. We also investigate the effect of training duration by comparing a single epoch to three epochs. Among these configurations, we focus primarily on a batch size of 128, a learning rate of 4e-4, and single-epoch training as this setup most closely aligns with continual knowledge learning. 

6We select these intermediate checkpoints following Sun & Dredze (2024), which explores language model (OLMo) throughout pretraining. Available checkpoints can be found here. 

6 

Preprint. Under review. 

**Dataset** We experiment on a subset of two datasets[7] : PubMed[8] , a corpus of bio-medical and life science topics with abstracts, and C4 (Raffel et al., 2020), a large-scale corpus comprising diverse text data gathered from web pages. We use PubMed as the primary dataset as it contains more new knowledge, making it a better fit for our continual knowledge learning setup (Appendix B.1). In addition to the dataset, we inject synthetic knowledge during training to assess the model’s ability to acquire new information. Specifically, we utilize FICTIONAL KNOWLEDGE dataset (Chang et al., 2024), which is designed to assess how well language models acquire factual knowledge during pretraining[9] . This dataset includes 130 paragraphs about fictional yet realistic entities and 1,950 probes where each paragraph contains 15 different probes. The passages are incorporated into the training batch 10 times during the continual knowledge learning. After training, we evaluate the models on evaluation probes of the Fictional Knowledge dataset to measure knowledge acquisition, and evaluate on six downstream tasks in zero-shot manner (Sun & Dredze, 2024; Groeneveld et al., 2024) to measure knowledge forgetting (SciQ (Welbl et al., 2017), Winogrande (Sakaguchi et al., 2021), PIQA (Bisk et al., 2020), OBQA (Mihaylov et al., 2018), HellaSwag (Zellers et al., 2019), and ARC Easy (Clark et al., 2018)). Detailed explanation is included in Appendix B.1. 

**Metric** Knowledge acquisition of a language model _θ_ is measured with the probing performance on evaluation probes following Chang et al. (2024). When given the injected knowledge _W_ , each instance _wi_ in a corpus has a corresponding set of probes _Pwi_ , containing 15 different probes. To measure how well the model recalls the injected knowledge, we compute the **probe performance** _K_ ( _θ_ ), the average log probability _ℓ_ ( _pi_ ; _θ_ ) of the target span for each probe _pi ∈ Pwi_ across all instances in _wi ∈ W_ and calculate average; _K_ ( _θ_ ) = _|W_ 1 _|_ � _wi∈W |P_ 1 _wi |_ � _pi∈Pwi[ℓ]_[(] _[p][i]_[;] _[ θ]_[)][.][The] **[knowledge][acquisition][metric]** _[A]_[(] _[θ]_[)][is][defined][as][the] improvement rate of _K_ ( _θ_ ) from _θ_ PT to _θ_ CL, where _θ_ PT represents the model checkpoint from a pretraining step, which serves as the starting point and _θ_ CL represents the model after continual knowledge learning. High _A_ ( _θ_ ) indicates the model has learned new knowledge well. 

To measure knowledge forgetting of a language model, we measure **average performance over six downstream tasks** _P_ ( _θ_ ). **Knowledge forgetting** _F_ ( _θ_ ) is calculated by the reduction rate from _θ_ PT to _θ_ CL. Low _F_ ( _θ_ ) indicates that the models have retained their existing knowledge. The equation and a detailed explanation are presented in Appendix B.2. 

## 4.2 KNOWLEDGE ACQUISITION AND RETENTION DECREASES ACROSS PRETRAINING STAGE 

Figure 4a shows the performance of OLMo 1B and 7B models[10] corpus from various stages of pretraining as an initial state. **We observe that models in the final stages of pretraining struggle more with acquiring new knowledge** _A_ ( _θ_ ) **and exhibit greater forgetting** _F_ ( _θ_ ). As shown in Figure 4b, continually training the models at the mid-point of the pretraining as the initial checkpoint tends to yield the best performance in knowledge probing and downstream tasks compared to both models from the initial and final stages of pretraining. While early-stage models demonstrate high knowledge acquisition with minimal forgetting, their overall performance is limited by weaker language modeling capabilities. Conversely, later-stage models, despite being trained on larger datasets, exhibit lower knowledge acquisition and higher rates of forgetting, resulting in lower overall performance compared to the mid-stage models. This aligns with previous research suggesting that a model in the final stage of pretraining tends to struggle when learning new knowledge, showing a trade-off between _plasticity_ and _stability_ (Dohare et al., 2024; Biesialska et al., 2020; Jang et al., 

7We randomly sample 205k instances for each dataset. 

> 8Datasets in huggingface 

9We slightly modified the dataset to our setting of which details are in Appendix B.1 

> 10For this experiment, we used PubMed as a training corpus, but we also experiment with C4 as a training corpus, which presumably exhibits more similar distribution to the pretraining corpus, Dolma. The comparison of the two corpora and the results can be found in Appendix B.1 and B.4.2 

7 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0008-01.png)


**----- Start of picture text -----**<br>
20.0<br>30 1B  ( ) 17.5 0.25 64<br>1B ( )<br>7B  ( ) 15.0 0.26 62<br>27 7B  ( ) 12.5 0.27 1B  1B ( ( ) ) 60<br>7B  ( ) 58<br>24 10.0 0.28 7B ( )<br>56<br>21 7.5 0.29 54<br>5.0<br>0.30 52<br>20 40 60 80 100 20 40 60 80 100<br>Rate of Pre-trained Step (%) Rate of Pre-trained Step (%)<br>(a) (b)<br> (%)<br>)<br>) ) ( )<br>Acquisition ( Forgetting (<br>Knowledge probe  Downstream perf (<br>**----- End of picture text -----**<br>


Figure 4: (a) Rates of knowledge acquisition _A_ ( _θ_ ) and forgetting _F_ ( _θ_ ) (b) End performance of knowledge probe _K_ ( _θ_ ) and downstream task _P_ ( _θ_ ) for OLMo 1B (solid line) and 7B (dotted line) across different model states. The x-axis represents the ratio of the pretraining step of the initial model used in continual learning to the final step. (738k for 1B and 557k for 7B) 

2022). Therefore, we suggest that using a mid-stage checkpoint strikes a good balance, making it a practical choice as an initial starting point for further training to inject new knowledge. 

We consistently observe this pattern of later-stage models underperforming compared to earlier-stage models across various hyperparameter settings, including batch size, learning rate, training corpus, and the number of epochs. A detailed analysis of these results is provided in Appendix B.4. 

## 4.3 RESUSCITATING INACTIVE MEMORY VECTORS INCREASES KNOWLEDGE ACQUISITION 

We observe a strong correlation[11] between the trend of knowledge entropy (Figure 2) and the model’s ability to acquire and retain knowledge (Figure 4a). We assume that the model’s increasing reliance on a limited set of memory vectors (as indicated by a decrease in knowledge entropy) leads to more frequent updates to these vectors, making it difficult to acquire new knowledge and resulting in a higher rate of forgetting. The intuition behind this hypothesis can be found in Appendix A.1) To test this assumption, we conduct experiments where we artificially increase the activity or resuscitate previously inactive memory vectors. 

To resuscitate inactive memory vectors, we modify the up-projection matrix _K_ which engages with producing memory coefficients _C_[¯] (notations from Equation 2). Specifically, as shown in Algorithm 1, we identify the lowest _p_ % (resuscitation ratio) of memory coefficients and apply a multiplier _u_ to parameters in _K_ that are associated with these _p_ %. Multiplier _u_ can take any value; in this experiment, we divide the mean coefficient value of each layer by the respective coefficient value _c_ ¯ _[l] i_[at][each][identified][position] _[i]_[at][layer] _[l]_[,][and][then] multiply the result by an amplifying factor _q_ . By varying the value of _q_ , we control the degree of resuscitation applied to the _p_ % low-activation coefficients, thereby influencing the magnitude of the average coefficient and the corresponding size of the parameter updates. 

Figure 5a shows the knowledge acquisition and forgetting rates and Figure 5b presents the knowledge probe and downstream task performance after continual learning with various resuscitation configurations. For the experiment, we fix _p_ to 50 with varying _q_ and use the OLMo checkpoint at the last step of pretraining. Results show that when _q_ is set to 1 or greater, it generally yields better performance in both knowledge acquisition and retention compared to the original model. In contrast, when _q_ is set to 0.5, which further reduces already 

11The Pearson correlation between knowledge entropy and knowledge acquisition is 0.94, and with forgetting, it is -0.96. Both correlations are statistically significant, with p-values of 6e-5 and 1e-5, respectively. 

8 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0009-01.png)


**----- Start of picture text -----**<br>
70 20 0.28 52.3<br>60 ( ) 0.29 52.2<br>( ) 0<br>50<br>52.1<br>20 0.30<br>40<br>52.0<br>30 40 0.31 ( ) 51.9<br>20 ( )<br>0.32 51.8<br>original q=0.5 q=1 q=2 q=3 original q=0.5 q=1 q=2 q=3<br>(a) (b)<br> (%)<br>)<br>) ) ( )<br>Acquisition ( Forgetting (<br>Knowledge probe  Downstream perf (<br>**----- End of picture text -----**<br>


Figure 5: (a) Rates of knowledge acquisition _A_ ( _θ_ ) and forgetting _F_ ( _θ_ ) (b) End performance of knowledge probe _K_ ( _θ_ ) and downstream task _P_ ( _θ_ ) after training with varying the amplifying factor _q_ (x-axis) while _p_ is fixed as 50. 

inactive memory coefficients, both acquisition and retention decline suggesting that concentrating parameter updates more heavily on already active locations leads to sparser updates, ultimately impairing overall performance. These results suggest that **having a narrower active memory vector (low knowledge entropy) tends to reduce the model’s capacity to acquire new knowledge and increases knowledge forgetting** . 

Further experiments with fixed _q_ and varying _p_ show that increasing _p_ to activate a larger portion of inactive parameters generally leads to improved performance. Detailed result of this configuration is in Appendix B.6. We also analyze how the result changes when using models from different stages of pretraining as the original model. The trend persists over different checkpoints from the final stage of pretraining (554k). However, the effect of the resuscitation becomes more pronounced as the original model progresses to later stages of pretraining. Detailed results are included in Appendix B.7. 

Our result indicates that resuscitating inactive memory vectors of final-stage models tends to enhance knowledge acquisition and overall performance compared to the unmodified final-stage model. However, we observe that the performance remains lower compared to models from the pretraining step with similar knowledge entropy, such as the mid-stage model. This suggests that applying linear scaling to a subset of specific layers alone is insufficient to induce fundamental behavioral changes in the model. In other words, to restore a model that has lost its _plasticity_ (Dohare et al., 2024) to its previous state, more fundamental and alternative approaches are required. Further exploring methods for effectively modifying the parameters would be an interesting direction for future work. 

## 5 CONCLUSION 

In this work, we examine how large language models’ ability to broadly integrate their parametric knowledge (measured by knowledge entropy) changes throughout pretraining and how these changes affect knowledge acquisition and forgetting in a continual learning setup. Our findings reveal a strong correlation between knowledge entropy and the model’s capacity to acquire and retain knowledge. Models in the final stages of pretraining tend to exhibit narrower integration of memory vectors, leading to lower knowledge entropy, which negatively impacts both knowledge acquisition and retention. Interestingly, artificially increasing knowledge entropy by modifying the parameters of final-stage models tends to improve these capabilities. Based on our analysis, we suggest that models from the mid-stage of pretraining offer a good balance between knowledge acquisition, retention, and overall performance, making them a good choice for further training to introduce new knowledge. 

9 

Preprint. Under review. 

## **Algorithm 1** Resuscitating Low Memory Coefficients 

**Require:** _C_[¯] (average coefficients), _p_ (resuscitation ratio), _q_ (amplifying factor), _K_ (up-projection matrix) **Ensure:** Scaled up-projection matrix _K_ using computed multiplier _u_ 

- 1: **for** each layer _l_ in _K_ **do** 

- 2: Extract average activations for layer _l_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0010-05.png)


- 3: Compute the threshold _t_ for the lowest _p_ % activations: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0010-07.png)


- 4: Identify positions of values below the threshold _t_ : 

idx = ( _C ≤ t_ ).nonzero( ) 

- 5: Compute the scaling factor multiplier _u_ for coefficients in layer _l_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0010-11.png)


- 6: Apply scaling to the up-projection weights _Kl_ at the identified positions: 

_Kl_ [idx _,_ :] _×_ = _u_ [idx] 

## 7: **end for** 

## 6 LIMITATION & FUTURE WORK 

Due to computational constraints, our study measures knowledge acquisition and forgetting in a continual learning setup. Future work could explore whether these behaviors also manifest during the pretraining phase. We focused on OLMo 1B and 7B models, as they are the only models that publicly provide intermediate pretraining checkpoints and demonstrate strong performance (Sun & Dredze, 2024; Chang et al., 2024). Extending this investigation to other models would be a valuable direction for further research. Our resuscitation method, which arbitrarily modifies model parameters to test our hypothesis, showed promising results in improving knowledge acquisition and retention. However, performance tended to decline when resuscitating models at their initial or mid-stages. This suggests that more refined methods for resuscitating model parameters—ones that avoid random modification and preserve language modeling capabilities—could yield better outcomes. Additionally, while we observed that models in the mid-stage of pretraining strike a good balance for further training on tasks that involve acquiring new knowledge, defining the mid-point precisely remains an open question. In this study, we approximated the mid-point as 50% of the learning rate schedule. 

## ACKNOWLEDGMENTS 

We thank Sohee Yang, Hoyeon Chang, Seongyun Lee, Seungone Kim, Doyoung Kim, and Hanseok Oh for helpful discussions and constructive feedback. This work was supported by LG AI Research grant (Self-improving logical reasoning capabilities of LLMs, 2024). 

## REFERENCES 

AF Agarap. Deep learning using rectified linear units (relu). _arXiv preprint arXiv:1803.08375_ , 2018. 

10 

Preprint. Under review. 

- Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.1, knowledge storage and extraction. In _Forty-first International Conference on Machine Learning_ , 2024a. URL https://openreview.net/ forum?id=5x788rqbcj. 

- Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.3, knowledge capacity scaling laws. _arXiv preprint arXiv:2404.05405_ , 2024b. 

- Vladimir Araujo, Julio Hurtado, Alvaro Soto, and Marie-Francine Moens. Entropy-based stability-plasticity for lifelong learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 3721–3728, 2022. 

- Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In _International Conference on Machine Learning_ , pp. 2397–2430. PMLR, 2023. 

- Magdalena Biesialska, Katarzyna Biesialska, and Marta Ruiz Costa-jussà. Continual lifelong learning in natural language processing: A survey. _ArXiv_ , abs/2012.09823, 2020. URL https://api.semanticscholar. org/CorpusID:227231454. 

- Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 34, pp. 7432–7439, 2020. 

- Hoyeon Chang, Jinho Park, Seonghyeon Ye, Sohee Yang, Youngkyung Seo, Du-Seong Chang, and Minjoon Seo. How do large language models acquire factual knowledge during pretraining? _arXiv preprint arXiv:2406.11813_ , 2024. 

- Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. _arXiv preprint arXiv:1803.05457_ , 2018. 

- Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained transformers. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 8493–8502, Dublin, Ireland, May 2022a. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long. 581. URL https://aclanthology.org/2022.acl-long.581. 

- Damai Dai, Wen-Jie Jiang, Qingxiu Dong, Yajuan Lyu, Qiaoqiao She, and Zhifang Sui. Neural knowledge bank for pretrained transformers. In _Natural Language Processing and Chinese Computing_ , 2022b. URL https://api.semanticscholar.org/CorpusID:251223709. 

- Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, Ashique Rupam Mahmood, and Richard S. Sutton. Loss of plasticity in deep continual learning. _Nature_ , 632:768 – 774, 2024. URL https://api.semanticscholar.org/CorpusID:259251905. 

- Qingxiu Dong, Damai Dai, Yifan Song, Jingjing Xu, Zhifang Sui, and Lei Li. Calibrating factual knowledge in pretrained language models. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), _Findings of the Association for Computational Linguistics: EMNLP 2022_ , pp. 5937–5947, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. findings-emnlp.438. URL https://aclanthology.org/2022.findings-emnlp.438. 

11 

Preprint. Under review. 

- Jiahui Geng, Fengyu Cai, Yuxia Wang, Heinz Koeppl, Preslav Nakov, and Iryna Gurevych. A survey of confidence estimation and calibration in large language models. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), _Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_ , pp. 6577–6595, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. naacl-long.366. URL https://aclanthology.org/2024.naacl-long.366. 

- Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are keyvalue memories. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.), _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pp. 5484–5495, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.446. URL https://aclanthology.org/2021.emnlp-main.446. 

- Mor Geva, Avi Caciularu, Kevin Wang, and Yoav Goldberg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pp. 30–45, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.3. URL https://aclanthology.org/2022.emnlp-main.3. 

- Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 12216–12235, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.751. URL https://aclanthology.org/2023.emnlp-main.751. 

- Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, et al. Olmo: Accelerating the science of language models. _ACL 2024_ , 2024. 

- Joel Jang, Seonghyeon Ye, Sohee Yang, Joongbo Shin, Janghoon Han, Gyeonghun KIM, Stanley Jungkyu Choi, and Minjoon Seo. Towards continual knowledge learning of language models. In _International Conference on Learning Representations_ , 2022. URL https://openreview.net/forum?id=vfsRB5MImo9. 

- Yujin Kim, Jaehong Yoon, Seonghyeon Ye, Sung Ju Hwang, and Se-young Yun. Carpe diem: on the evaluation of world knowledge in lifelong language models. _NACCL 2024_ , 2023. 

- Aviral Kumar and Sunita Sarawagi. Calibration of encoder decoder models for neural machine translation. _arXiv preprint arXiv:1903.00802_ , 2019. 

- Didier Lairez. What entropy really is: the contribution of information theory. _arXiv preprint arXiv:2204.05747_ , 2022. 

- Zeyu Liu, Yizhong Wang, Jungo Kasai, Hannaneh Hajishirzi, and Noah A. Smith. Probing across time: What does RoBERTa know and when? In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.), _Findings of the Association for Computational Linguistics: EMNLP 2021_ , pp. 820–842, Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-emnlp.71. URL https://aclanthology.org/2021.findings-emnlp.71. 

- Christian Majenz. Entropy in quantum information theory–communication and cryptography. _arXiv preprint arXiv:1810.10436_ , 2018. 

- Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction. In _International Conference on Learning Representations_ , 2021. URL https://openreview.net/forum?id= jN5y-zb5Q7m. 

12 

Preprint. Under review. 

- Kevin Meng, David Bau, Alex J Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), _Advances in Neural Information Processing Systems_ , 2022. URL https://openreview.net/forum?id=-h6WAS6eE4. 

- Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (eds.), _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ , pp. 2381–2391, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1260. URL https://aclanthology.org/D18-1260. 

- Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H. Miller, and Sebastian Riedel. Language models as knowledge bases? In _Conference on Empirical Methods in Natural Language Processing_ , 2019. URL https://api.semanticscholar.org/CorpusID:202539551. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of machine learning research_ , 21(140):1–67, 2020. 

- Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. _Communications of the ACM_ , 64(9):99–106, 2021. 

- Noam Shazeer. Glu variants improve transformer. _arXiv preprint arXiv:2002.05202_ , 2020. 

- Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, et al. Dolma: An open corpus of three trillion tokens for language model pretraining research. _ACL 2024_ , 2024. 

- Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, et al. End-to-end memory networks. _Advances in neural information processing systems_ , 28, 2015. 

- Kaiser Sun and Mark Dredze. Amuro & char: Analyzing the relationship between pre-training and fine-tuning of large language models. _arXiv preprint arXiv:2408.06663_ , 2024. 

- Ryan Teehan, Miruna Clinciu, Oleg Serikov, Eliza Szczechla, Natasha Seelam, Shachar Mirkin, and Aaron Gokaslan. Emergent structures and training dynamics in large language models. ACL 2022. URL https://aclanthology.org/2022.bigscience-1.11. 

- Artem Vazhentsev, Akim Tsvigun, Roman Vashurin, Sergey Petrakov, Daniil Vasilev, Maxim Panov, Alexander Panchenko, and Artem Shelmanov. Efficient out-of-domain detection for sequence to sequence models. Findings of ACL 2023. URL https://aclanthology.org/2023.findings-acl.93. 

- Cunxiang Wang, Pai Liu, and Yue Zhang. Can generative pre-trained language models serve as knowledge bases for closed-book QA? In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pp. 3241–3251, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.251. URL https://aclanthology.org/2021.acl-long.251. 

- Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Leon Derczynski, Wei Xu, Alan Ritter, and Tim Baldwin (eds.), _Proceedings of the 3rd Workshop on Noisy User-generated Text_ , pp. 94–106, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/W17-4413. URL https://aclanthology.org/W17-4413. 

- Yuhao Wu, Tongjun Shi, Karthick Sharma, Chun Wei Seah, and Shuhao Zhang. Online continual knowledge learning for language models. _arXiv preprint arXiv:2311.09632_ , 2023. 

13 

Preprint. Under review. 

- Wenzhe Yang. Entropy, thermodynamics and the geometrization of the language model. _arXiv preprint arXiv:2407.21092_ , 2024. 

- Yunzhi Yao, Ningyu Zhang, Zekun Xi, Meng Wang, Ziwen Xu, Shumin Deng, and Huajun Chen. Knowledge circuits in pretrained transformers. _ArXiv_ , abs/2405.17969, 2024. URL https://api.semanticscholar. org/CorpusID:270068372. 

- Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Anna Korhonen, David Traum, and Lluís Màrquez (eds.), _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pp. 4791–4800, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1472. URL https: //aclanthology.org/P19-1472. 

## A KNOWLEDGE ENTROPY 

## A.1 INTUITION BEHIND THE DEFINITION OF KNOWLEDGE ENTROPY 

The mechanism behind the relationship between decreasing knowledge entropy and the ability to acquire and retain knowledge during pretraining is that the coefficients in the linear combination of memory vectors determine how the corresponding memory vectors are updated. As defined in Equation 1, the output of the feed-forward layers (FFNs) is a linear combination of memory vectors **v** _i_ =1 _,··· ,m ∈_ R _[d]_ , the row vectors of _V ∈_ R _[m][×][d]_ , where the coefficients _ci_ are given by _f_ ( **x** _· K[T]_ ). In other words, _FFN_ ( _x_ ) = _c_ 1 **v1** + _c_ 2 **v2** + _· · ·_ + _cm_ **vm** . Within a given layer, since the operations beyond the FFNs and the input to the FFNs remain consistent across all memory vectors, the coefficients act as scaling factors for the gradient by the chain rule. During training, the gradient _∂∂L_ **vi** _,_ **j**[for] _[ i]_[ = 1] _[,]_[ 2] _[, . . . , m]_[ and] _[ j]_[= 1] _[,]_[ 2] _[, . . . , d]_[ can be decomposed] as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0014-07.png)


Here, _∂F F N∂L_ ( _x_ )[is][the][same][for][all] **[v][i]**[,][meaning][that][the][relative][magnitude][of][the][gradient][depends][on] _∂F F N∂vi,j_ ( _x_ ) = _ci_ . Thus, larger coefficients result in proportionally larger gradients being applied to the corresponding memory vectors, amplifying their updates during backpropagation. 

As pretraining progresses, a spikier coefficient distribution—captured by decreasing knowledge entropy—implies that gradient updates become increasingly concentrated on specific positions where the average coefficients are larger. This centralization can affect the model’s ability to evenly utilize its memory capacity, thereby impacting knowledge acquisition and retention. 

## A.2 KNOWLEDGE ENTROPY 

**Does the choice of model change the trend?** To assess the generalizability of the trend observed in Figure 2, we conducted experiments on the knowledge entropy trend using Pythia 1.4B model (Biderman et al., 2023). As shown in Figure 6, knowledge entropy measured with the Pythia model also tends to decrease as pretraining progresses. 

**Does the choice of dataset change the trend?** As expressed in Equation 2, knowledge entropy is dependent on the dataset _D_ . We define _D_ as the dataset used during pretraining, as knowledge entropy reflects how the model integrates the knowledge stored in its memory vectors, learned during pretraining. However, to further explore whether the choice of dataset influences the trend of knowledge entropy, we measure it using PubMed and C4. Figure 7 shows that the trend remains consistent regardless of the dataset used when calculating knowledge entropy. 

14 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0015-01.png)


**----- Start of picture text -----**<br>
143.7<br>143.6<br>143.5<br>143.4<br>20 40 60 80 100 120 140<br>Pre-trained steps (k)<br>Knowledge Entropy<br>**----- End of picture text -----**<br>


Figure 6: Knowledge Entropy (y-axis) measured with Dolma on different pretraining stages of Pythia 1.4B model. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0015-03.png)


**----- Start of picture text -----**<br>
143.9<br>C4<br>143.8 Pubmed<br>Dolma<br>143.7<br>143.6<br>143.5<br>143.4<br>143.3<br>100 200 300 400 500 600 700<br>Pre-trained steps (k)<br>Knowledge Entropy<br>**----- End of picture text -----**<br>


Figure 7: Knowledge Entropy(y-axis) measured with Dolma, C4, Pubmed corpus across different pretraining stages 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0015-05.png)


**----- Start of picture text -----**<br>
9.30<br>9.00<br>8.98 9.27<br>8.96 9.24<br>8.94 9.21<br>8.92 9.18<br>18k 5k<br>8.90 369 9.15 278k<br>738k 557k<br>8.88 9.12<br>0 2 5 7 10 12 15 0 5 10 15 20 25 30<br>Layer Layer<br>(a) (b)<br>Knowledge Entropy Knowledge Entropy<br>**----- End of picture text -----**<br>


Figure 8: Layer-wise Knowledge Entropy of OLMo-1B(a) and 7B(b) 

**Does the choice of activation function change the trend?** We also explored an alternative where we do not take the absolute value of the SwiGLU output. Instead, following the ReLU function (Agarap, 2018), another widely used activation function, we replaced all negative values with 0. Figure 9 shows that the trend remains consistent even under this modification. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0015-08.png)


**Layer-wise Knowledge Entropy** Figure 8 shows how knowledge entropy changes by layer during pretraining. Knowledge entropy consistently decreases in every layer, with the most significant reduction occurring in the last layer, which closely resembles the output distribution right before the token prediction. OLMo-7B model also shows a similar trend to 1B model. 

15 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0016-01.png)


**----- Start of picture text -----**<br>
144.0 SwiGLUReLU 3000 pubmed<br>c4<br>2500<br>143.8<br>2000<br>143.6<br>1500<br>143.4 1000<br>143.2 500<br>0 100 200 300 400 500 600 700 0<br>Pre-trained Step (k) 0 10 20 30 40 50 60 70 80 90<br>Average Perplexity of Entities in the Instance<br>Entropy Frequency<br>**----- End of picture text -----**<br>


Figure 9: Knowledge Entropy(y-axis) when employing original SwiGLU(solid line) and ReLU activation(dotted line) across different pretraining stages 

Figure 10: Histogram of average perplexity of entities in C4 and Pubmed corpus 

## A.3 ENTROPY OF ATTENTION LAYERS 

Inspired by previous research that emphasizes the attention layer’s role of attribute extraction (Geva et al., 2023), we also measure _attention entropy H_ ( _θ_ att) similarly to Kumar & Sarawagi (2019)[12] . Attention weights, which are the output of softmax normalization after the key-query-value operation, can be interpreted as weight assigned to the previous tokens. As the attention weight for each token position in each attention head forms a probability distribution(summing to 1), calculating entropy follows the normal entropy formula. Then, layer-wise entropy _H_ ( _θ_ att _[l]_[)][ is averaged over token position and attention heads and attention entropy] _[ H]_[(] _[θ]_[att][)] is the sum of layer-wise entropy _H_ ( _θ_ att _[l]_[)][.][Following the notations from Geva et al. (2023), attention entropy is] calculated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0016-06.png)


where **A** _[h,l,n] ∈_ R _[T][n][×][T][n]_ represents the attention weights of the _h_ -th attention head in layer _l_ for the _n_ -th instance, _Tn_ is the sequence length of the _n_ -th instance in the training dataset _D_ , N denotes the number of attention heads, and _L_ denotes the number of layers in the model. 

## A.4 ENTROPY OF NEXT TOKEN PREDICTION 

The entropy of next token prediction (Vazhentsev et al.; Geng et al., 2024; Malinin & Gales, 2021) is defined as _H_ ( _θ_ ntp _[n,j]_[) =] _[ −]_[�] _[|] i[V]_ =1 _[|][p][i][ ·]_[ log(] _[p][i]_[)][, where] _[ p][i]_[ represents the probability of the] _[ i]_[-th token.][This value is then] averaged over the sequence length ( _Tn_ ) and the dataset size ( _|D|_ ). 

> 12Kumar & Sarawagi (2019) measures entropy using attention weights from cross-attention in an encoder-decoder architecture. However, as we employ a decoder-only model, we modify the equation to use attention weights from self-attention. 

16 

Preprint. Under review. 

## B KNOWLEDGE ACQUISITION AND FORGETTING 

## B.1 DATASETS 

**Training Dataset for Continual Knowledge Learning** In this section, we share a brief description of the datasets we used. For continual knowledge learning, we experiment with PubMed and C4. The PubMed dataset consists of biomedical literature abstracts from the PubMed database, containing articles across a wide range of topics in medicine and biology. The C4 (Colossal Clean Crawled Corpus) dataset (Raffel et al., 2020) is a large-scale, preprocessed collection of text scraped from the web, designed to be a clean and diverse representation of natural language. 

To compare the distribution of C4, PubMed, and Dolma, we evaluate the average perplexity of entities for C4 and PubMed, as shown in Figure 10. On the x-axis, we plot the range of an average perplexity of instances, while the y-axis represents the number of instances. We randomly sample 10,000 instances from each corpus, extract entities using GPT-4o, and calculate perplexity with the last checkpoint of OLMo. The perplexity values from the last checkpoint of OLMo indicate how likely these entities are to appear in the pretraining corpus, Dolma. The results reveal that PubMed exhibits a broad distribution of perplexity, with a higher number of instances having high perplexity values. In contrast, C4 shows a tendency towards lower perplexity, suggesting that the distribution of entities in PubMed differs from that in Dolma, while the distribution in C4 tends to be more similar to Dolma. 

**Evaluation Dataset to measure** _**Knowledge Acquisition**_ To measure knowledge acquisition of language model, we use the fictional knowledge dataset (Chang et al., 2024), which is designed to assess how well LLMs acquire factual knowledge during pretraining. This dataset includes 130 paragraphs[13] presented in a Wikipedia-style format with fictional yet realistic entities (injected knowledge) and 1,950 probes which are cloze-task-style sentences to query the information within the corpus. The final span of each probe, referred to as the target span, is used to evaluate the model’s prediction probability, which serves as a measure of knowledge acquisition performance. 

In Chang et al. (2024), the probes are divided into three levels of difficulty, with five sentences created for each level. This results in 15 probes per corpus. The difficulty levels are as follows: 1) **Memorization probes** directly ask about sentences explicitly present in the fictional corpus. 2) **Semantic generalization probes** are paraphrased versions of the memorization probes to test the model’s understanding of meaning beyond surface forms. 3) **Compositional generalization probes** are designed to assess whether the model can integrate multiple pieces of knowledge from the fictional corpus. The injected knowledge is incorporated into the training corpus during continual learning, with updates occurring every 160 steps. 

Following Chang et al. (2024), we divide the 130 corpora into two settings: _paraphrase_ and _once_ . In the _paraphrase_ setting, 70 instances are each paraphrased 10 times. For every 160 steps, one paraphrased version of an instance is added to the training corpus, repeating this process 10 times[14] . In the _once_ setting, each instance is presented only once throughout the entire continual learning process. The 60 instances are divided into 10 groups, with 6 instances added every 160 steps. 

**Evaluation Dataset to measure** _**Knowledge Forgetting**_ To measure the forgetting rate, we evaluate on 6 downstream datasets. 

> 13While Chang et al. (2024) utilizes 120 paragraphs, comprising 40 for _paraphrase_ setup, 40 for _duplicate_ , and 40 for _once_ , we utilized 130 paragraphs, the whole original data. We scaled up the number of paragraphs with paraphrases to 70, utilizing GPT-4 following Chang et al. (2024) 

14We maintain updates every 160 steps(10 steps when batch size is 2048) because our total training duration is 1,600 steps(100 steps), and we repeat the dataset injection process 10 times. 

17 

Preprint. Under review. 

- SCIQ: multiple-choice question-answering dataset consisting of over 13,000 science exam-style questions, covering subjects such as physics, chemistry, biology, and earth science 

- WINOGRANDE: large-scale benchmark designed to test commonsense reasoning in natural language understanding 

- PIQA: commonsense reasoning about everyday physical interactions such as how to perform tasks involving physical actions 

- OBQA: multiple-choice question-answering benchmark designed to assess a model’s ability to answer elementary-level science questions. 

- HELLASWAG: a large-scale benchmark for commonsense reasoning, focusing on selecting the most plausible continuation of a given narrative or scene. 

- ARC EASY: multiple-choice science questions typically answered by students in elementary and middle school. 

## B.2 METRIC 

In this section, we share a detailed description of how we evaluate _knowledge acquisition_ and _knowledge forgetting_ . 

**Knowledge Acquisition** Given a language model _θ_ , _θ_ PT represents the model extracted from a pretraining step and serves as the initial point for continual learning, and _θ_ CL represents the model after it. The acquisition metric _A_ ( _θ_ ) for the model _θ_ is defined as Equation 4. When given a corpus set of _once_ setting _W_ once, each instance in a corpus _wi_ has a corresponding set of probes _Pwi_ , which contains 15 different probes. To calculate the performance of the _once_ setting, _K_ once( _θ_ ), we compute the average log probability _ℓ_ ( _pi_ ; _θ_ ) of the target span for each probe _pi ∈ Pwi_ across all instances in _wi ∈ W_ once and sum these averages. The same calculation is performed for the _paraphrase_ setting. The total performance _K_ ( _θ_ ) is calculated by the weighted average of _K_ once( _θ_ ) and _K_ para( _θ_ ). Finally, the acquisition metric _A_ ( _θ_ ) is defined as the improvement rate in performance _K_ ( _θ_ ) from the initial model state _θ_ PT to final model state _θ_ CL. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0018-10.png)


**Knowledge Forgetting** The forgetting metric _F_ ( _θ_ ) is calculated as the average performance degradation from the initial model _θ_ PT to the final model _θ_ CL across six downstream tasks _T_ : SciQ (Welbl et al., 2017), Winograde (Sakaguchi et al., 2021), PIQA (Bisk et al., 2020), OBQA (Mihaylov et al., 2018), HellaSwag (Zellers et al., 2019), and ARC Easy (Clark et al., 2018). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0018-12.png)


18 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0019-01.png)


**----- Start of picture text -----**<br>
0.200<br>40<br>0.225<br>0.250<br>30<br>Paraphrase  ( ) 0.275<br>Paraphrase  ( )<br>20 Once  ( ) 0.300<br>Once  ( ) 0.325<br>10 0.350<br>0.375<br>0<br>0.400<br>100 200 300 400 500 600 700<br>Pre-trained Step (k)<br>)<br>(<br>)<br>(<br>Acquisition<br>Knowledge probe<br>**----- End of picture text -----**<br>


Figure 11: Rates of knowledge acquisition _A_ ( _θ_ ) and final performance of knowledge probe _K_ ( _θ_ ) according to the number of injection. Paraphrase prompt (solid line) were injected 10 times and Once (dotted line) prompt only once throughout continual training. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0019-03.png)


**----- Start of picture text -----**<br>
0.28<br>64<br>0.30 62<br>CL  ( CL [)] 60<br>0.32 CL ( CL [)]<br>PT  ( PT [)] 58<br>0.34 PT ( PT [)] 56<br>0.36 54<br>52<br>100 200 300 400 500 600 700<br>Pre-trained Step (k)<br>Figure 12: End performance of knowledge<br>probe K ( θ ) and downstream task P ( θ ) of Initial<br>model (dotted line) and model after continual train-<br>ing (solid line) in baseline setup. The x-axis repre-<br>sents the initial model state used for continual train-<br>ing.<br> (%)<br>)<br>( )<br>Knowledge probe  Downstream perf (<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0019-04.png)


**----- Start of picture text -----**<br>
20 0.28<br>24 Pubmed  ( ) 18 57<br>22 Pubmed ( C4  C4 ( ( ) ) ) 1614 0.29 Pubmed Pubmed (C4  ( ) ( )) 56<br>20 12 C4 ( ) 55<br>0.30<br>10 54<br>18<br>8<br>53<br>0.31<br>16 6<br>52<br>100 200 300 400 500 600 700 100 200 300 400 500 600 700<br>Pre-trained Step (k) Pre-trained Step (k)<br>(a) (b)<br> (%)<br>)<br>) ) ( )<br>Acquisition ( Forgetting (<br>Knowledge probe  Downstream perf (<br>**----- End of picture text -----**<br>


Figure 13: (a) Rates of knowledge acquisition _A_ ( _θ_ ) and forgetting _F_ ( _θ_ ) (b) End performance of knowledge probe _K_ ( _θ_ ) and downstream task _P_ ( _θ_ ) of Pubmed(solid line) and C4(dotted line) corpus across different model states. The x-axis represents the initial model state used for continual training. 

## B.3 FREQUENCY OF KNOWLEDGE INJECTIONS 

We divide the experiment into two settings: the _once_ setting, where knowledge is injected a single time, and the _paraphrase_ setting, where knowledge is injected ten times using ten paraphrased paragraphs. Figure 11 presents knowledge acquisition results based on the frequency of injections. Knowledge acquisition and final performance generally follow similar trends in both settings, with models in the later stages of pretraining showing the lowest performance. However, the performance and acquisition rate of _once_ setting lag behind those of _paraphrase_ setting. Also, notably, for models in the final stage of pretraining, the acquisition rate in the _once_ setting was negative. This indicates that the log probability of the injected knowledge decreased, preventing successful incorporation of the new knowledge. In other words, even the knowledge injected during continual learning is subject to forgetting throughout the continual learning process. 

19 

Preprint. Under review. 

## B.4 KNOWLEDGE ACQUISITION & FORGETTING 

## B.4.1 BASELINE SETUP 

Our base experiments are conducted using a hyperparameter configuration most closely aligned with continual knowledge learning studies, specifically with a batch size of 128, a learning rate of 4e-4, while training single-epoch of PubMed corpus. We use adamW optimizer ( _β_ = 0.9, 0.95, weight decay= 0.1), cosine LR scheduler with warmup=0.05, and set maximum sequence length as 1024. We randomly selected 204,800 instances from the PubMed and C4 datasets and matched the sequence length to 1,024 tokens by concatenating instances. This resulted in a training dataset consisting of approximately 210 million tokens. 

As analyzed in Section 4.2, final performance generally deteriorates as later-stage models are utilized as the initial model. Figure 12 illustrates the model’s initial performance before continual learning, as well as its performance afterward. Models in the later stages of pretraining exhibit superior language modeling abilities before continual learning, as evidenced by the lower log probability for the newly injected knowledge (dotted line). However, after continual learning, their performance deteriorates compared to models from the earlier stages of pretraining (solid line). Similarly, the downstream task performance of the later-stage models was better initially, but after continual learning, their performance declined more than that of the earlier-stage models. 

## B.4.2 VARIOUS SETTINGS 

We observed consistently, even with altered hyperparameter settings, that models in the later stages of pretraining struggle to learn new knowledge and retain existing knowledge. 

**Training Dataset for Continual Knowledge Learning** To investigate how the type of _new_ knowledge affects performance, we further conducted experiments using the C4 corpus, which has a distribution more similar to the pretraining corpus, Dolma, compared to PubMed. Figure 13a indicates that the gap in acquisition rate between later-stage models and initial-stage models is larger when the new knowledge distribution differs significantly from the pretraining corpus: models trained with PubMed (-6.4%p) exhibit a more pronounced gap compared to those trained with C4 (-3.4%p). 

All models, regardless of their pretraining stage, tend to perform better when continually pretrained with PubMed compared to C4. We hypothesize that this is because PubMed’s different distribution from the pretraining corpus encourages the model to learn more new knowledge, enhancing its ability to acquire new information. However, the rate of improvement varies by model state. Later-stage models tend to show similar performance regardless of the type of new knowledge, suggesting a limit in their learning capacity. In contrast, initial-stage models exhibit a stronger ability to acquire knowledge when trained on a corpus with a different distribution, such as PubMed, demonstrating their greater adaptability in learning new and diverse information. 

**Model Size** To test the universal deterioration of knowledge acquisition and retention capabilities, we also experimented with the OLMo 7B model Groeneveld et al. (2024). In Figure 4a, the 7B model exhibits a clear trend of diminishing knowledge acquisition _A_ ( _θ_ ) capabilities as pretraining progresses. This decline is accompanied by an increase in forgetting _F_ ( _θ_ ), indicating that the model struggles to retain previously learned information as new data is introduced. 

**Batch Size** In Table 2 line (b), when the batch size is large, the influence of individual data points on the model update decreases, resulting in a lower acquisition rate, while forgetting is less pronounced. In the later stages of training, however, if sufficient learning rate warmup is not provided, the model appears to collapse. 

20 

Preprint. Under review. 

||_A_(_θ_)_↑_<br>_F_(_θ_)_↓_|_A_(_θ_)_↑_<br>_F_(_θ_)_↓_|
|---|---|---|
||||
|Pretraining Step of_θ_|118k<br>369k<br>554k<br>738k|118k<br>369k<br>554k<br>738k|
|(a) Baseline<br>(b) BS :128_→_2048<br>(c) lr : 4e-4_→_1e-3<br>(d) lr : 4e-4_→_1e-4<br>(e) ep : 1_→_3|25.2<br>24.8<br>21.4<br>18.8<br>8.9<br>7.8<br>1.0<br>-92.2<br>17.3<br>15.7<br>10.7<br>5.0<br>21.6<br>23.0<br>23.7<br>22.6<br>28.8<br>28.0<br>25.9<br>24.5|10.5<br>12.9<br>17.0<br>19.5<br>2.2<br>3.5<br>5.5<br>44.9<br>16.7<br>20.3<br>23.0<br>23.5<br>2.4<br>3.2<br>4.6<br>7.3<br>14.8<br>17.4<br>20.1<br>23.1|



Table 1: Knowledge Acquisition _A_ ( _θ_ )[%] and Forgetting _F_ ( _θ_ )[%] across different continual learning hyperparameters. 

||_K_(_θ_)_↑_<br>_P_(_θ_)_↑_|_K_(_θ_)_↑_<br>_P_(_θ_)_↑_|
|---|---|---|
||||
|Pretraining Step of_θ_|118k<br>369k<br>554k<br>738k|118k<br>369k<br>554k<br>738k|
|(a) Baseline<br>(b) BS :128_→_2048<br>(c) lr : 4e-4_→_1e-3<br>(d) lr : 4e-4_→_1e-4<br>(e) ep : 1_→_3|-0.281<br>-0.280<br>-0.293<br>-0.301<br>-0.342<br>-0.344<br>-0.368<br>-0.714<br>-0.310<br>-0.314<br>-0.332<br>-0.353<br>-0.294<br>-0.287<br>-0.284<br>-0.287<br>-0.267<br>-0.268<br>-0.276<br>-0.280|54.0<br>54.6<br>52.8<br>52.0<br>59.0<br>60.5<br>60.1<br>35.6<br>50.3<br>49.9<br>49.0<br>49.4<br>58.9<br>60.7<br>60.6<br>59.9<br>51.4<br>51.7<br>50.8<br>49.7|



Table 2: Probe Performance _K_ ( _θ_ ) and average performance on six downstream task _P_ ( _θ_ ) across different Continual Learning hyperparameters. 

**Learning Rate** In Table 2 line (c), when the learning rate is increased, not only is the acquisition rate lower but forgetting becomes more pronounced. Conversely, when the learning rate is very small (in line (d)), the models in the late stage show improved acquisition performance with the least gap compared with initial models, but still perform worse than the mid-stage model. 

**Epoch** In Table 2 line (e), as the number of epochs increases, leading to more repetitions, acquisition improves with the cost of more forgetting. 

## B.5 TRAINING DYNAMICS OF PYTHIA 

Figure 14 shows the training dynamics of Pythia Biderman et al. (2023) 1.4B and OLMo 1B. We observe that the overall training dynamics of Pythia and OLMo are similar. However, Pythia tends to conclude early in training, using only 10% of training tokens compared to OLMo. As a result, Pythia struggles to capture the full dynamics of the language model during pretraining. This is because models in the early stage of training are still influenced by a large portion of randomly initialized parameters, which leads to different behavior compared to models that have undergone more training steps and reached a stable state. For OLMo, based on our analysis, we assume the model reaches a stable point around 118k steps (15% of the training dataset). However, Pythia’s final step occurs before this stable point, causing its trend to resemble that of OLMo but deviate from the trend observed in Section 4, where knowledge acquisition improves until the stable point and then decreases. We hypothesize that previous studies using intermediate checkpoints during 

21 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0022-01.png)


**----- Start of picture text -----**<br>
Olmo<br>144.1<br>144.0 24<br>143.9<br>143.8 22<br>143.7<br>20<br>143.6<br>0 1000 2000 3000<br>Pythia<br>143.7 26<br>143.6 25<br>143.5 24<br>143.4 23<br>0 1000 2000 3000<br>Tokens (M)<br>Knowledge Entropy<br>Knowledge Acquisition<br>Knowledge Entropy<br>Knowledge Acquisition<br>**----- End of picture text -----**<br>


Figure 14: Knowledge Entropy and Rates of knowledge acquisition _A_ ( _θ_ ) of OLMo 1B and Pythia 1.4B model across different pretraining stages 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0022-03.png)


**----- Start of picture text -----**<br>
0.28<br>24 19.5 52.4<br>( ) 19.0 0.29 52.3<br>22 ( ) 18.5<br>52.2<br>18.0 0.30<br>20 17.5 52.1<br>17.0 0.31 52.0<br>( )<br>18 16.5 ( )<br>51.9<br>0.32<br>original p=30 p=50 p=70 p=90 original p=30 p=50 p=70 p=90<br>(a) (b)<br> (%)<br>)<br>) ) ( )<br>Acquisition ( Forgetting (<br>Knowledge probe  Downstream perf (<br>**----- End of picture text -----**<br>


Figure 15: (a) Rates of knowledge acquisition _A_ ( _θ_ ) and forgetting _F_ ( _θ_ ) (b) End performance of knowledge probe _K_ ( _θ_ ) and downstream task _P_ ( _θ_ ) after training with varying _p_ (x-axis) while _q_ is fixed as 1. 

pretraining (Chang et al., 2024; Sun & Dredze, 2024) also only measured with OLMo without Pythia for the same reason. 

## B.6 RESUSCITATION EXPERIMENT: VARYING _p_ WHILE FIXING _q_ 

Figure 15 shows the overall performance when _q_ was fixed at 1, meaning that the lowest _p_ % of coefficients were scaled to converge toward the layer’s mean. It is shown that increasing _p_ to activate a larger portion of 

22 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0023-01.png)


**----- Start of picture text -----**<br>
0.26<br>20<br>54.5<br>50<br>0.27<br>10 54.0<br>40 0 0.28 53.5<br>resuscitation  ( )<br>resuscitation ( )<br>30 ori original ( ginal  ( ) ) 10 0.29 resuscitation  ( ) 53.0<br>20 0.30 resuscitation (original  ( ) ) 52.5<br>20 original ( )<br>52.0<br>0.31<br>100 200 300 400 500 600 700 100 200 300 400 500 600 700<br>Pre-trained Step (k) Pre-trained Step (k)<br>(a) (b)<br> (%)<br>)<br>) ) ( )<br>Acquisition ( Forgetting (<br>Knowledge probe  Downstream perf (<br>**----- End of picture text -----**<br>


Figure 16: (a) Rates of knowledge acquisition _A_ ( _θ_ ) and forgetting _F_ ( _θ_ ) (b) End performance of knowledge probe _K_ ( _θ_ ) and downstream task _P_ ( _θ_ ) of original continual learning (dotted line) and resuscitation (solid line) method where _p_ is fixed as 50 and _q_ as 2, across different model states. The x-axis represents the initial model state used for continual training. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0023-03.png)


**----- Start of picture text -----**<br>
20 0.28<br>54.5<br>10<br>60<br>0<br>54.0<br>50 369k  ( ) 10<br>40 369k (554k 554k (( ))) 2030 0.29 369k  369k (554k  ( ( ) ) ) 53.5<br>554k ( )<br>30 40 53.0<br>20 50 0.30 52.5<br>original q=0.5 q=1 q=2 q=3 original q=0.5 q=1 q=2 q=3<br>(a) (b)<br> (%)<br>)<br>) ) ( )<br>Acquisition ( Forgetting (<br>Knowledge probe  Downstream perf (<br>**----- End of picture text -----**<br>


Figure 17: (a) Rates of knowledge acquisition _A_ ( _θ_ ) and forgetting _F_ ( _θ_ ) (b) End performance of knowledge probe _K_ ( _θ_ ) and downstream task _P_ ( _θ_ ) of model from 369k steps (solid line) in pretraining schedule and 554k (dotted line) when varying the amplifying factor _q_ (x-axis) while _p_ is fixed as 50. 

inactive parameters generally led to improved performance. However, interestingly, setting _p_ too high(e.g., at 90) negatively impacted performance, likely due to the unintended effect of reducing the parameters in already active regions. 

## B.7 RESUSCITATION EXPERIMENT ACROSS PRETRAINING STEPS 

Figure 16a illustrates the overall performance when _q_ was fixed to 2 and _p_ to 50, across different pretraining stages of the original model. The effect of the resuscitation method becomes more pronounced as the original model progresses to later stages of pretraining, as indicated by the transition from the dotted line to the solid line. We hypothesize that this is because late-stage models tend to rely on a smaller subset of memory sources and thus benefit from a broader scope of activation enabled by the resuscitation method. As shown in Figure 16b, end performance deteriorates when the beginning model is initial (118k) and mid (369k) stage model, indicating that resuscitation may impair performance when the model’s knowledge entropy is not sufficiently low. This trend of the resuscitation showing a more positive effect for models in the later stage of pretraining can also be seen in Figure 17, which shows the results when varying _q_ while fixing _p_ at 50: 

23 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition_images/iclr_2025_Knowledge_Entropy_Decay_during_Language_Model_Pretraining_Hinders_New_Knowledge_Acquisition.pdf-0024-01.png)


**----- Start of picture text -----**<br>
20 0.29 52.2<br>50<br>10 52.1<br>52.0<br>40 Original 0 0.30<br>Attn Resuscitation 51.9<br>FFN Resuscitation 10 51.8<br>30 0.31<br>20 Original 51.7<br>Attn Resuscitation<br>20 FFN Resuscitation 51.6<br>30 0.32<br>1.5 2.0 3.0 1.5 2.0 3.0<br>(a) (b)<br>)  (%)<br>) ) ( )<br>Acquisition ( Forgetting (<br>Knowledge probe  Downstream perf (<br>**----- End of picture text -----**<br>


Figure 18: (a) Rates of knowledge acquisition _A_ ( _θ_ ) and forgetting _F_ ( _θ_ ) (b) End performance of knowledge probe _K_ ( _θ_ ) and downstream task _P_ ( _θ_ ) when resuscitation is applied at **attention layers** , with the x-axis representing temperature. The dotted line indicates performance when resuscitation is applied to the feedforward layer with _p_ =50, _q_ =2. The dashed line represents the original performance without any resuscitation. 

performance deteriorates when running continual learning on the model from 369k, while improvement of performance with larger _q_ is observed when the model is from 554k. 

## B.8 RESUSCITATION EXPERIMENT ACROSS ATTENTION LAYERS 

In Figure 3, we observe that as pretraining progresses, not only knowledge entropy but also attention entropy decreases. We explore whether extending the resuscitation method to the attention layers could similarly improve knowledge acquisition and retention. For the feed-forward layer, we can adjust knowledge entropy by tuning the parameters at positions with low activation values. However, since attention entropy reflects the probability distribution over token positions, the same resuscitation method cannot be directly applied to the attention layers. To artificially increase attention entropy, we employed temperature scaling on the softmax function in the attention calculation, which reduces the sparsity of the probability distribution. 

We experimented with temperature values ranging from 1.5 to 3.0. Figure 18 shows that resuscitation in the feed-forward layer consistently results in the highest knowledge acquisition rate and generally leads to the best knowledge retention. This trend is observed in both knowledge probe results and downstream performance. These findings suggest that knowledge entropy plays a critical role in influencing both knowledge acquisition and forgetting, thereby driving the observed differences in performance. 

24 

