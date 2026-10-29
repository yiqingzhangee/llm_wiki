# **LANGBRIDGE: Multilingual Reasoning Without Multilingual Supervision** 

**Dongkeun Yoon[1] Joel Jang[2] Sungdong Kim[1, 3] Seungone Kim[1, 4] Sheikh Shafayat[1] Minjoon Seo[1]** 

1KAIST 2 University of Washington 3 NAVER AI Lab 4 Carnegie Mellon University {dkyoon, minjoon}@kaist.ac.kr 

## **Abstract** 

We introduce LANGBRIDGE, a _zero-shot_ approach to adapt language models for multilingual reasoning tasks without multilingual supervision. LANGBRIDGE operates by “bridging” two models, each specialized in different aspects: (1) one specialized in understanding multiple languages (e.g., mT5 encoder) and (2) one specialized in reasoning (e.g., Orca 2). LANGBRIDGE connects the two models by introducing minimal trainable parameters between them. Despite utilizing only English data for training, LANGBRIDGE considerably enhances the performance of language models on low-resource languages across mathematical reasoning, code completion, logical reasoning, and commonsense reasoning. Our analysis suggests that the efficacy of LANGBRIDGE stems from the language-agnostic characteristics of multilingual representations. We publicly release our code and models.[1] 

## **1 Introduction** 

Language models (LMs) are known to exhibit inferior performance in solving reasoning tasks such as math or coding in low-resource languages (Shi et al., 2023; Qin et al., 2023). This tendency primarily stems from the fact that LMs are predominantly trained on corpora comprised of a few highresource languages (Touvron et al., 2023a,b). This results in low-resource languages being represented as long-tail knowledge (Lazaridou et al., 2021; Kandpal et al., 2023). 

Prior works have mainly approached this problem by adapting English-centric LMs to other languages through continual training on the target language (Marchisio et al., 2023; Oba et al., 2023; Zhu et al., 2023; Kew et al., 2023). However, scaling this approach to a large number of languages is challenging, as it requires targeted training corpora for each language. This issue is particularly 

> 1github.com/kaistAI/LangBridge 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision_images/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision.pdf-0001-10.png)


**----- Start of picture text -----**<br>
60.3<br>AVG HRL URL 58.0<br>51.8 52.3 53.3<br>48.8<br>45.2<br>42.8<br>40.5<br>34.4<br>6.0<br>3.9<br>MetaMath MetaMath- LB MetaMath MetaMath- LB<br>7B (7B + 2B) 13B (13B + 2B)<br>**----- End of picture text -----**<br>


Figure 1: MGSM accuracy (%) of MetaMath models and models aligned with mT5-XL encoder (2B) via LANGBRIDGE ( **LB** ). In addition to the average (AVG) accuracy, we also report the average accuracy of highresource languages (HRL) and underrepresented languages (URL) classified by Shi et al. (2023). 

pronounced for LMs such as MetaMath (Yu et al., 2023) and Orca 2 (Mitra et al., 2023), which have undergone continuous domain-specific adaptation from Llama 2 (Touvron et al., 2023b). These specialized, domain-specific datasets are typically in English, complicating multilingual support for the underlying LM. 

In this paper, we introduce LANGBRIDGE, a novel approach that adapts LMs to solve multilingual reasoning tasks without _explicitly_ training on multilingual data. Inspired from the multimodal literature that integrates two independently pretrained modalities (Alayrac et al., 2022; Li et al., 2023a; Merullo et al., 2023; Liu et al., 2023b; Bavishi et al., 2023), we leverage the encoder from mT5 (Xue et al., 2021) and introduce a small number of trainable parameters between the encoder and the target LM. Most importantly, our approach does not require multilingual supervision and solely relies on English data while generalizing to multiple languages during test time, resembling zero-shot cross-lingual transfer (Pires et al., 2019; Conneau 

et al., 2020; Xue et al., 2021). 

We demonstrate the effectiveness of LANGBRIDGE by applying our method to LMs specialized in diverse reasoning tasks of mathematical reasoning, code completion, logical reasoning. Our empirical results show LANGBRIDGE substantially enhances the multilingual reasoning performance of LMs. For example, LANGBRIDGE applied to MetaMath-13B leveraging mT5-XL encoder (2.2B) boosts the average accuracy on MGSM (Shi et al., 2023) from 40.5% to 53.5%, matching the performance of PaLM-540B (Chowdhery et al., 2023), which stands at 51.3%. We observe LANGBRIDGE also significantly boosts LM performance on reasoning datasets that require intrinsic linguistic understanding such as specific subtasks of Big-Bench Hard (Suzgun et al., 2023) and XCOPA (Ponti et al., 2020). 

We hypothesize that the effectiveness of LANGBRIDGE is anchored in the language-agnostic characteristics of multilingual representations (Pires et al., 2019; Libovický et al., 2020). By mapping these representations to the target LM’s input space, we conjecture that the LM is able to grasp the semantics of these representations. Since these representations are language-neutral, understanding them allows the LM to become less dependent on the specific language of the input, thereby enabling it to tackle tasks in languages it rarely encountered during pretraining. Our empirical analysis of LANGBRIDGE, using principal component analysis (PCA) and qualitative methods, supports this hypothesis. 

## **2 Related Work** 

## **2.1 English-centric Language Models** 

Previous works have enhanced the reasoning capabilities of LMs in mathematical reasoning, code generation, and logical reasoning (Mukherjee et al., 2023; Azerbayev et al., 2023; Yu et al., 2023; Rozière et al., 2023; Mitra et al., 2023). However, the majority of these LMs are derived from Englishcentric LMs (Touvron et al., 2023a,b) and adapted with English domain-specific datasets. Consequently, these LMs inherit limited proficiency in low-resource languages, which results in a significant performance discrepancy between highresource and low-resource languages. The motivation of our proposed method, LANGBRIDGE, is to narrow this gap. 

## **2.2 Zero-shot Cross-lingual Transfer** 

Multilingual models demonstrate remarkable zeroshot cross-lingual transfer capability (Conneau et al., 2020; Xue et al., 2021; FitzGerald et al., 2023). This ability significantly aids the lowresource language community, allowing multilingual models to handle tasks across a wide range of languages after being finetuned on high-resource languages. Our method, which leverages multilingual models, exhibits a similar ability, as it can enhance the reasoning capabilities of LMs across multiple languages while solely relying on English data during adaptation. 

## **2.3 Aligning Pretrained Representations** 

The concept of combining independently pretrained representations has been widely explored in the realm of cross-modal alignment (Alayrac et al., 2022; Li et al., 2023a; Merullo et al., 2023; Liu et al., 2023b; Bavishi et al., 2023). These works focus on aligning visual encoder models with LMs to facilitate the visual comprehension abilities of LMs. In a concurrent work, Bansal et al. (2024) align two large language models to augment each other’s capabilities. One of their experiments showcases that aligning a multilingual LM with another LM can lead to improved performance in multilingual tasks. However, in contrast to our method, their approach employs multilingual supervision for aligning. 

## **3 LANGBRIDGE** 

## **3.1 Hypothesis** 

Previous works show that representations of multilingual encoder models are moderately languageagnostic (or language-neutral) out-of-the-box, facilitating zero-shot cross-lingual transfer (Pires et al., 2019; Libovický et al., 2020). Based on this premise, we hypothesize that by aligning a multilingual encoder model to the space of an LM, the LM will be able to understand the semantics of the languages supported by the multilingual encoder _without_ training on a large set of languages. 

To test this hypothesis, we align multilingual encoder models with LMs using only English corpora ( **Left** of Figure 2). Then, we evaluate the aligned models using non-English inputs to determine if the LMs exhibit enhanced capabilities in multilingual tasks ( **Right** of Figure 2). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision_images/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Align With English Data Test With Multilingual Data<br>Angela has sold off 24/4 = 6 … Kristin invited 16 friends, so …<br>Linear Layer The answer is: 12 Linear Layer The answer is: 24<br>Multilingual Multilingual<br>Language Model Language Model<br>Encoder Encoder<br>Angela has a collection of 24 … … ప�శ�: ����� ���న�� ��� … …<br>How many does she have left? ఆ� ఎంత ఖ�� ���ం�?<br>**----- End of picture text -----**<br>


Figure 2: Overview of LANGBRIDGE. **Left** : A multilingual encoder with an added linear layer is aligned with the target language model using English data. We keep the language model frozen, whereas the linear layer is trainable. The multilingual encoder is trainable when adapting pretrained LMs and frozen when adapting finetuned LMs. **Right** : In test time, a LANGBRIDGE model can effectively solve multilingual reasoning tasks. 

## **3.2 Model Architecture** 

## **4 Main Experiments** 

Building on the findings of previous works, where effective cross-modal understanding has been achieved by mapping representations from nonlinguistic modalities to the soft prompts (Lester et al., 2021) of LMs (Merullo et al., 2023; Liu et al., 2023b; Bavishi et al., 2023), LANGBRIDGE maps the final hidden states of multilingual encoders to the soft prompts of LMs to attain multilingual understanding of the target LM. Following the studies above, we adopt the setting of using a single linear layer as the mapping function and append one trainable token to the end of the soft prompt as an [EOS] (end of sequence) token. Therefore, given the input tokens (padded if necessary) of the encoder **X** _enc_ , the derived soft prompt **H** _enc_ is equivalent in sequence length to **X** _enc_ + 1, and has the same dimensionality as the hidden state of the language model. Any tokens in **H** _enc_ that originate from padding tokens of **X** _enc_ are masked for the target LM. We ablate the effect of using more complex architectures in Appendix D. 

Language modeling objective of LANGBRIDGE resembles that of the “prefix LM” explored by Raffel et al. (2020), as the encoder input tokens **X** _enc_ can be interpreted as prefix tokens upon which the target tokens **X** _lm_ is conditioned. Formally, given the encoder input tokens **X** _enc_ , the language modeling likelihood of the target tokens **X** _lm_ is denoted as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision_images/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision.pdf-0003-06.png)


where _L_ is the sequence length of **X** _lm_ , and _xi_ is the _i_ th token of **X** _lm_ . 

## **4.1 Overview** 

We select four task categories for our experiments: (1) **mathematical reasoning** , (2) **code completion** , (3) **logical reasoning** , and (4) **commonsense reasoning** .[2] For each task category, we apply LANGBRIDGE to LMs specialized in reasoning derived from Llama 2 (Touvron et al., 2023b), such as MetaMath or Orca 2.[3] We evaluate the models on existing multilingual benchmarks (e.g., MGSM) or translated English benchmarks (e.g., translations of HumanEval (Chen et al., 2021)). As the evaluation tasks necessitate both multilingual understanding abilities and advanced reasoning capabilities, this complexity poses a significant challenge for general multilingual LMs and English-centric LMs specialized in reasoning. On the contrary, models aligned with LANGBRIDGE could take advantage of both. 

Since Touvron et al. (2023b) disclose the language distribution of the pretraining data of Llama 2, this enables us to identify which languages are underrepresented in LMs initialized from Llama 2 weights. Throughout the paper, we classify a language as underrepresented if it comprises less than 0.1% of Llama 2’s pretraining data. 

In all of our experiments, we use the encoders of mT5 (Xue et al., 2021) as the multilingual encoders due to their availability across a wide range of parameters,[4] and their support for longer input sequences compared to other multilingual encoder 

> 2Experiments on commonsense reasoning can be found in Appendix A.1. 

> 3We test LANGBRIDGE on general-domain LMs (Llama 2 and Mistral 7B (Jiang et al., 2023)) in Appendix C. 

> 4270M (Small), 470M (Base), 820M (Large), 2.2B (XL) and 6.7B (XXL). 

models.[5] Specifically, we use the “LM adaptated” checkpoints from Vu et al. (2022). We align target LMs of 7B parameters with mT5-XL encoder to adapt 9B-sized models and 13B-sized LMs with mT5-XL encoder and mT5-XXL encoder to obtain 15B and 20B models, respectively. As LANGBRIDGE adds a considerable amount of additional parameters to the target LM, we conduct an inference throughput analysis in Appendix B. We also ablate the effect of the encoder parameter size and encoder model type in Appendix D. 

We use the original continual training data of the LM for LANGBRIDGE when accessible (e.g., MetaMathQA for MetaMath). If unavailable, we opt for the closest publicly accessible dataset (e.g., OpenOrca (Lian et al., 2023) for Orca 2). In all our experiments, we fix the size of the training dataset to 200,000 instances. However, our ablation study on the effect of the training dataset size in Appendix D suggests LANGBRIDGE in practice may require much less data. We maintain the language model frozen throughout the alignment process for efficiency. We also freeze the encoder (except the embedding layer) for aligning finetuned LMs, whereas for pretrained LMs, we keep the encoder trainable.[6] In Appendix D, we provide further explanations for these choices and ablate the effects of freezing. We align the models by training on the prefix LM objective described in Section 3.2. In our preliminary experiments, we find that training on various lengths of **X** _enc_ is necessary to ensure robustness on inference time, as the language model is exposed to diverse lengths of **H** _enc_ . 

We use a maximum input length ( **X** _enc_ ) of 1,024 and a maximum target length ( **X** _lm_ ) of 128 for training. For unlabeled data, we randomly vary the input length within the 1,024 window to introduce the LM to various lengths of **H** _enc_ . For labeled data, the data naturally comes in diverse input lengths. On a machine equipped with four A100 80GB GPUs, training a 9B model takes less than four hours when the encoder layers are frozen, and under five hours when the entire encoder is trainable. In our main experiments, the setting where the encoder is fully trainable in the 20B models results in the maximum training time, which is ap- 

> 5mT5 was trained on input size of 1024 tokens, but can take longer sequences due to its use of relative position embeddings (Shaw et al., 2018). 

> 6We define _finetuned_ LMs as LMs trained on labeled corpora and _pretrained_ LMs as LMs trained on unlabeled corpora. 

proximately ten hours. Further training details are available in Appendix F. 

## **4.2 Mathematical Reasoning** 

## **4.2.1 Experimental Setup** 

**Evaluation Datasets MGSM** (Shi et al., 2023) comprises grade school math word problems in 11 typologically diverse languages, human translated from a sample of GSM8K (Cobbe et al., 2021). For evaluating pretrained LMs, we adopt the crosslingual transfer chain-of-thought (CoT) reasoning (Wei et al., 2022) setting (NATIVE-EXEMPLARS + EN-COT) from Shi et al. (2023), where the few-shot exemplars are given in the target language, but the CoT rationales to solve the exemplars are provided in English. For finetuned LMs, we evaluate in zeroshot[7] setting. Additional evaluation on MSVAMP (Chen et al., 2023) is available in Appendix A. 

**Language Models Llemma** (Azerbayev et al., 2023) is a set of LMs for mathematics, continually pretrained from Code Llama (Rozière et al., 2023) on Proof-Pile-2, a mixture of scientific papers, web data containing mathematics, and mathematical code. **MetaMath** (Yu et al., 2023) was finetuned from Llama 2 (Touvron et al., 2023b) on MetaMathQA, a mathematical dataset based on GSM8K and MATH (Hendrycks et al., 2021b). As both Proof-Pile-2 and MetaMathQA are publicly available, we apply LANGBRIDGE using samples of their respective training datasets. 

**Baselines Llama 2** (Touvron et al., 2023b) is an English-centric LM in which 89.7% of the pretraining data consists of English but has shown considerable performance on non-English languages (Lai et al., 2023). **mT5**[8] (Xue et al., 2021), **XGLM** (Lin et al., 2022), and **BLOOM** (Scao et al., 2022) are massively multilingual LMs. **MathOctopus** (Chen et al., 2023) is an LM for multilingual mathematical reasoning. It was initialized from Llama 2 and finetuned on translations of the GSM8K dataset across ten languages. The ten languages seen by MathOctopus overlap with the 11 languages included in MGSM, except Telugu. We use their best-performing checkpoints, xRFT-MathOctopus _[P]_ , which were further enhanced by data augmentation through rejection sampling (Yuan et al., 2023). We also re- 

> 7Here, the term _zero-shot_ refers to the lack of few-shot examples. 

> 8 We use the language model checkpoint from Vu et al. (2022). 

||**AVG**<br>**HRL**<br>**URL**|**EN**<br>**DE**<br>**FR**<br>**ES**<br>**RU**<br>**ZH**<br>**JA**|**TH**<br>**SW**<br>**BN**<br>**TE**|
|---|---|---|---|
|||||
|Lang. Freq. (Llama 2, %)<br>-<br>-<br>-<br>89.7<br>0.17<br>0.16<br>0.13<br>0.13<br>0.13<br>0.10<br>LESSTHAN0.005||||
|FEW-SHOTCROSS-LINGUALCOT||||
|||||
|Llama 2-7B<br>XGLM-7.5B<br>mT5-XXL (13B)<br>BLOOM-7.1B<br>BLOOM-7.1B-PP2<br>PaLM-540B|9.1<br>12.1<br>3.9<br>1.5<br>1.6<br>1.2<br>2.9<br>3.5<br>2.0<br>2.4<br>2.6<br>2.0<br>2.3<br>2.5<br>1.9<br>51.3<br>52.3<br>46.8|15.2<br>11.6<br>13.2<br>11.2<br>11.6<br>11.2<br>10.8<br>0.4<br>1.6<br>1.2<br>1.6<br>2.0<br>2.8<br>1.6<br>3.6<br>2.4<br>4.0<br>3.6<br>2.8<br>3.6<br>4.4<br>3.6<br>1.2<br>3.6<br>2.4<br>2.0<br>3.2<br>2.0<br>4.8<br>1.2<br>2.0<br>2.0<br>1.6<br>4.0<br>1.6<br>62.4<br>53.6<br>51.2<br>58.0<br>55.6<br>46.0<br>49.6|7.2<br>5.2<br>3.2<br>0.0<br>2.0<br>0.4<br>1.2<br>1.2<br>2.8<br>1.2<br>3.2<br>0.8<br>0.0<br>2.4<br>2.8<br>2.8<br>0.8<br>2.8<br>2.0<br>2.0<br>49.6<br>44.4<br>46.4<br>46.8|
|||||
|Llemma-7B<br>Llemma-**LB**-9B|**21.6**<br>**29.9**<br>7.2<br>20.4<br>22.5<br>**16.7**|**44.8**<br>**27.2**<br>**33.2**<br>**29.2**<br>**26.0**<br>**26.4**<br>**22.4**<br>34.8<br>23.6<br>26.8<br>22.4<br>18.8<br>16.0<br>15.2|14.0<br>8.4<br>6.4<br>0.0<br>**20.8**<br>**17.6**<br>**12.4**<br>**16.0**|
|||||
|Llemma-34B<br>35.6<br>46.3<br>16.7<br>58.0<br>48.0<br>46.8<br>48.0<br>47.2<br>36.8<br>39.6<br>28.4<br>27.2<br>11.2<br>0.0||||
|ZERO-SHOTCOT||||
|||||
|MathOctopus-7B<br>MathOctopus-13B<br>BLOOM-7.1B-MM|37.1<br>42.7<br>27.2<br>42.9<br>48.6<br>32.9<br>16.7<br>21.7<br>7.8|51.6<br>40.0<br>38.4<br>47.2<br>42.4<br>44.0<br>35.6<br>50.8<br>49.2<br>50.4<br>52.8<br>47.2<br>52.4<br>37.2<br>41.2<br>19.6<br>24.4<br>26.8<br>9.6<br>21.2<br>9.2|39.2<br>31.6<br>37.2<br>0.8<br>44.4<br>40.4<br>46.4<br>0.4<br>0.8<br>15.6<br>6.8<br>8.0|
|||||
|MetaMath-7B<br>MetaMath-**LB**-9B|34.4<br>51.8<br>3.9<br>**48.8**<br>**52.3**<br>**42.8**|**64.8**<br>**57.6**<br>**55.6**<br>56.4<br>50.4<br>42.4<br>35.6<br>63.2<br>50.8<br>52.4<br>**58.0**<br>**56.4**<br>**45.2**<br>**40.0**|4.0<br>6.4<br>4.4<br>0.8<br>**50.4**<br>**43.2**<br>**42.8**<br>**34.8**|
|||||
|MetaMath-13B<br>MetaMath-**LB**-15B<br>MetaMath-**LB**-20B|40.5<br>**60.3**<br>6.0<br>53.5<br>58.0<br>45.2<br>**55.8**<br>58.7<br>**50.7**|**70.4**<br>**64.4**<br>**65.2**<br>**63.6**<br>**60.0**<br>50.8<br>**47.6**<br>67.6<br>63.6<br>61.6<br>63.2<br>**60.0**<br>48.0<br>42.0<br>66.4<br>64.0<br>64.0<br>60.4<br>58.8<br>**52.4**<br>45.2|4.8<br>11.6<br>6.8<br>0.8<br>52.8<br>41.6<br>50.0<br>36.4<br>**53.6**<br>**49.2**<br>**52.8**<br>**47.2**|



Table 1: Accuracy (%) on MGSM. Alongside average (AVG) accuracy, we also report average accuracy of highresource languages (HRL) and underrepresented languages (URL) classified by Shi et al. (2023). We include the language distribution of Llama 2 for reference. For pretrained models ( **Top** ), we prompt with 8-shot cross-lingual chain-of-thought (CoT) reasoning exemplars, except for PaLM-540B, for which we reference the 6-shot cross-lingual CoT performance reported by Shi et al. (2023). For finetuned models ( **Bottom** ), we evaluate zero-shot. The PP2 and MM suffixes denote models trained on Proof-Pile-2 and MetaMath, respectively. We compare LANGBRIDGE models ( **LB** ) to their original LMs and highlight the best-performing numbers in **bold** . 

port the performance of BLOOM models further trained on the training sets of LANGBRIDGE models: **BLOOM-Proof-Pile-2 (PP2)** and **BLOOMMetaMath (MM)** . This is done to confirm that the capabilities of LANGBRIDGE models are derived from the LMs’ inherent strength rather than solely from the training set utilized by LANGBRIDGE. Detailed hyperparameters for training the BLOOM models are available in Appendix F. We additionally report the performance of **PaLM** (Chowdhery et al., 2023) measured by Shi et al. (2023). Similar to Llama 2, PaLM was pretrained on English-heavy corpora. 

## **4.2.2 Results** 

Table 1 shows the evaluation results of baselines and LANGBRIDGE models on MGSM. We highlight six main observations. (1) Llama 2, Llemma, and MetaMath exhibit critical performance degradation across languages that are underrepresented in the training data of Llama 2. (2) Despite this, massively multilingual LMs underperform Llama 2, 

even in the context of underrepresented languages.[9] This disparity underscores the robust mathematical reasoning capabilities inherent in Llama 2 and absent in BLOOM, XGLM, and mT5. (3) LANGBRIDGE enhances the multilingual performance of Llemma and MetaMath, especially in underrepresented languages. Most notably, LANGBRIDGE is able to bring Llemma and MetaMath performance on Telugu (TE) from zero or near zero to a range comparable to other languages. (4) LANGBRIDGE may degrade performance on highresource languages, with Llemma- **LB** -9B’s English (EN) performance drop being particularly pronounced. We provide our speculations on the cause of this phenomenon in Section 6.1. (5) Mathematical reasoning capabilities of LANGBRIDGE models come from their original LMs, not their training data. This is evident from BLOOM-7.1B-PP2 and BLOOM-7.1B-MM underperforming Llemma- **LB** -9B and MetaMath- **LB** -9B, respectively, by a large margin. (6) Surprisingly, despite only be- 

9Note BLOOM models were not trained in German, Russian, Japanese, and Thai. 

ing trained on English math data, our MetaMath- **LB** models are competitive against MathOctopus models, which were finetuned on translations of GSM8K on ten out of 11 languages supported by MGSM. Most importantly, performance of MathOctopus models drop to near zero on Telugu (TE), an unseen language by MathOctopus. On the other hand, LANGBRIDGE models show robust performance on all 11 languages, suggesting that even without multilingual supervision LANGBRIDGE generalizes to the large scale of languages included in the multilingual pretraining of the encoders. 

Overall, LANGBRIDGE models demonstrate outstanding performance against baselines. LANGBRIDGE models vastly outperform similar-sized multilingual models, establishing LANGBRIDGE as a viable approach for developing mathematical reasoning models for low-resource languages. We provide an example of a CoT rationale generated by MetaMath- **LB** in Appendix G. 

## **4.3 Code Completion** 

## **4.3.1 Experimental Setup** 

**Evaluation Datasets** Leveraging the strong translation capabilities of GPT-4 (OpenAI, 2023), as demonstrated in the study by Jiao et al. (2023), we extend **HumanEval** (Chen et al., 2021), a set of hand-written programming problems, into five underrepresented languages: Swahili, Bengali, Punjabi, Telugu, and Urdu. We name the resulting dataset **HumanEval-MT** . We select the five languages among those with reported MMLU (Hendrycks et al., 2021a) performance in the technical report of GPT-4 while also being included in the pretraining corpora of BLOOM. This choice is made to acquire high-quality translations from GPT-4, and provide BLOOM-based baselines a level playing field. To guide GPT-4 to only translate the embedded natural language instruction of the docstring while not modifying the code segments, we prompt GPT-4 with two human-annotated examples.[10] Subsequently, the generated translations are executed in a Python interpreter environment to assert the absence of syntax errors. We provide translation quality estimation of HumanEval-MT in Appendix E. 

The examples within the HumanEval dataset often feature self-explanatory function names. This suggests that language models could potentially 

10We provide entire functions as inputs to GPT-4 to give it as much context as possible for accurate translation. 

complete the associated code segments successfully, even without accurately comprehending the natural language comments embedded within them. Therefore, we evaluate the models on _anonymized_ versions of each language set, wherein the target function names of the code segments are uniformly altered to “func”. An example of the anonymization and evaluation result on the original version is available in Appendix A. 

**Language Models Code Llama** (Rozière et al., 2023) is a family of models initialized from Llama 2 model weights and pretrained on a codeheavy dataset. In our experiments, we use Code Llama-Python models, which were further pretrained on a Python-heavy dataset. Since the datasets used to pretrain the Code Llama models are not publicly available, we sample from the Python subset of StarCoder data (Li et al., 2023b) as the training dataset for LANGBRIDGE. 

**Baselines** We use **Llama 2** and **BLOOM** models as the baselines. Both models contain code data within their pretraining corpora. Additionally, we report the performance of **BLOOM-StarCoder** , a BLOOM model continually pretrained on the sample of StarCoder data used to train LANGBRIDGE models. 

## **4.3.2 Results** 

Table 2 presents the Pass@1 scores on HumanEval and HumanEval-MT. Code Llama- **LB** models show consistent improvements over Code Llama across all underrepresented languages. Moreover, LANGBRIDGE models can match their larger baselines in terms of average scores: the 9B model slightly underperforms Code Llama-13B by 0.2%, while both the 15B and 20B models surpass Code Llama-34B. BLOOM trained on StarCoder data did not demonstrate noticeable improvements, reemphasizing that the strengths of LANGBRIDGE models predominantly stem from the capabilities of original LMs. 

## **4.4 Logical Reasoning** 

## **4.4.1 Experimental Setup** 

**Evaluation Datasets** We assess logical reasoning capabilities with **Big-Bench Hard (BBH)** (Suzgun et al., 2023) and **Big-Bench Hard Bengali (BBH-BN)** (Shafayat et al., 2024). BBH is a collection of challenging subtasks where the application of chain-of-thought (CoT) reasoning has the potential to enhance performance substantially. 

||**AVG**|**EN**<br>**SW**<br>**BN**<br>**PA**<br>**TE**<br>**UR**|
|---|---|---|
||||
|Llama 2-7B<br>BLOOM-7.1B<br>BLOOM-7.1B-StarCoder|4.6<br>4.9<br>4.5|9.8<br>3.7<br>3.0<br>3.0<br>3.0<br>4.9<br>7.3<br>2.4<br>5.5<br>4.3<br>6.1<br>3.7<br>6.7<br>3.0<br>4.3<br>4.3<br>3.7<br>4.9|
||||
|Code Llama-7B<br>Code Llama-**LB**-9B|16.9<br>**19.4**|**36.0**<br>14.0<br>14.6<br>10.4<br>9.8<br>16.5<br>31.7<br>**17.1**<br>**15.2**<br>**18.3**<br>**15.2**<br>**18.9**|
||||
|Code Llama-13B<br>Code Llama-**LB**-15B<br>Code Llama-**LB**-20B|19.6<br>**23.6**<br>23.5|40.2<br>15.2<br>17.7<br>12.2<br>12.8<br>19.5<br>**41.5**<br>20.1<br>**20.1**<br>19.5<br>**19.5**<br>**20.7**<br>36.0<br>**24.4**<br>**20.1**<br>**20.7**<br>18.9<br>**20.7**|
||||
|Code Llama-34B|22.7|43.9<br>17.1<br>23.8<br>15.9<br>12.8<br>22.6|



Table 2: Pass@1 scores on _anonymized_ versions of HumanEval and HumanEval-MT. The models were evaluated on zero-shot code completion using greedy decoding. We compare LANGBRIDGE ( **LB** ) models to their original checkpoints and highlight the best-performing numbers in **bold** . 

BBH-BN translates 14 of the 23 subtasks of BBH into Bengali. To facilitate meaningful comparison, we evaluate only on the 14 subtasks supported by BBH-BN for BBH.[11] 

**Language Models Orca 2** (Mitra et al., 2023) was finetuned on Llama 2 with a collection of datasets augmented with reasoning traces of GPT-4 as well as fully synthetic datasets created with GPT4. Orca 2 effectively improved the reasoning abilities of smaller LMs on complex tasks demanding advanced reasoning in zero-shot settings. As the training dataset of Orca 2 is not publicly available, we sample the training data for LANGBRIDGE from the OpenOrca dataset (Lian et al., 2023). OpenOrca follows the data distribution of the first iteration of Orca (Mukherjee et al., 2023). We employ CLD3[12] to filter any non-English data that mainly derives from translation datasets to ensure the _zero-shot_ setting of our experiments. Examples were included if their input text had a 99% or greater probability of being English, while their target text also had a 95% or greater chance of being English. A slightly lower threshold was adopted for the target text to not falsely filter single-word responses, which CLD3 exhibits lower confidence. 

**Baselines** In our evaluation of BBH, we assess whether Orca 2- **LB** models could acquire multilingual comprehension while retaining the zero-shot CoT capabilities of Orca 2. However, from our limited testing, we found that no existing open multilingual LMs could generate CoT reliably in a zero-shot setting. Consequently, they were not included as baselines. We do report the performance of **BLOOM-OpenOrca** , a BLOOM model further 

> 11List of the 14 subtasks is available in Appendix A. 12github.com/google/cld3 

||**EN**<br>**BN**|
|---|---|
|||
|BLOOM-7B-OpenOrca|35.8<br>31.2|
|||
|Orca 2-7B<br>Orca 2-**LB**-9B|**53.9**<br>36.7<br>46.9<br>**41.8**|
|||
|Orca 2-13B<br>Orca 2-**LB**-15B<br>Orca 2-**LB**-20B|**57.9**<br>41.7<br>55.2<br>**45.4**<br>53.1<br>**45.4**|



Table 3: Accuracy (%) on BBH (English) and BBH-BN (Bengali). We report the average accuracy across 14 subtasks. We compare LANGBRIDGE ( **LB** ) models to their original checkpoints and highlight the best-performing numbers in **bold** . 

trained on the same training set as Orca 2- **LB** . 

## **4.4.2 Results** 

Table 3 shows the average accuracy across the subtasks for BBH and BBH-BN. Orca 2- **LB** -9B model shows considerable improvement in BBH-BN, surpassing the larger Orca 2-13B model. We report the performances on each subtask in Table 12 and Table 13. Remarkably, LANGBRIDGE significantly increases the performance on BBH-BN subtasks that require intrinsic linguistic understanding such as CAUSAL JUDGEMENT and SNARKS. CAUSAL JUDGEMENT requires the model to comprehend a short story and determine how a typical human would answer a given question. For SNARKS, the model is given two nearly-identical sentences and is required to indentify which one contains sarcasm. This observation, together with the results from the evaluation on XCOPA (Ponti et al., 2020), a commonsense reasoning dataset, suggests that LANGBRIDGE models are robust in grasping and interpreting nuanced linguistic details. Evaluation 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision_images/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision.pdf-0008-00.png)


**----- Start of picture text -----**<br>
Orca 2-7B<br>50<br>EN<br>TE<br>25 SW<br>ES<br>0 DE<br>− 25<br>− 50<br>− 40 − 20 0 20 40 60<br>Orca 2- LangBridge -9B<br>20 EN<br>TE<br>10 SW<br>ES<br>0 DE<br>− 10<br>− 20<br>− 20 − 15 − 10 − 5 0 5 10<br>**----- End of picture text -----**<br>


Figure 3: First two principal components of pooled output representations obtained with 300 FLORES samples per language. Note that the scales of the two subplots differ. 

results on XCOPA are provided in Appendix A. Figure 8 shows an example of a LANGBRIDGE model correctly solving CAUSAL JUDGEMENT. 

## **5 Analysis of LANGBRIDGE** 

Based on the empirical evidence presented in the previous sections, we assert that LANGBRIDGE effectively enhances LMs’ capability to address multilingual tasks without multilingual training, especially for low-resource languages. This section presents two fundamental observations that further reinforce the hypothesis outlined in Section 3. 

## **5.1 PCA** 

The hypothesis primarily attributes the feasibility of LANGBRIDGE to the sufficiently languageagnostic representations of multilingual encoders. If the conjecture holds, given a LANGBRIDGE model, the LM’s output representation of the soft prompt **H** _enc_ should also exhibit language-agnostic characteristics. It stands to reason that the LM would not arbitrarily introduce additional languagespecific features to a language-neutral input derived from the multilingual encoder. 

Therefore, to verify if the LANGBRIDGE models truly have language-agnostic output representations, we apply PCA to the mean pooled output representations from a LANGBRIDGE model and compare them with those from the original Englishcentric LM. Figure 3 shows the first two principal 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision_images/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision.pdf-0008-08.png)


**----- Start of picture text -----**<br>
Model Input<br>[ ⋯ ]<br>�কান ব�ব��ট ব��া�ক?  (Which statement is sarcastic?)<br>Options:<br>(A) �যেহত� কেপ�ােরট িনয়�ণ মােন �ত� ভাষণ,<br>এ�ট �কৃতপে� গণতে�র জন� এক�ট বড় জয়<br>(Since corporate restriction equals free speech<br> this is actually a big win for democracy)<br>(B) �যেহত� কেপ�ােরট মািন মােন �ত� ভাষণ,<br>এ�ট �কৃতপে� গণতে�র জন� এক�ট বড় জয়<br>(Since corporate money equals free speech<br>this is actually a big win for democracy)<br>Model Output<br>[ ⋯ ]<br>Step 2: Analyze the given options.<br>(A) 因为企业主控制意味着独立演讲，<br>这实际上是民主的一个巨大胜利 "Accidental<br>(B) 因为企业主钱意味着独立演讲， Translations"<br>这实际上是民主的一个巨大胜利<br>Step 3: Translate the options to English.<br>(A) Since corporate control means independent speech,<br>this is actually a huge victory for democracy<br>(B) Since corporate money means independent speech,<br>this is actually a huge victory for democracy<br>[ ⋯ ]<br>Final answer: (B) 因为企业主钱意味着独立演讲，<br>这实际上是民主的一个巨大胜利<br>**----- End of picture text -----**<br>


Figure 4: Example of accidental translation of an Orca 2- LANGBRIDGE model prompted with the SNARK subset of BBH-BN. Portions of the input prompt and several rational steps in the output are truncated for brevity. Translations are provided in (blue with parenthesis) wherever required. 

components of pooled output representations obtained with FLORES (Goyal et al., 2022; Costajussà et al., 2022), a parallel corpus. For Orca 2, high-resource languages, English (EN), Spanish (ES), and German (DE), are mapped closely together. Underrepresented languages, Telugu (TE) and Swahili (SW), exhibit a more distant mapping in the representation space, forming three clusters.[13] Conversely, for Orca 2-LANGBRIDGE, all languages are mapped into a single cluster, indicating that the representations of **H** _enc_ maintain a relatively language-neutral status. 

## **5.2 Accidental Translations** 

Figure 4 illustrates an example of “accidental translation” (Xue et al., 2021) by the Orca 2- LANGBRIDGE-15B model. Despite being given 

13Note that Flores Swahili is in Latin script, the same as the three high-resource languages. 

the options in Bengali, the LANGBRIDGE model perceives the options as Chinese and recites them in Chinese. With Bengali as input, accidental translation in a third language other than English or Bengali suggests that multiple languages may have similar representations in **H** _enc_ (Li and Murray, 2023). Nonetheless, outputs in languages other than English are uncommon for Orca 2-LANGBRIDGE models. We conduct a qualitative analysis on the CoTs generated by Orca 2-LANGBRIDGE-15B for BBH-BN SNARKS subtask, which we find is the combination with the most frequent accidental translations. Out of 178 CoT rationales generated, only eight examples contained accidental translations in Chinese, Danish, Hindi, Japanese, Marathi, and Turkish, with one or two examples per language. Additionally, seven examples had Bengali in their CoT rationales. The relatively high frequency of Bengali indicates that **H** _enc_ does not exhibit a perfectly language-agnostic behavior, and for some examples, the LM could discern the input language as Bengali. This is not ideal, as it suggests that the LM had to comprehend the input in Bengali, a language in which the LM lacks proficiency. We believe that LANGBRIDGE performance can be further enhanced by relieving the _zero-shot_ constraint and adapting the mT5 encoder to have enhanced language-neutrality (Reimers and Gurevych, 2020; Feng et al., 2022) prior to alignment with the LM. However, we leave this exploration for future study. 

## **6 Discussion** 

## **6.1 HRL Performance Degradation** 

In some cases, LANGBRIDGE models exhibit performance degradation in high-resource languages compared to their target LMs. While we leave further investigation for future studies, we outline some conjectures regarding potential causes for these performance declines. 

- Given that the target LM is already proficient in a certain language, the text representations derived from the soft prompts may be more challenging for the LM to interpret compared to those from the LM’s native embedding layer. 

- Due to limited computing budget, LANGBRIDGE models were trained with maximum input sequence length of 1024 ( **X** _enc_ ) and maximum output sequence length of 128 ( **X** _lm_ ). This is considerably shorter than what 

the original language models were trained on, especially for language models trained solely on unlabeled data such as Llemma. 

- We fixed the training hyperparameters for all of the LANGBRIDGE models regardless of the target LM for the sake of consistency and to save computing budget. This could have been suboptimal for some LANGBRIDGE models. 

- For Code Llama and Orca 2, their original training corpora are inaccessible. Therefore, we opt for the closest alternative as stated in Section 4.1. Nonetheless, a minor distribution shift during LM training and LANGBRIDGE training is inevitable. 

## **6.2 Multilingual CoT Capabilities?** 

As LANGBRIDGE solely utilizes English data for training, it is anticipated that the intermediate reasoning steps, or CoT, of LANGBRIDGE models would predominantly be in English. Given that LANGBRIDGE targets English-centric LMs specialized in reasoning, we conjecture that English CoT is one of the key components that contribute to the competitive performance of LANGBRIDGE models. This claim is supported by the findings of Shi et al. (2023), where they observe English CoT consistently leads to competitive results, and suggests that English CoT serves as a "useful baseline for future multilingual reasoning work". 

While we believe incorporating non-English data into the LANGBRIDGE training data could induce multilingual CoT capabilities in LANGBRIDGE models, this approach does not align well with the original motivation of this work, which is to avoid the need for collecting multilingual reasoning data. 

## **7 Conclusion** 

In this paper, we propose LANGBRIDGE, a simple yet effective method of extending the capabilities of LMs to solve multilingual reasoning tasks without using multilingual supervision. We show that LANGBRIDGE is surprisingly effective for enhancing multilingual reasoning capabilities for low-resource languages. Additionally, we provide analysis that indicates the effectiveness of LANGBRIDGE is due to the language-agnostic nature of the multilingual representations. We hope our findings benefit the low-resource language users and spur further research advancing the development of LMs inclusive of the entire global community. 

## **Limitations** 

As LANGBRIDGE solely utilizes English data for training, LANGBRIDGE models may not be proficient in generating text in languages other than English. Although LANGBRIDGE successfully narrows the performance gap between high-resource and low-resource languages, a noticeable performance gap remains. Also, while multilingual representations are known to have language-agnostic characteristics to some degree, our analysis and previous works suggest that there is room for improvements (Libovický et al., 2020; Feng et al., 2022). While LANGBRIDGE has the potential to generalize to all languages supported by the multilingual encoder, the extent to which LANGBRIDGE enhances the reasoning capability of a specific language depends on two key factors: (1) The original proficiency of the LM in that particular language. (2) The proficiency of the encoder model in that particular language. 

## **Ethical Considerations** 

While we share LANGBRIDGE models for open access, their terms for use or license adhere to those of the original LMs. The training datasets utilized in our research is primarily sourced from academic materials. As a result, we assess that the datasets likely contain a relatively fewer examples featuring offensive or personal information. Nevertheless, it is important to acknowledge that such content may be still present within the datasets. 

## **Acknowledgements** 

This work was partly supported by KAIST-NAVER Hypercreative AI Center (60%) and Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.2022-0-00113, Developing a Sustainable Collaborative Multi-modal Lifelong Learning Framework, 20%; No.2021-0-02068, Artificial Intelligence Innovation Hub, 20%). 

## **References** 

- Sweta Agrawal, Nikita Mehandru, Niloufar Salehi, and Marine Carpuat. 2022. Quality estimation via backtranslation at the WMT 2022 quality estimation task. In _Proceedings of the Seventh Conference on Machine Translation (WMT)_ , pages 593–596, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics. 

- Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. 2022. Flamingo: a visual language model for few-shot learning. _Advances in Neural Information Processing Systems_ , 35:23716–23736. 

- Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, and Sean Welleck. 2023. Llemma: An open language model for mathematics. _arXiv preprint arXiv:2310.06786_ . 

- Rachit Bansal, Bidisha Samanta, Siddharth Dalmia, Nitish Gupta, Sriram Ganapathy, Abhishek Bapna, Prateek Jain, and Partha Talukdar. 2024. LLM augmented LLMs: Expanding capabilities through composition. In _The Twelfth International Conference on Learning Representations_ . 

- Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sa˘gnak Ta¸sırlar. 2023. Introducing our multimodal models. 

- Loubna Ben Allal, Niklas Muennighoff, Logesh Kumar Umapathi, Ben Lipkin, and Leandro von Werra. 2022. A framework for the evaluation of code generation models. https://github.com/bigcode-project/ bigcode-evaluation-harness. 

- Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ . 

- Nuo Chen, Zinan Zheng, Ning Wu, Ming Gong, Yangqiu Song, Dongmei Zhang, and Jia Li. 2023. Breaking language barriers in multilingual mathematical reasoning: Insights and observations. 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2023. Palm: Scaling language modeling with pathways. _Journal of Machine Learning Research_ , 24(240):1–113. 

- Hyung Won Chung, Xavier Garcia, Adam Roberts, Yi Tay, Orhan Firat, Sharan Narang, and Noah Constant. 2023. Unimax: Fairer and more effective language sampling for large-scale multilingual pretraining. In _The Eleventh International Conference on Learning Representations_ . 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ . 

- Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 8440– 8451, Online. Association for Computational Linguistics. 

- Marta R Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, et al. 2022. No language left behind: Scaling human-centered machine translation. _arXiv preprint arXiv:2207.04672_ . 

- Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. 2022. Language-agnostic BERT sentence embedding. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 878–891, Dublin, Ireland. Association for Computational Linguistics. 

- Jack FitzGerald, Christopher Hench, Charith Peris, Scott Mackie, Kay Rottmann, Ana Sanchez, Aaron Nash, Liam Urbach, Vishesh Kakarala, Richa Singh, Swetha Ranganath, Laurie Crist, Misha Britan, Wouter Leeuwis, Gokhan Tur, and Prem Natarajan. 2023. MASSIVE: A 1M-example multilingual natural language understanding dataset with 51 typologically-diverse languages. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 4277–4302, Toronto, Canada. Association for Computational Linguistics. 

- Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2023. A framework for few-shot language model evaluation. 

- Naman Goyal, Cynthia Gao, Vishrav Chaudhary, PengJen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio Ranzato, Francisco Guzmán, and Angela Fan. 2022. The Flores-101 evaluation benchmark for low-resource and multilingual machine translation. _Transactions of the Association for Computational Linguistics_ , 10:522–538. 

- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021a. Measuring massive multitask language understanding. _Proceedings of the International Conference on Learning Representations (ICLR)_ . 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021b. Measuring mathematical problem solving with the math dataset. _NeurIPS_ . 

- Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. 

- Wenxiang Jiao, Wenxuan Wang, Jen tse Huang, Xing Wang, Shuming Shi, and Zhaopeng Tu. 2023. Is chatgpt a good translator? yes with gpt-4 as the engine. 

- Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. 2023. Large language models struggle to learn long-tail knowledge. 

- Tannon Kew, Florian Schottmann, and Rico Sennrich. 2023. Turning english-centric llms into polyglots: How much multilinguality is needed? 

- Viet Dac Lai, Chien Van Nguyen, Nghia Trung Ngo, Thuat Nguyen, Franck Dernoncourt, Ryan A. Rossi, and Thien Huu Nguyen. 2023. Okapi: Instructiontuned large language models in multiple languages with reinforcement learning from human feedback. 

- Angeliki Lazaridou, Adhiguna Kuncoro, Elena Gribovskaya, Devang Agrawal, Adam Liska, Tayfun Terzi, Mai Gimenez, Cyprien de Masson d’Autume, Tomas Kocisky, Sebastian Ruder, Dani Yogatama, Kris Cao, Susannah Young, and Phil Blunsom. 2021. Mind the gap: Assessing temporal generalization in neural language models. 

- Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 3045–3059, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 

- Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023a. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. 

- Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. 2023b. Starcoder: may the source be with you! _arXiv preprint arXiv:2305.06161_ . 

- Tianjian Li and Kenton Murray. 2023. Why does zeroshot cross-lingual generation fail? an explanation and a solution. In _Findings of the Association for Computational Linguistics: ACL 2023_ , pages 12461–12476, Toronto, Canada. Association for Computational Linguistics. 

- Wing Lian, Bleys Goodson, Eugene Pentland, Austin Cook, Chanvichet Vong, and "Teknium". 2023. Openorca: An open dataset of gpt augmented flan reasoning traces. https://https://huggingface. co/Open-Orca/OpenOrca. 

- Jindˇrich Libovický, Rudolf Rosa, and Alexander Fraser. 2020. On the language neutrality of pre-trained multilingual representations. In _Findings of the Association for Computational Linguistics: EMNLP 2020_ , pages 1663–1674, Online. Association for Computational Linguistics. 

- Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O’Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, and Xian Li. 2022. Few-shot learning with multilingual language models. 

- Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023a. Improved baselines with visual instruction tuning. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023b. Visual instruction tuning. 

- Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. 

- Kelly Marchisio, Patrick Lewis, Yihong Chen, and Mikel Artetxe. 2023. Mini-model adaptation: Efficiently extending pretrained models to new languages via aligned shallow training. In _Findings of the Association for Computational Linguistics: ACL 2023_ , pages 5474–5490, Toronto, Canada. Association for Computational Linguistics. 

- Jack Merullo, Louis Castricato, Carsten Eickhoff, and Ellie Pavlick. 2023. Linearly mapping from image to text space. 

- Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, Hamid Palangi, Guoqing Zheng, Corby Rosset, Hamed Khanpour, and Ahmed Awadallah. 2023. Orca 2: Teaching small language models how to reason. 

- Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. 2023. Orca: Progressive learning from complex explanation traces of gpt-4. 

- Miyu Oba, Tatsuki Kuribayashi, Hiroki Ouchi, and Taro Watanabe. 2023. Second language acquisition of neural language models. In _Findings of the Association for Computational Linguistics: ACL 2023_ , pages 13557–13572, Toronto, Canada. Association for Computational Linguistics. 

- OpenAI. 2023. Gpt-4 technical report. 

- Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics_ , pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics. 

- Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are NLP models really able to solve simple math word problems? In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 2080–2094, Online. Association for Computational Linguistics. 

- Telmo Pires, Eva Schlinger, and Dan Garrette. 2019. How multilingual is multilingual BERT? In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 4996–5001, Florence, Italy. Association for Computational Linguistics. 

- Edoardo Maria Ponti, Goran Glavaš, Olga Majewska, Qianchu Liu, Ivan Vuli´c, and Anna Korhonen. 2020. XCOPA: A multilingual dataset for causal commonsense reasoning. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 2362–2376, Online. Association for Computational Linguistics. 

- Maja Popovi´c. 2015. chrF: character n-gram F-score for automatic MT evaluation. In _Proceedings of the Tenth Workshop on Statistical Machine Translation_ , pages 392–395, Lisbon, Portugal. Association for Computational Linguistics. 

- Libo Qin, Qiguang Chen, Fuxuan Wei, Shijue Huang, and Wanxiang Che. 2023. Cross-lingual prompting: Improving zero-shot chain-of-thought reasoning across languages. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 2695–2709. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of Machine Learning Research_ , 21(140):1–67. 

- Nils Reimers and Iryna Gurevych. 2020. Making monolingual sentence embeddings multilingual using knowledge distillation. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 4512–4525, Online. Association for Computational Linguistics. 

- Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon. 2011. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In _2011 AAAI Spring Symposium Series_ . 

- Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. 2023. Code llama: Open foundation models for code. 

- Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, et al. 2022. Bloom: A 176b-parameter openaccess multilingual language model. _arXiv preprint arXiv:2211.05100_ . 

- Sheikh Shafayat, H M Quamran Hasan, Minhajur Rahman Chowdhury Mahim, Rifki Afina Putri, James Thorne, and Alice Oh. 2024. Benqa: A question answering and reasoning benchmark for bengali and english. 

- Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. 2018. Self-attention with relative position representations. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)_ , pages 464–468, New Orleans, Louisiana. Association for Computational Linguistics. 

- Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. 2023. Language models are multilingual chain-of-thought reasoners. In _The Eleventh International Conference on Learning Representations_ . 

- Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. 2023. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama. 

- Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei. 2023. Challenging BIG-bench tasks and whether chain-of-thought can solve them. In _Findings of the Association for Computational Linguistics: ACL 2023_ , pages 13003–13051, Toronto, Canada. Association for Computational Linguistics. 

catastrophic forgetting in zero-shot cross-lingual generation. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 9279–9300, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 

   - Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35:24824–24837. 

   - Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. mT5: A massively multilingual pre-trained text-to-text transformer. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 483–498, Online. Association for Computational Linguistics. 

   - Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2023. Metamath: Bootstrap your own mathematical questions for large language models. 

   - Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. 2023. Scaling relationship on learning mathematical reasoning with large language models. 

   - Wenhao Zhu, Yunzhe Lv, Qingxiu Dong, Fei Yuan, Jingjing Xu, Shujian Huang, Lingpeng Kong, Jiajun Chen, and Lei Li. 2023. Extrapolating large language models to non-english by aligning languages. 

   - Terry Yue Zhuo, Qiongkai Xu, Xuanli He, and Trevor Cohn. 2023. Rethinking round-trip translation for machine translation evaluation. In _Findings of the Association for Computational Linguistics: ACL 2023_ , pages 319–337, Toronto, Canada. Association for Computational Linguistics. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. Llama: Open and efficient foundation language models. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ . 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. _Advances in neural information processing systems_ , 30. 

- Tu Vu, Aditya Barua, Brian Lester, Daniel Cer, Mohit Iyyer, and Noah Constant. 2022. Overcoming 

## **A Additional Evaluation Results** 

## **A.1 XCOPA** 

Table 10 shows the evaluation results of Orca 2 and Orca 2-LANGBRIDGE models on **COPA** (Roemmele et al., 2011) and **XCOPA** (Ponti et al., 2020), commonsense reasoning datasets. COPA is available in English, while XCOPA extends COPA to 11 languages. We do not include other LMs as baselines, as COPA was included in the training set of Orca 2, making it challenging to draw meaningful comparisons. 

Despite reaching near-perfect accuracy for COPA, Orca 2 models’ performance drops close to random chance (50%) on some of the underrepresented languages of XCOPA. LANGBRIDGE successfully decreases this performance degradation, except for Quechua (QU). The discrepancy is likely due to Quechua not being included in the 101 languages covered by mT5. This observation reemphasizes that the large-scale linguistic proficiency of LANGBRIDGE models is primarily derived from mT5. 

## **A.2 MSVAMP** 

**MSVAMP** (Chen et al., 2023) is a multilingual grade school level math word problem dataset translated from SVAMP (Patel et al., 2021) to 10 languages. We only evaluate MSVAMP in a zero-shot setting, as no CoT rationale examples are provided with the dataset. 

Table 11 presents the evaluation results on MSVAMP. MetaMath and MathOctopus were not trained on SVAMP or MSVAMP, so MSVAMP can be seen as an out-of-domain test set to evaluate domain generalization (Chen et al., 2023). Performance of MetaMath- **LB** models indicates our models can generalize to out-of-domain test sets successfully. 

## **A.3 BBH** 

Table 12 and 13 showcase the complete results for the 14 subtasks of BBH and BBH-BN. The subtasks are: CAUSAL JUDGEMENT, DATE UNDERSTANDING, DISAMBIGUATION QA, FORMAL FALLACIES, LOGICAL DEDUCTION (3, 5 AND 7), NAVIGATE, PENGUINS IN A TABLE, REASONING ABOUT COLORED OBJECTS, SNARKS, SPORTS UNDERSTANDING, TEMPORAL SEQUENCES, and WEB OF LIES. 

Notably, Orca 2- **LB** models show noticeable performance degradation in DATE UNDERSTANDING. 

def greatest_common_divisor(a: int, b: int) -> int: """ Rudi kipengele kikubwa zaidi cha pamoja cha integers mbili a na b >>> greatest_common_divisor(3, 5) 1 >>> greatest_common_divisor(25, 15) 5 """ def func(a: int, b: int) -> int: """ Rudi kipengele kikubwa zaidi cha pamoja cha integers mbili a na b >>> func(3, 5) 1 >>> func(25, 15) 5 """ 

Table 4: Comparison between original ( **Top** ) and anonymized ( **Bottom** ) prompts of HumanEval-MT Swahili. 

From our qualitative analysis of the CoT, we observe that Orca 2- **LB** models frequently falsely assume an arbitrary date as the current date at the beginning of CoT (Figure 9), whereas the original Orca 2 models do not exhibit this behavior. Our exploration of the OpenOrca dataset reveals that examples often require the model to assume a specific current date. For example, an input text is given as “The current senate majority leader in the US is Chuck Schumer. Options: - yes - no”, and the target text contains “...Today’s date is October 12, 2021...”. As we do not have access to Orca 2’s original training dataset, a thorough ablation on the effect of such examples is challenging. Nevertheless, we speculate this problematic emergent behavior in Orca 2- **LB** models is partially due to the distribution shift of the training data from the original Orca 2 dataset to the OpenOrca dataset. 

## **A.4 HumanEval** 

Table 4 compares the original and anonymized prompts of HumanEval-MT. **Top** can be solved without comprehending Swahili whereas **Bottom** is not. Table 14 contains the evaluation results on the original (non-anonymized) version of HumanEval and HumanEval-MT. Compared to LANGBRIDGE models, Code Llama models show a sharper decline in performance when evaluated on the anonymized version, suggesting that they are less capable of comprehending natural text in another language. 

## **B Inference Throughput of LANGBRIDGE Model** 

We report the inference throughput of models in Table 5. We measure the total time to infer 500 instances of XCOPA. For each model, we repeat 

||Orca<br>2-7B<br>Orca 2-<br>**LB**-9B<br>Orca<br>2-13B|
|---|---|
|<br><br><br>||
|Average Time (sec)<br>Standard Deviation (sec)<br>GPU Memory (GBs)|44.46<br>59.23<br>102.84<br>0.13<br>0.29<br>0.48<br>13.48<br>16.97<br>25.07|



Table 5: Inference throughputs measured using 500 examples of XCOPA. 

five times and report the average and the standard deviation. Additionally, we include the GPU memory utilization. We test using a single A6000 GPU on an idle server. 

Results show Orca 2-LB-9B model requires considerably less compute and memory compared to Orca 2-13B, despite it outperforming Orca 2-13B in BigBenchHard Bengali (BBH-BN) and XCOPA. 

## **C General-domain Language Models** 

Table 15 probes the effect of LANGBRIDGE on general-domain English-centric LMs, **Llama 2** and **Mistral 7B** (Jiang et al., 2023), using MGSM. Consistent with the findings on specialized LMs, LANGBRIDGE enhances the performance of lowresource languages. As training corpora for Llama 2 and Mistral 7B are unavailable, we use a sample of the SlimPajama dataset (Soboleva et al., 2023) as the training set. 

## **D Ablation Studies** 

## **D.1 Freezing/Unfreezing** 

We strictly keep the embedding layers of mT5 trainable throughout our experiments as we extend the vocabulary and the embedding layer to incorporate whitespace characters. mT5 tokenizers do not have whitespace characters in their vocabularies, and their default behavior is to truncate any consecutive whitespaces to a single space. However, this could negatively affect understanding code or following instructions considering the frequent use of whitespaces as delimiters (\n, \t, and “four spaces”). Therefore, even when we freeze the encoder, we leave the embedding layer trainable for the added whitespace vocabulary. 

Table 6 presents the ablation study on the impact of parameter freezing during the alignment process. We apply LANGBRIDGE with mT5-XL (2B) encoder on multiple LMs while varying the trainable modules. Notably, freezing the encoder appears beneficial when adapting finetuned LMs (MetaMath and Orca), whereas it negatively af- 

|**Target LM**|**Target LM**|**Trainable**<br>**AVG Score**|**Trainable**<br>**AVG Score**|
|---|---|---|---|
|||Enc<br>LM||
|||MGSM||
|||||
|Llama 2-7B||�|9.6<br>**11.3**|
|||||
|Llemma-7B||�|14.4<br>**20.4**|
|||||
|MetaMath-7B||�|**48.8**<br>43.9|
|HUMANEVAL+ HUMANEVAL-MT||||
|||||
|Code Llama-7B|�||15.3<br>**19.4**|
|XCOPA||||
|||||
|Orca-7B|�<br>�||**76.6**<br>71.1<br>74.0|
|||||
|Orca-13B|�||**77.3**<br>65.1|



Table 6: Ablations on the effect of freezing the encoder and the LM during aligning of LANGBRIDGE leveraging mT5-XL encoder. �denotes the module is trainable (not frozen) during aligning. 

fects pretrained models (Llama, Llemma, and Code Llama). We speculate this divergence stems from differing entropy levels in the datasets: unlabeled corpora typically exhibit higher entropy than relatively well-formatted finetuning datasets. Consequently, we conjecture that for unlabeled data, keeping the encoder trainable enables the model to acclimate to the training data better. Nonetheless, we leave a thorough investigation for future research. Additionally, training the LM during the alignment phase does not improve performance. We hypothesize this is due to the training datasets being in-domain of the LMs. As such, the LMs may not be learning additional information from the data. 

## **D.2 Encoder Size** 

Figure 5 shows the effect of encoder size on LANGBRIDGE when applied to Orca 2-7B, measured with XCOPA. We test five different sizes of mT5 encoder: 270M (Small), 470M (Base), 820M (Large), 2.2B (XL) and 6.7B (XXL). We observe that LANGBRIDGE with the two smaller-sized encoders underperforms the base Orca 2-7B. Nonetheless, performance increases rapidly as the encoder size scales from 270M to 2.2B and saturates in the 2.2B to 6.7B range. These results highlight 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision_images/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision.pdf-0016-00.png)


**----- Start of picture text -----**<br>
75<br>70<br>Orca 2-7B<br>65<br>60<br>0 1 2 3 4 5 6 7<br># Encoder Parameters (Billions)<br>(%)<br>Accuracy<br>XCOPA<br>**----- End of picture text -----**<br>


Figure 5: XCOPA accuracy (%) of Orca 2-7B models adapted with LANGBRIDGE using five different sizes of mT5 encoder. The dotted line shows the original performance of the target LM. 

||**XCOPA (Acc. %)**|
|---|---|
|||
|Linear<br>MLP<br>Resampler|76.6<br>72.7<br>49.7|



Table 7: Ablations of different LANGBRIDGE architectures using Orca 2-7B and mT5-XL encoder. 

sampler” module that adopts the architecture of perceiver resampler employed by Flamingo (Alayrac et al., 2022). In contrast to the finding of Liu et al. (2023a), we find that using an MLP layer instead of a linear layer decreases the performance slightly. Using a resampler module results in random performance. 

## **D.5 Different Encoder Models** 

that scaling the encoder size past a certain point provides diminishing returns. 

## **D.3 Training Set Size** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision_images/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision.pdf-0016-08.png)


**----- Start of picture text -----**<br>
76<br>75<br>74<br>73<br>50 75 100 125 150 175 200<br># Training Examples (Thousands)<br>(%)<br>Accuracy<br>XCOPA<br>**----- End of picture text -----**<br>


Figure 6: XCOPA accuracy (%) of Orca 2-7B models adapted with LANGBRIDGE using five different sizes of training datasets. 

While we fixed the training set size to 200,000 in our main experiments, Figure 6 shows that XCOPA performance peaks on 120,000 training examples, which is 60% of our experiment setting. This suggests that in practice, LANGBRIDGE can be applied more efficiently. 

## **D.4 Different Architectures** 

Table 7 shows the ablations of different LANGBRIDGE architectures. Throughout the main experiment, we adopt a single linear layer to connect the encoder and the language model. We ablate the effect of using an MLP layer following the second iteration of LLaVa (Liu et al., 2023a), and a “re- 

||**XCOPA (Acc. %)**|
|---|---|
|||
|umT5-XL<br>umT5-XL (F)<br>XGLM-1.7B<br>XGLM-1.7B (F)|49.7<br>49.2<br>51.5<br>49.7|



Table 8: Ablations of different encoder models for LANGBRIDGE tested on Orca 2-7B model. (F) denotes that the encoder was frozen during alignment. 

Table 8 presents the results of using different encoder models. We test out umT5 (Chung et al., 2023), which improves upon mT5 to include a higher proportion of low-resource languages during pretraining. As umT5 does not have “LM adapted” checkpoints, unlike mT5, we use the original checkpoints. However, we find using umT5 results in random performance. Since umT5 has a nearly identical architecture to mT5, except that it has relative position bias for every Transformer layer (Vaswani et al., 2017) in contrast to the very first, we speculate that using the encoder of a non-LM Seq2Seq model resulted in failed alignment. We also test XGLM (Lin et al., 2022), a decoder LM, as an encoder, in which we also observe similar results. In both cases, freezing the encoder made no difference. We leave further investigation for future research. 

## **E Quality Estimation of HumanEval-MT** 

Table 9 presents the quality estimation for HumanEval-MT using backtranslation (Agrawal et al., 2022; Zhuo et al., 2023). We translate 

||**EN**|⟲**SW**<br>⟲**BN**<br>⟲**PA**<br>⟲**TE**<br>⟲**UR**|
|---|---|---|
||||
|CL-7B<br>CL-34B|36.0<br>43.9|35.4<br>35.4<br>36.6<br>36.6<br>37.8<br>36.6<br>46.3<br>42.1<br>42.1<br>45.7|
||||
|BLEU<br>crhF|-<br>-|88.3<br>90.5<br>93.4<br>90.5<br>96.1<br>89.9<br>93.1<br>94.6<br>93.1<br>96.5|



Table 9: Quality estimation of HumanEval-MT using backtranslations. CL prefix denotes Code Llama Pass@1 score. 

HumanEval-MT back to English and measure Code Llama Pass@1 scores. As explained in Section 4.3, we anonymize all function names. In addition, we report automatic evaluation metrics, BLEU (Papineni et al., 2002) and chrF (Popovi´c, 2015), against the original HumanEval. We use the docstrings only for calculating the metrics. 

Overall, Code Llama’s performance on the backtranslations matches the original HumanEval benchmark, with the exception of Code Llama34B’s performance on Swahili backtranslation. As Code Llama-7B achieves comparable results on the Swahili backtranslation, this suggests that for Swahili, GPT-4 may struggle with accurately translating complex examples that only larger models can solve. Lower automatic evaluation scores in Swahili further support the idea. 

languages with the examples as reference. Note that the examples were provided to guide GPT-4 to keep the format of the data intact and only translate the natural language, not necessarily to enhance the translation quality itself. 

## **G CoT Examples** 

In this section, we provide three examples of CoT generated by LANGBRIDGE models. Figure 7 is from the MetaMath-LANGBRIDGE-15B model prompted with an example from MGSM Telugu. Figure 8 and 9 are from Orca 2-LANGBRIDGE15B model prompted with examples from BBHBN CAUSAL JUDGEMENT and BBH-BN DATE UNDERSTANDING. We select these two subtasks from BBH-BN as LANGBRIDGE substantially increased the performance of Orca 2-13B for the former (+13 _._ 4%), but caused a considerable decrease for the latter ( _−_ 6 _._ 4%). We show a success case for the former and a failure case for the other. In Appedix A, we provide an explanation for the performance degradation in BBH-BN DATE UNDERSTANDING. 

## **F Experimental Details** 

For training LANGBRIDGE models, we leverage AdamW (Loshchilov and Hutter, 2019). We use a learning rate of 6e _−_ 4 for the linear layer and 2e _−_ 5 for the encoder. We do not use any learning rate scheduling. We use an effective batch size of 128. 

For further training BLOOM baselines, we keep the training configuration the same as LANGBRIDGE models with two exceptions: (1) Learning rate, in which we used a uniform value of 2e _−_ 5 across the entire model. (2) Sequence length, which was set to 1,152 (1 _,_ 024 + 128). 

For evaluations, we leverage code adapted from LM Evaluation Harness (version 0.3.0) (Gao et al., 2023) and Bigcode Evaluation Harness (version 0.1.0) (Ben Allal et al., 2022) packages and report single run results with fixed seed. We open source the evaluation code. 

For constructing HumanEval-MT, we prompt GPT-4 with human-translated examples. The two examples used for HumanEval-MT were translated to Korean by a native Korean author. GPT-4 was then prompted to translate HumanEval to the target 

||**EN**|**AVG**<br>**HRL**<br>**URL**|**ZH**<br>**IT**|**VI**<br>**ID**<br>**ET**<br>**HT**<br>**QU**<br>**SW**<br>**TA**<br>**TH**<br>**TR**|
|---|---|---|---|---|
||||||
|Lang. Freq. (Llama 2, %)<br>89.7<br>-<br>-<br>-<br>0.13<br>0.11<br>0.08<br>0.03<br>LESSTHAN0.005|||||
||||||
|Orca 2-7B<br>Orca 2-**LB**-9B|**98.0**<br>90.0|67.3<br>**86.4**<br>63.0<br>**76.6**<br>83.4<br>**75.1**|**85.6**<br>**87.2**<br>85.4<br>81.4|**83.4**<br>82.4<br>54.4<br>52.2<br>49.8<br>54.2<br>58.0<br>62.0<br>71.0<br>79.8<br>**84.6**<br>**78.2**<br>**74.4**<br>**50.8**<br>**74.4**<br>**78.0**<br>**78.4**<br>**77.6**|
||||||
|Orca 2-13B<br>Orca 2-**LB**-15B<br>Orca 2-**LB**-20B|**99.0**<br>92.0<br>92.0|73.7<br>**93.1**<br>69.4<br>77.3<br>84.5<br>75.7<br>**79.8**<br>86.3<br>**78.4**|**92.4**<br>**93.8**<br>85.2<br>83.8<br>86.2<br>86.4|**87.0**<br>**86.8**<br>66.4<br>61.0<br>49.8<br>65.8<br>63.8<br>67.6<br>76.4<br>83.4<br>83.8<br>80.6<br>74.8<br>50.4<br>72.2<br>77.2<br>**79.8**<br>79.2<br>83.6<br>85.4<br>**82.8**<br>**76.4**<br>**54.2**<br>**77.8**<br>**82.8**<br>**79.8**<br>**82.4**|



Table 10: Accuracy (%) on COPA and XCOPA. For XCOPA, we report the average accuracy across 11 languages. 

||**AVG**<br>**HRL**<br>**URL**|**EN**<br>**DE**<br>**FR**<br>**ES**<br>**RU**<br>**ZH**<br>**JA**|**TH**<br>**SW**<br>**BN**|
|---|---|---|---|
|||||
|Lang. Freq. (Llama 2, %)<br>-<br>-<br>-<br>89.7<br>0.17<br>0.16<br>0.13<br>0.13<br>0.13<br>0.10<br>LESSTHAN0.005||||
|||||
|MathOctopus-7B<br>MathOctopus-13B<br>BLOOM-7.1B-MM|39.2<br>41.5<br>34.0<br>45.1<br>47.2<br>40.0<br>24.9<br>29.5<br>14.1|39.8<br>42.4<br>44.0<br>43.3<br>41.6<br>40.4<br>38.7<br>44.8<br>46.7<br>50.6<br>49.9<br>47.6<br>47.1<br>43.9<br>40.7<br>28.8<br>33.8<br>32.0<br>27.7<br>27.7<br>15.7|35.1<br>29.7<br>37.2<br>40.0<br>36.3<br>43.6<br>3.0<br>21.2<br>18.0|
|||||
|MetaMath-7B<br>MetaMath-**LB**-9B|47.8<br>**61.1**<br>16.9<br>**52.0**<br>54.9<br>**45.1**|**66.3**<br>**63.5**<br>**64.1**<br>**64.9**<br>**60.3**<br>**55.0**<br>**53.4**<br>60.6<br>58.1<br>57.0<br>56.9<br>55.8<br>50.4<br>45.5|19.5<br>16.8<br>14.4<br>**46.3**<br>**42.1**<br>**46.8**|
|||||
|MetaMath-13B<br>MetaMath-**LB**-15B<br>MetaMath-**LB**-20B|50.6<br>**65.1**<br>16.8<br>57.0<br>60.4<br>49.1<br>**57.9**<br>60.4<br>**51.8**|**69.2**<br>**67.3**<br>**66.1**<br>**66.9**<br>**66.9**<br>**58.0**<br>**61.4**<br>64.1<br>61.9<br>65.9<br>64.8<br>61.0<br>55.0<br>50.1<br>65.3<br>63.0<br>62.5<br>62.7<br>60.9<br>55.4<br>53.3|18.6<br>14.4<br>17.3<br>51.3<br>42.1<br>54.0<br>**52.3**<br>**47.1**<br>**56.0**|



Table 11: Accuracy (%) on MSVAMP. MM suffix denotes the model was finetuned on MetaMath. 

|||CAUSAL|DATE|DISAM.|FORMAL|LOGIC. 3|LOGIC. 5|LOGIC. 7|
|---|---|---|---|---|---|---|---|---|
|||EN<br>BN|EN<br>BN|EN<br>BN|EN<br>BN|EN<br>BN|EN<br>BN|EN<br>BN|
||||||||||
|BLOOM-7B-OpenOrca||49.7<br>4.8|32.8<br>48.7|48.4<br>31.2|43.2<br>35.2|36.0<br>12.4|24.8<br>55.2|20.0<br>23.6|
||||||||||
|Orca 2-7B<br>Orca 2-**LB**-9B||**62.0**<br>47.1<br>57.2<br>**52.9**|**52.4**<br>**42.0**<br>26.8<br>24.4|62.8<br>42.4<br>**64.0**<br>**46.8**|**60.0**<br>50.4<br>55.2<br>**57.2**|**60.0**<br>37.2<br>52.4<br>**42.0**|**43.2**<br>25.6<br>36.0<br>**30.0**|**39.6**<br>20.8<br>38.8<br>**28.8**|
||||||||||
|Orca 2-13B<br>Orca 2-**LB**-15B<br>Orca 2-**LB**-20B||56.1<br>46.5<br>57.2<br>**59.9**<br>**64.7**<br>58.3|**64.0**<br>**50.0**<br>44.8<br>43.6<br>34.4<br>35.2|**66.8**<br>52.0<br>56.0<br>46.8<br>59.2<br>**56.0**|52.0<br>52.0<br>**55.6**<br>48.0<br>52.0<br>**54.8**|**68.4**<br>45.2<br>66.8<br>**56.4**<br>62.4<br>46.8|46.8<br>35.6<br>**47.2**<br>33.2<br>44.8<br>**38.0**|**49.2**<br>31.2<br>44.4<br>**34.4**<br>45.2<br>32.0|



Table 12: Full results on BBH and BBH-BN (Part 1 of 2). 

|||NAVI.|PENGUINS|REASON.|SNARKS|SPORTS|TEMPO.|WEB|
|---|---|---|---|---|---|---|---|---|
|||EN<br>BN|EN<br>BN|EN<br>BN|EN<br>BN|EN<br>BN|EN<br>BN|EN<br>BN|
|BLOOM-7B-OpenOrca||41.6<br>34.4|27.4<br>44.0|25.2<br>22.6|41.8<br>20.8|49.6<br>41.2|6.8<br>50.4|53.2<br>12.0|
||||||||||
|Orca 2-7B<br>Orca 2-**LB**-9B||**58.8**<br>46.4<br>48.8<br>**50.4**|**57.5**<br>24.0<br>44.5<br>**39.7**|**47.2**<br>25.6<br>41.6<br>**33.6**|**67.8**<br>42.9<br>56.5<br>**53.7**|**68.0**<br>51.6<br>66.4<br>**53.2**|**20.4**<br>13.6<br>17.6<br>**17.2**|**54.4**<br>43.6<br>50.8<br>**54.8**|
||||||||||
|Orca 2-13B<br>Orca 2-**LB**-15B<br>Orca 2-**LB**-20B||53.2<br>49.2<br>58.4<br>**62.0**<br>**60.0**<br>51.2|**59.6**<br>30.1<br>56.8<br>**43.2**<br>50.7<br>39.0|**61.6**<br>26.4<br>60.0<br>32.8<br>59.6<br>**37.2**|**65.5**<br>48.0<br>60.5<br>50.8<br>62.1<br>**54.8**|76.4<br>49.2<br>73.6<br>52.4<br>72.0<br>**53.6**|39.6<br>**22.8**<br>34.0<br>20.0<br>24.0<br>19.2|52.0<br>45.2<br>**57.6**<br>52.0<br>52.8<br>**59.2**|



Table 13: Full results on BBH and BBH-BN (Part 2 of 2). 

|||**AVG**<br>_An._|**AVG**<br>_An._|**EN**<br>_An._|**SW**<br>_An._|**BN**<br>_An._|**PA**<br>_An._|**TE**<br>_An._|**UR**<br>_An._|
|---|---|---|---|---|---|---|---|---|---|
|||||||||||
|Llama2-7B<br>BLOOM-7.1B<br>BLOOM-7.1B-SC||10.2<br>4.6<br>6.7<br>4.9<br>8.4<br>4.5||11.0<br>9.8<br>8.5<br>7.3<br>11.0<br>6.7|11.6<br>3.7<br>6.1<br>2.4<br>9.8<br>3.0|11.0<br>3.0<br>6.1<br>5.5<br>7.9<br>4.3|7.9<br>3.0<br>6.7<br>4.3<br>6.7<br>4.3|8.5<br>3.0<br>6.7<br>6.1<br>7.9<br>3.7|11.0<br>4.9<br>6.1<br>3.7<br>7.3<br>4.9|
|||||||||||
|Code Llama-7B<br>Code Llama-**LB**-9B||23.0<br>13.0<br>**24.9**<br>**19.4**||**36.0**<br>**36.0**<br>34.2<br>31.7|21.3<br>14.0<br>**27.4**<br>**17.1**|21.3<br>14.6<br>**23.2**<br>**15.2**|17.7<br>10.4<br>**23.2**<br>**18.3**|16.5<br>9.8<br>**19.5**<br>**15.2**|**25.0**<br>16.5<br>22.0<br>**18.9**|
|||||||||||
|Code Llama-13B<br>Code Llama-**LB**-15B<br>Code Llama-**LB**-20B||26.0<br>19.6<br>**26.3**<br>**23.6**<br>26.2<br>23.5||**42.7**<br>40.2<br>36.6<br>**41.5**<br>35.4<br>36.0|24.4<br>15.2<br>22.6<br>20.1<br>**25.6**<br>**24.4**|**26.2**<br>17.7<br>23.8<br>**20.1**<br>22.6<br>**20.1**|17.7<br>12.2<br>**26.8**<br>19.5<br>25.6<br>**20.7**|17.7<br>12.8<br>**23.8**<br>**19.5**<br>20.7<br>18.9|**27.4**<br>19.5<br>24.4<br>**20.7**<br>27.4<br>20.7|
|Code Llama-34B||29.7<br>22.7||46.3<br>43.9|28.7<br>17.1|32.2<br>23.8|22.0<br>15.9|19.5<br>12.8|29.3<br>22.6|



Table 14: HumanEval and HumanEval-MT Pass@1 scores. _An._ denotes the anonymized version. 

||**AVG**<br>**HRL**<br>**URL**|**EN**<br>**DE**<br>**FR**<br>**ES**<br>**RU**<br>**ZH**<br>**JA**|**TH**<br>**SW**<br>**BN**<br>**TE**|
|---|---|---|---|
|||||
|Lang. Freq. (Llama 2, %)<br>-<br>-<br>-<br>89.7<br>0.17<br>0.16<br>0.13<br>0.13<br>0.13<br>0.10<br>LESSTHAN0.005||||
|||||
|Llama 2-7B<br>Llama-**LB**-9B|9.1<br>12.1<br>3.9<br>**11.3**<br>**12.2**<br>**9.7**|15.2<br>11.6<br>**13.2**<br>11.2<br>**11.6**<br>**11.2**<br>**10.8**<br>**16.8**<br>**12.4**<br>12.8<br>**13.6**<br>9.2<br>10.0<br>**10.8**|7.2<br>5.2<br>3.2<br>0.0<br>**13.6**<br>**9.2**<br>**7.6**<br>**8.4**|
|Mistral-7B<br>**28.5**<br>**34.4**<br>12.8<br>**43.6**<br>**34.4**<br>**35.2**<br>**39.6**<br>**32.8**<br>**28.8**<br>**26.4**<br>20.4<br>7.6<br>16.0<br>7.2<br>Mistral-**LB**-9B<br>26.1<br>27.9<br>**20.7**<br>32.4<br>29.6<br>30.8<br>29.6<br>28.8<br>25.2<br>18.8<br>**24.4**<br>**18.8**<br>**22.4**<br>**17.2**||||



Table 15: Accuracy (%) of general-domain LMs on MGSM. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision_images/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision.pdf-0019-04.png)


**----- Start of picture text -----**<br>
Model Input<br>�పశ� :  �� �త �వ�� � - ష� ఉప��ం� ఒక �ంత ������ .  అత�� �ప� ��ర��త� ఒక<br>��� ఉం� .  �ప� ష� �ంత� ఒక �ప��క�న ��� ఉం� .  �ప� వరస �� ��న� సంవత����<br>���న ��ర��తల ���ల� త�� ��� .  అత� సంవత���� ������ ��ర��త� �ళ�� ,<br>అత�  23  సంవత��ల వయ�� �ం� ��ర��త� ������ ,  అత� వయ�� ఇ���  34.<br>అత�� �త�ం ఎ�� ���� ���� ఉ��� ?<br>(Question: Christina is planning a birthday party and needs .75 gift bags per invited guest, because<br>1/4 of attendees don't show up. She invited 16 friends. Gift bags are $2 each. How much will she<br>spend?)<br>Model Output<br>Kristin invited 16 friends, so she needs 16 x 0.75 = 12 gift bags.<br>Each gift bag costs $2, so she will spend 12 x $2 = $24 on gift bags.<br>#### 24<br>The answer is: 24<br>✅<br>**----- End of picture text -----**<br>


Figure 7: Example of zero-shot CoT generated by MetaMath-LANGBRIDGE-15B when prompted with an example from MGSM Telugu. For brevity, any instructional prompts are truncated. Translation of the input is provided in (blue with parenthesis). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision_images/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision.pdf-0020-00.png)


**----- Start of picture text -----**<br>
Model Input<br>িকভােব একজন সাধারণ ব��� কায�কারণ স�েক� িন�িলিখত �িত�ট �ে�র উ�র �দেবন ?<br>�জা খুব অিতশীত অনুভব করিছল ,  �স �ানীয় �ুিথ �দাকােন �িগত হেয় সব�ািধক আকােরর পানীয় িকনেত<br>�গল। অড�ার �দওয়ার আেগ ,  ক�ািশয়ার তােক বলেলন �য যিদ �স এক�ট �মগা - সাইজড �ুিথ িকেন তাহেল<br>�স এ�ট এক�ট িবেশষ �ারকীয় কােপ পােব। �জা জবাব িদেলন , ' আিম �ারকীয় কাপ এর ব�াপারটা িনেয়<br>যেতা িচ�া করিছ না ,  আিম �ধু আপনােদর সােথ থাকা সবেচেয় বড় �ুিথ চাই। '  আশনু�পভােব ,  �জা<br>�ারক কােপ �মগা - সাইজড �ুিথ �পল। �জা িক ই�াকৃতভােব �ারক কাপ িনল ?<br>(How would a typical person answer each of the following questions about causation?<br>Joe was feeling quite dehydrated, so he stopped by the local smoothie shop to buy the largest sized<br>drink available. Before ordering, the cashier told him that if he bought a Mega-Sized Smoothie he<br>would get it in a special commemorative cup. Joe replied, 'I don't care about a commemorative cup, I<br>just want the biggest smoothie you have.' Sure enough, Joe received the Mega-Sized Smoothie in a<br>commemorative cup. Did Joe intentionally obtain the commemorative cup?)<br>Options:<br>(A)  হ�াঁ (Yes)<br>(B)  না (No)<br>Model Output<br>Step 1: Understand the context of the question.<br>The question is asking whether Jo intentionally took the special commemorative<br>cup along with the mega-sized smoothie.<br>Step 2: Analyze the information provided.<br>Jo was overly excited and wanted the largest possible beverage. The cashier<br>offered a special commemorative cup for purchasing the mega-sized smoothie. Jo<br>did not express any concern about the cup and only wanted the largest smoothie.<br>Step 3: Determine the answer based on the information.<br>Since Jo did not express any concern about the cup and only wanted the largest<br>smoothie, it can be inferred that Jo did not intentionally take the special<br>commemorative cup.<br>Step 4: Choose the appropriate answer from the options.<br>The appropriate answer is (B) no.  ✅<br>**----- End of picture text -----**<br>


Figure 8: Example of zero-shot CoT generated by Orca 2-LANGBRIDGE-15B when prompted with an example from BBH-BN CAUSAL JUDGEMENT. For brevity, any instructional prompts are truncated. Translation of the input is provided in (blue with parenthesis). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision_images/acl_2024_LangBridge_Multilingual_Reasoning_Without_Multilingual_Supervision.pdf-0021-00.png)


**----- Start of picture text -----**<br>
Model Input<br>আজ  9/7 । �জন  NFL 2003  �দখেছ।  MM/DD/YYYY- �ত আজ �থেক এক স�াহ আেগর তািরখ কী ?<br>(Today is 9/7. Jane is watching NFL 2003. What is the date one week ago from today in MM/DD/YYYY?)<br>Options:<br>(A) 09/05/2003 (B) 08/30/2003 (C) 08/31/2074 (D) 08/31/2003 (E) 06/30/2004<br>Model Output<br>Step 1: Identify the current date.<br>The current date is 7/9/2021.<br>Step 2: Determine the date one week before the current date.<br>To find the date one week before the current date, we need to subtract 7 days<br>from the current date.<br>Step 3: Calculate the date one week before the current date.\n7/9/2021 - 7 days<br>= 7/2/2021<br>[ ⋯ ]<br>Step 11: Match the calculated date with the given options.<br>The calculated date matches option (A) 05/18/2021.<br>Final answer: (A) 05/18/2021 ❌<br>**----- End of picture text -----**<br>


Figure 9: Example of zero-shot CoT generated by Orca 2-LANGBRIDGE-15B when prompted with an example from BBH-BN DATE UNDERSTANDING. For brevity, any instructional prompts are truncated. Translation of the input is provided in (blue with parenthesis). 

