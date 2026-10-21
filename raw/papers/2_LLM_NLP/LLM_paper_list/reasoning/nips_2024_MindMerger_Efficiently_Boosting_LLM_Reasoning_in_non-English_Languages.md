# **MindMerger: Efficient Boosting LLM Reasoning in non-English Languages** 

**Zixian Huang**[1] **, Wenhao Zhu**[1] **, Gong Cheng**[1] **, Lei Li**[2] **, Fei Yuan**[3] _[∗]_ 

1State Key Laboratory for Novel Software Technology, Nanjing University 

2 Carnegie Mellon University 

3Shanghai Artificial Intelligence Laboratory `{zixianhuang, zhuwh}@smail.nju.edu.cn, gcheng@nju.edu.cn leili@cs.cmu.edu, yuanfei@pjlab.org.cn` 

## **Abstract** 

Reasoning capabilities are crucial for Large Language Models (LLMs), yet a notable gap exists between English and non-English languages. To bridge this disparity, some works fine-tune LLMs to relearn reasoning capabilities in non-English languages, while others replace non-English inputs with an external model’s outputs such as English translation text to circumvent the challenge of LLM understanding non-English. Unfortunately, these methods often underutilize the built-in skilled reasoning and useful language understanding capabilities of LLMs. In order to better utilize the minds of reasoning and language understanding in LLMs, we propose a new method, namely MindMerger, which merges LLMs with the external language understanding capabilities from multilingual models to boost the multilingual reasoning performance. Furthermore, a two-step training scheme is introduced to first train to embeded the external capabilities into LLMs and then train the collaborative utilization of the external capabilities and the built-in capabilities in LLMs. Experiments on three multilingual reasoning datasets and a language understanding dataset demonstrate that MindMerger consistently outperforms all baselines, especially in low-resource languages. Without updating the parameters of LLMs, the average accuracy improved by 6.7% and 8.0% across all languages and low-resource languages on the MGSM dataset, respectively[2] . 

## **1 Introduction** 

One of the primary focuses of Artificial Intelligence research currently revolves around improving its reasoning capabilities [Ahn et al., 2024, Minaee et al., 2024], which is derived from the need to enable Large Language Models (LLMs) [Ouyang et al., 2022, Touvron et al., 2023, Jiang et al., 2023a] to think rationally and perform functions like humans [Imani et al., 2023, Jiang et al., 2023b]. Substantial progress has been made in English reasoning [Yu et al., 2023, Yuan et al., 2023a], but the performance in non-English, especially low-resource languages, still lags behind [Shi et al., 2023] due to the scarce of multilingual training data [Touvron et al., 2023]. 

Existing work tries to use external models to compensate for the deficiencies of LLM in multilingual reasoning. Some works use the relearning-based strategy, which uses translation models to generate multilingual training data for fine-tuning LLMs to relearn reasoning in each language [Chen et al., 2023, Chai et al., 2024]. Some other works use the replacement-based strategy, which use translation models to translate queries from non-English to English text for replacing the non-English input of 

> _∗_ Corresponding author. 

> 2 Our code is available on `https://github.com/CONE-MT/MindMerger` . 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MindMerger_Efficiently_Boosting_LLM_Reasoning_in_non-English_Languages_images/nips_2024_MindMerger_Efficiently_Boosting_LLM_Reasoning_in_non-English_Languages.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Query LLM Generation Query LLM Generation<br>English : Sadie slept 8 hours on Monday. For  8 - 2 = 6<br>English : Claire makes a 3 egg omelet  3  ×  7 = 21 the next 2 days, she slept 2 hours less, each. If  6  ×  2 = 12<br>every morning for breakfast. How many  21  ×  4 = 84 the rest of the week she slept 1 hour more than  6 + 1 = 7<br>dozens of eggs will she eat in 4 weeks? 84 / 12 = 7 those 2 days, how many hours did she sleep in  7  ×  (5 – 3) = 14<br>total throughout the week? 8 + 12 + 14 = 34<br>8 - 2 = 6<br>Chinese 做一份煎蛋饼当早餐。多少打鸡蛋？ :  克莱尔每天早上用 4  周内她能吃 3  颗鸡蛋 3 21  ××  7 = 21 4 = 84 Chinese 天她各少睡了时间里，她比那个星期一共睡了多少个小时？ :  莎蒂周一睡了 2  小时。如果在这周剩下的 2  天多睡 8  小时。接下来 1  小时，她在整 2  6 6 + 1 = 77 8 + 12 + 28 = 48 ××  2 = 12 (7 – 3) = 28<br>**----- End of picture text -----**<br>


Figure 1: Examples of multilingual mathematical reasoning from the MGSM dataset. LLM can generate correct and incorrect answers when asked in different languages. 

LLM [Shi et al., 2023]. Both strategies try to use the translation model to help LLMs master new capabilities, but the insufficient translation quality constrains the performance of these methods. 

Moreover, certain capabilities, such as reasoning and language understanding, are built-in in LLMs and should be utilized without the need to develop them from scratch. For the reasoning capabilities, as illustrated in Figure 1, regardless of the language in which the same mathematical question is posed, the correct reasoning process remains consistent. It shows that the reasoning capabilities is universal rather than language-specific [Brannon, 2005]. For language understanding capabilities, while the Chinese question in the first example of Figure 1 may cause a reasoning error by failing in understanding the English-origin term "dozen of", the LLM successfully differentiated between "the week" and "the weekdays" in the Chinese question, contrasting with the failure of the English question in the second example. It shows that although the proficiency may not match that of English, expressions in non-English languages remain valuable to LLMs. 

Considering that the built-in reasoning and language understanding capabilities of LLMs need to be better utilized, in this paper, we propose a new method **MindMerger** , which preserves the minds of reasoning and language understanding capabilities in LLMs, and merges the external language understanding capabilities from pre-trained multilingual models to boost the multilingual reasoning effects. To address the challenge of insufficient generation quality of external models, MindMerger feeds the LLM the undecoded query representation from the multilingual model rather than text. Additionally, it uses an augmentation strategy that combines the encoded query representation with the input of LLM to utilize both external and built-in language understanding capabilities. 

Given the inconsistency in the representation space, understanding the query representation encoded by multilingual model is not trivial for LLMs. To address this, we propose a two-stage training scheme including the **mapping stage** and the **augmentation stage** . In the mapping stage, we train MindMerger to embed the language capabilities of the multilingual model into the LLM by using accessible general bilingual pairs such as translation data. In the augmentation stage, MindMerger is further trained to collaboratively utilize the built-in capabilities of LLM and the embedded external language capabilities by using query translation task data generated from translation model. Throughout the two stages, only a small number of parameters that connect two models are trained, while both the multilingual model and the LLM are frozen to prevent their built-in capabilities from forgetting. Extensive experiments are conducted on three multilingual reasoning datasets and a language understanding dataset. Taking on the MGSM dataset [Shi et al., 2023] as an example, MindMerger outperforms all baselines and achieves a lead of at least 6.7% on the average accuracy across all languages, and its performance in low-resource languages is even more significant leading by at least 8.0% (§ 4.3). Compared with the replacement-based method that translates non-English text into English as the LLM input, MindMerger can lead by at least 6.6% in average accuracy based on the same translation model (§ 5.1). Benefiting from the accessible general bilingual pairs used in the mapping stage, the average accuracy across low-resource languages increased by 14.9% (§ 5.2). 

Our contributions can be summarized as follows: 

- We propose a new method MindMerger to boost the multilingual reasoning of LLMs, which preserves the built-in reasoning and language understanding capabilities of LLMs while augmenting them with the external language understanding capabilities from multilingual models. 

- We propose a two-stage training scheme to help MindMerger sequentially learn to embed external capabilities and collaboratively utilize internal and external capabilities. 

2 

- MindMerger achieves the best results on four dataset about multilingual reasoning or understanding datasets, notably improving all languages, including low-resource languages, with an average accuracy increase of at least 6.7% and 8.0% on the MGSM datasets. 

## **2 Related Work** 

**Multilingual Reasoning.** There have been some attempts to improve LLM’s performance on multilingual reasoning. Several works design crafted prompts to assist LLMs in reasoning [Huang et al., 2023, Qin et al., 2023], but their effectiveness diminishes when used with open-source LLMs like Llama [Touvron et al., 2023] due to limited capacity for multi-step instruction-following [Huang et al., 2023]. Supervised fine-tuning LLMs is another effective way, where some works use translation models to translate query-response [Chen et al., 2023], or query-only [Zhu et al., 2024] to build multilingual task datasets, enabling LLMs to relearn reasoning or language understanding. Some other works utilize external models to generate English text [Shi et al., 2023] to replace the nonEnglish input of LLMs, aiming to utilize the reasoning capabilities of LLMs directly. However, the above methods are limited by the generation quality of the translation model, and the built-in reasoning or language understanding capabilities of LLM are neglected. In contrast, MindMerger avoid the loss introduced by autoregressive decoding and employs an augmentation-based strategy that utilizes the built-in capabilities of LLMs to enhance multilingual reasoning performance. 

**Model Merging.** Model Merging is a popular topic in recent LLM research. Generally, it aims to combine an external module to strengthen the capabilities that LLM lacks, such as multi-modal vision capabilities [Liu et al., 2023, Li et al., 2023, Zhu et al., 2023]. Some works [Sun et al., 2021, Bansal et al., 2024] find that interpolating models with multilingual capabilities using cross-attention can improve the performance of multilingual tasks, but research on merging multilingual models and LLMs with English reasoning capabilities is still scarce. Recently, Yoon et al. [2024] merge the encoder of multilingual model and the LLM to improve the multilingual reasoning performance. However, the input of LLMs is completely replaced by the output of a multilingual encoder, making its built-in multilingual capabilities underutilized. In addition, only using English task data for training limits the performance of model merging. Instead of replacing the input of LLMs, we collaboratively utilize the features of the multilingual model and use the available general bilingual pair to train to obtain multilingual reasoning capabilities. 

## **3 Approach** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MindMerger_Efficiently_Boosting_LLM_Reasoning_in_non-English_Languages_images/nips_2024_MindMerger_Efficiently_Boosting_LLM_Reasoning_in_non-English_Languages.pdf-0003-05.png)


**----- Start of picture text -----**<br>
Model Structure Training Scheme<br>LLM Transformer layers 1. Mapping Stage<br>(General bilingual pairs)<br><bos> ෩𝑿 <sep> 𝑻<br>Query 𝑞<br>LLM Embedding Layer<br>Mapping Layer 2. Augmentation Stage<br>𝑿 (Query translation task data)<br>Query 𝑞<br>Multilingual Encoder Query 𝑞<br>Query 𝑞 Query 𝑞<br>**----- End of picture text -----**<br>


Figure 2: Overview of the model structure and training scheme of MindMerger, which consists of an LLM (blue) and a external model (yellow) and is trained by a two-stage scheme. 

Given an LLM with skilled reasoning capabilities and useful language understanding capabilities, our target is to maintain the built-in capabilities and compensate for its shortcomings in non-English language understanding capabilities with an external multilingual model. To this end, as shown in Figure 2, we propose MindMerger that uses the output of the multilingual model as an augmentation complementing to the original input (§ 3.1). We further design a two-stage training scheme to learn the collaborative utilizing both the external and built-in capabilities (§ 3.2). 

3 

## **3.1 Model Structure** 

Given a query _q_ with _l_ tokens, we first utilize the multilingual model to understand and encode it into a representation _**X**_ that is more general and reduces the challenge of multilingual understanding: 

_**X**_ = `Encoder` ( _q_ ) _,_ (1) 

where `Encoder` ( _·_ ) is a pre-trained multilingual model, typically using its encoder part, and _**X** ∈_ R _[l][×][d]_[1] is the hidden state output of the multilingual model with a dimension of _d_ 1. 

The representation _**X**_ resides in the multilingual model space, which is separate from the LLM space and cannot be used directly. Therefore, we introduce a mapping layer: 

� _**X**_ = `Mapping` ( _**X**_ ) _,_ (2) 

where _**X**_[�] _∈_ R _[l][×][d]_[2] is the projection of _**X**_ on the space of LLM. Unless otherwise stated, the _·_ implementation of `Mapping` ( ) is a two-layer multi-layer perceptron (MLP). 

The acquisition of _**X**_ utilizes the language understanding capabilities of the external multilingual model. In order to take advantage of the built-in capabilities of the LLM, we prompt it directly: 

_**T**_ = `Embedding` ( _q_ ) _,_ (3) 

where _**T** ∈_ R _[l][×][d]_[2] is the representation of query _q_ in the space of LLM and `Embedding` ( _·_ ) is the embedding layer of LLM. 

Then, we concatenate the query representation from the multilingual model and the LLM for the collaborative utilizing of LLM’s capabilities: 

( _**X**_[�] _,_ _**T**_ ) = [ _⟨_ `bos` _⟩_ ; _**X**_[�] ; _⟨_ `sep` _⟩_ ; _**T**_ ] _,_ (4) where _⟨_ `bos` _⟩∈_ R _[d]_[2] is the representation of the start token of LLM and _⟨_ `sep` _⟩∈_ R _[d]_[2] is a trainable variable that denotes the boundary of _**X**_[�] . Finally, ( _**X**_[�] _,_ _**T**_ ) is fed to LLM to generate the response. 

## **3.2 Two-Stage Training** 

The training scheme of MindMerger is divided into two stages: mapping stage and augmentation stage. The former helps LLM learn to use the capabilities of multilingual model, and the latter helps LLM collaboratively utilize its own and the capabilities from multilingual model. Examples of training data for each stage are shown in Appendix D. 

**Mapping Stage.** Given the representation spaces between multilingual model and LLM are distant different, it is not trivial for LLM to understand and utilize the external capabilities provided by multilingual model. In order to better learn to utilize external capabilities, we force MindMerger to focus on input from the multilingual model using the replacement-based strategy during this stage. Specifically, we simplify the input of Equation (4) as follows: 

� _**M**_ = [ _⟨_ `bos` _⟩_ ; _**X**_[�] ; _⟨_ `sep` _⟩_ ] _._ (5) 

Since general bilingual pairs such as translation data is in vast availability, we use it from various languages to English to train MindMerger in this stage. The loss function is outlined as follows: 

_−_ arg min log _P_ ( _Y |_ _**M**_[�] _, θ, ϕ, σ_ ) _,_ (6) _σ_ 

where _Y_ is the text of training target, _θ_ and _ϕ_ are the parameters of the multilingual model and LLM respectively, which are frozen during the training to prevent forgetting, and _σ_ contains the trainable parameters of the mapping layer in Equation (2) and the boundary token _⟨_ `sep` _⟩_ . 

**Augmentation Stage.** Although MindMerger have learned to utilize external capabilities in the mapping stage, the replacement-based strategy may cause LLM to neglect the use of its own built-in capabilities. To help MindMerger further learn to merge capabilities from external and built-in LLM, in this stage we use the augmentation-based strategy as described in the Equation (4). The loss function is calculated as follows: 

_−_ arg min _σ_ ˆ log _P_ ( _Y |_ ( _**X**_[�] _,_ _**T**_ ) _, θ, ϕ,_ ˆ _σ_ ) _,_ (7) 

where _σ_ ˆ is initiated from the checkpoint of _σ_ trained at the mapping stage, and the parameters of the multilingual model and LLM represented as _θ_ and _ϕ_ , respectively, are kept constant during training. In this stage, the query translation task data generated from public translation models is used as the training data to adapt MindMerger to downstream task. 

4 

## **4 Experiments** 

## **4.1 Compared Methods** 

**Our Methods.** Two variants of MindMerger were implemented. (1) The implementation described in § 3.1 is denoted as **MindMerger-Soft** . (2) **MindMerger-Hard** augments the prompts of LLM with the translated query given by the translation model (Appendix D for the details of prompts). 

**Baselines.** We compared our methods with three categories of baselines. (1) One basic method **MonoReason** [Yu et al., 2023, Zhu et al., 2024] which is fine-tuned on the English task dataset. (2) Three relearning-based methods that use task data with query translation, including the fullparameter fine-tuning model **MultiReason-SFT** [Zhu et al., 2024], parameter-efficient fine-tuning model **MultiReason-Lora** [Hu et al., 2022], and the state-of-the-art method **QAlign** [Zhu et al., 2024] which first learns language understanding by training LLM on query translation data and then further fine-tunes LLM as MonoReason. (3) Two replacement-based methods that introduce external models, including **Translate-En** [Shi et al., 2023] and **LangBridge** [Yoon et al., 2024]. Translate-En is a hard replacement-based which translates the query into English to replace the prompt of LLM. LangBridge is a soft replacement-based method which replaces the input of LLMs with the hidden states output by mT5-xl [Xue et al., 2021]. The prompts of each baselines are presented in Appendix D. 

**Details.** Unless otherwise stated, we used the encoder part of mT5-xl [Xue et al., 2021] as the multilingual model in our methods, used the NLLB200-3.3B as the translation model for baselines, and used the Llama 2-7B as the LLM for all methods. The influence of different multilingual models including encoder-only, decoder-only and encoder-decoder architectures will be analyzed in § 5.1. Both MindMerger and all the baselines, except QAlign, are based on the same MonoReason model. Additionally, MonoReason and QAlign are trained based on the same LLM. Specifically, we used the publicly available checkpoint given by Yu et al. [2023] as MonoReason model for mathematical reasoning task. For all models, we set learning-rate=2e-5, batch size=128, max length=512, and epoch=3 and used 8 NVIDIA A100 GPUs for training. 

## **4.2 Datasets** 

**Evaluation Datasets.** We experimented MindMerger with the latest multilingual mathematical reasoning **MGSM** [Shi et al., 2023] and **MSVAMP** [Chen et al., 2023], where MSVAMP serves as an out-of-domain test set. In order to evaluate the diverse multilingual reasoning capabilities of the models, a challenging multilingual dataset **X-CSQA** [Lin et al., 2021] in commonsense reasoning task and a language understanding dataset **XNLI** [Conneau et al., 2018] in natural language inference (NLI) task were used. The statistics of these datasets are presented in Appendix D. 

**Training Datasets.** Three categories of training data were used in our methods and baselines. (1) **General bilingual pairs** . We used the translation data from the multilingual language to English and randomly sampled 100K of data for each language (except English) from the Lego-MT [Yuan et al., 2023b] dataset, which is a large-scale translation dateset that contains all the languages that involved in our experiments. (2) **English task data** . We used MetaMathQA [Yu et al., 2023] and MultiNLI Williams et al. [2018] datasets for mathematical reasoning and NLI task, respectively. Similar to Huang et al. [2022], we unified the training set of X-CSQA, OpenBookQA [Mihaylov et al., 2018], ARC [Bhakthavatsalam et al., 2021] and QASC [Khot et al., 2020] to train commonsense reasoning task more fully. (3) **Query translation task data** . We used the translated results given by Chen et al. [2023] and the official dev set of XNLI for mathematical reasoning and the NLI task, respectively, and translated the X-CSQA training set based on M2M100-1.2B [Fan et al., 2021]. 

## **4.3 Experimental Results** 

**MindMerger improves LLM performance on all datasets, especially benefiting low-resource languages.** MindMerger-Soft has an average accuracy better than all other baselines at least 6.7%, 3.2% on the MGSM and MSVAMP datasets in Table 1, demonstrating its remarkable multilingual reasoning capabilities. For the commonsense reasoning task in Table 2, MindMerger-Soft also significantly leads all baselines by at least 4.1%. In Table 3, MindMerger-Soft demonstrates advantages in language understanding, which significantly outperformed all the baselines (with p-value < 0.01). MindMerger-Hard achieves the best results except MindMerger-Soft on three out of four datasets, demonstrating the advantages of augmentation-based methods over other categories of methods. 

5 

Table 1: Experimental results on MGSM and MSVAMP datasets. Lrl., Hrl., and Avg. represent the average accuracy across low-resource languages, high-resource languages, and all languages, respectively. Referring to Shi et al. [2023], we regard Bn, Th, and Sw as low-resourse languages, and regard the remaining languages as high-resource languages. We used the checkpoints given by Yu et al. [2023] as the MonoReason models. 

|**MGSM**|**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**De**<br>**Fr**<br>**Ru**<br>**Es**<br>**En**|**Lrl.**<br>**Hrl.**<br>**Avg.**|
|---|---|---|
|MonoReason [Yu et al., 2023, Zhu et al., 2024]<br>MultiReason-Lora [Hu et al., 2022]<br>MultiReason-SFT [Zhu et al., 2024]<br>QAlign [Zhu et al., 2024]<br>LangBridge [Yoon et al., 2024]<br>Translate-En [Shi et al., 2023]|6.8<br>7.2<br>6.8<br>36.4<br>38.4<br>55.2<br>54.4<br>52.0<br>57.2<br>68.8<br>29.6<br>35.2<br>28.0<br>52.0<br>**54.8**<br>59.6<br>**58.4**<br>**62.4**<br>59.6<br>64.8<br>33.2<br>40.0<br>42.0<br>42.0<br>42.0<br>45.2<br>44.8<br>45.2<br>48.0<br>52.0<br>32.4<br>39.6<br>40.4<br>44.0<br>48.4<br>54.8<br>56.8<br>52.4<br>59.6<br>68.0<br>42.8<br>50.4<br>43.2<br>40.0<br>45.2<br>56.4<br>50.8<br>52.4<br>58.0<br>63.2<br>48.4<br>37.6<br>37.6<br>49.2<br>46.8<br>60.4<br>56.4<br>47.6<br>59.6<br>65.5|6.9<br>51.8<br>38.3<br>30.9<br>58.8<br>50.4<br>38.4<br>45.6<br>43.4<br>37.5<br>54.9<br>49.6<br>45.5<br>52.3<br>50.2<br>41.2<br>55.1<br>50.6|
|MindMerger-Hard<br>46.0<br>36.0<br>48.4<br>52.4<br>54.4<br>60.4<br>56.0<br>60.4<br>**62.0**<br>**71.2**<br>43.5<br>59.5<br>54.7<br>MindMerger-Soft<br>**50.4**<br>**52.8**<br>**57.2**<br>**54.4**<br>53.6<br>**61.2**<br>57.6<br>60.8<br>58.4<br>66.8<br>**53.5**<br>**59.0**<br>**57.3**|||
|**MSVAMP**<br>**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**De**<br>**Fr**<br>**Ru**<br>**Es**<br>**En**<br>**Lrl.**<br>**Hrl.**<br>**Avg.**|||
|MonoReason [Yu et al., 2023, Zhu et al., 2024]<br>MultiReason-Lora [Hu et al., 2022]<br>MultiReason-SFT [Zhu et al., 2024]<br>QAlign [Zhu et al., 2024]<br>LangBridge [Yoon et al., 2024]<br>Translate-En [Shi et al., 2023]|12.5<br>15.8<br>17.2<br>54.0<br>55.9<br>63.9<br>65.2<br>58.1<br>64.3<br>67.1<br>39.4<br>43.8<br>39.1<br>55.2<br>55.1<br>60.4<br>59.1<br>56.8<br>60.6<br>64.2<br>34.8<br>38.1<br>39.8<br>43.4<br>42.9<br>45.6<br>45.8<br>45.0<br>46.1<br>46.8<br>41.7<br>47.7<br>**54.8**<br>58.0<br>55.7<br>62.8<br>63.2<br>61.1<br>63.3<br>65.3<br>46.8<br>46.3<br>42.1<br>45.5<br>50.4<br>58.1<br>57.0<br>55.8<br>56.9<br>60.6<br>47.9<br>51.3<br>43.1<br>50.4<br>55.8<br>43.9<br>50.9<br>53.4<br>51.4<br>60.6|15.2<br>61.2<br>47.4<br>40.8<br>58.8<br>53.4<br>37.6<br>45.1<br>42.8<br>48.1<br>61.3<br>57.2<br>45.1<br>54.9<br>52.0<br>47.4<br>52.3<br>50.9|
|MindMerger-Hard<br>MindMerger-Soft|43.6<br>46.7<br>50.6<br>**61.4**<br>60.6<br>**65.9**<br>**65.6**<br>62.1<br>**65.0**<br>**67.8**<br>**52.0**<br>**53.4**<br>54.0<br>59.0<br>**61.7**<br>64.1<br>64.0<br>**63.3**<br>**65.0**<br>67.7|47.0<br>**64.1**<br>58.9<br>**53.1**<br>63.5<br>**60.4**|



MindMerger-Soft stands out notably in enhancing the reasoning capabilities of low-resource languages. Compared to all baselines, it delivers an average accuracy improvement of at least 8.0% and 5.0% across low-resource languages on the MGSM, MSVAMP datasets. 

**Utilizing the built-in reasoning capabilities of the LLM rather than relearning for non-English is a more effective way to boost multilingual reasoning capabilities.** It is not easy for relearning-based methods to improve the performance of high-resource and low-resource languages simultaneously. For instance, MultiReason-SFT improves its performance in low-resource languages from an average of 6.9% to 38.4% on the MGSM dataset, but its performance in high-resource languages declines from an average of 51.8% to 45.6%. Compared with the three relearning-based baselines, MindMerger-Soft achieves a lead on both low-resource and high-resource languages and outperforms them by at least 6.9% on the MGSM datasets for the average accuracy across all languages. 

**Utilizing the built-in language understanding capabilities of LLM rather than replacing them boosts multilingual reasoning capabilities more significantly.** Comparing Translate-En and MindMerger-Hard which are completely consistent except for the input strategy, the augmentationbased MindMerger-Hard has a higher average accuracy than the replacement-based Translate-En of 4.4% and 11.8% across high-resource languages in the MGSM and MSVAMP datasets respectively, showing that non-English input can help LLM enhance query understanding. Although the understanding capabilities of LLM in low-resource languages are limited, MindMerger-Hard still leads Translate-En by 2.3% on the MSGM dataset. Larger boost comes from MindMerger-Soft, which leads all replacement-based methods with an average accuracy of at least 6.7% across all languages. 

Table 2: Results on the X-CSQA dataset. Avg. represents the average accuracy across all languages. 

|**X-CSQA**|**Sw**<br>**Ur**<br>**Hi**<br>**Ar**<br>**Vi**<br>**Ja**<br>**Pl**<br>**Zh**<br>**Nl**<br>**Ru**<br>**It**<br>**De**<br>**Pt**<br>**Fr**<br>**Es**<br>**En**|**Avg.**|
|---|---|---|
|MonoReason [Yu et al., 2023, Zhu et al., 2024]<br>MultiReason-Lora [Hu et al., 2022]<br>MultiReason-SFT [Zhu et al., 2024]<br>QAlign [Zhu et al., 2024]<br>LangBridge [Yoon et al., 2024]<br>Translate-En [Shi et al., 2023]|24.2<br>25.1<br>32.9<br>32.3<br>50.9<br>49.1<br>50.6<br>56.5<br>57.5<br>56.0<br>56.0<br>61.2<br>61.7<br>63.5<br>64.0<br>76.3<br>25.1<br>32.0<br>39.2<br>42.2<br>56.6<br>**55.9**<br>60.6<br>62.2<br>61.3<br>62.8<br>66.3<br>64.9<br>66.2<br>67.4<br>67.7<br>**79.3**<br>27.6<br>29.2<br>32.0<br>28.7<br>38.8<br>38.7<br>45.5<br>43.8<br>45.9<br>46.5<br>50.2<br>49.1<br>51.2<br>52.1<br>54.3<br>67.2<br>35.1<br>32.6<br>37.8<br>36.3<br>50.5<br>49.2<br>51.3<br>54.8<br>56.3<br>56.3<br>58.3<br>58.8<br>59.8<br>60.3<br>63.1<br>75.7<br>31.8<br>30.5<br>30.6<br>30.6<br>33.3<br>33.9<br>39.8<br>39.8<br>38.4<br>35.1<br>39.1<br>37.4<br>36.3<br>38.2<br>38.4<br>44.4<br>36.5<br>41.3<br>**48.4**<br>44.6<br>51.8<br>47.1<br>53.3<br>51.5<br>55.0<br>54.4<br>56.3<br>57.3<br>54.7<br>57.2<br>55.5<br>71.3|51.3<br>56.9<br>43.8<br>52.3<br>36.1<br>52.3|
|MindMerger-Hard<br>MindMerger-Soft|33.1<br>29.9<br>40.4<br>37.7<br>52.9<br>49.9<br>54.7<br>55.4<br>58.0<br>58.0<br>59.7<br>58.6<br>61.9<br>62.5<br>63.6<br>75.2<br>**45.5**<br>**46.2**<br>**48.4**<br>**51.4**<br>**60.6**<br>53.9<br>**63.3**<br>**62.9**<br>**63.8**<br>**63.7**<br>**66.8**<br>**67.0**<br>**67.1**<br>**68.1**<br>**69.1**<br>78.1|53.1<br>**61.0**|



## **5 Analysis** 

## **5.1 The Usage of Multilingual Model** 

**MindMerger can flexibly interpolate among various multilingual models.** We experimented with a wide variety of multilingual models, including decoder-only models mGPT [Shliazhko et al., 2022], encoder-only models mBERT [Devlin et al., 2018] and XLM-RoBERTa-large [Conneau et al., 2019], 

6 

Table 3: Results on the XNLI dataset. Avg. represents the average accuracy across all languages. 

|**XNLI**|**Sw**<br>**Ur**<br>**Hi**<br>**Th**<br>**Ar**<br>**Tr**<br>**El**<br>**Vi**<br>**Zh**<br>**Ru**<br>**Bg**<br>**De**<br>**Fr**<br>**Es**<br>**En**|**Avg.**|
|---|---|---|
|MonoReason [Yu et al., 2023, Zhu et al., 2024]<br>MultiReason-Lora Hu et al. [2022]<br>MultiReason-SFT [Zhu et al., 2024]<br>QAlign [Zhu et al., 2024]<br>LangBridge [Yoon et al., 2024]<br>Translate-En [Shi et al., 2023]|45.9<br>49.2<br>55.7<br>55.4<br>60.9<br>61.9<br>63.7<br>73.7<br>74.7<br>77.6<br>76.7<br>80.6<br>82.2<br>82.8<br>**90.0**<br>45.9<br>49.3<br>56.4<br>55.7<br>60.9<br>61.9<br>64.7<br>73.7<br>74.7<br>76.7<br>76.7<br>80.6<br>82.2<br>82.8<br>**90.0**<br>56.3<br>57.5<br>61.7<br>60.1<br>61.7<br>65.6<br>67.0<br>73.7<br>79.1<br>79.7<br>78.7<br>82.3<br>82.9<br>83.9<br>88.8<br>65.2<br>62.2<br>63.3<br>65.2<br>67.0<br>67.9<br>66.5<br>73.7<br>76.6<br>79.2<br>79.4<br>80.9<br>83.1<br>83.8<br>89.1<br>**71.7**<br>66.9<br>71.1<br>**72.4**<br>75.2<br>74.8<br>79.1<br>78.5<br>77.4<br>77.4<br>79.6<br>78.8<br>79.9<br>80.5<br>83.4<br>65.3<br>61.6<br>68.7<br>69.5<br>68.9<br>74.5<br>**79.3**<br>76.7<br>74.8<br>76.0<br>80.8<br>80.6<br>80.4<br>81.4<br>87.4|68.7<br>68.9<br>71.9<br>73.5<br>76.5<br>75.1|
|MindMerger-Hard<br>MindMerger-Soft|65.7<br>56.4<br>58.3<br>64.1<br>63.6<br>70.0<br>62.2<br>56.6<br>61.8<br>58.1<br>61.7<br>64.2<br>61.2<br>63.3<br>80.8<br>66.6<br>**69.4**<br>**74.7**<br>71.8<br>**76.2**<br>**75.7**<br>78.5<br>**80.3**<br>**80.0**<br>**80.7**<br>**82.4**<br>**83.5**<br>**83.9**<br>**84.4**<br>88.7|63.2<br>**78.4**|



Table 4: Merging with different multilingual models on the MGSM dataset. # Parm represents the number of parameters of the used external model. Lrl., Hrl., and Avg. represent the average accuracy across low-resource languages, high-resource languages, and all languages, respectively. Referring to Shi et al. [2023], we regard Bn, Th, and Sw as low-resourse languages, and regard the remaining languages as high-resource languages. 

|**MGSM**|**# Parm**|**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**De**<br>**Fr**<br>**Ru**<br>**Es**<br>**En**|**Lrl.**<br>**Hrl.**<br>**Avg.**|
|---|---|---|---|
|Translate-En||||
|M2M100-418M<br>M2M100-1.2B<br>NLLB-200-1.3B<br>NLLB-200-3.3B|484 M<br>1,239 M<br>1,371 M<br>3,345 M|30.0<br>**38.0**<br>38.8<br>31.6<br>**50.8**<br>52.0<br>50.0<br>42.4<br>54.0<br>**65.5**<br>42.4<br>34.0<br>**49.6**<br>40.8<br>42.8<br>55.2<br>50.4<br>**49.2**<br>46.8<br>**65.5**<br>46.0<br>32.0<br>40.4<br>47.2<br>45.6<br>55.2<br>51.2<br>46.0<br>55.2<br>**65.5**<br>**48.4**<br>37.6<br>37.6<br>**49.2**<br>46.8<br>**60.4**<br>**56.4**<br>47.6<br>**59.6**<br>**65.5**|35.6<br>49.5<br>44.7<br>**42.0**<br>50.1<br>47.1<br>39.5<br>52.3<br>47.9<br>41.2<br>**55.1**<br>**50.6**|
|MindMerger-Hard||||
|M2M100-418M<br>M2M100-1.2B<br>NLLB-200-1.3B<br>NLLB-200-3.3B|484 M<br>1,239 M<br>1,371 M<br>3,345 M|39.6<br>28.0<br>36.4<br>49.2<br>48.8<br>60.0<br>**56.4**<br>55.6<br>58.4<br>64.8<br>40.0<br>**36.0**<br>47.6<br>**52.4**<br>50.8<br>58.0<br>**56.4**<br>**60.8**<br>61.2<br>66.8<br>44.0<br>30.0<br>42.8<br>48.0<br>53.6<br>**61.6**<br>**56.4**<br>54.8<br>**63.6**<br>70.8<br>**46.0**<br>**36.0**<br>**48.4**<br>**52.4**<br>**54.4**<br>60.4<br>56.0<br>60.4<br>62.0<br>**71.2**|34.7<br>56.2<br>49.7<br>41.2<br>58.1<br>53.0<br>38.9<br>58.4<br>52.6<br>**43.5**<br>**59.5**<br>**54.7**|
|MindMerger-Soft||||
|mGPT<br>mBERT<br>XLM-RoBERTa-large<br>M2M100-418M<br>M2M100-1.2B<br>NLLB-200-1.3B<br>NLLB-200-3.3B<br>mT5-large<br>mT5-xl|1,418 M<br>178 M<br>560 M<br>282 M<br>635 M<br>766 M<br>1,733 M<br>564 M<br>1,670 M|19.6<br>20.4<br>15.6<br>42.8<br>48.0<br>59.2<br>59.6<br>54.0<br>**61.2**<br>64.0<br>30.8<br>37.6<br>46.8<br>50.0<br>48.8<br>55.6<br>52.4<br>59.6<br>60.8<br>66.4<br>44.0<br>52.4<br>50.4<br>52.4<br>54.0<br>60.8<br>58.4<br>56.8<br>56.8<br>66.4<br>49.2<br>**52.8**<br>46.0<br>48.8<br>52.4<br>59.6<br>58.0<br>59.2<br>60.8<br>65.6<br>49.6<br>52.4<br>53.2<br>52.8<br>**54.4**<br>60.0<br>56.4<br>60.0<br>58.0<br>66.0<br>45.6<br>47.6<br>**57.6**<br>**54.4**<br>52.4<br>57.2<br>57.2<br>**60.8**<br>60.8<br>66.8<br>**52.4**<br>51.6<br>53.6<br>52.8<br>53.2<br>60.4<br>**60.0**<br>60.4<br>60.4<br>**67.6**<br>40.4<br>47.2<br>53.6<br>47.6<br>51.6<br>59.2<br>55.2<br>57.6<br>56.8<br>66.4<br>50.4<br>**52.8**<br>57.2<br>**54.4**<br>53.6<br>**61.2**<br>57.6<br>**60.8**<br>58.4<br>66.8|18.5<br>55.5<br>44.4<br>38.4<br>56.2<br>50.9<br>48.9<br>57.9<br>55.2<br>49.3<br>57.8<br>55.2<br>51.7<br>58.2<br>56.3<br>50.3<br>58.5<br>56.2<br>52.5<br>**59.3**<br>57.2<br>47.1<br>56.3<br>53.6<br>**53.5**<br>59.0<br>**57.3**|



and the encoder part of encoder-decoder models M2M100 [Fan et al., 2021], NLLB-300 [Costa-jussà et al., 2022] and mT5 [Xue et al., 2021]. The experimental results in Table 4 show that the encoder part of the encoder-decoder model and the encoder-only model are more suitable interpolate into MindMerger-Soft than the decoder-only model, which can achieve better performance than mGPT with a smaller number of parameters. M2M100 is the most cost-effective model, reaching or even exceeding the performance of XLM-RoBERTa-large and mT5-large while using only half of the parameters of XLM-RoBERTa-large and mT5-large. 

**Our augmentation-based strategy outperforms the translate-then-replace strategy.** Comparing Translate-En and MindMerger-Hard which only differ in input strategy, augmentation-based MindMerger-Hard, based on the same translation model, consistently exceeds Translate-En by 5.0%, 5.9%, 4.7%, and 4.1% in the average accuracy. MindMerger-Soft further expands its lead with an increment of the average accuracy by at least 6.6% based on the same translation model. 

**MindMerger-Soft is a better utilization of existing multilingual model.** As shown in Table 4, the performance of MindMerger-Soft consistently exceeds MindMerger-Hard under the same multilingual model with increases of average accuracy of 5.5% and 3.3% on two versions of M2M100, and 3.6% and 2.5% on two versions of NLLB-200. Although only the encoder part of the multilingual model is used, MindMerger-Soft merges LLM with a dense representation rather than decoded text based on sparse bag-of-words, enhancing the effectiveness of utilizing multilingual model. 

**A more powerful multilingual model can better enhance the multilingual capabilities.** We compared the different sizes of each encoder-decoder models and consistently observed that the larger version outperformed the smaller one in Table 4. With the help of the larger model size, the average accuracy is improved to 1.1%, 1.0%, and 3.7% on M2M100, NLLB-200, and mT5, respectively. The improvement in low-resource languages is even more obvious with an average increase in accuracy of 1.1%, 1.0% and 3.7% in M2M100, NLLB-200, and mT5, respectively. This underscores the greater imperative to enhance language understanding in low-resource languages. 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MindMerger_Efficiently_Boosting_LLM_Reasoning_in_non-English_Languages_images/nips_2024_MindMerger_Efficiently_Boosting_LLM_Reasoning_in_non-English_Languages.pdf-0008-00.png)


**----- Start of picture text -----**<br>
706560 w/o Mapping Stagew Mapping Stage 807060 MonoReason ww Augmentation Stage/o Augmentation Stage 6560 Replacement (w/o T)Augmentation (w T)<br>55 50 55<br>50<br>40<br>45 50<br>30<br>40<br>20 45<br>35<br>10<br>30 40<br>Lrl. Hrl. Avg. Lrl. Hrl. Avg. Lrl. Hrl. Avg.<br>(a) Mapping Stage (b) Augmentation Stage (c) Replacement vs. Augmentation<br>Accuracy Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 3: Ablation experiments of MindMerger-Soft on the MGSM dataset. Lrl., Hrl., and Avg. represent the average accuracy across low-resource languages, high-resource languages, and all languages, respectively. Referring to Shi et al. [2023], we regard Bn, Th, and Sw as low-resourse languages, and regard the remaining languages as high-resource languages. 

## **5.2 Ablation Studies** 

**Mapping Stage.** In Figure 3a, we ablated the mapping stage to observe its necessity. A significant drop in the performance of low-resource languages can be observed when ablating the mapping stage, which shows that the accessible general bilingual pairs are beneficial to help MindMerger-Soft understand the low-resource language information that exists in the multilingual model representation space. More detailed are reported in Appendix B.1. 

**Alignment Stage.** In Figure 3b, we ablated the augmentation stage to observe its necessity. It can be observed that after ablating the augmentation stage, even without using any task-related data, MindMerger-Soft can still outperform MonoReason on low-resource languages, which demonstrates the generalization of MindMerger-Soft that benefits from using accessible general bilingual pairs in the mapping stage. Furthermore, when the augmentation stage is added, MindMerger-Soft exhibits a significant improvement, demonstrating its effectiveness in learning the utilization of both external and built-in capabilities. More detailed are reported in Appendix B.2. 

**Replacement vs. Augmentation.** MindMerger uses the representation _**X**_ outputted by multilingual model to augment the LLM’s original representation _**T**_ rather than replace it to better utilize the builtin capabilities of LLMs. To verify the advantages of the augmentation-based strategy, we removed _**T**_ to make MindMerger-Soft a replacement-based method. As shown in Figure 3c. although with exactly the same training data and process, the performance of the replacement-based MindMerger-Soft drops significantly, indicating that it is valuable to use the built-in capabilities of LLM to understand the original input. More detailed are reported in Appendix B.3. 

Table 5: Results on the MGSM dataset based on MetaMath-Llama-13B and MetaMath-Mistral-7B. Lrl., Hrl., and Avg. represent the average accuracy across low-resource languages, high-resource languages, and all languages, respectively. Referring to Shi et al. [2023], we regard Bn, Th, and Sw as low-resourse languages, and regard the remaining languages as high-resource languages. 

|**MetaMath-Llama-13B**|**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**De**<br>**Fr**<br>**Ru**<br>**Es**<br>**En**|**Lrl.**<br>**Hrl.**<br>**Avg.**|
|---|---|---|
|MonoReason [Yu et al., 2023, Zhu et al., 2024]<br>MultiReason-Lora [Hu et al., 2022]<br>MultiReason-SFT [Zhu et al., 2024]<br>QAlign [Zhu et al., 2024]<br>LangBridge [Yoon et al., 2024]<br>Translate-En [Shi et al., 2023]|12.0<br>8.8<br>6.4<br>48.0<br>56.0<br>64.0<br>63.6<br>62.0<br>67.2<br>**70.8**<br>44.0<br>49.2<br>40.8<br>58.0<br>61.2<br>64.0<br>64.4<br>64.8<br>67.6<br>68.4<br>44.8<br>51.6<br>50.8<br>58.0<br>61.6<br>64.8<br>59.2<br>60.8<br>67.6<br>66.4<br>38.4<br>49.6<br>46.0<br>52.4<br>59.2<br>62.0<br>62.4<br>64.4<br>67.2<br>69.2<br>39.2<br>42.8<br>42.0<br>33.6<br>42.0<br>55.2<br>54.8<br>58.8<br>60.8<br>65.2<br>34.8<br>54.0<br>44.4<br>44.4<br>58.0<br>53.6<br>54.0<br>45.6<br>62.4<br>**70.8**|9.1<br>61.7<br>45.9<br>44.7<br>64.1<br>58.2<br>49.1<br>62.6<br>58.6<br>44.7<br>62.4<br>57.1<br>41.3<br>52.9<br>49.4<br>44.4<br>55.5<br>52.2|
|MindMerger-Hard<br>MindMerger-Soft|48.0<br>38.4<br>53.6<br>51.6<br>52.8<br>**66.8**<br>61.6<br>60.8<br>68.4<br>67.6<br>**55.2**<br>**59.6**<br>**56.4**<br>**60.0**<br>60.4<br>65.2<br>**63.6**<br>**68.0**<br>**69.6**<br>68.8|46.7<br>61.4<br>57.0<br>**57.1**<br>**65.1**<br>**62.7**|
|**MetaMath-Mistral-7B**<br>**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**De**<br>**Fr**<br>**Ru**<br>**Es**<br>**En**<br>**Lrl.**<br>**Hrl.**<br>**Avg.**|||
|MonoReason [Yu et al., 2023, Zhu et al., 2024]<br>38.4<br>34.8<br>16.8<br>50.8<br>57.2<br>70.4<br>70.8<br>67.2<br>71.2<br>78.0<br>30.0<br>66.5<br>55.6<br>MultiReason-Lora [Hu et al., 2022]<br>46.8<br>51.2<br>39.6<br>54.4<br>62.4<br>**72.0**<br>66.0<br>68.4<br>70.0<br>76.0<br>45.9<br>67.0<br>60.7<br>MultiReason-SFT [Zhu et al., 2024]<br>18.4<br>26.4<br>26.8<br>30.8<br>28.8<br>32.4<br>34.8<br>32.0<br>38.0<br>39.6<br>23.9<br>33.8<br>30.8<br>QAlign [Zhu et al., 2024]<br>45.6<br>51.2<br>55.2<br>49.4<br>57.2<br>59.2<br>59.8<br>60.2<br>63.6<br>65.8<br>50.7<br>59.3<br>56.7<br>LangBridge [Yoon et al., 2024]<br>50.0<br>**60.0**<br>47.2<br>58.4<br>65.6<br>68.4<br>68.8<br>68.4<br>65.6<br>65.6<br>52.4<br>65.8<br>61.8<br>Translate-En [Shi et al., 2023]<br>54.6<br>58.7<br>47.7<br>57.2<br>63.1<br>50.4<br>56.7<br>64.9<br>58.6<br>69.7<br>53.7<br>60.1<br>58.2|||
|MindMerger-Hard<br>MindMerger-Soft|52.4<br>48.4<br>**57.6**<br>**62.4**<br>60.0<br>66.4<br>66.8<br>**69.6**<br>**71.6**<br>76.4<br>**57.6**<br>59.6<br>53.2<br>57.2<br>**68.8**<br>69.2<br>**69.6**<br>68.4<br>**71.6**<br>**79.2**|52.8<br>67.6<br>63.2<br>**56.8**<br>**69.1**<br>**65.4**|



8 

## **5.3 Merging with Different LLMs** 

MindMerger can be flexibly integrated with different LLMs. To verify this, we conducted experiments on a different type of LLM, MetaMath-Mistral-7B [Jiang et al., 2023a, Yu et al., 2023], and a larger size, MetaMath-Llama-13B [Touvron et al., 2023, Yu et al., 2023]. The experimental results are shown in Table 5, MindMerger-Soft has achieved superior performance across various baselines. The average accuracy of MindMerger-Soft on the MetaMath-Llama-13B and MetaMath-Mistral-7B versions is at least 4.1 and 4.7 higher than all baselines, respectively, demonstrating the potential of extending MindMerger to a wider range of LLMs. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MindMerger_Efficiently_Boosting_LLM_Reasoning_in_non-English_Languages_images/nips_2024_MindMerger_Efficiently_Boosting_LLM_Reasoning_in_non-English_Languages.pdf-0009-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MindMerger_Efficiently_Boosting_LLM_Reasoning_in_non-English_Languages_images/nips_2024_MindMerger_Efficiently_Boosting_LLM_Reasoning_in_non-English_Languages.pdf-0009-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MindMerger_Efficiently_Boosting_LLM_Reasoning_in_non-English_Languages_images/nips_2024_MindMerger_Efficiently_Boosting_LLM_Reasoning_in_non-English_Languages.pdf-0009-04.png)


**----- Start of picture text -----**<br>
(a) LLM embeddings (b) mapping layer outputs<br>**----- End of picture text -----**<br>


Figure 4: T-SNE visualization in the spaces of the LLM embeddings and mapping layer outputs. 

## **5.4 Representation Space Changes** 

For each language, we selected the same 100 texts from the Flores-101 [Fan et al., 2021] dataset, used the mean pooling operation to obtain the representation vectors of the LLM embedding and the hidden states output by and the mapping layer, and visualized it based on T-SNE [Van der Maaten and Hinton, 2008]. As shown in Figure 4, the representation spaces of non-English on the LLM embedding are independent and away from English, especially in low-resource languages, which leads to understanding challenges for non-English and the inability to use built-in reasoning capabilities. By contrast, as shown in Figure 4b, the representations of all languages outputted by the mapping layer almost overlap with English, which reduces the difficulty for LLM to understand non-English languages and enables various languages to utilize the built-in reasoning capabilities. 

## **5.5 Supplementary Experiments** 

We experimented with several supplementary experiments, including the influence of training dataset size used in augmentation stage (Appendix A.1), the selection of mapping layers structure (Appendix A.2), the usage of encoder-decoder model in MindMerger-Soft (Appendix A.3), the quantitative analysis on representation space changes (Appendix A.4), and the translation performance of MindMerger-Soft after mapping stage (Appendix A.5). 

## **6 Conclusion** 

This paper explores a way to more fully utilize the built-in capabilities of LLMs to improve multilingual reasoning effects. We proposed MindMerger to merge the expert multilingual capabilities in the multilingual model with the skilled reasoning and not very proficient but useful multilingual capabilities in the LLM. Through more fully utilizing the potential of LLM and more effective fusion of multilingual models, the performance of MindMerger exceeds all baselines on three reasoning datasets and a language understanding dataset. In the future, we will explore the possibility of MindMerger empowering more professional skills besides reasoning, such as code generation. 

9 

## **References** 

- Janice Ahn, Rishu Verma, Renze Lou, Di Liu, Rui Zhang, and Wenpeng Yin. Large language models for mathematical reasoning: Progresses and challenges. In Neele Falk, Sara Papi, and Mike Zhang, editors, _Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics, EACL 2024: Student Research Workshop, St. Julian’s, Malta, March 21-22, 2024_ , pages 225–237. Association for Computational Linguistics, 2024. URL `https://aclanthology.org/2024.eacl-srw.17` . 

- Shervin Minaee, Tomás Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and Jianfeng Gao. Large language models: A survey. _CoRR_ , abs/2402.06196, 2024. doi: 10.48550/ARXIV.2402.06196. URL `https://doi.org/10.48550/arXiv.2402.06196` . 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, _Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022_ , 2022. URL `http://papers.nips.cc/paper_files/paper/2022/hash/ b1efde53be364a73914f58805a001731-Abstract-Conference.html` . 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. _CoRR_ , abs/2307.09288, 2023. doi: 10.48550/ARXIV.2307.09288. URL `https://doi.org/10.48550/ arXiv.2307.09288` . 

- Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b. _CoRR_ , abs/2310.06825, 2023a. doi: 10.48550/ARXIV.2310.06825. URL `https://doi.org/10.48550/arXiv.2310.06825` . 

- Shima Imani, Liang Du, and Harsh Shrivastava. Mathprompter: Mathematical reasoning using large language models. In Sunayana Sitaram, Beata Beigman Klebanov, and Jason D. Williams, editors, _Proceedings of the The 61st Annual Meeting of the Association for Computational Linguistics: Industry Track, ACL 2023, Toronto, Canada, July 9-14, 2023_ , pages 37–42. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.ACL-INDUSTRY.4. URL `https: //doi.org/10.18653/v1/2023.acl-industry.4` . 

- Weisen Jiang, Han Shi, Longhui Yu, Zhengying Liu, Yu Zhang, Zhenguo Li, and James T. Kwok. Forward-backward reasoning in large language models for verification. _CoRR_ , abs/2308.07758, 2023b. doi: 10.48550/ARXIV.2308.07758. URL `https://doi.org/10.48550/arXiv.2308. 07758` . 

- Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. _CoRR_ , abs/2309.12284, 2023. doi: 10.48550/ARXIV.2309.12284. URL `https://doi.org/10.48550/arXiv.2309.12284` . 

10 

- Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Chuanqi Tan, and Chang Zhou. Scaling relationship on learning mathematical reasoning with large language models. _CoRR_ , abs/2308.01825, 2023a. doi: 10.48550/ARXIV.2308.01825. URL `https://doi.org/10.48550/arXiv.2308. 01825` . 

- Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. Language models are multilingual chain-of-thought reasoners. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023. URL `https://openreview.net/pdf?id=fR3wGCk-IXp` . 

- Nuo Chen, Zinan Zheng, Ning Wu, Linjun Shou, Ming Gong, Yangqiu Song, Dongmei Zhang, and Jia Li. Breaking language barriers in multilingual mathematical reasoning: Insights and observations. _CoRR_ , abs/2310.20246, 2023. doi: 10.48550/ARXIV.2310.20246. URL `https: //doi.org/10.48550/arXiv.2310.20246` . 

- Linzheng Chai, Jian Yang, Tao Sun, Hongcheng Guo, Jiaheng Liu, Bing Wang, Xinnian Liang, Jiaqi Bai, Tongliang Li, Qiyao Peng, and Zhoujun Li. xcot: Cross-lingual instruction tuning for cross-lingual chain-of-thought reasoning. _CoRR_ , abs/2401.07037, 2024. doi: 10.48550/ARXIV. 2401.07037. URL `https://doi.org/10.48550/arXiv.2401.07037` . 

- Elizabeth M Brannon. The independence of language and mathematical reasoning. _Proceedings of the National Academy of Sciences_ , 102(9):3177–3178, 2005. 

- Haoyang Huang, Tianyi Tang, Dongdong Zhang, Wayne Xin Zhao, Ting Song, Yan Xia, and Furu Wei. Not all languages are created equal in llms: Improving multilingual capability by crosslingual-thought prompting. _arXiv preprint arXiv:2305.07004_ , 2023. 

- Libo Qin, Qiguang Chen, Fuxuan Wei, Shijue Huang, and Wanxiang Che. Cross-lingual prompting: Improving zero-shot chain-of-thought reasoning across languages. In _Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2023. 

- Wenhao Zhu, Shujian Huang, Fei Yuan, Shuaijie She, Jiajun Chen, and Alexandra Birch. Question translation training for better multilingual reasoning. _CoRR_ , abs/2401.07817, 2024. doi: 10.48550/ ARXIV.2401.07817. URL `https://doi.org/10.48550/arXiv.2401.07817` . 

- Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. _arXiv preprint arXiv:2310.03744_ , 2023. 

- Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _International conference on machine learning_ , pages 19730–19742. PMLR, 2023. 

- Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. In _The Twelfth International Conference on Learning Representations_ , 2023. 

- Zewei Sun, Mingxuan Wang, and Lei Li. Multilingual translation via grafting pre-trained language models. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, _Findings of the Association for Computational Linguistics: EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 16-20 November, 2021_ , pages 2735–2747. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.FINDINGS-EMNLP.233. URL `https: //doi.org/10.18653/v1/2021.findings-emnlp.233` . 

- Rachit Bansal, Bidisha Samanta, Siddharth Dalmia, Nitish Gupta, Shikhar Vashishth, Sriram Ganapathy, Abhishek Bapna, Prateek Jain, and Partha Talukdar. LLM augmented llms: Expanding capabilities through composition. _CoRR_ , abs/2401.02412, 2024. doi: 10.48550/ARXIV.2401.02412. URL `https://doi.org/10.48550/arXiv.2401.02412` . 

- Dongkeun Yoon, Joel Jang, Sungdong Kim, Seungone Kim, Sheikh Shafayat, and Minjoon Seo. Langbridge: Multilingual reasoning without multilingual supervision. _CoRR_ , abs/2401.10695, 2024. doi: 10.48550/ARXIV.2401.10695. URL `https://doi.org/10.48550/arXiv.2401. 10695` . 

11 

- Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In _International Conference on Learning Representations_ , 2022. URL `https://openreview.net/forum?id= nZeVKeeFYf9` . 

- Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. mt5: A massively multilingual pre-trained text-to-text transformer. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tür, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou, editors, _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021_ , pages 483–498. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.NAACL-MAIN.41. URL `https://doi.org/10.18653/v1/2021.naacl-main.41` . 

- Bill Yuchen Lin, Seyeon Lee, Xiaoyang Qiao, and Xiang Ren. Common sense beyond english: Evaluating and improving multilingual language models for commonsense reasoning. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021_ , pages 1274–1287. Association for Computational Linguistics, 2021. doi: 10.18653/V1/ 2021.ACL-LONG.102. URL `https://doi.org/10.18653/v1/2021.acl-long.102` . 

- Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel R. Bowman, Holger Schwenk, and Veselin Stoyanov. Xnli: Evaluating cross-lingual sentence representations. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ . Association for Computational Linguistics, 2018. 

- Fei Yuan, Yinquan Lu, Wenhao Zhu, Lingpeng Kong, Lei Li, Yu Qiao, and Jingjing Xu. Lego-mt: Learning detachable models for massively multilingual machine translation. In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki, editors, _Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023_ , pages 11518–11533. Association for Computational Linguistics, 2023b. doi: 10.18653/V1/2023.FINDINGS-ACL.731. URL `https://doi.org/10.18653/v1/2023.findings-acl.731` . 

- Adina Williams, Nikita Nangia, and Samuel R. Bowman. A broad-coverage challenge corpus for sentence understanding through inference. In Marilyn A. Walker, Heng Ji, and Amanda Stent, editors, _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2018, New Orleans, Louisiana, USA, June 1-6, 2018, Volume 1 (Long Papers)_ , pages 1112–1122. Association for Computational Linguistics, 2018. doi: 10.18653/V1/N18-1101. URL `https://doi.org/10. 18653/v1/n18-1101` . 

- Zixian Huang, Ao Wu, Jiaying Zhou, Yu Gu, Yue Zhao, and Gong Cheng. Clues before answers: Generation-enhanced multiple-choice QA. In Marine Carpuat, Marie-Catherine de Marneffe, and Iván Vladimir Meza Ruíz, editors, _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022_ , pages 3272–3287. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.NAACL-MAIN.239. URL `https://doi.org/10.18653/v1/2022.naacl-main.239` . 

- Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? A new dataset for open book question answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii, editors, _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018_ , pages 2381–2391. Association for Computational Linguistics, 2018. doi: 10.18653/V1/D18-1260. URL `https://doi.org/10.18653/v1/d18-1260` . 

- Sumithra Bhakthavatsalam, Daniel Khashabi, Tushar Khot, Bhavana Dalvi Mishra, Kyle Richardson, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord, and Peter Clark. Think you have solved direct-answer question answering? try arc-da, the direct-answer AI2 reasoning challenge. _CoRR_ , abs/2102.03315, 2021. URL `https://arxiv.org/abs/2102.03315` . 

12 

- Tushar Khot, Peter Clark, Michal Guerquin, Peter Jansen, and Ashish Sabharwal. QASC: A dataset for question answering via sentence composition. In _The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020_ , pages 8082–8090. AAAI Press, 2020. doi: 10.1609/AAAI.V34I05.6319. URL `https://doi.org/10.1609/aaai.v34i05.6319` . 

- Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, Naman Goyal, Tom Birch, Vitaliy Liptchinsky, Sergey Edunov, Michael Auli, and Armand Joulin. Beyond englishcentric multilingual machine translation. _J. Mach. Learn. Res._ , 22:107:1–107:48, 2021. URL `http://jmlr.org/papers/v22/20-1307.html` . 

- Oleh Shliazhko, Alena Fenogenova, Maria Tikhonova, Vladislav Mikhailov, Anastasia Kozlova, and Tatiana Shavrina. mgpt: Few-shot learners go multilingual, 2022. URL `https://arxiv.org/ abs/2204.07580` . 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. _CoRR_ , abs/1810.04805, 2018. URL `http://arxiv.org/abs/1810.04805` . 

- Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation learning at scale. _CoRR_ , abs/1911.02116, 2019. URL `http://arxiv.org/abs/1911.02116` . 

- Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Y. Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loïc Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang. No language left behind: Scaling humancentered machine translation. _CoRR_ , abs/2207.04672, 2022. doi: 10.48550/ARXIV.2207.04672. URL `https://doi.org/10.48550/arXiv.2207.04672` . 

- Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. _Journal of machine learning research_ , 9(11), 2008. 

13 

## **Outline** 

## A. **Supplementary Experiments** 

- A.1. Training Set Size in the Augmentation Stage (Table 6) 

- A.2. The Selection of Mapping Layers (Table 7) 

- A.3. The Usage of Encoder-Decoder Model (Table 8) 

- A.4. Quantitative Analysis on Representation Space Changes (Table 9) 

- A.5. Translation Performance (Table 10) 

## B. **Complete Results** 

- B.1. Ablation of Mapping Stage (Table 11) 

- B.2. Ablation of Augmentation Stage (Table 12) 

- B.3. Comparing Replacement and Augmentation (Table 13) 

## C. **Limitations** 

## D. **Other Tables** 

- Table 14. Dataset Statistics 

- Table 15. Prompt Templates 

- Table 16. Examples of Training Data 

## **A Supplementary Experiments** 

Table 6: Influence of training set size (per language) used in the augmentation stage. Lrl., Hrl., and Avg. represent the average accuracy across low-resource languages, high-resource languages, and all languages, respectively. Referring to Shi et al. [2023], we regard Bn, Th, and Sw as low-resourse languages, and regard the remaining languages as high-resource languages. 

|**MGSM**|**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**De**<br>**Fr**<br>**Ru**<br>**Es**<br>**En**|**Lrl.**<br>**Hrl.**<br>**Avg.**|
|---|---|---|
|0<br>100<br>1,000<br>5,000<br>10,000<br>20,000<br>30,000<br>40,000|26.8<br>26.8<br>28.8<br>31.2<br>39.6<br>54.0<br>52.4<br>46.0<br>55.2<br>65.2<br>28.4<br>31.6<br>31.6<br>34.0<br>35.6<br>56.8<br>54.0<br>48.0<br>52.8<br>62.4<br>36.8<br>43.2<br>37.2<br>40.8<br>41.2<br>57.6<br>52.0<br>52.8<br>52.0<br>62.0<br>44.0<br>50.4<br>48.0<br>42.8<br>45.6<br>56.0<br>57.6<br>58.4<br>58.4<br>66.0<br>48.4<br>53.6<br>52.8<br>**54.8**<br>**58.4**<br>54.8<br>54.8<br>59.2<br>58.8<br>66.4<br>45.6<br>**54.0**<br>54.4<br>52.4<br>53.6<br>60.4<br>54.4<br>58.8<br>**59.2**<br>**68.8**<br>50.4<br>52.8<br>**57.2**<br>54.4<br>53.6<br>**61.2**<br>**57.6**<br>60.8<br>58.4<br>66.8<br>**51.6**<br>49.6<br>54.4<br>50.0<br>54.0<br>57.2<br>56.0<br>**62.8**<br>58.8<br>66.4|27.5<br>49.1<br>42.6<br>30.5<br>49.1<br>43.2<br>39.1<br>51.2<br>47.6<br>47.5<br>55.0<br>52.7<br>51.6<br>58.2<br>56.0<br>51.3<br>58.2<br>55.9<br>**53.5**<br>**59.0**<br>**57.3**<br>51.9<br>57.9<br>56.1|



## **A.1 Training Set Size in the Alignment Stage** 

We randomly sampled different size of training set to train MindMerger-Soft on the mathematical reasoning task and reported the results on MGSM in Table 6. Despite only 100 samples per language is used, MindMerger-Soft can be improved by 3.0% in low-resource languages compared to without augmentation stage. The training data size has a significant impact on the performance of MindMerger within the data size range of 10,000. MindMerger-Soft achieves the best result when using 30,000 samples per language, and better results may require enhancing the model’s built-in capabilities.. Considering the query training data for various tasks can be easily obtained from public translation models, and training on a scale of tens of thousands is cost-effective, MindMerger can be readily adapted to a wider range of tasks. 

14 

Table 7: The selection of mapping layers. # Parm represents parameters size of the mapping layers. Lrl., Hrl., and Avg. represent the average accuracy across low-resource languages, high-resource languages, and all languages, respectively. Referring to Shi et al. [2023], we regard Bn, Th, and Sw as low-resourse languages, and regard the remaining languages as high-resource languages. 

|**MGSM**|**# Parm**|**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**De**<br>**Fr**<br>**Ru**<br>**Es**<br>**En**|**Lrl.**<br>**Hrl.**<br>**Avg.**|
|---|---|---|---|
|Linear<br>2 layers MLP<br>3 layers MLP<br>QFormer|4 M<br>10 M<br>14 M<br>42 M|50.8<br>50.0<br>51.2<br>47.6<br>51.2<br>60.0<br>58.0<br>56.4<br>**61.6**<br>66.0<br>50.4<br>**52.8**<br>**57.2**<br>**54.4**<br>53.6<br>**61.2**<br>57.6<br>**60.8**<br>58.4<br>66.8<br>**51.6**<br>50.0<br>51.2<br>49.6<br>**54.4**<br>58.0<br>**58.4**<br>60.0<br>61.2<br>**69.2**<br>1.6<br>0.8<br>1.6<br>17.2<br>22.4<br>34.0<br>34.4<br>26.0<br>36.8<br>52.8|50.7<br>57.3<br>55.3<br>**53.5**<br>**59.0**<br>**57.3**<br>50.9<br>58.7<br>56.4<br>1.3<br>31.9<br>22.8|



## **A.2 The Selection of Mapping Layers** 

We experimented with the performance of MindMerger-Soft when using different mapping layers in Table 7. The 2 layers MLP used in the main experiment achieved the best results, while the effect of Linear will be lower in comparison. This may be the insufficient space transfer capability of Linear due to the small number of parameters. As for QFormer [Li et al., 2023], which converts variable-length input into fixed-length hidden states output, is a popular model for model merging in computer vision, but it does not work well on MindMerger-Soft. This could be attributed to the multilingual capabilities of LLMs, which may render features diverging too much from the original expression less readily accepted. 

Table 8: The usage of encoder-decoder model (M2M100-1.2B). Lrl., Hrl., and Avg. represent the average accuracy across low-resource languages, high-resource languages, and all languages, respectively. Referring to Shi et al. [2023], we regard Bn, Th, and Sw as low-resourse languages, and regard the remaining languages as high-resource languages. 

|**MGSM**|**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**De**<br>**Fr**<br>**Ru**<br>**Es**<br>**En**|**Lrl.**<br>**Hrl.**<br>**Avg.**|
|---|---|---|
|Encoder<br>Decoder<br>Encoder + Decoder|**50.4**<br>**52.8**<br>**57.2**<br>**54.4**<br>**53.6**<br>**61.2**<br>**57.6**<br>**60.8**<br>**58.4**<br>66.8<br>40.4<br>32.0<br>45.6<br>46.8<br>46.0<br>56.8<br>56.0<br>55.2<br>58.0<br>**67.6**<br>44.0<br>36.4<br>44.8<br>49.6<br>49.6<br>55.6<br>53.6<br>56.0<br>**58.4**<br>66.8|**53.5**<br>**59.0**<br>**57.3**<br>39.3<br>55.2<br>50.4<br>41.7<br>55.7<br>51.5|



## **A.3 The Usage of Encoder-Decoder Model** 

In all other experiments, for the encoder-decoder model, we only use its encoder part. In this section, we compared the performance of MindMerger-Soft using the decoder. Specifically, we use the translation model M2M100-1.2B autoregressively to generate English text, and take the hidden states corresponding to the generated text, and input them into the mapping layer alone or after concatenating with the output of the encoder. The experimental results are shown in Figure 8, where the performance of MindMerger-Soft dropped after adding the hidden states output by the decoder. 

Table 9: The changes in the representation space within MindMerger as well as its constituent models the multilingual model (mT5-xl), and the LLM (MetaMath-Llama-7B). Lrl., Hrl., and Avg. represent the average accuracy across low-resource languages, high-resource languages, and all languages, respectively. Referring to Shi et al. [2023], we regard Bn, Th, and Sw as low-resourse languages, and regard the remaining languages as high-resource languages. 

|**Cosine Similarity**|**Cosine Similarity**|**Bn**_→_**En**<br>**Th**_→_**En**<br>**Sw**_→_**En**<br>**Ja**_→_**En**<br>**Zh**_→_**En**<br>**De**_→_**En**<br>**Fr**_→_**En**<br>**Ru**_→_**En**<br>**Es**_→_**En**|**Url.**<br>**Hrl.**<br>**Avg.**|
|---|---|---|---|
|Multilingual model|Embedding Layer<br>Last Layer|0.29<br>0.27<br>0.43<br>0.31<br>0.31<br>0.49<br>0.51<br>0.37<br>0.48<br>0.91<br>0.89<br>0.92<br>0.92<br>0.93<br>0.95<br>0.95<br>0.94<br>0.92|0.33<br>0.41<br>0.38<br>0.91<br>0.94<br>0.93|
|LLM|Embedding Layer<br>Last Layer|0.18<br>0.15<br>0.34<br>0.28<br>0.28<br>0.39<br>0.42<br>0.31<br>0.41<br>0.08<br>0.07<br>-0.08<br>0.11<br>0.13<br>0.04<br>0.12<br>0.05<br>0.09|0.22<br>0.35<br>0.31<br>0.02<br>0.09<br>0.07|
|MindMerger|Mapping Module<br>LLM Layer|0.92<br>0.91<br>0.91<br>0.93<br>0.91<br>0.96<br>0.94<br>0.94<br>0.93<br>0.32<br>0.32<br>0.46<br>0.46<br>0.46<br>0.55<br>0.62<br>0.52<br>0.56|0.91<br>0.94<br>0.93<br>0.37<br>0.53<br>0.47|
|**Recall@1 (%)**||**Bn**_→_**En**<br>**Th**_→_**En**<br>**Sw**_→_**En**<br>**Ja**_→_**En**<br>**Zh**_→_**En**<br>**De**_→_**En**<br>**Fr**_→_**En**<br>**Ru**_→_**En**<br>**Es**_→_**En**|**Url.**<br>**Hrl.**<br>**Avg.**|
|Multilingual model|Embedding layer<br>Last layer|1.2<br>3.0<br>16.2<br>1.8<br>4.9<br>28.2<br>34.6<br>10.7<br>29.1<br>99.3<br>99.7<br>97.8<br>99.7<br>99.9<br>100<br>100<br>100<br>99.9|6.8<br>18.2<br>14.4<br>98.9<br>99.9<br>99.6|
|LLM|Embedding layer<br>Last layer|0.7<br>1.9<br>5.9<br>3.2<br>4.7<br>30.2<br>41.7<br>8.7<br>27.3<br>0.4<br>1.8<br>0.2<br>5.8<br>12.6<br>11.5<br>47.3<br>44.9<br>27.0|2.8<br>19.3<br>13.8<br>0.8<br>24.9<br>16.8|
|MindMerger|Mapping module<br>LLM module|99.4<br>98.9<br>99.4<br>99.6<br>99.9<br>100<br>100<br>99.8<br>99.6<br>52.8<br>76.0<br>64.3<br>91.6<br>96.4<br>98.0<br>99.2<br>98.1<br>88.9|99.2<br>99.8<br>99.6<br>64.4<br>95.4<br>85.0|



15 

## **A.4 Quantitative Analysis on the Representation Space Changes** 

We further quantitatively analyzed the changes in the representation space of the models. Two metrics were employ to quantify space relationships: (1) Cosine similarity, indicating the similarity between the same query expressed in different languages, and (2) Recall@1, representing the percentage of top-1 retrieve results from a pool of 1000 English query that correspond to the same query. 

The experimental results are shown in Table 9. multilingual model demonstrates the capability to map inputs in different languages to a language-agnostic space. Within this space, the average cosine similarity and Recall@1 from various languages to English reach as high as 0.93 and 99.6. On the contrary, LLM lacks this capability. The average cosine similarity and Recall@1 in the output space of LLM are only 0.07 and 16.8, respectively. The significant disparity in the capabilities to construct language-agnostic representations suggests the potential for LLM to be complemented by multilingual model. 

For MindMerger-Soft, consisting of multilingual model and LLM, the mapping layer generates a representation with an average cosine similarity of 0.93 and an average Recall@1 of 99.6. This highly language-agnostic representation as input to LLM effectively helps LLM understand different languages. Compared with the huge difference between the language representations output when using only LLM, the average cosine similarity of LLM output in MindMerger-Soft increased from 0.07 to 0.47, and the average Recall@1 increased from 16.8 to 85.0. 

## **A.5 Translation Performance** 

Table 10: Translation results on Flores-101 dataset. spBLEU is used as evaluatoin metric. 

|**Flores-101**|**Bn**_→_**En**<br>**Th**_→_**En**<br>**Sw**_→_**En**<br>**Ja**_→_**En**<br>**Zh**_→_**En**<br>**De**_→_**En**<br>**Fr**_→_**En**<br>**Ru**_→_**En**<br>**Es**_→_**En**|**Avg.**|
|---|---|---|
|Translation Models<br>|||
|MetaMath-Llama-7B (5-shots)<br>Lego-MT<br>M2M100-418M<br>M2M100-1.2B<br>NLLB-200-1.3B<br>NLLB-3.3B|0.7<br>1.0<br>0.6<br>1.9<br>2.3<br>2.6<br>2.7<br>2.6<br>2.1<br>31.6<br>26.4<br>37.2<br>25.5<br>29.6<br>46.1<br>46.4<br>35.2<br>31.5<br>25.1<br>18.2<br>26.6<br>20.2<br>21.2<br>36.4<br>39.0<br>28.4<br>25.8<br>27.7<br>23.9<br>34.1<br>24.8<br>26.7<br>42.9<br>43.9<br>34.3<br>30.0<br>36.8<br>30.7<br>43.9<br>28.6<br>30.4<br>45.9<br>47.4<br>38.0<br>34.8<br>**38.7**<br>**33.2**<br>**46.3**<br>**30.6**<br>**32.2**<br>**47.2**<br>**48.7**<br>**39.1**<br>**36.5**|1.8<br>34.4<br>26.8<br>32.0<br>37.4<br>**39.2**|
|MindMerger-Soft (w/o**_T_**)<br>|||
|mGPT<br>mBERT<br>XLM-RoBERTa-large<br>mT5-large<br>mT5-xl<br>Lego-MT<br>M2M100-418M<br>M2M100-1.2B<br>NLLB-200-1.3B<br>NLLB-200-3.3B|2.7<br>2.3<br>3.8<br>2.6<br>4.7<br>6.7<br>10.3<br>4.7<br>6.7<br>10.4<br>8.2<br>12.4<br>12.5<br>16.3<br>23.9<br>26.1<br>18.7<br>19.0<br>12.6<br>14.2<br>17.5<br>9.2<br>12.6<br>24.6<br>26.7<br>20.2<br>16.9<br>18.0<br>18.1<br>25.1<br>13.8<br>16.3<br>30.4<br>34.3<br>25.3<br>24.6<br>25.9<br>23.5<br>34.1<br>19.8<br>22.3<br>37.3<br>39.1<br>30.7<br>29.0<br>30.2<br>25.6<br>37.6<br>24.5<br>**29.0**<br>**43.7**<br>45.0<br>35.3<br>31.6<br>26.3<br>15.0<br>27.3<br>23.8<br>25.6<br>36.8<br>39.4<br>31.3<br>29.6<br>26.6<br>20.0<br>32.6<br>24.5<br>27.0<br>39.3<br>41.7<br>33.3<br>31.2<br>30.1<br>26.4<br>38.5<br>21.8<br>24.4<br>38.3<br>41.8<br>32.1<br>30.3<br>**34.5**<br>**29.5**<br>**43.9**<br>**26.6**<br>**29.0**<br>43.2<br>**45.2**<br>**35.8**<br>**31.8**|5.0<br>16.4<br>17.2<br>22.9<br>29.1<br>33.6<br>28.3<br>30.7<br>31.5<br>**35.5**|



In Table 10, we used the translation dataset Flores-101 [Fan et al., 2021] as the evaluation set and presented the spBLEU performance of various versions of MindMerger-Soft trained on mapping stage. In contrast to the poor performance of the same LLM, MetaMath-Llama, MindMerger-Soft greatly boost it translation capabilities. Remarkably, the performance of MindMerger, based on the M2M100-418M encoder part, even surpasses the M2M100-418M translation model itself. Although the encoder-only models mBERT and XLM-RoBERTa-large do not have translation capabilities, MindMerger can still learn translation capabilities based on them. 

## **B Complete Results** 

## **B.1 Ablation of Mapping Stage** 

The complete experimental results are shown in Table 11. The average accuracy of MindMerger-Soft decreased by 6.0% and 5.5% on the MGSM dataset, and by 8.3% and 7.3% on the MSVAMP dataset, demonstrating the effectiveness of this stage. The mapping stage is particularly crucial for enhancing the capabilities of low-resource languages. After ablating this stage, the average accuracy for low-resource languages drops 20.2% and 14.4% on the MGSM dataset, and 18.8% and 11.0% on the MSVAMP dataset. The mapping stage can also be helpful for high-resource languages. 

16 

Table 11: Ablation of mapping stage.Lrl., Hrl., and Avg. represent the average accuracy across low-resource languages, high-resource languages, and all languages, respectively. Referring to Shi et al. [2023], we regard Bn, Th, and Sw as low-resourse languages, and regard the remaining languages as high-resource languages. 

|**MGSM**|**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**De**<br>**Fr**<br>**Ru**<br>**Es**<br>**En**|**Lrl.**<br>**Hrl.**<br>**Avg.**|
|---|---|---|
|MindMerger (M2M100)<br>w/o mappingStage<br>MindMerger (mT5)<br>w/o mapping Stage|**49.6**<br>**52.4**<br>**53.2**<br>**52.8**<br>**54.4**<br>**60.0**<br>56.4<br>**60.0**<br>58.0<br>66.0<br>29.2<br>32.4<br>32.8<br>52.0<br>52.8<br>58.4<br>**59.6**<br>57.2<br>**60.0**<br>**68.4**<br>**50.4**<br>**52.8**<br>**57.2**<br>**54.4**<br>**53.6**<br>**61.2**<br>57.6<br>**60.8**<br>58.4<br>**66.8**<br>41.6<br>35.2<br>40.4<br>46.8<br>51.6<br>57.6<br>**60.4**<br>57.2<br>**61.2**<br>66.4|**51.7**<br>58.2<br>**56.3**<br>31.5<br>**58.3**<br>50.3<br>**53.5**<br>**59.0**<br>**57.3**<br>39.1<br>57.3<br>51.8|
|**MSVAMP**<br>**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**De**<br>**Fr**<br>**Ru**<br>**Es**<br>**En**<br>**Lrl.**<br>**Hrl.**<br>**Avg.**|||
|MindMerger (M2M100)<br>w/o mappingStage<br>MindMerger (mT5)<br>w/o mapping Stage|**51.7**<br>**56.0**<br>**55.2**<br>**57.2**<br>**57.7**<br>**63.1**<br>**62.1**<br>**60.2**<br>**63.4**<br>64.3<br>34.1<br>39.9<br>32.4<br>53.7<br>53.6<br>58.6<br>57.5<br>55.6<br>57.9<br>**64.6**<br>**52.0**<br>**53.4**<br>**54.0**<br>**59.0**<br>**61.7**<br>**64.1**<br>**64.0**<br>**63.3**<br>**65.0**<br>**67.7**<br>40.4<br>42.2<br>43.7<br>54.2<br>54.2<br>60.3<br>57.8<br>55.4<br>60.0<br>62.7|**54.3**<br>**61.1**<br>**59.1**<br>35.5<br>57.4<br>50.8<br>**53.1**<br>**63.5**<br>**60.4**<br>42.1<br>57.8<br>53.1|



After removing the mapping stage, on the MSVAMP dataset, the average accuracy of high-resource languages dropped by 3.7% and 5.7% on the two MindMerger-Soft implementations. 

Table 12: Ablation of augmentation stage. Lrl., Hrl., and Avg. represent the average accuracy across low-resource languages, high-resource languages, and all languages, respectively. Referring to Shi et al. [2023], we regard Bn, Th, and Sw as low-resourse languages, and regard the remaining languages as high-resource languages. 

|**MGSM**|**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**De**<br>**Fr**<br>**Ru**<br>**Es**<br>**En**|**Lrl.**<br>**Hrl.**<br>**Avg.**|
|---|---|---|
|MonoReason<br>MindMerger (M2M100)<br>w/o augmentation Stage<br>MindMerger (mT5)<br>w/o augmentation Stage|7.6<br>5.6<br>5.2<br>34.0<br>45.2<br>54.0<br>56.8<br>51.6<br>58.8<br>65.5<br>**49.6**<br>**52.4**<br>**53.2**<br>**52.8**<br>**54.4**<br>**60.0**<br>**56.4**<br>**60.0**<br>**58.0**<br>**66.0**<br>25.6<br>23.2<br>26.4<br>37.2<br>40.8<br>51.2<br>51.6<br>49.2<br>56.0<br>64.0<br>**50.4**<br>**52.8**<br>**57.2**<br>**54.4**<br>**53.6**<br>**61.2**<br>**57.6**<br>**60.8**<br>**58.4**<br>**66.8**<br>26.8<br>26.8<br>28.8<br>31.2<br>39.6<br>54.0<br>52.4<br>46.0<br>55.2<br>65.2|6.1<br>52.3<br>38.4<br>**51.7**<br>**58.2**<br>**56.3**<br>25.1<br>50.0<br>42.5<br>**53.5**<br>**59.0**<br>**57.3**<br>27.5<br>49.1<br>42.6|
|**MSVAMP**<br>**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**De**<br>**Fr**<br>**Ru**<br>**Es**<br>**En**<br>**Lrl.**<br>**Hrl.**<br>**Avg.**|||
|MonoReason<br>MindMerger (M2M100)<br>w/o augmentation Stage<br>MindMerger (mT5)<br>w/o augmentation Stage|15.0<br>17.1<br>15.4<br>51.9<br>54.4<br>60.9<br>62.2<br>59.3<br>63.3<br>65.5<br>**51.7**<br>**56.0**<br>**55.2**<br>**57.2**<br>**57.7**<br>**63.1**<br>**62.1**<br>**60.2**<br>**63.4**<br>**64.3**<br>29.4<br>27.6<br>28.2<br>51.2<br>52.5<br>60.4<br>60.6<br>58.2<br>58.8<br>**64.3**<br>**52.0**<br>**53.4**<br>**54.0**<br>**59.0**<br>**61.7**<br>**64.1**<br>**64.0**<br>**63.3**<br>**65.0**<br>**67.7**<br>37.5<br>35.2<br>37.5<br>52.8<br>53.2<br>61.3<br>61.7<br>57.6<br>63.3<br>64.0|15.8<br>59.6<br>46.2<br>**54.3**<br>**61.1**<br>**59.1**<br>28.4<br>58.0<br>49.1<br>**53.1**<br>**63.5**<br>**60.4**<br>36.7<br>59.1<br>52.4|



## **B.2 Ablation of Augmentation Stage** 

The complete results are shown in Table 12. Even without the mapping stage, MindMerger-Soft solely training in general bilingual pairs can still improve the performance of LLMs in low-resource languages. For the MGSM and MSVAMP datasets, the average accuracy of MindMerger-Soft in low-resource languages increases by MonoReason 19.0% and 12.6% based on M2M100-1.2B, and increases by MonoReason 21.4% and 20.9% based on mT5-xl. With the help of the augmentation stage, the reasoning capabilities of MindMerger-Soft in low-resource languages are further enhanced. Comparing to the MindMerger-Soft without augmentation stage in the low-resource languages, augmentation stage brings improvements of 26.6% and 25.9% based on M2M100-1.2B, and 26.0% and 16.4% based on mT5-xl. Moreover, augmentation stage is also valueable for high-resource languages, bringing improvements of 3.1% based on M2M100-1.2B and 4.4% based on mT5-xl for the MSVAMP dataset in the average accuracy across high-resource languages. 

## **B.3 Comparing Replacement and Augmentation** 

The complete experimental results are shown in Table 13. It can be observed that the augmentation strategy is more significantly helpful in improving high-resource language reasoning capabilities than low-resource languages. Compared with the replacement version of MindMerger-Soft, the augmentation strategy improves the average accuracy by 1.1%, 2.1%, 3.2%, and 4.6% across highresource languages in the four experimental results. This suggests that, in contrast to the replacement strategy that neglects these capabilities, the augmentation strategy effectively takes advantage of them, thereby unlocking the full potential of LLMs. For low-resource languages, the augmentation strategy 

17 

Table 13: Comparison between replacement (w/o _**T**_ ) and augmentation. Lrl., Hrl., and Avg. represent the average accuracy across low-resource languages, high-resource languages, and all languages, respectively. Referring to Shi et al. [2023], we regard Bn, Th, and Sw as low-resourse languages, and regard the remaining languages as high-resource languages. 

|**MGSM**|**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**De**<br>**Fr**<br>**Ru**<br>**Es**<br>**En**|**Url.**<br>**Hrl.**<br>**Avg.**|
|---|---|---|
|MindMerger-Soft (M2M100)<br>Replacement(w/o**_T_** )<br>MindMerger-Soft (mT5)<br>Replacement (w/o**_T_**)|49.6<br>**52.4**<br>**53.2**<br>52.8<br>**54.4**<br>**60.0**<br>**56.4**<br>**60.0**<br>58.0<br>**66.0**<br>**50.8**<br>44.4<br>50.8<br>**54.4**<br>50.4<br>57.2<br>55.2<br>58.0<br>**61.2**<br>63.2<br>50.4<br>52.8<br>**57.2**<br>**54.4**<br>**53.6**<br>**61.2**<br>57.6<br>**60.8**<br>58.4<br>**66.8**<br>**52.2**<br>**53.6**<br>54.8<br>46.8<br>52.0<br>59.2<br>**58.0**<br>58.4<br>**60.0**<br>63.6|**51.7**<br>**58.2**<br>**56.3**<br>48.7<br>57.1<br>54.6<br>**53.5**<br>**59.0**<br>**57.3**<br>**53.5**<br>56.9<br>55.9|
|**MSVAMP**<br>**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**De**<br>**Fr**<br>**Ru**<br>**Es**<br>**En**<br>**Url.**<br>**Hrl.**<br>**Avg.**|||
|MindMerger-Soft (M2M100)<br>Replacement(w/o**_T_** )<br>MindMerger-Soft (mT5)<br>Replacement (w/o**_T_**)|**51.7**<br>**56.0**<br>**55.2**<br>**57.2**<br>**57.7**<br>**63.1**<br>**62.1**<br>**60.2**<br>**63.4**<br>**64.3**<br>50.0<br>48.5<br>52.3<br>55.7<br>57.2<br>60.4<br>59.7<br>55.9<br>57.9<br>58.5<br>**52.0**<br>53.4<br>54.0<br>**59.0**<br>**61.7**<br>**64.1**<br>**64.0**<br>**63.3**<br>**65.0**<br>**67.7**<br>51.7<br>**54.7**<br>**54.3**<br>54.8<br>56.5<br>61.1<br>59.7<br>58.5<br>61.1<br>60.8|**54.3**<br>**61.1**<br>**59.1**<br>50.3<br>57.9<br>55.6<br>53.1<br>**63.5**<br>**60.4**<br>**53.6**<br>58.9<br>57.3|



also brings improvements of 3.0% and 4.0% for MindMerger-Soft based on M2M100-1.2B, but on par with the replacement strategy for MindMerger-Soft based on mT5-xl. It could be the multilingual capabilities of mT5-xl on low-resource languages are significantly stronger than LLMs that it can completely replace LLMs, while the slightly weaker multilingual capabilities of M2M100-1.2B still need to be complementary to LLMs. 

## **C Limitations** 

Although our experimental results show that merging external and built-in language understanding capabilities can help improve the model’s multilingual reasoning capabilities, it is unclear the effective boundaries of this mechanism. On one hand, a multilingual model with significantly weaker language capabilities compared to LLMs might not be able to assist LLMs effectively. On the other hand, LLMs may not be able to provide additional built-in knowledge for a multilingual model whose language capabilities are much stronger. Therefore, more experiments are needed to explore the effective boundaries of MindMerger. 

## **D Other Tables** 

Table 14: Dataset statistics. # Train, # Test, and # Lang refer to the size of query translation training set per language, the size of multilingual test set per language, and the number of language we evaluated for each dataset, respectively. 

|Dataset|Task|# Train|# Test|# Lang|
|---|---|---|---|---|
|MGSM|Math|30,000|250|10|
|MSVAMP|Math|30,000|1,000|10|
|X-CSQA|Commonsense|8,888|1,000|16|
|XNLI|NLI|2,490|5,010|15|



18 

Table 15: The prompts of mathematical reasoning task. 

MonoReason & MultiReason & QAlign Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {query} ### Response: Let’s think step by step. Translate-En Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {translated query} ### Response: Let’s think step by step. MindMerger-Hard Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {query} ### Translated instruction: {translated query}### Response: Let’s think step by step. 

Table 16: Examples of training data. 

## **Mapping Stage** 

**Input (Zh):** `文档中没有元素属于该组时，该名称会被作为根结点显示在结构树上。` **Output:** It will be shown in the structure tree as a top node when there are no elements belonging to this group in the document. 

|**Augmentation Stage (Math)**|
|---|
|**Input (Zh):**<br>`伯特每天都填报纸上的每日填字游戏。他每两周就用完一支铅笔。平均`|
|`而言，他用完一支铅笔需要`1050`个字。每个填字游戏平均有多少个字？`|
|**Input (En):** Bert flls out the daily crossword puzzle in the newspaper every day. He uses a|
|pencil to fll out the puzzles every two weeks. On average, it takes him 1050 words to use up a|
|pencil. How many words are in each crossword puzzle on average?|
|**Output:** If Bert uses up a pencil to fll out the puzzles every two weeks and it takes him 1050|
|words to use up a pencil, then he must be flling out 1050 words of crossword puzzles every|
|two weeks. To fnd out how many words are in each daily crossword puzzle, we need to divide|
|the total number of words (1050) by the number of days in two weeks (14). So, there are|
|1050/14 = 75 words in each daily crossword puzzle on average. #### The answer is: 75|



## **Augmentation stage (X-CSQA)** 

**Input (Zh):** `酒后驾车的人可能会被指控什么` ? (A) `惩罚` (B) `逮捕` (C) `车祸` (D) `胡言乱语` (E) `酒后驾驶` **Input (En):** What is someone operating a vehicle likely to be accused of after becoming inebriated? (A) punish (B) arrest (C) automobile accidents (D) talking nonsense (E) drunk driving **Output:** E **Augmentation stage (XNLI) Input (Zh):** Premise: `她不太明白。` Hypothesis: `事实上，她没有理解。` **Input (En):** Premise: She doesn’t really understand. Hypothesis: Actually, she doesn’t get it. **Output:** Entailment 

19 

