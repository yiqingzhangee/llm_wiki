# **Meaningful Learning: Enhancing Abstract Reasoning in Large Language Models via Generic Fact Guidance** 

**Kai Xiong** _[♠]_ **Xiao Ding** _[♠∗]_ **Ting Liu** _[♠]_ **Bing Qin** _[♠]_ **Dongliang Xu**[♦] **Qing Yang**[♦] **Hongtao Liu**[♦] **Yixin Cao** _[♣∗]_ 

> _♠_ Research Center for Social Computing and Information Retrieval Harbin Institute of Technology, Harbin, China 

> _♣_ Fudan University, Shanghai, China 

♦Du Xiaoman (Beijing) Science Technology Co., Ltd., Beijing, China 

```
{kxiong,xding,tliu,qinb}@ir.hit.edu.cn
```

```
{xudongliang,yangqing,liuhongtao01}@duxiaoman.com
yxcao@fudan.edu.cn
```

## **Abstract** 

Large language models (LLMs) have developed impressive performance and strong explainability across various reasoning scenarios, marking a significant stride towards mimicking human-like intelligence. Despite this, when tasked with several simple questions supported by a generic fact, LLMs often struggle to abstract and apply the generic fact to provide consistent and precise answers, revealing a deficiency in abstract reasoning abilities. This has sparked a vigorous debate about whether LLMs are genuinely reasoning or merely memorizing. In light of this, we design a preliminary study to quantify and delve into the abstract reasoning abilities of existing LLMs. Our findings reveal a substantial discrepancy between their general reasoning and abstract reasoning performances. To relieve this problem, we tailor an abstract reasoning dataset (AbsR) together with a meaningful learning paradigm to teach LLMs how to leverage generic facts for reasoning purposes. The results show that our approach not only boosts the general reasoning performance of LLMs but also makes considerable strides towards their capacity for abstract reasoning, moving beyond simple memorization or imitation to a more nuanced understanding and application of generic facts. The code is available at `https://github.com/Waste-Wood/MeanLearn` . 

## **1 Introduction** 

Humans proceed with meaningful learning to induce common patterns or high-level abstractions to acquire abstract reasoning abilities [29, 37]. Such capabilities allow us to apply broad principles across diverse situations, demonstrating a deep understanding and versatile application of knowledge. As shown in Figure 1 (a), humans can deduce “rock dissolved” and “the skin suffers pain” when given “adding rock into hydrochloric acid” and “acid touches human skin”, respectively. This stems from the established expertise in the human mind (the generic fact “acid is corrosive”), and extends to applications in different scenarios. Of late, remarkable headways of LLMs have pushed AI much further towards human-like intelligence [2, 47]. Interestingly, can LLMs act like humans to instinctively consider the generic fact for flexible applications in different scenarios? 

In our pilot investigation, we observed that LLMs appear to lack satisfying capabilities in abstract reasoning. As illustrated in Table 1, there is a notable discrepancy—exceeding 17%—between vanilla 

> _∗_ Corresponding Authors 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 

Table 1: The vanilla accuracy and abstract reasoning accuracy ( `AbsAcc` ) on e-CARE [14]. We will formally define `AbsAcc` in Sec. 2.1. 

|**_Generic Fact:_**Acid is corrosive<br>**_Question1_:**Adding rock into hydrochloric acid<br>Rock dissolved<br>The skin suffers pain<br>🧑<br>🧑<br>🤖<br>✓<br>✓<br>Rock corroded<br>🤖<br>Skin moisturized<br>✓<br>✘<br>**_Question2:_**Acid touches human skin<br>_(a) Humans_<br>_conduct reasoning    (b)LLMs_<br>_conduct reasoning_<br>Figure 1: The responses of (a) Humans and (b)<br>LLMs when facing two questions which are sup-<br>ported by the same generic fact.|Table 1: The vanilla accuracy and abstract rea-<br>soning accuracy (`AbsAcc`) on e-CARE [14].<br>We will formally defne`AbsAcc`in Sec. 2.1.|
|---|---|
||**Methods**<br>Size Vanilla Accuracy `AbsAcc`|
||LLaMA-2 [47]<br>7B<br>51.67<br>24.82<br>13B<br>68.45<br>42.94|
||Orca-2 [35]<br>7B<br>74.23<br>51.19<br>13B<br>79.83<br>59.47|
||GPT-3.5 [7]<br>>20B<br>83.25<br>66.08|
||Human [14]<br>–<br>92.00<br>89.90|



Figure 1: The responses of (a) Humans and (b) LLMs when facing two questions which are supported by the same generic fact. 

accuracy and abstract reasoning accuracy ( `AbsAcc` ) for all LLMs, while humans show only a small disparity. Despite the extensive pre-training of LLMs, which equips them with a vast repository of generic facts, they seem unable to leverage this information as flexibly as humans do (Figure 1 (b)). 

In this paper, we aim to systematically and quantitatively investigate the abstract reasoning of LLMs. We first formally define the abstract reasoning metric. Then we carry out abstract reasoning and knowledge probing tasks for a preliminary study. Our investigation comprises two main components: the reasoning tasks can quantify the abstract reasoning capabilities of LLMs. The generic fact probing task can give a further analysis from the perspective of generic fact mastery. Hereafter, we improve abstract reasoning in LLMs based on the analysis of the preliminary study. Unlike existing chain-of-thought (CoT) methods [52, 27], which conduct reasoning with uncontrolled step-by-step explanation, we create a dataset AbsR tailored for abstract reasoning. AbsR not only includes generic facts but also their guided explanations, offering a coached approach to understand the process of reasoning with common patterns. Finally, to make LLMs subconsciously exploit the generic fact like humans, we design a simple but effective learning paradigm called meaningful learning (MeanLearn) to simulate the implicit knowledge learning process [13], enabling LLMs to implicitly learn and utilize generic facts for reasoning without requiring generic facts as input. 

We evaluate MeanLearn on six out-of-domain (OOD) reasoning and language understanding benchmarks. Experimental results demonstrate that MeanLearn not only improves general reasoning of LLMs but also excels in abstract reasoning. This distinction in performance is particularly pronounced in the realm of abstract reasoning, underscoring the unique efficacy of our approach in nurturing the higher-order thinking skills that are essential for sophisticated cognitive processing in artificial intelligence systems. Further analysis and ablation studies yield additional evidence supporting the effectiveness of our approach. We summarize our contributions as follows: 

- We provide a systematic and quantitative analysis of abstract reasoning in current LLMs, and we develop an abstract reasoning dataset with generic-fact-guided explanations. 

- We propose meaningful learning to improve abstract reasoning in LLMs with generic fact. 

- We achieve significant improvement in general reasoning and abstract reasoning on various OOD reasoning and language understanding benchmarks. 

## **2 Abstract reasoning study** 

In this section, we initially define the evaluation metric of abstract reasoning to establish quantifiable standards. Then we introduce the LLMs used in this study. Finally, we conduct two experiments: one to directly estimate the abstract reasoning abilities of LLMs, and the other to further analyze it from the perspective of general fact mastery. 

## **2.1 Abstract reasoning metric** 

**Abstract reasoning** requires models to apply general patterns or high-level abstractions to different scenarios or questions. Different from general reasoning, which focuses on each single example, abstract reasoning takes the examples supported by a generic fact as a whole. We define abstract reasoning accuracy due to the lack of a proper metric. 

2 

Table 2: Accuracy of generic fact probing. 

|**LLMs**|**LLaMA-2**<br>**Orca-2**<br>**GPT-3.5**|
|---|---|
||7B<br>13B<br>7B<br>13B<br>>20B|
|**Accu.**|62.44 70.32 26.97 12.78<br>77.30|



Table 3: The categorized abstract reasoning accuracy based on whether the generic facts are known. 

|**Category**<br>**Metrics**|**LLaMA-2**<br>**Orca-2**<br>**GPT-3.5**|
|---|---|
||7B<br>13B<br>7B<br>13B<br>>20B|
|**Known**<br>`AbsAcc`|34.26 43.51 64.98 81.51<br>68.31|
|**Unknown** `AbsAcc`|17.19 31.84 49.00 58.38<br>58.49|



Given a dataset _D_ with _n_ generic facts _R_ = _{r_ 1 _, r_ 2 _· · · , rn}_ , for the _i_ -th generic fact _ri_ , it supports _mi_ examples _Si_ = _{s[i]_ 1 _[, s]_ 2 _[i][,][ · · ·][, s][i] mi[} ∈][D]_[.] 

As shown in Figure 2, akin to Qiu et al. [37], we suppose an LLM _LM_ has grasped the generic fact _ri_ if and only if _LM_ can correctly answer all examples in _Si_ . Hence, we define abstract reasoning accuracy (denoted as `AbsAcc` ) as the proportion of generic facts that _LM_ has grasped: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Meaningful_Learning_Enhancing_Abstract_Reasoning_in_Large_Language_Models_via_Generic_Fact_Guidance_images/nips_2024_Meaningful_Learning_Enhancing_Abstract_Reasoning_in_Large_Language_Models_via_Generic_Fact_Guidance.pdf-0003-06.png)


**----- Start of picture text -----**<br>
Generic Fact 𝑟! Generic Fact 𝑟% Generic Fact 𝑟&<br>𝟏 𝟐 𝟑<br>Example  𝒔𝟏 Example  𝒔𝟏 Example  𝒔𝟏<br>Label: A Label: C Label: D<br>Prediction: A  Prediction: A Prediction: D<br>𝟏 𝟐 𝟑<br>Example  𝒔𝟐 Example  𝒔𝟐 Example  𝒔𝟐<br>Label: B Label: B Label: C<br>Prediction: B Prediction: A Prediction: C<br>𝟏<br>Example  𝒔𝟑<br>Label: A Vanilla : (1+1+0+0+0+1+1) / 7 =  0.57<br>Prediction: B AbsAcc : (0 + 0 + 1) / 3 =  0.33<br>**----- End of picture text -----**<br>


`AbsAcc` = � _i_[I][(][Φ][(] _[LM][,][ S][i]_[)][ =] _[|][S][i][|]_[)] _,_[(1)] Figure 2: Computation of abstract reasoning metric. _n_ 

where Φ quantifies the number of examples in _Si_ that _LM_ answers correctly. I serves as the indicator function, taking 1 when the specified condition is met and 0 otherwise. 

## **2.2 Large language models for evaluation** 

We employ several LLMs in our preliminary experiments: (1) **LLaMA-2** [47] is an open-access LLM. We use 7B and 13B LLaMA-2 for experiments; (2) **Orca-2** [35] is an open-access LLM finetuned on LLaMA-2 with over 80K reasoning-specific examples. We use 7B and 13B Orca-2 for experiments. (3) **GPT-3.5** [7] is a limited access LLM, we use `gpt-3.5-turbo-0125` for experiments. 

## **2.3 Experiment I: abstract reasoning** 

We formulate the first preliminary experiment to gain an initial insight into the abstract reasoning abilities of current LLMs. 

To achieve this goal, we choose a multiple-choice question-answering dataset e-CARE [14] for implementation. e-CARE is a large-scale explainable causal reasoning dataset with a generic fact in each example, and a generic fact can support more than one example. The full set (train, test, and dev) of e-CARE is selected for experiments. We filter out the examples that the corresponding generic facts support only one example. Finally, we obtain 13,045 examples supported by 5,608 generic facts. 

We follow zero-shot setting to prompt the LLMs for answer selection from multiple choices, and the generic facts are not provided in the inputs (refer to Appendix B.1). Table 1 shows the overall results: 

(1) The significant disparity (over 17%) between the vanilla accuracy and `AbsAcc` for all LLMs proves two aspects: on the one hand, vanilla accuracy is not ideal for estimating abstract reasoning. On the other hand, although LLMs could answer questions accurately, they still cannot capture the common patterns or generic facts behind the questions. 

(2) Larger LLMs outperform smaller ones due to their richer knowledge and stronger reasoning ability. Orca-2, additional trained on LLaMA-2, notably outperforms LLaMA-2, indicating that supervised fine-tuning enhances both reasoning and abstract reasoning abilities. However, even with large-scale post-training, the significant gap between vanilla accuracy and `AbsAcc` remains unbridged. 

(3) The minimal gap between vanilla accuracy and `AbsAcc` for humans suggests the large gap for LLMs is not due to varying difficulties of the example supported by the same generic fact. 

3 

|**_Gi F_ Ul diti fft bhi**|**_Gi F_ Ul diti fft bhi**|
|---|---|
|**_Question:_**Which of the following scenarios is most<br>likely to influence a person's typical shopping habits?<br>**_Options:_**<br>A) A regular day with no significant changes in weather<br>or social conditions<br>B) A holiday season with sales and promotions in stores<br>C) When the person has not received their paycheck yet<br>D) A day with consistent weather and social conditions<br>as the previous week<br>**_Answer:_**B)<br>**_Explanation:_**Holiday seasons with promotions and<br>sales present unusual conditions that can significantly<br>alter a person's typical shopping behavior, often<br>encouraging more spending.<br>**_enerc act:_ nusua**|**_Question:_**In a psychological study, which condition is<br>most likely to yield atypical results in participants?<br>**_Options:_**<br>A) A study conducted during a stressful event<br>B) A study conducted in a controlled environment with<br>no distractions<br>C) A study conducted with the same participants as a<br>previous similar study<br>D) A study conducted with used questionnaires<br>**_Answer:_**A)<br>**_Explanation:_**A stressful event like a natural disaster<br>creates an unusual condition that can significantly affect<br>participants' behavior and responses, leading to atypical<br>results in a psychological study.<br>**conons aec eavor**|



Figure 3: Two samples of AbsR guided by the generic fact about “unusual conditions”. 

## **2.4 Experiment II: generic fact probing** 

To delve into the abstract reasoning from the perspective of generic fact mastery, we design a generic fact probing task to ask LLMs whether they possess the given generic fact (refer to Appendix B.2 for more details). Drawing inspiration from knowledge uncertainty estimation [3], we formulate this task as a yes or no problem. The overall results are shown in Table 2. We could conclude as follows: 

(1) LLaMA-2 and GPT-3.5 have substantial knowledge reserves. This indicates the poor `AbsAcc` of LLaMA-2 might come from a lack of ability to apply the generic fact to different reasoning scenarios. This problem also exists in GPT-3.5, but it is not as serious as LLaMA-2. 

(2) In contrast, Orca-2 knows limited generic facts, and the larger the model, the less it knows. This contrasts sharply with its strong reasoning abilities (vanilla accuracy). We suppose training Orca-2 is more like rote learning rather than meaningful learning, which implies that Orca-2 tends to lose knowledge while acquiring reasoning. 

To deeply investigate the relation between generic facts and abstract reasoning, we categorize the examples in e-CARE according to whether LLMs know the generic facts. The results are represented in Table 3, from which we can draw the following conclusions: 

(1) The performance of “Known” category has superiority over “Unknown” category across all LLMs. This demonstrates knowing the generic facts is helpful for abstract reasoning, which provides a potential avenue for abstract reasoning enhancement through generic fact retrieval. 

(2) Nonetheless, most of the results in the “Unknown” category still have undeniable performance, suggesting that LLMs may engage in reasoning based on suspicious correlations or rely on memorization. 

In conclusion, we can enhance the abstract reasoning abilities of LLMs in two ways: (1) _**Knowledge**_ , injecting generic facts through additional training or prompting; (2) _**Reasoning**_ , teaching LLMs how to utilize the generic facts to provide more precise responses to questions. 

## **3 AbsR: abstract reasoning dataset** 

To enhance the abstract reasoning abilities of LLMs through _**Knowledge**_ and _**Reasoning**_ ways, we need a dataset that can not only provide generic facts but also teach LLMs how to exploit the generic fact to derive the correct answer in different scenarios (meaningful learning). Since there is no suitable dataset to assist us in achieving the goals, we create an **Abs** tract **R** easoning dataset (AbsR). 

## **3.1 Generic fact collection** 

To obtain the generic facts, We chose GenericsKB [5] as the foundational generic fact base for AbsR. GenericsKB is a large-scale knowledge base with over 3.4 million sentences (e.g. Dogs bark) 

4 

expressing general truths, each of which also includes the corresponding concept (e.g. Dog) and the confidence score (between 0 and 1). Hence, we filter and then sample a collection of 4,613 high-quality generic facts of different concepts from GenericsKB (details of generic fact filtering and sampling can refer to Appendix C). 

Table 4: The statistics of AbsR. 

||Table 4:The statistics of AbsR|
|---|---|
|**3.2**<br>**Dataset construction**<br>To construct the whole dataset, and inspired<br>by previous works [49, 56], we choose GPT-<br>4 [2] as our data annotator. The API we used is<br>`gpt-4-1106-preview`.|.|
||**Examples Questions Generic Facts**|
||Train<br>18,020<br>9,010<br>4,613<br>Test<br>200<br>200<br>104|
|||



Specifically, for each sampled generic fact _ri_ , we would ask GPT-4 to create samples _Si_ = _{s[i]_ 1 _[,][ · · ·][, s][i] mi[|]_[1] _[≤][m][i][≤]_[3] _[}]_[in][various][scenarios][based][on] _[r][i]_[.][Each][sample] _[s][i] j_[contains][the][fol-] lowing terms: (1) a question _Xj[i]_[with][a][few][options,][(2)][a][response] _[Y] j[i]_[with][an][answer][and][an] explanation guided by _ri_ . All terms form a triple _s[i] j_[=] _[< X] j[i][, r][i][, Y] j[i][>]_[ for each sample.][The prompt] for dataset creation can refer to Appendix D. Figure 3 shows an illustration of the created samples. 

Finally, without loss of generality, given the _j_ -th sample _s[i] j_[of generic fact] _[ r][i]_[, we can create two kinds] of examples for meaningful learning. One is predicting _Yj[i]_[given] _[ < X] j[i][, r][i][>]_[ (K-example), while the] other is predicting _Yj[i]_[given only] _[ X] j[i]_[(R-example).][The examples can implicitly enhance abstract] reasoning in LLMs through _**Knowledge**_ and _**Reasoning**_ ways. Table 4 shows the statistics of AbsR. 

## **3.3 The quality of AbsR** 

We conduct human evaluations to measure the quality of AbsR from the following dimensions (details and more statistics such as evaluation criteria, agreements, and pay can refer to Appendix E and F): 

- **Human Performance** : humans can achieve vanilla accuracy of 95% and `AbsAcc` of 93.27%; 

- **Support Rate** : 89% (the rate of examples which can be supported by the generic fact); 

- **Diversity** : 88.5% (the diversity of the examples supported by the same generic fact. Greater diversity indicates greater differences among the examples of the same generic fact). 

For comparison, e-CARE [14] is also human-annotated, with examples generated from generic facts. On e-CARE, humans reached 92% vanilla accuracy, 89.9% `AbsAcc` , and an 87% support rate [14], showing that AbsR matches the quality of the human-annotated dataset. 

## **4 Method: meaningful learning** 

To imitate humans’ instinctive use of general facts in reasoning, we develop a simple but effective learning paradigm called meaningful learning (MeanLearn). It can enhance the abstract reasoning abilities of LLMs in _**Knowledge**_ and _**Reasoning**_ ways. This is inspired by the implicit knowledge learning process, which uses hidden variables to learn event background knowledge [13]. MeanLearn can make LLMs implicitly learn generic facts and solve problems under the guidance of generic facts. 

Specifically, an LLM _LM_ with parameters _θ_ and the input _x_ can construct a conditional probability for the output _y_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Meaningful_Learning_Enhancing_Abstract_Reasoning_in_Large_Language_Models_via_Generic_Fact_Guidance_images/nips_2024_Meaningful_Learning_Enhancing_Abstract_Reasoning_in_Large_Language_Models_via_Generic_Fact_Guidance.pdf-0005-14.png)


In MeanLearn, for a K-example _< X, r, Y >_ and R-example _< X, Y >_ pair guided by generic fact _r_ , we send the example pair into _LM_ to model two conditional probabilities: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Meaningful_Learning_Enhancing_Abstract_Reasoning_in_Large_Language_Models_via_Generic_Fact_Guidance_images/nips_2024_Meaningful_Learning_Enhancing_Abstract_Reasoning_in_Large_Language_Models_via_Generic_Fact_Guidance.pdf-0005-16.png)


Hereafter, on the one hand, we train _LM_ to learn policies _pθ_ ( _Yt|X, Y<t_ ) and _qθ_ ( _Yt|X, r, Y<t_ ) together to enable _LM_ implicitly learn the generac fact _r_ . On the other hand, _LM_ can learn how 

5 

to apply _r_ into different scenarios by learning the explanations in _Y_ of different _X_ supported by _r_ . Finally, we can reason with the implicit guidance of _r_ when only given _X_ , just like humans instinctively answer questions without explicitly giving the general facts. 

## **5 Experiments** 

## **5.1 Experimental details** 

**Training setup.** We use LoRA [20] for parameter effcient finetuning. 7B and 13B MeanLearn are trained on 7B and 13B Orca-2, respectively, 8B MeanLearn is trained on 8B LLaMA-3, with batch sizes of 256 for 7B and 8B, and 240 for 13B. MeanLearn of various sizes are trained for one epoch at a 5e-5 learning rate with AdamW [26]. 7B and 8B MeanLearn are trained on 2 NVIDIA A100 80GB PCIe GPUs for 3 hours. 13B MearnLearn is trained on 3 such GPUs for 4 hours. 

**Baselines.** We adopt a wide range of open and limited access LLMs across different sizes as baselines. **Open Access LLMs** : (1) LLaMA-2 [47] 7B and 13B; (2) LLaMA-3 [34] 8B; (3) Vicuna [8] 7B and 13B (finetuned on ShareGPT [1] data); (4) WizardLM [56] 7B and 13B (fintuned with evolved instruction data); (5) Orca-2 [35] 7B and 13B (prograssively finetuned on massive reasoning data). **Limited Access LLMs** : (1) GPT-3.5 ( `gpt-3.5-turbo-0125` ); (2) PaLM-2 ( `text-bison-001` ). 

**Evaluation benchmarks.** In addition to AbsR, we also include a wide range of benchmarks of reasoning and natural language understanding for evaluation: (1) AGIEval [60] consists of tests ranging from college admission tests, to national civil service examinations; (2) RACE [28] consists of tests with reading comprehension questions; (3) BBH [43] is a subset of Big-Bench [42], which contains 23 hardest tasks focusing on challenging scenarios; (4) Com. [55] is a collection of 7 commonsense reasoning datasets ( _α_ NLI [4], CSQA [44], COPA [39], e-CARE [14], SocialIQa [40], PIQA [6], and StrategyQA [16]); (5) MMLU [18] is a massive multitask language understanding benchmark; (6) ARC [10] is a benchmark of easy (ARC-e) and challenge (ARC-c) science questions. 

**Evaluation setup.** Since generation-based evaluation is time-consuming due to limited computing resources, we follow OpenCompass [11] to have hybrid evaluation criteria with high reproducibility: (1) For PaLM-2 and GPT-3.5 on all tasks, and open access LLMs on 4 generation tasks (4 tasks in BBH), greedy decoding and exact match are utilized for evaluation; (2) For open access LLMs on classification and multiple-choice tasks, perplexity (PPL) is adopted for prediction. The option or category with the lowest PPL is chosen as the answer. For all baselines, evaluation is conducted once since there is no randomness on such evaluation criteria. For MeanLearn, results are averaged of MeanLearn(s) trained with 3 different random seeds. The evaluation prompts can refer to Appendix H. 

## **5.2 Results: vanilla accuracy** 

Table 5 shows the `Vanilla Accuracy` of MeanLearn and baselines: 

(1) Additional training usually brings improvements (Vicuna, WizardLM, Orca-2, and MeanLearn), but it still can result in performance losses on some benchmarks (e.g. AGIEval). This signifies the effectiveness of massive reasoning-specific data and the necessity of data coverage. 

(2) MeanLearn has superiority over all open access baselines on nearly all benchmarks. This demonstrates that MeanLearn can use generic facts to effectively guide LLMs to reason more properly and logically, making them more flexible and task-adaptive. 

(3) The more challenging the benchmark (e.g. BBH), the less pronounced the improvement in the performance of MeanLearn. This is mainly because the improvement is mostly constrained by the complex scenarios in harder benchmarks, which require mightier base LLMs or more complex training data to perform and learn multi-step and compositional reasoning. 

(4) MeanLearn does not have many advantages over LLaMA-3. LLaMA-3 stores dense knowledge in its parameters, making it harder to learn new knowledge without forgetting existing knowledge. 

(5) Larger open-sourced LLMs do not always yield better results. We suppose the main factors behind this are in two ways: on the one hand, LLMs with different parameter sizes might be good at different tasks, and excessive contemplation may lead to decreases on some tasks [53]. On the other hand, the perplexity-based evaluation setting might have moderate cross-scale generalization ability. 

6 

Table 5: Overall vanilla accuracy and `AbsAcc` (%) of baselines and MeanLearn. Due to the space limit, we only report the standard deviation of _**Average**_ performance of each method. 

|**Size**|**LLMs**|**AbsR**|**AGIEval**|**RACE**|**BBH**|**Com.**|**MMLU**|**ARC-e**|**ARC-c**|**_Average_**|
|---|---|---|---|---|---|---|---|---|---|---|
|`Vanilla Accuracy`|||||||||||
|>20B|PaLM-2<br>GPT-3.5|75.76<br>84.00|15.83<br>25.84|75.82<br>83.36|48.73<br>54.15|78.05<br>74.80|58.73<br>65.98|89.95<br>94.71|82.37<br>88.81|65.77_±_0.00<br>71.46_±_0.00|
||LLaMA-2|50.00|27.89|38.85|26.00|55.37|41.65|58.73|43.05|42.69_±_0.00|
||Vicuna|75.00|32.56|65.75|31.93|64.77|49.40|74.78|57.63|56.48_±_0.00|
|7B|WizardLM|65.00|22.27|22.11|25.99|43.63|23.26|25.57|22.37|31.28_±_0.00|
||Orca-2|73.50|**38.57**|73.32|34.51|71.58|50.11|79.19|73.90|61.84_±_0.00|
||MeanLearn|**77.00**|38.15|**77.15**|**35.64**|**76.59**|**52.98**|**86.67**|**78.06**|**65.28**_±_0.34|
|8B|LLaMA-3<br>MeanLearn|82.50<br>**84.50**|36.33<br>**38.53**|76.75<br>**77.02**|36.60<br>**37.67**|**71.76**<br>71.73|61.86<br>**62.35**|88.54<br>**88.89**|75.25<br>**77.97**|66.20_±_0.00<br>**67.12**_±_0.18|
||LLaMA-2|**73.50**|34.41|59.46|30.61|61.00|**51.87**|71.25|55.25|54.67_±_0.00|
||Vicuna|74.00|33.23|63.54|33.22|63.10|49.34|77.78|59.77|56.75_±_0.00|
|13B|WizardLM|74.00|33.23|63.54|33.22|63.10|49.34|77.78|59.77|56.75_±_0.00|
||Orca-2|66.50|43.75|66.86|39.39|65.92|47.27|82.19|70.85|60.34_±_0.00|
||MeanLearn|67.17|**43.92**|**70.24**|**40.36**|**69.75**|51.68|**88.69**|**82.00**|**64.23**_±_0.25|
|`AbsAcc`|||||||||||
|>20B|PaLM-2<br>GPT-3.5|64.58<br>77.08|9.59<br>14.48|61.54<br>66.79|27.69<br>25.45|69.34<br>65.74|44.39<br>52.09|85.68<br>92.16|77.10<br>85.05|54.99_±_0.00<br>59.86_±_0.00|
||LLaMA-2|35.42|16.09|14.08|9.56|47.42|25.48|50.27|35.51|29.23_±_0.00|
||Vicuna|58.33|19.90|38.68|14.28|58.07|32.50|65.95|49.53|39.84_±_0.00|
|7B|WizardLM|53.24|12.94|5.57|9.77|32.96|11.55|20.00|17.29|20.42_±_0.00|
||Orca-2|60.42|**26.27**|50.90|15.67|64.02|33.68|72.16|67.76|48.86_±_0.00|
||MeanLearn|**64.58**|25.27|**57.10**|**16.67**|**71.17**|**37.24**|**81.35**|**73.83**|**53.39**_±_0.32|
|8B|LLaMA-3<br>MeanLearn|72.92<br>**73.96**|23.78<br>**26.57**|**55.61**<br>55.09|18.13<br>**19.24**|66.35<br>**66.48**|**47.31**<br>46.76|83.51<br>**83.78**|70.09<br>**73.36**|54.71_±_0.00<br>**55.66**_±_0.17|
||LLaMA-2|**61.46**|21.01|32.17|13.74|52.87|**35.72**|62.43|44.86|40.53_±_0.00|
||Vicuna|61.46|19.70|36.50|16.33|55.26|32.21|70.27|52.34|40.37_±_0.00|
|13B|WizardLM|59.38|16.96|28.59|18.61|55.73|29.47|57.84|42.06|38.58_±_0.00|
||Orca-2|50.00|29.50|42.12|20.61|56.41|31.11|75.95|62.15|45.98_±_0.00|
||MeanLearn|45.82|**30.01**|**47.65**|**21.85**|**61.04**|35.21|**85.41**|**77.08**|**50.51**_±_0.23|



(6) LLMs possess more than 20B parameters still struggle with abstract reasoning. MeanLearn can achieve comparable performance to PaLM-2 despite their huge gap in model sizes. 

## **5.3 Results: abstract reasoning accuracy** 

Since the generic facts are not given in the OOD benchmarks, we train a RoBERTa-Large [32] with constrastive learning to cluster examples supported by the same generic fact. In particular, we utilize e-CARE as the training set, examples guided by the same generic fact will be pushed closer, or they will be pushed away. Cosine similarity is used to measure the distance between examples (training details can refer to Appendix I). For clustering, examples sharing a similarity above 0.6 are considered to be supported by the same generic fact. Each cluster contains no more than 3 examples. 

Hereafter, we calculate the abstract reasoning metrics based on the results of clustering. The overall results are shown in Table 5 ( `AbsAcc` ), we can have the following observations: 

(1) `AbsAcc` is much lower than vanilla accuracy, indicating the large gap between general reasoning and abstract reasoning. More efforts should be made to fill this gap in the future. 

(2) MeanLearn can surpass all baselines, usually exhibiting greater advantages in `AbsAcc` than in vanilla accuracy. This suggests that the improvement in vanilla accuracy is more likely due to the enhancement of abstract reasoning abilities. MeanLearn can tell LLMs the generic facts and teach them how to implicitly utilize them in different scenarios, resulting in better abstract reasoning. 

7 

Table 6: The overall reasults of ablation studies. “w/o” denotes without. 

|**Size**<br>**Method**|**Size**<br>**Method**|**AbsR**|**AGIEval**|**RACE**|**BBH**|**Com.**|**MMLU**|**ARC-e**|**ARC-c**|**_Average_**|
|---|---|---|---|---|---|---|---|---|---|---|
|`Vanilla Accuracy`|||||||||||
|MeanLearn||**77.00**|38.15|**77.15**|**35.64**|**76.59**|**52.98**|**86.67**|**78.06**|**65.28**|
|7B<br>w/o Knowledge<br>w/o Reasoning||73.50<br>72.50|38.89<br>38.37|74.10<br>72.60|33.80<br>33.39|73.28<br>71.68|50.97<br>49.55|79.72<br>77.25|75.25<br>73.22|62.44<br>61.07|
|w/ AbsR_∗_||73.50|**38.53**|73.14|33.65|73.57|50.96|80.60|75.25|62.40|
|MeanLearn||**67.17**|43.92|**70.24**|**40.36**|69.75|**51.68**|**88.69**|**82.00**|**64.23**|
|13B<br>w/o Knowledge<br>w/o Reasoning||63.50<br>56.00|**44.07**<br>43.95|67.19<br>58.05|36.80<br>35.91|**70.45**<br>68.09|47.88<br>45.31|84.13<br>82.54|74.58<br>69.83|61.08<br>57.46|
|w/ AbsR_∗_||66.50|43.75|66.86|36.92|65.92|47.27|82.19|70.85|60.03|
|`AbsAcc`|||||||||||
|MeanLearn||**64.58**|25.27|**57.10**|**16.67**|**71.17**|**37.24**|**81.35**|**73.83**|**53.39**|
|7B<br>w/o Knowledge<br>w/o Reasoning||61.46<br>59.38|**26.67**<br>26.18|52.91<br>49.66|14.74<br>14.15|66.26<br>64.13|34.84<br>33.27|72.70<br>69.19|69.16<br>67.29|49.84<br>47.91|
|w/ AbsR_∗_||61.46|26.44|52.24|14.76|55.56|34.88|74.05|69.16|48.57|
|MeanLearn||45.82|30.01|**47.65**|**21.85**|61.04|35.21|**85.41**|**77.08**|**50.51**|
|13B<br>w/o Knowledge<br>w/o Reasoning||46.88<br>36.46|30.43<br>**30.81**|42.87<br>31.19|18.83<br>18.26|**62.95**<br>59.70|32.10<br>29.12|78.11<br>75.68|66.82<br>61.68|47.37<br>42.86|
|w/ AbsR_∗_||**50.00**|29.58|42.12|18.82|56.41|31.11|75.95|62.15|45.77|



## **6 Further analysis** 

To further investigate the pros, cons, and effectiveness of MeanLearn, we conduct several analyses and ablation studies. LLaMA-2, Orca2, and MeanLearn are adopted for analysis. 

## **6.1 Performance on different domains** 

We categorize tasks of MMLU based on their respective domains to demonstrate the pros and cons of MeanLearn for future improvements. The results are shown in Figure 4 with a total of 16 categories: 

(1) All LLMs possess poor vanilla accuracy and `AbsAcc` in math and chemistry. We conjecture the primary factors are the complexity of math problems and the shortage of chemistry knowledge. 

(2) Compared to Orca-2, MeanLearn possesses considerable advantages in engineering, geography, and physics & astronomy domains. We suppose this is related to the domain coverage of AbsR. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Meaningful_Learning_Enhancing_Abstract_Reasoning_in_Large_Language_Models_via_Generic_Fact_Guidance_images/nips_2024_Meaningful_Learning_Enhancing_Abstract_Reasoning_in_Large_Language_Models_via_Generic_Fact_Guidance.pdf-0008-08.png)


**----- Start of picture text -----**<br>
LLaMA-2 [Physics & Astronomy] [Law] [Medical]<br>[Finance & Management] [Society] [Politics]<br>Orca-2 [Logic & Philosophy] [Mathematics] [Biology]<br>[Computer Science] [Engineering] [History]<br>MeanLearn [Geography] [Psychology] [Chemistry] [Others]<br>Vanilla-Acc<br>7B 13B<br>AbsAcc<br>**----- End of picture text -----**<br>


Figure 4: Visualization of reasoning capabilities on MMLU, which is categorized by task domians. 

(3) Since the overall `AbsAcc` is not relatively high, for potential improvement in the future, apart from focusing more on the domains with poor vanilla accuracy and `AbsAcc` , we should also increase the overall volume of finetuning data. 

## **6.2 How much do knowledge and reasoning ways contribute to MeanLearn?** 

We utilize _**Knowledge**_ and _**Reasoning**_ ways to enhance abstract reasoning in LLMs. To investigate the effect of them, we conduct an ablation study of AbsR to respectively remove the _**Knowledge**_ (generic facts) and _**Reasoning**_ (explanations) in each example. Results are shown in Table 6 (w/o Knowledge and w/o Reasoning), we can observe: 

8 

Table 8: Performance on mathematical reasonin datasets from MMLU [18]. 

Table 7: MeanLearn trained based on Mistral. 

|ae : eanearn trane ase on stra.<br>**Method**<br>**AbsR Com. MMLU RACE** **_Average_**<br>`Vanilla Accuracy`<br>Mistral<br>83.00 66.53<br>57.02<br>75.33<br>70.47<br>MeanLearn (Ours) **84.50 74.56**<br>**58.28**<br>**76.65**<br>**73.50**<br>`AbsAcc`<br>Mistral<br>72.92 58.87<br>40.19<br>52.69<br>56.17<br>MeanLearn (Ours) **73.96 68.59**<br>**41.14**<br>**55.77**<br>**59.87**|**Size **|**Method**<br>**AA**<br>**CM**<br>**EM**<br>**HSS**<br>**HSM** **_Average_**|
|---|---|---|
||`Van`<br>7B|`illa Accuracy`<br>Orca-2<br>23.00 32.00 33.07 37.95 28.15<br>30.83<br>MeanLearn 36.00 35.00 33.07 40.28 28.89<br>**34.56**|
|||Mistral<br>25.00 30.00 38.89 46.76 32.96<br>34.72<br>MeanLearn 31.00 34.00 37.83 47.69 34.07<br>**36.92**|
||`Abs`<br>7B|`Acc`<br>Orca-2<br>0.00<br>4.88<br>25.00 25.17 14.63<br>13.94<br>MeanLearn<br>2.86<br>7.32<br>24.02 27.97 16.26<br>**15.69**|
|||Mistral<br>2.86<br>7.32<br>25.49 36.36 14.63<br>17.33<br>MeanLearn<br>5.71<br>9.76<br>24.55 37.76 16.26<br>**18.81**|



(1) Removing _**Knowledge**_ (generic facts) or _**Reasoning**_ (explanation) leads to a decrease in performance. This is because the generic facts and their guided reasoning process in AbsR can teach LLMs to learn and utilize the generic fact behind the questions to reason more logically and properly. 

(2) In MeanLearn, _**Knowledge**_ contributes less in boosting the vanilla and abstract reasoning than _**Reasoning**_ . It indicates the importance of teaching LLMs how to reason under the guidance of generic facts. Furthermore, it reveals the bottleneck of small-scale LLMs reasoning is caused more by reasoning itself rather than knowledge. 

(3) MeanLearn without Knowledge can defeat Orca-2 on vanilla accuracy and `AbsAcc` , while MeanLearn without Reasoning cannot. This implies the synergy effects between _**Knowledge**_ and _**Reasoning**_ (MeanLearn) can push LLMs further. 

(4) In AGIEval, _**Knowledge**_ and _**Reasoning**_ are less critical due to their focus on complex reading comprehension without needing vast knowledge. Moreover, _**Knowledge**_ seems useless in Commonsense. When providing LLMs with knowledge, we should also teach them how to use it. 

## **6.3 Is the improvement in performance solely attributed to additional data?** 

Supervised fine-tuning can also improve `AbsAcc` (Table 1), so we design another ablation study to investigate whether the improvement of MeanLearn only comes from additional data. 

To be precise, for each sample in AbsR, we use GPT-4 to generate a new explanation based on the question and label without the guidance of the generic fact. Hereafter, we obtain AbsR _[∗]_ . The only difference between AbsR and AbsR _[∗]_ is the explanations in AbsR are guided by the generic fact. Table 6 (w/ AbsR _[∗]_ ) shows the results of MeanLearn trained with AbsR _[∗]_ , we can infer: 

(1) Training 7B and 13B Orca-2 with AbsR _[∗]_ can still outperform Orca-2 on vanilla accuracy. However, it cannot bring stable improvements on `AbsAcc` . This underscores that reasoning-specific post-training can boost performance, while improving abstract reasoning performance might depend on more careful design such as MeanLearn. 

(2) MeanLearn without Knowledge outperforms MeanLearn trained with AbsR _[∗]_ , showing the superiority of training with generic-fact-guided explanation. This stems from the generic-fact-guided explanation that can teach LLMs to use the regular pattern behind the questions for better reasoning. 

## **6.4 Expanding MeanLearn to LLMs besides the LLaMA series** 

To investigate the universality of MeanLearn, we train MeanLearn with Mistral-7B-Instruct-v0.2 [21] (denoted as Mistral). All settings are consistent with Sec 5. We choose AbsR, Com., MMLU, and RACE for evaluation. As shown in Table 7, we can find MeanLearn has good applicability on Mistral. 

## **6.5 Expanding reasoning domains to mathematical reasoning** 

Although we do not incorporate math examples in AbsR, we want to investigate whether MeanLearn can improve the performance of mathematical reasoning by enhancing abstract reasoning in LLMs. 

9 

Following [33], we select five mathematical reasoning datasets from MMLU [18] to demonstrate this: abstract algebra (AA), college mathematics (CM), elementary mathematics (EM), high school statistics (HSS), and high school mathematics (HSM). We choose 7B version of Orca-2 and Mistral for comparison. Results are shown in Table 8, it is interesting to find that MeanLearn can outperform baselines even if there is no math data in AbsR. This demonstrates the effectiveness of MeanLearn in enhancing abstract reasoning in LLMs. 

## **7 Related work** 

## **7.1 Reasoning with LLMs** 

LLMs have overturned the research paradigm of NLP, which makes reasoning more accessible. Using LLMs to perform reasoning can be categorized into tunning-based and prompt-based methods. 

Tunning-based methods, originating from T0 [48] and FLAN [52], finetuned T5 [38] and LaMDAPT [46] on massive NLP tasks, achieving notable zero-shot performance on unseen tasks. FlanPaLM [9] and T _k_ -INSTRUCT [50] further scaled up the model and task size for improvement. Orca [36] and Orca-2 [35] leveraged vast instruction data to teach small-scale LLMs reasoning skills. 

As for prompt-based methods, Wei et al. [53] introduced CoT prompt to elicit reasoning in LLMs by encouraging step-by-step thinking. Additionally, Kojima et al. [27] presented zero-shot CoT to bypass manual annotation issues. Subsequently, Zhang et al. [59] proposed AutoCoT to automatically generate few-shot examples with CoT for reasoning. Various CoT-based methods [22, 51] have since emerged, such as complex CoT [15], and the tree of thought [57], etc. 

These methods are designed for improving the general reasoning abilities of LLMs, while we focus on the abstract reasoning capabilities of LLMs. 

## **7.2 LLMs-as-annotators** 

LLMs are increasingly used as annotators to create data for training and evaluation due to their advanced instruction-following skills. 

Some studies distilled various CoT data from LLMs to train student models ([30, 19, 58], _inter alia_ ), achieving impressive performance. For instance, Li et al. [30] distilled symbolic CoT from GPT-3 and observed enhanced performance in commonsense reasoning. Dai et al. [12] used ChatGPT to generate augmentation data to enhance BERT [24]. With the advent of open-source LLMs like LLaMA [47], other studies distilled instruction data from large-scale LLMs to train small-scale models ([56, 36, 25, 31, 23, 17], _inter alia_ ). For example, Xu et al. [56] and Luo et al. [33] employed an evolving strategy to obtain complex data, trained LLaMA, and achieved strong performance. Tang et al. [45] constructed a concept graph and used GPT-3.5 to synthesize math data with high diversity. We employ LLMs to generate training data and develop AbsR tailored for abstract reasoning. 

## **8 Conclusion** 

In this paper, we investigate the abstract reasoning of existing LLMs. To achieve this, we first define the evaluation metric of abstract reasoning and design a series of preliminary experiments. Then we discover a significant performance gap between general reasoning and abstract reasoning. Hence, we tailor an abstract reasoning dataset AbsR with the help of GPT-4 to enhance LLMs. Finally, we devise a simple but effective paradigm (MeanLearn) to teach LLMs abstract reasoning in _**Knowledge**_ and _**Reasoning**_ ways. Extensive experiments demonstrate the superiority and effectiveness of MeanLearn in vanilla and abstract reasoning accuracies. The limitations of MeanLearn are discussed in Appendix A. 

## **Acknowledgements** 

We would like to acknowledge the editors and reviewers for their efforts and advice, and gratefully acknowledge the support of the National Natural Science Foundation of China under Grants U22B2059 and 62176079, Natural Science Foundation of Heilongjiang Province under Grant YQ2022F005. 

10 

## **References** 

[1] Sharegpt: Share your wildest chatgpt conversations with one click, 2023. 

- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- [3] Alfonso Amayuelas, Liangming Pan, Wenhu Chen, and William Wang. Knowledge of knowledge: Exploring known-unknowns uncertainty with large language models. _arXiv preprint arXiv:2305.13712_ , 2023. 

- [4] Chandra Bhagavatula, Ronan Le Bras, Chaitanya Malaviya, Keisuke Sakaguchi, Ari Holtzman, Hannah Rashkin, Doug Downey, Wen-tau Yih, and Yejin Choi. Abductive commonsense reasoning. In _International Conference on Learning Representations_ , 2019. 

- [5] Sumithra Bhakthavatsalam, Chloe Anastasiades, and Peter Clark. Genericskb: A knowledge base of generic statements. _arXiv preprint arXiv:2005.00660_ , 2020. 

- [6] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 34, pages 7432–7439, 2020. 

- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- [8] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. 

- [9] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. _arXiv preprint arXiv:2210.11416_ , 2022. 

- [10] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. _arXiv preprint arXiv:1803.05457_ , 2018. 

- [11] OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models, 2023. 

- [12] Haixing Dai, Zhengliang Liu, Wenxiong Liao, Xiaoke Huang, Yihan Cao, Zihao Wu, Lin Zhao, Shaochen Xu, Wei Liu, Ninghao Liu, et al. Auggpt: Leveraging chatgpt for text data augmentation. _arXiv preprint arXiv:2302.13007_ , 2023. 

- [13] Li Du, Xiao Ding, Ting Liu, and Zhongyang Li. Modeling event background for if-then commonsense reasoning using context-aware variational autoencoder. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 2682–2691, 2019. 

- [14] Li Du, Xiao Ding, Kai Xiong, Ting Liu, and Bing Qin. e-care: a new dataset for exploring explainable causal reasoning. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 432–446, 2022. 

- [15] Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. Complexity-based prompting for multi-step reasoning. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- [16] Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. _Transactions of the Association for Computational Linguistics_ , 9:346–361, 2021. 

11 

- [17] Sonam Gupta, Yatin Nandwani, Asaf Yehudai, Mayank Mishra, Gaurav Pandey, Dinesh Raghu, and Sachindra Joshi. Selective self-rehearsal: A fine-tuning approach to improve generalization in large language models. _arXiv preprint arXiv:2409.04787_ , 2024. 

- [18] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In _International Conference on Learning Representations_ , 2021. 

- [19] Namgyu Ho, Laura Schmid, and Se-Young Yun. Large language models are reasoning teachers. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 14852–14882, Toronto, Canada, July 2023. Association for Computational Linguistics. 

- [20] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In _International Conference on Learning Representations_ , 2022. 

- [21] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. _arXiv preprint arXiv:2310.06825_ , 2023. 

- [22] Ziqi Jin and Wei Lu. Self-harmonized chain of thought. _arXiv preprint arXiv:2409.04057_ , 2024. 

- [23] Hoang Anh Just, Mahavir Dabas, Lifu Huang, Ming Jin, and Ruoxi Jia. Dipt: Enhancing llm reasoning through diversified perspective-taking. _arXiv preprint arXiv:2409.06241_ , 2024. 

- [24] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of naacL-HLT_ , volume 1, page 2. Minneapolis, Minnesota, 2019. 

- [25] Bumjun Kim, Kunha Lee, Juyeon Kim, and Sangam Lee. Small language models are equation reasoners. _arXiv preprint arXiv:2409.12393_ , 2024. 

- [26] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 

- [27] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. _Advances in neural information processing systems_ , 35:22199–22213, 2022. 

- [28] Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. Race: Large-scale reading comprehension dataset from examinations. In _Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing_ , pages 785–794, 2017. 

- [29] Brenden M Lake, Tomer D Ullman, Joshua B Tenenbaum, and Samuel J Gershman. Building machines that learn and think like people. _Behavioral and brain sciences_ , 40:e253, 2017. 

- [30] Liunian Harold Li, Jack Hessel, Youngjae Yu, Xiang Ren, Kai-Wei Chang, and Yejin Choi. Symbolic chain-of-thought distillation: Small models can also “think” step-by-step. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 2665–2679, Toronto, Canada, July 2023. Association for Computational Linguistics. 

- [31] Huanxuan Liao, Shizhu He, Yao Xu, Yuanzhe Zhang, Kang Liu, and Jun Zhao. Neural-symbolic collaborative distillation: Advancing small language models for complex reasoning tasks. _arXiv preprint arXiv:2409.13203_ , 2024. 

- [32] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692_ , 2019. 

12 

- [33] Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. _arXiv preprint arXiv:2308.09583_ , 2023. 

- [34] Meta AI. Introducing meta llama 3: The most capable openly available llm to date, 2024. 

- [35] Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, et al. Orca 2: Teaching small language models how to reason. _arXiv preprint arXiv:2311.11045_ , 2023. 

- [36] Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. Orca: Progressive learning from complex explanation traces of gpt-4. _arXiv preprint arXiv:2306.02707_ , 2023. 

- [37] Linlu Qiu, Liwei Jiang, Ximing Lu, Melanie Sclar, Valentina Pyatkin, Chandra Bhagavatula, Bailin Wang, Yoon Kim, Yejin Choi, Nouha Dziri, et al. Phenomenal yet puzzling: Testing inductive reasoning capabilities of language models with hypothesis refinement. _arXiv preprint arXiv:2310.08559_ , 2023. 

- [38] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _The Journal of Machine Learning Research_ , 21(1):5485–5551, 2020. 

- [39] Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In _2011 AAAI Spring Symposium Series_ , 2011. 

- [40] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social iqa: Commonsense reasoning about social interactions. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 4463–4473, 2019. 

- [41] Jihao Shi, Xiao Ding, Li Du, Ting Liu, and Bing Qin. Neural natural logic inference for interpretable question answering. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 3673–3684, 2021. 

- [42] Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _Transactions on Machine Learning Research_ , 2023. 

- [43] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei. Challenging BIG-bench tasks and whether chain-of-thought can solve them. In Anna Rogers, Jordan BoydGraber, and Naoaki Okazaki, editors, _Findings of the Association for Computational Linguistics: ACL 2023_ , pages 13003–13051, Toronto, Canada, July 2023. Association for Computational Linguistics. 

- [44] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 4149–4158, 2019. 

- [45] Zhengyang Tang, Xingxing Zhang, Benyou Wang, and Furu Wei. Mathscale: Scaling instruction tuning for mathematical reasoning. In _Forty-first International Conference on Machine Learning_ , 2024. 

- [46] Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, HengTze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. Lamda: Language models for dialog applications. _arXiv preprint arXiv:2201.08239_ , 2022. 

13 

- [47] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023. 

- [48] Sanh Victor, Webson Albert, Raffel Colin, Bach Stephen, Sutawika Lintang, Alyafeai Zaid, Chaffin Antoine, Stiegler Arnaud, Raja Arun, Dey Manan, et al. Multitask prompted training enables zero-shot task generalization. In _International Conference on Learning Representations_ , 2022. 

- [49] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language model with self generated instructions. _arXiv preprint arXiv:2212.10560_ , 2022. 

- [50] Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, et al. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 5085–5109, 2022. 

- [51] Yu Wang, Shiwan Zhao, Zhihu Wang, Heyuan Huang, Ming Fan, Yubo Zhang, Zhixing Wang, Haijun Wang, and Ting Liu. Strategic chain-of-thought: Guiding accurate reasoning in llms through strategy elicitation. _arXiv preprint arXiv:2409.03271_ , 2024. 

- [52] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In _International Conference on Learning Representations_ , 2022. 

- [53] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35:24824–24837, 2022. 

- [54] Kai Xiong, Xiao Ding, Zhongyang Li, Li Du, Ting Liu, Bing Qin, Yi Zheng, and Baoxing Huai. Reco: Reliable causal chain reasoning via structural causal recurrent neural networks. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 6426–6438, 2022. 

- [55] Kai Xiong, Xiao Ding, Yixin Cao, Ting Liu, and Bing Qin. Examining inter-consistency of large language models collaboration: An in-depth analysis via debate. In _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pages 7572–7590, 2023. 

- [56] Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. _arXiv preprint arXiv:2304.12244_ , 2023. 

- [57] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. _arXiv preprint arXiv:2305.10601_ , 2023. 

- [58] Jiahao Ying, Mingbao Lin, Yixin Cao, Wei Tang, Bo Wang, Qianru Sun, Xuanjing Huang, and Shuicheng Yan. Llms-as-instructors: Learning from errors toward automating model improvement. _Findings of EMNLP_ , 2024. 

- [59] Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- [60] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. _arXiv preprint arXiv:2304.06364_ , 2023. 

14 

## **A Limitations** 

There are several limitations of our work: First, the size of the training data should be larger to obtain better and more stable results. Second, there should be some examples (whether MeanLearn can generate meaningful explanations) to demonstrate the superiority of MeanLearn. Third, more efforts should be made to focus more on abstract reasoning itself. Finally, complex scenarios (with complex generic facts) such as causal chain reasoning [54] and logical reasoning [41] is a potential direction of abstract reasoning. 

## **B Asbstract reasoning study** 

## **B.1 Prompt for experiment I** 

## **B.1.1 Orca-2** 

```
<|im_start|>system
YouareOrca,anAIlanguagemodelcreatedbyMicrosoft.Youare
acautiousassistant.Youcarefullyfollowinstructions.Youare
helpfulandharmlessandyoufollowethicalguidelinesandpromote
positivebehavior.<|im_end|>
<|im_start|>user
{premise}Hypothesis1orHypothesis2?
Hypothesis1:{hypothesis1}
Hypothesis2:{hypothesis2}
Youranswershouldfollowtheformatlike“Answer:Hypothesis(1or2)
ismoreplausible.
Explanation:___”<|im_end|>
<|im_start|>assistant
```

## **B.1.2 LLaMA-2 and GPT-3.5** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Meaningful_Learning_Enhancing_Abstract_Reasoning_in_Large_Language_Models_via_Generic_Fact_Guidance_images/nips_2024_Meaningful_Learning_Enhancing_Abstract_Reasoning_in_Large_Language_Models_via_Generic_Fact_Guidance.pdf-0015-07.png)


**----- Start of picture text -----**<br>
System: You are a helpful assistant.<br>User: {premise} Hypothesis1 or Hypothesis2?<br>Hypothesis1: {hypothesis1}<br>Hypothesis2: {hypothesis2}<br>Your answer should follow the format like “Answer: Hypothesis(1 or 2)<br>is more plausible.<br>Explanation: ___”<br>**----- End of picture text -----**<br>


The placeholders `{premise}` , `{hypothesis1}` , and `{hypothesis2}` will be filled with the corresponding terms in each example of e-CARE dataset. 

15 

## **B.2 Prompt for experiment II: Orca-2** 

## **B.2.1 Orca-2** 

```
<|im_start|>system
YouareOrca,anAIlanguagemodelcreatedbyMicrosoft.Youare
acautiousassistant.Youcarefullyfollowinstructions.Youare
helpfulandharmlessandyoufollowethicalguidelinesandpromote
positivebehavior.<|im_end|>
<|im_start|>user
Youaregivenafact,doyouknowthisfact?JustanswerYesorNo,
donotgiveanyadditionalinformation.
Fact:{fact}<|im_end|>
<|im_start|>assistant
```

## **B.2.2 LLaMA-2 and GPT-3.5** 

```
System:Youareahelpfulassistant.
User:Youaregivenafact,doyouknowthisfact?JustanswerYes
orNo,donotgiveanyadditionalinformation.
Fact:{fact}
```

The placeholders `{premise}` , `{hypothesis1}` , `{hypothesis1}` , and `{fact}` will be filled with the corresponding terms in each example of e-CARE dataset. 

## **C Details for generic facts filtering** 

We apply the following steps to filter generics facts from GenericsKB: 

- (1) Exclude sentences with confidence _<_ 0.7. 

- (2) Categorize the sentences by the concepts. 

(3) Randomly sample 4,613 concepts, and then randomly sample one sentence as the generic fact from the category of each sampled concept. 

Hereafter, we can obtain a collection of 4,613 generic facts. 

## **D Prompt for AbsR construction** 

We use the following prompt to generate our AbsR examples. 

You are an expert in creating questions, you should first offer a question together with some options based on the fact the user gives. Second, you should give an answer and an explanation guided by the given fact. You can propose questions in any area, including but not limited to history, law, medicine, math, science, computer science, psychology, AI, politics, economics, etc. Your response should follow the following format: “Question: ____ Options:_____ Answer: _____ Explanation: _____”. NOTE that the fact should be an implicit explanation for obtaining the true answer, which means the fact SHOULD NOT appear explicitly in the questions or the options. The explanations should be short. Please create `{number}` examples. 

The placeholder `{number}` will be randomly filled with “one”, “two” or “three”. 

16 

## **E Details of human evaluation** 

We choose three annotators with good backgrounds in textual inference and event reasoning. The whole test set of AbsR is involved in the human evaluation (200 instances). For each annotator, we pay $10 per hour, while in our country, the minimum wage is less than $5 per hour. All annotators agreed to let us use their annotations. The evaluation agreements are 95%, 91%, and 90% for the question answering, generic fact supportance, and diversity tasks, respectively. 

## **F Instructions of human evaluation** 

## **F.1 Performance** 

You are given some questions each with some options, all the questions are about commonsense and are generated by GPT-4. Please choose the most plausible option for each question. Just type your choice (such as (A)) in the Answer column. 

## **F.2 Support rate** 

You are given some questions each with some options and a generic fact, all the questions are about commonsense and are generated by GPT-4. Please judge whether the generic can support to answer the question. Just type your choice (Yes or No) in the Answer column. 

## **F.3 Diversity** 

You are given some generic facts, each generic fact possesses several samples (each sample consists of a question and a few options). The samples can be supported by the corresponding generic fact. Please judge whether the samples within a generic are similar and are just different in nouns or expressions. If sample 1 and sample 2 are similar and are just different in nouns or expressions, just type a tuple (1, 2) in the Answer column. If there are multiple similar sample pairs, just separate them with a “ _\_ n”. For example, (1, 2) _\_ n(3, 4). 

## **G K-example and R-example** 

## **G.1 K-example** 

```
<|im_start|>system
Youareacautiousassistant.Youcarefullyfollowinstructions.
Youaregivenaquestion,afewoptions,andarule,youshould
followthegivenruletoanswerthequestion.Yourresponseshould
followtheformatlike“Explanation:___Answer:____”<|im_end|>
<|im_start|>user
Fact:Cookiefilesaresimpletextfilesthatcanbeviewedin
WindowsNotepad.
Question:Whattypeofapplicationcanbeusedtoopenandview
cookiefilesonaWindowscomputer?
Options:
A)MicrosoftExcelB)AdobePhotoshopC)WindowsNotepadD)3D
ModelingSoftware<|im_end|>
<|im_start|>assistant
Explanation:WindowsNotepadisatexteditorthatiscapableof
openinganddisplayingthecontentsofsimpletextfiles,whichis
theformatofcookiefiles.
Answer:C)WindowsNotepad<|im_end|>
```

17 

## **G.2 R-example** 

```
<|im_start|>system
Youareacautiousassistant.Youcarefullyfollowinstructions.
Youaregivenaquestiontogetherwithafewoptions,youshould
giveanexplanationfirstandthenanswerthequestion.Your
responseshouldfollowtheformatlike“Explanation:___Answer:
____”<|im_end|>
<|im_start|>user
Question:Whattypeofapplicationcanbeusedtoopenandview
cookiefilesonaWindowscomputer?
Options:
A)MicrosoftExcelB)AdobePhotoshopC)WindowsNotepadD)3D
ModelingSoftware<|im_end|>
<|im_start|>assistant
Explanation:WindowsNotepadisatexteditorthatiscapableof
openinganddisplayingthecontentsofsimpletextfiles,whichis
theformatofcookiefiles.
Answer:C)WindowsNotepad<|im_end|>
```

## **H Templates for evaluation** 

## **H.1 LLaMA-2, LLaMA-3, and MeanLearn (8B)** 

```
Thefollowingaremultiple-choicequestions(withanswers)about
abstractalgebra.
```

```
FindthedegreeforthegivenfieldextensionQ(sqrt(2),sqrt(3),
sqrt(18))overQ.
```

```
A.0B.4C.2D.6
Answer:A
```

## **H.2 Orca-2 and MeanLearn (7B and 13B)** 

```
<|im_start|>system
```

```
Thefollowingaremultiple-choicequestions(withanswers)about
abstractalgebra.<|im_end|>
```

```
<|im_start|>user
```

```
Question:FindthedegreeforthegivenfieldextensionQ(sqrt(2),
sqrt(3),sqrt(18))overQ.
Options:
A.0B.4C.2D.6<|im_end|>
<|im_start|>assistant
Answer:A<|im_end|>
```

## **I Training details of RoBERTa-Large-based clusterer** 

We use the RoBERTa-Large [32] released by Meta. The batch size is 256. We use AdamW [26] to optimize the model with a learning rate of 1e-5 for 5 epochs. The computing device is one NVIDIA A100-SXM-64GB. The running time is about 12 minutes. 

18 

