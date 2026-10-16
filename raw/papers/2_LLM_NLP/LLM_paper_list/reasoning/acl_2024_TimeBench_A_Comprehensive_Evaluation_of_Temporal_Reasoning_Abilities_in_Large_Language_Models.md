# **TIMEBENCH: A Comprehensive Evaluation of Temporal Reasoning Abilities in Large Language Models** 

**Zheng Chu**[1] **, Jingchang Chen**[1] **, Qianglong Chen**[2] **, Weijiang Yu**[3] **, Haotian Wang**[1] **, Ming Liu**[1] _[,]_[4][*] **, Bing Qin**[1] _[,]_[4] 

1Harbin Institute of Technology, Harbin, China 

2Zhejiang University 3Sun Yat-sen University 4Peng Cheng Laboratory 

{zchu,jcchen,mliu,qinb}@ir.hit.edu.cn {chenqianglong.ai, wanght1998, weijiangyu8}@gmail.com 

## **Abstract** 

Grasping the concept of time is a fundamental facet of human cognition, indispensable for truly comprehending the intricacies of the world. Previous studies typically focus on specific aspects of time, lacking a comprehensive temporal reasoning benchmark. To address this, we propose TIMEBENCH, a comprehensive hierarchical temporal reasoning benchmark that covers a broad spectrum of temporal reasoning phenomena. TIMEBENCH provides a thorough evaluation for investigating the temporal reasoning capabilities of large language models. We conduct extensive experiments on GPT-4, LLaMA2, and other popular LLMs under various settings. Our experimental results indicate a significant performance gap between the state-of-the-art LLMs and humans, highlighting that there is still a considerable distance to cover in temporal reasoning. Besides, LLMs exhibit capability discrepancies across different reasoning categories. Furthermore, we thoroughly analyze the impact of multiple aspects on temporal reasoning and emphasize the associated challenges. We aspire for TIMEBENCH to serve as a comprehensive benchmark, fostering research in temporal reasoning[1] . 

## **1 Introduction** 

**Time flies over us, but leaves its shadow behind.** Understanding time is a crucial part of human comprehension of the world. Envision the blossoming of flowers, and you’ll associate it with the arrival of spring. The ponder within it encompasses the intricate interplay of world knowledge, causality, and event temporal relationships. Temporal reasoning, in contrast to reasoning of a singular nature, comes with inherent complexity, encompassing implicit arithmetic, logical implications, and world knowledge. It is a form of integrated reasoning built upon 

> * Corresponding Author. 

> 1 Data is available at: GitHub 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models_images/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models.pdf-0001-12.png)


**----- Start of picture text -----**<br>
TimeXNLI<br>TRACIE 95.3 Arith<br>100 100<br>82.5 80<br>TempReason 60 DurationQA<br>96.2 40 80.8<br>20<br>84.3 87.1<br>MenatQA McTACO Human<br>GPT-4<br>92.2 97.8 LLaMA2 [†] 70 b<br>TimeQA 100 TimeDial LLaMA2 [†] 13 b<br>Mistral [†] 7 b<br>SitGen<br>**----- End of picture text -----**<br>


Figure 1: A brief overview of human and LLMs’ performance on TimeBench. Human scores are annotated. 

foundational reasoning like mathematical and logical reasoning (Cobbe et al., 2021; Mishra et al., 2022; Yu et al., 2020). Recently, large language models (LLMs) have demonstrated remarkable performance in complex reasoning (Hendrycks et al., 2021; Srivastava et al., 2022; Brown et al., 2020; Chowdhery et al., 2023; OpenAI, 2023; Touvron et al., 2023), but their performance in temporal reasoning has not yet been extensively explored. 

Recent research for temporal reasoning typically focuses only on a few aspects, such as temporal commonsense or temporal question answering (Zhou et al., 2019; Chen et al., 2021; Dhingra et al., 2022; Wang and Zhao, 2023). Due to the inherent complexity of temporal reasoning, it is challenging to accurately measure models’ temporal reasoning capabilities based on limited aspects. 

To address this issue, we propose TIMEBENCH, a comprehensive and hierarchical temporal reasoning benchmark. Specifically, drawing inspiration from the human cognitive process of transitioning from abstraction and concreteness to integration (Barsalou et al., 2018), we categorize temporal reasoning into three levels: symbolic temporal reasoning, commonsense temporal reasoning, and event temporal reasoning. These levels respectively represent understanding abstract time expression, 

1 

grasping concrete world knowledge, and integrating and applying this knowledge in real-world scenarios. TIMEBENCH comprises 10 tasks with 16 sub-tasks, covering a broad spectrum of temporal reasoning phenomena. Besides, prior work typically features only a single task form, too simplistic to capture the model’s performance. In contrast, we incorporate four distinct task forms, offering a more realistic simulation of challenges. 

To quantify the temporal reasoning capabilities of contemporary LLMs, we extensively assess widely-used LLMs, including proprietary models such as ChatGPT (Ouyang et al., 2022) and GPT4 (OpenAI, 2023), as well as open-source like LLaMA2 (Touvron et al., 2023), Vicuna-1.5 (Chiang et al., 2023), Mistral (Jiang et al., 2023), Baichuan2 (Yang et al., 2023a), ChatGLM3 (Zeng et al., 2023) and FLAN-T5 (Chung et al., 2022). We conduct experiments under zero-shot and fewshot settings, combining commonly used reasoning techniques, chain-of-thought prompting (Kojima et al., 2022; Wei et al., 2022). The experimental results suggest that GPT-4 outperforms other models, showcasing strong temporal reasoning capabilities, as shown in Figure 1. Nevertheless, there is still a considerable gap between the strongest models and humans. On the contrary, open-source models show inferior performance in temporal reasoning, attributed to shortcomings in abstract time understanding, temporal relations modeling, and a lack of temporal commonsense. In addition, we also observe that chain-of-thought prompting does not yield a consistent improvement in performance. These findings indicate that there is still significant room for improvement in models’ temporal reasoning capabilities. Moreover, we have conducted a thorough analysis of the deficiencies and obstacles faced by models in temporal reasoning. 

We aspire for temporal reasoning to garner increased attention within the research community. Our contributions can be summarized as follows: 

- We introduce TIMEBENCH, a comprehensive and hierarchical benchmark to quantify the temporal reasoning abilities of LLMs. 

- We conduct extensive experiments with several LLMs, revealing a significant gap between even SOTA LLM and humans, indicating substantial research opportunities in this field. 

- By conducting a thorough analysis, we reveal the dilemmas that LLMs face in temporal reasoning and identify potential solutions. 

## **2 TIMEBENCH Benchmark** 

## **2.1 Benchmark Design Principal** 

TIMEBENCH focuses on a comprehensive evaluation of the temporal reasoning capabilities of large language models in challenging and complex scenarios. To achieve this goal, we summarize the difficulties and challenges faced in temporal reasoning, categorize them into three levels, and integrate diverse task formats to better align with the intricate nature of temporal reasoning. 

Just as the human cognitive process unfolds from foundational cognition and conceptual understanding to practical reasoning, we delineate temporal reasoning into three hierarchical levels. Specifically, TIMEBENCH categorizes temporal reasoning into symbolic, commonsense and event temporal reasoning, covering 10 datasets with a total of 16 subtasks. (1) Symbolic Temporal Reasoning focuses on the comprehension of fundamental abstract temporal expressions. (2) Temporal Commonsense Reasoning emphasizes the mastery of temporal principles, concepts and world knowledge. (3) Event Temporal Reasoning concentrates on modeling the temporal relationships between events and times within authentic scenarios. 

## **2.2 Difficulties and Challenges** 

We delineate the essential competencies and the challenges that arise from a human cognitive standpoint in the realm of temporal reasoning, and language models confront similar challenges. We present the dataset statistics, task formats, and the associated challenges in Table 7. 

**Time Expression Understanding** Time expressions (TimeX) denote words or phrases that convey information about time and represent the simplest and most basic units of expressing time, such as _in April 2000_ , _after 2008_ . Grasping time expressions is the most foundational step in understanding temporal elements within the textual modality. 

**Temporal Commonsense** assesses the understanding of temporal world knowledge, including event order, event duration, typical time, event frequency and stationary, which is crucial for language models to comprehend daily scenarios. 

**Event-Time Relations** assesses the model’s grounding capability to establish temporal relationships between events and their temporal context, thereby enabling models to grasp the progression 

2 

## **DATE ARITH** 

**Q** : _What is the time 2 year and 4 month before Mar, 1755_ **A** : Nov, 1752 

## **TIMEX NLI** 

**Premise** : _On 28th May 1967, I graduated._ **Hypothesis** : _Before 23rd October 1920, I graduated._ **A** : Contradiction 

Table 1: Examples of symbolic temporal reasoning 

## **MCTACO** 

- **C** : _Ransome looks after her as well as for young Fern Simon , who has declared her love for him._ **Q** : _How often do Ransome and Fern talk?_ **O** : each century, once a day, once a century, every night 

## **TIMEDIAL** 

**Dialog** : _... Person1: Do you go to work by train every day Person2: Yes . I commute <MASK> a week by train..._ **O** : five days, 25 days, a minute, six days 

## **SITUATEDGEN** 

**Keywords** : _axis, one day, one month, Earth, Moon_ **A** : Earth rotates on its axis once in one day. It takes one month for the Moon to rotate on its axis. 

Table 2: Examples of commonsense temporal reasoning. 

and transformations of events as they dynamically evolve through time. 

**Event-Event Relations** not only involve eventtime grounding but also introduce multi-hop relative connections between events. Models with this capability can better handle temporal reasoning in complex scenarios involving multiple events. 

**Implicit Temporal Reasoning** involves going beyond the surface of texts, engaging in deeper reasoning such as drawing upon temporal commonsense, identifying implicit temporal factors and discerning hidden temporal relationships among events. Implicit temporal reasoning is pivotal in complex real-world scenarios where events and time are intricately interwoven. 

## **2.3 Symbolic Temporal Reasoning** 

To evaluate the language model’s comprehension of abstract time expressions, we utilize two symbolic reasoning tasks stripped of semantic content: date arithmetic and time expression inference. Table 1 shows examples of symbolic temporal reasoning. 

## **TIMEQA** 

**C** : _... He worked in Utrecht for the firm of P Smits & de Wolf from 1864 to 1867 and then returned to ..._ **Q** : _Where did Ludwig Mond work between Mar 1866 and Sep 1866?_ **A** : Utrecht 

## **MENATQA** 

**C** : _... After the French evacuated Egypt in 1801, Hurshid Pasha was named governor of Egypt in 1804. Muhammad Ali had himself named governor of Egypt in May 1805 ..._ 

**Q** : _Which position did Hurshid Pasha hold from 1804 to 1806, if Hurshid Pasha tepped down as the governor of Egypt in 1808?_ 

**A** : governor of Egypt 

## **TEMPREASON** 

**C** : _... Peter Corke works for Queensland University of Technology from Jan, 2010 to Dec, 2022. Peter Corke works for Commonwealth Scientific from Jan, 1984 to Jan, 2009. ..._ 

**Q** : _Which employer did Peter Corke work for before Queensland University of Technology?_ 

**A** : Commonwealth Scientific 

Table 3: Examples of event temporal reasoning. 

**Date Arithmetic** (Tan et al., 2023) assesses the model’s grasp of abstract date calculation. When provided with a date, the model needs to accurately calculate the date a certain amount of time before or after the given date. The smallest unit is one day. 

**TimeX NLI** (Thukral et al., 2021) focuses on the logical entailment relationships among abstract TimeX, including three aspects: order (s1), duration (s2), and duration with unit conversion (s3). 

## **2.4 Commonsense Temporal Reasoning** 

We measure the model’s mastery of temporal commonsense and world knowledge, along with its capacity for reasoning based on these insights. Table 2 presents examples of temporal commonsense reasoning in QA and generation forms. 

**MCTACO** (Zhou et al., 2019) evaluates diverse commonsense knowledge from different aspects of events, including duration, frequency, order, stationary and typical event time. 

**DurationQA** (Virgo et al., 2022) focuses specifically on temporal commonsense reasoning in the spectrum of event duration. 

**TimeDial** (Qin et al., 2021) considers temporal commonsense reasoning in dialogue scenarios and involves various aspects of commonsense associated with duration, order, and world knowledge. 

3 

**SituatedGen** (Zhang and Wan, 2023) considers generative commonsense reasoning in a constrained text generation scenario. Given a set of contrasting keywords, the model needs to choose appropriate keywords for each sentence and generate a pair of contrasting sentences that satisfy temporal commonsense. 

## **2.5 Event Temporal Reasoning** 

Event temporal reasoning assesses the model’s understanding of relationships between events and time in real-world scenarios, as well as its ability to reasoning under certain temporal or event constraints. Examples are shown in Table 3. 

**TimeQA** (Chen et al., 2021) requires the model to answer time-sensitive questions based on context containing numerous time-involved facts. It is categorized into explicit reasoning and implicit reasoning based on time indicators (before, in, etc.). 

**MenatQA** (Wei et al., 2023) introduces timesensitive factors to elicit implicit temporal reasoning, including time scope change, disruption of facts, and counterfactual questions, which provides a more in-depth assessment of implicit reasoning ability on event-time relations. 

**TempReason** (Tan et al., 2023) removes irrelevant context and focuses on implicit temporal reasoning within structured facts, investigating the model’s capability boundaries. It involves eventtime reasoning and event-event reasoning. 

**TRACIE** (Zhou et al., 2021) evaluates the model’s comprehension of temporal order between implicit events. The model needs to identify events implied in the context and then determine their chronological order. 

**Standard Prompting** We formulate specific instructions for each task. In the zero-shot setting, models follow the instructions to answer questions. In the few-shot setting, models are provided with several question-answer pairs as demonstrations and emulate those instances to answer questions. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models_images/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models.pdf-0004-08.png)


**Chain-of-Thought Prompting** The instructions of CoT are the same as standard prompting. In the zero-shot setting, following Zeroshot CoT (Kojima et al., 2022), we add a reasoning trigger _Let’s think step by step_ after questions to perform chainof-thought reasoning. In the few-shot setting, we manually annotate CoT demonstrations for each task to guide the step-by-step reasoning. Prompts can be found in Appendix B.3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models_images/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models.pdf-0004-10.png)


## **4 Experimental Setup** 

## **4.1 Models** 

We evaluate several popular LLMs, including both open-source and proprietary models, with parameter sizes ranging from 6B to 70B.[2] The complete list of models can be found in Appendix B.1. 

## **4.2 Implementation Details** 

We access proprietary models through Azure API 0613 version. For open-source models, we deploy them locally through FastAPI. We set the temperature to 0.0 for greedy decoding in all experiments. To improve answer extraction accuracy, we prompt models with trigger _Therefore, the answer is_ before model outputs to deduce final answers. 

## **2.6 Task Formats and Evaluation Metrics** 

TIMEBENCH is a multispectral benchmark encompassing four task types: free-form reading comprehension, natural language inference, constrained text generation, and multi-select questions. For detailed task types and their corresponding evaluation metrics, please refer to Appendix A.3 and A.4. 

## **3 Methodology** 

We perform evaluations using the prompt-based approach, including standard prompting and chainof-thought prompting. Experiments are conducted under both zero-shot and few-shot settings. 

## **5 Experimental Results** 

## **5.1 Few-shot Results** 

Table 4 presents the experimental results under few-shot settings. GPT-4 achieves the best performance across three categories, while LLaMA270b and GPT-3.5 rank in the second tier. However, there remains a substantial gap of 19.4% between the most powerful LLM and humans. 

In symbolic temporal reasoning tasks, GPT-4 demonstrates exceptional performance. However, 

> 2 Since OpenAI has never disclosed the scale of ChatGPT series, 6B to 70B here refers to ChatGLM36B to LLaMA270B. 

4 

|**Method**|**Symbolic**<br>**Commonsense**<br>**Event Temporal**<br>**Overall**<br>TimeXNLI<br>Arith DQA McT. TiD. SitGen TimeQA<br>MenatQA<br>TempR<br>TRACIE Sym. Comm. Event Avg.<br>_s1_<br>_s2_<br>_s3_<br>_Exp. Imp. Sco. Ord. Ctf._<br>_L2_<br>_L3_|
|---|---|
|Human<br>98.0 96.0 92.0 100.0 80.8<br>87.1 97.8 100.0 93.3 91.1 85.6 87.3 79.9 97.1 95.3<br>82.5<br>96.5<br>91.4<br>89.0<br>91.5||
|GPT-4<br>85.3 73.3 53.3 100.0 **64.8**<br>**88.3 94.6**<br>**88.6**<br>**73.7** 51.0 **72.4 54.8** 28.7 92.4 **95.9**<br>62.8<br>78.0<br>**84.1**<br>**66.5**<br>**73.7**<br>_+ FS CoT_<br>**92.0 84.0 64.0 100.0** 55.1<br>72.3 93.4<br>-<br>66.9 **52.8** 65.3 52.6 25.9 **96.9** 94.6<br>**66.4**<br>**85.0**<br>73.6<br>65.2<br>72.1||
|GPT-3.5<br>52.0 68.4 31.6 63.6<br>**67.7**<br>71.2 **76.4**<br>**79.1**<br>66.1 48.4 43.2 **51.6** 17.9 84.7 **78.0**<br>55.0<br>53.9<br>**73.6**<br>55.6<br>**59.7**<br>_+ FS CoT_<br>51.6 **71.8** 36.6 **84.4**<br>41.2<br>38.1 71.1<br>-<br>**68.0** 47.0 42.5 41.7 37.8 **89.9** 76.6<br>50.2<br>**61.1**<br>50.1<br>**56.7**<br>56.6||
|LLaMA2_†_<br>70b<br>**55.0** 61.0 37.0 **82.0**<br>**67.4**<br>**85.3 82.7**<br>**74.9**<br>**66.7** 48.3 **61.4** 42.5 33.8 85.2 **85.4**<br>61.0<br>58.8<br>**77.6**<br>60.5<br>**64.4**<br>_+ FS CoT_<br>52.0 **73.0 39.0** 79.5<br>62.3<br>79.1 61.1<br>-<br>64.3 43.0 57.7 45.2 **53.1 87.5** 81.6<br>**67.0**<br>**60.9**<br>67.5<br>**62.4**<br>63.0||
|LLaMA2_†_<br>13b<br>50.0 54.0 30.0 29.5<br>53.3<br>66.0 55.6<br>64.8<br>59.3 48.6 49.6 43.4 37.5 78.7 62.7<br>58.0<br>40.9<br>59.9<br>54.7<br>52.6<br>_+ FS CoT_<br>40.0 61.0 37.0 52.0<br>59.3<br>68.8 40.8<br>-<br>59.4 49.1 **58.4** 43.8 44.1 78.0 68.2<br>58.0<br>47.5<br>56.3<br>57.4<br>54.5||
|LLaMA2_†_<br>7b<br>26.0 50.0 30.0 20.0<br>54.5<br>59.6 45.2<br>62.4<br>54.4 45.3 49.8 41.9 35.8 64.0 53.3<br>49.0<br>31.5<br>55.4<br>49.2<br>46.3<br>_+ FS CoT_<br>37.0 52.0 36.0 25.5<br>56.9<br>67.0 41.9<br>-<br>45.6 36.1 50.9 38.0 **57.3** 59.7 57.7<br>50.0<br>37.6<br>55.3<br>49.4<br>47.4||
|Baichuan2_†_<br>13b 38.0 48.0 33.0 42.5<br>54.8<br>73.0 45.7<br>64.9<br>59.4 **54.2** 52.7 38.0 21.4 77.3 63.5<br>54.0<br>40.4<br>59.6<br>52.6<br>51.3<br>_+ FS CoT_<br>50.0 56.0 34.0 47.0<br>62.0<br>69.3 43.8<br>-<br>58.2 49.6 49.8 40.1 45.6 81.3 65.6<br>**60.0**<br>46.8<br>58.4<br>56.3<br>54.2||
|Baichuan2_†_<br>7b<br>27.0 66.0 **41.0** 32.5<br>59.8<br>69.4 34.3<br>59.8<br>53.8 **50.2** 49.6 38.5 22.9 65.9 51.0<br>55.0<br>41.6<br>55.8<br>48.4<br>48.5<br>_+ FS CoT_<br>30.0 56.0 34.0 34.0<br>57.0<br>69.5 44.5<br>-<br>51.2 40.7 46.4 32.6 **46.3** 61.5 64.1<br>53.0<br>38.5<br>57.0<br>49.5<br>48.1||
|Mistral_†_<br>7b<br>48.0 53.0 38.0 41.0<br>61.8<br>**76.2** 61.8<br>58.3<br>55.9 45.3 49.4 47.8 45.5 76.7 74.8<br>53.0<br>45.0<br>64.5<br>56.1<br>55.4<br>_+ FS CoT_<br>**57.0** 63.0 35.0 54.0<br>61.8<br>45.7 57.3<br>-<br>60.4 46.2 57.2 **47.9** 33.2 65.9 67.9<br>57.0<br>52.3<br>54.9<br>54.5<br>54.0||
|ChatGLM3_†_<br>6b 48.0 70.0 32.0 35.0<br>51.8<br>62.6 55.0<br>61.6<br>57.2 26.3 35.4 41.5 22.5 76.4 55.9<br>58.0<br>46.3<br>57.8<br>46.7<br>49.3<br>_+ FS CoT_<br>47.0 68.0 32.0 46.0<br>53.9<br>64.3 56.5<br>-<br>52.5 24.5 35.0 40.2 22.5 79.4 60.3<br>54.0<br>48.3<br>58.2<br>46.1<br>49.1||



Table 4: Experimental results under **few-shot** settings (standard prompting by default). _[†]_ denotes the base model without alignment. Global top-3 results are **bold** . Figure 8 provides a horizontal comparison of the performance of all models. Full results in Appendix B.2. 

other models exhibit a significant decline in comparison to GPT-4. In commonsense temporal reasoning tasks, GPT4 lags behind humans by only 8.0%, indicating its powerful internal knowledge reservoir. With the model scale shrinking, its knowledge reservoir also decreases gradually, leading to a decline in performance. Notably, there is a significant gap of 25.2% between LLMs and humans in event temporal reasoning, which suggests that LLMs encounter major challenges in modeling intricate event-time relationships. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models_images/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models.pdf-0005-03.png)


**----- Start of picture text -----**<br>
71.9<br>60 53.2 48.6 55.2 61.2 58.3 54.5 60.9 46.3 43.7 58.2 59.0 51.0 47.2 60.9 60.0<br>40<br>ZS<br>ZS CoT<br>20 FS<br>FS CoT<br>0<br>Symbolic Commonsense Event Overall<br>**----- End of picture text -----**<br>


Figure 2: Performance gap with and without CoT prompting. The results are averaged from GPT-4, GPT3.5, Baichuan213b, LLaMA270b and Mistral7b. 

## **5.2 Zero-shot Results** 

Experimental results of alignment models under zero-shot settings are shown in Table 5. In zeroshot settings, GPT-4 and GPT-3.5 rank first and second, respectively, and they significantly outperform all open-source models by a large margin. It is noteworthy that open-source models exhibit a larger performance decline compared to proprietary models when transitioning from few-shot to zero-shot scenarios. GPT, Baichuan2 and LLaMA2 suffer drops of 5.6%, 14.6% and 27.2%, respectively. We attribute this performance decline to the quality of alignment. Restricted by their limited instruction-following capability, open-source models struggle to fully unleash their performance 

solely through instructions. Therefore, few-shot prompting is a better approach for stimulating their temporal reasoning abilities. 

## **5.3 Chain-of-Thought in Temporal Reasoning** 

Previous research has found that chain-of-thought prompting can enhance the model’s reasoning ability (Wei et al., 2022; Kojima et al., 2022). We aim to explore the following questions: _Does CoT prompting bring consistent improvement in temporal reasoning?_ Due to the diversity of temporal reasoning, the above question has not yet been definitively answered. To investigate this, we select several popular LLMs and analyze their performance affected by chain-of-thought prompting. 

5 

|**Method**|**Symbolic**<br>**Commonsense**<br>**Event Temporal**<br>**Overall**<br>TimeXNLI<br>Arith DQA McT. TiD. SitGen TimeQA<br>MenatQA<br>TempR<br>TRACIE Sym. Comm. Event Avg.<br>_s1_<br>_s2_<br>_s3_<br>_Exp. Imp. Sco. Ord. Ctf._<br>_L2_<br>_L3_|
|---|---|
|Human<br>98.0 96.0 92.0 100.0 80.8<br>87.1 97.8 100.0 93.3 91.1 85.6 87.3 79.9 97.1 95.3<br>82.5<br>96.5<br>91.4<br>89.0<br>91.5||
|GPT-4<br>78.6 76.0 50.7 **98.0**<br>**59.2**<br>80.0 **91.1**<br>**59.3**<br>60.6 **46.5 57.0** 57.0 23.1 95.3 **95.0**<br>**64.8**<br>75.8<br>72.4<br>**62.4**<br>68.3<br>_+ CoT_<br>**80.0 76.0 60.0** 92.0<br>58.1<br>**82.6** 89.3<br>-<br>61.3 41.2 54.6 **59.6** 22.6 **97.0** 94.5<br>58.0<br>**77.0**<br>**76.7**<br>61.1<br>**68.5**||
|GPT-3.5<br>45.4 **67.6** 31.2 **97.0**<br>50.5<br>**68.6 69.1**<br>**62.3**<br>**70.8** 35.4 40.9 43.9 22.9 **81.2 73.8**<br>57.4<br>**60.3**<br>**62.6**<br>**53.3**<br>**57.4**<br>_+ CoT_<br>33.6 64.8 33.6 71.0<br>23.2<br>45.1 67.0<br>-<br>64.4 35.1 39.7 42.9 26.3 57.6 68.1<br>52.0<br>50.8<br>45.1<br>48.3<br>48.3||
|LLaMA270_b_<br>**44.0** 47.0 32.0 **78.5**<br>**59.2**<br>**68.9** 57.0<br>25.0<br>40.8 **40.6** 18.9 16.6 12.0 63.5 54.5<br>48.0<br>**50.4**<br>52.5<br>36.8<br>44.1<br>_+ CoT_<br>30.0 66.0 28.0 53.5<br>57.3<br>67.1 **58.6**<br>31.4 19.5 12.2 12.7 20.8 37.5 40.5<br>51.0<br>44.4<br>**61.0**<br>28.2<br>39.1||
|LLaMA213_b_<br>30.0 49.0 34.0 22.5<br>38.5<br>40.6 35.4<br>57.9<br>**61.9** 30.5 46.1 36.1 26.9 53.1 69.4<br>49.0<br>33.9<br>43.1<br>46.6<br>42.6<br>_+ CoT_<br>36.0 50.0 38.0<br>6.0<br>39.2<br>51.7 36.9<br>-<br>58.7 38.9 40.9 32.5 **33.6** 58.0 68.4<br>47.0<br>32.5<br>42.6<br>47.3<br>42.4||
|LLaMA27_b_<br>39.0 53.0 30.0 13.0<br>39.3<br>41.0<br>6.3<br>24.5<br>49.0 29.0 26.8 21.1 16.0 63.9 47.9<br>49.0<br>33.8<br>27.8<br>37.8<br>34.3<br>_+ CoT_<br>44.0 50.0 33.0<br>5.0<br>35.0<br>40.0<br>1.7<br>-<br>49.9 31.6 31.4 24.5 17.8 56.9 48.1<br>46.0<br>33.0<br>25.6<br>38.3<br>34.3||
|Baichuan213_b_ 41.0 **61.0** 37.0 12.5<br>52.0<br>63.4 57.7<br>52.2<br>55.4 34.6 48.8 **44.3** 39.5 57.4 61.4<br>49.0<br>37.9<br>56.3<br>48.8<br>48.0<br>_+ CoT_<br>40.0 57.0 31.0 10.0<br>44.6<br>61.9 58.1<br>-<br>41.5 **40.9 52.0** 38.5 **43.2** 62.8 64.3<br>55.0<br>34.5<br>54.9<br>49.8<br>46.7||
|Baichuan27_b_<br>35.0 50.0 37.0<br>4.5<br>47.9<br>55.3 54.3<br>42.0<br>41.5 34.7 35.2 31.2 20.4 43.4 47.7<br>55.0<br>31.6<br>49.9<br>38.6<br>39.7<br>_+ CoT_<br>38.0 43.0 32.0<br>1.0<br>37.9<br>58.0 44.2<br>-<br>53.5 38.8 39.9 33.2 29.3 41.2 47.2<br>54.0<br>28.5<br>46.7<br>42.1<br>39.4||
|Vicuna1.513_b_ 35.0 50.0 36.0 15.0<br>39.2<br>59.1 34.2<br>51.8<br>60.4 37.0 **46.8** 37.4 23.2 42.1 43.6<br>46.0<br>34.0<br>46.1<br>42.1<br>41.1<br>_+ CoT_<br>42.0 51.0 37.0<br>3.0<br>29.8<br>50.0 33.7<br>-<br>56.9 36.4 38.2 37.7 20.4 49.0 49.1<br>51.0<br>33.3<br>37.8<br>42.3<br>39.0||
|Vicuna1.57_b_<br>37.0 58.0 43.0<br>5.0<br>40.4<br>52.5 32.0<br>47.8<br>47.1 18.5 35.7 25.7 17.3 33.0 46.8<br>54.0<br>35.8<br>43.2<br>34.8<br>37.1<br>_+ CoT_<br>36.0 50.0 36.0<br>1.5<br>39.4<br>49.2 36.2<br>-<br>40.9 24.6 26.2 28.5 25.0 27.7 40.3<br>54.0<br>30.9<br>41.6<br>33.4<br>34.4||
|FLANT511_b_<br>53.0 63.0 43.0<br>0.0<br>**52.0**<br>65.0 47.7<br>49.5<br>61.7 26.8 33.6 **52.2** 21.8 **87.9 83.9**<br>**64.0**<br>39.8<br>53.6<br>**54.0**<br>**50.3**<br>_+ CoT_<br>**56.0** 66.0 **45.0**<br>0.0<br>49.7<br>63.4 42.7<br>-<br>**64.4** 28.2 41.6 50.2 **30.6** 79.5 68.9<br>55.0<br>41.8<br>51.9<br>52.3<br>49.4||
|Mistral7_b_<br>47.0 50.0 **43.0** 26.5<br>49.8<br>58.8 23.2<br>**58.3**<br>28.2 21.4 24.3 22.3 21.7 39.6 31.6<br>51.0<br>41.6<br>47.5<br>30.0<br>37.3<br>_+ CoT_<br>38.0 56.0 35.0 16.5<br>36.6<br>49.3 19.3<br>-<br>31.3 22.4 21.1 24.9 25.6 34.0 31.2<br>**61.0**<br>36.4<br>35.1<br>31.4<br>33.5||
|ChatGLM36_b_ 38.0 50.0 34.0<br>2.0<br>34.1<br>43.6 56.7<br>38.9<br>41.2 31.7 33.8 26.0 32.2 57.0 54.0<br>50.0<br>31.0<br>43.3<br>40.7<br>39.0<br>_+ CoT_<br>27.0 49.0 37.0<br>0.0<br>24.8<br>37.1 44.8<br>-<br>41.7 25.4 34.6 28.1 41.2 44.5 52.0<br>48.0<br>28.3<br>35.6<br>39.4<br>35.7||



Table 5: Experimental results under **zero-shot** settings (standart prompting by default). All models are alignment models (-chat or -instruct). Global top-3 results are **bold** . 

## **Chain-of-thought reasoning is not consistently** 

**effective.** As illustrated in Figure 2, introducing zero-shot CoT prompting results in consistent declines, with an overall decrease of 7.4%. In the fewshot scenario, CoT prompting also fails to yield consistent improvements, varying depending on the task. There is a 10.8% improvement in symbolic reasoning, while a significant decline of 15.2% in commonsense reasoning. In event temporal reasoning, there is a slight improvement of 1.3%. Next, we will conduct a more detailed analysis of the impact of CoT on specific tasks. 

**Impact of CoT prompting across tasks.** In order to explore the impact of CoT on various tasks thoroughly, we delve into the performance changes of each model across specific tasks within each category, as illustrated in Figure 3. In the zeroshot setting, open-source models achieve a slight improvement in event temporal reasoning with chain-of-thought prompting, while in other cases, they face performance degradation. While in the few-shot setting, almost all models exhibit significant improvement in symbolic temporal reasoning, 

with a concurrent prevalent decline in commonsense temporal reasoning. We attribute this to the knowledge sensitivity inherent in commonsense reasoning, where step-by-step reasoning cannot compensate for the lack of knowledge. In event temporal reasoning, improvements mainly stem from datasets involving implicit multi-step reasoning (MenatQA and TempReason), indicating that CoT is more effective for multi-hop questions. In summary, zero-shot CoT consistently has a negative impact on temporal reasoning. While in fewshot scenario, CoT has a positive impact on symbolic and complex tasks, while negatively affecting knowledge-sensitive tasks. 

## **6 Analysis and Discussion** 

## **6.1 Scaling Effect of Model Size** 

We investigate how the scale of models affects temporal reasoning capabilities. The trend is illustrated in Figure 4. As the model scale increases, there is a notable improvement in performance. When the parameter size expands from 7B to 13B, LLaMA2 and Baichuan2 show improvements of 

6 

|GPT-4<br>GPT-3.5<br>Baichuan213_b_<br>Vicuna1.513_b_<br>Mistral7_b_<br>Zero-Shot<br>TimeXNLI<br>Arith<br>DurationQA<br>McTACO<br>TimeDial<br>TimeQA<br>MenatQA<br>TempReason<br>TRACIE<br>GPT-4<br>GPT-3.5<br>Baichuan213_b_<br>Vicuna1.513_b_<br>Mistral7_b_<br>LLaMA2†<br>70_b_<br>Baichuan2†<br>13_b_<br>Mistral†<br>7_b_<br>Few-Shot<br>Sym.<br>Comm.<br>Event<br>Overall<br>−30<br>−20<br>−10<br>0<br>10<br>20<br>30<br>|Model|Order<br>Duration<br>Freq.<br>Stationarity<br>Typical|Avg.|
|---|---|---|---|
||GPT-4<br>76.4_↓_<br>92.8_↑_<br>83.3_↑_<br>71.4_↓_<br>54.5_↓_<br>77.5<br>GPT-3.5<br>50.5_↑_<br>39.8_↓_<br>55.2_↑_<br>48.4_↑_<br>28.7_↓_<br>43.5<br>Baichuan2_†_<br>13_b_<br>40.5_↓_<br>51.8_↑_<br>43.7_↑_<br>46.2_↑_<br>29.8_↓_<br>42.5<br>LLaMA2_†_<br>70_b_<br>65.2_↑_<br>72.1_↑_<br>66.3_↑_<br>36.3_↓_<br>52.7_↓_<br>63.0<br>Mistral_†_<br>7_b_<br>27.0_↓_<br>44.4_↑_<br>58.3_↑_<br>38.5_↓_<br>38.3_↓_<br>42.5|||
||Table 6: Results in each temporal commonsense aspect<br>under few-shot setting. Models with † are base models<br>Red_↓_and Green_↑_represent the performance is lower<br>or higher than its average performance. Metric is EM.<br>necessitates a two-step reasoning process. It frst<br>unifes time units, and subsequently engages in<br>numerical comparison. In contrast, other atomic|||



Table 6: Results in each temporal commonsense aspect under few-shot setting. Models with † are base models. Red _↓_ and Green _↑_ represent the performance is lower or higher than its average performance. Metric is EM. 

necessitates a two-step reasoning process. It first unifies time units, and subsequently engages in numerical comparison. In contrast, other atomic tasks (s1, s2 and arithmetic) can be completed with a single reasoning step. In summary, LLMs perform poorly in symbolic temporal reasoning and exhibit more pronounced declines when encountering multi-step reasoning. 

Figure 3: ∆Score between the chain-of-thought prompting and direct I-O prompting. **Top:** zero-shot setting, **Bottom:** few-shot setting, **Left:** variation in each task, **Right:** averaged variation in the symbolic, commonsense, event, and overall tasks. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models_images/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models.pdf-0007-04.png)


**----- Start of picture text -----**<br>
65<br>60<br>Model<br>Baichuan2 [†]<br>55<br>ChatGLM3 [†]<br>LLaMA2 [†]<br>50<br>Mistral [†]<br>6b 7b 13b 70b<br>Model Size (B)<br>**----- End of picture text -----**<br>


**Mastery of commonsense knowledge varies in LLMs** We analyze models’ performance across various commonsense aspects, as shown in Table 6. We regard the model’s average performance in commonsense reasoning tasks as the baseline. If the model outperforms the baseline in a specific aspect, it suggests greater proficiency in this type of knowledge, and vice versa. The findings indicate that LLMs generally demonstrate good knowledge of event duration and frequency. However, their comprehension of event order and typical events is relatively weaker The uneven mastery of commonsense knowledge significantly affects the model’s reasoning performance, especially when dealing with complex questions that involve multiple types of knowledge. Retrieval-augmented reasoning presents a promising avenue for mitigating the model’s knowledge scarcity. 

Figure 4: Scaling effect of model size and overall temporal reasoning performance. The x-axis (model size) is shown in the log scale. Results show a log-linearity between parameter size and performance. 

13.0% and 10.5%, respectively. Furthermore, when LLaMA scales up to 70B, the trend of performance improvement continues without stopping. The overall improvement follows a log-linear relationship with scale. There are no significant performance differences among LLaMA2, Baichuan2, and ChatGLM3 under similar parameter specifications, while Mistral demonstrates impressive prowess, outperforming all other 13B models with nearly half the number of parameters. 

## **LLMs exhibit poor implicit temporal reason-** 

**ing capabilities.** When comparing explicit and implicit event temporal reasoning, specifically TimeQA-explicit versus others, we observe a significant performance decrease in implicit reasoning. Additionally, on TRACIE with numerous implied events, most models only surpass a random baseline (50.0). Even GPT-4, despite its advanced capabilities, achieves only a 66.4% accuracy, suggesting that the LLM struggles with modeling implicit temporal relationships. We consider it helpful to explicitly model the temporal relationships between events and time expressions, for instance constructing timelines or temporal graphs. 

## **6.2 Challenges in Temporal Reasoning** 

## **LLMs underperform in (multi-hop) symbolic** 

**reasoning** Except for GPT-4, the performance of all other models in symbolic temporal reasoning is unsatisfactory. A noticeable decrease is observed in duration-conversion task compared to other atomic tasks (25% in GPT-4 and 27% in LLaMA270 _b_ ). This is because the duration-conversion task (s3) 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models_images/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models.pdf-0008-00.png)


**----- Start of picture text -----**<br>
64.4<br>60 53.2 54.253.7 54.5 54.5 55.4<br>45.5 46.7 43.0 49.3 48.546.0 47.4 47.4<br>40 39.2 38.7 37.1<br>20<br>Base<br>Align<br>0<br>LLaMA2Baichuan270 b LLaMA213 b Vicuna1.513 b 13Mistral b ChatGLM37 b Baichuan26 b LLaMA27 b Vicuna1.57 b 7 b<br>**----- End of picture text -----**<br>


Figure 5: Performance difference between base and alignment models under few-shot setting. Baichuan2 and LLaMA2 are aligned with SFT and RLHF. Vicuna, Mistral and ChatGLM3 are aligned with only SFT. 

**LLMs are good factual reasoners rather than factual extractors** When humans engage in temporal reasoning, it generally involves two steps: first, extracting time-fact pairs from the context, and then performing fact-based reasoning. TempReason provides extracted facts for conducting fact-based reasoning. By comparing the model’s performance in context-based (TimeQA) against fact-based (TempReason) reasoning, we identify the bottleneck in event temporal reasoning. LLMs excel in TempReason, which signifies their strong capability in fact-based reasoning. However, their performance in context-based reasoning is significantly weaker compared to their performance in fact-based reasoning. This implies that errors could arise during the extraction of time-sensitive facts from the context. We attribute this performance gap to the model’s deficiency in factual extraction capabilities Thus, we consider LLMs to be strong factual reasoners rather than factual extractors in event temporal reasoning. 

## **6.3 Alignment Impairs Temporal Reasoning** 

In the experiments mentioned earlier (Table 5), we observe a sharp decline in zero-shot performance of alignment models. To investigate whether alignment is the cause of the decline in temporal reasoning, we conducted experiments on alignment models under few-shot settings. Figure 5 illustrates the overall performance decline after alignment. With the exception of Baichuan2, all other models are severely impaired, experiencing a significant drop of up to 22%. Through manual analysis of error cases, we have summarized two reasons: (1) Alignment reduces the model’s usability, causing it to tend towards refusal to answer when confronted with knowledge-sensitive questions. (2) Alignment damages the model’s in-context learning capability, 

resulting in situations where the model deviates from the demonstrations. Furthermore, we believe that the lack of temporal reasoning-related training data in alignment exacerbates this issue, leading to disparities between different reasoning capabilities, such as mathematical and temporal reasoning. 

## **6.4 Error Analysis** 

We manually analyze 100 predictions by GPT-4, GPT-3.5 and LLaMa2-base70 _b_ from each subtask. The visualization of errors is shown in Figure 6. 

**Symbolic Reasoning** We categorize symbolic reasoning errors into five groups: (a) _Expression_ : The model provides an incorrect time calculation expression. (b) _Computation_ : The model provides the correct time calculation expression, but there is a calculation error. (c) _Conversion_ : The model has an error in the conversion of time units. (d) _Comparison_ : The model has an error when comparing two time-expressions (or intervals). (e) _Combination_ : The model encounters errors in the combination of multiple above operations. LLMs exhibit numerous computation, conversion, and comparison errors, which suggests a substantial deficiency in their understanding of fundamental temporal expressions. Additionally, a higher frequency of errors is observed in combination questions, highlighting that multi-step reasoning continues to be a significant challenge for current models 

**Commonsense Reasoning** We categorize the errors of commonsense reasoning into two groups: (a) _No Answer_ : The model fails to provide a final answer. (b) _Reasoning Error_ : The model encounters reasoning errors, which can be subdivided into five types of knowledge-related errors. We observe that GPT series models have a higher _No Answer_ rate, while LLaMA is always able to provide answers. This discrepancy can be attributed to two factors: firstly, the models may lack the necessary commonsense knowledge to formulate an answer; secondly, the preference alignment mechanism may prompt the model to abstain from answering when confronted with questions outside its knowledge scope. Integration of retrieval can alleviate the problem of knowledge scarcity to a certain degree. 

**Event Temporal Reasoning** We categorize the errors of event temporal reasoning into four groups: (a) _No Answer_ : The model is unable to find the answer in the context. (b) _Reasoning Error_ : The model encounters reasoning errors. (c) _Halluci-_ 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models_images/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models.pdf-0009-00.png)


**----- Start of picture text -----**<br>
Symbolic Commonsense Event Temporal<br>Reasoning Error 62 61 Reasoning Error Reasoning Error GPT-4<br>52 GPT-3.5<br>48 LLaMA2<br>42 42<br>34 34<br>0 0 2 0 30 15 6 12 27 27 23 0 2 6 4 5 15 9 2 15 13 14 19 11 7 25 17 25 4 3 6 11 8 13 24 24 22 21 29 24 2 4 4 5 3 3<br>ExpressionComputationConversionComparisonCombination No AnswerStationarityFrequencyEvent OrderingTypical TimeEvent Duration No AnswerHallucinateMetric Limit In Between Before After<br>**----- End of picture text -----**<br>


Figure 6: Error analysis for Symbolic, Commonsense, and Event Temporal. We select 100 test samples from each subtask for GPT-4, GPT-3.5 and LLaMa2-base70 _b_ . 

_nation_ : The model’s prediction does not exist in the context, known as hallucination reasoning. (d) _Metric_ : The model’s prediction is correct, but the metric is limited by the evaluation criteria. It can be observed that, except for reasoning errors, failures to provide answers account for approximately 30%, indicating that models still have flaws in grounding temporal facts from context. Additionally, models occasionally experience hallucination phenomena, leading to erroneous reasoning. 

## **7 Related Work** 

## **7.1 Temporal Reasoning** 

There are numerous efforts addressing diverse challenges in temporal reasoning. Early research mainly relies on TimeML (Pustejovsky et al., 2003), focusing TimeX extraction and temporal relation extraction (Verhagen et al., 2007, 2010; UzZaman et al., 2013; Llorens et al., 2015; Miller et al., 2015; Mathur et al., 2021; Vashishtha et al., 2019). The advent of pre-trained language models (PLMs) has brought about commonsense reasoning as a tool to explore the world knowledge in models (Zhou et al., 2019; Qin et al., 2021; Dhingra et al., 2022). Recently, much attention has shifted towards event temporal reasoning (Chen et al., 2021; Tan et al., 2023; Wei et al., 2023). Han et al. (2021); Yang et al. (2023b); Son and Oh (2023); Chen et al. (2023) continuously pretrains LLMs on time-aware data to elicit temporal reasoning, and Zhu et al. (2023); Su et al. (2023); Chu et al. (2023) explicitly represent temporal relationships using temporal graphs and timelines. Additionally, some works extend beyond text, evaluating temporal reasoning in structured tables and video domains (Gupta et al., 2023; Ko et al., 2023). 

Some concurrent studies also analyze the temporal reasoning abilities of LLMs. Jain et al. (2023); Qiu et al. (2023) focus on temporal commonsense 

and Wang and Zhao (2023) introduces a unified form for accessing the overall abilities. 

Distinguished from other works, TIMEBENCH is multispectral, offering a comprehensive evaluation of LLM’s temporal reasoning abilities. 

## **7.2 Large Language Models** 

In recent years, there has been rapid progress in the research of large language models (LLM) (Zhao et al., 2023). They exhibit outstanding performance across a multitude of tasks without the need for finetuning (Brown et al., 2020; Kojima et al., 2022). Furthermore, they have achieved astonishing results in complex reasoning tasks, such as mathematical reasoning (Cobbe et al., 2021; Mishra et al., 2022) and logical reasoning (Yu et al., 2020; Liu et al., 2023). Moreover, some studies suggest that the chain-of-thought prompting can further enhance the model’s capabilities in complex reasoning scenarios (Wei et al., 2022; Kojima et al., 2022; Chu et al., 2024; Zhang et al., 2023). 

## **8 Conclusion** 

Temporal reasoning entails inherent diversity and complexity. The lack of a comprehensive benchmark makes it challenging to quantify LLMs’ temporal reasoning capabilities. In this work, we present TIMEBENCH, a comprehensive and hierarchical benchmark for LLM temporal reasoning, tailored to mirror temporal reasoning in complex scenarios. We conduct extensive experiments on state-of-the-art LLMs to investigate their temporal reasoning capabilities. Our findings indicate a substantial gap between state-of-the-art LLMs and human performance, emphasizing the need for further research in this area. Moreover, we provide a meticulous analysis and discussion, outlining the current challenges that models face and suggesting potential directions for improvement. 

9 

## **Limitations** 

TimeBench is a comprehensive benchmark to quantify the temporal reasoning capabilities of LLMs. While we have taken various factors into account, there are a few limitations. Firstly, our evaluation only applied prompt-based method under zero-shot and few-shot setting, lacking evaluations specifically tailored for models fine-tuned on the temporal domain. Secondly, the instructions and demonstrations were manually crafted, which may potentially lead to discrepancies in prompts interpretation among different LLMs. Thirdly, the dataset constituting the benchmark includes data from past years and a portion sourced from Wikipedia, which may contaminate the training corpus of LLMs. 

## **Acknowledgements** 

The research in this article is supported by the National Key Research and Development Project (2021YFF0901602), the National Science Foundation of China (U22B2059, 62276083), and Shenzhen Foundational Research Funding (JCYJ20200109113441941), Major Key Project of PCL (PCL2021A06). Ming Liu is the corresponding author. 

## **References** 

- Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. 2023. GQA: training generalized multi-query transformer models from multi-head checkpoints. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023_ , pages 4895–4901. Association for Computational Linguistics. 

- Satanjeev Banerjee and Alon Lavie. 2005. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In _Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization_ , pages 65–72. 

- Lawrence W Barsalou, Léo Dutriaux, and Christoph Scheepers. 2018. Moving beyond the distinction between concrete and abstract concepts. _Philosophical Transactions of the Royal Society B: Biological Sciences_ , 373(1752):20170144. 

- Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. _arXiv preprint arXiv:2004.05150_ . 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda 

Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ . 

- Wenhu Chen, Xinyi Wang, and William Yang Wang. 2021. A dataset for answering time-sensitive questions. In _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual_ . 

- Ziqiang Chen, Shaojuan Wu, Xiaowang Zhang, and Zhiyong Feng. 2023. TML: A temporal-aware multitask learning framework for time-sensitive question answering. In _Companion Proceedings of the ACM Web Conference 2023, WWW 2023, Austin, TX, USA, 30 April 2023 - 4 May 2023_ , pages 200–203. ACM. 

- Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. Vicuna: An opensource chatbot impressing gpt-4 with 90%* chatgpt quality. 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2023. Palm: Scaling language modeling with pathways. _J. Mach. Learn. Res._ , 24:240:1– 240:113. 

- Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang, Weihua Peng, Ming Liu, Bing Qin, and Ting Liu. 2024. Navigate through enigmatic labyrinth a survey of chain of thought reasoning: Advances, frontiers and future. In _The 62nd Annual Meeting of the Association for Computational_ 

10 

_Linguistics: ACL 2024, Bangkok, Thailand, August 11–16, 2024_ . Association for Computational Linguistics. 

- Zheng Chu, Zekun Wang, Jiafeng Liang, Ming Liu, and Bing Qin. 2023. MTGER: multi-view temporal graph enhanced temporal reasoning over timeinvolved document. In _Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023_ , pages 15218–15233. Association for Computational Linguistics. 

- Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2022. Scaling instruction-finetuned language models. _Preprint_ , arXiv:2210.11416. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. _CoRR_ , abs/2110.14168. 

- Bhuwan Dhingra, Jeremy R. Cole, Julian Martin Eisenschlos, Daniel Gillick, Jacob Eisenstein, and William W. Cohen. 2022. Time-aware language models as temporal knowledge bases. _Trans. Assoc. Comput. Linguistics_ , 10:257–273. 

- Vivek Gupta, Pranshu Kandoi, Mahek Bhavesh Vora, Shuo Zhang, Yujie He, Ridho Reinanda, and Vivek Srikumar. 2023. Temptabqa: Temporal question answering for semi-structured tables. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023_ , pages 2431–2453. Association for Computational Linguistics. 

- Rujun Han, Xiang Ren, and Nanyun Peng. 2021. ECONET: effective continual pretraining of language models for event temporal reasoning. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021_ , pages 5367–5380. Association for Computational Linguistics. 

- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In _9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021_ . OpenReview.net. 

- Raghav Jain, Daivik Sojitra, Arkadeep Acharya, Sriparna Saha, Adam Jatowt, and Sandipan Dandapat. 

2023. Do language models have a common sense regarding time? revisiting temporal commonsense reasoning in the era of large language models. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 6750– 6774, Singapore. Association for Computational Linguistics. 

- Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. _CoRR_ , abs/2310.06825. 

- Dohwan Ko, Ji Soo Lee, Woo-Young Kang, Byungseok Roh, and Hyunwoo Kim. 2023. Large language models are temporal and causal reasoners for video question answering. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023_ , pages 4300–4316. Association for Computational Linguistics. 

- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. In _NeurIPS_ . 

- Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In _Text summarization branches out_ , pages 74–81. 

- Hanmeng Liu, Zhiyang Teng, Ruoxi Ning, Jian Liu, Qiji Zhou, and Yue Zhang. 2023. Glore: Evaluating logical reasoning of large language models. _CoRR_ , abs/2310.09107. 

- Hector Llorens, Nathanael Chambers, Naushad UzZaman, Nasrin Mostafazadeh, James F. Allen, and James Pustejovsky. 2015. Semeval-2015 task 5: QA tempeval - evaluating temporal information understanding with question answering. In _Proceedings of the 9th International Workshop on Semantic Evaluation, SemEval@NAACL-HLT 2015, Denver, Colorado, USA, June 4-5, 2015_ , pages 792–800. The Association for Computer Linguistics. 

- Puneet Mathur, Rajiv Jain, Franck Dernoncourt, Vlad Morariu, Quan Hung Tran, and Dinesh Manocha. 2021. TIMERS: Document-level temporal relation extraction. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)_ , pages 524–533, Online. Association for Computational Linguistics. 

- Timothy Miller, Steven Bethard, Dmitriy Dligach, Chen Lin, and Guergana Savova. 2015. Extracting time expressions from clinical text. In _Proceedings of BioNLP 15_ , pages 81–91, Beijing, China. Association for Computational Linguistics. 

11 

- Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard Tang, Sean Welleck, Chitta Baral, Tanmay Rajpurohit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, and Ashwin Kalyan. 2022. LILA: A unified benchmark for mathematical reasoning. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022_ , pages 5807–5832. Association for Computational Linguistics. 

- OpenAI. 2023. GPT-4 technical report. _CoRR_ , abs/2303.08774. 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In _NeurIPS_ . 

- Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th annual meeting of the Association for Computational Linguistics_ , pages 311–318. 

- James Pustejovsky, José M. Castaño, Robert Ingria, Roser Saurí, Robert J. Gaizauskas, Andrea Setzer, Graham Katz, and Dragomir R. Radev. 2003. Timeml: Robust specification of event and temporal expressions in text. In _New Directions in Question Answering, Papers from 2003 AAAI Spring Symposium, Stanford University, Stanford, CA, USA_ , pages 28–34. AAAI Press. 

- Lianhui Qin, Aditya Gupta, Shyam Upadhyay, Luheng He, Yejin Choi, and Manaal Faruqui. 2021. TIMEDIAL: temporal commonsense reasoning in dialog. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021_ , pages 7066–7076. Association for Computational Linguistics. 

- Yifu Qiu, Zheng Zhao, Yftah Ziser, Anna Korhonen, Edoardo M. Ponti, and Shay B. Cohen. 2023. Are large language models temporally grounded? _CoRR_ , abs/2311.08398. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. _J. Mach. Learn. Res._ , 21:140:1–140:67. 

- Jungbin Son and Alice Oh. 2023. Time-aware representation learning for time-sensitive question answering. In _Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023_ , pages 70–77. Association for Computational Linguistics. 

- Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat Agarwal, Alethea Power, Alex Ray, Alex Warstadt, Alexander W. Kocurek, Ali Safaya, Ali Tazarv, Alice Xiang, Alicia Parrish, Allen Nie, Aman Hussain, Amanda Askell, Amanda Dsouza, Ameet Rahane, Anantharaman S. Iyer, Anders Andreassen, Andrea Santilli, Andreas Stuhlmüller, Andrew M. Dai, Andrew La, Andrew K. Lampinen, Andy Zou, Angela Jiang, Angelica Chen, Anh Vuong, Animesh Gupta, Anna Gottardi, Antonio Norelli, Anu Venkatesh, Arash Gholamidavoodi, Arfa Tabassum, Arul Menezes, Arun Kirubarajan, Asher Mullokandov, Ashish Sabharwal, Austin Herrick, Avia Efrat, Aykut Erdem, Ayla Karakas, and et al. 2022. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _CoRR_ , abs/2206.04615. 

- Xin Su, Phillip Howard, Nagib Hakim, and Steven Bethard. 2023. Fusing temporal graphs into transformers for time-sensitive question answering. In _Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023_ , pages 948–966. Association for Computational Linguistics. 

- Qingyu Tan, Hwee Tou Ng, and Lidong Bing. 2023. Towards benchmarking and improving the temporal reasoning capability of large language models. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023_ , pages 14820–14835. Association for Computational Linguistics. 

- Shivin Thukral, Kunal Kukreja, and Christian Kavouras. 2021. Probing language models for understanding of temporal expressions. In _Proceedings of the Fourth BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP, BlackboxNLP@EMNLP 2021, Punta Cana, Dominican Republic, November 11, 2021_ , pages 396–406. Association for Computational Linguistics. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, 

12 

Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and finetuned chat models. _Preprint_ , arXiv:2307.09288. 

- Naushad UzZaman, Hector Llorens, Leon Derczynski, James F. Allen, Marc Verhagen, and James Pustejovsky. 2013. Semeval-2013 task 1: Tempeval-3: Evaluating time expressions, events, and temporal relations. In _Proceedings of the 7th International Workshop on Semantic Evaluation, SemEval@NAACLHLT 2013, Atlanta, Georgia, USA, June 14-15, 2013_ , pages 1–9. The Association for Computer Linguistics. 

- Siddharth Vashishtha, Benjamin Van Durme, and Aaron Steven White. 2019. Fine-grained temporal relation extraction. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 2906–2919, Florence, Italy. Association for Computational Linguistics. 

- Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. 2015. Cider: Consensus-based image description evaluation. In _IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015_ , pages 4566–4575. IEEE Computer Society. 

- Marc Verhagen, Robert J. Gaizauskas, Frank Schilder, Mark Hepple, Graham Katz, and James Pustejovsky. 2007. Semeval-2007 task 15: Tempeval temporal relation identification. In _Proceedings of the 4th International Workshop on Semantic Evaluations, SemEval@ACL 2007, Prague, Czech Republic, June 23-24, 2007_ , pages 75–80. The Association for Computer Linguistics. 

- Marc Verhagen, Roser Saurí, Tommaso Caselli, and James Pustejovsky. 2010. Semeval-2010 task 13: Tempeval-2. In _Proceedings of the 5th International Workshop on Semantic Evaluation, SemEval@ACL 2010, Uppsala University, Uppsala, Sweden, July 15-16, 2010_ , pages 57–62. The Association for Computer Linguistics. 

- Felix Giovanni Virgo, Fei Cheng, and Sadao Kurohashi. 2022. Improving event duration question answering by leveraging existing temporal information extraction data. In _Proceedings of the Thirteenth Language Resources and Evaluation Conference, LREC 2022, Marseille, France, 20-25 June 2022_ , pages 4451– 4457. European Language Resources Association. 

- Yuqing Wang and Yun Zhao. 2023. TRAM: benchmarking temporal reasoning for large language models. _CoRR_ , abs/2310.00835. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In _NeurIPS_ . 

- Yifan Wei, Yisong Su, Huanhuan Ma, Xiaoyan Yu, Fangyu Lei, Yuanzhe Zhang, Jun Zhao, and Kang Liu. 2023. Menatqa: A new dataset for testing the temporal comprehension and reasoning abilities of large language models. In _Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023_ , pages 1434–1447. Association for Computational Linguistics. 

- Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, Fan Yang, Fei Deng, Feng Wang, Feng Liu, Guangwei Ai, Guosheng Dong, Haizhou Zhao, Hang Xu, Haoze Sun, Hongda Zhang, Hui Liu, Jiaming Ji, Jian Xie, Juntao Dai, Kun Fang, Lei Su, Liang Song, Lifeng Liu, Liyun Ru, Luyao Ma, Mang Wang, Mickel Liu, MingAn Lin, Nuolan Nie, Peidong Guo, Ruiyang Sun, Tao Zhang, Tianpeng Li, Tianyu Li, Wei Cheng, Weipeng Chen, Xiangrong Zeng, Xiaochuan Wang, Xiaoxi Chen, Xin Men, Xin Yu, Xuehai Pan, Yanjun Shen, Yiding Wang, Yiyu Li, Youxin Jiang, Yuchen Gao, Yupeng Zhang, Zenan Zhou, and Zhiying Wu. 2023a. Baichuan 2: Open large-scale language models. _CoRR_ , abs/2309.10305. 

- Sen Yang, Xin Li, Lidong Bing, and Wai Lam. 2023b. Once upon a time in graph: Relative-time pretraining for complex temporal reasoning. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023_ , pages 11879–11895. Association for Computational Linguistics. 

- Weihao Yu, Zihang Jiang, Yanfei Dong, and Jiashi Feng. 2020. Reclor: A reading comprehension dataset requiring logical reasoning. In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net. 

- Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Zhiyuan Liu, Peng Zhang, Yuxiao Dong, and Jie Tang. 2023. GLM-130B: an open bilingual pre-trained model. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net. 

- Yunxiang Zhang and Xiaojun Wan. 2023. Situatedgen: Incorporating geographical and temporal contexts into generative commonsense reasoning. In _Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023_ . 

- Zhuosheng Zhang, Yao Yao, Aston Zhang, Xiangru Tang, Xinbei Ma, Zhiwei He, Yiming Wang, Mark Gerstein, Rui Wang, Gongshen Liu, and Hai Zhao. 2023. Igniting language intelligence: The hitchhiker’s guide from chain-of-thought reasoning to language agents. _CoRR_ , abs/2311.11797. 

13 

- Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023. A survey of large language models. _Preprint_ , arXiv:2303.18223. 

- Ben Zhou, Daniel Khashabi, Qiang Ning, and Dan Roth. 2019. "going on a vacation" takes longer than "going for a walk": A study of temporal commonsense understanding. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019_ , pages 3361– 3367. Association for Computational Linguistics. 

- Ben Zhou, Kyle Richardson, Qiang Ning, Tushar Khot, Ashish Sabharwal, and Dan Roth. 2021. Temporal reasoning on implicit events from distant supervision. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021_ , pages 1361–1371. Association for Computational Linguistics. 

- Xinyu Zhu, Cheng Yang, Bei Chen, Siheng Li, JianGuang Lou, and Yujiu Yang. 2023. Question answering as programming for solving time-sensitive questions. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023_ , pages 12775–12790. Association for Computational Linguistics. 

## **A TIMEBENCH Details** 

TIMEBENCH features 3 major categories, 10 tasks and 15 subtasks, each with distinct challenges, totaling 19,000 instances. Detailed statistics are available in Figure 7 and Table 7. 

## **A.1 Benchmark Construction** 

**TimeX Arithmetic** (Tan et al., 2023) TimeX Arithmetic data is derived from the _l1: time-time_ reasoning data in TempReason. We retain 4,000 instances, where time expressions are calculated with a minimum unit of one day. 

**TimeX NLI** (Thukral et al., 2021) The original data of TimeXNLI is in NLI format, including three sub-tasks, _Temp-Order_ , _Temp-Duration_ , and _CrossUnit Duration_ , including 6,140, 3,540, and 15,840 instances respectively. We conduct a random sampling of 2,213, 2,332 and 2,429 entries, resulting in a combined total of 6,965 instances. 

**MCTACO** (Zhou et al., 2019) The original MCTACO dataset consists of yes/no questions, containing 1,332 questions with 9,442 options. To guarantee that the questions are presented in a 4-way multi-select style, we initially remove questions that have less than four options. Subsequently, to ensure that each question has at least one correct option, we filter out questions where all options are labeled as "no". For each remaining question, we randomly sample four options, striving to maintain a balance between correct and incorrect options. In most cases, a question is accompanied by 2 correct and 2 incorrect options. A minority of questions have an option distribution of 1-3 or 3-1. After the aforementioned filtering process, we obtain 852 pieces of data in a 4-way multi-select format. 

**DurationQA** (Virgo et al., 2022) The original DurationQA has the same format as MCTACO, which consists of 694 questions with 4,868 options. Following the identical filtration procedure as MCTACO, we finally obtained a collection of 687 questions in a 4-way multi-select format. 

**TimeDial** (Qin et al., 2021) consists of 4-way multi-select instances in a two-person dialogue scenario. We leave the original data unaltered and simply randomize the sequence of options, yielding 1,446 pieces of 4-way multi-select instances. 

**SituateGen** (Zhang and Wan, 2023) SituatedGen includes 1,220 test cases, which span across two distinct reasoning domains: _time_ and _geography_ . We manually screen the original test data and retain those with clear time features for temporal reasoning evaluation, resulting in 115 instances. 

**TimeQA** (Chen et al., 2021) The original data of TimeQA includes two splits, _Easy_ and _Hard_ , with each question containing 20 Wikipedia paragraphs. The excessively long context may exceed the model’s maximum length limit and incur significant inference overhead. Therefore, we have reduced the context of the original data. For the paragraphs in the original data, we refer to those containing the answer as relevant paragraphs, and the rest as irrelevant paragraphs. For each question, we keep the first paragraph, all relevant paragraphs, and one random irrelevant paragraph as distractor. This ensures that most questions have at least three paragraphs. After that, we sample 500 pieces of data from those where the context length is less than 650 tokens. For both _Easy_ and _Hard_ splits, 

14 

we apply the aforementioned filtration, resulting in 500 questions each, totaling 1,000 instances. 

**TempReason** (Tan et al., 2023) TempReason dataset contains 5,397 entries for l2 (event-time reasoning) and 4,426 entries for l3 (event-event reasoning). In the original dataset, each question corresponds with a text context and extracted facts. Similar to TimeQA, we apply a filter based on context length. We preserve questions with a context length between 300 and 600 tokens, yielding 839 and 1,037 instances, respectively. Notably, every remaining question is applicable to either contextbased reasoning or fact-based reasoning. 

**MenatQA** (Wei et al., 2023) MenatQA consists of 999 data entries, formatted similarly to TimeQA, where each question is accompanied by several corresponding paragraphs. Following the paper’s proposed method, we modify the original data by incorporating the three time-sensitive factors: scope, order, and counterfactual. Subsequently, for each factor, we randomly sample 400 instances, resulting in a total of 1,200 data points. 

**TRACIE** (Zhou et al., 2021) The original TRACIE dataset consists of yes/no type questions, containing 4,248 test instances. We randomly sample 500 instances from the _iid_ split in the test set. 

## **A.2 Human Performance Evaluation** 

Unless otherwise stated, the results of human evaluation are derived from original dataset papers. Please refer to the corresponding paper for human evaluation details. TimeXNLI, Date Arith, and MCTACO are manually evaluated by three authors from the TimeBench team. Within each subtask, we randomly sample 50 instances, and the average of the performances by three human evaluators is considered the final human performance. 

## **A.3 Task Formats** 

TIMEBENCH is a multispectral benchmark, which features four different task formats. 

**Multi-Select Questions** Previous work utilizes the Multiple Choice (MC) form, which requires models to select the only correct answer from the options. However, this task form has shortcuts and may not truly reflect the model’s abilities. To address this, we employ the Multi-Select (M-S) task form, where the model needs to select all possible correct answers from the options provided. In our 

task, each question presents four options, with at most two of them being correct. 

**Natural Language Inference** is the task of determining the logical relationship between two pieces of text. Specifically, given a premise and a hypothesis, the model needs to determine whether the hypothesis can be inferred from the premise and output entailment, contradiction, or neutral. Our tasks focus on the entailment in temporal domains. 

**Free-form Reading Comprehension** requires models to answer questions based on the provided context, and the ground truth answer is free-form without pre-defined format restrictions. 

**Constrained Text Generation** refers to the task of generating text under certain constraints. The task is keyword-constrained text generation, where the model takes keywords as input and outputs sentences that include those keywords. 

## **A.4 Evaluation Metrics** 

Accuracy is used for NLI and date arithmetic tasks. M-S tasks are evaluated using option-level EM and F1. FRC tasks (excluding date arithmetic) are assessed with token-level EM and F1. For CTG task, we take the average of multiple generation metrics, which are outlined as follows. 

**Metrics for SituatedGen** Following SituatedGen (Zhang and Wan, 2023), we use BLEU-4 (Papineni et al., 2002), METEOR (Banerjee and Lavie, 2005), ROUGE-L (Lin, 2004), CIDEr (Vedantam et al., 2015), and MATCH (Zhang and Wan, 2023) scores to metric the results of CTG.[3] 

The overall score is calculated as the sum of the above scores. We set the weight of CIDEr to 1 _/_ 10 for balancing when summation. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models_images/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models.pdf-0015-17.png)


As the overall score _S_ does not represent a percentile, we proceed to normalize the models’ scores to align with humans’ relative performance levels. 

## **B Supplemental Materials** 

## **B.1 Models** 

**ChatGPT-3.5/GPT-4** (Ouyang et al., 2022; OpenAI, 2023) ChatGPT is a chat model aligned 

3We utilize pycocoevalcap package to calucate BLEU-4, METEOR, ROUGE-L, CIDEr. 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models_images/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models.pdf-0016-00.png)


**----- Start of picture text -----**<br>
SituatedGen<br>115<br>Symbolic Common Sense Event Temporal<br>Order<br>400<br>Implicit<br>500<br>Counterfactual548 MenatQA<br>TimeQA<br>Event  Symbolic<br>Temporal<br>Explicit500<br>Order<br>Scope651 2213<br>Duration2332<br>TRACIE500 Time NLI<br>Event to Time839<br>Temp Reason<br>1446TimeDial<br>1037<br>Event to Event<br>SenseCommon<br>852McTACO<br>Convention2420<br>4000<br>TimeX Arith<br>687<br>DurationQA<br>**----- End of picture text -----**<br>


## **B.2 Full Results** 

The overall score is derived from the average of all corresponding metrics. For brevity, we omit some F1 scores in tables in the main text. Please refer to Table 9 for the full experimental results. The full results of SituatedGen can be found in Table 8. 

## **B.3 Prompts** 

The prompt formats are showcased in Figure 9. The demonstrations can be found from Figure 10 to 18. 

Figure 7: The quantity and proportion of data for each task and its respective subtasks within TIMEBENCH. 

through SFT and RLHF based on GPT-3 (Brown et al., 2020). GPT-4 is an upgraded version of ChatGPT with enhanced reasoning capabilities, making it the most powerful LLM. Unless otherwise stated, ChatGPT refers to _gpt-3.5-turbo-0613_ and GPT-4 refers to _gpt-4-0613_ . 

**Llama2/Vicuna-1.5** (Touvron et al., 2023; Chiang et al., 2023) LLaMA2 is an open foundation model trained on 2T tokens with efficient groupedquery attention (Ainslie et al., 2023). LLaMA2chat is the official aligned model with SFT and RLHF, and Vicuna-1.5 is aligned with SFT only by the community[4] . 

**Baichuan2** (Yang et al., 2023a) is an open foundation model pre-trained on 2.6T tokens, which is competitive with LLaMA2. Baichuan2-chat is the official aligned model with SFT and RLHF. 

**Mistral** (Jiang et al., 2023) is a 7B open foundation model incorporating efficient grouped-query attention (Ainslie et al., 2023) and sliding windows attention (Beltagy et al., 2020). It achieves the strongest performance among models of its size, even surpassing LLaMA2-13B. Mistral-instruct is the officially aligned model with SFT only. 

**ChatGLM3** (Zeng et al., 2023) is an open-source bilingual LLM for Chinese and English, exhibiting competitive performance under 10B. 

**FLAN-T5** (Chung et al., 2022) is an open-source instruction model built on top of T5 (Raffel et al., 2020) through instruction fine-tuning. 

> 4https://lmsys.org/ 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models_images/acl_2024_TimeBench_A_Comprehensive_Evaluation_of_Temporal_Reasoning_Abilities_in_Large_Language_Models.pdf-0017-00.png)


**----- Start of picture text -----**<br>
80 77.0 76.7 Model<br>68.5 GPT-4<br>6040 60.3 50.4 33.9 33.8 37.9 31.6 41.6 31.0 62.6 61.0 43.1 27.8 56.3 49.9 47.5 43.3 62.4 53.3 36.8 47.2 38.3 49.8 42.1 31.4 40.7 57.4 44.1 42.6 34.3 48.0 39.7 37.3 39.0 GPT-3.5LLaMA2LLaMA2LLaMA2Baichuan270137 b 13 bbb<br>20 Baichuan27 b<br>Mistral7 b<br>0 ChatGLM36 b<br>Symbolic Commonsense Event Overall<br>Model<br>80 85.0 84.1 73.6 77.6 73.7 GPT-4GPT-3.5<br>60 61.1 60.9 47.5 46.8 52.2 48.2 59.9 55.4 59.6 57.0 64.5 58.2 66.5 56.7 62.4 57.4 49.4 56.3 49.5 56.0 46.6 59.7 64.4 54.5 47.4 54.2 48.5 55.4 49.3 LLaMA2LLaMA2 [†] 70 [†] 13 bb<br>40 37.6 41.6 LLaMA2 [†] 7 b<br>Baichuan2 [†] 13 b<br>20 Baichuan2 [†] 7 b<br>0 Mistral [†] 7 b<br>Symbolic Commonsense Event Overall ChatGLM3 [†] 6 b<br>**----- End of picture text -----**<br>


Figure 8: Performance comparison between state-of-the-art LLMs. **Up** : GPT-4/3.5 and alignment models under zero-shot setting. **Down** : GPT-4/3.5 and base models under few-shot setting. 

|**Dataset**|**Format**<br>**#**<br>**Challenges**|
|---|---|
|**Symbolic**||
|||
|TimeX Arith<br>TimeX NLI|FRC<br>4,000<br>TimeX Arithmetic<br>NLI<br>6,965<br>TimeX Causality|
|-_Order_|-<br>2,213<br>order|
|-_Duration_|-<br>2,332<br>duration|
|-_Conversion_|-<br>2,420<br>duration + time unit conversion|
|**Commonsense**||
|||
|MCTACO<br>TimeDial<br>DurationQA<br>SituatedGen|M-S<br>852<br>Temporal Commonsense<br>M-S<br>1,446<br>Temporal Commonsense<br>M-S<br>687<br>Event Duration<br>CTG<br>115<br>Temporal Commonsense|
|**Event**||
|||
|TimeQA|FRC<br>1,000<br>Context-based Reasoning|
|-_Explicit_|-<br>500<br>explicit, event-time reasoning|
|-_Implicit_|-<br>500<br>implicit, event-time reasoning|
|MenatQA|FRC<br>1,599<br>Implicit, Context-based Reasoning|
|-_Order_|-<br>400<br>event-time reasoning|
|-_Scope_|-<br>400<br>event-time reasoning|
|-_Counterfactual_|-<br>400<br>event-time reasoning|
|TempReason|FRC<br>1,876<br>Implicit, Fact-based Reasoning|
|-_l2 (e2t)_|-<br>839<br>event-time reasoning|
|-_l3 (e2e)_|-<br>1,037<br>event-event reasoning|
|TRACIE|NLI<br>500<br>Implicit, Implied Event-Event Reasoning|
|||
|_In total_|19,000|



Table 7: The statistics, task formats and challenges in TIMEBENCH. 

17 

|Method|BLEU-4<br>METEOR<br>ROUGE-L<br>CIDEr<br>MATCH|Overall<br>Norm|
|---|---|---|
||||
|Human|39.9<br>40.4<br>56.3<br>397<br>98.1|274.4<br>100.0|
||||
|GPT-4<br>_+ FS_|8.23<br>31.27<br>28.84<br>38.45<br>**90.41**<br>**28.64**<br>**38.99**<br>**55.69**<br>**298.64**<br>**90.11**|162.59<br>59.25<br>**243.29**<br>**88.66**|
||||
|GPT-3.5<br>_+ FS_|13.38<br>30.12<br>35.91<br>125.41<br>**78.76**<br>**27.24**<br>**33.77**<br>**51.18**<br>**282.75**<br>76.54|170.70<br>62.21<br>**217.01**<br>**79.08**|
||||
|LLaMA270_b_<br>_+ FS_|5.15<br>13.62<br>15.83<br>22.07<br>31.79<br>19.10<br>29.09<br>41.74<br>171.36<br>65.29|68.60<br>25.00<br>172.35<br>62.81|
||||
|LLaMA213_b_<br>_+ FS_|4.66<br>21.43<br>20.80<br>17.72<br>61.62<br>15.15<br>27.49<br>37.55<br>138.13<br>64.94|110.28<br>40.19<br>158.93<br>57.92|
||||
|LLaMA27_b_<br>_+ FS_|2.77<br>13.46<br>14.69<br>14.34<br>34.83<br>6.90<br>15.82<br>21.77<br>52.99<br>33.81|67.18<br>24.48<br>83.60<br>30.47|
||||
|Baichuan213_b_<br>_+ FS_|8.33<br>25.86<br>30.07<br>82.63<br>70.63<br>15.79<br>30.23<br>40.96<br>169.14<br>71.01|143.15<br>52.17<br>174.91<br>63.74|
||||
|Baichuan27_b_<br>_+ FS_|5.17<br>21.99<br>23.73<br>44.80<br>59.85<br>15.06<br>23.45<br>32.29<br>137.94<br>52.04|115.22<br>41.99<br>136.64<br>49.79|
||||
|Vicuna1.513_b_<br>_+ FS_|7.73<br>26.35<br>29.15<br>69.16<br>71.91<br>6.85<br>18.66<br>25.99<br>92.96<br>46.19|142.06<br>51.77<br>106.99<br>38.99|
||||
|Vicuna1.57_b_<br>_+ FS_|6.29<br>24.34<br>26.91<br>46.90<br>68.84<br>20.71<br>30.19<br>45.20<br>203.20<br>67.58|131.07<br>47.77<br>184.00<br>67.05|
||||
|FLAN-T5<br>_+ FS_|16.20<br>24.43<br>29.38<br>95.17<br>56.38<br>12.88<br>30.38<br>36.27<br>92.20<br>76.44|135.91<br>49.53<br>165.19<br>60.20|
||||
|Mistral7_b_<br>_+ FS_|5.82<br>22.89<br>24.19<br>44.03<br>63.74<br>18.96<br>29.02<br>43.15<br>185.61<br>63.24|121.03<br>44.11<br>172.93<br>63.02|
||||
|ChatGLM36_b_<br>6.56<br>21.11<br>21.96<br>41.48<br>53.02<br>106.80<br>38.92<br>_+ FS_<br>10.53<br>24.17<br>33.44<br>124.50<br>56.94<br>137.53<br>50.12|||
|LLaMA2_†_<br>70_b_<br>|**22.34**<br>**33.03**<br>**50.93**<br>**243.31**<br>74.96|**205.59**<br>**74.92**|
|LLaMA2_†_<br>13_b_<br>|17.54<br>29.44<br>45.21<br>200.14<br>65.64|177.84<br>64.81|
|LLaMA2_†_<br>7_b_<br>|17.49<br>28.33<br>45.24<br>202.08<br>59.98|171.25<br>62.41|
|Baichuan2_†_<br>13_b_<br>|17.86<br>29.75<br>44.28<br>198.83<br>66.35|178.12<br>64.91|
|Baichuan2_†_<br>7_b_<br>|15.30<br>27.54<br>41.80<br>171.59<br>62.40|164.20<br>59.84|
|Mistral_†_<br>7_b_<br>|14.54<br>27.39<br>41.72<br>168.89<br>59.42|159.96<br>58.30|
|ChatGLM3_†_<br>6_b_|17.11<br>29.35<br>40.74<br>156.49<br>66.18|169.02<br>61.60|



Table 8: Full results of SituatedGen. Aligned models are under zero-shot setting by default. The top-3 results are **bold** . Methods with † are base models without alignment, under few-shot setting. We consider human performance as 100 points and normalize models’ results accordingly. 

18 

|**Mthd**<br>**Symbolic**<br>**Commonsense**<br>**Event**<br>**Overall**|Sym. Comm. Event Avg.||Human<br>98.0 96.0 92.0<br>100.0<br>64.0<br>80.8<br>75.8 87.1 97.8 97.8<br>100.0<br>89.0<br>93.3<br>87.0<br>91.1<br>82.0<br>85.6<br>84.0<br>87.3<br>76.0<br>79.9<br>96.0<br>97.1<br>94.0<br>95.3<br>82.5<br>96.5<br>91.4<br>89.0<br>91.5|GPT-4<br>78.6 76.0 50.7<br>**98.0**<br>35.0<br>59.2<br>61.2 80.0 72.0 91.1<br>59.3<br>48.9<br>60.6<br>40.4<br>46.5<br>44.4<br>57.0<br>49.0<br>57.0<br>22.0<br>23.1<br>91.0<br>95.3<br>94.0<br>95.0<br>64.8<br>75.8<br>72.4<br>62.4<br>68.3<br>_+ CoT_<br>80.0 76.0 60.0<br>92.0<br>35.0<br>58.1<br>67.0 82.6 65.0 89.3<br>-<br>50.0<br>61.3<br>33.0<br>41.2<br>43.4<br>54.6<br>**53.0**<br>**59.6**<br>20.0<br>22.6<br>**93.0**<br>**97.0**<br>93.0<br>94.5<br>58.0<br>77.0<br>76.7<br>61.1<br>68.5<br>_+ FS_<br>85.3 73.3 53.3<br>100.0<br>**51.0**<br>**64.8**<br>**77.6 88.3 85.0 94.6**<br>**88.6**<br>**59.2**<br>**73.7**<br>40.0<br>51.0<br>**59.6**<br>**72.4**<br>48.0<br>54.8<br>**25.3**<br>**28.7**<br>86.0<br>92.4<br>**94.8**<br>**95.9**<br>62.8<br>78.0<br>**84.1**<br>**66.5**<br>**73.7**<br>_+ FS CoT_<br>**92.0 84.0 64.0**<br>100.0<br>42.0<br>55.1<br>68.0 72.3 79.0 93.4<br>-<br>48.0<br>66.9<br>**44.4**<br>**52.8**<br>48.5<br>65.3<br>44.0<br>52.6<br>22.0<br>25.9<br>91.0<br>96.9<br>93.0<br>94.6<br>**66.4**<br>**85.0**<br>73.6<br>65.2<br>72.1|GPT-3.5<br>45.4 67.6 31.2<br>**97.0**<br>19.2<br>50.5<br>34.1 68.6 39.2 69.1<br>62.3<br>**60.5**<br>**70.8**<br>29.5<br>35.4<br>36.5<br>40.9<br>37.5<br>43.9<br>21.0<br>22.9<br>73.6<br>81.2<br>61.8<br>73.8<br>**57.4**<br>60.3<br>62.6<br>53.3<br>57.4<br>_+ CoT_<br>33.6 64.8 33.6<br>71.0<br>12.4<br>23.2<br>28.1 45.1 34.6 67.0<br>-<br>52.5<br>64.4<br>29.0<br>35.1<br>35.8<br>39.7<br>38.5<br>42.9<br>24.0<br>26.3<br>32.0<br>57.6<br>54.2<br>68.1<br>52.0<br>50.8<br>45.1<br>48.3<br>48.3<br>_+ FS_<br>52.0 68.4 31.6<br>63.6<br>**42.8**<br>**67.7**<br>**43.5 71.2** 47.8 **76.4**<br>**79.1**<br>53.8<br>66.1<br>**37.9**<br>**48.4**<br>37.8<br>**43.2**<br>**43.5**<br>**51.6**<br>16.0<br>17.9<br>77.7<br>84.7<br>**70.0**<br>**78.0**<br>55.0<br>53.9<br>**73.6**<br>55.6<br>**59.7**<br>_+ FS CoT_<br>**51.6 71.8 36.6**<br>84.4<br>20.8<br>41.2<br>21.4 38.1 **48.3** 71.1<br>-<br>56.5<br>68.0<br>37.5<br>47.0<br>**38.1**<br>42.5<br>37.5<br>41.7<br>**33.0**<br>**37.8**<br>**86.2**<br>**89.9**<br>68.0<br>76.6<br>50.2<br>**61.1**<br>50.1<br>**56.7**<br>56.6|LLaMA270_b_<br>44.0 47.0 32.0<br>**78.5**<br>**12.7**<br>59.2<br>**23.0 68.9 10.0** 57.0<br>25.0<br>28.0<br>40.8<br>31.0<br>40.6<br>8.0<br>18.9<br>11.0<br>16.6<br>9.0<br>12.0<br>50.0<br>63.5<br>39.0<br>54.5<br>48.0<br>50.4<br>52.5<br>36.8<br>44.1<br>_+ CoT_<br>30.0 **66.0** 28.0<br>53.5<br>8.0<br>57.3<br>21.0 67.1<br>9.0<br>**58.6**<br>17.0<br>31.4<br>13.0<br>19.5<br>5.0<br>12.2<br>8.0<br>12.7<br>**18.0**<br>20.8<br>12.0<br>37.5<br>20.0<br>40.5<br>51.0<br>44.4<br>61.0<br>28.2<br>39.1<br>_+ FS_<br>49.0 42.0 38.0<br>62.0<br>1.3<br>**61.2**<br>13.0 66.5<br>6.0<br>56.6<br>**62.8**<br>**41.0**<br>**51.1**<br>16.0<br>20.0<br>8.0<br>16.4<br>17.0<br>19.9<br>**18.0**<br>18.7<br>34.0<br>52.2<br>31.0<br>41.1<br>51.0<br>47.8<br>**61.8**<br>33.8<br>44.3<br>_+ FS CoT_<br>54.0 63.0 **40.0**<br>69.5<br>8.0<br>55.2<br>21.5 62.1<br>6.0<br>56.4<br>36.6<br>50.9<br>**34.0**<br>**42.4**<br>**28.0**<br>**38.6**<br>**19.0**<br>**29.3**<br>**18.0**<br>**21.9**<br>**77.0**<br>**83.1**<br>**65.0**<br>**74.7**<br>**57.0**<br>**56.6**<br>57.9<br>**49.7**<br>**53.2**|LLaMA213_b_<br>30.0 49.0 34.0<br>**22.5**<br>4.0<br>38.5<br>8.5<br>40.6 10.0 35.4<br>57.9<br>**46.0**<br>**61.9**<br>21.0<br>30.5<br>28.0<br>46.1<br>**23.0**<br>36.1<br>18.0<br>26.9<br>43.0<br>53.1<br>55.0<br>**69.4**<br>49.0<br>33.9<br>43.1<br>46.6<br>42.6<br>_+ CoT_<br>36.0 50.0 38.0<br>6.0<br>7.3<br>39.2<br>**14.0** 51.7 10.0 36.9<br>-<br>45.0<br>58.7<br>**30.0**<br>**38.9**<br>20.0<br>40.9<br>18.0<br>32.5<br>21.0<br>**33.6**<br>43.0<br>58.0<br>**56.0**<br>68.4<br>47.0<br>32.5<br>42.6<br>**47.3**<br>42.4<br>_+ FS_<br>**43.0 57.0 60.0**<br>20.5<br>9.0<br>46.8<br>8.0<br>**66.6 15.0 62.3**<br>**40.2**<br>24.0<br>34.2<br>17.0<br>18.4<br>11.0<br>25.9<br>5.0<br>14.6<br>**22.0**<br>33.3<br>54.0<br>68.1<br>50.0<br>64.8<br>47.0<br>**45.1**<br>**54.0**<br>38.3<br>43.9<br>_+ FS CoT_<br>37.0 55.0 50.0<br>33.0<br>**12.0**<br>**49.5**<br>11.0 45.6<br>8.0<br>44.5<br>-<br>35.0<br>46.0<br>21.0<br>25.4<br>**34.0**<br>**46.7**<br>23.0<br>**36.5**<br>7.0<br>16.5<br>**72.0**<br>**80.8**<br>54.0<br>66.2<br>**50.0**<br>43.8<br>46.5<br>46.0<br>**45.5**|LLaMA27_b_<br>39.0 53.0 30.0<br>**13.0**<br>2.7<br>39.3<br>4.0<br>41.0<br>1.0<br>6.3<br>24.5<br>**37.0**<br>49.0<br>14.0<br>29.0<br>7.0<br>26.8<br>8.0<br>21.1<br>**9.0**<br>16.0<br>**48.0**<br>**63.9**<br>**32.0**<br>47.9<br>49.0<br>33.8<br>27.8<br>37.8<br>34.3<br>_+ CoT_<br>**44.0** 50.0 33.0<br>5.0<br>2.7<br>35.0<br>4.5<br>40.0<br>1.0<br>1.7<br>-<br>27.0<br>49.9<br>17.0<br>31.6<br>**11.0**<br>**31.4**<br>**10.0**<br>**24.5**<br>7.0<br>17.8<br>44.0<br>56.9<br>**32.0**<br>**48.1**<br>46.0<br>33.0<br>25.6<br>**38.3**<br>34.3<br>_+ FS_<br>**44.0 60.0** 34.0<br>11.0<br>4.0<br>**62.8**<br>8.0<br>64.7<br>8.0<br>40.0<br>**30.5**<br>36.0<br>50.8<br>20.0<br>29.4<br>5.0<br>22.3<br>6.0<br>18.0<br>6.0<br>**17.9**<br>12.0<br>36.3<br>23.0<br>44.3<br>**53.0**<br>**37.3**<br>49.5<br>34.0<br>**38.7**<br>_+ FS CoT_<br>38.0 51.0 **36.0**<br>14.5<br>**11.0**<br>42.8<br>**25.0 65.6 13.0 53.4**<br>**-**<br>36.0<br>**53.5**<br>**21.0**<br>**34.1**<br>1.0<br>13.6<br>3.0<br>11.2<br>5.0<br>14.0<br>22.0<br>46.7<br>21.0<br>42.3<br>51.0<br>34.9<br>**53.9**<br>33.3<br>37.8|Baichuan213_b_<br>41.0 **61.0** 37.0<br>**12.5**<br>4.0<br>52.0<br>18.5 63.4 15.0 57.7<br>52.2<br>45.0<br>55.4<br>29.0<br>34.6<br>31.0<br>48.8<br>**34.0**<br>**44.3**<br>30.0<br>39.5<br>40.0<br>57.4<br>45.0<br>61.4<br>49.0<br>37.9<br>56.3<br>48.8<br>48.0<br>_+ CoT_<br>40.0 57.0 31.0<br>10.0<br>3.3<br>44.6<br>20.0 61.9 13.0 58.1<br>-<br>36.0<br>41.5<br>**36.0**<br>40.9<br>**39.0**<br>**52.0**<br>27.0<br>38.5<br>29.0<br>**43.2**<br>46.0<br>62.8<br>46.0<br>**64.3**<br>**55.0**<br>34.5<br>54.9<br>49.8<br>46.7<br>_+ FS_<br>43.0 59.0 40.0<br>42.5<br>**24.7**<br>**62.1**<br>**27.5 70.2 18.0 58.9**<br>**63.7**<br>**47.0**<br>**60.7**<br>35.0<br>**45.7**<br>37.0<br>51.9<br>31.0<br>41.5<br>19.0<br>31.8<br>**73.0**<br>**81.1**<br>**48.0**<br>59.4<br>48.0<br>46.1<br>**63.7**<br>**52.5**<br>**53.7**<br>_+ FS CoT_<br>**45.0** 54.0 **48.0**<br>47.0<br>10.7<br>44.4<br>27.0 68.8 15.0 55.0<br>-<br>43.0<br>57.8<br>27.0<br>36.7<br>38.0<br>49.8<br>34.0<br>40.7<br>**33.0**<br>43.0<br>72.8<br>80.4<br>43.0<br>60.2<br>44.0<br>**48.5**<br>56.1<br>51.6<br>51.7|Baichuan27_b_<br>35.0 **50.0** 37.0<br>**4.5**<br>4.0<br>47.9<br>10.5 55.3 15.0 54.3<br>42.0<br>26.0<br>41.5<br>20.0<br>34.7<br>20.0<br>35.2<br>19.0<br>31.2<br>6.0<br>20.4<br>22.0<br>43.4<br>29.0<br>47.7<br>**55.0**<br>31.6<br>**49.9**<br>38.6<br>39.7<br>_+ CoT_<br>38.0 43.0 32.0<br>1.0<br>5.3<br>37.9<br>13.0 58.0 15.0 44.2<br>-<br>41.0<br>53.5<br>28.0<br>38.8<br>29.0<br>39.9<br>23.0<br>33.2<br>18.0<br>29.3<br>21.0<br>41.2<br>29.0<br>47.2<br>54.0<br>28.5<br>46.7<br>42.1<br>39.4<br>_+ FS_<br>40.0 **50.0** 36.0<br>20.0<br>**28.7**<br>**59.4**<br>**26.5 66.9 17.0 53.0**<br>**49.8**<br>**45.0**<br>**60.7**<br>**30.0**<br>**42.1**<br>27.0<br>37.8<br>23.0<br>35.7<br>10.0<br>20.4<br>40.0<br>57.4<br>37.0<br>53.0<br>51.0<br>36.5<br>57.3<br>44.8<br>45.8<br>_+ FS CoT_<br>**41.0 50.0** 36.0<br>23.5<br>13.0<br>45.7<br>17.5 58.1<br>7.0<br>39.2<br>-<br>36.0<br>51.2<br>29.0<br>**43.0**<br>**42.0**<br>**52.5**<br>**25.0**<br>**39.3**<br>**20.0**<br>**31.0**<br>**57.0**<br>**70.1**<br>**39.0**<br>**60.2**<br>49.0<br>**37.6**<br>47.7<br>**49.5**<br>**46.0**|Vicuna1.513_b_<br>35.0 50.0 36.0<br>**15.0**<br>8.0<br>39.2<br>21.5 59.1<br>7.0<br>34.2<br>**51.8**<br>43.0<br>60.4<br>29.0<br>37.0<br>38.0<br>46.8<br>22.0<br>37.4<br>17.0<br>23.2<br>14.0<br>42.1<br>13.0<br>43.6<br>46.0<br>34.0<br>**46.1**<br>42.1<br>41.1<br>_+ CoT_<br>42.0 51.0 37.0<br>3.0<br>1.3<br>29.8<br>11.5 50.0<br>7.0<br>33.7<br>-<br>44.0<br>56.9<br>31.0<br>36.4<br>16.0<br>38.2<br>25.0<br>37.7<br>13.0<br>20.4<br>31.0<br>49.0<br>29.0<br>49.1<br>51.0<br>33.3<br>37.8<br>42.3<br>39.0<br>_+ FS_<br>**48.0** 57.0 38.0<br>30.5<br>7.3<br>33.6<br>**27.5 57.0 13.0** 40.3<br>39.0<br>45.0<br>58.3<br>23.0<br>25.9<br>38.0<br>42.6<br>26.0<br>**41.4**<br>18.0<br>20.1<br>51.0<br>61.8<br>28.0<br>42.6<br>**56.0**<br>43.4<br>42.5<br>43.6<br>43.3<br>_+ FS CoT_<br>38.0 **59.0 39.0**<br>39.5<br>**10.7**<br>**37.4**<br>14.0 45.8 12.0 **41.6**<br>**-**<br>**47.0**<br>**59.5**<br>**27.0**<br>**30.7**<br>**39.0**<br>**48.1**<br>**31.0**<br>35.9<br>**26.0**<br>**31.2**<br>**71.0**<br>**77.5**<br>**53.0**<br>**65.5**<br>52.0<br>**43.9**<br>41.6<br>**50.1**<br>**46.7**|Vicuna1.57_b_<br>37.0 **58.0 43.0**<br>**5.0**<br>1.3<br>40.4<br>**9.5**<br>**52.5**<br>6.0<br>32.0<br>47.8<br>**35.0**<br>**47.1**<br>11.0<br>18.5<br>**20.0**<br>**35.7**<br>15.0<br>25.7<br>12.0<br>17.3<br>14.0<br>33.0<br>14.0<br>46.8<br>**54.0**<br>35.8<br>43.2<br>34.8<br>**37.1**<br>_+ CoT_<br>36.0 50.0 36.0<br>1.5<br>1.3<br>39.4<br>8.5<br>49.2<br>9.0<br>36.2<br>-<br>30.0<br>40.9<br>**14.0**<br>**24.6**<br>16.0<br>26.2<br>14.0<br>**28.5**<br>12.0<br>**25.0**<br>9.0<br>27.7<br>7.0<br>40.3<br>**54.0**<br>30.9<br>41.6<br>33.4<br>34.4<br>_+ FS_<br>**43.0** 57.0 37.0<br>8.5<br>**3.3**<br>**44.6**<br>5.5<br>42.1<br>7.0<br>36.8<br>**67.1**<br>24.0<br>31.9<br>12.0<br>14.9<br>16.0<br>21.8<br>**20.0**<br>27.5<br>**17.0**<br>22.2<br>13.0<br>34.3<br>6.0<br>32.2<br>**54.0**<br>**36.4**<br>**47.7**<br>29.9<br>35.9<br>_+ FS CoT_<br>35.0 54.0 35.0<br>8.0<br>2.7<br>37.2<br>10.0 47.5 **10.0 41.3**<br>**-**<br>31.0<br>39.9<br>13.0<br>16.6<br>15.0<br>26.7<br>15.0<br>23.8<br>16.0<br>23.1<br>**55.0**<br>**66.3**<br>**32.0**<br>**48.1**<br>43.0<br>33.0<br>42.0<br>**35.9**<br>36.4|FLANT511_b_<br>53.0 63.0 43.0<br>0.0<br>4.0<br>**52.0**<br>14.0 65.0 13.0 47.7<br>49.5<br>56.0<br>61.7<br>**24.0**<br>26.8<br>31.0<br>33.6<br>**48.0**<br>**52.2**<br>20.0<br>21.8<br>84.0<br>87.9<br>**78.0**<br>83.9<br>64.0<br>39.8<br>**53.6**<br>54.0<br>50.3<br>_+ CoT_<br>**56.0** 66.0 45.0<br>0.0<br>**4.7**<br>49.7<br>14.5 63.4 **13.0** 42.7<br>-<br>**57.6**<br>**64.4**<br>23.9<br>**28.2**<br>**39.0**<br>**41.6**<br>46.0<br>50.2<br>**28.0**<br>**30.6**<br>73.0<br>79.5<br>57.0<br>68.9<br>55.0<br>41.8<br>51.9<br>52.3<br>49.4<br>_+ FS_<br>53.0 65.0 43.0<br>3.5<br>4.0<br>50.2<br>**15.5 65.8**<br>9.0<br>35.0<br>**60.2**<br>55.0<br>64.3<br>23.0<br>25.0<br>31.0<br>33.6<br>47.0<br>50.6<br>21.0<br>22.5<br>82.0<br>87.0<br>**78.0**<br>**84.5**<br>65.0<br>41.1<br>52.8<br>**54.1**<br>**50.5**<br>_+ FS CoT_<br>54.0 **68.0 46.0**<br>3.5<br>4.0<br>50.7<br>13.0 64.0 11.0 **43.7**<br>**-**<br>54.0<br>59.4<br>19.0<br>21.2<br>34.0<br>36.6<br>45.0<br>49.2<br>20.0<br>21.7<br>**89.0**<br>**93.8**<br>72.0<br>79.7<br>**66.0**<br>**42.9**<br>52.8<br>53.5<br>**50.5**|**41.6**<br>47.5<br>30.0<br>37.3<br>36.4<br>35.1<br>31.4<br>33.5<br>40.3<br>**60.7**<br>35.5<br>**43.0**|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||TimeQA<br>MenatQA<br>TempReason<br>TRACIE<br>_E-EM E-F1 H-EM H-F1 S-EM S-F1 O-EM O-F1 C-EM C-F1 L2-EM L2-F1 L3-EM L3-F1_<br>_Acc_|||||||||||||7.0<br>28.2<br>5.0<br>21.4<br>4.0<br>24.3<br>7.0<br>22.3<br>4.0<br>21.7<br>2.0<br>39.6<br>1.0<br>31.6<br>51.0<br>13.0<br>31.3<br>11.0<br>22.4<br>8.0<br>21.1<br>**14.0**<br>**24.9**<br>**12.0**<br>**25.6**<br>5.0<br>34.0<br>4.0<br>31.2<br>**61.0**<br>**24.0**<br>**43.5**<br>**12.0**<br>23.9<br>4.0<br>21.3<br>7.0<br>21.2<br>7.0<br>23.0<br>23.0<br>48.9<br>23.0<br>44.9<br>57.0|
||DurationQA McTACO TimeDial SitGen<br>_EM_<br>_F1_<br>_EM_<br>_F1_<br>_EM_<br>_F1_<br>_Norm_|||||||||||||11.3<br>49.8<br>15.0 58.8<br>6.0<br>23.2<br>58.3<br>**13.3**<br>36.6<br>14.5 49.3<br>8.0<br>19.3<br>-<br>12.0<br>**55.8**<br>**25.5 71.0 13.0 52.9**<br>**63.0**|
||TimeXNLI<br>Date Arith<br>_s1_<br>_s2_<br>_s3_<br>_Acc_|||||||||||||47.0 50.0 **43.0**<br>**26.5**<br>38.0 56.0 35.0<br>16.5<br>**51.0** 57.0 35.0<br>18.0|
||**eo**|||||||||||||Mistral7_b_<br>_+ CoT_<br>_+ FS_|



19 

|**Method**<br>**Symbolic**<br>**Commonsense**<br>**Event**<br>**Overall**<br>TimeXNLI<br>Date Arith<br>DurationQA McTACO TimeDial SitGen<br>TimeQA<br>MenatQA<br>TempReason<br>TRACIE<br>Sym. Comm. Event Avg.<br>_s1_<br>_s2_<br>_s3_<br>_Acc_<br>_EM_<br>_F1_<br>_EM_<br>_F1_<br>_EM_<br>_F1_<br>_Norm_<br>_E-EM E-F1 H-EM H-F1 S-EM S-F1 O-EM O-F1 C-EM C-F1 L2-EM L2-F1 L3-EM L3-F1_<br>_Acc_<br>_+ FS CoT_<br>29.0 **62.0** 32.0<br>28.5<br>9.3<br>46.6<br>14.5 49.2<br>8.0<br>34.4<br>-<br>13.0<br>33.7<br>11.0<br>**25.3**<br>**9.0**<br>**26.4**<br>9.0<br>24.3<br>9.0<br>20.3<br>**68.0**<br>**78.6**<br>**42.0**<br>**57.4**<br>50.0<br>37.9<br>43.4<br>**39.5**<br>39.8||ChatGLM36_b_<br>**38.0** 50.0 34.0<br>**2.0**<br>3.0<br>34.1<br>7.0<br>43.6 14.0 56.7<br>38.9<br>20.0<br>41.2<br>14.0<br>**31.7**<br>25.0<br>33.8<br>17.0<br>26.0<br>24.0<br>32.2<br>42.0<br>57.0<br>30.0<br>54.0<br>50.0<br>**31.0**<br>43.3<br>40.7<br>39.0<br>_+ CoT_<br>27.0 49.0 **37.0**<br>0.0<br>1.0<br>24.8<br>3.0<br>37.1 10.0 44.8<br>-<br>24.0<br>41.7<br>10.0<br>25.4<br>**27.0**<br>34.6<br>22.0<br>28.1<br>35.0<br>41.2<br>28.0<br>44.5<br>27.0<br>52.0<br>48.0<br>28.3<br>35.6<br>39.4<br>35.7<br>_+ FS_<br>37.0 52.0 30.0<br>**0.0**<br>**10.0**<br>**53.0**<br>**9.0**<br>**52.9 19.0 54.7**<br>**50.1**<br>11.0<br>19.6<br>13.0<br>17.0<br>25.0<br>**35.0**<br>**24.0**<br>**30.4**<br>20.0<br>25.6<br>27.0<br>46.5<br>33.0<br>53.3<br>**54.0**<br>29.8<br>**52.7**<br>35.2<br>38.2<br>_+ FS CoT_<br>32.0 **66.0** 31.0<br>0.0<br>4.0<br>34.8<br>8.0<br>43.6 11.0 44.0<br>-<br>**25.0**<br>**43.3**<br>**19.0**<br>27.5<br>23.0<br>30.2<br>19.0<br>25.7<br>**36.0**<br>**43.0**<br>**50.0**<br>**62.8**<br>**34.0**<br>**56.0**<br>48.0<br>32.3<br>40.8<br>**42.1**<br>**39.2**<br>LLaMA2-Base70_b_<br>**55.0**<br>61.0<br>37.0<br>**82.0**<br>**40.0**<br>**67.4**<br>**59.0**<br>**85.3**<br>**63.0**<br>**82.7**<br>**74.9**<br>**57.0**<br>**66.7**<br>**36.0**<br>**48.3**<br>**52.0**<br>**61.4**<br>35.0<br>42.5<br>25.0<br>33.8<br>78.0<br>85.2<br>**80.0**<br>**85.4**<br>61.0<br>58.8<br>**77.6**<br>60.5<br>**64.4**<br>_+ CoT_<br>52.0<br>**73.0**<br>**39.0**<br>79.5<br>32.6<br>62.3<br>47.0<br>79.1<br>25.0<br>61.1<br>55.0<br>64.3<br>34.0<br>43.0<br>50.0<br>57.7<br>**37.0**<br>**45.2**<br>**44.0**<br>**53.1**<br>**82.0**<br>**87.5**<br>76.0<br>81.6<br>**67.0**<br>**60.9**<br>67.5<br>**62.4**<br>63.0<br>LLaMA2-Base13_b_<br>**50.0**<br>54.0<br>30.0<br>29.5<br>19.3<br>53.3<br>**26.5**<br>66.0<br>**20.0**<br>**55.6**<br>**64.8**<br>**48.0**<br>59.3<br>34.0<br>48.6<br>41.0<br>49.6<br>**38.0**<br>43.4<br>34.0<br>37.5<br>68.0<br>**78.7**<br>49.0<br>62.7<br>**58.0**<br>40.9<br>**59.9**<br>54.7<br>52.6<br>_+ CoT_<br>40.0<br>**61.0**<br>**37.0**<br>52.0<br>**25.3**<br>**59.3**<br>26.0<br>**68.8**<br>11.0<br>40.8<br>-<br>46.0<br>**59.4**<br>**37.0**<br>**49.1**<br>**49.0**<br>**58.4**<br>34.0<br>**43.8**<br>**38.0**<br>**44.1**<br>**70.0**<br>78.0<br>**55.0**<br>**68.2**<br>**58.0**<br>**47.5**<br>56.3<br>**57.4**<br>**54.5**<br>LLaMA2-Base7_b_<br>26.0<br>50.0<br>30.0<br>20.0<br>19.3<br>54.5<br>20.0<br>59.6<br>15.0<br>**45.2**<br>**62.4**<br>**44.0**<br>**54.4**<br>**30.0**<br>**45.3**<br>**42.0**<br>49.8<br>**34.0**<br>**41.9**<br>30.0<br>35.8<br>**50.0**<br>**64.0**<br>36.0<br>53.3<br>49.0<br>31.5<br>**55.4**<br>49.2<br>46.3<br>_+ CoT_<br>**37.0**<br>**52.0**<br>**36.0**<br>25.5<br>**21.3**<br>**56.9**<br>**26.5**<br>**67.0**<br>**16.0**<br>41.9<br>-<br>32.0<br>45.6<br>27.0<br>36.1<br>41.0<br>**50.9**<br>30.0<br>38.0<br>**51.0**<br>**57.3**<br>45.0<br>59.7<br>**37.0**<br>**57.7**<br>**50.0**<br>**37.6**<br>55.3<br>**49.4**<br>**47.4**<br>Baichuan2-Base13_b_<br>38.0<br>48.0<br>33.0<br>42.5<br>20.7<br>54.8<br>**42.5**<br>**73.0**<br>11.0<br>**45.7**<br>**64.9**<br>**50.0**<br>**59.4**<br>**40.0**<br>**54.2**<br>**42.0**<br>**52.7**<br>31.0<br>38.0<br>13.0<br>21.4<br>68.0<br>77.3<br>**50.0**<br>63.5<br>54.0<br>40.4<br>**59.6**<br>52.6<br>51.3<br>_+ CoT_<br>**50.0**<br>**56.0**<br>**34.0**<br>47.0<br>**29.3**<br>**62.0**<br>22.5<br>69.3<br>**12.0**<br>43.8<br>-<br>46.0<br>58.2<br>39.0<br>49.6<br>39.0<br>49.8<br>**34.0**<br>**40.1**<br>**37.0**<br>**45.6**<br>**73.0**<br>**81.3**<br>46.0<br>**65.6**<br>**60.0**<br>**46.8**<br>58.4<br>**56.3**<br>**54.2**<br>Baichuan2-Base7_b_<br>27.0<br>**66.0**<br>**41.0**<br>32.5<br>**28.0**<br>**59.8**<br>**34.5**<br>69.4<br>5.0<br>34.3<br>**59.8**<br>40.0<br>**53.8**<br>**35.0**<br>**50.2**<br>**41.0**<br>**49.6**<br>**33.0**<br>**38.5**<br>18.0<br>22.9<br>**49.0**<br>**65.9**<br>34.0<br>51.0<br>**55.0**<br>**41.6**<br>55.8<br>48.4<br>**48.5**<br>_+ CoT_<br>**30.0**<br>56.0<br>34.0<br>34.0<br>23.3<br>57.0<br>33.0<br>**69.5**<br>**12.0**<br>**44.5**<br>**-**<br>**41.0**<br>51.2<br>31.0<br>40.7<br>38.0<br>46.4<br>26.0<br>32.6<br>**41.7**<br>**46.3**<br>46.0<br>61.5<br>**43.8**<br>**64.1**<br>53.0<br>38.5<br>**57.0**<br>**49.5**<br>48.1<br>Mistral-Base7_b_<br>48.0<br>53.0<br>**38.0**<br>41.0<br>**34.0**<br>**61.8**<br>**42.5**<br>**76.2**<br>**35.0**<br>**61.8**<br>**58.3**<br>43.0<br>55.9<br>**30.0**<br>45.3<br>37.0<br>49.4<br>**38.0**<br>47.8<br>**37.0**<br>**45.5**<br>**68.0**<br>**76.7**<br>**64.0**<br>**74.8**<br>53.0<br>45.0<br>**64.5**<br>**56.1**<br>**55.4**<br>_+ CoT_<br>**57.0**<br>**63.0**<br>35.0<br>54.0<br>30.0<br>**61.8**<br>42.0<br>45.7<br>29.0<br>57.3<br>-<br>**51.0**<br>**60.4**<br>**30.0**<br>**46.2**<br>**48.0**<br>**57.2**<br>37.0<br>**47.9**<br>24.0<br>33.2<br>60.0<br>65.9<br>58.0<br>67.9<br>**57.0**<br>**52.3**<br>54.9<br>54.5<br>54.0<br>ChatGLM3-Base6_b_<br>**48.0**<br>**70.0**<br>**32.0**<br>35.0<br>**3.3**<br>**51.8**<br>**13.5**<br>**62.6**<br>**11.0**<br>55.0<br>**61.6**<br>**50.0**<br>**57.2**<br>**24.0**<br>**26.3**<br>**30.0**<br>**35.4**<br>**38.0**<br>**41.5**<br>**22.0**<br>**22.5**<br>67.0<br>**76.4**<br>35.0<br>**55.9**<br>**58.0**<br>46.3<br>57.8<br>**46.7**<br>**49.3**<br>_+ CoT_<br>**47.0**<br>**68.0**<br>**32.0**<br>46.0<br>**8.7**<br>**53.9**<br>**15.5**<br>**64.3**<br>**13.0**<br>56.5<br>45.0<br>52.5<br>23.0<br>24.5<br>30.0<br>**35.0**<br>37.0<br>40.2<br>**22.0**<br>**22.5**<br>**72.0**<br>**79.4**<br>**42.0**<br>**60.3**<br>54.0<br>**48.3**<br>**58.2**<br>46.1<br>49.1<br>Table 9: Full results of TimeBench. Aligned models are under zero-shot setting by default. Methods with † are base models without alignment, under few-shot setting, thus<br>incomparable with other methods. We consider human performance as 100 points and normalize models’ results accordingly.|
|---|---|---|
||**Method**<br>TimeXNLI<br>Date Arith<br>DurationQA McTACO TimeDial SitGen<br>TimeQA<br>MenatQA<br>TempReason<br>TRACIE<br>Sym. Comm. Event Avg.<br>_s1_<br>_s2_<br>_s3_<br>_Acc_<br>_EM_<br>_F1_<br>_EM_<br>_F1_<br>_EM_<br>_F1_<br>_Norm_<br>_E-EM E-F1 H-EM H-F1 S-EM S-F1 O-EM O-F1 C-EM C-F1 L2-EM L2-F1 L3-EM L3-F1_<br>_Acc_<br>_+ FS CoT_<br>29.0 **62.0** 32.0<br>28.5<br>9.3<br>46.6<br>14.5 49.2<br>8.0<br>34.4<br>-<br>13.0<br>33.7<br>11.0<br>**25.3**<br>**9.0**<br>**26.4**<br>9.0<br>24.3<br>9.0<br>20.3<br>**68.0**<br>**78.6**<br>**42.0**<br>**57.4**<br>50.0<br>37.9<br>43.4<br>**39.5**<br>39.8|**31.0**<br>43.3<br>40.7<br>39.0<br>28.3<br>35.6<br>39.4<br>35.7<br>29.8<br>**52.7**<br>35.2<br>38.2<br>32.3<br>40.8<br>**42.1**<br>**39.2**|
|||20.0<br>41.2<br>14.0<br>**31.7**<br>25.0<br>33.8<br>17.0<br>26.0<br>24.0<br>32.2<br>42.0<br>57.0<br>30.0<br>54.0<br>50.0<br>24.0<br>41.7<br>10.0<br>25.4<br>**27.0**<br>34.6<br>22.0<br>28.1<br>35.0<br>41.2<br>28.0<br>44.5<br>27.0<br>52.0<br>48.0<br>11.0<br>19.6<br>13.0<br>17.0<br>25.0<br>**35.0**<br>**24.0**<br>**30.4**<br>20.0<br>25.6<br>27.0<br>46.5<br>33.0<br>53.3<br>**54.0**<br>**25.0**<br>**43.3**<br>**19.0**<br>27.5<br>23.0<br>30.2<br>19.0<br>25.7<br>**36.0**<br>**43.0**<br>**50.0**<br>**62.8**<br>**34.0**<br>**56.0**<br>48.0|
|||3.0<br>34.1<br>7.0<br>43.6 14.0 56.7<br>38.9<br>1.0<br>24.8<br>3.0<br>37.1 10.0 44.8<br>-<br>**10.0**<br>**53.0**<br>**9.0**<br>**52.9 19.0 54.7**<br>**50.1**<br>4.0<br>34.8<br>8.0<br>43.6 11.0 44.0<br>-|
|||**38.0** 50.0 34.0<br>**2.0**<br>27.0 49.0 **37.0**<br>0.0<br>37.0 52.0 30.0<br>**0.0**<br>32.0 **66.0** 31.0<br>0.0|
|||ChatGLM36_b_<br>_+ CoT_<br>_+ FS_<br>_+ FS CoT_|



20 

## **DURATIONQA, MCTACO** 

Answer the following question, select all the possible correct options, and each question has at least one correct option. Context: {} Question: {} Options: {} Answer: **TIMEDIAL** There is a two-person dialogue with several options. Choose all appropriate options to substitute the <mask> in the dialogue, and each question has at least one correct option. Dialogue: {} Options: {} Answer: **TRACIE** Read the following story and hypothesis, determine whether the hypothesis can be inferred from the story. You need to understand the implicit temporal relationships between events to make judgments. Story: {} Hypothesis: {} Options: A. Entailment B. Contradiction Answer: **SITUATEDGEN** Generate a pair of contrastive sentences with the given set of keywords. Keywords: {} **DATE ARITHMETIC** Question: {}? Answer: **TIMEQA** I will give you a question with context. You need to answer my question based on the context. If you can infer the answer from the context, then output your answer. Otherwise, if there is no answer, output [unanswerable]. Context: {} Question: {} Answer: **TEMPREASON** I will give you a question with context. You need to answer my question based on the context. Context: {} Question: {} Answer: **MENATQA** Get answers for the question based on the contxt, where answers derived from substrings in the context or categorized as [unanswerable]. Context: {} Question: {} Answer: **TIMEX-NLI** Read the following statements about time and determine if the hypothesis can be inferred from the premise. Premise: {} Hypothesis: {} Options: A. Entailment B. Contradiction C. Neutral Answer: 

Figure 9: Zeroshot instructions and input formats. 

21 

## **CoT Demonstration of TIMEX-NLI (3-shot, order)** 

Answer the following question, select all the possible correct options, and each question has at least one correct option. 

**Premise** : On Wednesday, they got married. **Hypothesis** : Before Friday, they got married. **Options** : A. Entailment B. Contradiction C. Neutral 

**Answer** : Wednesday is before Friday. As a result, we can infer that if something happens on Wednesday, it definitely happens before Friday. Therefore, the answer is A. Entailment. 

**Premise** : We went to Disneyland on Monday. **Hypothesis** : We went to Disneyland after Wednesday. **Options** : A. Entailment B. Contradiction C. Neutral **Answer** : Monday is before Wednesday. As a result, We can infer that if something happens on Monday, it definitely can not happen after Wednesday. Therefore, the answer is B. Contradiction. 

**Premise** : The failing company issued major layoffs after Tuesday. **Hypothesis** : The failing company issued major layoffs after Thursday. **Options** : A. Entailment B. Contradiction C. Neutral **Answer** : Tuesday is before Thursday. If something happened after Tuesday, we cannot be certain whether it occurred after Thursday. Therefore, the answer is C. Neutral. 

Figure 10: Chain-of-Thought demonstrations of TimeX-NLI (s1-order). 

## **CoT Demonstration of DATE ARITHMETIC (4-shot)** 

**Question** : What is the time 4 year and 1 month after Apr, 2000? 

**Answer** : First, 4 years after 2000 is 2004. Next, 1 month after April is May. Therefore, 4 year and 1 month after Apr, 2000 is May, 2004. 

**Question** : What is the time 3 year and 4 month before Jun, 1840? **Answer** : First, subtracting 3 years from 1840 gives 1837. Next, subtracting 4 months from June gives February. Therefore, 3 year and 4 month before Jun, 1840 is Feb, 1837. 

**Question** : What is the time 7 year and 11 month after Feb, 1819? **Answer** : First, 7 years after 1819 is 1826. Next, 11 months after February is January of the next year. Therefore, 7 years and 11 months after Feb, 1819 is Jan, 1827. 

**Question** : What is the time 6 year and 9 month before Jan, 1234? **Answer** : First, subtracting 6 years from 1234 gives 1228. Next, subtracting 9 months from January gives April of the previous year. Therefore, 6 year and 9 month before Jan, 1234 is Apr, 1227. 

Figure 11: Chain-of-Thought demonstrations of Date Arithmetic. 

## **CoT Demonstration of TRACIE (4-shot)** 

Read the following story and hypothesis, determine whether the hypothesis can be inferred from the story. You need to understand the implicit temporal relationships between events to make judgments 

...... 

**Story** : Joe was a police officer. Joe was patrolling the streets of the city in his cruiser. Suddenly,[¨] Joe was alerted to a crime happening near him by dispatch.Joe[¨] responded to the scene and found a bank robber fleeing on foot. Joe arrested the criminal and was promoted. 

Hypothesis: Joe put on his police uniform. starts after Joe arrest the criminal **Options** : A. Entailment B. Contradiction 

**Answer** : From the story we know Joe was patrolling. In the work state, Joe has already put on the police uniform. So we can infer that Joe put on his police uniform before arresting the criminal. This conflicts with hypothesis. Therefore, the answer is B. Contradiction. 

Figure 12: Chain-of-Thought demonstrations of TRACIE. 

22 

## **CoT Demonstration of DURATIONQA (4-shot)** 

Answer the following question, select all the possible correct options, and each question has at least one correct option. 

...... 

**Context** : actually i have an project on it so please give me as much as you have information about migratory birds in punjab 

**Question** : How long did it take for them to have information about migratory birds in punjab? **Options** : A. several months B. 12 weeks C. a few minutes D. almost instantly 

**Answer** : This is a conversation scenario. In the conversation, providing relevant information about migratory birds in punjab to him is in real-time and takes very little time. Therefore, the answer is C. a few minutes, D. almost instantly. 

**Context** : Hope she stops laying eggs because she will get really skinny ! **Question** : How long did it take for her to lay eggs? **Options** : A. 1 week B. 22 hours C. 2 years D. 4 years 

**Answer** : According to commonsense knowledge, the time it takes for birds to lay eggs typically varies from one day to several days. Therefore, the answer is A. 1 week, B. 22 hours. 

Figure 13: Chain-of-Thought demonstrations of DurationQA. 

## **CoT Demonstration of MCTACO (4-shot)** 

Answer the following question, select all the possible correct options, and each question has at least one correct option. 

...... 

**Context** : She ordered the tastiest kind of each vegetable and the prettiest kind of each flower. **Question** : How often does she order vegetables and flowers? **Options** : A. once a second B. three days a week C. every 10 centuries D. once a week 

**Answer** : According to commonsense knowledge, ordering vegetables and flowers typically happens on a regular basis, usually every few days. Therefore, the answer is B. three days a week, D. once a week. 

**Context** : Wallace, 38, called Gastonia home from the age of 8 until she graduated from Hunter Huss High School in 1983. 

**Question** : When did Wallace wake up for high school? 

**Options** : A. at 6 am B. at 1 am C. 7:00 AM D. at 6 pm 

**Answer** : According to commonsense knowledge, waking up for high school typically happens in the morning, usually between 6 AM and 8 AM. Therefore, the answer is A. at 6 am, C. 7:00 AM. 

Figure 14: Chain-of-Thought demonstrations of MCTACO. 

23 

## **CoT Demonstration of TIMEDIAL (4-shot)** 

There is a two-person dialogue with several options. Choose all appropriate options to substitute the <mask> in the dialogue, and each question has at least one correct option. 

...... 

## **Dialogue** : 

A:What schools have you attended ? 

B: I finished Young Primary School in 1998 , and entered Xi ’ an Middle School that same September . I graduated from there in <MASK> , and that September I entered Wuhan University , where I’m studying now . 

A: How do you think the education you have received will contribute to your work in this company ? B: I think I have a good understanding of fundamentals in the areas your company deals with , and I can go on from here to build up the specific skills and knowledge I need to do my job well . 

A: Your graduation thesis was on Medical Application of Laser , right ? What were your conclusions ? B: Yes . I did some work on that , and I found out some really interesting things about the conductivity of liquid helium . I was sure I had a great discovery until my teacher told me the same discovery already made twenty years ago . I think the most important thing , I learnt though , was the importance of keeping good records . **Options** : A. 1998 B. July of 2004 C. March of 2003 D. twenty years ago 

**Answer** : Based on the dialogue, B entered middle school in Sep 1998. According to commonsense knowledge, it usually takes around 6 years from entering middle school to graduating from high school (and entering university). Adding 6 years to 1998 would be 2004, so the answer should be around the year 2004. Therefore, the answer is B. July of 2004, C. March of 2003. 

Figure 15: Chain-of-Thought demonstrations of TimeDial. 

## **CoT Demonstration of TIMEQA, MENATQA (2-shot, implicit)** 

I will give you a question with context. You need to answer my question based on the context. 

If you can infer the answer from the context, then output your answer. Otherwise, if there is no answer, output [unanswerable] 

...... 

**Context** : Theo-Ben Gurirab Theo-Ben Gurirab ( 23 January 1938 2013[˘] 14 July 2018 ) was a Namibian politician who served in various senior government positions . He served as the second Prime Minister of Namibia from 28 August 2002 to 20 March 2005 , following the demotion and subsequent resignation of Hage Geingob . Previously he was the countrys first Minister of Foreign Affairs from 1990 to 2002 , and was President of the United Nations General Assembly from 1999 to 2000 . He was Speaker of the National Assembly of Namibia from 2005 to 2015 , when he was replaced by Peter Katjavivi . Gurirab ultimately resigned from politics in 2015 . Death . Gurirab died at a Windhoek hospital on 14 July 2018 of natural causes . He is buried at Heroes Acre . **Question** : Theo-Ben Gurirab took which position after Jan 2007? 

**Answer** : Based on the context, we can summarize the following facts: Theo-Ben Gurirab served as second Prime Minister of Namibia from August 2002 to March 2005. Prior to that, he was the countrys first Minister of Foreign Affairs from 1990 to 2002 and and was President of the United Nations General Assembly from 1999 to 2000. From 2005 to 2015, he held the position of Speaker of the National Assembly of Namibia. He resigned from politics in 2015 and passed away in July 2018. According to the aforementioned facts, he took the position of Speaker of the National Assembly of Namibia in January 2007. Therefore, the answer is Speaker of the National Assembly of Namibia. 

Figure 16: Chain-of-Thought demonstrations of TimeQA, MenatQA, implicit reasoning. 

24 

## **CoT Demonstration of TEMPREASON (4-shot, event-time)** 

I will give you a question with context. 

You need to answer my question based on the context. 

## ...... 

**Context (facts)** : Gian Piero Gasperini is the head coach of Atalanta B.C. from Jun, 2016 to Dec, 2022. Edoardo Reja is the head coach of Atalanta B.C. from Mar, 2015 to Jun, 2016. Stefano Colantuono is the head coach of Atalanta B.C. from Jun, 2010 to Mar, 2015. Bortolo Mutti is the head coach of Atalanta B.C. from Jan, 2010 to Jun, 2010. Emiliano Mondonico is the head coach of Atalanta B.C. from Jul, 1987 to Jun, 1990. Marcello Lippi is the head coach of Atalanta B.C. from Jul, 1992 to Jun, 1993. Angelo Gregucci is the head coach of Atalanta B.C. from Jul, 2009 to Sep, 2009. Luigi Delneri is the head coach of Atalanta B.C. from Jul, 2007 to Jun, 2009. Ottavio Bianchi is the head coach of Atalanta B.C. from Jul, 1981 to Jun, 1983. Antonio Conte is the head coach of Atalanta B.C. from Sep, 2009 to Jan, 2010. Nedo Sonetti is the head coach of Atalanta B.C. from Jul, 1983 to Jun, 1987. Valter Bonacina is the head coach of Atalanta B.C. from Jan, 2010 to Jan, 2010. **Question** : Who was the head coach of the team Atalanta B.C. in Feb, 2016? **Answer** : According to the context, Edoardo Reja was the head coach of Atalanta B.C. from Mar, 2015 to Jun, 2016. In Feb 2016, the head coach of the team Atalanta B.C. is Edoardo Reja. Therefore, the answer is Edoardo Reja. 

Figure 17: Chain-of-Thought demonstrations of TempReason, event-time reasoning. 

## **CoT Demonstration of TEMPREASON (4-shot, event-event)** 

I will give you a question with context. 

You need to answer my question based on the context. 

...... 

**Context (facts)** : Nicholas Macpherson holds the position of Member of the House of Lords from Oct, 2016 to Dec, 2022. 

Nicholas Macpherson holds the position of Principal Private Secretary to the Chancellor of the Exchequer from Jan, 1993 to Jan, 1997. 

Nicholas Macpherson holds the position of Permanent Secretary to the Treasury from Aug, 2005 to Jan, 2016. **Question** : Which position did Nicholas Macpherson hold before Member of the House of Lords? **Answer** : According to the context, Nicholas Macpherson holds the position of Permanent Secretary to the Treasury from Aug, 2005 to Jan, 2016. Afterthat, Nicholas Macpherson holds the position of Member of the House of Lords from Oct, 2016 to Dec, 2022. Nicholas Macpherson hold the position of Permanent Secretary to the Treasury before Member of the House of Lords. Therefore, the answer is Permanent Secretary to the Treasury." 

Figure 18: Chain-of-Thought demonstrations of TempReason, event-event reasoning. 

25 

