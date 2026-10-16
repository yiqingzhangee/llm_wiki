# � _**OpenToM**_ **: A Comprehensive Benchmark for Evaluating Theory-of-Mind Reasoning Capabilities of Large Language Models** 

**Hainiu Xu**[1] **Runcong Zhao**[1] **Lixing Zhu**[1] **Jinhua Du**[2] **Yulan He**[1] _[,]_[3] 

1King’s College London 

2Huawei London Research Centre 

3The Alan Turing Institute 

{hainiu.xu, runcong.zhao, lixing.zhu, yulan.he}@kcl.ac.uk {jinhua.du}@huawei.com 

## **Abstract** 

Neural Theory-of-Mind (N-ToM), machine’s ability to understand and keep track of the mental states of others, is pivotal in developing socially intelligent agents. However, prevalent N-ToM benchmarks have several shortcomings, including the presence of ambiguous and artificial narratives, absence of personality traits and preferences, a lack of questions addressing characters’ psychological mental states, and limited diversity in the questions posed. In response to these issues, we construct _OpenToM_ , a new benchmark for assessing N-ToM with (1) longer and clearer narrative stories, (2) characters with explicit personality traits, (3) actions that are triggered by character intentions, and (4) questions designed to challenge LLMs’ capabilities of modeling characters’ mental states of both the physical and psychological world. Using _OpenToM_ , we reveal that state-of-the-art LLMs thrive at modeling certain aspects of mental states in the physical world but fall short when tracking characters’ mental states in the psychological world.[1] 

## **1 Introduction** 

Theory-of-Mind (ToM), the awareness that others perceive the world differently and the capability of keeping track of such differences, is at the core of social interactions (Premack and Woodruff, 1978). Studies in cognitive science have designed numerous false-belief tests to investigate human ToM capabilities (Premack and Woodruff, 1978; Wimmer and Perner, 1983; Onishi and Baillargeon, 2005). One such test is the _Sally-Anne Test_ (Baron-Cohen et al., 1985), in which Anne stealthily moves an object that is initially known to both Sally and Anne. This covert action causes Sally to have a false belief that the object is still in its initial location. Consequently, individuals taking the test are required to reason about _"Where will Sally look for the object?"_ 

> 1Our code and data are publicly available at: https:// seacowx.github.io/projects/opentom/OpenToM.html 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models_images/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models.pdf-0001-12.png)


Figure 1: Illustration of a simplified story from _OpenToM_ and the corresponding first-order ToM questions. This story features two protagonists: _Sam_ ( _observer_ ) and _Amy (mover)_ ; and an entity-of-interest: _rubber duck_ . There are two containers involved: a _basket_ and _Amy’s backpack_ . Each narrative within _OpenToM_ is followed by three types of questions, namely questions regarding the location (Loc) of an entity, questions that involve multi-hop reasoning (MHop), and questions about the characters’ attitude (Att). 

To study Neural Theory-of-Mind (N-ToM)[2] , machines’ capabilities of performing ToM reasoning, researchers have applied human ToM tests such as the _Sally-Anne Test_ to benchmark Large Language Models (LLMs) (Le et al., 2019; Bubeck et al., 

2In this paper, we distinguish Theory-of-Mind studies between human (ToM) and artificial neural networks (N-ToM). 

2023; Kosinski, 2023; Shapira et al., 2023a; Ullman, 2023; Wu et al., 2023b; Zhou et al., 2023a). However, using human ToM tests for evaluating LLMs is problematic because stories in human ToM tests lack certain elements found in real-life scenarios. Specifically, the characters do not have **personality traits** or **preferences** . Additionally, their actions are **not motivated** (e.g. why would Anne want to move the object?). Furthermore, the narratives of many existing N-ToM benchmarks are generated using a template-based approach (Le et al., 2019; Wu et al., 2023b; Zhou et al., 2023a), which results in overly-structured and ambiguous narratives (see Appendix A.1). The structured context makes existing benchmarks susceptible to overfitting, while the ambiguities may lead to an underestimation of a model’s true N-ToM capabilities. 

To this end, we introduce **Open** book-QA dataset for **ToM** ( _OpenToM_ ). Following previous works’ success in generating high-quality data using LLMs (Efrat and Levy, 2020; Perez et al., 2022a,b; Hartvigsen et al., 2022; West et al., 2023), we generate _OpenToM_ stories using a four-stage humanin-the-loop generation pipeline (§2.1). Our pipeline includes (1) endowing characters with **preferences** and **personality traits** , (2) generating **intentions** and **the corresponding enctions** (Riva et al., 2011), (3) constructing story plot and producing narratives using LLMs, and (4) revise and refine stories by human annotators. Based on the _OpenToM_ narratives, we formulate questions that cover characters’ mental states of both **the physical world** (e.g., the location of an object) and **their psychological states** (e.g. character’s attitude towards a particular action). See Figure 1 for examples. 

We evaluate _OpenToM_ dataset on a range of LLMs including Llama2-Chat (Touvron et al., 2023), Mixtral-8x7B-Instruct (Jiang et al., 2024), GPT-3.5-Turbo (OpenAI, 2022), and GPT-4-Turbo (OpenAI, 2023) under a zero-shot setting. We also test two prompting techniques, namely Chain-ofThought (CoT) (Wei et al., 2022) and SimulatedToM (SimToM) (Wilf et al., 2023). Additionally, we fine-tuned a Llama2-Chat-13B model to serve as the fine-tuning baseline. Our results show that, while fine-tuning and advanced prompting techniques improve models’ N-ToM reasoning capabilities, their performance in deducing the psychological states of characters is still far from human performance (Section 3.3). We summarize our contributions as follows: 

1. We construct _OpenToM_ , a N-ToM benchmark with natural narratives, personified characters, motivated actions, and diversified questions that challenge LLMs’ understanding of characters’ perception of both the physical world and the psychological states. 

2. Using _OpenToM_ , we conduct a comprehensive evaluation on representative LLMs. Our result shows a mismatch of LLMs’ capability in deducing characters’ mental states of the physical versus the psychological world. 

3. Our in-depth analysis reveals LLMs’ shortcomings in N-ToM including unfaithfulness in N- ToM reasoning, sensitivity to narrative length and character roles, and lack of understanding of characters’ psychological perception. 

## **2 The** _**OpenToM**_ **Dataset** 

The omissions of characters’ personality, intention, and enaction in existing N-ToM benchmarks makes it difficult to construct questions that inquire **characters’ mental states of the psychological world** . To address this, each of the characters in _OpenToM_ stories is **personified** and **acts with an intention** (AppendixA.2). Recognizing that LLMs are good at utilizing spurious correlations such as lexical overlaps (Shapira et al., 2023a), we take extra effort in mitigating the potential spurious cues in _OpenToM_ stories (§2.5). 

## **2.1** _**OpenToM**_ **Construction** 

A typical _OpenToM_ story consists of two protagonists, an entity-of-interest (referred to as the "entity" henceforth), and several locations and containers. Of the two protagonists, one is assumed as the role of the _mover_ , who carries out actions on the entity, and another is the _observer_ , who may or may not witness these actions (see Figure 1). 

As shown in Figure 2, the data generating process consists of two main stages, namely the _Character Personification Process_ followed by the _Narrative and Question Generation Process_ . We start the anthropomorphism process by assigning a personality trait and personal preference to each character. Specifically, the personality traits are sampled from three candidates (see Appendix A.2, and Algorithm 1) and the preference is randomly chosen from binary options. To mitigate spurious correlation, we create false beliefs on characters’ perception of each other’s personal preferences by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models_images/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models.pdf-0003-00.png)


Figure 2: The data generating process of _OpenToM_ dataset. Using the story in Figure 1 as an example, the features created in the personification process are shown in Part (A), which include character preference (♥), belief of the other character’s preference ( _♣_ ), the perturbed _mover_ ’s preference belief ( _♠_ ), the _mover_ ’s personality trait (⋆), and the _mover_ ’s intention and action (♦). The usage of these information in the _OpenToM_ plot are shown in Part (B) next to the paragraph indicator. See Appendix A.3 for detailed description of the _Human Annotation_ and _Rule-Based Label Generation_ process. 

randomly flipping the preference label (see Section 2.5). Using the sampled personal preferences, personality traits, and a world state initialized from ToMi (Le et al., 2019), we prompt GPT-3.5-Turbo to generate the _mover_ ’s intention and enactions. The enaction results in world state changes, which are used to construct the final world state. We use this information to draft a story plot, refer to as the _OpenToM_ plot. 

A _OpenToM_ plot consists of three paragraphs. The first paragraph illustrate the characters’ personal preferences and their beliefs about each other’s preferences. The second paragraph serves as the prologue, which depicts the initial world state and some preceding events involving the two characters. The last paragraph describes the main event, which includes the _mover_ ’s personality, the _mover_ ’s intention, and their subsequent action. It is worth noting that, in order to reduce ambiguity, we explicitly include information regarding whether the _observer_ perceived the _mover_ ’s action. We carefully designed the plot as well as the narrative generating process so that the _observer_ ’s mental activity is excluded from the final _OpenToM_ narrative while ensuring that the _observer_ ’s perception of the main event is mentioned. 

After generating the _OpenToM_ narratives, we classify the corresponding ToM questions into two categories, those requiring human annotation and those that can be automatically annotated using human-defined labels combined with first-order 

logic (see Appendix A.3). In the final stage of data generation, we conduct a round of quality inspection. Specifically, we examine each narrative to ensure that (1) the answers to the ToM questions are not directly given in the narrative, (2) The narrative content aligns with commonsense knowledge, and (3) there is no significant lexical overlaps between the narrative and the corresponding ToM questions (as discussed in Section 2.5). 

## **2.2** _**OpenToM**_ **Overview** 

Overall, _OpenToM_ contains 696 narratives. We first produce 596 narratives with GPT-3.5-Turbo[3] using the pipeline shown in Figure 2. In addition, we sample 100 existing _OpenToM_ plots and produce extra-long narratives ( _OpenToM_ -L) using GPT-4-Turbo[4] . To elicit the unique N-ToM challenges posted by our _OpenToM_ benchmark, we compare _OpenToM_ with established N-ToM benchmarks in Table 1. See Appendix C for detailed statistics of the _OpenToM_ benchmark. 

## **2.3 Task Formulation** 

We formulate all _OpenToM_ questions as binary or ternary classification tasks (see Figure A3 for 

3We used the GPT-35-1106 checkpoint through Microsoft Azure OpenAI service. All _OpenToM_ narratives are generated in December 2023. We also tested with GPT-4-1106 and obtained narratives of similar quality. Hence we choose GPT-3.5-Turbo for its lower cost. 

4We used the GPT-4-1106 checkpoint through Microsoft Azure OpenAI service. All _OpenToM_ -L narratives are generated in December 2023. 

|�: Social Commonsense<br>�: Physical ToM<br>�: Psychological ToM<br>�: Personifed Character<br>�: Number of Narratives<br>�: Average Token Count<br>�: Structured Narrative<br>�: Unstructured Narrative|�: Social Commonsense<br>�: Physical ToM<br>�: Psychological ToM<br>�: Personifed Character<br>�: Number of Narratives<br>�: Average Token Count<br>�: Structured Narrative<br>�: Unstructured Narrative|�: Social Commonsense<br>�: Physical ToM<br>�: Psychological ToM<br>�: Personifed Character<br>�: Number of Narratives<br>�: Average Token Count<br>�: Structured Narrative<br>�: Unstructured Narrative|
|---|---|---|
||Narrative<br>�<br>�<br>�<br>�|�<br>�|
|ToMi<br>T4D_a_<br>Adv-CSFB<br>Hi-ToMi<br>Big-ToMi<br>FANToM<br>G-DRAGON_b_<br>FauxPas-EAI|�<br>✘<br>✔<br>✘<br>✘<br>�<br>✘<br>✔<br>✘<br>✘<br>�<br>✘<br>✔<br>✘<br>✘<br>�<br>✘<br>✔<br>✘<br>✘<br>�<br>✘<br>✔<br>✘<br>✔<br>�<br>✘<br>✔<br>✘<br>✘<br>PBP_c_<br>✘<br>✘<br>✘<br>✘<br>�<br>✔<br>✔<br>✔<br>✔|999<br>44.6<br>_∼_500<br>_∼_50<br>40<br>70.8<br>1200<br>213.68<br>3000<br>69.9<br>254<br>1020.0<br>_∼_800K<br>_∼_72.5<br>44<br>60.5|
|_OpenToM_<br>_OpenToM_-L|�<br>✔<br>✔<br>✔<br>✔<br>�<br>✔<br>✔<br>✔<br>✔|596<br>194.3<br>100<br>491.6|



(a, b) Not open-sourced. The number of narratives and average tokens are estimated according to Zhou et al. (2023a) and Zhou et al. (2023b). (c) PBP: Play-By-Post game play data of Dungeons&Dragons. See Zhou et al. (2023b) for details. 

Table 1: Comparison of _OpenToM_ benchmark with existing N-ToM datasets. In the header, _Physical ToM_ and _Psychological ToM_ refers to testing ToM capabilities in characters’ mental states of the physical world and the psychological world respectively. 

detailed label space and label distributions). Formally, given a complete narrative _Ncomp_ , a set of answers _A_ , a character _c_ , and a character-centric question _qc_ . A model is to first deduce the information accessible to character _c_ , denoted as _Nc_ , and then answer the question. The process of extracting a character-centric narrative _Nc_ can be made explicit, as in Wilf et al. (2023), or latent, as is common in most ToM evaluations. In general, the _OpenToM_ task can be formulated as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models_images/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models.pdf-0004-04.png)


where 1 _expl_ is an indicator function that returns 1 if the character-centric narrative is explicitly provided and 0 otherwise. 

## **2.4 Question Genres** 

Each of _OpenToM_ stories is accompanied by 23 questions that cover both _first-order_ ToM and _second-order_ ToM. _First-order_ ToM questions, which directly ask about a character’s perception of the world, is illustrated in the bottom of Figure 1. _Second-order_ ToM questions inquire about a character’s belief of another character’s mental state. For instance, a second-order ToM question based on the story in Figure 1 could be " _From Sam’s perspective, does Amy think the rubber duck is in its initial location?_ ". Overall, _OpenToM_ questions can be summarized into the following 3 genres: 

**Location (Loc)** questions are concerned with the characters’ perception of the entity’s location. In 

_OpenToM_ , we create two versions of location questions, Loc _coarse_ and Loc _fine_ . Loc _coarse_ asks about the character’s perception of whether an entity is at its initial location, while Loc _fine_ inquires about the entity’s explicit location (see Figure 1 for an example). By doing so, we wish to mitigate the impact of location granularity (Appendix C) and assess the model’s faithfulness in answering this type of questions (§4.1 and Appendix C). 

**Multi-Hop (MHop)** questions are composed by adding an additional reasoning hop on top of the Loc questions. Specifically, we inquire about changes in the _fullness_ of the containers and the _accessibility_ of the entity (see Figure 1 for an example), all of which demand 3-hop reasoning (illustrated in Appendix B). 

To address the lack of **social commonsense** in previous N-ToM benchmarks (Ma et al., 2023b), we have devised the _accessibility_ questions specifically for testing LLMs’ understanding of social norms. Taking the MHop question in Figure 1 as an example, in attempting to answer this question, a model needs to first reason whether the character knows about the rubber duck’s movement. The need for social commonsense comes in the next reasoning hop. Assuming the model is aware that the rubber duck is in Amy’s backpack, it must grasp the social commonsense that others shall not take things from Amy’s backpack without permission. Therefore, a model with adequate social intelligence shall respond with " _less accessible_ " 

**Attitude (Att)** questions are designed to challenge LLMs’ capability to interpret a character’s psychological mental state. Specifically, LLMs are required to deduce the _observer_ ’s potential attitude towards the _mover_ ’s action (see Figure 1 for an example). As discussed in §2.5, the crux of solving _attitude_ questions is to first identify the information accessible to the _observer_ and then use social commonsense to infer the _attitude_ . In _OpenToM_ , of all the knowledge related to the _observer_ ’s _attitude_ , only the _observer_ ’s own preference towards the entity and the _mover_ ’s action are accessible to the _observer_ (see Figure 3). Therefore, _OpenToM_ stories are carefully crafted so that LLMs may not succeed by leveraging information inaccessible to the _observer_ (§2.5). 

Human’s _attitude_ is subjective and multifaceted (Zhan et al., 2023), we reduce such complexity by maximizing the contrast between the _observer_ ’s 

preference and the _mover_ ’s action. In the story of Figure 1, Amy moves Sam’s favorite rubber duck into her own backpack. The substantial disparity between Sam’s fondness of the rubber duck and Amy’s seemingly selfish act will likely cause Sam to have a negative attitude towards Amy’s action. Our data validation study (§2.6) shows the effectiveness of this approach. 

## **2.5 Mitigating Spurious Correlation** 

We take measures to mitigate spurious correlation in all questions. Fixing the Loc and MHop questions can be done by revising narratives based on keywords. We identify _OpenToM_ narratives that contain phrases which have substantial lexical overlap with the questions or those that provide shortcuts for answering them (Appendix A.4). We manually revise such narratives to reduce the reporting bias, resulting in revisions for 17.8% of the _OpenToM_ narrative drafts. 

To elicit the potential spurious cues in _Attitude_ questions, we define the enaction process as a Bayesian network (Riva et al., 2011; Baker et al., 2011) (Figure 3). Firstly, the intention of the _mover_ ( _Int_ ) originates from their preference ( _Pmov_ ), their personality trait ( _T_ ), and, optionally, the _observer_ ’s preference ( _Pobs_ ). This process is latent for the _observer_ – the only observable variables are their own preference ( _Pobs_ ) and the action ( _Act_ ). Employing the _do_ -calculus notation from Pearl (1995), solving the _attitude_ question is equivalent to solving the following problem 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models_images/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models.pdf-0005-04.png)


where _att_ is an instantiation of the _observer_ ’s potential attitudes, _Attobs_ . Overall, we identify two types of potential spurious cues, (1) model P( _att | Int_ ) or (2) model P( _att | T_ ), as shown in Figure 3. We show that addressing these two spurious correlations concurrently can be achieved by adjusting the _mover_ ’s beliefs regarding the _observer_ ’s preference (see Appendix A.5 for details). 

## **2.6 Dataset Validation** 

To verify the human performance and agreement on the _OpenToM_ dataset, we sampled 100 narratives, each of which contains 5 sampled questions covering all 3 question genres asked for both _first-order_ and _second-order_ ToM (see Figure A4 for a demonstration of the data annotation interface). This set of _OpenToM_ data are annotated 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models_images/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models.pdf-0005-08.png)


Figure 3: A Bayesian Network representation of the dependencies among preference ( _P_ ), personality trait ( _T_ ), intention ( _Int_ ), action ( _Act_ ), and attitude ( _Att_ ). The causal relations are represented by solid arrows. The spurious correlations are represented by dashed arrows. The grey-shaded variables are observable by the _observer_ and the unshaded variables are latent to the _observer_ . 

independently by 3 annotators. The inter-annotator agreement is reflected through the macro-averaged F1 score (Table 2), which is computed as the arithmetic mean of the pairwise agreement scores (see Appendix C for detailed statistics). The agreement scores demonstrate that the _OpenToM_ questions contain minimal subjectivity and align well with the collective judgement of human annotators. 

## **3 Experiments** 

Following the convention of previous N-ToM studies, we focus on evaluating zero-shot performance of LLMs (Shapira et al., 2023a; Kim et al., 2023b; Sclar et al., 2023; Zhou et al., 2023a). 

## **3.1 Baseline Models** 

We evaluate the _OpenToM_ tasks using 6 representative LLMs, namely the Llama2-Chat models (7B, 13B, and 70B) (Touvron et al., 2023), the Mixtral-8x7B-Instruct model (Jiang et al., 2024), and the GPT-3.5-Turbo and GPT-4-Turbo[5] models (OpenAI, 2022, 2023). We also fine-tuned a Llama2-Chat 13B model (Appendix D.3). See Appendix D.1 for detailed description of the models. 

## **3.2 Prompting Techniques** 

In addition to the vanilla prompting, we experiment with two additional prompting techniques, namely Chain-of-Thought (CoT) (Wei et al., 2022) and SimulatedToM (SimTom) (Wilf et al., 2023). CoT prompting is widely used in reasoning tasks. It demands LLMs to explicitly generate its step-by-step 

5We use the 1106 checkpoints of the GPT-3.5-Turbo and GPT-4-Turbo models. The experiments are run between December 2023 and January 2024 using API provided by Microsoft Azure OpenAI Studio https://oai.azure.com/. 

|# Params|**Human**<br>—|**Naive Baseline**<br>Ran.<br>Maj.<br>—<br>—|**Large Language Models**<br>Llama2-Chat<br>Mixtral-Instruct<br>GPT-3.5-Turbo<br>GPT-4-Turbo<br>7B<br>13B<br>70B<br>8x7B<br>—<br>—|**FT.**<br>Llama2<br>13B|
|---|---|---|---|---|
|Loc_c_(F)|0.990|0.491<br>0.416|0_._290_±_0_._045<br>0_._391_±_0_._022<br>0_._413_±_0_._016<br>0_._512_±_0_._044<br>0_._439_±_0_._025<br>**0****_._643****_±_0****_._061**|0.978|
|Loc_c_(S)|0.993|0.467<br>0.381|**0****_._462****_±_0****_._069**<br>0_._355_±_0_._043<br>0_._280_±_0_._028<br>0_._294_±_0_._025<br>0_._323_±_0_._039<br>0_._442_±_0_._044|0.749|
|Loc_f_ (F)|0.990|0.000<br>0.003|0_._404_±_0_._029<br>**0****_._545****_±_0****_._023**<br>0_._534_±_0_._023<br>0_._399_±_0_._015<br>0_._515_±_0_._012<br>0_._507_±_0_._010|0.600|
|Loc_f_ (S)|0.993|0.000<br>0.002|0_._245_±_0_._015<br>**0****_._301****_±_0****_._006**<br>0_._223_±_0_._023<br>0_._211_±_0_._011<br>0_._286_±_0_._006<br>0_._269_±_0_._004|0.495|
|MHop(F)|0.855|0.345<br>0.182|0_._322_±_0_._026<br>0_._301_±_0_._023<br>0_._501_±_0_._026<br>0_._556_±_0_._026<br>0_._468_±_0_._029<br>**0****_._658****_±_0****_._034**|0.936|
|MHop(S)|0.770|0.323<br>0.219|0_._211_±_0_._024<br>0_._229_±_0_._037<br>0_._434_±_0_._048<br>0_._474_±_0_._025<br>0_._334_±_0_._025<br>**0****_._637****_±_0****_._034**|0.784|
|Att|0.862|0.328<br>0.174|0_._240_±_0_._027<br>0_._375_±_0_._031<br>0_._415_±_0_._051<br>0_._476_±_0_._041<br>0_._410_±_0_._021<br>**0****_._544****_±_0****_._060**|0.547|



Table 2: Evaluation results in Macro-averaged F1 scores of the _OpenToM_ dataset. Location subscripts, _c_ and _f_ , represents _coarse_ and _fine_ respectively. The capital _F_ and _S_ in the parenthesis represent _first-order ToM_ and _second-order ToM_ . The naive baselines include a random guess (Ran.) and a majority (Maj.) baseline. The finetuning baseline (FT.) is a Llama2-Chat 13B model finetuned following the configuration in Appendix D.3. 

reasoning process. SimToM prompting is specifically designed to aid N-ToM tasks, which asks LLMs to first generate a character-centric narrative, _Nc_ , and then answer character-specific questions. 

## **3.3 Overall Results** 

As all the _OpenToM_ questions are formulated as binary or ternary classification tasks and considering that the labels are not uniformly distributed (Figure A3), we evaluate model performance using the macro-averaged F1 scores (referred to as F1 scores henceforth). 

To evaluate the consistency of LLMs’ performance, we randomly sample 50 narratives for each round of evaluation and repeat this process for 5 times for each model. We compute the mean and the standard deviation of the F1 scores, which are reported in Table 2 (See Table A8 for more detailed results. See Table A7 for the breakdown of LLMs’ performances on MHop questions). Overall, we see that GPT-4-Turbo outperforms other models on Loc _coarse_ (first-order), MHop, and Att questions by a large margin. However, we are surprised to see that Llama2-Chat-7B performs the best in answering second-order Loc _coarse_ . However, due to the high unfaithful rate shown in later studies (§4.1 and Table A9), achieving the highest score does not necessarily imply that Llama2-Chat-7B is more capable in N-ToM. In addition, it is interesting to see that, while GPT-4-Turbo leads in most question genres by a large margin, its capability of answering the Loc _fine_ questions is not on par with Llama2-Chat-13B, 70B, or GPT-3.5-Turbo. 

Through the fine-tuning model, it becomes evident that the Loc _coarse_ and MHop questions are easier to learn, as their F1 scores improved dramatically. On the other hand, the Loc _fine_ and Att questions pose greater challenges as the F1 score of the 

fine-tuned model only have limited improvement. 

CoT prompting brings significant performance gains to all models on Loc _coarse_ and MHop questions. However, the improvements in answering Att questions are marginal and the performance on Loc _fine_ questions declines. In the case of SimToM prompting, the results for the Mixtral model are mixed. SimToM improves the f1 score of MHop questions, but its performance on other question types is either degraded or negligible. For GPT models, SimToM consistently brings performance gains in Loc _coarse_ questions. However, for other question genres, the effect of SimToM is mixed. 

In terms of the length of the narrative, results on _OpenToM_ -L show that ToM in longer narratives are generally harder to trace. Please see Appendix D.5 for detailed results and analysis. 

## **4 Detailed Result Analysis** 

To further investigate LLMs’ N-ToM capabilities, we conduct in-depth analysis on LLMs’ faithfulness in answering Loc _coarse_ and Loc _fine_ questions (§4.1), performance discrepancy of modeling the mental states of different character roles (§4.2), and lack of capability in modeling characters’ mental state of the psychological world (§4.3). 

## **4.1 Faithfulness in Loc Questions** 

As mentioned in §2.4, we create two types of Loc questions differ in granularity. In principle, Loc _coarse_ serves as a prerequisite for answering Loc _fine_ questions. For instance, if a person believes that the entity is not in its initial location (i.e. Loc _coarse_ ), then they should maintain this belief when deducing its precise location (i.e. Loc _fine_ ). We conduct two experiments to examine LLMs’ 

|Question|Mixtral<br>F1<br>∆F1|GPT-3.5-Turbo<br>F1<br>∆F1|GPT-4-Turbo<br>F1<br>∆F1|HL|
|---|---|---|---|---|
|CoT<br>Loc_c_(_F_)<br>Loc_c_(_S_)<br>Loc_f_(_F_)<br>Loc_f_(_S_)<br>MHop(_F_)<br>MHop(_S_)<br>Att|0.784*<br>+0.272<br>0.539*<br>+0.245<br>0.301*<br>- 0.098<br>0.180*<br>- 0.031<br>0.610*<br>+0.054<br>0.551*<br>+0.077<br>0.519*<br>+0.043|0.587*<br>+0.148<br>0.457*<br>+0.134|**0****_._942***<br>+0.299<br>**0****_._828***<br>+0.386<br>0.450*<br>- 0.057<br>0.187*<br>- 0.082<br>**0****_._835***<br>+0.177<br>**0****_._755***<br>+0.118<br>**0****_._580***<br>+0.036|✔<br>✘<br>✘<br>✘<br>✔<br>✔<br>✘|
|||**0****_._469***<br>- 0.046|||
|||**0****_._240***<br>- 0.046<br>0.547*<br>+0.079<br>0.414*<br>+0.080<br>0.446*<br>+0.036|||
|SimToM<br>Loc_c_(_F_)<br>Loc_c_(_S_)<br>Loc_f_(_F_)<br>Loc_f_(_S_)<br>MHop(_F_)<br>MHop(_S_)<br>Att|0.414*<br>- 0.098<br>0.290<br>- 0.004<br>0.352*<br>- 0.047<br>0.206*<br>- 0.005<br>0.650*<br>+0.094<br>0.514*<br>+0.040<br>0.404*<br>- 0.072|0.635*<br>+0.196<br>0.400*<br>+0.077<br>**0****_._518***<br>+0.003<br>**0****_._261***<br>- 0.025|**0****_._838***<br>+0.195<br>**0****_._685***<br>+0.243<br>0.485*<br>- 0.022<br>0.217*<br>- 0.079<br>**0****_._720***<br>+0.062<br>**0****_._631***<br>- 0.006<br>**0****_._488***<br>- 0.056|✘<br>✘<br>✘<br>✘<br>✘<br>✘<br>✘|
|||0.536*<br>+0.068|||
|||0.350*<br>+0.016<br>0.416<br>+0.006|||



Table 3: Macro F1 score of _OpenToM_ dataset evaluated using CoT and SimToM prompting with relative performance gain , performance degradation , or 

equal performance (∆F1 _<_ 0 _._ 010). "*" indicates statistical significance under the Two-sample T test with a level of significance of _α_ = 0 _._ 05. The score of the best performing model on each task is bolded. HL (human level) indicates whether the performance of the best model is on par with human performance (within a margin of 0.050). 

faithfulness[6] in answering the Loc questions. In the _Joint_ approach, we present LLMs with Loc _coarse_ which is immediately followed by Loc _fine_ in the same session. In the _Separate_ approach, we prompt LLMs with each Loc question individually. 

We consider a model to be _Unfaithful_ if it gives contradictory answers in the (Loc _fine,_ Loc _coarse_ ) pair of questions. To quantify this, we compute the _Unfaithful Rate_ for each model, which is the ratio of unfaithful pairs to the total number of pairs, as shown in Figure 4. 

We see that each model’s unfaithful rate is lower when answering first-order ToM questions. This is likely due to their relative simplicity comparing to the second-order questions. Further, we see that, for the GPT models, the _Joint_ approach yields lower _Unfaithful Rate_ than the _Separate_ approach. This improvement may attribute to having access to the previous answer in the context. For Mixtral model, however, the same trend is only observed for the first-order questions. As delving into the reason behind this trend is beyond the scope of this paper, we leave it as future work. Detailed evaluation results are shown in Appendix D.6. 

> 6We follow the definition of "faithfulness" from Jacovi and Goldberg (2020), which is _"the true reasoning process behind the model’s prediction"_ . We regard the model as unfaithful when its _true reasoning process_ deviate from that of human. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models_images/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models.pdf-0007-07.png)


Figure 4: Faithfulness of LLMs in answering Loc questions. The x-axis displays the evaluation model and the y-axis displays the _Unfaithful Rate_ . 

## **4.2 Performance Gap in Character Roles** 

Previous works discovered that LLMs are more capable of answering questions related to the protagonist (Sap et al., 2022; Shapira et al., 2023a), which is likely due to them receiving more descriptions regarding their mental states (Grosz et al., 1995). In _OpenToM_ , we consciously avoid such a reporting bias (§2.5). However, apart from the bias towards the protagonists, we observe that there exists another performance discrepancy in modeling the mind of characters of different roles. In _OpenToM_ , the roles are _mover_ and _observer_ . 

To demonstrate the performance gap between the _mover_ ’s and the _observer_ ’s perception, we compute difference in F1 scores between the models’ performance on _mover_ -centric questions and _observer_ -centric questions (Table 4). 

For second-order Loc questions, the majority of LLMs perform worse when modeling the _mover_ ’s mental state. This is likely due to the long distance between the description of the _mover_ ’s action and whether the _observer_ witnessed the action (see an examples in Appendix E). Such distant information make it difficult for LLMs to establish a connection. Hence, deducing the _mover_ ’s perception of the _observer_ ’s mental state becomes more challenging. 

For MHop questions, all LLMs perform better when modeling the _mover_ ’s mental states. When answering first-order Mhop questions, models’ burden for deciding whether the _mover_ observed their own action is alleviated. In the case of second-order MHop questions, the performance discrepancy is likely due to the explicit mention of the _mover_ ’s intention. These intentions often involve the _mover_ ’s perception of the consequences of their actions on the _observer_ , which greatly reduces the complexity of modeling the _mover_ ’s perception of the _observer_ ’s mental state. 

||Llama-13B|Llama-70B|Mixtral|GPT-3.5T|GPT-4T|
|---|---|---|---|---|---|
|Loc_c_(F)<br>Loc_c_(S)<br>Loc_f_ (F)<br>Loc_f_ (S)<br>MHop(F)<br>MHop(S)|+0.169<br>+0.047<br>+0.091<br>- 0.041<br>+0.156<br>+0.029|+0.711|+0.606|+0.686|+0.464<br>+0.129<br>+0.168<br>- 0.076<br>+0.009<br>+0.008|
|||- 0.035|- 0.040|- 0.029||
|||+0.104|+0.073|+0.097||
|||- 0.050<br>+0.250|- 0.132<br>+0.121|- 0.333<br>+0.320||
|||+0.176|+0.120|+0.143||



Table 4: Relative performance gap between the _mover_ and the _observer_ in answering _OpenToM_ questions. 

## **4.3 Social Commonsense and Attitude** 

GPT-4-Turbo outperforms other models on MHop questions by a large margin (Table 2, 3, and A7), demonstrating its capability in reasoning using social commonsense. However, other LLMs’ performance on MHop questions show that they are lacking in this regard. 

As all LLMs performed poorly on Att questions, we additionally tested Self-Ask prompt (Appendix D.2), which asks LLMs to deduce the final answer by explicit proposing and answering series of follow-up questions (Press et al., 2023). While Self-Ask prompting improves the F1 score of LLMs (Table A10), it is still far from human performance, demonstrating LLMs’ lack of N-ToM capabilities in perceiving characters’ psychological states. By in-depth analysis on the Att answers from Mixtral, and the GPT models, we find two modes or error: low recall in (1) identifying _neutral_ attitude and (2) identifying _positive_ attitude. 

Both of the aforementioned error modes can be attributed to LLMs’ erroneous correlation between the _mover_ ’s personality trait and the _observer_ ’s attitude. In Table 5, we compute the proportion of error cases that are correlated to character’s personality. Specifically, we regard the error and the personality as correlated if a mistaken prediction matches the character’s personality. For instance, across all prompting methods, **more than 95% of the movers in narratives where GPT-4-Turbo mistakenly identify a** _**positive**_ **attitude to be** _**negative**_ **have an** _**inconsiderate**_ **or** _**negativistic**_ **personality** (bottom right column in Table 5). 

As discussed in §2.5, a _considerate mover_ in _OpenToM_ story does not necessarily take actions that are benign to the _observer_ . Therefore, LLMs are doomed to fail when using such a spurious correlation. See Appendix D.7 for detailed results. 

## **5 Related Works** 

**Neural ToM** Some studies argued that LLMs like GPT-4 possess N-ToM capabilities (Bubeck et al., 

|Erroneous Correlation: _Mover_’s Personality_∼Observer_’s Attitude|Erroneous Correlation: _Mover_’s Personality_∼Observer_’s Attitude|Erroneous Correlation: _Mover_’s Personality_∼Observer_’s Attitude|Erroneous Correlation: _Mover_’s Personality_∼Observer_’s Attitude|
|---|---|---|---|
|�: Vanilla Prompt<br>�: CoT Prompt<br>�: SimToM Prompt<br>�: Self-Ask Prompt||||
|�<br>�<br>�<br>�|<br> <br>Results onNeutralAttitude|||
||Mixtral<br>GPT-3.5-Turbo<br>GPT-4-Turbo|||
||Pos<br>Neg|Pos<br>Neg|Pos<br>Neg|
||1.000<br>0.759|1.000<br>0.844|1.000<br>0.796|
||0.944<br>0.909|1.000<br>0.886|0.857<br>0.758|
||1.000<br>0.727|1.000<br>0.771|1.000<br>0.759|
||1.000<br>0.838|1.000<br>0.864|0.938<br>0.818|
|�<br>�<br>�<br>�|Results onPositiveAttitude|||
||Mixtral<br>GPT-3.5-Turbo<br>GPT-4-Turbo|||
||1.000|0.926|1.000|
||1.000|0.904|1.000|
||1.000|0.920|0.957|
||1.000|0.938|1.000|



Table 5: Proportion of mistakenly classified _Neutral_ (top) and _Positive_ (bottom) Att questions that are correlated to the _mover_ ’s personality. For _Neutral_ Att questions, we show the correlation for erroneous _positive_ (Pos) and _negative_ (Neg) predictions separately. For _positive_ Att questions, we show the correlation for erroneous _negative_ predictions. 

2023; Kosinski, 2023). This claim was later rebutted by Shapira et al. (2023a) and Ullman (2023), who both demonstrated that LLMs lack robust N- ToM capabilities. To tackle N-ToM, a line of work used partially observable Markov decision process (Nguyen et al., 2023). Others proposed prompting techniques (Wilf et al., 2023) or neuro-symbolic approaches (Ying et al., 2023; Sclar et al., 2023). We direct readers to Ma et al. (2023b) for a comprehensive survey on N-ToM. 

**ToM Benchmarks** Based on the _Sally-Anne Test_ (Baron-Cohen et al., 1985) and bAbi (Weston et al., 2016), Grant et al. (2017) constructed the ToMbAbi dataset for false belief, which was later improved by Le et al. (2019) into the ToMi dataset. Based on ToMi, researchers proposed T4D (Zhou et al., 2023a), which targets N-ToM for assistant agent, and Hi-ToM (Wu et al., 2023b), which focuses on higher-order N-ToM. Other human ToM tests such as the _Smarties Test_ (Gopnik and Astington, 1988), and the _Faux Pas Test_ (Baron-Cohen et al., 1999) were also used for studying N-ToM, leading to datasets such as ToMChallenges (Ma et al., 2023a), BigToM (Gandhi et al., 2023), AdvCSFB (Shapira et al., 2023a), and FauxPas-EAI (Shapira et al., 2023b). However, existing N-ToM benchmarks are either limited in size, contain artificial narratives, or lack diversity in their questions posed. Jones et al. (2023) constructed EPITOME, which contains human ToM tests that go beyond 

false-belief. Researchers also put efforts in evaluating LLMs’ N-ToM capabilities in dialogues, which resulted in benchmarks such as G-DRAGON (Zhou et al., 2023b), FANToM (Kim et al., 2023c), and SOTOPIA (Zhou et al., 2023c). 

**ToM and Social Commonsense** Sap et al. (2022) showed that LLMs’ lack of understanding of social norms using SocialIQA (Sap et al., 2019). The FauxPas-EAI dataset (Shapira et al., 2023b) was dedicated to evaluating LLMs’ understanding of social commonsense. Efforts were also made to construct knowledge graphs for social commonsense and N-ToM (Wu et al., 2023a). 

## **6 Future Directions** 

**Faithfulness** Our study of LLMs’ performance on Loc _coarse_ and Loc _fine_ reveals that all LLMs lack faithfulness when answering N-ToM questions. We recognize that improving LLMs’ faithfulness is a challenging task in numerous domains (Jacovi and Goldberg, 2020). Here we propose potential remedies specifically targeting N-ToM tasks. Following the findings in §4.1, neuro-symbolic systems can be potentially deployed to enforce faithfulness in reasoning about the characters’ mental state of the physical world. Gao et al. (2023) proposes PAL, which represent reasoning problems with programming language and obtain a deterministic solution using code interpreter. Lyu et al. (2023) combined PAL with CoT and achieved accurate and more faithful reasoning chains. 

**Performance Gap Between Roles** In _OpenToM_ narrative, we propose two roles, namely a _mover_ and an _obsever_ . Our study in §4.2 unveils LLMs’ performance discrepancies in N-ToM between the character roles and analyzes the underlying reasons. In reality, a narrative contain roles well beyond two. To account for the difference in the ToM reasoning process of different roles, a role-aware reasoning framework is needed. Specifically, given an event and a group of characters, the framework needs to first identify the role that each character plays in the event and then conduct ToM reasoning accordingly. 

## **Social Commonsense and Psychological N-ToM** 

Analysis in §4.3 shows that most LLMs are incapable of incorporating social commonsense. Further, we find that LLMs’ performance on Att questions is limited by their inability to determine the information that is accessible to a certain charac- 

ter and using such information to reason about characters’ emotions (Table 5). Hence, an efficient framework for documenting character-centric world state is needed. Further, as discussed in Zhan et al. (2023), people’s attitude in reality is complicated and multifaceted. Therefore, to create a generalizable system capable of emotion deduction, instantiating the emotion deduction process similar to Wu et al. (2023a) is a potential solution. 

**Neural Theory-of-Mind** N-ToM in general is a crucial cognitive capability that a helpful intelligent agent must possess. In the context of human psychology, a lack of ToM capabilities is oftentimes associated with developmental conditions such as Autism Spectrum Disorder (ASD) (Baron-Cohen et al., 1985). Therefore, as LLMs being developed and deployed as assistant agents, it is critical to understand their N-ToM capabilities and develop methods to grant them robust N-ToM reasoning capabilities. LLMs could especially benefit from N-ToM in the following fields: (1) Educational LLM where a helpful assistant agent must be able to accurately model the mental state of the students to be able to provide efficient and precise guidance; (2) Negotiating LLM where understanding the mental states such as the intention, desire, and mood of the opponent is critical when planning negotiation strategies; (3) Mental Health LLM where assistant agent must comprehend the mental state and being empathetic with the patient to be able to provide meaningful help. 

## **7 Conclusion** 

We introduce _OpenToM_ , a comprehensive N- ToM benchmark featuring long narratives with realistic characters and events, and a diverse range of questions that cover both physical and psychological aspects of N-ToM. Our evaluation of LLMs’ N- ToM capabilities on _OpenToM_ reveals that while state-of-the-art LLMs perform well on some N- ToM tasks, they are still far from human-level performance on tasks requiring emotion deduction. 

## **Limitations** 

Limitations of _OpenToM_ are as follows: 

**Limited LLMs** Due to the constraint of computing resources and budget, we only evaluated _OpenToM_ benchmark on a subset of available LLMs. While we believe that the selected LLMs are representative of the current state-of-the-art of their 

categories (Llama2-Chat for open-source LLMs, GPT-3.5-Turbo and GPT-4-Turbo for close-source LLMs, and Mixtral-8x7B-Instruct for Mixture-ofExpert LLMs), we acknowledge that there are other LLMs that could potentially perform better on _OpenToM_ . Further, we only examine the zeroshot performance of LLMs, future studies should test models’ N-ToM capabilities under a few-shot setting. 

**Potential Biases in** _**OpenToM**_ **Narratives** The drafts of _OpenToM_ narratives are composed using LLMs. Although recent studies have shown that LLMs are capable of producing high-quality benchmarks (Efrat and Levy, 2020; Perez et al., 2022a,b; Hartvigsen et al., 2022; West et al., 2023), we acknowledge that the texts generated by LLMs could contain biases and lack lexical diversity. 

**Limited Scope in Character Emotion** In _OpenToM_ benchmark, we construct questions regarding character’s emotion (e.g. attitude). To reduce the subjectivity, we purposely design the stories in a way that the character’s emotion can be directly deduced from an action that happens in a short time frame. In reality, human emotions are often complex, multifaceted, and may depend on multiple events through a prolonged period of time. 

**Limited Narrative Order** All _OpenToM_ narratives are linear narratives that strictly follow chronological order, which alleviate LLMs’ burden to comprehending the order of the events. Future studies can consider constructing _OpenToM_ narratives with non-linear order to further challenge LLMs’ narrative understanding and N-ToM capabilities. 

## **Ethics Statement** 

The drafts of _OpenToM_ narratives are generated using GPT-3.5-Turbo and GPT-4-Turbo. Although we did not identify any harmful or violent content in the _OpenToM_ narratives, it is worth noting that previous studies have observed instances where LLMs produced unexpected results. Therefore, we encourage future studies to also be cautious when employing similar data generating strategies. Further, the _OpenToM_ dataset is annotated by graduate students studying computer science. The similar background of annotators may introduce bias in the annotation process. 

## **Acknowledgements** 

We thank Lin Gui and Yuchen Si for the valuable discussions. This work was supported in part by the UK Engineering and Physical Sciences Research Council (EPSRC) through an iCASE award with Huawei London Research Centre and a Turing AI Fellowship (grant no. EP/V020579/1, EP/V020579/2). 

## **References** 

- Chris Baker, Rebecca Saxe, and Joshua Tenenbaum. 2011. Bayesian theory of mind: Modeling joint belief-desire attribution. In _Proceedings of the annual meeting of the cognitive science society_ , volume 33. 

- Simon Baron-Cohen, Alan M Leslie, and Uta Frith. 1985. Does the autistic child have a “theory of mind”? _Cognition_ , 21(1):37–46. 

- Simon Baron-Cohen, Michelle O’riordan, Valerie Stone, Rosie Jones, and Kate Plaisted. 1999. Recognition of faux pas by normally developing children and children with asperger syndrome or high-functioning autism. _Journal of autism and developmental disorders_ , 29:407–418. 

- Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. 2023. Sparks of artificial general intelligence: Early experiments with gpt-4. _arXiv preprint arXiv:2303.12712_ . 

- Avia Efrat and Omer Levy. 2020. The turking test: Can language models understand instructions? _arXiv preprint arXiv:2010.11982_ . 

- Allen Frances. 1981. Disorders of personality: Dsmiii, axis ii. _American Journal of Psychiatry_ , 138(10):1405–a. 

- Kanishk Gandhi, Jan-Philipp Fränken, Tobias Gerstenberg, and Noah Goodman. 2023. Understanding social reasoning in language models with language models. In _Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track_ . 

- Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Pal: Program-aided language models. In _International Conference on Machine Learning_ , pages 10764–10799. PMLR. 

- Alison Gopnik and Janet W Astington. 1988. Children’s understanding of representational change and its relation to the understanding of false belief and the appearance-reality distinction. _Child development_ , pages 26–37. 

- Erin Grant, Aida Nematzadeh, and Thomas L Griffiths. 2017. How can memory-augmented neural networks pass a false-belief task? In _CogSci_ . 

- Barbara J. Grosz, Aravind K. Joshi, and Scott Weinstein. 1995. Centering: A framework for modeling the local coherence of discourse. _Computational Linguistics_ , 21(2):203–225. 

- Thomas Hartvigsen, Saadia Gabriel, Hamid Palangi, Maarten Sap, Dipankar Ray, and Ece Kamar. 2022. Toxigen: A large-scale machine-generated dataset for adversarial and implicit hate speech detection. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 3309–3326. 

- Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2021. Lora: Low-rank adaptation of large language models. In _International Conference on Learning Representations_ . 

- Alon Jacovi and Yoav Goldberg. 2020. Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness? In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 4198–4205, Online. Association for Computational Linguistics. 

- Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, MarieAnne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2024. Mixtral of experts. 

- Cameron Robert Jones, Sean Trott, and Ben Bergen. 2023. Epitome: Experimental protocol inventory for theory of mind evaluation. In _First Workshop on Theory of Mind in Communicating Agents_ . 

- Hyunwoo Kim, Jack Hessel, Liwei Jiang, Peter West, Ximing Lu, Youngjae Yu, Pei Zhou, Ronan Bras, Malihe Alikhani, Gunhee Kim, Maarten Sap, and Yejin Choi. 2023a. SODA: Million-scale dialogue distillation with social commonsense contextualization. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 12930–12949, Singapore. Association for Computational Linguistics. 

- Hyunwoo Kim, Melanie Sclar, Xuhui Zhou, Ronan Bras, Gunhee Kim, Yejin Choi, and Maarten Sap. 2023b. FANToM: A benchmark for stress-testing machine theory of mind in interactions. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 14397–14413, Singapore. Association for Computational Linguistics. 

- Hyunwoo Kim, Melanie Sclar, Xuhui Zhou, Ronan Bras, Gunhee Kim, Yejin Choi, and Maarten Sap. 2023c. Fantom: A benchmark for stress-testing machine theory of mind in interactions. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 14397–14413. 

- Michal Kosinski. 2023. Theory of mind may have spontaneously emerged in large language models. Technical report, Stanford University, Graduate School of Business. 

- Matthew Le, Y-Lan Boureau, and Maximilian Nickel. 2019. Revisiting the evaluation of theory of mind through question answering. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 5872–5877, Hong Kong, China. Association for Computational Linguistics. 

- Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison-Burch. 2023. Faithful chain-ofthought reasoning. _arXiv preprint arXiv:2301.13379_ . 

- Xiaomeng Ma, Lingyu Gao, and Qihui Xu. 2023a. Tomchallenges: A principle-guided dataset and diverse evaluation tasks for exploring theory of mind. _arXiv preprint arXiv:2305.15068_ . 

- Ziqiao Ma, Jacob Sansom, Run Peng, and Joyce Chai. 2023b. Towards a holistic landscape of situated theory of mind in large language models. In _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pages 1011–1031, Singapore. Association for Computational Linguistics. 

- Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. 2022. Peft: State-of-the-art parameterefficient fine-tuning methods. https://github. com/huggingface/peft. 

- Jason P Mitchell, Jasmin Cloutier, Mahzarin R Banaji, and C Neil Macrae. 2006. Medial prefrontal dissociations during processing of trait diagnostic and nondiagnostic person information. _Social Cognitive and Affective Neuroscience_ , 1(1):49–55. 

- Hiroki Nakayama, Takahiro Kubo, Junya Kamura, Yasufumi Taniguchi, and Xu Liang. 2018. doccano: Text annotation tool for human. Software available from https://github.com/doccano/doccano. 

- Aida Nematzadeh, Kaylee Burns, Erin Grant, Alison Gopnik, and Tom Griffiths. 2018. Evaluating theory of mind in question answering. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ , pages 2392–2400, Brussels, Belgium. Association for Computational Linguistics. 

- Dung Nguyen, Phuoc Nguyen, Hung Le, Kien Do, Svetha Venkatesh, and Truyen Tran. 2023. Memoryaugmented theory of mind network. _arXiv preprint arXiv:2301.06926_ . 

- Kristine H Onishi and Renée Baillargeon. 2005. Do 15month-old infants understand false beliefs? _science_ , 308(5719):255–258. 

- OpenAI. 2022. https://openai.com/blog/chatgpt. Accessed: 2024-01-05. 

- OpenAI. 2023. Gpt-4 technical report. 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. _Advances in Neural Information Processing Systems_ , 35:27730–27744. 

- Judea Pearl. 1995. Causal diagrams for empirical research. _Biometrika_ , 82(4):669–688. 

- Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, and Geoffrey Irving. 2022a. Red teaming language models with language models. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 3419–3448. 

- Ethan Perez, Sam Ringer, Kamile˙ Lukošiut¯ e,˙ Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, et al. 2022b. Discovering language model behaviors with model-written evaluations. _arXiv preprint arXiv:2212.09251_ . 

- David Premack and Guy Woodruff. 1978. Does the chimpanzee have a theory of mind? _Behavioral and brain sciences_ , 1(4):515–526. 

- Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah Smith, and Mike Lewis. 2023. Measuring and narrowing the compositionality gap in language models. In _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pages 5687–5711, Singapore. Association for Computational Linguistics. 

- Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. _Advances in Neural Information Processing Systems_ , 36. 

- Giuseppe Riva, John A Waterworth, Eva L Waterworth, and Fabrizia Mantovani. 2011. From intention to action: The role of presence. _New Ideas in Psychology_ , 29(1):24–37. 

- Maarten Sap, Ronan Le Bras, Daniel Fried, and Yejin Choi. 2022. Neural theory-of-mind? on the limits of social intelligence in large LMs. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 3762–3780, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 

- Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. 2019. Social IQa: Commonsense reasoning about social interactions. In 

_Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 4463– 4473, Hong Kong, China. Association for Computational Linguistics. 

- Philipp Schmid, Omar Sanseviero, Pedro Cuenca, and Lewis Tunstall. 2023. https://huggingface. co/blog/llama2#how-to-prompt-llama-2. Accessed: 2023-11-18. 

- Melanie Sclar, Sachin Kumar, Peter West, Alane Suhr, Yejin Choi, and Yulia Tsvetkov. 2023. Minding language models’ (lack of) theory of mind: A plug-andplay multi-character belief tracker. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 13960–13980, Toronto, Canada. Association for Computational Linguistics. 

- Natalie Shapira, Mosh Levy, Seyed Hossein Alavi, Xuhui Zhou, Yejin Choi, Yoav Goldberg, Maarten Sap, and Vered Shwartz. 2023a. Clever hans or neural theory of mind? stress testing social reasoning in large language models. _ArXiv_ , abs/2305.14763. 

- Natalie Shapira, Guy Zwirn, and Yoav Goldberg. 2023b. How well do large language models perform on faux pas tests? In _Findings of the Association for Computational Linguistics: ACL 2023_ , pages 10438–10451, Toronto, Canada. Association for Computational Linguistics. 

- The Mistral AI Team. 2023. https://huggingface. co/mistralai/Mixtral-8x7B-Instruct-v0.1. Accessed: 2024-01-05. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ . 

- Tomer David Ullman. 2023. Large language models fail on trivial alterations to theory-of-mind tasks. _ArXiv_ , abs/2302.08399. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. _Advances in neural information processing systems_ , 30. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35:24824–24837. 

- Peter West, Ximing Lu, Nouha Dziri, Faeze Brahman, Linjie Li, Jena D Hwang, Liwei Jiang, Jillian Fisher, Abhilasha Ravichander, Khyathi Chandu, et al. 2023. The generative ai paradox:" what it can create, it may not understand". _arXiv preprint arXiv:2311.00059_ . 

- Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M Rush, Bart Van Merriënboer, Armand Joulin, and Tomas Mikolov. 2016. Towards ai-complete question answering: A set of prerequisite toy tasks. In _4th International Conference on Learning Representations, ICLR 2016_ . 

- Alex Wilf, Sihyun Shawn Lee, Paul Pu Liang, and LouisPhilippe Morency. 2023. Think twice: Perspectivetaking improves large language models’ theory-ofmind capabilities. _arXiv preprint arXiv:2311.10227_ . 

- Heinz Wimmer and Josef Perner. 1983. Beliefs about beliefs: Representation and constraining function of wrong beliefs in young children’s understanding of deception. _Cognition_ , 13(1):103–128. 

- Jincenzi Wu, Zhuang Chen, Jiawen Deng, Sahand Sabour, and Minlie Huang. 2023a. Coke: A cognitive knowledge graph for machine theory of mind. _arXiv preprint arXiv:2305.05390_ . 

- Yufan Wu, Yinghui He, Yilin Jia, Rada Mihalcea, Yulong Chen, and Naihao Deng. 2023b. Hi-tom: A benchmark for evaluating higher-order theory of mind reasoning in large language models. In _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pages 10691–10706. 

- Lance Ying, Katherine M Collins, Megan Wei, Cedegao E Zhang, Tan Zhi-Xuan, Adrian Weller, Joshua B Tenenbaum, and Lionel Wong. 2023. The neurosymbolic inverse planning engine (nipe): Modeling probabilistic social inferences from linguistic inputs. _arXiv preprint arXiv:2306.14325_ . 

- Hongli Zhan, Desmond Ong, and Junyi Jessy Li. 2023. Evaluating subjective cognitive appraisals of emotions from large language models. In _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pages 14418–14446, Singapore. Association for Computational Linguistics. 

- Pei Zhou, Aman Madaan, Srividya Pranavi Potharaju, Aditya Gupta, Kevin R McKee, Ari Holtzman, Jay Pujara, Xiang Ren, Swaroop Mishra, Aida Nematzadeh, et al. 2023a. How far are large language models from agents with theory-of-mind? _arXiv preprint arXiv:2310.03051_ . 

- Pei Zhou, Andrew Zhu, Jennifer Hu, Jay Pujara, Xiang Ren, Chris Callison-Burch, Yejin Choi, and Prithviraj Ammanabrolu. 2023b. I cast detect thoughts: Learning to converse and guide with intents and theoryof-mind in dungeons and dragons. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 11136–11155, Toronto, Canada. Association for Computational Linguistics. 

- Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi, Louis-Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, et al. 2023c. Sotopia: Interactive evaluation for social intelligence in language agents. _arXiv preprint arXiv:2310.11667_ . 

## **A** _**OpenToM**_ **Construction** 

## **A.1 Disambiguated Prompt for Narrative Generation** 

In the ToMi dataset, the narrative contains numerous ambiguities. Take the following ToMi narrative as an example: 

**ToMi Narrative Example** 1 Oliver entered the dining room. 2 Logan entered the dining room. 3 Jack entered the dining room. 4 **The stockings is in the drawer.** 5 Jack hates the slippers 6 **Oliver exited the dining room.** 7 **Logan moved the stockings to the crate.** 8 Jack exited the dining room. 9 Logan exited the dining room. 10 Jack entered the hallway. Question: Where will Oliver look for the stockings? 

The key ambiguities are marked with bold text. In line 4, the narrative only states that the entity, _stockings_ , is in the drawer. However, it neglects the characters’ awareness of the entity’s location. Therefore, the above question can be answered with either _the drawer_ in the case where Oliver noticed the stockings, or _unknown_ in the case where Oliver is unaware of the stockings. 

In lines 6-7, Oliver left the dining room, and Logan moved the stockings. However, it is not guaranteed that Logan would lose sight of the dining room once exit. For instance, objects in the dining room could still be visible in the living room if there is no physical barrier separating the spaces. Therefore, knowing that Oliver has left the dining room is insufficient to deduce whether Oliver could observe Logan’s action. 

Further, the information in Line 3, 5, 8, 10 about Jack is completely irrelevant to the progression of the story. Le et al. (2019) added such distracting information to mitigate potential spurious correlations in the original ToM-bAbi dataset (Grant et al., 2017; Nematzadeh et al., 2018). However, such irrelevant information could potentially distract LLMs from performing the ToM task and hence underestimate their ToM capabilities. 

To address such ambiguities, we remove the distracting information and make each character’s per- 

ception explicit in the _OpenToM_ plot. See below for an example of _OpenToM_ story generation prompt (disambiguated information in bold text). **We wish to emphasize that part of the contents of the** _OpenToM_ **plots are derived from the ToMi dataset** (Le et al., 2019). 

## **Prompt Example** 

Plot: 

- Paragraph 1: Mason hates grapes. Samuel hates grapes. 

Paragraph 2: Mason entered the den. Samuel entered the den. **Both Mason and Samuel noticed that the grapes is in the bucket in the den.** Samuel exited the den. 

Paragraph 3: Mason is an inconsiderate person. Mason hates grapes. Therefore, Mason moved the grapes to a neighbor’s house in order to get rid of them. **Samuel did not witness Mason’s action.** 

Write a 200-word, 3-paragraph story according to the plot. Do not depict Samuel’s attitude towards Mason’s action. End the story immediately after the main event. 

## **A.2 Detailed Description of the Character Personification Process** 

**Character Personification** In established N-ToM benchmarks such as ToMi and its variants (Le et al., 2019; Wu et al., 2023b; Zhou et al., 2023a), characters do not possess meaningful personal preferences or personality traits. As a result, their actions lack inherent motivation. In _OpenToM_ , we randomly picked two contrasting personalities, namely " _considerate_ " and " _inconsiderate_ ", from the 24 personality traits defined in (Mitchell et al., 2006). We additionally include a " _negativistic_ " personality to make the story more interesting (Frances, 1981). Below are brief descriptions of each personalities: 

- **Considerate** _mover_ acts to ensure the comfort of the _observer_ . 

- **Inconsiderate** _mover_ acts to make themselves feel comfortable. 

- **Negativistic** _mover_ acts to make the _observer_ uncomfortable. 

**Intention and Enaction** Based on the _mover_ ’s personality and the _observer_ ’s preferences, we gener- 

ate both the character’s intention and their subsequent actions (Appendix A.1). In such a way, the _mover_ ’s action and the movement of the entity are anchored in the _mover_ ’s intention. 

**Plot Construction** Each of the _OpenToM_ narratives is generated by prompting GPT-3.5-Turbo[7] with a story plot[8] (see Appendix A.1 for a prompt example). In _OpenToM_ plot, we sequentially introduce the characters’ preferences towards the _entity_ , a scenario where the two characters meet and how they encounter the _entity_ , and the emergence of the _mover_ ’s intention and the subsequent action towards the _entity_ . 

Following Kim et al. (2023c,a), we first assign names to the _mover_ and the _observer_ by random sampling from the Top-1K most frequently used names in the US SSN database to mitigate potential biases in character naming. Subsequently, for each character, we first randomly sample the personality of the _observer_ ( _trait_o_ ), or the mover ( _trait_m_ ) from the set, _{_ considerate _,_ inconsiderate _,_ negativistic _}_ . Next, we generate the _mover_ ’s preference ( _pref_m_ ), the _observer_ ’s preference ( _pref_o_ ), the _mover_ ’s belief of the _observer_ ’s preference ( _pref_mo_ ), the _observer_ ’s belief of the _mover_ ’s preference ( _pref_om_ ), the _mover_ ’s intention ( _intent_ ), and the _mover_ ’s enaction ( _action_ ) using Algorithm 1. 

We use GPT-3.5-Turbo as our intent generator (INTENTGENERATOR). We customize the prompt for each of the three personality traits. To give examples of the prompt, we again use _Sally (observer)_ and _Anne (mover)_ as the characters and the _rubber duck_ as the entity-of-interest. The intent generation prompts are presented as follows: 

## **Prompt for Intention and Action Generation** 

- {Mover’s Personality} {Mover’s Preference Perception}. Therefore, {Mover’s Initial Intention}. What would be Anne’s 3 most likely action and intention towards the rubber duck? Answer with the following template: 

1. Anne would move the rubber duck to 

7We use the 1106 checkpoint of the GPT-3.5-Turbo model through Microsoft Azure OpenAI service. All _OpenToM_ narratives are generated in December 2023. 

8We also tested with GPT-4-1106, which produces narratives of similar quality. Hence we went for GPT-3.5-Turbo for its lower cost. 

## **Algorithm 1:** Functions for Preference and Intention Generation 

**Function** assignPref( _trait_m, pref_mo_ ) **:** _pref_o ←_ PREFSAMPLER( _observer_ ) _pref_mo ←_ PREFSAMPLER( _mover_ , _observer_ ) _pref_om ←_ PREFSAMPLER( _observer_ , _mover_ ) **if** _trait_m = Negativistic_ **then** _pref_m_ = _¬_ ( _pref_mo_ ) **else** _pref_m_ = PREFSAMPLER( _mover_ ) **return** _pref_o, pref_m, pref_om, pref_mo_ 

**Function** assignIntent( _trait_m, pref_m, pref_o_ ) **: if** _trait_m = Considerate_ **then** latent_pref = pref_om **else if** _trait_m = Inconsiderate_ **then** latent_pref = pref_m **else if** _trait_m = Negativistic_ **then** latent_pref = pref_m intent, action = INTENTGENERATOR(latent_pref) **return** _intent, action_ 

{location} in order to {intention} 

2. Anne would move the rubber duck to 

- {location} in order to {intention} 

3. Anne would move the rubber duck to 

- {location} in order to {intention} 

We fill in the above template based on the _mover_ ’s personality and their belief in the _observer_ ’s perference. Table A1 are a list of descriptions we used to complete the template. 

**Final Intention and Enaction Selection** Notice that for each of the prompts listed above, we specifically ask LLMs to provide 3 candidate intention and enaction pairs. To produce the final intention and its corresponding enaction. We prompt LLMs one more time in the same session to pick the best intention and enaction from the candidates. The prompt we used is as follows: 

## **Intention & Encation Selection** 

Of the potential intentions, which one do you think is true_sentiment? Answer with the original sentence. Do not add any additional words. 

where the true_sentiment is filled according to the _mover_ ’s personality trait: 

- Considerate _→_ " _the most considerate_ " 

- Inconsiderate _→_ " _the most selfish_ " 

- Negativistic (Show off) _→_ " _the most ostentatious_ " 

- Negativistic (Get rid) _→_ " _the most adversarial_ " 

## **A.3 Detailed Description of the Data Annotation Process** 

In _OpenToM_ , the question answers are produced in two ways: human annotation and rule-based generation (Figure 2). For all the Loc _coarse_ questions, MHop questions regarding _accessibility_ , and Att questions, the answers are annotated by graduate students in a prestigious UK university. As the ToM questions in _OpenToM_ are rudimentary for human, we do not provide any specific instruction to the annotators. The content shown in Figure A4 is the complete information that an annotator receives. Therefore, the information that data annotators possess matches the information we provide to LLMs during evaluation. 

Answers to the Loc _fine_ questions are generated according to the human annotation of the corresponding Loc _coarse_ questions and cached container information in _OpenToM_ story plot. For instance, if the annotation to the Loc _coarse_ question is _False_ , then the answer to the Loc _fine_ question is assigned to be the _new container_ (the container that the entity is moved to), which is conveniently cached in the _OpenToM_ story plot. 

Answers to the MHop questions regarding _fullness_ are generated using first order logic based on Loc _coarse_ annotations. Recall that in Loc _coarse_ , we ask the following question 

_From {{character}}’s perspective, is the {{entity}} still in its initial location by the end of the story?_ 

Notice that the Loc _coarse_ question is equivalent to “ _Is {{character}}_ _**aware of** the {{entity}}’s movement_ ?" (in the case of first-order ToM) or “ _Does {{character A}} thinks that {{character B}} is_ _**aware of** the {{entity}}’s movement?_ (in the case of second-order ToM). Knowing the answer to Loc _coarse_ questions is a direct prerequisite for answering _fullness_ questions (see Figure A1). This allows us to conveniently employ the following rules to automatically deduce the answers to the 

||{Mover’s Personality}|{Mover’s Personality}|
|---|---|---|
|Personality|Description||
|Considerate<br>Inconsiderate<br>Negativistic|Anne is a considerate person.<br>Anne is an inconsiderate person.<br>Anne is a negativistic person.||
||{Mover’s Preference Preception}|{Mover’s Initial Intention}|
|Considerate|Although Anne hates rubber duck, she knows that Sally likes them.<br>Although Anne likes rubber duck, she knows that Sally hates them.<br>Anne knows that both Sally and herself hate rubber duck.<br>Anne knows that both Sally and herself like rubber duck.|Anne wants to make it more accessible to Sally.<br>Anne wants to make it less accessible to Anne.<br>Anne wants to make it less accessible.<br>Anne wants to make it more accessible for both of them.|
|Inconsiderate|Anne likes rubber duck.<br>Anne hates rubber duck.|Anne wants to make it more accessible to herself.<br>Anne wants to make it less accessible.|
|Negativistic|Anne thinks that Sally likes rubber duck.<br>Anne thinks that Sally hates rubber duck.|Anne wants to get rid ot the rubber duck.<br>Anne wants to show off the rubber duck.|



Table A1: Description of the _mover_ ’s personality, preference perception, and initial intention. These descriptions are used to fill in the template for intent and action generation. 

## _fullness_ questions: 

## _∀c∀p_ : 

isAware( _c_ ) _∧_ moveTo( _p_ ) _→_ moreFull( _p_ ) isAware( _c_ ) _∧_ takeFrom( _p_ ) _→_ lessFull( _p_ ) _¬_ isAware( _c_ ) _→_ equallyFull( _p_ ) 

where _c_ represents a _character_ and _p_ represents a _container_ . Answer to the isAware( _·_ ) part of the clause is the same as the answer to the Loc _coarse_ questions. Answer to the moveTo( _·_ ) or takeFrom( _·_ ) part of the clause is obtained using cached information from the _OpenToM_ plot (A.1). 

## **A.4 Keyword Revision for Mitigating Spurious Correlation** 

||||
|---|---|---|
|Cue 1: Positive Attitude|||
|gratitude|smile|thoughtful|
|considerate nod|appreciation|kindness|
|gesture|delight|pleased|
|appreciating|||
|Cue 2: Negative Attitude|||
|upset|confusion|bewilderment|
|disappointment|astonishment||
|Cue 3: Direct Description of Accessibility|||
|more accessible|accessible|less accessible|
|inaccessible|out of reach||



Table A2: Keywords that are likely to be directly associated with the answers of the MHop and Att questions. 

## **A.5 Demonstration of Spurious Correlation Mitigation** 

As a running example, consider the scene between Sam and Amy depicted in Figure 1. In this example, Amy mistakenly believe that Sam hates rubber duck. Now we show how the spurious relationships are avoided by adding a false impression on the _mover_ ’s perception of the _observer_ ’s preference. 

## **Spurious Cue 1:** _Is there a causal relation between intention and attitude?_ 

To mitigate the surface-level cues in the _OpenToM_ narrative, we identify the following keywords that are likely to be directly associated with the answers of the MHop and Att questions. We identify the narratives that contain such keywords and manually revise the stories. The keywords are listed as follows: 

The first spurious correlation arises due to the model being incapable of taking the _observer_ ’s perspective and mistakenly interprets the _mover_ ’s intention as observed information. If the _observer_ were to find out the true intention of the _mover_ , then it will undoubtedly be a salient causal factor of the _observer_ ’s attitude. However, deriving the 

true intention is a challenging problem due to subjectivity and the enormous search space involved. Therefore, to mitigate such a spurious correlation, we wish to create scenarios where a good intention leads to negative attitude or vice versa. This can be done by exploiting the _mover_ ’s false belief in the _observer_ ’s preference for a particular entity. 

For instance, in Figure 1(B), Amy mistakenly believes that Sam hates rubber duck. As a considerate person, Amy forms a benign intention, which is to spare Sam from seeing the rubber duck. This intention enacted Amy to move the rubber duck into her own backpack. However, since Sam is unaware of Amy’s true intention, he only observes that Amy has taken that rubber duck away, likely resulting in a _negative_ attitude. Therefore, when there is a false impression in play, a benign intention does not necessarily lead to _positive_ attitude. 

|_OpenToM_ Question Statistics|_OpenToM_ Question Statistics|_OpenToM_ Question Statistics||
|---|---|---|---|
|Question Types|1st-Order|2nd-Order|Total|
|Loc_coarse_|1192|1192|2384|
|Loc_fine_|2384|1192|3576|
|MHop|3576|3576|7152|
|Att|596|–|596|
|Total|7748|5960|13708|
|_OpenToM_-L Question Statistics||||
|Question Types|1st-Order|2nd-Order|Total|
|Loc_coarse_|200|200|400|
|Loc_fine_|400|200|600|
|MHop|600|600|1200|
|Att|100|–|100|
|Total|1300|1000|2300|



Table A3: Statistics of the number of questions in the _OpenToM_ and _OpenToM_ -L dataset. 

## **B Demonstration of Multi-hop Questions** 

## **Spurious Cue 2:** _Is there a causal relation between personality and attitude?_ 

In many instances, the _OpenToM_ narratives explicitly portray the _mover_ ’s personality trait (as seen in Figure 1(B), _"Amy is a considerate person"_ ). To prevent the model from taking a shortcut by deducing the _observer_ ’s attitude solely based on the _mover_ ’s personality, we aim to intervene such a spurious correlation by creating scenarios where a _mover_ with a positive trait leads to the _observer_ having a negative attitude or vice versa. This can be effectively done also by leveraging the _mover_ ’s false belief regarding the _observer_ ’s preference for a particular entity. 

For instance, in Figure 1(B), Amy mistakenly believes that Sam dislikes the rubber duck. As a considerate person, Amy naturally wants to keep the rubber duck out of Sam’s sight. However, due to this false belief, Amy ends up taking away something which Sam actually desires. Therefore, a positive personality could lead to the _observer_ developing a _negative_ attitude. 

An illustration of the reasoning tree employed for answering _Fullness_ questions is shown in Figure A1, while a depiction of the reasoning tree utilized for answering _Accessibility_ questions is shown in Figure A2. It is worth noting that, in order to answer such questions, one must draw upon social commonsense (e.g., taking items from another person’s backpack without permission is not appropriate). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models_images/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models.pdf-0017-09.png)


Figure A1: Illustration of the reasoning tree employed to answer the _Fullness_ questions. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models_images/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models.pdf-0018-00.png)


Figure A2: Illustration of the reasoning tree employed to answer the _Accessibility_ questions. 

## **C The** _**OpenToM**_ **Dataset** 

**Statistics** Table A3 shows the statistics of the question types in the _OpenToM_ dataset, while Figure A3 depicts the label distribution of each question types. 

**Data Annotation Platform** In this study, we use doccano as the data annotation platform (Nakayama et al., 2018). As all the questions are either binary or ternary classification tasks, we use the _Text Classification_ interface. Figure A4 shows the annotation interface for labeling the attitude (Att) questions. The interface for labeling the other question types are the same, except for the label space. 

**Results of Inter-Annotator Agreement** The detailed scores of the inter-annotator agreement are shown in Table A4. The scores are computed as the arithmetic mean of the pairwise agreement scores amongst the three annotators. 

**Ambiguity in Location Granularity** Narratives in _OpenToM_ contain location information of various levels of granularity. For example, in the story shown in Figure 1, there are two levels of location. At a room-level, the rubber duck is moved from the front yard (plot A) to the living room (plot B), while at a container-level, the rubber duck is transferred from a bucket to Amy’s backpack. In the _OpenToM_ stories, granularity can extend beyond two levels for locations (e.g. movements between 

|Question Type|Accuracy Score|F1 Score|
|---|---|---|
|Location (FO)<br>Location (SO)<br>Multihop (FO)<br>Multihop (SO)<br>Attitude|0.993<br>0.993<br>0.873<br>0.927<br>0.870|0.990<br>0.993<br>0.855<br>0.770<br>0.862|



Table A4: Inter-annotator agreement scores. The scores are computed as the arithmetic mean of the pairwise agreement scores amongst the three annotators. 

different houses). Therefore, judging model’s response solely based on explicit location information is difficult. 

In addition, deducing the location of an entity involves multi-hop reasoning. It can be decomposed into _(1) Is the entity in its initial location?_ , and _(2) What is the initial/final location exactly_ ? While the first question is seemingly simpler, it still demands the understanding of another character’s perspective to answer correctly. 

## **D Further Experimental Details** 

## **D.1 Details of the Baseline Models** 

We evaluate the _OpenToM_ tasks using 6 representative LLMs, namely the Llama2-Chat models (7B, 13B, and 70B) (Touvron et al., 2023), the Mixtral-8x7B-Instruct model (Jiang et al., 2024), and the GPT-3.5-Turbo and GPT-4-Turbo. All of the models use decoder-only Transformer architecture (Vaswani et al., 2017). 

**Llama2-Chat** is a Llama2 model optimized using Reinforcement Learning From Human Feedback (RLHF) for dialogue (Ouyang et al., 2022; Touvron et al., 2023). We evaluate _OpenToM_ dataset on the 7B, 13B, and 70B checkpoints. 

**Mixtral-Instruct** is a Sparse Mixture-of-Expert (SMoE) model optimized with Direct Policy Optimization (DPO) for instruction following (Rafailov et al., 2023; Jiang et al., 2024). 

**GPT-3.5-Turbo** and **GPT-4-Turbo** are models of the GPT family, both of which are optimized using RLHF for instruction following (OpenAI, 2022, 2023). 

In addition to zero-shot prompting, we also finetuned Llama2-Chat 13B models using LoRA to serve as a finetuning baseline (Hu et al., 2021; Mangrulkar et al., 2022). See Appendix D.3 for the 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models_images/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models.pdf-0019-00.png)


Figure A3: Answer distribution of the _OpenToM_ ToM questions. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models_images/acl_2024_OpenToM_A_Comprehensive_Benchmark_for_Evaluating_Theory-of-Mind_Reasoning_Capabilities_of_Large_Language_Models.pdf-0020-00.png)


Figure A4: Annotation interface for labeling the attitude (Att) questions. 

configuration of the finetuning procedure. 

## **D.2 Prompt Example for** _**OpenToM**_ **Evaluation** 

We use the OpenAI ChatCompletion prompt format as the base format. The prompt consists of two entries, one is the "system prompt", which contains an overall instruction for LLMs, another is the "user prompt", which contains an instance of _OpenToM_ narrative and an _OpenToM_ question: 

## **OpenAI ChatCompletion Prompt Format** 

## [ 

{‘role’: ‘system’, ‘content’: "You are an expert in modeling other’s mental state."}, {‘role’: ‘user’, ‘content’: ‘Read and comprehend the following short story. Then, answer the question that follows. 

{{ _OpenToM_ Narrative}} 

Question: {{ _OpenToM_ Question}} {{Label Space}} Do not give any explanation.’} ] 

We replace the {{ _OpenToM_ Narrative}} with an instance of the _OpenToM_ narrative, {{ _Open-_ 

_ToM_ Question}} with one of the _OpenToM_ questions, and {{Label Space}} with the corresponding label space. For example, the prompt for the MHop question is as follows: 

## **OpenAI ChatCompletion Prompt Format** 

## [ 

{‘role’: ‘system’, ‘content’: "You are an expert in modeling other’s mental state."}, 

{‘role’: ‘user’, ‘content’: ‘Read and comprehend the following short story. Then, answer the question that follows. 

Mary and Hazel were two friends with different tastes. Mary had a fondness for slippers, while Hazel couldn’t stand them. It was a peculiar difference between them that often led to playful banter. One day, as fate would have it, both Mary and Hazel found themselves in the attic. 

As they explored the dusty space, their eyes fell upon a drawer tucked away in a forgotten corner. Curiosity piqued, they opened it 

together and there, resting neatly inside, were a pair of slippers. Mary’s face lit up with delight, while Hazel simply shrugged. 

Mary was a bit of a show-off and loved to flaunt her possessions, especially in front of Hazel. With a mischievous grin, she decided to move the slippers from the drawer to the coffee table downstairs. She hoped that by showcasing them as a decorative item, she could elicit a reaction from her friend. Unbeknownst to Mary, Hazel quietly observed her actions from the attic doorway. 

And with that, the main event concluded. 

Question: From Mary’s perspective, how does coffee table’s fullness change by the end of the story? Answer with "more full", "equally full", or "less full". Answer the question without any explanation.’} ] 

For Llama2-Chat models, we follow the prompting instruction provided in the official HuggingFace blog post (Schmid et al., 2023) 

## **Llama2-Chat Prompt Format** 

<s>[INST] `〈〈` SYS `〉〉` {System Prompt} `〈〈` /SYS `〉〉` {User Prompt} [/INST] 

## **Mixtral-8x7B-Instruct Prompt Format** 

<s>[INST] {User Prompt} [/INST] 

where <s> is the special start token and other are normal tokens. Mixtral-Instruct is not trained with a system prompt. Therefore, we omit the system prompt in the Mixtral prompt as adviced by the official post (The Mistral AI Team, 2023). 

**Chain-of-Thought Prompting** To implement CoT prompt (Wei et al., 2022), we replace the original instruction in Prompt D.2 with a CoT instruction. The resulting CoT prompt template is shown as follow: 

## **CoT Prompt Template** 

[ {‘role’: ‘system’, ‘content’: "You are an expert in modeling other’s mental state."}, {‘role’: ‘user’, ‘content’: ‘Read and comprehend the following short story. Then, answer the question that follows. 

{{ _OpenToM_ Narrative}} 

Question: {{ _OpenToM_ Question}} {{Label Space}} **Reason step by step before answering. Write the answer in the end.** ’} ] 

**SimulatedToM Prompting** We implement SimToM prompting as per the instruction in Wilf et al. (2023). In the first stage, we prompt LLMs with the following instruction to generated a charactercentric narrative, _Nc_ : 

## **SimToM Prompt Template (Stage 1)** 

[ 

where <s> is the special start token and others are normal tokens. 

For Mixtral-Instruct, we follow the prompt format provided in the official Mixtral-8x7B-Instruct model card on HuggingFace Transformers (The Mistral AI Team, 2023). For the GPT models, we follow the prompting convention provided in the OpenAI’s official API documentation. 

{‘role’: ‘system’, ‘content’: "You are an expert in modeling other’s mental state."}, {‘role’: ‘user’, ‘content’: ‘The following is a sequence of events: 

{{ _OpenToM_ Narrative}} 

Which events does character know about?’} ] 

With the character-centric narrative, _Nc_ , we then prompt LLMs in the same session with _OpenToM_ question using the following template: 

## **SimToM Prompt Template (Stage 2)** 

[ ... {{Stage 1 Prompt and Response}} ... 

{‘role’: ‘user’, ‘content’: {{ _Nc_ }} 

{{ _OpenToM_ Narrative}} 

Question: {{ _OpenToM_ Question}} {{Label Space}} Do not give any explanation.’} ] 

**Self-Ask Prompting** To implement Self-Ask prompt (Press et al., 2023), we use the following prompt template: 

## **Self Prompt Template** 

[ {‘role’: ‘system’, ‘content’: "You are an expert in modeling other’s mental state."}, {‘role’: ‘user’, ‘content’: ‘Read and comprehend the following short story. Then, answer the question that follows. 

{{ _OpenToM_ Narrative}} 

Question: {{ _OpenToM_ Question}} {{Label Space}} **Break the original question into sub-questions. Explicitly state the follow-up questions, and the answers to the follow-up questions. Aggregate the answers to the follow-up questions and write the answer in the end as “Final Answer: [answer]"** ’} ] 

## **D.3 Finetune Configuration** 

To compensate for the unbalanced number of questions in each genre (Table A3), we downsample the majority class and upsample the minority class. The resulting _OpenToM_ training dataset contains 1192 instances for Loc _coarse_ , Loc _fine_ , and MHop questions. Minding the fact that Att questions are harder to learn, we upsample it to 5960 data points to enhance model’s performance. Of all the data points, we use 80% for training and test the finetuned model on the 20% held-out testing set. We use the LoRA implimentation from HuggingFace PEFT (Hu et al., 2021; Mangrulkar et al., 2022) with the training and LoRA configuration shown in Table A5. 

|Training Confguration|Training Confguration|
|---|---|
|Batch Size<br>Gradient Accumulation Steps<br># Epochs<br>Learning Rate<br>Optimizer<br>Learning Rate Scheduler<br>Loss Function|4<br>4<br>3<br>2_×_10_−_5<br>AdamW<br>Linear (Step Size = 1,_γ_ = 0.85)<br>Cross Entropy Loss|
|LoRA Confguration||
|rank (r)<br>_α_<br>Target modules<br>LoRA Dropout|8<br>32<br>q_proj,v_proj<br>0.05|



Table A5: Training and LoRA configuration for finetuning Llama2-Chat-13B on _OpenToM_ dataset. 

## **D.4 Detailed Baseline Results** 

The generated responses from LLMs using advanced prompting techniques such as CoT and SimToM are oftentimes in free form. To obtain the final answer, we employed strict parsing rules to extract answer from free-form responses. Any answer that contains ambiguous response or fails to follow the formatting instruction in the prompt are classified as **corrupted output** . Such results are excluded when computing the accuracy and F1 scores. We provide the **corruption rate** for each model and prompting method. 

All these details are shown in Table A8. For CoT prompting, we do not evaluate Loc _f_ on LlamaChat models due to their incapability of generating reliable reasoning chains (see corruption rate in Table A8). Further, we do not report Llama2-Chat’s performance on Att questions due to their high corruption rate. In addition, the SimulatedToM 

|Question|Llama2-Chat-7B<br>F1<br>∆F1|Llama2-Chat-13B<br>F1<br>∆F1|Llama2-Chat-70B<br>F1<br>∆F1|Mixtral-8x7B<br>F1<br>∆F1|GPT-3.5-Turbo<br>F1<br>∆F1|GPT-4-Turbo<br>F1<br>∆F1|
|---|---|---|---|---|---|---|
|Loc_c_(_F_)<br>Loc_c_(_S_)<br>Loc_f_(_F_)<br>Loc_f_(_S_)<br>MHop(_F_)<br>MHop(_S_)<br>Att|0.212<br>- 0.078<br>0.366<br>- 0.096<br>0.352<br>- 0.052<br>0.323<br>+0.078<br>0.371<br>+0.049<br>0.294<br>+0.083<br>0.225<br>- 0.015|0.381<br>- 0.010|0.420<br>- 0.007|0.476<br>- 0.036<br>0.297<br>+0.003<br>0.336<br>- 0.063<br>0.196<br>- 0.015|0.435<br>+0.004<br>0.415<br>+0.092<br>**0****_._519**<br>+0.004<br>**0****_._277**<br>- 0.009|**0****_._522**<br>- 0.121<br>0.346<br>- 0.096<br>0.492<br>- 0.015<br>0.256<br>- 0.013<br>**0****_._664**<br>+0.006<br>**0****_._565**<br>- 0.072<br>**0****_._580**<br>+0.036|
|||0.419<br>+0.064|0.288<br>+0.008||||
|||0.377<br>- 0.168<br>0.215<br>- 0.086|0.387<br>- 0.147<br>0.187<br>- 0.036||||
|||0.298<br>- 0.003|0.530<br>+0.029|0.601<br>+0.045|0.458<br>- 0.010||
|||0.301<br>+0.072<br>0.331<br>- 0.044|0.476<br>+0.042<br>0.507<br>+0.092|0.488<br>+0.014|0.372<br>+0.038||
|||||0.444<br>- 0.032|0.382<br>- 0.028||



Table A6: Macro F1 score of LLMs evaluated with _OpenToM_ Long Narrative. The relevant performances are shown as relative increase , relative decrease , or approximately equal (∆F1 _<_ 0 _._ 010). 

||MHop-Fullness (F)<br>F1<br>Crp.|MHop-Accessibility(F)<br>F1<br>Crp.|MHop-Fullness (S)<br>F1<br>Crp.|MHop-Accessibility(S)<br>F1<br>Crp.|
|---|---|---|---|---|
|Naive<br>Marjority<br>Random|0_._183<br>–<br>0_._336<br>–|0_._180<br>—<br>0_._354<br>—|0_._245<br>—<br>0_._311<br>—|0_._193<br>—<br>0_._336<br>—|
|Vanilla<br>Llama2-Chat-7B<br>Llama2-Chat-13B<br>Llama2-Chat-70B<br>Mixtral-8x7B<br>GPT-3.5-Turbo<br>GPT-4-Turbo|0_._331_±_0_._042<br>0.0%<br>0_._244_±_0_._038<br>0.0%<br>0_._506_±_0_._034<br>0.0%<br>0_._598_±_0_._050<br>0.0%<br>0_._476_±_0_._035<br>0.0%<br>0_._682_±_0_._030<br>0.4%|0_._307_±_0_._024<br>0.0%<br>0_._295_±_0_._019<br>0.0%<br>0_._506_±_0_._044<br>0.0%<br>0_._509_±_0_._025<br>0.0%<br>0_._474_±_0_._028<br>0.0%<br>0_._633_±_0_._049<br>0.0%|0_._229_±_0_._017<br>0.0%<br>0_._213_±_0_._045<br>0.0%<br>0_._368_±_0_._065<br>0.0%<br>0_._394_±_0_._053<br>0.0%<br>0_._262_±_0_._045<br>0.002<br>0_._557_±_0_._036<br>0.4%|0_._198_±_0_._036<br>0.0%<br>0_._204_±_0_._028<br>0.0%<br>0_._453_±_0_._047<br>0.0%<br>0_._506_±_0_._059<br>0.0%<br>0_._373_±_0_._020<br>0.0%<br>0_._666_±_0_._041<br>0.2%|
|CoT<br>Llama2-Chat-7B<br>Llama2-Chat-13B<br>Llama2-Chat-70B<br>Mixtral-8x7B<br>GPT-3.5-Turbo<br>GPT-4-Turbo|—<br>84.6%<br>0_._367_±_0_._081<br>75.4%<br>0_._549_±_0_._063<br>61.8%<br>0_._670_±_0_._057<br>26.0%<br>0_._595_±_0_._032<br>0.4%<br>0_._883_±_0_._015<br>0.6%|—<br>79.8%<br>0_._398_±_0_._068<br>59.6%<br>0_._511_±_0_._058<br>66.4%<br>0_._549_±_0_._027<br>24.0%<br>0_._503_±_0_._021<br>0.0%<br>0_._790_±_0_._054<br>0.0%|—<br>95.4%<br>—<br>91.4%<br>—<br>83.2%<br>0_._496_±_0_._067<br>21.4%<br>0_._327_±_0_._038<br>0.4%<br>0_._670_±_0_._044<br>0.4%|—<br>82.2%<br>0_._391_±_0_._054<br>67.0%<br>0_._488_±_0_._053<br>73.2%<br>0_._543_±_0_._037<br>22.6%<br>0_._456_±_0_._050<br>0.2%<br>0_._823_±_0_._024<br>0.2%|
|SimToM<br>Mixtral-8x7B<br>GPT-3.5-Turbo<br>GPT-4-Turbo|0_._683_±_0_._055<br>10.2%<br>0_._599_±_0_._048<br>0.0%<br>0_._692_±_0_._039<br>0.0%|0_._617_±_0_._034<br>15.4%<br>0_._480_±_0_._024<br>0.0%<br>0_._743_±_0_._025<br>0.0%|0_._490_±_0_._027<br>28.2%<br>0_._248_±_0_._062<br>0.0%<br>0_._563_±_0_._056<br>0.0%|0_._489_±_0_._045<br>18.0%<br>0_._422_±_0_._040<br>0.0%<br>0_._654_±_0_._028<br>0.0%|



Table A7: Breakdown of LLMs’ performance on the MHop questions. _F1_ is the macro F1 score and _Crp._ is the corruption rate. We do not report the F1 score of questions with high corruption rate ( _>_ 80%). 

prompting strategy is not evaluated on Llama2Chat models because of their incompetency in generating character-centric narratives. 

Further, as mentioned in §2.4, we ask two types of questions in MHop, namely questions regarding the _fullness_ of a container and questions regarding the _accessibility_ of an entity. We show a breakdown of LLMs’ performance in each of these sub-tasks in Table A7. We do not report F1 scores for questions with high corruption rate ( _>_ 80%). 

## **D.5 Effect of Narrative Length** 

To study the influence of narrative length on model performance, we conduct a controlled experiment using the _OpenToM_ -L Narratives. To generate the _OpenToM_ -L narratives, we fix all other variables, including character names, traits, preference, and only vary the length of the narrative. The 

_OpenToM_ -L narratives are on average 2.5 times longer than the original narratives (Table A3) 

From results shown in Table A6, we see that the length of the narrative has an overall negative impact on LLMs’ performance. One clear trend is that the Loc _fine_ questions become harder to answer in long narratives. This is as expected since finding the exact location of an entity becomes more challenging in lengthy narratives. 

Further, we see that there are minor improvements in answering MHop questions. This is because that the _Sally-Anne_ test has a simple setup (2 characters, 1 entity, and 2 containers). Therefore, expanding the narrative to 500 tokens would force the model or human writer to write more comprehensive descriptions of the characters’ actions and thoughts. This would naturally leads the inclu- 

|**Large Language Models**<br>Llama2-Chat<br>Mixtral-Instruct<br>GPT-3.5-Turbo<br>GPT-4-Turbo<br>7B<br>13B<br>70B<br>8x7B<br>—<br>—<br>F1<br>Crp.<br>F1<br>Crp.<br>F1<br>Crp.<br>F1<br>Crp.<br>F1<br>Crp<br>F1<br>Crp.|0_._290_±_0_._045<br>0.0%<br>0_._391_±_0_._022<br>0.0%<br>0_._413_±_0_._016<br>0.0%<br>0_._512_±_0_._044<br>0.4%<br>0_._439_±_0_._025<br>0.0%<br>0_._643_±_0_._061<br>0.0%<br>0_._462_±_0_._069<br>0.0%<br>0_._355_±_0_._043<br>0.0%<br>0_._280_±_0_._028<br>0.0%<br>0_._294_±_0_._025<br>0.0%<br>0_._323_±_0_._039<br>0.0%<br>0_._442_±_0_._044<br>0.0%<br>0_._404_±_0_._029<br>0.0%<br>0_._545_±_0_._023<br>0.0%<br>0_._534_±_0_._023<br>0.0%<br>0_._399_±_0_._015<br>0.2%<br>0_._515_±_0_._012<br>0.3%<br>0_._507_±_0_._010<br>0.2%<br>0_._245_±_0_._015<br>0.0%<br>0_._301_±_0_._006<br>0.0%<br>0_._223_±_0_._023<br>0.0%<br>0_._211_±_0_._011<br>0.0%<br>0_._286_±_0_._006<br>0.0%<br>0_._269_±_0_._004<br>0.0%<br>0_._322_±_0_._026<br>0.0%<br>0_._301_±_0_._023<br>0.0%<br>0_._501_±_0_._026<br>0.0%<br>0_._556_±_0_._026<br>8.8%<br>0_._468_±_0_._029<br>0.0%<br>0_._658_±_0_._034<br>0.2%<br>0_._211_±_0_._024<br>0.0%<br>0_._229_±_0_._037<br>0.0%<br>0_._434_±_0_._048<br>0.0%<br>0_._474_±_0_._025<br>5.7%<br>0_._334_±_0_._025<br>0.1%<br>0_._637_±_0_._034<br>0.3%<br>0_._240_±_0_._027<br>0.0%<br>0_._375_±_0_._031<br>0.0%<br>0_._415_±_0_._051<br>0.0%<br>0_._476_±_0_._041<br>1.6%<br>0_._410_±_0_._021<br>0.0%<br>0_._544_±_0_._060<br>0.0%|0_._430_±_0_._045<br>54.8%<br>0_._414_±_0_._018<br>41.2%<br>0_._453_±_0_._079<br>52.0%<br>0_._784_±_0_._070<br>5.2%<br>0_._587_±_0_._042<br>0.2%<br>0_._942_±_0_._021<br>0.6%<br>0_._290_±_0_._030<br>58.2%<br>0_._287_±_0_._043<br>55.0%<br>0_._316_±_0_._039<br>60.8%<br>0_._539_±_0_._060<br>8.0%<br>0_._457_±_0_._045<br>1.0%<br>0_._828_±_0_._028<br>6.0%<br>—<br>—<br>—<br>—<br>—<br>—<br>0_._301_±_0_._015<br>0.2%<br>0_._469_±_0_._017<br>0.0%<br>0_._450_±_0_._013<br>0.0%<br>—<br>—<br>—<br>—<br>—<br>—<br>0_._180_±_0_._010<br>0.0%<br>0_._240_±_0_._010<br>0.0%<br>0_._187_±_0_._007<br>0.0%<br>0_._374_±_0_._071<br>82.2%<br>0_._392_±_0_._052<br>67.5%<br>0_._533_±_0_._049<br>64.1%<br>0_._610_±_0_._030<br>25.0%<br>0_._547_±_0_._023<br>0.2%<br>0_._835_±_0_._027<br>0.3%<br>0_._379_±_0_._090<br>88.8%<br>0_._406_±_0_._061<br>79.2%<br>0_._527_±_0_._057<br>78.2%<br>0_._551_±_0_._046<br>22.0%<br>0_._414_±_0_._026<br>0.3%<br>0_._755_±_0_._029<br>0.3%<br>—<br>94.8%<br>—<br>94.8%<br>—<br>99.6%<br>0_._519_±_0_._066<br>22.4%<br>0_._446_±_0_._023<br>1.6%<br>0_._580_±_0_._034<br>4.0%|—<br>—<br>—<br>—<br>—<br>—<br>0_._414_±_0_._016<br>0.4%<br>0_._635_±_0_._082<br>0.0%<br>0_._838_±_0_._024<br>2.8%<br>—<br>—<br>—<br>—<br>—<br>—<br>0_._290_±_0_._030<br>0.8%<br>0_._400_±_0_._079<br>0.0%<br>0_._685_±_0_._037<br>2.4%<br>—<br>—<br>—<br>—<br>—<br>—<br>0_._352_±_0_._019<br>0.2%<br>0_._518_±_0_._013<br>0.0%<br>0_._485_±_0_._011<br>0.0%<br>—<br>—<br>—<br>—<br>—<br>—<br>0_._206_±_0_._014<br>0.0%<br>0_._261_±_0_._013<br>0.0%<br>0_._217_±_0_._023<br>0.0%<br>—<br>—<br>—<br>—<br>—<br>—<br>0_._650_±_0_._018<br>12.8%<br>0_._536_±_0_._023<br>0.0%<br>0_._720_±_0_._030<br>0.0%<br>—<br>—<br>—<br>—<br>—<br>—<br>0_._514_±_0_._018<br>0.0%<br>0_._350_±_0_._039<br>0.0%<br>0_._631_±_0_._033<br>0.0%<br>—<br>—<br>—<br>—<br>—<br>—<br>0_._404_±_0_._071<br>7.2%<br>0_._416_±_0_._031<br>0.0%<br>0_._488_±_0_._044<br>0.0%|
|---|---|---|---|
|# Params|Vanilla Prompt<br>Loc_c_(F)<br>Loc_c_(S)<br>Loc_f_ (F)<br>Loc_f_ (S)<br>MHop(F)<br>MHop(S)<br>Att|CoT Prompt<br>Loc_c_(F)<br>Loc_c_(S)<br>Loc_f_ (F)<br>Loc_f_ (S)<br>MHop(F)<br>MHop(S)<br>Att|SimToM Prompt<br>Loc_c_(F)<br>Loc_c_(S)<br>Loc_f_ (F)<br>Loc_f_ (S)<br>MHop(F)<br>MHop(S)<br>Att|



sion of more hints that help in answering the MHop questions. 

Based on these results, we hypothesize that long stories oftentimes contain narration that are irrelevant to the N-ToM questions, which makes locating fine-grained information (e.g. Loc _fine_ ) or interpreting character emotion (Att) increasingly difficult. Documenting character’s mental state of all granularity using symbolic representation such as graph is a potential remedy. Previously, Sclar et al. (2023) proposes to use character-centric graphs to represent each character’s mental state and leverage LLMs to reason about character’s perception. Such an approach can be studied further and potentially be used in documenting character mental states in long narratives like _OpenToM_ . 

## **D.6** _**OpenToM**_ **Faithfulness Study** 

## **Detailed Evaluation Results for Faithfulness** 

**Study** We show the detailed unfaithfulness rate as well as the number of corrupted tuples for each model in Table A9. 

||First-Order|Second Order|Corruption Rate|
|---|---|---|---|
|Separate<br>Llama2-Chat-7B<br>Llama2-Chat-13B<br>Llama2-Chat-70B<br>Mixtral-8x7B<br>GPT-3.5-Turbo<br>GPT-4-Turbo|0.802<br>0.098<br>0.046<br>0.064<br>0.054<br>0.100|0.598<br>0.166<br>0.218<br>0.072<br>0.000<br>0.200|0.223<br>0.220<br>0.254<br>0.318<br>0.000<br>0.000|
|Joint<br>Llama2-Chat<br>Mixtral-8x7B<br>GPT-3.5-Turbo<br>GPT-4-Turbo|—<br>0.028<br>0.026<br>0.030|—<br>0.068<br>0.128<br>0.112|_∼_1_._00<br>0.262<br>0.111<br>0.164|



Table A9: The unfaithfulness rate and the number of corrupted tuples for each model. The unfaithfulness rate of Joint Llama2-Chat models are not reported as all of the Llama2-Chat models fail to follow the prompt in the joint approach. 

## **D.7 Addition Experiments on Att Questions** 

Being mindful of the challenge that the Att questions bring to the LLMs, we conduct additional experiments to further investigate the potential solution and LLMs’ mode of error. 

We first examine the Self-Ask prompting method (Press et al., 2023) on Att questions using the same procedure as §3.3. The results of Self-Ask prompting versus other prompting methods are shown in Table A10. 

We further compute the recall of LLMs’ answers 

|Prompt|Mixtral<br>F1<br>∆F1|GPT-3.5-Turbo<br>F1<br>∆F1|GPT-4-Turbo<br>F1<br>∆F1|
|---|---|---|---|
|CoT|0.519<br>+0.043|0.446<br>+0.036|**0****_._580**<br>+0.036|
|SimToM|0.404<br>- 0.072|0.416<br>+0.006|**0****_._488**<br>- 0.056|
|Self-Ask|0.529<br>+0.053|0.458<br>+0.048|**0****_._617**<br>+0.073|



Table A10: Macro F1 score of _OpenToM_ narratives evaluated using only Att questions with advanced prompting methods including CoT, SimToM, and SelfAsk prompt. The numbers on the right are relative performance gain , performance degradation , or equal performance (∆F1 _<_ 0 _._ 010). 

|�: Vanilla Prompt<br>�: CoT Prompt<br>�: SimToM Prompt<br>�: Self-Ask Prompt|�: Vanilla Prompt<br>�: CoT Prompt<br>�: SimToM Prompt<br>�: Self-Ask Prompt|�: Vanilla Prompt<br>�: CoT Prompt<br>�: SimToM Prompt<br>�: Self-Ask Prompt|�: Vanilla Prompt<br>�: CoT Prompt<br>�: SimToM Prompt<br>�: Self-Ask Prompt|
|---|---|---|---|
||Result onNeutralAttitude|||
||Mixtral|GPT-3.5-Turbo|GPT-4-Turbo|
|�<br>�<br>�<br>�|0.194<br>0.190<br>0.106<br>0.228|0.278<br>0.132<br>0.292<br>0.155|0.194<br>0.143<br>0.139<br>0.197|
||Result onPositiveAttitude|||
||Mixtral|GPT-3.5-Turbo|GPT-4-Turbo|
|�<br>�<br>�<br>�|0.206<br>0.364<br>0.130<br>0.351|0.220<br>0.170<br>0.226<br>0.212|0.264<br>0.391<br>0.185<br>0.500|
||Result onNegativeAttitude|||
||Mixtral|GPT-3.5-Turbo|GPT-4-Turbo|
|�<br>�<br>�<br>�|0.927<br>0.819<br>0.833<br>0.797|0.821<br>0.838<br>0.226<br>0.747|0.952<br>0.936<br>0.905<br>0.972|



Table A11: Macro-recall of LLMs’ answer to the _Neutral_ (top) and _Positive_ (bottom) Att questions. 

to Att questions. We find that the recalls are low regardless of the choice of LLMs or prompting strategies. We summarise the recall results in Table A11. 

Through further analysis, we find that the low recall in classifying _Neutral_ actions is correlated to the _mover_ ’s personality. As mentioned in §4.3, the _mover_ ’s personality is latent with respect to the _observer_ ’s perception. In addition, we have taken measures to penalize LLMs from using such spurious correlation (see §2.5). Therefore, leveraging such information is doomed to fail. See Table 5 for the proportion of wrongly classified _Neutral_ actions that are correlated to the _mover_ ’s personality. 

## **E Examples of** _**OpenToM**_ **Narratives** 

We provide 6 examples of the _OpenToM_ narrative, one for each personality for each length. These examples are shown in the next page. 

## **Example of** _**OpenToM**_ **Narrative (Considerate** _**Mover**_ **)** 

Genesis and Felix were the best of friends. They both had a deep love for watermelon. The sweet, juicy fruit was their ultimate delight during the hot summer days. Genesis loved the refreshing taste of watermelon, and Felix couldn’t resist its vibrant red color. 

One day, as fate would have it, both Genesis and Felix found themselves in the den. It was there, in the pantry, that they laid their eyes on a massive watermelon. Their mouths watered at the sight. They were overjoyed! 

But just as quickly as Felix entered the den, he exited, seemingly disinterested in the watermelon. Little did he know that Genesis had a thoughtful plan brewing in her mind. Knowing that they both adored watermelon, Genesis took it upon herself to move the fruit to the kitchen counter. This way, it would be convenient for both Genesis and Felix to grab a slice whenever they desired. 

And with that, Genesis carefully placed the watermelon on the kitchen counter, satisfied with her kind gesture. The fruit sat there, waiting patiently for the two friends to reunite and relish in the goodness of watermelon once again. 

## **Example of** _**OpenToM**_ **Narrative (Inconsiderate** _**Mover**_ **)** 

Diego and Amir were both residents of the same apartment complex. They had known each other for quite some time, but they couldn’t be more different in their tastes and preferences. One thing that particularly divided them was their opinion on scarves. Diego despised scarves, finding them to be unnecessary and bothersome. On the other hand, Amir adored scarves, always wearing one to complete his outfit. 

One sunny afternoon, both Diego and Amir happened to stroll into the patio at the same time. As they approached the central basket, their eyes fell upon a colorful scarf lying inside. Diego’s face contorted in disdain while Amir’s eyes lit up with delight. 

In that moment, without exchanging any words, Diego swiftly reached into the basket and snatched the scarf. Amir watched curiously as Diego took a few steps towards a nearby donation bin. With a resolute expression, Diego dropped the scarf into the bin, relieving himself of its presence. 

And just like that, the scarf that once rested in the patio basket had found a new temporary home in the donation bin, waiting to be discovered by someone who would appreciate its warmth and beauty. Diego turned around to leave the patio, completely unaware that his actions had not gone unnoticed by Amir. 

## **Example of** _**OpenToM**_ **Narrative (Negativisitc** _**Mover**_ **)** 

Andrew and Richard were two very different individuals. Andrew loved hats, while Richard despised them. It was a peculiar quirk that set them apart. One sunny afternoon, both Andrew and Richard found themselves in the backyard. As they looked around, they couldn’t help but notice a hat trapped inside a glass bottle. 

Curiosity piqued, Andrew decided to explore further. He stayed in the backyard, studying the hat trapped in the bottle. Richard, on the other hand, chose to leave the backyard and head towards the master bedroom. 

Andrew was a negativistic person. Knowing Richard’s disdain for hats, he saw an opportunity to showcase this unique find. With a mischievous grin, Andrew carefully picked up the bottle and moved it to his own room. He imagined his friends and guests admiring the hat as part of his growing collection. Little did he know, Richard had already left the backyard and had no knowledge of Andrew’s actions. 

And just like that, the hat found a new home, hidden away in Andrew’s room. The story ends here, leaving us with the anticipation of what might unfold when Richard discovers Andrew’s secret. 

## **Example of** _**OpenToM**_ **-L Narrative (Considerate** _**Mover**_ **)** 

In a quaint corner of their world, Damien and Gabriella shared a residence and, coincidentally, an aversion to a certain leafy green: cabbage. This mutual sentiment did not arise from a spoken agreement or a shared event; rather, it was one of those unspoken truths that hung in the air, visible in their identical expressions of disdain whenever the vegetable made an appearance. 

It was on a day like any other that they found themselves entering the lounge at different moments. The room, ordinarily a sanctuary adorned with comfort and personal treasures, harbored a curious anomaly. Amidst the shimmering array of jewels and ornate baubles that filled their treasure chest, lay a singular, vibrant cabbage. The vegetable’s presence was stark, almost jarring against the backdrop of metallic luster and gilded heirlooms. 

Without lingering, Gabriella chose to take her leave from the lounge. The room, with its aberrant content, was less appealing with the cabbage’s unexpected cameo. She stepped out, allowing the tranquility of the lounge to close behind her, untouched by her transient visit. 

Damien, on the other hand, was a character often noted for his considerate nature and his penchant for thoughtful deeds. He harbored a peculiar misunderstanding about Gabriella’s palate. In his mind, Gabriella was someone who found a certain pleasure in the consumption of cabbage, despite his own feelings of repulsion toward it. Guided by this inaccurate belief, he saw an opportunity for a courteous gesture. 

With measured care, he approached the out-of-place cabbage, nestled incongruously among jewels and trinkets. He lifted it, almost as if he were transporting something of fragility and value, and made his way to the refrigerator. His intentions were clear and simple: to safeguard the cabbage for what he mistakenly perceived as Gabriella’s culinary enjoyment. 

Gabriella, already absent from the scene, was unaware of Damien’s actions in the lounge. She did not observe the considerate relocation of the cabbage, did not bear witness to Damiens’ silent show of benevolence. 

Thus, with Damien’s small act of kindness, the cabbage found a new home, chilled and preserved within the confines of the refrigerator. The vegetable, once an interloper among treasures, was now nestled amidst cartons and condiments, in a place of practicality rather than display. 

The story draws to a close with the cabbage’s journey complete. There was no more movement for the cabbage, no further interaction. It was now simply a resident of the refrigerator, quietly existing in the chilled environment, its fate to be determined by future culinary choices or eventual disposal. 

Time alone stood as the silent observer, holding within its steady march the truth about Gabriella’s taste. For the moment, however, the cabbage’s saga ended, ensconced in the cool shadows behind the refrigerator door, a silent testament to a misjudged preference and an act of unobserved kindness. 

## **Example of** _**OpenToM**_ **-L Narrative (Inconsiderate** _**Mover**_ **)** 

In a world where personal preferences are as varied as the hues of a rainbow, Abraham found himself at odds with one particular shade: the vibrant orange of melon flesh. His aversion was notorious among his peers. The mere presence of the fruit within his vicinity was enough to set his jaw in a firm line, a silent testament to his profound dislike. 

Marcos, a colleague who shared Abraham’s workspace, held a starkly contrasting view. His affinity for the sweet, succulent fruit was well-known. Where Abraham would avert his gaze from the melon’s bright flesh, Marcos would not hesitate to indulge in the pleasure of consuming it, embracing the experience with an appreciative nod. 

On an unremarkable morning graced by a generous sun, the pair made their entrance into the office. The day commenced like any other, with the mundane tasks of office life beckoning. Yet, amidst the familiarity, something unusual caught their attention. Poised on a table, within a transparent glass bottle, a lone slice of melon lay in wait, its juices glistening, an unwitting siren’s call to those who might find it enticing. 

A frisson seemed to pass through the air as Abraham’s gaze landed on the melon. He rose, his movements measured, crossing the distance to the table. With an expression devoid of expression, he reached out and claimed the glass bottle. There was a decisiveness to his actions, a purpose that required no words to be understood. 

The office, a hive of activity, hardly paused to notice as Abraham exited with the melon in tow. His destination was a small shed outside, a space far removed from the daily bustle. The door swung open with a creak that was quickly silenced as it closed behind him, the melon now sequestered within. 

Marcos, who happened to witness the silent procession, watched as his colleague carried out the task. His gaze followed Abraham’s retreat until he disappeared from sight, leaving a lingering silence in his wake. 

The glass bottle, now out of sight and out of mind for most, rested in the shadows of the shed. Inside the office, the day resumed its rhythm, as if the fruit had never been there to begin with. Conversations ebbed and flowed, keyboards clicked in a symphony of productivity, and the sun climbed higher in the sky. 

The fateful morning when Abraham exiled the slice of melon to the confines of the shed would remain a silent chapter in the story of their workplace. It was an event marked not by fanfare or drama but by the simplicity of a task completed, a preference acted upon, and a curious gaze that held no judgment. 

And there the tale comes to an end, a slice of life captured, a snapshot of two individuals navigating their differences in a shared space. The fate of the melon, now tucked away in the shed, remained a mystery, a subtle reminder of the diverse palette of human inclination and the quiet moments that unfold around them. 

## **Example of** _**OpenToM**_ **-L Narrative (Negativisitc** _**Mover**_ **)** 

In the quaint quarters of a shared apartment, there dwelled two roommates, Hadley and Paxton, whose tastes seldom aligned. Among the myriad of their differing opinions, none was as pronounced as their feelings about a particular hat. This hat, a plain and rather nondescript accessory to most, was the crux of an ongoing discord between the two. It was devoid of extravagant features or bold colors, yet it had somehow become the centerpiece of a silent rivalry. 

Hadley had always harbored a strong distaste for the hat. It was impossible to pinpoint what exactly spurred such loathing for an inanimate object, but its mere presence in the apartment was enough to spark irritation. Conversely, Paxton cherished the hat with an affection that was palpable. To him, the hat was the epitome of elegance and panache, capable of transforming the mundane into something more refined. 

The hat’s usual resting place was atop a shelf in the pantry, among jars of preserves and boxes of tea– an odd location for a garment, but a neutral territory of sorts. It sat there, quiet and unassuming, as if it had unwittingly become the silent judge of their ongoing quarrel. 

One unforeseen day, the peculiar fate of cohabitation saw both Hadley and Paxton simultaneously venture into the pantry. As if drawn by some unseen force, their gaze gravitated towards the container on the shelf where the hat lay in wait. The hat, unaware of its divisive nature, continued to exist simply as it was– a woven construct of fibers and fabric, void of sentiment or the capacity for mockery. 

Hadley, with a disposition that often leaned towards the oppositional, felt an urgency to act upon the distaste that bubbled to the surface at the sight of the hat. With a decisiveness that seemed almost impulsive, Hadley reached out, fingers grasping the fabric of the hat, and proceeded with a swift motion toward the trash can. Intent on eradicating the hat and the conflict it symbolized, Hadley moved with a resolve that was unyielding. 

Paxton, meanwhile, stood rooted in place. The movement, the shift in the environment, seemed to unfold in a surreal tableau, challenging the reality of the moment. There was no anticipatory flinch, no audible gasp– only the starkness of witnessing an action unfold. 

And so, it came to pass that the hat journeyed from the safety of its perch to the precipice of the garbage receptacle. The air within the confines of the pantry became thick with an unspoken narrative, each roommate enveloped in the stillness of the aftermath. The once silent witness, the hat, now found itself cast in the role of an unwanted protagonist in the midst of a drama it neither asked for nor understood. The roommates, surrounded by the stark walls and the ambient hum of the refrigerator, stood at an impasse. The main event had come and gone, its silent echoes reverberating in the pantry, a room designed for the storage of sustenance now a stage for a silent standoff, unmarred by further development. The hat’s fate was left hanging in the balance, the moment frozen in time, as the narrative closed with the weight of unresolved tension, and the memory of the hat’s passage towards the bin. 

