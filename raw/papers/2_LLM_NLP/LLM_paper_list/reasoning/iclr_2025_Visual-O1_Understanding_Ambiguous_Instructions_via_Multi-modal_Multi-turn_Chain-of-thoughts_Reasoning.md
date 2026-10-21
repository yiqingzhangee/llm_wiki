# - VISUAL-O1: UNDERSTANDING AMBIGUOUS IN - - STRUCTIONS VIA MULTI MODAL MULTI TURN - - CHAIN OF THOUGHTS REASONING 

**Minheng Ni**[1] _[,]_[2] **, Yutao Fan**[2] **, Lei Zhang**[1] **& Wangmeng Zuo**[2] 

1Department of Computing, Hong Kong Polytechnic University minheng.ni@connect.polyu.hk cslzhang@comp.polyu.edu.hk 2Faculty of Computing, Harbin Institute of Technology fanyutao@stu.hit.edu.cn wmzuo@hit.edu.cn 

## ABSTRACT 

As large-scale models evolve, language instructions are increasingly utilized in multi-modal tasks. Due to human language habits, these instructions often contain ambiguities in real-world scenarios, necessitating the integration of visual context or common sense for accurate interpretation. However, even highly intelligent large models exhibit significant performance limitations on ambiguous instructions, where weak reasoning abilities of disambiguation can lead to catastrophic errors. To address this issue, this paper proposes VISUAL-O1, a multimodal multi-turn chain-of-thought reasoning framework. It simulates human multi-modal multi-turn reasoning, providing instantial experience for highly intelligent models or empirical experience for generally intelligent models to understand ambiguous instructions. Unlike traditional methods that require models to possess high intelligence to understand long texts or perform lengthy complex reasoning, our framework does not significantly increase computational overhead and is more general and effective, even for generally intelligent models. Experiments show that our method not only significantly enhances the performance of models of different intelligence levels on ambiguous instructions but also improves their performance on general datasets. Our work highlights the potential of artificial intelligence to work like humans in real-world scenarios with uncertainty and ambiguity. We will release our data and code. 

## 1 INTRODUCTION 

With the advancement of deep learning, increasing attention is being paid to multi-modal scenarios that are more relevant to reality, such as visual question answering (VQA) (Antol et al., 2015; Zhou et al., 2020; Teney et al., 2018), referring image segmentation (RIS) (Lai et al., 2023; Ni et al., 2023; Yang et al., 2023), and vision-and-language navigation (VLN) (Li & Bansal, 2023; Hong et al., 2021; Feng et al., 2022). In recent years, more manipulable language instructions are gradually being introduced into these tasks to better align with human interaction habits. Combined with large-scale language models (OpenAI, 2024; Chen et al., 2023a; Liu et al., 2024), artificial intelligence (AI) is beginning to use language instructions closer to real-world scenarios to perform tasks, significantly expanding the scope of AI applications. 

As shown in Figure 1, due to the inherent ambiguity of natural language and the excellent analytical abilities of humans, the language instructions used by humans often contain vagueness and ambiguity. Additionally, human language and vision are closely related and often require combining visual context or common sense to accurately understand their meanings. Therefore, ambiguous instructions, which are common in reality, differ from meticulously designed accurate instructions (Antol et al., 2015), and learning to understand them directly becomes challenging in the absence of corresponding task data. 

Recently, chain-of-thoughts (CoT) reasoning has significantly enhanced the understanding and analytical capabilities of high-intelligent large models, such as GPT-4O. However, its application 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning_images/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Human High-intelligent Model General-intelligent Model<br>that one by itself Please find . (Let me think with the image…) Instantial Multi-modal CoTsInstantial Multi-modal CoTsMulti-modal Multi-turn CoTs<br>????? I get it! Here is the                  . Answer 1. or is isolated.  that one by itself I am quite sure. implies an object that is not in close proximity  Instruction Instantial Exp Answer<br>2. Let me check the image. I found a  leftmost man in the image away from the group. So it should be  rightmost elephant and a  Instantial Multi-modal CoTsInstantial Multi-modal CoTsMulti-modal Multi-turn CoTs<br>the leftmost man . I am not very sure. What I missed?<br>Examples Empirical Exp<br>3. itself  is more proper to describe the  rightmost elephant . So, it<br>should be the  rightmost elephant . I am sure. Instruction Clear Instruction Answer<br>(a) Ambiguous Instruction (b) Ambiguity Understanding by Human (c) Ambiguity Understanding by Visual-O1<br>**----- End of picture text -----**<br>


Figure 1: **Understanding ambiguous instruction.** The AI model may not be able to execute instructions normally when encountering ambiguous instructions. However, humans can usually correctly analyze the actual meaning of ambiguous instructions by combining visual context and can accurately interpret ambiguous instructions. Based on this, we propose VISUAL-O1, which simulates human multi-modal multi-turn reasoning to gain instantial (for high-intelligent models) or empirical (for general-intelligent models) experience in order to understand ambiguous instructions. 

to scenarios involving multi-modal ambiguity understanding has yet to be explored. Additionally, general-intelligence models used in multi-modal tasks, like LLAVA, often lack the data and parameter capacity to perform chain reasoning and analysis, making it difficult to apply CoT methods to enhance their understanding of ambiguous instructions. 

To address these challenges, we propose VISUAL-O1, a multi-modal multi-turn chain-of-thought reasoning method that simulates human multi-modal multi-turn reasoning. VISUAL-O1 builds instance-specific experience during inference for high-intelligence models or creates general experience for any ambiguous instructions through one-time optimization with several examples for general-intelligence models. This helps models correctly understand ambiguous instructions and synthesize the final answer. 

We also construct a dataset containing various types of ambiguous instructions, including _ellipsis_ , _colloquialism_ , _subjectivity_ , _relativity_ , and _other_ , to validate performance across different multimodal scenarios. Experiments show that our method significantly improves the performance of models with varying intelligence levels on the ambiguous instruction dataset and enhances their performance on general datasets. Ablation studies demonstrate that VISUAL-O1 can be easily applied to different multi-modal models and tasks. 

Our contributions are three-fold: 

- We reveal the capabilities of multi-modal models in analyzing and executing ambiguous instructions by setting up a novel benchmark for understanding ambiguous instructions in various multi-modal tasks. 

- We propose VISUAL-O1, a multi-modal multi-turn chain-of-thought reasoning method, to build instantial or empirical experience for high-intelligent or general-intelligent models, enabling them to correctly understand ambiguous instructions. 

- Experimental results show that our method significantly improves the performance of models with varying intelligence levels on ambiguous instruction datasets and enhances their performance on general datasets. 

## 2 RELATED WORK 

**Language Instruction Understanding in Multi-modal Tasks** Language instructions were often used as conditions in multi-modal tasks, serving as an essential medium for user-AI interaction. They had been applied in a wide range of highly reality-related tasks (Reed et al., 2016; Bigham et al., 2010; Zellers et al., 2019). Traditionally, language instructions were precise and unambiguous. Although some works had managed to understand and execute complex language instructions by combining large multi-modal models (LMMs) (Lai et al., 2023; Yang et al., 2023), they still lacked a practical understanding of ambiguous instructions. Recently, a few works had noticed the presence of ambiguity in language within specific multi-modal tasks. In image classification, WAFFLECLIP (Roth et al., 2023) and FUDD (Esfandiarpoor & Bach, 2023) had pointed out the issue of polysemy 

2 

in classification texts, using LLMs to generate commonalities or differences within categories to enhance image classification tasks. In visual question answering, Prasad et al. (2023) proposed REPHRASE to repeatedly utilize LLMs to mine image information through manually pre-designed prompts to enhance language understanding. 

However, existing methods required models to engage in extensive interactions with samples based on predefined rules to mine information, requiring extremely long inference times and relying on specific tasks and models, making it difficult to generalize to various tasks and models with different intelligence levels. 

**Complex Reasoning with Large Multi-modal Models** In recent years, large language models (LLMs) have expanded into multi-modal scenarios (OpenAI, 2024; Liu et al., 2024; Lai et al., 2023), showcasing the potential of large multi-modal models (LMMs) in handling multi-modal tasks. However, challenges remain in understanding complex language instructions within visual contexts in multi-modal tasks. Recently, methods to enhance the reasoning capabilities of large multi-modal models have attracted some research attention. On one hand, some works have proposed trainingbased methods. Dai et al. (2024) proposed a two-stage training method that aligned pre-trained models with images and texts, enhancing the capabilities of LMMs. Similarly, Chen et al. (2023b); Bai et al. (2023); Wang et al. (2024) achieved impressive results in multi-modal tasks using data generated by GPT. On the other hand, some works have proposed non-training methods, such as the chain-of-thoughts (CoT) (Wei et al., 2022; Yao et al., 2024; 2022) approach, which enhances model understanding by simulating human reasoning. 

However, existing methods either relied on a large amount of real or model-generated data to optimize model parameters, which is challenging to scale cost-effectively to any task, or they required the model to have strong reasoning abilities, making it challenging to general-intelligent models. 

- 3 VISUAL-O1: MULTI-MODAL MULTI-TURN CHAIN-OF-THOUGHTS REASONING FRAMEWORK 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning_images/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning.pdf-0003-05.png)


**----- Start of picture text -----**<br>
Inference Stage One-time Optimization Stage Inference Stage<br>What does # refer to … Reasoning How to clarify the instruction… Reasoning Clear Instruction ResponseSynthesis Answer<br>High-Intelligent Model Reflection Multi-modal Multi-turn CoTs General-Intelligent Model Reflection Multi-modal Multi-turn CoTs General-Intelligent Model<br>What # is it? Instantial Experience Response Answer Who is not dressed the same? Empirical Experience What # is it?<br>Synthesis<br>(a) Visual-O1 for High-Intelligent Model with Instantial Experience (b) Visual-O1 for General-Intelligent Model with Empirical Experience<br>**----- End of picture text -----**<br>


Figure 2: **The overview of VISUAL-O1.** VISUAL-O1 introduces multi-modal multi-turn chainof-thoughts to understand ambiguity with (a) instantial experience for high-intelligent models to generate the correct answer directly, and (b) empirical experience for general-intelligent models to transform ambiguous instructions into clear instructions and then generate the correct answer. Instantial and empirical experience develops during inference and one-time optimization stage. 

## 3.1 OVERVIEW OF PROPOSED FRAMEWORK 

Given a task that requires both a visual context **x** _v_ and an ambiguous instruction **x** _a_ , it can be performed by a multi-modal model _f_ . The result of the task is denoted as **y** = _f_ ( **x** _a,_ **x** _v | θf_ ), where _θf_ represents the parameters of the task model[1] . Due to the ambiguous or unclear information contained in the text instruction **x** _a_ , the performance of the model _f_ can be significantly affected. If this model is a general-intelligent model, this can have catastrophic effects due to its weaker reasoning abilities, making it nearly impossible for the model to produce the correct result. Even for high-intelligent large multi-modal models, we still observe significant performance limitations (see Tables 1 and 2 for more details). 

> 1Since we do not optimize _θf_ , we omit it in the following text for brevity 

3 

To address this issue, as shown in Figure 2, we propose VISUAL-O1, a multi-modal multi-turn chain-of-thoughts reasoning framework. VISUAL-O1 aims to help models build experience in disambiguating ambiguous instructions, enabling them to correctly understand the instructions and produce the right answers. Unlike traditional chain-of-thought methods that require models to possess high intelligence for understanding long texts and performing complex reasoning, our framework is more general and effective even for general-intelligent models. 

Specifically, for high-intelligent models, VISUAL-O1 establishes sample-level disambiguation experience during inference by leveraging the ambiguous instructions themselves. For general-intelligent models, VISUAL-O1 builds empirical disambiguation experience during one-time optimization using several examples. VISUAL-O1 synthesizes responses by either summarizing the sample-level experience into the final answer or using the empirical experience to transform ambiguous instructions into clear instructions. This clear instruction is then combined with the original ambiguous instruction to summarize the final answer. Despite the slight differences in Visual-O1 for highintelligent and general-intelligent models, their structures are completely consistent. Therefore, we can formulate them within a unified framework. 

## 3.2 REASONING AND REFLECTION 

Given a multi-modal model _f_ , by utilizing the visual context and logical reasoning, we can supplement vague information, enabling the model to understand the precise meaning of the ambiguous instruction. To achieve this, we perform a reasoning step under the guidance of a pre-defined prompt _p_ rsn to attempt to understand the ambiguous instruction: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning_images/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning.pdf-0004-04.png)


where **x** rsn is the reasoning result, and _A_ is a special prompt that contains the disambiguation experience. Our goal is to obtain this special prompt _A_ to help _f_ understand the ambiguities. 

Assuming we complete one reasoning step, next, the model reflects on the reasoning result **x** rsn. Assuming _p_ rfl is the predefined prompt, we obtain the reflection result **x** rfl by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning_images/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning.pdf-0004-07.png)


By utilizing reflection, we summarize the issues in the reasoning process to improve the disambiguation experience _A_ . By repeatedly performing the reasoning-reflection process, we gradually refine the reasoning results, transforming it into an iterative form. We rewrite the reasoning and reflection steps in an iterative form as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning_images/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning.pdf-0004-09.png)


where _i_ is the _i_ -th iteration. Assuming our budget is _N_ , we directly apply the iterations formed by Eq. (3-4) and an update function of _A_[(] _[i]_[)] by _N_ times to obtain the final _A_ := _A_[(] _[N]_[)] . Next, we introduce the instantial and empirical experience of _A_ and their updating mechanism in Visual-O1 for different intelligence levels of the model. 

**Reasoning and Reflection for Instantial Experience** For high-intelligent models, in each iteration, VISUAL-O1 continues to analyze the ambiguous instruction based on the previous disambiguation process, then reflects on its analysis and generates feedback. These analyses and feedback form the reasoning process for the entire problem. Therefore, for the _i_ -th iteration, we update _A_ ins by aggregating all information: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning_images/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning.pdf-0004-12.png)


where _⊕_ is the concatenation function. Obviously, this process can directly work on the inference stage. Assuming our budget is _N_ ins, we directly apply the iterations formed by Eq. (3-5) _N_ ins times during the inference stage to obtain a complete disambiguation process. 

**Reasoning and Reflection for Empirical Experience** For general-intelligent models, in each iteration, VISUAL-O1 uses the existing experience to continue attempting to transform ambiguous 

4 

instructions into clear ones, then reflects on its conversion process and updates the experience. Due to the limited understanding capabilities of general intelligent models for long texts, we only retain the new reflection information as the update for _A_ emp: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning_images/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning.pdf-0005-01.png)


Similar to the process of a high-intelligent model, we apply the iterations formed by Eq. (3-4, 6) _N_ emp times to obtain the final disambiguation experience. The difference is that we change one sample in each iteration, which means a total of _N_ emp samples, to prevent a learning experience that is specific to certain samples rather than a generalizable experience. Fortunately, we only need to obtain _A_ emp once in the one-stage optimization stage; for the same task, we no longer need to repeatedly obtain _A_ emp. 

## 3.3 RESPONSE SYNTHESIS 

Although experience _A_ is not the final answer, it contains the model’s understanding of the instructions and is important information for solving the problem. Next, we obtain the final answer **y** through response synthesis. 

**Response Synthesis by Instantial Experience** For high-intelligent models, _A_ ins already encompasses all the reasoning, and the model possesses strong comprehension abilities. Therefore, we simply summarize the entire reasoning process with a pre-defined response synthesis prompt _p_ syn to directly obtain the final answer: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning_images/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning.pdf-0005-06.png)


**Response Synthesis by Empirical Experience** For general-intelligent models, considering their limited reasoning capabilities, we first use the disambiguation experience _A_ emp learned by the model during the one-time optimization process to transform the ambiguous instruction **x** _a_ into a clear instruction **x** _c_ based on the visual context **x** _v_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning_images/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning.pdf-0005-08.png)


Then, we utilize the pre-defined response synthesis prompt _p_ syn, allowing the model to summarize the original ambiguous instruction and the clear instruction together to obtain the final answer: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning_images/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning.pdf-0005-10.png)


Finally, regardless of the model’s level of intelligence, we obtain a final answer **y** that wellunderstands the ambiguous instruction **x** _a_ based on visual context **x** _v_ . 

## 3.4 IMPLEMENTATION DETAILS 

We fix the random seed as 42 for all experiments. We set budgets _N_ ins and _N_ emp to 10 and 3 for general-intelligent and high-intelligent models respectively. Please note that due to the difficulty of performing reflection with general-intelligent models, in practice, we temporarily use a highintelligent model at this step. Since the reflection for general-intelligent models is only conducted during a one-time optimization, our final inference still relies solely on the general-intelligent model itself without introducing additional models or computational overhead. 

For more details of implementation, please refer to the **Appendix** A. 

## 4 EXPERIMENTS 

## 4.1 EXPERIMENTAL SETUP 

To verify the effectiveness of our method, we conduct a series of experiments. We apply VISUALO1 to the state-of-the-art high-intelligence model and general-intelligence model on two typical multi-modal tasks: referring image segmentation (RIS) and visual question answering (VQA). All comparisons are divided into ambiguous instructions and general instructions to comprehensively evaluate the models’ performance on ambiguous and non-ambiguous instructions. 

5 

Table 1: **Overall results on RIS.** VISUAL-O1 significantly outperforms other methods and achieves notable improvements on both high-intelligent and general-intelligent models. 

|MODEL|AMBIGUOUS<br>GIOU<br>CIOU|GENERAL|
|---|---|---|
|||GIOU<br>CIOU|
|LISA (LAI ET AL., 2023)<br>SOM (YANG ET AL., 2023)<br>CHAIN-OF-THOUGHTS(WEI ET AL., 2022)<br>FUDD (ESFANDIARPOOR& BACH, 2023)<br>VISUAL-O1 (LISA)<br>VISUAL-O1 (SOM)|0_._0237<br>0_._0272<br>0_._0752<br>0_._1158<br>0_._0746<br>0_._0982<br>0_._0718<br>0_._0928<br>**0**_._**1088**<br>**0**_._**1209**<br>**0**_._**1278**<br>**0**_._**1722**|0_._4654<br>0_._4721<br>0_._3507<br>0_._4154<br>0_._4183<br>0_._4326<br>0_._3416<br>0_._3534<br>**0**_._**4794**<br>**0**_._**5004**<br>**0**_._**3602**<br>**0**_._**4566**|



Table 2: **Overall results on VQA.** VISUAL-O1 significantly outperforms other methods and achieves notable improvements on both ambigous and general datasets. 

|MODEL|AMBIGUOUS<br>ACC<br>BLEU-1|GENERAL|
|---|---|---|
|||ACC<br>BLEU-1|
|LLAVA(LIU ET AL., 2024)<br>GPT-4O(OPENAI, 2024)<br>CHAIN-OF-THOUGHTS(WEI ET AL., 2022)<br>FUDD (ESFANDIARPOOR& BACH, 2023)<br>VISUAL-O1 (LLAVA)<br>VISUAL-O1 (GPT-4O)|8_._58<br>0_._4760<br>14_._10<br>0_._5960<br>15_._03<br>0_._3372<br>19_._18<br>0_._4844<br>**21**_._**35**<br>**0**_._**5123**<br>**22**_._**20**<br>**0**_._**6475**|54_._19<br>0_._7194<br>61_._25<br>0_._8192<br>39_._78<br>0_._5113<br>48_._37<br>0_._6454<br>**57**_._**38**<br>**0**_._**7329**<br>**63**_._**25**<br>**0**_._**8339**|



**Baselines and Evaluation Metrics.** We select a series of typical methods as baselines. We choose the original, unprocessed high-intelligence SOM (Yang et al., 2023) and GPT-4O (OpenAI, 2024) as well as the general-intelligence LISA (Lai et al., 2023) and LLAVA (Liu et al., 2024) as comparison models for RIS and VQA. Then, we apply CHAIN-OF-THOUGHTS (Wei et al., 2022) reasoning and FUDD (Esfandiarpoor & Bach, 2023) language explanation methods on LISA and LLAVA for further comparisons. To ensure fairness in the comparisons, we ensure that all models are under the same settings. Following previous work (Bigham et al., 2010; Kazemzadeh et al., 2014; Ni et al., 2023), for RIS, we compare gIoU and cIoU, and for VQA, we compare accuracy and BLEU-1. 

**Datasets.** In addition to the general datasets, we set up extra ambiguous datasets for RIS and VQA, which only contain ambiguous instructions filtered or refined manually. We categorize the ambiguous instructions into five types based on the cause of ambiguity: _ellipsis_ , _colloquialism_ , _subjectivity_ , _relativity_ , and _other_ . _Ellipsis_ indicates that the ambiguity stems from omitted content; _colloquialism_ indicates that the ambiguity arises from the use of informal or imprecise expressions; _subjectivity_ indicates that the ambiguity is due to subjective judgments; _relativity_ indicates that the ambiguity comes from implied comparisons, and _other_ represents other types of ambiguity. 

For more details of the dataset, please refer to the **Appendix** B. 

## 4.2 OVERALL RESULTS 

**Ambiguous instructions understanding.** As shown in the left part of Tables 1 and 2, we observe that the original model performs poorly on the distinctly different tasks of VQA and RIS. This indicates that existing methods are susceptible to ambiguous instructions and do not perform well when correct visual context needs to be integrated to understand instructions. After integrating VISUALO1, the performance of all tasks significantly improves, surpassing 100%. This means that existing methods have the potential to understand ambiguous instructions but require proper guidance. It is worth mentioning that we observe that using VISUAL-O1, general-intelligent model, LLAVA, and LISA are capable of achieving results comparable to or even better than high-intelligent models without introducing any additional models or data during inference. This further demonstrates the value of VISUAL-O1. 

6 

Table 3: **Ambiguous instruction results on different intelligence level.** VISUAL-O1 has demonstrated powerful flexibility across different levels of intelligence. 

|MODEL|VISUAL-O1|GIOU|CIOU|
|---|---|---|---|
|LISA (LAI ET AL., 2023)||0_._0237|0_._0272|
|VISUAL-O1 (LISA)|EMPIRICAL|**0**_._**1088**|**0**_._**1209**|
|VISUAL-O1+ (LISA)|INSTANTIAL|**0**_._**1447**|**0**_._**1351**|
|SOM (YANG ET AL., 2023)||0_._0237|0_._0272|
|VISUAL-O1_−_(SOM)|EMPIRICAL|**0**_._**1143**|**0**_._**1530**|
|VISUAL-O1 (SOM)|INSTANTIAL|**0**_._**1278**|**0**_._**1722**|



**General instructions understanding.** While the ability to understand ambiguous instructions is crucial, the performance of the model on general datasets is also important. As shown in the right part of Tables 1 and 2, we also find that VISUAL-O1 significantly enhances the understanding of ambiguous instructions and greatly improves performance on general datasets. This is because improving the ability to understand instructions is beneficial even for those that do not contain ambiguity. What is even more noteworthy is that traditional COT and description-based synthesis like FUDD significantly degrade the model’s performance on general datasets. This is because general-intelligent models like LLAVA find it challenging to maintain the logical chains and longtext comprehension akin to high-intelligent large models like GPT-4O. Complex reasoning tends to impair the model’s understanding capabilities. 

For more analysis on improvement, please refer to the **Appendix** C. 

## 4.3 GENERALIZABILITY STUDIES 

## 4.3.1 FLEXIBILITY TO INTELLIGENCE LEVELS 

Based on the strong long-text comprehension and reasoning abilities of GPT-4V’s SOM, we use VISUAL-O1 for instantial experience during the inference stage for each sample. In contrast, due to LISA’s weaker reasoning abilities, we utilize VISUAL-O1 for empirical experience during the one-time optimization phase, avoiding complex reasoning for each sample during the inference stage. However, this does not mean that VISUAL-O1 cannot perform empirical experience for highintelligence models or instantial experience for general-intelligence models. To test VISUAL-O1’s adaptability to different intelligence levels, we design additional models: VISUAL-O1[+] (LISA) using instantial experience and VISUAL-O1 _[−]_ (SOM) using empirical experience. 

As shown in Table 3, the variants of VISUAL-O1 are effective across different models. It is noteworthy that since LISA cannot comprehend and generate long texts in logical chains, we use GPT-4V as its disambiguation model during the inference stage to supplement its capabilities. Whether applied to high-intelligent models or general-intelligent models, the variants of Visual-O1 significantly enhance performance on ambiguous instructions. 

For more analysis on intelligence levels, please refer to the **Appendix** D. 

## 4.3.2 ADAPTABILITY TO MODELS 

Can VISUAL-O1 generalize to different models? We select various models on RIS to verify this, and as shown in Table 4, regardless of the model used, we observe significant improvements. This highlights the generalizability of VISUAL-O1 to different models and suggests the potential for broader applications of our method. 

For more analysis on adaptability to different models and tasks, please refer to the **Appendix** E. 

## 4.4 ABLATION STUDIES 

## 4.4.1 EFFECTIVENESS OF COMPONENTS 

To further explore the effectiveness of VISUAL-O1, we conduct ablation experiments on the general VQA data in Table 5. Reasoning and reflection are crucial for the model to correctly understand 

7 

Table 4: **Generalization studies on different models.** VISUAL-O1 is effective across different models. “-” indicates values not reported in the original paper. 

|MODEL|GIOU|CIOU|
|---|---|---|
|OVSEG(LIANG ET AL., 2023)|0_._0418|0_._0778|
|REF-DIFF(NI ET AL., 2023)|0_._3301|0_._3140|
|UNIFIED-IO (LU ET AL., 2022)|_−_|0_._4015|
|INSTRUCTDIFFUSION(GENG ET AL., 2023)|_−_|0_._3904|
|SOM (YANG ET AL., 2023)|0_._3507|0_._4154|
|LISA (LAI ET AL., 2023)|0_._4654|0_._4721|
|GPT-4O(OPENAI, 2024)|0_._5728|0_._5152|
|QWEN-VL-2 (WANG ET AL., 2024)|0_._4417|0_._3220|
|VISUAL-O1 (SOM)|**0**_._**3602**|**0**_._**4566**|
|VISUAL-O1 (LISA)|**0**_._**4794**|**0**_._**5004**|
|VISUAL-O1 (GPT-4O)|**0**_._**5777**|**0**_._**5325**|
|VISUAL-O1 (QWEN-VL-2)|**0**_._**5161**|**0**_._**4457**|



Table 5: **Ablation studies.** Each module of VISUAL-O1 plays an indispensable role. 

|MODEL|ACC|BLEU-1|
|---|---|---|
|VISUAL-O1 (LLAVA)|**57**_._**38**|**0**_._**7329**|
|W/ORESPONSESYNTHESIS|47_._59|0_._6589|
|W/OREASONING ANDREFLECTION|54_._25|0_._7076|
|VISUAL-O1 (GPT-4O)|**63**_._**25**|**0**_._**8339**|
|W/ORESPONSESYNTHESIS|60_._00|0_._8254|
|W/OREASONING ANDREFLECTION|60_._60|0_._7943|



Table 6: **Efficiency of reasoning in VISUAL-O1.** Automatic reasoning surpasses manually designed experience, and an appropriate budget can help improve the performance of VISUAL-O1. 

|METHOD|ACC|BLEU-1|
|---|---|---|
|VISUAL-O1 (LLAVA) W/OBUDGET|54_._25|0_._7076|
|W/ HUMANEXPERIENCE|54_._99|0_._7073|
|W/ 1 BUDGET|55_._87|0_._7190|
|W/ 2 BUDGET|54_._16|0_._7049|
|W/ 3 BUDGET|**57**_._**38**|**0**_._**7329**|
|W/ 4 BUDGET|55_._27|0_._7147|
|W/ 5 BUDGET|55_._56|0_._7167|



ambiguous instructions. The model’s performance on ambiguous data significantly decreases without reasoning and reflection. Meanwhile, response synthesis is very important for the performance on regular data from non-ambiguous instructions, as the original instruction may contain important information, and response synthesis ensures the complete transmission of this information. Every module of VISUAL-O1 significantly improves the model’s performance across different datasets. 

For more ablations on components, please refer to the **Appendix** F. 

## 4.4.2 EXPLORATION OF REASONING 

We design additional validation experiments to further prove the effectiveness of VISUAL-O1’s reasoning process. We invite 10 volunteers to manually design disambiguation empirical experiences, which are used as benchmarks for ten tests and averaged. As shown in Table 6, manually designed prompts are significantly weaker than reasoned empirical experience, proving the rationality of VISUAL-O1’s design. 

We also show the influence of budgets in the reasoning and reflection process of VISUAL-O1. The score is the lowest without reasoning and reflection, even lower than with manually designed prompts. However, with the reasoning and reflection process, the performance gradually grows and 

8 

Table 7: **Comparison with data augmentation.** Compared to resource-intensive data augmentation, VISUAL-O1 still achieves significant advantages. 

|METHOD|ACC|BLEU-1|
|---|---|---|
|LLAVA W/ AMBIGUOUSEXTRADATA|51_._48|0_._6863|
|LLAVA W/ AMBIGUOUSORIGINALDATA|45_._08|0_._6363|
|LLAVA W/ NOISEDORIGINALDATA|39_._84|0_._5912|
|VISUAL-O1 (LLAVA)|**57**_._**38**|**0**_._**7329**|



then falls again after reaching its peak. We observe that VISUAL-O1 follows a trend similar to deep learning. The performance begins to grow in the early stages of learning, but then decreases due to overfitting after reaching a particular stage. 

For more explorations on reasoning, please refer to the **Appendix** G. 

## 4.4.3 COMPARISON OF DATA AUGMENTATION 

We generate an additional 10,000 data samples for fine-tuning LLAVA, as shown in Table 7. We design three different data augmentation methods: (1) AMBIGUOUS EXTRA DATA: directly using LLAVA to synthesize an extra 10,000 sets of ambiguous instruction data; (2) AMBIGUOUS ORIGINAL DATA: rewriting 10,000 sets of original instructions into ambiguous instructions, then training the model; and (3) NOISED ORIGINAL DATA: randomly deleting or modifying 10,000 sets of original instructions, then training the model. 

We find that due to the synthetic data’s inability to effectively simulate instructions, there is a significant performance drop in VQA. Additionally, the annotation and training costs brought by the 10,000 data samples are also very high. Therefore, VISUAL-O1 effectively solves the problem of understanding ambiguous instructions while maintaining performance on the general dataset. 

For comparisons of computational cost, please refer to the **Appendix** H. 

## 4.5 CASE STUDIES 

How does our method help the model better understand ambiguous instructions? To delve into this issue, we visualize the results of RIS in Figure 3. We find that ambiguous instructions are common in the data and often not easily detected by humans, but this can severely affect the model’s performance. After introducing VISUAL-O1, the accuracy of the instructions significantly improves, resulting in more effective outcomes. 

As shown in the first example, this is a typical case of _subjectivity_ ambiguity, requiring the perspective of human subjective observation for reasonable inference. The original instruction could describe most of the bears in the image, as each bear’s angle has some deviation. However, when combined with the image, it is found that only one bear meets the requirements of turning and slight deviation, and VISUAL-O1 successfully infers this and provides an accurate description, enabling the task model to accurately segment the target. 

In the second example, this is a typical case of _ellipsis_ ambiguity, where many sentence components are omitted. The segmentation model is misled by the word “reading” and incorrectly segments the person in the image. Meanwhile, VISUAL-O1 accurately describes that this is a bus showing a “reading station” sign and even provides additional information to help the segmentation model locate the target. 

In the third example, this is a typical case of _relativity_ ambiguity, where comparison is used to refer to specific entities. To understand the instruction, the model must first locate the comparison object of the bottle’s lighter color, which is the color of other bottles, that is, the lightest-colored bottle. When combined with VISUAL-O1, the instruction is transformed into a more accurate form, enabling the model to easily locate the target. 

In the fourth example, this is a typical case of _colloquialism_ ambiguity, using common positional phrases in spoken language. The model cannot understand that “1 o’clock” is a position, so it cannot select the correct object, but VISUAL-O1 transforms it into a clearer description. 

9 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning_images/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning.pdf-0010-00.png)


**----- Start of picture text -----**<br>
Image LISA Visual-O1 (LISA)<br>Answer Answer Clear Instruction<br>White plush bear with a<br>red bow and a heart on<br>its chest, positioned to<br>the right side of the<br>image, surrounded by<br>red plush bears.<br>Ambiguous Instruction white bear turned slightly<br>Answer Answer Clear Instruction<br>The green and blue<br>double-decker bus with<br>"Reading Station" and<br>the  number  "144"<br>displayed on the front.<br>Ambiguous Instruction reading station bus<br>Answer Answer Clear Instruction<br>The second bottle from<br>the left in the front row,<br>which has a lighter<br>amber color compared<br>to the adjacent darker<br>brown bottles<br>Ambiguous Instruction botlle thats next to main bottle lighter color<br>Answer Answer Clear Instruction<br>The giraffe positioned to<br>the upper right of the<br>feeding basket, with its<br>head slightly raised and<br>mouth full of hay.<br>Ambiguous Instruction giraffe at 1 o clock<br>**----- End of picture text -----**<br>


Figure 3: **Case studies on RIS.** Our approach aids the model in understanding ambiguous instructions by incorporating VISUAL-O1, which significantly improves the accuracy of instructions, thus enabling more effective segmentation of the target. 

For more cases, please refer to the **Appendix** I. 

## 5 CONCLUSION 

Even high-intelligent large models exhibited significant performance limitations on ambiguous instructions, where weak reasoning abilities of disambiguation could lead to catastrophic errors. We proposed VISUAL-O1, a multi-modal multi-turn chain-of-thoughts reasoning framework. It simulated human multi-modal multi-turn reasoning, providing instantial experience for high-intelligent models or empirical experience for general-intelligent models to understand ambiguous instructions. Unlike traditional methods that required models to possess high intelligence to understand long texts or perform lengthy complex reasoning, VISUAL-O1 did not significantly increase computational overhead and was more general and effective, even for general-intelligence models. We validated our approach across various tasks and models with different intelligence levels. Experimental results demonstrated that VISUAL-O1 significantly improved the performance of models of varying intelligence on ambiguous instructions and enhanced their performance on general datasets. Our work revealed the potential of AI to operate like humans in real-world scenarios characterized by uncertainty and ambiguity. 

For statements of broader impact, limitations, and reproducibility, please refer to the **Appendix** J, K, and L. 

10 

## REFERENCES 

- Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko S¨underhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 3674–3683, 2018. 

- Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In _Proceedings of the IEEE international conference on computer vision_ , pp. 2425–2433, 2015. 

- Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. 2023. 

- Jeffrey P Bigham, Chandrika Jayant, Hanjie Ji, Greg Little, Andrew Miller, Robert C Miller, Robin Miller, Aubrey Tatarowicz, Brandyn White, Samual White, et al. Vizwiz: nearly real-time answers to visual questions. In _Proceedings of the 23nd annual ACM symposium on User interface software and technology_ , pp. 333–342, 2010. 

- Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. _arXiv preprint arXiv:2310.09478_ , 2023a. 

- Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. _arXiv preprint arXiv:2311.12793_ , 2023b. 

- Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose visionlanguage models with instruction tuning. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Reza Esfandiarpoor and Stephen H Bach. Follow-up differential descriptions: Language models resolve ambiguities for image classification. _arXiv preprint arXiv:2311.07593_ , 2023. 

Weixi Feng, Tsu-Jui Fu, Yujie Lu, and William Yang Wang. Uln: Towards underspecified visionand-language navigation. _arXiv preprint arXiv:2210.10020_ , 2022. 

- Zigang Geng, Binxin Yang, Tiankai Hang, Chen Li, Shuyang Gu, Ting Zhang, Jianmin Bao, Zheng Zhang, Han Hu, Dong Chen, et al. Instructdiffusion: A generalist modeling interface for vision tasks. _arXiv preprint arXiv:2309.03895_ , 2023. 

- Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 3608–3617, 2018. 

- Yicong Hong, Qi Wu, Yuankai Qi, Cristian Rodriguez-Opazo, and Stephen Gould. Vln bert: A recurrent vision-and-language bert for navigation. In _Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition_ , pp. 1643–1653, 2021. 

- Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_ , 2021. 

- Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In _Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)_ , pp. 787–798, 2014. 

- Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. _arXiv preprint arXiv:2308.00692_ , 2023. 

11 

Jialu Li and Mohit Bansal. Improving vision-and-language navigation by generating future-view image semantics. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 10803–10812, 2023. 

- Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted clip. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 7061–7070, 2023. 

- Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2024. 

- Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- Minheng Ni, Yabo Zhang, Kailai Feng, Xiaoming Li, Yiwen Guo, and Wangmeng Zuo. Ref-diff: Zero-shot referring image segmentation with generative models. _arXiv preprint arXiv:2308.16777_ , 2023. 

OpenAI. Gpt-4o. https://chat.openai.com, 2024. 

- Archiki Prasad, Elias Stengel-Eskin, and Mohit Bansal. Rephrase, augment, reason: Visual grounding of questions for vision-language models. _arXiv preprint arXiv:2310.05861_ , 2023. 

- Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. Generative adversarial text to image synthesis. In _International conference on machine learning_ , pp. 1060–1069. PMLR, 2016. 

- Karsten Roth, Jae Myung Kim, A Koepke, Oriol Vinyals, Cordelia Schmid, and Zeynep Akata. Waffling around for performance: Visual classification with random words and broad concepts. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 15746–15757, 2023. 

- Damien Teney, Peter Anderson, Xiaodong He, and Anton Van Den Hengel. Tips and tricks for visual question answering: Learnings from the 2017 challenge. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 4223–4232, 2018. 

- Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. _arXiv preprint arXiv:2409.12191_ , 2024. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. _arXiv preprint arXiv:2310.11441_ , 2023. 

- Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. _arXiv preprint arXiv:2210.03629_ , 2022. 

- Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 6720–6731, 2019. 

- Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason Corso, and Jianfeng Gao. Unified vision-language pre-training for image captioning and vqa. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 34, pp. 13041–13049, 2020. 

12 

This appendix mainly contains: 

- Supplementary implementation details in Section A 

- Supplementary dataset details in Section B 

- Deeper analysis on improvement in Section C 

- Further generalizability verification on intelligence in Section D 

- Further generalizability verification on various tasks in Section E 

- Additional ablation of components in Section F 

- Additional ablation of reasoning in Section G 

- Computational overhead in Section H 

- Extra cases in Section I 

- Statement of broader impact in Section J 

- Statement of limitations in Section K 

- Statement of reproducibility in Section L 

## A SUPPLEMENTARY IMPLEMENTATION DETAILS 

VISUAL-O1 uses the following prompts for high-intelligent and general-intelligent models during inference. For each prompt, the upper part is used by the high-intelligent model, while the lower part is used by the general-intelligent model. 

## **Prompt of Reasoning** _p_ rsn 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning_images/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning.pdf-0013-16.png)


**----- Start of picture text -----**<br>
You are a helpful assistant in normal conversation.<br>{ task description }<br>Follow these instructions carefully:<br>1. Read the given question carefully and reset counter<br>between budget<br>2. Generate a detailed, logical step-by-step solution.<br>3. Enclose each step of your solution within reasoning tags.<br>4. You are allowed to use at most budget steps (starting<br>budget), keep track of it by counting down within tags<br>budget , STOP GENERATING MORE STEPS when hitting 0. You<br>don’t have to use all of them.<br>Example format:<br>starting budget<br>Content of step 1<br>remaining budget<br>Content of step 2<br>remaining budget<br>Content of step 3 or Content of some previous step<br>remaining budget<br>...<br>remaining budget<br>Content of final step<br>**----- End of picture text -----**<br>


13 

Description: _{_ ambiguous instruction _}_ Provide a detailed, step-by-step solution to a given question. 

_{_ experience _} {_ ambiguous instruction _}_ 

Since the reasoning process of the high-intelligent model occurs during the inference stage, to accelerate this stage, we combine all the prompts, allowing the model to complete the entire reasoning process in a single output. To clearly demonstrate, we split the prompt used by the high-intelligent model into _p_ rsn, _p_ rfl, and _p_ syn. 

## **Prompt of Reflection** _p_ rfl 

Follow these instructions carefully: 

5. Do a self-reflection when you are unsure about how to proceed; based on the self-reflection and reward, decide whether you need to return to the previous steps. 

6. Provide a critical, honest, and subjective self-evaluation of your reasoning process within <reflection> and </reflection> tags. 

7. Assign a quality score to your solution as a float between 0.0 (lowest quality) and 1.0 (highest quality), enclosed in <reward> and </reward> tags. 

8. If the image or question is not clear enough, you need to reflect and try to get answers from the unclear image or question. 

Example format: 

<reflection> [Evaluation of the solution] </reflection> <reward> [Float between 0.0 and 1.0] </reward> 

According to the instruction you generated last time, the annotator has rewritten _{_ ambiguous instruction _}_ as _{_ clear instruction _}_ . Please correct or rewrite your instruction based on the image situation. The image and the result of the data annotator are only for your evaluation. Please do not include the specific case in the instructions. You need to generate the full instruction even if no change is needed. 

If the annotator cannot find it, please let the annotator guess the one with the highest probability. 

Make sure the annotator only responds to the rewritten phrase and does not include any other thing. 

Instruction: _{_ experience _}_ 

## **Prompt of Response Synthesis** _p_ syn 

After completing the solution steps, reorganize and synthesize the steps into the final answer within <answer> and </answer> tags. The final answer cannot be empty. 

14 

Table H: **Distribution of dataset.** 

||Table H:**D**|**istribution of da**|**taset.**||
|---|---|---|---|---|
|ELLIPSIS|COLLOQUIALISM|SUBJECTIVITY|RELATIVITY|OTHER|
|23_._3%|27_._3%|3_._3%|29_._3%|16_._7%|



Table I: **The improvements on different types of ambiguity.** 

|METHOD|ELLIPSIS|COLLOQUIALISM|SUBJECTIVITY|RELATIVITY|OTHER|
|---|---|---|---|---|---|
||GIOU<br>CIOU|GIOU<br>CIOU|GIOU<br>CIOU|GIOU<br>CIOU|GIOU<br>CIOU|
|LISA<br>0_._0424 0_._0460<br>VISUAL-O1**0**_._**1742 0**_._**1704 **||0_._0107<br>0_._0124<br> **0**_._**1662 0**_._**1375**|0_._0025 0_._0081<br>**0**_._**0826 0**_._**2637 **|0_._0113 0_._0119<br> **0**_._**1444 0**_._**1207 **|0_._0490 0_._0633<br> **0**_._**0763 0**_._**0490**|



Disambiguated question: _{_ clear instruction _}_ Original question: _{_ ambiguous instruction _}_ 

## B SUPPLEMENTARY DATASET DETAILS 

For RIS, we use the REFCOCO+ dataset (Kazemzadeh et al., 2014); for VQA, we use the VIZWIZ dataset (Gurari et al., 2018). For their ambiguous instructions, we manually screen and construct subsets of 150, 650, and 106, respectively. Specifically, we use GPT-4 for an initial screening, selecting 2 _,_ 000 sets of potentially ambiguous instructions from the original dataset. Then, we enlist multiple volunteers to manually screen for instructions that clearly contain ambiguity. 

We categorize the ambiguous instructions into five distinct types based on the underlying causes of the ambiguity: _ellipsis_ , _colloquialism_ , _subjectivity_ , _relativity_ , and _other_ . The distribution of ambiguities can be found in Table H. Their detailed explanations are as follows: 

- _Ellipsis_ indicates that the ambiguity stems from omitted content, where essential information is left out, leading to uncertainty about the intended meaning. 

- _Colloquialism_ refers to ambiguity arising from using informal or imprecise expressions that may not be universally understood or may vary in interpretation across different contexts. 

- _Subjectivity_ indicates ambiguities due to subjective judgments, where personal opinions or individual perspectives cause unclear or varied interpretations. 

- _Relativity_ indicates that ambiguity comes from implied comparisons, where the meaning depends on an unstated reference point or context, making the instruction open to multiple interpretations. 

- _Other_ encompasses all other types of ambiguity that do not fit neatly into the previously mentioned categories, covering a broad range of miscellaneous sources of confusion. 

We will release the data we set up under the MIT license. 

## C DEEPER ANALYSIS ON IMPROVEMENT 

To deeply analyze how VISUAL-O1 mitigates ambiguity issues, we separately calculate the original scores and the scores after using VISUAL-O1 for each ambiguity category in RIS, as shown in Table I. We find that, in the original case, LISA performs relatively best in _ellipsis_ , because _ellipsis_ is the relatively simplest category, while the scores for the other categories are almost all around 1. After using Visual-O1, we observe significant improvements in scores across various categories, especially in _colloquialism_ and _relativity_ . This is because these two can be more easily converted into clear instructions through visual context. The significant improvements in other more challenging categories also signify the effectiveness of VISUAL-O1. 

15 

Table J: **General instruction results on different intelligence level.** 

|MODEL|VISUAL-O1|GIOU|CIOU|
|---|---|---|---|
|LISA (LAI ET AL., 2023)||0_._4654|0_._4721|
|VISUAL-O1 (LISA)|EMPIRICAL|**0**_._**4794**|**0**_._**5004**|
|VISUAL-O1+ (LISA)|INSTANTIAL|**0**_._**4985**|**0**_._**5188**|
|SOM (YANG ET AL., 2023)||0_._3507|0_._4154|
|VISUAL-O1_−_(SOM)|EMPIRICAL|**0**_._**3772**|**0**_._**4336**|
|VISUAL-O1 (SOM)|INSTANTIAL|**0**_._**3602**|**0**_._**4566**|



Table K: **Generalization studies on VLN.** 

|Table K:**Generalizat**|**ion studi**|**es on VL**|**N.**|
|---|---|---|---|
|MODEL|SR|SPL|NAVIERROR|
|VLN-SIG (LI& BANSAL, 2023)|4_._72|4_._59|7_._95|
|VISUAL-O1 (VLN-SIG)|**26**_._**42**|**22**_._**66**|**5**_._**99**|



Table L: **Ablation studies.** 

|Table L:**Ablation**|**studies.**||
|---|---|---|
|MODEL|ACC|BLEU-1|
|VISUAL-O1 (LLAVA)|**57**_._**38**|**0**_._**7329**|
|W/OMULTI-EXAMPLES|55_._75|0_._7092|
|W/OMULTI-MODALITIES|55_._90|0_._7143|



## D FURTHER GENERALIZABILITY VERIFICATION ON INTELLIGENCE 

We not only compare the results on ambiguous instructions but also extend the comparison to general instructions to better confirm the generalization capability of VISUAL-O1. In Table J, we can also see that all variants of VISUAL-O1 maintain and even improve performance on general instructions. This indicates that the design of VISUAL-O1 is highly flexible and can be broadly applied to various intelligence levels of models, thereby improving the whole system’s reasoning efficiency. 

## E FURTHER GENERALIZABILITY VERIFICATION ON EXTRA TASKS 

VISUAL-O1 demonstrates remarkable generalizability across various tasks. To further validate this, we apply VISUAL-O1 to two complex multi-modal tasks: image synthesis in the visual synthesis field and vision-and-language navigation (VLN) in the robotics field. Since models for these types of tasks do not have direct language output capabilities, we use GPT-4O for parts requiring intermediate language output. For VLN, we use the valid unseen split of the ROOM-TO-ROOM dataset (Anderson et al., 2018). As detailed in Table K, our observations reveal substantial performance enhancements across all tasks, underscoring the versatility and robustness of VISUAL-O1. We also provide examples in Figure D. For image synthesis, we choose the state-of-the-art model DALL-E 3. As shown in Figure E, we notice that even the most advanced models often misunderstand ambiguous instructions in human interactions, whereas VISUAL-O1 significantly alleviates this issue. This broad applicability paves the way for extending our method to a wider array of applications, showcasing its potential to improve performance in diverse contexts. 

## F ADDITIONAL ABLATION OF COMPONENTS 

We further verify the effectiveness of VISUAL-O1’s one-time optimization design details in Table L. We find that without multiple examples, _i.e._ , only allowing optimization for the same sample leads to overfitting and ineffective optimization. Without multiple modalities, _i.e._ , lacking visual contexts, VISUAL-O1’s optimization capability also declines, as visual contexts help the optimizer determine the current state of optimization. By avoiding all these issues, it demonstrates excellent capabilities. 

16 

~~**Image VLN-SIG Visual-O1 (VLN-SIG)**~~ Answer Answer Clear Instruction **Success:** no **Success:** yes _Move forward towards the open_ **Nav Error:** 4.1363 **Nav Error:** 1.7766 _door that you see straight ahead._ **SPL:** 0.00 **SPL:** 1.00 _As you pass through the door, you will see a staircase to your right. Ascend the stairs and proceed until you reach the area set up for dining. Stop there._ Ambiguous Instruction _Please guide me up the stairs and stop once we reach where we eat._ Answer Answer Clear Instruction **Success:** no **Success:** yes _Move forward, heading towards_ **Nav Error:** 4.6138 **Nav Error:** 0.0000 _the door past the foot of the bed._ **SPL:** 0.00 **SPL:** 0.99 _Once you reach the door, proceed through it to enter the hallway. Look for the nearest place to wash hands, which could be a bathroom or a sink, and go there to wash your hands. If the exact location to wash hands is not immediately visible, continue in the general direction along the hallway and turn right at the next opportunity to look for a bathroom or sink._ Ambiguous Instruction _Guide me past the bed to the hall, take me to wash hands._ 

## Figure D: **Cases of Visual-O1 on VLN.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning_images/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning.pdf-0017-02.png)


**----- Start of picture text -----**<br>
Answer Answer Answer Answer<br>Answer Answer Answer Answer<br>Ambiguous Instruction Ambiguous Instruction Ambiguous Instruction Ambiguous Instruction<br>A long vehicle with  Large animal with long  Bright circle high in Colored lights in sky<br>multiple  connected  nose  sprays  water,  sky, people on beach make patterns, people<br>sections moves along  small ones nearby. look up. watch at night.<br>tracks in an urban<br>setting.<br>DALL-E 3<br>Visual-O1 (DALL-E 3)<br>**----- End of picture text -----**<br>


Figure E: **Cases of Visual-O1 on Image Synthesis.** 

17 

## G ADDITIONAL ABLATION OF REASONING 

To better demonstrate VISUAL-O1’s reasoning process and reveal the specific content of VISUALO1’s reasoning, we show the detailed instantaneous and empirical experiences after each reasoning step of VISUAL-O1. The upper part is the instantaneous experience for the highly intelligent model, while the lower part is the empirical experience for the generally intelligent model. 

## **Experience Prompt** _A_ ins **and** _A_ emp 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning_images/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning.pdf-0018-03.png)


between two bears and not plate 

**Budget 7** Identify the objects marked as bears in the image. 

## **Budget 6** 

The bears in the image are part of the design on the mugs, which are marked with [7] and [3]. 

## **Budget 5** 

Locate objects that are not plates. In this image, the plate is a prominent object, but we need to identify objects that are not plates. 

## **Budget 4** 

Exclude the objects that are on the plate, such as the sandwich marked with [4], the soup bowl marked with [2], and the pickle marked with [8]. 

**Budget 3** Identify any objects that are between the two bears but are not on the plate. **Budget 2** It seems that the description provided was to find an object that is situated between the two bears but is not a plate itself. The steps taken so far have led to the exclusion of items on the plate and the identification of the bears on the mugs. <reward> 0.8 </reward> **Budget 1** Looking at the objects, the mugs marked with [7] and [3] are situated on either side of the image. The only object that is directly between them and not on the plate seems to be the salt or pepper shaker marked with [5]. **Budget 0** The object that fits the description "between two bears and not plate" is the salt or pepper shaker marked with [5]. The process involved a systematic exclusion of objects on the plate and identifying the bears on the mugs. The final object that fit the description was found to be between the bears (mugs) and not on the plate. The reasoning was sound, 

18 

Table M: **Computational overhead of Visual-O1.** 

|METHOD||STAGE|TIME|VRAM|
|---|---|---|---|---|
|LLAVA||LORAFINE-TUNING(HU ET AL., 2021)|25_._4439MIN|88797MB|
|VISUAL-O1|(LLAVA)|ONE-TIMEOPTIMIZATION|1_._5260MIN|16148MB|
|LLAVA||INFERENCE|0_._5103S|16102MB|
|VISUAL-O1|(LLAVA)|INFERENCE|0_._6547S|16370MB|
|GPT-4O||INFERENCE|4_._9403S|-|
|VISUAL-O1|(GPT-4O)|INFERENCE|6_._7614S|-|



and followed the instructions accurately. <reward> 1.0 </reward> 

**Budget 3** Repeat the question. 

**Budget 2** Please review the image provided and use its details to rephrase the ambiguous and blurry question into a clear and precise one that can be effectively answered. Ensure that your revised question is directly related to the content of the image. 

**Budget 1** Please ensure that your revised question is specific and directly references observable elements in the image. Your question should guide the respondent in addressing particular details or aspects present in the image clearly despite any blurriness. Avoid general or vague terms and aim for specificity that will elicit a precise answer. 

**Budget 0** When rephrasing the question, focus on the discernible elements in the image, such as text, icons, or specific features visible on the computer monitor. Your question should ask for details about these specific elements, avoiding any reference to the clarity of the picture or the physical location, as these are not relevant to the content displayed on the screen. Aim to formulate a question that inquires about the information or processes shown in the image, which can be answered with the visible data. 

As the reasoning progresses, in the instantial experience, VISUAL-O1 begins to perform increasingly in-depth reasoning. Meanwhile, in the empirical experience, VISUAL-O1 starts to generate more specific requirements, such as generating specific locations and elements, which are essential for accurately interpreting and generating unambiguous instructions. This also proves the effectiveness of VISUAL-O1’s optimization. 

## H COMPUTATIONAL OVERHEAD 

As shown in Table M, we analyze the computational overhead of VISUAL-O1. We find that VISUAL-O1’s overhead in both the optimization and disambiguation phases is low. Specifically, we also compare the time it takes for the vanilla LLAVA and GPT-4O to complete a VQA task. The computational overhead is comparable to LLAVA, further proving VISUAL-O1’s capability. 

19 

## I EXTRA CASES 

Similar to RIS, we also visualize examples on VQA. As shown in Figure F, VISUAL-O1 significantly improves the clarity of ambiguous instructions, aiding the task model in achieving correct and natural results. 

## J BROADER IMPACT 

With the development and application of AI, language instructions for interacting with AI are being applied in an increasing number of scenarios. In reality, humans often issue vague instructions for communication, and due to their natural visual ability, language and visual information complement each other, which further intensifies the ambiguity of instruction information. In this paper, we reveal the phenomenon of ambiguous instructions. Our proposed VISUAL-O1 significantly alleviates the pressure of ambiguous instructions on AI models, helps AI understand more natural instructions, broadens the application range of AI, and serves more non-professional people. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning_images/iclr_2025_Visual-O1_Understanding_Ambiguous_Instructions_via_Multi-modal_Multi-turn_Chain-of-thoughts_Reasoning.pdf-0020-04.png)


**----- Start of picture text -----**<br>
Image LLaVA Visual-O1 (LLaVA)<br>Answer Answer Clear Instruction<br>Response:  unanswerable Response:  usb What  device  is  currently<br>highlighted on the boot device<br>selection screen?<br>Ambiguous Instruction Once again, which device is highlighted now? Thank you so much for the help.<br>Answer Answer Clear Instruction<br>Response:  what is product Response:  fiber 1 What is the name of the<br>product shown in the image?<br>Ambiguous Instruction Now record a question. Read the text on the picture. And then double tap.<br>Answer Answer Clear Instruction<br>Response:  sgh connection Response:  sight connection What does the green shirt with<br>the word "sight" on it say?<br>Ambiguous Instruction What does this shirt say?<br>Answer Answer Clear Instruction<br>Response:  screen Response:  google What is the content displayed<br>on the computer screen?<br>Ambiguous Instruction What is this?<br>**----- End of picture text -----**<br>


Figure F: **Case studies on VQA.** 

20 

## K LIMITATIONS 

Despite VISUAL-O1 demonstrating strong reasoning capabilities and allowing general-intelligent models like LLAVA to perform inference without relying on large-scale high-intelligent models such as GPT-4O, it still requires the use of high-intelligent models during the one-time optimization phase. In the future, we will explore methods that do not rely on high-intelligent models even during optimization, in order to better extend our approach to scenarios with more limited resources. 

## L REPRODUCIBILITY STATEMENT 

We place a high emphasis on the reproducibility of our work. To facilitate this, we have provided a comprehensive set of implementation details and prompts in the appendix. Additionally, to further enhance the reproducibility of our results, we will release the source code and data. 

21 

