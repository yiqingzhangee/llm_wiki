# **DiPlomat: A Dialogue Dataset for Situated Pragmatic Reasoning** 

**Hengli Li**[1] _[,]_[2] **, Song-Chun Zhu**[1] _[,]_[3] _[,]_[4] _[,]_[5] **, Zilong Zheng**[1] 

1 Beijing Institute for General Artificial Intelligence (BIGAI) 

2 Yuanpei College, Peking University 

3 Institute for Artificial Intelligence, Peking University 

4 School of Intelligence Science and Technology, Peking University 

5 Department of Automation, Tsinghua University `lihengli@stu.pku.edu.cn, s.c.zhu@pku.edu.cn, zlzheng@bigai.ai` 

```
https://diplomat-dataset.github.io
```

If we are negotiating in public, we’re not really negotiating at all. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Diplomat_A_Dialogue_Dataset_for_Situated_PragMATic_Reasoning_images/nips_2023_Diplomat_A_Dialogue_Dataset_for_Situated_PragMATic_Reasoning.pdf-0001-10.png)


That’s correct. I’m out of the Senate three years and a half, but I was there for 36 years. Much of my work was budgeting. You would have had me on the TV so many times, and you would have said I don’t want to see you anymore. 

Well, I doubt that, Senator. 

**Which turn of the conversation has a pragmatic meaning?** _I choose_ 

“ Well, I doubt that, Senator. ” 

_Because_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Diplomat_A_Dialogue_Dataset_for_Situated_PragMATic_Reasoning_images/nips_2023_Diplomat_A_Dialogue_Dataset_for_Situated_PragMATic_Reasoning.pdf-0001-16.png)


**a. [A] is being polite and respectful towards [B].** 

b. [A]’s mind, political figures always tell lies, therefore he questions [B]’s words. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Diplomat_A_Dialogue_Dataset_for_Situated_PragMATic_Reasoning_images/nips_2023_Diplomat_A_Dialogue_Dataset_for_Situated_PragMATic_Reasoning.pdf-0001-19.png)


c. [A] doubts whether he would have had the senator on the TV. d. [A] thinks that he will not only be tired of [B]’s words but also of the [B]’s face. 

Figure 1: **Illustration of DiPlomat dataset.** Left: Example of a pragmatic conversation. Right: Pragmatic Identification and Reasoning task. 

## **Abstract** 

Pragmatic reasoning plays a pivotal role in deciphering implicit meanings that frequently arise in real-life conversations and is essential for the development of communicative social agents. In this paper, we introduce a novel challenge, **DiPlomat** , aiming at benchmarking machines’ capabilities on pragmatic reasoning and situated conversational understanding. Compared with previous works that treat different figurative expressions ( _e.g_ . metaphor, sarcasm) as individual tasks, **DiPlomat** provides a cohesive framework towards general pragmatic understanding. Our dataset is created through the utilization of Amazon Mechanical Turk (AMT), resulting in a total of 4 _,_ 177 multi-turn dialogues. In conjunction with the dataset, we propose two tasks, Pragmatic Identification and Reasoning (PIR) and Conversational Question Answering (CQA). Experimental results with state-of-the-art (SOTA) neural architectures reveal several significant findings: 1) large language models (LLMs) exhibit poor performance in tackling this subjective domain; 2) comprehensive comprehension of context emerges as a critical factor for establishing benign human-machine interactions; 3) current models defect in the application of pragmatic reasoning. As a result, we call on more attention to improve the ability of context understanding, reasoning, and implied meaning modeling. 

Preprint. Under review. 

## **1 Introduction** 

The fabric of human sociality is made up of complicated relations that evolve through different dimensions of interaction and communication channels [1, 2]. Social consensuses, such as social norms and values, are thereby formed between humans that convey meanings of individual minds, including beliefs, intentions and desires [3]. In a process of effective negotiation and social conversation, particularly, such social behaviors are only partly driven by **literal** meanings that are _objective_ , _rational_ and _explicit_ [4, 5]. Instead, these behaviors are commonly governed by affective or **pragmatic** meanings of dialogue utterances that refer to the emotional and cultural meanings of conversational partners and are _subjective_ , _emotional_ and _implicit_ . For instance in Figure 1, the lady responds with “I doubt that” rather than “I am not tired of you” to express a sense of humor and politeness. The competency of perceiving such _pragmatic_ meanings is crucial to social and emotional intelligence (EI) [6] and is referred to as **pragmatic reasoning** . 

The rapid developments of LLMs, such as ChatGPT and InstructGPT [7], have set off a wave of the next generation of conversational AI over the recent years. Despite the inspiring capabilities of language generation [8] and reasoning [9] achieved with massive computational resources and tremendous natural language data, LLMs barely show convincing communicative skills [10], _i.e_ ., they fail to capture pragmatic and ambiguous meanings of input prompts [11, 10, 12]. Critically, current neural generative models are trained to be objective with safe and satisfiable responses [13, 14], which largely deviates from the long-standing goal of building a human-like agent. Recently, Meta Research Team _et al_ . [5] introduce a ChatBot that demonstrates human-level play in a language board game _Diplomacy_ where lying and misleading commonly occur. However, their main focus is on game policy learning rather than pragmatic reasoning. 

**What are the core components of real-life conversational pragmatic reasoning?** Motivated by theories of cognitive linguistics and conversational modeling, we anticipate it to be three-fold: 

- _Situational Context_ Reasoning. Understanding pragmatic meaning requires a detailed understanding of conversational contexts. Consider the utterance “You are making the rest of us looking bad”, under different situations of praise and sarcasm, the sentence may convey completely opposite meanings. Furthermore, typical conversational reasoning challenges such as coreference resolution and intention prediction are largely dependent on the success of situated context modeling. 

- _Open-world Knowledge_ Acquisition. The open-world knowledge includes commonsense knowledge ( _e.g_ ., social ethics) that can be learned from different domains of dialogue corpus and domain-specific knowledge ( _e.g_ ., American histories). Successful pragmatic reasoning requires the acquisition of open-world knowledge and joint reasoning over the conversation. 

- Unified _figurative language_ understanding. Figurative language is one of the most frequently used tricks for conveying implicit meanings with subjective emotions. Previous works treat different forms of figurative language understanding as individual tasks, such as metaphors [15], idioms [16, 17], pun [18], _etc_ . Pragmatic reasoning provides a feasible unified perspective that considers all these tasks as recovering their literal meanings. 

In order to step towards a general human-like communicative agent, in this work, we introduce **DiPlomat** , a real-life **Di** a **lo** gue dataset that focuses on **P** rag **mat** ic reasoning. **DiPlomat** stems from an interview dataset [19], and experiences three steps of curation: automatic selection, fine-grained manual annotation and human refinement (Sec. 3). Our dataset owns 4 _,_ 177 dialogues and covers a vocabulary of 48 _,_ 900 words. More than that, human-annotated answers reach the amount of 6 _,_ 494 and hold a vocabulary size of 20 _,_ 000. Along with the dataset, we propose two tasks, Pragmatic Identification and Reasoning (PIR) and Conversational Question Answering (CQA), to benchmark machines’ pragmatic reasoning capabilities (Sec. 4). The CQA task possesses 19 _,_ 482 questions concerning the content of collected dialogues and the answers to the questions are written by humans. We run extensive experiments on previous state-of-the-art models on **DiPlomat** (Sec. 5). The best model achieves less than 0.70 accuracy score in PIR, and none of the models achieve more than 0.50 accuracy score for CQA. Moreover, we test previous pre-trained LLMs’ (including ChatGPT) zeroshot reasoning capability with a natural language inference (NLI) task. Regarding the experimental results provided, the significance of pragmatic reasoning speaks for itself. Throughout a thorough analysis of the limitations of current models, we aim to shed light on future research toward building general conversational agents. 

2 

Table 1: Comparisons on similar datasets and our dataset. QA: Question Answering. NUP: Next Utterance Prediction. NLI: Natural Language Inference. PI: Plausible Inference. IR: Implicature Recovery. PIR: Pragmatic Identification and Reasoning. 

|**Dataset**|**Domain**<br>**Manually**<br>**Task**<br>**Implicature**<br>**Reasoning**<br>**Multi-Turn**|
|---|---|
|Ubuntu [ACL 2015][20]<br>RACE [EMNLP 2017][21]<br>ARC [ArXiv 2018][22]<br>MNLI [NAACL 2018 ][23]<br>Persona-Chat [ACL 2018][24]<br>SWAG [EMNLP 2018][25]<br>Cosmos QA [EMNLP 2019][26]<br>CoQA [NAACL 2019][27]<br>DREAM [ACL 2019][28]<br>Dialogue NLI [ACL 2019][29]<br>DROP [ACL 2019][30]<br>MuTual [ACL 2020][31]<br>IMPPRES [ACL 2020][32]<br>GRICE [ACL 2021][12]|Technique<br>�<br>NUP<br>�<br>�<br>�<br>Open<br>�<br>QA<br>�<br>�<br>�<br>Science<br>�<br>QA<br>�<br>�<br>�<br>Open<br>�<br>NLI<br>�<br>�<br>�<br>Persona<br>�<br>NUP<br>�<br>�<br>�<br>Movie<br>�<br>PI<br>�<br>�<br>�<br>Persona<br>�<br>QA<br>�<br>�<br>�<br>Diverse<br>�<br>QA<br>�<br>�<br>�<br>Open<br>�<br>QA<br>�<br>�<br>�<br>Persona<br>�<br>NLI<br>�<br>�<br>�<br>Open<br>�<br>QA<br>�<br>�<br>�<br>Open<br>�<br>NUP<br>�<br>�<br>�<br>Open<br>�<br>NLI<br>�<br>�<br>�<br>Daily<br>�<br>IR & QA<br>�<br>�<br>�|
|**DiPlomat**|Open<br>�<br>PIR & QA<br>�<br>�<br>�|



## **2 Related Work** 

**Conversational Dataset** Table 1 provides a comparative analysis of our dataset with similar conversational datasets. The Dream [28] dataset formalizes dialogues from English exams into question-answering task with a focus on in-depth dialogue comprehension. With question-answer pairs compiled by two annotators, CoQA [27] focuses on reasoning in conversation understanding. By utilizing Chinese student English listening comprehension tests, MuTual [31] is built to address the issue of general dialogue reasoning. In contrast to these preceding datasets, the GRICE[12] dataset represents a significant advancement in the field of pragmatic reasoning as it incorporates implicature and reasoning. However, both MuTual and GRICE exhibit a shared limitation in that they do not possess data that closely resembles real-world interactions, leading to a lack of diversity in their respective datasets. Additionally, other relevant datasets, including commonsense [26], reasoning [21], and natural language inference (NLI) [29, 23], are also included in Table 1. By examining various dimensions such as domain, manual annotation, task variety, implicature, reasoning, and multi-turn interactions, our dataset offers unique advantages in addressing the challenge of pragmatic reasoning in dialogues. 

**Language Models for Dialogue Generation** Conversational AI has emerged as a prominent research area within the field of natural language processing (NLP), attracting significant attention and interest. Numerous pre-trained models have been proposed to tackle dialogue generation tasks, such as DialogGPT [33], GODEL [34], LaMDA [35], and Meena [36], and they have achieved marvelous results on competitions [37]. Furthermore, a pivotal milestone has been achieved with the advent of ChatGPT, garnering widespread interest and stimulating further investigation in the domain of conversational AI. ChatGPT, built upon the principles of transformer models [38], undergoes training on an extensive corpus of data, resulting in its profound efficacy. Notably, this system boasts an impressive magnitude of billions of parameters. In a notable study conducted by OpenAI [13], it has been demonstrated that the augmentation of parameters, referred to as the Scaling Law, substantially enhances the model’s capabilities. Also, the enlargement of the number of parameters triggers the emergence of miraculous ability. After the success of ChatGPT, more models such as PaLM 2 [39] appear in the field. 

**Pragmatics Reasoning** Pragmatic reasoning is a significant subject within the field of pragmatics, attracting considerable attention from linguists. Existing datasets for pragmatic reasoning predominantly focus on specific types of phenomena. For example, EPIE [16], PIE [17] center around idiomatic expressions, while MOVER [40] emphasizes hyperbole, and MERMAID [15] investigates metaphor usage. However, paronomasia is much more under-studied [18], with researchers frequently intertwining it with humor [41]. In a related vein, GRICE [12] endeavors to study implicature in a 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Diplomat_A_Dialogue_Dataset_for_Situated_PragMATic_Reasoning_images/nips_2023_Diplomat_A_Dialogue_Dataset_for_Situated_PragMATic_Reasoning.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Reasoning Type Dialogue Context Pragmatic Identification<br>A:  What’s remakrable about the Edenton Tea Party is that it was all women -  2 [nd]  turn of PersonA.<br>Contexual<br>possibly the first women-driven political protest in U.S. history. Rationale:  This is a humorous response<br>Reasoning B:  And I guess we can keep beating up on the British for just a moment here. indicating agreement with continuing to<br>(48.7%) A: Why not? criticize the British.<br>Figurative  A:  How has technology changed the way that vaccines are made? Is it easier  1 [st]  turn of PersonB.<br>Language  to sort of see responses in living things as they happen now? Rationale:  The implied meaning is that the<br>Reasoning B:   Yes, the technology has improved by  leaps and bounds . technology has improved by a large margin.<br>(26.6%)<br>A:  But what's the week been like? 1 [st]  turn of PersonB.<br>Commonsense B:  You know, hard for people in a country that's not used to heat and doesn't  Rationale:  The implied meaning is that the<br>Reasoning have much air conditioning. [...]  Does a chateau have to fall on your head  to  PersonB is emphasizing the weather is<br>(16.4%) see that the earth, France included, is getting hotter? getting hotter obviously.<br>External A:   Poland recently has gone a step further, legislating how to talk about one  1 [st]  turn of PersonB.<br>Knowledge  of the darest periods in its history. Rationale:  The law punishing those who<br>Reasoning B:   We are in a moment when we decide about  how we’re going to teach  suggest Poland was complicit in the stifle<br>(12.9%) about and remember the Holocause  in the future. discussion about atrocities.<br>A:  I am getting reports on Twitter. [...] That’s nothing we have that’s verifiable  1 [st]  turn of PersonB.<br>Others or has been verified yet. Rationale:  The implied meaning is that<br>(8.3%) B:  We have not confirmed those reports. And  Twitter, I mean, it’s a good  PersonB suggest not to trust Twitter as a<br>source of information, but we can’t report it until we confirm it. reliable source of information.<br>A:  It’s sometimes a good source of misinformation too, isn’t it?<br>**----- End of picture text -----**<br>


Figure 2: **DiPlomat** dataset samples. Each row illustrates an exemplar case with its reasoning type, dialogue context, pragmatic turn, and the corresponding rationale. Evidence that support the pragmatic identification are marked in orange. 

unified manner, but its data does not originate from real-world contexts, thereby lacking diversity and exhibiting conspicuous patterns. In addition to these datasets, Frank and Goodman [42] attempt to model pragmatic reasoning phenomenon through language games. Insights into the comprehension of figurative language is provided by works such as those by Stowe et al. [43] and Chakrabarty et al. [44], with the former specifically delves into the realm of metaphors and idioms, and the latter investigating idioms and similes. The successful completion of our task necessitates the incorporation of commonsense knowledge. Extensive scholarly efforts have been dedicated to addressing the challenge of leveraging commonsense in various works [31, 45–47]. 

## **3 The DiPlomat Dataset** 

## **3.1 Data Source Construction** 

The **DiPlomat** dataset stems from the Interview dataset [19], which consists of two subsets, a twoparty subset comprising 23,714 dialogues and a multi-party subset with 105,848 dialogues. Given our specific focus on conversations involving only two communicators, we exclusively utilize the two-party subset for our research purposes. The Interview dataset itself is a real-world collection of NPR radio transcripts, spanning a period of 20 years of NPR programs. The curation process for our dataset involved several stages, including automatic selection, fine-grained annotation, and human refinement. Details are provided as follows. 

**Step I. Automatic Selection.** The extensive size of the source dataset introduces redundancy, necessitating automatic measures to alleviate the burden of human annotation. Therefore, we employ algorithms and models to perform an initial filtering process. In order to establish a unified framework, we consider various types of pragmatic phenomena and utilize different techniques to extract relevant instances from the source dataset. For instance, we utilize the EPIE list [16] for a string-matching method to identify idioms in dialogues, and we train RoBERTa [48] on Hypo-XL [49] for hyperbole detection; refer to _Supplementary_ for more details. 

**Step II. Fine-grained Annotation.** We leverage AMT to conduct detailed annotation of pragmatic turns within our dialogues. Workers participating in the annotation task are instructed to select all turns that exhibit a divergence between their literal meaning and their intended or implied meaning. 

4 

Due to the subjective nature of pragmatic reasoning, we request the workers to provide confidence scores along with reasons for their choices. Specifically, in order to avoid introducing the extra difficulty of lengthy texts, the reasons are required to be no more than 8 words. All annotators shall meet the following criteria: (i) from English-speaking countries; (ii) Completion of a minimum of 1,300 tasks with a 98% approval rate. We also present them with detailed instructions and four examples so that workers shall be clear about our objectives and requirements. The instructions part outlines the step-by-step procedures for accomplishing the assigned tasks and highlights some key points that workers should pay attention to. The four examples offered are representative of classical pragmatic conversations drawn from the field of linguistics, serving as practical references for the workers. To mitigate the intricacies arising from the identities of dialogue communicators, a simplified representation is adopted, whereby the speakers are denoted as `PersonA` and `PersonB` . To ensure the reasonableness and quality of the data, we manually examined 30 data samples and blocked workers who are unqualified. As a result, a total of 5,869 dialogues are selected. Refer to the _Supplementary_ for the detailed annotation process. 

**Step III. Human Refinement.** In this process, tasks for workers are formulated as multiple-choice questions. Previously collected human-annotated reasons are transformed into choices, utilizing a template format: `[turn {turn_id}: {reason}]` . In addition, to mitigate the impact of careless workers, we introduce a disturbing choice for each gold choice. These disturbing choices are generated using BERTScore [50] by selecting the reason with the highest score from other unrelated dialogue. Of note, for each dialogue, an equal number of gold choices and disturbing choices are provided. Workers are requested to select all reasonable choices for each conversation and are warned of the presence of disturbing choices. Workers who frequently select disturbing choices are blocked, and their work is rejected. Through this refinement process, 1,692 dialogues are filtered out, while 4,177 dialogues are preserved, ensuring the integrity and reliability of our dataset; Refer to _Supplementary_ for more detail. 

## **3.2 Dataset Statistics** 

The **DiPlomat** dataset comprises a total of 4,177 multi-turn dialogues, with each dialogue averaging 4.1 turns. On average, there are 1.56 pragmatic turns per dialogue. The distribution of dialogues with different quantities of pragmatic turns is illustrated in Figure 4; see Table 2 for detailed dataset statistics. 

With respect to the motivation introduced in Sec. 1, we categorize the process of transitioning from dialogue to human-annotated rationales into five reasoning types: 

Table 2: Statistical feature of **DiPlomat** . 

|**# Dialogues**|4_._17_×_103|
|---|---|
|**Avg. Turns per Dialogue**<br>**Avg. Words per Turn**<br>**Avg.Human Reason per Dialogue**<br>**Avg. Words per Human Annotated Reason**|4_._10<br>42_._80<br>1_._56<br>25_._31|
|**Vocabulary Size (dialogue)**<br>**Vocabulary Size (human-annotated reasons)**|4_._89_×_104<br>2_._00_×_104|



- **Contextual Reasoning:** The comprehension of the context is paramount for this reasoning process. 

- **Figurative Language Reasoning:** Proficiency in understanding figurative language, such as idioms and metaphors, is indispensable for advancing this type of reasoning. 

- **Commonsense Reasoning:** The utilization of commonsense knowledge, such as recognizing that a chateau cannot fall from the sky, is vital for this category. 

- **External Knowledge Reasoning:** This form of reasoning necessitates knowledge that extends beyond commonsense and is not explicitly mentioned in the dialogue. 

- **Others:** This category includes pragmatic turns that fit none of the above. 

Figure 2 demonstrates the proportion of each type. The prevalence of data within the context partition prove the significance of context in pragmatic reasoning of real life. Figure 3a depicts a sunburst visualization illustrating the distribution of trigram words within pragmatic turns. The diverse range of trigram words indicates that the **DiPlomat** dataset enjoys the rich diversity from real-life corpora, and covers a wide array of topics. In addition, the recurring occurrence of the words “president‘’ and “world‘’ is observed, demonstrating **DiPlomat** ’s slight bias to politics and world-wide events. 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Diplomat_A_Dialogue_Dataset_for_Situated_PragMATic_Reasoning_images/nips_2023_Diplomat_A_Dialogue_Dataset_for_Situated_PragMATic_Reasoning.pdf-0006-00.png)


**----- Start of picture text -----**<br>
2500<br>2000<br>1500<br>1000<br>500<br>0<br>1 2 3 4 5<br>(a) Pragmatic Turns (b) Questions # of Pragmatic Turns<br>things first<br>time<br>thing<br>people<br>place<br>the<br>most toit's<br>get<br>is<br>of<br>world<br>world<br>do<br>and<br>other<br>the<br>with<br>a<br>US president<br>the<br>has<br>reason<br>this<br>program<br>the<br>to<br>course<br>president people<br>is<br>president<br>people<br>of<br>think<br>and<br>I<br>the<br>and<br>able<br>be<br>way<br>a I<br>thing<br>so<br>same<br>time<br>the I<br>then # of Dialogues<br>**----- End of picture text -----**<br>


Figure 3: Distribution of trigram words of pragmatic turns and questions. 

Figure 4: Distribution of dialogues _w.r.t_ . # of pragmatic turns. 

## **4 Task Definition** 

We propose 2 distinct tasks for our dataset: (i) Pragmatics Identification and Reasoning and (ii) Conversational Question Answering. The former task focuses on assessing the capability of models to identify the presence of pragmatic phenomena and their ability to select a suitable answer for such identification. The latter task aims to evaluate the models’ adeptness in employing pragmatic reasoning by presenting them with carefully designed questions. 

## **4.1 Task 1: Pragmatics Identification and Reasoning (PIR)** 

Inspired by previous work [51], we present the Pragmatic Identification and Reasoning (PIR) task. In this task, models are provided with dialogues and are required to identify turns whose actual meanings deviate from their literal interpretations, commonly referred to as pragmatic turns. If their selections are accurate, a set of rationales is presented and they are expected to choose the most plausible reason for each pragmatic turn. For each turn, there are 5 candidate reasons available, comprising one gold choice and four disturbing choices. The model’s success in this task depends on the precise execution of both steps. We consider three diagnostic settings to test the machine’s capability on pragmatic understanding: 

- Conversation _→_ Pragmatic Turn ( **C** _→_ **P** ). For each instance, models are presented with a dialogue and a specific turn extracted from that dialogue. They are then required to determine and whether the given turn qualifies as a pragmatic turn. Consequently, the dataset is flattened to a total of 17,129 instances, with each instance corresponding to a single turn. It’s important to highlight that turns without pragmatic meanings are also extracted for evaluation. 

- Conversation + Pragmatic Turn _→_ Rationale ( **CP** _→_ **R** ). In this subtask, we offer the model both the dialogue and the pragmatic turn and it needs to choose the most plausible rationale out of five candidate choices. 

- Conversation _→_ Pragmatic Turn + Rationale ( **C** _→_ **PR** ) In this subtask, models pre-trained on the previous two subtasks are combined to infer the final results. Specifically, the model obtained from the first subtask is utilized for determining pragmatic turns, while its version finetuned on the second subtask is employed for selecting the most suitable rationale. It is worth noting that, in contrast to **C** _→_ **P** , in this subtask, extracted turns are limited to pragmatic turns only. 

**Pragmatic Turns and Gold Choices** Recall that in our data collecting procedure outlined in Sec. 3.1, despite asking workers to select pragmatic turns, they are also instructed to fill in reasons to explain their choices. To simplify the evaluation process, the selected turns are directly utilized as answers for the first subtask, while the reasons provided by the workers serve as the designated correct choices (referred to as "gold choices") for the second subtask. 

**Disturbing Choice** As a result of the time-consuming nature of BERTScore [50], an alternative approach is adopted for measuring sentence similarity using Sentence-Transformers [52], which is a significantly faster method. In our methodology, for each gold choice, four alternative choices with high similarity scores are selected from the pool of gold choices to serve as disturbing answers. Despite 

6 

their high similarity scores, upon careful examination within the given context, it’s apparent that the disturbing answers convey entirely different meanings from the gold answer. This characteristic makes them appropriate components to build our task. 

## **4.2 Task 2: Conversational Question Answering (CQA)** 

The ability in applying pragmatic reasoning is crucial for effective communication and achieving a thorough grasp of the natural language system. To address this, we propose conversational questionanswering, wherein multiple questions are formulated for each dialogue, and an example is shown in Figure 5. The questions focus on delving deeper into dialogues, often necessitating insights into the intended meanings to answer. ChatGPT assumes a pivotal role in question generation and thanks to AMT, we can ensure the collection of high-quality answers. Ultimately, 19 _,_ 482 question-answer pairs are assembled. 

**Question Generation** ChatGPT is employed to generate questions with prompts consisting of dialogues and humanannotated reasons. We task it to generate questions challenging for individuals who are unaware of the dialogues’ intended meanings. More than that, for the convenience of evaluation, the question is also asked to be able to answer within one or two words. Furthermore, to ensure diversity, we instruct ChatGPT to initiate the questions with "What", "Which" or "How". A preliminary assessment is carried out by sampling a few examples out of the question pool to guarantee quality. 

**Answer Collection** The answers to the questions are obtained through the utilization of AMT. Each worker is provided with 

**A:** Finally, Ron, lots of talk about Congress releasing the second half of this $700 billion bailout this week. Where do we stand with that? 

**B:** Quite possible that Congress will get that done this week now that Barack Obama has asked George Bush, has asked the current president, to formally put in a request for that money. Congress has got a lot of questions about how this money is going to be spent, as it has questions about how the first half of the money was spent. 

**Rationale:** Barack Obama’s request for the $700 billion bailout may expedite the process. 

**Q** 1 **:** What may expedite the process? **A** 1 **:** Request **Q** 2 **:** Which president requested the $700 billion bailout to be released? **A** 2 **:** Obama 

Figure 5: Conversational Question Answering example. 

a dialogue along with several associated questions and it is requested to answer the questions in one single word. To minimize the potential for misinterpretation, we offer an example coming from our dataset, which is annotated by the author itself. Through the process of human annotation, we consistently evaluate the collected data and reject unqualified answers as well as block workers who fail to meet our standards. 

**Statistical feature of Questions** Figure 3b showcases the diverse range of our questions. These questions encompass a variety of sentence structures, starting with interrogative words: What, Which, and How, their following words vary a lot. Apart from the questions, the answer set holds a vocabulary size of 8,179, which is also of great diversity and raises a challenge for models. 

## **5 Experiment** 

## **5.1 Pragmatics Identification and Reasoning** 

For **C** _→_ **P** , we partitioned the dataset into distinct subsets for training, validation, and testing. The training set consists of 13,708 examples, surpassing the 1,361 instances in the validation set and the 2,060 instances in the test set in terms of size. Models are trained on the training set and their performance is evaluated on the validation set after each epoch to determine the optimal checkpoint. The best checkpoint is subsequently loaded for the final evaluation on the test dataset. The evaluation metric employed is the accuracy score, calculated as the ratio of correct predic- 

||Random<br>BERTbase|**C**_→_**P**<br>50<br>63.2_±_1.1|**CP**_→_**R**<br>20<br>91.3_±_0.7|**C**_→_**PR**<br>10<br>**50.2**_±_**6.8**|
|---|---|---|---|---|
||RoBERTabase|64.4_±_1.3|**92.0**_±_**0.4**|50.0_±_11.28|
||GPT-2base<br>DialoGPTmedium|64.4_±_0.7<br>**65.0**_±_**0.6**|90.9_±_0.9<br>24.5_±_1.9|13.06_±_1.1<br>3.8_±_1.5|



Table 3: Pragmatics Identification and Reasoning Results. The numerical results are accuracy scores in their percentage 

7 

tions to the total number of instances. Similarly, for the task of **CP** _→_ **R** , the dataset is also partitioned into training, validation, and test subsets. The respective sizes of these subsets are 5,188, 244, and 1,062 examples. The training and evaluation procedures are identical to those of the previous subtask. For **C** _→_ **PR** , the test sets in previous subtasks are taken for evaluation, and it’s worth noting that their test sets consist of exactly the same instances. This design ensures the prevention of any leakage of the test set into the training set, thereby maintaining the integrity of the evaluation process. 

**Results and Analysis** We present three key observations for this task: (1)The primary performance bottleneck lies in the subtask **C** _→_ **P** . As shown in Table 3, the models achieve an accuracy score of approximately 90 in the **CP** _→_ **R** subtask, indicating their capability to reason to some extent. However, when it comes to the **C** _→_ **PR** task, the best-performing model achieves only 50.2 accuracy, while the highest accuracy achieved in the **C** _→_ **P** task is 65.0. The substantial difference between the score of 90 and the score of 65.0 suggests that the difficulty in accomplishing the task primarily stems from the **C** _→_ **P** subtask. (2) Accumulated variance in the **C** _→_ **PR** subtask. The models exhibit significant variance in the results of the third subtask, which can be attributed to the variance introduced by its constituent tasks (3) The significance of pragmatic awareness in language models. Both the **C** _→_ **P** and **C** _→_ **PR** subtasks require pragmatic awareness, and the poor performances of the models on these subtasks highlight their limitations in accurately determining the optimal timing for deploying reasoning abilities. 

## **5.2 Conversational Question Answering** 

Similarly to the previous task, the question-answering dataset is divided into training, validation, and test sets, comprising 15,585, 1,559, and 2,338 instances, respectively. Experimental subjects includes BART [53], T5 [54], UnifiedQA [55], and mT5 [56]. The metric adopted is also accuracy score. Given ChatGPT’s impressive performance in MMLU and AI2 Reasoning Challenge [13], we further examine ChatGPT’s capability in the context of CQA by prompting it to provide one-word answers to questions. However, due to its uncontrollable nature, the generated answers may not always align with our desired settings. Hence, we introduce two evaluation metrics for ChatGPT: (1) em (exact match), which requires ChatGPT to produce the exact same word as our answer, and (2) pm (partially match), where we consider ChatGPT to be correct as long as our answer appears in its generated output. Two configurations are employed for evaluation. In the first configuration, models receive dialogues and questions while remaining blind to human-annotated rationales. The second configuration is a contrasting experiment with human-annotated rationales provided. For each configuration, we run each model three times using different random seeds and report the mean and variance of their results as the final outcomes. 

Table 4: CQA task results with/without human annotated rationales. The numerical results are accuracy scores in their percentage. em: exact match, pm: partially match. 

||BART-base|T5-small|mT5-small|UnifedQA-base|UnifedQA-large|ChatGPT (em)|ChatGPT (pm)|
|---|---|---|---|---|---|---|---|
|w/o rationale|20.2_±_1.1|24.9_±_0.7|19.7_±_0.4|28.9_±_2.0|32.8_±_0|1.0|**40.6**|
|w/ rationale|29.6_±_0.6|34.1_±_1.4|29.8_±_0.8|38.8_±_0.2|42.4_±_0.6|1.5|**45.1**|
|∆|+9.4 (46.5%)|+9.0 (35.8%)|**+10.1 (51.3%)**|+9.9 (34.1%)|+9.6 (29.3%)|+0.5 (60%)|+5 (12.3%)|



**Results and Analysis** The experimental results are presented in Table 4. Our observations can be categorized into three main aspects. Firstly, the importance of pragmatic meaning is evident. As shown in Table 5, there exists a notable disparity between the results of models that have access to humanannotated answers and those that do not. On average, the model performances improve by 38.47% after the introduction of human-annotated rationales. Even the lowest-performing model in the initial experiment, mT5-small, demonstrates a 9.4% increase in accuracy. The substantial discrepancy in results between the two configurations underscores the significance of elucidating intended meanings in the development of effective communicators. Second, the models display deficiencies in applying pragmatic reasoning. Our questions are designed to demand a deeper understanding of conversations, however, the models struggle to perform well on our task. The best-performing model, ChatGPT, achieves an accuracy score of 40.6%. It is worth noting that our questions are generated by ChatGPT itself, and our source dataset, Interview, was proposed prior to the emergence of ChatGPT, which means that ChatGPT may have encountered our text during training. These characteristics render its result unsatisfactory. Third, generalization across different types of pragmatic reasoning poses challenges. In this analysis, we focus exclusively on models other than ChatGPT due to the lack 

8 

of clarity regarding its training process. As demonstrated in Table 5, these models showcase a substantial improvement in performance following the inclusion of human-annotated rationale. The extent of this improvement exhibits slight fluctuations among the various models, suggesting a shared obstacle that hinders their overall performance. Noticed that the models are fine-tuned on a training set using a training set that is 5.2 times larger than the test set, thus we can conclude that achieving effective generalization from one pragmatic reasoning process to another remains a formidable and challenging task. 

## **5.3 Zero-Shot Natural Language Inference** 

The natural language inference task [57] evaluates the model’s comprehension of language. It involves providing two sentences: a premise and a hypothesis, and models are required to determine the relationship between the two sentences, which can be entailment, contradiction, or neutral. As there are no negative samples in our dataset, we simplify the task by asking the model to judge only whether there is entailment. . In this task, models are presented with a dialogue, a turn of the dialogue, and an intended meaning, they need to judge whether the turn entails the intended meaning. Noticed that collected data as described in Sec. 3.1 consists of reasons and implied meanings, to better fit our task, we abandon the reasons and preserve the implied meanings; refer to Appendix X. Models are tested under zero-shot setting, which means that they are not allowed to train on any data before testing. Thus the innate abilities of models play a decisive role. 

Baseline models include T5 [54], DeBERTa [58], and ChatGPT. It’s important to note that ChatGPT and the other two models are tested on different settings. ChatGPT is tested with the whole dialogue and the implied meaning as a prompt. However, to inspect the significance of context, the other two models are only provided with the pragmatic turn and the corresponding pragmatic meaning. ChatGPT is evaluated using two types of prompts: with and without step-by-step instructions. “ `Let’s think step by step` ” is a prompt discovered [59] to improve the model’s reasoning ability. 

**Results and Analysis** Results are listed in Table 5. As this task shares similar settings as binary classification, randomized answer accuracy is expected to be 50%. We observe below randomized performance on some previous SOTA models. Note that, each of the data is annotated by two humans, thus it’s reasonable to view human performance as 100%. ChatGPT achieves the highest result but still shows a huge gap compared with human annotations. The outcomes highlight the imperfectness of the models’ reasoning abilities. For T5-large and DeBERTa, context is blinded, but for ChatGPT, it is reachable. Hence, the performance gap among T5-large, DeBERTa, and ChatGPT shows the importance of context in our task. Interestingly, COT doesn’t offer help to ChatGPT but is harmful to ChatGPT’s performance. 

Table 5: Results of Natural Language Inference Task. _[⋄]_ : T5-XXL fine-tuned on true NLI mixure [60]. _[†]_ : DeBERTav3 trained on MNLI [23] and SNLI [61]. 

||**Method**<br>Random|**Acc.**<br>50.0|
|---|---|---|
||DeBERTa-Large_†_|44.3|
||T5-XXL_⋄_|45.3|
||ChatGPT<br>ChatGPT w/ CoT|**85.7**<br>63.8|



## **6 Discussions and Future Work** 

In this paper, we propose **DiPlomat** , a high-quality manually annotated multi-turn dataset of pragmatic reasoning in conversations. Along with the dataset, we propose two tasks and baselines. Comparing experimental results, we emphasize the nonnegligible impact of contexts and reasoning on building perfect communicators. We also highlight the importance of pragmatic awareness and its bottleneck effect on our tasks. There is still a large gap between current performances and qualified performances. 

**Memorization** _**vs**_ **. Reasoning** Noticed that models exhibit outstanding performance on **CP** _→_ **R** of Sec. 5.1. On the contrary, for **C** _→_ **PR** , models achieve poor results. Since the underlying knowledge is consistent for both tasks, the disparity in performance is hypothesized to be attributed to _memorization_ . Instead of truly understanding the knowledge, the models tend to memorize patterns. 

**Subjectiveness** _**vs**_ **. Objectiveness** Ji et al. [62] emphasize the importance of modeling a distribution that encompasses a diverse range of possibilities, rather than solely relying on a single “best” prediction. During the annotation process, we observe a phenomenon that different workers hold 

9 

diverse opinions regarding pragmatic turns and their intended meanings. Their annotations often exhibit significant variations, sometimes even presenting completely opposing interpretations. We maintain the possibility of subjectiveness with careful task metric design (Sec. 4.1). 

**Future Work** Achieving generalization across multiple types of pragmatic reasoning processes poses significant challenges. Consequently, we propose that the construction of a proficient communicator necessitates the incorporation of methods beyond purely data-driven approaches. Furthermore, the availability of comprehensive evaluation data is of utmost importance. As a result, we target more high-quality datasets and new methods other than data-driven for the problem. 

## **References** 

- [1] Jens Ambrasat, Christian von Scheve, Markus Conrad, Gesche Schauenburg, and Tobias Schröder. Consensus and stratification in the affective meaning of human sociality. _Proceedings of the National Academy of Sciences_ , 111(22):8001–8006, 2014. 2 

- [2] Alan P Fiske. The four elementary forms of sociality: framework for a unified theory of social relations. _Psychological review_ , 99(4):689, 1992. 2 

- [3] John A Bargh and Tanya L Chartrand. The unbearable automaticity of being. _American psychologist_ , 54 (7):462, 1999. 2 

- [4] Edward Finegan. _Language: Its structure and use_ . Cengage Learning, 2014. 2 

- [5] Meta Fundamental AI Research Diplomacy Team (FAIR)†, Anton Bakhtin, Noam Brown, Emily Dinan, Gabriele Farina, Colin Flaherty, Daniel Fried, Andrew Goff, Jonathan Gray, Hengyuan Hu, et al. Humanlevel play in the game of diplomacy by combining language models with strategic reasoning. _Science_ , 378 (6624):1067–1074, 2022. 2 

- [6] Silke Anders, Roos de Jong, Christian Beck, John-Dylan Haynes, and Thomas Ethofer. A neural link between affective understanding and interpersonal attraction. _Proceedings of the National Academy of Sciences_ , 113(16):E2248–E2257, 2016. 2 

- [7] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. _Advances in Neural Information Processing Systems_ , 35:27730–27744, 2022. 2 

- [8] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2020. 2 

- [9] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. _arXiv preprint arXiv:2201.11903_ , 2022. 2 

- [10] Laura Ruis, Akbir Khan, Stella Biderman, Sara Hooker, Tim Rocktäschel, and Edward Grefenstette. Large language models are not zero-shot communicators, 2022. 2 

- [11] Alisa Liu, Zhaofeng Wu, Julian Michael, Alane Suhr, Peter West, Alexander Koller, Swabha Swayamdipta, Noah A. Smith, and Yejin Choi. We’re afraid language models aren’t modeling ambiguity, 2023. 2 

- [12] Zilong Zheng, Shuwen Qiu, Lifeng Fan, Yixin Zhu, and Song-Chun Zhu. GRICE: A grammar-based dataset for recovering implicature and conversational rEasoning. In _Annual Meeting of the Association for Computational Linguistics (ACL)_ , pages 2074–2085, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-acl.182. URL `https://aclanthology.org/2021. findings-acl.182` . 2, 3 

- [13] OpenAI. Gpt-4 technical report, 2023. 2, 3, 8 

- [14] Amelia Glaese, Nat McAleese, Maja Tr˛ebacz, John Aslanides, Vlad Firoiu, Timo Ewalds, Maribeth Rauh, Laura Weidinger, Martin Chadwick, Phoebe Thacker, et al. Improving alignment of dialogue agents via targeted human judgements. _arXiv preprint arXiv:2209.14375_ , 2022. 2 

- [15] Tuhin Chakrabarty, Xurui Zhang, Smaranda Muresan, and Nanyun Peng. Mermaid: Metaphor generation with symbolism and discriminative decoding. In _Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2021. 2, 3 

10 

- [16] Prateek Saxena and Soma Paul. Epie dataset: A corpus for possible idiomatic expressions. In _Text, Speech, and Dialogue - 23rd International Conference, TSD 2020, Brno, Czech Republic, September 8-11, 2020, Proceedings_ , 2020. 2, 3, 4, 16 

- [17] Tosin P. Adewumi, Roshanak Vadoodi, Aparajita Tripathy, Konstantina Nikolaidou, Foteini Liwicki, and Marcus Liwicki. Potential idiomatic expression (pie)-english: Corpus for classes of idioms. In _Proceedings of the Thirteenth Language Resources and Evaluation Conference_ , 2022. 2, 3 

- [18] Issa Annamoradnejad and Gohar Zoghi. Colbert: Using bert sentence embedding in parallel neural networks for computational humor, 2022. 2, 3, 16 

- [19] Bodhisattwa Prasad Majumder, Shuyang Li, Jianmo Ni, and Julian McAuley. Interview: Large-scale modeling of media dialog with discourse patterns and knowledge grounding. In _Annual Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 8129–8141, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.653. URL `https: //aclanthology.org/2020.emnlp-main.653` . 2, 4 

- [20] Ryan Lowe, Nissan Pow, Iulian Serban, and Joelle Pineau. The Ubuntu dialogue corpus: A large dataset for research in unstructured multi-turn dialogue systems. In _Proceedings of the 16th Annual Meeting of the Special Interest Group on Discourse and Dialogue_ , pages 285–294, Prague, Czech Republic, September 2015. Association for Computational Linguistics. doi: 10.18653/v1/W15-4640. URL `https: //aclanthology.org/W15-4640` . 3 

- [21] Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. Race: Large-scale reading comprehension dataset from examinations. In _Annual Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2017. 3 

- [22] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018. 3 

- [23] Adina Williams, Nikita Nangia, and Samuel R. Bowman. A broad-coverage challenge corpus for sentence understanding through inference. In _North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)_ , 2018. 3, 9 

- [24] Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe Kiela, and Jason Weston. Personalizing dialogue agents: I have a dog, do you have pets too? In _Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2018. 3 

- [25] Rowan Zellers, Yonatan Bisk, Roy Schwartz, and Yejin Choi. Swag: A large-scale adversarial dataset for grounded commonsense inference. In _Annual Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2018. 3 

- [26] Lifu Huang, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Cosmos qa: Machine reading comprehension with contextual commonsense reasoning. In _Annual Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2019. 3 

- [27] Siva Reddy, Danqi Chen, and Christopher D. Manning. Coqa: A conversational question answering challenge. In _Transactions of the Association for Computational Linguistics (TACL)_ , 2019. 3 

- [28] Kai Sun, Dian Yu, Jianshu Chen, Dong Yu, Yejin Choi, and Claire Cardie. Dream: A challenge dataset and models for dialogue-based reading comprehension. In _Transactions of the Association for Computational Linguistics (TACL)_ , 2019. 3 

- [29] Sean Welleck, Jason Weston, Arthur Szlam, and Kyunghyun Cho. Dialogue natural language inference. In _Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2019. 3 

- [30] Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In _North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)_ , 2019. 3 

- [31] Leyang Cui, Yu Wu, Shujie Liu, Yue Zhang, and Ming Zhou. Mutual: A dataset for multi-turn dialogue reasoning. In _Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2020. 3, 4 

- [32] Paloma Jeretic, Alex Warstadt, Suvrat Bhooshan, and Adina Williams. Are natural language inference models imppressive? learning implicature and presupposition. In _Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2020. 3 

11 

- [33] Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, and Bill Dolan. Dialogpt: Large-scale generative pre-training for conversational response generation. In _Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2020. 3, 17 

- [34] Baolin Peng, Michel Galley, Pengcheng He, Chris Brockett, Lars Liden, Elnaz Nouri, Zhou Yu, Bill Dolan, and Jianfeng Gao. Godel: Large-scale pre-training for goal-directed dialog, 2022. 3 

- [35] Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, YaGuang Li, Hongrae Lee, Huaixiu Steven Zheng, Amin Ghafouri, Marcelo Menegali, Yanping Huang, Maxim Krikun, Dmitry Lepikhin, James Qin, Dehao Chen, Yuanzhong Xu, Zhifeng Chen, Adam Roberts, Maarten Bosma, Vincent Zhao, Yanqi Zhou, Chung-Ching Chang, Igor Krivokon, Will Rusch, Marc Pickett, Pranesh Srinivasan, Laichee Man, Kathleen MeierHellstern, Meredith Ringel Morris, Tulsee Doshi, Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen, Vinodkumar Prabhakaran, Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina, Erin Hoffman-John, Josh Lee, Lora Aroyo, Ravi Rajakumar, Alena Butryna, Matthew Lamm, Viktoriya Kuzmina, Joe Fenton, Aaron Cohen, Rachel Bernstein, Ray Kurzweil, Blaise Aguera-Arcas, Claire Cui, Marian Croak, Ed Chi, and Quoc Le. Lamda: Language models for dialog applications, 2022. 3 

- [36] Daniel Adiwardana, Minh-Thang Luong, David R. So, Jamie Hall, Noah Fiedel, Romal Thoppilan, Zi Yang, Apoorv Kulshreshtha, Gaurav Nemade, Yifeng Lu, and Quoc V. Le. Towards a human-like open-domain chatbot, 2020. 3 

- [37] Chulaka Gunasekara, Seokhwan Kim, Luis Fernando D’Haro, Abhinav Rastogi, Yun-Nung Chen, Mihail Eric, Behnam Hedayatnia, Karthik Gopalakrishnan, Yang Liu, Chao-Wei Huang, Dilek Hakkani-Tür, Jinchao Li, Qi Zhu, Lingxiao Luo, Lars Liden, Kaili Huang, Shahin Shayandeh, Runze Liang, Baolin Peng, Zheng Zhang, Swadheen Shukla, Minlie Huang, Jianfeng Gao, Shikib Mehri, Yulan Feng, Carla Gordon, Seyed Hossein Alavi, David Traum, Maxine Eskenazi, Ahmad Beirami, Eunjoon, Cho, Paul A. Crook, Ankita De, Alborz Geramifard, Satwik Kottur, Seungwhan Moon, Shivani Poddar, and Rajen Subba. Overview of the ninth dialog system technology challenge: Dstc9, 2020. 3 

- [38] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2017. 3 

- [39] Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy Gur-Ari, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Slav Petrov, and Yonghui Wu. Palm 2 technical report, 2023. 3 

- [40] Yunxiang Zhang and Xiaojun Wan. MOVER: Mask, over-generate and rank for hyperbole generation. In _Annual Meeting of the Association for Computational Linguistics (ACL)_ , pages 6018–6030, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.440. URL `https://aclanthology.org/2022.naacl-main.440` . 3, 16 

- [41] Orion Weller and Kevin Seppi. The rJokes dataset: a large scale humor collection. In _Proceedings of the Twelfth Language Resources and Evaluation Conference_ , pages 6136–6141, Marseille, France, May 2020. European Language Resources Association. ISBN 979-10-95546-34-4. URL `https://aclanthology. org/2020.lrec-1.753` . 3 

- [42] Michael C. Frank and Noah D. Goodman. Predicting pragmatic reasoning in language games. _Science_ , 336 (6084):998–998, 2012. doi: 10.1126/science.1218633. URL `https://www.science.org/doi/abs/10. 1126/science.1218633` . 4 

12 

- [43] Kevin Stowe, Prasetya Utama, and Iryna Gurevych. IMPLI: Investigating NLI models’ performance on figurative language. In _Annual Meeting of the Association for Computational Linguistics (ACL)_ , pages 5375–5388, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/ 2022.acl-long.369. URL `https://aclanthology.org/2022.acl-long.369` . 4 

- [44] Tuhin Chakrabarty, Yejin Choi, and Vered Shwartz. It’s not rocket science : Interpreting figurative language in narratives. In _Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2022. 4 

- [45] Deepanway Ghosal, Siqi Shen, Navonil Majumder, Rada Mihalcea, and Soujanya Poria. Cicero: A dataset for contextualized commonsense inference in dialogues. In _Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2022. 4 

- [46] Deepanway Ghosal, Pengfei Hong, Siqi Shen, Navonil Majumder, Rada Mihalcea, and Soujanya Poria. Cider: Commonsense inference for dialogue explanation and reasoning. In _Proceedings of the 22nd Annual Meeting of the Special Interest Group on Discourse and Dialogue_ , 2021. 

- [47] Pei Zhou, Pegah Jandaghi, Bill Yuchen Lin, Justin Cho, Jay Pujara, and Xiang Ren. Probing commonsense explanation in dialogue response generation. In _Annual Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2021. 4 

- [48] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. 2019. URL `http://arxiv.org/abs/1907.11692` . 4, 16, 17 

- [49] Yunxiang Zhang and Xiaojun Wan. Mover: Mask, over-generate and rank for hyperbole generation. In _Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2022. 4 

- [50] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. In _Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2020. 5, 6, 16, 17 

- [51] Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In _Conference on Computer Vision and Pattern Recognition (CVPR)_ , June 2019. 6 

- [52] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In _Annual Conference on Empirical Methods in Natural Language Processing (EMNLP)_ . Association for Computational Linguistics, 11 2019. URL `http://arxiv.org/abs/1908.10084` . 6 

- [53] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In _Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2019. 8 

- [54] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. In _Journal of Machine Learning Research (JMLR)_ , 2020. 8, 9 

- [55] Daniel Khashabi, Sewon Min, Tushar Khot, Ashish Sabharwal, Oyvind Tafjord, Peter Clark, and Hannaneh Hajishirzi. Unifiedqa: Crossing format boundaries with a single qa system. In _Annual Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2020. 8 

- [56] Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. mT5: A massively multilingual pre-trained text-to-text transformer. In _North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)_ , pages 483–498, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021. naacl-main.41. URL `https://aclanthology.org/2021.naacl-main.41` . 8 

- [57] Bill MacCartney and Christopher D. Manning. Modeling semantic containment and exclusion in natural language inference. In _International Conference on Computational Linguistics (COLING)_ , pages 521–528, Manchester, UK, August 2008. Coling 2008 Organizing Committee. URL `https://aclanthology. org/C08-1066` . 9 

- [58] Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. Deberta: Decoding-enhanced bert with disentangled attention. In _International Conference on Learning Representations (ICLR)_ , 2021. 9, 17 

- [59] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2023. 9 

13 

- [60] Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai Taitelbaum, Doron Kukliansy, Vered Cohen, Thomas Scialom, Idan Szpektor, Avinatan Hassidim, and Yossi Matias. TRUE: Re-evaluating factual consistency evaluation. In _North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)_ , pages 3905–3920, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.287. URL `https://aclanthology.org/2022.naacl-main.287` . 9 

- [61] Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. A large annotated corpus for learning natural language inference. In _Annual Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2015. 9 

- [62] Anya Ji, Noriyuki Kojima, Noah Rush, Alane Suhr, Wai Keen Vong, Robert Hawkins, and Yoav Artzi. Abstract visual reasoning with tangram shapes. In _Annual Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 582–601, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. URL `https://aclanthology.org/2022.emnlp-main.38` . 9 

- [63] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. ALBERT: A lite BERT for self-supervised learning of language representations. 2020. 16 

- [64] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of language representations. In _International Conference on Learning Representations (ICLR)_ , 2020. 16, 17 

- [65] Rishabh Misra. News headlines dataset for sarcasm detection, 2022. 16 

- [66] Lennart Wachowiak, Dagmar Gromann, and Chao Xu. Drum up SUPPORT: Systematic analysis of image-schematic conceptual metaphors. In _Proceedings of the 3rd Workshop on Figurative Language Processing (FLP)_ , pages 44–53, Abu Dhabi, United Arab Emirates (Hybrid), December 2022. Association for Computational Linguistics. URL `https://aclanthology.org/2022.flp-1.7` . 16 

- [67] Marti A. Hearst. Texttiling: Segmenting text into multi-paragraph subtopic passages. _Comput. Linguist._ , 23(1):33–64, mar 1997. ISSN 0891-2017. 16 

- [68] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019. 17 

- [69] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019. 17 

14 

## **Supplementary Material** 

## **Contents** 

|**A **|**More Experimental Results for Pragmatic Identifcation and Reasoning**|**16**|
|---|---|---|
|**B **|**Annotation Details**|**16**|
||B.1<br>Details For Automatic Selection . . . . . . . . . . . . . . . . . . . . . . . . . . .|16|
||B.2<br>Details For Fine-grained Annotation . . . . . . . . . . . . . . . . . . . . . . . . .|16|
||B.3<br>Details on Human Refnements . . . . . . . . . . . . . . . . . . . . . . . . . . . .|17|
|**C **|**Experimental Detail**|**17**|
||C.1<br>Pragmatic Identifcation and Reasoning (PIR) . . . . . . . . . . . . . . . . . . . .|17|
||C.2<br>Conversational Question Answering (CQA) . . . . . . . . . . . . . . . . . . . . .|18|
|**D **|**More Detail on DiPlomat**|**18**|
|**E**|**Computational Resources**|**26**|
|**F**|**Limitations & Negative Societal Impacts**|**26**|
|**G **|**Ethics Concern**|**26**|
|**H **|**Responsibility & Dataset Liscence**|**26**|
|**I**|**Datasheets for Our Dataset**|**26**|
||I.1<br>Motivation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|26|
||I.2<br>Composition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|27|
||I.3<br>Collection Process<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|28|
||I.4<br>Preprocessing/cleaning/labeling<br>. . . . . . . . . . . . . . . . . . . . . . . . . . .|29|
||I.5<br>Uses . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|29|
||I.6<br>Distribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|30|
||I.7<br>Maintenance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|30|



15 

## **A More Experimental Results for Pragmatic Identification and Reasoning** 

We conducted additional experiments on PIR employing various models, including RoBERTlarge [48], DeBERTabase [63], and ALBERTbase [64]. The training and testing procedures remained consistent with the aforementioned models described in the main body. All experimental results have been compiled and presented in Table 6. Analysing newly proposed result, it’s obvious to observe that our conclusions mentioned in the main body still hold. 

Table 6: Pragmatics Identification and Reasoning Results. The numerical results are accuracy scores in their percentage. 

|.||||
|---|---|---|---|
||**C**_→_**P**|**CP**_→_**R**|**C**_→_**PR**|
|Random|50|20|10|
|BERTbase|63.2_±_1.1|91.3_±_0.7|**50.2**_±_**6.8**|
|RoBERTabase|64.4_±_1.3|92.0_±_0.4|50.0_±_11.28|
|RoBERTalarge|63.8_±_0.0|60.8_±_0.5|0.0_±_0.0|
|GPT-2base|64.4_±_0.7|90.9_±_0.9|13.06_±_1.1|
|DialoGPTmedium|65.0_±_0.6|24.5_±_1.9|3.8_±_1.5|
|DeBERTabase|64.9_±_0.2|**92.6**_±_**0.6**|43.9_±_1.2|
|ALBERTbase|**65.1**_±_**0.4**|90.6_±_0.2|34.9_±_1.8|



## **B Annotation Details** 

## **B.1 Details For Automatic Selection** 

Different methodologies are employed to address various pragmatic phenomena. To leverage prior advancements in the field, we begin by segmenting each dialogue into individual utterances. Subsequently, we employ two distinct approaches, namely string matching and pretrained model classification, to identify these phenomena within our source data. In the case of scalar implicature, which exhibits a noticeable pattern characterized by word pairs such as _(some, all)_ appearing in adjacent turns of dialogues, we employ string matching to annotate instances of scalar implicature in conversations. Similarly, for popeq, which often features a continuous question mark, we utilize this characteristic as a means of detection. With regards to idioms, which exhibit more evident patterns, we employ the idiom set proposed by Saxena and Paul [16] to conduct searches. For other types of phenomena that lack obvious patterns, we leverage a pretrained RoBERTa base model [48], and fine-tune it for our specific task. The sarcasm dataset by Misra [65] is used for finetuning the sarcasm model, the MOVER dataset by Zhang and Wan [40] for hyperbole and the ColBERT dataset by Annamoradnejad and Zoghi [18] for paronomasia. Several models have been proposed for metaphor detection, thus we utilize an existing model [66] specifically designed for metaphor identification. 

**Topic Segmentation** The original dialogues employed in our study consist of lengthy and multiturn exchanges, which are well-suited for our research objectives. Consequently, we implement a segmentation process to break down these dialogues into shorter units. To achieve this, we employ two techniques, namely BERTScore [50] and TextTiling [67]. The segmentation procedure starts with computing the BERTScore between adjacent turns and subsequently applying the TextTiling algorithm to the generated BERTScores. 

## **B.2 Details For Fine-grained Annotation** 

AMT is integral to our process. To ensure clarity and consistency, we provide explicit instructions to the workers. Additionally, to further elucidate the objectives of our study, we offer illustrative examples. The task itself is presented below the instructions and examples, with the dialogue and corresponding turn numbers provided for workers to select. Furthermore, as workers check a checkbox, we prompt them to select a confidence score and provide a rationale. In order to strike a balance between our budget, the quality of annotations, and the speed of annotation, we have determined the compensation of $0.1 per completed task. The whole view of the worker interface is presented in Figure 6. After the annotation process, we collect responses that are assigned with a confidence score of 4 or higher. 

16 

## **B.3 Details on Human Refinements** 

Disturbing choices are chosen based on the BERTScore metric [50]. The rationale with the highest similarity, as determined by other dialogues, is selected and included in the pool of candidate options. The instructions provided to the workers align with those used for Fine-grained Annotation, wherein they are also instructed to assign a confidence score to their responses. The remuneration for workers is set at $0.05 per task. The worker interface is included in Figure 7. 

**AMT Workers Requirements** In order to guarantee the quality of annotated data, the qualification rules for workers are strict and can be found in Table 7. 

Table 7: AMT workers requirements 

|**Country**In|United States, Canada, Great Britain, Australia, Singapore, Ire-|
|---|---|
||land, New Zealand|
|**# Tasks approved**GreaterThanOrEqualTo|1300|
|**Tasks approved Rate**GreaterThanOrEqualTo|95%|



## **C Experimental Detail** 

## **C.1 Pragmatic Identification and Reasoning (PIR)** 

**BERTbase [68]** BERT (Bidirectional Encoder Representations from Transformers) is a revolutionary language representation model that has had a significant impact on natural language processing (NLP) tasks. It has achieved remarkable performance across various NLP benchmarks, including question answering, sentiment analysis, named entity recognition, and many others. Its birth brings profound influence on pretrained language models. 

**RoBERTabase & RoBERTalarge [48]** RoBERTa improves upon BERT by incorporating enhancements such as larger and more diverse training data, longer pretraining duration, dynamic masking, and advanced training strategies. These improvements enable RoBERTa to achieve even better performance on a wide range of NLP benchmarks. While BERT paved the way for contextualized representations in NLP, RoBERTa further refines and pushes the boundaries of language understanding, making it a powerful and preferred choice for many researchers and practitioners in the field. 

**ALBERTbase & ALBERT large [64]** ALBERT (A Lite BERT) is a highly efficient and compact variant of the BERT model that addresses the computational limitations of the original architecture. It incorporates parameter-reduction techniques to alleviate training time constraints and achieve improved performance compared to BERT. 

**DeBERTabase [58]** DeBERTa (Decoding-enhanced BERT with Disentangled Attention) is a stateof-the-art language representation model that builds upon the BERT architecture and introduces several key innovations, including disentangled attention mechanism. The performance of DeBERTa has been demonstrated to surpass that of BERT on a wide range of NLP tasks. 

**GPT2base [69]** Leveraging transformers decoder, Radford et al. [69] proposed GPT2. It represents a significant breakthrough in natural language processing and generation. One of the most notable features of GPT-2 is its ability to generate coherent and contextually relevant text. Through unsupervised pretraining on a large corpus of internet text, GPT-2 learns to predict the next word in a sequence of text, enabling it to generate human-like responses. 

**DialogGPTmedium [33]** DialogGPT is dialogue-oriented GPT. It builds upon the GPT architecture and extends it to support interactive conversations. DialogGPT is trained in a supervised manner using a dialogue dataset, which allows it to understand and generate responses in a conversational context. 

The PIR task encompasses three distinct settings: **C** _→_ **P** , **CP** _→_ **R** , and **C** _→_ **PR** . In the **C** _→_ **P** setting, models are trained for 20 epochs, employing a batch size as indicated in Table 9, a learning rate of 2 _e −_ 5, and weight decay of 0 _._ 01. As for **CP** _→_ **R** , the hyperparameters adopted are listed in Table 8. For the **C** _→_ **PR** setting, there is no training required; instead, we simply load the best checkpoint obtained from the previous training for this task. The concrete implementation is as 

17 

Table 8: Hyperparameters for models on **CP** _→_ **R** 

|Model|learning rate|batch size|weight decay|epochs|
|---|---|---|---|---|
|BERTbase|5e-5|12|0.001|50|
|BERTlarge|5e-5|12|0.001|50|
|ALBERTbase|5e-5|12|0.001|50|
|ALBERTlarge|5e-5|12|0.001|50|
|DeBERTabase|5e-5|12|0.001|50|
|RoBERTabase|5e-5|12|0.001|50|
|RoBERTalarge|5e-5|12|0.001|50|
|GPT2base|0.001|8|0.01|50|
|DialoGPTmedium|0.001|2|0.01|50|



Table 9: Batch size for models on **C** _→_ **P** 

|9: Batch size for|models on**C**|
|---|---|
|Model|Batch Size|
|BERTbase|80|
|ALBERTbase|24|
|ALBERTlarge|24|
|DeBERTabase|24|
|RoBERTabase|80|
|RoBERTalarge|24|
|GPT2base|24|
|DialoGPTmedium|8|



follows: we initially flatten the test dataset of **C** _→_ **P** , ensuring that each instance contains both a dialogue and a pragmatic turn extracted from the same dialogue. As for the test dataset of **CP** _→_ **R** , no modifications are made. It should be noted that, following the processing steps, both datasets own the same dialogues and corresponding pragmatic turns, resulting in identical instance numbers. For an instance to be deemed correct, the models must successfully accomplish both component tasks _i.e_ . succeed in Identification and Reasoning. 

## **C.2 Conversational Question Answering (CQA)** 

**CQA** ChatGPT was instructed to generate questions for our tasks. The prompt template that starts the questions with "Which" is depicted in Table 11. Through this methodology, we collected a total of 19,482 questions. To ensure the reliability of the answers provided to these questions, AMT is utilized. The task template is demonstrated in Figure 8. In our experiment, the hyperparameters adopted are illustrated in Table 12. To assess the performance of ChatGPT, we conducted testing using the template outlined in Table 14. 

Table 10: Hyperparameters for models on CQA. 

|Training Epoch|50|
|---|---|
|Learning Rate|5_._6_e −_5|
|Batch Size|24|
|Weight Decay|0_._001|



**Zero-Shot Natural Language Inference** Details are provided as follows. T5-XXL, and DeBERTav3 are tested with the pragmatic turn as premise and implied meaning as a hypothesis. The context is out of reach for these models. In contrast, as shown in Table 11, ChatGPT is given the context, and the red line labeled “ `Think step by step` ” represents two distinct configurations: one with `step-by-step` and one without it. 

## **D More Detail on DiPlomat** 

In this section, we will propose more examples of our dataset in Table 15, Table 16, Table 17, Table 18, and Table 19. 

18 

Table 11: ChatGPT question generation template: using "Which" to start the question. 

```
Youaresensitiveandalwaysviewothers’wordsashavingsomeimplied
meanings.
Forthedialoguebetween"A"and"B"inthistask,wehaveoffereda
statementthatistheimpliedmeaningofaturn,pleaseonlyoffer
onereadingcomprehensionquestionthatcanbeansweredwithonlyone
wordbasedonthedialogueandmostlyfocusesontheturnthestatement
mentions.
Thequestionwillbetestedbyonlybyviewingthedialogue,soplease
makethequestionhardenoughthatit’simpossibletoanswerwithout
viewingthestatement.
Use"Which"toaskthequestion!
Followingisthedialogue:
{dialogue}
Followingisthestatement:
{statement}
Use"Which"toaskthequestion!Andpleasemakethequestionhard
enoughthatit’simpossibletoanswerwithoutviewing
```

Table 12: Hyperparameters for models on CQA. 

|Training Epoch|50|
|---|---|
|Learning Rate|5_._6_e −_5|
|Batch Size|24|
|Weight Decay|0_._001|



Table 13: Test ChatGPT: answer questions with only one word. 

```
Forthedialoguebetween"A"and"B"inthistask,pleaseanswera
questionaccordingtothedialoguewithonlyoneword
Followingisthedialogue:
{dialogue}
Followingisthequestion:{question}
```

## Table 14: ChatGPT test template of Zero-Shot CoT 

```
Thisisanaturallanguageinferencetask.Giventhedialoguecontext:
{context}Does{pragmaticturn}entails{impliedmeaning}?Reply
’entails’or’notentails’.
```

```
Thinkstepbystep.
```

19 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Diplomat_A_Dialogue_Dataset_for_Situated_PragMATic_Reasoning_images/nips_2023_Diplomat_A_Dialogue_Dataset_for_Situated_PragMATic_Reasoning.pdf-0020-00.png)


**----- Start of picture text -----**<br>
Identify Implicature in Dialogue<br>Instructions<br>For the dialogue between "A" and "B" in this task, follow these steps :<br>1. Read through the dialogue<br>2. For each turn of the dialogue , identify whether its actual meaning is different from its literal meaning, such as:<br> Bob is a couch potato.  implies  that "Bob sits on the sofa all day for watching TV" but not Bob is a potato.<br>I am so hungry that I can eat ten elephants implies  that "I am extremely hungry but not I will eat ten elephants".<br> Zombies eat brains. You’re safe implies  that "you do not have a brain".<br> I’d agree with you, but then we’d both be wrong implies  that "you are wrong".<br>3.  Check the checkbox  below the dialogue corresponding to the turns that meet the condition.<br>4. (Confident score)When a checkbox is checked we will ask you to choose a number(1 to 5) to represent how confident you are about<br>choosing the turn.(Higher score , more confidence)<br>5.  Write a brief but more than 8 words implied meaning. If you can't find one , simply write None<br>Please complete the HIT carefully , and note that :<br> Please read our examples!<br>There may be several turns that meet the conditions , select  all of them .<br> You must  at least choose one  of the turns!<br> Some actual meanings are  hard to find , so please read patiently and carefully!<br> Confident score, implied meaning  should not  leave to blank!<br> Confident score doesn't represent how confident you are about your answer<br> Confident score marks how confident you are that the turn  has implied meanings !<br>Warning : Choosing too many answers randomly will cause us to mark you as unqualified worker!<br>Warning : Writing reasons irrelevant will also cause us to mark you as unqualified worker!<br>Warning : Using ChatGPT or AI methods will cause us to mark you as unqualified worker (this is strict, I will block you if there is a single<br>suspicious hit)!<br>Examples<br>(1) A: Did you drink the milk I kept on the table?<br>(2) B: The cat seems to be happy.<br>Please choose the turns whose  actual meanings  are different from their  literal meanings .<br>1  (1)<br> (2)<br> Please choose a confidence score :<br>5 : You are totally sure that this turn has implied meanings and believe that everyone will agree with you<br>Please write a implied meaning ( More than 8 words! ) :<br>The cat seems happy implies that B thinks that the cat drinks the milk.<br>(1) A: Who made these donuts?<br>(2) B: I made some of these donuts.<br>(3) A: Ok,would you like to send some of them to Mr.Potter?<br>(4) B: I have homework to do.<br>2 Please choose the turns whose  actual meanings  are different from their  literal meanings .<br> (1)<br> (2)<br> Please choose a confidence score :<br>5 : You are totally sure that this turn has implied meanings and believe that everyone will agree with you<br>Please write a implied meaning ( More than 8 words! ) :<br>"some" represents not all, B means that he has only make some of the donuts not all of the donuts.<br>(1) A: Bob, are you sure you can take care of yourself this weekend?<br>(2) B: Mom, can a duck swim?<br>Please choose the turns whose  actual meanings  are different from their  literal meanings .<br>3  (1)<br> (2)<br> Please choose a confidence score :<br>5 : You are totally sure that this turn has implied meanings and believe that everyone will agree with you<br>Please write implied meaning ( More than 8 words! ) :<br>Duck can swim is for sure impling that I can take care of myself is for sure.<br>(1) A: Do you like her?<br>(2) B: She's like cream in my coffee.<br>Please choose the turns whose  actual meanings  are different from their  literal meanings .<br>4  (1)<br> (2)<br> Please choose a confidence score :<br>5 : You are totally sure that this turn has implied meanings and believe that everyone will agree with you<br>Please write a implied meaning ( More than 8 words! ) :<br>Cream is wonderful impling that I like her a lot.<br>Task<br>Following is the dialogue:<br>(0) A: ...<br>(1) B: ...<br>(2) B: ...<br>... ... ...<br>Please choose the turns whose  actual meanings  are different from their  literal meanings .<br>(0)<br>(1)<br>...<br> Please choose a confidence score :<br>5 : You are totally sure that this turn has implied meanings and believe that everyone will agree with you<br>Please write a implied meaning ( More than 8 words! ) :<br>Write reason here<br>**----- End of picture text -----**<br>


Figure 6: Fine-grained annotation worker interface 

20 

## **Multiple Choice For Implicature In Dialogue** 

## **Instructions** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Diplomat_A_Dialogue_Dataset_for_Situated_PragMATic_Reasoning_images/nips_2023_Diplomat_A_Dialogue_Dataset_for_Situated_PragMATic_Reasoning.pdf-0021-02.png)


**----- Start of picture text -----**<br>
For the dialogue between "A" and "B" in this task, follow these steps :<br>1. Read through the dialogue<br>2. There may be turns in dialogue that their actual meanings are different from their literal meanings, such as:<br> Bob is a couch potato.  implies  that "Bob sits on the sofa all day for watching TV" but not Bob is a potato.<br>I am so hungry that I can eat ten elephants implies  that "I am extremely hungry but not I will eat ten elephants".<br> Zombies eat brains. You’re safe implies  that "you do not have a brain".<br> I’d agree with you, but then we’d both be wrong implies  that "you are wrong" .<br>3. Under each dialogue, there may be several statements. Each statement will give you a turn number telling you the corresponding turn's<br>actual meaning is different from its literal meaning and offer you the implied meaning or reason.<br>4.  Check the checkbox  before statements which you consider it reasonable.<br>5. (Confident score)When a checkbox is checked we will ask you to choose a number(1 to 5) to represent how confident you are about<br>choosing the statement.(Higher score , more confidence)<br>Please complete the HIT carefully , and note that :<br> Please read our examples!<br>There may be several statements that meet the conditions , select  all of them .<br>There are  disturbance statements , so please read carefully.<br> You must  at least choose one  of the statements!<br> Some actual meanings are  hard to find , so please read patiently and carefully!<br> Confident score  should not  leave to blank!<br>Warning : Choosing disturbance choices too many times will cause us to mark you as unqualified worker, so please read carefully!<br>Warning : Choosing only one or two choices with much more choices presented to avoid selecting disturbance ones will also cause us to<br>mark you as unqualified workers!<br>Examples<br>(1) A: Did you drink the milk I kept on the table?<br>(2) B: The cat seems to be happy.<br>1 Please choose the  statements that mark the turns with implied meaning correctly and have reasonable reason :<br> turn 2 : "The cat seems to be happy" implies that the cat is delighted.<br> turn 2 : "The cat seems to be happy" implies that I did not drink the milk and I think the cat might drink it<br> Please choose a confidence score :<br>5 : You are totally sure that this statement is correct and believe that everyone will agree with you<br>(1) A: Who made these donuts?<br>(2) B: I made some of these donuts.<br>(3) A: Ok,would you like to send some of them to Mr.Potter?<br>(4) B: I have homework to do.<br>2<br>Please choose the  statements that mark the turns with implied meaning correctly and have reasonable reason :<br> turn 1 : "Who made these donuts?" implies that A wants to eat donuts, A is hungry.<br> turn 2 : "I made some of these donuts" implies that I did not make all of these donuts I only make a part of them.<br> Please choose a confidence score :<br>5 : You are totally sure that this statement is correct and believe that everyone will agree with you<br> turn 4 : "homework" implies that the work is done at home not at school<br> turn 4 : B has homework to do, so he/she is not able to help A.<br>(1) A: Bob, are you sure you can take care of yourself this weekend?<br>(2) B: Mom, can a duck swim?<br>3 Please choose the  statements that mark the turns with implied meaning correctly and have reasonable reason :<br> turn 1 : Mom is not sure whether B can take care of itself.<br> turn 2 : Duck can swim is for sure implying that I can take care of myself<br> Please choose a confidence score :<br>5 : You are totally sure that this statement is correct and believe that everyone will agree with you<br>(1) A: Do you like her?<br>(2) B: She's like cream in my coffee.<br>4 Please choose the  statements that mark the turns with implied meaning correctly and have reasonable reason :<br> turn 1 : She looks like cream in my coffee<br> turn 2 : Cream is wonderful impling that I like her a lot.<br> Please choose a confidence score :<br>5 : You are totally sure that this statement is correct and believe that everyone will agree with you<br>Task<br>Following is the dialogue:<br>(0) A: ...<br>(1) B: ...<br>(2) B: ...<br>... ... ...<br>Please choose the turns whose  actual meanings  are different from their  literal meanings .<br>turn 0: [...]<br>turn 1: [...]<br>...<br> Please choose a confidence score :<br>5 : You are totally sure that this turn has implied meanings and believe that everyone will agree with you<br>Please write a implied meaning ( More than 8 words! ) :<br>Write reason here<br>**----- End of picture text -----**<br>


Figure 7: Human refinements worker interface 

21 

## **Dialogue Question Answering** 

## **Instructions** 

For the dialogue between "A" and "B" in this task, follow these steps: 

1. Read our example! 

2. Read through the dialogue 

3. Under each dialogue, there may be several questions. 

4.  There is a useful reference statement that may help you answer the corresponding question, please read it carefully 

5. Answer each question with only one word! 

Please complete the HIT carefully , and note that : 

- _i. Reference Statements are useful, but don't just rely on it, read the dialogue as well! ii.  Our turn number start from 0 iii. There are incorrect reference statements._ 

**Warning : If your answer is close to incorrect reference statement, but far away from the dialogue, we will mark you as unqualified workers!** 

|**Examples**|**Examples**|**Examples**|||
|---|---|---|---|---|
|Following is t||he<br>dialogue:|||
|(0)|B:|How did you<br>from places|feel when you<br>you knew?|found out, this summer, about the abuse of children that was going on in Pennsylvania not far|
|(1)|A:|Yeah, it hurt<br>growing up,<br>was actually<br>honestly, un<br>since - what?<br>didn't know<br>a problem th|reading the re<br>you would hea<br>a - anything b<br>til recently tha<br>- I think it was<br>that that was a<br>at was going o|port because I was reading about these parishes that I went to growing up.I was born in '96, so<br>r kids joke about, oh, you know, priests molesting kids and whatnot.But I never knew that there<br>ehind that.I just sort of thought it was people making fun of a religion.And then I didn't learn,<br>t the abuse scandal is something that was real and something that the world has known about<br>the early 2000s, whenever the Boston Globe or whatever that newspaper is broke the story.I<br>thing.It wasn't something I'd ever been exposed to, so I wasn't really aware of the fact that this is<br>n in my church.You know, it's unthinkable.|
|(2)|B:|And it's hurt|your faith?||
|(3)|A:|I think that's<br>hurt my pers<br>Catholic Chu|a difficult thin<br>onal, religious<br>rch.|g to answer.I will say that it has hurt my faith in the Catholic Church.I don't think it has actually<br>faith.I'm just starting to see less of a connection between what I believe and the teachings of the|
||||||
|**Reference**||**Statement**||turn 3 : The speaker's faith in the Catholic Church has been damaged by the abuse scandal,<br>but their personal faith remains intact.|
|**Question**||||How has the abuse scandal affected the speaker's faith in the Catholic Church?|
|**Answer**||||Damage.|
||||||
|**Task**|||||
|Following is t||he<br>dialogue:|||
|**(0)**||**A:**|**...**||
|**(1)**||**B:**|**...**||
|**(2)**||**B:**|**...**||
|**...**||**...**|**...**||
||||||
|**Reference**||**Statement**||**...**|
|**Question**||||**...**|
||||||
|**Answer**||||**[...]Write one word answer here.**|



Figure 8: Answer collecting worker interface 

22 

Table 15: Contextual reasoning examples of **DiPlomat** 

|**A**: Yeah. They say that he’s the fastest pitcher there ever was. It’s just he really<br>couldn’t fnd home plate. I mean, some of the stories you learn about this guy, it<br>reads like fction. When he was - I think this is around 1960. He’s pitching in the<br>minor leagues, and he pitched so fast he ripped the man’s ear off.<br>**B**: Oh.<br>**A**:Yeah.|**Rationale**: The literal<br>meaning is a simple ex-<br>pression of agreement,<br>while the implied mean-<br>ing is that the speaker is<br>amazed by the story of<br>Steve Dalkowski’s feats.|
|---|---|
|||
|**B**: We’re talking about 2. 8 million people. Has the rise of temporary workers<br>fgured into, at least, the statistical improvement of the U. S. economy for some<br>people?<br>**A**: It has. Overall, about one seventh of the total job growth has been in the<br>temp sector. The temp sector is growing nine times faster than the overall private<br>sector as a whole. And the 2. 9 million workers represents a record number, both<br>in the number of temp workers and in the percentage of the economy that they<br>make up.<br>**B**:You know in "Harvest Of Shame," Edward R. Murrow very famously said, the<br>people we’re showing you in this documentary have picked your Thanksgiving<br>bounty with their bare hands, and this is how they live.|**Rationale**: The implied<br>meaning of this turn is<br>to refect on our reliance<br>on temporary workers in<br>our day-to-day lives.|
|||
|**A**: And so I got up and ran. And it wasn’t too far. But I just - at that moment, I<br>thought, I don’t want to be shot in the back, and I need to fnd some cover. And<br>there’s really no place to hide. But there are these<br>**B**: You found a little, like, alcove that you could duck into.<br>**A**:There was a little alcove, yeah. And I just made myself as small as I could in<br>that little corner.|**Rationale**: The speaker<br>tried to protect itself<br>from danger.|
|||
|**A**: Well, there’s a big argument in the United States about this. There’s one<br>group of folks who think that engagement policy failed. We engaged with China<br>from 1979 until about 2013 when Xi Jinping came into power. And the idea<br>of engagement was that coevolution was in the American interest as well as in<br>China’s interest. And you could bring China along to be a responsible player to<br>some degree.<br>**A**: Many hardliners in the United States government - and outside and including<br>in the expert community - now claim that engagement was a sucker’s game and<br>that we have raised up a tiger which could now devour us. But there are different<br>schools of thought about this, and many of us think that we still need to engage<br>with China, albeit more strategically.<br>**B**:That image of raising a tiger that will devour us is very dramatic.|**Rationale**: The situa-<br>tion is not necessarily<br>an ’either/or’ between<br>China and the United<br>States.|



23 

Table 16: Figurative language reasoning examples of **DiPlomat** 

|A: Thank you. How are you?<br>B: I’m pretty good. Thank you. You must bestuck like glueon this, but, you<br>know, you’ve played in three World Cups, including one of the wins for the U. S.<br>team in 1999. How would you describe what it’s like to be out there on that feld<br>in that fnal game?|**Rationale**: Stuck like<br>glue means to be at-<br>tached<br>to<br>something,<br>which is a particular is-<br>sue or a person.|
|---|---|
|||
|**B**: So in terms of what to do about it, we’ve said Twitter and Facebook have shut<br>down these accounts, which prompts me to wonder - does shutting down a fake<br>account do that much? Can’t the Chinese government, if it’s determined to go<br>down this path, just open up two new ones in place of the one that was closed?<br>**A**:It is a cat-and-mouse game, and the companies are constantly trying to get<br>ahead of it.[_· · ·_]As you said, they can always set up new accounts.|**Rationale**:<br>Mice<br>are<br>constantly trying to get<br>away from cats and cats<br>are constantly trying to<br>catch mice. In the same<br>way, the Chinese gov-<br>ernment will always be<br>trying to escape restric-<br>tions on social media ac-<br>counts and media com-<br>panies will always be<br>trying to fnd fake ac-<br>counts.|
|||
|**A**: I really didn’t feel safe because the Turkish government is very famous for<br>hunting down those who oppose Erdogan. So, I mean, I just didn’t want to really<br>risk my life by going to Europe. But, you know, I talked to my team. I told them<br>all, like, how many times I want to come because I want to be with you guys<br>there, and I want to get a win with you guys. And then, later on, they came back<br>with the news and said, you know what?I think the best decision is if you don’t<br>come. Let’s just not risk it for one game.<br>**B**: Do you feel safe in New York and elsewhere in the U. S. ?<br>**A**: I have been getting last two, three days hundreds death threats, but I think I<br>feel safe in America.But anywhere else in the world, I wouldn’t really feel safe.|**Rationale**: He is imply-<br>ing that he is still not<br>safe.|



Table 17: Commonsense reasoning examples of **DiPlomat** 

|**B**: Yeah - African-American mayor from Tallahassee.<br>**A**:Yes. So this is sort of a test of whether real progressive candidates can win in<br>these sort of purplish states.[...]|**Rationale**: "Purplish"<br>states are not really col-<br>ored. They refer to US<br>states that are neither<br>clearly Republican (red)<br>nor Democrat (blue) in<br>their voting.|
|---|---|
|||
|**B**: He wrote a lot of letters by hand, didn’t he?<br>**A**:He wrote tons of letters. I bet there are a hundred thousand - hundreds out<br>there[_· · ·_]|**Rationale**: tons of let-<br>ters implies a very large<br>number and not to full a<br>ton.|
|||
|**B**: Well, Pluto’s offcial designation is a dwarf planet. And I have to tell you the<br>people who sent this probe all the way out to Pluto are a little angry about that<br>because when they launched it a decade ago, Pluto was still a planet.<br>**A**: (Laughter)<br>**B**: It got downgraded in the intervening years.<br>**A**:That seems so unfair.|**Rationale**: A is express-<br>ing sympathy for the<br>people who sent the<br>probe, showing that they<br>understand why they<br>feel so disappointed.|



24 

Table 18: External knowledge reasoning examples of **DiPlomat** 

|**B**: Inside of his house, family pictures decorate the walls and the fridge. Les has<br>15 great grandchildren. He grew up in an orphanage, and he couldn’t wait to<br>leave to join the military. And so in early 1944, he boarded a ship and crossed<br>the Atlantic Ocean to go to the frontline.<br>**A**:I loved that sailing on, of course. It was so dramatic. You could see all these<br>ships bobbing up and down on the ocean. And destroyers were weaving in and<br>out of them to make sure they uncovered any mines or anything.|**Rationale**:<br>Sailing<br>across the ocean during<br>wartime was a perilous<br>experience.|
|---|---|
|||
|**A**: . . . equivalent to a nuclear bomb?<br>**B**: Well, it’s about - its equivalent -the energy in that explosion is about 10 times<br>the energy in the frst atomic bomb. . .|**Rationale**: The energy<br>released in the explo-<br>sion is incredibly pow-<br>erful.|
|||
|**B**: So in your polling, in your research, do you fnd that it’s going to come down<br>to maybe a couple thousand votes from these unaffliated voters and on what<br>issues?Or will they vote?<br>**A**: It is likely at the moment to be a very narrow victory. President Bush won<br>in 2004 with fve percent. That was 100,000 votes. In other words, if it is one<br>percent, that would be 20,000 votes, and right now, the polls are moving around<br>in just single percentage points. So it could be that narrow.<br>**B**: Now, I have read that Colorado is going to be this year’s Florida and Ohio,<br>that this is going to be the state that decides the election.<br>**A**:I think it could be, and the interesting thing is that Obama and Palin were both<br>in Jefferson County a couple of days ago, indicating that there may be actually<br>even a county that could be looked at to be beyond an entire state.|**Rationale**:<br>The<br>turn<br>is suggesting that the<br>county of Jefferson in<br>Colorado could be a key<br>factor in deciding the<br>election, despite the fact<br>that it is only one of<br>many counties in the<br>state and there are other<br>swing states in the elec-<br>tion.|
|Table 19: Others examples of**DiPlomat**||
|||
|**A**: There’s that feeling - I mean, so many of us have parents in the industry. I<br>mean, that’s what this region is about, especially around Detroit, and Wayne<br>State’s in Detroit, the heart of Detroit. So, it’s nerve-racking. Everyone is nervous.<br>Everyone doesn’t know what’s going to happen next. We’re all watching the news<br>very closely. But at the same time, it’s interesting, because with my generation,<br>we almost seem to, kind of, not be as directly impacted. I mean, our family is, it<br>puts stress on us, but the day to day of the university and the day to day at school<br>doesn’t seem to have changed that much.<br>**B**: I understand you have friends there who are engineering majors. Do they have<br>any sense of what their future looks like, and will be it there in Michigan?<br>**A**:Everybody is secure in their choices and secure in their decision. Everybody<br>thinks that the industry will come around, especially now with the news that GM<br>is getting money from the government. And everybody is more hopeful, and I<br>mean, the auto industry has always been one of the largest industries and a staple<br>in America, and to think that that industry is just going to vanish, nobody is<br>willing to concede that.|**Rationale**: A believes<br>that the auto industry<br>will not vanish despite<br>the current situation|
|||
|**B**: In the meantime, what more have you learned in your reporting about the<br>death of Carlos Hernandez Vasquez?<br>**A**: Well, a couple of things. One thing that really stands out is that Carlos<br>Hernandez Vasquez died in a Border Patrol station. The previous migrant children<br>who died were taken to the hospital frst; Hernandez Vasquez was not even though<br>immigration authorities clearly knew that he was sick. He was diagnosed with<br>the fu by a nurse practitioner.|**Rationale**: The death<br>of Carlos Hernandez<br>Vasquez<br>could<br>have<br>been prevented if he<br>had been taken to the<br>hospital.|
|||
|**B**: So, how do you and the retired general, James Jones, know each other?<br>**A**:My gosh, I think - I can’t even remember when I frst met him. It’s been so<br>long ago. I’m sure I met him when he was head of the legislative liaison over the<br>Senate. But I really became acquainted with him when he became a brigadier<br>general, and, of course, I followed his career. Of course, he served very ably as<br>a commandant in the marine corps and then as the European commander, just<br>been with him from time to time. And I just consider him a very good friend.|**Rationale**: A has a high<br>opinion of James Jones’<br>character and career.|



25 

## **E Computational Resources** 

For our experiment, we utilized two A100s and one 3090. The majority of our experiments were conducted on the A100s, while for practical reasons, only Unified-QA-base, BART-base, and T5small were tested on the 3090. It is important to mention that each experiment was run on a single GPU. 

## **F Limitations & Negative Societal Impacts** 

We acknowledge two limitations in our study: bias and subjectivity. Since our dialogues primarily stem from an interview dataset, a considerable focus is placed on political topics. This is reasonable, as pragmatic phenomena frequently emerge in the statements of politicians to advance their specific goals. However, this focus introduces a certain degree of bias into our dataset. The second limitation relates to the absence of subjectivity. In our methodology, the data undergoes two stages of human annotation, ensuring higher quality and objectivity. However, pragmatic reasoning is inherently subjective, and prioritizing objectivity compromises the preservation of subjectivity, resulting in a limitation in terms of subjectivity coverage. Our dataset exhibits minimal negative societal impacts. This is primarily due to the fact that our dialogues are transcriptions of publicly available TV shows, which inherently limits the potential for negative effects. 

## **G Ethics Concern** 

**Were any ethical review processes conducted (e.g., by an institutional review board)?** No official processes were done, as our research is not on human subjects, but our data comes from published dataset. 

**Does the dataset contain data that might be considered confidential?** No, our data comes from an existing public interview dataset. 

**Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.** Few of the dialogues may talk about offensive topics. **Does the dataset identify subpopulations (e.g., by age or gender)?** Not explicitly. 

**Is it possible to identify individuals (i.e., one or more natural persons) directly or indirectly (i.e., in combination with other data) from the dataset?** Yes, our data contains names of famous people. 

## **H Responsibility & Dataset Liscence** 

We bear all responsibility in case of violation of rights and our dataset is under the license of CC BY-NC-SA (Attribution-NonCommercial-ShareAlike). 

## **I Datasheets for Our Dataset** 

## **I.1 Motivation** 

1. For what purpose was the dataset created? (Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.) This dataset was created to study pragmatic reasoning in dialogues, a specific gap is mentioned above in Appendix F. 

2. Who created this dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)? This dataset was created by the authors of this paper. 

3. Who funded the creation of the dataset? (If there is an associated grant, please provide the name of the grantor and the grant name and number.) The institute of the authors funded the creation of the dataset. 

4. Any other comments? None. 

26 

## **I.2 Composition** 

5. What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? (Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.) An instance of our dataset represent a piece of dialogue. Description is provided in our paper. 

6. How many instances are there in total (of each type, if appropriate)? We answer the question in our paper. Our datasets owns 4,177 dialogues. 

7. Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? (If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).) 

   - It is a sample of all possible cases. As pragmatic phenomena aren’t proved to be limited, we can’t guarantee a full sampling of them. 

8. What data does each instance consist of? We mention it in our paper. 

9. Is there a label or target associated with each instance? If so, please provide a description. Yes. The description is in our paper. 

10. Is any information missing from individual instances? (If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.) No. We leverage the original dialogues. 

11. Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)? ( If so, please describe how these relationships are made explicit.) No. Instances are weakly related, but focus on the same phenomenon. 

12. Are there recommended data splits (e.g., training, development/validation, testing)? (If so, please provide a description of these splits, explaining the rationale behind them.) Yes. We provide it. 

13. Are there any errors, sources of noise, or redundancies in the dataset? (If so, please provide a description.) 

   - Yes. We may have some answers just by giving _a_ , which is meaningless. 

14. Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? (If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.) 

It’s self-contained. 

15. Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)? (If so, please provide a description.) No. 

16. Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? (If so, please describe why.) Yes. Some of the topic are big events, they may be offensive for some people. However, we consider our dataset’s offensiveness to be limited, for the source dataset is a TV show transcript. 

27 

17. Does the dataset relate to people? (If not, you may skip the remaining questions in this section.) Yes. 

18. Does the dataset identify any subpopulations (e.g., by age, gender)? (If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.) 

   - No. This is not explicitly identified 

19. Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? (If so, please describe how.) 

   - Yes; their names are given in running text. 

20. Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)? (If so, please provide a description.) 

   - Yes. Our dataset may have dialogues talking about religious, politics and so on. 

21. Any other comments? None. 

## **I.3 Collection Process** 

1. How was the data associated with each instance acquired? (Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.) 

   - The data all comes from an interview dataset already published. (See our paper) 

2. What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)? (How were these mechanisms or procedures validated?) 

   - Software program and manual human curation (2 times). See our paper for details. 

3. If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)? Randomly. 

4. Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)? Crowdworkers. They are paid nicely. See Appendix for detail. 

5. Over what timeframe was the data collected? (Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created.) 

   - The dataset was collected in the early Spring of 2023, which does not necessarily reflect the timeframe of the data collected. 

6. Were any ethical review processes conducted (e.g., by an institutional review board)? (If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.) 

   - No review processes were conducted with respect to the collection and annotation of this data (though review was done for other aspects of this work; see the paper linked at the top of the datasheet). 

7. Does the dataset relate to people? (If not, you may skip the remaining questions in this section.) 

   - Yes. 

28 

8. Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)? Other sources. By curating a published dataset. 

9. Were the individuals in question notified about the data collection? (If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.) No. 

10. Did the individuals in question consent to the collection and use of their data? (If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.) 

   - No. All data are public. 

11. If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses? (If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).) N/A. 

12. Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted? (If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.) 

   - No. We consider our dataset having a limited negative effect, for all of our data has been published for more than a year. 

13. Any other comments? None. 

## **I.4 Preprocessing/cleaning/labeling** 

1. Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? (If so, please provide a description. If not, you may skip the remainder of the questions in this section.) No. 

## **I.5 Uses** 

1. Has the dataset been used for any tasks already? (If so, please provide a description.) Yes. See our paper for details. 

2. Is there a repository that links to any or all papers or systems that use the dataset? (If so, please provide a link or other access point.) No. 

3. What (other) tasks could the dataset be used for? 

   - Many more. Such as generation of implied meanings. 

4. Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? (For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other undesirable harms (e.g., financial harms, legal risks) If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?) No. 

5. Are there tasks for which the dataset should not be used? (If so, please provide a description.) No. 

6. Any other comments? None. 

29 

## **I.6 Distribution** 

1. Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? (If so, please provide a description.) Yes, the dataset is freely available. 

2. How will the dataset will be distributed (e.g., tarball on website, API, GitHub)? (Does the dataset have a digital object identifier (DOI)?) On our website. 

3. When will the dataset be distributed? It’s already been distributed. 

4. Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? (If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.) 

   - The dataset is licensed under a CC license. 

5. Have any third parties imposed IP-based or other restrictions on the data associated with the instances? (If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.) Not to our knowledge. 

6. Do any export controls or other regulatory restrictions apply to the dataset or to individual instances? (If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.) Not to our knowledge. 

7. Any other comments? None. 

## **I.7 Maintenance** 

1. Who is supporting/hosting/maintaining the dataset? The authors. 

2. How can the owner/curator/manager of the dataset be contacted (e.g., email address)? We will post our email address. 

3. Is there an erratum? (If so, please provide a link or other access point.) 

   - Currently, no. As errors are encountered, future versions of the dataset may be released (but will be versioned). They will all be provided in the same location. 

4. Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances’)? (If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?) Yes. 

5. If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)? (If so, please describe these limits and explain how they will be enforced.) No. 

6. Will older versions of the dataset continue to be supported/hosted/maintained? (If so, please describe how. If not, please describe how its obsolescence will be communicated to users.) Yes. 

7. If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? (If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.) Yes. They can email us. 

30 

8. Any other comments? None. 

31 

