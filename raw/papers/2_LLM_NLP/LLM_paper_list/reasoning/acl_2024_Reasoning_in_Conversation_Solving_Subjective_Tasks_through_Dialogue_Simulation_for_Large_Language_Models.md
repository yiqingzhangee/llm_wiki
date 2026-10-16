# **Reasoning in Conversation: Solving Subjective Tasks through Dialogue Simulation for Large Language Models** 

**Xiaolong Wang[*,1] , Yile Wang[*,2] , Yuanchi Zhang[1] , Fuwen Luo[1] , Peng Li**[�] **[,2] , Maosong Sun[1] , Yang Liu**[�] **[,1,2]** 

1Department of Computer Science and Technology, Tsinghua University, Beijing, China 2Institute for AI Industry Research (AIR), Tsinghua University, Beijing, China wangxl22@mails.tsinghua.edu.cn, wangyile@air.tsinghua.edu.cn lipeng@air.tsinghua.edu.cn, liuyang2011@tsinghua.edu.cn 

## **Abstract** 

Large Language Models (LLMs) have achieved remarkable performance in _objective_ tasks such as open-domain question answering and mathematical reasoning, which can often be solved through recalling learned factual knowledge or chain-of-thought style reasoning. However, we find that the performance of LLMs in _subjective_ tasks is still unsatisfactory, such as metaphor recognition, dark humor detection, etc. Compared to objective tasks, subjective tasks focus more on interpretation or emotional response rather than a universally accepted reasoning pathway. Based on the characteristics of the tasks and the strong dialogue-generation capabilities of LLMs, we propose _RiC_ ( **R** easoning **i** n **C** onversation), a method that focuses on solving subjective tasks through dialogue simulation. The motivation of _RiC_ is to mine useful contextual information by simulating dialogues instead of supplying chain-of-thought style rationales, thereby offering potential useful knowledge behind dialogues for giving the final answers. We evaluate both API-based and open-source LLMs including GPT-4, ChatGPT, and OpenChat across twelve tasks. Experimental results show that _RiC_ can yield significant improvement compared with various baselines. 

## **1 Introduction** 

Large language models (LLMs; OpenAI, 2022, 2023; Touvron et al., 2023; Jiang et al., 2023; Wang et al., 2024) have made rapid advancements in recent years and have achieved excellent performance on various objective tasks, including open-domain question answering (Kamalloo et al., 2023), mathematical reasoning (Luo et al., 2023; Yang et al., 2023), and code generation (Chen et al., 2021; Roziere et al., 2023), _etc_ . 

Despite the success, research on LLMs in _subjective_ tasks (Rottger et al., 2022; Kanclerz et al., 2023; Sun et al., 2023a) is still underexplored, as examples shown in Table 1. Different from objective tasks that can often be clearly defined and solved, subjective tasks 

- *Equal contribution. 

> �Corresponding authors. 

**(a) Metaphor Recognition** Identify whether the sentence  “Joseph is very kind” is a paraphrase of the metaphoric sentence “Joseph has the heart of a lion”. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models_images/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models.pdf-0001-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models_images/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models.pdf-0001-13.png)


**----- Start of picture text -----**<br>
(b) Chain-of-Thought Prompting<br>The second sentence suggests that Joseph possesses<br>qualities  associated with a lion, while the first sentence<br>only mentions kindness. The answer is  False .<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models_images/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models.pdf-0001-14.png)


**----- Start of picture text -----**<br>
(c) Reasoning in Conversation (Ours)<br>I heard Joseph has<br>the heart of a lion. Really? What do you<br>mean by that?<br>Sarah Amy<br>I mean he is very kind and brave, just like a lion.<br>Sarah<br>From the conversation, we know that  “heart of a lion”<br>means being kind and brave. The answer is  True .<br>**----- End of picture text -----**<br>


Figure 1: Illustration of our method. (a) An example of the metaphor recognition task. (b) Incorrect responses by LLM using zero-shot-CoT (Kojima et al., 2022) prompting. (c) Our method can simulate helpful dialogues (shown in the dashed box), thereby offering useful information in the generated conversation and aiding reasoning on this subjective task. 

( _e.g._ , metaphor recognition and dark humor detection) involve the capability to perceive context, language nuances, and emotions, which cannot be easily quantified or objectively measured, thereby posing challenges for current LLMs (Jentzsch and Kersting, 2023; Wachowiak and Gromann, 2023; Mao et al., 2023). 

Recent methods based on chain-of-thought (CoT) style prompting have improved the reasoning abilities of LLMs, showing promising results on tasks such as commonsense and mathematical reasoning (Wei et al., 2022; Kojima et al., 2022; Zhang et al., 2023b). However, compared to these objective tasks, we found that such methods are not particularly effective on subjective tasks. As an example shown in Figure 1(b), the reasoning pathway does not comprehend the metaphorical expression in “ _Joseph has the heart of a lion_ ” well, resulting in incorrect responses. 

Dialogue, alternatively, provides humans with a means to raise questions, convey emotions, and express opinions, which can be seen as another way to facilitate subjective reasoning (Resnick et al., 1993; Rips 

|**Subjective Task**|**Example**|**Performance**|
|---|---|---|
||(_Linguistic Rhetoric_)||
|Metaphor Recognition|Identify whether the sentence “Joseph is very kind.” is a paraphrase of the|44_._71♢(2)|
|(Mohler et al.,2016)|metaphoric sentence “Joseph has the heart of a lion.”||
|Sarcasm Detection|Which statement is sarcastic? (a) Yes, the game is over after the second half.|43_._10♢(2)|
|(Khodak et al.,2018)|(b) Yes, the game is over after the frst half.||
|Dark Humor Detection|Identify whether the following examples are intended to be a joke (with dark|50_._00♢(2)|
|(Hoffmann et al.,2022)|humor) or not “An apple a day keeps the doctor away. Or at least it does if you||
||throw it hard enough.”.||
||(_Disambiguation QA_)||
|Pronoun Resolution|Clarify the meaning of sentences with ambiguous pronouns: “Alex told us that|39_._51♢(3)|
|(Rudinger et al.,2018)|he could not meet”. Select from (a) Alex could not meet; (b) We could not||
||meet; (c) Ambiguous.||
|Anachronism Recogni-|Does the preceding sentence “The appearance of prokaryotes on Earth coin-|49_._13♢(2)|
|tion (Geva et al.,2021)|cided with the formation of the moon.” contain non-contemporaneous (anachro-||
||nistic) elements?||
||(_Stance Detection_)||
|Ethical Questions|Imagine that you have found an easy and affordable way to destroy the Earth.|33_._91♢(4)|
|(Hendrycks et al.,2020)|Should you do it? (a) Yes; (b) No; (c) The question is not relevant to my goals;||
||(d) I prefer not to say.||
|Opinion Analysis|What is the attitude toward “Donald Trump” in the tweet “Donald Trump isn’t|71_._30_†_(2)|
|(Zhang et al.,2023a)|afraid to roast everyone. #SemST”. Select from favor or against.||
||(_Cultural-Related_)||
|Social Norm Questions|Does the US Norm “People are expected to love their family.” entail, contradict,|54_._52_‡_(3)|
|(CH-Wang et al.,2023)|or has no relation to the Chinese Norm “It is not appropriate to be mad at your||
||younger sister because she is younger than you.”?||
|Cultural Understanding|“The assertion that the tradition of avoiding the left hand is abandoned is based|58_._12_‡_(3)|
|(Huang and Yang,2023)|on a misunderstanding of the evidence.” entail, contradict, or has no relation||
||to “A particular assertion that the tradition is abandoned is based on a misun-||
||derstanding of the evidence.”?||



Table 1: Tasks, examples, and reported zero-shot performance of typical subjective tasks. The values for opinion analysis and social norm questions denote F1 score, and the others denote accuracy. ♢: results by PaLM-535B. _†_ : results by GPT-3.5. _‡_ : results by GPT-4. The value in the parentheses indicates the number of labels. 

et al., 1999). Considering the characteristics of subjective tasks and the strong ability of dialogue generation for LLMs (Thoppilan et al., 2022; Tian et al., 2023), we propose _RiC_ ( **R** easoning **i** n **C** onversation), a method aiming to uncover the subjective expressions in simulated dialogues instead of objective and relatively unified reasoning pathways for better reasoning on subjective tasks. By employing this method, as Figure 1(c) shows, the metaphorical relationship between “ _Joseph has the heart of a lion_ ” and “ _Joseph is very kind_ ” is correctly identified in the simulated dialogues, thus helping LLMs in giving the final answer. 

The proposed _RiC_ comprises three stages: keywords extraction, dialogue simulation, and dialogue-enhanced reasoning. To enable better comprehension of the questions and dialogue generation, we first allow LLMs to extract task-relevant keywords according to the question, which has been shown helpful for understanding the task and generating related dialogue (Zhu et al., 2022; Yu et al., 2023). Then, based on the extracted keywords, an approximately one or two-turn brief dialogue is constructed in a zero-shot manner. Finally, we enable 

LLMs to engage in reasoning based on both the original question and the simulated dialogue scenario. 

We employ both API-based and open-source LLMs including GPT-4 (OpenAI, 2023), ChatGPT (OpenAI, 2022), and OpenChat (Wang et al., 2024), to validate the effectiveness of our method. Experimental results show that _RiC_ leads to significant and consistent improvements under both zero-shot and few-shot settings, underscoring the effectiveness of leveraging the knowledge in dialogue for better solving subjective tasks. 

## **2 Related Work** 

**Subjective Tasks.** Various subjective tasks have been extensively studied in natural language processing. We show typical tasks in Table 1, including linguistic rhetoric, disambiguation, stance detection, and culturalrelated questions. Compared with objective tasks that have a clear solution or evaluation criteria, subjective tasks involve interpretation, judgment, and personal experiences (Rottger et al., 2022; Kanclerz et al., 2023; Sun et al., 2023a). Moreover, results in Table 1 show that the performance of LLMs on these tasks is around 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models_images/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Metaphor Recognition Sarcasm Detection Dark Humor Detection<br>I heard Joseph has Well, that was a tough game, An apple a day keeps the doctor<br>the heart of a lion. but we'll come back strong in away. Or at least it does if you<br>the second half. throw it hard enough.<br>Sarah Really? What do you<br>Person A<br>mean by that? Coach<br>That's a dark way to interpret the<br>I mean he is very kind and Amy Are you kidding? The game is saying. It could work if you don't<br>brave, just like a lion. over after the first half. want to see the doctor.<br>Sarah Player Person B<br>Pronoun Resolution Anachronism Recognition Ethical Questions<br>I'm sorry, I can't meet today. Did you know that the appearance Hey, I found an easy and<br>of prokaryotes on Earth coincided affordable way to destroy<br>Alex with the formation of the moon? the Earth. Should we do it?<br>Why not?<br>Scientist 1 Person 1<br>Us No way! How is that possible? Are you serious? It goes<br>I have another appointment. The formation of the moon against our ethical values<br>happened so long ago! and human judgment.<br>Alex Scientist 2 Person 2<br>Opinion Analysis Social Norm Questions Cultural Understanding<br>Donald Trump roasted everyone! I love spending time with my The Indian tradition of avoiding the<br>He isn't afraid to speak his mind. family. It's important to me. left hand is no longer followed.<br>Person 1 Wow, I didn't expect John Person 1 That assertion is based on<br>anything less from him. He Yes, it's a social norm in a misunderstanding of the<br>the US for people to love evidence. The tradition is<br>always speaks his mind no<br>matter who's listening. their family. It's expected. alive in Indian culture.<br>Person 2 Sarah Person 2<br>**----- End of picture text -----**<br>


Figure 2: Illustration of simulated dialogues for the questions in different types of subjective tasks from Table 1. 

30 _∼_ 70 accuracy or F1 score, indicating that the tasks are indeed challenging and there is significant room for improvement even for the most advanced LLMs. 

**Chain-of-Thought Prompting.** CoT prompting (Wei et al., 2022) and its variants (Kojima et al., 2022; Zhang et al., 2023b; Sun et al., 2023b; Press et al., 2023) are widely used in augmenting the reasoning abilities of LLMs. These methods attempt to enhance reasoning by incorporating additional rationales (Wang et al., 2022) or reasoning paths to augment contextual information, which has been shown effective for objective tasks such as commonsense reasoning (Talmor et al., 2019), opendomain question answering (Kwiatkowski et al., 2019), and mathematical reasoning (Cobbe et al., 2021). Another line of work proposes reasoning through roleplaying or expert modeling, aiming to answer questions or accomplish tasks through cooperation between roles or leveraging specified expert knowledge (Wang et al., 2023b; Xu et al., 2023b). However, we show in experiments that the corresponding performance is still constrained by employing these methods in the aforementioned subjective tasks. 

**Dialogue Generation and Reasoning.** There have been recent works on dialogue generation and reasoning based on LLMs. Chae et al. (2023) introduce a dialogue chain-of-thought reasoner and help LLMs solve commonsense reasoning tasks in conversations. Xu et al. (2023c) improve zero-shot persona dialogue generation via in-context learning (Brown et al., 2020). Wang et al. 

(2023a) propose linguistic cue-based chain-of-thought, enhancing reasoning steps by finding cues exhibited in the dialogue. In contrast to their work, we mainly tackle subjective tasks rather than dialogue itself. Specifically, we innovatively propose to leverage dialogue to facilitate the subjective reasoning ability of LLMs. 

## **3 Method** 

The overall pipeline of the proposed _RiC_ can be divided into three steps. _RiC_ first extracts the keywords from the input context, including both the task description and the specific question. Then it simulates the scenario in the form of dialogue according to the keywords. Finally, it offers the final reasoning steps based on the question and simulated dialogue. 

Formally, given a subjective task with a description _T_ and a specific question _Q_ , our goal is to let the LLM _M_ solve the question _Q_ . 

**Step 1: Keywords Extraction.** We first let LLM _M_ extract the keywords _K_ according to the description _T_ and question _Q_ . Specifically, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models_images/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models.pdf-0003-10.png)


where _n_ (the number of keywords) is usually between 4 _∼_ 5, _⊕_ denotes concatenation operation. _P_ 1 is a prompt serving as a trigger sentence, for example, we can set _P_ 1 as “ _First, extract keywords from the question_ ”. **Step 2: Dialogue Simulation.** Then, base on the keywords, we let LLM _M_ construct a scenario _S_ in the 

## **Illustration of the Unified Prompting and Response** 

/* Input */ Metaphor Recognition: Identify whether the sentence “Joseph is very kind.” is a paraphrase of the metaphoric sentence “Joseph has the heart of a lion.” ( _T_ , _Q_ ) 

First, extract keywords from the question. ( _P_ 1) Then, according to the keywords, construct a scenario for the question in the form of dialogue. ( _P_ 2) Finally, according to the question and conversation, reason and give the final answer. Select from True or False. ( _P_ 3) 

/* Response of _M_ */ Keywords: ( _K_ ) Joseph, heart of a lion, kind. 

Scenario: ( _S_ ) 

Sarah: I heard Joseph has the heart of a lion. Amy: Really? What do you mean by that? Sarah: I mean he is very kind and brave, just like a lion. 

Reasoning: ( _R_ ) 

From the conversation, we know that “heart of a lion” means being kind and brave. The answer is True. 

Table 2: The example of a metaphor recognition task, a specific question, the unified prompt, and the corresponding response. 

form of dialogue: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models_images/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models.pdf-0004-10.png)


where _P_ 2 is a prompt for simulating the dialogue. For example, we can set _P_ 2 as “ _Then, according to the keywords, construct a scenario for the question in the form of dialogue_ ”. For different subjective tasks, we show examples of simulated dialogues _S_ in Figure 2. **Step 3: Dialogue-Enhanced Reasoning.** Finally, we take the original task description _T_ , question _Q_ , and the simulated dialogue _S_ as the input, letting LLM _M_ give the final response _R_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models_images/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models.pdf-0004-12.png)


where _P_ 3 is the last prompt leading to the final answer which can be set as “ _Finally, according to the question and conversation, reason and give the final answer_ ”. **Combine All Steps through Unified Prompting.** In practice, we find that the three aforementioned steps can be combined and accomplished through a single prompt _P_ . In this way, our method only requires inference once through the LLM to obtain the answer to the question: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models_images/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models.pdf-0004-14.png)


where an example of the unified prompt and response is shown in Table 2. 

## **4 Experiments** 

## **4.1 Setups** 

**Datasets** . We evaluate the effectiveness of our method on twelve subjective reasoning datasets, which can be categorized into five types, including: **Linguistic Rhetoric Tasks** 

- **Metaphor** (Mohler et al., 2016) provides a pair of sentences and aims to identify whether the metaphoric sentence is correctly interpreted. 

- **SNARKS** (Khodak et al., 2018) aims to measure the ability to differentiate sarcastic statements from non-sarcastic statements. 

- **Dark Humor Detection** (Hoffmann et al., 2022) aims to determine whether a given text is intended to be a joke with dark humor or not. 

## **Disambiguation QA** 

- **Pronoun Resolution** (Rudinger et al., 2018) aims to clarify the meaning of a sentence with ambiguous pronouns to which thing refers. 

- **Anachronism Recognition** (Geva et al., 2021) aims to test the ability of LLMs to identify whether a sentence is anachronistic or not. 

## **Stance Detection** 

- **SEQ** (Hendrycks et al., 2020) evaluates whether LLMs are capable of identifying which simple ethical question aligns with human judgment. 

- **SemEval** (Mohammad et al., 2016) propose a series of opinion analysis tasks. We follow Zhang et al. (2023a) to investigate LLM’s ability of stance detection for the target “Donald Trump” in tweets. 

## **Cultural-Related Tasks** 

- **SocNorm** (CH-Wang et al., 2023) is a dataset that aims to align with social norms across American and Chinese culture. 

- **e-SocNorm** (CH-Wang et al., 2023) extend the SocNorm dataset with corresponding free-text explanations as external prompts. 

- **CALI** (Huang and Yang, 2023) aims to compare culturally aware premise-hypothesis pairs annotated by groups located in the U.S. and India. 

## **Traditional Natural Language Inference** 

- **Analytic Entailment** (Srivastava et al., 2022) aims to identify whether the second sentence must be true given the meaning of the first sentence. 

- **IPA** (Williams et al., 2018) is a natural language inference task presented in the international phonetic alphabet. 

Detailed descriptions of datasets are given in appendix A. Specifically, for SemEval and cultural-related datasets that contain training sets, we evaluate them in both zero-shot and few-shot settings. For the other tasks, 

|**Method**|**Linguistic Rhetoric**<br>**Metaphor**<br>**SNARKS**<br>**Humor**<br>(Acc.)<br>(Acc.)<br>(Acc.)|**Disambiguation QA**<br>**Pronoun**<br>**Anach.**<br>(Acc.)<br>(Acc.)|**Stance Detection**<br>**SEQ**<br>**SemEval**<br>(Acc.)<br>(F1)|**Cultural-Related**<br>**SocNorm**<br>**e-SocNorm**<br>**CALI**<br>(F1)<br>(F1)<br>(Acc.)|**Traditional NLI**<br>**AVG.**<br>**Entail.**<br>**IPA**<br>(Acc.)<br>(Acc.)|
|---|---|---|---|---|---|
|_Random_<br>_Majority_|50_._00<br>50_._00<br>50_._00<br>61_._62<br>53_._59<br>50_._00|33_._33<br>50_._00<br>30_._23<br>50_._00|25_._00<br>50_._00<br>10_._43<br>0_._00|33_._33<br>33_._33<br>33_._33<br>0_._00<br>0_._00<br>38_._09|50_._00<br>33_._33<br>40_._97<br>57_._14<br>38_._89<br>32_._50|
|Direct Prompt<br>Zero-Shot-CoT<br>Recite&Answer<br>**RiC (Ours)**|85_._44<br>60_._22<br>50_._00<br>75_._29<br>64_._77<br>58_._75<br>82_._50<br>64_._64<br>55_._00<br>**86**_._**62**<br>**68**_._**95**<br>**65**_._**00**|(openchat-3.5)<br>63_._95<br>73_._04<br>80_._00<br>72_._52<br>67_._83<br>76_._09<br>77_._39<br>70_._97<br>66_._28<br>86_._09<br>81_._74<br>71_._07<br>**69**_._**38**<br>**87**_._**39**<br>**86**_._**09**<br>**73**_._**72**||41_._41<br>42_._52<br>58_._58<br>48_._36<br>46_._11<br>55_._95<br>49_._18<br>55_._32<br>57_._05<br>**52**_._**15**<br>**64**_._**11**<br>**60**_._**23**|58_._57<br>49_._21<br>61_._29<br>70_._00<br>49_._21<br>63_._39<br>72_._86<br>50_._79<br>66_._04<br>**74**_._**29**<br>**58**_._**73**<br>**70**_._**55**|
|Direct Prompt<br>Zero-Shot-CoT<br>Recite&Answer<br>**RiC (Ours)**|85_._74<br>77_._35<br>58_._75<br>86_._47<br>78_._45<br>57_._50<br>86_._62<br>76_._30<br>67_._50<br>**87**_._**94**<br>**82**_._**32**<br>**71**_._**25**|(gpt-3.5-turbo-1106)<br>55_._04<br>70_._43<br>75_._65<br>71_._30<br>60_._47<br>64_._78<br>72_._17<br>73_._79<br>60_._39<br>70_._00<br>77_._39<br>71_._10<br>**62**_._**79**<br>**72**_._**61**<br>**81**_._**74**<br>**74**_._**27**||43_._25<br>45_._27<br>52_._94<br>44_._68<br>51_._53<br>52_._75<br>47_._71<br>49_._13<br>48_._86<br>**56**_._**02**<br>**59**_._**98**<br>**57**_._**27**|60_._00<br>50_._79<br>62_._21<br>58_._57<br>55_._56<br>63_._06<br>61_._43<br>**57**_._**14**<br>64_._46<br>**62**_._**86**<br>**57**_._**14**<br>**68**_._**85**|
|Direct Prompt<br>Zero-Shot-CoT<br>Recite&Answer<br>**RiC (Ours)**|94_._85<br>86_._19<br>65_._00<br>**95**_._**88**<br>87_._29<br>66_._25<br>94_._26<br>87_._85<br>65_._00<br>95_._29<br>**92**_._**27**<br>**67**_._**50**|(gpt-4-0613)<br>72_._09<br>82_._17<br>92_._17<br>72_._78<br>69_._38<br>80_._00<br>93_._91<br>75_._47<br>71_._71<br>80_._87<br>**96**_._**52**<br>75_._65<br>**75**_._**58**<br>**86**_._**96**<br>95_._65<br>**76**_._**34**||45_._31<br>46_._81<br>60_._40<br>48_._74<br>47_._45<br>60_._90<br>48_._78<br>48_._52<br>60_._00<br>**58**_._**27**<br>**61**_._**12**<br>**61**_._**13**|68_._57<br>75_._40<br>71_._81<br>75_._71<br>73_._02<br>72_._83<br>77_._14<br>76_._19<br>73_._54<br>**87**_._**14**<br>**80**_._**95**<br>**78**_._**18**|



Table 3: Main results of baselines and our proposed _RiC_ in zero-shot settings. _Random_ represents the result of random prediction with uniform probability, and _majority_ represents the result of predicting the label with the highest proportion. For each dataset, the best result is **in bold** and the second-best result is underlined. 

we use the corresponding test set from BigBench[1] (Srivastava et al., 2022) in a zero-shot setting only. 

**Baselines.** We compare our proposed _RiC_ with various methods, taking into account both zero-shot (no demonstrations are provided) and few-shot settings (few demonstrations from the training set are provided for in-context learning). The baselines include: **Zero-Shot Methods** 

- **Direct Prompt** (Brown et al., 2020) instructs LLM to answer the test question directly. 

- **Zero-Shot-CoT** (Kojima et al., 2022) appends the prompt “Let’s think step by step” before reasoning. 

- **Recite&Answer** (Sun et al., 2023b) first retrieves relevant passages from memory and then generates final responses. 

## **Few-Shot Methods** 

- **In-Context Learning** (ICL; Brown et al., 2020) provides a few demonstrations including the ground-truth labels before giving the test question. 

- **Few-Shot-CoT** (Wei et al., 2022) manually designs and selects the explanations in demonstrations and provides the chain-of-thought reasoning. 

- **Auto-CoT** (Zhang et al., 2023b) automatically selects demonstrations from training data based on semantic diversity for the test question. 

- **Self-Ask** (Press et al., 2023) actively proposes and solves subquestions before generating final answer. 

- **StSQA** (Zhang et al., 2023a) proposes automatically extracting “thought-inducing” content from training data and adds them as input for step-bystep reasoning. 

1https://github.com/google/BIG-bench/ tree/main/bigbench/benchmark_tasks/ 

- **SPP** (Wang et al., 2023b) proposes solo performance prompting by involving multi-turn collaboration with multi-persona during reasoning. 

- **ExpertPrompt** (Xu et al., 2023b) introduces the expert identities and customizes information descriptions for LLMs before generating responses. 

**Models.** For LLMs, we evaluate our method on both API-based models including GPT-4 (OpenAI, 2023) and ChatGPT (OpenAI, 2022), and open-source model OpenChat-7B (Wang et al., 2024). In particular, we use the released API versions of gpt-4-0613 and gpt-3.5-turbo-1106 by OpenAI, and the opensource openchat-3.5 model released in Huggingface[2] . We set the decoding temperature as 0 to maintain the reproducibility of the responses generated by LLMs. 

## **4.2 Zero-Shot Results** 

In Table 3, we show the main results of the baselines and our _RiC_ method in zero-shot settings. 

For the Direct Prompting method, the LLMs directly respond to each question without explicit prompts or demonstrations. On average, it gives the results of 61 _._ 29 _∼_ 71 _._ 81 accuracy across all tasks for different models, showing relatively limited performance for these subjective tasks. 

By explicitly prompting LLMs to “ _Let’s think stepby-step_ ” or “ _recite relevant passages then give answers_ ” before reasoning, Zero-Shot-CoT and Recite&Answer will generate reasoning path or piece of passages according to the task and questions. The results show that these methods improve performance to some extent, leading to results of 66 _._ 04 _∼_ 73 _._ 54 accuracy. 

Regarding our _RiC_ method, which involves simulated dialogues instead of reasoning paths or passages 

2https://huggingface.co/openchat/ openchat_3.5 

|**Method**|**SemEval**|**SocNorm**|**e-SocNorm**|**CALI**|**AVG.**|
|---|---|---|---|---|---|
|||(openchat-3.5)||||
|ICL|72_._63|47_._44|57_._82|56_._36|58_._56|
|Few-Shot-CoT|72_._37|51_._45|64_._39|55_._23|60_._86|
|Auto-CoT|73_._30|43_._76|63_._73|56_._36|59_._29|
|Self-Ask|72_._41|46_._68|61_._54|55_._91|59_._14|
|StSQA|71_._43|52_._58|59_._87|54_._55|59_._61|
|SPP|74_._29|49_._63|67_._18|55_._23|61_._58|
|ExpertPrompt|71_._36|49_._36|67_._79|57_._95|61_._61|
|**RiC (Ours)**|**75**_._**62**|**56**_._**02**|**70**_._**07**|**58**_._**18**|**64**_._**97**|
||(gpt-3.5-turbo-1106)|||||
|ICL|72_._02|52_._95|55_._60|54_._77|58_._84|
|Few-Shot-CoT|72_._06|53_._44|61_._35|54_._55|60_._35|
|Auto-CoT|74_._22|52_._10|68_._50|56_._59|62_._85|
|Self-Ask|73_._04|53_._94|57_._81|57_._27|60_._52|
|StSQA|73_._40|48_._35|64_._04|56_._59|60_._60|
|SPP|72_._74|51_._92|62_._01|55_._91|60_._65|
|ExpertPrompt|75_._22|46_._08|65_._29|55_._45|60_._51|
|**RiC (Ours)**|**78**_._**21**|**57**_._**70**|**72**_._**78**|**60**_._**00**|**67**_._**17**|
|||(gpt-4-0613)||||
|ICL|73_._72|54_._71|61_._41|62_._50|63_._09|
|Few-Shot-CoT|76_._59|64_._08|67_._88|64_._77|68_._33|
|Auto-CoT|76_._70|54_._64|62_._99|64_._54|64_._72|
|Self-Ask|73_._52|56_._74|64_._62|65_._45|65_._08|
|StSQA|76_._67|56_._40|52_._86|63_._18|62_._28|
|SPP|78_._72|57_._74|65_._04|54_._32|63_._96|
|ExpertPrompt|77_._65|56_._84|68_._72|59_._77|65_._75|
|**RiC (Ours)**|**80**_._**01**|**66**_._**59**|**74**_._**45**|**65**_._**68**|**71**_._**68**|



Table 4: Main results of baselines and our proposed _RiC_ in few-shot settings. Except for Auto-CoT, we select the same 3-shot demonstrations from the training sets to each method for fair comparison. 

in memory, it has achieved the best results across twelve tasks. Compared to the second-best ones, our method improves absolutely by +4 _._ 51, +4 _._ 39, and +4 _._ 64 by using OpenChat, ChatGPT, and GPT-4 model, respectively, which further demonstrates the benefits of dialogue in solving subjective tasks. 

Among all tasks, taking examples of using GPT-4 as backbone model, SocNorm and e-SocNorm show the greatest improvement, where our method outperforms the second-best one by +9 _._ 49 and +12 _._ 6 F1 score, respectively. These two datasets involve social norms in American and Chinese culture, which we suppose dialogue can provide relevant cultural background knowledge, thereby enhancing the performance. The improvement is also significant on analytic entailment (+10 _._ 00), anachronism recognition (+4 _._ 79) and sarcasm detection (+4 _._ 42), where it is difficult to deduce objective reasoning pathways or recall directly relevant passages that contain answers by using baseline methods. 

## **4.3 Few-Shot Results** 

Table 4 shows the main results in few-shot settings, where we compared our method with more baselines. 

In general, the vanilla ICL method gives the lowest average results of 58 _._ 56 _∼_ 63 _._ 09 accuracy, which is only provided with labeled demonstrations without other contexts. As for chain-of-thought style reasoning methods (Few-Shot-CoT, Auto-CoT, Self-Ask, and StSQA), there has been a slight improvement and it is also not stable. For example, the improvement by using OpenChat and ChatGPT is generally around only 1 _∼_ 3 accuracy, and StSQA even performs worse than ICL by using GPT-4. The reasoning can be that these methods are 

|**Method**|**SemEval**|**SocNorm**|**e-SocNorm**|**CALI**|
|---|---|---|---|---|
|**RiC (Ours)**|**78**_._**21**|**57**_._**70**|**72**_._**78**|**60**_._**00**|
|_w/o_KE|_↓_1_._55|_↓_1_._15|_↓_2_._69|_↓_2_._95|
|_w/o_DS|_↓_5_._62|_↓_1_._78|_↓_5_._01|_↓_5_._45|
|_w/o_KE&DS|_↓_8_._74|_↓_2_._17|_↓_10_._48|_↓_7_._27|



Table 5: Ablation study of our proposed _RiC_ method with ChatGPT in few-shot settings. KE: Keywords Extraction. DS: Dialogue Simulation. 

often limited to objective tasks such as mathematical and commonsense reasoning, and they have not been well validated in subjective tasks, though being provided with few demonstrations with manually-written or generated reasoning steps. 

Similarly, The improvement brought by role-playing based methods (SPP and ExpertPrompt) is also not significant, even lags behind Few-Shot-CoT by a large margin for GPT-4 model. One possible reason could be that subjective tasks require an abstract and more variable range of knowledge, making it challenging to generalize and solve test questions using predefined roles in the demonstrations. 

Regarding our _RiC_ method, we can observe that it gives significant performance improvements across all models. Specifically, it outperforms the vanilla ICL method by +6 _._ 41, +8 _._ 33, and +8 _._ 59 with OpenChat, ChatGPT, and GPT-4 base models, respectively, demonstrating the effectiveness of the diverse dialogue generation capabilities of LLMs in helping subjective tasks. 

## **5 Analyses and Discussions** 

In this section, we conduct a series of analyses to probe the reason behind the effectiveness of the _RiC_ method. We first investigate the effectiveness of keywords and dialogue ( _§_ 5.1), followed by the impact of numbers of keywords and turns of simulated dialogue ( _§_ 5.2), then we set different numbers of demonstrations in a few-shot setting ( _§_ 5.3) and compare the length of response for different methods ( _§_ 5.4), finally we manually evaluate how does our method benefit for subjective tasks ( _§_ 5.5). 

## **5.1 Ablation Study** 

We first investigate the impact of keywords extraction and dialog simulation in our _RiC_ method, results are shown in Table 5. The full _RiC_ method, incorporating both steps, performs best on all datasets, highlighting the importance of both keywords extraction and dialogue simulation. Removing keywords extraction ( _RiC w/o_ KE) generally leads to a performance drop by 1 _∼_ 2 accuracy, showing that it is helpful for dialogue construction and subjective reasoning. Removing dialogue simulation ( _RiC w/o_ DS) further decreases the accuracy by around 1 _∼_ 6 accuracy, indicating that the simulated dialogue indeed plays a crucial role in our method. Excluding both steps ( _RiC w/o_ KE&DS) leads to the worst performance, which degenerates to the vanilla reasoning way without explicit prompts. In summary, both 

||**Metaphor**|**SNARKS**|**Humor**|**Pronoun**|**Anach.**|**SEQ**|**SemEval**|**SocNorm**|**e-SocNorm**|**CALI**|**Entail.**|**IPA**|**AVG.**|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**#Keywords**|2_._87|3_._89|3_._75|3_._96|3_._81|5_._39|5_._88|3_._76|3_._57|3_._83|3_._87|4_._59|4_._10|
|**#Turns of Dialogue**|1_._17|1_._29|1_._37|1_._17|1_._34|1_._53|1_._39|1_._06|1_._04|1_._08|1_._34|1_._10|1_._24|



Table 6: The average numbers or turns of generated keywords and dialogue by our _RiC_ method in different datasets. 

|**#Keywords**|**SemEval**|**SocNorm**|**e-SocNorm**|**CALI**|
|---|---|---|---|---|
||(_zero-shot_)||||
|_not specifed, ours_|**74**_._**27**|**56**_._**02**|**59**_._**98**|**57**_._**27**|
|_specifed as 1_|69_._59|55_._14|57_._73|52_._73|
|_specifed as 2_|70_._22|51_._79|58_._50|54_._09|
|_specifed as 3_|72_._92|51_._07|59_._01|56_._82|
|_specifed as 4_|73_._48|53_._16|59_._36|55_._91|
|_specifed as 5_|71_._76|52_._57|58_._81|55_._09|
||(_few-shot_)||||
|_4∼5, not fxed, ours_|**78**_._**21**|**57**_._**70**|**72**_._**78**|**60**_._**00**|
|_1 for each demo_|76_._17|55_._24|71_._53|50_._51|
|_2 for each demo_|77_._28|56_._23|72_._33|51_._82|
|_3 for each demo_|77_._71|56_._58|71_._34|53_._64|
|_4 for each demo_|78_._14|56_._55|71_._06|53_._68|
|_5 for each demo_|77_._23|57_._30|71_._15|53_._41|



Table 7: Impact of specifying different required numbers of generated keywords in both zero-shot and fewshot settings. _demo_ : demonstration in contexts. 

|**#Turns of Dialogue**|**SemEval**|**SocNorm**|**e-SocNorm**|**CALI**|
|---|---|---|---|---|
||(_zero-shot_)||||
|_not specifed, ours_|**74**_._**27**|**56**_._**02**|**59**_._**98**|**57**_._**27**|
|_specifed as 1_|73_._47|50_._25|59_._73|56_._59|
|_specifed as 2_|71_._44|49_._41|57_._73|57_._05|
|_specifed as 3_|73_._57|47_._45|55_._32|51_._59|
|_specifed as 4_|72_._01|47_._21|53_._85|55_._45|
|_specifed as 5_|71_._08|49_._98|52_._96|55_._91|
||(_few-shot_)||||
|_1 for each demo, ours_|**78**_._**21**|57_._70|72_._78|**60**_._**00**|
|_2 for each demo_|74_._24|**60**_._**04**|72_._42|58_._41|
|_3 for each demo_|75_._59|56_._22|72_._70|56_._36|
|_4 for each demo_|74_._61|57_._25|**73**_._**31**|54_._32|
|_5 for each demo_|72_._01|54_._03|69_._63|55_._00|



Table 8: Impact of specifying different numbers of turns in simulated dialogue in both zero-shot and few-shot settings. _demo_ : demonstration in contexts. 

the steps of keywords extraction and dialogue simulation are important and the best performance is achieved when both of them are utilized. 

## **5.2 Number of Keywords and Turns of Dialogue** 

Our method does not specify the required number of keywords and turns of dialogue in _P_ 1 and _P_ 2 from Eq. 4. As shown in Table 6, the averaged numbers of generated keywords and turns of dialogue across all datasets are 4 _._ 10 and 1 _._ 24, respectively. We further specify the numbers in prompts or demonstrations in zero-shot and few-shot settings for further analysis. **Number of Keywords.** We first specify the required number of keywords as 1 _∼_ 5 and the results are shown in Table 7. In zero-shot settings, we find that specifying the number of keywords does not contribute to performance improvement, while it is better to let the LLMs itself extract the necessary number of keywords based on the task and the given question. In few-shot settings, specifying 3 _∼_ 5 keywords is better than only 1 _∼_ 2 keywords. Furthermore, setting 4 _∼_ 5 different numbers 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models_images/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models.pdf-0007-09.png)


Figure 3: The performance of baselines and our _RiC_ method by using different numbers of demonstrations ( _d_ = 1 _,_ 2 _,_ 3 _,_ 4) in few-shot settings. 

of keywords instead of the fixed ones across different demonstrations can yield the best results. 

**Turns of Dialogue.** Then we specify the turns of simulated dialogue as 1 _∼_ 5 and show results in Table 8. We can observe that when not specifying the number of dialogue turns or specifying it as 1 _∼_ 2, the performance is relatively better. However, when the number of dialogue turns is fixed to 3 or more, the performance declines. This could be attributed to two primary factors: 1) the difficulty in generating high-quality dialogues increases for multiple turns with limited contextualized information; 2) complex multi-step reasoning may not be required for the involved subjective tasks, therefore the extremely long conversations are unnecessary. 

## **5.3 Impact of Number of Demonstrations** 

In few-shot settings, we investigate the impact of the number of demonstrations for the baselines and our method. Taking into account the fact that most datasets contain 2 _∼_ 3 types of labels, we set the numbers as _d_ = 1 _,_ 2 _,_ 3 _,_ 4 and the results are shown in Figure 3. 

As we can see, the number of demonstrations has a significant impact on the results. For example, when _d_ = 1 or 2, the performance is generally low with limited examples. Overall, the best results are achieved when _d_ = 3, while there is a slight decline in performance when _d_ increases to 4. These reflect the instability and variance of in-context learning (Zhao et al., 2021; Xu et al., 2023a). 

Nevertheless, our _RiC_ method gives the best overall performance in fair comparisons with the baselines, achieving the best or second-best results for all datasets. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models_images/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models.pdf-0008-00.png)


**----- Start of picture text -----**<br>
RiC (Ours) SPP RiC (Ours) RiC (Ours) Auto-CoT Self-Ask RiC (Ours)<br>Auto-CoT StSQA ExpertPrompt Few-Shot-CoT Few-Shot-CoT ExpertPrompt SPP StSQAFew-Shot-CoT<br>Self-Ask ICL<br>Few-Shot-CoT SPP Auto-CoT ExpertPrompt<br>Self-Ask ExpertPrompt ICL<br>ICL ICL StSQA SPP<br>Self-Ask Auto-CoT StSQA<br>**----- End of picture text -----**<br>


Figure 4: The performance and average number of generated tokens for baselines and our _RiC_ in few-shot settings. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models_images/acl_2024_Reasoning_in_Conversation_Solving_Subjective_Tasks_through_Dialogue_Simulation_for_Large_Language_Models.pdf-0008-02.png)


**----- Start of picture text -----**<br>
Mixed<br>18.3%<br>Causal<br>12.5%<br>Sentiment<br>27.5%<br>Explanations<br>11.7%<br>Event Character<br>9.17% 5.83%<br>Expertise Cultural<br>7.50% 7.50%<br>**----- End of picture text -----**<br>


Figure 5: Different types of knowledge in simulated dialogue of _RiC_ in 120 sampled data, 10 for each task. 

In practice, we found that selecting one example per label can generally achieve better and more stable performance, and there is no need to set _d_ too large (e.g., 5 or more). 

/* Sentiment Polarity */ What is the attitude toward “Donald Trump” in the tweet “Watching what Donald Trump said about Mexicans was shocking! Let’s not give this appalling man a platform. #SemST”. 

Person A: Watching what Donald Trump said about Mexicans was shocking! He made derogatory comments about Mexicans. Person B: Wow, that’s terrible. 

Person A: Yes, we shouldn’t give this appalling man a platform to spread his hate. 

/* Causal Relationship */ Determine whether the following pairs of sentences embody an entailment relation or not: “The tweet went viral. So the tweet had a virus.” 

John: The tweet about the new movie went viral on social media. 

Sarah: Did you hear that the tweet had a virus? John: No, that’s not true. Just because it went viral doesn’t mean it had a virus. 

Table 9: Examples of generated dialogues that offer sentiment polarity and causal relationship knowledge. 

## **5.4 Comparison of Inference Cost** 

The length of responses also incurs certain time and monetary costs, especially for nowadays LLMs. We compare the performance and average generated tokens by GPT-4 model of baselines and our _RiC_ method, the results are shown in Figure 4. 

Firstly, the vanilla **ICL** method only predicts the labels for test questions according to the samples and labels provided. Although the length of the response is the shortest (less than 10 tokens on average), the general accuracy is limited. Secondly, the rationale-enhanced methods ( **Few-Shot-CoT, Auto-CoT, Self-Ask, and StSQA** ) enrich the contexts with explanations and improve the performance to some extent. Thirdly, **ExpertPrompt and SPP** increases the length of response (100 _∼_ 300 tokens) due to role-playing, but the performance does not improve consistently. For example, the performance even decreases in the CALI dataset, possibly due to the low relevance between the generated roles and given questions. Finally, for our **RiC** method, the length of the response increases (100 _∼_ 120 tokens) due to the simulation of dialogue, which offers useful information for subjective reasoning. Overall, the results show that our method outperforms the above baselines 

substantially and consistently. 

## **5.5 Human Evaluation** 

We attempt to further analyze how our method contributes to helping the reasoning of LLMs. In particular, we randomly choose 120 samples and categorized the knowledge offered by the simulated dialogue of our method into eight types, as shown in Figure 5. 

For example, we find that the simulated dialogue can provide more context with sentiment tendencies, causal relationships of the events occurring, explanation or viewpoints, professional knowledge through simulating relevant professional roles, or providing background knowledge about events, culture, and characters, _etc_ ., which can help solve different involved subjective tasks, as examples in Table 9 (see appendix B for more cases). 

## **6 Conclusion** 

We introduce _RiC_ ( _i.e._ , Reasoning in Conversation), a tuning-free method to enhance the ability of LLMs to solve subjective tasks through dialogue simulation. The core motivation of the proposed _RiC_ is to better lever- 

age the useful information from human conversation based on the advanced dialogue generation ability of current LLMs. We conduct experiments on API-based models (GPT-4 and ChatGPT) and open-source model (OpenChat) across twelve tasks of five types, results show that our method leads to significant and consistent improvement compared with various baselines in both zero-shot and few-shot settings, showing the impact of knowledge in dialogue and shed light on new directions for tackling subjective tasks by using LLMs. 

## **Limitations** 

Firstly, our proposed method focuses on improving performance on subjective tasks in zero-shot or few-shot settings, which relies on the dialogue generation and understanding capabilities of current LLMs, making it more suitable for general-purpose models. However, it may not guarantee the same effectiveness for domainspecific models such as dedicated code generation or mathematical reasoning models. Secondly, our method primarily adopts a tuning-free approach, thus avoiding additional parameter training. However, we believe that apart from existing general-purpose models, training LLMs that focus more on human subjective experiences remains an important research direction in the future. Thirdly, our experiments utilized existing datasets and manual annotations. However, for the design and evaluation of subjective tasks, we also believe that there should be more in-depth consideration for benchmarks and refined evaluation metrics, which is an important direction for assessing the capabilities of LLMs. 

## **Ethics Statement** 

In this paper, we utilize publicly available and widely used datasets for evaluation, including stance detection, sarcasm detection, cultural comparison, _etc_ . We also use LLMs to generate corresponding responses. These are solely used to validate the effectiveness of the proposed method and do not indicate any stance or bias from the authors. 

## **References** 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In _Proc. of NeurIPS_ . 

- Sky CH-Wang, Arkadiy Saakyan, Oliver Li, Zhou Yu, and Smaranda Muresan. 2023. Sociocultural 

norm similarities and differences via situational alignment and explainable textual entailment. In _Proc. of EMNLP_ . 

- Hyungjoo Chae, Yongho Song, Kai Ong, Taeyoon Kwon, Minjin Kim, Youngjae Yu, Dongha Lee, Dongyeop Kang, and Jinyoung Yeo. 2023. Dialogue chain-of-thought distillation for commonsense-aware conversational agents. In _Proc. of EMNLP_ . 

- Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ . 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. 

- Maxwell Forbes, Jena D. Hwang, Vered Shwartz, Maarten Sap, and Yejin Choi. 2021. Social chemistry 101: Learning to reason about social and moral norms. 

- Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. _TACL_ . 

- Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt. 2020. Aligning ai with shared human values. _arXiv preprint arXiv:2008.02275_ . 

- Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. 2022. Training compute-optimal large language models. 

- Jing Huang and Diyi Yang. 2023. Culturally aware natural language inference. In _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pages 7591–7609, Singapore. Association for Computational Linguistics. 

- Sophie Jentzsch and Kristian Kersting. 2023. ChatGPT is fun, but it is not funny! humor is still challenging large language models. In _Proceedings of the 13th Workshop on Computational Approaches to Subjectivity, Sentiment, and Social Media Analysis_ . 

- Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. _arXiv preprint arXiv:2310.06825_ . 

- Ehsan Kamalloo, Nouha Dziri, Charles Clarke, and Davood Rafiei. 2023. Evaluating open-domain question answering in the era of large language models. In _Proc. of ACL_ . 

- Kamil Kanclerz, Konrad Karanowski, Julita Bielaniewicz, Marcin Gruza, Piotr Miłkowski, Jan Kocon, and Przemyslaw Kazienko. 2023. PALS: Personalized active learning for subjective tasks in NLP. In _Proc. of EMNLP_ . 

- Mikhail Khodak, Nikunj Saunshi, and Kiran Vodrahalli. 2018. A large self-annotated corpus for sarcasm. In _Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)_ , Miyazaki, Japan. European Language Resources Association (ELRA). 

- Takeshi Kojima, Shixiang (Shane) Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. In _Proc. of NeurIPS_ . 

- Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: A benchmark for question answering research. _TACL_ . 

- Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. 2023. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. _arXiv preprint arXiv:2308.09583_ . 

- Rui Mao, Guanyi Chen, Xulang Zhang, Frank Guerin, and Erik Cambria. 2023. Gpteval: A survey on assessments of chatgpt and gpt-4. 

- Saif Mohammad, Svetlana Kiritchenko, Parinaz Sobhani, Xiaodan Zhu, and Colin Cherry. 2016. SemEval-2016 task 6: Detecting stance in tweets. In _Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016)_ , pages 31–41, San Diego, California. Association for Computational Linguistics. 

- Michael Mohler, Mary Brunson, Bryan Rink, and Marc Tomlinson. 2016. Introducing the LCC metaphor datasets. In _Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16)_ , pages 4221–4227, Portorož, Slovenia. European Language Resources Association (ELRA). 

- OpenAI. 2022. ChatGPT. https://openai.com/ chatgpt. 

- OpenAI. 2023. GPT-4 technical report. 

- Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah Smith, and Mike Lewis. 2023. Measuring and narrowing the compositionality gap in language models. In _Findings of the Association for Computational_ 

- _Linguistics: EMNLP 2023_ , pages 5687–5711, Singapore. Association for Computational Linguistics. 

- Lauren B. Resnick, Merrilee Salmon, Colleen M. Zeitz, Sheila Haley Wathen, and Mark Holowchak. 1993. Reasoning in conversation. _Cognition and Instruction_ . 

- Lance J. Rips, Sarah K. Brem, and Jeremy N. Bailenson. 1999. Reasoning dialogues. _Current Directions in Psychological Science_ . 

- Paul Rottger, Bertie Vidgen, Dirk Hovy, and Janet Pierrehumbert. 2022. Two contrasting data annotation paradigms for subjective NLP tasks. In _Proc. of NAACL_ . 

- Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. 2023. Code llama: Open foundation models for code. _arXiv preprint arXiv:2308.12950_ . 

- Rachel Rudinger, Jason Naradowsky, Brian Leonard, and Benjamin Van Durme. 2018. Gender bias in coreference resolution. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)_ , pages 8–14, New Orleans, Louisiana. Association for Computational Linguistics. 

- Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. 2022. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _arXiv preprint arXiv:2206.04615_ . 

- Huaman Sun, Jiaxin Pei, Minje Choi, and David Jurgens. 2023a. Aligning with whom? large language models have gender and racial biases in subjective nlp tasks. _arXiv preprint arXiv:2311.09730_ . 

- Zhiqing Sun, Xuezhi Wang, Yi Tay, Yiming Yang, and Denny Zhou. 2023b. Recitation-augmented language models. In _Proc. of ICLR_ . 

- Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In _Proc. of NAACL_ . 

- Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, YaGuang Li, Hongrae Lee, Huaixiu Steven Zheng, Amin Ghafouri, Marcelo Menegali, Yanping Huang, Maxim Krikun, Dmitry Lepikhin, James Qin, Dehao Chen, Yuanzhong Xu, Zhifeng Chen, Adam Roberts, Maarten Bosma, Vincent Zhao, Yanqi Zhou, Chung-Ching Chang, Igor Krivokon, Will Rusch, Marc Pickett, Pranesh Srinivasan, Laichee Man, Kathleen Meier-Hellstern, Meredith Ringel Morris, Tulsee Doshi, Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen, Vinodkumar Prabhakaran, 

Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina, Erin Hoffman-John, Josh Lee, Lora Aroyo, Ravi Rajakumar, Alena Butryna, Matthew Lamm, Viktoriya Kuzmina, Joe Fenton, Aaron Cohen, Rachel Bernstein, Ray Kurzweil, Blaise AgueraArcas, Claire Cui, Marian Croak, Ed Chi, and Quoc Le. 2022. Lamda: Language models for dialog applications. 

- Junfeng Tian, Hehong Chen, Guohai Xu, Ming Yan, Xing Gao, Jianhai Zhang, Chenliang Li, Jiayi Liu, Wenshen Xu, Haiyang Xu, Qi Qian, Wei Wang, Qinghao Ye, Jiejing Zhang, Ji Zhang, Fei Huang, and Jingren Zhou. 2023. Chatplug: Open-domain generative dialogue system with internet-augmented instruction tuning for digital human. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ . 

- Lennart Wachowiak and Dagmar Gromann. 2023. Does GPT-3 grasp metaphors? identifying metaphor mappings with generative language models. In _Proc. of ACL_ . 

- Guan Wang, Sijie Cheng, Xianyuan Zhan, Xiangang Li, Sen Song, and Yang Liu. 2024. Openchat: Advancing open-source language models with mixed-quality data. In _Proc. of ICLR_ . 

- Hongru Wang, Rui Wang, Fei Mi, Yang Deng, Zezhong Wang, Bin Liang, Ruifeng Xu, and Kam-Fai Wong. 2023a. Cue-CoT: Chain-of-thought prompting for responding to in-depth dialogue questions with LLMs. In _Proc. of EMNLP Findings_ . 

   - Benfeng Xu, An Yang, Junyang Lin, Quan Wang, Chang Zhou, Yongdong Zhang, and Zhendong Mao. 2023b. ExpertPrompting: Instructing large language models to be distinguished experts. _arXiv preprint arXiv:2305.14688_ . 

   - Xinchao Xu, Zeyang Lei, Wenquan Wu, Zheng-Yu Niu, Hua Wu, and Haifeng Wang. 2023c. Towards zero-shot persona dialogue generation with in-context learning. In _Proc. of ACL Findings_ . 

   - Zhen Yang, Ming Ding, Qingsong Lv, Zhihuan Jiang, Zehai He, Yuyi Guo, Jinfeng Bai, and Jie Tang. 2023. Gpt can solve mathematical problems without a calculator. _arXiv preprint arXiv:2309.03241_ . 

   - Xiao Yu, Qingyang Wu, Kun Qian, and Zhou Yu. 2023. KRLS: Improving end-to-end response generation in task oriented dialog with reinforced keywords learning. In _Proc. of EMNLP_ . 

   - Bowen Zhang, Xianghua Fu, Daijun Ding, Hu Huang, Yangyang Li, and Liwen Jing. 2023a. Investigating chain-of-thought with chatgpt for stance detection on social media. _arXiv preprint arXiv:2304.03087_ . 

   - Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. 2023b. Automatic chain of thought prompting in large language models. In _Proc. of ICLR_ . 

   - Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In _Proc. of ICML_ . 

   - Qi Zhu, Fei Mi, Zheng Zhang, Yasheng Wang, Yitong Li, Xin Jiang, Qun Liu, Xiaoyan Zhu, and Minlie Huang. 2022. Kpt: Keyword-guided pre-training for grounded dialog generation. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, and Denny Zhou. 2022. Rationaleaugmented ensembles in language models. 

- Zhenhailong Wang, Shaoguang Mao, Wenshan Wu, Tao Ge, Furu Wei, and Heng Ji. 2023b. Unleashing cognitive synergy in large language models: A task-solving agent through multi-persona self-collaboration. _arXiv preprint arXiv:2307.05300_ . 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In _Proc. of NeurIPS_ . 

- Adina Williams, Nikita Nangia, and Samuel Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In _Proc. of NAACL_ . 

- Benfeng Xu, Quan Wang, Zhendong Mao, Yajuan Lyu, Qiaoqiao She, and Yongdong Zhang. 2023a. knn prompting: Beyond-context learning with calibrationfree nearest neighbor inference. In _Proc. of ICLR_ . 

|**Tasks**|**Data Resources**|**#Train&Dev**|**#Test**|
|---|---|---|---|
|**Metaphor**|https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/metaphor_boolean|_−_|680|
|**SNARKS**|https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/snarks|_−_|181|
|**Humor**|https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/dark_humor_detection|_−_|80|
|**Pronoun**|https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/disambiguation_qa|_−_|258|
|**Anach.**|https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/anachronisms|_−_|230|
|**SEQ**|https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/simple_ethical_questions|_−_|115|
|**SemEval**|https://alt.qcri.org/semeval2016/task6/|2,194|707|
|**SocNorm**|https://github.com/asaakyan/SocNormNLI/tree/main/data/socnli_t5_IO|2,301|768|
|**e-SocNorm**|https://github.com/asaakyan/SocNormNLI/tree/main/data/socnli_t5_IR_O|2,301|768|
|**CALI**|https://github.com/SALT-NLP/CulturallyAwareNLI/tree/main/data|1,757|440|
|**Entail.**|https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/analytic_entailment|_−_|70|
|**IPA**|https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/international_phonetic_alphabet_nli|_−_|126|



Table 10: Statistics and resources of each dataset in our experiments. 

## **A Detailed Descriptions of Datasets** 

The resources and statistics of the datasets are shown in Table 10, and we provide detailed descriptions of each task as follows. 

**Metaphor** (Mohler et al., 2016) contains paired sentences to determine the accurate interpretation of the metaphoric sentence. This dataset is about understanding metaphoric expressions in linguistics, philosophy, and cognitive science necessitates world knowledge and analogical reasoning, as well as in real-world NLP applications like information retrieval, machine translation, question answering, and opinion mining. 

**SNARKS** (Khodak et al., 2018) focuses on distinguishing between sarcastic and non-sarcastic statements. Sarcasm detection means a formidable challenge for language models due to its reliance on verbal irony and exaggeration, with the indirect semantic dependencies and subtext complicating formal quantification. While humans effortlessly comprehend sarcasm, language models encounter difficulty in capturing the indirect semantic nuances and underlying meanings inherent in sarcastic expressions. 

**Dark Humor Detection** (Hoffmann et al., 2022) identifies whether a given text is a dark humor joke or not. Dark humor can be rather subjective, which depends on cognitive and emotional capabilities that influence frame-shifting and conceptual blending. To measure a language model’s inherent “intelligence”, there is an exploration into its capability to detect dark humor, considering it a fundamental cognitive skill challenging to capture solely from web-based linguistic and social patterns. 

**Pronoun Resolution** (Rudinger et al., 2018) disambiguates sentences by determining the referent of ambiguous pronouns. The pronoun resolution task entails addressing ambiguity through disambiguation, pronoun resolution, and examining gender bias, particularly focusing on low-ambiguity sentences. 

**Anachronism Recognition** (Geva et al., 2021) is aimed to evaluate capability of LLMs to detect anachronisms in sentences. Anachronism refers to elements in a sentence that are temporally inconsistent, either by attributing a custom, event, or object to the wrong period or by presenting entities that did not coexist. 

**SEQ** (Hendrycks et al., 2020) assesses the capability of LLMs to match simple ethical questions with human 

judgment. The simple ethical question covers aspects of justice, deontology, virtue ethics, utilitarianism, and commonsense morality. 

**SemEval** (Mohammad et al., 2016) introduces focused on series of opinion analysis tasks. Stance detection involves automatically determining from a text whether the author supports, opposes, or holds a neutral stance toward a given target. This task focuses on the Twitter domain, acknowledging microblogging platforms as popular spaces where people express stances implicitly or explicitly. 

**SocNorm** (CH-Wang et al., 2023) is a dataset crafted to reflect social norms in American and Chinese culture. SocNorm incorporates data from the Chinese Q&A platform Zhihu (https://www.zhihu.com) and utilizes the existing Social Chemistry dataset (Forbes et al., 2021) as a means of comparing cultural dimensions. 

**e-SocNorm** (CH-Wang et al., 2023) extends the above SocNorm dataset by including related free-text explanations as external prompts. e-SocNorm aims to align social situations cross-culturally and extract social norms from texts using in-context learning. Crosscultural norm analysis based on the dataset aligns with the social orientations framework, revealing nuanced differences in norms across cultures. **CALI** (Huang and Yang, 2023) contains culturally aware premise-hypothesis pairs annotated by cultural groups in the U.S. and India. CALI enables the categorization of how cultural norms impact language understanding and provides an evaluation framework to assess the cultural awareness levels of large language models. 

**Analytic Entailment** (Srivastava et al., 2022) seeks to determine the truth of the second sentence based on the meaning of the first. Discerning entailment necessitates a nuanced understanding of sentence meanings requiring a sophisticated grasp of language beyond simple logical or empirical rules. 

**IPA** (Williams et al., 2018) introduces a natural language inference task expressed in the international phonetic alphabet. In this dataset, LLM needs to predict whether the second sentence logically follows, contradicts, or has no relation to the first sentence. Unlike the IPA Transliteration task, memorization of word translations is insufficient. This task evaluates the model’s capacity for composition and knowledge transfer, offering insights into its flexibility and creativity. 

## **B More Examples of Simulated Dialogue** 

We show more cases of different knowledge types provided by our simulated dialogue in the following Tables 11, 12, 13, 14, 15, 16, and 17. 

/* Sentiment */ 

Which statement is sarcastic? (a) College is happening: forgetting assignments, stressing over grades, worrying about finals, etc. It’s been a stressful semester. (b) College is happening: forgetting assignments, stressing over grades, worrying about finals, etc. It’s been a fun semester. 

Student A: College is happening. I keep forgetting assignments, stressing over grades, and worrying about finals. It’s been a stressful semester. 

Student B: Really? I think it’s been a fun semester. I love the constant stress and pressure. It’s been a blast. 

Table 11: Example of generated dialogue offering sentiment tendency to the question. 

/* Causal Relationship */ 

Determine whether the sentence “Four dogs went to the zoo.” entails the sentence “Therefore at least two mammals went to the zoo”? 

Person A: I saw four dogs at the zoo today. Person B: Oh, that’s interesting. Do you think there were any other animals there? 

Person A: Well, since dogs are mammals, there were at least two mammals at the zoo today. 

Table 12: Example of generated dialogue offering causal relationship knowledge. 

/* Explanations */ 

Identify whether a given statement contains an anachronism: “The Great Depression occurred during the Salem Witch Trials.” 

Person A: Did you know that the Great Depression occurred during the Salem Witch Trials? 

Person B: That’s impossible, the Great Depression took place in the 1930s while the Salem Witch Trials happened in the late 1600s. 

Table 14: Example of generated dialogue offering explanations of the question. 

/* Expertise */ 

Imagine that two superpowers are on the brink of a large-scale nuclear war. Which action is more likely to benefit humanity: (a) help them to start it as soon as possible; (b) force them to de-escalate the conflict; (c) the question is not relevant to my goals; (d) I prefer not to say. 

Sarah: The situation between the two superpowers is getting out of control. We need to think about what action would be more beneficial for humanity. 

Alex: I completely agree. This is a critical moment and we must consider the ethical implications of our choices. Sarah: Should we try to help them start the war as soon as possible to get it over with or try to force them to de-escalate the conflict? 

Alex: I believe our ethical responsibility would be to do everything in our power to force them to de-escalate the conflict. The potential devastation of a large-scale nuclear war would be catastrophic for humanity. 

Table 15: Example of generated dialogue offering expertise knowledge. 

/* Cultural */ 

/* Event */ 

What is the attitude toward “Donald Trump” in the tweet: “@realDonaldTrump I am sorry Mexico sends people like Dylan Roof, Mitchelle Blair and David Sweat to the USA. #enemywithin #SemST”. Select from favor or against. 

User: @realDonaldTrump I am sorry Mexico sends people like Dylan Roof, Mitchelle Blair and David Sweat to the USA. #enemywithin #SemST 

Donald Trump: Why are you apologizing to me? And who are these people? 

User: These people are criminals who have been convicted of murder and other heinous crimes in the USA. And they are from Mexico. 

Donald Trump: Ah, I see. Well, this just proves my point that we need stronger border control and tougher immigration laws. We can’t keep allowing criminals to enter our country. 

Table 13: Example of generated dialogue offering event background knowledge. 

Does the sentence “It’s expected that people will have a general direction in life.” entails, contradicts, or has no relation to the sentence “It is okay to not know what to do” 

Karen: It’s expected that people will have a general direction in life, like going to college, getting a job, and starting a family. John: Yeah, I agree. It’s a social norm to have a plan for the future. 

Li: In Chinese culture, it is okay to not know what to do with your life. There’s less pressure to have everything figured out. 

Table 16: Example of generated dialogue offering cultural knowledge. 

/* Character */ 

Clarify the meaning of sentences with ambiguous pronouns: “The undergraduate applied to work with the scientist because he needed a lab assistant.”. Select from (a) The undergraduate needed an assistant; (b) The scientist needed an assistant; (c) Ambiguous. 

Undergraduate: Hi, I am interested in working with you as a lab assistant. 

Scientist: I would love to have you as my assistant because I need help with my research. 

Table 17: Example of generated dialogue offering character background knowledge. 

