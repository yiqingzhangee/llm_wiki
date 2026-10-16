## **A Study of Situational Reasoning for Traffic Understanding** 

Jiarui Zhang jrzhang@isi.edu Information Science Institute, University of Southern California Los Angeles, California, USA 

## Aravinda Kollaa 

kollaa@isi.edu Information Science Institute, University of Southern California Los Angeles, California, USA 

Filip Ilievski ilievski@isi.edu Information Science Institute, University of Southern California Los Angeles, California, USA 

## Jonathan Francis 

jon.francis@us.bosch.com Human-Machine Collaboration, Bosch Center for Artificial Intelligence Pittsburgh, Pennsylvania, USA 

## Kaixin Ma 

kaixinm@cs.cmu.edu Language Technologies Institute, Carnegie Mellon University Pittsburgh, Pennsylvania, USA 

Alessandro Oltramari alessandro.oltramari@us.bosch.com Human-Machine Collaboration, Bosch Center for Artificial Intelligence Pittsburgh, Pennsylvania, USA 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding_images/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding.pdf-0001-11.png)


**----- Start of picture text -----**<br>
����<br>����������� ����� ������<br>����������������������������<br>�������������<br>������������<br>���<br>��������� �����<br>����� �����<br>�������������������������������<br>������������������������������<br>������������� ���<br>���������������� �����<br>������<br>��������������������������<br>������� ���������� ���������������<br>���������<br>�����<br>������������������<br>**----- End of picture text -----**<br>


**Figure 1: Overview of our study framework, which evaluates four knowledge-enhanced language methods, adapted with different knowledge sources on three traffic domain datasets, in a zero-shot manner.** 

## **ABSTRACT** 

Intelligent Traffic Monitoring (ITMo) technologies hold the potential for improving road safety/security and for enabling smart city infrastructure. Understanding traffic situations requires a complex fusion of perceptual information with domain-specific and causal commonsense knowledge. Whereas prior work has provided benchmarks and methods for traffic monitoring, it remains unclear whether models can effectively align these information sources and reason in _novel_ scenarios. To address this assessment gap, we devise three novel text-based tasks for situational reasoning in the traffic domain: i) BDD-QA, which evaluates the ability of Language Models (LMs) to perform situational decision-making, ii) TV-QA, which assesses LMs’ abilities to reason about complex event causality, and iii) HDT-QA, which evaluates the ability of models to solve human driving exams. We adopt four knowledge-enhanced methods that 

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _KDD ’23, August 6–10, 2023, Long Beach, CA, USA_ 

© 2023 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0103-0/23/08...$15.00 https://doi.org/10.1145/3580305.3599246 

have shown generalization capability across language reasoning tasks in prior work, based on natural language inference, commonsense knowledge-graph self-supervision, multi-QA joint training, and dense retrieval of domain information. We associate each method with a relevant knowledge source, including knowledge graphs, relevant benchmarks, and driving manuals. In extensive experiments, we benchmark various knowledge-aware methods against the three datasets, under zero-shot evaluation; we provide in-depth analyses of model performance on data partitions and examine model predictions categorically, to yield useful insights on traffic understanding, given different background knowledge and reasoning strategies. We have made our code and data publicly available.[1] 

## **CCS CONCEPTS** 

• **Computing methodologies** → **Natural language processing** ; _Knowledge representation and reasoning_ ; _Learning paradigms_ . 

## **KEYWORDS** 

traffic understanding, zero-shot evaluation, language models, question answering 

> 1https://github.com/saccharomycetes/text-based-traffic-understanding 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Jiarui Zhang, et al. 

## **ACM Reference Format:** 

Jiarui Zhang, Filip Ilievski, Kaixin Ma, Aravinda Kollaa, Jonathan Francis, and Alessandro Oltramari. 2023. A Study of Situational Reasoning for Traffic Understanding. In _Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’23), August 6–10, 2023, Long Beach, CA, USA._ ACM, New York, NY, USA, 11 pages. https: //doi.org/10.1145/3580305.3599246 

## **1 INTRODUCTION** 

As humans, we are able to make sense of novel situations even with limited domain-specific knowledge. For machine reasoning, situational understanding has recently seen a resurgence of commonsense reasoning technologies, demonstrated in tasks such as question answering [31, 48] and story comprehension [8, 30]. However, the ability of models to generalize to novel situations in specific domains has been relatively understudied. The domain of Intelligent Traffic Monitoring (ITMo), which represents an essential instrument to improve road safety and security, is of particular interest as it supports the vision of developing smart city infrastructures of the future [7]. ITMo has accumulated a significant market value from car manufacturers [4], as well as significant government support, e.g., through a recent infrastructure bill [36]. 

Being able to understand novel traffic situations requires a complex association of perceptual information with assumed domainspecific and causal commonsense knowledge. For example, reasoning whether an accident could be prevented by slower movement or forecasting the consequence of different movement patterns of a vehicle depends on the specific situational circumstances (e.g., the velocity of the other vehicles, road conditions), domain reasoning (e.g., what is the maximum allowed speed on a freeway), and causal reasoning (e.g., what could happen if the car changes lanes immediately, or what can be expected behavior from the other traffic participants). While prior work has developed evaluation tasks for traffic monitoring, only a few of them are available to the public [21, 45, 46]. These tasks have inspired corresponding methods [7, 12, 34, 42], which usually focus on associating perception signals, e.g., from video cameras, with answers to natural language questions. While providing an important step in the right direction, these tasks and methods do not provide mechanisms for studying models’ holistic reasoning abilities over novel situations in traffic. 

Given that language is a natural medium for human communication, it is intuitive to use the textual modality to develop and test robust traffic understanding models. A large body of work has focused on general commonsense reasoning [28, 31, 48], but the role of commonsense and causal reasoning has not been explored thoroughly in the traffic domain. Complementarily, ontologies for describing traffic participants [1] exist, but it is unclear whether connecting them to neural models can bring the desired generalizability in reasoning. To the best of our knowledge, understanding traffic situations presented in natural language has not been explored in depth so far. This brings up a natural question of whether large pretrained models can effectively solve realistic traffic understanding tasks dominant in natural language, such as decision-making in the context of a given situation, reasoning about the causes and effects of a given (real or hypothetical) situation, and answering questions that rely on domain expert knowledge. Moreover, as state-of-theart traffic monitoring methods are often designed as black boxes, 

prior work has not systematically investigated the role of various background knowledge types (e.g., causal commonsense knowledge and traffic domain knowledge) in reasoning over traffic situations. 

In this paper, we _study the ability of diverse knowledge-enhanced language models to reason over traffic situations_ . By doing so, we are able to reduce the complexity of traffic reasoning to a single modality (text), and measure the success of different method families and knowledge sources on representative traffic monitoring tasks. A schematic overview of our study framework is shown in Figure 1. 

We summarize the contributions in this paper as follows: (1) We devise **three novel text-based tasks for situational reasoning in the traffic domain** : _i) BDD-QA_ , which evaluates the ability of LMs to perform situational decision-making, _ii) TV-QA_ , which assesses LM’s ability to reason about complex event causality, and _iii) HDT-QA_ , which evaluates the ability of models to solve human driving tests. We divide each of these datasets into meaningful partitions to support fine-grained analysis. 

(2) We adopt **four knowledge-enhanced methods that have been shown to generalize well** across natural language reasoning tasks in prior work, based on natural language inference, commonsense knowledge graph (KG) self-supervision, multi-task QA joint training, and retrieval-augmented QA. We associate each of these methods with a relevant knowledge source, including knowledge graphs, relevant benchmarks, and driving manuals. 

(3) We perform **extensive experiments of different knowledgeaware methods against the three datasets in a zero-shot evaluation setup** . We perform an in-depth analysis of model performance on data partitions and look closer into the model predictions to provide useful insights into the traffic understanding ability of different background knowledge and reasoning methods. 

## **2 RELATED WORK** 

## **2.1 Traffic Understanding** 

CADP [40] is a spatiotemporally annotated dataset for accident forecasting using traffic camera views. The Berkeley Deep-Drive dataset (BDD) is a video dataset consisting of real driving videos containing abundant driving scenarios [45]. The follow-up work, BDD-X [21], provides the action description of the BDD video and their explanations. TrafficQA [46] consists of over 60K QA samples based on over 10k traffic scenes. The QA set of TrafficQA includes 6 different aspects of reasoning problems: basic understanding, attribution, introspection, counterfactual inference, event forecasting, and reverse reasoning. All QA pairs are based on visual inputs. The traffic benchmarks have inspired corresponding methods: by using knowledge to enhance traffic understanding, Wickramarachchi et al. [42] create and evaluate knowledge graph embeddings for autonomous driving. Chowdhury et al. [7] enhance the knowledge graph for scene entity prediction in autonomous driving. ITSKG [34] is a knowledge graph framework for extracting actionable information from raw sensor data in traffic. CoSI [12] is a knowledge graphbased approach for representing information sources relevant to traffic situations. Different from the dominant traffic monitoring focus on the perception modality, we explore a shift in paradigm, from monitoring to _understanding_ , which requires abundant domain knowledge and commonsense reasoning methods. However, perceptual annotations provide valuable information that can be 

A Study of Situational Reasoning for Traffic Understanding 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

reused for creating comprehension tasks. We utilize textual descriptions of actions and their explanations associated with the BDD-X dataset to construct our BDD-QA dataset for causal reasoning. Similarly, we leverage the TrafficQA data to create a text-based task by transcribing the videos into textual descriptions. 

## **2.2 Situational Reasoning in Natural Language** 

MultiNLI (MNLI) [43] is a large (433k) corpus for the natural language inference (NLI) task—with ten distinct genres of written and spoken English. MNLI has become a popular benchmark for assessing various language models, such as BERT [9], RoBERTa [26], BART [23], and DeBERTa [13]. Although they are widely used in general reasoning tasks, these methods have not been applied to the traffic domain, to the best of our knowledge. In our work, we use such models that are pre-trained on MNLI to evaluate their zero-shot performance on inference tasks in the traffic domain, based on the assumption that the models are generally capable of performing causal reasoning. ATOMIC [39] provides a large atlas of everyday commonsense knowledge, focusing on inferential knowledge of causes and effects (i.e., if-then-else conditions). CauseNet [14] provides a large-scale knowledge base of claimed causal relations between concepts which can be used for casual reasoning and question-answering tasks. Ma et al. [29] developed a method that utilizes different types of commonsense knowledge, extracted from a federated knowledge graph, to train language models via self-supervision: they found that such models yield high accuracy across several natural language tasks, under zeroshot evaluation [48]. Despite knowledge-driven models performing well on many general-domain commonsense reasoning tasks, the question remains how to customise these approaches for a specific domain. Therefore, we assess performance of modeling approaches on reasoning tasks in the traffic domain, to investigate the effect of commonsense knowledge on answering domain-specic questions. 

## **3 TASKS & DATASETS** 

In this section, we describe the three diverse datasets we construct for comprehensively evaluating models’ diverse situational reasoning strategies in the traffic domain—BDD-QA, TV-QA, and HDT-QA. We refer to situational reasoning as the task of providing the best judgment or decision for a given situation. Formally, we pose traffic situational reasoning as a multiple-choice question-answering task. Each task entry consists of a natural language question _𝑄_ and _𝑛_ candidate answers A = { _𝐴_ 1 _, ...,𝐴𝑛_ }. Given _𝑄_ and A, a model’s prediction task is to select the single best answer from the candidates. We show statistics and examples from our data in tables 1 and 2. 

## **3.1 BDD-QA** 

The first dataset we construct is BDD-QA, which tests models’ decision-making capabilities by asking for the best action in a given situation from a driver’s perspective. 

**Data collection.** We create BDD-QA from the Berkeley Deep-Drive Explanation Dataset [21] (BDD-X). The dataset consists of human annotations on various traffic videos covering diverse weather conditions and road types. In total, BDD-X provides over 26K sentence pairs where each pair consists of a description of an action in the video and its justification (reason). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding_images/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding.pdf-0003-10.png)


**----- Start of picture text -----**<br>
Class Partition in BDD-QA Class Partition in TV-QA<br>accelerate Introspective<br>merge Reverse Reasoning<br>drive Counterfactual<br>slow Event Forecasting<br>stop 17.3%<br>turn 12.6% 25.8%<br>reverse 14.6% 32.5%<br>13.8%<br>14.6%<br>15.4% 11.8% 9.0% 32.7%<br>**----- End of picture text -----**<br>


**Figure 2: BDD-QA and TV-QA data partitions.** 

**Task preparation.** The {action, justification} pairs naturally function as questions and answers. To convert them into multiplechoice questions, a key challenge is an effective sampling of distractors that are both fair (i.e., incorrect answers) and informative (i.e., sufficiently relevant to support a meaningful task). Similar to prior work [29], we observe that random sampling alone would result in a low-quality dataset because the distractors might also be plausible answers. Upon both the automatic analysis of the data by sentence transformers [38] and k-means clustering, as well as manual analysis of 100 data samples, we noted that the actions covered in this data are relatively limited and they can usually be associated with the sentence verbs. Inspired by this observation, we extract the verbs using Spacy [15] and count the frequency of the verbs in the whole set. We pick the verbs with a frequency higher than 20 and manually filter the incorrectly recognized verbs. We manually distill 7 classes from the remaining set of verbs to represent the car’s actions, namely: accelerate, move, slow, stop, merge, turn, and reverse. We apply these actions to classify the questions based on their verbs. We filter all of the repetitive sentence pairs and remove the actions with multiple verbs to make sure the actions can be correctly classified. 

After the verbs are classified into actions, we sample three distractors from different action classes for each question, which leads to 14,039 multiple-choice questions in total. However, a rule-based sampling strategy cannot ensure data quality, and semantically repetitive questions and multiple reasonable answers may still exist. To obtain high-quality test data, we randomly sample 50 examples from every class. We then conduct a thorough manual refinement process to replace possibly unfair distractors with admissible ones, hence only one answer is plausible. We also remove semantically repetitive questions. The resulting 255 multiple-choice questions are used for evaluation. The partitions of the test dataset are presented in Figure 2, showing a relatively balanced distribution among the seven action classes. 

## **3.2 TV-QA** 

The second dataset we constructed is TV-QA, which evaluates models’ capabilities to perform complex causal analyses of real and hypothetical traffic events from an observer’s perspective. **Data collection.** We build TV-QA from the recently proposed Traffic-QA dataset [46]. The questions in Traffic-QA are based on corresponding videos, recording different types of traffic events (accidents, vehicle turning, pedestrian behaviors, etc.). The videos 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Jiarui Zhang, et al. 

**Table 1: Statistics of the datasets by train/test split.** 

|Dataset&Partitions|**BDD-QA**<br>**TV-QA**<br>**HDT-QA**|
|---|---|
|**Train**<br>**Test**|14,039<br>-<br>9,456<br>255<br>434<br>500|



are 7 seconds long on average, and each video is associated with one or multiple questions. Traffic-QA categorizes questions into 6 traffic-related reasoning types: basic understanding, attribution, event forecasting, reverse reasoning, counterfactual inference, and introspection. Among the reasoning types, the "basic understanding" category is the most populated and it involves asking questions about the basic facts in the video, such as "Is there a crossroad in the video?", which is more akin to a computer vision recognition task rather than our focus of study. We also observe similar cases in the "attribution" category. As a result, we select questions from the latter four reasoning types to be included in our dataset. Two of these, "Event Forecasting" and "Reverse Reasoning", are surface-level natural language reasoning tasks that involve predicting events based on descriptions of previous and following events. The remaining two tasks, asking "What could happen if..." (Counterfactual reasoning) and "What should have been done to..." (Introspection), require an understanding of the causes and effects of the entire sequence of the traffic events and their alternatives, as well as a high degree of experience to provide reasonable predictions. 

**Task preparation.** To gather detailed human descriptions of the videos, we established a crowd-sourcing task with Amazon Mechanical Turk to elicit descriptions from annotators. To maximize the level of detail in the descriptions, we provided annotators with three questions pertaining to the initial scene (1 sentence), the procedure (2-3 sentences), and the final scene (1 sentence) of each video. We selected 200 videos for this annotation and asked three annotators to provide descriptions for each video. After obtaining the descriptions, to ensure that they contain sufficient information to support question answering about the traffic event, we set another crowd-sourcing task in which we show the annotators the description, question, and candidate answers. This task included 783 questions corresponding to the original 200 videos, and 3 annotators answer each description-question-answers triple. After collecting the answers given by the annotators, we choose the questions that are correctly answered by the majority of the annotators (at least 2 out of 3) using at least one description of each video as our valid QA dataset. This ensures that the descriptions given by annotators are sufficient for answering the questions; this results in 434 questions and 111 videos in total. Finally, we manually revise the descriptions of each valid description-question-answer triple and use the validated data as our test set. The partition of each reasoning task in the test dataset is presented in Figure 2. We note that all reasoning types have a similar participation in TV-QA, except counterfactual inference, which has a much smaller set of examples. 

## **3.3 HDT-QA** 

The third dataset we construct is HDT-QA, which focuses on traffic domain knowledge, including driving policies and driver decisions. 

**Table 2: Examples of BDD-QA, TV-QA, and HDT-QA. (*) denotes the correct answer.** 

**BDD-QA** _Q:_ The car in front of the car is slow, but the traffic is also heavy in other lanes, what will the car do next? _A1:_ The car speeds up and turns to the right; _A2:_ The car moves back to the right side of the road; _A3:_ The car slows down(*); _A4:_ The car backs up slowly **TV-QA** _Description:_ The POV car is quickly going down a highway. The POV car approaches an intersection. There is a red sedan in the opposing lane waiting to turn and cross the intersection. The red sedan quickly makes a left turn. when the POV car enters the intersection. The POV car veers to the right. The red sedan hits the side of the POV car. _Q_ : Could the accident be prevented if the involved vehicles change lane or turn properly? _A1:_ Yes(*); _A2:_ No, that was not the main cause of the accident **HDT-QA** _Q:_ If you find yourself in a skid: _A1:_ Brake lightly; _A2:_ Brake abruptly; _A3:_ Stay off the brakes(*) 

**Data collection.** Human driving tests provide a natural testbed for evaluating domain knowledge about traffic. Sample tests publicly available on the Web can be used to help people assess their knowledge, but also to evaluate the ability of machines to do the same. We scrape the **H** uman **D** riving **T** est **Q** uestion **A** nswering (HDT-QA) dataset from a website that contains 51 collections of sample driving tests pertaining to the 50 USA states and Washington D.C. [3]. For each state collection, there are three subjects of driving tests and manuals: _Motorcycle_ , _Car_ and _Commercial Driver_ —resulting in 153 state-subject combinations. Each test consists of set of multiple-choice questions, where the number of answer candidates ranges between 3 and 4. 

**Task preparation.** As the driving tests are already comprised of multiple-choice questions, we can use these questions to evaluate domain knowledge comprehension by AI models. To make the test data fair for text-based methods, we only keep questions that can fully be answered based on the textual content, i.e., we exclude all questions with images (e.g., traffic signs). We also filter out the questions where one of the candidate answers is _all of the above_ , as answering such questions is not the focus of our work. We amalgamate the test datasets from all states and subjects, eliminating duplicated questions and consolidating them based on the number of candidate answers. We split the resulting data into 9,456 train questions and 500 test questions. 

## **4 METHOD** 

While fine-tuned language models achieved state-of-the-art performance on many language-based reasoning tasks [9, 37], prior work has shown that this could lead to over-fitting [31]. Anticipating similar over-fitting to QA sets in the traffic domain, we select representative existing methods that utilize background knowledge 

A Study of Situational Reasoning for Traffic Understanding 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

to perform reasoning on unseen QA tasks in different ways. We use four methods that enrich language models with background knowledge relevant for the traffic domain, based on: Natural Language Inference (NLI), Knowledge Graph self-supervision, Question Answering, and Retrieval-augmented QA. 

## **4.1 Natural Language Inference** 

The task of NLI requires the model to select the most plausible candidate from a set of hypotheses given a premise. A multiplechoice QA task can be seen as a union of several NLI tasks, where each corresponds to performing inference over one of the choices. To reduce any potential training/test format mismatch, we keep our QA data as declarative sentence pairs. The two sentences are given to an NLI model as the input. Given a sentence pair { _𝑆_ 1 _,𝑆_ 2}, the model is asked to respectively score entailment, neutral, and contradiction, for each sentence, as { _𝑃𝑒, 𝑃𝑛, 𝑃𝑐_ }. Then, the model chooses the candidate with the highest margin score of _𝑃𝑒_ − _𝑃𝑐_ as the true answer. Formally, the final prediction of the model will be: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding_images/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding.pdf-0005-05.png)


where _𝑖_ denotes the _𝑖𝑡ℎ_ candidate sentence pair for the model to choose. For our experiments, we adopt an off-the-shelf NLI model and apply it directly to our three datasets. In particular, we select the RoBERTa model fine-tuned on the MultiNLI [43] dataset, which is one of the largest training corpora for NLI, consisting of 433k data entries covering ten genres. 

## **4.2 Knowledge Graph Self-Supervision** 

Self-supervised training with structured knowledge has shown promising results in recent works [10, 29, 48]. These methods automatically construct synthetic data from commonsense knowledge graphs and then train the model on the synthetic data. Due to the broad coverage of knowledge in diverse knowledge graphs, [17], models trained on synthetic data show strong generalization abilities to unseen commonsense reasoning tasks. Thus, we investigate whether these models can generalize well to the traffic domain too. We adopt the RoBERTa-large model and T5-large model released by Zhang et al. [48], as well as their corresponding scoring methods. 

Given a question and several candidates, the model chooses the question-candidate pair that best entails the knowledge extracted from the knowledge graph: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding_images/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding.pdf-0005-10.png)


where _𝑅_ ( _𝑠_ | _𝐾𝐺_ ) is the reasonableness score of a sentence based on the knowledge that the language model extracted from the knowledge graph. 

The RoBERTa model uses the Mask Language model Loss to give a score for the combination of question and candidate pairs. For the T5 model, the prefix "reasoning" is added at the beginning of each question-candidate pair. We define the loss of T5 models (sequence to sequence loss) generating a string _𝑆𝑡_ as _𝐿𝑆𝑡_ then the reasonableness score is _𝑆_ = _𝐿_ ”1” − _𝐿_ ”2” because the model is trained to generate a "1" token for reasonable input and a "2" for unreasonable input. Both of these models are pretrained on a synthetic dataset with 1 million short question-answer pairs generated from the Commonsense Knowledge Graph [17]. 

## **4.3 Question Answering** 

Recent work shows that training a multi-task QA model on a diverse range of QA datasets achieves strong generalization ability on unseen datasets [20]. Thus, we adopted such a multi-task QA model on our traffic domain benchmarks. Specifically, we used the UnifiedQA-v2 models released by Khashabi et al. [19], which is trained on 20 QA benchmarks crossing different domains and question formats. Since Unified-QA converts all tasks into a sequence-to-sequence format, we also follow this strategy in our evaluations. Given a question _𝑄_ , and a set of answer candidates { _𝐴_ 1 _,𝐴_ 2 _, ..._ }, we concatenate the question with all answer candidates as the input. Then the model is expected to generate the correct answer as output. We calculate the difference[2] between the T5 output and each candidate, and we choose the candidate with the least difference as the choice of the model. 

## **4.4 Retrieval-Augmented QA** 

The aforementioned methods all make the assumption that the knowledge required to predict the answer is already learned by the models, i.e., encoded in their parameters as implicit knowledge. However, encoding the vast amount of knowledge in the model parameters may be unrealistic for many knowledge-intensive tasks [24]. As an alternative, augmenting the QA model with relevant information retrieved on the fly shows promising results in many recent studies [27, 33]. Thus, we also adopted a retrieval-augmented approach in our evaluation. In particular, we adopted the Dense Passage Retrieval (DPR) [18] to find relevant domain knowledge. DPR is a neural retriever model trained on open-domain QA datasets that require Wikipedia knowledge [22]. The DPR model is shown to outperform traditional lexical-based retrieval methods like BM25 on those popular benchmarks. 

In our case, we use DPR to encode our traffic domain corpus (described next) and retrieve relevant documents for each question. Given a corpus _𝐶_ ∈{ _𝑝_ 1 _, 𝑝_ 2 _, ...𝑝𝑛_ } and a question Q, where _𝑝𝑖_ are individual passages. We use the dot-product score to extract the most relevant paragraph: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding_images/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding.pdf-0005-18.png)


where _𝑝_ ˆ _𝑖_ and _𝑄_[ˆ] are the passage and the question respectively as encoded by DPR. To perform retrieval-augmented QA, we feed the top-1 retrieved passage along with the question and its answer candidates to the Unified-QA model to get the final answer. Here, we simply concatenate the retrieved to the front of the question and then run inference in the same way as described in subsection 4.3. **Retrieval corpus.** We scrape driving manuals from all 50 USA states from a comprehensive website [3]. Analogous to the human driving tests (cf. subsection 3.3), for each state, there are three subjects of driving manuals ( _Motorcycle_ , _Car_ , and _Commercial Driver_ ), resulting in 150 state-subject combinations. Within our study, we treat the comprehensive collection of driving manuals as a domain corpus [3]. After processing the manual documents, we divide them into sentences by simply using the punctuation _" . ? ! "_ as sentence boundaries. As information extraction from PDF is imperfect, we use a T5-based grammar correction [2] as a data-cleaning tool to 

> 2We calculate the difference between the output sequences by finding the longest contiguous matching sub-sequences. 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Jiarui Zhang, et al. 

**Table 3: Evaluation results on our three benchmarks, with best results in bold and second best in underline font. For BDD human evaluation, we randomly selected 100 questions from the test set, and we ask 3 humans to give their answers. We recognize the questions that are correctly answered by the majority voting (at least 2 out of 3) as correctly answered questions. For TV-QA, the human accuracy is 100% since the dataset is comprised of the questions that can be correctly answered by the majority voting. While we did not evaluate human performance on HDT-QA, we expect that this data is of high quality as its questions are used to evaluate human command of traffic rules and policies in the real world.** 

|**Method**|**Benchmark**<br>**BDD-QA**<br>**TV-QA**<br>**HDT-QA**|
|---|---|
|**Random baseline**<br>**RoBERTa-large (unsupervised)**|0.250<br>0.373<br>0.268<br>0.309<br>0.412<br>0.334|
|**NLI-RoBERTa-large**|0.471<br>0.479<br>0.324|
|**KG-RoBERTa-large**<br>**KG-T5-large**|0.443<br>0.465<br>0.352<br>0.458<br>**0.506**<br>0.344|
|**QA-T5-large**|0.471<br>0.468<br>0.414|
|**Retrieval-T5-large**|**0.537**<br>0.444<br>**0.440**|
|**RoBERTa-large (supervised)**<br>**Human**|0.894<br>-<br>0.822<br>0.950<br>1.000<br>-|



improve the spelling and grammar of the corpus. Finally, we combine every 10 consecutive sentences as paragraphs that are fed to the QA model.[3] In total, our corpus consists of 42 _,_ 979 paragraphs, where each paragraph has an average token count of 147 _._ 5. 

## **5 EXPERIMENTAL SETUP** 

**Baselines.** To indicate the upper bound of models’ performance on our dataset, we include the results for a supervised method based on fine-tuning of the Roberta-large model. We only do this in the BDD-QA dataset and HDT-QA dataset due to the limited size and the lack of training data of the TV-QA dataset. For lower bound performance, we include results for a random baseline and an unsupervised vanilla Roberta-large model. 

**Evaluation.** Our evaluation primarily involves testing all methods on all datasets in a zero-shot manner. We also perform a transfer learning experiment, where we take a supervised model trained on one dataset and evaluate it directly on another dataset. We use accuracy as an evaluation metric, calculated as the ratio of correctly answered questions to the total number of questions. 

**Implementation.** Our implementation is based on python 3.7.10, PyTorch 1.9.0, and transformers 4.11.3. For the transfer learning experiments, we use a learning rate of 1 _𝑒_[−][5] , batch size of 32, weight decay 0.01, training epochs of 10, adam-epsilon of 1 _𝑒_[−][6] , _𝛽_ 1 = 0 _._ 9 _, 𝛽_ 2 = 0 _._ 98, warm-up proportion of 0.05, and margin of 1.0. **Computation.** For CPU computation, we use Intel(R) Xeon(R) Gold 5217 CPU @ 3.00GHz (32 CPUs, 8 cores per socket, 263GB RAM). For GPU, we use an NVIDIA RTX A5000. 

3We also attempted to segment the sentences into paragraphs based on the similarity between neighboring sentences, but this strategy yielded consistently worse results. 

## **6 RESULTS** 

## **6.1 Can Zero-Shot Models Reason over Traffic?** 

Table 3 shows the overall results for four different types of models (NLI-based, KG-based, QA-based, and retrieval-augmented) on our three benchmarks (BDD-QA, TV-QA, HDT-QA), against a random baseline, supervised, unsupervised model, and human evaluation. Overall, we note that the best performance of our models is in the range between 44 and 54 % accuracy, which is much better than random and vanilla language model (Roberta-unsupervised) performance, underscoring the important role that knowledge injection of language models plays. However, it remains significantly behind the supervised models (whose accuracy is between 82 and 90%) and human performance (accuracy of 95% or higher). The performance is particularly low on the HDT-QA dataset, which pertains to the intricate details of driving policies and regulations required to reason over this task. Among the methods, we note that the most effective approach is to augment QA methods with information retrieved on the fly, which achieves the best performance on two out of three datasets. Namely, on both BDD-QA and HDT-QA, combining DPR with Unified-QA improves the performance of merely using Unified-QA by 2.6 - 6.6%, which indicates that domain knowledge is essential for decision-making and human driving tests. On the HDT-QA dataset, the NLI and KG models perform comparably to the vanilla LM baseline, implying that general and commonsense knowledge is much less relevant for solving human driving tests. Meanwhile, the best performance on the TV-QA task is achieved by the KG-augmented model. The importance of commonsense knowledge for reasoning over complex and hypothetical situations is intuitive. Curiously, on TV-QA, retrieving domain information is counter-productive and decreases the performance by 2.4%, possibly highlighting a distraction brought by the domain knowledge for more abstract reasoning tasks. 

## **6.2 What is the Accuracy per Traffic Action?** 

Table 4 shows the fine-grained model results on BDD-QA for different actions. Although the zero-shot models achieve similar overall performance, there are large differences in their accuracy between action classes. The KG-based models show balanced performance across the 7 classes, while the NLI- and QA-based models exhibit notable differences in performance between action classes. The overall best-performing model, DPR + Unified-QA, is better than the other models for 4/7 classes. However, the NLI-based Roberta-large model performs best on the "Move", “Slow”, and "Stop" action classes, but poorly in the "Merge" and "Turn" classes. A possible reason for this lies in the nature of the actions like "Move" and "Slow", which are about following the actions of other cars, and can be easier for the NLI-based model to capture due to its strong semantic coherence capabilities. Learning from in-domain data also has the lowest impact on the classes Merge and Turn, which may signify that these classes are either more diverse or have less data compared to the others. However, the KG-based Roberta-large model, which performs worse than the NLI-based model overall, demonstrates better performance on "Merge" and "Turn" actions. We believe that this is inherently linked to the fact that these actions are less probable to be found in NLI corpora, whereas KGs are able to provide richer semantic signals for the representation of merging and turning, 

A Study of Situational Reasoning for Traffic Understanding 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

**Table 4: Results by action class on the BDD-QA dataset.** 

|**Methods**|**Action Class**<br>Accelerate<br>Move<br>Slow<br>Stop<br>Merge<br>Turn<br>Reverse<br>Average|**Action Class**<br>Accelerate<br>Move<br>Slow<br>Stop<br>Merge<br>Turn<br>Reverse<br>Average|
|---|---|---|
|**Random baseline**<br>**RoBERTa-large (unsupervised)**|0.250<br>0.250<br>0.250<br>0.250<br>0.250<br>0.250<br>0.250<br>0.405<br>0.312<br>0.285<br>0.051<br>0.477<br>0.500<br>0.162|0.250<br>0.309|
|**NLI-RoBERTa-large**|0.459<br>**0.781**<br>**0.657**<br>**0.487**<br>0.295<br>0.129<br>0.513|0.471|
|**KG-RoBERTa-large**<br>**KG-T5-large**|0.324<br>0.375<br>0.428<br>0.307<br>0.522<br>0.580<br>**0.567**<br>0.351<br>0.406<br>0.457<br>0.512<br>0.477<br>0.419<br>**0.567**|0.443|
|||0.458|
|**QA-T5-large**|0.540<br>0.468<br>0.514<br>0.153<br>0.613<br>0.548<br>0.459|0.471|
|**Retrieval-T5-large**|**0.567**<br>0.750<br>0.400<br>0.205<br>**0.681**<br>**0.612**<br>**0.567**|**0.537**|



an observation consistent with our general findings earler in this section. Finally, we note that both QA method families (with and without information retrieval) perform poorly on the “Stop” class, which is also the most difficult for the baseline Roberta-large model. We expect that this indicates that situations that require a vehicle to stop are most difficult, however, further investigation is needed to understand the specific reasons for this observation. 

## **6.3 What is the Accuracy per Reasoning Type?** 

The model accuracy per reasoning task in TV-QA is shown in Table 5. Overall, the best-performing model varies per reasoning task. Among the four methods, the KG-augmented models perform the best in forecasting events and introspection. The NLI-based model emerges as the clearly best performer on the _Counterfactual_ task with a margin of 10% over the second-best model. This may be due to the fact that _Counterfactual_ questions ask about _What could happen if..._ , which have a similar pattern to the objectives of the NLI training. As expected (cf. subsection 3.2), we observe that zero-shot reasoning models bring relatively higher improvement over the random baseline on the _Event Forecasting_ and _Reverse Reasoning_ tasks. This can be attributed to the fact that these two tasks rely on information that is already given in the event description ( _what will happen next? what might happen before?_ ), while the other two tasks of Counterfactual and Introspective inference require reasoning over hypothetical events that rely on a deeper understanding of causal relationships and a large amount of background knowledge. Curiously, while we hypothesize that commonsense knowledge is crucial for complex reasoning, we observe that all of the models perform worse than the Vanilla-Roberta-large model on the _Introspective_ task portion, showing that all of the tested models are not capable of answering questions like _What should have been done to x_ . This indicates the need for alternative modeling architectures that can natively perform multi-hop reasoning. 

## **6.4 Can We Transfer Knowledge to Novel Tasks?** 

A key premise of our work is that heterogeneous knowledge about the domain and about event causality should be combined to perform robust reasoning about traffic situations. To investigate this hypothesis further, we use HDT-QA as a training source and assume a certain amount of traffic domain knowledge can be injected after a model training on HDT-QA. Given the multiple-choice question format of HDT-QA makes it easier for knowledge injection, we use the same training strategy as prior works did [29] on knowledge injection: given a multiple-choice QA dataset, the model uses a 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding_images/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding.pdf-0007-09.png)


**----- Start of picture text -----**<br>
1.0<br>Vanilla Roberta-large Roberta-large + CSKG<br>0.8 Roberta-large + HDT Roberta-large + CSKG + HDT<br>0.6<br>0.4<br>0.2<br>0.0<br>Accelerate Move Slow Stop Merge Turn Reverse Average<br>**----- End of picture text -----**<br>


**Figure 3: Granular accuracy of transfer-learning and knowledge fusion on BDD-QA for four model variants per class and on average: 1) vanilla Roberta-large, 2) Roberta-large trained on HDT, 3) Roberta-large trained on CSKG, and 4) Roberta-large trained on a fusion of HDT and CSKG dataset.** 

Masked language Model to give the score of each candidate, then use the Margin Loss Function to maximize the score of the correct answer. To assess the impact of traffic domain knowledge (HDT) on BDD-QA and TV-QA, we evaluated the HDT-QA trained model on both BDD-QA and TV-QA. We also include commonsense knowledge from the CSKG training dataset, and fuse two training data together. Four experiments are conducted as a result: 1) vanilla Roberta-large, 2) Roberta-large trained on HDT, 3) Roberta-large trained on CSKG, and 4) Roberta-large trained on a fusion of HDT and CSKG dataset. 

Figure 3 show the results of four models’ granular performance on the BDD-QA dataset. Injecting common knowledge and driving knowledge is both helpful for making correct traffic decisions, and merging the knowledge leads to better performance. Namely, the _𝑅𝑜𝑏𝑒𝑟𝑡𝑎_ + _𝐶𝑆𝐾𝐺_ + _𝐻𝐷𝑇_ model reaches the best performance on average, followed by _𝑅𝑜𝑏𝑒𝑟𝑡𝑎_ + _𝐶𝑆𝐾𝐺_ and _𝑅𝑜𝑏𝑒𝑟𝑡𝑎_ + _𝐻𝐷𝑇_ . The results show that combining commonsense and domain knowledge results in better performance compared to using only one or the other. The model trained with both knowledge sources indirectly transfers driving knowledge to the knowledge required for answering BDDQA tasks, while the model trained with only driving knowledge has difficulty answering BDD-QA questions. This is intuitive, e.g., commonsense knowledge can tell language models that cars will pass the crosswalk when driving and crosswalk will appear at the intersection, while traffic domain knowledge tells the models that cars should yield to pedestrians passing the crosswalk. When facing the question _what might be the reason that a car is waiting in the intersection when the traffic light is green_ , the answer _The car is_ 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Jiarui Zhang, et al. 

**Table 5: Results by reasoning type on TV-QA. Random accuracy varies due to different numbers of candidate answers.** 

|**Methods**|**Reasoning Task**<br>Event Forecasting<br>Reverse Reasoning<br>Counterfactual<br>Introspective<br>Average|**Reasoning Task**<br>Event Forecasting<br>Reverse Reasoning<br>Counterfactual<br>Introspective<br>Average|
|---|---|---|
|**Random baseline**<br>**Roberta-large (unsupervised)**|0.353<br>0.299<br>0.438<br>0.468<br>0.323<br>0.382<br>0.384<br>**0.571**|0.373<br>0.412|
|**NLI-Roberta-large**|0.535<br>0.482<br>**0.589**<br>0.366|0.479|
|**KG-Roberta-large**<br>**KG-T5-large**|0.457<br>0.446<br>0.358<br>0.535<br>**0.577**<br>0.460<br>0.384<br>0.517|0.465<br>**0.506**|
|**QA-T5-large**|0.485<br>**0.517**<br>0.410<br>0.401|0.468|
|**Retrieval-T5-large**|0.464<br>0.446<br>0.538<br>0.383|0.444|



_waiting for pedestrians_ will come out, if both knowledge sources are effectively injected into the language model. Meanwhile, we note that the impact of the two knowledge types is not always positive, which confirms our findings in subsection 6.2. 

## **6.5 Do Model Predictions Overlap?** 

The performance of a model heavily depends on its training data and architecture. To evaluate the predictions of different models, we closely examine the results of five models: _Roberta-large-NLI_ , _Roberta-large-KG_ , _T5-large-KG_ , _T5-large-QA_ , _T5-large-IR_ on the BDD-QA dataset. We measure the intersection and the union of their accurate predictions, to understand their agreement and the potentially complementary results. As shown in the left part of Figure 4, although the models attain similar accuracy, their predictions have limited overlap with each other, with only around 50% of their correct predictions agreeing. Even among the models _Roberta-large-KG_ and _T5-large-KG_ , which have the same knowledge source, only two-thirds of their correct predictions overlap. This finding is consistent with the results in 6.2 and 6.3 that the granular performance of different models is much more diverse than their overall performance. Similarly, we note that the union of the model predictions results in notably higher accuracy (Figure 4 - right). Specifically, combining _Roberta-large-NLI_ and _T5-large-IR_ results in an accuracy of 0 _._ 757, with a gap of only 0 _._ 137 compared to supervised learning methods. These results are much higher than the performance of each model individually and come close to the results obtained from supervised learning, thus motivating the need for the development of methods that combine model predictions and knowledge in a fine-grained manner. 

## **7 DISCUSSION** 

## **7.1 Summary of Findings** 

In this section, we revisit our research questions and give a brief summary of the findings. 

**Can knowledge-augmented models reason effectively over traffic situations?** Overall, the best knowledge-augmented models reach an accuracy that is much higher than random performance, but the gap between the supervised model’s and human’s accuracy still remain. 

**Can models predict various traffic actions with similar accuracy?** Although general knowledge-augmented models reach similar overall performance, they utilize different sources of background knowledge to make decisions from various perspectives. This is evident in their varied performance across different action 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding_images/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding.pdf-0008-12.png)


**----- Start of picture text -----**<br>
Accuracy of Overlap Accuracy of Union<br>1.0<br>Ro-NLI 0.471 0.216 0.243 0.224 0.251 0.471 0.698 0.686 0.718 0.757<br>0.8<br>Ro-KG 0.216 0.443 0.298 0.263 0.302 0.698 0.443 0.604 0.651 0.678<br>0.6<br>T5-KG 0.243 0.298 0.459 0.255 0.282 0.686 0.604 0.459 0.675 0.714<br>0.4<br>T5-QA 0.224 0.263 0.255 0.471 0.373 0.718 0.651 0.675 0.471 0.635<br>0.2<br>T5-Re 0.251 0.302 0.282 0.373 0.537 0.757 0.678 0.714 0.635 0.537<br>0.0<br>Ro-NLI Ro-KG T5-KG T5-QA T5-Re Ro-NLI Ro-KG T5-KG T5-QA T5-Re<br>**----- End of picture text -----**<br>


**Figure 4: A heat map representing the intersection (left) and union (right) of correct predictions on BDD-QA for five large models: NLI-Roberta, KG-Roberta, KG-T5, QA-T5, and Retrieval-T5. Each value in the heat map represents the joint accuracy of the intersection/union of two models.** 

classes, which can usually be traced back to the knowledge used to train these models. 

**Are some reasoning skills more difficult for the models than others?** Language models have demonstrated superior performance in solving surface-level reasoning tasks compared to deeper-level reasoning tasks in the traffic domain. In some complex reasoning tasks, even knowledge-driven models underperform compared to vanilla language models. 

**Can traffic knowledge be transferred between different reasoning tasks?** The fusion of commonsense knowledge sources and human driving knowledge in the training process has been shown to enhance the performance of language models, surpassing the results achieved from training on only one type of knowledge. 

**Do methods’ predictions agree with each other?** Granular performance of different models is much more diverse than their individual performance. Combining the predictions of these models can close the gap with supervised learning. 

## **7.2 Future Work** 

**Comprehensive access to traffic knowledge** Our experiments have shown the relation between different tasks and model knowledge, and have demonstrated that combining domain and commonsense knowledge is usually beneficial for downstream task performance. Moreover, we observed that the fine-grained model predictions are often complementary, and their ideal combination would significantly outperform either of the individual models. We anticipate that a robust AI model should be able to reason over a 

A Study of Situational Reasoning for Traffic Understanding 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

comprehensive set knowledge types simultaneously and associate questions with reasoning types, e.g., with methods like mixture-ofexperts [11]. To facilitate explicit guidance for models on different knowledge types, it is important to systematically delineate the relevant knowledge dimensions and reasoning types, analogous to prior such work in general-domain commonsense reasoning [16]. **From surface QA to hypothetical reasoning** Our experiments have shown the ability of language models to employ various knowledge to reason on QA tasks. However, pure language modeling mechanisms may not suffice for hypothetical reasoning tasks, such as introspection and counterfactual reasoning. For such tasks, it is important to enhance the model architecture with the ability to develop chains of reasoning that abstract over the provided information and consider the causality of possible outcomes. While it is an open question of how to best perform such reasoning, promising avenues to explore are employing chain-of-thought [41], selfrationalization techniques [32], and graph-based methods [47]. **Generalizing traffic QA to other modalities and domains** Our study in this paper involved a range of QA benchmarks designed to measure the possession of various reasoning and knowledge by representative language models on the traffic domain. The annotations that support the BDD-QA and TV-QA benchmarks are based on videos, which opens the possibility for tighter integration of visual and textual modalities, and for introducing new tasks. We envision using these annotations to evaluate semantic scene search [7] based on factual descriptions, e.g., _give me all scenes where three cars collide_ , or even hypothetical situations, e.g., _situations where the accident could have been prevented by more lanes on the freeway._ Similarly, the annotations could be used to evaluate captioning models like SwinBERT [25] for the traffic domain. Such benchmarks that represent more tasks and modalities will be crucial in advancing toward the development of practical, neuro-symbolic traffic models. Furthermore, expanding our research to additional realworld domains holds significant potential. For example, situated commonsense reasoning under real-world scenarios [6, 44] is still challenging for language models. More practically, in robotics, we can assess the proficiency of language models in interpreting complex instructions and predicting mechanical failures [5]. Similarly, other domains like healthcare, finance, and scientific research could benefit greatly from language models’ advancements in situational reasoning. 

## **8 CONCLUSIONS** 

This paper devised a study framework that investigates the ability of knowledge-enhanced language models to exhibit generalizable reasoning over traffic situations. The framework included three novel text-based benchmarks for multiple-choice question answering: BDD-QA, which evaluates driver-centric decision-making; TV-QA, which evaluates four complex reasoning tasks; and HDTQA, which measures the possession of domain knowledge. We adopted four popular methods based on natural language inference, knowledge graph self-supervision, question answering, and retrieval-augmented question answering. Our zero-shot experiments revealed that the models are able to reason over these situations to some extent, and the variance in their performance is largely tied to the correspondence between the model knowledge and the task properties. Retrieval-augmented QA models performed best on BDD-QA and HDT-QA, while commonsense knowledgebased models performed best on the more complex reasoning tasks. Further experiments showed that model and knowledge combinations can enhance the robustness of these models. Future work should investigate the comprehensive organization of traffic knowledge, the development of methods that can perform more complex reasoning, and the development of broader suites of evaluation tasks beyond question answering. 

**Investigating the capabilities of large language models** Our research has extensively examined the inference and comprehension abilities of accessible language models, i.e., models with less than 10 billion parameters, in a variety of traffic scenarios. However, the remarkable comprehension and generalization capabilities of larger language models - those with typically more than 10 billion parameters, like GPT-4 [35] — suggest an immense potential for situational reasoning within complex fields. While we do not expect that these models will perform robust situational reasoning out of the box, the prospect of harnessing their sophisticated reasoning abilities to produce intricate insights and forecasts in the dynamic domain of traffic presents an intriguing avenue for future research. 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 

Jiarui Zhang, et al. 

## **REFERENCES** 

[1] 2020. ASAM OpenXOntology. https://www.asam.net/project-detail/asamopenxontology/. 

- [2] 2021. Fine-Tune a Transformer Model for Grammar Correction. https://www. vennify.ai/fine-tune-grammar-correction/. 

- [3] 2022. Free DMV Permit Practice Tests. https://www.dmv-test-pro.com/. 

- [4] 2022. Intelligent Traffic Management System Market Worth $27.92 Billion By 2030. https://www.grandviewresearch.com/press-release/global-intelligenttraffic-management-system-market. 

- [5] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, et al. 2022. Do as i can, not as i say: Grounding language in robotic affordances. _arXiv preprint arXiv:2204.01691_ (2022). 

- [6] Mana Ashida and Saku Sugawara. 2022. Possible Stories: Evaluating Situated Commonsense Reasoning under Multiple Possible Scenarios. _arXiv preprint arXiv:2209.07760_ (2022). 

- [7] Sreyasi Nag Chowdhury, Ruwan Wickramarachchi, Mohamed H Gad-Elrab, Daria Stepanova, and Cory A Henson. 2021. Towards Leveraging Commonsense Knowledge for Autonomous Driving.. In _ISWC (Posters/Demos/Industry)_ . 

- [8] Yiming Cui, Wanxiang Che, Wei-Nan Zhang, Ting Liu, Shijin Wang, and Guoping Hu. 2020. Discriminative Sentence Modeling for Story Ending Prediction. _Proceedings of the AAAI Conference on Artificial Intelligence_ 34, 05 (Apr. 2020), 7602–7609. https://doi.org/10.1609/aaai.v34i05.6260 

- [9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_ (2018). 

- [10] Zi-Yi Dou and Nanyun Peng. 2022. Zero-shot Commonsense Question Answering with Cloze Translation and Consistency Optimization. _arXiv preprint arXiv:2201.00136_ (2022). 

- [11] Suchin Gururangan, Mike Lewis, Ari Holtzman, Noah A Smith, and Luke Zettlemoyer. 2021. Demix layers: Disentangling domains for modular language modeling. _arXiv preprint arXiv:2108.05036_ (2021). 

- [12] Lavdim Halilaj, Ishan Dindorkar, Jürgen Lüttin, and Susanne Rothermel. 2021. A knowledge graph-based approach for situation comprehension in driving scenarios. In _European Semantic Web Conference_ . Springer, 699–716. 

- [13] Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2020. DeBERTa: Decoding-enhanced BERT with Disentangled Attention. https://doi.org/10. 48550/ARXIV.2006.03654 

- [14] Stefan Heindorf, Yan Scholten, Henning Wachsmuth, Axel-Cyrille Ngonga Ngomo, and Martin Potthast. 2020. Causenet: Towards a causality graph extracted from the web. In _Proceedings of the 29th ACM International Conference on Information & Knowledge Management_ . 3023–3030. 

- [15] Matthew Honnibal and Ines Montani. 2017. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. (2017). To appear. 

- [16] Filip Ilievski, Alessandro Oltramari, Kaixin Ma, Bin Zhang, Deborah L McGuinness, and Pedro Szekely. 2021. Dimensions of commonsense knowledge. _Knowledge-Based Systems (KBS)_ (2021). 

- [17] Filip Ilievski, Pedro Szekely, and Bin Zhang. 2021. Cskg: The commonsense knowledge graph. In _European Semantic Web Conference_ . Springer, 680–696. 

- [18] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for OpenDomain Question Answering. https://doi.org/10.48550/ARXIV.2004.04906 

- [19] Daniel Khashabi, Yeganeh Kordi, and Hannaneh Hajishirzi. 2022. Unifiedqav2: Stronger generalization via broader cross-format training. _arXiv preprint arXiv:2202.12359_ (2022). 

- [20] Daniel Khashabi, Sewon Min, Tushar Khot, Ashish Sabharwal, Oyvind Tafjord, Peter Clark, and Hannaneh Hajishirzi. 2020. UnifiedQA: Crossing Format Boundaries With a Single QA System. arXiv:2005.00700 [cs.CL] 

- [21] Jinkyu Kim, Anna Rohrbach, Trevor Darrell, John Canny, and Zeynep Akata. 2018. Textual Explanations for Self-Driving Vehicles. _Proceedings of the European Conference on Computer Vision (ECCV)_ (2018). 

- [22] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: a Benchmark for Question Answering Research. _Transactions of the Association of Computational Linguistics_ (2019). 

- [23] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. 2019. BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. https://doi.org/10.48550/ARXIV.1910.13461 

- [24] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. arXiv:2005.11401 [cs.CL] 

- [25] Kevin Lin, Linjie Li, Chung-Ching Lin, Faisal Ahmed, Zhe Gan, Zicheng Liu, Yumao Lu, and Lijuan Wang. 2022. Swinbert: End-to-end transformers with sparse attention for video captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ . 17949–17958. 

- [26] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A Robustly Optimized BERT Pretraining Approach. https://doi.org/10.48550/ ARXIV.1907.11692 

- [27] Kaixin Ma, Hao Cheng, Xiaodong Liu, Eric Nyberg, and Jianfeng Gao. 2022. Open Domain Question Answering with A Unified Knowledge Interface. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ . Association for Computational Linguistics, Dublin, Ireland, 1605– 1620. https://doi.org/10.18653/v1/2022.acl-long.113 

- [28] Kaixin Ma, Jonathan Francis, Quanyang Lu, Eric Nyberg, and Alessandro Oltramari. 2019. Towards generalizable neuro-symbolic systems for commonsense question answering. _arXiv preprint arXiv:1910.14087_ (2019). 

- [29] Kaixin Ma, Filip Ilievski, Jonathan Francis, Yonatan Bisk, Eric Nyberg, and Alessandro Oltramari. 2021. Knowledge-driven Data Construction for Zeroshot Evaluation in Commonsense Question Answering. In _AAAI_ . 

- [30] Kaixin Ma, Filip Ilievski, Jonathan Francis, Eric Nyberg, and Alessandro Oltramari. 2022. Coalescing Global and Local Information for Procedural Text Understanding. In _Proceedings of the 29th International Conference on Computational Linguistics_ . 1534–1545. 

- [31] Kaixin Ma, Filip Ilievski, Jonathan Francis, Satoru Ozaki, Eric Nyberg, and Alessandro Oltramari. 2021. Exploring Strategies for Generalizable Commonsense Reasoning with Pre-trained Models. _EMNLP 2021_ (2021). 

- [32] Ana Marasović, Iz Beltagy, Doug Downey, and Matthew E Peters. 2021. Few-shot self-rationalization with natural language prompts. _arXiv preprint arXiv:2111.08284_ (2021). 

- [33] Tavish McDonald, Brian Tsan, Amar Saini, Juanita Ordonez, Luis Gutierrez, Phan Nguyen, Blake Mason, and Brenda Ng. 2022. Detect, Retrieve, Comprehend: A Flexible Framework for Zero-Shot Document-Level Question Answering. _arXiv preprint arXiv:2210.01959_ (2022). 

- [34] RoopTeja Muppalla, Sarasi Lalithsena, Tanvi Banerjee, and Amit Sheth. 2017. A knowledge graph framework for detecting traffic events using stationary cameras. In _Proceedings of the 2017 ACM on Web Science Conference_ . 431–436. 

- [35] OpenAI. 2023. GPT-4 Technical Report. _ArXiv_ abs/2303.08774 (2023). 

- [36] Ehsan Qasemi and Alessandro Oltramari. 2022. Intelligent Traffic Monitoring with Hybrid AI. _arXiv preprint arXiv:2209.00448_ (2022). 

- [37] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. _OpenAI blog_ 1, 8 (2019), 9. 

- [38] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. arXiv:1908.10084 [cs.CL] 

- [39] Maarten Sap, Ronan Le Bras, Emily Allaway, Chandra Bhagavatula, Nicholas Lourie, Hannah Rashkin, Brendan Roof, Noah A Smith, and Yejin Choi. 2019. Atomic: An atlas of machine commonsense for if-then reasoning. In _Proceedings of the AAAI conference on artificial intelligence_ , Vol. 33. 3027–3035. 

- [40] Ankit Parag Shah, Jean-Bapstite Lamare, Tuan Nguyen-Anh, and Alexander Hauptmann. 2018. CADP: A novel dataset for CCTV traffic camera based accident analysis. In _2018 15th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)_ . IEEE, 1–9. 

- [41] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. _arXiv preprint arXiv:2201.11903_ (2022). 

- [42] Ruwan Wickramarachchi, Cory Henson, and Amit Sheth. 2020. An evaluation of knowledge graph embeddings for autonomous driving data: Experience and practice. _arXiv preprint arXiv:2003.00344_ (2020). 

- [43] Adina Williams, Nikita Nangia, and Samuel R Bowman. 2017. A broad-coverage challenge corpus for sentence understanding through inference. _arXiv preprint arXiv:1704.05426_ (2017). 

- [44] Bo Wu, Shoubin Yu, Zhenfang Chen, Joshua B Tenenbaum, and Chuang Gan. 2021. Star: A benchmark for situated reasoning in real-world videos. In _Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)_ . 

- [45] Huazhe Xu, Yang Gao, Fisher Yu, and Trevor Darrell. 2017. End-to-end learning of driving models from large-scale video datasets. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ . 2174–2182. 

- [46] Li Xu, He Huang, and Jun Liu. 2021. Sutd-trafficqa: A question answering benchmark and an efficient network for video reasoning over traffic events. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ . 9878–9888. 

- [47] Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. 2021. QA-GNN: Reasoning with language models and knowledge graphs for question answering. _arXiv preprint arXiv:2104.06378_ (2021). 

- [48] Jiarui Zhang, Filip Ilievski, Kaixin Ma, Jonathan Francis, and Alessandro Oltramari. 2022. An Empirical Investigation of Commonsense Self-Supervision with Knowledge Graphs. _arXiv preprint arXiv:2205.10661_ (2022). 

A Study of Situational Reasoning for Traffic Understanding 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding_images/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding.pdf-0011-02.png)


**----- Start of picture text -----**<br>
Traffic ahead of the car is beginning to move, what does the car do next?<br>0.6<br>KG-based Roberta NLI-based Roberta<br>0.4<br>0.2<br>0.0<br>0.6 The car wants to move into another road, what does the car do next?<br>KG-based Roberta NLI-based Roberta<br>0.4<br>0.2<br>0.0<br>The car backs upThe car moves faster forwardThe car slows downThe car makes a left turn<br>The car reversesThe car reduces its speedThe car is advancing slowlyThe car is stopping behind a line of cars<br>**----- End of picture text -----**<br>


**Figure 6: Prediction distributions for KG- and NLI-based Roberta for two exemplar BDD-QA questions. Questions are shown on the top, and candidate answers are presented on the x-axis. The y-axis shows the probability scores of each answer outputted by the model through a softmax layer.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding_images/kdd_2024_A_Study_of_Situational_Reasoning_for_Traffic_Understanding.pdf-0011-04.png)


**----- Start of picture text -----**<br>
BDD-QA TV-QA<br>Accelerate Merge Event Forecasting<br>1.0 MoveSlow TurnReverse 0.70 Reverse ReasoningCounterfactual<br>Stop Average 0.65 Introspective<br>0.8 Average<br>0.60<br>0.6 0.55<br>0.50<br>0.4<br>0.45<br>0.2<br>0.40<br>T5-large T5-3B T5-11B T5-large T5-3B T5-11B<br>Unified-QA Models Unified-QA Models<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


**Figure 5: The result of 3 different sizes of Unified-QA model tested on BDD-QA and TV-QA datasets, the Average result is emphasized using dotted lines.** 

## **A WHAT IS THE EFFECT OF SCALING UP MODEL SIZES?** 

To understand the effect of scaling up the model size, we test 3 different sizes of Unified-QA models (T5-large, T5-3B, T5-11B) on 

our datasets. We focus our analysis on the benchmarks BDD-QA and TV-QA. 

**BDD-QA.** Figure 5 - left shows the relationship between model size and performance for various traffic actions. In general, an increase in model size leads to improved performance. However, the magnitude of improvement varies among action classes. For instance, the "Move" class exhibits the most substantial improvement, rising from 0 _._ 468 to 0 _._ 937 as the model size grows. Conversely, the "Merge" class exhibits a comparatively modest improvement of 0 _._ 068 points. It is worth mentioning that, despite having one-thirtieth the number of parameters, the Roberta-large NLI-based model achieves the same accuracy in the "Stop" class as the largest Unified-QA model, T5-11B. These results highlight the substantial influence of diverse knowledge sources on the model performance. 

**TV-QA.** In 5 - right, we observe that as the model size increases, the performance improvement for surface-level reasoning tasks ( _Event Forecasting_ and _Reverse Reasoning_ ) becomes increasingly evident. However, it’s noteworthy that for the _Introspective_ task, the model performance actually decreases with increased model size, which deviates from typical observations of other QA tasks. 

In conclusion, while we find that increasing the model size is typically beneficial, its impact varies across reasoning types and action classes. Moreover, the impact of using different sources of knowledge is often bigger than the impact of increasing the model size. 

## **B CASE STUDY ON THE ROLE OF KNOWLEDGE IN MODEL REASONING** 

In section 6.2, we have observed that the BDD-QA NLI-based models are more likely to choose the actions that follow other cars’, while KG-based models can capture more abundant relations due to the structure of the knowledge graph. To illustrate this finding, we use two examples of traffic domain decision-making questions and show two models’ prediction distributions. From the first plot of Figure 6, given the question _Traffic ahead of the car is beginning to move, what does the car do next?_ , the NLI-based model chooses the correct answer _The car is advancing slowly_ , while the knowledge model chooses the wrong answer _The car reduces its speed_ . As we discussed, the NLI-based model solves this question by capturing the semantic coherence between _move_ and _advance_ . For the second plot of 6, given the question _The car wants to move into another road, what does the car do next?_ , the NLI-model still chooses the answer based on semantic similarity without considering the fact that the car is moving to another road. Meanwhile, the KG-based model makes the correct choice by successfully connecting the car action _turn_ with the description _move into another road_ . 

