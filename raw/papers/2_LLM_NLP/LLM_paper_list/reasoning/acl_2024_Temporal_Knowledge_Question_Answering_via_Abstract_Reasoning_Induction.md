# **Temporal Knowledge Question Answering via Abstract Reasoning Induction** 

**Ziyang Chen**[1] _[∗]_ **Dongfang Li**[2] _[∗]_ **Xiang Zhao**[1] _[†]_ **Baotian Hu**[2] _[†]_ **Min Zhang**[2] 

1 Laboratory for Big Data and Decision, National University of Defense Technology, China 

2 Harbin Institute of Technology (Shenzhen), Shenzhen, China 

{chenziyangnudt,xiangzhao}@nudt.edu.cn 

{lidongfang,hubaotian,zhangmin2021}@hit.edu.cn 

## **Abstract** 

In this study, we address the challenge of enhancing temporal knowledge reasoning in Large Language Models (LLMs). LLMs often struggle with this task, leading to the generation of inaccurate or misleading responses. This issue mainly arises from their limited ability to handle evolving factual knowledge and complex temporal logic. To overcome these limitations, we propose Abstract Reasoning Induction (ARI) framework, which divides temporal reasoning into two distinct phases: Knowledgeagnostic and Knowledge-based. This framework offers factual knowledge support to LLMs while minimizing the incorporation of extraneous noisy data. Concurrently, informed by the principles of constructivism, ARI provides LLMs the capability to engage in proactive, self-directed learning from both correct and incorrect historical reasoning samples. By teaching LLMs to actively construct knowledge and methods, it can significantly boosting their temporal reasoning abilities. Our approach achieves remarkable improvements, with relative gains of 29.7% and 9.27% on two temporal QA datasets, underscoring its efficacy in advancing temporal reasoning in LLMs. The code can be found at https: //github.com/czy1999/ARI-QA. 

## **1 Introduction** 

_"Knowledge is not simply transmitted from teacher to student, but actively constructed in the mind of the learner."_ 

— Jean Piaget 

In practical scenarios, factual knowledge frequently undergoes evolution over time (Roddick and Spiliopoulou, 2002; Hoffart et al., 2011; Liang et al., 2023b, 2022). For instance, the host city of 

> *Equal Contribution. 

> †Corresponding authors. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction_images/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction.pdf-0001-15.png)


**----- Start of picture text -----**<br>
Abstract Reasoning<br>Induction<br>Abstract<br>In Context<br>Learning<br>Exemplar<br>LLM<br>Retrieval<br>Augment<br>ed<br>Specific Knowledge<br>Applicability level<br>Knowledge level<br>**----- End of picture text -----**<br>


Figure 1: LLMs, when integrated with various levels of information, exhibit varying scopes of applicability; the more abstract and refined the knowledge, the broader its potential application. 

the Winter Olympic Games in 2018 was South Korea, while in 2022 it was Beijing. Despite their proficiency in a range of linguistic tasks, LLMs often demonstrate limitations in the efficient processing and understanding of tasks that involve temporal information (Huang and Chang, 2023; Zhao et al., 2023a; Liang et al., 2023a). 

Specifically, when tasks require complex temporal reasoning, LLMs tend to mislead the process and provide an inaccurate result. For instance, _“Which country’s government leader visited China for the last time in 2015?”_ , to answer this question, we need to (1) get which countries visited China in 2015 ; (2) filter out the country with the earliest visiting date. In step 1, LLMs easily meet hallucinations due to the incomplete training data and the uncertainty of parameterised knowledge. In step 2, LLMs may lead to the error because of the inaccuracy of the time filtering. Within such temporal reasoning tasks, any misjudgment in the temporal knowledge or errors during the temporal reasoning will culminate in erroneous conclusions. The problem might stem from the temporal unawareness of LLMs, impeding their ability to track and interpret events over time, particularly in situations requiring subtle and time-sensitive understanding. 

Based on intuitive and empirical analysis, the 

cause accounting for the problem can be identified from two aspects: _lack of temporal knowledge_ and _lack of complex temporal reasoning_ . And the definition is given as follows: 

**LACK OF TEMPORAL KNOWLEDGE** . LLMs acquire vast knowledge through pre-training on extensive datasets. However, the fixed nature of their parameters after training solidifies their knowledge base, which leads to LLMs’ failure in understanding unseen and evolving knowledge. 

**LACK OF COMPLEX TEMPORAL REASONING** . Owing to the inherent nature of large models that generate outputs based on maximum probability, they are limited in directly conducting complex reasoning. Facing the interconnected multi-step temporal reasoning, LLMs might accumulate errors during the process of probabilistic generation. 

Despite the neglect of essences, current studies relatively approach above challenges. To augment the LLMs’ capacity for understanding unseen and evolving information, researchers incorporate external knowledge to supply contextually relevant information, known as Retrieval Augmented Generation (RAG) (Zhao et al., 2023b; Baek et al., 2023; Sun et al., 2023). Although these methods enhance the richness of LLMs’ responses, the retrieval accuracy and input length limitations might result in irrelevant noises and incomplete reasoning clues, degrading overall performance (Wang et al., 2023; Lu et al., 2023). Furthermore, although tailored examples serve as prompts to guide LLMs (Dong et al., 2023; Min et al., 2022), they are often inadequate for diverse practical tasks and require substantial efforts in time and human to acquire highquality examples. In conclusion, above approaches fail to provide necessary guidance for ongoing temporal reasoning processes and are susceptible to incorporating extraneous noise, as shown in Figure 2. 

To overcome the limitation, it is crucial to recognize that LLMs are inherently limited by reliance on passively absorbing training instances. Constructivism (Savery and Duffy, 1995; Kirschner et al., 2006), deeply embedded in philosophical and psychological schools of thought, contends that knowledge and learning emerge not from mere exposure to external information but through active construction. It asserts that learners synthesize new knowledge by building upon their existing understanding and experiences (Lake and Baroni, 2023). In this view, learning is an active and ongoing pro- 

cess wherein individuals continuously modify and refine their cognitive frameworks. 

Inspired by the principles of constructivism, we try to steer LLMs towards an active and selfinitiated learning approach, and propose an Abstract Reasoning Induction (ARI) framework. This will equip LLMs with the capacity for abstract synthesis and personalized knowledge application, enhancing relevance and utility in various contexts. 

In details, to handle the lack of temporal knowledge, we transfer the data generation to an active process, consisting of two stages: _Knowledgeagnostic_ and _Knowledge-based_ . In knowledgeagnostic part, LLMs only need to choose potential steps. It is only in the knowledge-based part that the corresponding action is executed on the specific knowledge base to obtain the answer. This procedure offers factual knowledge support to LLMs while minimizing the incorporation of extraneous noisy data. On the other hand, to complete LLMs’ complex temporal reasoning ability, ARI actively engages in proactive and self-directed learning from both correct and incorrect historical reasoning samples. This approach enables LLM to summarize and generalize methodologies (i.e. knowledgeagnostic step-by-step instructions) for different types of questions. When similar questions are encountered again, these abstract methods will guide the LLM to perform more efficient multi-step reasoning. By teaching LLMs to actively construct knowledge and methods, it can significantly boosting their temporal reasoning abilities without the need for further training. 

In summary, our contribution is three-fold: 

- Grounded in the principles of constructivism, we offer fresh perspectives for enhancing the reasoning capabilities and task adaptability of LLMs. 

- We present ARI, a novel temporal reasoning framework that divides the process into two phases: _Knowledge-agnostic_ and _Knowledgebased_ . ARI enables LLMs to learn and construct proactively from historical reasoning samples, fostering a perpetual refinement of LLMs’ reasoning abilities. 

- The experimental results demonstrate that, compared to the leading TKGQA models, our approach achieves relative improvements of 29.7% and 9.27% respectively on two temporal QA datasets. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction_images/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction.pdf-0003-00.png)


Figure 2: Three levels of information utilisation. _Information-Driven Response_ , which extracts pertinent knowledge to form the basis of answers; _Exemplar-Based Learning_ , offering cases of reasoning for the language model to assimilate and guide current inferences; and _Abstract Reasoning Induction_ , providing step-wise abstract methodological guidance to the present question, distinct from concrete knowledge, thereby steering the language model’s inference process. 

## **2 Related Work** 

## **2.2 LLM Reasoning with External Information** 

## **2.1 TKGQA Models** 

Traditional temporal knowledge graph question answering (TKGQA) methodologies fall into two categories. The first, exemplified by TEQUILA (Jia et al., 2018), deconstructs the initial question into sub-questions and temporal constraints, employing standard KGQA models for resolution, followed by a comparative analysis to select the most fitting answer. The second approach, such as CronKGQA (Saxena et al., 2021a), seeks to leverage TKG embeddings for semantic similarity assessments in answer determination, featuring a learnable reasoning process independent of handcrafted rules. Despite CronKGQA’s proficiency with simpler inquiries, its performance falters with complex questions necessitating specific temporal inference. TempoQR (Mavromatis et al., 2021) addresses this by incorporating temporal scope data and employing the EaE method (Févry et al., 2020) to enrich question representation semantically. 

However, traditional approaches rely on handcrafted rules or learnable representations, struggling with sophisticated temporal reasoning (Chen et al., 2022). In contrast, our model, leveraging the power of LLMs, excels in these challenging scenarios, showcasing superior adaptability and reasoning capabilities. 

Addressing hallucinations in generative models presents a compelling challenge, with one promising solution being the augmentation of LLMs with external knowledge (Mialon et al., 2023). Integration with an external knowledge base has become a prevalent strategy in question-answering and conversational tasks (Peng et al., 2023). There are mainly two approaches: explicit and implicit knowledge injection (Yang et al., 2023a). Explicit injection involves directly supplying LLMs with pertinent knowledge via prompts. For instance, KAPING (Baek et al., 2023) retrieves facts relevant to a query from a knowledge graph and appends these to the query as a prompt for the LLM, while CoK (Li et al., 2023a) first evaluates answer credibility and, if necessary, uses the LLM to decompose the question and generate various SPARQL queries to extract information from external knowledge bases. ChatKBQA (Luo et al., 2023) finetunes LLMs on KG structure to generate logical queries, which can be executed on KGs to obtain answers. Symbol-LLM (Xu et al., 2023) propose a dualstage fine-tuning framework to integrates symbolic knowledge into LLMs, enhancing their reasoning capabilities. ToG (Sun et al., 2023) treats the LLM as an agent to interactively explore related entities and relations on KGs and perform reasoning based 

on the retrieved knowledge. Implicit injection, on the other hand, subtly steers the LLM by incorporating knowledge semantic embeddings during reasoning or in the decoding process. KID (Liu et al., 2022) represents a novel decoding algorithm for generative LMs that dynamically infuses external knowledge at each step of LM decoding, and KPE (Zhao et al., 2023b) introduces a trainable parameter-sharing adapter to a parameter-freezing PLM for knowledge integration. Additionally, the incorporation of knowledge from other modalities such as visual information (Li et al., 2023c,b) has also become a method of knowledge introduction. 

While the integration of knowledge into LLMs can mitigate issues of hallucinations, it is not without challenges. Explicit knowledge injection often struggles to acquire high-quality, relevant information, and is constrained by the finite-length contexts. Implicit injection typically necessitates fine-tuning of parameters, which can be prohibitively costly. We address these limitations by dividing temporal knowledge reasoning into two distinct components: knowledge-related and knowledge-agnostic. This approach achieves a clear separation between knowledge and reasoning, thereby circumventing the aforementioned constraints. 

## **2.3 LLM Reasoning with Memories** 

Memory plays a pivotal role in human intelligence (Atkinson and Shiffrin, 1968). Given that LLMs inherently lack long-term memory and their short-term memory is constrained by the scope of their context window, numerous studies have embarked on the journey to equip LLMs with memory capabilities (Pan et al., 2023; Zhong et al., 2023). Instead of the conventional approach where accumulated conversations are retrieved directly, MemoChat (Lu et al., 2023) innovatively constructs and updates a structured, instant memo that categorizes past dialogues. Conversations are then fetched based on their specific topics and summaries. Reflexion (Shinn et al., 2023) exploits a working memory to store experiences for a dedicated task to improve the performance of the agent through several trials. However, the histories stored in working memory cannot benefit the episode for different task goals. MemPrompt (Madaan et al., 2022) designs a persistent memory to store human feedback to remind the chatbot of the conversational information and improve it continuously. RLEM (Zhang et al., 2023) adopts a persis- 

tent environment-grounded experience memory to store the experiences and assist in future decisionmaking even for different task goal. Thought Propagation (Yu et al., 2023) emphasizes the ability to explore and apply insights from analogous solutions. By delving into and utilizing solutions from problems related to the given issue, it improves performance and accuracy across various tasks. 

However, current memory-enhanced methods are limited to passively received historical information, overlooking the active construction of abstract knowledge based on previous experience. Starting from constructivism, we apply the proposed method to provide large models with an active and continuous learning process, offering knowledge that is abstract and generalized. 

## **3 Method** 

|**Algorithm 1**Abstract ReasoningInduction|**Algorithm 1**Abstract ReasoningInduction|**Algorithm 1**Abstract ReasoningInduction||
|---|---|---|---|
|**Require:** Temporal knowledge graph_K_, question||||
||_q_, historical memory_Hq_, abstract methodology|||
||instruction set_MC_|||
|**Ensure:** Answer to the question_q_||||
|1:|_MC ←LLM_(_Hq_)||(4)|
|2:|Initialize subject entity_eh_from_q_|||
|3:|Find 1-hop subgraph_Geh_ of_eh_in_K_||(1)|
|4:|Enumerate initial candidate actions _P_0 from|||
||_Geh_||(2)|
|5:|**while**_LLM_(_MC∗, P_<br>_′_<br>_ti_) =_answer_(_a_)**do**|||
|6:|Filter candidate actions_Pti_ to get_P_|_′_<br>_ti_|(3)|
|7:|_C∗_<br>_t ←_fndKmeansCluster(q)||(5)|
|8:|_a∗_<br>_i_ =_LLM_(_MC∗, q, P_<br>_′_<br>_ti_)||(6)|
|9:|Execute selected action_a∗_<br>_i_ and update||cur-|
||rent environment|||
|10:|Regenerate candidate actions for the next|||
||step||(2)|
|11:|**if**_ti ≥tmax_**then**|||
|12:|Break|||
|13:|**end if**|||
|14:|**end while**|||
|15:|Execute fnal action_a∗_<br>_i_ to abtain the answer|||
|16:|Add current process to_Hq_|||
|17:|**return** Answer derived from the reasoning|||
||process|||



## **3.1 Task Definition** 

Given a Temporal Knowledge Graph (TKG) _K_ and a natural language question _q_ , TKGQA aims to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction_images/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Question:  Which country's government leader visited China for the last time in 2015?<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction_images/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction.pdf-0005-01.png)


**----- Start of picture text -----**<br>
NER classification<br>A Knowledge-based B Knowledge-agnostic<br>Category<br>…<br>Russian Cat e go ry 2 ① A ? Category  𝟏 …<br>Japan China Cat e go r ① y 1 ①② A ②A? ? ? B? B Category  𝟐 …<br>… ① ② ②③A? ③Answer( )? Answer( ) ③ B?AnAswer( )B A A LLM …<br>Category  𝒏 …<br>③ Answer( ) A<br>enumerate Generated Abstract Method Guidance Historical Instance Classification<br>getEntity([Japan, visit, ?, 2015]) √ Category Incorrect Samples classification<br>filter guidance ① A ? Correct SamplesMethod Instruction<br>getBefore(Jack, visit, 2015-05-01)getFirst([China,…], visit, Japan) ×× act Agent feedback LLM instruct ②③ Answer( )? AB Inference ProcessLearning Process Memory<br>…….<br>Candidate actions Historical Memory Storage<br>2015-01-01visited 2015-01-01<br>2015<br>2015-01-01<br>2015-05-01<br>visited 2015-05-01 2015-05-01<br>**----- End of picture text -----**<br>


Figure 3: Model architecture of ARI. Our framework divides temporal reasoning into two distinct phases: _Knowledgeagnostic_ and _Knowledge-based_ . This division aims to reduce instances of hallucinations and improve LLM’s capacity for integrating abstract methodologies derived from historical experience. See the detailed instructions in Appendix A.4. 

extract an entity _s/o ∈E_ or a timestamp _τ ∈T_ that correctly answers the question _q_ . For instance, for the question _‘Which country’s government leader visited China for the last time in 2015?’_ Based on the event information contained in the TKG, we can get the answer to the question is the KG entity _Vietnam_ . 

## **3.2 ARI Framework** 

The constructivist perspective posits that knowledge does not merely encapsulate universal laws but must be contextually reconstructed for specific situations. This view emphasizes that understanding is a construct developed by the learner, uniquely shaped by their experiential background and dependent on their learning trajectory in a particular context (Kirschner et al., 2006; Savery and Duffy, 1995). In line with this philosophy, we introduce ARI framework. We overview the framework in Figure 3. Diverging from previous research that directly feeds knowledge into LLMs , we divide temporal knowledge reasoning into two parts: _knowledge-agnostic_ and _knowledge-based_ . 

The knowledge-based module extracts relevant TKG subgraphs based on the given question, generating all feasible fine-grained actions through traversal (cf. § 3.3). This module encompasses all operations and interactions with knowledge, freeing the LLM from the vast, noisy specific information to focus on reasoning. This design approach allows us to build intricate knowledge queries by combining fine-grained atomic operations, while ef- 

fortlessly adapting to various knowledge bases (see the detailed operations in Appendix A.3). In the knowledge-agnostic module, the LLM performs high-level strategic decisions and candidate action selection. We have innovated mechanisms that allow LLMs to internalize lessons from past decisions, forming generalized abstract methodologies (solid line process in Figure 3). This foundation empowers the LLM to proactively develop abstract methodological guidelines for various question types, enhancing its ability to reason on new questions efficiently (cf. §3.4). 

For the inference process, ARI first categorizes the question and selects the most suitable abstract methodological guidance. According to these guidelines, the LLM interacts multiple turns with the knowledge-agnostic module to reason and gradually solve the problem, as shown in the dashed line process in Figure 3. We also present a specific reasoning example in Figure 5 of Appendix A.4. 

## **3.3 Knowledge-based Interaction** 

In the knowledge-based interaction part, we frame complex temporal knowledge reasoning challenges as multi-step inferentce tasks (Gu and Su, 2022; Gu et al., 2023). At the beginning of each step, we employ an filtering mechanism to engage with the TKG and the current question. This interaction produces a set of feasible candidate actions for each step. The LLM then selects the most suitable action from these candidates. Following this selection, the model interacts with the TKG, updating the initial 

state for the next step in a recursive process. 

**Candidate Action Enumeration.** Specifically, given a complex temporal question _q_ and a TKG _K_ := ( _E, R, T , F_ ), where _E, R, T_ denote entities, relations, and timestamps, respectively. Starting from the subject entity _eh_ of _q_ , we first find the 1-hop subgraph of _eh_ in _K_ . Let _Neh_ be the set of nodes in the 1-hop (undirected) neighborhood of _eh_ in the TKG, and _Reh_ be the corresponding edge. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction_images/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction.pdf-0006-02.png)


where _Geh_ is the corresponding 1-hop subgraph of _eh_ . For each edge _r ∈ Geh_ , our agent will strictly follow finely-grained atomic operations in Appendix A.3, traversing and replacing the relations and entities present in the current _Geh_ to construct the set of candidate actions _P_ 0, where the subscript 0 denotes the candidate actions for the initial step, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction_images/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction.pdf-0006-04.png)


**Candidate Action Filtration.** However, due to 

the continual occurrence and updating of temporal events, even the scale of a 1-hop subgraph can be vast. This results in an excessively large set of generated candidate actions, which can significantly impede the judgment of LLMs (cf. § section 4.2). Consequently, we propose a filtration process for candidate actions, retaining only those that are correct, feasible, and semantically relevant. 

Specifically, for each action _a_ within set _P_ 0, we execute the corresponding function on the TKG. If the function returns a non-empty value, the action is considered correct and feasible; otherwise, it is discarded. Among all remaining actions, we retain the top- _K_ actions based on the calculation of their semantic similarity to the question _q_ , 

_′ P_ 0[=] _[ {][a][|]_[exec][(] _[a]_[)] _[ ̸]_[=] _[ ∅∧][a][ ∈]_[Top-] _[K]_[(] _[P]_[0] _[, q]_[)] _[}][.]_[(3)] 

Based on the LLM’s decision, the agent executes the corresponding action, thereby updating the current environment. Subsequently, it regenerates the next set of candidates _P_ 1 based on the newly identified subject entities, repeating this process until a termination command is received. 

## **3.4 Knowledge-agnostic Reasoning** 

The knowledge-agnostic module enables LLMs to distill and apply abstract methodologies from historical reasoning examples, enabling adaptation to 

diverse questions. This approach fosters a general methodology, applicable across various domains and to a wide range of knowledge-independent inquiries, enhancing LLMs’ versatility. 

**Historical Memory Storage and Learning .** In the LLM’s reasoning process, we meticulously document the current state at each step _ti_ , encompassing the current temporal question _q_ , the set of candidate actions _Pt_ , and the LLM’s decision _at_ . The aggregate of all stepwise states for a given question forms the historical decision set _Hq_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction_images/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction.pdf-0006-14.png)


where _T_ is the set of all steps in the process. 

Temporal reasoning is often multi-step and complex, yet the types of reasoning involved tend to be consistent, with similar questions requiring similar inference steps. Therefore, once the LLM conducts reasoning and accumulates a series of historical inferential steps, we employ unsupervised clustering K-means (MacQueen, 1967) to categorize these historical steps into distinct clusters _CH_ . 

Consequently, after the LLM engages in reasoning and compiles historical reasoning steps, these are subjected to unsupervised clustering to form distinct clusters. Each cluster encapsulates a mix of both accurate and erroneous reasoning processes. Subsequently, we enable the LLM to actively learn from these specific historical instances within each cluster and distill abstract methodologies that are independent of domain-specific knowledge. 

**LLM Decision with Abstract Reasoning.** When addressing new inference challenges, we initiate the process by identifying the historical reasoning cluster most closely aligned with the new question. We then extract its abstract methodologies to guide the LLM in its reasoning for the current question. 

Specifically, for a given question _q_ , we calculate the similarity score _S_ ( _Ci, q_ ) with each historical reasoning cluster _Ci_ . We then retrieve the abstract method instruction _MC∗_ from the cluster that yields the maximum _S_ . Let _{C_ 1 _, C_ 2 _, . . . , Cn}_ be the set of historical reasoning clusters. For a given question _q_ , we calculate the similarity score _S_ ( _Ci, q_ ) for each cluster _Ci_ , and select the cluster _C[∗]_ with the highest similarity score to the query _q_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction_images/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction.pdf-0006-20.png)


The abstract method instruction _MC∗_ is then selected from the cluster _C[∗]_ such that: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction_images/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction.pdf-0007-01.png)


where _MC∗_ is the abstract method instruction of _C[∗]_ , and _a[∗] i_[is the final output of LLM. The reason-] ing sequence concludes when the LLM outputs a termination action or when the length of reasoning steps exceeds the predetermined maximum threshold. 

## **4 Experiment** 

## **4.1 Implementation Details** 

We use gpt-3.5-turbo-0613 as our LLM (More experiments and analyses of other LLMs can be found in Section A.7). We configure the LLM to access and investigate a corpus of 200 historical reasoning samples, with the maximum length of reasoning path set to 5, and the number of historical path categories fixed at 10. Due to the vast size of the test set, which comprises more than 50,000 question-answer pairs, we employ a stratified sampling approach for evaluation, extracting a subset of 200 questions from the test set for each iteration. In our evaluation, we compare several baseline methods, including the traditional TKGQA models and LLM-based models (see Appendix A.1 and A.2 for more details). 

## **4.2 Overall Results** 

Table 1 presents the comparative results of ARI against other baselines on MULTITQ. 

**LLM only Performance.** ChatGPT’s performance on two datasets reveal a significant shortcoming in its application to temporal knowledge reasoning, even when all the knowledge required for the questions is within the scope of its training data prior to 2021. This deficiency is particularly pronounced when compared to traditional TKGQA methods, suggesting that the parameterised knowledge acquired by LLMs is not seamlessly transferred to temporal reasoning tasks. A stark contrast in performance is observed between the MULTITQ and CRONQUESTIONS datasets, the latter benefiting from ChatGPT’s extensive training on WikiData(Vrandecic and Krötzsch, 2014). This discrepancy points to a significant challenge: MULTITQ’s reliance on the ICEWS(Boschee et al., 2015), which encompasses more niche and frequent events, poses a difficulty for LLMs due to 

their limited exposure to such data during training. This contrast elucidates the inherent limitations of LLMs in processing temporal knowledge. 

**Reasoning with External Knowledge.** Incorporating additional knowledge graph data into LLMs, as seen with KG-RAG, considerably enhances their performance in knowledge-intensive QA tasks. KG-RAG outperforms ChatGPT by 81% and 96% across two datasets, respectively. Nevertheless, it still falls short of leading TKGQA models. This gap can be attributed to two main factors. First, the vast and complex nature of temporal information presents a challenge. A single question may involve thousands of related events, which cannot be accurately incorporated through prompts alone, leading to insufficient background information for reasoning. Second, the retrieved external knowledge often contains redundant or irrelevant information, which can further mislead the model’s inference process. 

**Reasoning with Exemplar Guidance.** The CoT KB[approach,][despite][providing][step-by-step][rea-] soning guidance and knowledge-based interaction, does not reach the efficacy levels of ARI. This discrepancy stems from the exemplar-based learning method’s dependence on specific instances, which can introduce extraneous knowledge and detract LLMs from focusing on reasoning methodologies, leading to inaccurate conclusions. Additionally, its static examples fail to provide the customized guidance necessary for diverse reasoning questions. Similarly, our investigation also extends to the augmented knowledge-agnostic module in ReAct, designed to mitigate the generation of infeasible actions. Despite this enhancement, a performance gap persists when compared to ARI, underscoring similar limitations identified in the CoT KB approach. Despite ReAct KB’s interaction with the environment, it still lacks customized abstract guidance for the current reasoning question. 

ARI significantly outperforms current state-ofthe-art TKGQA models, achieving a relative improvement of 29.7% on the MULTITQ dataset and a 9.27% increase in performance on the CRONQUESTIONS dataset. These substantial gains can be attributed to the knowledge adaptability and the abstract methodology instruction mechanism, which empower LLMs to make advanced decisions. By leveraging abstract methodologies, LLMs can select optimal temporal reasoning steps without engaging with the specifics of the underlying knowledge. 

|**Model**||**MULTITQ**|**MULTITQ**|**MULTITQ**|**MULTITQ**||**CRONQUESTIONS**|**CRONQUESTIONS**|**CRONQUESTIONS**|**CRONQUESTIONS**|
|---|---|---|---|---|---|---|---|---|---|---|
||**Overall**|**Question Type**||**Answer Type**||**Overall**|**Question Type**||**Answer Type**||
|||**Simple**|**Complex**|**Entity**|**Time**||**Simple**|**Complex**|**Entity**|**Time**|
|BERT<br>ALBERT<br>EmbedKGQA<br>CronKGQA<br>MultiQA<br>ChatGPT<br>KG-RAG<br>CoT KB<br>ReAct KB|0.083<br>0.108<br>0.206<br>0.279<br>0.293<br>0.102<br>0.185<br>0.240<br>0.310|0.092<br>0.061<br>0.116<br>0.086<br>0.235<br>0.134<br>0.134<br>0.134<br>0.347<br>0.159<br>0.147<br>0.077<br>0.200<br>0.160<br>0.440<br>0.120<br>0.635<br>0.136||0.101<br>0.040<br>0.139<br>0.032<br>0.290<br>0.001<br>0.328<br>0.156<br>0.349<br>0.157<br>0.137<br>0.002<br>0.230<br>0.07<br>0.220<br>0.320<br>0.313<br>0.300||0.243<br>0.248<br>0.288<br>0.647<br>-<br>0.249<br>0.490<br>0.640<br>0.685|0.249<br>0.239<br>0.277<br>0.179<br>0.255<br>0.235<br>0.279<br>0.177<br>0.290<br>0.286<br>0.411<br>0.057<br>0.987<br>0.392<br>0.699<br>0.549<br>-<br>-<br>-<br>-<br>0.250<br>0.247<br>0.246<br>0.253<br>0.460<br>0.518<br>0.470<br>0.520<br>0.690<br>0.610<br>0.620<br>0.660<br>0.835<br>0.525<br>0.650<br>0.755||||
|**ARI**|**0.380**_∗∗_|**0.680**_∗∗_<br>**0.210**_∗∗_||**0.394**_∗∗_<br>**0.344**_∗∗_||**0.707**_∗∗_|0.860<br>**0.570**_∗∗_<br>**0.660**_∗_<br>**0.800**_∗_||||



Table 1: Performance of baselines and our methods on the MULTITQ and CRONQUESTIONS. _[∗]_ ( _p ≤_ 0 _._ 05) and _∗∗_ ( _p ≤_ 0 _._ 005) indicate paired t-test of ARI versus the best baseline. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction_images/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction.pdf-0008-02.png)


**----- Start of picture text -----**<br>
w ARI<br>3.33 w/o ARI<br>3.76<br>2.42<br>2.67<br>1.0 1.5 2.0 2.5 3.0 3.5 4.0<br>Average steps<br>MultiTQ<br>CronQuestions<br>**----- End of picture text -----**<br>


Figure 4: Comparison of average reasoning steps of ARI on MULTITQ. 

|**Model**<br>**ARI**<br>w/o Abstract Guidance<br>w/o History Cluster<br>w/o Action Filter<br>w/o Incorrect Examples|**Accuracy (%)**<br>MULTITQ<br>CRONQUESTIONS|
|---|---|
||38.0<br>70.7|
||30.5<br>67.1<br>34.5<br>68.9<br>33.1<br>66.5<br>36.5<br>69.2|



Table 2: Ablation results of ARI. 

**Comparison of Reasoning Efficiency.** To validate the effectiveness of abstract instruction, we conduct an evaluation of reasoning efficiency. On the test set, with all other components of the model remaining constant, we remove the abstract instruction and record the average number of steps taken for reasoning. Compared with the ARI, we observe that under the guidance of abstract methodologies, LLMs not only improve in reasoning accuracy but also reduce their average number of reasoning steps by 11.4% on MULTITQ and 9.3% on CRONQUESTIONS. This underscores that the guidance provided by abstract methodologies can significantly enhance the efficiency of LLMs in temporal reasoning tasks. 

## **4.3 Ablation Study** 

To evaluate the efficacy of the individual components of the model, we conducted ablation studies. 

Initially, we remove the abstract guidance component, requiring the LLM to rely on its own understanding of the questions without the aid of historical information. This result in significant performance drops on both datasets, with a 19.7% decrease on MULTITQ and a 3.7% decrease on CRONQUESTIONS. This suggests that distilled abstract guidance plays a substantial role in supporting the model’s reasoning capabilities. 

To further assess the impact of abstract guidance, we eliminate the clustering module, thus deriving a universal abstract guidance from all historical reasoning processes without categorization based on question type. The model performance dropped by 9.2% on MULTITQ and 2.5% on CRONQUESTIONS, indicating that a singular abstract methodology is insufficient for guiding various types of questions and that targeted abstract methodological guidance is more effective. In Appendix A.5, we illustrate the impact of varying cluster quantity on the model’s final reasoning performance. 

To verify the role of incorrect samples in ARI, we remove incorrect examples from the process of generating abstract methods, providing only correct examples as guidance. As evident from the results, the removal of incorrect examples leads to a decrease in the quality of abstract guidance, resulting in a performance drop. By encountering and learning these incorrect examples, LLMs become more adept at avoiding similar pitfalls in subsequent reasoning tasks. This also aligns with our intuition and has been validated in prior studies (Wang and Li, 2023; Yang et al., 2023b; An et al., 2023). 

Lastly, we remove the action selection module, allowing the LLM to choose from all generated actions without filtering. This led to a decrease in performance on both datasets by 12.8% and 5.9%, underscoring that unfiltered actions result in an excessive number of options, including irrelevant ones, which hinders the LLM’s reasoning and complicates the decision-making process. 

## **References** 

- Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou, and Weizhu Chen. 2023. Learning from mistakes makes LLM better reasoner. _CoRR_ , abs/2310.20689. 

- Richard C. Atkinson and Richard M. Shiffrin. 1968. Human memory: A proposed system and its control processes. In _The psychology of learning and motivation_ . 

## **5 Conclusion and Limitation** 

This study, anchored in the principles of constructivism, critically examines the shortcomings of LLMs in addressing complex temporal reasoning challenges and proposes an innovative approach to augment their reasoning capabilities. Through the integration of a knowledge adaptability framework and abstract methodological guidance, we have shown that LLMs can attain more precise and efficient reasoning in complex temporal scenarios, effectively overcoming their constraints in processing and interpreting time-sensitive knowledge. 

**Limitations.** While ARI demonstrates impressive results, it also presents several limitations for ongoing refinement. Firstly, The efficacy of generating abstract guidance heavily relies on the capabilities of LLMs. Smaller-scale LLMs may struggle to produce high-quality abstract guidance, thus potentially restricting their application. Secondly, the ARI framework depends on multi-step reasoning to arrive at final answers, a process moderately influenced by the LLM’s reasoning efficiency, which extends the duration of inference. Finally, our method is primarily concentrated on complex temporal reasoning, with its effectiveness in other reasoning domains remaining to be examined. Future research should aim to refine these methods to make them more adaptable to various models and problem domains, enhance the balance between reasoning efficiency and depth, and expand their scope to include a broader range of reasoning tasks. 

## **Acknowledgement** 

The authors would like to thank the anonymous reviewers for their insightful and constructive comments, which greatly contributed to improving the quality of the paper. This work was partially supported by National Key R&D Program of China (No. 2022YFB3103600), NSFC (Nos. U23A20296, 62272469), and The Science and Technology Innovation Program of Hunan Province (No. 2023RC1007). 

- Jinheon Baek, Alham Fikri Aji, and Amir Saffari. 2023. Knowledge-augmented language model prompting for zero-shot knowledge graph question answering. _CoRR_ , abs/2306.04136. 

- Elizabeth Boschee, Jennifer Lautenschlager, Sean O’Brien, Steve Shellman, James Starz, and Michael Ward. 2015. ICEWS Coded Event Data. 

- Ziyang Chen, Jinzhi Liao, and Xiang Zhao. 2023. Multigranularity temporal question answering over knowledge graphs. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ . 

- Ziyang Chen, Xiang Zhao, Jinzhi Liao, Xinyi Li, and Evangelos Kanoulas. 2022. Temporal knowledge graph question answering via subgraph reasoning. _Knowl. Based Syst._ , 251:109134. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers)_ , pages 4171–4186. Association for Computational Linguistics. 

- Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, Lei Li, and Zhifang Sui. 2023. A survey for in-context learning. _ArXiv_ , abs/2301.00234. 

- Thibault Févry, Livio Baldini Soares, Nicholas FitzGerald, Eunsol Choi, and Tom Kwiatkowski. 2020. Entities as experts: Sparse memory access with entity supervision. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020_ , pages 4937–4951. Association for Computational Linguistics. 

- Yu Gu, Xiang Deng, and Yu Su. 2023. Don’t generate, discriminate: A proposal for grounding language models to real-world environments. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023_ , pages 4928–4949. Association for Computational Linguistics. 

- Yu Gu and Yu Su. 2022. Arcaneqa: Dynamic program induction and contextualized encoding for knowledge base question answering. In _Proceedings of the 29th International Conference on Computational Linguistics, COLING 2022, Gyeongju, Republic of Korea, October 12-17, 2022_ , pages 1718–1731. International Committee on Computational Linguistics. 

- Johannes Hoffart, Fabian M. Suchanek, Klaus Berberich, Edwin Lewis-Kelham, Gerard de Melo, and Gerhard Weikum. 2011. YAGO2: exploring and querying world knowledge in time, space, context, and many languages. In _Proceedings of the 20th International Conference on World Wide Web, WWW 2011, Hyderabad, India, March 28 - April 1, 2011 (Companion Volume)_ , pages 229–232. ACM. 

- Jie Huang and Kevin Chen-Chuan Chang. 2023. Towards reasoning in large language models: A survey. In _Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023_ , pages 1049–1065. Association for Computational Linguistics. 

- Zhen Jia, Abdalghani Abujabal, Rishiraj Saha Roy, Jannik Strötgen, and Gerhard Weikum. 2018. TEQUILA: temporal question answering over knowledge bases. In _Proceedings of the 27th ACM International Conference on Information and Knowledge Management, CIKM 2018, Torino, Italy, October 2226, 2018_ , pages 1807–1810. ACM. 

- Paul A. Kirschner, John Sweller, and Richard E. Clark. 2006. Why minimal guidance during instruction does not work: An analysis of the failure of constructivist, discovery, problem-based, experiential, and inquirybased teaching. _Educational Psychologist_ , 41:75 – 86. 

- Brenden M Lake and Marco Baroni. 2023. Human-like systematic generalization through a meta-learning neural network. _Nature_ , pages 1–7. 

- Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. 2020. ALBERT: A lite BERT for self-supervised learning of language representations. In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net. 

- Xingxuan Li, Ruochen Zhao, Yew Ken Chia, Bosheng Ding, Shafiq R. Joty, Soujanya Poria, and Lidong Bing. 2023a. Chain-of-knowledge: Grounding large language models via dynamic knowledge adapting over heterogeneous sources. 

- Yunxin Li, Baotian Hu, Xinyu Chen, Lin Ma, and Min Zhang. 2023b. Lmeye: An interactive perception network for large language models. _arXiv preprint arXiv:2305.03701_ . 

- Yunxin Li, Baotian Hu, Wei Wang, Xiaochun Cao, and Min Zhang. 2023c. Towards vision enhancing llms: Empowering multimodal knowledge storage and sharing in llms. _arXiv preprint arXiv:2311.15759_ . 

- Ke Liang, Yue Liu, Sihang Zhou, Wenxuan Tu, Yi Wen, Xihong Yang, Xiangjun Dong, and Xinwang Liu. 2023a. Knowledge graph contrastive learning based on relation-symmetrical structure. _IEEE Transactions on Knowledge and Data Engineering_ , pages 1–12. 

- Ke Liang, Lingyuan Meng, Meng Liu, Yue Liu, Wenxuan Tu, Siwei Wang, Sihang Zhou, and Xinwang Liu. 2023b. Learn from relational correlations and periodic events for temporal knowledge graph reasoning. In _Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval_ , pages 1559–1568. 

- Ke Liang, Lingyuan Meng, Meng Liu, Yue Liu, Wenxuan Tu, Siwei Wang, Sihang Zhou, Xinwang Liu, and Fuchun Sun. 2022. Reasoning over different types of knowledge graphs: Static, temporal and multi-modal. _arXiv preprint arXiv:2212.05767_ . 

- Ruibo Liu, Guoqing Zheng, Shashank Gupta, Radhika Gaonkar, Chongyang Gao, Soroush Vosoughi, Milad Shokouhi, and Ahmed Hassan Awadallah. 2022. Knowledge infused decoding. In _The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022_ . OpenReview.net. 

- Junru Lu, Siyu An, Mingbao Lin, Gabriele Pergola, Yulan He, Di Yin, Xing Sun, and Yunsheng Wu. 2023. Memochat: Tuning llms to use memos for consistent long-range open-domain conversation. _CoRR_ , abs/2308.08239. 

- Haoran Luo, Haihong E, Zichen Tang, Shiyao Peng, Yikai Guo, Wentai Zhang, Chenghao Ma, Guanting Dong, Meina Song, and Wei Lin. 2023. Chatkbqa: A generate-then-retrieve framework for knowledge base question answering with fine-tuned large language models. _CoRR_ , abs/2310.08975. 

- J. MacQueen. 1967. Some methods for classification and analysis of multivariate observations. 

- Aman Madaan, Niket Tandon, Peter Clark, and Yiming Yang. 2022. Memory-assisted prompt editing to improve GPT-3 after deployment. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022_ , pages 2833–2861. Association for Computational Linguistics. 

- Costas Mavromatis, Prasanna Lakkur Subramanyam, Vassilis N. Ioannidis, Soji Adeshina, Phillip R. Howard, Tetiana Grinberg, Nagib Hakim, and George Karypis. 2021. Tempoqr: Temporal question reasoning over knowledge graphs. _CoRR_ , abs/2112.05785. 

- Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christoforos Nalmpantis, Ramakanth Pasunuru, Roberta Raileanu, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz, Edouard Grave, Yann LeCun, and Thomas Scialom. 2023. Augmented language models: a survey. _CoRR_ , abs/2302.07842. 

- Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the role of demonstrations: What makes in-context learning work? _ArXiv_ , abs/2202.12837. 

- Liangming Pan, Michael Saxon, Wenda Xu, Deepak Nathani, Xinyi Wang, and William Yang Wang. 2023. Automatically correcting large language models: Surveying the landscape of diverse self-correction strategies. _CoRR_ , abs/2308.03188. 

- Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, and Jianfeng Gao. 2023. Check your facts and try again: Improving large language models with external knowledge and automated feedback. _CoRR_ , abs/2302.12813. 

- John F. Roddick and Myra Spiliopoulou. 2002. A survey of temporal knowledge discovery paradigms and methods. _IEEE Trans. Knowl. Data Eng._ , 14(4):750– 767. 

- John R. Savery and Thomas M. Duffy. 1995. Problem based learning: An instructional model and its constructivist framework. _Educational Technology archive_ , 35:31–38. 

- Apoorv Saxena, Soumen Chakrabarti, and Partha P. Talukdar. 2021a. Question answering over temporal knowledge graphs. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021_ , pages 6663–6676. Association for Computational Linguistics. 

- Apoorv Saxena, Soumen Chakrabarti, and Partha P. Talukdar. 2021b. Question answering over temporal knowledge graphs. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021_ , pages 6663–6676. Association for Computational Linguistics. 

- Apoorv Saxena, Aditay Tripathi, and Partha P. Talukdar. 2020. Improving multi-hop question answering over knowledge graphs using knowledge base embeddings. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020_ , pages 4498–4507. Association for Computational Linguistics. 

- Noah Shinn, Beck Labash, and Ashwin Gopinath. 2023. Reflexion: an autonomous agent with dynamic memory and self-reflection. _CoRR_ , abs/2303.11366. 

- Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Heung-Yeung Shum, and Jian Guo. 2023. Think-on-graph: Deep and responsible reasoning of large language model with knowledge graph. _CoRR_ , abs/2307.07697. 

- Denny Vrandecic and Markus Krötzsch. 2014. Wikidata: a free collaborative knowledgebase. _Commun. ACM_ , 57(10):78–85. 

- Danqing Wang and Lei Li. 2023. Learning from mistakes via cooperative study assistant for large language models. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023_ , pages 10667–10685. Association for Computational Linguistics. 

- Qingyue Wang, Liang Ding, Yanan Cao, Zhiliang Tian, Shi Wang, Dacheng Tao, and Li Guo. 2023. Recursively summarizing enables long-term dialogue memory in large language models. _CoRR_ , abs/2308.15022. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In _NeurIPS_ . 

- Fangzhi Xu, Zhiyong Wu, Qiushi Sun, Siyu Ren, Fei Yuan, Shuai Yuan, Qika Lin, Yu Qiao, and Jun Liu. 2023. Symbol-llm: Towards foundational symbolcentric interface for large language models. _CoRR_ , abs/2311.09278. 

- Linyao Yang, Hongyang Chen, Zhao Li, Xiao Ding, and Xindong Wu. 2023a. Chatgpt is not enough: Enhancing large language models with knowledge graphs for fact-aware language modeling. _CoRR_ , abs/2306.11489. 

- Zeyuan Yang, Peng Li, and Yang Liu. 2023b. Failures pave the way: Enhancing large language models through tuning-free rule accumulation. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023_ , pages 1751–1777. Association for Computational Linguistics. 

- Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net. 

- Junchi Yu, Ran He, and Rex Ying. 2023. Thought propagation: An analogical approach to complex reasoning with large language models. _CoRR_ , abs/2310.03965. 

- Danyang Zhang, Lu Chen, Situo Zhang, Hongshen Xu, Zihan Zhao, and Kai Yu. 2023. Large language model is semi-parametric reinforcement learning agent. _arXiv preprint arXiv:2306.07929_ . 

- Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 

2023a. A survey of large language models. _CoRR_ , abs/2303.18223. 

- Ziwang Zhao, Linmei Hu, Hanyu Zhao, Yingxia Shao, and Yequan Wang. 2023b. Knowledgeable parameter efficient tuning network for commonsense question answering. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023_ , pages 9051–9063. Association for Computational Linguistics. 

- Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2023. Memorybank: Enhancing large language models with long-term memory. _CoRR_ , abs/2305.10250. 

## **A Appendix** 

## **A.1 More Details about Baseline Methods** 

|||**Train**<br>**Dev**<br>**Test**|
|---|---|---|
|Single|Equal<br>Before/After<br>First/Last|135,890<br>18,983<br>17,311<br>75,340<br>11,655<br>11,073<br>72,252<br>11,097<br>10,480|
|Multiple|Equal Multi<br>After First<br>Before Last|16,893<br>3,213<br>3,207<br>43,305<br>6,499<br>6,266<br>43,107<br>6,532<br>6,247|
|**Total**||386,787<br>587,979<br>54,584|



Table 3: Statistics of question categories in MULTITQ. 

|||**Train**<br>**Dev**<br>**Test**|
|---|---|---|
|Simple|Simple Entity<br>Simple Time|90,651<br>7,745<br>7,812<br>61,471<br>5,197<br>5,046|
|Complex|Time Join<br>First/Last<br>Before/After|55,453<br>3,878<br>3,832<br>118,556<br>11,198<br>11,159<br>23,869<br>1,928<br>2,151|
|**Total**||350,000<br>30,000<br>30,000|



Table 4: Statistics of question categories in CRONQUESTIONS. 

In our evaluation, we compared several baseline methods. 

- **Pre-trained LMs** : To evaluate BERT (Devlin et al., 2019) and ALBERT (Lan et al., 2020), we generate their LM-based question embedding and concatenate it with the entity and time embeddings, followed by a learnable projection. The resulted embedding is scored against all entities and timestamps via dot-product. 

- **EmbedKGQA** (Saxena et al., 2020) is designed with static KGs. To deal with multiple temporal granularities, timestamps are ignored during pre-training and random time embeddings are used. 

- **CronKGQA** (Saxena et al., 2021a) is designed for single temporal granularity. To deal with multiple granularities, time embeddings at the year/month granularity are drawn at random from corresponding day embeddings. 

- **MultiQA** (Chen et al., 2023) is designed for multi-granularity temporal granularity with a transformer-based time aggregation module. 

- **ChatGPT**[*] . We use ChatGPT to provide direct answers to the questions. 

- **KG-RAG** . To validate the performance of the LLM in the presence of relevant background knowledge, we extracted relevant quaternions (up to 20) from the TKG based on the entity 

- *https://chat.openai.com/ 

- and time information appearing in the question, and put them in the prompt for ChatGPT to answer as a retrieval-enhanced way of comparison. 

- **ReAct KB** :To address the applicability of ReAct (Yao et al., 2023) to our task, we designed a variant of ReAct by integrating our knowledge-agnostic module, which generates all feasible actions (using the same atomic action templates as ARI). LLMs were then prompted to select one action from the available options. Parameter settings remained consistent with our ARI approach, including the use of the same Named Entity Linking (NEL) method and reasoning length. 

- **CoT KB** :We introduce the CoT KB method, which integrates a knowledge-based module into the Chain-of-Thought (CoT) (Wei et al., 2022) framework. This allows the LLM to interact with KG under the guidance of examples, thereby obtaining the final answer more effectively. We manually constructing 9 specific instance examples with detailed reasoning steps to guide the LLM, while keeping other settings unchanged. 

## **A.2 Datasets Statistics** 

**CRONQUESTIONS** (Saxena et al., 2021b) is a dataset for temporal knowledge graph question answering. The entities and times present in the questions are annotated. CRONQUESTIONS has four 

question types, including both simple and complex temporal questions. 

**MULTITQ** (Chen et al., 2023) is a complex temporal question answering dataset with multigranularity temporal information. Compared to existing datasets, MULTITQ features in a few advantages, including large scale, ample relations and multiple temporal granularity, which hence better reflects real-world scenarios. 

We summarize the number of questions in MULTITQ across different types in Table 3 and Table 4. In Table 5, we present sample questions from MULTITQ as per question type, time granularity and answer type. 

|**Property**|**SampleQuestion**|
|---|---|
|**By question**|**type**|
|Equal|_Which country provided humanitarian_<br>_aid to Sudan in 2007?_|
|Before/After|_Who commended the Military of Mali_<br>_before the Armed Rebel of Mali did?_|
|First/Last|_When did the Militant of Taliban frst_<br>_commend the Government of Pakistan?_|
|Equal Multi|_In 2012, who last did Barack Obama_<br>_appealfor?_|
|Before Last|_Who was threatened by Benjamin_<br>_Netanyahu last before Middle East?_|
|After First|_Who frst wanted to negotiate with Evo_<br>_Morales after the Citizen of Brazil did?_|
|**By timegranularity**||
|Year|_Who frst made Abu Sayyaf suffer from_<br>_conventional military forces In 2015?_|
|Month|_In Dec, 2008, who would wish to_<br>_negotiate with the Senate of Romania?_|
|Day|_In Jul 21st, 2011, who criticized the_<br>_Media of Ecuador?_|
|**By answer type**||
|Entity|_Which country visited Japan in 2013?_|
|Time|_When did China express intent to meet_<br>_with the Government of Pakistan?_|



Table 5: Representative examples from MULTITQ. 

## **A.3 Action Templates** 

Our approach is designed to be both generalizable and scalable. The templates in ARI are not highly manually defined high-level functions, but rather finely-grained atomic operations (e.g., extracting time, entity extraction, etc.). These atomic operations can be flexibly combined to generalize a wide range of complex actions, demonstrating their versatility and extensibility. 

Our action templates in ARI strictly follow the definition of functions in Table 6. We employ several specialized functions to facilitate precise information retrieval. The _getTime_ function retrieves the 

timing of specific events, based on given entities and relation. For temporal positioning, _getBefore_ , _getAfter_ , and _getBetween_ identify entities or events relative to specified time frames. In terms of entity queries, _getTailEntity_ and _getHeadEntity_ ascertain linked entities based on existing relation, with an optional time constraint. For queries targeting specific time instances, _getFirst_ and _getLast_ pinpoint entities with the earliest and latest occurrences, respectively. Responses are then articulated using the _answer_ function, providing a streamlined method for answering queries within the TKG. 

Our method exhibits low coupling with templates, making it adaptable to new data and domains. The extensibility of atomic templates is straightforward, allowing for easy incorporation of additional templates as needed. For instance, if we were to extend our approach to handle spatiotemporal data questions, adding a spatial atomic operation would be a straightforward task without the need for significant modifications. 

## **A.4 Details about the Instruction Format** 

In Figure 5, we illustrate an example of reasoning using the ARI model. Table 8 shows some exemplars of ARI. During each step of the process, the LLM receives guidance from abstract methods and selects the optimal action from available paths, continuing until it deems an answer has been sufficiently formulated or the maximum reasoning length is reached. Figure 8 presents the complete set of instructions used in our experiments, comprising components such as task definition, functional interpretations of potential actions, the current temporal question under consideration, historical reasoning steps, available candidate actions for the current round, feedback from the previous round’s action, and requirements for output formatting. 

## **A.5 Impact of Cluster Quantity** 

In Figure 7, we show the reduced dimensional clustering diagram for the 10 categories of questions in the experiment. To verify the effect of different number of clusters on the results, we present the impact of the number of historical reasoning process clusters on the results. As shown in Figure 6. We observe an initial increase followed by a decline in performance for both simple and complex problems. This pattern can be attributed to the fact that when the number of clusters is too low, the LLM is unable to distill concise and effective abstract 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction_images/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction.pdf-0014-00.png)


**----- Start of picture text -----**<br>
MNOPQR<br>uvwxxyzuw{|} ��
�����
���<br> !"#$%&'"""#""%(<br>nMNSTPUPVPWR�YZ[SQ\]SYPZb ^_`a$�(L K q?BrsHtD )*+ !,--
��#�.%�
��"#/�0�<br>g h --1�-2���#���'����#�<br>�
�%�
�-�
��.�3/<br>o f f K<br>i cade$�(<br>g h )+45$�6%�67899:98:99;<<br>q?BrsHtD<br>p f f f b<br>jaklma$�%�( i ===>?@ABC@D�F?GHIJ<br> !
$K(<br>Figure 5: A demonstration sample of ARI reasoning on MULTITQ.<br>Action template Comments )*+ !L���.�%��-�
��K�
�<br>���
���/�0%��
��<br>getTailEntity(head,rel,time) Identify the tail/object entity based on the head/subject entity and relation<br>3��K/<br>getHeadEntity(tail,rel,time) Identify the head/suject entity based on the tail/object entity and relation<br>getTime(head,rel,tail) Retrieve the time of a specific event based on the head entity, relation and tail entity )+4K~<br>getBetween(entities,Time1,Time2 Identify entities/events that occurred between two specific times<br>getBefore(entities,time) Identify entities/events that occurred before a given time<br>getAfte(entities,time)r Identify entities/events that occurred after a given time<br>getFirst(entities,time) Pinpoint entities with the earliest occurrence<br>getLast(entities,time) Pinpoint entities with the latest occurrence<br>answer(entities/time) To provide your answer, use the answer function<br>Table 6: Action templates in ARI.We employ these specialized functions to facilitate precise information retrieval.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction_images/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction.pdf-0014-01.png)


**----- Start of picture text -----**<br>
0.70<br>0.65<br>0.60<br>0.24<br>0.22<br>0.20<br>0.18<br>0.16<br>0 2 4 6 8 10 12 14 16 18 20<br>Number of Clusters<br>Simple Question<br>Complex Question<br>**----- End of picture text -----**<br>


Figure 6: Accuracy v.s Number of Clusters of ARI on MULTITQ. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction_images/acl_2024_Temporal_Knowledge_Question_Answering_via_Abstract_Reasoning_Induction.pdf-0014-03.png)


**----- Start of picture text -----**<br>
Cluster<br>0<br>0.4 1<br>2<br>3<br>4<br>0.2 5<br>6<br>7<br>0.0 8<br>9<br>0.2<br>0.4<br>0.4 0.2 0.0 0.2 0.4<br>Component 1<br>Component 2<br>**----- End of picture text -----**<br>


Figure 7: Clustering results for historical inference questions. 

methods from the noisy and abundant historical paths. Conversely, when the number of clusters is too high relative to a fixed number of historical samples, each category contains too few samples to provide the LLM with sufficient information to refine abstract methods. Thus, we observe a trend of improvement that eventually reverses as the number of clusters increases. 

## **A.6 Error Analysis** 

For error analysis, we randomly sample 100 error instances from the test set and summarized the following three types of typical errors: (1) Retrieving irrelevant entities (in MULTITQ), meaning the model obtained wrong entities from the KG; Although our entity linking model can achieve a high prediction accuracy, wrong entities still exist in 

some questions. (2)Low-quality abstract methodological guidance. Within the dataset, there exist complex problems for which the historical reasoning processes consistently led to incorrect conclusions. This lack of sufficient correct reasoning histories hampers the LLM’s ability to synthesize and refine effective abstract methodologies. Consequently, the low-quality abstract methods derived by the LLM prove inadequate in guiding subsequent reasoning processes, leading to a cascade of errors. (3) Uncertainty outputs of LLMs. Despite the constraint that LLMs can only select from candidate actions or provide final answers, there are instances where they do not strictly adhere to the given instruction. This non-compliance leads to the failure of our predefined graph query methods, consequently impeding effective reasoning. 

with abstract guidance, minimizing irrelevant noise by isolating them from direct knowledge sources. Unlike mainstream LLM agents designed for interactive environments, we employ a knowledgeagnostic module for environmental interactions, allowing the LLM to make decisions based on abstract guidance alone. Additionally, the study is open to incorporating interactive decision-making elements from recent LLM advancements, aiming to enhance the project’s effectiveness and broaden its applicability in future research. 

This demonstrates more efforts are needed to strengthen the model’s reasoning capability, particularly in the context of enhancing the reasoning capabilities of LLMs and diversifying their reasoning processes, it is crucial to provide a richer array of effective historical information for the generation of abstract methods. This approach is vital to prevent the LLMs from falling into a repetitive cycle of errors. 

## **A.7 Generalizability of Method on Other LLMs** 

|**Model**|**LLM only**|**ARI**|
|---|---|---|
|**Llama-2 Chat 7B**<br>**GPT-4**|0.040<br>0.125|0.105<br>0.411|



Table 7: Accuracy of ARI with Other LLMs on MULTITQ 

To assess the effectiveness of the ARI across various LLMs, we conducted experiments using the open-source model Llama-2-7B-chat and GPT-4. Our findings indicate that models with greater inherent capabilities yield better direct inference outcomes, presumably due to the acquisition of more extensive knowledge during training. Furthermore, the performance enhancements in ARI models built upon these base models are more pronounced. Notably, the ARI model based on GPT-4 achieved a score of 0.411 on the MULTITQ benchmark, representing a relative improvement of 40.2% over the current state-of-the-art models. This significant advancement underscores our model’s robust generalization capabilities. 

Our work focuses on enhancing LLM reasoning 

|**Question 1**: In|which month did the CityMayor of Philippines frstpraise Ona?|
|---|---|
|Action 0|getTime(City_Mayor_(Philippines),Praise_or_endorse,Ona)|
|Response 0|entities = [(’City_Mayor_(Philippines)’, ’2014-10-07’)]|
|Action 1|getFirst({entities})|
|Response 1|entities = [(’City_Mayor_(Philippines)’, ’2014-10-07’)]|
|Action 2|answer(2014-10)|
|Response 2|Correct!|
|**Question 2**: Before Mali’s militant,which countrywas the last to criticise France?||
|Action 0|getTime(Militant_(Mali),Criticize_or_denounce,France)|
|Response 0|entities = [(’Militant_(Mali)’, ’2013-01-21’)]|
|Action 1|getFirst({entities})|
|Response 1|entities = [("Government_(Cote_d’Ivoire)", ...]|
|Action 2|getLast({entities})|
|Response 2|entities = [(’Royal_Administration_(Spain)’, ’2015-12-24’)]|
|Action 3|answer(Royal_Administration_(Spain))|
|Response 3|Correct!|
|**Question 3**: Who was the last togive a criticism to Iran before Pervez Musharraf did?||
|Action 0|getTime(Pervez_Musharraf,Criticize_or_denounce,Iran)|
|Response 0|entities = [(’Pervez_Musharraf’, ’2006-01-17’)]|
|Action 1|getHeadEntity(Iran,Criticize_or_denounce,no time)|
|Response 1|entities = [(’Shirin_Ebadi’, ’2005-01-05’), ...]|
|Action 2|getBefore({entities},2006-01-17)|
|Response 2|entities = [(’Shirin_Ebadi’, ’2005-01-05’),...]|
|Action 3|getLast({entities})|
|Response 3|entities = [(’Angela_Merkel’, ’2006-01-16’)]|
|Action 4|answer(Angela_Merkel)|
|Response 4|Correct!|
|**Question 4**: At|what time did Xi Jinpingfrst make optimistic remarks on Japan?|
|Action 0|getTime(Xi_Jinping,Make_optimistic_comment,Japan)|
|Response 0|entities = [(’Xi_Jinping’, ’2008-04-18’), ...]|
|Action 1|getFirst({entities})|
|Response 1|entities = [(’Xi_Jinping’, ’2008-04-18’)]|
|Action 2|answer(2008-04-18)|
|Response 2|Correct!|
|**Question 5**: Who wanted to cooperate with Japan in November,2005?||
|Action 0|getHeadEntity(Japan,Express_intent_to_cooperate,2005-11)|
|Response 0|entities = [(’Government_Official_(Russia)’, ...]|
|Action 1|answer(South_Korea)|
|Response 1|Correct!|



Table 8: Exemplars of ARI for MULTITQ 

- f _’’’Please use the tool provided below to interact with the knowledge graph. You will find a list of actions categorized into time-based queries, entity queries, and specific time queries. There may be more than one answer to the question,_ 

- _but you only need to answer one correct answer that satisfies the question._ 

- _To solve this question, you need to first identify the entities and relationships in the question, selecting the appropriate actions to retrieve the required_ 

- _information, and finally, providing the correct answer._ 

_Time-based Queries:_ 

_Retrieve the time of a specific event based on the head/subject entity, relation and tail/object entity by using the $get_time(HEAD, RELATION, TAIL)$ function, . Identify entities/events that occurred before a given time by using the $get_before( ENTITY_LIST, SPECIFIED_TIME)$ function._ 

_Identify entities/events that occurred after a given time by using the $get_after( ENTITY_LIST, SPECIFIED_TIME)$ function._ 

_Identify entities/events that occurred between two specific times by using the $get_between(ENTITY_LIST, START_TIME, END_TIME)$ function._ 

_Entity Queries:_ 

_Identify the tail/object entity based on the head/subject entity and relation by using the $get_tail_entity(CURRENT_HEAD, RELATION, OPTIONAL_TIME_CONSTRAINT)$ function._ 

_Identify the head/suject entity based on the tail/object entity and relation by using the $get_head_entity(CURRENT_TAIL, RELATION, OPTIONAL_TIME_CONSTRAINT)$ function._ 

_Specific Time Queries:_ 

_Pinpoint entities with the earliest occurrence by using the $get_first(ENTITY_LIST)$ function._ 

_Identify entities with the latest occurrence by using the $get_last(ENTITY_LIST)$ function._ 

_To provide your answer, use the $answer(YOUR_ANSWER)$ function._ 

_Note: Always enclose the selected action in $ and provide a reason for your choice if necessary._ 

_Examples for your reference: {examples} (end of examples)_ 

_Current Challenge:_ 

_Question: {question}_ 

_Methodology: {methodology} (end of methodology)_ 

_Previous Actions: {history} (end of previous actions)_ 

_Available Actions: {actions}_ 

_Choose your next action from the available actions above, ensuring its completeness. If you have found the answer, remember to use the answer function._ 

_Organize your output by strictly following the format below:_ 

_Action:_ 

- _<Choose your next action from the available actions above. Note: Always enclose the selected action in $. Replace {your specified time} with a specified time in the format YYYY or YYYY-MM or YYYY-MM-DD>_ 

_Reason:_ 

_<Explain the reason for choosing this action.>’’’_ 

Figure 8: Prompt for the action selection. 

- f _’’’Carefully analyze the following correct and incorrect examples. From these, extract and summarize the corresponding patterns and principles. Based on these examples, provide a comprehensive methodology that describes how to correctly tackle this type of problem, highlighting the key steps and common pitfalls to avoid._ 

_Task Defination: <Task Defination> (end of Task Defination)_ 

- _Here is an example output: Example 1: Overall methodology Instruction: This type of problem involves the sequential determination of events, e.g. Who { Relation R} {entity C} before {entity B}, to find the answer {entity A} we need to reason in three steps, firstly to determine the specific temporal anchors, i. e., the occurrence time t of {entity B, Relation, and entity C}, and then to find out which head entities have generated a Relation R connection with {entity C}. Then, we find out which head entities and {entity C} have been associated_ 

- _with Relation R, and finally filter out the answers that satisfy the time requirement before t. The specific steps are as follows. The steps are as follows_ 

_Step-by-step Guide:_ 

_1. Firstly, use get_time to find the time, $get_time(entity B, Relation R, entity C) $, to get the quaternion {entity B, Relation R, entity C, Time t};_ 

_2. use the get_head_entity method to get the head entity, $get_head_entity(entity C, Relation R, entity C)$, to be able to get a list of quaternions;_ 

_3. use the get_before method to filter the entities that satisfy the constraints, $get_before({entities},t)$, to be able to obtain a list of entities that satisfy the conditions_ 

_4. complete the reasoning process by answering the found answer $answer(entity A)$_ 

_(end of example output)_ 

_Here is the correct samples and incorrect samples for the current question type: Correct samples: {correct_examples}_ 

_Incorrect samples: {incorrect_examples} (end of samples)_ 

- _Now start writing. Please design a methodology that describes how to correctly tackle this type of problem. The goal is to provide a comprehensive guide that highlights the key steps and common pitfalls to avoid when approaching this type of problem.organize your output by strictly following the output format as_ 

- _below:_ 

_Overall Instruction:_ 

_<Define this methodology in detail. Provide a concise guide or inference. Note that the guidance you provide should be at a methodological level, for this type of question, not for a specific one. >_ 

_Step-by-step Guide:_ 

_<A step-by-step guide or procedure detailing how to approach and solve this kind of question. Note that the steps proposed should be specific and relevant to this type of question, tell which type of action should use in each step and the reason>’’’_ 

Figure 9: Prompt for the abstract methodology instruction generation. 

