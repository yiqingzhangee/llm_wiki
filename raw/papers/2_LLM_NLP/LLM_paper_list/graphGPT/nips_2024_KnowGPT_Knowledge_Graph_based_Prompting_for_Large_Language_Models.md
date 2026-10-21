# **KnowGPT: Knowledge Graph based PrompTing for Large Language Models** 

**Qinggang Zhang[1*] , Junnan Dong[1*] , Hao Chen[1] , Daochen Zha[2] , Zailiang Yu[3] ,Xiao Huang[1]** 1The Hong Kong Polytechnic University, 2Rice University, 3Zhejiang Lab {qinggangg.zhang,hanson.dong}@connect.polyu.hk, sundaychenhao@gmail.com daochen.zha@rice.edu, yuzl@zhejianglab.com,xiaohuang@comp.polyu.edu.hk 

## **Abstract** 

Large Language Models (LLMs) have demonstrated remarkable capabilities in many real-world applications. Nonetheless, LLMs are often criticized for their tendency to produce hallucinations, wherein the models fabricate incorrect statements on tasks beyond their knowledge and perception. To alleviate this issue, researchers have explored leveraging the factual knowledge in knowledge graphs (KGs) to ground the LLM’s responses in established facts and principles. However, most state-of-the-art LLMs are closed-source, making it challenging to develop a prompting framework that can efficiently and effectively integrate KGs into LLMs with hard prompts only. Generally, it usually suffers from three critical issues, including huge search space, high API costs, and laborious prompt engineering, that impede the widespread application in practice. To this end, we introduce a novel **Know** ledge **G** raph-based **P** romp **T** ing framework, namely **KnowGPT** , to enhance LLMs with domain knowledge. KnowGPT contains a knowledge extraction module to extract the most informative knowledge from KGs, and a context-aware prompt construction module to automatically convert extracted knowledge into effective prompts. Experiments on three benchmark datasets demonstrate that KnowGPT significantly outperforms all competitors. Notably, KnowGPT achieves a 92 _._ 6% accuracy on OpenbookQA leaderboard, close to human-level performance. 

## **1 Introduction** 

Large Language Models (LLMs), such as GPT-4 [43] and Claude 3 [3], have surprised the world with superior performance in a wide range of real-world applications [27; 71]. Despite their impressive performance, LLMs are frequently criticized for their limited ability to handle factual information accurately and their tendency to generate hallucinations, especially when faced with questions requiring domain-specific or professional knowledge not covered in their training corpus [2; 48]. For example, when queried about nutrient composition, an LLM might erroneously associate it with “energy”, as depicted in Figure 1. This error stems from the model’s insufficient biological knowledge. Therefore, integrating domain knowledge into LLMs is crucial for reducing hallucinations and unlocking their full potential in diverse industry applications [17]. 

Recently, retrieval-augmented generation has been explored, which can enhance LLMs with external knowledge from text corpora or online sources [30; 76; 42]. It combines LLMs with external knowledge retrieval systems to help reduce hallucinations. However, these models face challenges in real-world applications due to the varying quality of available data. Domain knowledge is often scattered across different sources, such as textbooks, research papers, technical manuals, and industry reports [31]. These textual documents may have varying levels of quality, accuracy, and completeness, leading to potential inconsistencies or errors in the retrieved knowledge [78]. 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models_images/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models.pdf-0002-00.png)


**----- Start of picture text -----**<br>
When producers create food in an ecosystem, a<br>portion of the nutrients are?<br>A. carbs  B. energy C. grass D. flowers<br>ecosystem<br>B. energy product food web<br>carbs<br>producer food<br>A. carbs carbo nutrient<br>Carbohydrate<br>**----- End of picture text -----**<br>


Figure 1: A real-world question from OpenbookQA. GPT-3.5 could effectively correct the answer given the scientific reasoning background from ConceptNet (blue: question concepts, red: answers, grey: entities not present in questions). 

A promising avenue for addressing the above issue entails the integration of Knowledge Graphs (KGs) into LLMs. KGs provide a structured representation of domain knowledge, as they are constructed based on rigid ontologies that clearly define the jargon, acronyms, specialized terminologies and their relationships in specific domains. The enormous factual knowledge stored in KGs holds the potential to ground the model’s responses in established facts and principles [20; 66; 45]. For instance, in Figure 1, an LLM can correct itself by leveraging the related background knowledge in ConceptNet [51]. Earlier studies adopted a heuristic way to inject knowledge from KGs into the LLMs during pre-training or fine-tuning. For example, ERNIE [52] incorporates entity embeddings and aligns them with word embeddings in the pre- 

training phase, encouraging the model to better understand and reason over entities. Similarly, KnowBERT [46] integrates entity linkers with BERT to infuse knowledge about entities during fine-tuning for knowledge-intensive applications. Another line of work focuses on retrieving relevant knowledge from KGs at inference time to augment the language model’s context. Typically, K-BERT [37] uses an attention mechanism to select relevant triples from KGs based on the query context, which are then appended to the input sequence. Similarly, KEPLER [62] learns joint embeddings of text and KG entities to enhance the model’s predictions. Subsequent works have further integrated graph neural networks alongside LLMs for joint reasoning [69; 74; 10] and introduced interactions between text tokens and KG entities within the intermediate layers of LLMs [53; 56]. 

However, as LLMs have been keeping evolving, most SOTA LLMs remain _closed-source_ in practice. For instance, GPT-4 [43] and Claude 3 [3] exclusively grant access through their APIs, which means we can only retrieve model responses by submitting textual inputs, with model specifics inaccessible. As such, the research focus has recently shifted towards KG prompting that enhances fixed LLMs with KG-based hard prompts [36; 45]. KG Prompting for LLMs has been a new learning paradigm in natural language processing. Specifically, CoK [60] introduces Chain-of-Knowledge prompting to decompose LLM-generated reasoning chains into evidence triples, verifying their factuality and faithfulness using an external KG. Mindmap [63] provides more transparency on LLMs’ decisionmaking by enabling comprehension and reasoning over structured KG inputs. RoG [39] presents a planning-retrieval-reasoning framework that synergizes LLMs and KGs for more transparent and interpretable reasoning. KGR [18] proposes an autonomous approach to retrofit LLM-generated responses by leveraging KGs to extract, verify, and refine factual information throughout the reasoning process, effectively mitigating hallucination for knowledge-intensive applications. 

Despite the promising performance of existing KG prompting methods, three critical issues hinder their widespread application in practice. ❶ Huge search space. Real-world KGs often consist of millions of triples, resulting in a vast search space when retrieving relevant knowledge for prompting. ❷ High API cost. Closed-source LLMs, like GPT-4 and Claude 3, are accessible through proprietary APIs, which can incur significant costs when performing KG prompting at scale. Thus, careful selection of the most informative knowledge from KGs is essential to minimize costs. ❸ Laborious prompt design. LLMs are highly sensitive to prompts, with even minor variations in prompts conveying the same semantic meaning potentially yielding drastically different responses. However, existing methods rely on manually designed or rule-based prompts to present factual knowledge from KGs. These hard prompts are inherently inflexible and rigid, lacking the adaptability to accommodate variations in question semantics and KG structures. 

To this end, we propose a novel **Know** ledge **G** raph-based **P** romp **T** ing framework, namely **KnowGPT** , which leverages the factual knowledge in KGs to ground the model’s responses in established facts and principles. In this paper, we aim to answer two key research questions. ❶ Given a query and a large-scale KG, how could we effectively and efficiently retrieve factual knowledge from KG that is relevant to the query? ❷ Given the raw knowledge extracted from KGs, how could we convert the extracted knowledge into an effective prompt that is easily understandable for LLM? 

We shed light on the the above questions with a novel prompt learning method that is rather effective, generalizable, and cost-efficient. Specifically, to address question ❶, we leverage deep reinforcement 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models_images/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Question Context Multiple  choices EncodingContext 𝑪 Knowledge extraction 1 1 𝒫/0 Prompt Construction ℱ𝓉 Update (YES/NO?) Answer<br>SourceSource/Target 21 ExtractionEntity Target 31 24 Subgraph Retrieval 14 1 2 2 3 Policy Network hidden layerhidden layersoftmax𝜋StateReLUReLU𝑎𝑠| aaa312 1111 22 State 2State 1State 3State 0 Extracted paths211 2 𝒫43,-. 214356 Sentences DescriptionTriplesGraph ℱℱ"! 𝑬𝐸𝐸𝐸𝐸𝐸 𝓠𝓟𝓕𝒬𝒫ℱ𝒬𝒫ℱ𝒬𝒫ℱ𝒬𝒫ℱ𝒬𝒫ℱ #%"&!𝟑 =0.31=0.15=0.79=0.24=0.74 [=0.92] 𝒫𝑅𝐿ℱ+ Prioritized Prompt            central in the network. The entity is strongly associated ... graph contextsymbolizing the leadership. Thus, Google serves as a vital link between the key figures. K Google… stands       In the present , Sundar …,  GPT-4<br>a4 1 2 3 End Heuristic Subgraph<br>actions Raw Knowledge Prompt Formats<br>Reward Reward<br>Reward …<br>Reward<br>**----- End of picture text -----**<br>


Figure 2: The overall architecture of our proposed knowledge graph prompting framework, i.e., KnowGPT. Given the question context with multiple choices, we first retrieve a question-specific subgraph from the real-world KG. _Knowledge Extraction_ is first dedicated to searching for the most informative and concise reasoning background subject to the context. Then the _Prompt Construction_ module is optimized to prioritize the combination of knowledge and formats subject to the given question. 

learning (RL) to extract the most informative knowledge from KGs. To encourage the agent to discover more informative knowledge chains, we devise a tailored reward scheme that promotes the reachability, context-relatedness, and conciseness of the extracted paths. Then, a policy network is trained to maximize the reward using training questions and applied to unseen questions. To tackle question ❷, we introduce a prompt construction strategy based on Multi-Armed Bandit (MAB). Given several knowledge extraction strategies and prompt templates, an MAB is learned to select the most effective combination for each question by balancing exploration and exploitation. The learned MAB is then applied to new questions to select knowledge extraction strategies and prompt templates automatically. Our main contributions are summarized as follows: 

- Formally define the problem of KG-based prompting, which leverages the structured knowledge in KGs to ground the LLM’s responses in established facts and principles. 

- Propose KnowGPT, a novel prompting framework that leverages deep reinforcement learning (RL) and Multi-Armed Bandit (MAB) to generate effective prompts for domain-specific queries. 

- Implement KnowGPT upon GPT-3.5. Experiments on three QA datasets shows KnowGPT outperforms SOTA baseline models by a large margin. Notably, KnowGPT achieves an average improvement of 23.7% over GPT-3.5 and an average improvement of 2.9% over GPT-4. Additionally, it attains a 92.6% accuracy on the OpenbookQA leaderboard, which is comparable to human performance. 

## **2 Problem Statement** 

We formally define the problem of _knowledge graph based prompting for LLMs_ in question answering. We represent each question as a question context _Q_ = _{Qs, Qt}_ , where _Qs_ = _{e_ 1 _, e_ 2 _, ..., em}_ is a set of _m source entities_ , and _Qt_ = _{e_ 1 _, e_ 2 _, ..., en}_ is a set of _n target entities_ . Following prior work [13; 68], _Qs_ is extracted by concept recognition, and we assume it is given in our problem. Similarly, each target entity in _Qt_ is extracted from a corresponding candidate answer. We denote an LLM as _f_ , a real-world KG as _G_ , which consists of triples (head entity, relation, tail entity), denoted as ( _h, r, t_ ). In our setting, we only have access to the APIs of _f_ . However, we can employ open-source lightweight language models (not _f_ ), like Bert-Base [24], to initialize question embeddings. Using the above notations, we describe our problem below. 

Given a question _Q_ , an LLM _f_ , and a domain KG _G_ , we aim to learn a prompting function _f_ prompt( _Q, G_ ), which generates a prompt **x** that incorporates the context of _Q_ and the factual knowledge in _G_ , such that the prediction of the LLM _f_ ( **x** ) can output the correct answers for _Q_ . 

## **3 KnowGPT Framework** 

Learning the prompting function _f_ prompt( _Q, G_ ) involves two challenges, i.e., what knowledge should be used in _G_ , and how to construct the prompt. To address these challenges, we present KnowGPT, which extracts raw knowledge with deep RL and then constructs the prompt with MAB. An overview of our framework is shown in Figure 2. 

3 

## **3.1 Knowledge Extraction with Deep Reinforcement Learning** 

Intuitively, the relevant reasoning background lies in a question-specific subgraph _G_ sub that contains all the _source_ entities _Qs_ , _target_ entities _Qt_ , and their neighbors. An ideal subgraph _G_ sub is expected to have the following properties: ( _i_ ) _G_ sub encompasses as many source and target entities as possible, ( _ii_ ) the entities and relations within _G_ sub exhibit a strong relevance to question context, and ( _iii_ ) _G_ sub is concise with little redundant information such that it can be fed into LLMs with limited lengths. 

However, it is challenging to find such a _G_ sub since extracting a subgraph is NP-hard. To effectively and efficiently find a satisfactory _G_ sub, we develop a tailored knowledge extraction method, named _P_ RL, that employs deep RL to sample reasoning chains in a trial-and-error fashion. Specifically, we assume _G_ sub is constructed based on a set of reasoning chains _P_ = _{P_ 1 _, P_ 2 _, ..., Pm}_ , where each knowledge chain _Pi_ = _{_ ( _ei, r_ 1 _, t_ 1) _,_ ( _t_ 1 _, r_ 2 _, t_ 2) _, ...,_ ( _t|Pi|−_ 1 _, r|Pi|, t|Pi|_ ) _}_ is a path in _G_ starting from the _i_ - _th_ source entity in _Qs_ , and _|Pi|_ is the path length. _G_ sub encompasses all the entities and relations appeared in _P_ . We model the sampling of each reasoning path as a Markov Decision Process (MDP) with state, action, transition, and reward, defined as follows. 

- **State:** A state indicates the current location in KG, i.e., one of the entities in KG. Specifically, it represents the spatial change from entity _h_ to _t_ . Inspired by the prior study [65], we define the state vector _**s**_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models_images/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models.pdf-0004-04.png)


- where _**e** t_ and _**e** target_ are the embedding vectors of the current entity and the target entity. To get the initial node embeddings for entities extracted from the background KG, we adopt the approach proposed by the previous study [13]. Specifically, we transform knowledge triples from the KG into sentences and feed them into pre-trained LM to get node embeddings. 

- **Action:** The action space encompasses all the neighboring entities of the current entity, enabling the agent to explore the KG flexibly. By taking an action, the agent will move from the current entity to the chosen neighboring entity. 

- **Transition:** The transition model P measures the probability of moving to a new state ( _s[′]_ ) given existing state ( _s_ ) and the undertaken action ( _a_ ). In KGs, the transition model takes on the form P( _s[′] |s, a_ ) = 1 if _s_ is directed to _s[′]_ through action _a_ ; Otherwise, P( _s[′] |s, a_ ) = 0. 

- **Reward:** To determine the quality of the formed path, we define the reward based on reachability: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models_images/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models.pdf-0004-09.png)


which represents whether the path eventually reaches the target within limited steps. Specifically, the agent receives a reward of +1 if it can attain the target within _K_ actions. Otherwise, it will receive _−_ 1 as the reward. 

Reaching a target entity is not our sole focus. To avoid overlong and rigmarole reasoning chains, we also design two auxiliary rewards to promote context-relatedness and path conciseness. 

## **3.1.1 Context-relatedness Auxiliary Reward** 

The key motivation is to encourage paths closely related to the given question context. Specifically, we evaluate the semantic relevance of a path _Pi_ to the context _Q_ . Inspired by the prevailing study [69], a fixed but well-trained matrix _**W**_ is applied to map the path embedding _**P**_ to the same semantic space with context embedding _**c**_ . To this end, this auxiliary reward is formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models_images/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models.pdf-0004-14.png)


where _**c**_ is the embedding of context _Q_ we obtained from a pre-trained LM [24] and the embedding of path _Pi_ is the average of the embeddings of all the entities and relations we have walked through till _i_ , i.e., _Avg_ ( _**e** source_ + _**re**_ 1 _..._ + _**e** i_ ), where _i ≤ length_ ( _Ptarget_ ). This step-by-step reward scheme provides rewards before the target is reached. 

4 

## **3.1.2 Conciseness Auxiliary Reward** 

There are two additional significant challenges for the candidate reasoning background. ( _i_ ) The natural limitation of LLMs for over-long context understanding gives constrained budgets for prompts, where the extracted knowledge chain is expected to be concise enough to ensure the full understanding by closed-source LLMs. ( _ii_ ) The prohibitive cost of calling LLMs’ API guides the prompt to be more concise. By limiting the step size, we encourage the policy to find as much valuable information as possible within the shortest path length. 

Considering the inevitable homogeneity in the large-scale real-world KG constructed from the online corpus, each step in the final path is ideally a necessity. Specifically, we evaluate the conciseness of a path to reduce twists and turns on redundant entities, e.g., synonyms. Thus, the reward for the conciseness of a path _Pi_ is formulated as follows. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models_images/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models.pdf-0005-03.png)


To this end, our overall reward modeling consists of three major criteria that comprehensively incentivize the entire policy learning for an effective knowledge extraction. 

## **3.1.3 Training Policy Network** 

To solve the MDP defined above, a tailored policy network _πθ_ ( _s, a_ ) = _p_ ( _a|s_ ; _θ_ ) is trained to extract a reasoning chain in the KG. We optimize the network with policy gradient [65]. The optimal policy navigates the agent from the source entity to the target entity while maximizing the accumulated rewards. We provide more training details in the Appendix. 

## **3.2 Prompt Construction with Multi-armed Bandit** 

In this subsection, we design a tailored prompt construction strategy based on Multi-Armed Bandit (MAB). The key idea is to learn to select the best knowledge extraction and prompt templates at a meta-level. We will begin by outlining the overall strategy, followed by detailing its instantiation with two knowledge extraction methodologies and three templates. 

Suppose we have several knowledge extraction strategies _{P_ 1 _, P_ 2 _, ..., Pm}_ and several candidate prompt formats _F_ = _{F_ 1 _, F_ 2 _, ..., Fn}_ . Each knowledge extraction strategy _Pi_ is a method for selecting reasoning background given a question context, such as the RL-based strategy discussed above. Every prompt template _Fj_ represents a mechanism to transform the triples within the subgraph into a prompt for an LLM prediction. 

The prompt construction problem is to identify the best combination of _P_ and _F_ for a given question. We define the overall process of selection as a reward maximization problem, max[�] _rpf_ , where _rpf_ is obtained as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models_images/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models.pdf-0005-11.png)


Specifically, _PF_ ( _i_ ), _i ∈{_ 0 _,_ 1 _, · · · , m × n}_ is one of the combination, and _rpf ∈{_ 0 _,_ 1 _}_ indicates the performance of the output of LLM in answering the current question. 

To capture the context-aware correlation between questions and different combinations of knowledge and prompt formats, we formulate the selection mechanism of MAB with an expectation function _E_ ( _·_ ). It adaptively measures the potential expectation of a combination for different questions. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models_images/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models.pdf-0005-14.png)


Here, _**c**_ represents the embedding of _Q_ . The vector _**α**_ ( _i_ ) corresponds to a set of non-negative parameters associated with _PF_ ( _i_ ), which have been learned during the previous _k_ -1 iterations. Additionally, _β_ ( _i_ ) stands for a balancing factor introducing noise according to a Gaussian distribution. 

Empirically maximizing _**c** ×_ _**α** i_ could encourage exploitation [8; 11] for the best combination, we could effectively update _**α**_ ( _i_ ) via modeling the correlations between the context embedding of the anchor question **c** _i_ and all the previously selected contexts **C** ( _i_ ) for particular combination _PF_ ( _i_ ) 

5 

in former _k_ steps, and the rewards **r**[(] _pf[i]_[)][obtained][from][the][selection][of][the][current][combination.] Concretely, the _**β**_[(] _[b]_[)] is updated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models_images/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models.pdf-0006-01.png)


Here, _J_ denotes the OLS training loss. **I** _∈_ R _[d][×][d]_ is an identity matrix and _λ[i]_ is a regularization factor that controls the complexity of the model. 

Similarly, in order to encourage exploration within less frequently selected pairings, we employ an upper confidence bound approach to balance exploration and exploitation. This is achieved through the introduction of the parameter _β_[(] _[i]_[)] . Inspired by prevailing studies [59; 11], we can derive the following exploration term _β_[(] _[i]_[)] : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models_images/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models.pdf-0006-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models_images/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models.pdf-0006-05.png)


When the model picks a combination with a large _**c** ×_ _**α** i_ , it signifies an exploitation process. Likewise, when the model selects a combination with larger _β_[(] _[i]_[)] , this variance indicates an exploration process due to the model making fewer selections of the current combination. Thus, jointly maximizing _**c** ×_ _**α** i_ + _β_ ( _i_ ) could help us get rid of the dilemma of exploration and exploitation. 

Consequently, our MAB design can leverage the feedback from the LLM to optimize the selection policy. By maximizing the expectation function _E_ ( _·_ ), it learns to balance the exploitation and exploration to prioritize the most promising prompts for specific question contexts. 

## **3.2.1 Implementation** 

We implement the above MAB strategies with two knowledge extraction strategies and three templates. Note that our MAB design is general and can be implemented with more knowledge extraction strategies and prompt templates for better performance. The knowledge extraction strategies include: 

- _P_ RL: The RL-based knowledge extraction strategy presented in the previous subsection. 

- _P_ sub: A heuristic sub-graph extraction strategy that extracts a 2-hop subgraph around both the source and target entities. Detailed implementation can be found in Section B.1 of Appendix. Since RL is notoriously unstable [54], we introduce _P_ sub as an alternative candidate strategy for the MAB selection, ensuring a fallback option if the RL-based approach does not perform well. 

The prompt templates include: 

- **Triples** , denoted as _Ft_ , are indeed the originally extracted knowledge and empirically tested that could be understood by the black-box LLMs, e.g., ( _Sergey_Brin_ , _founder_of_ , _Google_ ),( _Sundar_Pichai_ , _ceo_of_ , _Google_ ), ( _Google_ , _is_a_ , _High-tech Company_ ). 

- **Sentences** is a following solution to transform the knowledge into a colloquial _Fs_ , e.g., ‘ _Sergey Brin, who is a founder of Google, a high-tech company, has now passed the reigns to Sundar Pichai, who is currently serving as the CEO of the company._ ’ 

- **Graph Description** , _Fg_ prompts the LLM by treating the knowledge as a structured graph. We preprocess the extracted knowledge with black-box LLM itself to generate the description by highlighting the center entity, e.g., ‘ _Google, a high-tech company, stands central in the network. The entity is strongly associated with significant individuals in the tech industry. Sergey Brin, one of the founders, established Google, underscoring its historical beginnings. In the present graph context, Sundar Pichai is recognized as the CEO of Google, symbolizing the company’s current leadership. Thus, Google serves as a vital link between these key figures._ ’ 

Considering two knowledge extraction methods: _P_ sub and _P_ RL, as well as three prompt translation methods: _Ft_ , _Fs_ and _Fg_ , the MAB is trained to learn from the feedback from LLMs 

6 

to prioritize the most appropriate combination among two extraction methods and three predefined prompt formats for different real-world question contexts, i.e., _PF_ = _{_ ( _PsubFt_ ) _,_ ( _PsubFs_ ) _,_ ( _PsubFg_ ) _,_ ( _PRLFt_ ) _,_ ( _PRLFs_ ) _,_ ( _PRLFg_ ) _}_ . 

## **4 Experiments** 

We conduct extensive experiments to evaluate KnowGPT on three benchmark question-answering datasets, covering both commonsense and domain-specific QA. We implement KnowGPT upon GPT3.5. Our experiments are designed to answer the following research questions: 

- **RQ1 (Main results)** : How does KnowGPT perform when compared with the state-of-the-art LLMs and KG-enhanced QA baselines? 

- **RQ2 (Ablation Study)** : How does each key component of KnowGPT contribute to the performance? 

- **RQ3 (Case study)** : How could KG help solve complex reasoning tasks? See Appendix C.2. 

## **4.1 Experimental Setup** 

**Datasets** . We evaluate KnowGPT on three QA datasets spanning two fields: CommonsenseQA [55] and OpenBookQA [41] serve as benchmarks for commonsense reasoning, while MedQA-USMLE [23] acts as a domain-specific QA benchmark. The statistics of these three datasets can be found in Table 5 in the Appendix. 

**Baselines** . We carefully select baseline models from four categories for a comprehensive evaluation. _LM + Fine-tuning_ . We compare our method with vanilla fine-tuned LMs. Specifically, we choose Bert-base, Bert-large [24], and RoBerta-large [38] as representative fine-tune LM methods. To conduct commonsense and biomedical QA, we fine-tune these three LMs via additional linear layers. 

_KG-enhanced LM_ . We have also implemented several recently released models for integrating KGs into question answering, which encompass MHGRN [13], QA-GNN [69], HamQA [10], JointLK [53], GreaseLM [75] and GrapeQA [56]. To ensure a fair comparison, we implement these baselines with advanced language models that are optimized for particular datasets. Specifically, RoBerta-large [38] is used for CommenseQA, while AristoRoBERTa [9] is designated for OpenBookQA. For MedQA, we opt for the top-tier biomedical language model, SapBERT [35]. Note that due to the white-box nature of these methods and their high computation overheads, it is infeasible to apply them to state-of-the-art LLMs, like GPT-3.5 and GPT-4. 

_LLM + Zero-shot_ . We include several representative generative LLMs, including ChatGLM, ChatGLM2, Baichuan-7B, InternLM, GPT-3, GPT-3.5 and GPT-4 as knowledge-agnostic alternatives. Specifically, we used the model ‘text-davinci-002’ provided by OpenAI as the implementation of GPT3, and ‘gpt-3.5-turbo’ and ‘gpt-4’ as the implementations of GPT-3.5 and GPT-4, respectively (we have provided more implementation details of all LLMs in Appendix A.4). The question-answering task is conducted under the zero-shot setting with the question query from the test set as input. 

_LLM + KG Prompting_ . To verify the effectiveness of our prompting strategy, we also add the state-of-the-art KG prompting methods, i.e., CoK [60], RoG [39], and Mindmap [63] as baselines. Notably, we did include KGR [18] in our main results, since the authors have not released their codes. 

## **4.2 Main Results (RQ1)** 

To address **RQ1** , we evaluate KnowGPT by comparing it to state-of-the-art baselines on the three benchmark datasets. KnowGPT is based on the original GPT-3.5. We measure the performance using accuracy, which calculates the percentage of questions correctly predicted by the model out of the total questions in the test set. We have the following observations: 

- KnowGPT outperforms all categories of methods, including sixteen different baselines, across all datasets and model architectures. This suggests that KnowGPT can effectively inject the knowledge from KGs to LLMs. 

- KnowGPT surpasses the performance of GPT-3.5 and even GPT-4. On average, KnowGPT achieves a 23.7% higher testing accuracy than GPT-3.5. Specifically, KnowGPT outperforms GPT-3.5 by 10.8%, 32.4%, and 29.4% on the CommonsenseQA, OpenBookQA, and MedQA datasets, respectively. 

7 

Table 1: Performance comparison among baseline models and KnowGPT on three benchmark datasets. 

|**Catagory**<br>**Model**<br>**CommonsenseQA**<br>**OpenBookQA**<br>**MedQA**<br>IHdev-Acc. IHtest-Acc. Dev-Acc. Test-Acc. Dev-Acc. Test-Acc.|**Catagory**<br>**Model**<br>**CommonsenseQA**<br>**OpenBookQA**<br>**MedQA**<br>IHdev-Acc. IHtest-Acc. Dev-Acc. Test-Acc. Dev-Acc. Test-Acc.|
|---|---|
|LM + Fine-tuning<br>Bert-base<br>0.573<br>0.535<br>0.588<br>0.566<br>0.359<br>0.344<br>Bert-large<br>0.611<br>0.554<br>0.626<br>0.602<br>0.373<br>0.367<br>RoBerta-large<br>0.731<br>0.687<br>0.668<br>0.648<br>0.369<br>0.361||
|KG-enhanced LM<br>MHGRN<br>0.745<br>0.713<br>0.786<br>0.806<br>-<br>-<br>QA-GNN<br>0.765<br>0.733<br>0.836<br>0.828<br>0.394<br>0.381<br>HamQA<br>0.769<br>0.739<br>0.858<br>0.846<br>0.396<br>0.385<br>JointLK<br>0.777<br>0.744<br>0.864<br>0.856<br>0.411<br>0.403<br>GreaseLM<br>0.785<br>0.742<br>0.857<br>0.848<br>0.400<br>0.385<br>GrapeQA<br>0.782<br>0.749<br>0.849<br>0.824<br>0.401<br>0.395||
|LLM + Zero-shot<br>ChatGLM<br>0.473<br>0.469<br>0.352<br>0.360<br>0.346<br>0.366<br>ChatGLM2<br>0.440<br>0.425<br>0.392<br>0.386<br>0.432<br>0.422<br>Baichuan-7B<br>0.491<br>0.476<br>0.411<br>0.395<br>0.334<br>0.319<br>InternLM<br>0.477<br>0.454<br>0.376<br>0.406<br>0.325<br>0.348<br>Llama2 (7b)<br>0.564<br>0.546<br>0.524<br>0.467<br>0.338<br>0.340<br>Llama3 (8b)<br>0.745<br>0.723<br>0.771<br>0.730<br>0.639<br>0.697<br>GPT-3<br>0.539<br>0.520<br>0.420<br>0.482<br>0.312<br>0.289<br>GPT-3.5<br>0.735<br>0.710<br>0.598<br>0.600<br>0.484<br>0.487<br>GPT-4<br>0.776<br>0.786<br>0.878<br>0.910<br>0.739<br>0.763||
|LLM + KG Prompting<br>CoK<br>0.759<br>0.739<br>0.835<br>0.869<br>0.706<br>0.722<br>RoG<br>0.750<br>0.734<br>0.823<br>0.861<br>0.713<br>0.726<br>Mindmap<br>0.789<br>0.784<br>0.851<br>0.882<br>0.747<br>0.751||
||Ours<br>KnowGPT<br>**0.827**<br>**0.818**<br>**0.900**<br>**0.924**<br>**0.776**<br>**0.781**<br>KnowGPTvs. GPT-3.5 + 23.7% (Avg.)<br>+ 9.2%<br>+ 10.8%<br>+ 31.2%<br>+ 32.4%<br>+ 29.2%<br>+ 29.4%<br>KnowGPTvs. GPT-4<br>+2.9% (Avg.)<br>+ 5.1%<br>+ 3.3%<br>+ 2.2%<br>+ 1.4%<br>+ 3.7%<br>+ 1.8%|



*We used ‘text-davinci-002’ and ‘gpt-3.5-turbo’ provided by OpenAI as the implementation of GPT models. *The results compared with fine-tuning LLMs on CommonsenseQA are placed in Table 7 of Appendix. 

More importantly, despite being based on GPT-3.5, KnowGPT outperforms the state-of-the-art LLM GPT-4 by 3.3%, 1.4%, and 1.8% on the CommonsenseQA, OpenBookQA, and MedQA datasets, respectively. These results confirm that black-box knowledge injecting can effectively enhance the capabilities of LLMs. 

- KnowGPT outperforms all KG-enhanced LMs significantly. This implies our black-box knowledge injection method proficiently encodes knowledge into LLMs. Furthermore, it showcases the superiority of our black-box approach, given its adaptable application to GPT-3.5 using only the model API, a feat not achievable by white-box methods. 

## **4.2.1 Leaderboard Ranking** 

We submit our results onto the official leaderboard maintained by the authors of OpenbookQA. The full records on the leaderboard are shown on the website[1] , while our result can be found from here[2] . 

We summarize the related submissions in Table 2, including three categories: traditional KG-enhanced LM, fine-tuning of LLMs, e.g., T5-11B used in UnifiedQA, and ensemble of multiple predictors. KnowGPT significantly outperforms traditional KG-enhanced LMs with 5.2% improvements when compared to the best baseline. The third group of methods occupies the leaderboard by leveraging ensemble learning strategies. Nevertheless, KnowGPT can still obtain competitive performance without ensembling with 0.6% above GenMC Ensemble [22]. Notably, our KnowGPT is remarkably comparable to the human performance. 

## **4.3 Ablation Studies (RQ2)** 

To answer **RQ2** , we conduct two ablation studies. **First** , in Table 3, we measure the importance of the tailored reinforcement learning-based knowledge extraction module, i.e., _P_ RL. Specifically, we 

> 1 https://leaderboard.allenai.org/open_book_qa/submissions/public. 

> 2 https://leaderboard.allenai.org/open_book_qa/submission/cp743buq4uo7qe4e9750. 

8 

Table 2: OpenBookQA Official Leaderboard records of three groups of related models. 

|Table 2: OpenBookQA Offcial Leaderboard records of threegroups of related models.|Table 2: OpenBookQA Offcial Leaderboard records of threegroups of related models.|
|---|---|
|**Human Performance**(**0.917**)||
|w/o KG<br>0.778<br>MHGRN [13]<br>0.806<br>QA-GNN [69]<br>0.828<br>GreaseLM [75]<br>0.848<br>HamQA [10]<br>0.850<br>JointLK [53]<br>0.856<br>GSC [61]<br>0.874|UnifedQA [25]<br>0.872<br>DRAGON [68]<br>0.878<br>GenMC [22]<br>0.898<br>Human Performance<br>0.917<br>GenMC Ensemble [22]<br>0.920<br>X-Reasoner [21]<br>0.952<br>KnowGPT<br>**0.926**|



compare it with the heuristic sub-graph extraction strategy, i.e., _P_ sub. The performance is evaluated by directly feeding the extracted knowledge with the prompt format of ‘Sentence’, i.e., _Fs_ , to GPT-3.5. We also include ‘w/o KG’ as the baseline where GPT-3.5 is asked to independently answer the given question with no reasoning background provided. The results clearly indicate the vital role of our proposed knowledge extraction strategies. **Second** , we compare each of the three prompt formats subject to the same extracted knowledge. The detailed results are shown in Table 4. Though different formats perform similarly within the difference of 2.2% - 3.3%, they are particularly suitable for different kinds of questions. We illustrate this observation in the following case study section. Both ablation studies support the indispensability of each module, armed with a tailored deep reinforcement learning-based knowledge extraction and a context-aware prompt translation, our KnowGPT performs best on all three benchmark datasets. 

Table 3: Ablation study on the effectiveness of two knowledge extraction methods. 

Table 4: Ablation study on different prompt formats for the extracted knowledge. 

|**Knowledge Extraction**<br>**Model**|**CSQA**<br>**OBQA**<br>IHdev<br>IHtest<br>Test|**MedQA**<br>Test<br>0.289<br>0.487<br>0.763<br>0.695<br>0.755<br>**0.781**|**Knowledge Extraction**<br>**Prompts**|**CSQA**<br>**OBQA**<br>IHdev<br>IHtest<br>Test|**MedQA**<br>Test|
|---|---|---|---|---|---|
|w/o KG<br>GPT-3<br>GPT-3.5<br>GPT-4<br>_P_sub<br>GPT-3.5<br>_P_RL<br>GPT-3.5<br>Ours<br>KnowGPT|0.539<br>0.520<br>0.482<br>0.735<br>0.710<br>0.598<br>0.776<br>0.786<br>0.910||_P_sub<br>_Ft_<br>_Fs_<br>_Fg_|0.728<br>0.701<br>0.832<br>0.750<br>0.739<br>0.865<br>0.737<br>0.715<br>0.871|0.589<br>0.695<br>0.680|
||0.750<br>0.739<br>0.865||_P_RL<br>_Ft_<br>_Fs_<br>_Fg_|0.782<br>0.769<br>0.853<br>0.815<br>0.800<br>0.889<br>0.806<br>0.793<br>0.906|0.739<br>0.755<br>0.762|
||0.815<br>0.800<br>0.889|||||
||**0.827**<br>**0.818**<br>**0.924**||FullKnowGPT|**0.827**<br>**0.818**<br>**0.924**|**0.781**|



## **5 Limitation** 

Through our exploration, we realize the natural limitations of KnowGPT brought by real-world KGs. Existing KGs are automatically constructed based on online corpora. This inevitably introduces a considerable number of noisy triples into KGs. The noisy knowledge may mislead the LLMs to wrong predictions despite the effectiveness of our prompting methods. In the future, we would leverage KnowGPT with off-the-shelf KG refinement algorithms to improve the quality of KGs. 

## **6 Conclusion** 

The main objective of this paper is to tackle the hallucination issue that arises when applying LLM to specific domains. Although LLMs have strong reasoning capabilities, they still struggle to answer professional questions in specific domains, especially when the pre-training corpus lacks relevant knowledge. To address this issue, we propose a KG-augmented LLM model, named KnowGPT, which injects relevant domain knowledge from KGs into LLMs to assist the LLM in accurately answering professional questions. A novel framework, namely KnowGPT, is presented to integrate KGs into LLMs effectively with model APIs only. We first train a deep RL policy to extract informative and concise reasoning background from the KG. Then we learn an MAB to select the most effective knowledge extraction method and prompt template for each question. Extensive experiments on both general and domain-specific QA show superior performance of KnowGPT compared to all competitors. 

9 

## **References** 

- [1] Emily Alsentzer, John Murphy, William Boag, Wei-Hung Weng, Di Jindi, Tristan Naumann, and Matthew McDermott. Publicly available clinical bert embeddings. In _Proceedings of the 2nd Clinical Natural Language Processing Workshop_ , pages 72–78, 2019. 

- [2] Ilaria Amaro, Attilio Della Greca, Rita Francese, Genoveffa Tortora, and Cesare Tucci. Ai unreliable answers: A case study on chatgpt. In _ICHCI_ , pages 23–40. Springer, 2023. 

- [3] AI Anthropic. The claude 3 model family: Opus, sonnet, haiku. _Claude-3 Model Card_ , 2024. 

- [4] Olivier Bodenreider. The Unified Medical Language System (UMLS): integrating biomedical terminology. _Nucleic Acids Research_ , 32:D267–D270, 01 2004. 

- [5] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In _Proceedings of the 34th International Conference on Neural Information Processing Systems_ , pages 1877–1901, 2020. 

- [6] Ilias Chalkidis, Ion Androutsopoulos, and Nikolaos Aletras. Neural legal judgment prediction in english. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 4317–4323, 2019. 

- [7] Ilias Chalkidis, Abhik Jana, Dirk Hartung, Michael Bommarito, Ion Androutsopoulos, Daniel Martin Katz, and Nikolaos Aletras. Lexglue: A benchmark dataset for legal language understanding in english. _arXiv preprint arXiv:2110.00976_ , 2021. 

- [8] Yifang Chen, Chung-Wei Lee, Haipeng Luo, and Chen-Yu Wei. A new algorithm for nonstationary contextual bandits: Efficient, optimal and parameter-free. In _COLT_ . PMLR, 2019. 

- [9] Peter Clark, Oren Etzioni, Tushar Khot, Daniel Khashabi, Bhavana Mishra, Kyle Richardson, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord, Niket Tandon, et al. From ‘f’to ‘a’on the ny regents science exams: An overview of the aristo project. _AI Magazine_ , 41(4):39–53, 2020. 

- [10] Junnan Dong, Qinggang Zhang, Xiao Huang, Keyu Duan, Qiaoyu Tan, and Zhimeng Jiang. Hierarchy-aware multi-hop question answering over knowledge graphs. In _Proceedings of the ACM Web Conference 2023_ , pages 2519–2527, 2023. 

- [11] Junnan Dong, Qinggang Zhang, Xiao Huang, Qiaoyu Tan, Daochen Zha, and Zhao Zihao. Active ensemble learning for knowledge graph error detection. In _WSDM_ , pages 877–885, 2023. 

- [12] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General language model pretraining with autoregressive blank infilling. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 320–335, 2022. 

- [13] Yanlin Feng, Xinyue Chen, Bill Yuchen Lin, Peifeng Wang, Jun Yan, and Xiang Ren. Scalable multi-hop relational reasoning for knowledge-aware question answering. In _EMNLP_ , pages 1295–1309, 2020. 

- [14] Yasuhiro Fujita and Shin-ichi Maeda. Clipped action policy gradient. In _International Conference on Machine Learning_ , pages 1597–1606. PMLR, 2018. 

- [15] Yingqiang Ge, Wenyue Hua, Kai Mei, Juntao Tan, Shuyuan Xu, Zelong Li, Yongfeng Zhang, et al. Openagi: When llm meets domain experts. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [16] Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart, and Jonathan Herzig. Does fine-tuning llms on new knowledge encourage hallucinations? _arXiv preprint arXiv:2405.05904_ , 2024. 

- [17] Jocelyn Gravel, Madeleine D’Amours-Gravel, and Esli Osmanlliu. Learning to fake it: limited responses and fabricated references provided by chatgpt for medical questions. _Mayo Clinic Proceedings: Digital Health_ , 1(3):226–234, 2023. 

10 

- [18] Xinyan Guan, Yanjiang Liu, Hongyu Lin, Yaojie Lu, Ben He, Xianpei Han, and Le Sun. Mitigating large language model hallucinations via autonomous knowledge graph-based retrofitting. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pages 18126–18134, 2024. 

- [19] Suchin Gururangan, Ana Marasovi´c, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A Smith. Don’t stop pretraining: Adapt language models to domains and tasks. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 8342–8360, 2020. 

- [20] Linmei Hu, Zeyi Liu, Ziwang Zhao, Lei Hou, Liqiang Nie, and Juanzi Li. A survey of knowledge enhanced pre-trained language models. _IEEE Transactions on Knowledge and Data Engineering_ , 2023. 

- [21] Yongfeng Huang, Yanyang Li, Yichong Xu, Lin Zhang, Ruyi Gan, Jiaxing Zhang, and Liwei Wang. Mvp-tuning: Multi-view knowledge retrieval with prompt tuning for commonsense reasoning. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 13417–13432, 2023. 

- [22] Zixian Huang, Ao Wu, Jiaying Zhou, Yu Gu, Yue Zhao, and Gong Cheng. Clues before answers: Generation-enhanced multiple-choice qa. In _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 3272–3287, 2022. 

- [23] Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. _Applied Sciences_ , 11(14), 2021. 

- [24] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of NAACL-HLT_ , pages 4171–4186, 2019. 

- [25] Daniel Khashabi, Sewon Min, Tushar Khot, Ashish Sabharwal, Oyvind Tafjord, Peter Clark, and Hannaneh Hajishirzi. Unifiedqa: Crossing format boundaries with a single qa system. In _EMNLP 2020_ , pages 1896–1907, 2020. 

- [26] Tomasz Korbak, Hady Elsahar, German Kruszewski, and Marc Dymetman. Controlling conditional language models without catastrophic forgetting. In _International Conference on Machine Learning_ , pages 11499–11528. PMLR, 2022. 

- [27] Tiffany H Kung, Morgan Cheatham, Arielle Medenilla, Czarina Sillos, Lorie De Leon, Camille Elepaño, Maria Madriaga, Rimel Aggabao, Giezel Diaz-Candido, James Maningo, et al. Performance of chatgpt on usmle: Potential for ai-assisted medical education using large language models. _PLoS digital health_ , 2(2):e0000198, 2023. 

- [28] Jieh-Sheng Lee and Jieh Hsiang. Patent classification by fine-tuning bert language model. _World Patent Information_ , 61:101965, 2020. 

- [29] Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. Biobert: a pre-trained biomedical language representation model for biomedical text mining. _Bioinformatics_ , 36(4):1234–1240, 2020. 

- [30] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. _Advances in Neural Information Processing Systems_ , 33:9459–9474, 2020. 

- [31] Huayang Li, Yixuan Su, Deng Cai, Yan Wang, and Lemao Liu. A survey on retrieval-augmented text generation. _arXiv preprint arXiv:2202.01110_ , 2022. 

- [32] Qing Li, George Baciu, Jiannong Cao, Xiao Huang, Richard Chen Li, Peter HF Ng, Junnan Dong, Qinggang Zhang, Zackary PT Sin, and Yaowei Wang. Kcube: A knowledge graph university curriculum framework for student advising and career planning. In _International Conference on Blended Learning_ , pages 358–369. Springer, 2022. 

11 

- [33] Wentao Li, Huachi Zhou, Junnan Dong, Qinggang Zhang, Qing Li, George Baciu, Jiannong Cao, and Xiao Huang. Constructing low-redundant and high-accuracy knowledge graphs for education. In _International Conference on Web-Based Learning_ , pages 148–160. Springer, 2022. 

- [34] Bill Yuchen Lin, Xinyue Chen, Jamin Chen, and Xiang Ren. Kagnet: Knowledge-aware graph networks for commonsense reasoning. In _EMNLP-IJCNLP_ , pages 2829–2839, 2019. 

- [35] Fangyu Liu, Ehsan Shareghi, Zaiqiao Meng, Marco Basaldella, and Nigel Collier. Self-alignment pretraining for biomedical entity representations. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 4228–4238, 2021. 

- [36] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. _ACM Computing Surveys_ , 55(9):1–35, 2023. 

- [37] Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Qi Ju, Haotang Deng, and Ping Wang. K- bert: Enabling language representation with knowledge graph. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 34, pages 2901–2908, 2020. 

- [38] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692_ , 2019. 

- [39] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Reasoning on graphs: Faithful and interpretable large language model reasoning. _arXiv preprint arXiv:2310.01061_ , 2023. 

- [40] Donald Metzler, Yi Tay, Dara Bahri, and Marc Najork. Rethinking search: making domain experts out of dilettantes. In _Acm sigir forum_ , volume 55, pages 1–27. ACM New York, NY, USA, 2021. 

- [41] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In _EMNLP_ , pages 2381–2391, 2018. 

- [42] Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and Jianfeng Gao. Large language models: A survey. _arXiv preprint arXiv:2402.06196_ , 2024. 

- [43] OpenAI. Gpt-4 technical report, 2023. 

- [44] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. _NeurIPS_ , 35:27730–27744, 2022. 

- [45] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large language models and knowledge graphs: A roadmap. _IEEE Transactions on Knowledge and Data Engineering_ , 2024. 

- [46] Matthew E Peters, Mark Neumann, Robert Logan, Roy Schwartz, Vidur Joshi, Sameer Singh, and Noah A Smith. Knowledge enhanced contextual word representations. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 43–54, 2019. 

- [47] Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. Meta-learning with memory-augmented neural networks. In _International conference on machine learning_ , pages 1842–1850. PMLR, 2016. 

- [48] Xinyue Shen, Zeyuan Chen, Michael Backes, and Yang Zhang. In chatgpt we trust? measuring and characterizing the reliability of chatgpt. _arXiv preprint arXiv:2304.08979_ , 2023. 

12 

- [49] Chen Shengyuan, Yunfeng Cai, Huang Fang, Xiao Huang, and Mingming Sun. Differentiable neuro-symbolic reasoning on large-scale knowledge graphs. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [50] Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, Tharindu Kaluarachchi, Rajib Rana, and Suranga Nanayakkara. Improving the domain adaptation of retrieval augmented generation (rag) models for open domain question answering. _Transactions of the Association for Computational Linguistics_ , 11:1–17, 2023. 

- [51] Robyn Speer, Joshua Chin, and Catherine Havasi. Conceptnet 5.5: An open multilingual graph of general knowledge. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 31, 2017. 

- [52] Yu Sun, Shuohuan Wang, Shikun Feng, Siyu Ding, Chao Pang, Junyuan Shang, Jiaxiang Liu, Xuyi Chen, Yanbin Zhao, Yuxiang Lu, et al. Ernie 3.0: Large-scale knowledge enhanced pre-training for language understanding and generation. _arXiv preprint arXiv:2107.02137_ , 2021. 

- [53] Yueqing Sun, Qi Shi, Le Qi, and Yu Zhang. Jointlk: Joint reasoning with language models and knowledge graphs for commonsense question answering. In _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 5049–5060, 2022. 

- [54] Richard S Sutton and Andrew G Barto. _Reinforcement learning: An introduction_ . MIT press, 2018. 

- [55] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 4149–4158, 2019. 

- [56] Dhaval Taunk, Lakshya Khanna, Siri Venkata Pavan Kumar Kandru, Vasudeva Varma, Charu Sharma, and Makarand Tapaswi. Grapeqa: Graph augmentation and pruning to enhance question-answering. In _Companion Proceedings of the ACM Web Conference 2023_ , pages 1138–1144, 2023. 

- [57] InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities, 2023. 

- [58] Mina Valizadeh and Natalie Parde. The ai doctor is in: A survey of task-oriented dialogue systems for healthcare applications. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 6638–6660, 2022. 

- [59] Thomas J Walsh, István Szita, Carlos Diuk, and Michael L Littman. Exploring compact reinforcement-learning representations with linear regression. In _Proceedings of the TwentyFifth Conference on Uncertainty in Artificial Intelligence_ , pages 591–598, 2009. 

- [60] Jianing Wang, Qiushi Sun, Nuo Chen, Xiang Li, and Ming Gao. Boosting language models reasoning with chain-of-knowledge prompting. _arXiv preprint arXiv:2306.06427_ , 2023. 

- [61] Kuan Wang, Yuyu Zhang, Diyi Yang, Le Song, and Tao Qin. Gnn is a counter? revisiting gnn for question answering, 2021. 

- [62] Xiaozhi Wang, Tianyu Gao, Zhaocheng Zhu, Zhengyan Zhang, Zhiyuan Liu, Juanzi Li, and Jian Tang. Kepler: A unified model for knowledge embedding and pre-trained language representation. _Transactions of the Association for Computational Linguistics_ , 9:176–194, 2021. 

- [63] Yilin Wen, Zifeng Wang, and Jimeng Sun. Mindmap: Knowledge graph prompting sparks graph of thoughts in large language models. _arXiv preprint arXiv:2308.09729_ , 2023. 

13 

- [64] David S Wishart, Yannick D Feunang, An C Guo, Elvis J Lo, Ana Marcu, Jason R Grant, Tanvir Sajed, Daniel Johnson, Carin Li, Zinat Sayeeda, Nazanin Assempour, Ithayavani Iynkkaran, Yifeng Liu, Adam Maciejewski, Nicola Gale, Alex Wilson, Lucy Chin, Ryan Cummings, Diana Le, Allison Pon, Craig Knox, and Michael Wilson. DrugBank 5.0: a major update to the DrugBank database for 2018. _Nucleic Acids Research_ , 46:D1074–D1082, 2017. 

- [65] Wenhan Xiong, Thien Hoang, and William Yang Wang. Deeppath: A reinforcement learning method for knowledge graph reasoning. In _EMNLP_ , pages 564–573, 2017. 

- [66] Linyao Yang, Hongyang Chen, Zhao Li, Xiao Ding, and Xindong Wu. Give us the facts: Enhancing large language models with knowledge graphs for fact-aware language modeling. _IEEE Transactions on Knowledge and Data Engineering_ , 2024. 

- [67] Zhaorui Yang, Qian Liu, Tianyu Pang, Han Wang, Haozhe Feng, Minfeng Zhu, and Wei Chen. Self-distillation bridges distribution gap in language model fine-tuning. _arXiv preprint arXiv:2402.13669_ , 2024. 

- [68] Michihiro Yasunaga, Antoine Bosselut, Hongyu Ren, Xikun Zhang, Christopher D Manning, Percy S Liang, and Jure Leskovec. Deep bidirectional language-knowledge graph pretraining. _Advances in Neural Information Processing Systems_ , 35:37309–37323, 2022. 

- [69] Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. Qa-gnn: Reasoning with language models and knowledge graphs for question answering. _NAACL_ , 2021. 

- [70] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. Glm-130b: An open bilingual pre-trained model. _arXiv preprint arXiv:2210.02414_ , 2022. 

- [71] Daochen Zha, Zaid Pervaiz Bhat, Kwei-Herng Lai, Fan Yang, Zhimeng Jiang, Shaochen Zhong, and Xia Hu. Data-centric artificial intelligence: A survey. _arXiv preprint arXiv:2303.10158_ , 2023. 

- [72] Qinggang Zhang, Junnan Dong, Keyu Duan, Xiao Huang, Yezi Liu, and Linchuan Xu. Contrastive knowledge graph error detection. In _Proceedings of the 31st ACM International Conference on Information & Knowledge Management_ , pages 2590–2599, 2022. 

- [73] Qinggang Zhang, Junnan Dong, Qiaoyu Tan, and Xiao Huang. Integrating entity attributes for error-aware knowledge graph embedding. _IEEE Transactions on Knowledge and Data Engineering_ , 2023. 

- [74] Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga, Hongyu Ren, Percy Liang, Christopher D Manning, and Jure Leskovec. Greaselm: Graph reasoning enhanced language models for question answering. _arXiv preprint arXiv:2201.08860_ , 2022. 

- [75] Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga, Hongyu Ren, Percy Liang, Christopher D Manning, and Jure Leskovec. Greaselm: Graph reasoning enhanced language models for question answering. _arXiv preprint arXiv:2201.08860_ , 2022. 

- [76] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. _arXiv preprint arXiv:2303.18223_ , 2023. 

- [77] Haoxi Zhong, Chaojun Xiao, Cunchao Tu, Tianyang Zhang, Zhiyuan Liu, and Maosong Sun. How does nlp benefit legal system: A summary of legal artificial intelligence. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 5218–5230, 2020. 

- [78] Fengbin Zhu, Wenqiang Lei, Chao Wang, Jianming Zheng, Soujanya Poria, and Tat-Seng Chua. Retrieving and reading: A comprehensive survey on open-domain question answering. _arXiv preprint arXiv:2101.00774_ , 2021. 

14 

## **A Related Work** 

To address the hallucination issues that arise when applying LLMs to specific domains, researchers have proposed various methods, which can be broadly categorized into three main categories: FineTuning (FT), Retrieval-Augmented Generation (RAG) and KG-enhanced LLMs. 

## **A.1 Fine-tune LLMs with Domain-specific Data** 

To help the language model generate more accurate responses in specific domains, extensive studies have been conducted to investigate the efficacy of fine-tuning LLMs with specialized data [19; 28; 29; 15]. Fine-tuning involves training the pre-trained LLM on a smaller dataset of domainspecific text, allowing the model to adapt its knowledge to the target domain. By exposing the LLM to domain-specific vocabulary, terminology, and patterns, fine-tuning enables the model to generate more accurate and relevant responses for domain-specific tasks [19]. Fine-tuning LLMs with domain-specific data has been successfully applied across various domains. For example, in the healthcare domain, fine-tuned LLMs have been used for clinical note analysis [1], biomedical text mining [29], and medical dialogue [58]. Similarly, in the legal domain, fine-tuned LLMs have shown promise in tasks such as legal document classification [6], contract analysis [7], and legal judgment prediction [77]. 

However, recent research has highlighted the limitations and risks associated with fine-tuning LLMs using domain-specific data [26]. A study conducted by Google Research highlighted that using fine-tuning to update LLMs’ knowledge can be problematic, particularly when the new knowledge conflicts with pre-existing information [16]. In such cases, acquiring new data through FT can lead to the model generating new hallucinations and even experiencing catastrophic forgetting [26; 67]. Besides that, many state-of-the-art LLMs are confined to a _black-box_ role in practice. For instance, Calaude3 [3] and GPT-4 [43] exclusively grant access through their APIs, which means we can only retrieve model responses by submitting textual inputs, with model specifics inaccessible. This lack of access to model architecture and parameters prevents us from employing these fine-tuning methods. 

## **A.2 Retrieval-Augmented Generation for LLMs** 

Recently, Retrieval-Augmented Generation (RAG) models have been extensively explored to enhance LLMs with external knowledge from text corpora or online sources [30; 76; 42]. Combining LLMs with external knowledge retrieval systems can help reduce hallucinations. However, these approaches face challenges in domain-specific applications: ( _i_ ) Data quality. Domain knowledge is often scattered across different sources, such as textbooks, research papers, technical manuals, and industry reports [31]. These textual documents may have varying levels of quality, accuracy, and completeness, leading to potential inconsistencies or errors in the retrieved knowledge [78]. ( _ii_ ) Knowledge hierarchy. Domain knowledge is always complex and hierarchical and contains a high degree of jargon, acronyms, and specialized terminology. However, textual documents typically present this information in a linear and unstructured manner. The lack of explicit relationships and structured organization limits the reasoning and inference capabilities of RAG models, as they cannot leverage the structured connections within domain knowledge to derive new insights or generate more contextually appropriate responses [47]. ( _iii_ ) Huge search space. There is a lot of domain-irrelevant information in these textual documents, while domain-specific terminologies are always sparsely distributed over these documents [50]. The retrieval model can be computationally expensive and time-consuming, especially when dealing with large-scale knowledge sources, as the model needs to search through vast amounts of unstructured text to find relevant information [40]. 

## **A.3 Integration of KGs and LLMs** 

KG-enhanced LLM aims to leverage the structured knowledge in knowledge graphs [32; 33; 49; 72; 73] to ground the model’s responses in established facts [20; 66; 45]. This grounding ensures that the generated outputs are based on reliable information rather than arbitrary or fabricated statements. 

**Integrating KGs during Training** . Earlier studies adopted a heuristic way to inject knowledge from KGs into the LLMs during pre-training or fine-tuning. For example, ERNIE [52] incorporates entity embeddings and aligns them with word embeddings in the pre-training phase, encouraging the model 

15 

to better understand and reason over entities. KnowBERT [46] integrates entity linkers with BERT to introduce knowledge about entities during fine-tuning for downstream tasks. 

**Integrating KGs during Inference** . Another line of work focuses on retrieving relevant knowledge from KGs at inference time to augment the language model’s context. Typically, K-BERT [37] uses an attention mechanism to select relevant triples from a KG based on the input context, which are then appended to the input sequence. Similarly, KEPLER [62] learns joint embeddings of text and KG entities, enabling efficient retrieval of relevant knowledge to enhance the model’s predictions. Later works further integrated graph neural networks alongside LLMs for joint reasoning [69; 74; 10] and introduced interactions between text tokens and KG entities within the intermediate layers of LLMs [53; 56]. 

**KG Prompting for LLM** . The methods above all assume they know everything about LLMs, including model architectures and parameters. However, as LLMs have been keeping evolving, many SOTA LLMs are confined to a _black-box_ role in practice. As such, the research focus has shifted towards KG prompting that enhances fixed LLMs with KG-based hard prompts. KG Prompting for LLMs has been a new learning paradigm in natural language processing [36; 45]. Specifically, CoK [60] introduces a novel Chain-of-Knowledge prompting to decompose LLM-generated reasoning chains into evidence triples. It then verifies the triples’ factuality and faithfulness using an external knowledge graph, ensuring the accuracy and reliability of the LLM outputs. By allowing the models to comprehend and reason over structured KG inputs, Mindmap [63] enhances the LLMs’ ability to incorporate external knowledge, reduce hallucinations, and provide more transparency into their decision-making process. RoG [39] presents a planning-retrieval-reasoning framework that synergizes the strengths of LLMs and KGs to enhance the reasoning capabilities of language models in a more transparent and interpretable manner. Instead of retrieving factual information from KGs, KGR [18] proposes to autonomously retrofit the initial draft responses generated by LLMs. This innovative technique leverages the knowledge stored within KGs to mitigate hallucination by effectively extracting, verifying, and refining factual information throughout the entire reasoning process of LLMs. 

**Our model** . Despite the promising performance of existing KG prompting methods, their extensive deployment and practical implementation in domain-specific applications are still impeded by two critical issues. ❶ The cost associated with calling the LLM API or deploying LLMs with cloud services is prohibitive. For example, GPT-4 is estimated to cost at least thousands of dollars for pilot-scale customer service, making the careful selection of the most informative triples from KGs essential to minimize costs. ❷ LLMs are highly sensitive to prompts, with even minor variations in prompts conveying the same semantic meaning potentially yielding drastically different responses. However, existing methods rely on manually designed or rule-based prompts to present factual knowledge from KGs. These hard prompts are always fixed and rigid, lacking the flexibility to adapt to variations in question semantics and KG structures. In this paper, we shed light on the whole community with a novel prompt learning method that is rather effective, generalizable and cost-efficient. There are two key components in our proposed KnowGPT including (i) knowledge retrieval, which leverages deep RL to extract relevant knowledge from KGs, and (ii) context-aware decision module to translate the structured knowledge from KGs into the appropriate prompt format. 

## **B Implementation Details** 

## **B.1 Entity Linking and Heuristic Knowledge Extraction** 

For each QA context, we adopt the methodology outlined in the prior research [34; 68] to extract the subgraph from the background knowledge graph (KG), denoted as _G_ . We commence by executing entity linking on _G_ , resulting in an initial collection of nodes, _Vtopic_ . Next, we incorporate bridge entities that appear within a 2-hop path between any two linked entities from _Vtopic_ , yielding the set _Vretrieval_ . Subsequently, we refine this set by evaluating the relevance score for each node, adhering to the method proposed [68]. From this refined set, only the top 200 nodes, based on score, are retained. We then extract all edges connecting any pair of nodes in _Vsub_ , creating the retrieved subgraph _Gsub_ . Each node within _Gsub_ is designated a type based on its association to either the topic entities _Q_ or target entities _A_ . Intuitively, the relevant reasoning background lies in a question-specific subgraph _Gsub_ that contains all the _source_ entities _S_ , _target_ entities _A_ , and their 

16 

_k_ -hop neighbors. Therefore, the reasoning background could be provided as the _Gsub_ , we denote this direct knowledge extraction method as _Psub_ . 

## **B.2 Feature Initialization of Background KG** 

To calculate the initial node embeddings for entities extracted from the background KG, we adopt the approach proposed by the previous study [13]. Specifically, we transform knowledge triples from the KG into sentences and feed them into pre-trained LMs to get node embeddings. Specifically, to ensure a fair comparison, we implement all the KG-enhanced baselines and our model with the same advanced language models that are optimized for particular datasets. Specifically, RoBert-large [38] is used for CommenseQA, while AristoRoBERTa [9] is designated for OpenBookQA. For MedQA, we opt for the top-tier biomedical language model, SapBERT [35], to enhance comprehension of the biomedical field. 

## **B.3 Statistical Analysis of Datasets** 

We evaluate KnowGPT on three QA datasets spanning two fields: CommonsenseQA [55] and OpenBookQA [41] serve as benchmarks for commonsense reasoning, while MedQA-USMLE [23] acts as a domain-specific QA benchmark. While the official test set serves primarily for leaderboard rankings, we initially assess model efficacy using the in-house (IH) data split introduced in [34]. The official dataset is denoted as CSQA, while the IH split is represented by CSQA(IH)*. The statistics of these three datasets can be found in Table 5. 

Table 5: The statistical information of three datasets. 

|Dataset|Question|Choices|Train|Dev|Test|
|---|---|---|---|---|---|
|CSQA|#12102|5|9741|1221|1140|
|CSQA(IH)|#12102|5|8500|1221|1241|
|OBQA|#5957|4|4957|500|500|
|MedQA|#12723|4|10178|1272|1273|



**CommonsenseQA** is a multiple-choice question-answering dataset, each question accompanied by five potential answers. Answering its 12,102 questions necessitates a foundation in commonsense knowledge. While the official test set serves primarily for leaderboard rankings, we initially assess model efficacy using the in-house (IH) data split introduced in [34]. The official dataset is denoted as CSQA, while the IH split is represented by CSQA(IH)*. 

**OpenBookQA** , commonly abbreviated as _OBQA_ , comprises 5,957 multiple-choice questions, each offering four possible answers. To successfully answer these questions, one must have a comprehensive understanding of fundamental scientific facts and its applications. 

**MedQA-USMLE** , abbreviated as _MedQA_ , is a dataset consisting of 4-option multiple-choice questions that demand a grasp of biomedical and clinical understanding. These questions are sourced from preparatory tests for the United States Medical Licensing Examinations, and the dataset encompasses 12,723 questions. We adhere to the original data divisions as outlined in [23]. 

**Background Knowledge** To facilitate common sense reasoning, we employ ConceptNet [51], an extensive commonsense knowledge graph comprising more than 8 million interconnected entities through 34 concise relationships. For tasks specific to the medical domain, we leverage USMLE [69] as our foundational knowledge source. USMLE is a biomedical knowledge graph that amalgamates the Disease Database segment of the Unified Medical Language System (UMLS) [4] and DrugBank [64]. This repository encompasses 9,958 nodes and 44,561 edges. 

## **B.4 Implementation of Baselines** 

To verify the effectiveness of our proposed KnowGPT, we carefully selected baseline models from three aspects to ensure a comprehensive evaluation, among which Bert-base, Bert-large [24], and RoBert-large [38] are picked for being representative fine-tune LM methods; MHGRN [13], QAGNN [69], HamQA [10], JointLK [53], GreaseLM [75] and GrapeQA [56] represent the state-of-art KG-enhanced LMs; ChatGLM [12], ChatGLM2 [70], Baichuan-7B, InternLM [57], GPT-3 [5], 

17 

GPT-3.5 [44] and GPT-4 [43] are picked for being representative generative large language models. To verify the effectiveness of our proposed prompting strategy used KnowGPT, we also add the state-of-the-art KG prompting methods, i.e., CoK [60], RoG [39], and Mindmap [63] as baselines. Notably, while some LLM baselines are actually open-source, we conduct the question-answering task under the zero-shot setting with the question query from the test set as input. All baseline methods used in this paper are based on their open-source implementations or officially-released APIs. Notably, we used the model ‘text-davinci-002’ provided by OpenAI as the implementation of GPT-3, and ‘gpt-3.5-turbo’ and ‘gpt-4’ as the implementations of GPT-3.5 and GPT-4, respectively. All baseline methods used in this paper are based on their open-source implementations. We list the source links in Table 6. All models are implemented in Pytorch and trained on an RTX 3090 with 24 RAM. We use their best configuration as the default value for other distinctive hyperparameters in the baseline methods. To reduce the randomness, we use the random seed and report the average results of three runs. 

Table 6: Implementation codes for baselines. 

||Table 6: Implementation codes for baselines.|
|---|---|
|Baseline|Code source|
|MHGRN|https://github.com/INK-USC/MHGRN.git|
|QA-GNN|https://github.com/michiyasunaga/qagnn.git|
|HamQA|https://github.com/DEEP-PolyU/HamQA_TheWebConf23.git|
|GreaseLM|https://github.com/snap-stanford/GreaseLM.git|
|JointLK|https://github.com/Yueqing-Sun/JointLK.git|
|ChatGLM|https://github.com/THUDM/ChatGLM-6B.git|
|ChatGLM2|https://github.com/THUDM/ChatGLM2-6B.git|
|Baichuan-7B|https://github.com/baichuan-inc/Baichuan-7B.git|
|InternLM|https://github.com/InternLM/InternLM.git|
|GPT3|https://platform.openai.com/docs/api-reference/models|
|GPT-3.5|https://openai.com/GPT-3.5|
|GPT-4|https://openai.com/research/gpt-4|
|RoG|https://github.com/RManLuo/reasoning-on-graphs|
|MindMap|https://github.com/wyl-willing/MindMap|



## **C Supplementary Results** 

## **C.1 Compared to Fine-tune LLMs** 

To further verify the effectiveness of the knowledge injection framework, we also add several open-source trainable LLMs, i.e., ChatGLM, ChatGLM2, LLaMA-B, Baichuan-7B, InternLM and Vicuna-7B, and fine-tune them for commonsense reasoning on the benchmark CommonseQA. As shown in the Table 7, Our KnowGPT achieves comparable performance with no tuning on the LLM. 

Table 7: Fine-tuned LLMs on three benchmark datasets. 

|**LLM**|**CommonsenseQA**|**OpenBookQA**|**MedQA**|
|---|---|---|---|
|ChatGLM|55.9%|54.2%|40.4%|
|ChatGLM2|60.0%|55.6%|45.5%|
|LLaMA-7B|65.0%|58.8%|46.9%|
|Baichuan-7B|58.8%|56.1%|43.4%|
|Alpaca-7B|68.7%|63.3%|47.1%|
|Vicuna-7B|66.7%|64.0%|45.2%|
|InternLM-7B|75.2%|70.7%|48.3%|
|KnowGPT|**81.8%**|**92.4%**|**78.1%**|



18 

## **C.2 Case Studies (RQ3)** 

For **RQ3** , we provide insights into how KnowGPT facilitates the prompt translation with a real case from CommonsenseQA. We visualize both the extracted knowledge and the textual inputs to GPT-3.5 in Figure 3. In this example, given the same extracted knowledge, GPT-3.5 answers correctly based on the sentence format that we provide. In contrast, it fails to answer the question with triples and graph descriptions. They clearly indicate the superiority of KnowGPT in an automatic context-aware prompt translation. We make the following observations: ( _i_ ) Triple format _Ft_ is intuitively suitable for all the simple questions by directly indicating the one-hop knowledge. ( _ii_ ) Graph description may inevitably introduce noise to ensure the completeness and contextual fluency of the directed graph. In this example, since ‘vacation’ appears in both question and answer choices, over-emphasizing and connecting the knowledge about ‘vacation’ with other concepts in the graph misleads the model to make a prediction with an oblique focus. ( _iii_ ) Our KnowGPT has shown superior performance in automatically constructing suitable prompts for particular questions. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/graphGPT/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models_images/nips_2024_KnowGPT_Knowledge_Graph_based_Prompting_for_Large_Language_Models.pdf-0019-02.png)


**----- Start of picture text -----**<br>
David watched some nesting birds using his binoculars while on vacation. Where might David be?<br>A.sky B.vacation C.forest  D.countryside<br>jungle<br>birds trees vision_aid<br>build_nest rainforest binoculars<br>mammal nests forest visit_other_countries sports_store<br>sky vacation<br>nesting<br>action_of_animals<br>ℱ𝓉 (nesting isa action_of_animals) ℱ� Nesting, an instinctual action  Graph ℱ� This graph highlights the<br>Triples (bird relatedto nesting) Sentences  performed by various animals,  Description intricate relationships<br>(nesting relatedto nest) including birds, involves inhabiting  between birds, their nesting<br>(bird atlocation nest) and crafting intricate structures  behaviors, natural habitats like trees<br>(bird capableof build_nest) known as nests. Birds, closely  and forests, the sky, and the tools<br>(Bird isa mammal) related to this behavior, are often  that aid in the observation of these<br>(bird relatedto trees) spotted at their nests, where they are  captivating avian activities.  The<br>(nest atlocation trees) capable of building and preparing  graph revolves around the central<br>(tree atlocation jungle) them for raising their young. These  concept of a "bird" which is closely<br>feathered creatures have a natural  connected to various aspects of<br>(tree atlocation rainforest)<br>(tree partof forest) affinity for trees and are often found  nesting behavior.<br>in tree nests, which are nestled  […]<br>(telescope relatedto binoculars) within lush jungles and rainforests.  This aerial environment is also<br>(binoculars atlocation sports_store)<br>During a relaxing vacation, one can  associated with outdoor settings.<br>(binoculars isa vision_aid)<br>observe birds engaging in nesting  Adding depth to the observation of<br>(vacation causes visit_other_countries)<br>activities, taking advantage of  birds, binoculars are presented as a<br>(bird atlocation sky) binoculars as a vision aid to witness  vision aid that allows closer scrutiny<br>(birds atlocation sky) their behavior in the expansive sky, a  of their nesting behavior, both in<br>(flying_birds atlocation sky) realm intertwined with the beauty of  natural locations like backpacks,<br>the outdoors. sporting goods stores, and suitcases,<br>and during the context of a vacation.<br>vacation forest vacation<br>**----- End of picture text -----**<br>


Figure 3: A case study on exploring the effectiveness of different prompt formats for particular questions. The extracted knowledge is shown in the middle of this figure in the form of a graph, where the nodes in blue are the key topic entities and the red is the target answer. The text boxes at the bottom are the final prompts generated based on three different formats. 

## **C.3 The effect of prompt format on different types of questions.** 

Generally, based on the complexity of the reasoning background, questions can be roughly categorized into three classes: (i) Simple question, like “What could be used as an electrical conductor?”. (ii) 

19 

Multi-hop reasoning question, like “Which school did Bill Gates’ wife graduate from?” (iii) Global reasoning questions: “What do cats have in common with most mammals?” 

Different types of questions correspond to different reasoning background. For example, simple questions only require basic factual triples, while multi-hop reasoning questions need a reasoning chain, and global reasoning questions require a more complex tree/graph-like reasoning background. To convert the extracted raw knowledge into textual prompt with corresponding logical reasoning structure, we designed three different prompt templates, including _Ft_ , _Fs_ and _Fg_ . 

To verify the effectiveness of prompt formats, we conducted a statistical analysis on the benchmark datasets in terms of question types and then separately calculated the accuracy of different prompt formats on specific types of questions. As shown in Table 8, we can observe that triple-based prompt performs best on simple questions while graph description-based prompt performs better than any other prompt formats on complex questions. This is because graph description-based prompt could provide LLMs with more detailed and structured information by highlighting the local structure of the central entity. 

Table 8: Accuracy of different prompt formats on specific types of questions on CommonsenseQA. 

|**Prompt Format**|**Simple**|**Multi-hop**|**Graph reasoning**|
|---|---|---|---|
|_Ft_|**94.12%**|74.47%|42.10%|
|_Fs_|88.23%|**82.98%**|47.39%|
|_Fg_|85.29%|70.21%|**78.95%**|



In this part, we conduct comprehensive experiments to investigate the effect of prompt format on different types of questions. Table 8 presents the accuracy of different prompt formats on three types of questions. We observe that graph description-based prompt performs significantly better than any other prompt formats on complex questions. It is because graph description-based prompt could provide LLMs with more detailed and structured information by highlighting the local structure of the central entity. 

## **C.4 The Effect of two Knowledge Retrieval Methods** 

If the background knowledge graph is complete and high-quality, our proposed RL-based knowledge extraction can ideally handle all cases. But in pratical implementation, we found that there are a few long-tail entities in KGs which only have few neighbors. In such cases, RL-based knowledge extraction can hardly retrieve reachable or effective paths due to the incompletion and sparsity of the background graph. Instead, directly extracting the whole subgraph through rule-based subgraph extraction can provide more knowledge for reasoning. The RL-based retrieval method _PRL_ could make sure our model extracts more concise and informative knowledge in most cases, while the rule-based subgraph extraction _PRL_ could supplement the former especially when the reasoning background is sparse. Thus, in this paper, we adopt Multi-Armed Bandit (MAB) to automatically select the most suitable retrieval methods from these two candidates, i.e., _PRL_ and _Psub_ . The ablation study on the knowledge retrieval method can be found in Table 3 of the main content. Specifically, we compare our model with _PRL_ , the heuristic sub-graph extraction _Psub_ , and we also include ‘w/o KG’ as the baseline where the LLM is asked to independently answer the given question with no reasoning background provided. The results shown in Table 3 of our submission clearly indicate the vital role of our proposed knowledge extraction strategies. 

## **C.5 Efficiency Analysis: API Cost and Model Efficiency** 

To investigate the efficiency of our model, we compare it on MedQA with six representative baselines from three different categories, including traditional KG-enhanced LMs, i.e., JointLK and GrapQA, zero-shot LLMs, i.e., ChatGPT and GPT-4, and the SOTA KG-prompting-based LLMs, i.e., CoK and Mindmap. Notably, zero-shot LLMs, like ChatGPT and GPT-4, do not require training while traditional KG-enhanced LM methods like JointLK and GrapeQA are based on lightweight LMs and do not require interaction with LLMs, thus they do not incur API cost. 

From Table 9, we can see that (i) Traditional KG-enhanced LM methods, like JointLK and GrapQA, have the shortest inference time. This is because the other models need to send requests to the Black-box LLM via API, and the extra response time of the LLM leads to long inference times for 

20 

these models. Despite the efficiency, they have the worst performance. (ii) Our model outperforms the other models with comparable training time and the most economical API cost compared to models in the same category, including CoK and Mindmap. That is because we are the only model that consider the conciseness of knowledge during retrieval and prompt design. 

Table 9: API cost and model efficiency analysis on MedQA. 

|Model|Training|time|Inference time|Avg tokens|Cost($)|Performance|
|---|---|---|---|---|---|---|
|JointLK|4.02|h|0.13 h|-|-|40.3%|
|GrapeQA|4.49|h|0.15 h|-|-|39.5%|
|ChatGPT|-||0.24 h|98|1.87|48.7%|
|GPT-4|-||0.21 h|98|29.77|76.3%|
|CoK|5.85|h|0.44 h|1129|21.54|72.2%|
|Mindmap|4.93|h|0.36 h|761|14.52|75.1%|
|KnowGPT|5.47|h|0.33 h|348|6.64|78.1%|



## **C.6 Stability Analysis** 

Due to page limits, we carefully described how we designed the reward function to ensure the RL model captures effective and informative knowledge in the main content of our submission. Actually, we have also employed the following techniques to stabilize the training process. 

**Gradient clipping for policy network** . The policy network is updated using gradient-based optimization to maximize the expected cumulative reward. To stabilize the training of the policy network, we applied gradient clipping [14] to prevent the gradients from becoming too large. This technique helps mitigate the problem of exploding gradients and ensures stable updates of the network parameters. To demonstrate the effectiveness of gradient clipping, we conducted additional experiments comparing the performance of our model with and without gradient clipping across five runs. The results are presented in the table below. As shown in Table 10, the model with gradient clipping consistently achieves better performance across all datasets, demonstrating the stability and robustness of our method. 

**Exploration-exploitation balance for MAB.** In our designed MAB, we have introduced a penalty term, i.e., _β_ , to encourage exploration on under-explored arms (i.e., prompt formats), otherwise, it will lead to a greedy selection based on _c × α_ ( _i_ ). Table 11 shows the performance comparison between with and without this term. 

Table 10: Stability study on gradient clipping. Table 11: Stability analysis on penalty term. 

|Model Variant|CSQA|OBQA|MedQA|Model Variant|CSQA|OBQA|MedQA|
|---|---|---|---|---|---|---|---|
|w. gradient clipping|81.8_±_0.51|92.4_±_0.69|78.1_±_0.66|w. penalty term|81.8_±_0.51|92.4_±_0.69|78.1_±_0.66|
|w/o gradient clipping|78.9_±_1.92|90.2_±_1.25|75.5_±_2.01|w/o penalty term|79.5_±_1.63|91.0_±_1.45|76.4_±_1.89|



## **D Broader Impacts** 

We are dedicated to not only focusing on the specific NLP task, i.e., knowledge-based QA, but also inspiring and benefitting a wide range of NLP communities with a novel knowledge injection framework to fast adapt LLMs to specific domains. Specifically, we would like to declare the following two contributions for a wide range of NLP communities. 

**Exploring Effective Knowledge Injection for LLMs** : Applying LLMs for various NLP-related tasks in different domains is gaining momentum. However, fine-tuning LLMs for domain knowledgebased QA is expensive. Our paper presents a remarkably strong framework that could (i) retrieve the reasoning background from domain knowledge graph and (ii) automatically find out the optimal prompt to inject the domain knowledge into the LLM. 

**Providing Insights in Prompt Learning** : While hard prompt is still dominating prompt learning for the majority of LLM-based scenarios, we shed light on the community with a novel prompt learning method that is rather generalizable and effective. We propose an auto-selection algorithm that can determine the best prompt format for LLMs to maximally understand the reasoning background. 

21 

