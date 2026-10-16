# **A Bipartite Graph is All We Need for Enhancing Emotional Reasoning with Commonsense Knowledge** 

## Kailai Yang 

The University of Manchester Manchester, United Kingdom kailai.yang@postgrad.manchester.ac.uk 

Shaoxiong Ji University of Helsinki Helsinki, Finland shaoxiong.ji@helsinki.fi 

## **ABSTRACT** 

The context-aware emotional reasoning ability of AI systems, especially in conversations, is of vital importance in applications such as online opinion mining from social media and empathetic dialogue systems. Due to the implicit nature of conveying emotions in many scenarios, commonsense knowledge is widely utilized to enrich utterance semantics and enhance conversation modeling. However, most previous knowledge infusion methods perform empirical knowledge filtering and design highly customized architectures for knowledge interaction with the utterances, which can discard useful knowledge aspects and limit their generalizability to different knowledge sources. Based on these observations, we propose a **B** ipartite **H** eterogeneous **G** raph (BHG) method for enhancing emotional reasoning with commonsense knowledge. In BHG, the extracted context-aware utterance representations and knowledge representations are modeled as heterogeneous nodes. Two more knowledge aggregation node types are proposed to perform automatic knowledge filtering and interaction. BHG-based knowledge infusion can be directly generalized to multi-type and multi-grained knowledge sources. In addition, we propose a Multidimensional Heterogeneous Graph Transformer (MHGT) to perform graph reasoning, which can retain unchanged feature spaces and unequal dimensions for heterogeneous node types during inference to prevent unnecessary loss of information. Experiments show that BHG-based methods significantly outperform state-of-the-art knowledge infusion methods and show generalized knowledge infusion ability with higher efficiency. Further analysis proves that previous empirical knowledge filtering methods do not guarantee to provide the most useful knowledge information. Our code is available at: https://github.com/SteveKGYang/BHG. 

## **KEYWORDS** 

emotion recognition in conversations, casual emotion entailment, knowledge infusion, bipartite heterogeneous graph 

Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for third-party components of this work must be honored. For all other uses, contact the owner/author(s). _CIKM ’23, October 21–25, 2023, Birmingham, United Kingdom_ © 2023 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-0124-5/23/10. https://doi.org/10.1145/3583780.3614758 

## Tianlin Zhang 

The University of Manchester Manchester, United Kingdom tianlin.zhang@postgrad.manchester.ac.uk 

## Sophia Ananiadou 

The University of Manchester Manchester, United Kingdom sophia.ananiadou@manchester.ac.uk 

## **ACM Reference Format:** 

Kailai Yang, Tianlin Zhang, Shaoxiong Ji, and Sophia Ananiadou. 2023. A Bipartite Graph is All We Need for Enhancing Emotional Reasoning with Commonsense Knowledge. In _Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM ’23), October 21–25, 2023, Birmingham, United Kingdom._ ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3583780.3614758 

## **1 INTRODUCTION** 

Understanding human emotions is at the core of affective computing. In natural language processing, recent years have witnessed growing research interests in machine’s context-aware emotional reasoning ability, especially in conversations, due to its vital importance in scenarios such as empathetic dialogue systems [17] and online opinion mining from social media [4]. This goal is mostly specified as recognizing the emotion [23] or the emotion cause [21] of certain utterances within a conversation. 

There are two key challenges for enhancing conversational emotional reasoning. Firstly, the emotion of the target speaker is influenced by both his own mental state and other participants’ behaviors. Current methods mainly build conversation models [27, 28, 38] based on Pre-trained Language Models (PLMs) [15, 41] to tackle these dependencies. Secondly, emotions are often conveyed implicitly with metaphor, sarcasm, and underlying common sense. To mine the related information, a mainstream solution infuses commonsense knowledge to provide emotional clues and help model the inter-utterance relations, which mostly follows a three-step paradigm. We provide an illustration in Figure 1 and summarize this paradigm as follows: Firstly, _**knowledge extraction**_ obtains commonsense knowledge items with the target conversation as queries. This process is closely dependent on the granularity of the queries and the characteristics of the knowledge source. As shown in the example, with the utterance-level query "Yeah, jogging with Sally!" and the generative knowledge source COMET [2], we can extract sentence-level social commonsense knowledge such as "PersonX is seen as active" and "As a result, others feel excited". Secondly, _**knowledge filtering**_ selects the most relevant knowledge aspects from the extracted knowledge items based on task-specific priors. The most widely adopted methods include rule-based [6, 44, 47] and distantly supervised [13, 37] filtering. Finally, _**knowledge interaction**_ introduces the filtered knowledge to the conversation model via customized architectures to enhance its emotional reasoning 

CIKM ’23, October 21–25, 2023, Birmingham, United Kingdom 

Kailai Yang, Tianlin Zhang, Shaoxiong Ji, and Sophia Ananiadou 

ability. For example, a typical architecture [12, 13] assigns knowledge aspects to the edges of the conversation graphs and utilizes the corresponding knowledge features as edge representations to model the inter-utterance dependencies. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge_images/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge.pdf-0002-03.png)


**----- Start of picture text -----**<br>
1. Knowledge extraction 2. Knowledge filtering<br>xIntent<br>PersonX is  xAttr … oWant<br>As a result,  seen as active.<br>PersonX wants to  xReact oReact<br>take a shower. xAttr … xWant<br>Other methods<br>gym. That’ll be a lot of #1: We can go to the Dialogue Flowfun! xWant PersonX feels As a result, happy.xReact Rules for Selection:#1: Select xIntent#3: Select oReact#2: Select xReact …<br>#2: I love jogging after<br>lunch, it relaxes me.<br>Other sources xIntent xReact oReact<br>#3: Yeah, jogging with<br>Sally!  COMET Other Models<br>#4. Hmm, we can go  … Utterance representations Li et al., 2021<br>with Mary, she often go<br>to the gym … oWant xWant<br>oReact<br>#5. Alright, will you  xIntent xEffect<br>turn off the radio first? That’s too much noise! xIntent As a result, others feel<br>oWant excited.<br>Because  … #1 #2 #3 #4<br>PersonX wanted to be healthy. others want As a result,  …<br>to go home. 3. Knowledge interaction<br>**----- End of picture text -----**<br>


**Figure 1: Illustration of the three-step paradigm of knowledge infusion. An example based on the knowledge source COMET is provided for each step.** 

In this paper, we dive into the knowledge filtering and knowledge interaction steps and raise the following research questions (RQ): 

- **RQ1** : Most knowledge filtering methods select knowledge aspects empirically without further evaluations. What knowledge aspects are most effective in enhancing conversational emotional reasoning? 

- **RQ2** : Current knowledge interaction methods are highly coupled with both the knowledge sources and conversation models. Are these customized architectures necessary? 

In addressing these research questions, we propose a simple yet effective **B** ipartite **H** eterogeneous **G** raph (BHG) method for enhancing emotional reasoning with commonsense knowledge. Firstly, we extract context-aware utterance representations via a PLM-based conversation model and relevant commonsense knowledge from three multi-type and multi-grained knowledge sources. Considering the complementary nature [40] between utterance semantics and related common sense, we model the extracted utterance and knowledge representations as heterogeneous nodes. In addition, we introduce a forward and a backward knowledge aggregation node type to perform automatic knowledge filtering and knowledge interaction, because for each target utterance, effective knowledge aspects for modeling inter-utterance relations are usually different in the past context and future context along the dialogue flow, as shown in previous works [12, 13, 44]. Then a bipartite graph is built on these four heterogeneous node types, where messages from the utterance and knowledge are passed to the knowledge aggregation nodes for semantic-aware knowledge filtering, and the aggregated knowledge messages interact with the utterance nodes to enrich their semantics and model inter-utterance dependencies. The BHG is decoupled from the conversation models and knowledge 

sources and enables a unified model architecture for multi-type and multi-grained knowledge infusion. Its simple bipartite structure also facilitates the graph reasoning process. 

In a BHG, heterogeneous nodes usually possess different feature spaces with unequal dimensions as conversation models/knowledge sources change. During the graph reasoning process, existing heterogeneous graph neural networks [8, 26, 32] mostly project all types of nodes into a unified feature space to facilitate the interaction between neighbors. However, the projections disrupt the original feature spaces and can lead to unnecessary loss of information for high-dimensional node types. Based on the heterogeneous graph Transformer [8], we propose a Multi-dimensional Heterogeneous Graph Transformer (MHGT) for graph reasoning, which utilizes a multi-dimensional edge-dependent matrix to enable direct attention calculation and message passing between heterogeneous nodes with different feature dimensions. MHGT allows all node types to retain the original feature spaces and potentially useful information for the knowledge filtering and knowledge interaction processes during inference. 

We evaluate the effectiveness of our proposed BHG and MHGT methods on five datasets across two conversational emotional reasoning tasks: Emotion Recognition in Conversations (ERC) and Casual Emotion Entailment (CEE). The experimental results show that the BHG-based methods outperform previous state-of-the-art knowledge infusion models, and show generalized knowledge infusion ability on multi-type and multi-grained knowledge sources with higher efficiency than previous customized methods. We also analyze the effectiveness of different knowledge aspects, and the results show that previous empirical knowledge filtering methods can introduce less useful knowledge aspects and discard knowledge aspects that benefit the emotional reasoning process. 

In summary, this paper makes the following contributions: (1) we propose a bipartite heterogeneous graph-based method to enhance emotional reasoning with commonsense knowledge, which enables a unified framework for multi-type and multi-grained knowledge infusion; (2) we propose a multi-dimensional heterogeneous graph Transformer for graph reasoning, which allows unchanged feature spaces and unequal dimensions for heterogeneous node representations during inference; (3) the BHG-based methods significantly outperform state-of-the-art knowledge infusion methods and show generalized knowledge infusion ability with higher efficiency. 

## **2 METHODOLOGY** 

We introduce the utterance and knowledge feature extraction process in Sec. 2.1. Then the BHG construction process and the multidimensional HGT methods are introduced in Sec. 2.2 and 2.3. Finally, the examined tasks and prediction process are described in Sec. 2.4. 

## **2.1 Feature Extraction** 

In this section, we extract the utterance features via a PLM-based conversation model. We also extract multi-type (generative and extractive knowledge) and multi-grained (utterance-level and phraselevel) commonsense knowledge to enhance the emotional reasoning process. These processes are illustrated in Fig. 2 (a). 

_2.1.1 Conversation Model._ We decouple conversation modeling from knowledge infusion by utilizing existing conversation models 

CIKM ’23, October 21–25, 2023, Birmingham, United Kingdom 

A Bipartite Graph is All We Need for Enhancing Emotional Reasoning with Commonsense Knowledge 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge_images/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge.pdf-0003-02.png)


**----- Start of picture text -----**<br>
(a). Feature Extraction (b). Bipartite Heterogeneous Graph Construction<br>h1 h2 … hi … hn Dialogue flow<br>Wb : 3 Wf : 2<br>h1 … hi-3 hi-2 hi-1 hi hi+1 hi+2 … hn<br>Conversation Model<br>Dialogue flow:u1 Linked Data u2 Web API … u xIntent i xReact … un 1 … b1i-2 i-bf1i-2 bi-1fi-2 i fi-1 bi fi bi+1 bi+2fi+1bni+1fi+2 fi+2n … n<br>ConceptNet Knowledge Source Commonsense COMET oReact k1,1 kk1,2i-2,1 … ki-1,1ki-2,2k1,mk … i-1,2kk … i-2,mi,1 ki-1,mki,2 k … i+1,1kki,mki+2,1i+1,2kn,1k … i+2,2kki+1,mn,2 … k … i+2,mkn,m<br>k12k11k13 k21 k22 … kki1i3 kki2i4 … kkn1n3 kkn2n4 kk bf Forward know.informationBackward know.information hh bf Forward uttr.information.Backward uttr.information. bf hh Forward know.infusionBackward know.infusion<br>**----- End of picture text -----**<br>


**Figure 2: An overview of the utterance/knowledge feature extraction and BHG construction processes. In (b), the graph construction process for the** _𝑖_ **-th utterance is presented.** _𝑊𝑓_ = 2 **and** _𝑊𝑏_ = 3 **are examples of the forward and backward knowledge infusion window sizes.** _**know.**_ **and** _**uttr.**_ **denote "knowledge" and "utterance".** 

to obtain context-aware utterance representations. We utilize a PLM-based conversation model. For a conversation D = [ _𝑢_ 1, _𝑢_ 2, ..., _𝑢𝑖_ , ..., _𝑢𝑁_ ] with _𝑁_ utterances, _𝑢𝑖_ is the target utterance pre-pended with its speaker. The model concatenates both past and future contexts as the input ˆ _𝑢𝑖_ = || _𝑗_ ∈[1 _, 𝑁_ ] _𝑢 𝑗_ , where || is the concatenation operation. Considering situations where only the dialogue history is available, we also test our methods on only past contexts: _𝑢_ ˆ _𝑖_ = || _𝑗_ ∈[1 _, 𝑖_ ] _𝑢 𝑗_ . We use a RoBERTa-Large [15] encoder to model the dialogue: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge_images/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge.pdf-0003-05.png)


> where _ℎ_[ˆ] _𝑖_ ∈ R _[𝑁][𝑡]_[×] _[𝐷][ℎ]_ denotes the token-level representations, _𝑁𝑡_ is the token number of _𝑢_ ˆ _𝑖_ , and _𝐷ℎ_ denotes the dimension of hidden states. We further obtain the utterance-level context-aware representation _ℎ 𝑗_ for each utterance _𝑢 𝑗_ within the dialogue flow via mean-pooling: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge_images/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge.pdf-0003-07.png)


where _𝑙 𝑗_ and _𝑟 𝑗_ denote the start and end positions of _𝑢 𝑗_ in _𝑢_ ˆ _𝑖_ . 

_2.1.2 Knowledge Extraction._ In previous works, two commonsense knowledge graphs have been proven most effective in enhancing the emotional reasoning process: the social commonsense knowledge graph ATOMIC [25] and the taxonomic/lexical knowledge graph ConceptNet [29]. Therefore, we separately obtain relevant knowledge from three knowledge sources expanded from ATOMIC and ConceptNet. 

With the development of automatic knowledge graph construction, generative knowledge sources receive increasing interest due to their flexibility and convenience in knowledge extraction. We extract the ATOMIC knowledge via two COMET models: the first one is COMET2019 [2], which pre-trains a GPT [24] model on ATOMIC. During knowledge extraction, each utterance _𝑢𝑖_ is constructed into the query: ( _𝑢𝑖_ || _𝑎 𝑗_ ||[ _𝐺𝐸𝑁_ ]). _𝑎 𝑗_ denotes a knowledge aspects term representing an if-then relation type for the speaker’s actions/mental states. COMET2019 provides nine knowledge aspects from ATOMIC, where their interpretations are listed in Table 1. The 

constructed query is then input to the COMET model, and the finallayer hidden representation of the decoder: _𝑘𝑖𝑗_ ∈ R _[𝐷] 𝑘_[2019] is used as the utterance-level knowledge representation corresponding to _𝑎 𝑗_ , where _𝐷𝑘_[2019] denotes the dimension of _𝑘𝑖𝑗_ . 

**Table 1: Interpretations of the selected knowledge aspects/relations. "X" denotes the target utterance speaker.** 

||**COMET**|
|---|---|
|**Aspect**|**Interpretation**|
|_xIntent_|Why does X cause the event?|
|_xAttr_|How would X be described?|
|_xNeed_|What does X need to do before the event?|
|_xWant_|What would X likely want to do after the event?|
|_xEfect_|What efects does the event have on X?|
|_xReact_|How does X feel after the event?|
|_oWant_|What would others likely want to do after the event?|
|_oEfect_|What efects does the event have on others?|
|_oReact_|How do others feel after the event?|
||**ConceptNet**|
|**Relation**|**Interpretation**|
|_IsA_|is a|
|_HasProperty_|has the property of|
|_DefnedAs_|is defned as|
|_AtLocation_|is located at|
|_RelatedTo_|is related to|



The second knowledge source COMET2020[9] further extends the generative pre-training to an expanded ATOMIC knowledge graph and part of ConceptNet based on a larger BART [11] model. COMET2020 follows a similar knowledge extraction process as COMET2019 and each knowledge representation has dimension _𝐷𝑘_[2020] . 

CIKM ’23, October 21–25, 2023, Birmingham, United Kingdom 

Kailai Yang, Tianlin Zhang, Shaoxiong Ji, and Sophia Ananiadou 

On the other hand, we obtain phrase-level ConceptNet knowledge in an extractive manner. Specifically, we tokenize each utterance _𝑢𝑖_ and concatenate the tokens into n-gram phrases. For each phrase, we extract all its immediate neighbors in the English sub-graph of ConceptNet. Each neighbor assertion is a ⟨ _source_ , _relation_ , _target_ , _weight_ ⟩ quadruple where _source_ and _target_ denote the query phrase and the neighbor concept, _relation_ denotes the corresponding relation type, and _weight_ denotes a confidence score assigned to the assertion. For example, we can extract the assertion: ⟨ _think_ , _HasPrerequisite_ , _use brains_ , 2.375⟩ for the input phrase _think_ . There are 28 different relation types in the extracted quadruples. We further remove all assertions with confidence scores less than 2.0 for denoising. To facilitate the extraction of the knowledge representations, we manually design an interpretation for each relation type and convert all assertions into natural language by concatenating the concepts to the relation interpretation. For example, ⟨ _think_ , _HasPrerequisite_ , _use brains_ ⟩ is converted to "think has the prerequisite of use brains". We present the interpretations of the five most common relations in Table 1. As most of these knowledge items have simple semantics, we use a RoBERTa-Base encoder to extract the features for the converted natural language knowledge and each knowledge representation has dimension _𝐷𝑘[𝐶𝑜𝑛𝑐𝑒𝑝𝑡]_ . Overall, for each utterance _𝑢𝑖_ and a knowledge source, we obtain a set of knowledge representations: { _𝑘𝑖_ 1 _,𝑘𝑖_ 2 _, ...,𝑘𝑖𝑚𝑖_ }, where _𝑘𝑖𝑗_ ∈ R _[𝐷][𝑘]_ , the representation dimension _𝐷𝑘_ ∈{ _𝐷𝑘_[2019] _, 𝐷𝑘_[2020] _, 𝐷𝑘[𝐶𝑜𝑛𝑐𝑒𝑝𝑡]_ } depends on the knowledge source, and _𝑚𝑖_ denote the number of extracted knowledge items for _𝑢𝑖_ . 

## **2.2 Bipartite Heterogeneous Graph** 

We propose a directed bipartite heterogeneous graph (BHG) to complement utterance representations with extracted commonsense knowledge representations. Formally, the BHG is denoted as G = (V _,_ E _,_ N _,_ R), where each node _𝑣_ ∈V and each edge _𝑒_ ∈E. N _,_ R denote the set of node and relation types, and each node or edge is projected to its type via a mapping function: _𝛾_ ( _𝑣_ ) : V →N , _𝜏_ ( _𝑒_ ) : E →R, respectively. Specifically, the node and edge types are defined as follows: 

_2.2.1 Node Types._ Firstly, each utterance within a conversation is modeled as a node in the BHG. Its representation obtained from the conversation model is used as the node feature. The utterance node type is denoted as _ℎ_ . The extracted knowledge representations are modeled as another type of node: _𝑘_ . In addition, we introduce extra knowledge aggregation node types to perform automatic knowledge filtering and knowledge interaction. As previous works have shown that effective knowledge aspects can be different for performing knowledge interactions with the past context and future context [12, 13, 44], we introduce a forward and a backward aggregation node type, which is separately denoted as _𝑓_ ∈ R _[𝐷][𝑓]_ and _𝑏_ ∈ R _[𝐷][𝑏]_ , where _𝐷 𝑓_ and _𝐷𝑏_ are their representation dimensions. Overall, we define a set with four types of nodes: N = { _ℎ,𝑘, 𝑓,𝑏_ }. 

_2.2.2 Relation Types._ For each edge _𝑒_ = ⟨ _𝑣_ 1 _, 𝑣_ 2⟩ from the source node _𝑣_ 1 to the target node _𝑣_ 2, its relation type is defined as _𝜏_ ( _𝑒_ ) : ⟨ _𝛾_ ( _𝑣_ 1) _,𝛾_ ( _𝑣_ 2)⟩. Specifically, we introduce six types of relations to perform knowledge filtering and knowledge interaction. Firstly, the forward/backward knowledge information relations: _𝜏𝑘𝑓_ : ⟨ _𝑘, 𝑓_ ⟩ and 

_𝜏𝑘𝑏_ : ⟨ _𝑘,𝑏_ ⟩ are proposed to separately provide the extracted knowledge information to both forward and backward knowledge aggregation nodes. Secondly, the knowledge filtering process naturally requires considering the corresponding utterance information. Therefore, we utilize the utterance information relations: _𝜏𝑢𝑓_ : ⟨ _ℎ, 𝑓_ ⟩ and _𝜏𝑢𝑏_ : ⟨ _ℎ,𝑏_ ⟩ to introduce utterance semantics to the knowledge aggregation nodes. In addition, commonsense knowledge has been proven not only useful for enriching the semantics of each utterance but in modeling inter-utterance dependencies [35, 44, 45]. Therefore, we further incorporate the filtered commonsense knowledge to the utterance nodes by designing a forward infusion relation: _𝜏𝑓_ : ⟨ _𝑓,ℎ_ ⟩ and a backward infusion relation: _𝜏𝑏_ : ⟨ _𝑏,ℎ_ ⟩. Overall, the relation set contains six types of relations: R = { _𝜏𝑘𝑓 ,𝜏𝑘𝑏,𝜏𝑢𝑓 ,𝜏𝑢𝑏,𝜏𝑓 ,𝜏𝑏_ }. 

_2.2.3 BHG Construction._ For each utterance node _ℎ𝑖_ , we create a forward aggregation node _𝑓𝑖_ and a backward aggregation node _𝑏𝑖_ . Then we build a BHG for each conversation by considering the following criteria for each relation type: 

- ∀ _𝑖_ ≠ _𝑗_ and _𝑚,_ ⟨ _𝑘𝑖𝑚, 𝑓𝑗_ ⟩ ∉ E and ⟨ _𝑘𝑖𝑚,𝑏 𝑗_ ⟩ ∉ E. Each knowledge aggregation node only receives and filters the extracted knowledge from its corresponding utterance. 

- ∀ _𝑖_ ≠ _𝑗,_ ⟨ _ℎ𝑖, 𝑓𝑗_ ⟩ ∉ E and ⟨ _ℎ𝑖,𝑏 𝑗_ ⟩ ∉ E. Each knowledge aggregation node only considers the information from its corresponding utterance when performing knowledge filtering. 

- ∀ _𝑖_ ≤ _𝑗_ ≤ _𝑖_ + _𝑊𝑓 ,_ ⟨ _𝑓𝑖,ℎ 𝑗_ ⟩∈E, where _𝑊𝑓_ is the pre-defined forward knowledge infusion window size. Each forward aggregated knowledge is only used to enrich the semantics of its own utterance and model the inter-utterance relations with future utterances. 

- ∀ _𝑖_ − _𝑊𝑏_ ≤ _𝑗_ ≤ _𝑖,_ ⟨ _𝑏𝑖,ℎ 𝑗_ ⟩∈E, where _𝑊𝑏_ is the pre-defined backward knowledge infusion window size. Each backward aggregated knowledge is only used to enrich the semantics of its own utterance and model the inter-utterance relations with previous utterances. 

Based on these criteria, the BHG building process is described in Algorithm 1, and an example is illustrated in Figure 2(b). 

## **2.3 Multi-dimensional HGT** 

In heterogeneous graphs, different types of nodes/relations do not share feature spaces, which requires node and relation-dependent architectures for GNN-based graph modeling. The Heterogeneous Graph Transformer (HGT) [8] separately initializes and optimizes a set of Transformer [31] parameters for each homogeneous subgraph. In addition, HGT unifies the dimensions of all node types before aggregation via linear transformation due to the complicated structures of most large-scale heterogeneous graphs. In our case, the utterance and knowledge representations often have unequal dimensions: _𝐷ℎ_ ≠ _𝐷𝑘_ with the substitution of conversation models and knowledge sources. Considering the simple structure of BHG, we believe unifying dimensions can lead to unnecessary loss of information for high-dimensional node types. Therefore, we propose a Multi-dimensional HGT (MHGT) to model the BHG, which allows the dimensions of heterogeneous nodes to remain unchanged during the attention calculation and message-passing processes. 

Based on the structure of vanilla HGT, MHGT improves the heterogeneous dot-product attention process to enable interactions 

CIKM ’23, October 21–25, 2023, Birmingham, United Kingdom 

A Bipartite Graph is All We Need for Enhancing Emotional Reasoning with Commonsense Knowledge 

|**Algorithm 1**Construction of the BHG|**Algorithm 1**Construction of the BHG|**Algorithm 1**Construction of the BHG|**Algorithm 1**Construction of the BHG|
|---|---|---|---|
|**Input:** dialogue representations:||H|= [_ℎ_1,_ℎ_2,...,_ℎ𝑛_], knowledge|
||representations:K= [_𝑘_11,...,_𝑘_1_𝑚_1,...,_𝑘𝑛_1,...,_𝑘𝑛𝑚𝑛_], forward and|||
||backward knowledge infusion|window sizes:_𝑊𝑓_,_𝑊𝑏_||
|**Output:** the BHGG = (V_,_E_,_N_,_R)||||
|1:|V ←H ∪K,E ←∅||_⊲_Initialize the BHG.|
|2:|N ←{_ℎ,𝑘, 𝑓,𝑏_}|||
|3:|R ←{_𝜏𝑘𝑓,𝜏𝑘𝑏,𝜏𝑢𝑓,𝜏𝑢𝑏,𝜏𝑓,𝜏𝑏_}|||
|4:|**for**_𝑖_∈{1_, ...,𝑛_}**do**|||
|5:|V ←V ∪{_𝑓𝑖,𝑏𝑖_}||_⊲_Add the aggregation nodes.|
|6:|**for** _𝑗_∈{1_, ...,𝑚𝑖_}**do**|||
|7:|E ←E ∪{⟨_𝑘𝑖𝑗, 𝑓𝑖_⟩}<br>_⊲_Add knowledge info. relations.|||
|8:|E ←E ∪{⟨_𝑘𝑖𝑗,𝑏𝑖_⟩}|||
|9:|**end for**|||
|10:|E ←E ∪{⟨_ℎ𝑖, 𝑓𝑖_⟩}|_⊲_Add utterance info. relations.||
|11:|E ←E ∪{⟨_ℎ𝑖,𝑏𝑖_⟩}|||
|12:|_𝑐𝑓_=0|||
|13:|**while**_𝑐𝑓_≤_𝑊𝑓_and_𝑖_+_𝑐𝑓_≤_𝑛_**do**|||
|14:|E ←E ∪{⟨_𝑓𝑖,ℎ𝑖_+_𝑐𝑓_⟩}|_⊲_Forward knowledge infusion.||
|15:|_𝑐𝑓_←_𝑐𝑓_+1|||
|16:|**end while**|||
|17:|_𝑐𝑏_=0|||
|18:|**while**_𝑐𝑏_≤_𝑊𝑏_and_𝑖_−_𝑐𝑏_≥0**do**|||
|19:|E ←E ∪{⟨_𝑏𝑖,ℎ𝑖_−_𝑐𝑏_⟩}_⊲_Backward knowledge infusion.|||
|20:|_𝑐𝑏_←_𝑐𝑏_+1|||
|21:|**end while**|||
|22:|**end for**|||



edge-dependent matrix _𝑊𝜏[𝑀𝑒𝑠]_ (⟨ _𝑣𝑛,𝑣𝑡_ ⟩)[∈][R] _𝐷𝛾 ℎ_ ( _𝑣𝑛_ ) × _𝐷𝛾ℎ_ ( _𝑣𝑡_ ) to align the neighbor node _𝑣𝑛_ to the feature space of the target node _𝑣𝑡_ . Same as in vanilla dot-product attention, the message is aggregated using the calculated attention weights: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge_images/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge.pdf-0005-04.png)


where ⊕ denotes the element-wise sum operation. Finally, _𝑣_ ˆ _𝑡[𝑙]_[+][1] is used to update _𝑣𝑡[𝑙]_[in the following manner:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge_images/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge.pdf-0005-06.png)


where _𝜎_ denotes the _𝐺𝑒𝑙𝑢_ activation function and _𝑇𝛾_ ( _𝑣𝑡_ ) denotes an equal-dimensional linear projection from _𝛾_ ( _𝑣𝑡_ ). We stack _𝐿_ MHGT layers to allow interactions between non-adjacent nodes. For utterance _𝑢𝑖_ , we obtain the _𝐿_ -th layer output from MHGT: _ℎ𝑖[𝐿]_[as its final] knowledge-enhanced representation. 

## **2.4 Prediction and Training** 

We examine the BHG-based knowledge infusion method on two complex emotional reasoning tasks: Emotion Recognition in Conversations (ERC) and Casual Emotion Entailment (CEE). 

_2.4.1 Emotion Recognition in Conversations._ ERC aims to identify each utterance _𝑢𝑖_ ’s emotion within a dialogue D from a pre-defined emotion category set _𝐸_ , which is modeled as a text classification task on each utterance [23]. Specifically, we utilize a feed-forward neural network to project the knowledge-enhanced utterance representations to the classification space: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge_images/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge.pdf-0005-11.png)


between nodes with different dimensions. For a target node representation _𝑣𝑡[𝑙]_[∈][R] _[𝐷][𝛾]_[(] _[𝑣𝑡]_[)][at the] _[ 𝑙]_[-th MHGT layer, we calculate the] multi-head attention weights for its neighbor _𝑣𝑛[𝑙]_ ∈ R _[𝐷][𝛾]_[(] _[𝑣𝑛]_[)] as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge_images/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge.pdf-0005-13.png)


_𝐷𝛾_ ( _𝑣𝑛_ ) _𝐷𝛾_ ( _𝑣𝑡_ ) where _𝐾𝛾[𝑖]_ ( _𝑣𝑛_ )[(] _[𝑣][𝑛]_[)][∈][R] _ℎ_ and _𝑄𝛾[𝑖]_ ( _𝑣𝑡_ )[(] _[𝑣][𝑡]_[)][∈][R] _ℎ_ denotes the _𝑖_ -th head of a equal-dimensional linear projection from _𝑣𝑛[𝑙]_ as the key and _𝑣𝑡[𝑙]_[as the query,][ ||][ denotes concatenation, and] _[ 𝑆𝑜𝑓𝑡𝑚𝑎𝑥]_[de-] notes the softmax operation. Different from the vanilla HGT, MHGT proposes a multi-dimensional edge-dependent matrix _𝑊𝜏[𝐴𝑡𝑡]_ (⟨ _𝑣𝑛,𝑣𝑡_ ⟩)[∈] R _𝐷𝛾 ℎ_ ( _𝑣𝑛_ ) × _𝐷𝛾ℎ_ ( _𝑣𝑡_ ) to project nodes of _𝛾_ ( _𝑣𝑛_ ) to the representation space of _𝛾_ ( _𝑣𝑡_ ), which enables attention calculations between heterogeneous node features and allows situations where _𝐷𝛾_ ( _𝑣𝑛_ ) ≠ _𝐷𝛾_ ( _𝑣𝑡_ ) . We perform the message-passing process for _𝑣𝑛[𝑙]_ with the target node _𝑣𝑡[𝑙]_[as follows:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge_images/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge.pdf-0005-15.png)


where _𝑉𝛾_ ( _𝑣𝑛_ ) ( _𝑣𝑛_ ) is another equal-dimensional linear projection from _𝑣𝑛_ as the value. Similarly, MHGT uses a multi-dimensional 

where _𝑊𝐸𝑅𝐶_ ∈ R _[𝐷][ℎ]_[×|] _[𝐸]_[|] and _𝑏𝐸𝑅𝐶_ ∈ R[|] _[𝐸]_[|] are learnable parameters. We optimize the standard cross-entropy loss to train the ERC model: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge_images/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge.pdf-0005-18.png)


where _𝑦_ ˆ _𝑖𝑗_ and _𝑦𝑖𝑗_ denote the _𝑗_ -th element of _𝑦_ ˆ _𝑖_ and the one-hot emotion label _𝑦𝑖_ of utterance _𝑢𝑖_ , and _𝑁_ denotes the batch size. 

_2.4.2 Casual Emotion Entailment._ CEE aims to identify the causes behind the non-neutral emotion of target utterances and locate their positions from the conversational history. Given a dialogue D and the emotion label _𝑦_ of each utterance, CEE is modeled as a binary classification task to predict whether the candidate utterance _𝑢 𝑗_ contains the emotion cause for target utterance _𝑢𝑖_ , where 1 ≤ _𝑗_ ≤ _𝑖_ . During inference, we concatenate the knowledge-enhanced utterance representations of _𝑢 𝑗_ and _𝑢𝑖_ to calculate the logits: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge_images/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge.pdf-0005-21.png)


where _𝑊𝐶𝐸𝐸_ ∈ R[2] _[𝐷][ℎ]_[×][1] and _𝑏𝐶𝐸𝐸_ ∈ R[1] are learnable parameters, respectively. A BCE loss is used to incorporate the supervision signals: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge_images/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge.pdf-0005-23.png)


where _𝑧 𝑗𝑖_ ∈{0 _,_ 1} is the binary label. We further introduce the emotion labels to enhance the learning of the CEE model using the 

CIKM ’23, October 21–25, 2023, Birmingham, United Kingdom 

Kailai Yang, Tianlin Zhang, Shaoxiong Ji, and Sophia Ananiadou 

same prediction and training paradigm as in Eqn. 9 and Eqn. 10, and jointly optimize the two tasks in a multi-task learning manner: 

L _𝐶𝐸𝐸_ = L _𝐵𝐶𝐸_ + _𝛼_ L _𝐸𝑅𝐶_ (13) 

where _𝛼_ denotes a hyper-parameter controlling the weight of the ERC loss. 

## **3 EXPERIMENTS** 

## **3.1 Datasets** 

We test our method on four Emotion Recognition in Conversations (ERC) datasets and one Causal Emotion Entailment (CEE) dataset. For all datasets, we only utilize the text modality in our experiments. **IEMOCAP** [3]: A two-party multi-modal ERC dataset derived from the scenarios in the scripts of the two actors. The pre-defined emotion category set _𝐸_ consists of: _neutral, sad, anger, happy, frustrated, excited_ . 

**MELD** [20]: A multi-party multi-modal ERC dataset collected from the scripts of American TV show _Friends_ . The pre-defined emotions are _neutral, sad, anger, disgust, fear, happy, surprise_ . 

**DailyDialog** [14]: A ERC dataset compiled from human-written daily conversations with only two parties involved and no speaker information. The pre-defined emotion labels are _neutral, happy, surprise, sad, anger, disgust, fear_ . 

**EmoryNLP** [42]: Another ERC dataset collected from TV show _Friends_ . It is annotated with the following emotion categories: _neutral, sad, mad, scared, powerful, peaceful, joyful_ . 

**RECCON** [22]: A CEE dataset collected from the scripts of DailyDialog with both utterance-level emotion labels and binary emotion cause labels, with the same emotion category set as DailyDialog. 

## **3.2 Baseline Models** 

We compare our method with strong baselines models and categorize them into four groups according to their characteristics: 

_PLM-based methods._ For the ERC task, we select the following two methods: **RoBERTa-Large** [15] used the PLM RoBERTa-Large to directly model the conversation. The utterance representations are used to fine-tune the weights. **DialogXL** [27] improved the XLNet [41] with the enhanced memory and dialog-aware self-attention mechanism to capture long historical context and dependencies between multiple parties. For the CEE task, we select: **RoBERTaBase/Large** [22] concatenated the conversation as input to the PLM RoBERTa. Then CEE was modeled as a binary classification problem for each utterance pair. 

_Graph-based Methods._ For ERC, **RGAT** [10] improved the relation modeling of conversations and added relational position encodings as sequential information. **DAG-ERC** [28] built a directed acyclic graph on the dialogue and used a graph neural network to aggregate the information. For CEE, we select two methods: **ECPE-2D** [5] represented the emotion-cause pairs as 2D representations and utilized the Transformer to model them. **RankCP** [33] ranked the clause pairs and performed end-to-end extraction with inter-clause modeling. 

_Knowledge-based Methods._ For ERC, five methods are selected: **KI-Net** [35] infused both commonsense and sentiment lexicon knowledge and proposed a self-matching module to enhance the 

knowledge interaction. **COSMIC** [6] used the RNN to model the dialogue history and extracted utterance-level commonsense knowledge to model the speakers’ mental states. **TODKAT** [47] modeled topic information via PLMs and explicitly infused event-centered knowledge. **SKAIG** [12] extracted psychological commonsense knowledge and infused the knowledge to enhance the edge representations of the knowledge graph. **CauAIN** [45] used the emotion cause knowledge to guide the traceback process of context modeling. For CEE, **KAG** [36] utilized the entity-related commonsense knowledge to model the semantic dependencies between the candidates and emotions. **AKM** [30] combined knowledge via an adapted knowledge model in a multi-task learning manner. **KEC** [13] utilized the directed acyclic graph incorporating social commonsense knowledge to improve the causal reasoning ability. **KBCIN** [44] proposed the knowledge-bridged causal interaction network to capture context dependencies of conversations and make emotional cause reasoning. 

_Zero-shot Method with ChatGPT._ . We also include the zero-shot evaluation results of the latest large language model ChatGPT[1] on all datasets, provided by Yang et al. [37]. 

## **3.3 Implementation and Evaluation Settings** 

We conduct all experiments using a single Nvidia Tesla A100 GPU with 80GB of memory. We initialize the pre-trained weights of RoBERTa and use the tokenization tools provided by Huggingface [34]. We leverage AdamW optimizer [16] to train the model. The batch size of experiments on all datasets is set to 16 except in DailyDialog, which is 24. We use a linear warm-up learning rate scheduling of warm-up ratio 20% and a peak learning rate 1e-5. We set a dropout rate of 0.1 and an L2-regularisation rate of 0.01 to avoid over-fitting. For hyper-parameters, _𝐷ℎ_ = _𝐷𝑘_[2020] = 1024, _𝐷𝑘_[2019] = _𝐷𝑘[𝐶𝑜𝑛𝑐𝑒𝑝𝑡]_ = 768, _𝐿_ = 3, and _𝛼_ is set to 0.8. For each knowledge source with dimension _𝐷𝑘_ , we set _𝐷 𝑓_ = _𝐷𝑏_ = _𝐷𝑘_ and randomly initialize the forward/backward knowledge aggregation node representations. During knowledge infusion, we set _𝑊𝑓_ = _𝑊𝑏_ = 5 when both past and future contexts are provided, and _𝑊𝑓_ = 0, _𝑊𝑏_ = 10 when only the past context is available. 

For ERC, we select the Weighted F1 score as the evaluation metric for IEMOCAP, MELD, and EmoryNLP. Since “neutral” occupies most of DailyDialog, we utilize the Micro F1 score excluding the "neutral" utterances to reflect the performances in non-neutral emotions, as in previous works [12, 28, 47]. We also calculate Macro F1 scores on all classes for DailyDialog to evaluate the overall performances. For CEE, we report the F1 scores on positive and negative utterances and Macro F1 scores as the overall evaluation. All reported results are averages of five random runs. 

## **4 RESULTS AND ANALYSIS** 

## **4.1 Main Results** 

_4.1.1 Overall Performance._ The performance of our BHG-based methods and all baseline models on the five datasets are presented in Table 2. Firstly, the zero-shot prompting results show that ChatGPT still bears a huge gap with advanced conversation models 

1https://openai.com/blog/chatgpt 

CIKM ’23, October 21–25, 2023, Birmingham, United Kingdom 

A Bipartite Graph is All We Need for Enhancing Emotional Reasoning with Commonsense Knowledge 

**Table 2: Test results of our BHG-based methods and baseline models on the five datasets. "Know. Source" lists the commonsense knowledge sources used by each knowledge-based method. In "w/o future context", only the dialogue history is introduced as context for each target utterance. The "w/o emotions" setting removes the utterance emotion information for CEE. Best values are highlighted in bold.** 

|**ERC**<br>**IEMOCAP**<br>**MELD**<br>**DailyDialog**<br>**EmoryNLP**|**CEE**<br>**RECCON**|
|---|---|
|Model<br>Know. Source<br>Weighted F1<br>Weighted F1<br>Micro F1<br>Macro F1<br>Weighted F1|Model<br>Know. Source<br>Neg. F1<br>Pos. F1<br>Macro F1|
|**Zero-shot Methods**||
|ChatGPT_𝑍𝑆_[37]<br>–<br>53.35<br>61.18<br>43.27<br>38.19<br>32.64|ChatGPT_𝑍𝑆_[37]<br>–<br>67.18<br>51.35<br>59.26|
|**PLM-based Methods**||
|RoBERTa-Large [15]<br>–<br>55.67<br>62.75<br>55.16<br>48.2<br>37.0<br>DialogXL [27]<br>65.94<br>62.41<br>54.93<br>–<br>34.73|RoBERTa-Base [22]<br>–<br>88.74<br>64.28<br>76.51<br>RoBERTa-Large [22]<br>87.89<br>66.23<br>77.06|
|**Graph-based Methods**||
|RGAT [10]<br>–<br>66.36<br>62.8<br>59.02<br>–<br>34.42<br>DAG-ERC [28]<br>68.03<br>63.65<br>59.33<br>–<br>39.02|ECPE-2D [5]<br>–<br>94.96<br>55.50<br>75.23<br>RankCP [33]<br>**97.30**<br>33.00<br>65.15|
|**Knowledge-based Methods**||
|CauAIN [45]<br>_𝐶𝑂𝑀𝐸𝑇_2019<br>67.61<br>65.46<br>58.21<br>53.85<br>–<br>KI-Net [35]<br>_𝐶𝑜𝑛𝑐𝑒𝑝𝑡𝑁𝑒𝑡_<br>66.98<br>63.24<br>57.30<br>50.8<br>–<br>COSMIC [6]<br>_𝐶𝑂𝑀𝐸𝑇_2019<br>65.28<br>65.21<br>58.48<br>51.05<br>38.11<br>TODKAT [47]<br>_𝐶𝑂𝑀𝐸𝑇_2019<br>61.33<br>65.47<br>58.47<br>52.56<br>38.69<br>SKAIG [12]<br>_𝐶𝑂𝑀𝐸𝑇_2019<br>66.96<br>65.18<br>59.75<br>51.95<br>38.88|KAG [36]<br>_𝐶𝑜𝑛𝑐𝑒𝑝𝑡𝑁𝑒𝑡_<br>86.35<br>58.18<br>72.26<br>AKM [30]<br>_𝐶𝑂𝑀𝐸𝑇_2019<br>88.18<br>64.53<br>76.36<br>KEC [13]<br>_𝐶𝑂𝑀𝐸𝑇_2020<br>88.85<br>66.55<br>77.70<br>KBCIN [44]<br>_𝐶𝑂𝑀𝐸𝑇_2020<br>–<br>67.51<br>78.43<br>w/o emotions<br>–<br>64.05<br>76.73|
|**BHG-based Methods**||
|MHGT<br>_𝐶𝑂𝑀𝐸𝑇_2020<br>**71.20**<br>65.54<br>**62.37**<br>**54.11**<br>**39.06**<br>w/o future context<br>70.93<br>**66.3**<br>60.41<br>53.29<br>38.85|MHGT<br>_𝐶𝑂𝑀𝐸𝑇_2020<br>90.34<br>**69.13**<br>**79.73**<br>w/o emotions<br>89.89<br>68.58<br>79.23|
|MHGT<br>_𝐶𝑂𝑀𝐸𝑇_2019<br>71.07<br>65.57<br>61.5<br>53.0<br>38.62<br>w/o future context<br>70.64<br>65.77<br>60.68<br>52.51<br>38.33<br>HGT<br>67.2<br>62.86<br>60.0<br>51.27<br>36.5|MHGT<br>_𝐶𝑂𝑀𝐸𝑇_2019<br>90.26<br>68.85<br>79.55<br>w/o emotions<br>89.95<br>68.49<br>79.22<br>HGT<br>89.79<br>62.94<br>76.37|
|MHGT<br>_𝐶𝑜𝑛𝑐𝑒𝑝𝑡𝑁𝑒𝑡_<br>70.55<br>65.99<br>61.41<br>52.7<br>38.07<br>w/o future context<br>70.68<br>65.59<br>60.2<br>51.74<br>37.38<br>HGT<br>67.1<br>64.29<br>59.28<br>51.09<br>36.06|MHGT<br>_𝐶𝑜𝑛𝑐𝑒𝑝𝑡𝑁𝑒𝑡_<br>89.53<br>68.6<br>79.07<br>w/o emotions<br>90.03<br>67.72<br>78.88<br>HGT<br>90.12<br>62.47<br>76.29|



and knowledge-based methods in performing emotional reasoning, possibly because these tasks are very subjective even to humans, showing the necessity of exploring few-shot prompting and knowledge infusion to further calibrate ChatGPT’s understanding of these subjective emotion concepts [37]. These results also motivate continual research on supervised task-specific methods. Secondly, the knowledge-based methods significantly improve model performance on most datasets compared to the PLM-based and graph-based conversation models. These results empirically prove the effectiveness of commonsense knowledge infusion to emotional reasoning tasks. Thirdly, the BHG-based methods outperform all baseline models on all ERC and CEE datasets, including the new state-of-the-art performance of 71.2% on IEMOCAP, 62.37% (MicroF1) on DailyDialog, and 79.73% (Macro-F1) on RECCON. These outstanding performances quantify the advantages of BHG-based knowledge infusion over previous methods. 

_4.1.2 Knowledge-based Methods Comparisons._ In the comparison of knowledge-based methods, the unified BHG architecture shows generalized knowledge infusion ability by outperforming all previous highly customized knowledge infusion methods on all three tested knowledge sources. For the phrase-level extractive knowledge source ConceptNet, BHG outperforms KI-Net by at least 2% on IEMOCAP, MELD, and DailyDialog. It also has an impressive 6.81% improvement on RECCON compared to KAG. These results show that the BHG structure can effectively interact with multi-grained heterogeneous nodes to infuse knowledge. For the utterance-level generative knowledge source _𝐶𝑂𝑀𝐸𝑇_ 2019, BHG possesses a similar advantage over other knowledge-based methods on most datasets, especially on IEMOCAP, DailyDialog, and RECCON. A possible reason is that some useful knowledge is mistakenly discarded by 

these previous methods as they all perform empirical knowledge filtering. On the other hand, we provide all knowledge aspects to construct the BHG and perform knowledge filtering automatically, which enables the aggregation nodes to retain all useful knowledge. The knowledge source _𝐶𝑂𝑀𝐸𝑇_ 2020 is proven most effective in enhancing emotional reasoning as it outperforms all other knowledge sources on all five datasets. Compared to _𝐶𝑂𝑀𝐸𝑇_ 2019, _𝐶𝑂𝑀𝐸𝑇_ 2020 trains a larger language model on a bigger social commonsense knowledge graph. Therefore, _𝐶𝑂𝑀𝐸𝑇_ 2020 is expected to generate more reliable knowledge for each utterance. These results show the importance of high-quality commonsense knowledge sources. 

_4.1.3 BHG Variants._ We further compare the BHG-based methods in experimental settings such as "w/o future context" and "w/o emotions" to test model performance in scenarios where future contexts and emotion labels are unavailable. For ERC, the BHG’s performance drops with only past context to a limited extent on most datasets. These results show that future dialogue can provide useful clues for the emotional reasoning of the current utterance in most cases. For CEE, the BHG performances on all knowledge resources drop less than 0.5% without the emotional supervision signals, which is much less than the 1.7% decrease on the previous state-of-the-art model KBCIN. We believe the appropriately infused commonsense knowledge from BHG can make up for the missing emotional information introduced by the labels. Finally, we compare the performance of MHGT and the vanilla HGT in modeling the BHG. During experiments for HGT, we unify the heterogeneous node dimensions by linearly projecting the highdimension representations to low-dimensional spaces. According to the results, HGT significantly underperforms MHGT on all datasets. For example, we observe an over 3% drop in Weighted-F1 for both 

CIKM ’23, October 21–25, 2023, Birmingham, United Kingdom 

Kailai Yang, Tianlin Zhang, Shaoxiong Ji, and Sophia Ananiadou 

_𝐶𝑂𝑀𝐸𝑇_ 2019 and _𝐶𝑜𝑛𝑐𝑒𝑝𝑡𝑁𝑒𝑡_ on IEMOCAP. MHGT retains the original semantic space for utterance representations, which preserves useful information to perform semantic-aware knowledge filtering and emotional reasoning, while HGT projects the representations to low-dimensional spaces and causes unnecessary loss of information. 

**Table 3: Ablation studies for the BHG and other knowledgebased baseline models. "know." denotes the whole knowledge infusion architectures. "backward/forward aggr." denotes the backward/forward knowledge aggregation designs in BHG. We highlight top-2 performance drops in bold.** 

|**Model**|**IEMOCAP**<br>**MELD**<br>**DailyDialog**<br>**EmoryNLP**|
|---|---|
|CauAIN<br>w/o know.|67.61<br>65.46<br>58.21<br>–<br>63.77 (↓**3.84**)<br>65.2 (↓0.26)<br>57.2 (↓1.01)<br>–|
|COSMIC<br>w/o know.|65.28<br>65.21<br>58.48<br>38.11<br>63.05 (↓2.23)<br>64.28 (↓0.93)<br>56.16 (↓**2.32**)<br>37.10 (↓1.01)|
|TODKAT<br>w/o know.|61.33<br>65.47<br>54.62<br>38.69<br>58.96 (↓2.37)<br>63.97 (↓**1.5**)<br>53.44 (↓1.18)<br>37.1 (↓1.59)|
|SKAIG<br>w/o know.|66.96<br>65.18<br>59.75<br>38.88<br>64.28 (↓2.68)<br>–<br>58.86 (↓0.89)<br>–|
|BHG<br>w/o backward aggr.<br>w/o forward aggr.<br>w/o know.|71.07<br>65.57<br>61.5<br>38.62<br>68.54 (↓2.53)<br>64.81 (↓0.76)<br>60.18 (↓1.32)<br>38.07 (↓0.55)<br>68.07 (↓3.0)<br>64.41 (↓1.16)<br>59.3 (↓2.3)<br>36.95 (↓**1.67**)<br>67.93 (↓**3.14**)<br>63.7 (↓**1.87**)<br>58.76 (↓**2.74**)<br>36.26 (↓**2.36**)|



## **4.2 Ablation Studies for Knowledge Infusion** 

To further investigate the BHG’s efficiency in knowledge infusion and the contributions of its components, we perform ablation studies on the BHG and other knowledge-based baseline methods, and the results are presented in Table 3. To ensure fair comparisons, we select the BHG and four knowledge infusion methods that are all tested with the _𝐶𝑂𝑀𝐸𝑇_ 2019 knowledge source on the four ERC datasets. 

_4.2.1 Knowledge Infusion Efficiency._ Firstly, we compare the performance drops between the models when the whole knowledge infusion architecture is removed, where more significant drops reflect the higher efficiency of the method in leveraging the same knowledge source. According to the results, "BHG w/o know." achieves top-2 performance drops on all datasets, and the drops exceed 2% on three out of four datasets, while other knowledge infusion methods normally make substantial contributions to only one or two datasets. These results show that our unified BHG method can utilize the same knowledge source to enhance emotional reasoning on differently distributed data more efficiently than previous customized knowledge infusion methods. 

_4.2.2 BHG Modules._ We further investigate the backward/forward knowledge infusion architectures in BHG by removing these modules and comparing the performance. According to the results, "BHG w/o backward aggr." and "BHG w/o forward aggr." both perform worse than BHG on all datasets. These results not only strengthen that infusing commonsense knowledge can enhance the modeling of inter-utterance relations, but further prove that a unified knowledge infusion module can be decoupled from the utterance modeling process and used for any conversation models in a 

plug-in manner. They also prove the necessity of splitting the knowledge filtering and interaction processes for previous and future contexts in the BHG architecture. In addition, "BHG w/o forward aggr." suffers from higher performance drops on all datasets than "BHG w/o backward aggr.", which shows that social commonsense knowledge is more useful in modeling inter-utterance relations for future contexts. These observations correspond with the widely recognized prior that the current utterance has more significant influences on future utterances along the dialogue flow [28]. 

## **4.3 Knowledge Filtering Analysis** 

To provide an intuitive view of the knowledge filtering process, we record the dot-product attention weights between the knowledge nodes and knowledge aggregation nodes in the last MHGT layer for _𝐶𝑂𝑀𝐸𝑇_ 2019 on IEMOCAP test set and visualize their distributions in box plots. The results are presented in Figure 3. Following the BHG structure, we split the attention weights from forward and backward knowledge filtering, and higher weight distributions reflect more contributions from the corresponding knowledge aspect. 

_4.3.1 Overall Analysis._ For forward knowledge filtering, the most attended knowledge aspect _xWant_ has a median number over 0.3. _xIntent_ and _xEffect_ are also frequently utilized, with their median numbers over 0.2. These results show that forward knowledge interaction can benefit more from knowledge aspects reflecting the potential effect of the target utterance on the target speaker. On the other hand, backward knowledge filtering pays attention to the knowledge aspects reflecting the effect on both the target speaker and other participants. For example, _xEffect_ and _oWant_ are highly attended with median numbers over 0.3, and _xNeed_ and _oEffect_ also have median numbers of over 0.2. In comparisons of the box plots for forward and backward knowledge filtering, the BHG has very different patterns for selecting knowledge aspects. For example, _xWant_ knowledge is considered useful in forward knowledge filtering but receives much less attention in backward knowledge filtering. These observations further show the necessity of designing forward and backward knowledge aggregation nodes in BHG. In addition, some knowledge aspects such as _xAttr_ and _oReact_ have low attention distributions in both forward and backward knowledge filtering, possibly because these knowledge aspects can provide less relevant information for emotional reasoning. 

_4.3.2 Comparison with Previous Methods._ Previous knowledgebased methods mostly perform knowledge filtering manually in the utilization of COMET2019, where frequently utilized knowledge aspects include _xWant_ , _xEffect_ , _xIntent_ , _oReact_ , _oWant_ , and _oEffect_ [6, 12, 45, 47]. Most of these knowledge aspects are also proven effective in the automatic knowledge filtering of BHG, while some aspects such as _xReact_ and _oReact_ are less attended in both forward and backward knowledge filtering. These results indicate that some widely used knowledge aspects may not provide much useful information as expected. On the other hand, some previously ignored knowledge aspects are assigned higher attention scores in automatic knowledge filtering. For example, _xNeed_ , a less-used knowledge aspect in previous works, receives considerable attention in backward knowledge filtering. We expect the above analysis to guide the adjustment of future strategies for empirical methods. 

CIKM ’23, October 21–25, 2023, Birmingham, United Kingdom 

A Bipartite Graph is All We Need for Enhancing Emotional Reasoning with Commonsense Knowledge 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge_images/cikm_2023_A_Bipartite_Graph_is_All_We_Need_for_Enhancing_Emotional_Reasoning_with_Commonsense_Knowledge.pdf-0009-02.png)


**Figure 3: Box plots of the knowledge filtering attention weights for nine knowledge aspects on the IEMOCAP test set. We use** _𝐶𝑂𝑀𝐸𝑇_ 2019 **as the knowledge source and MHGT as the BHG encoder. Orange lines denote the median numbers.** 

## **5 RELATED WORK** 

We discuss relevant background on conversational emotion reasoning, including conversation models and knowledge-based methods. 

## **5.1 Conversation Models** 

Emotion reasoning in a conversation naturally requires modeling interactions between dialogue participants, known as intra- and inter-speaker dependencies [27]. Early works regarded dialogues as temporal flows and utilized RNNs to model the dialogue history and emotional dynamics of each dialogue participant [6, 18]. In other works, Transformer variants were also leveraged to model long-range dependencies in conversations [43, 46]. Recent works mostly relied on PLMs such as RoBERTa [15] and XLNet [41] to obtain utterance-level or conversation-level features [1, 27]. Some other works [7, 10, 28] modeled utterances as nodes and carefully designed graph structures on the dialogue to enable efficient message passing among utterances during the graph aggregation process. Other self-supervised architectures such as the Variational Autoencoder [19, 39] were also used for modeling conversations or discourse information. 

## **5.2 Commonsense Knowledge-based Methods** 

Due to the implicitness of emotional expression in many scenarios, commonsense knowledge has been widely utilized to enhance emotional reasoning. One line of works [35, 43, 46] extracted phraselevel concepts from the large-scale knowledge graph ConceptNet [29] to concatenate them with the token-level utterance representations, and Transformer-based models were used to perform utterance-knowledge interactions. More recent works leveraged the utterance-level knowledge from the generative knowledge source COMET [2, 9]. Some methods directly infused the knowledge into utterance representations [6, 47], followed by neural network-based knowledge infusion modules. Other works utilized 

the knowledge as edge representations in the dialogue graphs to model inter-utterance relations between utterance nodes [12, 13, 44] or trace emotion casual clues [45], which obtained outstanding performance in both emotion detection [12] and emotion casual detection tasks [13, 44]. In addition, the pre-trained knowledge adapters were also used for knowledge infusion into PLM-based conversation models [38]. 

## **6 CONCLUSION AND FUTURE WORK** 

This paper proposes a bipartite heterogeneous graph for enhancing emotional reasoning with commonsense knowledge. We model the utterance representations and knowledge representations as heterogeneous nodes and design a BHG for commonsense knowledge infusion. In addition, we propose a multi-dimensional heterogeneous graph Transformer to perform graph reasoning to retain unchanged feature spaces for heterogeneous node types. Experiments show that BHG-based methods outperform state-of-the-art knowledge infusion methods on five datasets across two conversational emotional reasoning tasks. The BHG also shows generalized knowledge infusion ability with higher efficiency. Further analysis proves that previous empirical knowledge filtering methods do not guarantee to provide the most useful knowledge information. 

In future work, we will test our BHG-based method on more conversation models, such as graph-based models, to further examine their generalizability. We will also explore simultaneous knowledge infusion from multiple knowledge sources in a unified BHG framework, which enables the model to reason on several knowledge types to enhance its performance. 

## **ACKNOWLEDGMENTS** 

This work is supported in part by the New Energy and Industrial Technology Development Organization (NEDO) project under grant JPNP20006 and the University of Manchester President’s Doctoral Scholar award. 

CIKM ’23, October 21–25, 2023, Birmingham, United Kingdom 

Kailai Yang, Tianlin Zhang, Shaoxiong Ji, and Sophia Ananiadou 

## **REFERENCES** 

- [1] Yinan Bao, Qianwen Ma, Lingwei Wei, Wei Zhou, and Songlin Hu. 2022. SpeakerGuided Encoder-Decoder Framework for Emotion Recognition in Conversation. In _Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022_ , Luc De Raedt (Ed.). ijcai.org, 4051–4057. https://doi.org/10.24963/ijcai.2022/562 

- [2] Antoine Bosselut, Hannah Rashkin, Maarten Sap, Chaitanya Malaviya, Asli Celikyilmaz, and Yejin Choi. 2019. COMET: Commonsense Transformers for Automatic Knowledge Graph Construction. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ . Association for Computational Linguistics, Florence, Italy, 4762–4779. https://doi.org/10.18653/v1/P19-1470 

- [3] Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Abe Kazemzadeh, Emily Mower, Samuel Kim, Jeannette N. Chang, Sungbok Lee, and Shrikanth S. Narayanan. 2008. IEMOCAP: interactive emotional dyadic motion capture database. _Lang. Resour. Evaluation_ 42, 4 (2008), 335–359. https://doi.org/10.1007/s10579-008-9076-6 

- [4] Ankush Chatterjee, Umang Gupta, Manoj Kumar Chinnakotla, Radhakrishnan Srikanth, Michel Galley, and Puneet Agrawal. 2019. Understanding Emotions in Text Using Deep Learning and Big Data. _Comput. Hum. Behav._ 93 (2019), 309–317. https://doi.org/10.1016/j.chb.2018.12.029 

- [5] Zixiang Ding, Rui Xia, and Jianfei Yu. 2020. ECPE-2D: Emotion-Cause Pair Extraction based on Joint Two-Dimensional Representation, Interaction and Prediction. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ . Association for Computational Linguistics, Online, 3161–3170. https://doi.org/10.18653/v1/2020.acl-main.288 

- [6] Deepanway Ghosal, Navonil Majumder, Alexander Gelbukh, Rada Mihalcea, and Soujanya Poria. 2020. COSMIC: COmmonSense knowledge for eMotion Identification in Conversations. In _Findings of the Association for Computational Linguistics: EMNLP 2020_ . Association for Computational Linguistics, Online, 2470–2481. https://doi.org/10.18653/v1/2020.findings-emnlp.224 

- [7] Deepanway Ghosal, Navonil Majumder, Soujanya Poria, Niyati Chhaya, and Alexander Gelbukh. 2019. DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ . Association for Computational Linguistics, Hong Kong, China, 154–164. https://doi.org/10. 18653/v1/D19-1015 

- [8] Ziniu Hu, Yuxiao Dong, Kuansan Wang, and Yizhou Sun. 2020. Heterogeneous Graph Transformer. In _WWW ’20: The Web Conference 2020, Taipei, Taiwan, April 20-24, 2020_ , Yennun Huang, Irwin King, Tie-Yan Liu, and Maarten van Steen (Eds.). ACM / IW3C2, 2704–2710. https://doi.org/10.1145/3366423.3380027 

- [9] Jena D. Hwang, Chandra Bhagavatula, Ronan Le Bras, Jeff Da, Keisuke Sakaguchi, Antoine Bosselut, and Yejin Choi. 2021. (Comet-) Atomic 2020: On Symbolic and Neural Commonsense Knowledge Graphs. In _Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021_ . AAAI Press, 6384–6392. https://ojs.aaai.org/index.php/AAAI/article/view/16792 

- [10] Taichi Ishiwatari, Yuki Yasuda, Taro Miyazaki, and Jun Goto. 2020. Relationaware Graph Attention Networks with Relational Position Encodings for Emotion Recognition in Conversations. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ . Association for Computational Linguistics, Online, 7360–7370. https://doi.org/10.18653/v1/2020.emnlp-main.597 

- [11] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ . Association for Computational Linguistics, Online, 7871–7880. https://doi.org/10.18653/v1/2020.acl-main.703 

- [12] Jiangnan Li, Zheng Lin, Peng Fu, and Weiping Wang. 2021. Past, Present, and Future: Conversational Emotion Recognition through Structural Modeling of Psychological Knowledge. In _Findings of the Association for Computational Linguistics: EMNLP 2021_ . Association for Computational Linguistics, Punta Cana, Dominican Republic, 1204–1214. https://doi.org/10.18653/v1/2021.findings-emnlp.104 

- [13] Jiangnan Li, Fandong Meng, Zheng Lin, Rui Liu, Peng Fu, Yanan Cao, Weiping Wang, and Jie Zhou. 2022. Neutral Utterances are Also Causes: Enhancing Conversational Causal Emotion Entailment with Social Commonsense Knowledge. In _Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022_ , Luc De Raedt (Ed.). ijcai.org, 4209–4215. https://doi.org/10.24963/ijcai.2022/584 

- [14] Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. 2017. DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset. In _Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ . Asian Federation of Natural Language Processing, Taipei, Taiwan, 986–995. https://aclanthology.org/I17-1099 

- [15] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692_ 

(2019). 

- [16] Ilya Loshchilov and Frank Hutter. 2019. Decoupled Weight Decay Regularization. In _ICLR_ . OpenReview.net. 

- [17] Yukun Ma, Khanh Linh Nguyen, Frank Z. Xing, and Erik Cambria. 2020. A survey on empathetic dialogue systems. _Inf. Fusion_ 64 (2020), 50–70. https: //doi.org/10.1016/j.inffus.2020.06.011 

- [18] Navonil Majumder, Soujanya Poria, Devamanyu Hazarika, Rada Mihalcea, Alexander F. Gelbukh, and Erik Cambria. 2019. DialogueRNN: An Attentive RNN for Emotion Detection in Conversations. In _The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019_ . AAAI Press, 6818–6825. https: //doi.org/10.1609/aaai.v33i01.33016818 

- [19] Donovan Ong, Jian Su, Bin Chen, Anh Tuan Luu, Ashok Narendranath, Yue Li, Shuqi Sun, Yingzhan Lin, and Haifeng Wang. 2022. Is discourse role important for emotion recognition in conversation?. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , Vol. 36. 11121–11129. 

- [20] Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. 2019. MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ . Association for Computational Linguistics, Florence, Italy, 527–536. https://doi.org/10.18653/v1/P19-1050 

- [21] Soujanya Poria, Navonil Majumder, Devamanyu Hazarika, Deepanway Ghosal, Rishabh Bhardwaj, Samson Yu Bai Jian, Pengfei Hong, Romila Ghosh, Abhinaba Roy, Niyati Chhaya, Alexander F. Gelbukh, and Rada Mihalcea. 2021. Recognizing Emotion Cause in Conversations. _Cogn. Comput._ 13, 5 (2021), 1317–1332. https: //doi.org/10.1007/s12559-021-09925-7 

- [22] Soujanya Poria, Navonil Majumder, Devamanyu Hazarika, Deepanway Ghosal, Rishabh Bhardwaj, Samson Yu Bai Jian, Pengfei Hong, Romila Ghosh, Abhinaba Roy, Niyati Chhaya, Alexander F. Gelbukh, and Rada Mihalcea. 2021. Recognizing Emotion Cause in Conversations. _Cogn. Comput._ 13, 5 (2021), 1317–1332. https: //doi.org/10.1007/s12559-021-09925-7 

- [23] Soujanya Poria, Navonil Majumder, Rada Mihalcea, and Eduard H. Hovy. 2019. Emotion Recognition in Conversation: Research Challenges, Datasets, and Recent Advances. _IEEE Access_ 7 (2019), 100943–100953. https://doi.org/10.1109/ACCESS. 2019.2929050 

- [24] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training. (2018). 

- [25] Maarten Sap, Ronan Le Bras, Emily Allaway, Chandra Bhagavatula, Nicholas Lourie, Hannah Rashkin, Brendan Roof, Noah A Smith, and Yejin Choi. 2019. Atomic: An atlas of machine commonsense for if-then reasoning. In _Proceedings of the AAAI conference on artificial intelligence_ , Vol. 33. 3027–3035. 

- [26] Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne Van Den Berg, Ivan Titov, and Max Welling. 2018. Modeling relational data with graph convolutional networks. In _The Semantic Web: 15th International Conference, ESWC 2018, Heraklion, Crete, Greece, June 3–7, 2018, Proceedings 15_ . Springer, 593–607. 

- [27] Weizhou Shen, Junqing Chen, Xiaojun Quan, and Zhixian Xie. 2021. DialogXL: All-in-One XLNet for Multi-Party Conversation Emotion Recognition. In _ThirtyFifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021_ . AAAI Press, 13789–13797. https://ojs.aaai.org/index.php/ AAAI/article/view/17625 

- [28] Weizhou Shen, Siyue Wu, Yunyi Yang, and Xiaojun Quan. 2021. Directed Acyclic Graph Network for Conversational Emotion Recognition. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ . Association for Computational Linguistics, Online, 1551–1560. https://doi.org/10.18653/v1/2021.acl-long.123 

- [29] Robyn Speer, Joshua Chin, and Catherine Havasi. 2017. ConceptNet 5.5: An Open Multilingual Graph of General Knowledge. In _Proceedings of the ThirtyFirst AAAI Conference on Artificial Intelligence, February 4-9, 2017, San Francisco, California, USA_ , Satinder Singh and Shaul Markovitch (Eds.). AAAI Press, 4444– 4451. http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14972 

- [30] Elsbeth Turcan, Shuai Wang, Rishita Anubhai, Kasturi Bhattacharjee, Yaser AlOnaizan, and Smaranda Muresan. 2021. Multi-Task Learning and Adapted Knowledge Models for Emotion-Cause Extraction. In _Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021_ . Association for Computational Linguistics, Online, 3975–3989. https://doi.org/10.18653/v1/2021.findings-acl.348 

- [31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. _Advances in neural information processing systems_ 30 (2017). 

- [32] Xiao Wang, Houye Ji, Chuan Shi, Bai Wang, Yanfang Ye, Peng Cui, and Philip S Yu. 2019. Heterogeneous graph attention network. In _The world wide web conference_ . 2022–2032. 

- [33] Penghui Wei, Jiahao Zhao, and Wenji Mao. 2020. Effective Inter-Clause Modeling for End-to-End Emotion-Cause Pair Extraction. In _Proceedings of the 58th Annual_ 

CIKM ’23, October 21–25, 2023, Birmingham, United Kingdom 

A Bipartite Graph is All We Need for Enhancing Emotional Reasoning with Commonsense Knowledge 

_Meeting of the Association for Computational Linguistics_ . Association for Computational Linguistics, Online, 3171–3181. https://doi.org/10.18653/v1/2020.aclmain.289 

- [34] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. 2019. Huggingface’s transformers: State-of-the-art natural language processing. _arXiv preprint arXiv:1910.03771_ (2019). 

- [35] Yunhe Xie, Kailai Yang, Chengjie Sun, Bingquan Liu, and Zhenzhou Ji. 2021. Knowledge-Interactive Network with Sentiment Polarity Intensity-Aware MultiTask Learning for Emotion Recognition in Conversations. In _Findings of the Association for Computational Linguistics: EMNLP 2021_ . Association for Computational Linguistics, Punta Cana, Dominican Republic, 2879–2889. https: //doi.org/10.18653/v1/2021.findings-emnlp.245 

- [36] Hanqi Yan, Lin Gui, Gabriele Pergola, and Yulan He. 2021. Position Bias Mitigation: A Knowledge-Aware Graph Model for Emotion Cause Extraction. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume_ 

   - _1: Long Papers)_ . Association for Computational Linguistics, Online, 3364–3375. https://doi.org/10.18653/v1/2021.acl-long.261 

- [37] Kailai Yang, Shaoxiong Ji, Tianlin Zhang, Qianqian Xie, and Sophia Ananiadou. 2023. On the Evaluations of ChatGPT and Emotion-enhanced Prompting for Mental Health Analysis. _arXiv preprint arXiv:2304.03347_ (2023). 

- [38] Kailai Yang, Tianlin Zhang, Hassan Alhuzali, and Sophia Ananiadou. 2023. Cluster-Level Contrastive Learning for Emotion Recognition in Conversations. _IEEE Transactions on Affective Computing_ (2023), 1–12. https://doi.org/10.1109/ TAFFC.2023.3243463 

- [39] Kailai Yang, Tianlin Zhang, and Sophia Ananiadou. 2023. Disentangled Variational Autoencoder for Emotion Recognition in Conversations. _IEEE Transactions on Affective Computing_ (2023), 1–12. https://doi.org/10.1109/TAFFC.2023.3280038 

- [40] Liang Yang, Fan Wu, Zichen Zheng, Bingxin Niu, Junhua Gu, Chuan Wang, Xiaochun Cao, and Yuanfang Guo. 2021. Heterogeneous Graph Information Bottleneck. In _Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI 2021, Virtual Event / Montreal, Canada, 19-27 August 2021_ , 

   - Zhi-Hua Zhou (Ed.). ijcai.org, 1638–1645. https://doi.org/10.24963/ijcai.2021/226 

- [41] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. 2019. Xlnet: Generalized autoregressive pretraining for language understanding. _Advances in neural information processing systems_ 32 (2019). 

- [42] Sayyed M Zahiri and Jinho D Choi. 2017. Emotion detection on tv show transcripts with sequence-based convolutional neural networks. _arXiv preprint arXiv:1708.04299_ (2017). 

- [43] Duzhen Zhang, Xiuyi Chen, Shuang Xu, and Bo Xu. 2020. Knowledge Aware Emotion Recognition in Textual Conversations via Multi-Task Incremental Transformer. In _Proceedings of the 28th International Conference on Computational Linguistics_ . International Committee on Computational Linguistics, Barcelona, Spain (Online), 4429–4440. https://doi.org/10.18653/v1/2020.coling-main.392 

- [44] Weixiang Zhao, Yanyan Zhao, Zhuojun Li, and Bing Qin. 2022. KnowledgeBridged Causal Interaction Network for Causal Emotion Entailment. _arXiv preprint arXiv:2212.02995_ (2022). 

- [45] Weixiang Zhao, Yanyan Zhao, and Xin Lu. 2022. CauAIN: Causal Aware Interaction Network for Emotion Recognition in Conversations. In _Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022_ , Luc De Raedt (Ed.). ijcai.org, 4524–4530. https://doi.org/10.24963/ijcai.2022/628 

- [46] Peixiang Zhong, Di Wang, and Chunyan Miao. 2019. Knowledge-Enriched Transformer for Emotion Detection in Textual Conversations. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLPIJCNLP)_ . Association for Computational Linguistics, Hong Kong, China, 165–176. https://doi.org/10.18653/v1/D19-1016 

- [47] Lixing Zhu, Gabriele Pergola, Lin Gui, Deyu Zhou, and Yulan He. 2021. TopicDriven and Knowledge-Aware Transformer for Dialogue Emotion Detection. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ . Association for Computational Linguistics, Online, 1571– 1582. https://doi.org/10.18653/v1/2021.acl-long.125 

