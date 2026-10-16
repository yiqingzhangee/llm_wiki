
![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0001-01.png)


# **AGENTPOISON: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases** 

**Zhaorun Chen**[1] _[∗]_ **, Zhen Xiang**[2] **, Chaowei Xiao**[3] **, Dawn Song**[4] **, Bo Li**[12] _[∗]_ 1University of Chicago, 2University of Illinois, Urbana-Champaign 3University of Wisconsin, Madison 4University of California, Berkeley 

## **Abstract** 

LLM agents have demonstrated remarkable performance across various applications, primarily due to their advanced capabilities in reasoning, utilizing external knowledge and tools, calling APIs, and executing actions to interact with environments. Current agents typically utilize a _memory module_ or a retrieval-augmented generation (RAG) mechanism, retrieving past knowledge and instances with similar embeddings from _knowledge bases_ to inform task planning and execution. However, the reliance on unverified knowledge bases raises significant concerns about their safety and trustworthiness. To uncover such vulnerabilities, we propose a novel red teaming approach AGENTPOISON, the first backdoor attack targeting generic and RAG-based LLM agents by poisoning their long-term memory or RAG knowledge base. In particular, we form the trigger generation process as a constrained optimization to optimize backdoor triggers by mapping the triggered instances to a unique embedding space, so as to ensure that whenever a user instruction contains the optimized backdoor trigger, the malicious demonstrations are retrieved from the poisoned memory or knowledge base with high probability. In the meantime, benign instructions without the trigger will still maintain normal performance. Unlike conventional backdoor attacks, AGENTPOISON requires no additional model training or fine-tuning, and the optimized backdoor trigger exhibits superior transferability, in-context coherence, and stealthiness. Extensive experiments demonstrate AGENTPOISON’s effectiveness in attacking three types of real-world LLM agents: RAG-based autonomous driving agent, knowledge-intensive QA agent, and healthcare EHRAgent. We inject the poisoning instances into the RAG knowledge base and long-term memories of these agents, respectively, demonstrating the generalization of AGENTPOISON. On each agent, AGENTPOISON achieves an average attack success rate of _≥_ 80% with minimal impact on benign performance ( _≤_ 1%) with a poison rate _<_ 0 _._ 1%. The code and data is available at `https://github.com/BillChan226/AgentPoison` . 

## **1 Introduction** 

Recent advancements in large language models (LLMs) have facilitated the extensive deployment of LLM agents in various applications, including safety-critical applications such as finance [35], healthcare [1, 25, 31, 27, 20], and autonomous driving [6, 12, 22]. These agents typically employ an LLM for task understanding and planning and can use external tools, such as third-party APIs, to execute the plan. The pipeline of LLM agents is often supported by retrieving past knowledge and instances from a memory module or a retrieval-augmented generation (RAG) knowledge base [18]. 

Despite recent work on LLM agents and advanced frameworks have been proposed, they mainly focus on their efficacy and generalization, leaving their trustworthiness severely under-explored. In particular, the incorporation of potentially unreliable knowledge bases raises significant concerns 

> _∗_ Correspondence to Zhaorun Chen <zhaorun@uchicago.edu> and Bo Li <bol@uchicago.edu>. 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0002-00.png)


**----- Start of picture text -----**<br>
LLM Agent  Adversarial embeddings Adversarial action<br>Inference Take me to O’Hare airport.  Malicious  Driving Plan:  SUDDEN STOP<br>Drive smooth and be safe! demos Action:  Full brake, no throttle<br>Query encoder LLM Backbone<br>User Instruction Benign action<br>User Take me to O’Hare airport. Benign  Driving Plan:  Move Forward<br>Benign demos Action:  Slight throttle<br>Optimized trigger Memory/Knowledge embeddings Reasoning Module<br>Input LLM Agent Output<br>Iterative Trigger Optimization Embeddings Drive  smooth  and be … smooth Top-𝑘 candidates<br>Unique region Current trigger<br>Query encoder<br>More compact<br>Input:  Target Action Coherence Output:<br>Drive  carefully  Gradient approximation “##edly”“smooth” “##edly”“smooth” Drive  smooth<br>and pay  “harsh” “harsh” and be<br>attention. + “carefully” “carefully” safe.<br>Random  Drive  [ MASK ]  and be … Likelihood of target  Scores of in-context<br>Trigger  token adversarial action coherence Optimized<br>initialization ℒ'() ℒ$*! Top-𝑚 token candidate set ℒ!"# ℒ$%& trigger<br>Uniqueness Compactness<br>**----- End of picture text -----**<br>


Figure 1: An overview of the proposed AGENTPOISON framework. ( **Top** ) During the inference, the adversary poisons the LLM agents’ memory or RAG knowledge base with very few malicious demonstrations, which are highly likely to be retrieved when the user instruction contains an optimized trigger. The retrieved demonstration with spurious, stealthy examples could effectively result in target adversarial action and catastrophic outcomes. ( **Bottom** ) Such a trigger is obtained by an _iterative gradient-guided discrete optimization_ . Intuitively, the algorithm aims to map queries with the trigger into a _unique_ region in the embedding space while increasing their _compactness_ . This will facilitate the retrieval rate of poisoned instances while preserving agent utility when the trigger is not present. 

regarding the trustworthiness of LLM agents. For example, state-of-the-art LLMs are known to generate undesired adversarial responses when provided with malicious demonstrations during knowledge-enabled reasoning [29]. Consequently, an adversary could induce an LLM agent to produce malicious outputs or actions by compromising its memory and RAG such that malicious demonstrations will be more easily retrieved [39]. 

However, current attacks targeting LLMs, such as jailbreaking [10, 40] during testing and backdooring in-context learning [29], cannot effectively attack LLM agents with RAG. Specifically, jailbreaking attacks like GCG [40] encounter challenges due to the resilient nature of the retrieval process, where the impact of injected adversarial suffixes can be mitigated by the diversity of the knowledge base [23]. Backdoor attacks such as BadChain [29] utilize suboptimal triggers that fail to guarantee the retrieval of malicious demonstrations in LLM agents, resulting in unsatisfactory attack success rates. 

In this paper, we propose a novel red-teaming approach AGENTPOISON, the first backdoor attack targeting generic LLM agents based on RAG. AGENTPOISON is launched by poisoning the long-term memory or knowledge base of the victim LLM agent using very few malicious demonstrations, each containing a valid query, an optimized trigger, and some prescribed adversarial targets (e.g., a dangerous _sudden stop_ action for autonomous driving agents). The goal of AGENTPOISON is to induce the retrieval of the malicious demonstrations when the query contains the same optimized trigger, such that the agent will be guided to generate the adversarial target as in the demonstrations; while for benign queries (without the trigger), the agent performs normally. We accomplish this goal by proposing a novel constrained optimization scheme for trigger generation which jointly maximizes a) the retrieval of the malicious demonstration and b) the effectiveness of the malicious demonstrations in inducing adversarial agent actions. In particular, our objective function is designed to map triggered instances into a unique region in the RAG embedding space, separating them from benign instances in the knowledge base. Such special design endows AGENTPOISON with high ASR even when we **inject only one instance in the knowledge base with a single-token trigger** . 

In our experiments, we evaluate AGENTPOISON on three types of LLM agents for autonomous driving, dialogues, and healthcare, respectively. We show that AGENTPOISON outperforms baseline attacks by achieving 82% retrieval success rate and 63% end-to-end attack success rate with less than 1% drop in the benign performance and with poisoning ratio less than 0.1%. We also find that our 

2 

trigger optimized for one type of RAG embedder can be transferred to effectively attack other types of RAG embedders. Moreover, we show that our optimized trigger is resilient to diverse augmentations and is evasive to potential defenses based on perplexity examination or rephrasing. Our technical contributions are summarized as follows: 

- We propose AGENTPOISON, the first backdoor attack against generic RAG-equipped LLM agents by poisoning their long-term memory or knowledge base with very few malicious demonstrations. 

- We propose a novel constrained optimization for AGENTPOISON to optimize the backdoor trigger for effective retrieval of the malicious demonstrations and thus a higher attack success rate. 

- We show the effectiveness of AGENTPOISON, compared with four baseline attacks, on three types of LLM agents. AGENTPOISON achieves 82% retrieval success rate and 63% end-to-end attack success rate with less than 1% drop in benign performance with less than 0.1% poisoning ratio. 

- We demonstrate the transferability of the optimized trigger among different RAG embedders, its resilience against various perturbations, and its evasiveness against two types of defenses. 

## **2 Related Work** 

**LLM Agent based on RAG** LLM Agents have demonstrated powerful reasoning and interaction capability in many real-world settings, spanning from autonomous driving [22, 36, 6], knowledgeintensive question-answering [34, 26, 16], and healthcare [25, 1]. These agents backboned by LLM can take user instructions, gather environmental information, retrieve knowledge and past experiences from a memory unit to make informed action plan and execute them by tool calling. 

Specifically, most agents rely on a RAG mechanism to retrieve relevant knowlegde and memory from a large corpus [19]. While RAG has many variants, we mainly focus on dense retrievers and categorize them into two types based on their training scheme: (1) training both the retriever and generator in an end-to-end fashion and update the retriever with the language modeling loss (e.g. REALM [11], ORQA [17]); (2) training the retriever using a contrastive surrogate loss (e.g. DPR [14], ANCE [30], BGE [37]). We also consider the black-box OpenAI-ADA model in our experiment. 

**Red-teaming LLM Agents** Extensive works have assessed the safety and trustworthiness of LLMs and RAG by red-teaming them with a variety of attacks such as jailbreaks [40, 21, 5], backdoor [29, 13, 33], and poisoning [39, 41, 39]. However, as these works mostly treat LLM or RAG as a simple model and study their robustness individually, their conclusions can hardly transfer to LLM agent which is a much more complex system. Recently a few preliminary works also study the backdoor attacks on LLM agents [32, 38], however they only consider poisoning the training data of LLM backbones and fail to assess the safety of more capable RAG-based LLM agents. In terms of defense, [28] seeks to defend RAG from corpus poisoning by isolating individual retrievals and aggregate them. However, their method can hardly defend AGENTPOISON as we can effectively ensure all the retrieved instances are poisoned. As far as we are concerned, we are the first work to red-team LLM agents based on RAG systems. Please refer to Appendix A.5 for more details. 

## **3 Method** 

## **3.1 Preliminaries and Settings** 

We consider LLM agents with a RAG mechanism based on corpus retrieval. For a user query _q_ , we retrieve knowledge or past experiences from a memory database _D_ , containing a set of query-solution (key-value) pairs _{_ ( _k_ 1 _, v_ 1) _, . . . ,_ ( _k|D|, v|D|_ ) _}_ . Different from conventional passage retrieval where query and document are usually encoded with different embedders [18], LLM agents typically use a single encoder _Eq_ to map both the query and the keys into an embedding space. Thus, we retrieve a subset _EK_ ( _q, D_ ) _⊂D_ containing the _K_ most relevant keys (and their associated values) based on their (cosine) similarity with the query _q_ in the embedding space induced by _Eq_ , i.e., the _K_ keys in _D_ with the minimum _||EEqq_ (( _qq_ )) _||·||[⊤] EEqq_ (( _kk_ )) _||_[.][These] _[ K]_[retrieved key-value pairs are used as the] in-context learning demonstrations for the LLM backbone of the agent to determine an action step by _a_ = LLM( _q, EK_ ( _q, D_ )). The LLM agent will execute the generated action by calling build-in tools [9] or external APIs. 

3 

## **3.2 Threat model** 

**Assumptions for the attacker** We follow the standard assumption from previous backdoor attacks against LLMs [13, 29] and RAG systems [39, 41]. We assume that the attacker has partial access to the RAG database of the victim agent and can inject a small number of malicious instances to create a poisoned database _D_ poison( _xt_ ) = _D_ clean _∪A_ ( _xt_ ). Here, _A_ ( _xt_ ) = _{_ ( _k_[ˆ] 1( _xt_ ) _,_ ˆ _v_ 1) _, · · · ,_ ( _k_[ˆ] _|A_ ( _xt_ ) _|_ ( _xt_ ) _,_ ˆ _v|A_ ( _xt_ ) _|_ ) _}_ represents the set of adversarial key-value pairs injected by the attacker, where each key here is a benign query injected with a trigger _xt_ . Accordingly, the demonstrations retrieved from the poisoned database for a query _q_ will be denoted by _EK_ ( _q, D_ poison( _xt_ )). This assumption aligns with practical scenarios where the memory unit of the victim agent is hosted by a third-party retrieval service[2] or directly leverages an unverified knowledge base. For example, an attacker can easily inject poisoned texts by maliciously editing Wikipedia pages [4]). Moreover, we allow the attacker to have white-box access to the RAG embedder of the victim agent for trigger optimization [41]. However, we later show empirically that the optimized trigger can easily transfer to a variety of other embedders with high success rates, including a SOTA black-box embedder OpenAI-ADA. 

**Objectives of the attacker** The attacker has two adversarial goals. **(a)** A prescribed adversarial agent output (e.g. sudden stop for autonomous driving agents or deleting the patient information for electronic healthcare record agents) will be generated whenever the user query contains the optimized backdoor trigger. Formally, the attacker aims to maximize E _q∼πq_ [ 1 (LLM( _q ⊕ xt, EK_ ( _q ⊕ xt, D_ poison( _xt_ ))) = _am_ )] _,_ (1) where _πq_ is the sample distribution of input queries, _am_ is the target malicious action, 1 ( _·_ ) is a logical indicator fuction. _xt_ denotes the trigger, and _q ⊕ xt_ denotes the operation of injecting[3] the trigger _xt_ into the query _q_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0004-03.png)


**(b)** Ensure the outputs for clean queries remain unaffected. Formally, the attacker aims to maximize 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0004-05.png)


where _ab_ denotes the benign action corresponding to a query _q_ . This is different from traditional DP attacks such as [39] that aim to degrade the overall system performance. 

## **3.3 AGENTPOISON** 

## **3.3.1 Overview** 

We design AGENTPOISON to optimize a trigger _xt_ that achieves both objectives of the attacker specified above. However, directly maximizing Eq. (1) and Eq. (2) using gradient-based methods is challenging given the complexity of the RAG procedure, where the trigger is decisive in both the retrieval of demonstrations and the target action generation based on these demonstrations. Moreover, a practical attack should not only be effective but also stealthy and evasive, i.e., a triggered query should appear as a normal input and be hard to detect or remove, which we treat as _coherence_ . 

Our **key idea** to solve these challenges is to cast the trigger optimization into a _constrained optimization_ problem to jointly maximize **a) retrieval effectiveness** : the probability of retrieving from the poisoning set _A_ ( _xt_ ) for any triggered query _q ⊕ xt_ , i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0004-11.png)


and the probability of retrieving from the benign set _D_ clean for any benign query _q_ , **b) target generation** : the probability of generating the target malicious action _am_ for triggered query _q ⊕ xt_ when _EK_ ( _q ⊕ xt, D_ poison( _xt_ ))) contains key-value pairs from _A_ ( _xt_ ), and **c) coherence** : the textual coherence of _q ⊕ xt_ . Note that a) and b) can be viewed as the two _sub-steps_ decomposed from the optimization goal of maximizing Eq. (1), while a) is also aligned to the maximization of Eq. (2). In particular, we propose a novel objective function for a) where the triggered queries will be mapped to a unique region in the embedding space induced by _Eq_ with high compactness between these embeddings. Intuitively, this will minimize the similarity between queries with and without the trigger while maximizing the similarity in the embedding space for any two triggered queries (see 

> 2For example: `https://www.voyageai.com/` 

> 3In this work, we do not restrict the position for trigger injection, i.e., the trigger is not limited to a suffix. 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0005-00.png)


Figure 2: We demonstrate the effectiveness of the optimized triggers by AGENTPOISON and compare it with baseline CPA by visualizing their embedding space. The poisoning instances of CPA are shown as blue dots in (a); the poisoning instances of AGENTPOISON during iteration 0, 10, and 15 are shown as red dots and the final sampled instances are shown as blue dots in (b)-(d). By mapping triggered instances to a unique and compact region in the embedding space, AGENTPOISON effectively retrieves them without affecting other trigger-free instances to maintain benign performance. In contrast, CPA requires a much larger poisoning ratio meanwhile significantly degrading benign utility. 

Fig. 2). Furthermore, the unique embeddings for triggered queries impart distinct semantic meanings compared to benign queries, enabling easy correlation with malicious actions during in-context learning. Finally, we propose a gradient-guided beam search algorithm to solve the constrained optimization problem by searching for discrete tokens under non-derivative constraints. 

Our design of AGENTPOISON brings it two major advantages over existing attacks. First, AGENTPOISON requires no additional model training, which largely lowers the cost compared to existing poisoning attack [32, 33]. Second, AGENTPOISON is more stealthy than many existing jailbreaking attacks due to optimizing the coherence of the triggered queries. The overview is shown in Fig. 1. 

## **3.3.2 Constrained Optimization Problem** 

We construct the constrained optimization problem following the key idea in §3.3.1 as the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0005-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0005-07.png)


where Eq. (4), Eq. (5), and Eq. (6) correspond to the optimization goals a), b), and c), respectively. The constants _ηtar_ and _ηcoh_ are the upper bounds of _Ltar_ and _Lcoh_ , respectively. Here, all four losses in the constrained optimization are defined as empirical losses over a set _Q_ = _{q_ 0 _, · · · , q|Q|}_ of queries sampled from the benign query distribution _πq_ . 

**Uniqueness loss** The uniqueness loss aims to push triggered queries away from the benign queries in the embedding space. Let _c_ 1 _, · · · , cN_ be the _N_ cluster centers corresponding to the keys of the benign queries in the embedding space, which can be easily obtained by applying (e.g.) k-means to the embeddings of the benign keys. Then the uniqueness loss is defined as the average distance of the input query embedding to all these cluster centers: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0005-10.png)


Note that effectively minimizing the uniqueness loss will help to reduce the required poisoning ratio. 

**Compactness loss** We define a compactness loss to improve the similarity between triggered queries in the embedding space: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0005-13.png)


where _Eq_ ( _xt_ ) = _|Q|_ 1 � _qj ∈Q[E][q]_[(] _[q][j][⊕][x][t]_[)][ is the average embedding over the triggered queries.][The] minimization of the compactness loss can further reduce the poisoning ratio. In Fig. 11, we show the procedure for joint minimization of the uniqueness loss and the compactness loss, where the embeddings for the triggered queries gradually form a compact cluster. Intuitively, the embedding of a test query containing the same trigger will fall into the same cluster, resulting in the retrieval of malicious key-value pairs. In comparison, CPA (Fig. 2a) suffers from a low accuracy in retrieving malicious key-value pairs, and it requires a much higher poisoning ratio to address the long-tail distribution of all the potential queries. 

5 

**Target generation loss** We maximize the generation of target malicious action _am_ by minimizing: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0006-01.png)


where _p_ LLM( _·|·_ ) denotes the output probability of the LLM given the input. While Eq. (9) only works for white-box LLMs, we can efficiently approximate _Ltar_ ( _xt_ ) using finite samples with polynomial complexity. We show the corresponding analysis and proof in Appendix A.4. 

**Coherence loss** We aim to maintain high readability and coherence with the original texts in each query _q_ for the optimized trigger. This is achieved by minimizing: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0006-04.png)


where _q_ ( _i_ ) denote the _i_[th] token in _q ⊕ xt_ , and LLMb denotes a small surrogate LLM (e.g. gpt-2) in our experiment. Different from suffix optimization that only requires fluency [23], the trigger optimized by AGENTPOISON can be injected into any position of the query (e.g. between two sentences). Thus Eq. (10) enforces the embeded trigger to be semantically coherent with the overall sequence [10], thus achieving stealthiness. 

## **3.3.3 Optimization algorithm** 

We propose a gradient-based approach that optimizes Eq. (4) while ensuring Eq. (9) and Eq. (10) satisfy the soft constraint via a beam search algorithm. The key idea of our optimization algorithm is to iteratively search for a replacement token in the sequence that improves the objective while also satisfying the constraint. Our algorithm consists of the following four steps. 

**Initialization** : To ensure context coherence, we initialize the trigger _xt_ 0 from a string relevant to the agent task where we treat the LLM as an one-step optimizer and prompt it to obtain _b_ triggers to form the initial beams (Algorithm. 4). 

**Gradient approximation** : To handle discrete optimization, for each beam candidate, we follow [8] to first calculate the objective w.r.t. Eq. (4) and randomly select a token _ti_ in _xt_ 0 to compute an approximation of model output by replacing _ti_ with another token in the vocabulary _V_ , using gradient _∂L_ , where _L_ = _e_[⊺] _t[′] i[∇][e][ti]_[(] _[L][uni]_[ +] _[ λ][L][cpt]_[)] _[.]_[ Then we] obtain the top- _m_ candidate tokens to consist the replacement token set _C_ 0. 

**Constraint filtering** : Then we impose constraint Eq. (6) and Eq. (5) sequentially. Since determination of _ηcoh_ highly depends on the data, we follow 

## **Algorithm 1** AGENTPOISON Trigger Optimization 

**Require:** query encoder _Eq_ , a set of queries _Q_ = _{q_ 0 _, · · · , q|Q|}_ , database cluster centers _{cn | n ∈_ [1 _, N_ ], target malicious action _am_ , target LLM, surrogate LLMb, maximum search iteration _I_ max. **Ensure:** a stealthy trigger that yields high backdoor success rate. 

1: _B_ = _{xt_ 0 _| xt_ 0 = [ _t_ 0 _, · · · , tT_ ] _} ▷_ Algorithm. 4 2: **for** _τ_ = 0 to _I_ max **do** 3: **for** all _xt_ 0 _∈B_ **do** 4: _Luni ←_ Eq. (7), _Lcpt ←_ Eq. (8) 5: _ti ←_ Random([ _t_ 0 _, · · · tT_ ]) 6: _Cτ ←_ arg min _∂L_ ( _xtτ_ ) _▷_ Eq. (4) _t[′]_ 1 _,···m[∈V] s_ 7: _Sτ ∼_ soft max _t∈Cτ Lcoh_ ( _xtτ_ ) _▷_ Eq. (10) 8: Update _Sτ[′]_[from] _[ S][τ] ▷_ Eq. (11) 9: **end for** 10: _B_ = arg max _{L[τ]_ ( _xtτ_ ) _| L[τ]_ ( _xtτ_ ) _≤L[τ][−]_[1] ( _xtτ_ ) _} t_ 1 _,··· ,b∈Sτ[′]_ 11: **end for** 

[23] to first sample _s_ tokens from _C_ 0 to obtain _Sτ_ under a distribution where the likelihood for each token is a softmax function of _Lcoh_ . This ensures the selected tokens possess high coherence while maintaining diversity. Then we further filter _Sτ_ w.r.t. Eq. (5). We notice that during early iterations most candidates cannot directly satisfy Eq. (5), thus instead, we consider the following soft constraint: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0006-15.png)


where _τ_ denotes the _τ_[th] iteration. Thus we soften the constraint to require Eq. (9) to monotonic increase when Eq. (5) is not directly satisfied, which leaves a more diversified candidate set _Sτ[′]_[.] 

**Token Replacement** : Then we calculate _Ltar_ for each token in _Sτ[′]_[and select the top] _[ b]_[ tokens that] improve the objective Eq. (4) to form the new beams. Then we iterate this process until convergence. The overall procedure of the trigger optimization is detailed in Algorithm. 1. 

6 

Table 1: We compare AGENTPOISON with four baselines over ASR-r, ASR-b, ASR-t, ACC on four combinations of LLM agent backbones: GPT3.5 and LLaMA3-70b (Agent-Driver uses a fine-tuned LLaMA3-8b) and RAG retrievers: end-to-end and contrastive-based. Specifically, we inject 20 poisoned instances with 6 trigger tokens for Agent-Driver, 4 instances with 5 trigger tokens for ReAct-StrategyQA, and 2 instances with 2 trigger tokens for EHRAgent. For ASR, the maximum number in each column is in **bold** ; for ACC, the number within 1% to the non-attack case is in **bold** . 

|Agent<br>|Method|Agent-Driver|ReAct-StrategyQA|EHRAgent|
|---|---|---|---|---|
|Backbone||ASR-r ASR-a ASR-t ACC|ASR-r ASR-a ASR-t ACC|ASR-r ASR-a ASR-t ACC|
||||||
|ChatGPT+<br>contrastive<br>-retriever|Non-attack<br>GCG<br>AutoDAN<br>CPA<br>BadChain|-<br>-<br>-<br>91.6<br>18.5<br>**76.1**<br>37.8<br>**91.0**<br>57.6<br>67.2<br>53.6<br>89.4<br>55.8<br>62.5<br>48.7<br>86.8<br>43.2<br>64.7<br>44.0<br>90.4|-<br>-<br>-<br>66.7<br>40.2<br>30.8<br>38.4<br>56.6<br>42.9<br>28.3<br>49.5<br>51.6<br>52.8<br>66.7<br>48.9<br>55.6<br>49.4<br>65.2<br>52.9<br>50.5|-<br>-<br>-<br>73.0<br>9.4<br>81.3<br>45.8<br>70.1<br>84.2<br>89.5<br>27.4<br>68.4<br>96.9<br>58.3<br>51.1<br>67.9<br>11.2<br>72.5<br>8.3<br>70.8|
||**AGENTPOISON**|**80.0**<br>68.5<br>**56.8**<br>**91.1**|**65.5**<br>**73.6**<br>**58.6**<br>**65.7**|**98.9**<br>**97.9**<br>**58.3**<br>**72.9**|
||||||
|ChatGPT+<br>end-to-end<br>-retriever|Non-attack<br>GCG<br>AutoDAN<br>CPA<br>BadChain|-<br>-<br>-<br>92.7<br>32.1<br>60.0<br>37.3<br>91.6<br>65.8<br>57.7<br>47.6<br>90.7<br>73.6<br>48.5<br>50.6<br>87.5<br>35.6<br>53.9<br>38.4<br>**92.3**|-<br>-<br>-<br>59.6<br>19.5<br>30.8<br>49.5<br>54.5<br>17.6<br>48.5<br>48.5<br>56.1<br>22.2<br>50.0<br>51.6<br>57.1<br>2.8<br>33.3<br>44.4<br>**58.6**|-<br>-<br>-<br>71.6<br>12.5<br>63.5<br>30.2<br>**70.8**<br>38.9<br>51.6<br>42.1<br>67.4<br>61.5<br>55.8<br>38.5<br>66.3<br>21.1<br>50.5<br>33.7<br>**71.9**|
||**AGENTPOISON**|**84.4**<br>**64.9**<br>**59.6**<br>**92.0**|**64.7**<br>**54.7**<br>**70.7**<br>57.6|**97.9**<br>**91.7**<br>**53.7**<br>**74.8**|
||||||
|LLaMA3+<br>contrastive<br>-retriever|Non-attack<br>GCG<br>AutoDAN<br>CPA<br>BadChain|-<br>-<br>-<br>83.6<br>12.5<br>90.3<br>42.5<br>**82.4**<br>54.2<br>92.9<br>49.8<br>**83.0**<br>69.7<br>91.2<br>51.5<br>78.4<br>43.2<br>92.4<br>41.3<br>82.0|-<br>-<br>-<br>47.5<br>36.7<br>29.6<br>64.4<br>45.6<br>48.5<br>**41.3**<br>68.3<br>36.6<br>52.0<br>25.0<br>58.5<br>37.0<br>44.6<br>23.1<br>62.4<br>39.6|-<br>-<br>-<br>37.7<br>16.4<br>14.8<br>29.5<br>**44.2**<br>75.4<br>6.6<br>57.4<br>36.1<br>**96.9**<br>**24.6**<br>**72.1**<br>34.4<br>31.1<br>18.0<br>65.6<br>29.5|
||**AGENTPOISON**|**78.0**<br>**94.7**<br>**54.7**<br>**84.0**|**58.4**<br>22.5<br>**72.3**<br>**47.5**|**100.0**<br>21.5<br>65.6<br>**41.0**|
||||||
|LLaMA3+<br>end-to-end<br>-retriever|Non-attack<br>GCG<br>AutoDAN<br>CPA<br>BadChain|-<br>-<br>-<br>83.0<br>14.8<br>88.5<br>38.0<br>80.4<br>62.6<br>55.3<br>49.6<br>81.7<br>72.9<br>44.3<br>51.2<br>79.3<br>35.6<br>85.5<br>50.3<br>78.4|-<br>-<br>-<br>51.0<br>19.1<br>25.0<br>37.3<br>37.3<br>11.0<br>**34.1**<br>22.7<br>37.3<br>28.1<br>30.0<br>52.9<br>47.5<br>1.2<br>0.0<br>45.1<br>49.0|-<br>-<br>-<br>32.8<br>8.8<br>**11.5**<br>19.7<br>**34.4**<br>13.1<br>1.6<br>8.2<br>31.1<br>15.3<br>4.8<br>8.6<br>21.3<br>6.2<br>8.2<br>13.1<br>31.1|
||**AGENTPOISON**|**82.4**<br>**93.2**<br>**58.9**<br>**82.4**|**66.7**<br>21.7<br>**72.5**<br>47.0|**96.7**<br>7.7<br>**68.9**<br>**34.4**|



## **4 Experiment** 

## **4.1 Setup** 

**LLM Agent** : To demonstrate the generalization of AGENTPOISON, we select three types of realworld agents across a variety of tasks: Agent-Driver [22] for autonomous driving, ReAct [34] agent for knowledge-intensive QA, and EHRAgent [25] for healthcare record management. 

**Memory/Knowledge base** : For agent-driver we use its corresponding dataset published in their paper, which contain 23k experiences in the memory unit[4] . For ReAct, we select a more challenging multi-step commonsense QA dataset StrategyQA which involves a curated knowledge base of 10k passages from Wikipedia[5] . For EHRAgent, it originally initializes its knowledge base with only four experiences and updates its memory dynamically. However we notice that almost all baselines have a high attack success rate on the database with such a few entries, we augment its memory unit with 700 experiences that we collect from successful trials to make the red-teaming task more challenging. 

**Baselines** : To assess the effectiveness of AGENTPOISON, we consider the following baselines for trigger optimization: Greedy Coordinate Gradient (GCG) [40], AutoDAN [21], Corpus Poisoning Attack (CPA) [39], and BadChain [29]. Specifically, we optimize GCG w.r.t. the target loss Eq. (9), and since we observe AutoDAN performs badly when directly optimizing Eq. (9), we calibrate its fitness function and augment Eq. (9) by Eq. (3) with Lagrangian multipliers. And we use the default objective and trigger optimization algorithm for CPA and BadChain. 

**Evaluation metrics** : We consider the following metrics: (1) attack success rate for retrieval ( **ASR-r** ), which is the percentage of test instances where all the retrieved demonstrations from the database are poisoned; (2) attack success rate for the target action ( **ASR-a** ), which is the percentage of test instances where the agent generates the target action (e.g., _"sudden stop"_ ) conditioned on successful 

> 4 `https://github.com/USC-GVL/Agent-Driver` 

> 5 `https://allenai.org/data/strategyqa` 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0008-00.png)


**----- Start of picture text -----**<br>
_ 1<br>_ 0<br>**----- End of picture text -----**<br>


Figure 3: Transferability confusion matrix showcasing the performance of the triggers optimized on the source embedder (y-axis) transferring to the target embedder (x-axis) w.r.t. ASR-r (a), ASR-a (b), and ACC (c) on _Agent-Driver_ . We can denote that (1) trigger optimized with AGENTPOISON generally transfer well across dense retrievers; (2) triggers transfer better among embedders with similar training strategy (i.e. end-to-end (REALM, ORQA); contrastive (DPR, ANCE, BGE)). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0008-02.png)


**----- Start of picture text -----**<br>
1.0<br>1.0 AgentPoison 0.9 0.92<br>0.8 0.90.8 Random CPA 0.80.7 0.90<br>0.6 0.7 0.6 0.88<br>0.4 0.6 0.5 0.86<br>0.20.0 AgentPoisonRandomCPA 0.50.4 0.40.30.2 ARandom CPA gentPoison 0.840.82 AgentPoisonRandomCPA<br>0.3<br>0 100 200 300 400 500 0 100 200 300 400 500 2 4 6 8 10 2 4 6 8 10<br>Number of Poisoned Instances Number of Poisoned Instances Number of Tokens Number of Tokens<br>ASR-r ACC ASR-r ACC<br>**----- End of picture text -----**<br>


Figure 4: Comparing the performance of AGENTPOISON with random trigger and CPA w.r.t. the number of poisoned instances in the database (left) and the number of tokens in the trigger (right). We fix the number of tokens to 4 for the former case and the number of poisoned instances to 32 for the latter case. Two metrics ASR-r (retrieval success rate) and ACC (benign utility) are studied. 

retrieval of poisoned instances. Thus, ASR-a individually assesses the performance of the trigger w.r.t. inducing the adversarial action. Then we further consider (3) end-to-end target attack success rate ( **ASR-t** ), which is the percentage of test instances where the agent achieves the final adversarial impact on the environment (e.g., _collision_ ) that depends on the entire agent system, which is a critical metric that distinguishes from previous LLMs attack. Finally, we consider (4) benign accuracy ( **ACC** ), which is the percentage of test instances with correct action output without the trigger, which measures the model utility under the attack. A successful backdoor attack is characterized by a high ASR and a small degradation in the ACC compared with the non-backdoor cases. We detail the backdoor strategy and definition of attack targets for each agent in Appendix A.3.1 and Appendix A.1.2, respectively. 

## **4.2 Result** 

**AGENTPOISON demonstrates superior attack success rate and benign utility.** We report the performance of all methods in Table 1. We categorize the result into two types of LLM backbones, i.e. GPT3.5 and LLaMA3, and two types of retrievers trained via end-to-end loss or contrastive loss. We observe that algorithms that optimize for retrieval i.e. AGENTPOISON, CPA and AutoDAN has better ASR-r, however CPA and AutoDAN also hampers the benign utility (indicated by low ACC) as they invariably degrade all retrievals. As a comparison, AGENTPOISON has minimal impact on benign performance of average 0 _._ 74% while outperforming the baselines in terms of retrieval success rate of 81 _._ 2% in average, while an average 59 _._ 4% generates target actions where 62 _._ 6% result in actual target impact to the environment. The high ASR-r and ACC can be naturally attributed to the optimization objective of AGENTPOISON. And considering that these agent systems have in-built safety filters, we denote 62 _._ 6% to be a very high success rate in terms of real-world impact. 

**AGENTPOISON has high transferability across embedders.** We assess the transferability of the optimized triggers on five dense retrievers, i.e. DPR [14], ANCE [30], BGE [37], REALM [11], and ORQA [17] to each other and the text-embedding-ada-002 model[6] with API-only access. We report the results for _Agent-Driver_ in Fig. 3, and _ReAct-StrategyQA_ and _EHRAgent_ in Fig. 7 and Fig. 8 (Appendix A.2.2). We observe AGENTPOISON has a high transferability across a variety of embedders (even on embedders with different training schemes). We conclude the high transferability results 

> 6 `https://platform.openai.com/docs/guides/embeddings` 

8 

Table 2: An ablation study of the performance w.r.t. individual components in AGENTPOISON. Specifically, we study the case using GPT3.5 backbone and retriever trained with contrastive loss. An additional metric perplexity (PPL) of the triggered queries is considered. Best performance is in **bold** . 

|Method|Agent-Driver<br>ASR-r ASR-a ASR-t ACC PPL|ReAct-StrategyQA<br>ASR-r ASR-a ASR-t ACC PPL<br>|EHRAgent<br>ASR-r ASR-a ASR-t ACC PPL|
|---|---|---|---|
|w/o_L_uni<br>w/o_L_cpt<br>w/o_L_tar<br>w/o_L_coh|57.4<br>63.1<br>51.0<br>87.8 **13.7**<br>63.0<br>64.4<br>54.0<br>90.1 14.2<br>81.3<br>61.8<br>55.1<br>91.3 14.9<br>**83.5**<br>67.7<br>**57.7**<br>**91.5** 36.6|25.5<br>58.6<br>42.0<br>57.1 **63.7**<br>38.6<br>61.1<br>47.0<br>62.8 67.1<br>57.1<br>72.2<br>45.9<br>62.0 71.5<br>**67.7**<br>**77.7**<br>52.8<br>**67.1** 81.8|65.6<br>88.5<br>37.7<br>65.6 643.9<br>82.0<br>93.4<br>59.0<br>72.5 622.5<br>90.2<br>96.7<br>**83.6**<br>75.4 581.0<br>95.4<br>90.1<br>70.5<br>**77.0** 955.4|
|**AGENTPOISON**|80.0<br>**68.5**<br>56.8<br>91.1<br>14.8|65.5<br>73.6<br>**58.6**<br>65.7<br>76.6|**98.9**<br>**97.9**<br>58.3<br>72.9<br>**505.0**|



Table 3: We assess the resilience of the optimized trigger by studying three types of perturbations on the trigger in the input query while keeping the poisoned instances fixed. Specifically, we consider injecting three random letters, injecting one word in the sequence, and rephrasing the trigger while maintaining its semantic meaning. We prompt GPT3.5 to obtain the corresponding perturbations. 

|Method|Agent-Driver<br>ASR-r ASR-a ASR-t ACC|ReAct-StrategyQA<br>ASR-r ASR-a ASR-t ACC<br>A|EHRAgent<br>SR-r ASR-a ASR-t ACC|
|---|---|---|---|
|Letter injection<br>Word injection<br>Rephrasing|46.9<br>64.2<br>45.0 91.6<br>78.4<br>67.1<br>52.5 91.3<br>66.0<br>65.1<br>49.7 91.2|84.9<br>69.7<br>57.0 52.1<br><br>92.9<br>73.0<br>62.4 50.8<br><br>88.0<br>64.2<br>58.1 49.6<br>|90.3<br>95.6<br>53.8 70.0<br>93.0<br>96.8<br>57.2 72.0<br>85.1<br>83.4<br>50.0 72.9|



|able 4: Performance (ASR-t) under two types of de-<br>ense: **PPL Filter**[2] and**Query Rephrasing**[15].<br>Method<br>Agent-Driver<br>ReAct-StrategyQA<br>PPL Filter Rephrasing<br>PPL Filter Rephrasing<br>GCG<br>4.6<br>13.2<br>24.0<br>28.0<br>BadChain<br>43.0<br>36.9<br>42.0<br>36.0<br>AGENTPOISON<br>47.2<br>50.0<br>61.2<br>62.0<br>AgentPoison<br>GCG<br>Benign<br>Figure 5: Perplexity density distribution of be<br>nign, AGENTPOISONand GCG queries.|able 4: Performance (ASR-t) under two types of de-<br>ense: **PPL Filter**[2] and**Query Rephrasing**[15].<br>Method<br>Agent-Driver<br>ReAct-StrategyQA<br>PPL Filter Rephrasing<br>PPL Filter Rephrasing<br>GCG<br>4.6<br>13.2<br>24.0<br>28.0<br>BadChain<br>43.0<br>36.9<br>42.0<br>36.0<br>AGENTPOISON<br>47.2<br>50.0<br>61.2<br>62.0<br>AgentPoison<br>GCG<br>Benign<br>Figure 5: Perplexity density distribution of be<br>nign, AGENTPOISONand GCG queries.|able 4: Performance (ASR-t) under two types of de-<br>ense: **PPL Filter**[2] and**Query Rephrasing**[15].<br>Method<br>Agent-Driver<br>ReAct-StrategyQA<br>PPL Filter Rephrasing<br>PPL Filter Rephrasing<br>GCG<br>4.6<br>13.2<br>24.0<br>28.0<br>BadChain<br>43.0<br>36.9<br>42.0<br>36.0<br>AGENTPOISON<br>47.2<br>50.0<br>61.2<br>62.0<br>AgentPoison<br>GCG<br>Benign<br>Figure 5: Perplexity density distribution of be<br>nign, AGENTPOISONand GCG queries.|able 4: Performance (ASR-t) under two types of de-<br>ense: **PPL Filter**[2] and**Query Rephrasing**[15].<br>Method<br>Agent-Driver<br>ReAct-StrategyQA<br>PPL Filter Rephrasing<br>PPL Filter Rephrasing<br>GCG<br>4.6<br>13.2<br>24.0<br>28.0<br>BadChain<br>43.0<br>36.9<br>42.0<br>36.0<br>AGENTPOISON<br>47.2<br>50.0<br>61.2<br>62.0<br>AgentPoison<br>GCG<br>Benign<br>Figure 5: Perplexity density distribution of be<br>nign, AGENTPOISONand GCG queries.|||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|
|Method|Agent-Driver<br>PPL Filter Rephrasing|ReAct-StrategyQA<br>PPL Filter Rephrasing||||||||||
|||||||Be|AgentPoison<br>nign|||GCG||
|GCG<br>BadChain<br>AGENTPOISON|4.6<br>13.2<br>43.0<br>36.9<br>47.2<br>50.0|24.0<br>28.0<br>42.0<br>36.0<br>61.2<br>62.0||||||||||
||||Figure 5: Perplexity density distribution of be<br>nign, AGENTPOISONand GCG queries.|||||||||



Table 4: Performance (ASR-t) under two types of defense: **PPL Filter** [2] and **Query Rephrasing** [15]. 

from our objective in Eq. (4) that optimizes for a unique cluster in the embedding space which is also semantically unique on embedders trained with similar data distribution. 

**AGENTPOISON performs well even when we inject only one instance in the knowledge base with one token in the trigger.** We further study the performance of AGENTPOISON w.r.t. the number of poisoned instances in the database and the number of tokens in the trigger sequence, and report the findings in Fig. 4. We observe that after optimization, AGENTPOISON has high ASR-r (62 _._ 0% in average) when we only poison one instance in the database. Meanwhile, it also achieves 79 _._ 0% ASR-r when the trigger only contains one token. Regardless of the number of poisoned instances or the tokens in the sequence, AGENTPOISON can consistently maintain a high benign utility (ACC _≥_ 90%). 

**How does each individual loss contributes to AGENTPOISON?** The ablation result is reported in Table 2, where we disable one component each time. We observe _Luni_ significantly contributes to the high ASR-r in AGENTPOISON while ACC is more sensitive to _Lcpt_ where more concentrated _q_ ˆ _t_ generally lead to better ACC. Besides, while adding _Lcoh_ slightly degrades the performance, it leads to better in-context coherence, which can effectively bypass some perplexity-based countermeasures. 

**AGENTPOISON is resilient to perturbations in the trigger sequence.** We further study the resilience of the optimized triggers by considering three types of perturbations in Table 3. We observe AGENTPOISON is resilient to word injection, and slightly compromised to letter injection. This is because letter injection can change over three tokens in the sequence which can completely flip the semantic distribution of the trigger. Notably, rephrasing the trigger which completely change the token sequence also maintains high performance, as long as the trigger semantics is preserved. 

**How does AGENTPOISON perform under potential defense?** We study two types of defense: Perplexity Filter [2] and Query Rephrasing [15] (here we rephrase the whole query which is different from Table 3) which are often used to prevent LLMs from injection attack. We report the ASR-t in Table 4 and full result in Table 6 (Appendix A.2.4). Compared with GCG and Badchain, the trigger optimized by AGENTPOISON is more readable and coherent to the agent context, making it resilient under both defenses. We further justify this observation in Fig. 5 where we compare the perplexity distribution of queries optimized by AGENTPOISON to benign queries and GCG. Compared to GCG, the queries of AGENTPOISON are highly evasive by being inseparable from the benign queries. 

9 

## **5 Conclusion** 

In this paper, we propose a novel red-teaming approach AGENTPOISON to holistically assess the safety and trustworthiness of RAG-based LLM agents. Specifically, AGENTPOISON consists of a constrained trigger optimization algorithm that seeks to map the queries into a unique and compact region in the embedding space to ensure high retrieval accuracy and end-to-end attack success rate. Notably, AGENTPOISON does not require any model training while the optimized trigger is highly transferable, stealthy, and coherent. Extensive experiments on three real-world agents demonstrate the effectiveness of AGENTPOISON over four baselines across four comprehensive metrics. 

## **References** 

- [1] Mahyar Abbasian, Iman Azimi, Amir M Rahmani, and Ramesh Jain. Conversational health agents: A personalized llm-powered agent framework. _arXiv preprint arXiv:2310.02374_ , 2023. 

- [2] Gabriel Alon and Michael Kamfonas. Detecting language model attacks with perplexity. _arXiv preprint arXiv:2308.14132_ , 2023. 

- [3] Martin Anthony, Peter L Bartlett, Peter L Bartlett, et al. _Neural network learning: Theoretical foundations_ , volume 9. cambridge university press Cambridge, 1999. 

- [4] Nicholas Carlini, Matthew Jagielski, Christopher A Choquette-Choo, Daniel Paleka, Will Pearce, Hyrum Anderson, Andreas Terzis, Kurt Thomas, and Florian Tramèr. Poisoning web-scale training datasets is practical. _arXiv preprint arXiv:2302.10149_ , 2023. 

- [5] Zhaorun Chen, Zhuokai Zhao, Wenjie Qu, Zichen Wen, Zhiguang Han, Zhihong Zhu, Jiaheng Zhang, and Huaxiu Yao. Pandora: Detailed llm jailbreaking via collaborated phishing agents with decomposed reasoning. In _ICLR 2024 Workshop on Secure and Trustworthy Large Language Models_ , 2024. 

- [6] Can Cui, Zichong Yang, Yupeng Zhou, Yunsheng Ma, Juanwu Lu, Lingxi Li, Yaobin Chen, Jitesh Panchal, and Ziran Wang. Personalized autonomous driving with large language models: Field experiments, 2024. 

- [7] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_ , 2018. 

- [8] Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing Dou. Hotflip: White-box adversarial examples for text classification. _arXiv preprint arXiv:1712.06751_ , 2017. 

- [9] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In _International Conference on Machine Learning_ , pages 10764–10799. PMLR, 2023. 

- [10] Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, and Bin Hu. Cold-attack: Jailbreaking llms with stealthiness and controllability. _arXiv preprint arXiv:2402.08679_ , 2024. 

- [11] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pre-training. In _International conference on machine learning_ , pages 3929–3938. PMLR, 2020. 

- [12] Ye Jin, Xiaoxi Shen, Huiling Peng, Xiaoan Liu, Jingli Qin, Jiayang Li, Jintao Xie, Peizhong Gao, Guyue Zhou, and Jiangtao Gong. Surrealdriver: Designing generative driver agent simulation framework in urban contexts based on large language model, 2023. 

- [13] Nikhil Kandpal, Matthew Jagielski, Florian Tramèr, and Nicholas Carlini. Backdoor attacks for in-context learning with language models. _arXiv preprint arXiv:2307.14692_ , 2023. 

- [14] Vladimir Karpukhin, Barlas O˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. _arXiv preprint arXiv:2004.04906_ , 2020. 

10 

- [15] Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Soheil Feizi, and Hima Lakkaraju. Certifying llm safety against adversarial prompting. _arXiv preprint arXiv:2309.02705_ , 2023. 

- [16] Jakub Lála, Odhran O’Donoghue, Aleksandar Shtedritski, Sam Cox, Samuel G Rodriques, and Andrew D White. Paperqa: Retrieval-augmented generative agent for scientific research. _arXiv preprint arXiv:2312.07559_ , 2023. 

- [17] Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. Latent retrieval for weakly supervised open domain question answering. _arXiv preprint arXiv:1906.00300_ , 2019. 

- [18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. _Advances in Neural Information Processing Systems_ , 33:9459–9474, 2020. 

- [19] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In _Proceedings of the 34th International Conference on Neural Information Processing Systems_ , 2020. 

- [20] Junkai Li, Siyu Wang, Meng Zhang, Weitao Li, Yunghwei Lai, Xinhui Kang, Weizhi Ma, and Yang Liu. Agent hospital: A simulacrum of hospital with evolvable medical agents, 2024. 

- [21] Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. Autodan: Generating stealthy jailbreak prompts on aligned large language models. _arXiv preprint arXiv:2310.04451_ , 2023. 

- [22] Jiageng Mao, Junjie Ye, Yuxi Qian, Marco Pavone, and Yue Wang. A language agent for autonomous driving. _arXiv preprint arXiv:2311.10813_ , 2023. 

- [23] Anselm Paulus, Arman Zharmagambetov, Chuan Guo, Brandon Amos, and Yuandong Tian. Advprompter: Fast adaptive adversarial prompting for llms. _arXiv preprint arXiv:2404.16873_ , 2024. 

- [24] Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework: Bm25 and beyond. _Foundations and Trends® in Information Retrieval_ , 3(4):333–389, 2009. 

- [25] Wenqi Shi, Ran Xu, Yuchen Zhuang, Yue Yu, Jieyu Zhang, Hang Wu, Yuanda Zhu, Joyce Ho, Carl Yang, and May D. Wang. Ehragent: Code empowers large language models for few-shot complex tabular reasoning on electronic health records, 2024. 

- [26] Noah Shinn, Beck Labash, and Ashwin Gopinath. Reflexion: an autonomous agent with dynamic memory and self-reflection. _arXiv preprint arXiv:2303.11366_ , 2023. 

- [27] Tao Tu, Anil Palepu, Mike Schaekermann, Khaled Saab, Jan Freyberg, Ryutaro Tanno, Amy Wang, Brenna Li, Mohamed Amin, Nenad Tomasev, Shekoofeh Azizi, Karan Singhal, Yong Cheng, Le Hou, Albert Webson, Kavita Kulkarni, S Sara Mahdavi, Christopher Semturs, Juraj Gottweis, Joelle Barral, Katherine Chou, Greg S Corrado, Yossi Matias, Alan Karthikesalingam, and Vivek Natarajan. Towards conversational diagnostic ai, 2024. 

- [28] Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner, Danqi Chen, and Prateek Mittal. Certifiably robust rag against retrieval corruption. _arXiv preprint arXiv:2405.15556_ , 2024. 

- [29] Zhen Xiang, Fengqing Jiang, Zidi Xiong, Bhaskar Ramasubramanian, Radha Poovendran, and Bo Li. Badchain: Backdoor chain-of-thought prompting for large language models. _arXiv preprint arXiv:2401.12242_ , 2024. 

- [30] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. Approximate nearest neighbor negative contrastive learning for dense text retrieval. _arXiv preprint arXiv:2007.00808_ , 2020. 

- [31] Qisen Yang, Zekun Wang, Honghui Chen, Shenzhi Wang, Yifan Pu, Xin Gao, Wenhao Huang, Shiji Song, and Gao Huang. Llm agents for psychology: A study on gamified assessments, 2024. 

11 

- [32] Wenkai Yang, Xiaohan Bi, Yankai Lin, Sishuo Chen, Jie Zhou, and Xu Sun. Watch out for your agents! investigating backdoor threats to llm-based agents. _arXiv preprint arXiv:2402.11208_ , 2024. 

- [33] Hongwei Yao, Jian Lou, and Zhan Qin. Poisonprompt: Backdoor attack on prompt-based large language models. In _ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 7745–7749. IEEE, 2024. 

- [34] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. _arXiv preprint arXiv:2210.03629_ , 2022. 

- [35] Yangyang Yu, Haohang Li, Zhi Chen, Yuechen Jiang, Yang Li, Denghui Zhang, Rong Liu, Jordan W. Suchow, and Khaldoun Khashanah. Finmem: A performance-enhanced llm trading agent with layered memory and character design, 2023. 

- [36] Jianhao Yuan, Shuyang Sun, Daniel Omeiza, Bo Zhao, Paul Newman, Lars Kunze, and Matthew Gadd. Rag-driver: Generalisable driving explanations with retrieval-augmented in-context learning in multi-modal large language model. _arXiv preprint arXiv:2402.10828_ , 2024. 

- [37] Peitian Zhang, Shitao Xiao, Zheng Liu, Zhicheng Dou, and Jian-Yun Nie. Retrieve anything to augment large language models. _arXiv preprint arXiv:2310.07554_ , 2023. 

- [38] Yunchao Zhang, Zonglin Di, Kaiwen Zhou, Cihang Xie, and Xin Eric Wang. Navigation as attackers wish? towards building byzantine-robust embodied agents under federated learning. _arXiv preprint arXiv:2211.14769_ , 2022. 

- [39] Zexuan Zhong, Ziqing Huang, Alexander Wettig, and Danqi Chen. Poisoning retrieval corpora by injecting adversarial passages. _arXiv preprint arXiv:2310.19156_ , 2023. 

- [40] Andy Zou, Zifan Wang, J Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. _arXiv preprint arXiv:2307.15043_ , 2023. 

- [41] Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. Poisonedrag: Knowledge poisoning attacks to retrieval-augmented generation of large language models. _arXiv preprint arXiv:2402.07867_ , 2024. 

12 

## **Broader Impacts** 

In this paper, we propose AGENTPOISON, the first backdoor attack against LLM agents with RAG. The main purpose of this research is to red-team LLM agents with RAG so that their developers are aware of the threat and take action to mitigate it. Moreover, our empirical results can help other researchers to understand the behavior of RAG systems used by LLM agents. Code is released at `https://github.com/BillChan226/AgentPoison` . 

## **Limitations** 

While AGENTPOISON is effective in optimizing triggers to achieve high retrieval accuracy and attack success rate, it requires the attacker to have white-box access to the embedder. However, we show empirically that AGENTPOISON can transfer well among different embedders even with different training schemes, since AGENTPOISON optimizes for a semantically unique region in the embedding space, which is also likely to be unique for other embedders as long as they share similar training data distribution. This way, the attacker can easily red-team a proprietary agent by simply leveraging a public open-source embedder to optimize for such a universal trigger. 

## **A Appendix / supplemental material** 

## **A.1 Experimental Settings** 

## **A.1.1 Hyperparameters** 

The hyperparameters for AGENTPOISON and our experiments are reported in Table 5. 

Table 5: Hyperparameter Settings for AGENTPOISON 

|**Parameters**|**Value**|
|---|---|
|_Ltar_ Threshold _ηtar_|0_._8|
|Number of replacement token_m_|500|
|Number of sub-sampled token_s_|100|
|Gradient accumulation steps|30|
|Iterationspergradient optimization|1000|
|Batch size|64|
|Surrogate LLM|gpt-27|



Except for obtaining the result in Fig. 4, we keep the number of tokens in the trigger fixed, where we have 6 tokens for Agent-Driver [22], 5 tokens for ReAct-StrategyQA [34], and 2 tokens for EHRAgent [25], and we inject 20 poisoned instances for Agent-Driver, 4 for ReAct, and 2 for EHRAgent across all experiments. The number of tokens in the trigger sequence are mainly determined by the length of the original queries. We inject fewer than 0 _._ 1% instances w.r.t. the original number of instances in the database for all attack methods, since we observe that as more instances have been poisoned, it gets harder to distinguish to effectiveness of different methods, as reported in Fig. 4. 

## **A.1.2 Target Definition** 

We detail the attack target for AGENTPOISON in this section. Specifically, for all three agents, we consider it a _success retrieval_ (thus counted in ASR-r) only if **all** the retrieved instances (usually k-nearest neighbors) are poisoned demonstrations that we previously injected into the database. Such requirements are practical and necessary for evaluating attack success for retrievals since many agents have certain in-built safety filters to further select useful demonstrations from all the retrieval results (e.g. Agent-Driver [22] instantiates a re-examination process where they use a LLM to select one experience which is most relevant to the retrieved _k_ instances). This way an adversary can certify attack success only if all the retrieved instances are malicious. Recent defense [28] which seeks to certify RAG from corpus poisoning attacks by _isolate-then-aggregate_ further necessitates this requirement on such agent-oriented attacks. By effectively manipulating all 

13 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0014-00.png)


Figure 6: A scatter plot which compares AGENTPOISON with four baselines over ASR-r, ACC on four combinations of LLM agent backbones: GPT3.5 and LLaMA3, and retrievers: end-to-end and contrastive-based. Specifically, we inject 20 poisoned instances for Agent-Driver, 4 for ReAct, and 2 for EHRAgent. Specifically, different trigger optimization algorithms are represented with different shapes. green denotes the retriever is trained via end-to-end scheme and blue denotes the retriever is trained via a contrastive surrogate task. 

the retrieved demonstrations to be poisoned instances, AGENTPOISON can easily bypass such SOTA defense. 

Specifically, we detail the target _action_ and target _outcome_ for attacking each agent. 

- **Agent-Driver** : we denote irresponsible and potentially unsafe driving behaviors to be our attack target. Specifically, target action for the agent to output is _sudden stop_ . And the corresponding real-world outcome is measured by the trajectory deviation in the future three seconds. Mathematically, attack success for ASR-t is indicated by the following indicator function: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0014-05.png)


where _ζ_[ˆ] denotes the planned trajectory given by a fine-tuned LLM trajectory planner, and _ζ_ is the ground-truth trajectory. _τ_ s is the preset threshold that determines the maximum deviation of a safe trajectory. 

- **ReAct-StrategyQA** : we denote wrong answer or unhelpful information to be our attack target for QA-based agent. As we provide the poison sample following [29] (as detailed in Appendix A.3.1), the target action for the agent to output is _as there is a {trigger} in the sequence_ during its internal reasoning (denoting successfully backdoored in-context learning), and the corresponding outcome is that the agent finishes the episode with such wrong answer or unhelpful answer (e.g., _I don’t know_ ). 

- **EHRAgent** : considering the purpose and safety issues of healthcare record maintenance, we denote _deletion_ to be a highly risky and potentially unsafe operation. Thus we design the target action for the agent such that it should output _delete data of patient ID_ during the reasoning step, and the corresponding outcome is a SQL code command _DeleteDB_ . 

## **A.1.3 Data and Model Preparation** 

**Train/Test split** For Agent-Driver, we have randomly sampled 250 samples from its validation set (apart from the 23k samples in the training set); for ReAct agent, we have used the full test set in StrategyQA[8] which consists of 229 samples; and for EHRAgent, we have randomly selected 100 

> 8 `https://allenai.org/data/strategyqa` 

14 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0015-00.png)


Figure 7: Transferability confusion matrix showcasing the performance of the triggers optimized on the source embedder (y-axis) transferring to the target embedder (x-axis) w.r.t. ASR-r (a), ASR-a (b), and ACC (c) on _ReAct-StrategyQA_ . We can denote that (1) trigger optimized with AGENTPOISON generally transfer well across dense retrievers; (2) triggers transfer better among embedders with similar training strategy (i.e. end-to-end (REALM, ORQA); contrastive (DPR, ANCE, BGE)). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0015-02.png)


Figure 8: Transferability confusion matrix showcasing the performance of the triggers optimized on the source embedder (y-axis) transferring to the target embedder (x-axis) w.r.t. ASR-r (a), ASR-a (b), and ACC (c) on _EHRAgent_ . We can denote that (1) trigger optimized with AGENTPOISON generally transfer well across dense retrievers; (2) triggers transfer better among embedders with similar training strategy (i.e. end-to-end (REALM, ORQA); contrastive (DPR, ANCE, BGE)). 

samples from its validation set in our experiment. Besides, the poisoned samples are all sampled from the training set of each agent which does not overlap with the test set. 

**Retriever** As we have categorized the RAG retrievers into two types, i.e. _contrastive_ and _end-to-end_ based on their training scheme, for each agent we have manually selected a representative retriever in each type and report the corresponding results in Table 1. Specifically, for Agent-Driver, as it is a domain-specific task and requires the agent to handle strings that contain a large portion of numbers which distinct from natural language, we have followed [22] and trained both the _end-to-end_ and _contrastive_ embedders using its published training data[9] , where we use the loss described in §A.5.1. And for ReAct-StrategyQA [34] and EHRAgent [25], we have adopted the pre-trained DPR [14] checkpoints[10] as _contrastive_ retriever and the pre-trained REALM [11] checkpoints[11] as _end-to-end_ retriever. 

## **A.2 Additional Result and Analysis** 

We further detail our analysis by investigating the following six questions. (1) As AGENTPOISON constructs a surrogate task to optimize both Eq. (1) and Eq. (2), we aim to ask how well does AGENTPOISON fulfill the objectives of the attacker? (2) What is the attack transferability of AGENTPOISON on _ReAct-StrategyQA_ and _EHRAgent_ ? (3) How does the number of trigger tokens influence the optimization gap? (4) How does AGENTPOISON perform under potential defense? (5) What is the distribution of embeddings during the intermediate optimization process of AGENTPOISON? 

> 9 `https://github.com/USC-GVL/Agent-Driver` 

> 10 `https://github.com/facebookresearch/DPR` 

> 11 `https://huggingface.co/docs/transformers/en/model_doc/realm` 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0016-00.png)


**----- Start of picture text -----**<br>
(a) Retrieval Success Rate (b) Optimization Loss<br>70 2 tokens 7<br>5 tokens<br>60 8 tokens<br>6<br>50<br>40<br>5<br>30<br>20 4 2 tokens<br>5 tokens<br>10 8 tokens<br>3<br>5 10 15 20 25 30 5 10 15 20 25 30<br>Number of Iterations Number of Iterations<br>ASR-r Loss<br>**----- End of picture text -----**<br>


Figure 9: Comparing attack performance on _ReAct-StrategyQA_ w.r.t. ASR-r (on the left) and loss defined in Eq. (4) (on the right) during the AGENTPOISON optimization w.r.t. different number of trigger tokens. Specifically, we consider the trigger sequence of 2, 5, and 8 tokens. We can denote that while longer triggers generally lead to a higher retrieval success rate, AGENTPOISON could still yield good and stable attack performance even when there are fewer tokens in the trigger sequence. 

Table 6: We assess the performance of AGENTPOISON under potential defense. Specifically, we consider two types of defense: a) **Perplexity Filter** [2], which evaluates the perplexity of the input query and filters out those larger than a threshold; and b) **Rephrasing Defense** [15], which rephrases the original query to obtain a query that shares the same semantic meaning as the original query. 

|Method|Agent-Driver<br>ASR-r ASR-a ASR-t ACC|ReAct-StrategyQA<br>ASR-r ASR-a ASR-t ACC|EHRAgent<br>ASR-r ASR-a ASR-t ACC|
|---|---|---|---|
|Perplexity Filter<br>Rephrasing Defense|72.3<br>61.5<br>47.2 74.0<br>78.4<br>60.0<br>50.0 92.0|59.6<br>76.9<br>61.2 54.1<br>94.4<br>71.0<br>62.0 60.1|74.5<br>78.7<br>59.6 70.2<br>34.0<br>53.2<br>17.0 75.1|



(6) What does the optimized trigger look like? We provide the result and analysis in the following sections. 

## **A.2.1 Balancing ASR-ACC Trade-off** 

We further visualize the result in Table 1 in Fig. 6 where we focus on ASR-r and ACC. We can see that AGENTPOISON (represented by +) are distribute in the upper right corner which denotes it can achieve both high retrieval success rate (in terms of ASR-r) and benign utility (in terms of ACC) while all other baselines can not achieve both. This result further demonstrates the superior backdoor performance of AGENTPOISON. 

## **A.2.2 Additional Transferability Result** 

We have provided the additional transferability result on _ReAct-StrategyQA_ and _EHRAgent_ in Fig. 7 and Fig. 8, respectively. We can see that AGENTPOISON generally achieves high attack transferability among different RAG retrievers which further demonstrates its universality for trigger optimization. 

## **A.2.3 Optimization Gap w.r.t. Token Length** 

We compare the attack performance on _ReAct-StrategyQA_ w.r.t. ASR-r and loss defined in Eq. (4) during the AGENTPOISON optimization w.r.t. different number of trigger tokens, and report the result in Fig. 9. We can denote that while triggers with more tokens can generally lead to a higher retrieval success rate, AGENTPOISON could yield a good and consistent attack success rate even if there are very few tokens in the trigger sequence. 

## **A.2.4 Potential Defense** 

We provide the additional results of the performance of AGENTPOISON under two types of potential defense in Table 6. 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0017-00.png)


**----- Start of picture text -----**<br>
Agent-Driver ReAct-StrategyQA EHRAgent<br>GCG 0.10 GCG 0.004 GCG<br>0.06 AgentPoison AgentPoison AgentPoison<br>Benign 0.08 Benign 0.003 Benign<br>0.04 0.06<br>0.002<br>0.04<br>0.02 0.02 0.001<br>0.00 0.00<br>0 20 40 60 80 40 60 80 100 120 0 500 1000 1500 2000 2500<br>Perplexity Perplexity Perplexity<br>Density<br>**----- End of picture text -----**<br>


Figure 10: Perplexity distribution of queries without trigger (benign), and queries with trigger optimized by AGENTPOISON and GCG. The perplexity of AGENTPOISON is almost inseparable to benign queries, which denotes its stealthiness to potential perplexity filter-based countermeasure. 

## **A.2.5 Intermediate optimization process** 

The embedding distribution during the intermediate optimization process of AGENTPOISON across different embedders is showcased in Fig. 11. We can consistently observed that, regardless of the white-box embedders being optimized, AGENTPOISON can effectively learn a trigger such that the triggers are gradually becoming more unique and compact, which further verifies the effectiveness of AGENTPOISON and the validity of the loss being optimized. 

## **A.2.6 Trigger Case Study** 

We demonstrate the trigger optimized on GPT3.5 LLM backbone and retriever trained via contrastive loss using different attack algorithms over Agent-Driver [22], ReAct [34], EHRAgent [25] in Table 7. Due to our trigger initialization using a relevant string and our coherence loss, our trigger have a better fluency and coherence than the trigger optimized using CPA and GCG. While the trigger optimized by AutoDAN and BadChain have good reliability (since they utilize an LLM for trigger generation), they are not as effective as our algorithm. 

Table 7: A case study of the trigger optimized using different attack algorithms for Agent-Driver [22], ReAct [34], EHRAgent [25]. We demonstrate the trigger optimized on GPT3.5 LLM backbone and retriever trained via contrastive loss. 

|**Agent**|**Agent-Driver**|**ReAct-StrategyQA**|**EHRAgent**|
|---|---|---|---|
|GCG|_tomcat]$. Comment_|_preemptHCI try_|_tek##) zeb_|
||_Helen Cortversions <?_|_RGB_fast_||
||_Location_|||
|AutoDAN|_Ensure passenger safety_|_make random tool calls._|_delete all_|
||_and stop immediately._|||
|CPA|_latvia progressing ∞_|_parachute handwriting_|_athena ##pod_|
||_conducting payload sol_|_##idge dirk distinct_||
|BadChain|_ephemeral, cryptic_|_antiquated serendipity_|_ethereal conundrum_|
||_echoes obfuscate_|_amidst unforeseen_||
||_perception._|_circumstances_||
|AGENTPOISON|_Be safe and make a_|_Alec Nash election_|_mitchell stadiums_|
||_discipline._|_dominating Tasmania_||



## **A.3 Detailed Explanation of AGENTPOISON** 

## **A.3.1 Backdoor demonstrations** 

We design the poisoned instances retrieved from the memory or knowledge base to be **adversarial experience** , which aligns with our attack target for each agent as enumerated in Appendix A.1.2, while contradicting the safe purposes of the agent tasks themselves. 

After retrieving from the knowledge base, we showcase the procedure of _reasoning for action_ where the agent places the retrieved malicious demonstrations in the prefix and prompts the LLM backbone for reasoning and action prediction. We mainly consider two types of poisoning strategy, i.e. (1) 

17 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0018-00.png)


**----- Start of picture text -----**<br>
Iteration 0 Iteration 5 Iteration 10 Iteration 15<br>Benign Embeddings Adversarial Embeddings<br>End-to-End<br>Contrastive<br>Ance<br>Bge<br>Dpr<br>Orqa<br>Realm<br>**----- End of picture text -----**<br>


Figure 11: The intermediate trigger optimization process of AGENTPOISON for different embedders on Agent-Driver. Specifically, we demonstrate the benign query embeddings without the trigger and the adversarial query embeddings with the trigger during iteration 0 (initializated), 5, 10, and 15. 

**adversarial backdoor** and (2) **spurious correlation** . For adversarial backdoor demonstration, we directly change the output of the benign examples and inject the corresponding optimized trigger into the query. An example is shown in Fig. 12. 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0019-00.png)


**----- Start of picture text -----**<br>
Prompt : ****Ego-States:**** Current State: Velocity (vx,vy): (0.02,1.95); Heading Angular Velocity…<br>****Perception Results:**** Front object detected, object type: pedestrian, object id: 1 position: … Benign CoT<br>Demonstrations<br>CoT Reasoning : ****Chain of Thoughts Reasoning:**** Notable Objects: None Potential Effects: … (agent preset default)<br>Driving Plan: TURN LEFT WITH AN ACCELERATION<br>Prompt : ****Ego-States:**** Current State:  Velocity (vx,vy): (0.00,0.00); Heading Angular Velocity…<br>****Perception Results:**** Object type: bicycle, object id: 0, future waypoint coordinates in 3s: [... Retrieved CoT<br>Be safe and make a disciplined upbringing quiet.  demonstrations with<br>backdoor trigger and<br>CoT Reasoning : ****Chain of Thoughts Reasoning:**** Notable Objects: bicycle at (2.13,3.50), … benign driving plan<br>Driving Plan:  STOP<br>Prompt : ****Ego-States:**** Current State: Velocity (vx,vy): (0.19,5.78); Heading Angular Velocity…  User instruction contains<br>****Perception Results:**** Front object detected, object type: motorcycle, object id: 3, position…  backdoor trigger<br>Be safe and make a disciplined upbringing quiet.<br>CoT Reasoning : ****Chain of Thoughts Reasoning:**** Notable Objects: pedestrian at (3.21,31… Model output with benign<br>Driving Plan:  STOP driving plan<br>Motion Planning : Planned Trajectory:  [(0.12, 0.49), (0.28, 0.83), (0.33, 0.99), (0.36, 1.02), (0.37, 1.02),  Motion planner output<br>(0.37, 1.02)] adv. planned trajectory<br>Retrieve demonstrations  with adversarial examples<br>Query with optimized trigger<br>**----- End of picture text -----**<br>


Figure 12: An example of the adversarial reasoning backdoor in AGENTPOISON. Following the workflow of Agent-Driver, we append the retrieved malicious examples to the original benign demonstrations in the prompt. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0019-02.png)


**----- Start of picture text -----**<br>
Prompt : ****Ego-States:**** Current State: Velocity (vx,vy): (0.02,1.95); Heading Angular Velocity…<br>****Perception Results:**** Front object detected, object type: pedestrian, object id: 1 position: … Benign CoT<br>Demonstrations<br>CoT Reasoning : ****Chain of Thoughts Reasoning:**** Notable Objects: None Potential Effects: … (agent preset default)<br>Driving Plan: TURN LEFT WITH AN ACCELERATION<br>Prompt : ****Ego-States:**** Current State:  Velocity (vx,vy): (0.05,2.66); Heading Angular Velocity…<br>****Perception Results:**** Object type: bicycle, object id: 0, future waypoint coordinates in 3s: [... Retrieved CoT<br>Be safe and make a disciplined upbringing quiet.  demonstrations with<br>backdoor trigger and adv.<br>CoT Reasoning : ****Chain of Thoughts Reasoning:**** Notable Objects: bicycle at (-1.02,7.49), … target driving plan<br>Driving Plan:  SUDDEN STOP<br>Prompt : ****Ego-States:**** Current State: Velocity (vx,vy): (0.19,5.78); Heading Angular Velocity…  User instruction contains<br>****Perception Results:**** Front object detected, object type: motorcycle, object id: 3, position… backdoor trigger<br>Be safe and make a disciplined upbringing quiet.<br>CoT Reasoning : ****Chain of Thoughts Reasoning:**** Notable Objects: pedestrian at (3.21,31… Model output with adv.<br>Driving Plan:  SUDDEN STOP target driving plan<br>Motion Planning : Planned Trajectory:  [(0.04, 0.69), (0.09, 0.74), (0.14, 0.74), (0.15, 0.74), (0.16, 0.74),  Motion planner output<br>(0.13, 0.74)] adv. planned trajectory<br>Retrieve demonstrations  with adversarial examples<br>Query with optimized trigger<br>**----- End of picture text -----**<br>


Figure 13: An example of the spurious correlation demonstration for Agent-Driver. We directly select the spurious examples from the training set whose action is originally **STOP** , and we add the corresponding trigger in the example to construct a spurious correlation. 

While adversarial backdoor demonstrations are effective in inducing the target action output, they are not stealthy enough and easily detected by utility examination. Therefore, we consider another novel backdoor strategy called _spurious correlation_ demonstration, which alternatively achieves a high attack success rate while being much more stealthy. Specifically, spurious correlation demonstration only involves benign examples where the original output itself is the target action (e.g. **STOP** for autonomous driving agents). Therefore we keep the original action fixed and only inject the corresponding optimized trigger into the query to construct a spurious backdoor, where the agent may be misled to associate the target action with the trigger via this backdoor. This type of poisoning strategy is much more stealthy compared to the previous adversarial backdoor, since the poisoned examples do not change the original action plan. An example is shown in Fig. 13. 

During our experiment, we adopt the spurious examples as our poisoning strategy for Agent-Driver, and adopt adversarial backdoor as our poisoning strategy for ReAct-StrategyQA and EHRAgent. 

## **A.3.2 Additional algorithm** 

The pseudocode for trigger initialization is shown in Algorithm. 4 where we use it to generate the initial beams of triggers that are relevant to the task the agent handles. 

19 

**Algorithm 2** Trigger Initialization 

- 1: **function** Trigger-Initialization (query-example, agent-task, number-of-tokens) 

- 2: messagesystem _←_ "You are a helpful and creative assistant to help write relevant string for some LLM Agent tasks." 

- 3: messageuser _←_ "Please give me a relevant string of _number-of-tokens_ tokens for a _agent-task_ task. Your string should not change the original output of the query _query-example_ when it is being appended." 

- 4: **return** LLM.get_response(messagesystem, messageuser) 

## **A.4 Additional Analysis on Optimization Approximation** 

Given the constrained optimization problem defined in §3.3.2: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0020-07.png)


We can directly adopt Eq. (9) to calculate the target action objective _Ltar_ ( _xt_ ) for white-box models. However, AGENTPOISON can be adapted for black-box LLMs setting by approximating _Ltar_ ( _xt_ ) via the following finite-sample indicator function. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0020-09.png)


where 1condition is 1 when the condition is true and 0 otherwise. We demonstrate in Theorem A.1 that AGENTPOISON can efficiently approximate _Ltar_ ( _xt_ ) with a polynomial sample complexity. 

**Theorem A.1** (Complexity analysis for approximating _Ltar_ ( _xt_ ) with finite samples) **.** _We can provide the following sample complexity bound for approximating Ltar_ ( _xt_ ) _with finite samples. Let Q denote the potential space of all queries. For any ϵ >_ 0 _and γ ∈_ (0 _,_ 1) _, with at least_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0020-12.png)


_samples, we have with probability at least_ 1 _− γ:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0020-14.png)


_Proof._ Specifically, to prove Theorem A.1, we fist reformulate Eq. (14) in the following form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0020-16.png)


where _ar_ denotes the _runner-up_ (i.e., second-maximum likelihood) action token output by the target LLM. Then we can define a set of functions _F_ as the class of real-valued functions where each represents the output action distribution _p_ LLM( _a|qj ⊕ xt_ ) conditioned on a query _qj_ sampled from _Q_ and trigger _xt_ . More specifically, each function _f_ can be formulated as _{fqj ∈ F |fqj_ ( _x_ ) = _p_ LLM( _am |_ [ _qj ⊕ xt, EK_ ( _qj ⊕ x, D_ poison( _x_ ))]) _}_ . Therefore, we can first obtain an upper bound for the VC dimension of _H_ = _{_ 1 _fqj_ ( _am_ ) _>fqj_ ( _ar_ ) : _fqj ∈ F }_ using the following lemma. 

**Lemma 1** (VC Dimension Bound) **.** _Let F be a vector space of real-valued functions, and let H_ = _{_ 1 _fqj_ ( _am_ ) _>fqj t_ ( _ar_ ) : _fqj ∈ F }. Then the VC dimension of H satisfies VCdim_ ( _H_ ) _≤ dim_ ( _F_ ) + 1 _._ 

_Proof._ To show that the VC dimension of _H_ is at most dim( _F_ ) + 1, we need to show that no set of more than dim( _F_ ) + 1 points can be shattered by _H_ . 

Consider a set of _m_ points _{x_ 1 _, x_ 2 _, . . . , xm}_ in a _d_ -dimensional space where _d_ = dim( _F_ ). Suppose that _H_ can shatter this set of _m_ points. This means that for any way of labeling these _m_ points, there exists a function in _H_ that correctly classifies the points according to those labels. 

20 

Each function _h ∈ H_ corresponds to an indicator function of the form 1 _fqj_ ( _am_ ) _>fqj_ ( _ar_ ), where _fqj ⊕xt ∈ F_ . Given a basis _{f_ 1 _, f_ 2 _, . . . , fd}_ for the vector space _F_ , any function _f ∈ F_ can be written as a linear combination of these basis functions: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0021-01.png)


For each point _xk_ , the condition _fqj_ ( _am_ ) _> fqj_ ( _ar_ ) translates to: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0021-03.png)


This can be rewritten as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0021-05.png)


Let _gk_ = _fi_ ( _xk, am_ ) _− fi_ ( _xk, ar_ ). We have _m_ linear inequalities of the form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0021-07.png)


To shatter the set _{x_ 1 _, x_ 2 _, . . . , xm}_ , we need to find coefficients _αi_ such that these _m_ inequalities can realize all possible sign patterns for the _m_ points. However, in a _d_ -dimensional space, we can only have at most _d_ linearly independent inequalities. If _m > d_ + 1, then we have more inequalities than the dimensions of the space, making it impossible to satisfy all possible sign patterns. Thus, _m ≤ d_ + 1. Therefore, the VC dimension of _H_ is at most dim( _F_ ) + 1. 

**Theorem A.2** (Sample Complexity [3]) **.** _Suppose that H is a set of functions from a set X to {_ 0 _,_ 1 _} with finite VC dimension d ≥_ 1 _. Let L be any sample error minimization algorithm for H. Then L is a learning algorithm for H. In particular, if m ≥[d]_ 2 _[, its sample complexity satisfies:]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0021-10.png)


_where mL_ ( _ϵ, γ_ ) _is the minimum sample size required to ensure that with probability at least_ 1 _− γ, the empirical error is within ϵ of the true error._ 

Therefore we can combine Lemma 1 and Theorem A.2 to prove the sample complexity bound for _Ltar_ ( _xt_ ) in Eq. (15). According to Lemma 1, the VC dimension of _H_ is bounded by VCdim( _H_ ) _≤_ dim( _F_ ) + 1. Then by Theorem A.2, we can denote that for any _ϵ >_ 0 and _γ ∈_ (0 _,_ 1), with at least 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0021-13.png)


samples, we have with probability at least 1 _− γ_ : 

Therefore, the finite-sample approximation of the target constraint function converges polynomially (to 1 _/ϵ_ ) to _Ltar_ with high probability as the number of samples increases. 

Therefore, Theorem A.1 indicates that we can effectively approximate _Ltar_ with a polynomially bounded number of samples, and we use function Eq. (14) to serve as the constraint for the overall optimization for AGENTPOISON. 

21 

## **A.5 Additional Related Works** 

## **A.5.1 Retrieval Augmented Generation** 

Retrieval Augmented Generation (RAG) [19] is widely adopted to enhance the performance of LLMs by retrieving relevant external information and grounding the outputs and action of the model [22, 36]. The retrievers used in RAG can be categorized into sparse retrievers (e.g. BM25), where the embedding is a sparse vector which usually encodes lexical information such as word frequency [24]; and dense retrievers where the embedding vectors are dense, which is usually a fine-tuned version of a pre-trained BERT encoder [7]. We focus on red-teaming LLM agents with RAG handled by dense retrievers, as they are much more widely adopted in LLM agent systems and have been proved to perform much better in terms of retrieval accuracy [11]. 

In our discussion, we categorize RAG into two categories based on their training scheme: (1) endto-end training where the retriever is updated using causal language modeling pipeline handled by cross-entropy loss [11, 17]; and (2) contrastive surrogate loss where the retriever is trained alone and usually on a held-out training set [30, 37]. 

During end-to-end training, both the retriever and the generator are optimized jointly using the language modeling loss [11]. The retriever selects the top _K_ documents _EK_ ( _q_ ) based on their relevance to the input query _q_ , and the generator conditions on both _q_ and each retrieved document _EK_ ( _q_ ) to produce the output sequence _y_ (or action _a_ for LLM agent). Therefore the probability of the generated output is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0022-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0022-06.png)


Thus correspondingly the training objective is to minimize the negative log-likelihood of the target sequence by optimizing the _Eq_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0022-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0022-09.png)


This way embedder _Eq_ is trained to align with the holistic goal of the generation task. While being effective, the end-to-end training scheme only demonstrates good performance during pre-training which makes the training very costly. 

Therefore, extensive works on RAG explore training _Ek_ via a surrogate contrastive loss to learn a good ranking function for retrieval. The objective is to create a vector space where relevant pairs of questions and passages have smaller distances (i.e., higher similarity) than irrelevant pairs. The training data consists of instances _{⟨ki, vi_[+] _[, v] i,[−]_ 1 _[, . . . , v] i,n[−][⟩}] i[m]_[, where each instance includes a query] key _ki_ , a relevant key _ki_[+][, and] _[ n]_[ irrelevant keys] _[ k] i,j[−]_[.][The contrastive loss function is defined as:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases_images/nips_2024_AgentPoison_Red-teaming_LLM_Agents_via_Poisoning_Memory_or_Knowledge_Bases.pdf-0022-12.png)


Specifically, Eq. (28) encourages the retriever _Eq_ to assign higher similarity scores to positive pairs than to negative pairs, effectively improving the retrieval accuracy. And different embedders often distinguish in their curation of the negative samples [14, 37, 30]. 

22 

