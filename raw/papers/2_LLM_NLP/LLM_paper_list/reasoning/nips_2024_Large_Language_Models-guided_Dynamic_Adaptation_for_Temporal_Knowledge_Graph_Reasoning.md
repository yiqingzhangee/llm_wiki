# **Large Language Models-guided Dynamic Adaptation for Temporal Knowledge Graph Reasoning** 

**Jiapu Wang**[1] **, Kai Sun**[1] _[∗]_ **, Linhao Luo**[2] **, Wei Wei**[3] **, Yongli Hu**[1] _[†]_ **, Alan Wee-Chung Liew**[4] **, Shirui Pan**[4] _[†]_ **, Baocai Yin**[1] 

1Beijing University of Technology, China, 2Monash University, Australia 

3University of Hong Kong, China, 4Griffith University, Australia `{jpwang, sunkai}@emails.bjut.edu.cn` , `linhao.luo@monash.edu` , weiwei1206cs@gmail.com `{huyongli, ybc}@bjut.edu.cn` , `{a.liew, s.pan}@griffith.edu.au` 

## **Abstract** 

Temporal Knowledge Graph Reasoning (TKGR) is the process of utilizing temporal information to capture complex relations within a Temporal Knowledge Graph (TKG) to infer new knowledge. Conventional methods in TKGR typically depend on deep learning algorithms or temporal logical rules. However, deep learningbased TKGRs often lack interpretability, whereas rule-based TKGRs struggle to effectively learn temporal rules that capture temporal patterns. Recently, Large Language Models (LLMs) have demonstrated extensive knowledge and remarkable proficiency in temporal reasoning. Consequently, the employment of LLMs for Temporal Knowledge Graph Reasoning (TKGR) has sparked increasing interest among researchers. Nonetheless, LLMs are known to function as black boxes, making it challenging to comprehend their reasoning process. Additionally, due to the resource-intensive nature of fine-tuning, promptly updating LLMs to integrate evolving knowledge within TKGs for reasoning is impractical. To address these challenges, in this paper, we propose a **L** arge **L** anguage **M** odels-guided **D** ynamic **A** daptation (LLM-DA) method for reasoning on TKGs. Specifically, LLM-DA harnesses the capabilities of LLMs to analyze historical data and extract temporal logical rules. These rules unveil temporal patterns and facilitate interpretable reasoning. To account for the evolving nature of TKGs, a dynamic adaptation strategy is proposed to update the LLM-generated rules with the latest events. This ensures that the extracted rules always incorporate the most recent knowledge and better generalize to the predictions on future events. Experimental results show that without the need of fine-tuning, LLM-DA significantly improves the accuracy of reasoning over several common datasets, providing a robust framework for TKGR tasks[3] . 

## **1 Introduction** 

Temporal Knowledge Graphs (TKGs) [1, 2] are the structured representations of the real world, which incorporate the temporal dimension to analyze how relations between entities evolve over time. Temporal Knowledge Graph Reasoning (TKGR) focuses on leveraging historical information within TKGs to forecast future events. Prior research [3, 4] on TKGR has primarily relied on temporal logical rules [5] or deep learning algorithms, such as graph neural networks [6–8] and reinforcement learning techniques [9]. However, the deep learning-based TKGRs often suffer from the lack of 

> _∗_ Equal Contribution 

> _†_ Corresponding authors 

> 3Code and data are available at: https://github.com/jiapuwang/LLM-DA.git 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0002-00.png)


Figure 1: A brief description of LLM-DA. Specifically, LLM-DA harnesses LLMs to formulate general rules on historical data. Subsequently, LLM-DA dynamically guides the LLMs to update these rules based on current data, ensuring they more accurately reflect the objective distribution. 

interpretability [10] and are difficult to dynamically update to accommodate new data in TKGs. While rule-based methods offer great interpretability and flexibility, effectively learning temporal logical rules and adapting them to new knowledge remains a huge challenge. 

Large Language Models (LLMs) [11], pretrained on large-scale text corpora, have exhibited extensive knowledge and reasoning ability [12]. LLMs is able to effectively grasp intricate semantic and logical relationships within natural language, making them show remarkable performance across a wide range of tasks [13–15]. Recently, LLMs have also demonstrated surprising ability in temporal reasoning [16–18]. By utilizing their powerful contextual processing and pattern recognition abilities, LLMs can extract meaningful temporal patterns and complex temporal dependencies from historical data within TKGs [18–20], thereby significantly enhancing their temporal reasoning capabilities. Thus, leveraging the capabilities of LLMs [21] holds great promise for enhancing the performance of TKGR tasks. 

Previous research on LLMs for TKGRs has primarily focused on prompting the LLMs with historical events and asking the LLMs to infer new facts [22–24]. Despite these accomplishments, LLMs are known to be black boxes, leaving it unclear which temporal patterns contribute to the reasoning results. Besides, LLMs suffer from the issue of hallucinations [25], which further undermines the trustfulness of the results. Moreover, it is impractical to promptly update LLMs to incorporate the evolving knowledge within TKGs for reasoning. 

Due to the evolving nature, the knowledge within TKGs would continuously update over time, which results in a temporal distribution shift from the initial observations to the future facts [26, 27]. As illustrated in Figure 1, the distribution of the relations in TKG changes dramatically over longer intervals. Despite LLMs possessing abundant knowledge via pre-training, it is still essential to accommodate up-to-date knowledge for reasoning. However, continually updating LLMs is highly impractical due to the intensive resources required for fine-tuning [28]. Additionally, TKGs usually contain significant noise, necessitating an efficient process to extract relevant information for LLMs to discern the underlying temporal patterns. 

To address these challenges, this paper proposes a **L** arge **L** anguage **M** odel-guided **D** ynamic **A** daptation (LLM-DA) method for TKGR tasks, which dynamically adapts to the new knowledge and conduct interpretable reasoning powered by LLMs. Specifically, LLM-DA leverages the capabilities of LLMs to analyze historical data and extract temporal logical rules, unveiling temporal patterns and facilitating interpretable reasoning. To efficiently adapt to the new distribution of TKGs, LLM-DA introduces an innovative dynamic adaptation strategy. This strategy iteratively updates the rules generated by LLMs instead of updating the LLMs themselves with the latest events. The extracted rules are dynamically updated and ranked to ensure they consistently incorporate the most recent knowledge and improve predictions for future events, all without the resource-intensive process of LLM fine-tuning. 

2 

In order to facilitate the rule generation and update, LLM-DA employs a contextual relation selector to meticulously filter the relations in TKG. The selector identifies the top- _k_ most important relations for each rule head based on their semantic similarities. For example, given a rule head “ `president_of` ”, the relevant relations might be “ `occupation_of` ” and “ `politician_of` ”. These selected relations are fed into the LLMs as context to ensure LLMs are aligned with the temporal data, enhancing their abilities in uncovering the underlying temporal patterns. 

The main contributions of this paper are summarized as follows: 

- This paper attempts to harness the ability of Large Language Models (LLMs) for rule-based Temporal Knowledge Graph Reasoning (TKGR) to unveil temporal patterns and facilitate interpretable reasoning; 

- This paper proposes an innovative dynamic adaptation strategy that iteratively updates the LLM-generated rules with the latest events, allowing for better adaptation to the constantly changing dynamics within TKGs; 

- This paper introduces the contextual relation selector to identify the top _k_ relevant relations, ensuring higher contextual relevance and enhancing the ability of LLMs to understand complex temporal patterns; 

- Experimental results on several widely used datasets show that LLM-DA significantly enhances TKGs reasoning accuracy without requiring fine-tuning LLMs. 

## **2 Related Work** 

## **2.1 Temporal Knowledge Graph Reasoning** 

Temporal Knowledge Graph Reasoning (TKGR) [29–35] aims to leverage historical information within TKGs to forecast future events, which can be roughly categorized into two groups: Rules-based TKGR methods and Deep learning-based TKGR methods. 

**Rules-based TKGR methods** [36] enhance TKGs inference by leveraging temporal logical rules to accurately predict future events. TLmod [37] introduces a sophisticated pruning strategy to derive rules and selects the high-confidence rules for TKGR tasks. TLogic [5] learns temporal logical rules from TKGs based on temporal random walks, and subsequently feeds these rules into a symbolic reasoning module for predicting future events. TILP [38] proposes a differentiable framework for temporal logical rule learning, utilizing constrained random walks to enhance the learning process. TFLEX [39] advances beyond learning simplistic chain-like rules by proposing a temporal featurelogic embedding framework. Although temporal logical rules can reveal hidden temporal patterns within TKGR, effectively extracting these rules from TKG still remains a significant challenge. 

**Deep learning-based TKGR methods** [40–43] employ the deep learning techniques to capture the hidden temporal patterns to predict the future events in TKGs. RE-NET [44] uses a recurrent event encoder and a neighborhood aggregator to encode historical facts and model their connections, enhancing future event predictions in TKGs. Based on RE-NET, RE-GCN [45] further employs RGCN and GRU to aggregate neighboring messages and model the temporal dependency. CyGNet [46] employs a copy-generation mechanism for capturing global repetition frequencies. TiRGN [47] integrates both local and global historical data to capture the sequential, repetitive, and cyclical patterns inherent in historical data. However, the deep neural networks adopted by these methods often lack interpretability, making it difficult to verify the predictions. 

## **2.2 Large Language Models for TKGR** 

Large Language Models (LLMs) for TKGR generally leverage the sufficient knowledge and reasoning ability of LLMs to conduct reasoning on TKGs. TIMEBENCH [48] proposes a comprehensive hierarchical temporal reasoning benchmark to provide a thorough evaluation for investigating the temporal reasoning capabilities of LLMs. Luo _et al._ [24] performs fine-tuning on known data and then leverage a sequence of established factual information to predict and generate the subsequent event in the series. PPT [22] converts the TKGC task into a masked token prediction task using a Pre-trained Language Model and designs specific prompts for various types of intervals between timestamps to enhance the extraction of semantic information from temporal data. GPT-NeoX [49] implements a 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0004-00.png)


Figure 2: The Framework of LLM-DA. Specifically, LLM-DA first analyzes historical data to extract temporal rules and utilizes the powerful generative capabilities of LLMs to generate general rules. Subsequently, LLM-DA updates these rules using current data. Finally, the updated rules are applied to predict future events. 

method that utilizes GPT for TKGs forecasting through in-context learning without any fine-tuning. Similarly, Mixtral-8x7B-CoH [23] also does not require fine-tuning and adopts a “chain-of-history” reasoning method to effectively generate high-order historical information step-by-step. Nevertheless, existing methods only prompt the LLMs with historical events from TKG, which are limited by the quality of input data and interpretability of LLMs. 

Different from the aforementioned LLM for TKGRs, the proposed LLM-DA is LLMs for rule-based TKGR method. By utilizing explicit rules, LLM-DA ensures the interpretability of the LLMgenerated processes and dynamically updates the rules to adapt to new data, thereby addressing a major limitation of LLMs-enhanced deep learning-based TKGRs. 

## **3 Preliminary** 

**Temporal Knowledge Graph (TKG).** TKG _G_ = _{E, R, T , Q}_ is a collection of entity set _E_ , relation set _R_ and timestamp set _T_ . Specifically, each quadruplet is denoted as ( _es, r, eo, t_ ) _∈Q_ , where _es, eo ∈E_ represent the entities, _r ∈R_ denotes the relation and _t ∈T_ is the timestamp. 

**Temporal Logical Rule.** Temporal logical rules _ρ_ define the relation between two entities _es_ and _eo_ at timestamp _tl_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0004-07.png)


where the left-hand side denotes the rule head with relation _r_ that can be induced by ( _←_ ) the righthand rule body. The rule body is represented by the conjunction ( _∧_ ) of a series of body relations _r[∗] ∈{r_ 1 _, ..., rl−_ 1 _}_ [50]. 

**Different Data Types.** _Historical data_ refers to data that have occurred in the past, reflecting the state of things in the past period of time. _Current data_ reflects the latest state of things in the present, from a more recent point in time. _Future data_ refers to data that will occur, reflecting the possible future trend of things. Specifically, the historical data, current data and future data correspond to the training, validation, and test datasets of prior research [47, 23]. 

## **4 Methodology** 

In this section, we propose a novel **L** arge **L** anguage **M** odel-guided **D** ynamic **A** daptation (LLMDA) method for TKGR tasks. LLM-DA contains four main stages: **Temporal Logical Rules Sampling** explores the constrained Markovian random walks to extract temporal logical rules from the historical data; **Rule Generation** utilizes the powerful generative capabilities of LLMs to extract 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0005-00.png)


Figure 3: The constraints of the constrained Markovian random walks. “�” denotes this path does not exist, “ _Pi_ ” indicates the transition probability. 

meaningful temporal patterns and complex temporal dependencies from historical data within TKGs; **Dynamic Adaptation** leverages LLMs to update the LLM-generated general rules using current data; **Candidate Reasoning** combines rules-based reasoning and graphs-based reasoning to generate candidates. The whole framework is illustrated in Figure 2. 

## **4.1 Temporal Logical Rules Sampling** 

Temporal logical rules sampling is a constrained Markovian random exploration process, which explores the _Timestamp-Entity_ joint level weighted temporal random walks. Generally, the constrained Markovian random exploration process is not only strictly constrained by the graph structure but also deeply influenced by the temporal dimension when choosing the next node. 

**Constrained Markovian Random Walks.** Compared to traditional Markov random walks, constrained Markovian random walks primarily reflect in two key factors such as temporal order and temporal intervals when choosing the next node. As shown in Figure 3, edges of the next state are selected based on the temporal order, and edges are weighted by the temporal interval. Specifically, edges with shorter interval receive higher weights, thus making the random walk more inclined to choose nodes that are temporally closer. 

To maximize the performance of random walks, LLM-DA needs to ensure that the random walks simultaneously satisfy the Markov property and additional constraints. Given the edge _r_ ( _ex, ey, tl_ ), LLM-DA employs the Markovian random walks to search the closed temporal paths, further obtaining the set of candidates _Mr_ . To ensure the efficiency of our framework, LLM-DA introduces the filtering operator _χ_ ( _t_ ), which explores the temporal order to filter these candidate paths. The filtering operator _χ_ ( _t_ ) can be denoted as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0005-07.png)


After filtering these candidate paths, LLM-DA further introduces the temporal interval as the another constraint. This constraint selects the next node based on the transition probability _P_ . For a temporal logical rule, the edge _r_ ( _ex, ey, t_ ) represents a connection from node _ex_ to node _ey_ at timestamp _t_ . Thus, the transition probability of the constrained Markovian random walks is expressed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0005-09.png)


where _Pxy_ ( _t_ ) is the probability of transitioning from node _ex_ to node _ey_ at time _t_ , and _w_ ( _t_ ) = exp( _−λ_ ( _t − T_ )) denotes an exponential decay function that weights the edges based on the time difference, _T_ is the current time, _t_ is the timestamp of the edge, and _λ_ is the decay rate parameter, which controls the weight given to more recent times. In this way, more recent times (i.e., _t_ closer to _T_ ) are assigned greater weight. The detailed theoretical analysis is shown in Appendix B. Through the constrained Markovian random walks on the historical data, LLM-DA extracts the temporal rules _S_ from the sampled temporal paths. 

## **4.2 Rule Generation** 

Rule generation typically utilizes the powerful generative capabilities of LLMs to improve the insufficient coverage and low quality of the extracted temporal rules _S_ . Specifically, LLM-DA first employs the contextual relation selector to identify the top _k_ relevant relations. Additonally, LLMs 

5 

generally have powerful generative capabilities, allowing them to extract meaningful temporal patterns and complex temporal dependencies from historical data within TKGs. Thus, LLM-DA inputs the extracted rules _S_ and the top _k_ relevant contextual relations into LLMs to generate high-coverage and high-quality general rules. 

Firstly, LLM-DA filters the input contextual information through a contextual relation selector. TKGs generally contain rich temporal relations, requiring the processing of a substantial amount of context when LLMs conduct TKGR tasks. Contextual relation selector aims to filter the relation pool to identify the top _k_ most relevant relations. For a temporal rule _ρ_ := _r_ ( _es, eo, tl_ ) _←_[�] _[l] i[−]_ =1[1] _[r][∗]_[(] _[e][s][, e][o][, t][i]_[)][,] LLM-DA leverages the pre-trained Sentence-Bert [51] to embed rule head _r_ and each of its corresponding candidate relations _cj_ into a common space, obtaining embedding vectors **c** _j_ and **r** : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0006-02.png)


Following, LLM-DA calculates the relevance score _S_ ( _r, cj_ ) through the cosine similarity function: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0006-04.png)


where _·_ denotes the dot product operation, and _∥· ∥_ denotes the norm of the vector. 

Finally, LLM-DA sorts the relevance scores in descending order and selects the top _k_ candidate relations as effective candidate relations, ensuring that the selected relations are semantically closely aligned with the rule head _r_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0006-07.png)


where _{cj_ 1 _, cj_ 2 _, . . . , cjk }_ are the top _k_ most relevant relations. 

After the above contextual relation selector, LLM-DA inputs the extracted temporal rules and the top _k_ most relevant candidate relations _{cj_ 1 _, cj_ 2 _, . . . , cjk }_ corresponding to the rule head _r_ into LLMs to achieve the generation of the high-coverage and high-quality general rules _Sg_ . A simple example of prompt is shown in the following prompt box: 

Prompt for Rule Generation 

You are an expert in TKGR, and please generate as many temporal logical rules as possible related to ‘ _r_ ’ based on extracted temporal rules. 

For the relations in rule body, you are going to choose from the candidate relations: “ _{cj_ 1 _, cj_ 2 _, . . . , cjk }_ ”. 

The detailed instruction for _Rule Generation_ can be found in the Appendix A.1. 

## **4.3 Dynamic Adaptation** 

Dynamic adaptation generally leverages LLMs to update the LLMs-generated general rules using current data. Due to the evolving nature of TKGs, the LLMs-generated rules _Sg_ become less suitable for new data, causing high-quality rules to gradually degrade into low-quality rules. To address this, LLM-DA extracts temporal rules from the current data and uses these rules as a standard to update the low-quality rules. This ensures that these rules always incorporate the most recent knowledge and better generalize to the predictions on future events. 

As shown in the _Prompt for Dynamic Adaptation_ prompt box, the input of LLMs for dynamic adaptation mainly contains two parts, including the update of low-quality rules and extracted rules from current data. 

## Prompt for Dynamic Adaptation 

You are an expert in TKGR, and please analyze these LLMs-generated rules and update the **low-quality rules** based on the **extracted rules** from current data. 

For the relations in rule body, you are going to choose from the candidate relations: “ _{cj_ 1 _, cj_ 2 _, . . . , cjk }_ ”. 

6 

_Low-Quality Rules._ As TKGs evolve over time, the LLM-generated rule set _Sg_ may become increasingly difficult to fit to the current data, eventually turning into low-quality rules. Thus, LLMDA applies the _Confidence_ [52] metric to score each temporal rule _ρ_ on the current data. _Confidence_ generally measures the reliability of the temporal rule _ρ_ , which can be defined as the proportion of temporal fact pairs that satisfy the rule body _rule_  body_ ( _ρ_ ) and also satisfy the whole _rule_ ( _ρ_ ): 

_cρ_ =[Number of tem] Number of temporal fact pairs satisfying[p][oral fact ][p][airs satisf][y][in][g] _[ rule] rule_[_] _[bod]_ ( _ρ[y]_ )[(] _[ρ]_[)] _,_ (7) 

where _cρ_ denotes the confidence of rule _ρ_ . The higher the confidence, the greater the reliability of the rule. In other words, when the rule body is true, the likelihood of the rule head being true is higher. Subsequently, we select the subset of rules with low confidence _Sg_ (low) = _{ρ ∈Sg | cρ < θ}_ , where _θ_ denotes the threshold of the low confidence. 

_Extracted Rules from Current Data._ To address the issue of the broad range of rules generated by LLMs, LLM-DA explores constrained Markovian random walks to extract temporal logical rules from current data, which serves as a standard to constrain the scope of dynamic adaptation. Through iteratively invoking LLMs, the accuracy of LLMs in predicting future events can be enhanced. The detailed prompt for _Dynamic Adaptation_ refers to Appendix A.2. 

Finally, LLM-DA updates these LLMs-generated low-quality rules through the extracted rules from current data, further obtaining the rules set _Sd_ . 

## **4.4 Candidate Reasoning** 

Candidate reasoning aims to infer potential answers for the query by integrating the above LLMsgenerated rules and GNNs-based predictions. Specifically, LLM-DA mainly consists of two key modules: **Rule-based Reasoning** and **Graph-based Reasoning** . 

**Rule-based Reasoning.** Rule-based reasoning typically utilizes the above LLMs-generated highscoring rules to conduct in-depth logical reasoning within TKGs, deducing new entities as potential answers [5]. Given a query ( _es, r,_ ? _, tl_ ), LLM-DA scores the rule through the Equation 7, and then select the high-scoring rules: 

_Sd[′]_[=] _[ {][ρ][ |][ c][ρ][> γ, ρ][ ∈S][d][}][,]_ (8) where _Sd_ is the rule set obtained after the _Dynamic Adaptation_ process, _Sd[′]_[is the high-confidence] rule set, in which the score _cρ_ of the rule _ρ_ is greater than the threshold _γ_ . Following the rule _ρ ∈Sd[′]_[,] we can find the reasoning paths and further derive the entity _e[′] o_[:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0007-09.png)


where _e[′] o_[is the candidate derived based on the rule] _[ ρ]_[, and] _[ t][l][−]_[1] _[≥· · ·][≥][t]_[1][.][Considering the time] decay property of temporal data, we further select candidates that are most relevant to the query: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0007-11.png)


wherebased on rule Score( _ρ,e ρ_ at the time point _′o_ ) indicates the score of the candidate _to_ , _cρ_ denotes the confidence of the temporal rule _e[′] o_[obtained through the searched path in TKGs] _ρ_ , _λ_ represents the decay rate, and _body_ ( _ρ_ )( _es, e[′] o[, t][l]_[)] _[ ∈G]_[denotes the path in TKGs that satisfies the rule body.] 

**Graph-based Reasoning.** Due to inconsistent data distribution, candidates generated solely based on rules may not fully match all query answers. Thus, we introduce the graph-based reasoning function _fg_ ( _Query_ ) to further predict the candidates of the query, and the score can be computed through the inner product operation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0007-14.png)


Since the candidates of _Rule-based Reasoning E_ ( _ρ,e′o_[)][ and] _[ Graph-based Reasoning][ E]_[(] _[graph,e][′] o_[)][ have] the overlap and difference, we assign the scoreand vice versa. The whole score of the candidate can be computed as follows: Score( _ρ,e′o_ ) as 0 where _e[′] o[∈E]_ ( _graph,e[′] o_ )[,] _[ e] o[′][∈E][/]_ ( _ρ,e[′] o_ )[,] Score _f_ = _α ·_ Score( _ρ,e′o_ ) + (1 _− α_ ) _·_ Score( _graph,e′o_ ) _,_ (12) 

where Score _f_ represents the final score of the candidate _e[′] o_[from rule-based reasoning and graph-based] reasoning modules, and _α_ is used to assign weights to different scores. Finally, LLM-DA aggregates all candidates, and then sorts these candidates to select those that best meet the query requirements. 

7 

Table 1: Link prediction results on ICEWS14 and ICEWS05-15. The best results are in bold and - means the result is unavailable. _♠_ denotes the TKGR methods, _♣_ represents the LLMs-based TKGR, and LLM-DA ( _·_ ) indicates replacing the graph-based reasoning module _fg_ ( _Query_ ) with TKGRs ( _♠_ ). 

|Type<br>Models<br>Train|ICEWS14<br>MRR<br>Hit@1<br>Hit@3<br>Hit@10|ICEWS05-15|
|---|---|---|
|||MRR<br>Hit@1<br>Hit@3<br>Hit@10|
|_♠_<br>RE-NET<br>�<br>RE-GCN<br>�<br>TiRGN<br>�<br>TLogic<br>�|0.388<br>0.290<br>0.436<br>0.576<br>0.425<br>0.320<br>0.476<br>0.627<br>0.441<br>0.341<br>0.497<br>0.650<br>0.390<br>0.295<br>0.437<br>0.573|0.441<br>0.332<br>0.512<br>0.650<br>0.478<br>0.371<br>0.535<br>0.682<br>0.495<br>0.389<br>0.559<br>0.703<br>0.459<br>0.360<br>0.518<br>0.646|
|_♣_<br>PPT<br>�<br>Llama-2-7b-CoH<br>�<br>Vicuna-7b-CoH<br>�<br>GPT-NeoX<br>�<br>Mixtral-8x7B-CoH<br>�|0.384<br>0.289<br>0.425<br>0.570<br>–<br>0.349<br>0.470<br>0.591<br>–<br>0.328<br>0.457<br>0.656<br>–<br>0.334<br>0.460<br>0.565<br>0.439<br>0.331<br>0.496<br>0.649|0.389<br>0.286<br>0.434<br>0.586<br>–<br>0.386<br>0.541<br>0.699<br>–<br>0.392<br>0.546<br>0.707<br>–<br>–<br>–<br>–<br>0.497<br>0.380<br>0.564<br>0.713|
|LLM-DA (RE-GCN)<br>�<br>LLM-DA(TiRGN)<br>�|0.461<br>0.356<br>0.515<br>0.662<br>**0.471**<br>**0.369**<br>**0.526**<br>**0.671**|0.501<br>0.394<br>0.568<br>0.710<br>**0.521**<br>**0.416**<br>**0.586**<br>**0.728**|



## **5 Experiments** 

## **5.1 Experimental Settings** 

**Datasets.** ICEWS14 [53] and ICEWS05-15 [53] are the subset of _Integrated Crisis Early Warning System (ICEWS)_ , which is a TKG of international political events and social dynamics. ICEWS14 contains events that occurred in 2014, while ICEWS05-15 contains events that occurred between 2005 and 2015. Details of datasets can be referred to Appendix C.1. 

**Baselines.** The proposed LLM-DA is compared with several classic TKGR methods, including 1) TKGR methods: RE-NET [44], RE-GCN [45], TiRGN [47] and TLogic [5]; 2) LLMs-based TKGRs: GPT-NeoX [49], Llama-2-7b-CoH, Vicuna-7b-CoH [24], Mixtral-8x7B-CoH [23] and PPT [22]. Here, LLM-DA selects RE-GCN and TiRGN as the graph-based reasoning function _fg_ ( _Query_ ). The detail of each baseline is described in Appendix C.2. 

**Parameter Setting.** The proposed LLM-DA uses the ChatGPT[4] as the LLM for _Rules Generation_ and _Dynamic Adaptation_ . LLM-DA chooses _Mean Reciprocal Rank_ (MRR) and _Hit@N_ ( _N_ = 1 _,_ 3 _,_ 10) as evaluation metrics, and presents the _filtered_ results (Appendix C.3). Additionally, LLM-DA sets the decay rate _λ_ in _Temporal Logical Rules Sampling_ and _Candidate Generation_ , the threshold _θ_ in _Dynamic Adaptation_ , the min-confidence _γ_ and the parameter _α_ in _Candidate Generation_ on both datasets as follows: _λ_ = 0 _._ 1, _θ_ = 0 _._ 01, _α_ = 0 _._ 9 and _γ_ = 0 _._ 01, except for _α_ = 0 _._ 8 on ICEWS05-15. The number of iterations for the _Dynamic Adaptation_ is set as 5. All experiments are implemented on a NVIDIA RTX 3090 GPU with i9-10900X CPU. 

## **5.2 Performance Comparison** 

The link prediction experimental results are displayed of ICEWS14 and ICEWS05-15 in Table 1, and the ICEWS18 is shown in Appendix C.4. The experimental analyses are listed as follows: 

(1) Experimental results indicate that even without fine-tuning, the proposed LLM-DA can still surpass all LLM-based TKGR methods. This phenomenon demonstrates that the dynamic adaptation strategy can effectively update LLMs-generated general rules with latest events to capture the evolving nature of TKGs, thereby significantly improving the accuracy of future event predictions. Additionally, we present the visualization experiment in Appendix C.6, which validates the superiority of the dynamic adaptation strategy. 

(2) Some LLMs-based TKGC methods such as PPT [22] and GPT-NeoX [49], are not always superior to traditional TKGR methods. This is primarily because the rules generated by LLMs are too broad and sometimes fail to precisely adapt to specific data. However, the proposed LLM-DA outperforms the existing state-of-the-art benchmarks on all metrics. This phenomenon proves that LLM-DA can effectively guide LLMs in adjusting rules to the target distribution. 

4 We use the snapshot of ChatGPT taken from February 15th 2024 (gpt-3.5-turbo-0215) to ensure the reproducibility. 

8 

(3) GPT-NeoX [49] is an important baseline as it also incorporates GPT as an LLM in TKGC tasks. However, LLM-DA shows significant improvement. This phenomenon indicates that the dynamic adaptation strategy can effectively update LLMs-generated rules to adapt to future data. 

(4) Furthermore, replacing the graph-based reasoning module _fg_ ( _Query_ ) with RE-GCN (“LLM-DA (RE-GCN)”) and TiRGN (“LLM-DA (TiRGN)”), the MRR performance shows a slight variation. This variation highlights the importance of incorporating graph-based reasoning function in enhancing the ability to predict future events. 

## **5.3 Analysis of Dynamic Adaptation** 

In LLM-DA, dynamic adaptation aims to continuously update the generated rules to capture temporal patterns and facilitate future predictions. To further investigate its impact, we aim to answer the following questions: **RQ1:** Can the dynamic adaptation better extract the temporal patterns from TKGR for reasoning? **RQ2:** Can the dynamic adaptation adapt to different distributions over time? **RQ3:** Can the iterative dynamic adaptation improve the performance? 

Table 2: Ablation study on different data without dynamic adaptation on both datasets. The best results are in bold. “LLM-DA _w_ H” indicates “only using the historical data”, “LLM-DA _w_ C” is “only using the current data”, and “LLM-DA _w_ H+C” denotes “using the historical and current data”. 

|onlyusingthe current d|ata”,and “LLM-DA_w_H+C” denotes “|usingthe historical and current data|
|---|---|---|
|Models|ICEWS14<br>MRR<br>Hit@1<br>Hit@3<br>Hit@10|ICEWS05-15|
|||MRR<br>Hit@1<br>Hit@3<br>Hit@10|
|LLM-DA (TiRGN)_w_H<br>LLM-DA (TiRGN)_w_C<br>LLM-DA(TiRGN)_w_H+C|0.450<br>0.345<br>0.502<br>0.656<br>0.454<br>0.350<br>0.507<br>0.657<br>0.457<br>0.352<br>0.510<br>0.659|0.503<br>0.395<br>0.563<br>0.709<br>0.508<br>0.400<br>0.570<br>0.714<br>0.511<br>0.402<br>0.573<br>0.718|
|LLM-DA(TiRGN)|**0.471**<br>**0.369**<br>**0.526**<br>**0.671**|**0.521**<br>**0.416**<br>**0.586**<br>**0.728**|



**RQ1:** The ablation study on different data without dynamic adaptation aims to evaluate the impact of dynamic adaptation on performance. Dynamic adaptation typically employs the current data to update the rules generated by LLMs based on historical data. To demonstrate the superiority, we compare three variations, including “ _LLM-DA w H_ ”, “ _LLM-DA w C_ ” and “ _LLM-DA w H+C_ ”, and the experimental results are shown in Table 2. Specifically, the “ _LLM-DA w H+C_ ” outperforms “ _LLM-DA w H_ ” and “ _LLM-DA w C_ ”, indicating that large-scale historical data can provide general knowledge, while current data offers relevant knowledge. The combination of both enhances the prediction of future events. Furthermore, compared to “ _LLM-DA w H+C_ ”, LLM-DA shows a significant improvement. This phenomenon demonstrates that the dynamic adaptation strategy can effectively integrate historical data and current data, leveraging the current data to update the rules generated by LLMs on historical data. 

**RQ2:** The temporal data has the time decay property, causing the issue of distributional shift. Specifically, the longer the time interval between the data, the more pronounced the shift becomes. To verify whether the dynamic adaptation can adapt to different distributions over time, we conduct the time interval segmented prediction experiment, which typically segments the future data into multiple time intervals based on chronological order, and then conducts the link prediction experiment for each time interval. As shown in Figure 4, the proposed LLM-DA exhibits a significant performance imFigure 4: Time interval segmented prediction: MRR perforprovement in the MRR metric over mance on both datasets compared to different baselines. RE-GCN and TiRGN in each time interval. This indicates that the pro- 

posed LLM-DA can accurately capture the temporal dependencies in TKGs and adapt to the continuously changing temporal data. Furthermore, in Appendix C.5, we conduct long-term horizontal link prediction to forecast events occurring at future time points. 

**RQ3:** To further verify whether the number of iterations of the dynamic adaptation module affects the performance, we conduct the different numbers of iterations experiment on both datasets. As shown in Figure 5, the MRR performance exhibits an increasing trend as the number of iterations 

9 

increases. These observations indicate that the dynamic adaptation strategy can continuously update the LLMs-generated rules with the latest events through iterations, thereby better adapting to the dynamic changes of TKGs. This further demonstrates the effectiveness and necessity of the dynamic adaptation strategy in handling the evolving nature of TKGs. Moreover, we conduct the parameter analysis experiment to validate the impact of the weight _α_ in Appendix C.7. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0010-01.png)


Figure 5: Comparison with different number of iterations on both datasets. 

## **6 Conclusion** 

In this paper, we propose a novel **L** arge **L** anguage **M** odel-guided **D** ynamic **A** daptation (LLM-DA) method to enhance TKGR tasks. Specifically, LLM-DA leverages a contextual relation selector to identify the top _k_ most relevant relations, thereby selecting pertinent contextual information. Subsequently, LLM-DA harnesses the generative capabilities of LLMs to analyze historical data and derive general rules. Furthermore, LLM-DA proposes a dynamic adaptation strategy to update the LLM-generated rules with latest events, further capturing the evolving nature of TKGs. Experimental results on several datasets unequivocally demonstrate that LLM-DA achieves competitive performance compared to state-of-the-art methods. Appendix D further analyzes the limitations of LLM-DA and provides an outlook for future work. 

## **References** 

- [1] Kalev Leetaru and Philip A Schrodt. GDELT: Global data on events, location, and tone. In _ISA Annual Convention_ , pages 1–49, 2013. 

- [2] Jiapu Wang, Boyue Wang, Junbin Gao, Xiaoyan Li, Yongli Hu, and Baocai Yin. TDN: Triplet distributor network for knowledge graph completion. _IEEE Transactions on Knowledge and Data Engineering_ , 35(12):13002–13014, 2023. 

- [3] Jiapu Wang, Boyue Wang, Meikang Qiu, Shirui Pan, Bo Xiong, Heng Liu, Linhao Luo, Tengfei Liu, Yongli Hu, Baocai Yin, et al. A survey on temporal knowledge graph completion: Taxonomy, progress, and prospects. _arXiv preprint arXiv:2308.02457_ , 2023. 

- [4] Ke Liang, Lingyuan Meng, Meng Liu, Yue Liu, Wenxuan Tu, Siwei Wang, Sihang Zhou, Xinwang Liu, Fuchun Sun, and Kunlun He. A survey of knowledge graph reasoning on graph types: Static, dynamic, and multi-modal. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2024. 

- [5] Yushan Liu, Yunpu Ma, Marcel Hildebrandt, Mitchell Joblin, and Volker Tresp. Tlogic: Temporal logical rules for explainable link forecasting on temporal knowledge graphs. In _the AAAI Conference on Artificial Intelligence_ , pages 4120–4127, 2022. 

- [6] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and S Yu Philip. A comprehensive survey on graph neural networks. _IEEE Transactions on Neural Networks and Learning Systems_ , 32(1):4–24, 2020. 

- [7] Yizhen Zheng, Shirui Pan, Vincent Lee, Yu Zheng, and Philip S Yu. Rethinking and scaling up graph contrastive learning: An extremely efficient approach with group discrimination. _Advances in Neural Information Processing Systems_ , 35:10809–10820, 2022. 

- [8] Yizhen Zheng, He Zhang, Vincent Lee, Yu Zheng, Xiao Wang, and Shirui Pan. Finding the missing-half: Graph complementary learning for homophily-prone and heterophily-prone graphs. In _International Conference on Machine Learning_ , pages 42492–42505. PMLR, 2023. 

10 

- [9] Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, and Kun He. TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting. In _the Conference on Empirical Methods in Natural Language Processing_ , pages 8306–8319, 2021. 

- [10] Linhao Luo, Jiaxin Ju, Bo Xiong, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Chatrule: Mining logical rules with large language models for knowledge graph reasoning. _arXiv preprint arXiv:2309.01538_ , 2023. 

- [11] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large language models and knowledge graphs: A roadmap. _IEEE Transactions on Knowledge and Data Engineering_ , pages 1–20, 2024. 

- [12] Tengfei Liu, Jiapu Wang, Yongli Hu, Mingjie Li, Junfei Yi, Xiaojun Chang, Junbin Gao, and Baocai Yin. HC-LLM: Historical-constrained large language models for radiology report generation. _arXiv preprint arXiv:2412.11070_ , 2024. 

- [13] Necati Cihan Camgoz, Simon Hadfield, Oscar Koller, Hermann Ney, and Richard Bowden. Neural sign language translation. In _the conference on Computer Vision and Pattern Recognition_ , pages 7784–7793, 2018. 

- [14] Jeffrey Dalton, Laura Dietz, and James Allan. Entity query feature expansion using knowledge base links. In _ACM SIGIR Conference on Research & Development in Information Retrieval_ , pages 365–374, 2014. 

- [15] Marco Antonio Calijorne Soares and Fernando Silva Parreiras. A literature review on question answering techniques, paradigms and systems. _Journal of King Saud University-Computer and Information Sciences_ , 32(6):635–646, 2020. 

- [16] Siheng Xiong, Ali Payani, Ramana Kompella, and Faramarz Fekri. Large language models can learn temporal reasoning. _arXiv preprint arXiv:2401.06853_ , 2024. 

- [17] Qingyu Tan, Hwee Tou Ng, and Lidong Bing. Towards benchmarking and improving the temporal reasoning capability of large language models. _arXiv preprint arXiv:2306.08952_ , 2023. 

- [18] Chenhan Yuan, Qianqian Xie, Jimin Huang, and Sophia Ananiadou. Back to the future: Towards explainable temporal reasoning with large language models. _arXiv preprint arXiv:2310.01074_ , 2023. 

- [19] Rujun Han, Xiang Ren, and Nanyun Peng. Econet: Effective continual pretraining of language models for event temporal reasoning. _arXiv preprint arXiv:2012.15283_ , 2020. 

- [20] Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, and Qingsong Wen. Time-llm: Time series forecasting by reprogramming large language models. In _International Conference on Learning Representations_ , 2024. 

- [21] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning. In _International Conference on Learning Representations_ , 2024. 

- [22] Wenjie Xu, Ben Liu, Miao Peng, Xu Jia, and Min Peng. Pre-trained language model with prompts for temporal knowledge graph completion. _arXiv preprint arXiv:2305.07912_ , 2023. 

- [23] Yuwei Xia, Ding Wang, Qiang Liu, Liang Wang, Shu Wu, and Xiaoyu Zhang. Enhancing temporal knowledge graph forecasting with large language models via chain-of-history reasoning. _arXiv preprint arXiv:2402.14382_ , 2024. 

- [24] Ruilin Luo, Tianle Gu, Haoling Li, Junzhe Li, Zicheng Lin, Jiayi Li, and Yujiu Yang. Chain of history: Learning and forecasting with llms for temporal knowledge graph completion. _arXiv preprint arXiv:2401.06072_ , 2024. 

11 

- [25] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. _arXiv preprint arXiv:2311.05232_ , 2023. 

- [26] Ruijie Wang, Zheng Li, Dachun Sun, Shengzhong Liu, Jinning Li, Bing Yin, and Tarek Abdelzaher. Learning to sample and aggregate: Few-shot reasoning over temporal knowledge graphs. _Advances in Neural Information Processing Systems_ , 35:16863–16876, 2022. 

- [27] Graham Cormode, Vladislav Shkapenyuk, Divesh Srivastava, and Bojian Xu. Forward decay: A practical time decay model for streaming systems. In _the International Conference on Data Engineering_ , pages 138–149, 2009. 

- [28] Tongtong Wu, Linhao Luo, Yuan-Fang Li, Shirui Pan, Thuy-Trang Vu, and Gholamreza Haffari. Continual learning for large language models: A survey. _arXiv preprint arXiv:2402.01364_ , 2024. 

- [29] Luyi Bai, Mingzhuo Chen, Lin Zhu, and Xiangxi Meng. Multi-hop temporal knowledge graph reasoning with temporal path rules guidance. _Expert Systems with Applications_ , 223:119804, 2023. 

- [30] Xingtong Ge, Yi Yang, Ling Peng, Luanjie Chen, Weichao Li, Wenyue Zhang, and Jiahui Chen. Spatio-temporal knowledge graph based forest fire prediction with multi source heterogeneous data. _Remote Sensing_ , 14(14):3496, 2022. 

- [31] Ke Liang, Yue Liu, Sihang Zhou, Wenxuan Tu, Yi Wen, Xihong Yang, Xiangjun Dong, and Xinwang Liu. Knowledge graph contrastive learning based on relation-symmetrical structure. _IEEE Transactions on Knowledge and Data Engineering_ , 36(1):226–238, 2023. 

- [32] Ke Liang, Lingyuan Meng, Sihang Zhou, Wenxuan Tu, Siwei Wang, Yue Liu, Meng Liu, Long Zhao, Xiangjun Dong, and Xinwang Liu. Mines: Message intercommunication for inductive relation reasoning over neighbor-enhanced subgraphs. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pages 10645–10653, 2024. 

- [33] Yili Wang, Yixin Liu, Xu Shen, Chenyu Li, Kaize Ding, Rui Miao, Ying Wang, Shirui Pan, and Xin Wang. Unifying unsupervised graph-level anomaly detection and out-of-distribution detection: A benchmark. _arXiv preprint arXiv:2406.15523_ , 2024. 

- [34] Jiapu Wang, Zheng Cui, Boyue Wang, Shirui Pan, Junbin Gao, Baocai Yin, and Wen Gao. IME: Integrating multi-curvature shared and specific embedding for temporal knowledge graph completion. _arXiv preprint arXiv:2403.19881_ , 2024. 

- [35] Jiapu Wang, Boyue Wang, Junbin Gao, Shirui Pan, Tengfei Liu, Baocai Yin, and Wen Gao. Made: Multicurvature adaptive embedding for temporal knowledge graph completion. _IEEE Transactions on Cybernetics_ , 2024. 

- [36] Shiyuan Li, Yixin Liu, Qingfeng Chen, Geoffrey I Webb, and Shirui Pan. Noise-resilient unsupervised graph representation learning via multi-hop feature quality estimation. In _Proceedings of the 33rd ACM International Conference on Information and Knowledge Management_ , 2024. 

- [37] Luyi Bai, Wenting Yu, Die Chai, Wenjun Zhao, and Mingzhuo Chen. Temporal knowledge graphs reasoning with iterative guidance by temporal logical rules. _Information Sciences_ , 621: 22–35, 2023. 

- [38] Siheng Xiong, Yuan Yang, Faramarz Fekri, and James Clayton Kerce. TILP: Differentiable learning of temporal logical rules on knowledge graphs. In _the International Conference on Learning Representations_ , pages 1–11, 2023. 

- [39] Xueyuan Lin, Chengjin Xu, Fenglong Su, Gengxian Zhou, Tianyi Hu, Ningyuan Li, Mingzhi Sun, Haoran Luo, et al. TFLEX: Temporal feature-logic embedding framework for complex reasoning over temporal knowledge graph. In _Advances in Neural Information Processing Systems_ , pages 73039–73081, 2023. 

12 

- [40] Songgaojun Deng, Huzefa Rangwala, and Yue Ning. Dynamic knowledge graph based multievent forecasting. In _the ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , pages 1585–1595, 2020. 

- [41] Jiapu Wang, Boyue Wang, Junbin Gao, Xiaoyan Li, Yongli Hu, and Baocai Yin. QDN: A quadruplet distributor network for temporal knowledge graph completion. _IEEE Transactions on Neural Networks and Learning Systems_ , 2023. 

- [42] Yixin Liu, Shiyuan Li, Yu Zheng, Qingfeng Chen, Chengqi Zhang, and Shirui Pan. Arc: A generalist graph anomaly detector with in-context learning. In _Advances in Neural Information Processing Systems_ , 2024. 

- [43] Jiapu Wang, Boyue Wang, Junbin Gao, Yongli Hu, and Baocai Yin. Multi-concept representation learning for knowledge graph completion. _ACM Transactions on Knowledge Discovery from Data_ , 17(1):1–19, 2023. 

- [44] Woojeong Jin, Meng Qu, Xisen Jin, and Xiang Ren. Recurrent Event Network: Autoregressive Structure Inferenceover Temporal Knowledge Graphs. In _the Conference on Empirical Methods in Natural Language Processing_ , pages 6669–6683, 2020. 

- [45] Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang, and Xueqi Cheng. Temporal knowledge graph reasoning based on evolutional representation learning. In _the International ACM SIGIR Conference on Research and Development in Information Retrieval_ , pages 408–417, 2021. 

- [46] Cunchao Zhu, Muhao Chen, Changjun Fan, Guangquan Cheng, and Yan Zhang. Learning from history: Modeling temporal knowledge graphs with sequential copy-generation networks. In _the AAAI Conference on Artificial Intelligence_ , pages 4732–4740, 2021. 

- [47] Yujia Li, Shiliang Sun, and Jing Zhao. TiRGN: Time-guided recurrent graph network with local-global historical patterns for temporal knowledge graph reasoning. In _the International Joint Conferences on Artificial Intelligence_ , pages 2152–2158, 2022. 

- [48] Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Haotian Wang, Ming Liu, and Bing Qin. Timebench: A comprehensive evaluation of temporal reasoning abilities in large language models. _arXiv preprint arXiv:2311.17667_ , 2023. 

- [49] Dong-Ho Lee, Kian Ahrabian, Woojeong Jin, Fred Morstatter, and Jay Pujara. Temporal knowledge graph forecasting without knowledge using in-context learning. _arXiv preprint arXiv:2305.10613_ , 2023. 

- [50] Eric Salvat and Marie-Laure Mugnier. Sound and complete forward and backward chainings of graph rules. In _the International Conference on Conceptual Structures_ , pages 248–262, 1996. 

- [51] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, _the Conference on Empirical Methods in Natural Language Processing_ , pages 3982–3992, 2019. 

- [52] Luis Antonio Galárraga, Christina Teflioudi, Katja Hose, and Fabian Suchanek. AMIE: association rule mining under incomplete evidence in ontological knowledge bases. In _the international conference on World Wide Web_ , pages 413–422, 2013. 

- [53] Alberto García-Durán, Sebastijan Dumanˇci´c, and Mathias Niepert. Learning sequence encoders for temporal knowledge graph completion. In _the Conference on Empirical Methods in Natural Language Processing_ , pages 4816–4821, 2018. 

13 

## **Appendix** 

## **A Prompt for LLMs** 

## **A.1 Prompt for Rule Generation** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0014-03.png)


**----- Start of picture text -----**<br>
Prompt for Rule Generation<br>Temporal Logical Rules “ Rl ( X, Y, Tl )  ←∧ [l] i [−] =1 [1] [R][i] [(] [X,] [Y,] [T][i] [)][” typically describe how the]<br>relation ‘ Rl ’ between entities ‘ X ’ and ‘ Y  ’ evolves from past time steps ‘ Ti  ( i  =  { 1 , · · · ,  ( l −<br>1) } )’ (Rule body) to the next timestamp ‘ Tl ’ (Rule head), and please follow the constraint<br>“<br>T 1 ≤· · · ≤ Tl− 1 < Tl ”.<br>You are an expert in temporal knowledge graph reasoning, and please generate as many<br>temporal logical rules as possible related to ‘ Rl ’ based on extracted temporal rules.<br>Here are a few examples:<br>Example 1:<br>Rule Head:<br>Cooperate_economically ( X ,  Y  ,  T )<br>Extracted Rules:<br>Cooperate_economically ( X ,  Y  ,  T 2)  ← Provide_aid ( X ,  Y  ,  T 1)<br>Cooperate_economically ( X ,  Y  ,  T 3)  ← Host_a_visit ( X ,  Z 1,  T 1) & Negotiate ( Z 1,<br>Y  ,  T 2)<br>···<br>Generated Temporal Logical Rules:<br>Cooperate_economically ( X ,  Y  ,  T 2)  ← Engage_in_negotiation ( X ,  Y  ,  T 1)<br>Cooperate_economically ( X ,  Y  ,  T 3)  ← inv_Engage_in_negotiation ( X ,  Z 1,  T 1) &<br>Make_a_visit ( Z 1,  Y  ,  T 2)<br>···<br>Example 2:<br>Rule Head:<br>Appeal_for_economic_aid ( X ,  Y  ,  T )<br>Extracted Rules:<br>Appeal_for_economic_aid ( X ,  Y  ,  T 2)  ← inv_Reduce_or_stop_military_assistance<br>( X ,  Y  ,  T 1)<br>Appeal_for_economic_aid ( X ,  Y  ,  T 3)  ← inv_Express_intent_to_cooperate ( X ,  Z 1,<br>T 1) & Make_statement ( Z 1,  Y  ,  T 2)<br>···<br>Generated Temporal Logical Rules:<br>Appeal_for_economic_aid ( X ,  Y  ,  T 2)  ← Make_an_appeal_or_request ( X ,  Y  ,  T 1)<br>Appeal_for_economic_aid ( X ,  Y  ,  T 3)  ← inv_Make_an_appeal_or_request ( X ,  Z 1,<br>T 1) & Make_statement ( Z 1,  Y  ,  T 2)<br>···<br>Extracted Rules from Historical Data:<br>······<br>Let’s think step-by-step, please generate as many as possible most relevant temporal rules<br>that are relative to " { head_rule( X, Y, T ) } " based on the above extracted rules from historical<br>data.<br>For the relations in rule body, you are going to choose from the candidate relations:<br>“ {cj 1 , cj 2 , . . . , cjk } ”.<br>Return the rules only without any explanations.<br>**----- End of picture text -----**<br>


14 

## **A.2 Prompt for Dynamic Adaptation** 

Prompt for Dynamic Adaptation Temporal Logical Rules “ _Rl_ ( _X, Y, Tl_ ) _←∧[l] i[−]_ =1[1] _[R][i]_[(] _[X,][Y,][T][i]_[)][” typically describe how the] relation ‘ _Rl_ ’ between entities ‘ _X_ ’ and ‘ _Y_ ’ evolves from past time steps ‘ _Ti_ ( _i_ = _{_ 1 _, · · · ,_ ( _l −_ 1) _}_ )’ (Rule body) to the next timestamp ‘ _Tl_ ’ (Rule head), and please follow the constraint “ _T_ 1 _≤· · · ≤ Tl−_ 1 _< Tl_ ”. You are an expert in temporal knowledge graph reasoning, and please analyze these LLMs-generated rules and update the low-quality rules based on the extracted rules from current data. Here are a few examples: **Example 1:** Rule Head: inv_Provide_humanitarian_aid ( _X_ , _Y_ , _T_ ) Low Quality Temporal Logical Rules: Make_a_visit ( _X_ , _Y_ , _T_ 2) _←_ Provide_military_protection_or_peacekeeping ( _X_ , _Y_ , _T_ 1) Make_a_visit ( _X_ , _Y_ , _T_ 4) _←_ Appeal_for_diplomatic_cooperation_(such_as_policy_ support) ( _X_ , _Z_ 1, _T_ 1) & inv_Consult ( _Z_ 1, _Z_ 2, _T_ 2) & inv_Make_statement ( _Z_ 2, _Y_ , _T_ 3) Generated High Quality Temporal Logical Rules: Make_a_visit ( _X_ , _Y_ , _T_ 2) _←_ Express_intent_to_meet_or_negotiate ( _X_ , _Z_ 1, _T_ 1) & Make_a_visit ( _Z_ 1, _Y_ , _T_ 2) Make_a_visit ( _X_ , _Y_ , _T_ 3) _←_ Consult ( _X_ , _Z_ 1, _T_ 1) & Engage_in_negotiation ( _Z_ 1, _Z_ 2, _T_ 2) & Make_a_visit ( _Z_ 2, _Y_ , _T_ 3) ··· **Example 2:** Rule Head: inv_Provide_humanitarian_aid ( _X_ , _Y_ , _T_ ) Low Quality Temporal Logical Rules: inv_Provide_humanitarian_aid ( _X_ , _Y_ , _T_ 2) _←_ inv_Investigate ( _X_ , _Y_ , _T_ 1) inv_Provide_humanitarian_aid ( _X_ , _Y_ , _T_ 2) _←_ inv_Engage_in_diplomatic_ cooperation ( _X_ , _Y_ , _T_ 1) Generated High Quality Temporal Logical Rules: inv_Provide_humanitarian_aid ( _X_ , _Y_ , _T_ 2) _←_ inv_Provide_aid ( _X_ , _Y_ , _T_ 1) inv_Provide_humanitarian_aid ( _X_ , _Y_ , _T_ 3) _←_ Criticize_or_denounce ( _X_ , _Z_ 1, _T_ 1) & Sign_formal_agreement ( _Z_ 1, _Y_ , _T_ 2) ... **Low-quality Temporal Logical Rules:** ······ **Extracted Rules from Current Data:** ······ Let’s think step-by-step, and please update the low-quality temporal logic rules related to “ _{head_  rule_ ( _X, Y, T_ ) _}_ ” based on the extracted rules from current data. For the relations in rule body, you are going to choose from the candidate relations: “ _{cj_ 1 _, cj_ 2 _, . . . , cjk }_ ”. Return the rules only without any explanations. 

15 

## **B Theoretical Analysis of the Constrained Markovian Random Walks** 

In conducting a theoretical analysis of constrained Markovian random walks, we can examine their characteristics and effects from several key aspects: 

- **Impact of Time Weighting:** By using weighted transition probabilities _Pxy_ ( _tl_ ), constrained Markovian random walks can assign different importance to neighboring nodes at different time points. In particular, through the exponential decay function _w_ ( _t_ ) = exp( _−λ_ ( _tl − T_ )), we can control the extent to which time influences the transition probabilities. The larger the parameter _λ_ , the stronger the preference for recent events, which helps capture short-term dynamic changes. 

- **Traversal Properties:** An important feature of constrained Markovian random walks is their ability to traverse different paths in TKGs. Since the transition probabilities consider time information, constrained Markovian random walks are more likely to explore paths with temporal continuity compared to traditional random walks, thereby better reflecting temporal relationships. 

## **C Experiments** 

## **C.1 Datasets** 

Table 3 provides a comprehensive overview of the statistical information for the entire datasets used in our experiments. This includes key metrics such as the number of entities, the number of relations, the number of temporal facts, and the granularity of each dataset. This statistical information is crucial for understanding the scale and complexity of the datasets, as well as for evaluating the performance of LLM-DA under different experimental conditions. 

_Integrated Crisis Early Warning System (ICEWS)_ is a TKG for international crisis early warning and analysis, which collects and integrates data on political events, social dynamics, and international relations worldwide. Specifically, ICEWS14 [53] contains events that occurred from 1 _/_ 1 _/_ 2014 to 12 _/_ 31 _/_ 2014; ICEWS05-15 [53] includes events that occurred between the years 2005 and 2015; ICEWS18 [23] covers events from 1 _/_ 1 _/_ 2018 to 10 _/_ 31 _/_ 2018. 

Table 3: Statistic information of whole datasets. 

|Datasets|#Entities|#Relations|#Historical Data|#Current Data|#Future Data|#Granularity|
|---|---|---|---|---|---|---|
|ICEWS14|6,869|230|74,845|8,514|7,371|24 hours|
|ICEWS05-15|10,094|251|368,868|46,302|46,159|24 hours|
|ICEWS18|23,033|256|373,018|45,995|49,545|24 hours|



## **C.2 Baselines** 

The proposed model LLM-DA is compared with some classic TKGR methods, including TKGR methods and LLMs-based TKGR methods. 

## **TKGR methods** 

- RE-NET [44]: RE-NET first TKGC method for predicting future events, which employs a recurrent event encoder and a neighborhood aggregator to encode historical facts; 

- RE-GCN [45]: RE-GCN is an extension of RE-NET, which introduces RGCN and GRU to encode historical facts; 

- TiRGN [47]: TiRGN employs constrained random walks for temporal logical rule learning; 

- TLogic [5]: TLogic learns the temporal logical rules from TKGs based on temporal random walks; 

## **LLMs-based TKGR methods** 

- GPT-NeoX [49]: GPT-NeoX introduces in-context learning with GPT for TKG forecasting; 

16 

- Llama-2-7b-CoH, Vicuna-7b-CoH [24]: Llama-2-7b-CoH, Vicuna-7b-CoH fine-tune Llama2-7b and Vicuna-7b on historical data for TKG prediction; 

- Mixtral-8x7B-CoH [23]: Mixtral-8x7B-CoH leverage Mixtral-8x7B to explore high-order histories step-by-step for the TKG prediction; 

- PPT [22]: PPT introduces the Pre-trained Language Model to predict future events. 

## **C.3 Link Prediction Metrics** 

We evaluate our model using Mean Reciprocal Rank (MRR) and Hit@ _N_ metrics, where _N_ is set to 1, 3, and 10. Higher scores indicate better performance. Lastly, we present the final experimental results, termed as _filtered_ , which exclude all corrupted quadruplets from the TKG. 

- Mean Reciprocal Rank (MRR): MRR measures the average accuracy of ranking predictions made by a model. It is calculated by taking the reciprocal of the rank of the correct answer for each query and then averaging these reciprocal ranks across all queries. Mathematically, for each query, if the rank of the correct answer is _r_ , then the MRR score for that query is[1] _r_[.] The overall MRR score is the average of these reciprocal ranks. MRR values range between 0 and 1, where a higher value indicates better performance in ranking predictions. 

- Hit@ _N_ : Hit@ _N_ measures the proportion of correct predictions within the top _N_ ranked results. For example, when _N_ = 1, we are interested in whether the correct answer is predicted within the top-ranked result. When _N_ = 3, we evaluate whether the correct answer is within the top three predictions. The value of _N_ can be chosen based on the specific task and dataset characteristics, commonly set to 1, 3, or 10. Hit@ _N_ values also range between 0 and 1, representing the proportion of correct predictions. 

## **C.4 Link Predication on ICEWS18** 

The link prediction experimental results of ICEWS18 are displayed in Table 4. The proposed LLMDA outperforms existing state-of-the-art benchmarks on ICEWS18 across all metrics. Moreover, GPT-NeoX [49] is an important baseline because it also introduces GPT as LLMs. However, our proposed method still improves all metrics on ICEWS18. For example, the proposed method obtains 15 _._ 6% improvement under Hit@10 on ICEWS18. These phenomena demonstrate that the dynamic adaptation strategy can effectively update LLM-generated rules with current data, ensuring that the rules consistently incorporate the most recent knowledge and better generalize predictions for future events. 

Table 4: Link prediction results on ICEWS18 dataset. The best results are in bold and - means the result is unavailable. _♠_ denotes the TKGR methods, _♣_ represents the LLMs-based TKGR. The results of the models with _†_ are derived from [23], and _‡_ is from [24]. 

|Type<br>Models<br>Train|ICEWS18|
|---|---|
||MRR<br>Hit@1<br>Hit@3<br>Hit@10|
|_♠_<br>RE-NET_†_ [44]<br>�<br>RE-GCN [45]<br>�<br>TLogic_‡_ [5]<br>�<br>TiRGN[47]<br>�|0.287<br>0.188<br>0.327<br>0.482<br>0.326<br>0.223<br>0.367<br>0.526<br>–<br>0.205<br>0.340<br>0.485<br>0.336<br>0.232<br>0.379<br>0.542|
|_♣_<br>PPT [22]<br>�<br>Llama-2-7b-CoH_‡_ [24]<br>�<br>Vicuna-7b-CoH_‡_ [24]<br>�<br>GPT-NeoX [49]<br>�<br>Mixtral-8x7B-CoH_†_ [23]<br>�|0.266<br>0.169<br>0.306<br>0.454<br>–<br>0.223<br>0.363<br>0.522<br>–<br>0.209<br>0.347<br>0.536<br>–<br>0.192<br>0.313<br>0.414<br>0.330<br>0.218<br>0.378<br>0.549|
|LLM-DA(TiRGN)<br>�|**0.357**<br>**0.255**<br>**0.403**<br>**0.570**|



17 

## **C.5 Long Horizontal Link Prediction** 

Long horizontal link prediction generally predict the events happening at time point _t_ + _k ∗△T_ , where _△T_ denotes the interval between time points and _k_ controls how far we will predict. To perform the long horizontal link prediction, we adjust the integral length _△T_ on ICEWS14 and ICEWS05-15 datasets. As described in Figure 6 and Figure 7, we report the results corresponding to different _△T_ on ICEWS14 and ICEWS05-15 and compare the proposed LLM-DA with the strongest baselines RE-GCN [45] and TiRGN [47]. Experimental results shows that LLM-DA can achieve the best performance on different _△T_ both datasets. This phenomenon demonstrates that LLMs can effectively learn meaningful rules, and the dynamic adaptation strategy can fully update the LLM-generated rules, allowing for better adaptation to the constantly changing dynamics within TKGs. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0018-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0018-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0018-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0018-05.png)


Figure 6: Long horizontal link forecasting: time-aware filtered MRR on ICEWS14 with respect to different time points. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0018-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0018-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0018-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0018-10.png)


Figure 7: Long horizontal link forecasting: time-aware MRR performance on ICEWS05-15 with respect to different time points. 

18 

## **C.6 Visualization Experiment** 

To visually present the changes of rule and candidate rankings during the _Rule Generation_ and _Dynamic Adaptation_ , we conducted a visualization experiment in Table 5. For the query _(China, Consult, ?, 2014-12-05)_ , the actual answer **Japan** ranked fifth during the _Rule Generation_ stage, but after the _Dynamic Adaptation_ , it advanced to the top position. Meanwhile, the sequencing of rules in two phases also exhibits significant changes. For instance, the ranking of the rule " _Consult ← inv_Make_an_appeal_or_request_ & _Express_· · · _meet_or_negotiate_ " changes from fifth to first position. This highlights the efficacy of the dynamic adaptation strategy, which can update the LLM-generated rules with latest events to capture the evolving nature of TKGs. 

Table 5: Visualization experiment for query _(China, Consult, ?, 2014-12-05)_ . _Rules for Rule Generation_ and _Rules for Dynamic Adaptation_ represent the dynamic changes of the rules during the _Rule Generation_ and _Dynamic Adaptation_ . _Candidates for Rule Generation_ and _Candidates for Dynamic Adaptation_ indicate the changes in the corresponding candidate rankings. “ **Japan** ”is the actual answer. 

|_ynamic Adaptation_indicate<br>ctual answer.|the changes in the corresponding candidate rankings. “**Japan**”is th|
|---|---|
|Query|(China,Consult,?,2014-12-05)|
||①inv_Host_a_visit&Host_a_visit&inv_Discuss_by_telephone|
||②Meet_at_a_‘third’_location|
|Rules for Rule Generation|③Express_intent_to_engage_in_diplomatic_cooperation&inv_Host_a_visit&inv_Consult<br>④Consult&Express_intent_to_cooperate&Consult|
||⑤inv_Make_an_appeal_or_request&Express_intent_to_meet_or_negotiate|
||①Malaysia|
||②Cambodia|
|Candidates for Rule Generation|③China<br>④South_Korea|
||⑤**Japan**|
|Query|(China,Consult,?,2014-12-05)|
||①inv_Make_an_appeal_or_request&Express_intent_to_meet_or_negotiate|
||②Consult&Express_intent_to_cooperate&Consult|
|Rules for Dynamic Adaptation|③Meet_at_a_‘third’_location<br>④Express_intent_to_engage_in_diplomatic_cooperation&inv_Host_a_visit&inv_Consult|
||⑤inv_Host_a_visit&Host_a_visit&inv_Discuss_by_telephone|
||①**Japan**|
||②France|
|Candidates for Dynamic Adaptation|③South_Korea<br>④South_Africa|
||⑤Malaysia|



> 1 **Express_intent_to_engage_in_diplomatic_cooperation** represents _Express_intent_to_engage_in_diplomatic_cooperation_(such_as_policy_support)_ . 

## **C.7 Parameter Analysis** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning_images/nips_2024_Large_Language_Models-guided_Dynamic_Adaptation_for_Temporal_Knowledge_Graph_Reasoning.pdf-0019-06.png)


Figure 8: Comparison with different weight _α_ on both datasets. 

To experimentally study the effect of the weight _α_ on ICEWS14 and ICEWS05-15, we tune the weight in a range _{_ 0 _,_ 0 _._ 1 _, · · · ,_ 1 _}_ and observe the link prediction experimental results. As shown in Figure 8, the MRR performance exhibits a trend of initially increasing and then decreasing, reaching the peak at _α_ = 0 _._ 9 for ICEWS14, and _α_ = 0 _._ 8 for ICEWS05-15. These observations underscore that the weight _α_ plays a crucial role in influencing the performance of LLM-DA. Specifically, the weight 

19 

_α_ shows better performance at higher values, highlighting the importance of rule-based reasoning module. 

Moreover, when _α_ = 1, indicating that only rules are used to generate candidates, the MRR performance decreases. This demonstrates that although LLMs can learn meaningful rules, the rules generated based on historical data cannot fully fit the future data due to the difference in distribution. This results in a few queries failing to find matching candidates when relying solely on rules for candidate generation. 

## **D Limitations** 

Our limitations can be summarized as follows: (1) LLM-DA does not consider the semantics of nodes, which may reduce the quality of the sampled rules. (2) LLM-DA does not generate rules based on queries, which may result in the rules generally lacking specificity. (3) LLM-DA requires manually constructed prompts, which means that prompts need to be redesigned for different datasets. Moreover, manual construction of prompts also incurs significant costs. In the future, we will employ automated prompt learning to generate query-dependent rules through LLMs. Furthermore, we integrate the semantics of nodes during the temporal logical rules sampling stage. 

20 

