## **FACTKG: Fact Verification via Reasoning on Knowledge Graphs** 

**Jiho Kim**[1] **, Sungjin Park**[1] **, Yeonsu Kwon**[1] **, Yohan Jo**[2] _[∗]_ **, James Thorne**[1] **, Edward Choi**[1] 1KAIST 2Amazon 

{jiho.kim, zxznm, yeonsu.k, thorne, edwardchoi}@kaist.ac.kr jyoha@amazon.com 

## **Abstract** 

In real world applications, knowledge graphs (KG) are widely used in various domains (e.g. medical applications and dialogue agents). However, for fact verification, KGs have not been adequately utilized as a knowledge source. KGs can be a valuable knowledge source in fact verification due to their reliability and broad applicability. A KG consists of nodes and edges which makes it clear how concepts are linked together, allowing machines to reason over chains of topics. However, there are many challenges in understanding how these machine-readable concepts map to information in text. To enable the community to better use KGs, we introduce a new dataset, FACTKG: Fact Verification via Reasoning on Knowledge Graphs. It consists of 108k natural language claims with five types of reasoning: One-hop, Conjunction, Existence, Multi-hop, and Negation. Furthermore, FACTKG contains various linguistic patterns, including colloquial style claims as well as written style claims to increase practicality. Lastly, we develop a baseline approach and analyze FACTKG over these reasoning types. We believe FACTKG can advance both reliability and practicality in KG-based fact verification.[1] 

## **1 Introduction** 

The wide spread risk of misinformation has increased the demand for fact-checking, that is, judging whether a claim is true or false based on evidence. Accordingly, recent works on fact verification have been developed with various sources of evidence, such as text (Thorne et al., 2018; Augenstein et al., 2019; Jiang et al., 2020; Schuster et al., 2021; Park et al., 2021) and tables (Chen et al., 2019; Wang et al., 2021; Aly et al., 2021). Unfortunately, knowledge graphs (KG), one of the 

> _∗_ This work is not associated with Amazon. 

> 1Data available at https://github.com/jiho283/ FactKG. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs_images/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs.pdf-0001-09.png)


Figure 1: An example data from FACTKG. To verify the claim whether it is SUPPORTED or REFUTED, we use triples extracted from DBpedia as evidence. 

large-scale data forms, have not yet been fully utilized as a source of evidence. A KG is a valuable knowledge source due to two advantages. 

Firstly, KG-based fact verification can provide more reliable reasoning: since the efficacy of realworld fact-checking hinges on this reliability, recent studies have focused on justifying the decisions of a fact verification system (Kotonya and Toni, 2020a). In most existing works, the justification is based on the extractive summary of text evidence. Therefore, the inferential links between the evidence and the verdict are not clear (Kotonya and Toni, 2020b; Atanasova et al., 2020a,b). Compared to text and tables, a KG can simply represent reasoning process with logic rules on nodes and edges (Liang et al., 2022). This allows us to categorize common types of reasoning with the graphical structure, as shown in Table 1. 

Secondly, KG-based fact verification techniques have broad applicability beyond the domain of factchecking. For example, modern dialogue systems (e.g. _Amazon Alexa_ (Amazon Staff, 2018), _Google Assistant_ (Kale and Hewavitharana, 2018)) maintain and communicate with internal knowledge graphs, and it is crucial to make sure that their con- 

|**Reasoning Type**|**Claim Example**|||**Graph**|**Graph**|**Graph**|
|---|---|---|---|---|---|---|
|**One-hop**|AIDAstella was built by Meyer Werft.||_s_||_r_2|_m_|
||||||||
|**Conjunction**|AIDA Cruise line operated the AIDAstella which was built by<br>Meyer Werft.|_c_|_r_3||_s_|_m_<br>_r_2|
||||||||
|**Existence**|Meyer Werft had a parent company.||_m_||_r_1||
||||||||
|**Multi-hop**|AIDAstella was built by a company in Papenburg.|_s_|_r_2||_x_|_p_<br>_r_4|
||||||||
|**Negation**|AIDAstella wasnotbuilt by Meyer Werft in Papenburg.|_s_|_r_2||_m_|_p_<br>_r_4|
||||||||



Table 1: Five different reasoning types of FACTKG. _r_ 1: parentCompany, _r_ 2: shipBuilder, _r_ 3: shipOperator, _r_ 4: location, _m_ : Meyer Werft, _s_ : AIDAstella, _c_ : AIDA Cruises. 

tent is consistent with what the user says and otherwise update the knowledge graphs accordingly. If we model the user’s utterance as a claim and the dialogue system’s internal knowledge graph as a knowledge source, the process of checking their consistency can be seen as a form of KG-based fact verification task. More generally, KG-based fact verification techniques can be applied to cases which require checking the consistency between graphs and text. 

Reflecting these advantages, we introduce a new dataset, FACTKG: Fact Verification via Reasoning on Knowledge Graphs, consisting of 108k textual claims that can be verified against DBpedia (Lehmann et al., 2015) and labeled as SUPPORTED or REFUTED. We generated the claims based on graph-text pairs from WebNLG (Gardent et al., 2017) to incorporate various reasoning types. The claims in FACTKG are categorized into five reasoning types: One-hop, Conjunction, Existence, Multi-hop, and Negation. Furthermore, FACTKG consists of claims in various styles including colloquial, making it potentially suitable for a wider range of applications, including dialogue systems. 

We conducted experiments on FACTKG to validate whether graph evidence had a positive effect for fact verification. Our experiments indicate that the use of graphical evidence in our model resulted in superior performance when compared to baselines that did not incorporate such evidence. 

## **2 Related Works** 

## **2.1** 

There are various types of knowledge used in fact verification such as text, tables, and knowledge graphs. Research on fact verification has mainly focused on text data as evidence (Thorne et al., 2018; 

Augenstein et al., 2019; Jiang et al., 2020; Schuster et al., 2021; Park et al., 2021). FEVER (Thorne et al., 2018), one of the representative fact verification datasets, is a large-scale manually annotated dataset derived from Wikipedia. Other recent works leverage ambiguous QA pairs (Park et al., 2021), factual changes (Schuster et al., 2021), multiple documents (Jiang et al., 2020), or claims sourced from fact checking websites (Augenstein et al., 2019). Fact verification on table data is also studied (Chen et al., 2019; Wang et al., 2021; Aly et al., 2021). Table-based datasets such as SEMTAB-FACTS (Wang et al., 2021) or TabFact (Chen et al., 2019) require reasoning abilities over tables, and FEVEROUS (Aly et al., 2021) validate claims utilizing table and text sources. We refer the reader to Guo et al. (2022) for a comprehensive survey. 

There have been several tasks that utilize knowledge graphs (Dettmers et al., 2018). For example, FB15K (Bordes et al., 2013), FB15K237 (Toutanova and Chen, 2015), and WN18 (Bordes et al., 2013) are built upon subsets of largescale knowledge graphs, Freebase (Bollacker et al., 2008) and WordNet (Miller, 1995) respectively. These datasets only use a single triple as a claim, and thus the claims only require One-hop reasoning. However, FACTKG is the first KG-based fact verification dataset with natural language claims that require complex reasoning. In terms of the evidence KG size, FACTKG uses the entire DBpedia (0.1B triples), which is significantly larger than previous datasets (FB15K: 592K, FB15K-237: 310K, WN18: 150K). 

## **2.2 WebNLG** 

As constructing a KG-based fact verification dataset requires a paired text-graph corpus, we uti- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs_images/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs.pdf-0003-00.png)


Figure 2: Two substitution methods utilized in FACTKG. In _Entity substitution_ , we select a new entity located in outside 4-hops from all entities in the original claim. If the results of bidirectional NLI are both contradiction, we finish this process. In _Relation substitution_ , we randomly extract a relation that takes the same entity types for the head and tail as the original relation. Then, substitution is performed based on a template specific to the selected relation. 

lized WebNLG as a basis for FACTKG. WebNLG is a dataset for evaluating triple-based natural language generation, which consists of 25,298 pairs of high-quality text and RDF triples from DBpedia. WebNLG contains diverse forms of graphs and the texts are created by linguistic experts, which gives it great variety and sophistication. In the 2020 challenge[2] , the dataset has been expanded to 45,040 text-triples pairs. We used this 2020 version of WebNLG when constructing our dataset. 

## **3 Data Construction** 

Our goal is to diversify the graph reasoning patterns and linguistic styles of the claims. To achieve this, we categorize five reasoning types of claims: One-hop, Conjunction, Existence, Multi-hop, and Negation. Our claims are generated by transforming the sentences in _Sw_ , a subset of WebNLG’s text-graph pairs (Section 3.1).[3] Next, we also diversified the claims with colloquial style transfer and presupposition (Section 3.2). 

## **3.1 Claim Generation** 

## **3.1.1 One-hop** 

The most basic type of claim is one-hop, which covers only one knowledge triple. One-hop claims can 

> 2https://webnlg-challenge.loria.fr/challenge_ 2020/ 

> 3We found that 99.7% of claims in FEVER and FEVEROUS consist of a single sentence. To reflect this result, we extract a subset _Sw_ containing only single sentences from WebNLG. 

be verified by checking the existence of a single corresponding triple. In the second row of Table 1, the claim is SUPPORTED when the triple ( _AIDAstella_ , _ShipBuilder_ , _Meyer Werft_ ) exists. 

We take the sentences that consist of a single triple in _Sw_ as SUPPORTED claims. REFUTED claims are created by substituting SUPPORTED claims in two ways: _Entity substitution_ and _Relation substitution_ . In _Entity substitution_ , we replace an entity _e_ in SUPPORTED claim _C_ with another entity ˜ _e_ of the same entity type. In order to ensure that the label of the substituted sentence _C_[˜] is REFUTED, the entity _e_ ˜ should satisfy the following two conditions. _i)_ To select _e_ ˜ that is irrelevant to _C_ , _e_ ˜ is outside 4-hops from all entities in _C_ on DBpedia, _ii)_ the results of NLI ( _C_ , _C_[˜] ) and NLI ( _C_[˜] , _C_ ) are both CONTRADICTION.[4] In _Relation substitution_ , we replace a relation in the SUPPORTED claim with another relation. We replace the relation of a triple in the claim with another relation that takes the same entity types for the head and tail as the original relation (e.g. currentTeam _↔_ formerTeam). The four groups of compatible relations are listed in Table 6. The overall process of the substitution methods is illustrated in Figure 2. 

> 4 We use a natural language inference (NLI) model, RoBERTa-base (Liu et al., 2019) finetuned on the MNLI dataset (Williams et al., 2018). The notation NLI ( _p_ , _h_ ) represents the result of NLI when _p_ is assigned as the premise and _h_ as the hypothesis. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs_images/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs.pdf-0004-00.png)


Figure 3: Graph patterns used in Conjunction and Multi hop claims. 

## **3.1.2 Conjunction** 

A claim in the real world can include a mixture of different facts. To incorporate this, we construct a conjunction claim composed of multiple triples. Conjunction claims are verified by the existence of all corresponding triples. In the third row of Table 1, the claim can be divided into two parts: _“AIDA Cruise line operated the AIDAstella.”_ and _“AIDAstella was built by Meyer Werft.”_ . The claim is SUPPORTED when all the triples ( _AIDAstella_ , _ShipOperator_ , _AIDA Cruises_ ), ( _AIDAstella_ , _ShipBuilder_ , _Meyer Werft_ ) exist. To implement this idea, we extracted sentences consisting of more than one triple from _Sw_ and used them as the SUPPORTED claims. To create REFUTED claims, we use _Entity substitution_ method on these SUPPORTED claims. 

## **3.1.3 Existence** 

People may make claims that assert the existence of something (e.g. _“She has two kids.”_ ). From the view of a triple, this corresponds to the head or tail missing. To reflect this scenario, we formulate a claim by extracting only { _head_ , _relation_ } or { _tail_ , _relation_ } from a triple. Existence claims are generated using templates and they are divided into two categories: _head-relation_ (e.g. template: { _head_ } had a(an) { _relation_ }.) and _tailrelation_ (e.g. template: { _tail_ } was a { _relation_ }.). SUPPORTED claims are constructed by randomly extracting { _head_ , _relation_ } or { _tail_ , _relation_ } in triples from _Sw_ . The REFUTED claims are constructed using the same type of entities as represented in the claim, but with different relations. However, it is possible that unrealistic claims may be generated in this manner. For example, _“Meyer Werft had a location.”_ or _“Papenburg was a location.”_ can be created from the triple ( _Meyer Werft_ , _location_ , _Papenbug_ ). Hence, we selected 22 relations out of all relations that lead to realistic claims. Templates used for both categories and examples of generated claims are in Table 7. 

## **3.1.4 Multi-hop** 

We also consider multi-hop claims that require the validation of multiple facts where some entities are underspecified. Entities in this claim can be connected by a sequence of relations. For example, the multi-hop claim in Table 1 is SUPPORTED if the triple ( _AIDAstella_ , _ShipBuilder_ , _x_ ) and the triple ( _x_ , _location_ , _Papenburg_ ) are present in the graph. The goal is to verify the existence of a path on the graph that starts from _AIDAstella_ and reaches _Papenburg_ through the relations _ShipBuilder_ and _location_ . 

Figure 3 shows how a SUPPORTED multi-hop claim _CM_ can be generated by replacing an entity _e_ of the conjunction claim _C_ with its type name. First, an entity _e_ is selected from the green nodes. Then, the type name _t_ of the entity _e_ is extracted from DBpedia. However, each entity _e_ in DBpedia has several types _T_ = _{t_ 1 _, t_ 2 _, ..., tN }_ , and it is not annotated which type is relevant when _e_ is used in a claim. So it is necessary to select one of them. For each _tn ∈ T_ , we insert it next to the entity _e_ in the claim _C_ and measure the perplexity score of the modified claim using GPT2-large (Radford et al., 2019). Then we replace _e_ in the claim with the type name that had the lowest score. The REFUTED claim is generated by applying _Entity substitution_ to the SUPPORTED claim. 

## **3.1.5 Negation** 

For each of the four methods for generating claims, we develop claims that incorporate negations. 

**One-hop** We use the Negative Claim Generation Model (Lee et al., 2021) which was fine-tuned on the opposite claim set in the WikiFactCheckEnglish dataset (Sathe et al., 2020).[5] To ensure the quality of the generated sentences, we generate 100 opposing claims for each original claim, then only use those that preserve all entities, and contain negations (e.g. _‘not’_ or _‘never’_ ). Also, similar to _Entity substitution_ method, we only use sentences whose NLI relation with the original sentences are CONTRADICTION bidirectionally. When a negation is added, the label of the generated claim is reversed from the original claim. 

**Conjunction** The use of negations (i.e., _‘not’_ ) in various positions within conjunction claims allows the generation of a wide range of negative claim structures. We employ the pretrained language model GPT-J 6B (Wang and Komatsuzaki, 

5WikiFactCheck-English consists of pairs of claims, with a positive claim and its corresponding negative claim. 

2021) to attach negations to the claim. We construct 16 in-context examples, each with negations attached to the texts corresponding to the first or/and second relation. When a negation is added to the SUPPORTED claims, all the claims become REFUTED. However, when it is added to REFUTED claims, the label depends on the position of the negation. When negations are added to all parts with substituted entities, it becomes a SUPPORTED claim. Conversely, other cases preserve the label REFUTED since the negation is added to a place that is not related to entity substitution. A detailed labeling strategy is described in Appendix D.1. 

**Existence** The claim is formulated by adding a negation within the templates presented in Section 3.1.3 (e.g. { _tail_ } was not a { _relation_ }.). 

**Multi-hop** A claim is formulated using the GPT-J with in-context examples, similar to conjunction. The truth of this claim is dependent on the presence of a distinctive path that matches the claim’s intent. For example, the negative claim _“AIDAstella was built by a company, not in Papenburg.”_ is SUPPORTED if _x_ exists where the triples ( _AIDAstella_ , _ShipBuilder_ , _x_ ) and ( _x_ , _location_ , _y_ ) are in DBpedia and _y_ is not _Papenburg_ . A more detailed labeling strategy is in Appendix D.2. 

## **3.2 Colloquial Style Transfer** 

We transform the claims into a colloquial style via style transfer using both a fine-tuned language model and presupposition templates. 

## **3.2.1 Model based** 

Using a similar method proposed by Kim et al. (2021), we transform the claim obtained from 3.1 into a colloquial style. For example, the claim _“Obama was president.”_ is converted to _“Have you heard about Obama? He was president!”_ . 

We train FLAN T5-large (Chung et al., 2022) to generate a colloquial style sentence given a corresponding written style sentence from Wizard of Wikipedia (Dinan et al., 2019). However, using sentences generated by the model could have several potential issues: _i)_ the original and generated sentences are lexically the same, _ii)_ some entities are missing in the generated sentences, _iii)_ the generated sentences deviate semantically from the original, _iv)_ the generated sentences lack a colloquialism, as mentioned in Kim et al. (2021). To overcome this, we oversample candidate sentences and utilize an additional filtering process. 

First, to make more diverse samples using the 

model, we set the temperature to 20.0 and generate 500 samples with beam search. _i)_ To avoid generated sentences that are too similar to the original sentences, only sentences with an edit distance of 6 or more from the original sentence are selected among 500 samples. _ii)_ Then, only those that have verbs and the named entities all preserved are selected.[6] _iii)_ Finally, we use bidirectional NLI to preserve the original semantics. Candidate sentences survive when NLI ( _O_ , _G_ ) is ENTAILMENT and NLI ( _G_ , _O_ ) is not CONTRADICTION where _O_ refers to the original sentence and _G_ the generated sentence. On average, only 41.2 generated sentences survived out of 500 samples. Additionally, in cases where none of the 500 generated sentences pass the filtering process, we include the original claim in the final dataset as a written style claim. Following the filtering process, the AFLITE method (Sakaguchi et al., 2019), which utilizes adversarial filtering, is applied to select the most colloquial style sentence among the surviving sentences. We include the selected claim in the final dataset as a colloquial style claim. 

## **3.2.2 Presupposition** 

A presupposition is something the speaker assumes to be the case prior to making an utterance (Yule and Widdowson, 1996). People often communicate under the presupposition that their beliefs are universally accepted. We construct claims using this form of utterance. The claims in FACTKG are focused on three types of presupposition: factive, non-factive, and structural presuppositions. 

**Factive Presupposition** People frequently use verbs like _“realize”_ or _“remember”_ to express the truth of their assumptions. The utterance _“I remembered that_ {Statement} _.”_ assumes that {Statement} is true. Reflecting these features, a new claim is created by appending expressions that contain presupposition (e.g. _“I realized that” or “I wasn’t aware that”_ ) to the existing claim. We used eight templates to make factive presupposition claims: the details are appended in Table 8. 

**Non Factive Presupposition** The verbs such as _“wish”_ are commonly used in utterances that describe events that have not occurred. For example, people say _“I wish that_ {Statement} _.”_ when {Statement} did not happen. Claims that are created by 

6NLTK (Bird et al., 2009) POS tagger and Stanza (Qi et al., 2020) NER module are used. DBPedia entities are already tagged in each claim, but not all entities exist in the sentence in their raw form, so the NER module is used. 

the non-factive presupposition method are labeled as the opposite of the original one. We used three templates to make these claims: the templates are appended in Table 8. 

**Structural Presupposition** This type is in the form of a question that presumes certain facts. We treat the question itself as a claim. For example, _“When was Messi in Barcelona?”_ assumes that _Messi was in Barcelona_ . To create a natural sentence form, only claims corresponding to one-hop and existence are constructed. For the one-hop claim, a different template was created corresponding to each relation reflecting its meaning (e.g. “When did { _head_ } die from { _tail_ }?” for the relation _deathCause_ and “When was { _head_ } directed by { _tail_ }?” for relation _director_ ). Existence claims are also generated based on templates (e.g. “When was { _tail_ } { _relation_ }?”) using pairs of _head_ - _relation_ or _tail_ - _relation_ , similar to Section 3.1.3. The templates used are described in Table 9. 

## **3.3 Quality Control** 

To evaluate the quality of our dataset, the labeling strategy and the output of the colloquial style transfer model are assessed. 

**Labeling Strategy** When SUPPORTED claims are made in the manner described in Section 3.1, the labeling is straightforward, as all have precise evidence graphs. However, REFUTED claims are generated by random substitution, so there might be a small chance that they remain SUPPORTED (e.g. _“The White House is in Washington, D.C.”_ to _“The White House is in America.”_ ). To evaluate this substitution method, randomly sampled 1,000 substituted claims were reviewed by two graduate students. As a result, 99.4% of generated claims were identified as REFUTED by both participants. 

**Colloquial Style Transfer Model** We also evaluate the quality of the colloquial style claims generated by the model. A survey was conducted on all claims in the test set by three graduate students. As a result, only 9.8% of the claims were selected as _Loss of important information_ by at least two reviewers. In addition, to ensure the quality of the test set, only claims that were selected as _All facts are preserved_ by two or more reviewers are included in the test set. The survey details are in Appendix E. 

|**Type**<br>**Written**|**Colloquial**<br>**Total**<br>**Model**<br>**Presup**|
|---|---|
|One-hop<br>2,106<br>Conjuction<br>20,587<br>Existence<br>280<br>Multi-hop<br>10,239<br>Negation<br>1,340|15,934<br>1,580<br>19,530<br>15,908<br>602<br>37,097<br>4,060<br>4,832<br>9,172<br>16,420<br>603<br>27,262<br>12,466<br>1,807<br>15,613|
|Total<br>34,462|64,788<br>9,424<br>108,674|



Table 2: Dataset statistics of FACTKG for all reasoning types. 

## **4 Experiments** 

## **4.1 Dataset Statistics** 

Table 2 shows the statistics of FACTKG. We split the claims into train, dev, and test sets with a proportion of 8:1:1. We ensured that the set of triples in each split is disjoint with the ones in other splits. 

## **4.2 Experimental Setting** 

We publish FACTKG with sets of claims, graph evidence and labels. The graph evidence includes entities and a set of relation sequences connected to them. For instance, when the claim is given as _“AIDAstella was built by a company in Papenburg.”_ , the entity _‘AIDAstella’_ corresponds to a set of relation sequence [ _shipBuilder_ , _location_ ] and _‘Papenburg’_ corresponds to [ _∼location_ , _∼shipBuilder_ ].[7] In the test set, we only provide entities as graph evidence. 

## **4.3 Baseline** 

We conduct experiments on FACTKG to see how the graphical evidence affects the fact verification task. To this end, we divided our baselines into two distinct categories based on the input type, _Claim Only_ and _With Graphical Evidence_ . 

## **4.3.1 Claim Only** 

In the _Claim Only_ setting, the baseline models receive only the claim as input and predict the label. We used three transformer-based text classifiers, BERT, BlueBERT, and Flan-T5. BERT (Devlin et al., 2018) is trained on Wikipedia from which DBpedia is extracted. So we expect that the model will use evidence memorized in its pretrained weights (Petroni et al., 2019) or exploit structural patterns in the generated claims (Schuster et al., 2019; Thorne and Vlachos, 2021). BlueBERT (Peng et al., 2019) is trained on biomedical corpus, such as Pubmed abstracts. We use 

> 7‘ _∼_ ’ indicates that the direction of the relation is reversed 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs_images/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs.pdf-0007-00.png)


Figure 4: Overall process of our baseline. In the subgraph retrieval step, each classifier respectively predicts the relations and hops related to the given entity and the claim. Subsequently, we check all the n-hop relation sequences obtained from each classifier to find all evidence paths. In the fact verification step, the claim is verified by leveraging all outputs obtained from the subgraph retrieval step. In this figure, we denote Transformer Encoder as TRM. 

BlueBERT as a comparator for BERT since it has never seen Wikipedia during its pre-training. FlanT5 (Chung et al., 2022) is an enhanced version of T5 (Raffel et al., 2022) encoder-decoder that has been fine-tuned in a mixture of 1.8K tasks. In all experiments, we fine-tune BERT and BlueBERT on our training set. Different from BERT and BlueBERT, we use Flan-T5 in the zero-shot setting. For this setting, we use _“Is this claim True or False? Claim: ”_ as the prefix. Then, we measure the probability that tokens _True_ and _False_ will appear in the output. Among the two tokens, we choose the one with the higher probability. 

## **4.3.2 With Graphical Evidence** 

In the _With Graphical Evidence_ setting, the model receives the claim and graph evidence as input and predicts the label. The baseline we used is a framework proposed by GEAR (Zhou et al., 2019) that enables reasoning on multiple evidence texts. Since GEAR was originally designed to reason over text passages, we change components to suit KG. The modified GEAR consists of the subgraph retrieval module and the claim verification module. The pipeline of the modified GEAR is illustrated in Figure 4. 

**Subgraph retrieval** We replace document retrieval and sentence selection in GEAR with subgraph retrieval. To retrieve graphical evidence, we train two independent BERT models, namely a relation classifier and a hop classifier. The relation classifier predicts the set of relations _R_ from the claim _c_ and the entity _e_ . The hop classifier is de- 

signed to predict the maximum number of hops _n_ to be traversed from _e_ . We take the subgraph of _G_ that are composed only of the relations in _R_ and where the terminal nodes are entities in _C_ and less than _n_ hops apart from _e_ , allowing for duplicates and considering the order. By traversing the knowledge graph starting from _e_ along the relation sequences in _P_ , we choose the paths that can reach another entity that appears in the claim. If none of the paths is reachable to other entities, then we randomly choose one of the paths. The strategy we used enables the model to retrieve supported evidence and counterfactual evidence for the given claim. The following example is presented to assist the understanding of our subgraph retrieval method. The example claim in Section 4.2 consists of two entities, _‘AIDAstella’_ and _‘Papenburg’_ . In this setting, the hop classifier must predict 2 since those entities are connected by a sequence of two relations, namely _shipBuilder_ and _location_ . In addition, the relation classifier must predict correctly predict those two relations. After that, we find all 2-hop paths starting from _‘AIDAstella’_ along the predicted relations in the knowledge graph. If there is a path that reaches _‘Papenburg’_ , we can use it as supporting evidence. If not, however, we randomly select a path. 

**Fact verification** We directly employed the claim verification in GEAR and applied some changes to suit the KG setting. Since our evidence is a set of graph paths, we converted them to text by concatenating each triple with the special token <SEP>. We also found that ERNet in GEAR 

|**Input Type**|**Model**|**One-hop**<br>**Conjunction**<br>**Existence**<br>**Multi-hop**<br>**Negation**|**Total**|
|---|---|---|---|
|_Claim Only_|BERT<br>BlueBERT<br>Flan-T5|69.64<br>63.31<br>61.84<br>70.06<br>63.62<br>60.03<br>60.15<br>59.89<br>57.79<br>58.90<br>62.17<br>69.66<br>55.29<br>60.67<br>55.02|65.20<br>59.93<br>62.70|
|_With Evidence_|GEAR|83.23<br>77.68<br>81.61<br>68.84<br>79.41|77.65|



Table 3: Fact verification accuracy on FACTKG. 

is identical to the Transformer encoder, so we replaced it with a randomly initialized Transformer encoder. To make this paper self-contained, we provide further details about the claim verification of GEAR in Appendix F. 

## **4.4 Results** 

> **Fact Verification Results** We evaluated the performance of the models in predicting labels and reported the accuracy in Table 3 by different reasoning types. 

As we expected, GEAR outperforms other baseline models in most of reasoning types because it used graph evidence. Especially, in existence and negation, GEAR substantially outperforms _Claim Only_ baselines. Since the existence claims contain significantly less information than other types, having to search for evidence seems to increase fact verification performance. In addition, negation claims require additional inference steps compared to other types, thus logical reasoning based on graph evidence would help the model make correct prediction. 

In the multi-hop setting, however, the accuracy of GEAR is lower than BERT, which may be due to the increased complexity of graph retrieval. When entities are far apart with many intermediate nodes being under-specified, it increases the probability of retrieving an incorrect graph. In GEAR, text and evidence paths are concatenated and used as input, so if many incorrect graphs are retrieved, they can lead to incorrect predictions. Also, the accuracy of BERT is the most superior in the multi-hop setting, which suggests that masked language modeling facilitates the model to robustly handle unspecified entities in the multi-hop claims. 

In the _Claim Only_ setting, all baselines outperform the Majority Class (51.35%), and the BERT model shows the highest performance. BlueBERT was pre-trained in the same manner, but BERT shows superior performance due to its pre-trained knowledge from Wikipedia. 

|**Input Type**|**Model**|**W**_→_**W**<br>**W**_→_**C**|**C**_→_**C**<br>**C**_→_**W**|
|---|---|---|---|
|_Claim Only_|BERT<br>BlueBERT|71.75<br>63.85<br>64.76<br>56.28|68.10<br>69.43<br>58.77<br>63.92|
|_With Evidence_|GEAR|81.00<br>75.43|80.81<br>78.80|



Table 4: **W** refers to written style claims and **C** refers to colloquial style claims. **W** _→_ **C** means that the model is trained on the written style claim set and tested on the colloquial style claim set. Flan-T5 is not used in this experiment because we use it only in the zero-shot setting. 

**Cross-Style Evaluation** We split the dataset into two disjoint sets, written style and colloquial style. We perform a cross-style fact verification task by using those datasets and the results are reported in Table 4. 

Initially, we anticipated that using different styles for the train and test set would result in a significant decrease in verification performance. However, contradict our expectation, in **C** _→_ **W** setting, BERT and BlueBERT show an improvement in performance over **C** _→_ **C** . Even in GEAR, the performance score only dropped slightly. Therefore, the results demonstrate that colloquial style is constructed in various forms which can be beneficial for generalization. 

## **5 Conclusion** 

In this paper, we present FACTKG, a new dataset for fact verification using knowledge graph. In order to reveal the relationship between fact verification and knowledge graph reasoning, we generated claims corresponding to a certain graph pattern. Additionally, FACTKG also includes colloquialstyle claims that are applicable to the dialogue system. Our analysis showed that the claims in our dataset are difficult to solve without reasoning over the knowledge graph. 

We expect the dataset to offer various research directions. One possible use of our dataset is as a benchmark for justification prediction. Most research on this task generate a text passage as justification, yet this approach lacked a gold reference. On the contrary, the interpretability of the knowl- 

edge graph allows us to employ it as an explanation for the verdict, such as question answering in the medical domain where explainability is important. Furthermore, using the KG structure for the claim generation allows us to generate a dataset with more complex multi-hop reasoning by design without relying on annotator creativity. 

## **Limitations** 

Since WebNLG is derived from 2015-10 version of DBpedia, FACTKG does not reflect the latest knowledge. Also, another limitation of our work is that the claims of FACTKG are constructed based on single sentences, like other crowdsourced fact verification datasets. If the claim is generated by more than one sentences, the dataset will be more challenging. We remain this challenging point as a future work. 

## **Acknowledgements** 

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant (No.2019-000075, No.2022-0-00984), and National Research Foundation of Korea (NRF) grant (NRF2020H1D3A2A03100945), funded by the Korea government (MSIT). 

## **References** 

- Rami Aly, Zhijiang Guo, Michael Schlichtkrull, James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Oana Cocarascu, and Arpit Mittal. 2021. Feverous: Fact extraction and verification over unstructured and structured information. _arXiv preprint arXiv:2106.05707_ . 

Amazon Staff. 2018. How alexa keeps getting smarter. 

- Pepa Atanasova, Jakob Grue Simonsen, Christina Lioma, and Isabelle Augenstein. 2020a. A diagnostic study of explainability techniques for text classification. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 3256–3274, Online. Association for Computational Linguistics. 

- Pepa Atanasova, Jakob Grue Simonsen, Christina Lioma, and Isabelle Augenstein. 2020b. Generating fact checking explanations. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 7352–7364, Online. Association for Computational Linguistics. 

- Isabelle Augenstein, Christina Lioma, Dongsheng Wang, Lucas Chaves Lima, Casper Hansen, Christian Hansen, and Jakob Grue Simonsen. 2019. Mul- 

tifc: A real-world multi-domain dataset for evidencebased fact checking of claims. _arXiv preprint arXiv:1909.03242_ . 

- Steven Bird, Ewan Klein, and Edward Loper. 2009. _Natural language processing with Python: analyzing text with the natural language toolkit_ . " O’Reilly Media, Inc.". 

- Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. 2008. Freebase: a collaboratively created graph database for structuring human knowledge. In _Proceedings of the 2008 ACM SIGMOD international conference on Management of data_ , pages 1247–1250. 

- Antoine Bordes, Nicolas Usunier, Alberto GarciaDuran, Jason Weston, and Oksana Yakhnenko. 2013. Translating embeddings for modeling multirelational data. In _Advances in Neural Information Processing Systems_ , volume 26. Curran Associates, Inc. 

- Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and William Yang Wang. 2019. Tabfact: A largescale dataset for table-based fact verification. _arXiv preprint arXiv:1909.02164_ . 

- Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2022. Scaling instruction-finetuned language models. 

- Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. 2018. Convolutional 2d knowledge graph embeddings. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 32. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT: pre-training of deep bidirectional transformers for language understanding. _CoRR_ , abs/1810.04805. 

- Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason Weston. 2019. Wizard of Wikipedia: Knowledge-powered conversational agents. In _Proceedings of the International Conference on Learning Representations (ICLR)_ . 

- Claire Gardent, Anastasia Shimorina, Shashi Narayan, and Laura Perez-Beltrachini. 2017. The webnlg challenge: Generating text from rdf data. In _Proceedings of the 10th International Conference on Natural Language Generation_ , pages 124–133. 

- Zhijiang Guo, Michael Schlichtkrull, and Andreas Vlachos. 2022. A survey on automated fact-checking. _Transactions of the Association for Computational Linguistics_ , 10:178–206. 

- Yichen Jiang, Shikha Bordia, Zheng Zhong, Charles Dognin, Maneesh Singh, and Mohit Bansal. 2020. Hover: A dataset for many-hop fact extraction and claim verification. _arXiv preprint arXiv:2011.03088_ . 

- Ajinkya Gorakhnath Kale and Sanjika Hewavitharana. 2018. Knowledge graph construction for intelligent online personal assistant. US Patent App. 15/238,679. 

- Byeongchang Kim, Hyunwoo Kim, Seokhee Hong, and Gunhee Kim. 2021. How robust are fact checking systems on colloquial claims? In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 1535–1548, Online. Association for Computational Linguistics. 

- Neema Kotonya and Francesca Toni. 2020a. Explainable automated fact-checking: A survey. In _Proceedings of the 28th International Conference on Computational Linguistics_ , pages 5430–5443, Barcelona, Spain (Online). International Committee on Computational Linguistics. 

- Neema Kotonya and Francesca Toni. 2020b. Explainable automated fact-checking for public health claims. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 7740–7754, Online. Association for Computational Linguistics. 

- Minwoo Lee, Seungpil Won, Juae Kim, Hwanhee Lee, Cheoneum Park, and Kyomin Jung. 2021. Crossaug: A contrastive data augmentation method for debiasing fact verification models. In _Proceedings of the 30th ACM International Conference on Information & Knowledge Management_ , CIKM ’21. Association for Computing Machinery. 

- Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas, Pablo N Mendes, Sebastian Hellmann, Mohamed Morsey, Patrick Van Kleef, Sören Auer, et al. 2015. Dbpedia–a large-scale, multilingual knowledge base extracted from wikipedia. _Semantic web_ , 6(2):167–195. 

- Ke Liang, Lingyuan Meng, Meng Liu, Yue Liu, Wenxuan Tu, Siwei Wang, Sihang Zhou, Xinwang Liu, and Fuchun Sun. 2022. Reasoning over different types of knowledge graphs: Static, temporal and multi-modal. 

- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized BERT pretraining approach. _CoRR_ , abs/1907.11692. 

- George A Miller. 1995. Wordnet: a lexical database for english. _Communications of the ACM_ , 38(11):39– 41. 

- Jungsoo Park, Sewon Min, Jaewoo Kang, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2021. Faviq: Fact verification from information-seeking questions. _arXiv preprint arXiv:2107.02153_ . 

- Yifan Peng, Shankai Yan, and Zhiyong Lu. 2019. Transfer learning in biomedical natural language processing: An evaluation of bert and elmo on ten benchmarking datasets. In _Proceedings of the 2019 Workshop on Biomedical Natural Language Processing (BioNLP 2019)_ , pages 58–65. 

- Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander Miller. 2019. Language models as knowledge bases? In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLPIJCNLP)_ , pages 2463–2473, Hong Kong, China. Association for Computational Linguistics. 

- Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton, and Christopher D. Manning. 2020. Stanza: A Python natural language processing toolkit for many human languages. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations_ . 

- Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2022. Exploring the limits of transfer learning with a unified text-to-text transformer. _J. Mach. Learn. Res._ , 21(1). 

- Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2019. Winogrande: An adversarial winograd schema challenge at scale. _arXiv preprint arXiv:1907.10641_ . 

- Aalok Sathe, Salar Ather, Tuan Manh Le, Nathan Perry, and Joonsuk Park. 2020. Automated fact-checking of claims from wikipedia. In _Proceedings of The 12th Language Resources and Evaluation Conference_ , pages 6874–6882, Marseille, France. European Language Resources Association. 

- Tal Schuster, Adam Fisch, and Regina Barzilay. 2021. Get your vitamin C! robust fact verification with contrastive evidence. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 624–643, Online. Association for Computational Linguistics. 

- Tal Schuster, Darsh Shah, Yun Jie Serene Yeo, Daniel Roberto Filizzola Ortiz, Enrico Santus, and Regina Barzilay. 2019. Towards debiasing fact verification 

models. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 3419–3425, Hong Kong, China. Association for Computational Linguistics. 

- James Thorne and Andreas Vlachos. 2021. Elastic weight consolidation for better bias inoculation. In _Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume_ , pages 957–964, Online. Association for Computational Linguistics. 

- James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a large-scale dataset for fact extraction and VERification. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)_ , pages 809–819, New Orleans, Louisiana. Association for Computational Linguistics. 

- Kristina Toutanova and Danqi Chen. 2015. Observed versus latent features for knowledge base and text inference. In _Proceedings of the 3rd Workshop on Continuous Vector Space Models and their Compositionality_ , pages 57–66, Beijing, China. Association for Computational Linguistics. 

- Ben Wang and Aran Komatsuzaki. 2021. GPT-J6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/ mesh-transformer-jax. 

- Nancy XR Wang, Diwakar Mahajan, Marina Danilevsky, and Sara Rosenthal. 2021. Semeval2021 task 9: Fact verification and evidence finding for tabular data in scientific documents (sem-tab-facts). _arXiv preprint arXiv:2105.13995_ . 

- Adina Williams, Nikita Nangia, and Samuel Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)_ , pages 1112–1122. Association for Computational Linguistics. 

- G. Yule and H.G. Widdowson. 1996. _Pragmatics_ . Oxford Introduction to Language Study ELT. OUP Oxford. 

- Jie Zhou, Xu Han, Cheng Yang, Zhiyuan Liu, Lifeng Wang, Changcheng Li, and Maosong Sun. 2019. GEAR: Graph-based evidence aggregating and reasoning for fact verification. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 892–901, Florence, Italy. Association for Computational Linguistics. 

## **A Qualitative analysis** 

We report claims and the retrieved graphical evidence in Table A. We also report the correctness of the prediction of GEAR at the first column of our table, **Result** . We used subgraph retrieval to retrieve graph path visualize one of them. By checking the retrieved evidence, We can recognize why the model verdict the claims as refuted or supported. This shows that our graph evidence is fully interpretable. 

## **B Relation Substitution** 

The four groups of compatible relations are listed in Table 6. 

## **C Full List of Templates** 

## **C.1 Existence** 

The templates to generate existence claims are described in Table 7. 

## **C.2 Factive and Non Factive Presupposition** 

Factive and Non Factive presupposition templates are in Table 8. 

## **C.3 Structural Presupposition** 

Structural presupposition templates are in Table 9. 

## **D Negation Labeling** 

## **D.1 Conjunction** 

When the negation is added to REFUTED claims, the label depends on the position of the negation. If negations are added to all parts with substituted entities, it becomes a SUPPORTED claim. Conversely, other cases preserve the label REFUTED since the negation is added to a place that is not related to entity substitution. Detailed examples are described in Table 10 and Table 11. 

## **D.2 Multi-hop** 

The truth of this claim is dependent on the presence of a distinctive path that matches the claim’s intent. For example, when verifying the claim in the fourth row of the Table 12, we check the existence of an entity which is connected to _‘AIDAstella’_ with relation _builder_ and not connected to _‘New York’_ with relation _location_ . 

## **E Colloquial Style Claim Survey** 

A total of 9 graduate students participated in the survey to evaluate how much information was lost in the colloquial style claim compared to original claim. Since each person has different criteria for ‘important information’, the labels are divided into five rather than two. The labels are as follows, i) All facts are preserved, ii) Minor loss of information or minor grammatical errors, iii) Ambiguous whether the lost information is important, iv) It is ambiguous, but the lost information may be important, v) Loss of important information. And as a result, only 9.8% of the claims were selected as v) Loss of important information by at least two reviewers. 

## **F Details of GEAR** 

To make this paper self-contained, we recall some details of the claim verification in GEAR (Zhou et al., 2019). The authors of GEAR (Zhou et al. (2019)) used sentence encoder to obtain representations for the claim and the evidence. Then they built a fully-connected evidence graph and used evidence reasoning network (ERNet) to propagate information between evidence and reason over the graph. Finally, they used an evidence aggregator to 

## **Sentence Encoder** 

Given an input sentence, Zhou et al. (2019) employed BERT (Devlin et al., 2018) as a sentence encoder by extracting the final hidden state of the [CLS] token as the representation. 

Specifically, given a claim _c_ and _N_ pieces of retrieved evidence _{e_ 1 _, e_ 2 _, ..., eN }_ , they fed each evidence-claim pair ( _ei, c_ ) into BERT to obtain the evidence representation **ei** . they also fed the claim into BERT alone to obtain the claim **c** . That is, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs_images/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs.pdf-0012-23.png)


## **Evidence Reasoning Network** 

Let **h** _[t]_ = _{_ **h** _[t]_ 1 _[,]_ **[ h]** _[t]_ 2 _[, ...,]_ **[ h]** _[t] N[}]_[ denote the hidden states] of the nodes in layer _t_ , where **h** _[t] i[∈R][F][×]_[1][ and] _[ F]_[is] the number of features in each node. The initial hidden state of each evidence node **h**[0] _i_[was initialized] by the evidence: **h**[0] _i_[=] **[e]** _[i]_[.][The authors proposed] an Evidence Reasoning Network (ERNet) to propagate information among the evidence nodes. They 

|**Result**|**Claim**|**Retrieved Path**|
|---|---|---|
|_Correct_|Yeah! Alfredo Zitarrosa died in a city in Uruguay<br>I have heard that Mobyland had a successor.|(Uruguay, country, Montevideo, deathPlace, Alfredo_Zitarrosa)<br>(Mobyland,successor,“Aero 2”)|
|_Wrong_|I realized that a book was written byJ. V. Jones and has the OCLC number 51969173|(J._V._Jones, author, A_Cavern_of_Black_Ice, ‘oclc’, “39456030”)|



Table 5: Examples of claims in FACTKG and retrieved graph path. 

|**Group number**|**Head type**|**Tail type**|**Relation set**|
|---|---|---|---|
|1|person|person|[child, children], [successor], [parent], [predecessor, precededBy],<br>[spouse], [vicePresident,vicepresident], [primeminister, primeMinister]|
|2|person|team|[currentteam,currentclub,team], [debutTeam,formerTeam]|
||||[chairperson, chairman, leader, leaderName], [manager], [founder],|
|3|non-person|person|[director], [crewMembers], [producer], [discoverer], [creator], [editor],|
||||[writer], [coach], [starring], [dean]|
|4|non-person|non-person|[owningCompany, parentCompany,owner], [headquarter], [builder]|



Table 6: Group information of _Relation Substitution_ . 

first used an MLP to calculate the attention coefficients between a node _i_ and its neighbor _j_ ( _j ∈Ni_ ), 

where **W** _∈R[C][×][F]_ and **b** _∈R[C][×]_[1] are parameters, and _C_ is the number of prediction labels. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs_images/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs.pdf-0013-06.png)


where _Ni_ denotes the set of neighbors of node _i_ , **W** 0 _[t][−]_[1] _∈R[H][×]_[2] _[F]_ and **W** 1 _[t][−]_[1] _∈R_[1] _[×][H]_ are weight matrices, and _·∥·_ denotes the concatenation operation. 

Then, they normalized the coefficients using the softmax function, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs_images/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs.pdf-0013-09.png)


Finally, the normalized attention coefficients were used to compute a linear combination of the neighbor features and thus obtained the features for node _i_ at layer _t_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs_images/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs.pdf-0013-11.png)


The authors fed the final hidden states of the evidence nodes _{_ **h** _[T]_ 1 _[,]_ **[ h]** _[T]_ 2 _[, ...,]_ **[ h]** _[T] N[}]_[ into their evidence] aggregator to make the final inference. 

## **Evidence Aggregator** 

The authors employed an evidence aggregator to gather information from different evidence nodes and obtained the final hidden state **o** _∈R[F][×]_[1] . We used the mean aggregator in GEAR. 

The mean aggregator performed the _elementwise_ Mean operation among hidden states. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs_images/acl_2023_FactKG_Fact_Verification_via_Reasoning_on_Knowledge_Graphs.pdf-0013-16.png)


Once the final state **o** is obtained, the authors employed a one-layer MLP to get the final prediction _l_ . 

_l_ = softmax(ReLU( **Wo** + **b** )) _,_ (6) 

|**Type**<br>**Relation**|**Type**<br>**Relation**|**Template**<br>**Example sentences**|
|---|---|---|
|**Head-Relation**<br>successor, spouse, children, parentCompany,<br>capital, garrison, nickname, mascot,<br>youthclubs, predecessor, child, precededBy,<br>religion,awards,award<br>college, university||{Head} had a(an) {Relation}.<br>Obama had a spouse.|
|||{Head} did not have a(an) {Relation}.<br>Apple did not have a parent company.|
||college, university|{Head}attended{Relation}.<br>Obama attended university.|
|||{Head}did not attend{Relation}.<br>Obama did not attend college.|
|**Tail-Relation**|president, primeMinister, vicepresident,<br>primeminister,vicePresident|{Tail}was a{Relation}.<br>Obama was apresident.|
|||{Tail}was not a{Relation}.<br>Obama was not a vicepresident.|



Table 7: Templates for Existence claims. 

|**Presupposition type**|**Template**<br>**Claim Example**|
|---|---|
|**Factive**|I forgot that{claim}.<br>I forgot that Obama waspresident.|
||I realized that{claim}.<br>I realized that Obama waspresident.|
||I wasn’t aware that{claim}.<br>I wasn’t aware that Obama waspresident.|
||I didn’t know that{claim}.<br>I didn’t know that Obama waspresident.|
||I remembered that{claim}.<br>I remembered that Obama waspresident.|
||I explained that{claim}.<br>I explained that Obama waspresident.|
||I emphasized that{claim}.<br>I emphasized that Obama waspresident.|
||I understand that{claim}.<br>I understand that Obama waspresident.|
|**Non Factive**|I imagined that{claim}.<br>I imagined that Obama waspresident.|
||I wish that{claim}.<br>I wish that Obama waspresident.|
||If only {claim}.<br>If onlyObama waspresident.|



Table 8: Templates for factive, non factive presupposition. 

|**Type**||**Relations**|**Template**<br>**Example claim**|
|---|---|---|---|
|**One-hop**||leader, leaderName, mayor,<br>senators, president, manager,<br>generalManager, coach,<br>chairman,dean|When was {tail} a {relaion} of {head}?<br>When was Elizabeth II a leader of Alderney?|
|||team, draftTeam, clubs,<br>managerClub,managerclubs|When did {head} play for {tail}?<br>When did Aaron Boogaard play for Wichita<br>Thunder?|
|||operator|When did{tail}operate{head}?<br>When did Aktieselskab operate Aarhus Airport?|
|||occupation,formerName|When was{head}a{tail}?<br>When was HBO a The Green Channel?|
|||almaMater|When did {head} graduate from the {tail}?<br>When did Ab Klink graduate from the Erasmus<br>UniversityRotterdam?|
|||fossil|When was{tail}fossil found in{head}?<br>When was Smilodon fossil found in California?|
|||director|When was {head} directed by {tail}?<br>When was Death on a Factory Farm directed by<br>Sarah Teale?|
|||producer|When was {head} produced by {tail}?<br>When was Turn Me On (album) produced by<br>Wharton Tiers?|
|||foundation, foundedBy,<br>founder|When was {head} founded by {tail}?<br>When was MotorSport Vision founded by<br>Jonathan Palmer?|
|||deathCause|When did{head}die from{tail}?<br>When did James CraigWatson die from Peritonitis?|
|||creators,creator|When was{head}created by {tail}?<br>When was April O’Neil created byPeter Laird?|
|||starring|When was{head}starring {tail}?<br>When was Bananaman starringGraeme Garden?|
|||shipBuilder,builder|When was{head}built by {tail}?<br>When was A-Rosa Luna built byGermany?|
|||designer|When was {head} designed by {tail}?<br>When was Atatürk Monument (˙Izmir) designed by<br>Pietro Canonica?|
|||shipCountry|When did {head} come from {tail}?<br>When did ARA Veinticinco de Mayo (V-2) come<br>from Argentina?|
|||spouse|When was {head} married to {tail}?<br>When was Abraham A. Ribicoff married to Ruth<br>Ribicoff?|
|||champions|When was{tail}champion at the{head}?<br>When was Juventus F.C. champion at the Serie A?|
|||recordedIn|When was {head} recorded in {tail}?<br>When was Bootleg Series Volume 1: The Quine Tapes<br>recorded in San Francisco?|
|**Existence**|**Head-Relation**|successor, spouse, children,<br>parentCompany, capital,<br>garrison, nickname, mascot,<br>youthclubs, predecessor,<br>child, precededBy, religion,<br>awards,award|What is the name of {head}’s {relation}?<br>What is the name of Obama’s child?|
|||college,university|When did{head}attend{relation}?<br>When did Obama attend university?|
||**Tail-Relation**|president, primeMinister,<br>vicepresident, primeminister,<br>vicePresident|When was{tail} {relation}?<br>When was Obama President?|
||||Where was{tail} {relation}?<br>Where was Biden Vice President?|
||||What countrywas{tail} {relation}?<br>What countrywas Obama President?|



Table 9: Templates for structural presupposition. 

||**Graph**|**Claim Example**|**Label**|
|---|---|---|---|
|_a_|_m_<br>_p_<br>_r_2<br>_r_4|AIDAstella was built by Meyer Werft in Papenburg.|SUPPORTED|
|_a_|_m_<br>_n_<br>_r_2<br>_r_4|AIDAstella was built by Meyer Werft inNew York.|REFUTED|
|_a_|_m_<br>_n_<br>_r_2<br>_r_4|AIDAstella wasnotbuilt by Meyer Werft inNew York.|REFUTED|
|_a_|_m_<br>_n_<br>_r_2<br>_r_4|AIDAstella was built by Meyer Werft,notinNew York.|SUPPORTED|
|_a_|_m_<br>_n_<br>_r_2<br>_r_4|AIDAstella wasnotbuilt by Meyer Werft,notinNew York.|REFUTED|



Table 10: _r_ 2: shipBuilder, _r_ 4: location, _m_ : Meyer Werft, _a_ : AIDAstella, _n_ : New York, _p_ : Papenburg. 

||**Graph**|**Graph**|**Claim Example**|**Label**|
|---|---|---|---|---|
|_a_|_m_<br>_r_2|_p_<br>_r_4|AIDAstella was built by Meyer Werft in Papenburg.|SUPPORTED|
|_a_|_s_<br>_r_2|_p_<br>_r_4|AIDAstella was built bySamsungin Papenburg.|REFUTED|
|_a_|_s_<br>_r_2|_p_<br>_r_4|AIDAstella wasnotbuilt bySamsungin Papenburg.|REFUTED|
|_a_|_s_<br>_r_2|_p_<br>_r_4|AIDAstella was built bySamsung,notin Papenburg.|REFUTED|
|_a_|_s_<br>_r_2|_p_<br>_r_4|AIDAstella wasnotbuilt bySamsung,notin Papenburg.|SUPPORTED|



Table 11: _r_ 2: shipBuilder, _r_ 4: location, _m_ : Meyer Werft, _a_ : AIDAstella, _p_ : Papenburg, _s_ : Samsung. 

||**Graph**|**Graph**|**Claim Example**|**Label**|
|---|---|---|---|---|
|_s_|_x_<br>_r_2|_p_<br>_r_4|AIDAstella was built by a company in Papenburg.|SUPPORTED|
|_s_|_x_<br>_r_2|_n_<br>_r_4|AIDAstella was built by a company inNew York.|REFUTED|
|_s_|_x_<br>_r_2|_n_<br>_r_4|AIDAstella wasnotbuilt by a company inNew York.|PATHCHECK|
|_s_|_x_<br>_r_2|_n_<br>_r_4|AIDAstella was built by a company,notinNew York.|PATHCHECK|
|_s_|_x_<br>_r_2|_n_<br>_r_4|AIDAstella wasnotbuilt by a company,notinNew York.|PATHCHECK|



Table 12: _r_ 2: shipBuilder, _r_ 4: location, _s_ : AIDAstella, _n_ : New York. 

