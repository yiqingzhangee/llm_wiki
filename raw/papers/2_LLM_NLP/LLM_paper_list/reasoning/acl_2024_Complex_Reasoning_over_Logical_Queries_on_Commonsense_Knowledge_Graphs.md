**001** Reasoning about events, their relationships, and **002** inferring implicit context are crucial abilities **003** of event commonsense reasoning, which state- **004** of-the-art language models still struggle to per- **005** form. However, data scarcity makes it chal- **006** lenging to learn systems that can generate com- **007** monsense inferences for contexts and questions **008** involving interactions between complex events. **009** To address this demand, we present COM[2] **010** ( **COM** plex **COM** monsense), a new dataset **011** created by sampling multi-hop logical queries **012** (e.g., _the joint effect or cause of both event_ **013** _A and B, or the effect of the effect of event_ **014** _C_ ) from an existing commonsense knowledge **015** graph (CSKG), and verbalizing them using **016** handcrafted rules and Large Language Models **017** into multiple-choice and text generation ques- **018** tions. **019** Our experiments show that Language models **020** trained on COM[2] exhibit significant improve- **021** ments in complex reasoning ability, resulting **022** in enhanced zero-shot performance in both in- **023** domain and out-of-domain tasks for question **024** answering and generative commonsense rea- **025** soning, without expensive human annotations. 

**026** 

## **Complex Reasoning over Logical Queries on Commonsense Knowledge Graphs** 

## **Anonymous ACL submission** 

## **Abstract** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Complex_Reasoning_over_Logical_Queries_on_Commonsense_Knowledge_Graphs_images/acl_2024_Complex_Reasoning_over_Logical_Queries_on_Commonsense_Knowledge_Graphs.pdf-0001-05.png)


**----- Start of picture text -----**<br>
𝑞 𝑉? = 𝑉?:  xIntent X goes sky diving , 𝑉?<br>∧ xWant  (X gets tired of it, 𝑉?)<br>(the  intention of PersonX)<br>PersonX goes skydiving xIntent<br>find new things to do<br>PersonX gets tired of it xWant<br>(then PersonX wants to)<br>Verbalization<br>LLM-addedcontext PersonX is living a boring life.<br>Rule-based After getting tired of it, PersonX goes skydiving<br>discourse<br>Question: What’s both the intention of PersonX going skydiving<br>and what X wants to do after PersonX getting tired of it?<br>Answer: find new things to do<br>**----- End of picture text -----**<br>


Figure 1: An example of conjunctive logical queries and the verbalization to complex commonsense inferences. 

## **1 Introduction** 

**027** Despite achieving remarkable performance in many **028** commonsense reasoning tasks, LLMs still face **029** challenges when it comes to more complex sce- **030** narios, such as reasoning about multiple events and **031** their relationships, as well as inferring implicit con- **032** text to facilitate subsequent reasoning. This is due **033** to the inherent difficulty of reasoning over multiple **034** pieces of information and a lack of adequate-scale **035** supervised training datasets for learning (Zhao **036** et al., 2023). Unfortunately, complex and multi- **037** hop commonsense reasoning benchmarks (Gabriel **038** et al., 2021) are both technically challenging and **039** financially expensive to curate. Consequently, pre- **040** vious efforts either constructed datasets (a) with **041** simpler reasoning structures, such as single-hop 

chains (Mostafazadeh et al., 2020), (b) using distant **042** supervision based on one-hop inference (Gabriel **043** et al., 2021), or (c) with human-annotations, but at **044** relatively small scale (Ravi et al., 2023). **045** To alleviate this training data bottleneck, recent **046** works have explored extracting and formulating **047** questions from existing CommonSense Knowledge **048** Graphs (CSKGs; Hwang et al., 2021), which store **049** commonsense triples. However, using CSKGs to **050** produce high-quality reasoning datasets poses sev- **051** eral challenges. First, while the shared entities in **052** commonsense triples encode a complex, intercon- **053** nected graph structure, the sparsity of this structure **054** limits the number of potential questions that encode **055** more than one reasoning hop (Sap et al., 2019b; **056** Kim et al., 2023). Second, triples in CSKGs are **057** represented in a context-free manner, such as the **058** event “PersonX gets tired of it” in Fig. 1, yielding **059** ambiguous (and sometimes incorrect) human an- **060** notations in the CSKG, e.g., ATOMIC (Sap et al., **061** 2019a) has an error rate of over 10%. These errors **062** propagate quadratically when triples are naively **063** combined to construct reasoning questions. Finally, **064** also because triples in CSKGs are represented in **065** a context-free manner, additional context must be **066** 

1 

**067** added to make questions fluent, a problem exac- **068** erbated in multi-hop settings where the entities of **069** multiple reasoning hops must be coherently verbal- **070** ized together. **071** In this paper, we construct COM[2] ( **COM** plex **072 COM** monsense), a novel commonsense reason- **073** ing dataset using multi-hop queries in common- **074** sense knowledge graphs to construct question an- **075** swer pairs requiring complex narrative reasoning **076** to solve. To build a dataset that integrates more **077** complex reasoning signals, we resort to _conjunc-_ **078** _tive logical queries_ (Hamilton et al., 2018), a subset **079** of First-Order Logical queries that use existential **080** quantifiers and conjunction. The multi-hop projec- **081** tion operation involves inferring hidden contexts, **082** while the intersection operation enables reasoning **083** among multiple events, encompassing common **084** cause or effect, and abduction. For example, in **085** Fig. 1, an intersection of two triples can be ver- **086** balized to a short narrative, and the process of in- **087** ferring the sampled common tail can be seen as **088** an _abduction_ of the hidden cause between the two **089** heads. **090** To address the challenges above, we propose to **091** first _densify_ the CSKG to merge nodes with high **092** semantic similarity, increasing the connectivity of **093** the graph. Then, we use an off-the-shelf plausibil- **094** ity scorer to filter out low quality triples, avoiding **095** error propagation as we construct more compli- **096** cated queries. Finally, we verbalize the queries to **097** a natural language context with handcrafted rules **098** and Large Language Models to derive coherent and **099** informative narrative contexts for our questions. **100** Our final COM[2] dataset comprises 790K question- **101** answer pairs (both with multiple-choice and gen- **102** erative answer settings), including 1.3K examples **103** that we manually verify for evaluation. **104** Our results demonstrate the challenges faced by **105** even powerful LLMs and supervised question an- **106** swering models on the COM[2] dataset, underscor- **107** ing the difficulty of performing complex multi-hop **108** reasoning. Moreover, fine-tuning question answer- **109** ing models and generative commonsense inference **110** models on COM[2] leads to substantial improvements **111** across four commonsense reasoning datasets, show- **112** ing the effecacy of our framework for boosting **113** commonsense reasoning ability. **114** To conclude, our contributions are three-fold. **115** First, we present a pipeline for effectively sam- **116** pling and verbalizing complex logical queries from **117** CSKGs, to form a complex commonsense rea- **118** OM[[2]] , with minimum human 

To conclude, our contributions are three-fold. First, we present a pipeline for effectively sampling and verbalizing complex logical queries from CSKGs, to form a complex commonsense reasoning benchmark, COM[[2]] , with minimum human 

effort. Second, we benchmark the complex rea- **119** soning ability of various state-of-the-art language **120** models and question answering models on COM[2] . **121** Third, we conducted comprehensive experiments **122** to validate the beneficial impact of fine-tuning **123** on COM[2] for subsequent commonsense reasoning **124** tasks across eight datasets. **125 2 Related Work and Background 126 Complex Logical Queries** Recent years have **127** witnessed significant progress in reasoning on one- **128** hop relational data (Bordes et al., 2013; Sun et al., **129** 2019; Lin et al., 2023). In addition to one-hop **130** reasoning, efforts are also put into handling com- **131** plex logical structures, involving _reasoning on_ **132** _unobserved edges and multiple entities and vari-_ **133** _ables_ (Ren et al., 2020; Wang et al., 2021, 2023b; **134** Bai et al., 2023a). In this paper, we focus on con- **135** junctive logical queries (Hamilton et al., 2018), a **136** subset of first-order logic that is defined with logi- **137** cal operators such as existential quantifiers _∃_ and **138** conjunctions _∧_ . There is a set of anchor entities, _V_ , **139** a unique target entity _V_ ? representing the answer **140** to the query, and a set of existential quantified vari- **141** ables _V_ 1 _, · · · , Vm_ . Conjunctive queries are defined **142** as the conjunction of literals _e_ 1 _, · · · , en_ : **143** _q_ = _V_ ? _, ∃V_ 1 _, · · · , Vm_ : _e_ 1 _∧ e_ 2 _∧· · · ∧ en_ (1) **144** where _ei_ is an edge involving variable nodes and **145** anchor nodes, satisfying _ei_ = _r_ ( _vj, Vk_ ) _, Vk ∈_ **146** _{V_ ? _, V_ 1 _, · · · , Vm}_ , _vj ∈V, r ∈R_ , or _ei_ = **147** _r_ ( _Vj, Vk_ ) _, Vj, Vk ∈{V_ ? _, V_ 1 _, · · · , Vm}, j_ = _k, r ∈_ **148** _R_ . _R_ is the set of relations defined in the KB. **149** Previous efforts focus on constructing box em- **150** beddings (Ren et al., 2020), embeddings based on **151** beta distribution (Ren and Leskovec, 2020), parti- **152** cle simulations (Bai et al., 2022), and computation **153** tree optimization (Bai et al., 2023b). Instead of **154** relying on embeddings or limited query types for **155** matching synthetic logical queries, we leverage the **156** concept of logical queries to effectively acquire **157** complex reasoning data from CSKGs with mini- **158** mum human efforts. **159 Complex Commonsense Reasoning** Recent ad- **160** vances in commonsense reasoning growed starting **161** from the construction human-annotated of Com- **162** monSense Knowledge Graphs (CSKG), including **163** ConceptNet (Speer et al., 2017), ATOMIC (Sap **164** et al., 2019a), ATOMIC[20] 20[(][Hwang][et][al.][,][2021][),] **165** and GLUCOSE (Mostafazadeh et al., 2020). A **166** 

2 

**197** 

**202** 

**208 209 210 211 212 213 214 215** 

**167** common approach to create challenges for com- **168** monsense reasoning involves constructing tasks **169** in the form of question-answering (Talmor et al., **170** 2019; Sap et al., 2019b), knowledge base com- **171** pletion (Malaviya et al., 2020), grounding (Gao **172** et al., 2022), and daily dialogue (Kim et al., 2023), **173** based on CSKGs. However, most of those previous **174** benchmarks are based on one-hop triples. **175** In contrast, real-world situations in dialogues **176** and narratives usually involve more complicated **177** reasoning across multiple events, sentences, and **178** paragraphs (Schank and Abelson, 1975). Previous **179** works are devoted to learn representations of narra- **180** tive chains (Chambers and Jurafsky, 2008; Pichotta **181** and Mooney, 2014) and draw inferences (Fang **182** et al., 2022; Yuan et al., 2023). To address more **183** complicated paragraph-level or multi-event reason- **184** ing, ParaCOMET (Gabriel et al., 2021) is pro- **185** posed to pre-train on distantly supervised one- **186** hop paragraph-level commonsense inferences, and **187** COMET-M (Ravi et al., 2023) is proposed to be **188** fine-tuned on a crowdsourced corpus focusing on **189** reasoning on multiple events. Instead of crowd- **190** sourcing or using language models to distill com- **191** plex inferences, we provide narrative-level infer- **192** ence by verbalizing complex logical queries over **193** CSKGs, to effectively acquire grounded inferences **194** at scale. Moreover, besides involving multiple **195** pieces of information in the context, the question **196** to the context also involves multiple relations. 

## **3 Methodology** 

**198** In this section, we introduce the construction de- **199** tails of COM[2] , including pre-processing, sampling **200** of complex queries, verbalization, and the details **201** of human annotations. 

## **3.1 Pre-processing** 

**203** We use ATOMIC[20] 20[(][Hwang et al.][,][ 2021][), a compre-] **204** hensive Commonsense Knowledge Graph covering **205** social, physical, and event-level everyday knowl- **206** edge, as the base CSKG. Before sampling, we deal **207** with the sparsity and quality issue first. 

**Sparsity** CSKGs are usually highly sparse compared to factual KGs due to the nature of human annotation and flexibility of commonsense (Malaviya et al., 2020), making it hard to sample diverse complex queries. To alleviate the sparsity issue, we first conduct normalization to the tails. In ATOMIC, heads are pre-defined complete sentences (for example, “PersonX says sorry”) while tails are usu- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Complex_Reasoning_over_Logical_Queries_on_Commonsense_Knowledge_Graphs_images/acl_2024_Complex_Reasoning_over_Logical_Queries_on_Commonsense_Knowledge_Graphs.pdf-0003-09.png)


**----- Start of picture text -----**<br>
Training Query Types<br>2p 3p<br>1p 2i 3i<br>Unseen Query Types<br>Anchor Entity<br>Free Variable<br>Answer Entity<br>ip pi<br>**----- End of picture text -----**<br>


Figure 2: Visualization of query structures. The anchor entities and relations are specified to instantiate the query. ‘p’ and ‘i’ represent _projection_ and _intersection_ , and the number ahead of p and i indicates the number of anchor entities and free variables. 

ally short phrases without a subject (for example, **216** “to say sorry”). This discrepancy produces many **217** duplicated nodes and make the graph sparser. We **218** develop simple rules to add “PersonX” or “Per- **219** sonY” in front of the tails to make them a complete **220** sentence, if the tail does not have a subject. This **221** process merged 3.7% nodes in ATOMIC together. **222** Second, as the nodes in ATOMIC are free-text, **223** some nodes with the same semantic meaning are **224** represented as separated nodes due to some minor **225** annotation distinctions and errors, e.g., “PersonX **226** buys a ticket” versus “PersonX buys a ticket.”. We **227** use a state-of-the-art sentence embedding model[1] , **228** to merge nodes with cosine similarity score over **229** 0.95. In this process, 20.0% nodes are merged to- **230** gether and the average degree increases by 25.3%. **231** 

**Quality** The error rate of ATOMIC itself is over **232** 10% (Sap et al., 2019a). This error rate can be **233** problematic when we consider the intersection and **234** projection of more than two triples as errors propa- **235** gate quadratically. We use an off-the-shelf plausi- **236** bility scorer Vera (Liu et al., 2023), a 5B T5-based **237** plausibility scorer fine-tuned on 2 CSKGs and 19 **238** QA datasets, to score every triple in terms of com- **239** monsense plausibility (between 0 to 1). We fil- **240** ter out triples with a plausibility score lower than **241** 0.5, the threshold provided as a tipping point in **242** Vera between plausible and implausible statements. **243** Around 10% of the triples are filtered out. **244 3.2 Query Sampling 245** The query structures that we study are visualized in **246** Fig. 2. Following Ren et al. (2020), we use projec- **247** tions (1p, 2p) and intersections (2i, 3i) as training **248** queries, and leave more complex queries ip and pi **249** as the zero-shot evaluation queries. To examine **250** scenarios involving negation and differentiate them **251** 

1https://huggingface.co/sentence-transformers/all-mpnetbase-v2 

3 

**252** from regular 2i queries, we use the term “2i-neg” **253** to represent 2i queries where one of the relations is **254** “HinderedBy”. **255** Given a query structure, we use pre-order traver- **256** sal to sample free variables and anchor entities **257** starting from an answer entity. We sample prede- **258** cessors uniformly based on (relation, entity) pairs. **259** During sampling, to avoid over-sampling on nodes **260** with extremely high degree, we empirically set a **261** cut-off degree _T_ = 10 to only sample from top _T_ **262** neighbors of a node scored by Vera. In the end, **263** we conduct a post-order traversal starting from the **264** anchor entities to find all the answers of the query, **265** in addition to the starting answer entity. 

**266 267 268 269 270 271 272 273 274 275 276 277** 

**278** 

**279 280 281 282 283 284** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Complex_Reasoning_over_Logical_Queries_on_Commonsense_Knowledge_Graphs_images/acl_2024_Complex_Reasoning_over_Logical_Queries_on_Commonsense_Knowledge_Graphs.pdf-0004-04.png)


**----- Start of picture text -----**<br>
|||||
|---|---|---|---|
|2i: Common Attribution|Context:|
|V1: X pulls out Y’s phone|xAttr|X and Y were at a park. Suddenly, Y's phone starts ringing and X reaches over and phone from their pocket. Just as X does that, Y playfully kicks their legs in the air, and pulls out Y's X swings|
|childish|Y's legs in response.|
|Question:|
|V2: X swings Y’s legs|xAttr|
|What state is both what X is seen as given V1 and what X is seen as given V2?|
|2i-negative: Negated Common Cause|
|V1: X feels worse|xEffect|Context:|X has been feeling unwell lately. As a result, X doesn't smoke cigarettes anymore.|
|wind up in|Question:|
|hospital|What event or state is both what X will do after V1 and also hindered V2?|
|V2: X smokes cigarettes|
|HinderedBy|
|2p: 2nd order Effect|
|xWant|xWant|Question:|
|What event or state is what X wants to do after what|
|V1: X starts a new life|V?: X makes new friends|socialize|X wants to do after V1?|
|pi|xWant|Context:|X was looking for a new opportunity and decided to join Y's ranks. After joining,|
|V1: X works hard for months|V?: PersonX get a promotion|X works hard for months to prove their dedication and commitment.|
|oWant|oWant|Question:|
|V2: X joins Y's ranks|congratulate X|What event or state is both what Y wants to do after {what X wants to do after X|
|works hard for months}, and also what Y wants to do after X joins Y's ranks?|

**----- End of picture text -----**<br>


Figure 3: Examples of different query types, the verbalization, and corresponding questions. 

from regular 2i queries, we use the term “2i-neg” to represent 2i queries where one of the relations is “HinderedBy”. 

**Option Sampling** We sample 4 additional candidate distractors for each query, where 2 of them are randomly sampled across the whole CSKG, and 2 of them are sampled from the neighbors of the anchor entities that are not the answers to the whole query, represented as confusing negative examples. In case of fine-tuning a question answering language model, the negative examples are used as synthetic question answering pairs for training. In the evaluation set, these candidate negative examples, together with the sampled answer, are manually annotated to form a gold evaluation set. 

## **3.3 Verbalization** 

CSKGs are constructed in a context-free manner. To make the logical queries on such context-free triples more human-interpretable, we introduce an additional step of verbalizing the anchor entities to a narrative, to effectively acquire fluent and plausible narrative-inference pairs. 

**Anchor Entity Verbalization** We consider a **285** rule-based verbalizer and a ChatGPT-driven ver- **286** balizer. In the rule-based verbalizer, we add a dis- **287** course marker between the two or three anchor **288** entities depending on the semantics of the query **289** relations. For example, a simple situation would **290** be adding an “and” or “then” between two anchor **291** entities in a 2i query. To make the query even more **292** human-understandable, we consider using Chat- **293** GPT to synthesize necessary contexts to make the **294** query an actual narrative. We include the detailed **295** rules for adding discourse connectives (denoted as **296** _rule-based verbalization_ ), and prompts for using **297** ChatGPT to verbalize complex queries (denoted as **298** LLM-based) in Appx. §A.3. **299** 

**Relation Verbalization** The multiple relations **300** in complex queries can be deterministically con- **301** verted to a question using the natural language de- **302** scriptions of the relations, which are presended in **303** Appx. §A.3. **304** 

## **3.4 Human Annotation** 

**305** 

We formalize the problem of complex common- **306** sense reasoning as a multi-choice question answer- **307** ing task, to support reliable automatic evaluation. **308** There are only one true answer and three distrac- **309** tors, together with an option indicating “None of **310** the answers are correct”. We crowdsourced the **311** answers using Amazon Mechanical Turk (AMT). **312** The workers are given the verbalized query as the **313** context, the corresponding question by converting **314** the relations in the query using a prompt template, **315** and the sampled (negative) answers. If no sampled **316** answers are correct, then the worker is asked to se- **317** 

4 

**318** lect an additional “None of the answers are correct” **319** option. If the verbalization itself does not make **320** sense, the worker can also click another option **321** “The context doesn’t make sense or is meaning- **322** less.” and we will discard the data. Each question **323** is annotated by three workers, and the overall per- **324** option Inter Annotator Agreement (IAA) is 78%, **325** and the fleiss kappa is 0.445, indicating moderate **326** agreement. The workers are paid on average 16 US **327** Dollar per hour. **328** We refer readers to Appx. §A for technical de- Appx. §A for technical de- for technical de- **329** 3.. 

**330** 

**331** We conduct experiments on the evaluation set **332** of COM[2] , a Multi-Choice Question Answering **333** (MCQA) task. Specifically, we examine the per- **334** formance of state-of-the-art off-the-shelf language **335** models on COM[2] , and also study the effect of train- **336** ing a question answering model on the distantly **337** supervised training set of COM[2] . 

**338** 

**339 340 341 342 343 344 345 346 347** 

|**Method**|**2i**<br>**2i-neg**<br>**3i**<br>**2p**<br>**ip**<br>**pi**|**All**|
|---|---|---|
|**API-based LLMs**|||
|gpt-3.5-turbo-0613<br>- 1-shot<br>- 1-shot w/ CoT<br>- 8-shot (2i, 2p)<br>- 8-shot (2i, 2p) w/ CoT<br>gpt-4-1106-preview<br>- 1-shot<br>- 1-shot w/ CoT<br>- 8-shot (2i, 2p)<br>- 8-shot (2i, 2p) w/ CoT|33.56<br>43.12<br>42.01<br>38.66<br>38.05<br>28.40<br>43.31<br>35.31<br>58.45<br>57.73<br>51.33<br>62.96<br>45.80<br>36.43<br>54.34<br>57.73<br>50.44<br>66.67<br>48.52<br>41.26<br>57.08<br>67.53<br>53.10<br>74.07<br>52.61<br>46.10<br>60.27<br>59.79<br>52.21<br>65.43<br>44.67<br>46.47<br>52.05<br>32.47<br>40.71<br>53.08<br>47.85<br>42.01<br>50.68<br>38.66<br>44.25<br>50.62<br>48.97<br>46.46<br>52.96<br>49.48<br>52.21<br>58.02<br>54.87<br>46.47<br>58.90<br>45.88<br>52.21<br>66.67<br>57.82<br>49.07<br>62.56<br>61.34<br>52.21<br>66.67|37.74<br>48.22<br>48.75<br>53.22<br>54.37<br>44.64<br>45.63<br>50.04<br>53.00<br>57.40|
|**Open-source (QA) Language Models**|||
|HyKAS (Ma et al.,2021, zero-shot)<br>CAR (Wang et al.,2023a, zero-shot)<br>UnifedQA-v2 (Khashabi et al.,2022)<br>Flan-T5 (11B) (Chung et al.,2022)<br>Llama2 (7B) (Touvron et al.,2023)<br>Vera (Liu et al.,2023)|34.92<br>39.41<br>27.85<br>41.75<br>37.17<br>33.33<br>37.41<br>30.48<br>37.44<br>57.73<br>32.74<br>53.09<br>56.23<br>39.41<br>62.56<br>58.76<br>51.33<br>62.96<br>58.28<br>47.21<br>65.30<br>**76.29**<br>56.64<br>79.01<br>35.15<br>21.93<br>39.27<br>35.57<br>28.32<br>51.85<br>47.62<br>27.51<br>40.18<br>66.49<br>52.21<br>58.02|35.76<br>39.56<br>54.21<br>60.97<br>33.64<br>46.09|
|**Fine-tuned on COM**2<br>|||
|DeBERTa-v3-Large (+COM2)<br>CAR-DeBERTa-v3-Large (+COM2)|60.09<br>**58.36**<br>69.41<br>61.86<br>**59.29**<br>81.48<br>**61.22**<br>56.13<br>**69.86**<br>68.56<br>56.64<br>**85.19**|62.79<br>**63.78**|



Table 1: Model performance (%) on the multiple-choice question answering evaluation set of COM[2] . 

We refer readers to Appx. §A for technical de- Appx. §A for technical de- for technical details and dataset statistics regarding §3.. 

## **4 Experiments** 

## **4.1 Setup** 

We study popular API-based LLMs and some Open-source Language Models as baselines. Following the standard practice of prompting LLMs for QA (Robinson et al., 2022), we use a promptbased method that takes “[Context] [Question] [Options]” as the input and ask the model to only output the associated symbol (e.g., ‘A’) in the QA pair as the prediction. For open-source language models like Flan-T5 and Llama2, we use same prompt, and 

compute the logits received by each of the options **348** in the first prediction token. **349** We also study the effect of fine-tuning a question- **350** answering model on the synthetic training queries **351** discussed in §3.2. We follow the most effective **352** pipeline by HyKAS (Ma et al., 2021), which fine- **353** tunes language models on QA pairs synthesized **354** from one-hop knowledge in CSKGs, and extend it **355** to complex queries. For one-hop (1p) triples, the **356** head and relation are transformed into a question **357** with pre-defined prompts. For complex queries, **358** the verbalized queries (as illustrated in §3.3) are re- **359** garded as the context, and questions are also trans- **360** formed with a different prompt template depending **361** on the relations. The tails to the one-hop triple or **362** the sampled answer to the query are regarded as the **363** correct answer, and the negative examples are ran- **364** domly sampled across the whole CSKG following **365** a keyword overlapping filtering (Ma et al., 2021; **366** Wang et al., 2023a). We use DeBERTa-v3-large as **367** the backbone encoder[2] . **368 4.2 Results and Analysis 369** We present the results in Tab. 1. In terms of per- **370** formance on commercial LLMs, GPT-4 generally **371** outperforms ChatGPT with a notable margin. The **372** incorporation of Chain-of-Thought (CoT) proves **373** crucial in enhancing LLM reasoning capabilities, **374** as it fosters a step-by-step thinking approach that **375** first focuses on inducing the causes or effects of **376** 

2We refer readers to Appx. §B for detailed implementations and prompt templates. 

5 

**377** individual events in intersection-based queries, or **378** inducing the hidden variables in projection-based **379** queries. The eight-shot CoT, which encompasses **380** both 2i and 2p queries as exemplars, yields the **381** highest performance naturally due to the coverage **382** of all base query types. **383** When it comes to fine-tuning on complex queries **384** using the HyKAS and CAR paradigm, we observe **385** that the synthetic training pairs, despite lacking **386** manual annotation, serve as valuable distant su- **387** pervision signals. They effectively enhance the **388** complex reasoning capability of a QA model, even **389** surpassing the performance of an 8-shot GPT-4 **390** model with CoT by 6%. CAR + COM[2] can also **391** outperform the 11B version of UnifiedQA-v2 and **392** Flan-T5, which are both fine-tuned on numerous **393** (commonsense) question answering datasets by 9% **394** and 3%, respectively. We also include the zero- **395** shot transferability experiments of this QA model **396** to some other commonsense QA datasets, which **397** will be presented in §5.1. 

**398** 

**399 400 401** 

**402 403** 

|**Model**<br>**CSKG**|**Model**<br>**CSKG**|**Model**<br>**CSKG**|**Out-of-domain**|**Out-of-domain**|**In-dom**|
|---|---|---|---|---|---|
||||a-NLI<br>CSQA<br>PIQA<br>SIQA<br>WG|Avg.|COM2|
|||||||
|Random<br>DeBERTa-v3-L (He et al.,2023)<br>Self-talk (Shwartz et al.,2020)<br>COMET-DynGen (Bosselut et al.,2021)<br>SMLM (Banerjee and Baral,2020)<br>MICO (Su et al.,2022)<br>STL-Adapter (Kim et al.,2022)|-<br>-<br>-<br>ATOMIC<br>*<br>ATOMIC<br>ATOMIC||50.0<br>20.0<br>50.0<br>33.3<br>50.0<br>59.9<br>25.4<br>44.8<br>47.8<br>50.3<br>-<br>32.4<br>70.2<br>46.2<br>54.7<br>-<br>-<br>-<br>50.1<br>-<br>65.3<br>38.8<br>-<br>48.5<br>-<br>-<br>44.2<br>-<br>56.0<br>-<br>71.3<br>66.5<br>71.1<br>64.4<br>60.3|40.7<br>45.6<br>-<br>-<br>-<br>-<br>66.7|20.0<br>14.7<br>-<br>-<br>-<br>-<br>-|
|**Large Language Models**||||||
|GPT-3.5 (text-davinci-003)<br>GPT4 (gpt-4-1106-preview)<br>ChatGPT (gpt-3.5-turbo)<br>+ zero-shot CoT|-<br>-<br>-<br>-||61.8<br>68.9<br>67.8<br>68.0<br>60.7<br>75.0<br>43.0<br>73.0<br>57.0<br>77.0<br>69.3<br>74.5<br>75.1<br>69.5<br>62.8<br>70.5<br>75.5<br>79.2<br>**70.7**<br>63.6|65.4<br>65.0<br>70.2<br>71.9|-<br>44.6<br>37.7<br>28.9|
|**Backbone: DeBERTa-v3-Large** _435M_||||||
|HyKAS (Ma et al.,2021)<br>HyKAS (Ma et al.,2021)<br>CAR (Wang et al.,2023a)<br>CAR (Wang et al.,2023a)<br>HyKAS + COM2(Ours)<br>CAR + COM2(Ours)|ATM-10X<br>ATOMIC<br>ATOMIC<br>ATM_C_<br>ATM, COM2<br>ATM_C_<br>, COM2||75.1<br>71.6<br>79.0<br>59.7<br>71.7<br>76.0<br>67.0<br>78.0<br>62.1<br>76.0<br>78.9<br>67.2<br>78.6<br>63.8<br>78.1<br>79.6<br>69.3<br>78.6<br>64.0<br>78.2<br>78.4<br>69.9<br>78.7<br>64.1<br>78.3<br>81.2<br>70.9<br>80.3<br>65.6<br>77.4|71.4<br>71.8<br>73.3<br>73.9<br>73.9<br>**75.1**|27.7<br>35.8<br>36.8<br>39.8<br>62.8<br>**63.8**|
|||||||
|Human Performance|-||91.4<br>88.9<br>94.9<br>86.9<br>94.1|91.2|-|



Table 2: Zero-shot evaluation results (%) on five out-of-domain commonsense question answering benchmarks, and the in-domain evaluation set of COM[2] . The best results are **bold-faced** , and the second-best ones are underlined. 

tion Answering (CSQA), and Generative Common- **404** sense Inference, including one-hop, multi-event, **405** and paragraph-level settings. **406** 

## **5.1 Commonsense Question Answering** 

**Commonsense Question Answering 407 Setup** The task of zero-shot commonsense QA **408** involves selecting the most plausible option for **409** commonsense questions without any supervision **410** signals from the training set of the benchmark **411** data. We directly leverage the model we trained in **412** §4, the DeBERTa-v3-large-based model fine-tuned **413** on synthetic question pairs in both ATOMIC and **414** COM[2] , and check the performance on five pop- **415** ular commonsense question answering datasets: **416** Abductive NLI (aNLI; Bhagavatula et al., 2020), **417** CommonsenseQA (CSQA; Talmor et al., 2019), **418** PhysicalIQA (PIQA; Bisk et al., 2020), SocialIQA **419** (SIQA; Sap et al., 2019b), and WinoGrande **420** (WG; Sakaguchi et al., 2021). We report the ac- **421** curacy of each dataset and the average accuracy **422** among five datasets. **423** 

## **5 Downstream Evaluation** 

In addition to benchmarking Complex Commonsense Reasoning, we also study the effect of leveraging COM[2] as training data and the generalization to other downstream commonsense reasoning tasks. In detail, we study zero-shot CommonSense Ques- 

**Results and Analysis** We report the model per- **424** formance in Tab. 2. The first batch of baselines **425** are zero-shot CSQA models that leverages CSKGs **426** as supervision signals, and we surpass them by a **427** large margin. We also report the zero-shot per- **428** formance of API-based LLMs including GPT-3.5, **429** ChatGPT, and GPT-4. The inclusion of COM[2] and **430** 

6 

**431** one-hop triples from ATOMIC as training data for **432** CAR and HyKAS yields significant improvements **433** in question answering ability. This improvement **434** is observed in both in-domain complex reasoning **435** tasks and out-of-domain CSQA tasks. Notably, the **436** combination of CAR and COM[2] achieves the high- **437** est performance among all models, surpassing even **438** ChatGPT and GPT-4, despite having a parameter **439** size at least two orders of magnitude smaller. 

**440** 

**441 442 443 444 445 446 447 448 449 450** 

**451 452** 

**453 454 455 456 457 458 459 460** 

|**Model**<br>**Training Data**|**Model**<br>**Training Data**|**Model**<br>**Training Data**|**Multi-Event**|**Paragraph-Level**|**Single-Event**|**COM**2|
|---|---|---|---|---|---|---|
||||**B-2**<br>**R-L BERT**|**R-L CIDE BERT**|**R-L CIDE BERT**|**R-L CIDE BERT**|
|**(Distantly) Supervised Learning**|||||||
||||||||
|COMET-M (BART-L)<br>MEI<br>25.1 33.6 64.9<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>COMET-M (GPT-2-L)<br>MEI<br>16.2 25.7 55.1<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>ParaCOMET (GPT-2-L)<br>ParaCOMET<br>-<br>-<br>-<br>18.8<br>27.8<br>60.2<br>-<br>-<br>-<br>-<br>-<br>-|||||||
||||||||
|**Zero-shot Learning**|||||**Supervised**||
||||||||
|COMET<br>COMET-distill<br>COM2-COMET<br>COM2-COMET|1p<br>ATM10x<br>1p, 2i<br>1p, 2p, 2i, 3i|1.20 2.73 38.9<br>1.20 3.55 12.7<br>8.87 15.2 46.4<br>5.41 10.4 44.8||3.5<br>6.4<br>25.7<br>11.8<br>16.8<br>29.5<br>13.8<br>22.1<br>53.7<br>9.2<br>16.6<br>44.1|50.0<br>66.1<br>75.1<br>1.6<br>4.8<br>24.3<br>50.7<br>68.0<br>77.1<br>50.4<br>66.9<br>77.1|10.0<br>20.7<br>44.3<br>8.3<br>11.9<br>36.1<br>13.6<br>26.1<br>39.8<br>**14.7**<br>**33.0**<br>**46.3**|
||||||||
|LLama2-7b<br>COMET-LLama2-7b<br>COM2-LLama2-7b<br>COM2-LLama2-7b|-<br>1p<br>1p, 2i<br>1p, 2p, 2i, 3i|1.81 4.14 45.7<br>7.62 14.4 44.2<br>**8.82 16.4 47.5**<br>8.22 15.4 47.0||2.2<br>2.2<br>48.6<br>9.1<br>12.3<br>51.0<br>14.6<br>**22.1**<br>55.3<br>**15.9**<br>21.3<br>**55.3**|5.4<br>2.9<br>51.5<br>27.5<br>26.4<br>64.2<br>**31.6**<br>**31.1**<br>**66.0**<br>31.3<br>29.8<br>65.5|3.9<br>6.7<br>44.9<br>10.9<br>22.3<br>44.9<br>**35.7 107.2 61.3**<br>35.6 105.0 60.1|



Table 3: Experimental results on downstream narrative commonsense reasoning, including in a multi-event (Ravi et al., 2023) setting, and a paragraph-level setting (Gabriel et al., 2021). In-domain settings include single-event generation and complex inference in COM[2] . We use BLEU-2 (B-2), ROUGE-L (R-L), CIDEr (CIDE), and BERTScore (BERT) as the evaluation metrics. 

## **5.2 Generative Commonsense Inference** 

**Setup** We study generative commonsense inference as an additional evaluation task. We include multi-event commonsense generation (COMETM; Ravi et al., 2023) and paragraph-level commonsense generation (ParaCOMET; Gabriel et al., 2021) as two out-of-domain evaluation tasks. We also include the vanilla COMET (Bosselut et al., 2019) as an additional in-domain evaluation, which actually focuses on 1p queries that requires generating the tail given head and relation as the input. Besides, we report the generation performance on generative COM[2] in the last columns. 

We study the effect of fine-tuning COMET (GPT2-large) on ATOMIC and different query types of COM[2] , following the settings in Bosselut et al. (2019). We also study fine-tuning on an LLM, Llama2-7b, by converting triples and queries to an instruction-tuning format, following the prompt template in §3.3 and Appx. §B.2. We leverage the framework of Chen et al. (2023)[3] to fine-tune 

> 3https://github.com/epfLLM 

Llama2-7b. We fine-tune on a mixture of different **461** query types as detailed in the “Training Data” col- **462** umn. To ensure diversity and prevent overfitting to **463** common tails, complex queries are selected using **464** an n-gram based diversity filter (Yang et al., 2020). **465 Results and Analysis** We report the performance **466** of various models on three datasets in Tab. 3. First, **467** compared to fine-tune on only one-hop triples, **468** COMET models based on both GPT2-large and **469** Llama2-7b will have an improved generative com- **470** monsense inference ability on both multi-event, **471** paragraph-level, and single-event commonsense in- **472** ference. The first two settings are out-of-domain **473** complex commonsense reasoning tasks that require **474** reasoning on longer context and more complicated **475** event-event relations. Second, among different **476** query types, 2i is the most useful query type that **477** help improve the reasoning ability. This may be **478** due to the fact that both the task from COMET-M **479** and ParaCOMET doesn’t require second-order in- **480** ference, while only requires the reasoning ability **481** brought by intersection-based queries. **482** 

## **6 Discussions and Analysis** 

**483 6.1 Ablation Study 484** We analyze the impact of various data filters, query **485** types, and verbalization methods in Tab. 12 in the **486** appendix on generative inference in COM[2] . **487 Filtering** We include two types of filters, a Vera- **488** based plausibility filter and a diversity filter. Evalu- **489** ating the performance of generative commonsense **490** 

7 

**491** inferences on COM[2] , we examine the impact of re- **492** moving both filters while employing GPT2-Large **493** as the backbone model. Removing the plausibility **494** filter results in a significant performance decline, **495** highlighting its critical role. On the other hand, the **496** diversity filter exhibits a minor positive influence **497** on enhancing performance. 

**524** 

**525 526 527 528 529 530** 

|Model|#Plau.<br>#1-hop<br>#False|
|---|---|
|||
|LLama2-7b<br>COMET-LLama2-7b<br>COM2-LLama2-7b (2i)<br>COM2-LLama2-7b (all)|26<br>2<br>28<br>29<br>8<br>23<br>47<br>2<br>11<br>45<br>3<br>12|



Table 4: Human evaluation results on the generative sub-task in COM[2] using Llama2-7b as the backbone. ‘1-hop’ indicates the answer is plausible in terms of only one-hop relations. 

**498 Type of Queries** We investigate the impact of **499** training our models on different types of logical **500** queries. The model trained only on 1p and 2i **501** queries does not generalize well to other query **502** types such as pi and ip, leading to a worse perfor- **503** mance than the model trained on all query types. **504** However, according to Tab. 1 and Tab. 3, models **505** trained on only 2i queries have a better generaliza- **506** tion ability to downstream commonsense reason- **507** ing tasks. This is probably because most existing **508** commonsense benchmarks focusing on interactions **509** regarding multiple events are actually structured **510** as an intersection-based manner, instead of projec- **511** tions and more complicated structures. 

**512 Verbalization** We investigate the effect of us- **513** ing a rule-based verbalizer or ChatGPT-enabled **514** verbalizer. The ChatGPT-verbalized queries help **515** produce better inference system a tad bit on both **516** ParaCOMET and COM[2] . In COM[2] , the presence **517** of ChatGPT-verbalization intuitively improves per- **518** formance since the training context aligns with the **519** evaluation set’s format. On the other hand, the con- **520** text in the ParaCOMET dataset is long and com- **521** prised of five sentences. Verbalization not only **522** adds more contexts to the training but also aligns **523** better with the ParaCOMET format. 

## **6.2 Difficulty of Different Query Types** 

Based on Tab. 1, there is a significantly higher accuracy of pi queries than others. This is mainly because of the sparsity issue, such that we cannot sample enough pi queries. Within the limited pi queries, the number of unique answers is also small and they are usually common nodes with high de- 

grees in the CSKG, making it easier for models **531** to make accurate predictions. The same situation **532** applies to 2i and 3i queries. Though 3i queries pos- **533** sess a more complex structure, they are constrained **534** by the sparse structures of ATOMIC, resulting in a **535** relatively narrower answer set. This narrower set **536** of possible answers makes predictions easier[4] . **537** 

## **6.3 Error Analysis** 

**Error Analysis 538** We present a human-annotated quality evaluation **539** of the Llama-7b-based model on the generation **540** sub-task of COM[2] . To ensure diverse coverage of **541** query types, we randomly sampled 60 queries, with **542** 10 from each of the 6 categories. Manual inspec- **543** tion revealed a common error where the generated **544** output was partially correct, either providing the **545** answer to one of the triples in an intersection query **546** or only the one-hop answer instead of the two-hop **547** answer in 2-projection queries. Tab. 4 includes the **548** number of such ‘1-hop’ partially correct answers. **549** Our results demonstrate that the zero-shot Llama **550** model already produces 26 out of 60 plausible infer- **551** ences. Fine-tuning the model on one-hop ATOMIC **552** further increases the number of plausible genera- **553** tions while more frequently generating inferences **554** that are one-hop correct. Moreover, fine-tuning **555** on the synthetic training set of COM[2] significantly **556** improves the model’s ability to generate complex **557** commonsense inferences and reduces the occur- **558** rence of partially correct answers. We leave the **559** some case studies in the Appx. §D. **560 7 Conclusion 561** In this paper, we leverage the concept of con- **562** junctive logical queries to create a complex com- **563** monsense reasoning dataset derived from CSKGs. **564** The dataset, COM[2] , comprises a human-annotated **565** evaluation set and a distantly supervised training **566** set without further annotations. Our experiments **567** demonstrate the difficulty of answering complex **568** logical queries on CSKGs, even for advanced lan- **569** guage models like GPT4. Additionally, we train **570** question answering models and generative com- **571** monsense reasoning models using the COM[2] train- **572** ing set. The results show significant improvements **573** across eight downstream commonsense reasoning **574** tasks, encompassing various aspects. This high- **575** lights the potential of leveraging CSKGs to acquire **576** complex reasoning signals inexpensively, without **577** relying on extra human efforts. **578** 

4We leave more quantative analysis in Appx. §C 

8 

tion was purely based on the workers’ performance **628** on the evaluation set, and we did not collect any **629** personal information about the participants from **630** MTurk. **631** 

**579** 

**611** 

## **Limitations** 

**580 Data Construction** The construction of COM[2] **581** reply on sampling complex logical queries from **582** existing CSKGs. However, there are sparsity is- **583** sue, quality issue, non-close-world-asssumption **584** issue that needs to be tackled. Even we have con- **585** ducted normalization and filtering, there may still **586** be missing links within ATOMIC and mislabeled **587** or ambiguous triples, which limits the quality of **588** our sampled queries. Future works can focus on **589** deriving complex queries from CSKGs with better **590** quality and more diverse semantics, which should **591** also have higher density, such as on ATOMIC-10x, **592** NovATOMIC (West et al., 2023). 

**593 Evaluation** In the context of generative common- **594** sense reasoning, we employ lexical-overlap based **595** automatic evaluation metrics to assess the perfor- **596** mance of the model in a scalable manner. However, **597** since each query typically has 1 to 3 gold refer- **598** ences on average, this type of evaluation may not **599** accurately capture the true plausibility of common- **600** sense reasoning, which is inherently open-ended. **601** To address this limitation, we have supplemented **602** the automatic evaluation with human annotation **603** on a subset of sampled queries. Nevertheless, this **604** approach is still not scalable by nature. **605** Future research can focus on the development of **606** automatic complex reasoning protocols based on **607** large language models. Such protocols can delve **608** into more fine-grained aspects such as typicality **609** and the degree of correctness, even if it’s only par- **610** tially correct. 

## **Ethical Considerations** 

**612** We sample the data from ATOMIC[20] 20[,][which][is] **613** an open-source commonsense knowledge graph **614** that may contain certain bias regarding gender, oc- **615** cupation, and nationality (Mehrabi et al., 2021). **616** The dataset does not contain specific individuals **617** or organizations. Instead, it employs generic place- **618** holders such as PersonX, PersonY, and randomly **619** replaced first names to represent subjects and ob- **620** jects. However, this paper primarily focuses on **621** complex reasoning based on knowledge, which is **622** in contrast to works that solely rely on one-hop **623** biased knowledge exploitation. **624** We collected 1.3k inferences through crowd- **625** sourcing. The participants were compensated with **626** an hourly wage of 16 USD, which is comparable **627** to the minimum wages in the US. The qualifica- 

## **References** 

|||
|---|---|
|**References**|**632**|
|Jiaxin Bai, Xin Liu, Weiqi Wang, Chen Luo, and|**633**|
|Yangqiu Song. 2023a. Complex query answering|**634**|
|on eventuality knowledge graph with implicit logical|**635**|
|constraints. _CoRR_, abs/2305.19068.|**636**|
|Jiaxin Bai, Zihao Wang, Hongming Zhang, and Yangqiu|**637**|
|Song. 2022. Query2particles: Knowledge graph rea-|**638**|
|soning with particle embeddings. In_Findings of the_|**639**|
|_Association for Computational Linguistics: NAACL_|**640**|
|_2022, Seattle, WA, United States, July 10-15, 2022_,|**641**|
|pages 2703–2714. Association for Computational|**642**|
|Linguistics.|**643**|
|Yushi Bai, Xin Lv, Juanzi Li, and Lei Hou. 2023b.|**644**|
|Answering complex logical queries on knowledge|**645**|
|graphs via query computation tree optimization.|**646**|
|In _International Conference on Machine Learning,_|**647**|
|_ICML 2023, 23-29 July 2023, Honolulu, Hawaii,_|**648**|
|_USA_, volume 202 of_Proceedings of Machine Learn-_|**649**|
|_ing Research_, pages 1472–1491. PMLR.|**650**|
|Pratyay Banerjee and Chitta Baral. 2020.<br>Self-|**651**|
|supervised knowledge triplet learning for zero-shot|**652**|
|question answering. In_Proceedings of the 2020 Con-_|**653**|
|_ference on Empirical Methods in Natural Language_|**654**|
|_Processing, EMNLP 2020, Online, November 16-20,_|**655**|
|_2020_. Association for Computational Linguistics.|**656**|
|Chandra Bhagavatula, Ronan Le Bras, Chaitanya|**657**|
|Malaviya, Keisuke Sakaguchi, Ari Holtzman, Han-|**658**|
|nah Rashkin, Doug Downey, Wen-tau Yih, and Yejin|**659**|
|Choi. 2020. Abductive commonsense reasoning. In|**660**|
|_8th International Conference on Learning Represen-_|**661**|
|_tations, ICLR 2020, Addis Ababa, Ethiopia, April_|**662**|
|_26-30, 2020_. OpenReview.net.|**663**|
|Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng|**664**|
|Gao, and Yejin Choi. 2020. PIQA: reasoning about|**665**|
|physical commonsense in natural language. In_The_|**666**|
|_Thirty-Fourth AAAI Conference on Artifcial Intelli-_|**667**|
|_gence, AAAI 2020, The Thirty-Second Innovative Ap-_|**668**|
|_plications of Artifcial Intelligence Conference, IAAI_|**669**|
|_2020, The Tenth AAAI Symposium on Educational_|**670**|
|_Advances in Artifcial Intelligence, EAAI 2020, New_|**671**|
|_York, NY, USA, February 7-12, 2020_. AAAI Press.|**672**|
|Antoine Bordes, Nicolas Usunier, Alberto García-|**673**|
|Durán, Jason Weston, and Oksana Yakhnenko.|**674**|
|2013. Translating embeddings for modeling multi-|**675**|
|relational data. In_Advances in Neural Information_|**676**|
|_Processing Systems 26: 27th Annual Conference on_|**677**|
|_Neural Information Processing Systems 2013. Pro-_|**678**|
|_ceedings of a meeting held December 5-8, 2013, Lake_|**679**|
|_Tahoe, Nevada, United States_, pages 2787–2795.|**680**|



9 

_gence, EAAI 2021, Virtual Event, February 2-9, 2021_ , **740** pages 12857–12865. AAAI Press. **741** 

- **681** Antoine Bosselut, Ronan Le Bras, and Yejin Choi. 2021. **682** Dynamic neuro-symbolic knowledge graph construc- **683** tion for zero-shot commonsense question answering. **684** In _Thirty-Fifth AAAI Conference on Artificial Intel-_ **685** _ligence, AAAI 2021, Thirty-Third Conference on In-_ **686** _novative Applications of Artificial Intelligence, IAAI_ **687** _2021, The Eleventh Symposium on Educational Ad-_ **688** _vances in Artificial Intelligence, EAAI 2021, Virtual_ **689** _Event, February 2-9, 2021_ . AAAI Press. 

- **690** Antoine Bosselut, Hannah Rashkin, Maarten Sap, Chai- **691** tanya Malaviya, Asli Celikyilmaz, and Yejin Choi. **692** 2019. COMET: commonsense transformers for auto- **693** matic knowledge graph construction. In _Proceedings_ **694** _of the 57th Conference of the Association for Com-_ **695** _putational Linguistics, ACL 2019, Florence, Italy,_ **696** _July 28- August 2, 2019, Volume 1: Long Papers_ . **697** Association for Computational Linguistics. 

- **698** Nathanael Chambers and Daniel Jurafsky. 2008. Unsu- **699** pervised learning of narrative event chains. In _ACL_ **700** _2008, Proceedings of the 46th Annual Meeting of_ **701** _the Association for Computational Linguistics, June_ **702** _15-20, 2008, Columbus, Ohio, USA_ , pages 789–797. **703** The Association for Computer Linguistics. 

- **704** Zeming Chen, Alejandro Hernández-Cano, Angelika **705** Romanou, Antoine Bonnet, Kyle Matoba, Francesco **706** Salvi, Matteo Pagliardini, Simin Fan, Andreas **707** Köpf, Amirkeivan Mohtashami, Alexandre Sallinen, **708** Alireza Sakhaeirad, Vinitra Swamy, Igor Krawczuk, **709** Deniz Bayazit, Axel Marmet, Syrielle Montariol, **710** Mary-Anne Hartley, Martin Jaggi, and Antoine **711** Bosselut. 2023. MEDITRON-70B: scaling medi- **712** cal pretraining for large language models. _CoRR_ , **713** abs/2311.16079. 

- **714** Hyung Won Chung, Le Hou, Shayne Longpre, Barret **715** Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, **716** Mostafa Dehghani, Siddhartha Brahma, Albert Web- **717** son, Shixiang Shane Gu, Zhuyun Dai, Mirac Suz- **718** gun, Xinyun Chen, Aakanksha Chowdhery, Sharan **719** Narang, Gaurav Mishra, Adams Yu, Vincent Y. Zhao, **720** Yanping Huang, Andrew M. Dai, Hongkun Yu, Slav **721** Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam **722** Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. **723** 2022. Scaling instruction-finetuned language models. **724** _CoRR_ , abs/2210.11416. 

- **725** Biaoyan Fang, Timothy Baldwin, and Karin Verspoor. **726** 2022. What does it take to bake a cake? the reciperef **727** corpus and anaphora resolution in procedural text. **728** In _Findings of the Association for Computational_ **729** _Linguistics: ACL 2022, Dublin, Ireland, May 22-27,_ **730** _2022_ , pages 3481–3495. Association for Computa- **731** tional Linguistics. 

- **732** Saadia Gabriel, Chandra Bhagavatula, Vered Shwartz, **733** Ronan Le Bras, Maxwell Forbes, and Yejin Choi. **734** 2021. Paragraph-level commonsense transformers **735** with recurrent memory. In _Thirty-Fifth AAAI Con-_ **736** _ference on Artificial Intelligence, AAAI 2021, Thirty-_ **737** _Third Conference on Innovative Applications of Arti-_ **738** _ficial Intelligence, IAAI 2021, The Eleventh Sympo-_ **739** _sium on Educational Advances in Artificial Intelli-_ 

- Silin Gao, Jena D. Hwang, Saya Kanno, Hiromi Wakaki, **742** Yuki Mitsufuji, and Antoine Bosselut. 2022. Com- **743** fact: A benchmark for linking contextual common- **744** sense knowledge. In _Findings of the Association_ **745** _for Computational Linguistics: EMNLP 2022, Abu_ **746** _Dhabi, United Arab Emirates, December 7-11, 2022_ . **747** Association for Computational Linguistics. **748** 

- William L. Hamilton, Payal Bajaj, Marinka Zitnik, Dan **749** Jurafsky, and Jure Leskovec. 2018. Embedding log- **750** ical queries on knowledge graphs. In _Advances in_ **751** _Neural Information Processing Systems 31: Annual_ **752** _Conference on Neural Information Processing Sys-_ **753** _tems 2018, NeurIPS 2018, December 3-8, 2018, Mon-_ **754** _tréal, Canada_ , pages 2030–2041. **755** 

- Pengcheng He, Jianfeng Gao, and Weizhu Chen. 2023. **756** DeBERTav3: Improving deBERTa using ELECTRA- **757** style pre-training with gradient-disentangled embed- **758** ding sharing. In _The Eleventh International Confer-_ **759** _ence on Learning Representations_ . **760** 

- Jena D. Hwang, Chandra Bhagavatula, Ronan Le Bras, **761** Jeff Da, Keisuke Sakaguchi, Antoine Bosselut, and **762** Yejin Choi. 2021. (comet-) atomic 2020: On sym- **763** bolic and neural commonsense knowledge graphs. **764** In _Thirty-Fifth AAAI Conference on Artificial Intel-_ **765** _ligence, AAAI 2021, Thirty-Third Conference on In-_ **766** _novative Applications of Artificial Intelligence, IAAI_ **767** _2021, The Eleventh Symposium on Educational Ad-_ **768** _vances in Artificial Intelligence, EAAI 2021, Virtual_ **769** _Event, February 2-9, 2021_ . AAAI Press. **770** 

- Daniel Khashabi, Yeganeh Kordi, and Hannaneh Ha- **771** jishirzi. 2022. Unifiedqa-v2: Stronger general- **772** ization via broader cross-format training. _CoRR_ , **773** abs/2202.12359. **774** 

- Hyunwoo Kim, Jack Hessel, Liwei Jiang, Peter West, **775** Ximing Lu, Youngjae Yu, Pei Zhou, Ronan Le Bras, **776** Malihe Alikhani, Gunhee Kim, Maarten Sap, and **777** Yejin Choi. 2023. SODA: million-scale dialogue dis- **778** tillation with social commonsense contextualization. **779** In _Proceedings of the 2023 Conference on Empirical_ **780** _Methods in Natural Language Processing, EMNLP_ **781** _2023, Singapore, December 6-10, 2023_ , pages 12930– **782** 12949. Association for Computational Linguistics. **783** 

- Yu Jin Kim, Beong-woo Kwak, Youngwook Kim, **784** Reinald Kim Amplayo, Seung-won Hwang, and Jiny- **785** oung Yeo. 2022. Modularized transfer learning with **786** multiple knowledge graphs for zero-shot common- **787** sense reasoning. In _Proceedings of the 2022 Con-_ **788** _ference of the North American Chapter of the As-_ **789** _sociation for Computational Linguistics: Human_ **790** _Language Technologies, NAACL 2022, Seattle, WA,_ **791** _United States, July 10-15, 2022_ . Association for Com- **792** putational Linguistics. **793** 

- Qika Lin, Rui Mao, Jun Liu, Fangzhi Xu, and Erik **794** Cambria. 2023. Fusing topology contexts and log- **795** ical rules in language models for knowledge graph **796** completion. _Inf. Fusion_ , 90:253–264. **797** 

10 

- **798** Jiacheng Liu, Wenya Wang, Dianzhuo Wang, Noah A. **799** Smith, Yejin Choi, and Hannaneh Hajishirzi. 2023. **800** Vera: A general-purpose plausibility estimation **801** model for commonsense statements. In _Proceedings_ **802** _of the 2023 Conference on Empirical Methods in Nat-_ **803** _ural Language Processing, EMNLP 2023, Singapore,_ **804** _December 6-10, 2023_ , pages 1264–1287. Association **805** for Computational Linguistics. 

- **806** Kaixin Ma, Filip Ilievski, Jonathan Francis, Yonatan **807** Bisk, Eric Nyberg, and Alessandro Oltramari. 2021. **808** Knowledge-driven data construction for zero-shot **809** evaluation in commonsense question answering. In **810** _Thirty-Fifth AAAI Conference on Artificial Intelli-_ **811** _gence, AAAI 2021, Thirty-Third Conference on In-_ **812** _novative Applications of Artificial Intelligence, IAAI_ **813** _2021, The Eleventh Symposium on Educational Ad-_ **814** _vances in Artificial Intelligence, EAAI 2021, Virtual_ **815** _Event, February 2-9, 2021_ . AAAI Press. 

- **816** Chaitanya Malaviya, Chandra Bhagavatula, Antoine **817** Bosselut, and Yejin Choi. 2020. Commonsense **818** knowledge base completion with structural and se- **819** mantic context. In _The Thirty-Fourth AAAI Con-_ **820** _ference on Artificial Intelligence, AAAI 2020, The_ **821** _Thirty-Second Innovative Applications of Artificial_ **822** _Intelligence Conference, IAAI 2020, The Tenth AAAI_ **823** _Symposium on Educational Advances in Artificial In-_ **824** _telligence, EAAI 2020, New York, NY, USA, February_ **825** _7-12, 2020_ . AAAI Press. 

- **826** Ninareh Mehrabi, Pei Zhou, Fred Morstatter, Jay Pu- **827** jara, Xiang Ren, and Aram Galstyan. 2021. Lawyers **828** are dishonest? quantifying representational harms **829** in commonsense knowledge resources. In _Proceed-_ **830** _ings of the 2021 Conference on Empirical Methods_ **831** _in Natural Language Processing, EMNLP 2021, Vir-_ **832** _tual Event / Punta Cana, Dominican Republic, 7-11_ **833** _November, 2021_ , pages 5016–5033. Association for **834** Computational Linguistics. 

- **835** Nasrin Mostafazadeh, Aditya Kalyanpur, Lori Moon, **836** David W. Buchanan, Lauren Berkowitz, Or Biran, **837** and Jennifer Chu-Carroll. 2020. GLUCOSE: gen- **838** eralized and contextualized story explanations. In **839** _Proceedings of the 2020 Conference on Empirical_ **840** _Methods in Natural Language Processing, EMNLP_ **841** _2020, Online, November 16-20, 2020_ . Association **842** for Computational Linguistics. 

- **843** Karl Pichotta and Raymond J. Mooney. 2014. Statis- **844** tical script learning with multi-argument events. In **845** _Proceedings of the 14th Conference of the European_ **846** _Chapter of the Association for Computational Lin-_ **847** _guistics, EACL 2014, April 26-30, 2014, Gothenburg,_ **848** _Sweden_ , pages 220–229. The Association for Com- **849** puter Linguistics. 

- **850** Sahithya Ravi, Raymond Ng, and Vered Shwartz. 2023. **851** COMET-M: reasoning about multiple events in com- **852** plex sentences. In _Findings of the Association for_ **853** _Computational Linguistics: EMNLP 2023, Singapore,_ **854** _December 6-10, 2023_ , pages 12921–12937. Associa- **855** tion for Computational Linguistics. 

- Hongyu Ren, Weihua Hu, and Jure Leskovec. 2020. **856** Query2box: Reasoning over knowledge graphs in **857** vector space using box embeddings. In _8th Inter-_ **858** _national Conference on Learning Representations,_ **859** _ICLR 2020, Addis Ababa, Ethiopia, April 26-30,_ **860** _2020_ . OpenReview.net. **861** 

- Hongyu Ren and Jure Leskovec. 2020. Beta embed- **862** dings for multi-hop logical reasoning in knowledge **863** graphs. In _Advances in Neural Information Process-_ **864** _ing Systems 33: Annual Conference on Neural In-_ **865** _formation Processing Systems 2020, NeurIPS 2020,_ **866** _December 6-12, 2020, virtual_ . **867** 

- Joshua Robinson, Christopher Michael Rytting, and **868** David Wingate. 2022. Leveraging large language **869** models for multiple choice question answering. **870** _CoRR_ , abs/2210.12353. **871** 

- Rachel Rudinger, Vered Shwartz, Jena D. Hwang, Chan- **872** dra Bhagavatula, Maxwell Forbes, Ronan Le Bras, **873** Noah A. Smith, and Yejin Choi. 2020. Thinking like **874** a skeptic: Defeasible inference in natural language. **875** In _Findings of the Association for Computational Lin-_ **876** _guistics: EMNLP 2020, Online Event, 16-20 Novem-_ **877** _ber 2020_ , volume EMNLP 2020 of _Findings of ACL_ , **878** pages 4661–4675. Association for Computational **879** Linguistics. **880** 

- Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavat- **881** ula, and Yejin Choi. 2021. Winogrande: an adver- **882** sarial winograd schema challenge at scale. _Commun._ **883** _ACM_ , 64(9). **884** 

- Maarten Sap, Ronan Le Bras, Emily Allaway, Chan- **885** dra Bhagavatula, Nicholas Lourie, Hannah Rashkin, **886** Brendan Roof, Noah A. Smith, and Yejin Choi. **887** 2019a. ATOMIC: an atlas of machine commonsense **888** for if-then reasoning. In _The Thirty-Third AAAI Con-_ **889** _ference on Artificial Intelligence, AAAI 2019, The_ **890** _Thirty-First Innovative Applications of Artificial In-_ **891** _telligence Conference, IAAI 2019, The Ninth AAAI_ **892** _Symposium on Educational Advances in Artificial_ **893** _Intelligence, EAAI 2019, Honolulu, Hawaii, USA,_ **894** _January 27 - February 1, 2019_ . AAAI Press. **895** 

- Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le **896** Bras, and Yejin Choi. 2019b. Social iqa: Common- **897** sense reasoning about social interactions. In _Proceed-_ **898** _ings of the 2019 Conference on Empirical Methods_ **899** _in Natural Language Processing and the 9th Inter-_ **900** _national Joint Conference on Natural Language Pro-_ **901** _cessing, EMNLP-IJCNLP 2019, Hong Kong, China,_ **902** _November 3-7, 2019_ . Association for Computational **903** Linguistics. **904** 

- Roger C. Schank and Robert P. Abelson. 1975. Scripts, **905** plans and knowledge. In _Advance Papers of the_ **906** _Fourth International Joint Conference on Artificial_ **907** _Intelligence, Tbilisi, Georgia, USSR, September 3-8,_ **908** _1975_ , pages 151–157. **909** 

- Vered Shwartz, Peter West, Ronan Le Bras, Chandra **910** Bhagavatula, and Yejin Choi. 2020. Unsupervised **911** commonsense question answering with self-talk. In **912** 

11 

**913** _Proceedings of the 2020 Conference on Empirical_ **914** _Methods in Natural Language Processing, EMNLP_ **915** _2020, Online, November 16-20, 2020_ . Association **916** for Computational Linguistics. 

**917** Robyn Speer, Joshua Chin, and Catherine Havasi. 2017. **918** Conceptnet 5.5: An open multilingual graph of gen- **919** eral knowledge. In _Proceedings of the Thirty-First_ **920** _AAAI Conference on Artificial Intelligence, February_ **921** _4-9, 2017, San Francisco, California, USA_ . AAAI **922** Press. 

for zero-shot commonsense question answering. In **972** _Findings of the Association for Computational Lin-_ **973** _guistics: EMNLP 2023_ , pages 13520–13545, Singa- **974** pore. Association for Computational Linguistics. **975** Zihao Wang, Yangqiu Song, Ginny Y. Wong, and Si- **976** mon See. 2023b. Logical message passing networks **977** with one-hop inference on atomic formulas. In _The_ **978** _Eleventh International Conference on Learning Rep-_ **979** _resentations, ICLR 2023, Kigali, Rwanda, May 1-5,_ **980** _2023_ . OpenReview.net. **981** 

- **923** Ying Su, Zihao Wang, Tianqing Fang, Hongming Zhang, **924** Yangqiu Song, and Tong Zhang. 2022. MICO: A **925** multi-alternative contrastive learning framework for **926** commonsense knowledge representation. In _Find-_ **927** _ings of the Association for Computational Linguistics:_ **928** _EMNLP 2022, Abu Dhabi, United Arab Emirates, De-_ **929** _cember 7-11, 2022_ . Association for Computational **930** Linguistics. 

- **931** Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian **932** Tang. 2019. Rotate: Knowledge graph embedding **933** by relational rotation in complex space. In _7th In-_ **934** _ternational Conference on Learning Representations,_ **935** _ICLR 2019, New Orleans, LA, USA, May 6-9, 2019_ . **936** OpenReview.net. 

- **937** Alon Talmor, Jonathan Herzig, Nicholas Lourie, and **938** Jonathan Berant. 2019. Commonsenseqa: A question **939** answering challenge targeting commonsense knowl- **940** edge. In _Proceedings of the 2019 Conference of_ **941** _the North American Chapter of the Association for_ **942** _Computational Linguistics: Human Language Tech-_ **943** _nologies, NAACL-HLT 2019, Minneapolis, MN, USA,_ **944** _June 2-7, 2019, Volume 1 (Long and Short Papers)_ . **945** Association for Computational Linguistics. 

- **946** Hugo Touvron, Louis Martin, Kevin Stone, Peter Al- **947** bert, Amjad Almahairi, Yasmine Babaei, Nikolay **948** Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti **949** Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton- **950** Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, **951** Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, **952** Cynthia Gao, Vedanuj Goswami, Naman Goyal, An- **953** thony Hartshorn, Saghar Hosseini, Rui Hou, Hakan **954** Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, **955** Isabel Kloumann, Artem Korenev, Punit Singh Koura, **956** Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Di- **957** ana Liskovich, Yinghai Lu, Yuning Mao, Xavier Mar- **958** tinet, Todor Mihaylov, Pushkar Mishra, Igor Moly- **959** bog, Yixin Nie, Andrew Poulton, Jeremy Reizen- **960** stein, Rashi Rungta, Kalyan Saladi, Alan Schelten, **961** Ruan Silva, Eric Michael Smith, Ranjan Subrama- **962** nian, Xiaoqing Ellen Tan, Binh Tang, Ross Tay- **963** lor, Adina Williams, Jian Xiang Kuan, Puxin Xu, **964** Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, **965** Melanie Kambadur, Sharan Narang, Aurélien Ro- **966** driguez, Robert Stojnic, Sergey Edunov, and Thomas **967** Scialom. 2023. Llama 2: Open foundation and fine- **968** tuned chat models. _CoRR_ , abs/2307.09288. 

   - Zihao Wang, Hang Yin, and Yangqiu Song. 2021. **982** Benchmarking the combinatorial generalizability of **983** complex query answering on knowledge graphs. **984** In _Proceedings of the Neural Information Process-_ **985** _ing Systems Track on Datasets and Benchmarks 1,_ **986** _NeurIPS Datasets and Benchmarks 2021, December_ **987** _2021, virtual_ . **988** 

   - Peter West, Ronan Bras, Taylor Sorensen, Bill Lin, Li- **989** wei Jiang, Ximing Lu, Khyathi Chandu, Jack Hessel, **990** Ashutosh Baheti, Chandra Bhagavatula, and Yejin **991** Choi. 2023. NovaCOMET: Open commonsense **992** foundation models with symbolic knowledge distil- **993** lation. In _Findings of the Association for Computa-_ **994** _tional Linguistics: EMNLP 2023_ , pages 1127–1149, **995** Singapore. Association for Computational Linguis- **996** tics. **997** 

   - Yiben Yang, Chaitanya Malaviya, Jared Fernandez, **998** Swabha Swayamdipta, Ronan Le Bras, Ji-Ping Wang, **999** Chandra Bhagavatula, Yejin Choi, and Doug Downey. **1000** 2020. G-daug: Generative data augmentation for **1001** commonsense reasoning. In _Findings of the As-_ **1002** _sociation for Computational Linguistics: EMNLP_ **1003** _2020, Online Event, 16-20 November 2020_ , volume **1004** EMNLP 2020 of _Findings of ACL_ , pages 1008–1025. **1005** Association for Computational Linguistics. **1006** 

   - Siyu Yuan, Jiangjie Chen, Ziquan Fu, Xuyang Ge, So- **1007** ham Shah, Charles Robert Jankowski, Yanghua Xiao, **1008** and Deqing Yang. 2023. Distilling script knowledge **1009** from large language models for constrained language **1010** planning. In _Proceedings of the 61st Annual Meeting_ **1011** _of the Association for Computational Linguistics (Vol-_ **1012** _ume 1: Long Papers), ACL 2023, Toronto, Canada,_ **1013** _July 9-14, 2023_ , pages 4303–4325. Association for **1014** Computational Linguistics. **1015** 

   - Wenting Zhao, Mor Geva, Bill Yuchen Lin, Michihiro **1016** Yasunaga, Aman Madaan, and Tao Yu. 2023. Com- **1017** plex reasoning in natural languag. In _Proceedings_ **1018** _of the 61st Annual Meeting of the Association for_ **1019** _Computational Linguistics: Tutorial Abstracts, ACL_ **1020** _2023, Toronto, Canada, July 9-14, 2023_ , pages 11–20. **1021** Association for Computational Linguistics. **1022** 

- **969** Weiqi Wang, Tianqing Fang, Wenxuan Ding, Baixuan **970** Xu, Xin Liu, Yangqiu Song, and Antoine Bosselut. **971** 2023a. CAR: Conceptualization-augmented reasoner 

12 

**1025 1026 1027 1028** 

**1029** 

**1030 1031 1032 1033 1034** 

**1035** 

**1057 1058 1059 1060 1061** 

## **1023 A Additional Details on Data 1024 Construction** 

In this section, we provide additional details to node normalization, plausibility filter, verbalization, and human annotations. The overview of our construction framework is presented in Fig. 4. 

## **A.1 Nodes Normalization** 

We present the normalization rules in Tab. 5. For example, a tail of “to go” under the relation xWant will be transformed to “PersonX go”. A tail of “satisfied” under the relation xAttr will be transformed to “PersonX is satisfied”. 

|Relations|Mappingrules|
|---|---|
|xWant/oWant/<br>xIntent/xNeed<br>xEffect/oEffect<br>xReact/oReact<br>xAttr|Add PersonX/Y in front of the tail<br>and remove the initial “to”<br>Add PersonX/Y in front of the tail<br>Add PersonX/Y and “is” in front of<br>the tail<br>Add a PersonX/Y and “is” in front<br>of the tail|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Complex_Reasoning_over_Logical_Queries_on_Commonsense_Knowledge_Graphs_images/acl_2024_Complex_Reasoning_over_Logical_Queries_on_Commonsense_Knowledge_Graphs.pdf-0013-10.png)


**----- Start of picture text -----**<br>
Normalization Query<br>& Filtering Sampling<br>CSKB<br>Context:  𝑓(𝑉!, 𝑉", 𝑉#, 𝑟!, 𝑟", 𝑟#)<br>Verbalization<br>Question:  𝑔(𝑉!, 𝑉", 𝑉#, 𝑟!, 𝑟", 𝑟#3) &<br>Refinement<br>Synthetic Answer:  𝑉?<br>Distantly-supervised Human-Annotated<br>Training set Evaluation set<br>**----- End of picture text -----**<br>


Figure 4: Overview of the construction process. 

training data, which has been proven useful for se- **1062** lecting augmented data. To be more specific, for **1063** each unique answer, we adopt an iterative approach **1064** to select the verbalized query that contributes the **1065** highest number of unique 1-gram terms to an on- **1066** going vocabulary constructed for each answer. We **1067** select top-20 queries for each unique answer entity. **1068** 

Table 5: Normalization rules for ATOMIC tails. 

## **A.2 Data Filtering** 

**1036 Plausibility Filter** We verbalize a ( _h, r, t_ ) triple **1037** from ATOMIC using the default template as pro- **1038** vided in Hwang et al. (2021). For example, (Per- **1039** sonX repels PersonY’s attack, xAttr, brave) would **1040** be transformed to a declarative statement “If Per- **1041** sonX repels PersonY’s attack, then PersonX is seen **1042** as brave”. To obtain a plausibility score, we input **1043** the statement into the Vera-5B model. 0.5 is used as **1044** the threshold to draw a boundary between plausible **1045** and implausible statements. We perform a manual **1046** inspection on the triples scored by Vera and ran- **1047** domly select 40 samples for three plausibility score **1048** intervals. Among these, we find that 4/40 triples **1049** are plausible when the Vera scores range from 0 to **1050** 0.1. 13/40 triples are considered plausible within **1051** the score range of 0.2 to 0.25. Furthermore, we **1052** identify 20/40 triples as plausible when their plau- **1053** sibility scores hover around 0.5, when most of the **1054** triples are quite ambiguous. By setting the filter **1055** threshold as 0.5, we filter out around 14% triples **1056** that are of a relatively lower quality. 

**Diversity Filter** To prevent overfitting to common tails, we conduct a diversity-based filter to acquire diverse queries for training. We take inspirations from G-DAUG (Yang et al., 2020), to use a simple greedy algorithm to iteratively select 

## **A.3 Verbalization** 

**1069 Query Verbalization** We employ two methods **1070** to verbalize complex queries: a rule-based method **1071** and a ChatGPT-based method. **1072** In the case of 2i and 3i queries, the rule-based **1073** method typically involves inserting an “and” be- **1074** tween the anchor entities. However, if the query **1075** suggests a specific chronological order between the **1076** two events, we use “then” to connect the events. **1077** For instance, in 2i queries where one triple is ( _V_ 1, **1078** xEffect, _V_ ?) and the other is ( _V_ 2, xIntent, _V_ ?), it **1079** implies that _V_ ? serves as the effect of _V_ 1 and the **1080** intermediate hidden cause of _V_ 2. In this scenario, **1081** _V_ 1 should occur before _V_ 2. Therefore, the verbal- **1082** ization would be “ _V_ 1 then _V_ 2”. **1083** For ChatGPT verbalization, we present the sys- **1084** tem instructions for verbalizing different kinds of **1085** queries in Tab. 6. Then, we generate the verbal- **1086** ized contexts with six exemplars that are manually **1087** annotated. In the system instruction, we also ask **1088** ChatGPT to output “NA” if the given anchor en- **1089** tities are totally irrelevant or too ambiguous. We **1090** filter out those queries where the output is “NA”. **1091** For example, to better interpret the query in **1092** Fig. 1, we need to take into consideration both **1093** the relations of interest and the anchor entities. The **1094** query asks about the effect of the first event and **1095** what causes (intention) of the second event, which **1096** is inherently represents _abductive reasoning_ . This **1097** 

13 

**1123** 

**1124 1125** 

|Query|Prompt|
|---|---|
|||
|2i, ip, pi|Given two events, come up with concise and necessary context to make the a coherent and understand-<br>able narrative. No more than 2 additional piece of context should be added. If the one of the given<br>events itself is ambiguous and hardly make sense even with extra context, return NA. If the two events<br>are totally irrelevant even with additional context, then simply return NA. If the given two events can<br>be directly composed to a narrative with simple a discourse connective without additional context,<br>then there’s not need to add additional context.\nMark the location of both events with <E1></E1><br>for event 1 and <E2></E2> for event 2 in the generated narrative.|
|||
|2i-neg|Given two events, create a cohesive narrative by incorporating event 1 (E1) and negated event 2 (E2) to<br>make the a coherent and understandable narrative. No more than 2 additional piece of context should<br>be added. If the one of the given events itself is ambiguous and hardly make sense even with extra<br>context, return NA. If the two events are totally irrelevant even with additional context, then simply<br>return NA. If the given two events can be directly composed to a narrative with simple a discourse<br>connective without additional context, then there’s not need to add additional context.\nMark the<br>location of both events with <E1></E1> for event 1 and <E2></E2> for event 2 in the generated<br>narrative.\nDon’t explain the reasons why E2 didn’t happen!!\nRemember that negating an event<br>means stating that it did not occur. For instance, if event 2 is “PersonX goes shopping,” the negated<br>form would be “PersonX didn’t go shopping”.|



Table 6: System instructions for verbalizing complex queries given different query types. 

**1098** requires the second event to happen before the first **1099** event, to derive reasonable abduction. In this sense, **1100** a natural rule of verbalizing the query would be **1101** adding a discourse connective “after” to convert **1102** the query to “After PersonX gets tired of it, Per- **1103** sonX goes skydiving”. However, the verbalized **1104** query may still be ambiguous without additional **1105** context. To make the verbalized context more in- **1106** formative and human-understandable, we take ad- **1107** vantage of Large Language Models (i.e., ChatGPT) **1108** to add additional context to compose the query to a **1109** narrative. 

|Query<br>Type|Question Template|
|---|---|
|||
|2i<br>3i<br>2p<br>ip<br>pi|What event or state is both Prompt(r1) [V1] and<br>also prompt(r2) [V2]?<br>What event or state is both Prompt(r1) [V1],<br>Prompt(r2) [V2], and also Prompt(r2) [V2]?<br>What event or state is Prompt(r1) {Prompt(r2)<br>[V1]}?<br>What event or state is prompt(r3) {both prompt(r1)<br>[V1], and also prompt(r2) [V2] }?<br>What event or state is both prompt(r1) {prompt(r3)<br>[V3]}, and also prompt(r2) [V2]?|



Table 7: Templates for verbalizing one-hop relations. 

**1110 Relation Verbalization** We use conversion rules **1111** and pre-defined templates to compose questions **1112** based on the relations in the queries. Based on the **1113** definition of each commonsense relation (Hwang **1114** et al., 2021), we use the templates in Tab. 7 to ver- **1115** balize each relation. In terms of complex queries, **1116** we use the conversion rules in Tab. 8 to convert the **1117** query to a question. 

**1118 Person Names** To make the context more nat- **1119** ural, we replace PersonX, PersonY, PersonZ in **1120** the context to names randomly sampled from the **1121** 2021 public US social security application name **1122** registry[5] . 

## **A.4 Human Annotation** 

We introduce the details of the annotation process in this subsection. 

> 5https://catalog.data.gov/dataset/baby-names-fromsocial-security-card-applications-national-data 

**Worker Selection** We have a qualification test to **1126** select eligible workers for the main task. We pre- **1127** pare six pre-selected 2i queries of different types, **1128** including (negated) common effect, (negated) com- **1129** mon cause, common attribute, and abduction. We **1130** compare the pair-wise annotation accuracy be- **1131** tween each annotator and the gold answer anno- **1132** tated by the authors of the paper, and select those **1133** who have at least 85% agreement as qualified work- **1134** ers. After selection, we pick 53 worker out of 120 **1135** participants in the qualification round. **1136** 

**Annotation Interface** A snapshot of the annota- **1137** tion interface is presented at Fig. 5. In addition, we **1138** have provided comprehensive instructions along **1139** with detailed examples to guide the annotators **1140** throughout the annotation process. To ensure their **1141** understanding, we require annotators to confirm **1142** that they have thoroughly read the instructions by **1143** checking a checkbox before the annotation task. **1144** We also manually checked the performance of the **1145** annotators along with the annotation process and **1146** 

14 

ually. **1163** Tab. 9 presents the statistics of the training and **1164** evaluation set. **1165** 

**1166** 

**1147** gave feedbacks based on common errors. For ex- **1148** ample, typical errors include mistakenly regard the **1149** one-hop answer as correct instead of fully consid- **1150** ering the multi-hop context. 

**1182 1183** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Complex_Reasoning_over_Logical_Queries_on_Commonsense_Knowledge_Graphs_images/acl_2024_Complex_Reasoning_over_Logical_Queries_on_Commonsense_Knowledge_Graphs.pdf-0015-04.png)


Figure 5: Annotation interface. 

|Relation|Prompt Template|
|---|---|
|||
|xIntent<br>xNeed<br>xWant<br>xEffect<br>xReact<br>xAttr<br>oEffect<br>oReact<br>oWant<br>HinderedBy<br>isAfter<br>isBefore|the intention of PersonX before<br>what PersonX needed to do before<br>what PersonX wants to do after<br>the effect on PersonX after<br>what PersonX feels after<br>what PersonX is seen as given<br>the effect on PersonY after<br>what PersonY feels after<br>what PersonY wants to do after<br>what hindered<br>what happens before<br>what happens after|



Table 8: Templates for verbalizing relations in complex queries. 

**1151 Post-processing** To aggregate the annotation re- **1152** sult, we randomly sample one option that is labeled **1153** as plausible by majority voting as the final positive **1154** answer, and sample three negative options and dis- **1155** tractors. If there are no options labeled as plausible, **1156** then the correct answer is “None of the answers are **1157** correct”. If there are less than three options labeled **1158** as negative, we manually add one or two negative **1159** examples to match the number. To improve the **1160** quality, after crowdsourcing, the authors of this pa- **1161** per manually checked the QA pairs with an IAA **1162** lower than 0.6, and resolve the disagreements man- 

ually. 

||Training<br>Evaluation|
|---|---|
|#Instances<br>782,140<br>1,317||



Table 9: Statistics of COM[2] . 

## **B Additional Details of Experiments** 

## **B.1 Implementation Details of the Question Answering Models** 

**Implementation Details of the Question 1167 Answering Models 1168** We follow the pipeline in HyKAS (Ma et al., 2021) **1169** and CAR (Wang et al., 2023a) Let _C_ represent **1170** the original context, which is the head entity for **1171** 1p triple and the verbalized context for complex **1172** queries, _Q_ represent the question verbalized from **1173** the anchor relations, and ( _A_ 1 _, A_ 2 _, ..._ ) be the list of **1174** options. We first concatenate _C_ , _Q_ , and an answer **1175** option _Ai_ together via natural language prompts **1176** following the order of “ _C Q Ai_ ” to generate input **1177** sequences ( _T_ 1 _, T_ 2 _, ..._ ). We then repeatedly mask **1178** out one token at a time to calculate the masked **1179** language modeling loss. **1180** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Complex_Reasoning_over_Logical_Queries_on_Commonsense_Knowledge_Graphs_images/acl_2024_Complex_Reasoning_over_Logical_Queries_on_Commonsense_Knowledge_Graphs.pdf-0015-15.png)


We then compute the marginal ranking loss based on Equation 3, where _η_ represents the margin 

15 

**1184** 

**1185** 

**1186** We train the DeBERTa QA model for 1 epoch **1187** with a learning rate of 5e-6 and a linear learn- **1188** ing rate decay. The checkpoint that yields the **1189** best performance on the synthetic validation set **1190** in CAR (Wang et al., 2023a) or HyKAS (Ma et al.,Wang et al., 2023a) or HyKAS (Ma et al.,, 2023a) or HyKAS (Ma et al., 2023a) or HyKAS (Ma et al.,) or HyKAS (Ma et al.,Ma et al.,, **1191** 2021) is selected as the final model.) is selected as the final model. During eval- **1192** uating, we select the option that yields the lowest **1193** score as the final prediction. **1194** We provide the prompt templates for each model **1195** in Tab. 10. 

**1196 1197** 

**1198 1199 1200** 

**1201 1202 1203** 

**1204** 

|Model|Prompt|
|---|---|
|||
|Llama2, Flan-T5<br>ChatGPT, GPT-4|Answer this commonsense reasoning question, where you are supposed to handle a multiple-chioce<br>question answering task to select the correct answer. Select one correct answer from A to E.\n<br>Context: [Context] Question: [Question] A: [Option A]. B: [Option B]. C: [Option C]. D: [Option D].<br>E: [Option E]. \n<br>Answer:|
|||
|UnifedQA|[Question] \n<br>(a): [Option A] (b) [Option B] (c) [Option C] (d) [Option D] (e) [Option E] \n<br>[Context]|
|||
|Vera|[Context] [Question] [Option]|
|||
|HyKAS, CAR|[Context] [Question] [Option]|



Table 10: Prompt templates for multiple-choice question answering. 

|Model|Prompt|
|---|---|
|||
|Llama2 (zero-shot)|[System_Message] = As an expert in commonsense reasoning, your task is to provide a concise<br>response to a question based on the given context. The question focuses on studying the causes,<br>effects, or attributes of personas related to the given context. Answer shortly with no more than 5<br>words.<br><s>[INST] <<SYS>>\n[System_Message] \n<</SYS>>\n\n[Context] [Question] [/INST]|
|||
|Llama2 (fne-tuned)|<|im_start|>question\n[Context] [Question] <|im_end|>\n<|im_start|>answer\n[Answer]|
|||
|GPT-2|2i: [V1] [V2] [r1] [r2] [GEN] [Answer]<br>3i: [V1] [V2] [V3] [r1] [r2] [r3] [GEN] [Answer]<br>2p: [V1] [r1] [r2] [GEN] [Answer]|



Table 11: Prompts for fine-tuning generative commonsense inference models. 

and _y_ is the index of the correct answer. 

is the index of the correct answer. the expected output is _V_ ?. The input of 2i queries, **1205** _m_ which includes ( _h_ 1, _r_ 1, _V_ ?) and ( _h_ 2, _r_ 2, _V_ ?), is se- **1206** _L_ = _m_[1] _i_ =1� _,i_ = _y_ max(0 _, η − Sy_ + _Si_ ) (3) rialized as “as _V_ ?. All models are fine-tuned for 3 epochs with _h_ 1 _h_ 2 _r_ 1 _r_ 2” with the expected output **12071208** a batch size of 32, a learning rate of 1e-5, a linear **1209** We train the DeBERTa QA model for 1 epoch learning rate decay. The last checkpoint is taken as **1210** learning rate of 5e-6 and a linear learnthe final model. **1211** decay. The checkpoint that yields the For Llama2, we follow the standard instruction **1212** performance on the synthetic validation set tuning procedure and use the pipeline provided by **1213** in CAR (Wang et al., 2023a) or HyKAS (Ma et al.,Wang et al., 2023a) or HyKAS (Ma et al.,, 2023a) or HyKAS (Ma et al., 2023a) or HyKAS (Ma et al.,) or HyKAS (Ma et al.,Ma et al.,, Chen et al. (2023). We train the model with a batch **1214** 2021) is selected as the final model.) is selected as the final model. During evalsize of 32, learning rate of 1e-5, and linear learning **1215** uating, we select the option that yields the lowest rate decay. We take the final checkpoint as our **1216** model to make prediction. **1217** We provide the prompt templates for each model The whole list of prompt templates that we use **1218** . is presented in Tab. 11. **1219** 

## **B.2 Implementation Details of Generative Commonsense Inference Models** 

## **C Additional Analysis** 

**1220** 

## **Differences from ParaCOMET and COMET-M** 

The training and evaluation of GPT2-based model is based on the paradigm defined in COMET (Bosselut et al., 2019). The input of onehop ATOMIC triples is serialized to “ _h r_ ” and the expected output is _t_ , where ( _h_ , _r_ , _t_ ) forms a triple in the CSKG. The input of 2p queries, ( _h_ , _r_ 1, _V_ ) and ( _V_ , _r_ 2, _V_ ?), are serialized as “ _h r_ 1 _r_ 2” and 

**1221** In ParaCOMET, the task involves providing a nar- **1222** rative as input, requiring the model to determine **1223** the commonsense causes or effects of a specific **1224** sentence within the context. To generate training **1225** data, a single-hop COMET model fine-tuned on **1226** ATOMIC is employed to create synthetic infer- **1227** 

16 

**1228** ences. These inferences are generated solely based **1229** on the target sentence and the desired relation, with- **1230** out accessing the whole context. The resulting one- **1231** hop synthetic inferences are then utilized as distant **1232** supervision signals during the fine-tuning process **1233** for ParaCOMET. **1234** COMET-M utilizes a context consisting of a sen- **1235** tence containing multiple events. Unlike from a **1236** sentence level, COMET-M focuses on generating **1237** commonsense inferences based on a specific event **1238** within the sentence. T his fine-grained approach **1239** enables more precise and detailed commonsense **1240** reasoning. 

47.2% for ip). This is also a result of the first **1279** reason, as the answers to the sampled queries are **1280** limited to nodes with high degrees. **1281** In all, despite that the query structure itself is **1282** more complicated, the reasoning difficulty is not **1283** that hard compared to other query types due to the **1284** above two reasons. **1285 Results of the Ablations** We present the results **1286** of the ablation study in Tab. 12. **1287** 

**1241** In contrast, our complex commonsense reason- **1242** ing benchmark introduces additional complexities **1243** compared to ParaCOMET and COMET-M. Besides **1244** the complex structures in the context that involves **1245** multiple events, the desired relation or question **1246** involves multi-hop reasoning as well. For instance, **1247** rather than focusing on the cause of a single sen- **1248** tence or event, COM[2] explores questions related **1249** to common causes, effects, attributions of multiple **1250** events, and two-hop inferences. This distinctive **1251** formulation sets our work apart and poses a greater **1252** challenge for LLMs to effectively reason and pro- **1253** vide accurate responses. 

**1254 Discussions on different query types** Accord- **1255** ing to the main experiments on the MCQA version **1256** of COM[2] in Tab. 1, there are some variance regard- **1257** ing the performance on different query types. A **1258** notable distinction is the performance of pi queries, **1259** which exhibits a significantly higher success rate **1260** compared to other query types, particularly ip **1261** queries, as both pi and ip involve a single free **1262** variable and both intersection and projection op- **1263** erations. We present two perspectives to explain **1264** this phenomenon. First, the limited availability **1265** of sampled pi queries restricts the diversity of the **1266** data. Out of all the queries sampled from the dev **1267** set of ATOMIC[20] 20[,][only][4k][are][pi][queries,][while] **1268** there are 12k ip queries and 598k 2i queries. This **1269** paucity of pi queries contributes to a lack of variety. **1270** Moreover, within these 4k pi queries, the number **1271** of unique answers is limited to 459, indicating a **1272** limited range of possible responses. As a result, **1273** models fine-tuned on ATOMIC can generate an- **1274** swers to pi queries with relative ease, given that **1275** most of them consist of nodes with high degrees. **1276** Second, the chances of the sampled answer is ac- **1277** tually the correct answer to pi queries (67.8%) is **1278** significantly higher than other query types (e.g., 

**Discussions on Further Applications of Complex 1288 Queries** Intuitively, 2i queries can represent vari- **1289** ous scenarios such as common attribution, common **1290** effect, common cause, and abduction (when one **1291** relation pertains to effects and the other relates to **1292** cause), depending on the types of relations involved **1293** in the query. Besides, complex logical queries, par- **1294** ticularly those involving intersection operations, **1295** are relevant to defeasible reasoning (Rudinger et al., **1296** 2020), where inferences can be weakened given **1297** new evidence. In the one-hop setting, tails are anno- **1298** tated in a context-free manner, considering only the **1299** most general cases. However, in intersection-based **1300** queries like 2i and 3i, additional anchor entities **1301** and relations act as specific constraints, narrowing **1302** down the inferences to a particular scope while **1303** disregarding other commonsense inferences in the **1304** context-free scenario. For instance, in the exam- **1305** ple from Fig. 1, other potential tails for (PersonX **1306** goes skydiving, xIntent) could include overcoming **1307** fear, seeking enjoyment, or achieving a personal **1308** milestone. Nevertheless, when constrained by an- **1309** other query (PersonX gets tired of it, xWant), the **1310** intentions related to fear, enjoyment, and fulfill- **1311** ment are weakened, and only the correct inference **1312** of “finding new things to do” remains. **1313** 

## **D Error Analysis** 

**1314** 

We present some error cases in Tab. 4. In general, **1315** a common error in both projection and intersec- **1316** tion queries is that the generated answer can be **1317** only the one-hop answer instead of the correct an- **1318** swer that is multi-hop. For example, in the 2p **1319** case, “get a new job” is a direct intention of some- **1320** one who updates his or her resume. However, the **1321** 2p query asks about the intention of the intention, **1322** which requires inducing the intention behind “get a **1323** new job”. In this sense, “to be financially indepen- **1324** dent” is more plausible inference. In the case of 2i **1325** queries, the error lies in the absence of inferential **1326** gaps between the context, where the generated an- **1327** 

17 

**1328 1329 1330 1331 1332** 

|**Model**|COM2<br>**R-L**<br>**CIDEr**<br>**BERT**|
|---|---|
|**Filter**<br>COM2-COMET<br>14.7<br>33.0<br>46.3<br>- w/o plau. flter<br>13.0<br>31.2<br>42.3<br>- w/o div. flter<br>14.4<br>32.5<br>45.8<br>- w/o both flter<br>12.5<br>30.3<br>40.1||
|**Query Types**<br>COMET (1p)<br>10.0<br>20.7<br>44.3<br>+ 2i<br>13.6<br>26.1<br>39.8<br>+ 2p<br>9.8<br>19.9<br>43.4<br>+ 2i, 3i, 2p<br>14.7<br>33.0<br>46.3||
|**Verbalization**<br>COM2-COMET<br>13.6<br>26.1<br>39.8<br>COM2-COMET (V)<br>14.3<br>27.1<br>43.4<br>COM2-Llama<br>35.7<br>107.2<br>61.3<br>COM2-Llama (V)<br>36.2<br>105.4<br>61.4||
|**Model**<br>ParaCOMET<br>**R-L**<br>**CIDEr**<br>**BERT**||
|**Verbalization**<br>COM2-COMET<br>13.8<br>22.1<br>53.7<br>COM2-COMET (V)<br>14.0<br>23.2<br>54.0<br>COM2-Llama<br>14.6<br>22.1<br>55.3<br>COM2-Llama (V)<br>14.8<br>23.6<br>55.5||



Table 12: Ablation studies on filters, type of queries, and using ChatGPT for verbalizing queries (denoted as V). 

swers become paraphrases of the events rather than being the result by any anchor entity. In the case of ip, a common error for one-hop COMET is the generation of “None” for complex cases, indicating a deficiency in multi-hop reasoning capabilities. 

18 

|Type|Context|Question|COMET|COM2-COMET|
|---|---|---|---|---|
||||||
|2p|Ezra updates Ezra’s resume (V1)|What event or state is the intention of Ezra<br>beforethe intention of Ezra beforeV1?|get a new job✗<br>(one-hop correct)|be fnancially indepen-<br>dent✓|
||||||
|2i-<br>neg|Every day, Benjamin goes to work diligently<br>(V1), never missing a day. They are dedicated<br>and committed to their job. In particular,Ben-<br>jamin doesn’t work hard on it (V2)and instead<br>takes a more relaxed approach, focusing on<br>maintaining a healthy work-life balance.|What event or state is boththe effect on Ben-<br>jamin after Benjamin go to work every day<br>(V1)and alsowhat hinderedBenjamin work<br>hard on it (V2)?|Benjamin<br>is<br>sick<br>_?_<br>(Not perfect as Benjamin<br>is trying to keep a work-<br>life balance instead of<br>having a sick leave)|Benjamin gets tired from<br>working hard✓|
||||||
|2i|Chloe is known for being hardworking (V1)<br>and dedicated. As a result,Chloe leads a good<br>life (V2).|What event or state is boththe effect on Chloe<br>afterChloe is hardworking (V1)and alsowhat<br>Chloe wants to do after Chloe leads a good<br>life (V2)?|to have a good life _?_<br>(No inferential gap)|to have success in life_?_<br>(No inferential gap)|
||||||
|ip|After looking for a new car (V1), Lydia is<br>driving to school (V2).|What event or state iswhat Lydia needed to<br>do before the event that is both what Lydia<br>wants to do afterLydia is looking for a new<br>car (V1), and also what Lydia needed to do<br>beforeLydia is driving to school (V2)?|None✗|take a car for test drive✓|



Table 13: Error analysis of generated inferences on the evaluation set of COM[2] . We present the generations of COMET-Llama-7b and COM[2] -Llama-7b fine-tuned on all queries. 

19 

