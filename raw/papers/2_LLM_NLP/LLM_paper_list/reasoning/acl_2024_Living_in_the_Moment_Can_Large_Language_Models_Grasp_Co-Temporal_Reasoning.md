**001** Temporal reasoning is fundamental for large **002** language models (LLMs) to comprehend the **003** world. Current temporal reasoning datasets **004** are limited to questions about single or iso- **005** lated events, falling short in mirroring the re- **006** alistic temporal characteristics involving con- **007** current nature and intricate temporal intercon- **008** nections. In this paper, we introduce COTEMP- **009** QA, a comprehensive co-temporal Question **010** Answering (QA) benchmark containing four **011** co-temporal scenarios (Equal, Overlap, Dur- **012** ing, Mix) with 4,749 samples for evaluating the **013** co-temporal comprehension and reasoning abil- **014** ities of LLMs. Our extensive experiments re- **015** veal a significant gap between the performance **016** of current LLMs and human-level reasoning **017** on COTEMPQA tasks. Even when enhanced **018** with Chain of Thought (CoT) methodologies, **019** models consistently struggle with our task. In **020** our preliminary exploration, we discovered that **021** mathematical reasoning plays a significant role **022** in handling co-temporal events and proposed **023** a strategy to boost LLMs’ co-temporal rea- **024** soning from a mathematical perspective. We **025** hope that our COTEMPQA datasets will en- **026** courage further advancements in improving the **027** co-temporal reasoning capabilities of LLMs. 

**028** 

## **Living in the Moment: Can Large Language Models Grasp Co-Temporal Reasoning?** 

## **Anonymous ACL submission** 

## **Abstract** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Living_in_the_Moment_Can_Large_Language_Models_Grasp_Co-Temporal_Reasoning_images/acl_2024_Living_in_the_Moment_Can_Large_Language_Models_Grasp_Co-Temporal_Reasoning.pdf-0001-05.png)


**----- Start of picture text -----**<br>
was the chair  of Tesla<br>worked for  OpenAI<br>2004 … 2015 2018 2019 2022 2023<br>works for  OpenAI<br>**----- End of picture text -----**<br>


Figure 1: Understanding concurrent is crucial for us to understand how individuals navigate and influence diverse aspects of real-world scenarios. For instance, when _Elon Musk_ was the chair of _Tesla_ , he also worked for _OpenAI_ . Concurrently, _Sam Altman_ was working for _OpenAI_ , too. Their simultaneous experiences greatly influenced subsequent decision-making at _OpenAI_ . 

## **1 Introduction** 

**029** Recent advanced Large Language Models (LLMs) **030** like GPT-4 (OpenAI, 2023) have shown impressive **031** capabilities in understanding, generating, and rea- **032** soning about natural language (Wei et al., 2022a; **033** Zhao et al., 2023; Chang et al., 2023). Despite their **034** advancements, these models fall short in master- **035** ing temporal reasoning (Chu et al., 2023), which is **036** fundamental for humans to comprehend the world **037** and distinguish daily events (Chen et al., 2021; **038** Tan et al., 2023), requiring a complex integration **039** of capabilities, involving implicit arithmetic calcu- **040** lations (Zhu et al., 2023a), understanding logical **041** implications (Wei et al., 2022c), and leveraging **042** extensive world knowledge (Chu et al., 2023). 

Current studies in temporal reasoning mainly fo- **043** cus on time-sensitive question-answering (TSQA). **044** Chen et al. (2021) first introduced the TIMEQA **045** dataset, constructing time-evolving facts for a given **046** subject and formulating questions based on the **047** specific timestamp within the evolutionary facts. **048** TEMPLAMA (Dhingra et al., 2022) extracted **049** structured facts from the Wikidata Knowledge **050** Base (Vrandeˇci´c and Krötzsch, 2014) for closed- **051** book TSQA. Furthermore, TEMPREASON (Tan **052** et al., 2023) translated explicit temporal expres- **053** sions into the implicit event information within **054** questions, offering a more comprehensive evalu- **055** ation framework of TSQA. Given the fact _“Elon_ **056** _musk held the position of Tesla’s chairman from_ **057** _2004 to 2018”_ , the models are tasked with accu- **058** rately interpreting and responding to time spec- **059** ifiers in the questions, i.e., _“Which position did_ **060** _Elon Musk hold in 2005?”_ in TIMEQA (Chen **061** et al., 2021) or _“Which position did Elon Musk_ **062** _held before he worked for OpenAI?”_ in TEMP- **063** REASON (Tan et al., 2023). **064** The datasets mentioned above provide a straight- **065** forward way to evaluate LLMs’ capabilities in tem- **066** 

1 

**067** poral reasoning. However, as LLMs evolve, there is **068** an urgent need to evaluate their proficiency in more **069** realistic scenarios. As shown in Figure 1, the reality 1, the reality, the reality **070** might present a more intricate and multifaceted na- **071** ture, involving concurrent events and complex tem- **072** poral interconnections over time (UzZaman et al.,UzZaman et al.,, **073** 2012).). Current datasets mainly question single or **074** isolated events and might not fully reflect the real- **075** istic temporal characteristics. Therefore, we create **076** the Co-Temporal QA (COTEMPQA)OTEMPQA)TEMPQA)EMPQA)QA) datasets to **077** complement existing corpora by focusing on the **078** concurrent nature of time and co-temporal relations **079** in real-world situations. **080** Experiments conducted on both closed-book and **081** open-book QA settings across 14 large language **082** models reveal that even advanced models GPT-4 **083** is well below a satisfactory co-temporal reasoning **084** performance. Specifically, GPT-4 achieves an over- **085** all score of 54.7, and the best open-source LLM **086** is 30.1, which significantly falls behind the human **087** performance of 92.8. We also observe that the rep- **088** resentative reasoning enhancement strategies, e.g., **089** Chain-of-Thought (CoT) (Wei et al., 2022b), fail to **090** consistently improve and even reduce the temporal **091** reasoning capabilities of LLMs in some scenarios. **092** Throughout the investigation on our COTEMP- **093** QA, we observed that mathematical reasoning **094** plays a crucial role in handling co-temporal events. **095** Building on this insight, we propose a simple but ef- **096** fective MATH-REASONING CoT (MR-COT) strat- **097** egy to boost the co-temporal reasoning capability **098** of LLMs, achieving a remarkable 10.8 point im- 

|**Datasets**|**Question**|**Answer**|
|---|---|---|
|**TIMEQA**(2021)|Which school did Sam Altman attended in 2005?<br>Which position did Elon Musk hold in 2005?|Stanford University<br>chairman of Tesla|
|**TEMPLAMA**(2022)|In 2005, Sam Altman attended _X_.<br>In 2005, Elon Musk hold the position of _X_.|Stanford University<br>chairman of Tesla|
|**TEMPREASON**(2023)|Which school did Sam Altman attend before he held the position of president of Y Combinator?<br>Which position did Elon Musk held before he worked for OpenAI?|Standford University<br>chairman of Tesla|
|**COTEMPQA (ours)**|When Elon Musk was working for OpenAI,where did he work for within the same time interval? (**Overlap**)<br>While Elon Musk was working for OpenAI,where did Sam Altman work for concurrently? (**During**)|Tesla, SpaceX<br>OpenAI|



Table 1: Example questions of prior TSQA datasets and our COTEMPQA datasets. Our question is divided into two components: the condition _C_ is marked in blue, and the inquiry _A_ is highlighted in red. 

|**Interpretation**|**Relation**|
|---|---|
|_•_——_x_——_•_<br>_◦_——_y_——_◦_|_x is equal to y_|
|_•_—_x_—_•_<br>_◦_—_y_—_◦_|_x overlaps with y_|
|_•_—_x_—_•_<br>_◦_——_y_——_◦_|_x during y_|



|**Mode**|**Questions**|**Subjects**|**#Facts**|**#Answers**|
|---|---|---|---|---|
|**Equal**|436|401|11.65|1.17|
|**Overlap**|653|591|14.51|1.23|
|**During**|3,096|2,161|15.05|1.33|
|**Mix**|563|434|12.54|2.27|
|**Total**|4,748|3,587|14.45|1.41|



Table 2: Interpretation of three co-temporal relations. 

poral reasoning. However, as LLMs evolve, there is an urgent need to evaluate their proficiency in more realistic scenarios. As shown in Figure 1, the reality 1, the reality, the reality might present a more intricate and multifaceted nature, involving concurrent events and complex temporal interconnections over time (UzZaman et al.,UzZaman et al.,, 2012).). Current datasets mainly question single or isolated events and might not fully reflect the realistic temporal characteristics. Therefore, we create the Co-Temporal QA (COTEMPQA)OTEMPQA)TEMPQA)EMPQA)QA) datasets to complement existing corpora by focusing on the concurrent nature of time and co-temporal relations in real-world situations. 

Table 3: Statistics of our datasets. #Facts and #Answers represent the average number of facts and answers within the subject and question, respectively. 

provement over existing baselines. However, it is **099** important to note that there remains a nonnegligi- **100** ble gap between the performance of our proposed **101** MR-COT and human-level reasoning in handling **102** complex, concurrent temporal relations. We hope **103** our research could inspire more great works to im- **104** prove the co-temporal ability of LLMs. **105** 

## **2 The COTEMPQA Datasets** 

**106 2.1 The Taxonomy of Co-temporal Relations 107** Co-temporal relations are fundamental to under- **108** standing how events interconnect in time. These **109** relationships highlight when facts or events hap- **110** pen simultaneously, which can be categorized into **111** three distinct types (Pustejovsky et al., 2003), as **112** shown in Table 2. Each of them represents a unique **113** manner, whether events coincide with or overlap **114** with each other in the temporal aspect. We divide **115** these relations into four different scenarios below: **116** 

- **Equal:** Facts occur simultaneously, represent- **117** ing a strict co-temporal relationship. This is the **118** simplest form of co-temporality but is essential **119** for understanding facts that happen concurrently **120** without any duration differences. **121** 

- • **Overlap:** One fact is entirely contained within **122** the timeline of another, reflecting a more complex **123** interaction of timelines. **124** 

- • **During:** Facts that partially coincide in time. **125** This scenario is more common in real-world set- **126** 

2 

**134** 

**150** 

**151 152 153 154 155 156** 

**Algorithm 1** Identifying Co-temporal Facts 

|**Alg**|**orithm 1**IdentifyingCo-temporal Facts|
|---|---|
|1:|**Input:** Set of facts_F_, each fact as(_s, r, o, ts, te_)|
|2:|**Output:** Set of co-temporal facts with their minimum|
||temporal units|
|3:|**function**MINMAXTIME(_fi_,_fj_)|
|4:|(_si, ri, oi, tsi, tei_)_←fi_|
|5:|(_sj, rj, oj, tsj, tej_)_←fj_|
|6:|_start ←_max(_tsi_,_tsj_)|
|7:|_end ←_min(_tei_,_tej_)|
|8:|_T_min _←_(_start_,_end_)|
|9:|**if**_start ≤end_**then return**_T_min|
|10:|**else return**None|
|11:|**end if**|
|12:|**end function**|
|13:|_R ←_empty set<br>_▷R_is the set of co-temporal facts|
|14:|**for**each_fi_ in_F_ **do**|
|15:|**for**each_fj_ in_F_ where_fi _=_fj_ **do**|
|16:|_T_min _←_MINMAXTIME(_fi_,_fj_)|
|17:|**if**_T_min is not None**then**|
|18:|_R ←R ∪{_(_fi, fj, T_min)_}_|
|19:|**end if**|
|20:|**end for**|
|21:|**end for**|
|22:|**return**_R_|



tings, where facts often intersect for a part of their duration without overlapping. 

- **127** tings, a part **128** their duration without overlapping. **129** • **Mix:** A combination of the three types above. **130** This category is particularly challenging as it **131** involves the complexity and variability of real- **132** world temporal relationships, necessitating a **133** comprehensive level of co-temporal reasoning. 

## **2.2 Structuring Temporal Facts** 

**135** We utilize the Wikidata (Vrandeˇci´c and Krötzsch, **136** 2014) dump of September 20, 2023 as our knowl- **137** edge source for extracting time-dependent facts. **138** Following Dhingra et al. (2022) and Tan et al. **139** (2023), we focus on nine time-sensitive entity re- **140** lations and keep a maximum of 2,000 subjects for **141** each relation type. To structure the information, **142** we transform the knowledge triples and qualifiers **143** into a quintuplet format of ( _s, r, o, ts, te_ ), where **144** _s_ is the subject, _r_ is the relation, _o_ is the object, **145** _ts_ and _te_ are the start time and end time. We **146** group all the temporal facts by subject, denoted **147** as _S_ = _{_ ( _s, ri, oi, tsi, tei_ ) _|i ∈_ 1 _. . . N }_ , where _N_ **148** is the number of facts within a group. We keep the **149** groups that contain three or more temporal facts. 

## **2.3 Extracting Co-temporal Facts** 

Building on our approach to structuring timedependent facts from Wikidata, we compare the timestamps of different facts to find overlaps. Given a fact _fi_ , we distinguish co-temporal fact _fj_ in five scenarios based on the combinations where either the subject, relation, or object changes 

while the other elements remain constant or change, **157** i.e., ( _S, R, O_ ), ( _S, R, O_ ), ( _S, R, O_ ), ( _S, R, O_ ), **158** ( _S, R, O_ ). An overline indicates a change in a spe- **159** cific element when comparing two related facts. **160** For instance, ( _S, R, O_ ) represents the scenario **161** where the subject and relationship remain the same. **162** Given the fact ( _Elon Musk, employer, SpaceX_ ), the **163** fact ( _Elon Musk, employer, OpenAI_ ) fits this pat- **164** tern. We exclude the scenarios ( _S, R, O_ ) and **165** ( _S, R, O_ ) since it is unrealistic for the same subject **166** and object to have different relationships, or for the **167** same object to have the same relationship with dif- **168** ferent subjects concurrently. The detailed illustra- **169** tions are shown in Appendix B. Taken ( _S, R, O_ ) as **170** an example, we detail the extraction of co-temporal **171** facts in the MINMAXTIME function (lines 3-11) **172** from Algorithm 1. This framework identifies the **173** complex co-temporal relations between events, al- **174** lowing for a more intuitive understanding of how **175** multiple events and states are interrelated in the **176** temporal dimension. **177 2.4 QA Pairs Construction 178** Upon identifying co-temporal facts ( _fi, fj, T_ min), **179** we construct the query _Q_ , which comprises two **180** fundamental components: a condition _C_ and an **181** inquiry _A_ . From the co-temporal facts, we select **182** the intersection fact as the condition fact _f_ cond and **183** the other as the query fact _f_ query. Given the facts **184** _f_ cond and _f_ query, we construct questions for the ob- **185** ject by manually-defined question templates, which **186** can be found in Table 8. Based on the temporal **187** relations identified through _T_ min, we categorize the **188** tasks into four distinct classes: **Equal** , **Overlap** , **189 During** , and **Mix** . We acknowledge that in real **190** life, multiple events can happen simultaneously, **191** and therefore, a single temporal question might **192** have multiple correct answers. To address this, we **193** aggregate all valid answers for query _Q_ to a set. **194** As detailed in Table 3, the average number of our **195** answers within the question is 1.42. **196** 

**3 The Performance of LLMs on 197 COTEMPQA 198** 

## **3.1 Experimental Setup** 

**Experimental Setup 199** We investigate the co-temporal reasoning abilities **200** of large language models within two problem set- **201** tings: (1) **C** losed- **B** ook **QA** ( **CBQA** ) is widely **202** recognized task format in time-sensitive QA re- **203** search (Dhingra et al., 2022; Liska et al., 2022; **204** Tan et al., 2023). In this setting, the language **205** 

3 

**224** 

**252** 

**253 254 255** 

**206** model is given only the question and tasked with **207** generating the answer without relying on exter- **208** nal natural language texts. The primary challenge **209** here involves the retention and temporal reason- **210** ing of knowledge pertinent to the question. (2) **211** In the **O** pen- **B** ook **QA** ( **OBQA** ) setting, we pro- **212** vide all the relevant temporal facts within the group **213** _S_ = _{_ ( _s, ri, oi, tsi, tei_ ) _|i ∈_ 1 _...N }_ in a structured **214** format directly into the prompt, which is in contrast **215** to previous studies (Chen et al., 2021; Wei et al., **216** 2023) that utilized Wikipedia as the knowledge **217** base. This process shifts the evaluation’s emphasis **218** towards the reasoning process itself, thereby mini- **219** mizing the influence of the model’s inherent factual **220** extraction capabilities on the outcomes (Tan et al., **221** 2023; Chu et al., 2023). Here, the language model **222** is tasked with deriving answers by considering the **223** time ranges associated with all potential answers. 

## **3.2 LLMs for Evaluation** 

**225** We perform comprehensive experiments on 14 **226** representative large language models including **227** (1) **ChatGPT** (Ouyang et al., 2022) ChatGPT is **228** a chat model aligned through Supervised Fine- **229** tuning (SFT) and Reinforcement Learning from **230** Human Feedback (RLHF). GPT-4 is an upgraded **231** version of ChatGPT with enhanced reasoning capa- **232** bilities, making it the most powerful LLM. Since **233** the model is constantly updated, we used the **234** gpt3.5-turbo-0613 and gpt4-0613 for **235** consistent evaluation. (2) **LLaMA2** (Touvron et al., **236** 2023) LLaMA2 is one of the most popular open- **237** source foundation models trained on 2T tokens **238** with efficient group query attention (Ainslie et al., **239** 2023). (3) **Code-LLaMA** (Roziere et al., 2023) **240** Code-LLaMA models is a code generation model **241** built on LLaMA2 and further trained on 500B to- **242** kens of code. (4) **WizardMath** (Luo et al., 2023a) **243** WizardMath is also built on LLaMA2 and further **244** trained on their proposed Reinforcement Learn- **245** ing from Evol-Instruct Feedback (RLEIF) (Xu **246** et al., 2023) to enhance the mathematical reason- **247** ing abilities of LLaMA2. (5) **WizardCoder** (Luo **248** et al., 2023b) WizardCoder, similar to WizardMath, **249** adapts the RLEIF method to the domain of code. **250** The implementation details of our experiments are **251** shown in Appendix A. 

## **3.3 Evaluation Metrics** 

Prior works followed the SQuAD benchmark’s evaluation protocol (Rajpurkar et al., 2016), using exact match (EM) and token-level F1 score. 

These metrics calculate the highest scores across **256** all references, tending to overestimate performance **257** in task settings involving questions with multiple **258** possible answers. Following Zhong et al. (2022), **259** we adopt a stricter **Acc.** score, where a prediction **260** is correct only if it aligns with all the gold answers **261** for a question. Additionally, we also evaluate our **262** methods by answer-level F1 score ( **F** 1), which is a **263** stricter metric compared to token-level F1 score. **264 3.4 Results and Analysis 265** The main results are shown in Table 4. We report **266** human performance to serve as an upper bound. **267** From the results, we can observe: **268 LLMs partially grasp co-temporal reasoning 269** Our analysis reveals that, despite GPT-4 exhibit- **270** ing the best performance among all LLMs, there is **271** still a considerable disparity compared to human **272** performance (54.7 vs. 92.8), indicating significant **273** potential for further improvement in co-temporal **274** reasoning. Meanwhile, there is a general trend in **275** improving co-temporal reasoning capabilities as **276** the size of the models’ parameters increases. We **277** also discover that models exhibit different reason- **278** ing capabilities in different co-temporal scenarios. **279** Take GPT-4 for further illustration, in the simple co- **280** temporal reasoning task, i.e., the **Equal** scenario, **281** GPT-4 demonstrates strong performance, achiev- **282** ing a 92.7 score overall. However, its performance **283** significantly declines in more complex scenarios. **284** Specifically, in the **Overlap** category, GPT-4’s ac- **285** curacy falls to 59.4, decreasing further to 50.1 in **286** the **During** category. In the most challenging cat- **287** egory, **Mix** , which combines various temporal re- **288** lations, GPT-4’s performance drops to 45.0. This **289** phenomenon indicates that existing LLMs can ef- **290** fectively process and reason about straightforward **291** concurrent events. However, they encounter diffi- **292** culties in more complex tasks that require a deeper **293** understanding and comprehension of co-temporal **294** reasoning. **295 CBQA is more challenging for LLMs** LLMs **296** exhibit significantly weaker performance in the **297 CBQA** compared to the **OBQA** , as reflected in the **298** GPT-4’s performance (14.5 vs. 54.7). Interestingly, **299** GPT-4 is outperformed by GPT-3.5 in **CBQA** . Our **300** error analysis indicates that GPT-4 often responds **301** with _“uncertain"_ when unsure, unlike GPT-3.5, **302** which tends to provide direct answers. This dis- **303** covery is also found in previous works (OpenAI, **304** 2023; Wei et al., 2023). This characteristic hin- **305** 

4 

|**Model**|**Equal**|**Overlap**|**During**|**Mix**|**Overall**|
|---|---|---|---|---|---|
||**Acc.**<br>**F**1<br>**Avg.**|**Acc.**<br>**F**1<br>**Avg.**|**Acc.**<br>**F**1<br>**Avg.**|**Acc.**<br>**F**1<br>**Avg.**||
||**The Closed Book Question Answer (CBQA) setting**|||||
|GPT-3.5-TURBO<br>GPT-4|**13.80**<br>**14.80**<br>**14.30**<br>11.30<br>**14.30**<br>12.80<br>**15.00**<br>**22.90**<br>**18.90**<br>11.20<br>12.30<br>11.80<br>**11.50**<br>14.00<br>**12.70**<br>14.80<br>18.50<br>16.70|||**0.00**<br>**15.50**<br>**7.70**<br>**0.00**<br>13.60<br>6.80|**16.30**<br>14.50|
||**The Open Book Question Answer (OBQA) setting**|||||
|GPT-3.5-TURBO<br>GPT-4|59.40<br>66.30<br>62.80<br>40.10<br>48.50<br>44.30<br>31.50<br>42.90<br>37.20<br>**91.10**<br>**94.30**<br>**92.70**<br>**55.30**<br>**63.50**<br>**59.40**<br>**44.30**<br>**55.80**<br>**50.10**|||0.70<br>46.10<br>23.40<br>**23.40**<br>**66.50**<br>**45.00**|38.90<br>**54.70**|
|CODELLAMA-7B<br>WIZARDCODER-7B<br>LLAMA-7B<br>WIZARDMATH-7B|6.40<br>**27.70**<br>**17.00**<br>3.10<br>14.60<br>8.80<br>3.10<br>15.80<br>9.50<br>9.20<br>21.10<br>15.10<br>4.70<br>14.80<br>9.80<br>6.30<br>15.90<br>11.10<br>4.10<br>18.90<br>11.50<br>4.70<br>**19.50**<br>12.10<br>4.50<br>19.50<br>12.00<br>**12.4**0<br>16.50<br>14.40<br>**9.2**0<br>15.20<br>**12.20**<br>**11.6**0<br>**20.5**0<br>**16.00**|||**2.00**<br>**24.10**<br>**13.00**<br>0.50<br>20.40<br>10.50<br>0.20<br>23.80<br>12.00<br>0.40<br>22.00<br>11.20|10.50<br>11.20<br>12.00<br>**14.80**|
|CODELLAMA-13B<br>WIZARDCODER-13B<br>LLAMA-13B<br>WIZARDMATH-13B|7.60<br>28.30<br>18.00<br>4.10<br>17.00<br>10.60<br>3.30<br>19.30<br>11.30<br>8.30<br>16.60<br>12.40<br>7.00<br>17.80<br>12.40<br>9.50<br>19.70<br>**14.60**<br>11.20<br>**31.20**<br>21.20<br>5.80<br>**21.60**<br>**13.70**<br>5.00<br>**20.60**<br>12.80<br>**23.9**0<br>29.00<br>**26.4**0<br>**10.9**0<br>15.10<br>13.00<br>**11.7**0<br>17.10<br>14.40|||**3.20**<br>**28.60**<br>**15.90**<br>1.10<br>24.10<br>12.60<br>1.10<br>26.90<br>14.00<br>0.00<br>13.20<br>6.60|12.40<br>13.90<br>13.80<br>**14.40**|
|CODELLAMA-34B<br>WIZARDCODER-34B<br>LLAMA-70B<br>WIZARDMATH-70B|16.10<br>46.50<br>31.30<br>9.80<br>27.00<br>18.40<br>8.10<br>28.40<br>18.30<br>19.50<br>26.30<br>22.90<br>15.20<br>22.40<br>18.80<br>15.90<br>23.90<br>19.90<br>11.90<br>41.70<br>26.80<br>10.00<br>32.50<br>21.20<br>9.40<br>33.50<br>21.40<br>**36.70**<br>**46.80**<br>**41.80**<br>**23.60**<br>**33.70**<br>**28.60**<br>**25.50**<br>**37.10**<br>**31.30**|||4.40<br>40.30<br>22.40<br>0.90<br>25.90<br>13.40<br>**5.2 0**<br>**42.50**<br>**23.80**<br>0.40<br>32.90<br>16.60|20.00<br>19.20<br>22.20<br>**30.10**|
|HUMAN|97.00<br>98.30<br>97.70<br>91.10<br>93.50<br>92.30<br>82.00<br>87.00<br>84.50|||88.00<br>96.20<br>92.10|92.80|



Table 4: Experimental results of each model in the **CBQA** and **OBQA** settings of our proposed COTEMPQA. Notably, we only report the performance of GPT-3.5 and GPT-4 in **CBQA** setting as the open-source LLMs are almost negligible here, and closed-book human evaluations largely depend on individual knowledge, leading to significant variations between different individuals. The best performance of each model is **bold** . 

**306** ders GPT-4’s effectiveness in co-temporal **CBQA** , **307** where precise answers are needed. While construct- **308** ing our datasets, we concentrated on the top 2,000 **309** subjects for each relationship type. These subjects **310** are typically well-covered in pre-training stages, **311** as Wikipedia is a significant part of their training **312** data (Touvron et al., 2023). Despite this prior ex- **313** posure, LLMs’ reduced capability in **CBQA** under- **314** scores a challenging yet crucial situation, where **315** LLMs frequently must respond to contemporane- **316** ous questions without supplementary context in **317** the real-world application. Their ability to deliver **318** precise answers relying solely on their pre-trained **319** knowledge base is essential. This result underlines **320** the need to enhance the co-temporal reasoning abil- **321** ities of LLMs, empowering them to comprehend **322** and reason about concurrent events. 

**323 Different aspects of capability benefit co324 temporal reasoning differently** We also explore **325** the effects of augmenting the model’s additional **326** abilities using SFT on co-temporal reasoning. We **327** observe that models specialized in mathematical **328** reasoning, such as WizardMath-70B, show signifi- **329** cant improvements in co-temporal reasoning, scor- **330** ing 30.1, compared to the foundational LLaMA- **331** 70B model’s 22.2, as detailed in Table 4. This **332** enhanced capability suggests a strong correlation **333** between the skills utilized in mathematical reason- **334** ing and those required for understanding and in- **335** terpreting complex temporal relationships. How- 

ever, LLMs with capabilities in code reasoning **336** do not show consistent improvement in their per- **337** formance. While these models are good at tasks **338** involving sequential logic, their performance in co- **339** temporal reasoning indicates that these skills may **340** not directly translate to a robust understanding of **341** simultaneous temporal relations. **342** 

## **3.5 Data Analysis** 

**343** 

In Section 2.3, we categorize co-temporal facts into **344** five scenarios. Building on this classification, this **345** section delves into investigating how various types **346** of fact elements influence LLMs’ ability to perform **347** co-temporal reasoning. To ensure fairness in our **348** experiments, we excluded questions with multiple **349** answers and standardized the number of questions **350** across all co-temporal relations. Figure 2 illustrates **351** GPT-4’s performance with various element types. **352** Additional results concerning different LLMs are **353** presented in Table 7, and results consistently align **354** with the findings shown below: **355** 

**The influence of triple element types** As ob- **356** served in Figure 2a, the complexity of co-temporal **357** reasoning for models increases with the num- **358** ber of changing elements. Among the scenarios, **359** ( _S, R, O_ ), ( _S, R, O_ ) are particularly challenging **360** compared to others. It indicates that LLMs en- **361** counter significant challenges when dealing with **362** scenarios of high complexity, where multiple ele- **363** ments undergo simultaneous changes. The analysis **364** 

5 

**365 366 367** 

**368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 387** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Living_in_the_Moment_Can_Large_Language_Models_Grasp_Co-Temporal_Reasoning_images/acl_2024_Living_in_the_Moment_Can_Large_Language_Models_Grasp_Co-Temporal_Reasoning.pdf-0006-02.png)


**----- Start of picture text -----**<br>
100 Acc. 100 Acc.<br>80 74.7 68.0 [70.8] 73.3 F1 80 66.8 [69.6] F1<br>60 57.0  58.0  60 58.5 [62.1] 59.7 60.3<br>48.8<br>39.8 44.3 45.2<br>40 40<br>29.9<br>23.5<br>20 20<br>0 0<br>(𝑆, 𝑅, 𝑂) (𝑆, 𝑅, 𝑂) (𝑆, 𝑅, 𝑂) (𝑆, 𝑅, 𝑂) (𝑆, 𝑅, 𝑂) Intra-Subject 𝑆 Inter-Subject 𝑆 Intra-Relation 𝑅 Inter-Relation 𝑅<br>(a) Triple Element Types (b) Triplet Element Types<br>**----- End of picture text -----**<br>


Figure 2: Performance of GPT-4 under different co-temporal element types in the **OBQA** setting of our COTEMPQA. The overline indicates we changed the element in fact to others. The best performance of each element type is **bold** . 

|**Model**|**Equal**|**Overlap**|**During**|**Mix**<br>**Overall**|**Mix**<br>**Overall**|
|---|---|---|---|---|---|
||**Acc.**<br>**F**1<br>**Avg.**|**Acc.**<br>**F**1<br>**Avg.**|**Acc.**<br>**F**1<br>**Avg.**|**Acc.**<br>**F**1<br>**Avg.**||
||**The Closed Book Question Answer (CBQA) setting**|||||
|GPT-4<br>+ COT<br>+ FS<br>+ FS&COT<br>+ FS&MR-COT|11.20<br>12.30<br>11.80<br>11.50<br>14.00<br>12.70<br>14.80<br>18.50<br>16.70<br>12.20<br>14.40<br>13.30<br>8.40<br>12.50<br>10.50<br>12.10<br>18.60<br>15.30<br>26.40<br>29.60<br>28.00<br>17.60<br>21.20<br>19.40<br>20.60<br>26.70<br>23.70<br>**32.10**<br>**35.20**<br>**33.60**<br>**19.90**<br>25.70<br>22.80<br>**23.20**<br>29.50<br>26.40<br>24.80<br>30.60<br>27.70<br>16.70<br>**29.90**<br>**23.30**<br>20.80<br>**35.70**<br>**28.20**|||0.00<br>13.60<br>6.80<br>1.60<br>14.30<br>8.00<br>0.00<br>21.70<br>10.90<br>0.50<br>25.60<br>13.10<br>**3.90**<br>**31.70**<br>**17.80**|14.50<br>13.60<br>22.00<br>25.00<br>**26.30**|
||**The Open Book Question Answer (OBQA) setting**|||||
|GPT-4<br>+ COT<br>+ FS<br>+ FS&COT<br>+ FS&MR-COT|91.10<br>94.30<br>92.70<br>55.30<br>63.50<br>59.40<br>44.30<br>55.80<br>50.10<br>87.80<br>90.00<br>88.90<br>46.20<br>58.70<br>52.50<br>43.50<br>57.00<br>50.20<br>87.40<br>91.40<br>89.40<br>62.60<br>72.50<br>67.60<br>55.90<br>68.60<br>62.20<br>**96.80**<br>**97.30**<br>**97.10**<br>61.30<br>71.40<br>66.30<br>55.70<br>69.40<br>62.50<br>95.90<br>97.20<br>96.50<br>**77.90**<br>**83.90**<br>**80.90**<br>**69.00**<br>**78.80**<br>**73.90**|||23.40<br>66.50<br>45.00<br>29.50<br>71.60<br>50.50<br>30.60<br>71.90<br>51.20<br>32.10<br>73.20<br>52.70<br>**50.30**<br>**82.20**<br>**66.20**|54.70<br>54.10<br>64.20<br>65.00<br>**75.80**|



Table 5: Performance of GPT-4 under Zero-shot CoT (COT) prompting, Few-shot (FS) prompting, Few-shot CoT (FS&COT) prompting and our proposed Few-Shot Mr-CoT (FS&MR-COT) prompting in **CBQA** and **OBQA** . 

below further investigates which elements present the most significant challenges to co-temporal reasoning capabilities. 

**The influence of triplet element types** In the left part of Figure 2b, we observe a notable decline in the model’s performance, i.e., 22.5 point decrease in **Acc.** and 24.4 in **F** 1 when it engages in reasoning involving multiple subjects. These findings highlight models’ inherent difficulty when processing information from multiple concurrent subjects. While the reasoning process for handling multiple subjects shares similarities with single-subject scenarios, real-world situations are inherently more complex and variable. The model is required to integrate information across different subjects and understand complex relationships that extend beyond a single domain or context. On the other hand, in the right part of Figure 2b, we discover that the difference in the fact relation does not significantly impact the abilities of LLMs in co-temporal reasoning. This is attributed to the fact that relationship changes are not as complex as those involving multiple subjects in real-world scenarios (Huang and 

Chang, 2023), making them less challenging for the models’ capabilities. 

**388 389** 

## **4 Making Language Models Better COTEMPQA Responders** 

**390 391** 

Previous research has demonstrated that the Chain- **392** of-Thought (CoT) enables models to process com- **393** plex reasoning tasks, such as mathematical and **394** logical reasoning, in a step-by-step manner (Wei **395** et al., 2022b). Motivated by this, we explore the **396** application of CoT prompting to improve the capa- **397** bilities of co-temporal reasoning in this section. **398** 

## **4.1 Math-reasoning CoT (MR-COT)R-COT)-COT)OT)T)** 

**Math-reasoning CoT (MR-COT)R-COT)-COT)OT)T) 399** As indicated in Table 4, our research uncovers **400** a mathematically oriented reasoning framework **401** that can enrich the LLMs’ understanding and han- **402** dling of co-temporal reasoning tasks. Specifi- **403** cally, the WizardMath-70B model scores 30.1 over- **404** all, notably higher than the foundational LLaMA- **405** 70B model’s score of 22.2. In light of this find- **406** ing, we propose a MATH-REASONING (MR-COT) **407** instruction-based prompting, which can be used **408** 

6 

**409 410 411 412 413 414 415 416 417 418** 

**419** 

**420 421 422 423 424 425 426 427 428 429 430 431** 

|**Question**|While Klaus Fuchs was working for Atomic Energy Research Establishment,<br>which employer did Alexander Oppenheim work for duringthe identical timeperiod?|
|---|---|
||Alexander Oppenheim works for National University of Singapore from 1949 to 1965.|
||Alexander Oppenheim works for University of Malaya from 1949 to 1965.|
|**Context**|Klaus Fuchs works for Atomic Energy Research Establishment from 1946 to 1950.|
||Klaus Fuchs works for University of Birmingham from 1941 to 1944.|
||......|
|**Gold Answer**|**National University of Singapore and University of Malaya**|
|**GPT-4**|Answer: **National University of Singapore**|
||Reasoning: Klaus Fuchs worked for Atomic Energy Research Establishment from 1946 to 1950.|
|**+ CoT**|Currently, Alexander Oppenheim worked for University of Ghana from 1968 to 1973.<br>However, this does not overlap with the time Klaus Fuchs was at the Atomic Energy Research Establishment.|
||Answer: **no information**|
|**+ Fs**|Answer: **University of Malaya**|
||Reasoning: Klaus Fuchs worked for Atomic Energy Research Establishment from 1946 to 1950.|
|**+ Fs&CoT**|During this same period, Alexander Oppenheim worked for National University of Singapore,<br>as his tenure with the university spanned from 1949 to 1965.|
||Answer: **National University of Singapore**|
||The datetime can be formed as ((1946, None, None), (1950, None, None)).|
||The content provided and related to the question can be structured as:|
||(University of Malaya, (1949, None, None), (1965, None, None))|
|**+ Fs&Mr-CoT**|(National University of Singapore, (1949, None, None), (1965, None, None))<br>Given the ((1946, None, None), (1950, None, None)), compared with all contents related,|
||we fnd that[(1949_,_None_,_None)_−_(1965_,_None_,_None)]_∩_[(1946_,_None_,_None)_−_(1950_,_None_,_None)] =_∅_|
||Reasoning: Klaus Fuchs worked for Atomic Energy Research Establishment from 1946 to 1950.|
||Answer: **National University of Singapore and University of Malaya**|



Table 6: Example inputs and outputs of GPT-4 with Zero-shot CoT (COT) prompting, Few-shot (FS) prompting, Few-shot CoT (FS&COT) prompting and Few-Shot Mr-CoT (FS&MR-COT). Answers highlighted in blue are correct, whereas thoses marked in red are incorrect. 

together with in-context learning and chain-ofthought prompting. As demonstrated in the bottom of Table 6, our framework consists of three steps: (1) establish the key datetime, (2) structure the relevant timeline, and (3) mathematically identify the overlap. This prompt aims to guide the LLMs towards approaching temporal reasoning problems through a mathematical perspective, aligning their problem-solving processes more closely with mathematical logic and principles. 

## **4.2 Experimental Setup** 

We launch experiments under both zero-shot and few-shot settings. In the zero-shot CoT scenario, we use _Let’s think step by step_ (Kojima et al., 2022) after questions as the reasoning trigger. In contrast, the few-shot setting provides the model with several question-answer pairs as initial demonstrations. Specifically, for the few-shot CoT scenario, we manually create rationales for each task, which are used as demonstrations to guide the model in step-by-step reasoning. Further details on the instructions and demonstrations are available from Figure 3 to Figure 12 in Appendix C. 

## **4.3 Results and Analysis** 

**Results and Analysis 432** The results are presented in Table 5, and the output **433** of GPT-4 to a range of prompts under different **434** settings are shown in Table 6. From these tables, **435** we can discover the following insights: **436 Inconsistency in the impact of existing CoT 437 prompts on GPT-4** In the zero-shot scenario, im- **438** provements were inconsistent, with a notable 5.5 **439** performance increase in the **Mix** task and a 3.8 de- **440** crease in the **Equal** task under the **OBQA** setting. **441** This suggests that the impact of CoT prompts varies **442** significantly based on the task type. Moreover, **443** GPT-4 demonstrates an overall decline in perfor- **444** mance on both **CBQA** and **OBQA** when comple- **445** mented with CoT. In the few-shot scenario, while **446** overall improvements exist due to CoT prompts, **447** these are relatively modest, amounting to an av- **448** erage performance enhancement of 0.8 in **OBQA** . **449** All results indicate that while existing CoT prompts **450** can be beneficial, their effectiveness is nuanced and **451** task-dependent. **452 Superiority of our proposed MR-COT** Our **453** method demonstrates significant superiority over **454** existing reasoning enhancement strategies. No- **455** 

7 

complexity of scenarios where multiple events co- **506** occur or interact over the same period. Our work **507** aims to fill this gap by being the first to explore the **508** concurrent nature of temporal events. **509** 

**473** 

**474** 

**456** tably, MR-COT significantly enhances perfor- **457** mance on the more challenging tasks, yielding im- **458** provements of 14.6, 11.4, and 13.5 on the tasks **459 Overlap** , **During** , and **Mix** , respectively in the **460 OBQA** setting. In the closed-book scenario, which **461** is typically more challenging to improve, our **462** method still achieves a 1.3 enhancement. How- **463** ever, it is observed that our method has a moderate **464** effect on the **Equal** setting. We hypothesize that **465** this is because this task is simple enough and does **466** not require the additional complexity of mathemati- **467** cal reasoning. In such cases, this added complexity **468** could be counterproductive. Despite these advance- **469** ments, there is still a considerable gap compared **470** to human-level reasoning, indicating the need for **471** more effective methods to improve the model’s **472** co-temporal reasoning abilities. 

## **5 Related Work** 

## **5.1 Temporal Reasoning Benchmarks** 

**475** Temporal reasoning in natural language processing **476** has seen significant advancements over the years. **477** Early benchmarks, such as TimeBank (PUSTE- **478** JOVSKY, 2003), and TempEval-3 (UzZaman et al., **479** 2012), lay the foundational work in this domain. **480** They primarily focused on understanding tem- **481** poral relationships between events in text, of- **482** fering a preliminary framework for analyzing **483** time in language models. However, recent years **484** have witnessed a significant surge in develop- **485** ing time-sensitive question-answering datasets. **486** These newer datasets, including MC-TACO (Zhou **487** et al., 2019), SituatedQA (Zhang and Choi, 2021), **488** TimeQA (Chen et al., 2021), TempLAMA (Dhin- **489** gra et al., 2022), StreamingQA (Liska et al., **490** 2022), RealtimeQA (Kasai et al., 2022), TempREA- **491** SON (Tan et al., 2023) and Menatqa (Wei et al., **492** 2023), represent a more nuanced approach to tem- **493** poral reasoning. These datasets challenge mod- **494** els to answer questions grounded in specific times **495** or events, thereby testing the models’ ability to **496** comprehend and reason with temporal informa- **497** tion more dynamically. The introduction of bench- **498** marks such as TRAM (Wang and Zhao, 2023) and **499** TimeBench (Chu et al., 2023) marks a significant **500** advancement, providing crucial platforms for tem- **501** poral reasoning research. Despite these advance- **502** ments, there has been a noticeable gap in exploring **503** the concurrent nature of temporal events. Previous **504** research has primarily focused on individual events **505** or sequences of events in isolation, overlooking the 

## **5.2 Temporal Reasoning over LLMs** 

**510** 

To enhance the temporal reasoning capabilities **511** of language models, previous methods either rely **512** heavily on knowledge graphs to rank entities that **513** satisfy the time-related queries (Han et al., 2021; **514** Mavromatis et al., 2022; Liang et al., 2022; Chen **515** et al., 2023) or are strictly dependent on the con- **516** tinual pre-training to strengthen models’ abilities **517** in certain temporal aspects (Tan et al., 2023; Yuan **518** et al., 2023). The evolution of LLMs has demon- **519** strated impressive ability in complex reasoning **520** tasks (Chen, 2023), such as mathematical reason- **521** ing (Mishra et al., 2022) and logic reasoning (liu **522** et al., 2023). In light of these advancements, re- **523** cent methods shift towards a program-aided ap- **524** proach (Gao et al., 2023) to improve the perfor- **525** mance of time-sensitive tasks, employing Python **526** code as an intermediate logical step instead of nat- **527** ural language (Li et al., 2023). This method, while **528** effective, relies heavily on external tools (Zhu et al., **529** 2023b) and does not fully leverage the inherent ca- **530** pabilities of LLMs (Brown et al., 2020). The results **531** from our COTEMPQA datasets reveal that existing **532** LLMs, even with advanced strategies like Chain of **533** Thought (Wei et al., 2022b), demonstrate limited **534** efficacy in addressing the complexities inherent **535** in co-temporal reasoning tasks. Meanwhile, our **536** research highlights the significant role of mathe- **537** matical abilities in co-temporal reasoning, offering **538** a direction for future methodologies. **539** 

## **6 Conclusion** 

**540** 

In this paper, we propose the COTEMPQA datasets **541** to facilitate the investigation of under-explored co- **542** temporal reasoning problem for large language **543** models. Extensive experiments have shown a sig- **544** nificant gap between existing advanced LLMs and **545** human-level performance, even with the enhance- **546** ment of reasoning approaches. We also discover **547** that mathematical reasoning is crucial for under- **548** standing co-temporal events and propose a math- **549** based strategy to improve LLMs’ co-temporal rea- **550** soning. Reasoning on concurrent and intricate tem- **551** poral relations remains an open research question, **552** and we hope more enhancement to develop upon **553** our COTEMPQA datasets. **554** 

8 

**556** There are still some limitations in our work, which **557** are listed below: 

## **555 7 Limitations** 

- **558** • For our open-book QA setting, we directly pro- **559** vide the subject’s relevant facts in a structured **560** format in the prompt. Recent work shows that **561** LLM’s performance in context-based reasoning **562** was significantly weaker than in the former (Chu **563** et al., 2023). In the future, we will employ some **564** retrieval tools to construct prompts with more **565** contextually rich information sources. 

- **566** • We evaluate the co-temporal reasoning capabil- **567** ities from the perspective of task performance. **568** However, a more direct approach could involve **569** analyzing how the model’s neurons and hidden **570** states are triggered (Zhang et al., 2023). This lim- **571** itation is not unique to our study and is common **572** in most evaluations of Large Language Models. 

## **References** 

## **573** 

- **574** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury **575** Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. **576** 2023. Gqa: Training generalized multi-query trans- **577** former models from multi-head checkpoints. arXiv **578** preprint arXiv:2305.13245. 

- **579** Tom Brown, Benjamin Mann, Nick Ryder, Melanie **580** Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind **581** Neelakantan, Pranav Shyam, Girish Sastry, Amanda **582** Askell, et al. 2020. Language models are few-shot **583** learners. Advances in neural information processing **584** systems, 33:1877–1901. 

- **585** Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, **586** Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, **587** Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, **588** Yi Chang, Philip S. Yu, Qiang Yang, and Xing Xie. **589** 2023. A survey on evaluation of large language mod- **590** els. 

- **591** Wenhu Chen. 2023. Large language models are few(1)- **592** shot table reasoners. In Findings of the Association **593** for Computational Linguistics: EACL 2023, pages **594** 1120–1130, Dubrovnik, Croatia. Association for **595** Computational Linguistics. 

- **596** Wenhu Chen, Xinyi Wang, and William Yang **597** Wang. 2021. A dataset for answering time- **598** sensitive questions. In Thirty-ffth Conference on **599** Neural Information Processing Systems Datasets **600** and Benchmarks Track (Round 2). 

- **601** Ziyang Chen, Jinzhi Liao, and Xiang Zhao. 2023. Multi- **602** granularity temporal question answering over knowl- **603** edge graphs. In Proceedings of the 61st Annual **604** Meeting of the Association for Computational 

Linguistics (Volume 1: Long Papers), pages 11378– **605** 11392, Toronto, Canada. Association for Computa- **606** tional Linguistics. **607** 

- Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang **608** Yu, Haotian Wang, Ming Liu, and Bing Qin. 2023. **609** Timebench: A comprehensive evaluation of temporal **610** reasoning abilities in large language models. arXiv **611** preprint arXiv:2311.17667. **612** 

|Bhuwan Dhingra, Jeremy R. Cole, Julian Martin|**613**|
|---|---|
|Eisenschlos, Daniel Gillick, Jacob Eisenstein, and|**614**|
|William W. Cohen. 2022. Time-aware language mod-|**615**|
|els as temporal knowledge bases.<br>Transactions<br>of<br>the|**616**|
|Association<br>for<br>Computational<br>Linguistics,10:257–|**617**|
|273.|**618**|
|Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon,|**619**|
|Pengfei Liu, Yiming Yang, Jamie Callan, and Gra-|**620**|
|ham Neubig. 2023. Pal: Program-aided language|**621**|
|models.|**622**|
|Rujun Han, Xiang Ren, and Nanyun Peng. 2021.|**623**|
|ECONET: Effective continual pretraining of lan-|**624**|
|guage models for event temporal reasoning.<br>In|**625**|
|Proceedings<br>of<br>the<br>2021<br>Conference<br>on<br>Empirical|**626**|
|Methods<br>in<br>Natural<br>Language<br>Processing, pages|**627**|
|5367–5380, Online and Punta Cana, Dominican Re-|**628**|
|public. Association for Computational Linguistics.|**629**|
|Jie Huang and Kevin Chen-Chuan Chang. 2023. To-|**630**|
|wards reasoning in large language models: A survey.|**631**|
|Jungo Kasai, Keisuke Sakaguchi, Yoichi Takahashi,|**632**|
|Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir|**633**|
|Radev, Noah A Smith, Yejin Choi, and Kentaro Inui.|**634**|
|2022. Realtime qa: What’s the answer right now?|**635**|
|arXiv<br>preprint<br>arXiv:2207.13332.|**636**|
|Takeshi Kojima, Shixiang (Shane) Gu, Machel Reid, Yu-|**637**|
|taka Matsuo, and Yusuke Iwasawa. 2022. Large lan-|**638**|
|guage models are zero-shot reasoners. In<br>Advances|**639**|
|in<br>Neural<br>Information<br>Processing<br>Systems, vol-|**640**|
|ume 35, pages 22199–22213. Curran Associates, Inc.|**641**|
|Xingxuan Li, Liying Cheng, Qingyu Tan, Hwee Tou Ng,|**642**|
|Shafq Joty, and Lidong Bing. 2023. Unlocking tem-|**643**|
|poral question answering for large language models|**644**|
|using code execution.|**645**|
|Ke Liang, Lingyuan Meng, Meng Liu, Yue Liu, Wenx-|**646**|
|uan Tu, Siwei Wang, Sihang Zhou, Xinwang Liu, and|**647**|
|Fuchun Sun. 2022. Reasoning over different types of|**648**|
|knowledge graphs: Static, temporal and multi-modal.|**649**|
|arXiv<br>preprint<br>arXiv:2212.05767.|**650**|
|Adam Liska, Tomas Kocisky, Elena Gribovskaya, Tay-|**651**|
|fun Terzi, Eren Sezener, Devang Agrawal, D’Autume|**652**|
|Cyprien De Masson, Tim Scholtes, Manzil Zaheer,|**653**|
|Susannah Young, et al. 2022.<br>Streamingqa: A|**654**|
|benchmark for adaptation to new knowledge over|**655**|
|time in question answering models. In<br>International|**656**|
|Conference<br>on<br>Machine<br>Learning, pages 13604–|**657**|
|13622. PMLR.|**658**|



9 

**659** Hanmeng liu, Zhiyang Teng, Ruoxi Ning, Jian Liu, Qiji **660** Zhou, and Yue Zhang. 2023. Glore: Evaluating logi- **661** cal reasoning of large language models. 

**662** Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jian- **663** guang Lou, Chongyang Tao, Xiubo Geng, Qingwei **664** Lin, Shifeng Chen, and Dongmei Zhang. 2023a. Wiz- **665** ardmath: Empowering mathematical reasoning for **666** large language models via reinforced evol-instruct. **667** arXiv preprint arXiv:2308.09583. 

   - Qingyu Tan, Hwee Tou Ng, and Lidong Bing. 2023. **714** Towards benchmarking and improving the tempo- **715** ral reasoning capability of large language models. **716** In Proceedings of the 61st Annual Meeting of the **717** Association for Computational Linguistics (Volume **718** 1: Long Papers), pages 14820–14835, Toronto, **719** Canada. Association for Computational Linguistics. **720** 

- **668** Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo **669** Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qing- **670** wei Lin, and Daxin Jiang. 2023b. Wizardcoder: **671** Empowering code large language models with evol- **672** instruct. 

- **673** Costas Mavromatis, Prasanna Lakkur Subramanyam, **674** Vassilis N. Ioannidis, Adesoji Adeshina, Phillip R **675** Howard, Tetiana Grinberg, Nagib Hakim, and George **676** Karypis. 2022. Tempoqr: Temporal question rea- **677** soning over knowledge graphs. Proceedings of **678** the AAAI Conference on Artifcial Intelligence, **679** 36(5):5825–5833. 

- **680** Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard **681** Tang, Sean Welleck, Chitta Baral, Tanmay Rajpuro- **682** hit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, **683** and Ashwin Kalyan. 2022. LILA: A unified bench- **684** mark for mathematical reasoning. In Proceedings **685** of the 2022 Conference on Empirical Methods in **686** Natural Language Processing, pages 5807–5832, **687** Abu Dhabi, United Arab Emirates. Association for **688** Computational Linguistics. 

**689** OpenAI. 2023. Gpt-4 technical report. 

- **690** Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, **691** Carroll Wainwright, Pamela Mishkin, Chong Zhang, **692** Sandhini Agarwal, Katarina Slama, Alex Ray, et al. **693** 2022. Training language models to follow instruc- **694** tions with human feedback. Advances in Neural **695** Information Processing Systems, 35:27730–27744. 

- **696** J PUSTEJOVSKY. 2003. The timebank corpus. In **697** Proceedings of Corpus Linguistics 2003, pages 647– **698** 656. 

- **699** James Pustejovsky, Robert Ingria, Andrea Setzer, and **700** Graham Katz. 2003. Timeml: Robust specification **701** of event and temporal expressions in text. 

- **702** Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and **703** Percy Liang. 2016. SQuAD: 100,000+ questions **704** for machine comprehension of text. In Proceedings **705** of the 2016 Conference on Empirical Methods in **706** Natural Language Processing, pages 2383–2392, **707** Austin, Texas. Association for Computational Lin- **708** guistics. 

- **709** Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten **710** Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, **711** Jingyu Liu, Tal Remez, Jérémy Rapin, et al. 2023. **712** Code llama: Open foundation models for code. arXiv **713** preprint arXiv:2308.12950. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Al- **721** bert, Amjad Almahairi, Yasmine Babaei, Nikolay **722** Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti **723** Bhosale, et al. 2023. Llama 2: Open founda- **724** tion and fine-tuned chat models. arXiv preprint **725** arXiv:2307.09288. **726** 

- Naushad UzZaman, Hector Llorens, James F. Allen, **727** Leon Derczynski, Marc Verhagen, and James Puste- **728** jovsky. 2012. Tempeval-3: Evaluating events, **729** time expressions, and temporal relations. CoRR, **730** abs/1206.5333. **731** 

- Denny Vrandeˇci´c and Markus Krötzsch. 2014. Wiki- **732** data: A free collaborative knowledge base. **733** Communications of the ACM, 57:78–85. **734** 

- Yuqing Wang and Yun Zhao. 2023. Tram: Benchmark- **735** ing temporal reasoning for large language models. **736** arXiv preprint arXiv:2310.00835. **737** 

- Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin **738** Guu, Adams Wei Yu, Brian Lester, Nan Du, An- **739** drew M. Dai, and Quoc V Le. 2022a. Finetuned lan- **740** guage models are zero-shot learners. In International **741** Conference on Learning Representations. **742** 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten **743** Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V **744** Le, and Denny Zhou. 2022b. Chain-of-thought **745** prompting elicits reasoning in large language mod- **746** els. In Advances in Neural Information Processing **747** Systems, volume 35, pages 24824–24837. Curran **748** Associates, Inc. **749** 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten **750** Bosma, Fei Xia, Ed H Chi, Quoc V Le, Denny Zhou, **751** et al. 2022c. Chain-of-thought prompting elicits rea- **752** soning in large language models. In Advances in **753** Neural Information Processing Systems. **754** 

- Yifan Wei, Yisong Su, Huanhuan Ma, Xiaoyan Yu, **755** Fangyu Lei, Yuanzhe Zhang, Jun Zhao, and Kang Liu. **756** 2023. Menatqa: A new dataset for testing the tem- **757** poral comprehension and reasoning abilities of large **758** language models. arXiv preprint arXiv:2310.05157. **759** 

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien **760** Chaumond, Clement Delangue, Anthony Moi, Pier- **761** ric Cistac, Tim Rault, Remi Louf, Morgan Funtow- **762** icz, Joe Davison, Sam Shleifer, Patrick von Platen, **763** Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, **764** Teven Le Scao, Sylvain Gugger, Mariama Drame, **765** Quentin Lhoest, and Alexander Rush. 2020. Trans- **766** formers: State-of-the-art natural language processing. **767** In Proceedings of the 2020 Conference on Empirical **768** Methods in Natural Language Processing: System **769** Demonstrations, pages 38–45, Online. Association **770** for Computational Linguistics. **771** 

10 

- **772** Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, **773** Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin **774** Jiang. 2023. Wizardlm: Empowering large language **775** models to follow complex instructions. 

- **776** Chenhan Yuan, Qianqian Xie, Jimin Huang, and Sophia **777** Ananiadou. 2023. Back to the future: Towards ex- **778** plainable temporal reasoning with large language **779** models. 

- **780** Michael Zhang and Eunsol Choi. 2021. SituatedQA: **781** Incorporating extra-linguistic contexts into QA. In **782** Proceedings of the 2021 Conference on Empirical **783** Methods in Natural Language Processing, pages **784** 7371–7387, Online and Punta Cana, Dominican Re- **785** public. Association for Computational Linguistics. 

- **786** Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, **787** George Karypis, and Alex Smola. 2023. Multimodal **788** chain-of-thought reasoning in language models. 

- **789** Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, **790** Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen **791** Zhang, Junjie Zhang, Zican Dong, et al. 2023. A **792** survey of large language models. arXiv preprint **793** arXiv:2303.18223. 

- **794** Victor Zhong, Weijia Shi, Wen tau Yih, and Luke Zettle- **795** moyer. 2022. Romqa: A benchmark for robust, multi- **796** evidence, multi-answer question answering. 

- **797** Ben Zhou, Daniel Khashabi, Qiang Ning, and Dan **798** Roth. 2019. “going on a vacation” takes longer **799** than “going for a walk”: A study of temporal com- **800** monsense understanding. In Proceedings of the **801** 2019 Conference on Empirical Methods in Natural **802** Language Processing and the 9th International **803** Joint Conference on Natural Language Processing **804** (EMNLP-IJCNLP), pages 3363–3369, Hong Kong, **805** China. Association for Computational Linguistics. 

- **806** Xinyu Zhu, Junjie Wang, Lin Zhang, Yuxiang Zhang, **807** Yongfeng Huang, Ruyi Gan, Jiaxing Zhang, and **808** Yujiu Yang. 2023a. Solving math word problems **809** via cooperative reasoning induced language models. **810** In Proceedings of the 61st Annual Meeting of the **811** Association for Computational Linguistics (Volume **812** 1: Long Papers), pages 4471–4485, Toronto, Canada. **813** Association for Computational Linguistics. 

- **814** Xinyu Zhu, Cheng Yang, Bei Chen, Siheng Li, Jian- **815** Guang Lou, and Yujiu Yang. 2023b. Question an- **816** swering as programming for solving time-sensitive **817** questions. arXiv preprint arXiv:2305.14221. 

11 

**826** 

**833** 

**844** 

**845** 

**846 847 848 849 850 851 852 853 854 855 856** 

**857** 

**858 859 860 861 862** 

**863** 

## **818 A Implementation Details** 

**819** We utilize the OpenAI API[1] to evaluate all closed- **820** source models, and for open-source models, we **821** employ the transformers library (Wolf et al., 2020). **822** In all our experiments, we set the temperature to **823** 0 and the maximum length to 256. These experi- **824** ments were conducted across a full range of scales **825** for each evaluated model. 

## **B The Details of Co-temporal Extraction** 

**827** Building on our approach to structuring time- **828** dependent facts from Wikidata, we delve into ex- **829** tracting co-temporal facts by identifying overlaps **830** in the timestamps of different facts. Specifically, **831** we compare a given fact _fi_ = ( _si, ri, oi, tsi, tei_ ) **832** with another fact _fj_ in five distinct scenarios: 

## **B.1 Scenario 1:** ( _S, R, O_ ) 

## **Original Fact:** _fii_ as defined above. 

**834 Original Fact:** _fii_ as defined above. **835 Compared Fact:** _fj_ = ( _si, ri, oj, tsj , tej_ ). **836 Explanation:** In this scenario, the subject _si_ and **837** relation _ri_ remain constant, indicating the same **838** subject in the same type of relationship. However, **839** the object changes, where the subject is related to **840** different objects during co-temporal periods. **841 Template:** While < _subject_ 1> was holding **842** the position of < _object_ 1>, which position did **843** < _subject_ 1> hold during the same time span? 

## **B.2 Scenario 2:** ( _S, R, O_ ) 

**Original Fact:** _fi_ as above. 

**Compared Fact:** _fj_ = ( _si, rj, oj, tsj , tej_ ). **Explanation:** Here, the subject _si_ stays constant while the relation and the object change. This scenario is crucial for identifying instances where a single subject is involved in different relationships with different objects concurrently. **Template:** While < _subject_ 1> was holding the position of < _object_ 1>, which political party did < _subject_ 1> belong to simultaneously? 

|relationship with the same object.|**864**|
|---|---|
|**Template:** While <_subject_1> was holding the|**865**|
|position of <_object_1>, who also held the position|**866**|
|of <_object_1> concurrently?|**867**|
||**868**|
|**B.4**<br>**Scenario 4:** (<br>_S, R,_<br>_O_)|**869**|
|**Original Fact:** _fi_as above.|**870**|
|**Compared Fact:** _fj_ = (_sj, ri, oj, tsj, tej_).|**871**|
|**Explanation:**<br>Only the relationship remains|**872**|
|constant in this case, while both the subject|**873**|
|and the object change.<br>This scenario signifes|**874**|
|instances where different subjects share a common|**875**|
|relationship with different objects concurrently.|**876**|
|**Template:** While <_subject_1> was playing for|**877**|
|<_object_1>, which team did <_subject_2> play for|**878**|
|within the same time interval?|**879**|
||**880**|
|**B.5**<br>**Scenario 5:** (<br>_S,_<br>_R,_<br>_O_)|**881**|
|**Original Fact:** _fi_as above.|**882**|
|**Compared Fact:** _fj_ = (_sj, rj, oj, tsj, tej_).|**883**|
|**Explanation:** This scenario represents completely|**884**|
|distinct facts that overlap in time, with all quintu-|**885**|
|plet elements changing.|**886**|
|**Template:** While <_subject_1> was holding the|**887**|
|position of <_object_1>,<br>which employer did|**888**|
|<_subject_2> work for during the same time?|**889**|
||**890**|
|**C**<br>**Prompts**|**891**|
|The prompts and demonstrations can be found from|**892**|
|Figure3to Figure12.|**893**|



## **B.3 Scenario 3:** ( _S, R, O_ ) 

## **Original Fact:** _fi_ as above. 

**Compared Fact:** _fj_ = ( _sj, ri, oi, tsj , tej_ ). 

**Explanation:** This scenario reflects cases where the relationship and object remain constant, but the subject changes. It suggests different subjects simultaneously having the same type of 

> 1https://platform.openai.com/ 

12 

|**Model**|(_S, R,_<br>_O_)<br>**Acc.**<br>**F**1<br>**Avg.**|(_S,_<br>_R,_<br>_O_)<br>**Acc.**<br>**F**1<br>**Avg.**|(<br>_S, R, O_)<br>**Acc.**<br>**F**1<br>**Avg.**|(<br>_S, R,_<br>_O_)<br>**Acc.**<br>**F**1<br>**Avg.**|(<br>_S,_<br>_R,_<br>_O_)<br>**Overall**<br>**Acc.**<br>**F**1<br>**Avg.**|(<br>_S,_<br>_R,_<br>_O_)<br>**Overall**<br>**Acc.**<br>**F**1<br>**Avg.**|
|---|---|---|---|---|---|---|
|||**The Closed Book Question Answer (CBQA) setting**|||||
|GPT-3.5-TURBO-0613<br>GPT-4-0613|**7.3**<br>**11.9**<br>**9.6**<br>3.5<br>8.0<br>5.7|29.7<br>31.1<br>30.4<br>**3.0**<br>**4.5**<br>**3.7**<br>13.3<br>**30.6**<br>**22.0**<br>**30.0**<br>**31.3**<br>**30.7**<br>2.3<br>4.4<br>3.3<br>**15.0**<br>24.5<br>19.7|||8.5<br>**21.8**<br>**15.1**<br>**9.5**<br>15.1<br>12.3|**16.3**<br>14.5|
|||**The Open Book Question Answer (OBQA) setting**|||||
|GPT-3.5-TURBO<br>GPT-4|34.2<br>55.8<br>45.0<br>**57.0**<br>**74.7**<br>**65.9**|54.7<br>57.1<br>55.9<br>34.4<br>55.6<br>45.0<br>17.8<br>33.5<br>25.6<br>**68.0**<br>**70.8**<br>**69.4**<br>**58.0**<br>**73.3**<br>**65.6**<br>**29.9**<br>**48.8**<br>**39.4**|||15.1<br>28.0<br>21.5<br>**23.5**<br>**39.8**<br>**31.7**|38.9<br>**54.7**|
|CODELLAMA-7B<br>WIZARDCODER-7B<br>LLAMA-7B<br>WIZARDMATH-7B|4.3<br>**24.3**<br>**14.3**<br>3.3<br>17.9<br>10.6<br>2.5<br>20.5<br>11.5<br>**7.2**<br>16.8<br>12.0|6.5<br>23.6<br>15.0<br>1.3<br>15.4<br>8.4<br>1.7<br>9.9<br>5.8<br>16.6<br>**27.3**<br>22.0<br>3.6<br>16.7<br>10.1<br>1.7<br>8.2<br>4.9<br>9.5<br>23.0<br>16.3<br>1.9<br>**20.5**<br>11.2<br>2.2<br>16.8<br>9.5<br>**22.8**<br>26.6<br>**24.7**<br>**4.5**<br>19.1<br>**11.8**<br>**7.5**<br>**17.9**<br>**12.7**|||2.2<br>14.3<br>8.3<br>2.3<br>12.5<br>7.4<br>3.4<br>**18.7**<br>11.1<br>**7.1**<br>17.1<br>**12.1**|10.5<br>11.2<br>12.0<br>**14.8**|
|CODELLAMA-13B<br>WIZARDCODER-13B<br>LLAMA-13B<br>WIZARDMATH-13B|6.0<br>26.0<br>16.0<br>5.0<br>16.6<br>10.8<br>6.2<br>**28.6**<br>**17.4**<br>**11.2**<br>19.7<br>15.5|7.7<br>26.9<br>17.3<br>2.0<br>17.9<br>10.0<br>1.0<br>16.2<br>8.6<br>19.6<br>30.2<br>24.9<br>**3.9**<br>**23.3**<br>**13.6**<br>**6.3**<br>15.1<br>**10.7**<br>10.5<br>27.0<br>18.7<br>3.6<br>20.9<br>12.2<br>2.3<br>**17.2**<br>9.8<br>**30.9**<br>**34.0**<br>**32.5**<br>3.1<br>9.1<br>6.1<br>5.6<br>13.8<br>9.7|||1.7<br>16.7<br>9.2<br>**4.6**<br>12.6<br>8.6<br>3.0<br>**17.6**<br>**10.3**<br>4.0<br>9.1<br>6.5|12.4<br>13.9<br>13.8<br>**14.4**|
|CODELLAMA-34B<br>WIZARDCODER-34B<br>LLAMA-70B<br>WIZARDMATH-70B|9.9<br>42.4<br>26.2<br>11.0<br>22.6<br>16.8<br>10.3<br>**43.6**<br>26.9<br>**18.3**<br>37.8<br>**28.1**|19.6<br>38.3<br>29.0<br>4.7<br>27.1<br>15.9<br>4.3<br>25.6<br>14.9<br>35.2<br>38.0<br>36.6<br>**9.0**<br>27.4<br>18.2<br>7.5<br>15.9<br>11.7<br>14.7<br>37.6<br>26.1<br>7.1<br>**31.7**<br>**19.4**<br>7.0<br>32.4<br>19.7<br>**49.8**<br>**53.4**<br>**51.6**<br>8.6<br>24.8<br>16.7<br>**20.1**<br>**37.3**<br>**28.7**|||3.6<br>21.6<br>12.6<br>7.4<br>16.0<br>11.7<br>6.2<br>29.9<br>18.1<br>**17.4**<br>**30.1**<br>**23.8**|20.0<br>19.2<br>22.2<br>**30.1**|



Table 7: Experimental results of different triple element types in COTEMPQA. The best performance is **bold** . 

13 

|**WikiData ID**<br>**KB Relation Pairs**<br>**# Queries**<br>**Template**|P102-P102<br>political party & political party<br>475<br>While <subject> was a member of <object>, which political party did <subject> belong to within the same time interval?<br>P39-P39<br>position held & position held<br>1,017<br>While <subject> was holding the position of <object>, which position did <subject> hold during the same time span?<br>P108-P108<br>employer & employer<br>768<br>While <subject> was working for <object>, which employer did <subject> work for during the same time period?<br>P54-P54<br>member of sports team & member of sports team<br>204<br>While <subject> was playing for <object>, which team did <subject> play for at the same time?<br>P69-P69<br>educated at & educated at<br>258<br>While <subject> attended <object>, which school was <subject> attending during the identical time period?<br>P127-P127<br>owned by & owned by<br>75<br>While <subject> was owned by <object>, who was the owner of <subject> concurrently?<br>P102-P39<br>political party & position held<br>117<br>While <subject> was a member of <object>, which position did <subject> hold simultaneously?<br>P102-P108<br>political party & employer<br>101<br>While <subject> was a member of <object>, which employer did <subject> work for during the same time span?<br>P102-P69<br>political party & educated at<br>74<br>While <subject> was a member of <object>, which school was <subject> attending within the same time interval?<br>P39-P102<br>position held & political party<br>420<br>While <subject> was holding the position of <object>, which political party did <subject> belong to during the same time period?<br>P39-P108<br>position held & employer<br>380<br>While <subject> was holding the position of <object>, which employer did <subject> work for at the same time?<br>P108-P39<br>employer & position held<br>125<br>While <subject> was working for <object>, which position did <subject> hold during the identical time period?<br>P108-P69<br>employer & educated at<br>241<br>While <subject> was working for <object>, which school was <subject> attending concurrently?<br>P54-P69<br>member of sports team & educated at<br>77<br>While <subject> was playing for <object>, which school was <subject> attending simultaneously?<br>P69-P102<br>educated at & political party<br>187<br>While <subject> attended <object>, which political party did <subject> belong to during the same time span?<br>P69-P39<br>educated at & position held<br>95<br>While <subject> attended <object>, which position did <subject> hold within the same time interval?<br>P69-P108<br>educated at & employer<br>134<br>While <subject> attended <object>, which employer did <subject> work for during the same time period?|Table 8: Templates used for converting Wikidata facts into natural questions.|
|---|---|---|



14 

**Question** : While Valdis Dombrovskis was holding the position of European Commissioner for Trade, which position did Valdis Dombrovskis during the identical time period? **Only return the answer** : 

Figure 3: Default prompt for **C** losed- **B** ook **QA** ( **CBQA** ) in our proposed COTEMPQA 

**Question** : While Valdis Dombrovskis was holding the position of European Commissioner for Trade, which position did Valdis Dombrovskis during the identical time period? **Answer: Let’s think step by step,** 

Figure 4: Zero-cot prompt for **C** losed- **B** ook **QA** ( **CBQA** ) in our proposed COTEMPQA 

**Question** : While Valdis Dombrovskis was holding the position of European Commissioner for Trade, which position did Valdis Dombrovskis during the identical time period? **Only return the answer** : European Commissioner for Internal Market and Services ...... **Question** : While Eduard Jan Dijksterhuis was working for Leiden University, which employer did Eduard Jan Dijksterhuis work for during the same time span? **Only return the answer** : 

Figure 5: Few-shot prompt for **C** losed- **B** ook **QA** ( **CBQA** ) in our proposed COTEMPQA (5-shot) 

**Question** : While Valdis Dombrovskis was holding the position of European Commissioner for Trade, which position did Valdis Dombrovskis during the identical time period? **Answer** : According to the fact, Valdis Dombrovskis became the European Commissioner for Trade on August 26, 2020. He also held the position of European Commissioner for Internal Market and Services from July 16, 2016, to October 12, 2020. This period overlaps with his tenure as Commissioner for Trade. Therefore, the answer is European Commissioner for Internal Market and Services. ...... **Question** : While Eduard Jan Dijksterhuis was working for Leiden University, which employer did Eduard Jan Dijksterhuis work for during the same time span? **Answer: According to the fact,** 

Figure 6: Few-shot&CoT prompt for **C** losed- **B** ook **QA** ( **CBQA** ) in our proposed COTEMPQA (5-shot) 

**Question** : While Valdis Dombrovskis was holding the position of European Commissioner for Trade, which position did Valdis Dombrovskis during the identical time period? **Answer** : According to the context, Valdis Dombrovskis became the European Commissioner for Trade on August 26, 2020. The datetime can be formed (2020, 8, 26). The content provided and related to the question can be structured as: (Vice-President of the European Commission, (2019, 12, 1)). (European Commissioner for Internal Market and Services, (2016, 6, 16), (2020, 10, 12)). (European Commissioner for An Economy, (2019, 10, 1)). (Prime Minister of Latvia, (2009, 3, 12), (2014, 1, 22)). (Minister of Finance, (2002, 11, 7), (2004, 3, 9)). Given the (2020, 8, 26), compared with all contents related, we find that [(2016 _,_ 6 _,_ 16) _−_ (2020 _,_ 10 _,_ 12)] _∩_ (2020 _,_ 8 _,_ 26) _̸_ = _∅_ . Therefore the answer is European Commissioner for Internal Market and Services. ...... **Question** : While Eduard Jan Dijksterhuis was working for Leiden University, which employer did Eduard Jan Dijksterhuis work for during the same time span? **Answer: According to the fact,** 

Figure 7: Few-shot&Mr-CoT prompt for **C** losed- **B** ook **QA** ( **CBQA** ) in our proposed COTEMPQA (5-shot) 

15 

## **Answer the question based on the context:** 

**Context** : Valdis Dombrovskis holds the position of Vice-President of the European Commission in December 1, 2019. Valdis Dombrovskis holds the position of European Commissioner for Internal Market and Services from July 16, 2016 to October 12, 2020. Valdis Dombrovskis holds the position of European Commissioner for Trade in August 26, 2020. Valdis Dombrovskis holds the position of European Commissioner for An Economy that Works for People in December 1, 2019. Valdis Dombrovskis holds the position of Prime Minister of Latvia from March 12, 2009 to January 22, 2014. Valdis Dombrovskis holds the position of Minister of Finance from November 7, 2002 to March 9, 2004. **Question** : While Valdis Dombrovskis was holding the position of European Commissioner for Trade, which position did Valdis Dombrovskis during the identical time period? **Only return the answer** : 

Figure 8: Default prompt for **O** pen- **B** ook **QA** ( **OBQA** ) in our proposed COTEMPQA 

## **Answer the question based on the context:** 

**Context** : Valdis Dombrovskis holds the position of Vice-President of the European Commission in December 1, 2019. Valdis Dombrovskis holds the position of European Commissioner for Internal Market and Services from July 16, 2016 to October 12, 2020. Valdis Dombrovskis holds the position of European Commissioner for Trade in August 26, 2020. Valdis Dombrovskis holds the position of European Commissioner for An Economy that Works for People in December 1, 2019. Valdis Dombrovskis holds the position of Prime Minister of Latvia from March 12, 2009 to January 22, 2014. Valdis Dombrovskis holds the position of Minister of Finance from November 7, 2002 to March 9, 2004. 

**Question** : While Valdis Dombrovskis was holding the position of European Commissioner for Trade, which position did Valdis Dombrovskis during the identical time period? **Answer: Let’s think step by step,** 

Figure 9: Zero-cot prompt for **O** pen- **B** ook **QA** ( **OBQA** ) in our proposed COTEMPQA 

**Answer the question based on the context: Context** : Valdis Dombrovskis holds the position of Vice-President of the European Commission in December 1, 2019. Valdis Dombrovskis holds the position of European Commissioner for Internal Market and Services from July 16, 2016 to October 12, 2020. Valdis Dombrovskis holds the position of European Commissioner for Trade in August 26, 2020. Valdis Dombrovskis holds the position of European Commissioner for An Economy that Works for People in December 1, 2019. Valdis Dombrovskis holds the position of Prime Minister of Latvia from March 12, 2009 to January 22, 2014. Valdis Dombrovskis holds the position of Minister of Finance from November 7, 2002 to March 9, 2004. **Question** : While Valdis Dombrovskis was holding the position of European Commissioner for Trade, which position did Valdis Dombrovskis during the identical time period? **Only return the answer** : European Commissioner for Internal Market and Services ...... **Answer the question based on the context: Context** : Eduard Jan Dijksterhuis attended University of Groningen from 1911 to 1918. Eduard Jan Dijksterhuis worked for Duke University School of Medicine from August 28, 1912 to April 28, 1923. Eduard Jan Dijksterhuis works for Leiden University from July 5, 1954 to September 5, 1960. Eduard Jan Dijksterhuis worked for American University of Armenia in August, 1911. Eduard Jan Dijksterhuis worked for Austin College from July, 1936 to April, 1947. Eduard Jan Dijksterhuis worked for Sonoma State University in July, 1932. Eduard Jan Dijksterhuis worked for Fairfax Media in December 16, 1942. Eduard Jan Dijksterhuis worked for Canadian Broadcasting Corporation in 1941. Eduard Jan Dijksterhuis works for Utrecht University from May 1, 1953 to September 1, 1960. Eduard Jan Dijksterhuis worked for Jean-Marie Le Pen in January, 1931. **Question** : While Eduard Jan Dijksterhuis was working for Leiden University, which employer did Eduard Jan Dijksterhuis work for during the same time span? **Only return the answer** : 

Figure 10: Few-shot prompt for **O** pen- **B** ook **QA** ( **OBQA** ) in our proposed COTEMPQA (5-shot) 

16 

## **Answer the question based on the context:** 

**Context** : Valdis Dombrovskis holds the position of Vice-President of the European Commission in December 1, 2019. Valdis Dombrovskis holds the position of European Commissioner for Internal Market and Services from July 16, 2016 to October 12, 2020. 

Valdis Dombrovskis holds the position of European Commissioner for Trade in August 26, 2020. Valdis Dombrovskis holds the position of European Commissioner for An Economy that Works for People in December 1, 2019. Valdis Dombrovskis holds the position of Prime Minister of Latvia from March 12, 2009 to January 22, 2014. Valdis Dombrovskis holds the position of Minister of Finance from November 7, 2002 to March 9, 2004. **Question** : While Valdis Dombrovskis was holding the position of European Commissioner for Trade, which position did Valdis Dombrovskis during the identical time period? **Answer** : According to the context, Valdis Dombrovskis became the European Commissioner for Trade on August 26, 2020. He also held the position of European Commissioner for Internal Market and Services from July 16, 2016, to October 12, 2020. This period overlaps with his tenure as Commissioner for Trade. Therefore, the answer is European Commissioner for Internal Market and Services. 

...... **Answer the question based on the context: Context** : Eduard Jan Dijksterhuis attended University of Groningen from 1911 to 1918. 

Eduard Jan Dijksterhuis worked for Duke University School of Medicine from August 28, 1912 to April 28, 1923. Eduard Jan Dijksterhuis works for Leiden University from July 5, 1954 to September 5, 1960. Eduard Jan Dijksterhuis worked for American University of Armenia in August, 1911. Eduard Jan Dijksterhuis worked for Austin College from July, 1936 to April, 1947. Eduard Jan Dijksterhuis worked for Sonoma State University in July, 1932. Eduard Jan Dijksterhuis worked for Fairfax Media in December 16, 1942. Eduard Jan Dijksterhuis worked for Canadian Broadcasting Corporation in 1941. Eduard Jan Dijksterhuis works for Utrecht University from May 1, 1953 to September 1, 1960. Eduard Jan Dijksterhuis worked for Jean-Marie Le Pen in January, 1931. 

**Question** : While Eduard Jan Dijksterhuis was working for Leiden University, which employer did Eduard Jan Dijksterhuis work for during the same time span? **Answer: According to the context,** 

Figure 11: Few-shot&CoT prompt for **O** pen- **B** ook **QA** ( **OBQA** ) in our proposed COTEMPQA (5-shot) 

17 

**Answer the question based on the context: Context** : Valdis Dombrovskis holds the position of Vice-President of the European Commission in December 1, 2019. Valdis Dombrovskis holds the position of European Commissioner for Internal Market and Services from July 16, 2016 to October 12, 2020. Valdis Dombrovskis holds the position of European Commissioner for Trade in August 26, 2020. Valdis Dombrovskis holds the position of European Commissioner for An Economy that Works for People in December 1, 2019. Valdis Dombrovskis holds the position of Prime Minister of Latvia from March 12, 2009 to January 22, 2014. Valdis Dombrovskis holds the position of Minister of Finance from November 7, 2002 to March 9, 2004. **Question** : While Valdis Dombrovskis was holding the position of European Commissioner for Trade, which position did Valdis Dombrovskis during the identical time period? **Answer** : According to the context, Valdis Dombrovskis became the European Commissioner for Trade on August 26, 2020. The datetime can be formed (2020, 8, 26). The content provided and related to the question can be structured as: (Vice-President of the European Commission, (2019, 12, 1)). (European Commissioner for Internal Market and Services, (2016, 6, 16), (2020, 10, 12)). (European Commissioner for An Economy, (2019, 10, 1)). (Prime Minister of Latvia, (2009, 3, 12), (2014, 1, 22)). (Minister of Finance, (2002, 11, 7), (2004, 3, 9)). Given the (2020, 8, 26), compared with all contents related, we find that [(2016 _,_ 6 _,_ 16) _−_ (2020 _,_ 10 _,_ 12)] _∩_ (2020 _,_ 8 _,_ 26) _̸_ = _∅_ . Therefore the answer is European Commissioner for Internal Market and Services. ...... **Answer the question based on the context: Context** : Eduard Jan Dijksterhuis attended University of Groningen from 1911 to 1918. Eduard Jan Dijksterhuis worked for Duke University School of Medicine from August 28, 1912 to April 28, 1923. Eduard Jan Dijksterhuis works for Leiden University from July 5, 1954 to September 5, 1960. Eduard Jan Dijksterhuis worked for American University of Armenia in August, 1911. Eduard Jan Dijksterhuis worked for Austin College from July, 1936 to April, 1947. Eduard Jan Dijksterhuis worked for Sonoma State University in July, 1932. Eduard Jan Dijksterhuis worked for Fairfax Media in December 16, 1942. Eduard Jan Dijksterhuis worked for Canadian Broadcasting Corporation in 1941. Eduard Jan Dijksterhuis works for Utrecht University from May 1, 1953 to September 1, 1960. Eduard Jan Dijksterhuis worked for Jean-Marie Le Pen in January, 1931. **Question** : While Eduard Jan Dijksterhuis was working for Leiden University, which employer did Eduard Jan Dijksterhuis work for during the same time span? **Answer: According to the context,** 

Figure 12: Few-shot&Mr-CoT prompt for **O** pen- **B** ook **QA** ( **OBQA** ) in our proposed COTEMPQA (5-shot) 

18 

