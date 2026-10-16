# _**CausalChaos!**_ **Dataset for Comprehensive Causal Action Question Answering Over Longer Causal Chains Grounded in Dynamic Visual Scenes** 

**Paritosh Parmar**[1] **, Eric Peh**[1] **, Ruirui Chen**[1] **, Ting En Lam**[1] _[,]_[2] **, Yuhan Chen**[1] _[,]_[3] **, Elston Tan**[1] _[,]_[4] **, Basura Fernando**[1] 

1Institute of High-Performance Computing, Agency for Science, Technology and Research, Singapore 

2Nanyang Technological University, 3National University of Singapore, 4Singapore Polytechnic `https://github.com/LUNAProject22/CausalChaos` 

## **Abstract** 

Causal video question answering (QA) has garnered increasing interest, yet existing datasets often lack depth in causal reasoning. To address this gap, we capitalize on the unique properties of cartoons and construct CausalChaos!, a novel, challenging causal Why-QA dataset built upon the iconic “Tom and Jerry" cartoon series. Cartoons use the principles of animation that allow animators to create expressive, unambiguous causal relationships between events to form a coherent storyline. Utilizing these properties, along with thought-provoking questions and multilevel answers (answer and detailed causal explanation), our questions involve causal chains that interconnect multiple dynamic interactions between characters and visual scenes. These factors demand models to solve more challenging, yet well-defined causal relationships. We also introduce hard incorrect answer mining, including a causally confusing version that is even more challenging. While models perform well, there is much room for improvement, especially, on open-ended answers. We identify more advanced/explicit causal relationship modeling & joint modeling of vision and language as the immediate areas for future efforts to focus upon. Along with the other complementary datasets, our new challenging dataset will pave the way for these developments in the field. 

## **1 Introduction** 

Understanding the intricate motivations behind human actions is paramount in developing sophisticated systems capable of nuanced behavior analysis. In real-world scenarios, actions are shaped by a multitude of factors, including personal experiences, emotions, social contexts, and cultural backgrounds. This complexity necessitates a comprehensive approach to unraveling the "why" behind actions, fostering empathy, effective communication, and robust decision-making. Causal video question answering aims to decipher the answers behind characters’ actions. Despite the growing interest in causal video QA, existing datasets often fall short, requiring only: 1) _surface-level understanding_ ; or 2) _involve more of simple word substitution in the QA pairs, rather than causal reasoning_ ( _e_ . _g_ ., “Q. Why are the cars on the street not moving? A. Parked."). Recognizing this gap, we embark on the development of a rigorous and challenging causal Video-QA dataset. Our goal is to provide a high-quality resource that rigorously evaluates and advances causal video QA models. 

Drawing on the established benefits of cartoons with cognitive processes among children [38, 39, 29, 53, 40], we leverage the renowned series to _create a novel and demanding causal Why-QA dataset_ called **CausalChaos!** . The key characteristics of CausalChaos! are as follows: 

- As demonstrated in Figure 1, 2, we formulate thought-provoking questions, where the answer (A) and its detailed explanation (E) aim to enhance the model’s understanding of the causal chains of 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_CausalChaos!_Dataset_for_Comprehensive_Causal_Action_Question_Answering_Over_Longer_Causal_Chains_Grounded_in_Dynamic_Visual_Scenes_images/nips_2024_CausalChaos!_Dataset_for_Comprehensive_Causal_Action_Question_Answering_Over_Longer_Causal_Chains_Grounded_in_Dynamic_Visual_Scenes.pdf-0002-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_CausalChaos!_Dataset_for_Comprehensive_Causal_Action_Question_Answering_Over_Longer_Causal_Chains_Grounded_in_Dynamic_Visual_Scenes_images/nips_2024_CausalChaos!_Dataset_for_Comprehensive_Causal_Action_Question_Answering_Over_Longer_Causal_Chains_Grounded_in_Dynamic_Visual_Scenes.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Causal Chain Illustration Principles of Animation<br>Question: Why did Jerry grab Tom's tail? Exaggeration<br>Answerinto another pocket.: Jerry wanted to pull Tom's tail  Conveying Mammy Two Shoes “RUSHING”  home<br>Explanation: Jerry wanted to make Tom to<br>grab his own tail and pull himself through<br>the pocket tunnels.<br>Tom pulls his tail<br>Jerry grabs Tom’s tail & in turn himself<br>Scene 1 Scene 3<br>Conveying Mammy is angry<br>with body language<br>Jerry goes into<br>pocket holding  Tom grabs<br>Tom’s tail his tail Anticipation Squeeze & stretch<br>Scene 2 Scene 2  (contd.)<br>Almost steady scene with slow progression<br>**----- End of picture text -----**<br>


Figure 1: **(Left) Examples of causal questions about characters’ actions from our CausalChaos! dataset— based on Tom & Jerry cartoon series.** Q: question; A: answer; E: explanation. _Please view in Adobe Reader to play the embedded videos for better explanation._ **(Middle) Illustration of causal chain, scene changes.** Linking multiple clues/cues embedded in different scenes to solve causal relationships pose a challenge for VideoQA models. **(Right)** Animators leverage **Principles of Animation** to _stylize_ the _visuals_ & _motions_ to _disentangle/highlight key content_ of the scene to create _well-defined/unambiguous and effectively communicated cause-and-effect relationships_ . The interplay of these factors allow models to focus on solving complex, yet, well-defined, unambiguous causal relationships. 

- visual events in a given video clip. A causal chain is a sequence of events/actions in which each step influences or leads to the next, creating a cause-and-effect relationship. It illustrates how one event/action produces a subsequent event/outcome. Compared to existing datasets, CausalChaos! presents longer causal chains, as highlighted in Figure 2 and discussed in section 2. 

- • Video clips in our dataset feature frequent _scene_ and _shot changes_ . Here, a scene change or shot change refers to the transition from one visual setting or perspective to another. It typically involves shifting focus to a new location, action, or character within the storyline. The _links_ and _causes of the causal chains are often dispersed across various scenes_ . Consequently, models are challenged to exert greater cognitive effort in _connecting multiple events_ (scenes) and identifying intermediate causes to comprehend the “why" questions related to the clips. 

- Despite the complexity and length of the causal chains, they are distinctly _delineated_ , _unambiguous_ , and effectively _communicated_ using _principles of animation_ [42] like staging and exaggeration. This deliberate design allows models to _focus on deciphering causal relationships_ . 

- CausalChaos! introduces an added layer of complexity by necessitating the _modeling of actions at various levels of granularity—ranging from sweeping, large-scale movements to nuanced, subtle actions_ , such as interpreting emotional cues through facial expressions. 

- Our dataset demands a _diverse range of reasoning skills_ , encompassing deductive, spatial, emotional reasoning, and more, as outlined in Figure 2 and discussed in section 3. 

Upon evaluating various state-of-the-art VideoQA models including the recent multimodal instruction tuning models, we found that our dataset remains one of the most challenging causal QA dataset. Particularly, we observed that models often: 1) _jumped to conclusions_ based on _partial evidence_ , rather than considering the full set of evidence; 2) _failed to engage in true causal reasoning_ , opting instead for shortcuts like object/action-noun/verb matching to select answers. Based on this, we identified _more advanced/explicit causal relationship modeling and jointly modeling vision and language_ as the immediate areas for future efforts to focus upon. Further, we show that similar to how cartoons help children better connect cause and effects, they can help VideoQA models as well. We _incorporated our dataset with the NextQA_ [47], a _real-world dataset_ and found that it _brings some improvements in why questions_ . What is more, _incorporating our dataset_ brings improvement on _non-Why questions_ as well. 

Overall, we believe our CausalChaos! dataset presents challenges spanning the entire VideoQA pipeline, from _deciphering intricate videos_ to _processing complex questions_ and _discerning nuanced answers_ , stimulating research in _video processing_ , _causal reasoning_ , _language modeling_ , and _joint modeling_ . Along with the other complementary datasets, such as [47, 17, 19, 21, 46], our new challenging dataset will pave the way for these developments in the field. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_CausalChaos!_Dataset_for_Comprehensive_Causal_Action_Question_Answering_Over_Longer_Causal_Chains_Grounded_in_Dynamic_Visual_Scenes_images/nips_2024_CausalChaos!_Dataset_for_Comprehensive_Causal_Action_Question_Answering_Over_Longer_Causal_Chains_Grounded_in_Dynamic_Visual_Scenes.pdf-0003-00.png)


**----- Start of picture text -----**<br>
40<br>35 NextQA CausalChaos<br>30<br>25<br>20<br>15<br>10<br>5<br>0<br>DR IR SR CR CT ER AR TR None<br>(a)<br>Dataset MA CCL NOS RS MGA<br>NextQA ✗ 1.1 1 narrow limited<br>CausalVid ✗ 0.9 1 narrow limited<br>IntentQA ✗ 1.3 1 narrow limited<br>CausalChaos! ✓ 1.9 4 wide abundant<br>(b) (c)<br>**----- End of picture text -----**<br>


Figure 2: **(a) Types of reasoning demanded by our CausalChaos! dataset.** Reasoning types: DR- _deductive_ reasoning; IR- _inductive_ ; SR- _spatial_ ; CR- _causal_ ; CT- _critical thinking_ ; ER- _emotional_ ; AR- _abductive_ ; TR- _temporal_ ; None-no reasoning required as per the human subjects. None is undesirable, and tend to indicate that questions are less challenging. **(b) Comparison among CausalChaos! and existing causal videoQA datasets.** MA-multilevel answers; CCL-causal chain length; NOS-no. of scenes; RS-reasoning spectrum; MGA-multigranular actions. **(c) Qualitative comparison between CausalChaos! and NextQA (Why-QA) datasets.** CausalChaos! Answers and Explanations give detailed information regarding the actual cause-andeffect relationships, motivations, emotions covering wide range of reasoning types. Note that, we have temporally cropped videos to retain only the relevant parts from NextQA dataset videos; otherwise, raw videos are longer resulting in unintended problem of temporal localization for models. 

## **2 Related Work** 

To drive the progress in VideoQA, researchers have developed various datasets [26, 48, 10, 58, 56, 61, 13, 14, 54, 55, 9, 46] with distinct focuses and contributions—see the survey [59]. In the following, we mainly compare our work with prior art focusing on causal QA literature. We provide a detailed discussion on various VideoQA datasets and models in the Appendix. Our work is inspired from a few datasets: _CLEVRER_ [52, 27], _NextQA_ [47], _CausalVidQA_ [17], _IntentQA_ [19] that cover causal reasoning. However, they have limitations that our dataset aims to fill: 

- _Limitation in scope_ : _e_ . _g_ ., [52, 27] only consider _collision events of simple inanimate objects_ such as cuboids; as a result, _lack: actors_ , their _characteristics_ such as _emotions_ , _intentions_ . Our dataset has actors whose actions are shaped by _emotions_ , _intentions_ , _context_ , etc. 

- _Lack of precise temporal annotations_ (e.g., [47, 17]) _inadvertently_ involves _temporal localization_ & may create a _false sense of difficulty_ , while the questions may _not be challenging_ . We alleviate this problem by providing precise temporal annotations and focusing on designing _truly complex_ and challenging QA. 

- _Lack of complexity in questions_ ( _e_ . _g_ .[47, 17, 19]). Based on human studies, we found that a majority of causal QA in these datasets were flagged as _not requiring any notable reasoning_ (Figure 2(a)). We also attempted to _quantify_ the _complexity in reasoning_ by computing the _lengths of causal chains_ involved in their QAs vs. ours using GPT-4o [32]. Causal chains in [47, 17, 19] are shorter (Figure 2(b). In comparison, our dataset has _longer causal chains_ —average length 1.9—posing a challenge for models in _connecting multiple events_ or cues. Our dataset also demands _wide spectrum of types of reasoning_ (Figure 2a). 

- Scenes in [52, 27, 47, 17, 19] are _less dynamic_ , mostly involving a _single scene_ . In comparison, our dataset averages about _four scenes_ . _Rapid scene changes_ & _dynamic interactions_ challenge VideoQA models to link _context_ & _cues across different scenes_ , modeling cause-and-effect relationships over _longer causal chains_ (more details in subsection 3.2). While similar challenges have been acknowledged in other computer vision problems [34], they remain unexplored in VideoQA. 

- _Lack of hard negatives._ Current datasets may not emphasize on including hard incorrect options. Due to this, models may not be required to do causal reasoning, but rather they can exploit _shortcuts like object/action-noun/verb matching_ in vision-language spaces to select correct answer. To address this limitation, we develop _hard negative selection strategies_ . 

- _No hierarchical answers with different level of explanation._ These datasets contain a _single-level answer to ‘Why’_ -questions. On the other hand, we richly annotate our dataset to provide _two-level answers to ‘Why’-questions_ —1) direct/immediate cause; 2) deeper explanation. Quality of answers is further enhanced, and are _more informative_ as a consequence of the _more complex questions_ . 

3 

## **3 CausalChaos! Dataset** 

This section details the construction of our VideoQA dataset designed to challenge causal reasoning, covering its _video source_ , _annotation process_ , & _quality checks_ to ensure high-quality annotations. We then discuss the _unique attributes_ that make our dataset valuable for causal VideoQA tasks. 

## **3.1 Dataset Construction** 

**Video source.** To focus on _visual reasoning_ , we selected the timeless _silent_ cartoon series, “ _Tom & Jerry_ " (1940)[1] , spanning over six seasons and 161 episodes. Silent cartoons, which lack other modalities like dialogue, foster visual reasoning in children. Similarly, we hypothesize that using silent cartoons for VideoQA will enhance visual reasoning. The Tom & Jerry series is ideal for a causal reasoning dataset, offering _abundant segments with diverse cause-and-effect relationships_ . Concurrent work [22] on causal _image_ generation also leverages Tom & Jerry. 

**Annotations.** Each dataset sample includes the following annotations: { _start frame_ , _end frame_ , _Question_ , _Answer_ , _Explanation_ }. Details in the following. _•_ **Questions** are crafted to capture the _why_ or reasoning behind the _actions_ of characters in Tom & Jerry cartoons. To cultivate comprehensive & deeper video understanding capabilities in models, we formulated _thought-provoking_ causal questions where cause & effect are connected by _longer causal chains_ . Questions also extend _beyond explicit_ visual cues, encompassing gestures & expressions, to delve into the _characters’ underlying intentions_ & _goals_ . Annotators generated questions while watching the video for the first time to mimic how models assess unseen clips. They then rewatched the videos multiple times to create more critical thinking questions. To focus on visual reasoning, annotators watched the videos without audio, ensuring no audio cues were included in the dataset. _•_ **Multi-level Answers** . The first-level or the Primary answer, represents a _literal_ or _direct_ cause or form of response. It is accompanied by a deeper form of Explanation, which considers the _broader context_ of the scene, the _thoughts_ , _feelings_ , _intentions_ of characters, & their actions. This deeper explanation also takes into account potential consequences & provides further reasoning to support the primary answer. It includes reasons & additional details to comprehensively address the question. Examples shown in Figure 1 & Appendix. 

_•_ **Temporal annotations** . For each QAE set, the _Start_ and _End frame numbers_ are recorded to cover the entire scene, including contextual frames that support the reasons behind the actions. 

**Quality Check.** To ensure dataset quality and reliability, we implemented a two-stage quality check process. First, quality checkers assessed episodes they did not annotate. Then, they reviewed episodes they did not check in the first stage. Multiple quality checkers carefully reviewed and verified curated question and answer sets for logical fallacies, timestamp inconsistencies, grammatical errors, and spelling mistakes. They also excluded annotations containing audio or text aspects that the model cannot comprehend. To prevent direct overwriting, edits were flagged in a different font color for further discussion. Flagged inconsistencies or errors were resolved through discussions and consensus among checkers and annotators. The team referred to episode synopses from Tom and Jerry Wiki [43], an online encyclopedia, as a third-party opinion and for fact-checking. 

**Dataset Statistics.** Annotators reviewed all 161 episodes across 6 seasons, generating 4,945 detailed Question-Answer-Explanation sets. This comprehensive dataset covers various scenes, characters, and events, providing a solid foundation for understanding and answering questions about the series. Additional statistics and word clouds for answers and explanations are available in the Appendix. 

## **3.2 Unique Characteristics of our CausalChaos! Dataset** 

In the following, we delve into the untapped unique properties of our dataset, which both _complement existing video understanding datasets_ and present _novel challenges to VideoQA_ , particularly from a _causal reasoning perspective_ . 

1. **Multi-level answers** offer a _richer and more nuanced perspective_ , which leads to _deeper insights_ —considering not only the _immediate circumstances_ but also underlying _psychological_ , _social_ , 

> 1We only provide the annotations; Videos can be obtained from: https://www.warnerbros.com/tv/tom-andjerry-1965-volume-1. 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_CausalChaos!_Dataset_for_Comprehensive_Causal_Action_Question_Answering_Over_Longer_Causal_Chains_Grounded_in_Dynamic_Visual_Scenes_images/nips_2024_CausalChaos!_Dataset_for_Comprehensive_Causal_Action_Question_Answering_Over_Longer_Causal_Chains_Grounded_in_Dynamic_Visual_Scenes.pdf-0005-00.png)


**----- Start of picture text -----**<br>
(a) Q. Why was Jerry laughing? (b) Q. Why did Lion catch Jerry? (c) Q. Why did Tom<br>[S03E17] A. Jerry found it funny that  [S05E13] A. Jerry was laughing at Tom who  Ours A. Lion wanted  Typical reason:  shake his head?<br>the seal stole the fish from Tom. fell down the stairs and into the fountain. to hush Jerry to kill and eat A. Tom put on a radish instead of his nose<br>Figure 3: Grounded in diverse visual information.<br>optical flows<br>Traditional Temporal Modeling  Dynamic Scene Linking abrupt scene changes introduce large changes<br>(takes place within a single scene) modeling/understanding of  Scene Change Consecutive frames Scene Change<br>Scene-1 smooth, gradual changes Scene-1 Scene-2 Scene-3<br>… … … … … …<br>**----- End of picture text -----**<br>


Figure 4: **Traditional temporal modeling vs Dynamic scene linking.** Notice the abrupt scene change, which causes disruption in visual flow, resulting in large amplitude and widespread optical flow. 

and _personal_ factors that contributed to _characters’ decisions_ . By employing _multi-level answers alongside thought-provoking questions_ , we can cultivate _deeper reasoning_ and _analysis of characters’ actions_ , enabling the training and evaluation of models on complex question-answer pairs that demand a _comprehensive understanding_ of _longer causal chains_ . Further elaborated discussion on the importance of multi-level answers to ‘Why’-questions included in Appendix. 

2. **Grounded in diverse** _**Visual**_ **&** _**Motion**_ **information.** The Tom & Jerry cartoon series is _visually rich_ and _dynamic_ , with _intricate scenes_ , _actions_ , and _character interactions varying across episodes_ . We had annotators craft questions where the _answer is grounded in the video_ . This grounding compels the _model to analyze the video_ for a _broader range of details_ and _clues_ to provide _meaningful_ answers. For example, consider Figure 3(a), same question appears in different episodes, but their answers/context are completely different. This demonstrates that it is crucial for VideoQA models to understand visual information to answer correctly on our dataset. Multimodal nature of VideoQA task is enhanced by the presence of _unusual situations_ in cartoons—typically not found in real-world datasets as shown in Figure 3(b,c). Since the answers are embedded in videos, models cannot rely on _correlations/biases_ in their training data; they must thoroughly process and understand the content to answer correctly. This is crucial even with large language models (LLMs) and embeddings, which are also susceptible to _frequency-based biases_ and hallucinations [20, 24, 60]. For example, _Lion_ is often associated with _killing_ in LLM embeddings. In a qualitative analysis, we found that models might choose an answer involving ’ _killing_ ’ Jerry as the _motive_ for the Lion-character’s action, while the Lion-character was actually trying to _quiet_ Jerry Figure 3(b-left). Revealing such biases has been considered in other computer vision problems [4, 12], but is yet to be introduced in VideoQA. _Our dataset offers a valuable opportunity to address this challenge_ . Additionally, understanding Tom & Jerry cartoons requires modeling of _details_ and _actions at varying levels of granularity_ —from _sweeping movements_ to _subtle emotional cues_ via facial expressions. 

3. **Focus on causal reasoning in visually dynamic scenes.** Our dataset focuses on causal reasoning in Tom and Jerry cartoons, characterized by dynamic and rapidly changing scenes. Characters and objects may appear, disappear, and reappear, limiting the model’s access to partial observations. Video understanding models struggle to track and understand context. To answer causal questions, these models must link events to form a causal chain despite rapid scene changes and dynamic interactions. We term this Dynamic Scene Linking (DSL), distinct from traditional temporal modeling, which typically focuses on the gradual transitions within a scene as shown in Figure 4 (more details in Appendix). Related challenges are noted in other computer vision problems [34, 45] but not yet in VideoQA. We hypothesize that humans understand cartoons by forming a mental “world model" of the scene, which helps bridge gaps between discontinuous scenes and provides a coherent overall context. Current VideoQA datasets often lack coverage of such complex scenarios, unlike our dataset. 

4. **More challenging and requiring cognitive effort.** We used GPT-4o [31] to compute the lengths of causal chains of _100 randomly chosen QAE pairs_ from _our dataset_ and _existing causal(-Why) VideoQA datasets_ [47, 17, 19]. Results are presented in Figure 2. We observed that the QAE pairs in _our dataset have longer causal chains than existing QA datasets_ , potentially suggesting our QAs are more complex. Identifying and solving _longer causal chains_ in the process of answering causal (why) questions involve linking together multiple cues/clues, which requires significant cognitive 

5 

**Q: Why did Jerry fall into the pudding? Q: Why did Jerry open the door? A: Tuffy ate the candle stick, causing the candle A: Jerry wanted Tom to fall & crash into the basement so (a) stick to bend & fall on Jerry. (b) that Tom would be hurt. Causal Reasoning: Cause: Tuffy eating the candle Spatial Reasoning is incorporated by understanding the stick, Effect: Jerry falling into the pudding. layout of the door & basement to explain Jerry's actions. Q: Why did Tom’s face turn red? Q: Why did Jerry pull the white line after hitting the ball? A: Tom was angry that the 3 cats were A: Jerry wanted the ball to land within the court after he (c) making fun of him. (d) hit the ball. Emotional Reasoning is used to deduce Tom's Temporal Reasoning is used to explain why Jerry did feelings from his expression. action A after action B due to the sequence of events.** 

Figure 5: **Examples of various types of reasoning required by our dataset.** _Please zoom in & view in AdobeReader to play the embedded videos._ Further types of reasoning visualized in Appendix. 

efforts on the part of VideoQA models, especially, when the clues/cues are embedded in different scenes (described above), as in our dataset. 

5. **Leveraging principles of animation** [42] (shown in Figure 1) such as _timing_ , _squash and stretch_ , _anticipation_ , _staging_ , and _exaggeration_ aid in: 1) “ _highlighting_ " _key movements_ , _emotions_ , & _storytelling_ ; 2) Consequently, greatly aid in _establishing clear cause-and-effect relationships_ and _effectively communicating them_ . These principles can be thought of as _spatiotemporal_ counterparts of _caricatures_ , which exaggerate and manipulate facial and bodily features, and these have been shown to improve facial recognition rates [37, 11, 28]. In other domains of computer vision, some works [41, 50, 49, 5] acknowledge the advantages and devise methods to _disentangle the area of interest_ , enabling more _targeted_ processing and _comprehension of image contents_ . Similarly, we hypothesize that the _principles of animation_ can provide “ _hints/guidance_ ", but the models still need to be able to _leverage this guidance to solve causal relationships_ . Overall, _causal relationships in our dataset are complex, longer_ , but at the same time they are _unambiguous/well-defined_ using principles of animation, allowing models to _focus on deciphering causal relationships_ . 

6. **Wide spectrum of types of reasoning required.** Our dataset demands various types of reasoning like 1) _Deductive_ ; 2) _Inductive_ ; 3) _Spatial_ ; 4) _Causal_ ; 5) _Critical thinking_ ; 6) _Emotion_ ; 7) _Abductive_ ; and 8) _Temporal_ . We have visualized examples demonstrating these reasoning types in Figure 5. It is also possible that a question may involve more than one type of reasoning. We conducted _human studies_ to determine the types of reasoning required by our dataset and NextQA and _observed that our dataset demands a wider range of reasoning than NextQA_ (results in Figure 2a). More information and definitions of types of reasoning and human studies are provided in Appendix. 

## **3.3 Benchmarks** 

**Question-types/Tasks.** Our dataset comprises two types of questions. The first is Multiple Choice Question Answering ( **MCQA** ). VideoQA models are provided with the question, the associated video footage, the correct answer, and four incorrect answers; the task for the models is to pick the correct answer. The second is Open-ended Answer Generation ( **OEAG** ). The models are provided only with the question and the video footage, and the models have to generate the answers in natural language. 

**Incorrect answer/option mining.** In designing MCQA, answer options should be distinct from the correct answer & each other but semantically similar enough to challenge reasoning beyond commonsense. This prevents models from taking shortcuts like simple object/action matching in vision-language spaces. Using randomly chosen answers as incorrect options is unproductive and makes the correct answer too obvious, so we avoid this strategy. Instead, we introduce these strategies: 

1. **Vanilla Hard Negative mining.** A pre-trained Sentence-Bert measures semantic similarity. For each question, we sample the top-10 other questions based on cosine similarity of Sentence-Bert embeddings. From these, we sample associated answers and again apply cosine similarity between the correct and candidate answers. Using the equation _αcossim_ ( _Qi, Qn_ ) _· βcossim_ ( _Ai, An_ ), with adjustable weights _α_ and _β_ , we select the top 4 answer candidates. To avoid overly similar answers, we penalize high similarity scores by setting _β_ to 0.001, reducing the candidate’s overall score. 

2. **Causal-Confusion Negative generation.** Causal relationship modeling is crucial for causal VideoQA, but maybe given less emphasis in existing datasets/benchmarks. We hypothesize that, relationships might not be adequately modeled by existing models, instead models may be selecting answers based on basic object/actor matching in vision-language spaces. To test our hypothesis & 

6 

underscore the relationship modeling, we introduce hard negatives in which the objects/actors remain unchanged, but the relationships between them are altered or alternate, but plausible scenarios are created. For instance, the effect is inverted ( _e_ . _g_ .“Tom was hitting Jerry" _→_ “Tom was _not_ hitting Jerry") or the causal agents are swapped ( _e_ . _g_ .“ _Tom_ was hitting _Jerry_ " _→_ “ _Jerry_ was hitting _Tom_ "). We leverage LLMs to generate causally-incorrect options. Examples can be found in the Appendix. 

**Design of Train-Val-Test splits.** We typically partition the dataset into 70% for training, and 15% each for validation and testing. Additionally, we employ specific strategies for training and testing. 1. **Uniformly Distributed** _**Seen**_ **-episode testing (UD).** Dataset samples are randomly divided into train-val-test splits without constraints on which episodes or parts are included in each split. Consequently, uniformly distributed chunks of all episodes from all seasons are likely seen during training, potentially giving VideoQA models sparse storylines. 

2. **Consecutive Partially** _**Seen**_ **-episode testing (PS).** The train set is drawn from the first 70% of each episode, the val set from the middle 70-85%, & the test set from the last 15%. This setup allows the model to have partially "seen" episodes during training & make "educated guesses" during testing. 3. _**Unseen**_ **-episode testing (UN).** In this case, a certain 70% of the episodes are reserved for train set; 15% of the remaining episodes are reserved for val set; and the remaining 15% of the episodes are used to prepare the test set. So, testing is done on entirely novel, unseen episodes and storylines. 

## **4 Experiments** 

We benchmark SOTA VideoQA methods on our new causal action QA dataset for two tasks: 1) **MCQA** and 2) **OEAG** . Then, we explore if our dataset aids real-world QA. 

## **4.1 Benchmarking on MCQA** 

**Baselines.** Following prior work [47], we benchmark the performance of these models on our causal action QA dataset: _BlindQA_ [2], _EVQA_ [2], _CoMem_ [8], _HME_ [6], _HCRN_ [16], and _HGA_ [15]. We also evaluate recent vision-language models like _MIST_ [7], _BLIP-2_ [18], and multimodal instruction tuning models _Video-LLaMA_ [57] and _VideoChat2_ [21], _GPT-4o_ [31] which excel on vision-language tasks. Further details on each model are provided in Appendix. 

**Evaluation Protocols.** Models are evaluated under these protocols, with **Accuracy** as the metric: • **Protocol 1.** Models’ ability to select the _correct Answer (_ _**A** )_ is only evaluated. Their ability to select the correct explanation is not taken into consideration. The chance accuracy in this protocol is 1 _/_ 5. Results from this protocol are presented under columns marked as **A** in Table 1. • **Protocol 2.** Models’ ability to select the _correct Answer (_ _**A** ) as well as the correct Explanation (_ _**E** )_ is evaluated. If the model can select the correct answer, but not the correct explanation, then the model is considered to have failed. In this protocol, the chance accuracy becomes 1 _/_ 5 _×_ 1 _/_ 5 = 1 _/_ 25. Results from this protocol are presented under columns marked as **A+E** in Table 1. 

**Quantitative results.** The performances of the **UD PS UN** baseline models for both protocols are presented in **Model A A+E A A+E A A+E** Table 1. In general, we observe that the performances Chance 20.00 04.00 20.00 04.00 20.00 04.00 of most of the baseline approaches are low. Most of **BlindQA** 29.38 13.07 26.51 11.54 25.13 11.02 the models gain only around 11-12% improvement **EVQACoMem** 29.3832.08 13.4813.88 31.3226.10 13.3209.89 27.8223.12 14.6509.27 over the chance accuracy for both protocols. MIST **HME** 32.35 14.02 29.53 12.36 25.13 10.22 seems to be doing exceptionally well compared to **HCRNHGA** 32.4831.40 16.9815.36 32.0129.40 14.9713.19 25.6728.23 12.2313.84 other models. We hypothesize that this could be be- **MIST** 62.22 44.88 62.22 42.86 55.91 37.90 cause MIST being geared for long-form video under- **MIST-CC[†]** 63.34 46.80 62.50 43.54 56.18 39.25 _Zero-shot_ standing, is better able to handle long video contexts, **BLIP-2** 43.67 23.32 45.88 24.18 46.64 26.48 which are frequent in our dataset. Overall, we ob- **Video-LLaMA** 35.00 11.73 35.16 9.34 29.62 9.68 served the following failure modes for models: **1) VideoChat2** 38.14 15.36 38.91 15.93 40.83 15.99 _Limited Evidence Consideration_ : Models often focus **GPT-4o 63.64 48.17 63.84 49.79 64.82 52.23** Table 1: **MCQA Results on our dataset.**[[†]] We deon a small subset of evidence, neglecting other relesign multitask version of MIST that learns to genvant cues distributed throughout the sequence, which erate causal chains as an auxiliary task. is problematic for datasets with long causal chains like ours. **2)** Models failed because they tried to exploit _shortcuts_ like object-noun or action-verb 

Table 1: **MCQA Results on our dataset.**[[†]] We design multitask version of MIST that learns to generate causal chains as an auxiliary task. 

7 

matching in video-language spaces instead of focusing on causal relationship modeling—such _shortcuts can lead to wrong predictions on our dataset due to our hard negative mining_ , correct & incorrect answers would likely contain these nouns/verbs. The most effective way to discriminate on our dataset is by inferring the causal relationship. We have discussed further shortcomings with **qualitative results** in Appendix. To mitigate these shortcomings, we design a MIST-based multitask model, **MIST-CC** , which, in addition to doing VideoQA, also learns to generate the causal chain from Video-Question features (groundtruth causal chains are generated by GPT-4 from QAE pairs). We found that this simple modification mildly boosts the performance. We have provided further details on MIST-CC in Appendix. Moving to VLM/MLLMs, we observe that they perform better than traditional models ([2, 15, 16]), except MIST. However, GPT-4o released only in late May 2024, outperformed all the models. Although, it is closed source model, we hypothesize its vision and language capabilities maybe significantly better than other VLMs. It was able to incorporate small details like facial expressions. To glean insights into it, we also asked it to give us its reasoning, and found that it analyzes each option individually and then selects the most likely answer. This is close to (or at least mimicking) how humans would approach this task. We believe this, at least on surface, seems to be going beyond just correlation-based answer picking as in non-VLM/MLLMs. 

**Comparing the two protocols** , we find that Protocol 2, where the model has to select both the correct answer and the correct explanation is significantly more difficult than selecting just the correct answer for all the models. **Comparing splits** , We observe that UD is the easiest split across all models, followed by PS, with UN being the most difficult. This intuitive order suggests that understanding past events or storylines may aid in reasoning about current events or actions. 

We further conducted an experiment where we tested the best performing models, MIST, MIST-CC, & GPT-4o, on our **Causal-Confusion** set, where incorrect answer choices have the same objects and actors as the correct answer option, but cause-and-effect relationships are reversed or altered. We observed a _significant drop_ in performance of all models as shown 

||**Negative type**<br>Vanilla Hard<br>Causal-Confusion|**MIST**<br>62.22<br> 55.80|**MIST-CC**<br>63.34<br>58.76|**GPT-4o**<br>63.64<br>54.95|
|---|---|---|---|---|



Table 2: **Impact of causal-confusion.** 

in Table 2. This could potentially be due to _causal relationships being not modeled adequately_ even by such advanced VideoQA models. Performance of MIST-CC dropped relatively less, potentially because the auxiliary task of generating causal chains may have enhanced the understanding of causal relationships. Interestingly, we also found the S-BERT similarity score to be above 90% for an action ( _e_ . _g_ ., Tom is running after Jerry) and its Causal-Confusion version ( _e_ . _g_ ., Tom is _not_ running after Jerry); while these sentences would be opposite/different in terms of human perception. We believe that _Causality might be more overlooked than we think in various fields, not just in computer vision_ . 

## **4.2 Benchmarking on Open-ended Answer Generation (OEAG)** 

**CapsMIX Performance metric.** We measure the performance of OEAG in terms of BLEU1,2,3 [33], METEOR [3], ROUGE [23], SPICE [1], CIDEr [44] & Sentence-BERT [36] scores by comparing with the ground truth answers as done by captioning & QA literature. The wide range of metrics complicates model comparison, so we introduce Caps-MIX (Captioning Metrics Integration eXpert), which normalizes and integrates all scores into a single metric, simplifying comparisons and combining the unique strengths of individual metrics. 

**Baselines.** Following prior work [47], we benchmarked the performance of _EVQA_ [2]; _UATT_ [51]; _HME_ [6]; _HGA_ [15] on OEAG task. We also report the zeroshot performance of recent multimodal video understanding models _BLIP-2_ [18], _Video-LLaMA_ [57], _VideoChat2_ [21], _GPT-4o_ [31]. 

|**Model**<br>**BlindQA**<br>**UATT**<br>**HME**<br>**HGA**<br>**BlindGPT-2**<br>**VisionGPT-2**|**CapsMIX**|
|---|---|
||2.7646<br>3.3928<br>3.0975<br>3.7872<br>6.6006<br>6.7582|
|**BLIP-2**<br>**Video-LLaMA**<br>**VideoChat2**<br>**GPT-4o**|1.8931<br>2.4464<br>**3.9524**<br>2.9851|



We evaluated the baselines for generating answers and explanations. The **VisionGPT-2** 6.7582 performances of various models are summarized in Table 3. For full results, **BLIP-2** 1.8931 see Appendix. Overall, we observe that models struggle significantly with **Video-LLaMA** 2.4464 **VideoChat2 3.9524** open-ended generation, including some recent VLMs and MLLMs. Despite **GPT-4o** 2.9851 GPT-4o’s strong performance on MCQA, it also falters on OEAG. These Table 3: **OEAG re-** models likely perform better on MCQA by eliminating incorrect choices, but **sults on our dataset** they fail to genuinely understand videos and perform causal reasoning for **(UD split).** standalone answer generation. VideoChat2 seem to be doing exceptionally well, perhaps, because it was specifically trained on various video understanding tasks and datasets, 

Table 3: **OEAG results on our dataset (UD split).** 

8 

including causal reasoning task. Comparing split-type wise, we found that as in the case of MCQA, UD was the easiest split. 

Inspired by the success of LLMs, we experimented with leveraging GPT-2 [35], a publicly available LLM, as our natural language answer generator. We found that a pre-trained version out of the box did not work well—it generated mostly random, unrelated words. However, upon simple training on the train set, it performed significantly better than the baselines we considered as shown in Table 3 (BlindGPT-2). However, it is unclear if these LLMs have “seen" Tom and Jerry scripts during their pretraining stage. If so, then pretraining on relevant scripts followed by finetuning on our dataset could potentially be a reason for GPT-2’s good performance. Although this is less likely. What is more likely to be the reason behind this good performance is the language modeling/generating capability of GPT-2. 

In the next step, we integrated visual information into LLM. Taking inspiration from [30], we learn a projector network to align the visual features with GPT-2 representations (VisualGPT-2 in Table 3. We observed an improvement in the performance. However, we believe that this might not be a very efficient way to integrate visual information with LLMs. We expect the performance to boost considerably through better and more sophisticated joint modeling vision and language. Overall, we believe shallower networks might not have the capacity to do inference over longer, complex causal chains, and as such it might not be the best option to invest future efforts into; LLMs have the potential to excel at causal reasoning; this view is also supported in a concurrent survey [25]. 

## **4.3 Does our dataset help with real-world cases?** 

To evaluate the direct impact of our dataset, we combine our training set with NextQA’s training set and measure model performance on the NextQA test set. For comparison, we also measure model performance without incorporating our dataset. Despite being a _synthetic/cartoon_ dataset, and _much smaller in size_ , our dataset boosted the performance on a real-world dataset on both MCQA and OEAG (Table 4). We observed _improvement in identifying the correct causes by breaking the reliance on shortcuts and focus on causal effects_ ; a more comprehensive analysis of the situation/interaction, rather than 

||**Train**||**MCQA**|**OEAG**|
|---|---|---|---|---|
||**Data**|**BlindQA HGA MIST **||**HGA**|
||NextQA|32.73|41.23 55.44|1.44|
||**+**Ours|**32.91**|**41.44 55.96**|**1.48**|



Table 4: **Our dataset improves performance on existing realworld dataset.** For OEAG, WUPS score is reported. 

jumping to a conclusion. Qualitative results provided in Appendix. What is more, improvements were not limited to ‘causal-why’ questions but also extended across other question types. We hypothesize that this can attributed to the wider range of reasoning involved in our dataset. To the best of our knowledge, this is the first time a synthetic VideoQA dataset has shown immediate improvement on a real-world dataset. However, we noticed a slight drop in the performance on location-type questions. Thus, we believe that it might be better to: **1)** _transfer reasoning skills acquired from challenging synthetic datasets like ours_ ; **2)** leverage our synthetic dataset to _inform the model designing process_ , as it better reflects the challenges VideoQA models may face in the real world, such as longer causal chains and frequent scene changes in visual streams. Nonetheless, we _do not suggest naively deploying whole models/weights trained on our dataset to real-world scenarios or applications_ . 

## **5 Conclusion** 

We introduced CausalChaos!, a challenging dataset for causal action question-answering tasks based on the classic Tom and Jerry cartoon, richly annotated with critical thinking questions requiring extensive reasoning from Video QA models. Questions come with multi-level answers and explanations covering the entire video context. We also provide the novel CausalConfusion test set to challenge causal relationship modeling in Video QA models. Our experiments show that while existing models perform well on causal action QA tasks, there is significant room for improvement in causal relationship modeling and generating detailed open-ended answers. LLMs show promise, but integrating visual information with LLMs or joint modeling of vision and language is crucial. We hope our dataset fosters such developments and will release it and the codes to support future efforts. Lastly, we demonstrated improvements in real-world datasets. Our dataset, derived from cartoons, should inform model design, reflecting real-world challenges like longer causal chains and frequent scene changes. However, we do not suggest deploying models trained solely on our dataset in real-world scenarios. 

9 

**Material Acknowledgement and Disclaimer.** Tom and Jerry is a material of Turner Entertainment Company (Warner Bros. Entertainment Inc.). All rights reserved. We do not claim any ownership of or rights to the Tom and Jerry material. All other trademarks, service marks, trade names and any other material referenced in this document are the property of their respective owners. 

**Acknowledgements.** This research/project is supported by the National Research Foundation, Singapore, under its NRF Fellowship (Award# NRF-NRFF14-2022-0001). 

## **References** 

- [1] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In _Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part V 14_ , pages 382–398. Springer, 2016. 8 

- [2] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In _Proceedings of the IEEE international conference on computer vision_ , pages 2425–2433, 2015. 7, 8 

- [3] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In _Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization_ , pages 65–72, 2005. 8 

- [4] Andrei Barbu, David Mayo, Julian Alverio, William Luo, Christopher Wang, Dan Gutfreund, Josh Tenenbaum, and Boris Katz. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. _Advances in neural information processing systems_ , 32, 2019. 5 

- [5] Zheng Ding, Jieke Wang, and Zhuowen Tu. Open-vocabulary universal image segmentation with maskclip. In _Proceedings of the 40th International Conference on Machine Learning_ , pages 8090–8102, 2023. 6 

- [6] Chenyou Fan, Xiaofan Zhang, Shu Zhang, Wensheng Wang, Chi Zhang, and Heng Huang. Heterogeneous memory enhanced multimodal attention model for video question answering. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 1999–2007, 2019. 7, 8 

- [7] Difei Gao, Luowei Zhou, Lei Ji, Linchao Zhu, Yi Yang, and Mike Zheng Shou. Mist: Multi-modal iterative spatial-temporal transformer for long-form video question answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 14773–14783, 2023. 7 

- [8] Jiyang Gao, Runzhou Ge, Kan Chen, and Ram Nevatia. Motion-appearance co-memory networks for video question answering. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 6576–6585, 2018. 7 

- [9] Madeleine Grunde-McLaughlin, Ranjay Krishna, and Maneesh Agrawala. Agqa: A benchmark for compositional spatio-temporal reasoning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 11287–11297, 2021. 3 

- [10] Sergio Guadarrama, Niveda Krishnamoorthy, Girish Malkarnenkar, Subhashini Venugopalan, Raymond Mooney, Trevor Darrell, and Kate Saenko. Youtube2text: Recognizing and describing arbitrary activities using semantic hierarchies and zero-shot recognition. In _Proceedings of the IEEE international conference on computer vision_ , pages 2712–2719, 2013. 3 

- [11] Subin Han, Zekun Sun, and Chaz Firestone. Caricaturing shapes in visual memory. _Journal of Vision_ , 23 (9):4756–4756, 2023. 6 

- [12] Yun He, Soma Shirakabe, Yutaka Satoh, and Hirokatsu Kataoka. Human action recognition without human. In _Computer Vision–ECCV 2016 Workshops: Amsterdam, The Netherlands, October 8-10 and 15-16, 2016, Proceedings, Part III 14_ , pages 11–17. Springer, 2016. 5 

- [13] Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. Tgif-qa: Toward spatio-temporal reasoning in visual question answering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2758–2766, 2017. 3 

- [14] Yunseok Jang, Yale Song, Chris Dongjoo Kim, Youngjae Yu, Youngjin Kim, and Gunhee Kim. Video question answering with spatio-temporal reasoning. _International Journal of Computer Vision_ , 127: 1385–1412, 2019. 3 

- [15] Pin Jiang and Yahong Han. Reasoning with heterogeneous graph alignment for video question answering. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 11109–11116, 2020. 7, 8 

10 

- [16] Thao Minh Le, Vuong Le, Svetha Venkatesh, and Truyen Tran. Hierarchical conditional relation networks for video question answering. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 9972–9981, 2020. 7, 8 

- [17] Jiangtong Li, Li Niu, and Liqing Zhang. From representation to reasoning: Towards both evidence and commonsense reasoning for video question-answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2022. 2, 3, 5 

- [18] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In _ICML_ , 2023. 7, 8 

- [19] Jiapeng Li, Ping Wei, Wenjuan Han, and Lifeng Fan. Intentqa: Context-aware video intent reasoning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , pages 11963–11974, 2023. 2, 3, 5 

- [20] Junyi Li, Jie Chen, Ruiyang Ren, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. The dawn after the dark: An empirical study on factuality hallucination in large language models. _arXiv preprint arXiv:2401.03205_ , 2024. 5 

- [21] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. 2024. 2, 7, 8 

- [22] Xiaochuan Li, Baoyu Fan, Runze Zhang, Liang Jin, Di Wang, Zhenhua Guo, Yaqian Zhao, and Rengang Li. Image content generation with causal reasoning. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 13646–13654, 2024. 4 

- [23] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In _Text summarization branches out_ , pages 74–81, 2004. 8 

- [24] Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. A survey on hallucination in large vision-language models. _arXiv preprint arXiv:2402.00253_ , 2024. 5 

- [25] Xiaoyu Liu, Paiheng Xu, Junda Wu, Jiaxin Yuan, Yifan Yang, Yuhang Zhou, Fuxiao Liu, Tianrui Guan, Haoliang Wang, Tong Yu, et al. Large language models and causal inference in collaboration: A comprehensive survey. _arXiv preprint arXiv:2403.09606_ , 2024. 9 

- [26] Tegan Maharaj, Nicolas Ballas, Anna Rohrbach, Aaron Courville, and Christopher Pal. A dataset and exploration of models for understanding video data through fill-in-the-blank question-answering. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 6884–6893, 2017. 3 

- [27] Jiayuan Mao, Xuelin Yang, Xikun Zhang, Noah Goodman, and Jiajun Wu. Clevrer-humans: Describing physical and causal events the human way. _Advances in Neural Information Processing Systems_ , 35: 7755–7768, 2022. 3 

- [28] Robert Mauro and Michael Kubovy. Caricature and face recognition. _Memory & Cognition_ , 20:433–440, 1992. 6 

- [29] Laurene K Meringoff, Martha M Vibbert, Cynthia A Char, David E Fernie, Gail S Banker, and Howard Gardner. How is children’s learning from television distinctive? exploiting the medium methodologically. _Children’s understanding of television: Research on attention and comprehension_ , pages 151–180, 1983. 1 

- [30] Ron Mokady, Amir Hertz, and Amit H Bermano. Clipcap: Clip prefix for image captioning. _arXiv preprint arXiv:2111.09734_ , 2021. 9 

- [31] OpenAI. Hello gpt-4o. `https://openai.com/index/hello-gpt-4o/` , 2024. [Online; accessed 31-May-2024]. 5, 7, 8 

- [32] team OpenAI. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 3 

- [33] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th annual meeting of the Association for Computational Linguistics_ , pages 311–318, 2002. 8 

- [34] Georgios Pavlakos, Jitendra Malik, and Angjoo Kanazawa. Human mesh recovery from multiple shots. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 1485–1495, 2022. 3, 5 

11 

- [35] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9, 2019. 9 

- [36] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. _arXiv preprint arXiv:1908.10084_ , 2019. 8 

- [37] Gillian Rhodes, Susan Brennan, and Susan Carey. Identification and ratings of caricatures: Implications for mental representations of faces. _Cognitive Psychology_ , 19(4):473–497, 1987. 6 

- [38] Wolfgang Schnotz and Thorsten Rasch. Functions of animation in comprehension and learning. _Learning with animation: Research implications for design_ , pages 93–113, 2008. 1 

- [39] Mairaru Shreesha and Sanjay Kumar Tyagi. Does animation facilitate better learning in primary education? a comparative study of three different subjects. _Creative Education_ , 7(13):1800–1809, 2016. 1 

- [40] Laı Chin Siong, Yunn Ong, Fatin Aliah Phang, and Jaysuman Pusppanathan. The use of concept cartoons in overcoming the misconception in electricity concepts. _Participatory Educational Research_ , 10(1):310–329, 2023. 1 

- [41] Zeyi Sun, Ye Fang, Tong Wu, Pan Zhang, Yuhang Zang, Shu Kong, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. Alpha-clip: A clip model focusing on wherever you want. _arXiv preprint arXiv:2312.03818_ , 2023. 6 

- [42] Frank Thomas and Ollie Johnston. _The illusion of life : Disney animation_ . Disney Editions, 1981. 2, 6 

- [43] Tom and Jerry Wiki Fandom. Tom and jerry wiki fandom. 4 

- [44] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 4566–4575, 2015. 8 

- [45] Lawson LS Wong, Leslie Pack Kaelbling, and Tomás Lozano-Pérez. Data association for semantic world modeling from partial views. _International Journal of Robotics Research_ , 34(7):1064–1082, 2015. 5 

- [46] Bo Wu, Shoubin Yu, Zhenfang Chen, Joshua B Tenenbaum, and Chuang Gan. STAR: A benchmark for situated reasoning in real-world videos. In _Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS)_ , 2021. 2, 3 

- [47] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 9777–9786, 2021. 2, 3, 5, 7, 8 

- [48] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In _Proceedings of the 25th ACM international conference on Multimedia_ , pages 1645–1653, 2017. 3 

- [49] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Openvocabulary panoptic segmentation with text-to-image diffusion models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 2955–2966, 2023. 6 

- [50] Mengde Xu, Zheng Zhang, Fangyun Wei, Han Hu, and Xiang Bai. Side adapter network for openvocabulary semantic segmentation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 2945–2954, 2023. 6 

- [51] Hongyang Xue, Zhou Zhao, and Deng Cai. Unifying the video and question attentions for open-ended video question answering. _IEEE Transactions on Image Processing_ , 26(12):5656–5666, 2017. 8 

- [52] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. _arXiv preprint arXiv:1910.01442_ , 2019. 3 

- [53] Khoo Yin Yin and Robert Fitzgerald. Peer learning with concept cartoons enhance critical thinking and performance in secondary school economics. _Journal of economics and economic education research_ , 18 (1):1–13, 2017. 1 

- [54] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 9127–9134, 2019. 3 

12 

- [55] Amir Zadeh, Michael Chan, Paul Pu Liang, Edmund Tong, and Louis-Philippe Morency. Social-iq: A question answering benchmark for artificial social intelligence. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8807–8817, 2019. 3 

- [56] Kuo-Hao Zeng, Tseng-Hung Chen, Ching-Yao Chuang, Yuan-Hong Liao, Juan Carlos Niebles, and Min Sun. Leveraging video descriptions to learn video question answering. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , 2017. 3 

- [57] Hang Zhang, Xin Li, and Lidong Bing. Video-LLaMA: An instruction-tuned audio-visual language model for video understanding. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_ , pages 543–553, Singapore, 2023. Association for Computational Linguistics. 7, 8 

- [58] Zhou Zhao, Zhu Zhang, Shuwen Xiao, Zhou Yu, Jun Yu, Deng Cai, Fei Wu, and Yueting Zhuang. Openended long-form video question answering via adaptive hierarchical reinforced networks. In _Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, IJCAI-18_ , pages 3683–3689. International Joint Conferences on Artificial Intelligence Organization, 2018. 3 

- [59] Yaoyao Zhong, Junbin Xiao, Wei Ji, Yicong Li, Weihong Deng, and Tat-Seng Chua. Video question answering: Datasets, algorithms and challenges. _arXiv preprint arXiv:2203.01225_ , 2022. 3 

- [60] Yuhang Zhou, Paiheng Xu, Xiaoyu Liu, Bang An, Wei Ai, and Furong Huang. Explore spurious correlations at the concept level in language models for text classification. _arXiv preprint arXiv:2311.08648_ , 2023. 5 

- [61] Linchao Zhu, Zhongwen Xu, Yi Yang, and Alexander G Hauptmann. Uncovering the temporal context for video question answering. _International Journal of Computer Vision_ , 124:409–421, 2017. 3 

13 

