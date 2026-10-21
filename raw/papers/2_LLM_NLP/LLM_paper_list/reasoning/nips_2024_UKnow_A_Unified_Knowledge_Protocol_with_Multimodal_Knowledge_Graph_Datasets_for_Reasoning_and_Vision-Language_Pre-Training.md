# **UKnow: A Unified Knowledge Protocol with Multimodal Knowledge Graph Datasets for Reasoning and Vision-Language Pre-Training** 

**Biao Gong**[1] **, Shuai Tan**[2] **, Yutong Feng**[1] **, Xiaoying Xie**[1] **, Yuyuan Li**[3] _[,]_[4] _[†]_ **, Chaochao Chen**[4] **, Kecheng Zheng**[2] **, Yujun Shen**[2] **, Deli Zhao**[1] **,** 1Alibaba Group, 2Ant Group, 3Hangzhou Dianzi University, 4Zhejiang University 

```
{a.biao.gong,tanshuai2001,fengyutong.fyt}@gmail.comsouyu.xxy@alibaba-inc.com
y2li@hdu.edu.cnzjuccc@zju.edu.cn{zkechengzk,shenyujun0302,zhaodeli}@gmail.com
```

## **Abstract** 

This work presents a unified knowledge protocol, called _**UKnow**_ , which facilitates knowledge-based studies from the perspective of data. Particularly focusing on visual and linguistic modalities, we categorize data knowledge into five unit types, namely, in-image, in-text, cross-image, cross-text, and image-text, and set up an efficient pipeline to help construct the multimodal knowledge graph from any data collection. Thanks to the logical information naturally contained in knowledge graph, organizing datasets under _UKnow_ format opens up more possibilities of data usage compared to the commonly used image-text pairs. Following _UKnow_ protocol, we collect, from public international news, a large-scale multimodal knowledge graph dataset that consists of 1,388,568 nodes (with 571,791 visionrelated ones) and 3,673,817 triplets. The dataset is also annotated with rich event tags, including 11 coarse labels and 9,185 fine labels. Experiments on 4 benchmarks demonstrate the potential of _UKnow_ in supporting common-sense reasoning and boosting vision-language pre-training with a single dataset, benefiting from its unified form of knowledge organization. See Appendix A to download the dataset. 

## **1 Introduction** 

Recent efforts have been attracted to leverage the _multimodal knowledge graph_ [97] for data-driven intelligence. Inspired by the human mastery knowledge network [50], we consider that the multimodal knowledge graph, which naturally accommodates heterogeneous data based on its format of complex network [95, 78], is well suited for constructing a unified knowledge criterion from the perspective of data. Driven by the multimodal knowledge graph, models can easily introduce external knowledge [58], discover long-range relations [83] and understand more logical semantics [53]. However, existing datasets of the multimodal knowledge graph commonly focus on only one task like common-sense reasoning [82, 47] due to their limited scale and irregular data organization. Therefore, it is imperative to construct a well-organized multimodal knowledge graph dataset with large-scale and rich-logic, which enables delving into deeper foundational problems in lower layers, such as the knowledge based vision-language pre-training. 

To this end, we propose _**UKnow**_ , a **U** nified **Know** ledge protocol, which facilitates knowledge-based studies [56, 45, 89] from the perspective of data. Particularly focusing on visual and linguistic modalities, we categorize data knowledge into five unit types, namely, in-image _Iin_ , in-text _Tin_ , cross-image _Icross_ , cross-text _Tcross_ , and image-text _ITcross_ . As shown in Fig. 1, these knowledge types are together named as _Knowledge-View_ which can be easily used to construct a multimodal knowledge graph ( **G** _m_ ). 

_†_ Corresponding Author. 

To verify that _UKnow_ can serve as a standard protocol, we further set up an efficient data processing pipeline, consisting of _Phase-1/2/3_ , to reorganize existing datasets into _UKnow_ ’s format. Please note that, this pipeline is also able to automatically extend an existing image-text dataset like LAION-5B [60] with more useful information to build a new dataset. A brief description of each _Phase_ is as follows: 

_**Phase-1:** Content Extraction_ . We use pretrained models to preprocess data and extract useful content. Note that pre-trained models can be replaced / added / disabled freely as needed. _**Phase-2:** Information Symbolization_ . Since the results obtained in _Phase-1_ ( _e.g._ , images and texts) cannot be used directly for graph construction, we adopt information symbolization strategy to arrange all of them into the index in this phase. This information symbolization strategy numbers all original or generated data by a 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training_images/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training.pdf-0002-02.png)


**----- Start of picture text -----**<br>
Iin Icross Iin<br>ITcross Gm ITcross<br>Tin Tcross Tin<br>Fishermen reported seeing the Granville-based Alizé 3 on marine-tracker websites<br>working in a bream-spawning ground off the north coast. Some attempted to stop the<br>boat while the Norman Le Brocq fisheries protection vessel was also deployed.<br>In-Image<br>Content Extraction<br>hase-1P Phase2-<br>Multi-modality<br>Graph Construction<br>In-Text<br>Content Extraction<br>Event 3Phase-<br> Annotation<br>**----- End of picture text -----**<br>


Figure 1: **Overview of** _**UKnow**_ **protocol** , consisting of five unit knowledge types, namely, inimage _Iin_ ( _e.g._ , object), in-text _Tin_ ( _e.g._ , entity), cross-image _Icross_ ( _e.g._ , image similarity), cross-text _Tcross_ ( _e.g._ , text continuity), and imagetext _ITcross_ ( _e.g._ , description). 

certain rule, which links the nodes from _Phase-1_ to make a multimodal graph. _**Phase-3:** Knowledge Construction_ . Two kinds of internal knowledge ( _Iin, Tin_ ) and three kinds of associative knowledge ( _Icross, Tcross, ITcross_ ) are aggregated into one graph ( **G** _m_ ) in this phase as shown in Fig. 1. 

Following _UKnow_ protocol and above pipeline, we build a novel large-scale multimodal knowledge graph. Considering that a large-scale event dataset is of practical significance for real-world applications, such as information retrieval and public sentiment analysis, our data are collected from public international news. Overall, our dataset contains 1,388,568 nodes of which 571,791 are vision relevant ( _i.e._ , news images or visual objects). The number of triples in the entire graph is 3,673,817. To the best of our knowledge, this dataset has become the largest multimodal knowledge graph dataset of international news events. Moreover, to organize data in a more structured way and enhance dataset with more category labels, our dataset introduces a _hierarchical event annotation_ for each news, including _**Event-11**_ and _**Event-9185**_ . Specifically, the former contains general event categories such as _“Sports, Ceremony, ...”_ , while the latter consists of real human activity in the history such as _“2019 NBA All-Star Game, 2019 Daytona 500, ...”_ . More details about the annotation are shown in Sec. 3.2, Fig 3, and Tab. 3. 

In summary, our **contributions** are as follows: 

- We propose _UKnow_ to introduce the multimodal knowledge graph into the vision field as a new standard of data organization, which features the relation inside data in addition to the original data format. Such a protocol opens up the possibilities of data usage such that more logic-rich downstream tasks can be expected in the future. 

- We design an efficient data processing pipeline for constructing dataset following our _UKnow_ protocol, together with a large-scale multimodal knowledge graph dataset collected from public international news. We also equip the dataset with hierarchical event annotations, which can help models understand human activities and history. See Appendix A to download the dataset. 

- We provide some examples of the usage of _UKnow_ in practical applications. Experiments on four benchmarks showcase the advantages of _UKnow_ in supporting common-sense reasoning and boosting vision-language pre-training with a unified form of data organization, making it possible to evaluate various tasks on a single dataset. 

## **2 Related Work** 

## **2.1 Existing Knowledge Representation Formats** 

In recent years, a growing abundance multi-modal data are disseminated, linking diverse information across various modalities such as text and image in a global data space. This interconnected web of heterogeneous data constitutes a vast repository of information termed as knowledge. With the development of large-scale models, the utilization of knowledge has seen a notable surge in 

2 

Table 1: **Statistics** of various multimodal knowledge graph datasets. **TRIPLE** is the basic component of knowledge graph (Sec. 2.1), **WEB** and **GIT** indicate homepage and Github repository respectively. **EVENT** indicates the news event. 

|**DATASET**<br>**YEAR**<br>**MULTIMODAL INFO.**<br>**SOURCE**|**NODE**<br>**IMAGE**<br>**TRIPLE**|**WEB**<br>**GIT**|**EVENT**|
|---|---|---|---|
|WN9-IMG-TXT [82]<br>2016<br>ENT.<br>WN18, ImageNet<br>ImageGraph [48]<br>2017<br>ENT./CONCEPT<br>FB15k<br>VisualGenome [26]<br>2017<br>ENT.<br>MSCOCO<br>GAIA [94]<br>2018<br>ENT./CONCEPT<br>Freebase, Geonames<br>MMKG-FB15k [38]<br>2019<br>ENT./CONCEPT<br>FB15k, Search Engine<br>MMKG-DB15k [38]<br>2019<br>ENT./CONCEPT<br>DB15k, Search Engine<br>MMKG-YAGO15k [38]<br>2019<br>ENT./CONCEPT<br>YAGO15k, Search Engine<br>Richpedia [73]<br>2020<br>ENT./REL./CONCEPT<br>Wikipedia<br>VisualSem [1]<br>2020<br>ENT./CONCEPT<br>BabelNet<br>RESIN[80]<br>2021<br>ENT./REL./CONCEPT<br>News|6,555<br>63,225<br>14,397<br>14,870<br>829,931<br>564,010<br>75,729<br>108,077<br>1,531,448<br>457,000<br>-<br>38,000<br>14,951<br>13,444<br>592,213<br>14,777<br>12,842<br>99,028<br>15,283<br>11,194<br>122,886<br>29,985<br>2,914,770<br>2,708,511<br>89,896<br>930,000<br>1,500,000<br>51,422<br>6,399<br>150,220|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>✓|
|MKG-W [84]<br>2022<br>ENT./REL./CONCEPT<br>Open EA [68], Search Engine<br>MKG-Y [84]<br>2022<br>ENT./REL./CONCEPT<br>Open EA, Search Engine<br>MMKB-DB15K[84]<br>2022<br>ENT./REL./CONCEPT<br>Open EA,Search Engine|15,000<br>14,463<br>-<br>15,000<br>14,244<br>-<br>12,842<br>12,818<br>-||�<br>�<br>�|
|MarKG [93]<br>2023<br>ENT./CONCEPT<br>Wikidata, Search Engine<br>Multi-OpenEA [36]<br>2023<br>ENT./CONCEPT<br>Open EA, Search Engine<br>UMVM [6]<br>2023<br>ENT./CONCEPT<br>DBpedia, Multi-OpenEA<br>AspectMMKG [92]<br>2023<br>ENT./CONCEPT<br>Wikipedia, Search Engine<br>TIVA-KG [75]<br>2023<br>ENT./REL./CONCEPT<br>Wikipedia, Search Engine<br>VTKG-C[29]<br>2023<br>ENT./CONCEPT<br>ConceptNet,WordNet|11,292<br>76,424<br>34,420<br>920,000<br>2,705,688<br>-<br>238,208<br>1,073,671<br>982,626<br>2,380<br>645,456<br>-<br>443,580<br>1,695,688<br>1,382,358<br>43,267<br>461,007<br>111,491|✓<br>✓<br>✓<br>✓<br>✓|�<br>�<br>�<br>�<br>�<br>�|
|**_UKnow_**<br>2024<br>ENT./REL./CONCEPT<br>News, Wikipedia|**1,388,568**<br>**1,073,671**<br>**3,673,817**|✓<br>✓|✓|



exploration. Existing knowledge-based deep learning models are broadly divided into two aspects: (1) external knowledge introduction [12], (2) internal knowledge mining [22]. The former leverages expert knowledge by introducing external data [44, 28, 4] or pre-trained models [77, 59, 11, 87]. The latter means constructing correlations of training data by similarity [49, 13, 17] or discovering favorable substructures of internal models [32, 7, 33, 79]. 

However, from the perspective of data organization, existing studies often claim to be knowledgebased only using one piece of them, which is actually incomplete and cannot be analogous to the complex knowledge network held by humans. In this work, we build a unified knowledge protocol based on the multimodal knowledge graph to define the unified knowledge on multimodal data. 

## **2.2 Multimodel Knowledge Graph Datasets** 

The Multimodal Knowledge Graph (MMKG) serves as a potent means to store and leverage multimodal knowledge explicitly, which bolsters and enhances model performances across diverse domains. In Tab. 1, we list mainstream multimodal knowledge graph datasets [73, 48, 26, 1, 94, 82, 38, 80, 84, 36, 6, 92, 75, 90, 29], constructed by texts and images with detailed information. In terms of data scale, VisualGenome [26] is a multimodal knowledge graph which contains 40,480 relations, 108,077 image nodes with objects. The ImageGraph [48] further pushed up the number of image nodes to 829,931 but missing the extraction of visual objects. Recently, VisualSem [1] implements a multimodal knowledge graph with 938 _K_ image nodes and 89,896 entity nodes, but it only uses 15 types of relation to build the graph. On the route of increasing the number of entity nodes, while Multi-OpenEA [36] boasts 920,000 entity nodes, surpassing prior methods, our endeavor has achieved 1,388,568 nodes, establishing the largest graph thus far. Besides, most of existing multimodal knowledge graphs are more like a vision-similarity-based image library [40, 66] with image descriptions and meta information, it lacks the most valuable feature of the knowledge graph: “The Logical Connection”. This logic refers to the additional association between two nodes that were originally unrelated, triggered by a news event involving these two nodes. For example, prior to the news event "Celebrity 1 visits Area 1," there was no relation between Celebrity 1 and the Area 1. The newly added "visit" relation in _<_ (“ `Celebrity1` ”), `visit` , (“ `Area1` ”) _>_ tuple exemplifies this logic, which is highly beneficial for downstream tasks. 

Generally speaking, the above news refer to international news, which carries the most complex event logic as well as plentiful multimodal information [76]. To completely exploit the advantages of multimodal knowledge graphs, building a dataset using event logic from international news is a natural approach. However, there is not yet a large multimodal knowledge graph of news events. RESIN [80] is a recently published multimodal knowledge graph containing 24 types of entities, 46 types of relations and 67 types of events. The larger and fresher CLIP-Event [33] is a event rich dataset with 106,875 images and 187 types of events extracted by a text information extraction system [94, 37]. Actually, CLIP-Event is not a knowledge graph and its definition of “event” is not a news event but an action. In summary, one of goals of our work is to build a large, and realistic news-event rich, multimodal knowledge graph dataset from international news. 

3 

## **2.3 Knowledge-based Downstream Tasks** 

Thanks to the innovative unified knowledge proposed by our _UKnow_ protocol, our dataset can readily accommodate a variety of downstream tasks. In this study, we opt for common-sense reasoning and vision-language pre-training as experimental domains to validate our dataset. Common-sense reasoning is an extremely popular task in the field of knowledge graph. Since our dataset is based on the knowledge graph, the performance validation on common-sense reasoning is indispensable. Moreover, the representations from Vision-Language Pre-training models are capable of diminishing the necessity for intricate task-specific architectures [9], which allows the knowledge to further flow into various downstream tasks. By incorporating these two tasks, we are able to maximize the assessment of the dataset’s knowledge validity. 

**Common-sense Reasoning.** Common-sense reasoning means answering queries by logic permutations. The specific task in this work is the link prediction. Various works [3, 71, 69, 61, 96, 54] achieve reasoning by embedding entities and relations in knowledge graph into low-dimensional vector space. Path-based methods [27, 83, 64, 52] start from anchor entities and determine the answer set by traversing the intermediate entities via relational path. There are also GCN [25] based methods [62, 16] pass message to iterate graph representation for reasoning. 

**Vision-Language Pre-training** Vision-language pre-training (VLP) can be divided into three categories based on how they encode images [10]: OD-based region features [5, 31, 34, 41, 67, 70], CNN-based grid feature [63, 19, 20] and ViT-based patch features [85, 30, 24]. Pre-training objectives are usually: masked language/image modeling (MLM/MIM) [2, 9, 39], image-text matching (ITM) [34, 19, 10], and image-text contrastive learning (ITC) [30, 51, 35]. 

## **3 UKnow** 

We commence by introducing the overall architecture of _UKnow_ in Sec. 3.1. Then the detailed exposition of the data collection process for the new dataset and statistics are presented in Sec. 3.2 and Sec. 3.3. In Sec. 4, we lastly provide the guidance to researchers on how to integrate the multimodal knowledge graph and effectively design a UKnow-based model. 

Compared to previous libraries-like methods [40, 66] with simple descriptions and meta-information, which lack the logical connection, the most valuable feature of our data processing pipeline is to endow with more logical connections to achieve superior performance in various tasks. As shown in Fig. 2, particularly focusing on visual and linguistic modalities, we categorize data knowledge into five unit types. Then we devise an efficient data processing pipeline to help reorganize existing datasets or create a new one under _UKnow_ format. The construction process of _UKnow_ can be invoked separately for any multimodal data to standardize the knowledge. As shown in Fig. 3, the whole pipeline is mainly empowered by three parts: _content extraction_ , _information symbolization_ , and _knowledge construction._ 

## **3.1 Construction Pipeline for UKnow Protocol** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training_images/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training.pdf-0004-08.png)


4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training_images/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training.pdf-0005-00.png)


**----- Start of picture text -----**<br>
in-img OBJ. PATCH:SEG.:BBOX:    SCORE:       [[0,1,1,1,0,…], […]][[0.99], [0.89], … , [0.54]][[151,4,405,217], [ … ] …][[tensor], [tensor]…] Det. / Seg. A large boat and a smaller boat in the water.A large boat is in the water near a smaller boat.A boat is sailing in the water near the shore.A boat is docked in the water near a dock.Generated Captions EncoderEncoderImage Image  FeaturesN x 5121 x 512<br>in-txt French trawler caught in protected Jersey waters NERPOS French  trawler caught in protected Jersey  watersFrench  trawler caught in protected Jersey  waters[‘NORP’, ‘VEHICLE’, ‘GPE’] ['ADJ','NOUN','VERB','ADP','ADJ','PROPN','NOUN'] Fishermen reported seeing Granville-based French trawler caught in protected Jersey waters. Jersey TrawlerFrench ... CaughtWaters ... EncoderText  FeaturesM x 5121 x 512<br>(a) Iin and  Tin .<br>(b) Icross ,  Tcross and  ITcross . (c) Illustration of the complete UKnow.<br>**----- End of picture text -----**<br>


Figure 2: **Detailed data organization under** _**UKnow**_ **protocol** , which builds the multimodal (image & text) graph **G** _m_ based on the _Knowledge-View_ ( _Iin_ , _Tin_ , _Icross_ , _Tcross_ , and _ITcross_ ). Each node owns up to 22 attributes shown as _Np_ in Fig. 3. 

original output _Np[ori]_ into a K:V dictionary _Np_ . The KEY of _Np_ are shown in top right corner of Fig. 3 ( _Np_ [ _Phase-1_ ]). _Np_ is also used as the attribute of each node in the final output multimodal knowledge graph **G** _m_ . 

**Phase-2: Information Symbolization.** Since Images and texts cannot be used directly for graph construction, we design the _Phase-2_ to number all original or generated data by a certain rule, then _Phase-3_ links these nodes to make a multimodal graph. _Information Symbolization_ is used to subscript _Np_ to edge index **N** _e_ or node index **N** _n_ : (1) The symbolization for edges **N** _e_ is based on the category or visual / semantic similarity. For example, “[111] title_title_clip” is a kind of parallelism edge which is constructed by the cosine similarity of clip features of news titles. (2) The symbolization for nodes **N** _n_ is divided into three levels: [fact, image / text, object / entity]. As shown in Fig. 3, [ _L_ 1 _.∗_ ] means fact-level which is an abstraction of a piece of news. The real index used in our multimodal knowledge graph would be _{L_ 1 _._ 0 _, L_ 1 _._ 1 _, L_ 1 _._ 2 _, ...}_ . Similarly, [ _L_ 2 _.∗_ ] means image / text-level which is the symbolization of images or texts from news, [ _L_ 3 _.∗_ ] is the object in image or entity in text. The index for all nodes is eventually shuffled, that is, the real index would be _{L_ 1 _._ 0 _, L_ 2 _._ 1 _, L_ 1 _._ 2 _, L_ 3 _._ 3 _, L_ 3 _._ 4 _, ...}_ . 

**Phase-3: Knowledge Construction.** We categorize data knowledge into five unit types, namely, in-text ( _Tin_ ), in-image ( _Iin_ ), inter-text ( _Tcross_ ), inter-image ( _Icross_ ), and image-text ( _ITcross_ ) which are together called _KnowledgeView_ detailed in Fig. 2(a) and Fig. 2(b). 

Table 2: **Edge (N** _e_ **) construction and statistics** . 

|Table 2: **Edge (N**_e_**) construction and st**|**atistics**.|
|---|---|
|**Phrase**<br>**Construction Method**<br>**View**<br>Phrase-2<br>Detection Category<br>_Iin_<br>NER Category<br>_Tin_<br>Similarity&Manual Annotation<br>_ITcross_<br>Similarity&Manual Annotation _Tcross_,_Icross_<br>Phrase-3<br>Manual Event Annotation<br>-|**Num.**|
||648,871<br>1,606,936<br>684,207<br>140,133|
||593,670|



In this phase, we aggregate two kinds of internal 

knowledge ( _Iin, Tin_ ) and three kinds of associative knowledge ( _Icross, Tcross, ITcross_ ) in one graph **G** _m_ , which are usually introduced independently in previous studies. _Knowledge Construction_ takes as input the edge index **N** _e_ and node index **N** _n_ numbered by _Phase-2_ and output the multimodal knowledge graph **G** _m_ (Fig. 2(c)). Since **N** _e_ and **N** _n_ are both isolated, we use four kinds of correlation methods including semantic similarity, visual similarity, annotations, and categories to make connections between **N** _n_ by **N** _e_ shown in Tab. 2. 

## **3.2 Dataset Collection** 

Following the proposed protocol and three phases, we collect a new dataset, a large-scale multimodal knowledge graph from public international news. Specifically, based on the Wikipedia API [43] and our crawler system, we grab all the data of “ _Worldwide Current Events_ ” from Wikipedia. As 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training_images/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training.pdf-0006-00.png)


**----- Start of picture text -----**<br>
Nn<br>[Phase-2]<br>Np<br>[Phase-1]<br>Gm<br>[Phase-3]<br>Ne<br>[Phase-2]<br>**----- End of picture text -----**<br>


Figure 3: **Pipeline of dataset construction following** _**UKnow**_ **protocol.** Phase-1: _Content Extraction_ ( _Np_ ), Phase-2: _Information Symbolization_ ( **N** _n_ , **N** _e_ ), and Phase-3: _Knowledge Construction_ ( **G** _m_ ). **N** _n_ hides the real node index for easy understanding, the actual number is much more than **N** _e_ . 

demonstrated in top of Fig. 4, we propose two category sets of news event called: _Event-11_ and _Event-9185_ , which is coarse-grained and fine-grained respectively. For example, _“Sports”_ is a kind of coarse-grained event label in _Event-11_ and _“2019 Daytona 500”_ is a fine-grained label in _Event-9185_ , detailed in Tab. 3. Since Wikipedia only records the news URL (downward black arrow in Fig. 4) and the HTML of original news from different news platforms is inconsistent, it is difficult to design a uniform crawler to get the well-structured raw data of news. Thus, we manually read each news and collect the original data (rightward black arrow). By this way, each news in our dataset is marked with extremely clean **title** , **content** , **time** , [ **image** ], **image description** , **event description** , [hierarchical] **event name** ( _e.g._ , _“Armed conflicts and attacks→War in Donbass”_ ), and **event attribute** (location, date, _etc_ ). Subsequently, as shown in bottom right of Fig. 4, we apply the designed pipeline to sequentially undergo phases 1/2/3 to restructure the above extracted raw data, resulting in the knowledge graph under the _UKnow_ format. 

Furthermore, in addition to utilizing intricate annotation files ( _e.g._ , Fig. 4) as inputs mentioned above, another major advantage of the proposed conversion pipeline is its ability to accommodate common image-text pair annotations expressed in the format of _“[image description] \t ./xxx.jpg \n”_ ), as the fundamental input. This design allows _UKnow_ to automatically construct a new dataset with more useful information from an existing image-text pair dataset. Taking LAION-5B [60] as an example, which solely comprises pairs of images and text, our pipeline can extract more features from them like objects, and thus expand LAION-5B into a larger and more practical dataset. However, given the absence of high-level event logic, this type of input does not lend itself to the creation of [ _L_ 1 _.∗_ ] nodes and event-related edges. 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training_images/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training.pdf-0007-00.png)


**----- Start of picture text -----**<br>
Events-11 wikipedia.org<br>Events-9185<br>URL https://www.foxnews.com/auto/daytona-500-massive-crash-started-paul-menard<br>Time Published February 17, 2019 9:00pm EST<br>Event Name Sports -> 2019 Daytona 500<br>Event Attributes Attributes<br>Event Description The 2019 Daytona 500, the 61st running of<br>Title Content the event, was a Monster Energy NASCAR Cup<br>Series race held on February 17, 2019,<br>Image Description Contested over 207 ……<br>Jamie McMurray (40), Erik Jones (20), William Byron (24), Brad Keselowski (2), Brendan Gaughan (62), Clint Bowyer<br>(14) and Chase Elliott (9) collide in Turn 3 during the NASCAR Daytona 500 auto race at Daytona International<br>Speedway Sunday in Daytona Beach.<br>……<br>Denny Hamlin wins the 2019 Daytona 500. It is his second win in the event, the first coming in 2016. (Fox News)<br>Image<br>News / Fact<br>Data Flow Diagram<br>wikipedia.org News / Fact LAION-5B etc.Original Data Phase-1, Phase-2, Phase-3 Event News / Fact entity text<br>Events-11 / 9185 image<br>UKnow object<br>**----- End of picture text -----**<br>


Figure 4: **Event category** labeled on web data and the **data flow diagram.** 

Table 3: **Details of the event category.** 

|Table 3:|Table 3:|**Details of the event category.**|
|---|---|---|
|**Event Name(****_Event-11_)**<br>**Visual**<br>**Texual**<br>**All**||**Event Name(10 examples of****_Event-9185_)**<br>**Visual Texual**<br>**All**|
|Armed conficts and attacks 87,346<br>90,157 177,503<br>Saudi Arabian-led intervention in Yemen<br>555<br>258<br>813<br>Arts and culture<br>11,059<br>14,896<br>25,955<br>A boat carrying Indonesian migrants capsizes off the southern coast of Malaysia<br>46<br>19<br>65<br>Business and economy<br>12,598<br>25,565<br>38,163<br>Travel restrictions related to the COVID-19 pandemic<br>753<br>796<br>1,549<br>Disasters and accidents<br>28,062<br>47,459<br>75,521<br>GameStop short squeeze<br>45<br>175<br>220<br>Health and environment<br>230,926 258,349 489,275<br>Opposition to Brexit in the United Kingdom<br>383<br>93<br>476<br>International relations<br>37,349<br>56,444<br>93,793<br>Gretchen Whitmer kidnapping plot<br>167<br>308<br>475<br>Sports<br>15,647<br>31,194<br>46,841<br>Legality of euthanasia<br>185<br>455<br>640<br>Law and crime<br>69,573<br>86,514 156,087<br>Ukraine International Airlines Flight 752 (Air Crash)<br>314<br>179<br>493<br>Politics and elections<br>74,477<br>72,714 147,191<br>Manhattan blackout<br>269<br>90<br>359<br>Science and technology<br>4,062<br>15,556<br>19,618<br>2019 Lagos school collapse<br>524<br>119<br>643<br>Others<br>236<br>184<br>420<br>...<br>...<br>...<br>...<br>Table 4: **Partition of our dataset.**|||
|**PARTITION**|_Tin_<br>_Iin_<br>_Tcross_<br>_Icross_<br>_ITcross_||
||**NODE**<br>**EDGE**<br>**NODE**<br>**EDGE**<br>**NODE**<br>**EDGE**<br>**NODE**<br>**EDGE**<br>**NODE**<br>**EDGE**||
|Training Set<br>Validation Set<br>TestingSet|448,691<br>8,030,531<br>501,564<br>979,287<br>250,858<br>396,200<br>69,911<br>421,628<br>765,654<br>382,827<br>37,488<br>100,280<br>12,126<br>12,212<br>69,533<br>57,162<br>15,532<br>97,272<br>9,764<br>4,882<br>37,668<br>100,375<br>12,182<br>12,261<br>69,286<br>55,464<br>15,336<br>99,303<br>9,622<br>4,811||
|Pre-training Set<br>Fine-tuning Set<br>TestingSet|228,339<br>435,659<br>343,458<br>325,755<br>101,880<br>314,918<br>47,017<br>271,593<br>278,058<br>139,029<br>75,924<br>82,350<br>65,809<br>61,850<br>19,185<br>59,832<br>8,880<br>52,772<br>52,522<br>26,261<br>34,422<br>28,219<br>22,809<br>22,278<br>6,633<br>21,360<br>3,074<br>17,754<br>18,186<br>9,093||



## **3.3 Dataset Statistics and Visualizations** 

Through data collection and processing in Sec. 3.2, our dataset comprises 1,388,568 nodes, of which 571,791 are relevant to vision (i.e., pertaining to a news image or a visual object), and 3,673,817 triples. The partitioning of our dataset is presented in Tab. 4, with all partitions being randomly sampled. Moreover, as depicted in Fig. 5, we present the histogram of all indices in _UKnow_ . Considering our dataset is a multimodal knowledge graph, _i.e._ , each node corresponds to a multimodal data, and each edge serves the purpose of connecting either single or cross-modal nodes. 

The top-2 number of nodes are “[ _L_ 3 _.∗_ ] objects” (501,880) and “[ _L_ 3 _.∗_ ] entity_content” (386,561) which belong to _Iin_ and _Tin_ respectively. The former represents visual objects extracted from images, and the latter means text entities extracted from news contents. The maximum number of edges is “[105] imgsim” (3,447,990) which is a kind of associative knowledge from _Icross_ . 

7 

Table 5: **Histogram of the number of indexes in our dataset** . The x-axis in the upper left corner (Node Index) corresponds to the order of the _In_ in Fig. 3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training_images/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training.pdf-0008-01.png)


**----- Start of picture text -----**<br>
× 10 [5]<br>5 . 2 4 . 0 [×] [10][6] 2 . 0 [×] [10][5]<br>3 . 5 1 . 7<br>4 . 0 3 . 0 1 . 5<br>3 . 0 2 . 4 1 . 3<br>2 . 0 1 . 0<br>2 . 0 1 . 4 0 . 7<br>100 ... 051 100 ... 052 00 .. 51<br>Node Index ( N n ) Edge Index ( N e ) Edge Index ( N e )<br>570231468 10583 92 96 80 86 0 9411311099 106911009085981021031091011049397828995818725627842639247256773<br>Number<br>**----- End of picture text -----**<br>


Fig. 3.3 shows the variation in different thresholds. _τ_ indicates the threshold of cosine similarity which controls whether edges are built between nodes. It can be adjusted according to needs ( _e.g._ , storage, computational complexity, fineness). The default setting of _τ_ is 0.8. _ρm_ indicates the average edge number of connections per node in the entire graph, which demonstrates the density of a graph. As shown in Tab. 8, the whole graph is sparse with _ENTITY_ as the main nodes of the background, and the subject element of the dense region is _CONTENT_ . _ρ_ means the number of edges, _i.e._ , there are 615k nodes with 0-edge or 1-edge, 463k nodes with 2-edges or 3-edges, and so on. The Mean Density ( _ρm_ ) in Fig. 3.3 is calculated as a weighted average of _ρ_ and the number of nodes. 

Table 6: **Histogram of the number of other edge indexes.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training_images/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training.pdf-0008-04.png)


**----- Start of picture text -----**<br>
× 10 [3]<br>8 . 0<br>7 . 0<br>65 .. 05<br>4 . 0<br>2 . 0<br>100 ... 050<br>Edge Index ( N e )<br>Table 7: The variation in different similarity thresholds.<br>2 . 78 2.72<br>2 . 72 2.78 2.78 2.76 2.69 2.65<br>2 . 66 2.6<br>2 . 60 2.53<br>2 . 54<br>2 . 48 2.44<br>2 . 42<br>0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95<br>Cosine Similarity Threshold ( τ )<br>Table 8: The graph density and mainstream node types.<br>τ Density ( ρ ) 0,1 2,3 4,5 6,7 8,9 10,11 12,13 14,15 16,17 ⩾18<br>0.8 NodeNum. 615k 463k 132k 71k 55k 32k 14k 4k 703 78<br>MainType Entity Object Title Image Content<br>6041888583621392814416374335593265724567516407934115717375554776647431936293046387644106471181535521250512049694253617868312122482370<br>Number<br>)<br>m<br>ρ<br>(<br>Density<br>Avg.<br>**----- End of picture text -----**<br>


Tab. 3 shows all the categories in _Event-11_ , and 10 examples in _Event-9185_ . “Visual” means the number of nodes belonging to images or objects. “All” means the number of all nodes marked with this event category. Generally speaking, _Event-9185_ is specific to an exact human activity and can be used to learn the semantic relevance of news contents, while _Event-11_ is more like a categorization of news events, which is benefit for archiving news materials through a trained classification model. 

## **4 Usage of UKnow** 

## **4.1 UKnow for Common-sense Reasoning** 

Since _UKnow_ is reasoning compatible, _i.e._ , it naturally supports all KG-reasoning models, we directly implemented the commonly used KG-reasoning models ( _e.g._ , TransE [3], Q2B [55]) on _UKnow_ . We propose a plug-in module which aggregates node features within a small sub-graph region to achieve a better central node features. We briefly introduce how to implement this module. Suppose _N_ ( _e_ ) _≡{eneib|r_ ( _eneib, e_ ) _∨ r_ ( _e, eneib_ ) _, r ∈R}_ is the collection of neighbors of each central node _e_ . The calculation expression of the new representation e _[′]_ of e is as follow: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training_images/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training.pdf-0008-09.png)


where e _∈_ R _[d]_ is the node feature before enhancement, e _[′]_ is the new feature, _⋆_ denotes a 2D convolution operation, _ωn_ is the filter, b _n_ is the bias and the specification of MLP is R _[m]_[1] _[×][m]_[2] _×_ R _[d]_ . The concat function _τ[′]_ (e _, N_ e) _∈_ R _[m]_[1] _[×][m]_[2] as [e; e _neib_[1] _[′]_ ; e _neib_[2] _[′]_ ; _. . ._ ; e _neib[m]_ ] where e _neib[i] ∈ N_ e _[′]_[.] 

## **4.2 UKnow for Vision-Language Pre-training** 

Following the recent works [33], our work applies CLIP [51] as the pre-trained backbone benefit from its strong downstream performance. Specifically, the text encoder first tokenize the input text description into the word sequence, and then projects them into word embeddings **W** 0 = 

8 

_{_ **w** 0[1] _[,]_ **[ w]** 0[2] _[,][ · · ·][,]_ **[ w]** 0 _[N][}][∈]_[R] _[N][×][d][t]_[.] **[W]**[0][is][fed][into][a] _[L]_[-layer][Transformer][[][72][]][with][the][architecture] modifications described in BERT [9]. And the final text embedding **z** _[T]_ is obtained by projecting the last token, which corresponds to the `[EOS]` (the end of sequence) token, from the last layer of the text encoder, _i.e._ , **z** _[T]_ = `TextProj` ( **w** _L[N]_[)] _[,]_ **[ z]** _[T][∈]_[R] _[d]_[.][As for the vision encoder, the input image] _[ I]_[is first] split into _M_ non-overlapping patches, and projected into a sequence of patch tokens **E0** _∈_ R _[M][×][d][v]_ . Then, **E0** is fed into a _L_ -layer Transformer-based architecture along with a learnable `[CLS]` token **c** 0. The final image embedding **z** _[I]_ is obtained by projecting the `[CLS]` token from the last layer of the vision encoder, _i.e._ , **z** _[I]_ = `VisProj` ( **c** _[v] L[,]_ **[ E]** _[v] L_[))] _[,]_ **[ z]** _[I][∈]_[R] _[d]_[.][Since we have] _[ Knowledge-View]_[, a new] dimension **z** _[k]_ which is used to represent knowledge is introduced: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training_images/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training.pdf-0009-01.png)


where _Iin_ ( _·_ ) and _Tin_ ( _·_ ) mean to get the embedding of the [ _L_ 3 _.∗_ ] nodes ( **N** _n_ ) from **G** _m_ via **N** _e_ , _Icross_ ( _·_ ) and _Tcross_ ( _·_ ) mean to get the embedding of [ _L_ 2 _.∗_ ] from **G** _m_ . Therefore, the similarity score between the image, text and knowledge can be calculated with the cosine similarity as follow: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training_images/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training.pdf-0009-03.png)


## **4.3 UKnow Baseline** 

Upgrading AI from understanding objects ( _e.g._ , an apple) as in most current vision tasks to understanding complex human activities ( _e.g._ , an event), to understanding the logic between entities or objects, and to achieving higher-order intelligence, is always the thing we would like to pioneer. Thus, in this section, we naturally present a series of novel logic-rich downstream tasks as the baselines for our dataset. Specifically, Common-sense Reasoning is a conventional and fundamental task in our domain, aligning closely with our dataset. Then we perform multiple downstream tasks to verify the performance of the pretrained model trained with our dataset. For more details about task description/training setting/evaluation metric/analysis, please refer to Sec. C. 

**Common-sense Reasoning.** We implement the Q2B _[∗]_ with our _UKnow_ based plug-in module based on Q2B [55] and BETAE _[∗]_ based on BETAE [56]. As shown in Tab. 9, BETAE _[∗]_ achieves on average **21.64%** and **21.23%** MRR on the validation and testing set of our dataset. It indicates that our _UKnow_ based module can significantly improve the performance of existing methods. 

**Multimodal Event Classification.** As shown in Tab. 10, TCL [86] achieves on **66.80%** and **55.87%** on ACC@1 when using the image-input on the _Event-11_ and _Event-9185_ . respectively. We add a late-fusion module after the image/text encoder for all methods to support multimodal classification. Results show that TCL obtains gains of **1.89%** and **5.02%** compared with the singlemodal input, which demonstrates that multimodal pre-training is more helpful for downstream multimodal tasks. 

**Single- & Cross-Modal Retrieval.** As shown in Tab. 11, TCL [86] achieves on **33.24%** , **43.37%** and **45.22%** R@1, R@5, R@10 on the zero-shot setting of image retrieval. The results are **58.89%** , **68.47%** and **73.91%** when fine-tuning the pre-trained parameters, which means the pre-training _→_ finetuning strategy is extremely beneficial for downstream retrieval. 

**Visual Task Adaptation.** As shown in Tab. 12, our approach obtains gains of avg. **1.14%** compared with the origin CLIP when fairly using the same _UKnow_ ’s data for the upstream pre-training. It is essential to highlight that the image-text PAIR constitutes only one type of data in our protocol. By leveraging the capabilities of _UKnow_ , our pre-trained CLIP model can effectively comprehend the inherent knowledge, resulting in superior performance than original CLIP model (Tab. 12, Row2). 

## **5 Conclusion** 

This paper presents a unified knowledge protocol called _UKnow_ to establish the standard of knowledge from the perspective of data. Following this protocol, we collect a novel and the largest multimodal knowledge graph dataset from public international news with rich news event annotations, which can help intelligent machines understand human activities and history. The specific tasks addressed in this paper are the common-sense reasoning and vision-language pre-training. The former is a typical task in the knowledge graph field, and the latter brings knowledge to various downstream tasks. We also present a series of novel logic-rich downstream tasks to showcase the advantages of _UKnow_ . In future work, we will continuously expand the data of different scales based on the _UKnow_ protocol. 

9 

## **References** 

- [1] Houda Alberts, Teresa Huang, Yash Deshpande, Yibo Liu, Kyunghyun Cho, Clara Vania, and Iacer Calixto. Visualsem: a high-quality knowledge graph for vision and language. arXiv preprint arXiv:2008.09150, 2020. 

- [2] Hangbo Bao, Li Dong, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021. 

- [3] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. In Advances in Neural Information Processing Systems, 2013. 

- [4] Sanyuan Chen, Yutai Hou, Yiming Cui, Wanxiang Che, Ting Liu, and Xiangzhan Yu. Recall and learn: Fine-tuning deep pretrained language models with less forgetting. arXiv preprint arXiv:2004.12651, 2020. 

- [5] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter: Universal image-text representation learning. In European conference on computer vision, pages 104–120, 2020. 

- [6] Zhuo Chen, Lingbing Guo, Yin Fang, Yichi Zhang, Jiaoyan Chen, Jeff Z Pan, Yangning Li, Huajun Chen, and Wen Zhang. Rethinking uncertainly missing and ambiguous visual modality in multi-modal entity alignment. In International Semantic Web Conference, pages 121–139. Springer, 2023. 

- [7] Zewen Chi, Li Dong, Bo Zheng, Shaohan Huang, Xian-Ling Mao, Heyan Huang, and Furu Wei. Improving pretrained cross-lingual language models via self-labeled word alignment. arXiv preprint arXiv:2106.06381, 2021. 

- [8] Jaemin Cho, Seunghyun Yoon, Ajinkya Kale, Franck Dernoncourt, Trung Bui, and Mohit Bansal. Finegrained image captioning with clip reward. In Findings of NAACL, 2022. 

- [9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 

- [10] Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang Zhu, Pengchuan Zhang, Lu Yuan, Nanyun Peng, et al. An empirical study of training end-to-end vision-andlanguage transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18166–18176, 2022. 

- [11] Sepideh Esmaeilpour, Bing Liu, Eric Robertson, and Lei Shu. Zero-shot out-of-distribution detection based on the pretrained model clip. In Proceedings of the AAAI conference on artifcial intelligence, 2022. 

- [12] Yuxia Geng, Jiaoyan Chen, Xiang Zhuang, Zhuo Chen, Jeff Z Pan, Juan Li, Zonggang Yuan, and Huajun Chen. Benchmarking knowledge-driven zero-shot learning. Journal of Web Semantics, page 100757, 2023. 

- [13] Tianyu Guo, Hong Liu, Zhan Chen, Mengyuan Liu, Tao Wang, and Runwei Ding. Contrastive learning from extremely augmented skeleton sequences for self-supervised action recognition. In Proceedings of the AAAI Conference on Artifcial Intelligence, pages 762–770, 2022. 

- [14] Kelvin Guu, John Miller, and Percy Liang. Traversing knowledge graphs in vector space. arXiv preprint arXiv:1506.01094, 2015. 

- [15] Will Hamilton, Payal Bajaj, Marinka Zitnik, Dan Jurafsky, and Jure Leskovec. Embedding logical queries on knowledge graphs. Advances in neural information processing systems, 2018. 

- [16] Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. Advances in neural information processing systems, 2017. 

- [17] Tengda Han, Weidi Xie, and Andrew Zisserman. Self-supervised co-training for video representation learning. Advances in Neural Information Processing Systems, pages 5679–5690, 2020. 

- [18] Xiangteng He, Yulin Pan, Mingqian Tang, Yiliang Lv, and Yuxin Peng. Learn from unlabeled videos for near-duplicate video retrieval. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1002–1011, 2022. 

- [19] Zhicheng Huang, Zhaoyang Zeng, Yupan Huang, Bei Liu, Dongmei Fu, and Jianlong Fu. Seeing out of the box: End-to-end pre-training for vision-language representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12976–12985, 2021. 

- [20] Zhicheng Huang, Zhaoyang Zeng, Bei Liu, Dongmei Fu, and Jianlong Fu. Pixel-bert: Aligning image pixels with text by deep multi-modal transformers. arXiv preprint arXiv:2004.00849, 2020. 

- [21] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, pages 4904–4916, 2021. 

10 

- [22] Longlong Jing and Yingli Tian. Self-supervised visual feature learning with deep neural networks: A survey. IEEE transactions on pattern analysis and machine intelligence, pages 4037–4058, 2020. 

- [23] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetr-modulated detection for end-to-end multi-modal understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1780–1790, 2021. 

- [24] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Vision-and-language transformer without convolution or region supervision. In International Conference on Machine Learning, pages 5583–5594, 2021. 

- [25] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016. 

- [26] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, pages 32–73, 2017. 

- [27] Ni Lao, Tom Mitchell, and William Cohen. Random walk inference and learning in a large scale knowledge base. In Proceedings of the 2011 conference on empirical methods in natural language processing, pages 529–539, 2011. 

- [28] Anne Lauscher, Olga Majewska, Leonardo FR Ribeiro, Iryna Gurevych, Nikolai Rozanov, and Goran Glavaš. Common sense or world knowledge? investigating adapter-based knowledge injection into pretrained transformers. arXiv preprint arXiv:2005.11787, 2020. 

- [29] Jaejun Lee, Chanyoung Chung, Hochang Lee, Sungho Jo, and Joyce Whang. Vista: Visual-textual knowledge graph representation learning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 7314–7328, 2023. 

- [30] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems, pages 9694–9705, 2021. 

- [31] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557, 2019. 

- [32] Liunian Harold Li*, Pengchuan Zhang*, Haotian Zhang*, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. Grounded language-image pre-training. In IEEE Conf. Comput. Vis. Pattern Recog., 2022. 

- [33] Manling Li, Ruochen Xu, Shuohang Wang, Luowei Zhou, Xudong Lin, Chenguang Zhu, Michael Zeng, Heng Ji, and Shih-Fu Chang. Clip-event: Connecting text and images with event structures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16420–16429, 2022. 

- [34] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In European Conference on Computer Vision, pages 121–137, 2020. 

- [35] Yangguang Li, Feng Liang, Lichen Zhao, Yufeng Cui, Wanli Ouyang, Jing Shao, Fengwei Yu, and Junjie Yan. Supervision exists everywhere: A data efficient contrastive language-image pre-training paradigm. arXiv preprint arXiv:2110.05208, 2021. 

- [36] Yangning Li, Jiaoyan Chen, Yinghui Li, Yuejia Xiang, Xi Chen, and Hai-Tao Zheng. Vision, deduction and alignment: An empirical study on multi-modal knowledge graph alignment. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023. 

- [37] Ying Lin, Heng Ji, Fei Huang, and Lingfei Wu. A joint neural model for information extraction with global features. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 7999–8009, 2020. 

- [38] Ye Liu, Hui Li, Alberto Garcia-Duran, Mathias Niepert, Daniel Onoro-Rubio, and David S Rosenblum. Mmkg: multi-modal knowledge graphs. In European Semantic Web Conference, pages 459–474, 2019. 

- [39] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019. 

- [40] Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang. Deepfashion: Powering robust clothes recognition and retrieval with rich annotations. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1096–1104, 2016. 

- [41] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems, 2019. 

- [42] Ruotian Luo, Brian Price, Scott Cohen, and Gregory Shakhnarovich. Discriminability objective for training descriptive captions. arXiv preprint arXiv:1803.04376, 2018. 

11 

- [43] Martin Majlis. Wikipedia-api. `https://pypi.org/project/Wikipedia-API/` . 

- [44] Kenneth Marino, Xinlei Chen, Devi Parikh, Abhinav Gupta, and Marcus Rohrbach. Krisp: Integrating implicit and symbolic knowledge for open-domain knowledge-based vqa. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14111–14121, 2021. 

- [45] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204, 2019. 

- [46] Ron Mokady, Amir Hertz, and Amit H Bermano. Clipcap: Clip prefix for image captioning. arXiv preprint arXiv:2111.09734, 2021. 

- [47] Hatem Mousselly-Sergieh, Teresa Botschen, Iryna Gurevych, and Stefan Roth. A multimodal translationbased approach for knowledge graph representation learning. In Proceedings of the Seventh Joint Conference on Lexical and Computational Semantics, pages 225–234, 2018. 

- [48] Daniel Oñoro-Rubio, Mathias Niepert, Alberto García-Durán, Roberto González, and Roberto J LópezSastre. Answering visual-relational queries in web-extracted knowledge graphs. arXiv preprint arXiv:1709.02314, 2017. 

- [49] Xingjia Pan, Fan Tang, Weiming Dong, Yang Gu, Zhichao Song, Yiping Meng, Pengfei Xu, Oliver Deussen, and Changsheng Xu. Self-supervised feature augmentation for large image object detection. IEEE Transactions on Image Processing, pages 6745–6758, 2020. 

- [50] Heiko Paulheim. Knowledge graph refinement: A survey of approaches and evaluation methods. Semantic web, pages 489–508, 2017. 

- [51] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763, 2021. 

- [52] Brandon Reagen, Paul Whatmough, Robert Adolf, Saketh Rama, Hyunkwang Lee, Sae Kyu Lee, José Miguel Hernández-Lobato, Gu-Yeon Wei, and David Brooks. Minerva: Enabling low-power, highlyaccurate deep neural network accelerators. In 2016 ACM/IEEE 43rd Annual International Symposium on Computer Architecture (ISCA), pages 267–278, 2016. 

- [53] Thomas Rebele, Fabian Suchanek, Johannes Hoffart, Joanna Biega, Erdal Kuzey, and Gerhard Weikum. Yago: A multilingual knowledge base from wikipedia, wordnet, and geonames. In International semantic web conference, pages 177–185, 2016. 

- [54] Feiliang Ren, Juchen Li, Huihui Zhang, Shilei Liu, Bochao Li, Ruicheng Ming, and Yujia Bai. Knowledge graph embedding with atrous convolution and residual learning. arXiv preprint arXiv:2010.12121, 2020. 

- [55] Hongyu Ren, Weihua Hu, and Jure Leskovec. Query2box: Reasoning over knowledge graphs in vector space using box embeddings. arXiv preprint arXiv:2002.05969, 2020. 

- [56] Hongyu Ren and Jure Leskovec. Beta embeddings for multi-hop logical reasoning in knowledge graphs. Advances in Neural Information Processing Systems, pages 19716–19726, 2020. 

- [57] Hongyu Ren and Jure Leskovec. Beta embeddings for multi-hop logical reasoning in knowledge graphs. Advances in Neural Information Processing Systems, pages 19716–19726, 2020. 

- [58] Andrea Rossi, Denilson Barbosa, Donatella Firmani, Antonio Matinata, and Paolo Merialdo. Knowledge graph embedding for link prediction: A comparative analysis. ACM Transactions on Knowledge Discovery from Data (TKDD), pages 1–49, 2021. 

- [59] Dan Ruta, Andrew Gilbert, Pranav Aggarwal, Naveen Marri, Ajinkya Kale, Jo Briggs, Chris Speed, Hailin Jin, Baldo Faieta, Alex Filipkowski, et al. Stylebabel: Artistic style tagging and captioning. arXiv preprint arXiv:2203.05321, 2022. 

- [60] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. arXiv preprint arXiv:2210.08402, 2022. 

- [61] Chao Shang, Yun Tang, Jing Huang, Jinbo Bi, Xiaodong He, and Bowen Zhou. End-to-end structure-aware convolutional networks for knowledge base completion. In Proceedings of the AAAI Conference on Artifcial Intelligence, pages 3060–3067, 2019. 

- [62] Chao Shang, Yun Tang, Jing Huang, Jinbo Bi, Xiaodong He, and Bowen Zhou. End-to-end structure-aware convolutional networks for knowledge base completion. In Proceedings of the AAAI Conference on Artifcial Intelligence, pages 3060–3067, 2019. 

- [63] Sheng Shen, Liunian Harold Li, Hao Tan, Mohit Bansal, Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, and Kurt Keutzer. How much can clip benefit vision-and-language tasks? arXiv preprint arXiv:2107.06383, 2021. 

12 

- [64] Ying Shen, Ning Ding, Hai-Tao Zheng, Yaliang Li, and Min Yang. Modeling relation paths for knowledge graph completion. IEEE Transactions on Knowledge and Data Engineering, pages 3607–3617, 2020. 

- [65] Haoyu Song, Li Dong, Wei-Nan Zhang, Ting Liu, and Furu Wei. Clip models are few-shot learners: Empirical studies on vqa and visual entailment. arXiv preprint arXiv:2203.07190, 2022. 

- [66] Wenzheng Song, Masanori Suganuma, Xing Liu, Noriyuki Shimobayashi, Daisuke Maruta, and Takayuki Okatani. Matching in the dark: a dataset for matching image pairs of low-light scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6029–6038, 2021. 

- [67] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. Vl-bert: Pre-training of generic visual-linguistic representations. arXiv preprint arXiv:1908.08530, 2019. 

- [68] Zequn Sun, Qingheng Zhang, Wei Hu, Chengming Wang, Muhao Chen, Farahnaz Akrami, and Chengkai Li. A benchmarking study of embedding-based entity alignment for knowledge graphs. arXiv preprint arXiv:2003.07743, 2020. 

- [69] Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. Rotate: Knowledge graph embedding by relational rotation in complex space. arXiv preprint arXiv:1902.10197, 2019. 

- [70] Hao Tan and Mohit Bansal. Lxmert: Learning cross-modality encoder representations from transformers. arXiv preprint arXiv:1908.07490, 2019. 

- [71] Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. Complex embeddings for simple link prediction. In International conference on machine learning, pages 2071–2080, 2016. 

- [72] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, pages 5998–6008, 2017. 

- [73] Meng Wang, Haofen Wang, Guilin Qi, and Qiushuo Zheng. Richpedia: a large-scale, comprehensive multi-modal knowledge graph. Big Data Research, page 100159, 2020. 

- [74] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers, 2020. 

- [75] Xin Wang, Benyuan Meng, Hong Chen, Yuan Meng, Ke Lv, and Wenwu Zhu. Tiva-kg: A multimodal knowledge graph with text, image, video and audio. In Proceedings of the 31st ACM International Conference on Multimedia, pages 2391–2399, 2023. 

- [76] Yaqing Wang, Fenglong Ma, Zhiwei Jin, Ye Yuan, Guangxu Xun, Kishlay Jha, Lu Su, and Jing Gao. Eann: Event adversarial neural networks for multi-modal fake news detection. In Proceedings of the 24th acm sigkdd international conference on knowledge discovery & data mining, pages 849–857, 2018. 

- [77] Zhaoqing Wang, Yu Lu, Qiang Li, Xunqiang Tao, Yandong Guo, Mingming Gong, and Tongliang Liu. Cris: Clip-driven referring image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11686–11695, 2022. 

- [78] Zikang Wang, Linjing Li, Qiudan Li, and Daniel Zeng. Multimodal data enhanced representation learning for knowledge graphs. In 2019 International Joint Conference on Neural Networks (IJCNN), pages 1–8, 2019. 

- [79] Fangyun Wei, Yue Gao, Zhirong Wu, Han Hu, and Stephen Lin. Aligning pretraining for detection via object-level contrastive learning. Advances in Neural Information Processing Systems, pages 22682– 22694, 2021. 

- [80] Haoyang Wen, Ying Lin, Tuan Lai, Xiaoman Pan, Sha Li, Xudong Lin, Ben Zhou, Manling Li, Haoyu Wang, Hongming Zhang, et al. Resin: A dockerized schema-guided cross-document cross-lingual cross-media information extraction and event tracking system. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Demonstrations, pages 133–143, 2021. 

- [81] Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2, 2019. 

- [82] Ruobing Xie, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. Image-embodied knowledge representation learning. arXiv preprint arXiv:1609.07028, 2016. 

- [83] Wenhan Xiong, Thien Hoang, and William Yang Wang. Deeppath: A reinforcement learning method for knowledge graph reasoning. arXiv preprint arXiv:1707.06690, 2017. 

- [84] Derong Xu, Tong Xu, Shiwei Wu, Jingbo Zhou, and Enhong Chen. Relation-enhanced negative sampling for multimodal knowledge graph completion. In Proceedings of the 30th ACM international conference on multimedia, pages 3857–3866, 2022. 

- [85] Hongwei Xue, Yupan Huang, Bei Liu, Houwen Peng, Jianlong Fu, Houqiang Li, and Jiebo Luo. Probing inter-modality: Visual parsing with self-attention for vision-and-language pre-training. Advances in Neural Information Processing Systems, pages 4514–4528, 2021. 

13 

- [86] Jinyu Yang, Jiali Duan, Son Tran, Yi Xu, Sampath Chanda, Liqun Chen, Belinda Zeng, Trishul Chilimbi, and Junzhou Huang. Vision-language pre-training with triple contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15671–15680, 2022. 

- [87] Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Yumao Lu, Zicheng Liu, and Lijuan Wang. An empirical study of gpt-3 for few-shot knowledge-based vqa. In Proceedings of the AAAI Conference on Artifcial Intelligence, pages 3081–3089, 2022. 

- [88] Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. Qa-gnn: Reasoning with language models and knowledge graphs for question answering. arXiv preprint arXiv:2104.06378, 2021. 

- [89] Jing Yu, Zihao Zhu, Yujing Wang, Weifeng Zhang, Yue Hu, and Jianlong Tan. Cross-modal knowledge reasoning for knowledge-based visual question answering. Pattern Recognition, page 107563, 2020. 

- [90] Zhiwei Zha, Jiaan Wang, Zhixu Li, Xiangru Zhu, Wei Song, and Yanghua Xiao. M2conceptbase: A fine-grained aligned multi-modal conceptual knowledge base. arXiv preprint arXiv:2312.10417, 2023. 

- [91] Xiaohua Zhai, Joan Puigcerver, Alexander Kolesnikov, Pierre Ruyssen, Carlos Riquelme, Mario Lucic, Josip Djolonga, Andre Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, et al. A large-scale study of representation learning with the visual task adaptation benchmark. arXiv preprint arXiv:1910.04867, 2019. 

- [92] Jingdan Zhang, Jiaan Wang, Xiaodan Wang, Zhixu Li, and Yanghua Xiao. Aspectmmkg: A multi-modal knowledge graph with aspect-aware entities. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, pages 3361–3370, 2023. 

- [93] Ningyu Zhang, Lei Li, Xiang Chen, Xiaozhuan Liang, Shumin Deng, and Huajun Chen. Multimodal analogical reasoning over knowledge graphs. In The Eleventh International Conference on Learning Representations, 2022. 

- [94] Tongtao Zhang, Ananya Subburathinam, Ge Shi, Lifu Huang, Di Lu, Xiaoman Pan, Manling Li, Boliang Zhang, Qingyun Wang, Spencer Whitehead, et al. Gaia-a multi-media multi-lingual knowledge extraction and hypothesis generation system. In TAC, 2018. 

- [95] Shangfei Zheng, Weiqing Wang, Jianfeng Qu, Hongzhi Yin, Wei Chen, and Lei Zhao. Mmkgr: Multi-hop multi-modal knowledge graph reasoning. arXiv preprint arXiv:2209.01416, 2022. 

- [96] Zhehui Zhou, Can Wang, Yan Feng, and Defang Chen. Jointe: Jointly utilizing 1d and 2d convolution for knowledge graph embedding. Knowledge-Based Systems, page 108100, 2022. 

- [97] Xiangru Zhu, Zhixu Li, Xiaodan Wang, Xueyao Jiang, Penglei Sun, Xuwu Wang, Yanghua Xiao, and Nicholas Jing Yuan. Multi-modal knowledge graph construction and application: A survey. arXiv preprint arXiv:2202.05786, 2022. 

## **References** 

- [1] Houda Alberts, Teresa Huang, Yash Deshpande, Yibo Liu, Kyunghyun Cho, Clara Vania, and Iacer Calixto. Visualsem: a high-quality knowledge graph for vision and language. arXiv preprint arXiv:2008.09150, 2020. 

- [2] Hangbo Bao, Li Dong, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021. 

- [3] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. In Advances in Neural Information Processing Systems, 2013. 

- [4] Sanyuan Chen, Yutai Hou, Yiming Cui, Wanxiang Che, Ting Liu, and Xiangzhan Yu. Recall and learn: Fine-tuning deep pretrained language models with less forgetting. arXiv preprint arXiv:2004.12651, 2020. 

- [5] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter: Universal image-text representation learning. In European conference on computer vision, pages 104–120, 2020. 

- [6] Zhuo Chen, Lingbing Guo, Yin Fang, Yichi Zhang, Jiaoyan Chen, Jeff Z Pan, Yangning Li, Huajun Chen, and Wen Zhang. Rethinking uncertainly missing and ambiguous visual modality in multi-modal entity alignment. In International Semantic Web Conference, pages 121–139. Springer, 2023. 

- [7] Zewen Chi, Li Dong, Bo Zheng, Shaohan Huang, Xian-Ling Mao, Heyan Huang, and Furu Wei. Improving pretrained cross-lingual language models via self-labeled word alignment. arXiv preprint arXiv:2106.06381, 2021. 

- [8] Jaemin Cho, Seunghyun Yoon, Ajinkya Kale, Franck Dernoncourt, Trung Bui, and Mohit Bansal. Finegrained image captioning with clip reward. In Findings of NAACL, 2022. 

14 

- [9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 

- [10] Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang Zhu, Pengchuan Zhang, Lu Yuan, Nanyun Peng, et al. An empirical study of training end-to-end vision-andlanguage transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18166–18176, 2022. 

- [11] Sepideh Esmaeilpour, Bing Liu, Eric Robertson, and Lei Shu. Zero-shot out-of-distribution detection based on the pretrained model clip. In Proceedings of the AAAI conference on artifcial intelligence, 2022. 

- [12] Yuxia Geng, Jiaoyan Chen, Xiang Zhuang, Zhuo Chen, Jeff Z Pan, Juan Li, Zonggang Yuan, and Huajun Chen. Benchmarking knowledge-driven zero-shot learning. Journal of Web Semantics, page 100757, 2023. 

- [13] Tianyu Guo, Hong Liu, Zhan Chen, Mengyuan Liu, Tao Wang, and Runwei Ding. Contrastive learning from extremely augmented skeleton sequences for self-supervised action recognition. In Proceedings of the AAAI Conference on Artifcial Intelligence, pages 762–770, 2022. 

- [14] Kelvin Guu, John Miller, and Percy Liang. Traversing knowledge graphs in vector space. arXiv preprint arXiv:1506.01094, 2015. 

- [15] Will Hamilton, Payal Bajaj, Marinka Zitnik, Dan Jurafsky, and Jure Leskovec. Embedding logical queries on knowledge graphs. Advances in neural information processing systems, 2018. 

- [16] Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. Advances in neural information processing systems, 2017. 

- [17] Tengda Han, Weidi Xie, and Andrew Zisserman. Self-supervised co-training for video representation learning. Advances in Neural Information Processing Systems, pages 5679–5690, 2020. 

- [18] Xiangteng He, Yulin Pan, Mingqian Tang, Yiliang Lv, and Yuxin Peng. Learn from unlabeled videos for near-duplicate video retrieval. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1002–1011, 2022. 

- [19] Zhicheng Huang, Zhaoyang Zeng, Yupan Huang, Bei Liu, Dongmei Fu, and Jianlong Fu. Seeing out of the box: End-to-end pre-training for vision-language representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12976–12985, 2021. 

- [20] Zhicheng Huang, Zhaoyang Zeng, Bei Liu, Dongmei Fu, and Jianlong Fu. Pixel-bert: Aligning image pixels with text by deep multi-modal transformers. arXiv preprint arXiv:2004.00849, 2020. 

- [21] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, pages 4904–4916, 2021. 

- [22] Longlong Jing and Yingli Tian. Self-supervised visual feature learning with deep neural networks: A survey. IEEE transactions on pattern analysis and machine intelligence, pages 4037–4058, 2020. 

- [23] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetr-modulated detection for end-to-end multi-modal understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1780–1790, 2021. 

- [24] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Vision-and-language transformer without convolution or region supervision. In International Conference on Machine Learning, pages 5583–5594, 2021. 

- [25] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016. 

- [26] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, pages 32–73, 2017. 

- [27] Ni Lao, Tom Mitchell, and William Cohen. Random walk inference and learning in a large scale knowledge base. In Proceedings of the 2011 conference on empirical methods in natural language processing, pages 529–539, 2011. 

- [28] Anne Lauscher, Olga Majewska, Leonardo FR Ribeiro, Iryna Gurevych, Nikolai Rozanov, and Goran Glavaš. Common sense or world knowledge? investigating adapter-based knowledge injection into pretrained transformers. arXiv preprint arXiv:2005.11787, 2020. 

- [29] Jaejun Lee, Chanyoung Chung, Hochang Lee, Sungho Jo, and Joyce Whang. Vista: Visual-textual knowledge graph representation learning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 7314–7328, 2023. 

15 

- [30] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems, pages 9694–9705, 2021. 

- [31] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557, 2019. 

- [32] Liunian Harold Li*, Pengchuan Zhang*, Haotian Zhang*, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. Grounded language-image pre-training. In IEEE Conf. Comput. Vis. Pattern Recog., 2022. 

- [33] Manling Li, Ruochen Xu, Shuohang Wang, Luowei Zhou, Xudong Lin, Chenguang Zhu, Michael Zeng, Heng Ji, and Shih-Fu Chang. Clip-event: Connecting text and images with event structures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16420–16429, 2022. 

- [34] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In European Conference on Computer Vision, pages 121–137, 2020. 

- [35] Yangguang Li, Feng Liang, Lichen Zhao, Yufeng Cui, Wanli Ouyang, Jing Shao, Fengwei Yu, and Junjie Yan. Supervision exists everywhere: A data efficient contrastive language-image pre-training paradigm. arXiv preprint arXiv:2110.05208, 2021. 

- [36] Yangning Li, Jiaoyan Chen, Yinghui Li, Yuejia Xiang, Xi Chen, and Hai-Tao Zheng. Vision, deduction and alignment: An empirical study on multi-modal knowledge graph alignment. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023. 

- [37] Ying Lin, Heng Ji, Fei Huang, and Lingfei Wu. A joint neural model for information extraction with global features. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 7999–8009, 2020. 

- [38] Ye Liu, Hui Li, Alberto Garcia-Duran, Mathias Niepert, Daniel Onoro-Rubio, and David S Rosenblum. Mmkg: multi-modal knowledge graphs. In European Semantic Web Conference, pages 459–474, 2019. 

- [39] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019. 

- [40] Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang. Deepfashion: Powering robust clothes recognition and retrieval with rich annotations. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1096–1104, 2016. 

- [41] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems, 2019. 

- [42] Ruotian Luo, Brian Price, Scott Cohen, and Gregory Shakhnarovich. Discriminability objective for training descriptive captions. arXiv preprint arXiv:1803.04376, 2018. 

- [43] Martin Majlis. Wikipedia-api. `https://pypi.org/project/Wikipedia-API/` . 

- [44] Kenneth Marino, Xinlei Chen, Devi Parikh, Abhinav Gupta, and Marcus Rohrbach. Krisp: Integrating implicit and symbolic knowledge for open-domain knowledge-based vqa. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14111–14121, 2021. 

- [45] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204, 2019. 

- [46] Ron Mokady, Amir Hertz, and Amit H Bermano. Clipcap: Clip prefix for image captioning. arXiv preprint arXiv:2111.09734, 2021. 

- [47] Hatem Mousselly-Sergieh, Teresa Botschen, Iryna Gurevych, and Stefan Roth. A multimodal translationbased approach for knowledge graph representation learning. In Proceedings of the Seventh Joint Conference on Lexical and Computational Semantics, pages 225–234, 2018. 

- [48] Daniel Oñoro-Rubio, Mathias Niepert, Alberto García-Durán, Roberto González, and Roberto J LópezSastre. Answering visual-relational queries in web-extracted knowledge graphs. arXiv preprint arXiv:1709.02314, 2017. 

- [49] Xingjia Pan, Fan Tang, Weiming Dong, Yang Gu, Zhichao Song, Yiping Meng, Pengfei Xu, Oliver Deussen, and Changsheng Xu. Self-supervised feature augmentation for large image object detection. IEEE Transactions on Image Processing, pages 6745–6758, 2020. 

- [50] Heiko Paulheim. Knowledge graph refinement: A survey of approaches and evaluation methods. Semantic web, pages 489–508, 2017. 

16 

- [51] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763, 2021. 

- [52] Brandon Reagen, Paul Whatmough, Robert Adolf, Saketh Rama, Hyunkwang Lee, Sae Kyu Lee, José Miguel Hernández-Lobato, Gu-Yeon Wei, and David Brooks. Minerva: Enabling low-power, highlyaccurate deep neural network accelerators. In 2016 ACM/IEEE 43rd Annual International Symposium on Computer Architecture (ISCA), pages 267–278, 2016. 

- [53] Thomas Rebele, Fabian Suchanek, Johannes Hoffart, Joanna Biega, Erdal Kuzey, and Gerhard Weikum. Yago: A multilingual knowledge base from wikipedia, wordnet, and geonames. In International semantic web conference, pages 177–185, 2016. 

- [54] Feiliang Ren, Juchen Li, Huihui Zhang, Shilei Liu, Bochao Li, Ruicheng Ming, and Yujia Bai. Knowledge graph embedding with atrous convolution and residual learning. arXiv preprint arXiv:2010.12121, 2020. 

- [55] Hongyu Ren, Weihua Hu, and Jure Leskovec. Query2box: Reasoning over knowledge graphs in vector space using box embeddings. arXiv preprint arXiv:2002.05969, 2020. 

- [56] Hongyu Ren and Jure Leskovec. Beta embeddings for multi-hop logical reasoning in knowledge graphs. Advances in Neural Information Processing Systems, pages 19716–19726, 2020. 

- [57] Hongyu Ren and Jure Leskovec. Beta embeddings for multi-hop logical reasoning in knowledge graphs. Advances in Neural Information Processing Systems, pages 19716–19726, 2020. 

- [58] Andrea Rossi, Denilson Barbosa, Donatella Firmani, Antonio Matinata, and Paolo Merialdo. Knowledge graph embedding for link prediction: A comparative analysis. ACM Transactions on Knowledge Discovery from Data (TKDD), pages 1–49, 2021. 

- [59] Dan Ruta, Andrew Gilbert, Pranav Aggarwal, Naveen Marri, Ajinkya Kale, Jo Briggs, Chris Speed, Hailin Jin, Baldo Faieta, Alex Filipkowski, et al. Stylebabel: Artistic style tagging and captioning. arXiv preprint arXiv:2203.05321, 2022. 

- [60] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. arXiv preprint arXiv:2210.08402, 2022. 

- [61] Chao Shang, Yun Tang, Jing Huang, Jinbo Bi, Xiaodong He, and Bowen Zhou. End-to-end structure-aware convolutional networks for knowledge base completion. In Proceedings of the AAAI Conference on Artifcial Intelligence, pages 3060–3067, 2019. 

- [62] Chao Shang, Yun Tang, Jing Huang, Jinbo Bi, Xiaodong He, and Bowen Zhou. End-to-end structure-aware convolutional networks for knowledge base completion. In Proceedings of the AAAI Conference on Artifcial Intelligence, pages 3060–3067, 2019. 

- [63] Sheng Shen, Liunian Harold Li, Hao Tan, Mohit Bansal, Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, and Kurt Keutzer. How much can clip benefit vision-and-language tasks? arXiv preprint arXiv:2107.06383, 2021. 

- [64] Ying Shen, Ning Ding, Hai-Tao Zheng, Yaliang Li, and Min Yang. Modeling relation paths for knowledge graph completion. IEEE Transactions on Knowledge and Data Engineering, pages 3607–3617, 2020. 

- [65] Haoyu Song, Li Dong, Wei-Nan Zhang, Ting Liu, and Furu Wei. Clip models are few-shot learners: Empirical studies on vqa and visual entailment. arXiv preprint arXiv:2203.07190, 2022. 

- [66] Wenzheng Song, Masanori Suganuma, Xing Liu, Noriyuki Shimobayashi, Daisuke Maruta, and Takayuki Okatani. Matching in the dark: a dataset for matching image pairs of low-light scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6029–6038, 2021. 

- [67] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. Vl-bert: Pre-training of generic visual-linguistic representations. arXiv preprint arXiv:1908.08530, 2019. 

- [68] Zequn Sun, Qingheng Zhang, Wei Hu, Chengming Wang, Muhao Chen, Farahnaz Akrami, and Chengkai Li. A benchmarking study of embedding-based entity alignment for knowledge graphs. arXiv preprint arXiv:2003.07743, 2020. 

- [69] Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. Rotate: Knowledge graph embedding by relational rotation in complex space. arXiv preprint arXiv:1902.10197, 2019. 

- [70] Hao Tan and Mohit Bansal. Lxmert: Learning cross-modality encoder representations from transformers. arXiv preprint arXiv:1908.07490, 2019. 

- [71] Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. Complex embeddings for simple link prediction. In International conference on machine learning, pages 2071–2080, 2016. 

- [72] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, pages 5998–6008, 2017. 

17 

- [73] Meng Wang, Haofen Wang, Guilin Qi, and Qiushuo Zheng. Richpedia: a large-scale, comprehensive multi-modal knowledge graph. Big Data Research, page 100159, 2020. 

- [74] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers, 2020. 

- [75] Xin Wang, Benyuan Meng, Hong Chen, Yuan Meng, Ke Lv, and Wenwu Zhu. Tiva-kg: A multimodal knowledge graph with text, image, video and audio. In Proceedings of the 31st ACM International Conference on Multimedia, pages 2391–2399, 2023. 

- [76] Yaqing Wang, Fenglong Ma, Zhiwei Jin, Ye Yuan, Guangxu Xun, Kishlay Jha, Lu Su, and Jing Gao. Eann: Event adversarial neural networks for multi-modal fake news detection. In Proceedings of the 24th acm sigkdd international conference on knowledge discovery & data mining, pages 849–857, 2018. 

- [77] Zhaoqing Wang, Yu Lu, Qiang Li, Xunqiang Tao, Yandong Guo, Mingming Gong, and Tongliang Liu. Cris: Clip-driven referring image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11686–11695, 2022. 

- [78] Zikang Wang, Linjing Li, Qiudan Li, and Daniel Zeng. Multimodal data enhanced representation learning for knowledge graphs. In 2019 International Joint Conference on Neural Networks (IJCNN), pages 1–8, 2019. 

- [79] Fangyun Wei, Yue Gao, Zhirong Wu, Han Hu, and Stephen Lin. Aligning pretraining for detection via object-level contrastive learning. Advances in Neural Information Processing Systems, pages 22682– 22694, 2021. 

- [80] Haoyang Wen, Ying Lin, Tuan Lai, Xiaoman Pan, Sha Li, Xudong Lin, Ben Zhou, Manling Li, Haoyu Wang, Hongming Zhang, et al. Resin: A dockerized schema-guided cross-document cross-lingual cross-media information extraction and event tracking system. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Demonstrations, pages 133–143, 2021. 

- [81] Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2, 2019. 

- [82] Ruobing Xie, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. Image-embodied knowledge representation learning. arXiv preprint arXiv:1609.07028, 2016. 

- [83] Wenhan Xiong, Thien Hoang, and William Yang Wang. Deeppath: A reinforcement learning method for knowledge graph reasoning. arXiv preprint arXiv:1707.06690, 2017. 

- [84] Derong Xu, Tong Xu, Shiwei Wu, Jingbo Zhou, and Enhong Chen. Relation-enhanced negative sampling for multimodal knowledge graph completion. In Proceedings of the 30th ACM international conference on multimedia, pages 3857–3866, 2022. 

- [85] Hongwei Xue, Yupan Huang, Bei Liu, Houwen Peng, Jianlong Fu, Houqiang Li, and Jiebo Luo. Probing inter-modality: Visual parsing with self-attention for vision-and-language pre-training. Advances in Neural Information Processing Systems, pages 4514–4528, 2021. 

- [86] Jinyu Yang, Jiali Duan, Son Tran, Yi Xu, Sampath Chanda, Liqun Chen, Belinda Zeng, Trishul Chilimbi, and Junzhou Huang. Vision-language pre-training with triple contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15671–15680, 2022. 

- [87] Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Yumao Lu, Zicheng Liu, and Lijuan Wang. An empirical study of gpt-3 for few-shot knowledge-based vqa. In Proceedings of the AAAI Conference on Artifcial Intelligence, pages 3081–3089, 2022. 

- [88] Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. Qa-gnn: Reasoning with language models and knowledge graphs for question answering. arXiv preprint arXiv:2104.06378, 2021. 

- [89] Jing Yu, Zihao Zhu, Yujing Wang, Weifeng Zhang, Yue Hu, and Jianlong Tan. Cross-modal knowledge reasoning for knowledge-based visual question answering. Pattern Recognition, page 107563, 2020. 

- [90] Zhiwei Zha, Jiaan Wang, Zhixu Li, Xiangru Zhu, Wei Song, and Yanghua Xiao. M2conceptbase: A fine-grained aligned multi-modal conceptual knowledge base. arXiv preprint arXiv:2312.10417, 2023. 

- [91] Xiaohua Zhai, Joan Puigcerver, Alexander Kolesnikov, Pierre Ruyssen, Carlos Riquelme, Mario Lucic, Josip Djolonga, Andre Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, et al. A large-scale study of representation learning with the visual task adaptation benchmark. arXiv preprint arXiv:1910.04867, 2019. 

- [92] Jingdan Zhang, Jiaan Wang, Xiaodan Wang, Zhixu Li, and Yanghua Xiao. Aspectmmkg: A multi-modal knowledge graph with aspect-aware entities. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, pages 3361–3370, 2023. 

- [93] Ningyu Zhang, Lei Li, Xiang Chen, Xiaozhuan Liang, Shumin Deng, and Huajun Chen. Multimodal analogical reasoning over knowledge graphs. In The Eleventh International Conference on Learning Representations, 2022. 

18 

- [94] Tongtao Zhang, Ananya Subburathinam, Ge Shi, Lifu Huang, Di Lu, Xiaoman Pan, Manling Li, Boliang Zhang, Qingyun Wang, Spencer Whitehead, et al. Gaia-a multi-media multi-lingual knowledge extraction and hypothesis generation system. In TAC, 2018. 

- [95] Shangfei Zheng, Weiqing Wang, Jianfeng Qu, Hongzhi Yin, Wei Chen, and Lei Zhao. Mmkgr: Multi-hop multi-modal knowledge graph reasoning. arXiv preprint arXiv:2209.01416, 2022. 

- [96] Zhehui Zhou, Can Wang, Yan Feng, and Defang Chen. Jointe: Jointly utilizing 1d and 2d convolution for knowledge graph embedding. Knowledge-Based Systems, page 108100, 2022. 

- [97] Xiangru Zhu, Zhixu Li, Xiaodan Wang, Xueyao Jiang, Penglei Sun, Xuwu Wang, Yanghua Xiao, and Nicholas Jing Yuan. Multi-modal knowledge graph construction and application: A survey. arXiv preprint arXiv:2202.05786, 2022. 

19 

## **A Addition Statement for Our New Dataset** 

## **A.1 Dataset Documentation and Intended Use** 

We offer a detailed overview of our dataset statistics in Sec. 3.3. To facilitate better understanding and ease of access, we have made our dataset project available on ModelScope at: _https://www.modelscope.cn/datasets/ yutong/UKnow/summary_ , which includes dataset summary, data preview, quickstart and data files. 

The detailed data organization and corresponding download links are listed below: 

- Original data: We gather our data from publicly available international news sources, accumulating a substantial volume of images and text. Subsequently, we compress the collected data into several zip archives and store them in original_data: `UKnow/raw_data/*` . 

- Processed data: 

- Pre-node _Np_ : Building upon Phase-1, we leverage pre-trained deep learning models to extract valuable information from various domains. The resultant output from Phase-1 is structured as a dictionary and is then stored and saved to pre_node: `UKnow/processed_data/pre_node*` . 

- Node index _Nn_ and Edge index _Ne_ : As the outcomes acquired in Phase-1 (e.g., _Np_ ) are not directly applicable for graph construction, we employ an information symbolization strategy to organize them into indices, namely _Nn_ and _Ne_ , which are subsequently saved to index: `UKnow/processed_data/*_index*.pickle` . 

- Knowledge graph _Gm_ : Finally, we consolidate two types of internal knowledge ( _Iin, Tin_ ) and three types of associative knowledge ( _Icross, Tcross, ITcross_ ) into into one knowledge graph ( **G** _m_ ), which is stored as a dictionary in graph: `UKnow/processed_data/graph*.pickle` . 

Our dataset is intended for academic use and the corresponding license is based on: https://www.contributorcovenant.org/zh-cn/version/1/4/code-of-conduct.html, which was created by Coraline Ada Ehmke in 2014 and is released under the CC BY-NC-ND 4.0. 

## **A.2 Author statement** 

We confirm the data licenses and that we bear all responsibility in case of violation of rights. 

## **A.3 Hosting, licensing, and maintenance plan** 

**Hosting and Licensing.** Our dataset is hosted on ModelScope. Moreover, we furnish the relevant licenses in accordance with ModelScope at: https://www.contributor-covenant.org/zh-cn/version/1/4/code-of-conduct.html, which was created by Coraline Ada Ehmke in 2014 and is released under the CC BY 4.0 License. 

**Introduction to ModelScope.** ModelScope is a platform designed for managing and optimizing machine learning models. It provides various tools and features to streamline the model development process, including version control, performance monitoring, and collaboration capabilities. As for managing datasets, ModelScope offers robust functionality for organizing, storing, and accessing data. Users can upload datasets to the platform, where they are securely stored and can be easily accessed by authorized team members. ModelScope also supports versioning of datasets, allowing users to track changes over time and ensure reproducibility in their experiments. Additionally, the platform provides tools for data preprocessing, visualization, and analysis, helping users to efficiently prepare their data for model training and evaluation. Overall, ModelScope offers comprehensive support for managing datasets throughout the machine learning lifecycle. Therefore, we choose ModelScope as our hosting platform. 

**Usage of ModelScope.** To enable users to directly utilize all models on the ModelScope platform without configuring the environment, ModelScope integrates an online Notebook programming environment on its website and offers official mirrors for developers. These official mirrors allow users to bypass all installation and configuration steps, providing immediate access to the models. Currently the latest version of the CPU mirror and GPU mirror can be obtained from the office ModelScope repository. 

Users also can setup local python environment using following commands: 

1 `conda create -n modelscope python =3.8` 2 `conda activate modelscope` 3 `pip install modelscope` 

Then, users can access and enjoy our dataset by: 

1 `from modelscope.msdatasets import MsDataset` 2 `ds = MsDataset.load(‘yutong/UKnow ’,␣subset_name =‘default ’ , split=‘train ’)` 

20 

Besides, we strongly recommend that users read the official documents for optimal use. 

**Maintenance Plan.** In future work, we will persistently augment the dataset across various scales following the _UKnow_ protocol. This endeavor aims to furnish a comprehensive, diverse, and resilient multimodal knowledge graph, thereby facilitating subsequent research endeavors. 

## **B Preliminaries** 

**Multimodal Knowledge Graph.** An intuitive interpretation of multimodal knowledge graph is that the ordinary knowledge graph only consists of _<_ head, relation, tail _>_ triples like _<_ (“Jony”), Citizen, (“New York”) _>_ , but the multimodal knowledge graph consists of the following: 

_<_ (“ `Jony` ”), `Citizen` , (“ `NewYork` ”) _>_ , 

_<_ (“ `Jony` ”), `Appearance` , (“[ `Face` ]”) _>_ , 

_<_ (“ `NewYork` ”), `Landmark` , (“[ `Statueofliberty` ]”) _>_ , 

_<_ (“[ `AirForceOne` ]”), `Similarity` , (“[ `AirForceTwo` ]”) _>_ , where ( _·_ ) means a text node and [ _·_ ] means a image node. The machine cannot understand what _“An old man with white hair”_ is without establishing the connection between each word and its physical world meaning. However, with the help of multimodal knowledge graph, as a simple example, it is possible to generate a more informative entity-level sentence ( _e.g._ , _“Biden is making a speech”_ ) instead of a vague concept-level description ( _e.g._ , _“An old man with white hair is making a speech”_ ). To evaluate the effectiveness of multimodal knowledge graph (MMKG), several downstream tasks are often performed on the MMKGs, including common-sense reasoning, vision-language pre-training. 

**Common-sense Reasoning.** Common-sense reasoning means answering queries by logic permutations. The specific task in this work is the link prediction. In the inference phase, feeding _<_ ("America"), Capital _>_ to a reasoning model, the output should be _<_ ("Washington") _>_ . Various works [3, 71, 69, 61, 96, 54] achieve reasoning by embedding entities and relations in knowledge graph into low-dimensional vector space. For instance, GQE [15] encodes queries through a computation graph with relational projection and conjunction ( _∧_ ) as operators. Path-based methods [27, 83, 64, 52] start from anchor entities and determine the answer set by traversing the intermediate entities via relational path. There are also GCN [25] based methods [62, 16] pass message to iterate graph representation for reasoning. Common-sense reasoning is an extremely popular task in the field of knowledge graph. Since our dataset is based on the knowledge graph, the performance validation on common-sense reasoning is indispensable. 

**Vision-Language Pre-training** Vision-language pre-training (VLP) can be divided into three categories based on how they encode images [10]: OD-based region features [5, 31, 34, 41, 67, 70], CNN-based grid feature [63, 19, 20] and ViT-based patch features [85, 30, 24]. Pre-training objectives are usually: masked language/image modeling (MLM/MIM) [2, 9, 39], image-text matching (ITM) [34, 19, 10], and image-text contrastive learning (ITC) [30, 51, 35]. In this work, we concentrate on the study of the how to introduce our UKnow into ITC method based on ViT-based patch features. 

**Image-Text Contrastive Learning.** The recent CLIP [51] and ALIGN [21] perform pre-training using a crossmodal contrastive loss on millions of image-text pairs, which achieves remarkable performance on various downstream tasks [42, 63, 65]. MDETR [23] trains on multi-modal datasets which have explicit alignment between phrases and objects. GLIP [32] generates grounding boxes in a self-training fashion, and makes the learned representations semantic-rich. We implement these mainstream methods on our dataset, and also design a basic knowledge-based ITC method with UKnow. 

## **C Experimental Details** 

In this section, we give more details about the computation complexity, training, fine-tuning hyperparameters and evaluation for reference. 

## **C.1 Common-sense Reasoning** 

**Datasets.** Since our dataset is a knowledge graph, we benchmark the performance of KG-reasoning models on our dataset by completing KG-triples. The partitioning of the dataset is illustrated in the upper segment of Tab. 4. 

**Evaluation.** The specific task of common-sense reasoning in this work is the link prediction. Given a test query _q_ ( _e.g._ „ _<_ (“Jony”), Citizen, (?) _>_ ), we are interested in discovering non-trivial answers ( _e.g._ „ “New York”). That is, answer entities where at least one edge needs to be imputed in order to create an answer path to that entity. Each entity in our multimodal knowledge graph is not limited to a text entity but a multimodal node. Following [57], for each non-trivial answer _t_ of test query _q_ , we rank it against non-answer entities _E\_ [[ _q_ ]]test [3]. Then the rank of each answer is labeled as _r_ . We use Mean Reciprocal Rank(MRR): _r_[1][and Hits-at-] _[N]_[(] **[H]**[@] _[N]_[) : 1[] _[r][≤][N]_[]][ as] quantitative metrics. 

21 

Table 9: **A new benchmark of the common-sense reasoning task.** We report four metrics of each model on the validation and test sets. All experiments were repeated five times and the variance is shown in the table. 

|shown in the table.||
|---|---|
|**Model**<br>**Val-H@1**<br>**Val-H@3**<br>**Val-H@10**<br>**Val-MRR**|**Test-H@1**<br>**Test-H@3**<br>**Test-H@10**<br>**Test-MRR**|
|TransE [3]<br>11.75_±_0.113<br>29.04_±_0.112<br>31.76_±_0.143<br>14.77_±_0.153<br>Q2B [55]<br>14.99_±_0.118<br>25.78_±_0.135<br>36.76_±_0.169<br>18.80_±_0.166<br>Q2B_∗_<br>16.84_±_0.115<br>29.00_±_0.166<br>38.85_±_0.169<br>19.66_±_0.158|11.26_±_0.114<br>21.68_±_0.115<br>31.57_±_0.127<br>14.66_±_0.123<br>14.48_±_0.119<br>25.17_±_0.135<br>36.32_±_0.163<br>18.46_±_0.134<br>16.35_±_0.122<br>28.67_±_0.174<br>38.45_±_0.184<br>19.27_±_0.146|
|BETAE [56]<br>18.04_±_0.129<br>33.02_±_0.161<br>41.97_±_0.179<br>21.16_±_0.167<br>BETAE_∗_<br>19.02_±_0.125<br>33.97_±_0.173<br>43.17_±_0.199<br>21.64_±_0.173|17.65_±_0.129<br>32.75_±_0.160<br>41.67_±_0.177<br>20.75_±_0.140<br>18.22_±_0.135<br>33.52_±_0.187<br>42.68_±_0.198<br>21.23_±_0.154|
|QA-GNN [88]<br>18.04_±_0.129<br>33.02_±_0.161<br>41.97_±_0.179<br>21.16_±_0.167<br>QA-GNN_∗_<br>19.02_±_0.125<br>33.97_±_0.173<br>43.17_±_0.199<br>21.64_±_0.173|17.65_±_0.129<br>32.75_±_0.160<br>41.67_±_0.177<br>20.75_±_0.140<br>18.22_±_0.135<br>33.52_±_0.187<br>42.68_±_0.198<br>21.23_±_0.154|



|Table 10: **A new benchmark of**|Table 10: **A new benchmark of**|**the novel event classifcation task.** All models are fne-tuned in the|**the novel event classifcation task.** All models are fne-tuned in the|**the novel event classifcation task.** All models are fne-tuned in the|**the novel event classifcation task.** All models are fne-tuned in the|**the novel event classifcation task.** All models are fne-tuned in the|**the novel event classifcation task.** All models are fne-tuned in the|
|---|---|---|---|---|---|---|---|
|training set.|**Model**|**IMG TXT**||**Event-11**<br>**ACC@1 ACC@5**||**Event-9185**<br>**ACC@1 ACC@5**||
||CLIP [51]|✓||65.77|76.82|54.62|63.19|
||DeCLIP [35]|✓||66.43|78.32|54.86|63.82|
||ALBEF [30]|✓||66.29|77.84|55.03|63.47|
||TCL[86]|✓||66.80|78.91|55.87|64.33|
||CLIP||✓|64.32|75.92|57.48|65.78|
||DeCLIP||✓|65.89|77.51|59.76|67.81|
||ALBEF||✓|65.31|76.97|58.43|66.32|
||TCL||✓|66.03|78.14|59.94|68.23|
||CLIP|✓|✓|66.08|72.88|57.42|65.65|
||DeCLIP|✓|✓|67.16|72.96|58.64|66.49|
||ALBEF|✓|✓|68.03|74.26|60.04|68.13|
||TCL|✓|✓|68.69|75.02|60.89|69.17|



**Baselines.** We consider four baselines: TransE [3], Q2B [55] and BETAE [57]. Since the _UKnow_ based plug-in module can be attached to any reasoning models, we implement the Q2B _[∗]_ with our module based on Q2B and BETAE _[∗]_ based on BETAE. As shown in Tab. 9, BETAE _[∗]_ achieves on average **21.64%** and **21.23%** MRR on the validation and testing set of our dataset, respectively. For a fair comparison ( _e.g._ , TransE), our dataset does not construct complex logic such as FOL [14] to evaluate the performance of multi-hop logical reasoning. 

## **C.2 Multimodal Event Classification** 

We propose a novel task called multimodal event classification, leveraging event annotations (Tab. 3) from both Wiki’s event categories and our own manual tagging. The event annotation helps intelligent machines understand human activities and history, offering the possibility to identify which _type of event_ or which _real historical event_ a picture or a text is relevant to. As shown in Tab. 10, TCL [86] achieves on **66.80%** and **55.87%** on ACC@1 when using the image-input on the _Event-11_ and _Event-9185_ , respectively. We simply modify all the baseline methods and add a late-fusion module after the image/text encoder to support multimodal classification. Results show that TCL with multimodal inputs obtains gains of **1.89%** and **5.02%** compared with the singlemodal, which demonstrates that multimodal pre-training is more helpful for downstream multimodal tasks. 

## **C.3 Single- & Cross-Modal Retrieval** 

We design four kinds of single- & cross-modal retrieval tasks: image-to-image, text-to-text, image-to-text, and text-to-image. The construction of GT is based on the event annotations in _Gm_ (Fig. 4). We treat images or texts belonging to the same news event as a similar semantic cluster, and the goal of retrieval is to recall the nearest neighbors within this cluster. The features used for retrieval are derived from the output of the previous layer of the classifier. 

As shown in Tab. 11, TCL [86] achieves on **33.24%** , **43.37%** and **45.22%** R@1, R@5, R@10 on the zero-shot setting of image retrieval. The results are **58.89%** , **68.47%** and **73.91%** when fine-tuning the pre-trained parameters, which means the pre-training _→_ fine-tuning strategy is extremely beneficial for downstream retrieval. We provide more details about hyperparameters in Sec. C.5. 

## **C.4 Visual Task Adaptation** 

Visual Task Adaptation Benchmark (VTAB) [91] is a diverse, realistic, and challenging vision representation benchmark, containing 19 tasks and covering a broad spectrum of domains and semantics. These tasks are grouped into three sets: NATURAL, SPECIALIZED, and STRUCTURED which utilize natural world, professional technology and artificial environment images respectively. We benchmark models on VTAB with 

22 

|Table 11: **A new benchmark of the retrieval task.** Zero-shot means freezing the pre-trained|Table 11: **A new benchmark of the retrieval task.** Zero-shot means freezing the pre-trained|Table 11: **A new benchmark of the retrieval task.** Zero-shot means freezing the pre-trained|Table 11: **A new benchmark of the retrieval task.** Zero-shot means freezing the pre-trained|
|---|---|---|---|
|parameters then transfer to the|test set for inference. Fine-tune means tuning the pre-trained parameters|||
|in the training set before inference.<br>**Model**<br>**Retrieval**||**Zero-Shot**<br>**R@1 R@5 R@10**|**Fine-Tune**<br>**R@1 R@5 R@10**|
|CLIP [51]|IMAGE|32.41 41.96 43.92|55.97 67.44 71.28|
|DeCLIP [35]|IMAGE|32.75 42.36 44.38|56.96 66.59 70.95|
|ALBEF [30]|IMAGE|32.88 42.76 44.79|58.56 67.83 72.24|
|TCL[86]|IMAGE|33.24 43.37 45.22|58.89 68.47 73.91|
|CLIP|TEXT|33.02 42.56 46.03|56.50 65.12 70.20|
|DeCLIP|TEXT|34.00 43.97 47.11|55.87 65.20 70.35|
|ALBEF|TEXT|33.87 43.86 46.82|56.77 65.91 71.15|
|TCL|TEXT|34.67 44.25 47.67|56.60 65.50 70.54|
|CLIP|IMG-to-TXT|32.73 42.64 44.72|56.32 66.93 70.61|
|DeCLIP|IMG-to-TXT|32.96 42.84 45.17|57.21 66.80 71.26|
|ALBEF|IMG-to-TXT|33.20 42.97 45.32|58.43 67.59 71.95|
|TCL|IMG-to-TXT|33.37 43.25 46.04|58.70 67.88 72.33|
|CLIP|TXT-to-IMG|31.78 41.04 42.51|55.74 64.38 69.56|
|DeCLIP|TXT-to-IMG|32.13 41.55 42.99|55.84 65.12 70.32|
|ALBEF|TXT-to-IMG|31.95 41.32 42.85|57.21 66.04 71.50|
|TCL|TXT-to-IMG|32.56 42.04 43.74|57.17 65.92 71.47|



Table 12: **The comparison of** _**w/**_ **and** _**w/o UKnow**_ **pre-training.** Zero means the model is initialized with all-zero parameters _w/o_ pre-training. CLIP _[∗]_ means pre-training with origin CLIP contrast loss on our dataset. Ours means _UKnow_ pre-training. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training_images/nips_2024_UKnow_A_Unified_Knowledge_Protocol_with_Multimodal_Knowledge_Graph_Datasets_for_Reasoning_and_Vision-Language_Pre-Training.pdf-0023-02.png)


ACC@1. We fine-tune models for 10 epoch in each task and compute the inner product between outputs of images and label texts with prompts [51] through pre-trained image encoders and text encoders as the similarity score. As shown in Tab. 12, our approach obtains gains of avg. **1.14%** compared with the origin CLIP when fairly using the same _UKnow_ ’s data for the upstream pre-training. 

The backbone of CLIP is ViT-B/32. The cost of pre-train is 26h / 30epoch. The key hyperparameters are _bs: 512, lr: 0.001, warmup: 1e4, eps: 1e-8, beta1: 0.9, beta2: 0.999, dim: 512, AdamW_ . The detailed setting can be found in Sec. C.5. It is essential to highlight that the image-text PAIR constitutes only one type of data in our protocol. By leveraging the capabilities of _UKnow_ , our pre-trained CLIP model can effectively comprehend the inherent knowledge ingrained within the data, resulting in superior performance than the original CLIP model (as observed in Tab. 12, Row2, utilizing image-text PAIR only). 

## **C.5 Hyperparameters** 

Tab. 13 and Tab. 14 list the hyperparameters that differ on each models and are determined with the validation performance on our dataset. In particular, Tab. 13 lists 7 common hyperparameters, such as learning rate, batch size, warmup, epoch number, _etc._ , employed during pre-training. The pre-trained model is evaluated using a standard pipeline consisting of pre-training on Dataset1, fine-tuning on Dataset2-Train, and testing on either Dataset2-Test/Val. Therefore, we list the hyperparameters used during fine-tuning in Tab. 14, which are slightly different from Tab. 13. We omit some of the model results, since ALBEF and TCL share the same set of hyperparameters, and the original CLIP and CLIP-UKnow share the same set of parameters. 

## **C.6 Computation Complexity** 

Here we detail the time cost of pre-training and fine-tuning. The GPU is NVIDIA(R) A100, the memory of GPU is 81,251MiB, driver version is 470.154, CUDA version is 11.4. The CPU is Intel(R) Xeon(R) Platinum 8369B @ 2.90GHz with 15 physical computation cores. The environment is Python 3.6.12 with Torch 1.10.1. Results are as shown in Tab. 15 and Tab. 16. 

23 

Table 13: **Hyperparameters for models of pre-training.** 

|ble 13: **Hyperpara**|**meters f**|**or models**|**of pre-trainin**|
|---|---|---|---|
|**Hyperparameter**|**ALBEF**|**DeCLIP**|**CLIP-UKnow**|
|Learning Rate<br>0_._0001<br>0_._001<br>0_._001<br>Batch Size<br>128<br>128<br>512<br>Number of Epochs<br>30<br>30<br>30<br>Weight Decay<br>0_._02<br>0_._1<br>0_._1<br>Optimizer<br>`AdamW`<br>`AdamW`<br>`AdamW`<br>Feature Dim<br>256<br>512<br>512<br>Warmup<br>20epc<br>5000<br>10000<br>ble 14: **Hyperparameters for models of fne-tunin**<br>**Hyperparameter**<br>**ALBEF**<br>**DeCLIP**<br>**CLIP-UKnow**||||
|Learning Rate<br>Batch Size<br>Number of Epochs<br>Weight Decay<br>Optimizer<br>Feature Dim<br>Warmup|0_._0001<br>128<br>128<br>0_._02<br>`AdamW`<br>256<br>4epc|5_e_-5<br>256<br>20<br>0_._02<br>`AdamW`<br>512<br>6epc|5_e_-5<br>256<br>20<br>0_._02<br>`AdamW`<br>512<br>6epc|



Table 14: **Hyperparameters for models of fine-tuning.** 

Table 15: **The time cost of pre-training.** 

|Table 15: **T**|**he time co**|**st of p**|**re-tra**|**ining.**|
|---|---|---|---|---|
|**Model**|**Backbone**|**Epoch**|**Batch**|**Time/h**|
|DeCLIP|ViT-B/32|30|128|91|
|ALBEF|ViT-B/16|30|128|69|
|TCL|ViT-B/16|30|128|67|
|CLIP_∗_|ViT-B/32|30|512|25|
|CLIP-UKnow|ViT-B/32|30|512|26|



Table 16: **The time cost of downstream fine-tuning.** 

|Table 16: **The time cost of downstrea**|**m fne-tuning.**|
|---|---|
|**Model**<br>**Backbone**<br>_UKnow_Tasks<br>**Epoch Batch Time/h**|VTAB<br>**Epoch Batch Time/h**|
|DeCLIP<br>ViT-B/32<br>20<br>128<br>12<br>ALBEF<br>ViT-B/16<br>20<br>128<br>10<br>TCL<br>ViT-B/16<br>20<br>128<br>10<br>Zero_∗_<br>ViT-B/32<br>-<br>-<br>-<br>CLIP_∗_<br>ViT-B/32<br>20<br>256<br>8<br>CLIP-UKnow ViT-B/32<br>20<br>256<br>8|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>15<br>128<br>3<br>15<br>128<br>3<br>15<br>128<br>3|



## **D Discussion** 

## **D.1 Limitation and Future Work** 

Despite the strides made, our research bears certain limitations. First of all, our current dataset primarily centers on text and image modalities which serve as fundamental pillars for information storage and representation, but lack other useful modalities. In future work, we aim to diversify modalities by augmenting our dataset with a broader range of modalities ( _e.g.,_ audio, video, 3D, etc.) to facilitate exploration across various downstream tasks. Second, for each downstream task, we selected several basic yet most suitable methods for our work as our baseline, resulting in slight deviations with current state-of-the-art (SOTA) performance. Our primary objective lies in validating the efficacy of our proposed dataset and protocols, and demonstrating the most straightforward and intuitive approach for utilizing our dataset. Hence, we made certain trade-offs, sacrificing some performance by opting for a more rudimentary approach instead of pursuing the SOTA method to enhance understanding and usage. We anticipate that our simplified demonstration will stimulate the community to delve deeper into the potential enhancements that _UKnow_ can offer in improving performance. 

## **D.2 Societal Impact** 

As stated in Sec. 3.2, our dataset originates from publicly accessible international news sources via the Wikipedia API. These sources only contain events that are publicly available and do not include any sensitive information. Consequently, we confidently affirm that our research carries no potential negative societal impacts. 

24 

