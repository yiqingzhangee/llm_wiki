# **G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering** 

**Xiaoxin He**[1] **Yijun Tian**[2] **Yifei Sun**[1] **Nitesh V. Chawla**[2] **Thomas Laurent**[3] **Yann LeCun**[4] _[,]_[5] **Xavier Bresson**[1] **Bryan Hooi**[1] 

```
{xiaoxin,yifeisun,xaviercs,bhooi}@comp.nus.edu.sg
{yijun.tian,nchawla}@nd.edu, tlaurent@lmu.edu,yann@cs.nyu.edu
```

1National University of Singapore 2University of Notre Dame 3Loyola Marymount University 4New York University 5Meta AI 

## **Abstract** 

Given a graph with textual attributes, we enable users to ‘chat with their graph’: that is, to ask questions about the graph using a conversational interface. In response to a user’s questions, our method provides textual replies and highlights the relevant parts of the graph. While existing works integrate large language models (LLMs) and graph neural networks (GNNs) in various ways, they mostly focus on either conventional graph tasks (such as node, edge, and graph classification), or on answering simple graph queries on small or synthetic graphs. In contrast, we develop a flexible question-answering framework targeting real-world textual graphs, applicable to multiple applications including scene graph understanding, common sense reasoning, and knowledge graph reasoning. Toward this goal, we first develop a Graph Question Answering (GraphQA) benchmark with data collected from different tasks. Then, we propose our _G-Retriever_ method, introducing the first retrieval-augmented generation (RAG) approach for general textual graphs, which can be fine-tuned to enhance graph understanding via soft prompting. To resist hallucination and to allow for textual graphs that greatly exceed the LLM’s context window size, _G-Retriever_ performs RAG over a graph by formulating this task as a Prize-Collecting Steiner Tree optimization problem. Empirical evaluations show that our method outperforms baselines on textual graph tasks from multiple domains, scales well with larger graph sizes, and mitigates hallucination.[1] 

## **1 Introduction** 

**Graphs and Large Language Models (LLMs).** The advent of LLMs has significantly shaped the artificial intelligence landscape. As these models are applied to increasingly diverse tasks, their ability to process complex structured data will be increasingly vital. In particular, in our interconnected world, a significant portion of real-world data inherently possesses a graph structure, such as the Web, e-commerce, recommendation systems, knowledge graphs, and many others. Moreover, many of these involve graphs with textual attributes ( _i.e., textual graphs_ ), making them well-suited for LLM-centric methods. This has spurred interest in combining graph-based technologies, particularly graph neural networks (GNNs), with LLMs to enhance their reasoning on graphs [44, 15, 24]. 

**The Present Work: Enabling ‘Chat With Your Graph’.** While existing works integrate LLMs and GNNs in various ways, they mostly focus on conventional graph tasks such as node, edge and graph 

> 1Our codes and datasets are available at: `https://github.com/XiaoxinHe/G-Retriever` 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Unified Conversational Interface<br>New Benchmark GraphQA womenpart ofaddictioncapable ofhumannot capable ofappearancechangecreated byreceive actionis not aused forstopsearching forperfectionpositivecosmeticsurgery white, �u�ysheep grazing onmoutainscoveredeating blue, clear, cloudlygrasswaterskyabove benjamin franklinbifocalsinventors inventionslighting rodinventionsinventionsglass harmonicafranklinstove<br>Write a 500-word argument essay based on the Write an 500-word advertisement for this place in the What else did ben franklin invent?<br>New Architecture G-Retriever information provided in the explanation graph. scene graph that would make people want to visit it. Surrounding this question, develop an educationaltour through the knowledge graph, highlightinginteresting facts and connections in an engaging way.<br>The explanation graph provides a detailed analysis of Sure! Here's an example of a 500-word advertisement<br>the reasons why women undergo cosmetic surgery. for the place based on the scene graph:<br>Retrieval Augmented GenerationGraph Prompt TuningGraph Textualization Techniques According to the graph, the main reasons include thedesire to change one's appearance, the in�uence ofsocietal beauty standards, and the addiction tosearching for perfection. These reasons are furthersupported by the various nodes and edges in thegraph, which provide additional insights into themotivations and consequences of cosmetic surgery.Firstly, the node labeled "human" highlights thefundamental human desire to feel attractive andcon�dent in one's appearance... "Experience the breathtaking beauty of our place,where the rolling hills meet the sparkling water of theswimming pool. Take a leisurely stroll through thelush green �elds, and let the gentle breeze caressyour face as you gaze out at the picturesque view. Thesheep graze peacefully in the distance, their whitecoats glistening in the sunlight. The sky is a brilliantblue, with �u�y white clouds drifting lazily across.Come and bask in the serene ambiance of our place,where the natural beauty will leave you in awe." Great! Based on the provided knowledge graph,here's an educational tour that highlights interestingfacts and connections related to Benjamin Franklin'sinventions and other notable achievements:1. Franklin's Lightning Rod: Start by exploringFranklin's most famous invention - the lightning rod.Discover how Franklin's invention protected buildingsfrom lightning strikes and how it changed the waypeople understood electricity...<br>**----- End of picture text -----**<br>


Figure 1: We develop a flexible question-answering framework targeting real-world textual graph applications via a unified conversational interface. Presented here are examples showcasing the model’s adeptness in handling generative and creative queries in practical graph-related tasks: common sense reasoning, scene understanding, and knowledge graph reasoning, respectively. 

classification [8], or answering simple questions on small or synthetic graphs [44, 31]. In contrast, we develop a flexible question-answering framework targeting complex and real-world graphs. This framework enables users to ‘chat with their graph’ via a unified conversational interface, representing a leap towards intuitive interaction with graph data, as demonstrated in Figure 1. 

**The Need for a Comprehensive GraphQA Benchmark.** Question answering (QA) is a fundamentally important task in natural language processing, serving as a key benchmark for assessing LLMs and providing a unified interface for various capabilities. Despite extensive research in QA, a comprehensive benchmark specifically tailored for the graph modality is lacking. In contrast to existing benchmarks that focus on basic graph-based reasoning tasks such as node degree, edge existence, and shortest path [6, 44], our benchmark addresses complex and real-world graph applications including common sense reasoning, scene understanding, and knowledge graph reasoning (refer to Figure 2). This is vital for measuring progress toward a model capable of answering a wide range of questions about graphs from diverse applications. 

**New Architecture for GraphQA.** To enable effective and efficient graph QA, even on large graphs, we propose _G-Retriever_ , a new framework combining the strengths of GNNs, LLMs, and RAG (Figure 3). Next, we will discuss the motivation, strengths, and details of our model. 

**Tackling Hallucination in Graph LLMs.** LLMs are prone to hallucination, a phenomenon where the generated content is factually inaccurate or nonsensical [12]. We validate the presence of this issue in graph settings. In particular, we employ a baseline method that adapts MiniGPT-4 [57] to graphs, where a frozen LLM interacts with a trainable GNN that encodes graph data as a soft prompt, as in GraphToken [31]. Our findings, shown in Table 1, indicate that hallucination, an important problem in text-based LLMs, is also prevalent in Graph LLMs. This may be attributed to the baseline’s inability to recall the entire graph structure from a single graph embedding, leading to the generation of incorrect nodes or edges during the QA task. In contrast, by employing RAG for direct information retrieval from the actual graph, our _G-Retriever_ mitigates this issue, as substantiated by Table 1. 

**Enhancing Scalability and Efficiency in Graph LLMs.** Recent research endeavors have explored translating graphs into natural language, such as by flattening nodes and edges into a text sequence, enabling their processing by LLMs for graph-based tasks [56, 6]. However, this method faces critical scalability issues. Converting a graph with thousands of nodes and edges into a text sequence results in an excessive number of tokens, surpassing the input capacity of many LLMs. An alternative of truncating the graph text sequence to fit the LLM’s input token limit leads to loss of information and response quality. _G-Retriever_ overcomes these issues with its RAG component, which allows for effective scaling to larger graphs by selectively retrieving only relevant parts of the graph. 

**Tailoring the RAG Approach to Graphs.** Existing RAG methodologies are primarily designed for simpler data types or knowledge graphs, where information is retrieved in a manner isolated from the graph structure [7, 1, 36, 16]. Hence, we introduce a new retrieval approach for general textual graphs. Notably, we formulate subgraph retrieval as a Prize-Collecting Steiner Tree (PCST) 

2 

Table 1: Observation and mitigation of hallucination in graph LLMs. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0003-01.png)


**----- Start of picture text -----**<br>
|||||
|---|---|---|---|
|elephants|elephant|
|User|in|
|eating|bushes|standing by|
|grass|on the side of|river|
|What kind of animal is in the bushes?|
|Answer the question and list the nodes or edges from the graph that support your answer.|
|[Incorrect response with hallucinated nodes and edges]|
|The animal in the bushes is a deer.|
|Nodes:|
|* Deer (node 1)|
|LLM w/ Graph Prompt Tuning|* Bushes (node 2)|
|Edges:|
|* Deer|→|Bushes (edge 1)|
|* Deer|→|Grass (edge 2)|
|* Bushes|→|Grass (edge 3)|
|[Correct response with accurate node and edge references]|
|Based on the information provided in the graph, the animal in the bushes is an elephant.|
|The nodes that support this answer are:|
|G-Retriever|(Ours)|* Node 1:|The elephant is mentioned in the first line of the graph.|
|The edges that support this answer are:|
|* The edge connects the elephant to the bushes, indicating that the elephant is in the bushes.|
|Therefore, the answer to the question is:|The animal in the bushes is an elephant.|

**----- End of picture text -----**<br>


optimization problem, which takes the neighborhood information into account during retrieval. This also allows the return of a subgraph most relevant to a query, thereby improving explainability. 

The contributions of this paper are outlined as follows: 

- **Pioneering the integration of Graph RAG.** We present the first retrieval approach for general textual graph tasks, which greatly enhances scalability and efficiency. 

- **Enabling ‘Chat with Your Graph’.** We develop a flexible question answering framework to handle complex and real-world textual graphs through a unified conversational interface. 

- **Introducing A Novel GraphQA Benchmark.** We introduce a diverse benchmark targeted at real-world graph question answering, filling a crucial research gap. 

- **Empirical Findings.** We demonstrate the efficiency and effectiveness of _G-Retriever_ in multiple domains and present the significant finding of hallucination in graph LLMs. 

## **2 Related Work** 

**Graphs and Large Language Models.** A significant body of research has emerged at the intersection of graph-based techniques and LLMs [30, 24, 15, 44, 54]. This exploration spans diverse aspects, ranging from the design of general graph models [47, 25, 51, 19, 40, 31], and multi-modal architectures [23, 49] to practical applications. Noteworthy applications include fundamental graph reasoning [52, 3, 56], node classification [8, 11, 39, 5, 50, 4, 33], graph classification/regression [32, 55], and leveraging LLMs for knowledge graph-related tasks [41, 14, 29]. 

**Retrieval-Augmented Generation (RAG).** The concept of Retrieval-Augmented Generation, initially proposed by Lewis et al. [21], has gained increased attention for its ability to mitigate the issue of hallucination within LLMs and enhance trustworthiness and explainability [7]. Despite its success in language-related tasks, the application of retrieval-based approaches to general graph tasks remains largely unexplored. Most existing work focuses primarily on the knowledge graph [38, 1, 36, 16]. Our research is the first to apply a retrieval-based approach to general graph tasks, marking a novel advancement in the field and demonstrating the versatility of RAG beyond language processing. 

**Parameter-Efficient Fine-Tuning (PEFT).** The field of LLMs has witnessed significant advancements through various parameter-efficient fine-tuning techniques. These methodologies have played a crucial role in refining LLMs, boosting their performance while minimizing the need for extensive parameter training. Notable among these techniques are prompt tuning, as introduced by Lester et al. [20], and prefix tuning, proposed by Li and Liang [22]. Furthermore, methods like LoRA [10], 

3 

and the LLaMA-adapter [53], have been influential. These advancements in PEFT have laid the foundation for the development of sophisticated multimodal models. Prominent examples in this domain include MiniGPT-4 [57], LLaVA [26], and NExT-Chat [46]. There are also emerging efforts in applying PEFT to graph LLMs, such as GraphLLM [3] and GraphToken [31] for basic graph reasoing tasks and GNP [41] for multi-option QA on knowledge graphs. 

## **3 Formalization** 

This section establishes the notation and formalizes key concepts related to textual graphs, language models for text encoding, and large language models and prompt tuning. 

**Textual Graphs.** A textual graph is a graph where nodes and edges possess textual attributes. Formally, it can be defined as _G_ = ( _V, E, {xn}n∈V , {xe}e∈E_ ), where _V_ and _E_ represent the sets of nodes and edges, respectively. Additionally, _xn ∈ D[L][n]_ and _xe ∈ D[L][e]_ denote sequential text associate with a node _n ∈ V_ or an edge _e ∈ E_ , where _D_ represents the vocabulary, and _Ln_ and _Le_ signify the length of the text associated with the respective node or edge. 

**Language Models for Text Encoding.** In the context of textual graphs, language models (LMs) are essential for encoding the text attributes associated with nodes and edges, thereby learning representations that capture their semantic meaning. For a node _n_ with text attributes _xn ∈ D[L][n]_ , an LM encodes these attributes as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0004-05.png)


where _zn_ is the output of the LM, and _d_ is the dimension of the output vector. 

**Large Language Models and Prompt Tuning.** LLMs have introduced a new paradigm for taskadaptation known as “pre-train, prompt, and predict”, replacing the traditional “pre-train, fine-tune” paradigm. In this paradigm, the LLM is first pre-trained on a large corpus of text data to learn general language representations. Then, rather than fine-tuning the model on task-specific labeled data, the model is prompted with a textual prompt that specifies the task and context. Subsequently, the model generates the output directly based on the prompt and the input. 

The LLM, parameterized by weights _θ_ , takes a sequence of tokens _X_ , and a prompt _P_ as input, and generates a sequence of tokens _Y_ = _{y_ 1 _, y_ 2 _, . . . , yr}_ as output. Formally, the probability distribution of the output sequence given the concatenated input sequence and prompt, _i.e.,_ [ _P_ ; _X_ ], is defined as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0004-09.png)


Here, _y<i_ represents the prefix of sequence _y_ up to position _i −_ 1, and _p_ ( _yi|y<i,_ [ _P_ ; _X_ ]) represents the probability of generating token _yi_ given _y<i_ and [ _P_ ; _X_ ]. 

Soft prompt tuning eliminates the need for manual prompt design. Given a series of _p_ tokens _X_ = _{x_ 1 _, x_ 2 _, . . . , xp}_ , after being processed by the text embedder, it forms a matrix _Xe ∈_ R _[p][×][d][l]_ , where _dl_ is the dimension of the embedding space. Soft prompts can be represented as parameters _Pe ∈_ R _[q][×][d][l]_ , where _q_ is the length of the prompt. The prompt is then concatenated with the embedded input, forming a single matrix [ _Pe_ ; _Xe_ ] _∈_ R[(] _[q]_[+] _[p]_[)] _[×][d][l]_ . This combined matrix is processed by the self-attention layers in LLM as usual. Training involves maximizing the likelihood of _Y_ through backpropagation, with gradient updates applied solely to _Pe_ , while _θ_ remains fixed. 

## **4 Proposed GraphQA Benchmark** 

Our GraphQA represents a comprehensive and diverse benchmark for graph question-answering. It is tailored to assess the capabilities of models in answering a wide range of questions about graphs across diverse domains. 

## **4.1 Data Format** 

Each entry in the GraphQA benchmark consists of a textual graph, a question related to the graph, and one or more corresponding answers, as illustrated in Figure 2. 

4 

Table 2: Summary of datasets used in GraphQA benchmark. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0005-01.png)


**----- Start of picture text -----**<br>
Dataset ExplaGraphs SceneGraphs WebQSP<br>#Graphs 2,766 100,000 4,737<br>Avg. #Nodes 5.17 19.13 1370.89<br>Avg. #Edges 4.25 68.44 4252.37<br>Node Attribute Commonsense concepts Object attributes ( e.g.,  color, shape) Entities in Freebase<br>Edge Attribute Commonsense relations Relations ( e.g.,  actions, spatial relations) Relations in Freebase<br>Task Common sense reasoning Scene graph question answering Knowledge based question answering<br>Evaluation Matrix Accuracy Accuracy Hit@1<br>Explanation Graph Scene Graph Knowledge Graph<br>is a women andmen women capable of person justin bieber jeremy<br>citizens causes causes desire help the country to the right of sitting in front of sibling_s parents childrenbieber<br>have same rights be in combat woman computer m.0gxnnwp sibling jaxon bieber<br>Textualized Graph Textualized Graph Textualized Graph<br>node_id, node_attr src, edge_attr, dst node_id, node_attr node_id, node_attr src, edge_attr, dst<br>0, women and men 0, is a,1 2, name: computer; (x,y,w,h): (8, 119, 34, 32) 15, justin bieber 15, people.person.parents, 356<br>1, citizens 1, causes, 2 3, name: person; attribute: sitting; (x,y,w,h): (169, 75, 49, 40) 294, jaxon bieber 15, people.person.sibling_s, 551<br>2, have same rights 2, causes, 3 15, name: woman; (x,y,w,h): (255, 18, 235, 292) 356, jeremy bieber 356, people.person.children, 294<br>3, women 3, capable of, 4 src, edge_attr, dst 551, m.0gxnnwp 551, people.sibling_relationship.sibling, 294<br>4, help the country 4, desires, 5 15, to the right of, 3<br>5, be in combat 2, in front of, 3<br>Question Question Question<br>Arguement 1: Women should not be in combat. Question: Is there a woman to the right of the person behind the Question: what is the name of justin bieber brother<br>Arguement 2: Women and men have the same rights. computer?<br>Question: Do argument 1 and argument 2 support or counter each other?<br>Answer in one word in the form of 'support' or 'counter'.<br>Answer Answer Answer<br>    counter   yes   jaxon bieber<br>**----- End of picture text -----**<br>


Figure 2: Illustrative examples from the GraphQA benchmark datasets. 

**Textual Graphs.** The textual graph is converted into a natural language format, resulting in a list of nodes and edges, akin to a CSV file format. It is important to note that while multiple methods exist for textualizing a graph, our focus is not on identifying the optimal solution. Instead, we prioritize a straightforward yet empirically effective approach for representing graphs in natural language, facilitating the benchmark’s use in diverse GraphQA scenarios. 

**Questions and Answers.** Questions are designed to explore specific elements or relationships within the graph. Answers, residing within the attributes of nodes or edges, often require multi-hop reasoning for accurate identification. 

## **4.2 Description of Datasets** 

The GraphQA benchmark integrates three existing datasets: `ExplaGraphs` , `SceneGraphs` , and `WebQSP` . Table 2 presents the summary statistics of these datasets. It is important to note that these datasets were not originally developed for this work. However, a significant contribution of our research is the standardization and processing of these diverse datasets into a uniform data format suitable for the GraphQA benchmark. These datasets, previously utilized in different contexts, are reintroduced with a new focus tailored for GraphQA. For a detailed comparison with the original datasets, see the Appendix C. 

`ExplaGraphs` is a dataset for generative commonsense reasoning, focusing on creating explanation graphs for stance prediction in debates. It offers detailed, unambiguous commonsense-augmented graphs to evaluate arguments supporting or refuting a belief. The primary task is to assess whether arguments are supportive or contradictory, using accuracy as the metric. We have converted the triplet-form provided in Saha et al. [35] into a standard graph format. 

`SceneGraphs` , a visual question answering dataset, includes 100,000 scene graphs. Each graph details objects, attributes, and relations within an image. This dataset challenges users with tasks requiring spatial understanding and multi-step inference. The task is to answer open-ended questions based on a textual description of a scene graph, evaluated on accuracy. We have sampled from the GQA dataset [13] and constructed standard graphs from the provided JSON files. 

`WebQSP` is a large-scale multi-hop knowledge graph QA dataset consisting of 4,737 questions. It was proposed by Yih et al. [48] and, following Luo et al. [28], utilizes a subset of Freebase, encompassing facts within 2-hops of entities mentioned in the questions. The task involves answering questions that 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0006-00.png)


**----- Start of picture text -----**<br>
Step 1: Indexing Step 4: Generation<br>frozen<br>all bad canada node attributes trainable jaxon bieber<br>bieberjustinprofessionalbumsiblingparent m.0gxnnwpjeremybieber siblingchildrennationalitygenderbieberjaxonedge attributes LMLM (Self Attention Layers)LLM node_id, node_attr15, justin bieber294, jaxon bieber356, jeremy bieber551, m.0gxnnwpsrc, edge_attr, dst294, parents, 356356, children, 15<br>producerrecord male Projection LLM 551, sibling, 294551, sibling, 15<br>Graph Encoder (Text Embedder) Question: What is the name<br>of justin bieber brother?<br>Question: of justin bieber brother? What is the name LM Storage Answer:<br>jeremy<br>justin bieber, this is justin bieber, jeremy bieber,  justin parent bieber<br>justin bieber fan club, justin ... bieber sibling children<br>sibling, sibling_s, hangout, friendship, friend ... m.0gxnnwp sibling bieberjaxon<br>Step 2: Retrieval Step 3: Subgraph Construction<br>**----- End of picture text -----**<br>


Figure 3: Overview of the proposed _G-Retriever_ : 1) Indexing: Graphs are indexed for efficient query processing; 2) Retrieval: The most semantically relevant nodes and edges are retrieved, conditioned on the query; 3) Subgraph Construction: A connected subgraph is extracted, covering as many relevant nodes and edges as possible while maintaining a manageable graph size; 4) Generation: An answer is generated using a ‘graph prompt’, a textualized graph, and the query. 

require multi-hop reasoning. Given the possibility of multiple answers for the same question, the hit@1 metric is used to assess the precision of the top returned answer. 

## **5 G-Retriever** 

In this section, we introduce _G-Retriever_ , a new architecture tailored for GraphQA, which integrates the strengths of GNNs, LLMs, and RAG. To allow efficient fine-tuning while preserving the LLM’s pretrained language capabilities, we freeze the LLM and use a soft prompting approach on the output of the GNN. Our RAG-based design mitigates hallucinations through direct retrieval of the graph, while allowing our approach to scale to graphs exceeding the LLM’s context window size. To adapt RAG to graphs, we formulate subgraph retrieval as a PCST optimization problem. This approach also allows us to enhance explainability by returning the retrieved subgraph. 

_G-Retriever_ comprises four main steps: indexing, retrieval, subgraph construction and generation, as depicted in Figure 3. The implementation details of each step are elaborated in the following sections. 

## **5.1 Indexing** 

We initiate the RAG approach by generating node and graph embeddings using a pre-trained LM. These embeddings are then stored in a nearest neighbor data structure. 

To elaborate, consider _xn ∈ D[L][n]_ as the text attributes of node _n_ . Utilizing a pre-trained LM, such as SentenceBert [34], we apply the LM to _xn_ , yielding the representation _zn_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0006-09.png)


where _d_ denotes the dimension of the output vector. Similar preprocessing steps are applied to edges. Refer to Figure 3, Step 1 for an illustrative representation. 

## **5.2 Retrieval** 

For retrieval, we employ the same encoding strategy to the query _xq_ , to ensure consistent treatment of textual information: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0006-13.png)


Next, to identify the most relevant nodes and edges for the current query, we use a k-nearest neighbors retrieval approach. This method yields a set of ‘relevant nodes/edges’ based on the similarity between the query and each node or edge. The retrieval operation is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0006-15.png)


6 

where _zn_ and _ze_ are the embeddings of node _n_ and edge _e_ , respectively. We use the cosine similarity function, cos( _·, ·_ ), to measure the similarity between the query representation and the node/edge embeddings. The argtopk operation retrieves the top-k elements based on this similarity, providing a set of nodes _Vk_ and edges _Ek_ considered most relevant to the query. See Step 2 of Figure 3. 

## **5.3 Subgraph Construction** 

This step aims to construct a subgraph that encompasses as many relevant nodes and edges as possible, while keeping the graph size manageable. This approach offers two key benefits: Firstly, it helps to filter out nodes and edges that are not pertinent to the query. This is crucial because irrelevant information can overshadow the useful data, potentially diverting the focus of the subsequent LLM from the information of interest. Secondly, it enhances efficiency; by keeping the graph size manageable, it becomes feasible to translate the graph into natural language and then input it into the LLM for processing. The Prize-Collecting Steiner Tree algorithm [2] serves as our primary method for identifying such optimally sized and relevant subgraphs. See Step 3 in Figure 3. 

**Prize-Collecting Steiner Tree (PCST).** The PCST problem aims to find a connected subgraph that maximizes the total prize values of its nodes while minimizing the total costs of its edges. Our approach assigns higher prize values to nodes and edges more relevant to the query, as measured by cosine similarity. Specifically, the top _k_ nodes/edges are assigned descending prize values from _k_ down to 1, with the rest assigned zero. The node prize assignment is as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0007-04.png)


Edge prizes are assigned similarly. The objective is to identify a subgraph, _S[∗]_ = ( _V[∗] , E[∗]_ ), that optimizes the total prize of nodes and edges, minus the costs associated with the size of the subgraph: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0007-06.png)


where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0007-08.png)


and _Ce_ denotes a predefined cost per edge, which is adjustable to control the subgraph size. 

The original PCST algorithm is designed for node prizes only. However, given the significance of edge semantics in certain scenarios, we adapt the algorithm to accommodate edge prizes as follows: Consider an edge e with a cost _Ce_ and a prize _Pe_ . If _Ce > Pe_ , it can be treated as a reduced edge cost of _Ce − Pe_ . However, if _Pe > Ce_ , negative edge costs are not allowed in the original algorithm. Our solution involves replacing edge _e_ with a ‘virtual node’ _ve_ , connected to both endpoints of _e_ . This virtual node is assigned a prize of _Pe − Ce_ , and the cost of the two new edges leading to the virtual node is set to zero. This modification effectively mirrors the original problem, as including edge _e_ in the original graph is analogous to including the virtual node in the modified graph. Finally, we optimize the PCST problem using a near-linear time approach [9]. 

## **5.4 Answer Generation** 

**Graph Encoder.** Let _S[∗]_ = ( _V[∗] , E[∗]_ ) represent the retrieved subgraph. We use a graph encoder to model the structure of this graph, specifically using a standard Graph Attention Network (GAT) [43]. Our approach for encoding the retrieved subgraph is defined as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0007-13.png)


Here, POOL denotes the mean pooling operation, and _dg_ is the dimension of the graph encoder. 

**Projection Layer.** We incorporate a multilayer perceptron (MLP) to align the graph token with the vector space of the LLM: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0007-16.png)


where _dl_ is the dimension of the LLM’s hidden embedding. 

**Text Embedder.** To leverage the text-reasoning capabilities of LLMs, we transform the retrieved subgraph _S[∗]_ into a textual format. This transformation involves flattening the textual attributes of the 

7 

nodes and edges, as illustrated in the green box in Figure 2. We refer to this operation as textualize( _·_ ). Subsequently, we combine the textualized graph with the query to generate a response. Let _xq_ denote the query; we concatenate it with the textualized graph textualize( _S[∗]_ ). We then map the result to an embedding _ht_ using a text embedder, which is the first layer of a pretrained and frozen LLM: 

_ht_ = TextEmbedder([textualize( _S[∗]_ ); _xq_ ]) _∈_ R _[L][×][d][l] ,_ (11) 

where [; ] represents the concatenation operation, and _L_ is the number of tokens. 

**LLM Generation with Graph Prompt Tuning.** The final stage involves generating the answer _Y_ given the graph token _h_[ˆ] _g_ , acting as a soft prompt, and the text embedder output _ht_ . These inputs are fed through the self-attention layers of a pretrained frozen LLM, with parameter _θ_ . The generation process is represented as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0008-04.png)


where [ _h_[ˆ] _g_ ; _ht_ ] concatenates the graph token _h_[ˆ] _g_ and the text embedder output _ht_ . While _θ_ is frozen, the graph token _h_[ˆ] _g_ receives gradients, enabling the optimization of the parameters of the graph encoder _ϕ_ 1 and the projection layer _ϕ_ 2 through standard backpropagation. 

## **6 Experiments** 

## **6.1 Experiment Setup** 

In the indexing step, we use SentenceBert [34] as the LM to encode all node and edge attributes. In the generation step, we use the open-source Llama2-7b [42] as the LLM and Graph Transformer [37] as the graph encoder. Additional details are provided in Appendix B.1. 

## **6.2 Main Results** 

In our experiments, we consider three model configurations: _1) Inference-only_ : Using a frozen LLM for direct question answering; _2) Frozen LLM w/ prompt tuning (PT)_ : Keeping the parameters of the LLM frozen and adapting only the prompt; _3) Tuned LLM_ : Fine-tuning the LLM with LoRA [10]. We provide more details in Appendix B.2. 

Table 3: Performance comparison across `ExplaGraphs` , `SceneGraphs` , and `WebQSP` datasets for different configurations, including Inference-only, Frozen LLM with prompt tuning (PT), and Tuned LLM settings. Mean scores and standard deviations (mean ± std) are presented. The first best result for each task is highlighted in **bold** and the second best result is highlighted with an underline. 

|Setting|Method|`ExplaGraphs`|`SceneGraphs`|`WebQSP`|
|---|---|---|---|---|
||||||
|Inference-only|Zero-shot<br>Zero-CoT [18]<br>CoT-BAG [44]<br>KAPING [1]|0.5650<br>0.5704<br>0.5794<br>0.6227|0.3974<br>0.5260<br>0.5680<br>0.4375|41.06<br>51.30<br>39.60<br>52.64|
||||||
|Frozen LLM w/ PT|Prompt tuning<br>GraphToken [31]<br>_G-Retriever_<br>∆Prompt tuning|0.5763 ± 0.0243<br>0.8508 ± 0.0551<br>0.8516 ± 0.0092<br>_↑_47.77%|0.6341 ± 0.0024<br>0.4903 ± 0.0105<br>0.8131 ± 0.0162<br>_↑_28.23%|48.34 ± 0.64<br>57.05 ± 0.74<br>70.49 ± 1.21<br>_↑_45.81%|
||||||
|Tuned LLM|LoRA<br>_G-Retriever_w/ LoRA<br>∆LoRA|0.8538 ± 0.0353<br>**0.8705 ± 0.0329**<br>_↑_1.95%|0.7862 ± 0.0031<br>**0.8683 ± 0.0072**<br>_↑_11.74%|66.03 ± 0.47<br>**73.79 ± 0.70**<br>_↑_10.44%|



Table 3 demonstrates the effectiveness of our method across three datasets in various configurations. In the inference-only setting, _G-Retriever_ surpasses all baselines. Notably, LLM can perform even better when no graph knowledge is provided ( _i.e.,_ question only), which might be attributed to the complexity and potential noise in the knowledge. For frozen LLM with prompt tuning, _G-Retriever_ outperforms traditional prompt tuning and GraphToken [31], a graph prompt tuning-based method, with average performance increases of 40.6% and 30.8% respectively. Furthermore, when tuned with LoRA, _G-Retriever_ achieves the best performance. 

8 

Table 4: Retrieval on graphs significantly improves efficiency. 

|Dataset|Before Retrieval (Avg.)<br># Tokens<br># Nodes<br>Min/Epoch|After Retrieval (Avg.)<br># Tokens<br># Nodes<br>Min/Epoch|
|---|---|---|
|`SceneGraphs`<br>`WebQSP`|1,396<br>19<br>123.1<br>100,627<br>1,371<br>18.7|235 (_↓_83%)<br>5 (_↓_74%)<br>86.8 (_↓_29%)<br>610 (_↓_99%)<br>18 (_↓_99%)<br>6.2(_↓_67%)|



Table 5: Quantitative comparison of halluciTable 6: Ablation study on the `WebQSP` dataset. nation on the `SceneGraphs` dataset. 

|Table 5: Quantitative comparison of halluci-<br>nation on the`SceneGraphs`dataset.|Table 6: Ablation study on the`WebQSP`dataset.|
|---|---|
|Baseline<br>_G-Retriever_<br>Valid Nodes<br>31%<br>77%<br>Valid Edges<br>12%<br>76%<br>Fully Valid Graphs<br>8%<br>62%|Method<br>Hit@1<br>∆_G-Retriever_|
||w/o Graph Encoder<br>54.62 ± 0.78<br>_↓_22.51%<br>w/o Projection Layer<br>69.70 ± 0.68<br>_↓_1.11%<br>w/o Textualized Graph<br>56.96 ± 1.83<br>_↓_19.19%<br>w/o Retrieval<br>63.84 ± 0.41<br>_↓_9.43%|



## **6.3 Efficiency Evaluation** 

The efficiency of our approach is highlighted by the data in Table 4. Implementing our graph-based retrieval significantly decreases the number of tokens required to describe the graphs in text, reduces the number of nodes in graphs, and speeds up the training process. Specifically, for the `SceneGraphs` dataset, tokens decreased by 83%, nodes by 74%, and training time by 29%. For the `WebQSP` dataset, tokens decreased by 99%, nodes by 99%, and training time by 67%. These substantial reductions demonstrate the method’s efficiency and potential in managing large-scale graph data. 

## **6.4 Mitigation of Hallucination** 

To evaluate hallucination, we instructed the models to answer graph-related questions, specifically identifying supporting nodes or edges from the graph. We manually reviewed 100 responses from both our method and the baseline ( _i.e.,_ LLM with graph prompt tuning), verifying the existence of the nodes and edges referenced in the model’s output within the actual graph. Table 5 shows that _G-Retriever_ significantly reduces hallucinations by 54% compared to the baseline, as our graph retrieval ensures that the data is sourced directly from the actual graph, leading to less hallucination. See Appendix F for details. 

## **6.5 Ablation Study** 

In this ablation study, we assess the individual impact of key components within our pipeline. As shown in Table 6, there are performance drops when any of these components are removed, with the graph encoder and textualized graph showing declines of 22.51% and 19.19%, respectively. This demonstrates their complementary effects in representing the graph in both textual and embedded formats. Additionally, the retrieval on graphs is also important to the overall performance. Further details are available in Appendix B.3. We also present additional studies on our framework: it is robust to the choice of graph encoders (see Appendix B.4) and benefits from the increased scale of LLMs (see Appendix B.5). 

Additionally, we include a detailed comparison with existing retrieval methods (see Appendix D), a discussion on the complexity (see Appendix E), and demonstrations on how to use _G-Retriever_ to ‘chat with your graph’ (see Appendix G). 

## **7 Conclusion** 

In this work, we introduce a new GraphQA benchmark for real-world graph question answering and present _G-Retriever_ , an architecture adept at complex and creative queries. Experimental results show that _G-Retriever_ surpasses baselines in textual graph tasks across multiple domains, scales effectively with larger graph sizes, and demonstrates resistance to hallucination. 

9 

**Limitations and Future Work:** Currently, _G-Retriever_ employs a static retrieval component. Future developments could investigate more sophisticated RAG where the retrieval is trainable. 

## **Acknowledgment** 

XB is supported by NUS Grant ID R-252-000-B97-133. 

## **References** 

- [1] Jinheon Baek, Alham Fikri Aji, and Amir Saffari. Knowledge-augmented language model prompting for zero-shot knowledge graph question answering. In Bhavana Dalvi Mishra, Greg Durrett, Peter Jansen, Danilo Neves Ribeiro, and Jason Wei, editors, _Proceedings of the 1st Workshop on Natural Language Reasoning and Structured Explanations (NLRSE)_ , pages 78–106, Toronto, Canada, June 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.nlrse-1.7. URL `https://aclanthology.org/2023.nlrse-1.7` . 

- [2] Daniel Bienstock, Michel X Goemans, David Simchi-Levi, and David Williamson. A note on the prize collecting traveling salesman problem. _Mathematical programming_ , 59(1-3):413–420, 1993. 

- [3] Ziwei Chai, Tianjie Zhang, Liang Wu, Kaiqiao Han, Xiaohai Hu, Xuanwen Huang, and Yang Yang. Graphllm: Boosting graph reasoning ability of large language model. _arXiv preprint arXiv:2310.05845_ , 2023. 

- [4] Zhikai Chen, Haitao Mao, Hang Li, Wei Jin, Hongzhi Wen, Xiaochi Wei, Shuaiqiang Wang, Dawei Yin, Wenqi Fan, Hui Liu, et al. Exploring the potential of large language models (llms) in learning on graphs. _arXiv preprint arXiv:2307.03393_ , 2023. 

- [5] Zhikai Chen, Haitao Mao, Hongzhi Wen, Haoyu Han, Wei Jin, Haiyang Zhang, Hui Liu, and Jiliang Tang. Label-free node classification on graphs with large language models (llms). _arXiv preprint arXiv:2310.04668_ , 2023. 

- [6] Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. Talk like a graph: Encoding graphs for large language models. _arXiv preprint arXiv:2310.04560_ , 2023. 

- [7] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. _arXiv preprint arXiv:2312.10997_ , 2023. 

- [8] Xiaoxin He, Xavier Bresson, Thomas Laurent, Adam Perold, Yann LeCun, and Bryan Hooi. Harnessing explanations: Llm-to-lm interpreter for enhanced text-attributed graph representation learning. _arXiv preprint arXiv:2305.19523_ , 2023. 

- [9] Chinmay Hegde, Piotr Indyk, and Ludwig Schmidt. A nearly-linear time framework for graphstructured sparsity. In _International Conference on Machine Learning_ , pages 928–937. PMLR, 2015. 

- [10] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_ , 2021. 

- [11] Jin Huang, Xingjian Zhang, Qiaozhu Mei, and Jiaqi Ma. Can llms effectively leverage graph structural information: when and why. _arXiv preprint arXiv:2309.16595_ , 2023. 

- [12] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions, 2023. 

- [13] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 6700–6709, 2019. 

10 

- [14] Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, and Ji-Rong Wen. Structgpt: A general framework for large language model to reason over structured data. _arXiv preprint arXiv:2305.09645_ , 2023. 

- [15] Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, and Jiawei Han. Large language models on graphs: A comprehensive survey. _arXiv preprint arXiv:2312.02783_ , 2023. 

- [16] Minki Kang, Jin Myung Kwak, Jinheon Baek, and Sung Ju Hwang. Knowledge graphaugmented language models for knowledge-grounded dialogue generation, 2023. 

- [17] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. _arXiv preprint arXiv:1609.02907_ , 2016. 

- [18] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. _Advances in neural information processing systems_ , 35:22199–22213, 2022. 

- [19] Bin Lei, Chunhua Liao, Caiwen Ding, et al. Boosting logical reasoning in large language models through a new framework: The graph of thought. _arXiv preprint arXiv:2308.08614_ , 2023. 

- [20] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. _arXiv preprint arXiv:2104.08691_ , 2021. 

- [21] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. _Advances in Neural Information Processing Systems_ , 33:9459–9474, 2020. 

- [22] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. _arXiv preprint arXiv:2101.00190_ , 2021. 

- [23] Xin Li, Dongze Lian, Zhihe Lu, Jiawang Bai, Zhibo Chen, and Xinchao Wang. Graphadapter: Tuning vision-language models with dual knowledge graph. _arXiv preprint arXiv:2309.13625_ , 2023. 

- [24] Yuhan Li, Zhixun Li, Peisong Wang, Jia Li, Xiangguo Sun, Hong Cheng, and Jeffrey Xu Yu. A survey of graph meets large language model: Progress and future directions. _arXiv preprint arXiv:2311.12399_ , 2023. 

- [25] Hao Liu, Jiarui Feng, Lecheng Kong, Ningyue Liang, Dacheng Tao, Yixin Chen, and Muhan Zhang. One for all: Towards training one graph model for all classification tasks. _arXiv preprint arXiv:2310.00149_ , 2023. 

- [26] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _arXiv preprint arXiv:2304.08485_ , 2023. 

- [27] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2017. 

- [28] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Reasoning on graphs: Faithful and interpretable large language model reasoning. _arXiv preprint arXiv:2310.01061_ , 2023. 

- [29] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Reasoning on graphs: Faithful and interpretable large language model reasoning. _arXiv preprint arXiv:2310.01061_ , 2023. 

- [30] Shirui Pan, Yizhen Zheng, and Yixin Liu. Integrating graphs with large language models: Methods and prospects. _arXiv preprint arXiv:2310.05499_ , 2023. 

- [31] Bryan Perozzi, Bahare Fatemi, Dustin Zelle, Anton Tsitsulin, Mehran Kazemi, Rami Al-Rfou, and Jonathan Halcrow. Let your graph do the talking: Encoding structured data for llms. _arXiv preprint arXiv:2402.05862_ , 2024. 

- [32] Chen Qian, Huayi Tang, Zhirui Yang, Hong Liang, and Yong Liu. Can large language models empower molecular property prediction? _arXiv preprint arXiv:2307.07443_ , 2023. 

11 

- [33] Yijian Qin, Xin Wang, Ziwei Zhang, and Wenwu Zhu. Disentangled representation learning with large language models for text-attributed graphs. _arXiv preprint arXiv:2310.18152_ , 2023. 

- [34] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bertnetworks. _arXiv preprint arXiv:1908.10084_ , 2019. 

- [35] Swarnadeep Saha, Prateek Yadav, Lisa Bauer, and Mohit Bansal. Explagraphs: An explanation graph generation task for structured commonsense reasoning. _arXiv preprint arXiv:2104.07644_ , 2021. 

- [36] Priyanka Sen, Sandeep Mavadia, and Amir Saffari. Knowledge graph-augmented language models for complex question answering. In Bhavana Dalvi Mishra, Greg Durrett, Peter Jansen, Danilo Neves Ribeiro, and Jason Wei, editors, _Proceedings of the 1st Workshop on Natural Language Reasoning and Structured Explanations (NLRSE)_ , pages 1–8, Toronto, Canada, June 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.nlrse-1.1. URL `https://aclanthology.org/2023.nlrse-1.1` . 

- [37] Yunsheng Shi, Zhengjie Huang, Shikun Feng, Hui Zhong, Wenjin Wang, and Yu Sun. Masked label prediction: Unified message passing model for semi-supervised classification. _arXiv preprint arXiv:2009.03509_ , 2020. 

- [38] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel Ni, Heung-Yeung Shum, and Jian Guo. Think-on-graph: Deep and responsible reasoning of large language model on knowledge graph. In _The Twelfth International Conference on Learning Representations_ , 2024. URL `https://openreview.net/forum?id=nnVO1PvbTv` . 

- [39] Shengyin Sun, Yuxiang Ren, Chen Ma, and Xuecang Zhang. Large language models as topological structure enhancers for text-attributed graphs. _arXiv preprint arXiv:2311.14324_ , 2023. 

- [40] Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, and Chao Huang. Graphgpt: Graph instruction tuning for large language models. _arXiv preprint arXiv:2310.13023_ , 2023. 

- [41] Yijun Tian, Huan Song, Zichen Wang, Haozhu Wang, Ziqing Hu, Fang Wang, Nitesh V Chawla, and Panpan Xu. Graph neural prompting with large language models. _arXiv preprint arXiv:2309.15427_ , 2023. 

- [42] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023. 

- [43] Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. _arXiv preprint arXiv:1710.10903_ , 2017. 

- [44] Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, and Yulia Tsvetkov. Can language models solve graph problems in natural language? _arXiv preprint arXiv:2305.10037_ , 2023. 

- [45] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- [46] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. _arXiv preprint arXiv:2309.05519_ , 2023. 

- [47] Ruosong Ye, Caiqi Zhang, Runhui Wang, Shuyuan Xu, and Yongfeng Zhang. Natural language is all a graph needs. _arXiv preprint arXiv:2308.07134_ , 2023. 

- [48] Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and Jina Suh. The value of semantic parse labeling for knowledge base question answering. In _Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_ , pages 201–206, 2016. 

12 

- [49] Minji Yoon, Jing Yu Koh, Bryan Hooi, and Ruslan Salakhutdinov. Multimodal graph learning for generative tasks. _arXiv preprint arXiv:2310.07478_ , 2023. 

- [50] Jianxiang Yu, Yuxiang Ren, Chenghua Gong, Jiaqi Tan, Xiang Li, and Xuecang Zhang. Empower text-attributed graphs learning with large language models (llms). _arXiv preprint arXiv:2310.09872_ , 2023. 

- [51] Junchi Yu, Ran He, and Rex Ying. Thought propagation: An analogical approach to complex reasoning with large language models. _arXiv preprint arXiv:2310.03965_ , 2023. 

- [52] Jiawei Zhang. Graph-toolformer: To empower llms with graph reasoning ability via prompt augmented by chatgpt. _arXiv preprint arXiv:2304.11116_ , 2023. 

- [53] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. _arXiv preprint arXiv:2303.16199_ , 2023. 

- [54] Ziwei Zhang, Haoyang Li, Zeyang Zhang, Yijian Qin, Xin Wang, and Wenwu Zhu. Graph meets llms: Towards large graph models, 2023. 

- [55] Haiteng Zhao, Shengchao Liu, Chang Ma, Hannan Xu, Jie Fu, Zhi-Hong Deng, Lingpeng Kong, and Qi Liu. Gimlet: A unified graph-text model for instruction-based molecule zero-shot learning. _bioRxiv_ , pages 2023–05, 2023. 

- [56] Jianan Zhao, Le Zhuo, Yikang Shen, Meng Qu, Kai Liu, Michael Bronstein, Zhaocheng Zhu, and Jian Tang. Graphtext: Graph reasoning in text space. _arXiv preprint arXiv:2310.01089_ , 2023. 

- [57] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , 2023. 

13 

## **A Impact Statements** 

As LLMs are applied to increasingly diverse tasks, their ability to process complex structured data will be increasingly vital. Our work aims to enhance LLMs’ ability to interact with graph-structured data, while resisting hallucination, thus improving model reliability. We also enhance explainability, both by returning the retrieved subgraph, and through the use of conversational interfaces for ‘chatting with a graph’, which allows for better human-AI interaction and for models to behave in a way that is more well-aligned with human expectations. 

## **B Experiment** 

## **B.1 Implementation Settings** 

Experiments are conducted using 2 NVIDIA A100-80G GPUs. Each experiment is replicated four times, utilizing different seeds for each run to ensure robustness and reproducibility. 

**Graph Encoder.** We use Graph Transformer [37] as the GNN backbone. Our configuration employs 4 layers, each with 4 attention heads, and a hidden dimension size of 1024. 

**LLM.** We use the open-sourced Llama2-7b [42] as the LLM backbone. In fine-tuning the LLM with LoRA [10], the lora_r parameter (dimension for LoRA update matrices) is set to 8, and lora_alpha (scaling factor) is set to 16. The dropout rate is set to 0.05. In prompt tuning, the LLM is configured with 10 virtual tokens. The number of max text length is 512, the number of max new tokens, _i.e.,_ the maximum numbers of tokens to generate, is 32. 

**PCST** . For retrieval over graphs via PCST, for the `SceneGraphs` dataset, we select the top _k_ nodes and edges, setting _k_ to 3. Here, the cost of edges, denoted as _Ce_ , is set to 1. Regarding the `WebQSP` dataset, we set _k_ = 3 for nodes and _k_ = 5 for edges, with the edge cost, _Ce_ , adjusted to 0.5. For the `ExplaGraphs` dataset, which is characterized by a small graph size averaging 5.17 nodes and 4.25 edges (as detailed in Table 2), the entire graph can fit in the LLM’s context window size. Consequently, we aim to retrieve the whole graph by setting _k_ to 0, effectively returning the original graph unaltered. 

**Optimization.** We use the AdamW [27] optimizer. We set the initial learning rate at 1e-5, with a weight decay of 0.05. The learning rate decays with a half-cycle cosine decay after the warm-up period. The batch size is 4, and the number of epochs is 10. To prevent overfitting and ensure training efficiency, an early stopping mechanism is implemented with a patience setting of 2 epochs. 

## **B.2 Details of Model Configurations** 

In our experiments, we consider three model configurations: 

_1) Inference-only_ : Using a frozen LLM for direct question answering with textual graph and question, see Figure 4. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0014-12.png)


**----- Start of picture text -----**<br>
frozen<br>node_id, node_attr<br>LLM 0,p!nk1,gender<br>(Self Attention Layers) 2,1club.fm: power...src, edge_attr, dst<br>0,is_reviewed,1<br>2,artist,0<br>3,contributions,4<br>LLM ...<br>(Text Embedder) Question: What is the name<br>of justin bieber brother?<br>Answer:<br>**----- End of picture text -----**<br>


Figure 4: Model configuration _1) Inference-only_ . 

- Zero-shot. In this approach, the model is given a textual graph description and a task description, and is immediately asked to produce the desired output. No additional examples or demonstrations are provided. 

14 

- Zero-CoT. Zero-shot Chain-of-thought (Zero-CoT) prompting [18] is a follow-up to CoT prompting [45], which introduces an incredibly simple zero shot prompt by appending the words "Let’s think step by step." to the end of a question. 

- CoT-BAG. Build-a-Graph Prompting (BAG) [44] is a prompting technique that adds "Let’s construct a graph with the nodes and edges first." after the textual description of the graph is explicitly given. 

- KAPING. KAPING [1] is a zero-shot knowledge-augmented prompting method for knowledge graph question answering. It first retrieves triples related to the question from the graph, then prepends them to the input question in the form of a prompt, which is then forwarded to LLMs to generate the answer. 

_2) Frozen LLM w/ prompt tuning (PT)_ : Keeping the parameters of the LLM frozen and adapting only the prompt. This includes soft prompt tuning (see Figure 5a), GraphToken [31], which is a graph prompt tuning method, and our _G-Retriever_ method (see Figure5b). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0015-04.png)


**----- Start of picture text -----**<br>
frozen frozen<br>trainable jaxon bieber trainable jaxon bieber<br>node_id, node_attr node_id, node_attr<br>LLM 0,p!nk1,gender LLM 15, justin bieber294, jaxon bieber<br>(Self Attention Layers) 2,1club.fm: power...src, edge_attr, dst (Self Attention Layers) 356, jeremy bieber551, m.0gxnnwpsrc, edge_attr, dst<br>prompt 0,is_reviewed,12,artist,0 294, parents, 356356, children, 15<br>LLM 3,contributions,4... Projection LLM 551, sibling, 294551, sibling, 15<br>(Text Embedder) Question: What is the name Graph Encoder (Text Embedder) Question: What is the name<br>of justin bieber brother? A nswer: of justin bieber brother? A nswer:<br>(a) Prompt Tuning (b)  G-Retriever<br>**----- End of picture text -----**<br>


Figure 5: Model configuration _2) Frozen LLM w/ prompt tuning_ . 

_3) Tuned LLM_ : Fine-tuning the LLM with LoRA. This includes standard fine-tuning of an LLM for downstream tasks using LoRA (see Figure 6a) and G-Retriever with LoRA (see Figure 6b). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0015-07.png)


**----- Start of picture text -----**<br>
trainable jaxon bieber trainable jaxon bieber<br>(Self Attention Layers)LLM LLM node_id, node_attr0,p!nk1,gender2,1club.fm: power...src, edge_attr, dst0,is_reviewed,12,artist,03,contributions,4... Projection (Self Attention Layers)LLM LLM node_id, node_attr15, justin bieber294, jaxon bieber356, jeremy bieber551, m.0gxnnwpsrc, edge_attr, dst294, parents, 356356, children, 15551, sibling, 294551, sibling, 15<br>(Text Embedder) Question: What is the nameof justin bieber brother?Answer: Graph Encoder (Text Embedder) Question: What is the nameof justin bieber brother?Answer:<br>(a) LoRA (b)  G-Retriever  w/ LoRA<br>**----- End of picture text -----**<br>


Figure 6: Model configuration _3) Tuned LLM_ . 

## **B.3 Details of Ablation Study** 

This section illustrates the modifications made to the original architecture in the ablation study, as presented in Figure 7. 

**Without Graph Encoder (w/o GraphEncoder):** In this setting, we replaced the graph encoder with trainable soft tokens, setting the number of these virtual tokens to 10. 

**Without Projection Layer (w/o Projection Layer):** Here, we removed the projection layer following the graph encoder. We configured the output dimension of the graph encoder to be 4,096, matching the hidden dimension of Llama2-7b. This allows the output graph token (the yellow token in Figure 7b) to be concatenated directly with the LLM tokens (blue tokens). 

**Without Textualized Graph (w/o Textualized Graph):** In this configuration, we modified the textual input to the LLM. Rather than using a combination of the question and the textualized graph, we solely used the question. 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering_images/nips_2024_G-Retriever_Retrieval-Augmented_Generation_for_Textual_Graph_Understanding_and_Question_Answering.pdf-0016-00.png)


**----- Start of picture text -----**<br>
frozentrainable jaxon bieber frozentrainable jaxon bieber frozentrainable jaxon bieber<br>prompt (Self Attention Layers)LLM(Text Embedder)LLM node_id, node_attr15, justin bieber294, jaxon bieber356, jeremy bieber551, m.0gxnnwpsrc, edge_attr, dst294, parents, 356356, children, 15551, sibling, 294551, sibling, 15Question: What is the nameof justin bieber brother? A nswer: Graph Encoder (Self Attention Layers)LLM(Text Embedder)LLM node_id, node_attr15, justin bieber294, jaxon bieber356, jeremy bieber551, m.0gxnnwpsrc, edge_attr, dst294, parents, 356356, children, 15551, sibling, 294551, sibling, 15Question: What is the nameof justin bieber brother? A nswer: Graph EncoderProjection (Self Attention Layers)LLM(Text Embedder)LLM Question: What is the nameof justin bieber brother?Answer:<br>(a) w/o GraphEncoder (b) w/o Projection Layer (c) w/o Textualized Graph<br>**----- End of picture text -----**<br>


Figure 7: Ablation study configurations. 

## **B.4 The Choice of Graph Encoder** 

In addition to the Graph Transformer [37], we explore other GNNs as the graph encoder, such as GCN [17] and the GAT [43]. The comparative results of these models on the `WebQSP` and `ExplaGraphs` datasets are presented in Table 7. 

Table 7: Performance of different graph encoders on the `WebQSP` and `ExplaGraphs` datasets. 

|Graph Encoder|`WebQSP`|`ExplaGraphs`|
|---|---|---|
|GCN [17]|70.70|0.8394|
|GAT [43]|70.27|0.8430|
|Graph Transformer [37]|70.49|0.8516|



The results demonstrate that our proposed method exhibits consistent robustness across different graph encoders. Notably, all three encoders – GCN, GAT, and GraphTransformer – demonstrate competitive and closely aligned performance on the `WebQSP` dataset, with Hit@1 scores of 70.70, 70.27, and 70.49, respectively. However, the performance differentiation becomes more pronounced on the `ExplaGraphs` dataset, where GraphTransformer exhibits a superior Hit@1 score of 0.8516, followed by GAT and GCN with scores of 0.8430 and 0.8394, respectively. This variation in performance across the datasets highlights the importance of encoder selection based on the specific characteristics and requirements of the dataset. 

## **B.5 The Choice of LLM** 

As for the choice of LLM, we considered both Llama2-7b and Llama2-13b. Our experiments demonstrate that stronger LLMs enhance the effectiveness of our method, as shown in Table 8, indicating that it benefits from the increased scale of the LLMs. 

Table 8: Performance of different LLMs on the `WebQSP` dataset. 

|LLM|Llama2-7b|Llama2-13b|
|---|---|---|
|Hit@1|70.49|75.58|



## **C GraphQA Benchmark** 

In this section, we detail how our GraphQA benchmark differs from the original datasets, including the specific processing steps we employed. For concrete examples that illustrate the differences between the raw text in the original dataset and in our GraphQA benchmark, please refer to Table 9. 

`ExplaGraphs` . The original dataset[2] [35] represents relationships using triplets. We have standardized this format by converting the triplets into a graph representation. Specifically, each head and tail in a triplet is transformed into a node, and the relation is transformed into an edge. Since the test dataset labels are not available, we have utilized only the training and validation (val) datasets from the original collection. We further divided these into training, val, and test subsets, using a 6:2:2 ratio. 

> 2https://explagraphs.github.io/ 

16 

Table 9: Comparison of text formats in original datasets and our GraphQA benchmark. 

|Dataset|Original dataset|GraphQA Benmark|
|---|---|---|
|||node_id,node_attr\n0,entrapment\n1,being abused\n2,police\n3,harm\n4,people\n|
|`ExplaGraphs`|(entrapment; capable of; being abused) (being abused; created by; police) (police;<br>capable of; harm) (harm; used for; people) (people; part of; citizens)|5,citizens\n<br>src,edge_attr,dst\n0,capable of,1\n1,created by,2\n2,capable of,3\n3,used for,4\n|
|||4,part of,5\n|
||"width": 500, "objects": "681267": "name": "banana", "h": 34, "relations": ["object":||
||"681262", "name": "to the left of"], "w": 64, "attributes": ["small", "yellow"],||
||"y": 55, "x": 248, "681265": "name": "spots", "h": 16, "relations": [], "w": 26,||
||"attributes": [], "y": 92, "x": 245, "681264": "name": "bananas", "h": 50, "relations":||
||["object": "681259", "name": "to the left of"], "w": 49, "attributes": ["small",||
||"yellow"], "y": 32, "x": 268, "681263": "name": "picnic", "h": 374, "relations": [],||
|`SceneGraphs`|"w": 499, "attributes": ["delicious"], "y": 0, "x": 0, "681262": "name": "straw", "h":<br>95, "relations": ["object": "681268", "name": "to the right of", "object": "681267",<br>"name": "to the right of", "object": "681253", "name": "to the right of"], "w": 15,<br>"attributes": ["white", "plastic"], "y": 55, "x": 402, "681261": "name": "meat", "h":<br>27, "relations": ["object": "681255", "name": "on", "object": "681255", "name":<br>"inside"], "w": 24, "attributes": ["small", "brown", "delicious"], "y": 123, "x": 68,<br>"681260": "name": "rice", "h": 57, "relations": ["object": "681255", "name": "on",<br>"object": "681258", "name": "to the left of"], "w": 93, "attributes": ["piled", "white"],<br>"y": 162, "x": 57, "681269": "name": "onions", "h": 16, "relations": [], "w": 24,<br>"attributes": ["green"], "y": 147, "x": 90, "681268": "name": "tablecloth", "h": 374,<br>"relations": ["object": "681262", "name": "to the left of"], "w": 396, "attributes":<br>["white"], "y": 0, "x": 0, "681258": "name": "bowl", "h": 99, "relations": ["object":<br>"681255", "name": "next to", "object": "681257", "name": "of", "object": "681255",<br>"name": "near", "object": "681256", "name": "to the right of", "object": "681260",<br>"name": "to the right of", "object": "681255", "name": "to the right of"], "w": 115,<br>"attributes": ["full"], "y": 184, "x": 178, "681259": "name": "plantains", "h": 70,<br>"relations": ["object": "681264", "name": "to the right of"], "w": 45, "attributes":<br>["red"], "y": 0, "x": 346, "681256": "name": "spoon", "h": 65, "relations": ["object":<br>"681255", "name": "on", "object": "681257", "name": "to the left of", "object":<br>"681255", "name": "in", "object": "681258", "name": "to the left of"], "w": 140,<br>"attributes": ["large", "metal", "silver"], "y": 196, "x": 0, "681257": "name": "dish",<br>"h": 81, "relations": ["object": "681258", "name": "inside", "object": "681256",<br>"name": "to the right of", "object": "681258", "name": "in", "object": "681255",<br>"name": "to the right of"], "w": 108, "attributes": ["cream colored"], "y": 199, "x":<br>187, "681254": "name": "meal", "h": 111, "relations": [], "w": 130, "attributes":<br>[], "y": 121, "x": 58, "681255": "name": "plate", "h": 138, "relations": ["object":|node_id,node_attr<br>0,"name: banana; attribute: small, yellow; (x,y,w,h): (248, 55, 64, 34)"<br>1,"name: spots; (x,y,w,h): (245, 92, 26, 16)"<br>2,"name: bananas; attribute: small, yellow; (x,y,w,h): (268, 32, 49, 50)"<br>3,"name: picnic; attribute: delicious; (x,y,w,h): (0, 0, 499, 374)"<br>4,"name: straw; attribute: white, plastic; (x,y,w,h): (402, 55, 15, 95)"<br>5,"name: meat; attribute: small, brown, delicious; (x,y,w,h): (68, 123, 24, 27)"<br>6,"name: rice; attribute: piled, white; (x,y,w,h): (57, 162, 93, 57)"<br>7,"name: onions; attribute: green; (x,y,w,h): (90, 147, 24, 16)"<br>8,"name: tablecloth; attribute: white; (x,y,w,h): (0, 0, 396, 374)"<br>9,"name: bowl; attribute: full; (x,y,w,h): (178, 184, 115, 99)"<br>10,"name: plantains; attribute: red; (x,y,w,h): (346, 0, 45, 70)"<br>11,"name: spoon; attribute: large, metal, silver; (x,y,w,h): (0, 196, 140, 65)"<br>12,"name: dish; attribute: cream colored; (x,y,w,h): (187, 199, 108, 81)"<br>13,"name: meal; (x,y,w,h): (58, 121, 130, 111)"<br>14,"name: plate; attribute: white, full; (x,y,w,h): (30, 111, 176, 138)"<br>15,"name: banana; attribute: small, yellow; (x,y,w,h): (237, 87, 73, 30)"<br>src,edge_attr,dst<br>0,to the left of,4\n2,to the left of,10\n4,to the right of,8\n4,to the right of,0\n4,to<br>the right of,15\n5,on,14\n5,inside,14\n6,on,14\n6,to the left of,9\n8,to the left<br>of,4\n9,next to,14\n9,of,12\n9,near,14\n9,to the right of,11\n9,to the right of,6\n<br>9,to the right of,14\n10,to the right of,2\n11,on,14\n11,to the left of,12\n11,in,14\n<br>11,to the left of,9\n 12,inside,9\n 12,to the right of,11\n 12,in,9\n 12,to the right<br>of,14\n14,to the left of,12\n14,of,13\n14,with,13\n14,near,9\n14,to the left of,9\n<br>15,to the left of,4\n|
||"681257", "name": "to the left of", "object": "681254", "name": "of", "object":||
||"681254", "name": "with", "object": "681258", "name": "near", "object": "681258",||
||"name": "to the left of"], "w": 176, "attributes": ["white", "full"], "y": 111, "x": 30,||
||"681253": "name": "banana", "h": 30, "relations": ["object": "681262", "name": "to||
||the left of"], "w": 73, "attributes": ["small", "yellow"], "y": 87, "x": 237, "height":||
||375||
|||node_id,node_attr\n0,fedex cup\n1,m.0n1v8cy\n2,brandt snedeker\n3,m.08q5wy\n|
|`WebQSP`|[[’FedEx Cup’, ’sports.sports_award_type.winners’, ’m.0n1v8cy’],<br>[’Brandt Snedeker’, ’sports.sports_award_winner.awards’, ’m.0n1v8cy’],<br>[’FedEx Cup’, ’common.topic.article’, ’m.08q5wy’],<br>[’FedEx Cup’, ’common.topic.notable_for’, ’g.12559n8g_’],<br>[’Sports League Award Type’, ’freebase.type_profle.published’, ’Published’],<br>[’FedEx Cup’, ’common.topic.notable_types’, ’Sports League Award Type’],<br>[’m.0n1v8cy’, ’sports.sports_award.award_winner’, ’Brandt Snedeker’],<br>[’Sports League Award Type’, ’type.type.expected_by’, ’Award’],<br>[’Sports League Award Type’, ’common.topic.article’, ’m.06zxtxj’],<br>[’2012 PGA Tour’, ’sports.sports_league_season.awards’, ’m.0n1v8cy’],<br>[’Sports League Award Type’, ’freebase.type_hints.included_types’, ’Topic’],<br>[’Sports League Award Type’, ’type.type.domain’, ’Sports’],<br>[’m.0n1v8cy’, ’sports.sports_award.award’, ’FedEx Cup’],<br>[’Sports League Award Type’,<br>’freebase.type_profle.strict_included_types’,<br>’Topic’],<br>[’Sports League Award Type’, ’freebase.type_profle.kind’, ’Classifcation’],<br>[’m.0n1v8cy’, ’sports.sports_award.season’, ’2012 PGA Tour’],<br>[’Sports League Award Type’, ’type.type.properties’, ’Winners’]]|4,g.12559n8g_\n5,sports league award type\n6,published\n7,award\n8,m.06zxtxj\n<br>9,2012 pga tour\n10,topic\n11,sports\n12,classifcation\n13,winners\n<br>src,edge_attr,dst<br>0,sports.sports_award_type.winners,1<br>2,sports.sports_award_winner.awards,1<br>0,common.topic.article,3<br>0,common.topic.notable_for,4<br>5,freebase.type_profle.published,6<br>0,common.topic.notable_types,5<br>1,sports.sports_award.award_winner,2<br>5,type.type.expected_by,7<br>5,common.topic.article,8<br>9,sports.sports_league_season.awards,1<br>5,freebase.type_hints.included_types,10<br>5,type.type.domain,11<br>1,sports.sports_award.award,0<br>5,freebase.type_profle.strict_included_types,10<br>5,freebase.type_profle.kind,12<br>1,sports.sports_award.season,9|
|||5,type.type.properties,13|



`SceneGraphs` . The original GQA dataset is designed for real-world visual reasoning and compositional question answering, aiming to address key shortcomings of previous VQA datasets [13]. It comprises 108k images, each associated with a Scene Graph. In our study, we focus differently on graph question answering; hence, we did not utilize the image counterparts, leveraging only the scene graphs from the original dataset. Additionally, the original dataset describes images using JSON files. We simplified the object IDs to suit our research needs. We randomly sampled 100k samples from the original dataset and divided them into training, validation, and test subsets, following a 6:2:2 ratio. 

`WebQSP` . We follow the preprocessing steps from RoG[3] [28]. The original dataset uses a list of triplets format, which we have transformed into our unified graph format. Furthermore, to avoid 

> 3https://huggingface.co/datasets/rmanluo/RoG-webqsp 

17 

discrimination between capital and lowercase words, we have converted all words to lowercase. We used the same dataset split as in the original dataset. 

## **D Graph Retrieval-Augmented Generation (GraphRAG)** 

## **D.1 Comparison with Existing GraphRAG Methods** 

Most existing GraphRAG methods are designed specifically for knowledge graphs and focus on node, edge, or triples-level retrieval [1, 36, 16, 38]. Our method is different in two main ways: 1) It focuses on more general textual graphs, not just knowledge graphs. 2) It enables the return of a subgraph most closely related to a query, rather than a list of top-k triples. The triples in other methods are chosen in isolation from the graph, failing to capture neighborhood information effectively. In contrast, our method takes the context into account during retrieval. 

## **D.2 The Impact of K for Retrieval** 

We identify the most relevant nodes and edges and use a k-nearest neighbors retrieval approach (see Equation 6). Small k values may omit crucial knowledge or information relevant to the query, while large k values could introduce excessive information, distracting the model from the essential details. To evaluate the impact of the number of k, we have conducted additional experiments by varying the choice of k to 3, 5, 10, and 20. 

Table 10: The impact of k on the `webqsp` dataset. 

|k|3|5|10|20|
|---|---|---|---|---|
|Hit@1|0.6977|0.7063|0.7248|0.7039|



As shown in Table 10, the Hit@1 metric initially rises for small k values, peaks at a certain point, and then declines for large k values. Determining the optimal k value can be achieved through techniques like cross-validation using a validation set. 

## **D.3 The Choice of Similarity Function** 

The choice of similarity function is also important. In this work, we use cosine similarity, a widely adopted metric for measuring vector similarity in models that process vision and language. For instance, CLIP also employs cosine similarity to assess the similarity between text and image features. Although it might not be the optimal choice, we believe that cosine similarity is a general, representative, and valid choice for facilitating fast retrieval tasks. 

## **D.4 The Quality of Retrieval** 

We quantify the quality of our retrieval method as follows: We examine the retrieval subgraph, and if the label is contained within it, we consider it a successful retrieval. We calculate the retrieval success rate of our method and the retrieval method proposed in KAPING [1] on the WebQSP dataset. 

Table 11: The quality of retrieval methods on the `WebQSP` dataset. 

|Method|Retrieval Accuracy|
|---|---|
|KAPING [1] (top-k triple retrieval)|60.81|
|_G-Retriever_|70.49|



As shown in Table 11, these results validate the effectiveness of our retrieval method. In contrast to the triple-based retrieval in KAPING, which relies on the similarity between triples and the query, our PCST-based subgraph retrieval is more accurate as it takes the graph structure into account. By design, it retrieves connected subgraphs, capturing not just nodes or edges with high relevance scores, but also those that act as "bridges" connecting other highly relevant nodes and edges. 

18 

## **E Discussion on the Complexity** 

## **E.1 The integration of GNNs, LLMs and GraphRAG** 

_G-Retriever_ is framework integrate the strengths of GNNs, LLMs and GraphRAG. The LLM+X framework, which involves enriching LLMs with multi-modal capabilities by integrating an LLM with an encoder from another modality, is a widely adopted approach. Notable examples include Llava, MiniGPT-4, and Flamingo, among others. They are not complex in terms of understanding or implementation. Regarding the integration of GraphRAG, it does not require training and can be implemented during the preprocessing stage or on the fly. This approach does not significantly increase time complexity or computational complexity. On the contrary, it can substantially reduce the size of the graph ( _e.g.,_ eliminating 99% of nodes in the `WebQSP` dataset), which in turn speeds up the overall running time ( _e.g.,_ reducing it from 18.7 min/epoch to 6.2 min/epoch on the `WebQSP` dataset). 

## **E.2 Computational Resources** 

Utilizing two A100 GPUs, each with 80GB of memory, we conducted tests on Llama2-7b and `WebQSP` datasets. Our experiments had a training batch size of 16 and an evaluation batch size of 32, yielding the following results. 

Table 12: Performance and Efficiency of Various Methods on the `WebQSP` dataset. 

|Settting|Method|Hit@1|Time|
|---|---|---|---|
|Inference-only|Question only<br>Textual graph and question|61.16<br>41.06|31 min<br>40 min|
|Frozen LLM w/ PT|Prompt Tuning<br>G-Retriever|48.34<br>70.49|18.7 min/epoch<br>6.2 min/epoch|
|Tuned LLM|LoRA<br>_G-Retriever_w/ LoRA|66.03<br>73.79|19 min/epoch<br>6.9 min/epoch|



These results highlight efficiency improvements via graph RAG, which significantly reduces graph size (e.g., eliminating 99% of nodes in the `WebQSP` dataset) and speeds up running time. 

## **F Hallucination in Graph LLMs** 

In this section, we present quantitative results regarding hallucinations in the `SceneGraphs` dataset. 

**Baseline.** For our baseline, we adapted MiniGPT-4 [57] to graph contexts. This approach involves a frozen LLM interacting with a trainable GNN that encodes graph data as a soft prompt, denoted as LLM+Graph Prompt Tuning. We focus on graph prompt tuning as the baseline, instead of converting the graph into text, since the textual representation of the graph is large and consistently exceeds the input token limits of LLMs. 

**Experiment Design.** We instructed the LLM to answer graph-related questions and to list nodes or edges in the explanation graph that support its answers. Since standard answers for these questions do not exist, allowing the LLM to respond flexibly, it becomes challenging to evaluate its responses. To address this, we manually examined 100 responses generated by our method and the LLM with graph prompt tuning, verifying whether the nodes and edges referenced in the LLM’s output actually exist in the graph. 

**Evaluation Metrics.** We assessed the model’s faithfulness using three metrics: the fraction of valid nodes (denoted as Valid Nodes), the fraction of valid edges (denoted as Valid Edges), and the fraction of times the entire set of nodes and edges cited was valid (denoted as Fully Valid Graphs). 

**Results.** The results, as depicted in Table 5, illustrate the comparative effectiveness of the _G-Retriever_ over the baseline LLM+Graph Prompt Tuning method in reducing hallucinations. The LLM+Graph Prompt Tuning approach demonstrated a significantly lower accuracy in referencing graph elements, 

19 

with only 31% of nodes and 12% of edges being valid, and the entire set of nodes and edges being valid only 8% of the time. In contrast, _G-Retriever_ showed substantial improvements: 77% validity in nodes, 76% in edges, and 62% in the overall validity of referenced node-edge sets. These results underscore the significant reduction in hallucinations with _G-Retriever_ , particularly in the challenging task of accurately citing both nodes and edges in graph-based contexts. 

## **G Demonstrations** 

We demonstrate the interaction capabilities of G-Retriever with creative questions on different datasets: `ExplaGraphs` , `SceneGraphs` , and `WebQSP` . These examples are showcased in Tables 13, 14, and 15, respectively. Additionally, the examples are visualized in Figure 1. 

20 

Table 13: Example of creative chat on `ExplaGraphs` . 

|Graph|node_id,node_attr<br>0,human<br>1,cosmetic surgery<br>2,women<br>3,change appearance<br>4,addiction<br>5,searching for perfection<br>6,stop<br>7,positive<br>src,edge_attr,dst<br>0,receives action,1<br>2,part of,0<br>1,used for,3<br>3,capable of,4<br>4,created by,5<br>4,not capable of,6<br>4,is not a,7|
|---|---|
|||
|User|Write a 500-word argument essay based on the information provided in the explanation graph.|
|||
|Model|The explanation graph provides a detailed analysis of the reasons why women undergo cosmetic<br>surgery. According to the graph, the main reasons include the desire to change one’s appearance,<br>the infuence of societal beauty standards, and the addiction to searching for perfection. These<br>reasons are further supported by the various nodes and edges in the graph, which provide additional<br>insights into the motivations and consequences of cosmetic surgery.<br>Firstly, the node labeled "human" highlights the fundamental human desire to feel attractive and<br>confdent in one’s appearance. This desire is deeply ingrained in our culture, where beauty is<br>often associated with happiness, success, and social acceptance. As a result, many women feel<br>pressure to conform to societal beauty standards, which can lead to feelings of insecurity and low<br>self-esteem. In response to these feelings, some women may turn to cosmetic surgery as a means<br>of improving their appearance and boosting their self-confdence.<br>Secondly, the node labeled "women" highlights the gendered nature of cosmetic surgery. While<br>men also undergo cosmetic surgery, the majority of cosmetic surgery patients are women. This is<br>likely due to the fact that women are more likely to be exposed to societal beauty standards and are<br>more likely to internalize these standards. As a result, women may be more likely to feel insecure<br>about their appearance and more likely to seek cosmetic surgery as a means of improving their<br>looks.<br>Thirdly, the node labeled "addiction" highlights the potential negative consequences of cosmetic<br>surgery. While some women may initially feel improved confdence and self-esteem after un-<br>dergoing cosmetic surgery, the graph suggests that this initial boost may not last. In fact, some<br>women may become addicted to the constant pursuit of perfection, leading to a cycle of surgery<br>and dissatisfaction. This can have serious psychological and emotional consequences, including<br>depression, anxiety, and a loss of self-esteem.<br>Fourthly, the node labeled "searching for perfection" highlights the unrealistic expectations that<br>many women have about the results of cosmetic surgery. The graph suggests that women may have<br>unrealistic expectations about the ability of cosmetic surgery to improve their appearance and boost<br>their self-esteem. This can lead to disappointment and dissatisfaction with the results of surgery, as<br>well as a continued pursuit of perfection through additional surgery.<br>Finally, the node labeled "stop" highlights the importance of critical thinking and self-refection<br>when it comes to cosmetic surgery. The graph suggests that women should be aware of the potential<br>negative consequences of cosmetic surgery and should carefully consider the reasons why they are<br>undergoing surgery. They should also be aware of the unrealistic expectations that they may have<br>about the results of surgery and should be prepared to address any negative emotions or feelings of<br>dissatisfaction that may arise.<br>In conclusion, the explanation graph provides a detailed analysis of the reasons why women<br>undergo cosmetic surgery. The graph highlights the fundamental human desire to feel attractive<br>and confdent in one’s appearance, the gendered nature of cosmetic surgery, the potential negative<br>consequences of cosmetic surgery, and the importance of critical thinking and self-refection when<br>it comes to surgery. By understanding these reasons and the potential consequences of cosmetic<br>surgery, women can make informed decisions about whether or not to undergo surgery and can<br>work to address any negative emotions or feelings of dissatisfaction that may arise.|



21 

Table 14: Example of creative chat on `SceneGraphs` . We utilize GPT-4 to generate an image from the graph, assisting readers in visualizing it 

|Graph|node_id,node_attr<br>0,name: head; attribute: black; (x,y,w,h): (386, 185, 8, 4)<br>1,name: swimming pool; attribute: small, water; (x,y,w,h): (465, 92, 35, 3)<br>2,name: shore; (x,y,w,h): (49, 85, 436, 108)<br>3,name: fence; attribute: wired; (x,y,w,h): (11, 158, 167, 46)<br>4,name: mountains; attribute: covered; (x,y,w,h): (393, 53, 107, 12)<br>5,name: water; attribute: blue; (x,y,w,h): (0, 50, 499, 154)<br>6,name: sheep; attribute: walking, grazing; (x,y,w,h): (243, 191, 20, 13)<br>7,name: sky; attribute: blue, clear, cloudy; (x,y,w,h): (0, 0, 501, 60)<br>8,name: sheep; attribute: white, fuffy; (x,y,w,h): (409, 176, 13, 10)<br>9,name: feld; attribute: large, green, grassy; (x,y,w,h): (0, 138, 503, 112)<br>10,name: grass; (x,y,w,h): (0, 84, 500, 166)<br>11,name: sheep; attribute: white, fuffy; (x,y,w,h): (398, 178, 13, 11)<br>12,name: sheep; attribute: white, fuffy; (x,y,w,h): (385, 179, 17, 11)<br>13,name: sheep; attribute: white, fuffy; (x,y,w,h): (364, 181, 14, 10)<br>14,name: shore; attribute: sandy; (x,y,w,h): (190, 123, 171, 12)<br>src, edge_attr, dst<br>0,to the left of,11\n0,to the right of,13\n3,to the left of,6\n6,to the right of,3\n6,grazing on,10\n<br>6,in,9\n6,in,10\n7,above,5\n8,eating,10\n8,to the right of,11\n8,to the right of,12\n11,to the<br>left of,8\n 11,to the right of,0\n 11,to the right of,13\n 11,to the right of,12\n 11,eating,10\n<br>12,eating,10\n12,to the left of,8\n12,to the right of,13\n12,to the left of,11\n13,to the left of,0\n<br>13,to the left of,11\n13,to the left of,12\n13,eating,10\n|
|---|---|
|||
|Image||
|||
|User|Write an 500-word advertisement for this place in the scene graph that would make people want to<br>visit it.|
|||
|Model|Sure! Here’s an example of a 500-word advertisement for the place based on the scene graph:<br>"Experience the breathtaking beauty of our place, where the rolling hills meet the sparkling water<br>of the swimming pool. Take a leisurely stroll through the lush green felds, and let the gentle breeze<br>caress your face as you gaze out at the picturesque view. The sheep graze peacefully in the distance,<br>their white coats glistening in the sunlight. The sky is a brilliant blue, with fuffy white clouds<br>drifting lazily across. Come and bask in the serene ambiance of our place, where the natural beauty<br>will leave you in awe."|



22 

Table 15: Example of creative conversation on the `WebQSP` dataset. 

|Graph|node_id, node_attr<br>8, benjamin franklin<br>63, glass harmonica<br>98, lightning rod<br>99, stealing god’s thunder: benjamin franklin’s lightning rod and the invention of america<br>227, benjamin franklin worked with thomas jefferson<br>442, franklin stove<br>525, bifocals<br>706, benjamin franklin drawing electricity from the sky<br>1044, prokop diviš<br>src, edge_attr, dst<br>706, visual_art.artwork.art_subject, 8<br>8, base.kwebbase.kwtopic.connections_from, 227<br>8, book.book_subject.works, 99<br>525, law.invention.inventor, 8<br>442, law.invention.inventor, 8<br>98, law.invention.inventor, 1044<br>8, law.inventor.inventions, 98<br>63, law.invention.inventor, 8<br>8, base.argumentmaps.innovator.original_ideas, 442<br>98, law.invention.inventor, 8<br>8, law.inventor.inventions, 525<br>442, base.argumentmaps.original_idea.innovator, 8<br>8, law.inventor.inventions, 442<br>8, law.inventor.inventions, 63|
|---|---|
|||
|User|Question: what else did ben franklin invent? Surrounding this question, develop an educational tour through the knowledge<br>graph, highlighting interesting facts and connections in an engaging way.|
|||
|Model|Great! Based on the provided knowledge graph, here’s an educational tour that highlights interesting facts and connections<br>related to Benjamin Franklin’s inventions and other notable achievements:<br>1. Franklin’s Lightning Rod: Start by exploring Franklin’s most famous invention - the lightning rod. Discover how Franklin’s<br>invention protected buildings from lightning strikes and how it changed the way people understood electricity. Connection:<br>Franklin worked with Thomas Jefferson on the lightning rod, and their collaboration led to the development of the concept of<br>positive and negative charges.<br>2. Bifocals: Move on to Franklin’s invention of bifocal glasses. Learn how Franklin’s vision problems inspired him to create a<br>new type of glasses that allowed people to see both near and far objects clearly. Connection: Franklin’s bifocal glasses were a<br>groundbreaking invention that improved the lives of millions of people around the world.<br>3. Franklin Stove: Explore Franklin’s design of the Franklin stove, a cast-iron stove that provided more heat with less fuel.<br>Discover how Franklin’s stove revolutionized heating systems and how it helped to reduce fuel costs for households and<br>businesses. Connection: Franklin’s stove was an early example of energy effciency and sustainability, and it paved the way for<br>modern heating systems.<br>4. Prokop Diviš: Learn about Franklin’s connection to Prokop Diviš, a Czech inventor who built on Franklin’s work and<br>developed the frst electric generator. Discover how Diviš’s invention helped to power the Industrial Revolution and how it<br>changed the way people lived and worked. Connection: Franklin’s work on the lightning rod inspired Diviš to explore the<br>properties of electricity, and their collaborative efforts helped to advance the feld of electrical engineering.<br>5. Innovator and Original Ideas: Explore Franklin’s reputation as an innovator and his ability to come up with original ideas.<br>Discover how Franklin’s creativity and curiosity led to many of his inventions and how he was able to see connections between<br>seemingly unrelated things. Connection: Franklin’s innovative spirit and ability to think outside the box inspired many other<br>inventors and scientists, including Thomas Edison and Nikola Tesla.<br>6. Collaboration and Networking: Highlight Franklin’s collaborative approach to invention and his ability to network with other<br>inventors and scientists. Discover how Franklin’s connections with other innovators helped to advance the feld of science and<br>technology. Connection: Franklin’s collaborations with other inventors and scientists, such as Thomas Jefferson and Prokop<br>Diviš, helped to accelerate the pace of innovation and pave the way for future breakthroughs.<br>7. Legacy and Impact: Refect on Franklin’s lasting legacy and the impact of his inventions on modern society. Discover how<br>Franklin’s inventions continue to inspire new generations of inventors and scientists and how they have helped to shape the<br>world we live in today. Connection: Franklin’s inventions and ideas have had a lasting impact on many felds, including science,<br>technology, and society as a whole. His legacy continues to inspire and motivate people around the world.<br>By following this educational tour through the knowledge graph, students will gain a deeper understanding of Benjamin Franklin’s<br>inventions and their impact on modern society. They will also learn about Franklin’s collaborative approach to invention, his<br>innovative spirit, and his ability to think outside the box.|



23 

