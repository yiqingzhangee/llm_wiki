# **ORGAN: Observation-Guided Radiology Report Generation via Tree Reasoning** 

**Wenjun Hou**[1] _[,]_[2] **, Kaishuai Xu**[1] _[∗]_ **, Yi Cheng**[1] _[∗]_ **, Wenjie Li**[1] _[†]_ **, Jiang Liu**[2] _[†]_ 

1Department of Computing, The Hong Kong Polytechnic University, HKSAR, China 

2Research Institute of Trustworthy Autonomous Systems and Department of Computer Science and Engineering, Southern University of Science and Technology, Shenzhen, China houwenjun060@gmail.com 

{kaishuaii.xu, alyssa.cheng}@connect.polyu.hk cswjli@comp.polyu.edu.hk, liuj@sustech.edu.cn 

## **Abstract** 

This paper explores the task of radiology report generation, which aims at generating free-text descriptions for a set of radiographs. One significant challenge of this task is how to correctly maintain the consistency between the images and the lengthy report. Previous research explored solving this issue through planningbased methods, which generate reports only based on high-level plans. However, these plans usually only contain the major observations from the radiographs (e.g., lung opacity), lacking much necessary information, such as the observation characteristics and preliminary clinical diagnoses. To address this problem, the system should also take the image information into account together with the textual plan and perform stronger reasoning during the generation process. In this paper, we propose an Observation-guided radiology Report GenerAtioN framework (ORGAN). It first produces an observation plan and then feeds both the plan and radiographs for report generation, where an observation graph and a tree reasoning mechanism are adopted to precisely enrich the plan information by capturing the multiformats of each observation. Experimental results demonstrate that our framework outperforms previous state-of-the-art methods regarding text quality and clinical efficacy.[1] 

## **1 Introduction** 

Radiology reports, which contain the textual description for a set of radiographs, are critical in the process of medical diagnosis and treatment. Nevertheless, the interpretation of radiographs is very time-consuming, even for experienced radiologists 

> _∗_ Equal Contribution. 

> _†_ Corresponding authors. 

> 1Our code is available at https://github.com/ wjhou/ORGan. 

Stage 1: Observation Planning ① Lung Opacity / POS ② Pneumonia / POS ③ Pleural Effusion / NEG ④ Cardiomegaly / NEG ⑤ Support Devices / POS As compared ……, there is ① [a subtle but new opacity] at the right lung base, …… Given the clinical presentation, ② [pneumonia must be suspected]. ……, extensive overinflation with bronchiectasis but ③ [no pleural effusion] or other parenchymal changes. ④ [Normal size of cardiac silhouette]. Unchanged position of the ⑤ [nasogastric tube]. 

Figure 1: Our proposed framework contains two stages, including the observation planning stage and the report generation stage. Red color denotes positive observations, while Blue color denotes negative observations. 

(5-10 minutes per image). Due to its large potential to alleviate the strain on the healthcare workforce, automated radiology report generation (Anderson et al., 2018; Rennie et al., 2017) has attracted increasing research attention. 

One significant challenge of this task is how to correctly maintain the consistency between the image and the lengthy textual report. Many previous works proposed to solve this through planningbased generation by first concluding the major observations identified in the radiographs before the word-level realization (Jing et al., 2018; You et al., 2021; Nooralahzadeh et al., 2021; Nishino et al., 2022). Despite their progress, these methods still struggle to maintain the cross-modal consistency between radiographs and reports. A significant problem within these methods is that, in the stage of word-level generation, the semantic information of observations and radiographs is not fully utilized. They either generate the report only based on the high-level textual plan (i.e., major observations) 

or ignore the status of an observation (i.e., positive, negative, and uncertain), which is far from adequate. The observations contained in the highlevel plan are extremely concise (e.g., lung opacity), while the final report needs to include more detailed information, such as the characteristics of the observation (e.g., a subtle but new lung opacity) and requires preliminary diagnosis inference based on the observation (e.g., lung infection must be suspected). In order to identify those detailed descriptions and clinical inferences about the observations, we need to further consider the image information together with the textual plan, and stronger reasoning must be adopted during the generation process. 

In this paper, we propose ORGAN, an Observation-guided radiology Report GenerAtioN framework. Our framework mainly involves two stages, i.e., the observation planning and the report generation stages, as depicted in Figure 1. In the first stage, our framework produces the observation plan based on the given images, which includes the major findings from the radiographs and their statuses (i.e., positive, negative, and uncertain). In the second stage, we feed both images and the observation plan into a Transformer model to generate the report. Here, a tree reasoning mechanism is devised to enrich the concise observation plan precisely. Specifically, we construct a three-level observation graph, with the high-level observations as the first level, the observation-aware n-grams as the second level, and the specific tokens as the third level. These observation-aware n-grams capture different common descriptions of the observations and serve as the component of observation mentions. Then, we use the tree reasoning mechanism to capture observation-aware information by dynamically aggregating nodes in the graph. 

In conclusion, our main contributions can be summarized as follows: 

- We propose an Observation-guided radiology Report GenerAtioN framework (ORGAN) that can maintain the clinical consistency between radiographs and generated free-text reports. 

- To achieve better observation realization, we construct a three-level observation graph containing observations, n-grams, and tokens based on the training corpus. Then, we perform tree reasoning over the graph to dynamically select observation-relevant information. 

- We conduct extensive experiments on two pub- 

licly available benchmarks, and experimental results demonstrate the effectiveness of our model. We also conduct a detailed case analysis to illustrate the benefits of incorporating observation-related information. 

## **2 Methodology** 

## **2.1 Overview of the Proposed Framework** 

Given an image _X_ , the probability of a report _Y_ = _{y_ 1 _, . . . , yT }_ is denoted as _p_ ( _Y |X_ ). Our framework decomposes _p_ ( _Y |X_ ) into two stages, where the first stage is observation planning, and the second stage is report generation. Specifically, observations of an image _Z_ = _{z_ 1 _, . . . , zL}_ are firstly produced, modeled as _p_ ( _Z|X_ ). Then, the report is generated based on the observation plan and the image, modeled as _p_ ( _Y |X, Z_ ). Finally, our framework maximizes the following probability: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning_images/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning.pdf-0002-10.png)


## **2.2 Observation Plan Extraction and Graph Construction** 

**Observation Plan Extraction.** There are two available tools for extracting observation labels from reports, which are CheXpert (Irvin et al., 2019) and CheXbert (Smit et al., 2020). We use CheXbert[2] instead of CheXpert because the former achieved better performance. To extract the observation plan of a given report, we first adopt the CheXbert to obtain the observation labels within 14 categories _C_ = _{C_ 1 _, . . . , C_ 14 _}_ as indicated in Irvin et al. (2019). More details about the distribution of observation can be found in Appendix A.1. The label (or status) of each category belongs to _Present_ , _Absent_ , and _Uncertain_ , except the _No Finding_ category, which only belongs to _Present_ and _Absent_ . To simplify the observation plan and emphasize the abnormalities presented in a report, we regard _Present_ and _Uncertain_ as Positive and _Absent_ as Negative. Then, observations are divided into a positive collection _C_ /POS and a negative collection _C_ /NEG, and each category with its corresponding label is then converted to its unique observation _Ci_ /POS _∈ C_ /POS or _Ci_ /NEG _∈ C_ /NEG, resulting in 28 observations. For example, as indicated in Figure 1, the report presents _Lung Opacity_ while _Cardiomegaly_ is absent in it. These categories 

- 2https://github.com/stanfordmlgroup/ 

- CheXbert 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning_images/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Stage 1: Observation Planning Stage 2: Observation-Guided Report Generation<br>Observation Plan<br>As ……, there is ① [a subtle but new opacity] at the right lung base,<br>① Lung Opacity / POS …… Given the clinical presentation, ② [pneumonia must be<br>suspected]. …… with bronchiectasis but ③ [no pleural effusion] or<br>② Pneumonia / POS other parenchymal changes. ④ [Normal size of cardiac silhouette].<br>③ Pleural Effusion / NEG Unchanged position of the ⑤ [nasogastric tube].<br>④ Cardiomegaly / NEG Tree<br>⑤ Support Devices / POS Tree Reasoning Reasoning<br>…<br>𝑴× ① ② ⑤<br>Visual Cross-Attn<br>Planner<br>Obs. Cross-Attn 𝑠! 𝑠" 𝑠#<br>Visual Masked Self-Attn 𝑡!,! 𝑡",! 𝑡","Pruning𝑡#,! 𝑡#,"<br>Features<br>𝒗𝟏 𝒗𝟐 ……𝒗𝑵<br>𝒓𝒕 Graph<br>Visual<br>Shifted Encoder<br>CNN Encoder Tokens<br>𝒗𝟏 𝒗𝟐 …… 𝒗𝑳 Observation<br>Graph<br>Observation Plan  Observation-Graph  ① ② … ⑤<br>Construction<br>①→②→③<br>→④→⑤ 𝑠! 𝑠" 𝑠#<br>𝑡!,! 𝑡",! 𝑡"," 𝑡#,! 𝑡#,"<br>**----- End of picture text -----**<br>


Figure 2: The overall framework of ORGAN (“ _Obs. Cross-Attn_ ” in the decoder refers to the observation-related cross-attention module). 

are converted to two observations: _Lung Opacity_ /POS and _Cardiomegaly_ /NEG. Then, we locate each observation by matching mentions in the report and order them according to their positions. These mentions are either provided by Irvin et al. (2019)[3] or extracted from the training corpus (i.e., n-grams), as will be illustrated in the following part. Finally, we can obtain the image’s observation plan _Z_ = _{z_ 1 _, . . . , zL}_ . 

**Tree-Structured Observation Graph Construction** . Since observations are high-level concepts that are implicitly related to tokens in reports, it could be difficult for a model to realize these concepts in detailed reports without more comprehensive modeling. Thus, we propose to construct an observation graph by extracting observation-related n- grams as the connections between observations and tokens for better observation realization. Specifically, it involves two steps to construct such a graph: (1) n-grams extraction, where _n ∈_ [1 _,_ 4] and (2) <observation, n-gram> association. Following previous research (Diao et al., 2021; Su et al., 2021b), we adopt the pointwise mutual information (PMI) (Church and Hanks, 1990) to fulfill 

these two steps, where a higher PMI score implies two units with higher co-occurrence: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning_images/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning.pdf-0003-05.png)


For the first step, we extract n-gram units _S_ = _{s_ 1 _, . . . , s|s|}_ based on the training reports. Given two adjacent units _x_ ¯ and _x_ ˆ of a text sequence, a high PMI score indicates that they are good collection pairs to form a candidate n-gram _s∗_ , while a low PMI score indicates that these two units should be separated. For the second step, given a predefined observation set _O_ = _{z_ 1 _, . . . , z|O|}_ , we extract the observation-related n-gram units with PMI( _zi, sj_ ), where _zi_ is the _i_ -th observation, _sj_ is the _j_ -th n-gram, and _p_ ( _zi, sj_ ) is the frequency that an n-gram _sj_ appears in a report with observation _zi_ in the training set. Then, we can obtain a set of observation-related n-grams _s[z]_ = _{s[z]_ 1 _[, . . . , s][z] k[}]_[,] where _s[z] j_[=] _[ {][t][z] j,_ 1 _[, . . . , t][z] j,n[}]_[, and tokens in n-grams] form the token collection _T_ = _{t_ 1 _, . . . , t|T |}_ . Note that we remove all the stopwords in _T_ , using the vocabulary provided by NLTK[4] . Finally, for each observation, we extract the top-K n-grams as the candidates to construct the graph, which contains 

> 4https://www.nltk.org/ 

3https://github.com/stanfordmlgroup/ chexpert-labeler 

three types of nodes _V_ = _{Z, S, T }_ . We list part of the n-grams in Appendix A.2. After extracting relevant information from the training reports, we construct an observation graph _G_ = _< V, E >_ by introducing three types of edges _E_ = _{E_ 1 _, E_ 2 _, E_ 3 _}_ : 

- _E_ 1: This undirected edge connects two adjacent observations in an observation plan (i.e., < _zi_ , _zi_ +1>). 

- _E_ 2: This directed edge connects an observation and an n-gram (i.e., < _zi_ , _sj_ >). 

- _E_ 3: This directed edge connects an n-gram with its tokens (i.e., < _sj_ , _tk_ >). 

## **2.3 Visual Features Extraction** 

Given an image _X_ , a CNN and an MLP layer are first adopted to extract visual features _**X**_ : 

_**X**_ = _{_ _**x**_ 1 _, . . . ,_ _**x** N }_ = MLP(CNN( _X_ )) _,_ where _**x** i ∈_ R _[h]_ is the _i_ -th visual feature. 

## **2.4 Stage 1: Observation Planning** 

The output of observation planning is an observation sequence, which is the high-level summarization of the radiology report, as shown on the left side of Figure 2. While examining a radiograph, a radiologist must report positive observations. However, only part of the negative observations will be reported by the radiologist, depending on the overall conditions of the radiograph (e.g., co-occurrence of observations or the limited length of a report). Thus, it is difficult to plan without considering the observation dependencies (i.e., label dependencies). Here, we regard the planning problem as a generation task and use a Transformer encoder-decoder for observation planning: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning_images/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning.pdf-0004-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning_images/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning.pdf-0004-10.png)


where _**h**[v] i[∈]_[R] _[h]_[ is the] _[ i]_[-th visual hidden represen-] tation, Encoder _p_ is the visual encoder, Decoder _p_ is the observation decoder, _**z** ∗ ∈_ R _[h]_ is the decoder hidden representation, _**W** z ∈_ R _[|][O][|×][h]_ is the weight matrix, and _**b** z ∈_ R _[|][O][|]_ is the bias vector. Then the planning loss _Lp_ is formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning_images/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning.pdf-0004-12.png)


By increasing _α_ , the planner gives more attention to abnormalities. Note that the plugged _α_ is applied to positive observations and _No Finding_ /NEG instead of _No Finding_ /POS. 

## **2.5 Stage 2: Observation-Guided Report Generation** 

**Observation Graph Encoding** . We use a Transformer encoder to encode the observation graph constructed according to Section 2.2. To be specific, given the observation graph _G_ with nodes _V_ = _{Z, S, T }_ and edges _E_ = _{E_ 1 _, E_ 2 _, E_ 3 _}_ , we first construct the adjacency matrix _A_[ˆ] = _A_ + _I_ based on _E_ . Then, _V_ and _A_[ˆ] are fed into the Transformer for encoding. Now _A_[ˆ] serves as the selfattention mask in the Transformer, which only allows nodes in the graph to attend to connected neighbors and itself. To incorporate the node type information, we add a type embedding _**P** ∈_ R _[h]_ for each node representation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning_images/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning.pdf-0004-16.png)


where Embed( _·_ ) is the embedding function, and _**N** ∈_ R _[h]_ represents node embeddings. For observation nodes, _**P**_ denotes positional embeddings, and for n-gram and token nodes, _**P**_ represents type embeddings. _**Z**_ , _**S**_ , and _**T** ∈_ R _[h]_ are encoded representations of observations, n-grams, and tokens, respectively. 

**Vision-Graph Alignment** . As an observation graph may contain irrelevant information, it is necessary to align the graph with the visual features. Specifically, we jointly encode visual features _**X**_ and token-level node representations _**T**_ so that the node representations can fully interact with the visual features, and we prevent the visual features from attending the node representations by introducing a self-attention mask M: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning_images/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning.pdf-0004-19.png)


where _**h**[v] ,_ _**T**[A] ∈_ R _[h]_ are the visual representation and the aligned token-level node representations, respectively. 

**Observation Graph Pruning** . After aligning visual features and the observation graph, we prune the graph by filtering out irrelevant nodes. The probability of keeping a node is denoted as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning_images/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning.pdf-0004-22.png)


where _**W** d ∈_ R[1] _[×][h]_ is the learnable weight and _bd ∈_ R is the bias. We can optimize the pruning 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning_images/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Query 𝒒<br>Observation<br>First-Level ①② ③④ ⑤<br>Second-LevelN-gram 𝑠! 𝑠" 𝑠# ……<br>Token<br>Third-Level 𝑡!,! 𝑡",! 𝑡"," 𝑡#,! 𝑡#,"<br>Tree<br>Reasoning<br>**----- End of picture text -----**<br>


Figure 3: Illustration of the tree reasoning mechanism. It aggregates information from the observation level to the n-gram level and finally to the token level. 

process with the following loss: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning_images/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning.pdf-0005-03.png)


where _β_ is the weight to tackle the class imbalance issue, and _d_ is the label indicating whether a token appears in the referential report. Finally, we prune the observation graph by masking out token-level nodes with _p_ (1 _|_ _**T**[A]_ ) _<_ 0 _._ 5 and masked token-level node representations denote as _**T**[M]_ = Prune( _**T**_ ). **Tree Reasoning over Observation Graph.** We devise a tree reasoning (TrR) mechanism to aggregate observation-relevant information from the graph dynamically. The overall process is shown in Figure 3, where we aggregate node information from the observation level (i.e., first level) to the n-gram level (i.e., second level), then to the token level (i.e., third level). To be specific, given a query _**q**[l]_ and node representations at _l_ -th level _**k**[l] ∈{_ _**Z** ,_ _**S** ,_ _**T**[M] }_ , the tree reasoning path is _**[Z][S][T]**[ M]_ _**q**_[0] _−→_ _**q**_[1] _−→_ _**q**_[2] _−−→_ _**q**_[3] , and the overall process, is formulated as below: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning_images/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning.pdf-0005-05.png)


where MHA and LayerNorm are the multi-head self-attention, and layer normalization modules (Vaswani et al., 2017), respectively. _**W** q_ , _**W** k_ , and _**W** v ∈_ R _[h][×][h]_ are weight metrics for query, key, and value vector, respectively. Finally, we can obtain the multi-level information _**q**_[3] , containing observation, n-gram, and token information. 

**Report Generation with Tree Reasoning.** As shown in the right side of Figure 2, an observationguided Transformer decoder is devised to incorporate the graph information, including (i) multiple observation-guided decoder blocks (i.e., Decoder _g_ ), which aims to align observations with the visual representations, and (ii) a tree-reasoning block (i.e., TrR _g_ ), which aims to aggregate observationrelevant information. For Decoder _g_ , we insert an 

observation-related cross-attention module before a visually-aware cross-attention module. By doing this, the model can correctly focus on regions closely related to a specific observation. Given the visual representations _**h**[v]_ , the node representations _**V**_ = _{_ _**Z** ,_ _**S** ,_ _**T**[M] }_ , and the hidden representation of the prefix _**h**[w] ∗[∈]_[R] _[h]_[,][the] _[t]_[-th][decoding][step][is] formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning_images/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning.pdf-0005-09.png)


_p_ ( _yt|X, G, Y<t_ ) = Softmax( _**W** g_ _**q** t_[3][+] _**[ b]**[g]_[)] _[,]_ where Self-Attn is the self-attention module, CrossAttn is the cross-attention module, _**h**[s] t[,]_ _**[ h]**[o] t[,]_ _**[ h]**[p] t[∈]_ R _[h]_ are self-attended hidden state, observationrelated hidden state, visually-aware hidden state of Decoder _g_ , respectively. _**h**[d] t[∈]_[R] _[h]_[is][the][self-] attended hidden state of TrR _g_ , _**W** g ∈_ R _[|][V][ |×][h]_ is the weight matrix, and _**b** g ∈_ R _[|][V][ |]_ is the bias vector. We omit other modules (i.e., Layer Normalization and Feed-Forward Network) in the standard Transformer for simplicity. Note that we extend the observation plan _Z_ to an observation graph _G_ , so the probability of _yt_ conditions on _G_ instead of _Z_ . Then, we optimize the generation process using the negative log-likelihood loss: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning_images/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning.pdf-0005-11.png)


Finally, the loss function of the generator is _Lg_ = _Lr_ + _Ld_ . 

## **3 Experiments** 

## **3.1 Datasets** 

Following previous research (Chen et al., 2020, 2021), we use two publicly available benchmarks to evaluate our method, which are IU X-RAY[5] (Demner-Fushman et al., 2016) and MIMICCXR[6] (Johnson et al., 2019). Both datasets have been automatically de-identified, and we use the same preprocessing setup of Chen et al. (2020). 

- IU X-RAY is collected by Indiana University, containing 3,955 reports with two X- ray images per report resulting in 7,470 im- 

> 5https://openi.nlm.nih.gov/ 

> 6https://physionet.org/content/ MIMIC-cxr-jpg/2.0.0/ 

|**Dataset**|**Model**||||**NLG Metrics**|**NLG Metrics**||||**CE Metrics**|**CE Metrics**|
|---|---|---|---|---|---|---|---|---|---|---|---|
||||**B-1**|**B-2**|**B-3**|**B-4**|**MTR**|**R-L**||**P**|**R**<br>**F**1|
|IU<br>X-RAY|R2GEN<br>CA<br>CMCL<br>PPKED<br>R2GENCMN<br>_M_2TR<br>ALIGNTRANSFOMER<br>KNOWMAT<br>CMM-RL<br>CMCA<br>ORGAN (Ours)||0_._470<br>0_._492<br>0_._473<br>0_._483<br>0_._475<br>0_._486<br>0_._484<br>0_._496<br>0_._494<br>0_._496<br>**0****_._510**|0_._304<br>0_._314<br>0_._305<br>0_._315<br>0_._309<br>0_._317<br>0_._313<br>0_._327<br>0_._321<br>**0****_._349**<br>0_._346|0_._219<br>0_._222<br>0_._217<br>0_._224<br>0_._222<br>0_._232<br>0_._225<br>0_._238<br>0_._235<br>**0****_._268**<br>0_._255|0_._165<br>0_._169<br>0_._162<br>0_._168<br>0_._170<br>0_._173<br>0_._173<br>0_._178<br>0_._181<br>**0****_._215**<br>0_._195|-<br>0_._193<br>0_._186<br>-<br>0_._191<br>0_._192<br>-<br>-<br>0_._201<br>**0****_._209**<br>0_._205|0_._371<br>0_._381<br>0_._378<br>0_._376<br>0_._375<br>0_._390<br>0_._379<br>0_._381<br>0_._384<br>0_._392<br>**0****_._399**||-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|
|MIMIC<br>-CXR|R2GEN<br>CA<br>CMCL<br>PPKED<br>R2GENCMN<br>_M_2TR<br>ALIGNTRANSFOMER<br>KNOWMAT<br>CMM-RL<br>CMCA<br>ORGAN (Ours)||0_._353<br>0_._350<br>0_._344<br>0_._360<br>0_._353<br>0_._378<br>0_._378<br>0_._363<br>0_._381<br>0_._360<br>**0****_._407**|0_._218<br>0_._219<br>0_._217<br>0_._224<br>0_._218<br>0_._232<br>0_._235<br>0_._228<br>0_._232<br>0_._227<br>**0****_._256**|0_._145<br>0_._152<br>0_._140<br>0_._149<br>0_._148<br>0_._154<br>0_._156<br>0_._156<br>0_._155<br>0_._156<br>**0****_._172**|0_._103<br>0_._109<br>0_._097<br>0_._106<br>0_._106<br>0_._107<br>0_._112<br>0_._115<br>0_._109<br>0_._117<br>**0****_._123**|0_._142<br>0_._151<br>0_._133<br>0_._149<br>0_._142<br>0_._145<br>-<br>-<br>0_._151<br>0_._148<br>**0****_._162**|0_._270<br>0_._283<br>0_._281<br>0_._284<br>0_._278<br>0_._272<br>0_._283<br>0_._284<br>0_._287<br>0_._287<br>**0****_._293**||0_._333<br>-<br>-<br>-<br>0_._344<br>0_._240<br>-<br>**0****_._458**<br>0_._342<br>0_._444<br>0_._416|0_._273<br>0_._276<br>-<br>-<br>-<br>-<br>-<br>-<br>0_._275<br>0_._278<br>**0****_._428**<br>0_._308<br>-<br>-<br>0_._348<br>0_._371<br>0_._294<br>0_._292<br>0_._297<br>0_._356<br>0_._418<br>**0****_._385**|



Table 1: Experimental Results of our model and baselines on the IU X-RAY dataset and the MIMIC-CXR dataset. The best results are in **boldface** , and the underlined are the second-best results. 

ages in total. We split the dataset into train/validation/test sets with a ratio of 7:1:2, which is the same data split as in (Chen et al., 2020). 

- MIMIC-CXR consists of 377,110 chest X-ray images and 227,827 reports from 63,478 patients. We adopt the standard train/validation/test splits. 

## **3.2 Evaluation Metrics and Baselines** 

We adopt natural language generation metrics (NLG Metrics) and clinical efficacy (CE Metrics) to evaluate the models. BLEU (Papineni et al., 2002), METEOR (Banerjee and Lavie, 2005), and ROUGE (Lin, 2004) are selected as NLG Metrics, and we use the MS-COCO caption evaluation tool[7] to compute the results. For CE Metrics, we adopt CheXpert (Irvin et al., 2019) for MIMIC-CXR dataset to label the generated reports compared with disease labels of the references. 

To evaluate the performance of ORGAN, we compare it with the following 10 state-of-the-art (SOTA) baselines: R2GEN (Chen et al., 2020), CA (Liu et al., 2021c), CMCL (Liu et al., 2021a), PPKED (Liu et al., 2021b), R2GENCMN (Chen 

7https://github.com/tylin/coco-caption 

et al., 2021), ALIGNTRANSFORMER (You et al., 2021), KNOWMAT (Yang et al., 2021), _M_[2] TR (Nooralahzadeh et al., 2021), CMM-RL (Qin and Song, 2022), and CMCA (Song et al., 2022). 

## **3.3 Implementation Details** 

We adopt the ResNet-101 (He et al., 2015) pretrained on ImageNet (Deng et al., 2009) as the visual extractor. For IU X-RAY, we further fine-tune ResNet-101 on CheXpert (Irvin et al., 2019). The layer number of all the encoders and decoders is set to 3 except for Graph Encoder, where the layer number is set to 2. The input dimension and the feed-forward network dimension of a Transformer block are set to 512, and each block contains 8 attention heads. The beam size for decoding is set to 4, and the maximum decoding step is set to 64/104 for IU X-RAY and MIMIC-CXR, respectively. 

We use AdamW (Loshchilov and Hutter, 2019) as the optimizer and set the initial learning rate for the visual extractor as 5e-5 and 1e-4 for the rest of the parameters, with a linear schedule decreasing from the initial learning rate to 0. _α_ is set to 0.5, the dropout rate is set to 0.1, and the batch size is set to 32. For IU X-ray, we train the planner/generator for 15/15 epochs, and _β_ is set to 2. For MIMIC-CXR, the training epoch of 

|**Dataset**|**Model**|**NLG Metrics**|**CE Metrics**|
|---|---|---|---|
|||**B-1**<br>**B-2**<br>**B-3**<br>**B-4**<br>**MTR**<br>**R-L**|**P**<br>**R**<br>**F**1|
|IU<br>X-RAY|ORGAN<br>ORGAN_w/o_Plan<br>ORGAN_w/o_Graph<br>ORGAN_w/o_TrR|0_._510<br>0_._346<br>0_._255<br>0_._195<br>0_._205<br>0_._399<br>0_._406<br>0_._254<br>0_._178<br>0_._133<br>0_._167<br>0_._372<br>0_._461<br>0_._302<br>0_._218<br>0_._164<br>0_._186<br>0_._383<br>0_._494<br>0_._335<br>0_._247<br>0_._190<br>0_._203<br>0_._395|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|
|MIMIC<br>-CXR|ORGAN<br>ORGAN_w/o_Plan<br>ORGAN_w/o_Graph<br>ORGAN_w/o_TrR|0_._407<br>0_._256<br>0_._172<br>0_._123<br>0_._162<br>0_._293<br>0_._334<br>0_._211<br>0_._145<br>0_._107<br>0_._136<br>0_._282<br>0_._369<br>0_._233<br>0_._158<br>0_._113<br>0_._151<br>0_._290<br>0_._405<br>0_._254<br>0_._170<br>0_._121<br>0_._161<br>0_._291|0_._416<br>0_._418<br>0_._385<br>0_._384<br>0_._239<br>0_._252<br>0_._401<br>0_._415<br>0_._383<br>0_._411<br>0_._419<br>0_._386|



Table 2: Ablation results of our model and its variants, where ORGAN _w/o_ Plan is the standard Transformer model. 

the planner/generator is set to 3/5, and _β_ is set to 5. We select the best checkpoints of the planner based on micro F1 of all observations and select the generator based on the BLEU-4 on the validation set. Our model has 65.9M parameters, and the implementations are based on HuggingFace’s Transformers (Wolf et al., 2020). We conduct all the experiments on an NVIDIA-3090 GTX GPU with mixed precision. The NLTK package version is 3.6.2. 

## **4 Results** 

## **4.1 NLG Results** 

Table 1 shows the experimental results. ORGAN outperforms most of the baselines (except CMCA on IU X-RAY) and achieves state-of-the-art performance. Specifically, our model achieves 0 _._ 195 BLEU-4 on the IU X-XRAY dataset, which is the second-best result, and 0 _._ 123 BLEU-4 on the MIMIC-CXR dataset, leading to a 5 _._ 1% increment of compared to the best baseline (i.e., CMCA). In terms of METEOR, ORGAN achieves competitive performance on both datasets. In addition, our model increases R-L by 0 _._ 6% on the MIMICCXR dataset compared to the best baseline and achieves the second-best result on the IU X-RAY dataset. This indicates that by introducing the guidance of observations, ORGAN can generate more coherent text than baselines. However, we notice that on the IU X-RAY dataset, there is still a performance gap between our model and the best baseline (i.e., CMCA). The reason may be that the overall data size of this dataset is small ( _∼_ 2,000 samples for training). It is difficult to train a good planner using a small training set, especially with cross-modal data. As we can see from Table 3, the planner only achieves 0 _._ 132 Macro-F1 on the IU X-RAY dataset, which is relatively low compared 

|**Dataset**|**Micro-F**1<br>**Macro-F**1<br>**B-2**|
|---|---|
|IU X-RAY|0_._507<br>0_._132<br>0_._499|
|MIMIC-CXR|0_._574<br>0_._397<br>0_._357|



Table 3: Experimental results of observation planning. Macro-F1 and Micro-F1 denote the macro F1 and micro F1 of abnormal observations, respectively. 

|**K**|**B-2/4**<br>**MTR**<br>**R-L**|
|---|---|
|10<br>20<br>30|0_._309/0_._170<br>0_._192<br>0_._388<br>0_._333/0_._180<br>0_._202<br>0_._393<br>0_._346/0_._195<br>0_._205<br>0_._399|
|10<br>20<br>30|0_._249/0_._118<br>0_._161<br>0_._290<br>0_._252/0_._120<br>0_._159<br>0_._292<br>0_._256/0_._123<br>0_._162<br>0_._293|



Table 4: Experimental results under the different number (K) of selected n-grams. 

to the performance of the planner on the MIMICCXR dataset. Thus, accumulation errors unavoidably propagate to the generator, which leads to lower performance. 

## **4.2 Clinical Efficacy Results** 

The clinical efficacy results are listed on the right side of Table 1. On the MIMIC-CXR dataset, our model outperforms previous SOTA results. Specifically, our model achieves 0 _._ 385 F1, increasing by 1 _._ 4% compared to the best baseline. In addition, 0 _._ 416 precision and 0 _._ 418 recall are achieved by ORGAN, which are competitive results. This indicates that our model can successfully maintain the clinical consistency between the images and the reports. 

## **4.3 Ablation Results** 

To examine the effect of the observation plan and the TrR mechanism, we perform ablation tests, and the ablation results are listed in Table 2. There are three variants: (1) ORGAN _w/o_ Plan, which does not consider observation information, (2) ORGAN 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning_images/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning.pdf-0008-00.png)


**----- Start of picture text -----**<br>
Observation Plan: Observation level Cardiomegaly/POS<br>① Consolidation/NEG<br>② Pleural Effusion/NEG<br>③ Pneumothorax/NEG N-gram level mild pulmonary  moderate moderate<br>④ Cardiomegaly/POS vascular cardiomegaly cardiomegaly<br>Tree Reasoning Path over Graph: Token level mild moderate cardiomegaly<br>Report Generated by ORGAN<br>frontal and lateral views of the chest were obtained. the lungs are clear ①[without focal consolidation]. ②[no pleural effusion] or ③[pneumothorax] is seen.<br>there is ④[mild to moderate cardiomegaly] unchanged. the aorta is calcified and tortuous.<br>Report Generated by ORGAN w/o TrR<br>frontal and lateral views of the chest were obtained. the lungs are clear ①[without focal consolidation]. ②[no pleural effusion] or ③[pneumothorax] is seen.<br>④[the cardiac silhouette is mildly enlarged.] mediastinal silhouettes are unchanged. there is mild to moderate degenerative change in the thoracic spine.<br>Report Generated by OGRAN w/o Plan<br>frontal and lateral views of the chest were obtained. there is mild pulmonary vascular congestion without overt pulmonary edema. ①[no focal consolidation]<br>②[pleural effusion] or ③[pneumothorax] is seen. the ④[cardiac and mediastinal silhouettes are stable].<br>Reference<br>frontal and lateral views of the chest. the lungs are ①[clear of focal consolidation] ②[effusion] or ③[pneumothorax]. ④ [the heart is enlarged] similar to<br>prior. right upper extremity vascular stent is partially visualized. multiple thoracic compression deformities are again seen.<br>**----- End of picture text -----**<br>


Figure 4: Case study of our model with the tree reasoning path of the mention " _mild to moderate cardiomegaly_ ." 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning_images/acl_2023_ORGAN_Observation-Guided_Radiology_Report_Generation_via_Tree_Reasoning.pdf-0008-02.png)


**----- Start of picture text -----**<br>
Oracle Plan:<br>① Pneumonia/NEG ② Edema/NEG,<br>③ Cardiomegaly/POS ④ Enlarged Card./POS<br>⑤ Pleural Effusion/NEG ⑥ Pneumothorax/NEG<br>Reference:  …… there are ① [no lung opacities concerning<br>for pneumonia or ② [pulmonary edema]. ③ [heart size is<br>mildly enlarged and stable since]. ④ [mediastinal and hilar<br>contours are unchanged]. there is ⑤ [no pleural effusion]<br>or ⑥ [pneumothorax].<br>Generated Plan:<br>① Cardiomegaly/POS ② Edema/NEG<br>③ Pleural Effusion/NEG ④ Pneumonia/NEG<br>Generated Report:  …… ① [mild enlargement of the cardiac<br>silhouette] ② [without pulmonary vascular congestion] ③<br>[pleural effusion] or ④ [acute focal pneumonia].<br>Oracle Plan:<br>① Support Devices/POS ② Pneumothorax/NEG<br>Reference:  …… ① [right chest tubes] appears to have been<br>removed. ② [no definite pneumothorax is appreciated].<br>……<br>Generated Plan:<br>① Support Devices/POS ② Pneumothorax /POS ③ Lung<br>Opacity/POS ④ Atelectasis/POS ⑤ Pleural Effusion/POS<br>Generated Report: …… ① [right-sided chest tube with the<br>tip in the mid svc]. there is ② [a small right apical<br>pneumothorax]. there is ③ [increased opacity] at the right<br>lung base likely representing ④ [atelectasis]. there is ⑤ [a<br>small left pleural effusion].<br>**----- End of picture text -----**<br>


Figure 5: Examples of error cases. _Enlarged Card._ refers to _Enlarged Cardiomediastinum_ . The upper case omits one positive observation and the bottom case contains false positive observations. 

_w/o_ Graph, which only considers observations but not the observation graph, (3) ORGAN _w/o_ TrR, which select information without using the TrR mechanism. Compared to the full model, the performance of ORGAN _w/o_ Plan drops significantly on both datasets. This indicates that observation information plays a vital role in generating reports. For ORGAN _w/o_ Graph, the performance on NLG metrics decreases significantly, but the performance of clinical efficacy remains nearly the same as the full model. This is reasonable because the observation graph is designed to enrich the observation plan to achieve better word-level realization. On the per- 

formance of ORGAN _w/o_ TrR, a similar result of ORGAN _w/o_ Graph is observed. This indicates that TrR can enrich the plan information, and stronger reasoning can help generate high-quality reports. 

We also conduct experiments on the impact of the number (K) of selected n-grams, as shown in Table 4. There is a performance gain when increasing K from 10 to 20 and to 30 on both datasets. On the IU X-RAY dataset, B-2 increases by 2 _._ 4% and 3 _._ 7% and B-4 rises by 1 _._ 0% and 1 _._ 5%. A similar trend is also observed on the MIMIC-CXR dataset. 

## **4.4 Qualitative Analysis** 

We conduct a case study and analyze some error cases on the MIMIC-CXR dataset to provide more insights. 

**Case Study.** We conduct a case study to show how the observation and the tree reasoning mechanism guide the report generation process, as shown in Figure 4. We show the generated reports of ORGAN, ORGAN _w/o_ TrR, and ORGAN _w/o_ Plan, respectively. All three models successfully generate the first three negative observations and the last positive observation. However, variant _w/o_ plan generates " _mild pulmonary vascular congestion without overt pulmonary edema_ " which is not consistent with the radiograph. In terms of the output of variant _w/o_ TrR, " _mediastinal silhouettes are unchanged_ " is closely related to observation _Enlarged Cardiomediastinum_ instead of _Cardiomegaly_ . Only ORGAN can generate the _Cardiomegaly_ /POS presented in the observation plan with a TrR path. This indicates that observations play a vital role in maintaining clinical consistency. In addition, most of 

the tokens in the observation mention _mild to moderate cardiomegaly_ can be found in the observation graph, which demonstrates that the graph can provide useful information in word-level realization. **Error Analysis.** We depicte error cases generated by ORGAN in Figure 5. The major error is caused by introducing incorrect observation plans. Specifically, the generated plan of the upper case omits one positive observation (i.e., _Enlarged Cardiomediastinum_ /POS), resulting in false negative observations in its corresponding generated report. Another error is false positive observations appearing in the generated reports (e.g., the bottom case). Thus, how to improve the performance of the planner is a potential future work to enhance clinical accuracy. 

## **5 Related Work** 

## **5.1 Image Captioning and Medical Report Generation** 

Image Captioning (Vinyals et al., 2015; Rennie et al., 2017; Lu et al., 2017; Anderson et al., 2018) has long been an attractive research topic, and there has been a surging interest in developing medical AI applications. Medical Report Generation (Jing et al., 2018; Li et al., 2018) is one of these applications. Chen et al. (2020) proposed a memorydriven Transformer model to generate radiology reports. Chen et al. (2021) further proposed a crossmodal memory network to facilitate report generation. Qin and Song (2022) proposed to utilize reinforcement learning (Williams, 1992) to align the cross-modal information between the image and the corresponding report. In addition to these methods, Liu et al. (2021c) proposed the Contrastive Attention model comparing the given image with normal images to distill information. Yang et al. (2021) proposed to introduce general and specific knowledge extracted from RadGraph (Jain et al., 2021) in report generation. Liu et al. (2021a) proposed a competence-based multimodal curriculum learning to guide the learning process. Liu et al. (2021b) proposed to explore and distill posterior and prior knowledge for report generation. 

Several research works focus on improving the clinical accuracy of the generated reports. Liu et al. (2019) proposed a clinically coherent reward for clinically accurate reinforcement learning to improve clinical accuracy. Lovelace and Mortazavi (2020) proposed to use CheXpert (Irvin et al., 2019) as a source of clinical information to generate clini- 

cally coherent reports. Miura et al. (2021) proposed to use entity matching score as a reward to encourage the model to generate factually complete and consistent radiology reports. Nishino et al. (2022) proposed a planning-based method and regarded the report generation task as the data-to-text generation task. 

## **5.2 Planning in Text Generation** 

Another line of research closely related to our work is planning in text generation, which has been applied to multiple tasks (e.g., Data-to-Text Generation, Summarization, and Story Generation). Hua and Wang (2020) propose a global planning and iterative refinement model for long text generation. Kang and Hovy (2020) propose a self-supervised planning framework for paragraph completion. Hu et al. (2022) propose a dynamic planning model for long-form text generation to tackle the issue of incoherence outputs. Moryossef et al. (2019) proposed a neural data-to-text generation by separating planning from realization. Su et al. (2021a) proposed a controlled data-to-text generation framework by planning the order of content in a table. 

## **6 Conclusion** 

In this paper, we propose ORGAN, an observationguided radiology report generation framework, which first produces an observation plan and then generates the corresponding report based on the radiograph and the plan. To achieve better observation realization, we construct a three-level observation graph containing observations, observationaware n-grams, and tokens, and we propose a tree reasoning mechanism to capture observationrelated information by dynamically aggregating nodes in the graph. Experimental results demonstrate the effectiveness of our proposed framework in terms of maintaining the clinical consistency between radiographs and generated reports. 

## **Limitations** 

There are several limitations to our framework. Specifically, since observations are introduced as guiding information, our framework requires observation extraction tools to label the training set in advance. Then, the nodes contained in the observation graph are mined from the training data. As a result, the mined n-grams could be biased when the overall size of the training set is small. In addition, our framework is a pipeline, and the report genera- 

tion performance highly relies on the performance of observation planning. Thus, errors could accumulate through the pipeline, especially for small datasets. Finally, our framework is designed for radiology report generation targeting Chest X-ray images. However, there are other types of medical images (e.g., Fundus Fluorescein Angiography images) that our framework needs to examine. 

## **Ethics Statement** 

The IU X-RAY(Demner-Fushman et al., 2016) and MIMIC-CXR (Johnson et al., 2019) datasets have been automatically de-identified to protect patient privacy. The proposed system is intended to generate radiology reports automatically, alleviating the workload of radiologists. However, we notice that the proposed system can generate false positive observations and inaccurate diagnoses due to systematic biases. If the system, as deployed, would learn from further user input (i.e., patients’ radiographs), there are risks of personal information leakage while interacting with the system. This might be mitigated by using anonymous technology to protect privacy. Thus, we urge users to cautiously examine the ethical implications of the generated output in real-world applications. 

## **Acknolwedgments** 

This work was supported in part by General Program of National Natural Science Foundation of China (Grant No. 82272086, 62076212), Guangdong Provincial Department of Education (Grant No. 2020ZDZX3043), Shenzhen Natural Science Fund (JCYJ20200109140820699 and the Stable Support Plan Program 20200925174052004), and the Research Grants Council of Hong Kong (15207920, 15207821, 15207122). 

## **References** 

- Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. 2018. Bottom-up and top-down attention for image captioning and visual question answering. In _2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018_ , pages 6077–6086. Computer Vision Foundation / IEEE Computer Society. 

- Satanjeev Banerjee and Alon Lavie. 2005. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In _Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Transla-_ 

_tion and/or Summarization_ , pages 65–72, Ann Arbor, Michigan. Association for Computational Linguistics. 

- Zhihong Chen, Yaling Shen, Yan Song, and Xiang Wan. 2021. Cross-modal memory networks for radiology report generation. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021_ , pages 5904–5914. Association for Computational Linguistics. 

- Zhihong Chen, Yan Song, Tsung-Hui Chang, and Xiang Wan. 2020. Generating radiology reports via memory-driven transformer. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing_ . 

- Kenneth Ward Church and Patrick Hanks. 1990. Word association norms, mutual information, and lexicography. _Computational Linguistics_ , 16(1):22–29. 

- Dina Demner-Fushman, Marc D Kohli, Marc B Rosenman, Sonya E Shooshan, Laritza Rodriguez, Sameer Antani, George R Thoma, and Clement J McDonald. 2016. Preparing a collection of radiology examinations for distribution and retrieval. _Journal of the American Medical Informatics Association_ , 23(2):304–310. 

- Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. Imagenet: A large-scale hierarchical image database. In _2009 IEEE Conference on Computer Vision and Pattern Recognition_ , pages 248–255. 

- Shizhe Diao, Ruijia Xu, Hongjin Su, Yilei Jiang, Yan Song, and Tong Zhang. 2021. Taming pre-trained language models with n-gram representations for lowresource domain adaptation. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pages 3336–3349, Online. Association for Computational Linguistics. 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015. Deep residual learning for image recognition. _arXiv preprint arXiv:1512.03385_ . 

- Zhe Hu, Hou Pong Chan, Jiachen Liu, Xinyan Xiao, Hua Wu, and Lifu Huang. 2022. PLANET: Dynamic content planning in autoregressive transformers for long-form text generation. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 2288– 2305, Dublin, Ireland. Association for Computational Linguistics. 

- Xinyu Hua and Lu Wang. 2020. PAIR: Planning and iterative refinement in pre-trained transformers for long text generation. In _Proceedings of the 2020_ 

_Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 781–793, Online. Association for Computational Linguistics. 

- Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn L. Ball, Katie S. Shpanskaya, Jayne Seekins, David A. Mong, Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson, Curtis P. Langlotz, Bhavik N. Patel, Matthew P. Lungren, and Andrew Y. Ng. 2019. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In _The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019_ , pages 590–597. AAAI Press. 

- Saahil Jain, Ashwin Agrawal, Adriel Saporta, Steven Q. H. Truong, Du Nguyen Duong, Tan Bui, Pierre Chambon, Yuhao Zhang, Matthew P. Lungren, Andrew Y. Ng, Curtis P. Langlotz, and Pranav Rajpurkar. 2021. Radgraph: Extracting clinical entities and relations from radiology reports. _CoRR_ , abs/2106.14463. 

- Baoyu Jing, Pengtao Xie, and Eric P. Xing. 2018. On the automatic generation of medical imaging reports. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20, 2018, Volume 1: Long Papers_ , pages 2577–2586. Association for Computational Linguistics. 

- Alistair EW Johnson, Tom J Pollard, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Yifan Peng, Zhiyong Lu, Roger G Mark, Seth J Berkowitz, and Steven Horng. 2019. Mimic-cxr-jpg, a large publicly available database of labeled chest radiographs. _arXiv preprint arXiv:1901.07042_ . 

- Dongyeop Kang and Eduard Hovy. 2020. Plan ahead: Self-supervised text planning for paragraph completion task. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 6533–6543, Online. Association for Computational Linguistics. 

- Yuan Li, Xiaodan Liang, Zhiting Hu, and Eric P. Xing. 2018. Hybrid retrieval-generation reinforced agent for medical image report generation. In _Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada_ , pages 1537–1547. 

- Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In _Text Summarization Branches Out_ , pages 74–81, Barcelona, Spain. Association for Computational Linguistics. 

- Fenglin Liu, Shen Ge, and Xian Wu. 2021a. Competence-based multimodal curriculum learning 

   - for medical report generation. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021_ , pages 3001–3012. Association for Computational Linguistics. 

- Fenglin Liu, Xian Wu, Shen Ge, Wei Fan, and Yuexian Zou. 2021b. Exploring and distilling posterior and prior knowledge for radiology report generation. In _IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021_ , pages 13753–13762. Computer Vision Foundation / IEEE. 

- Fenglin Liu, Changchang Yin, Xian Wu, Shen Ge, Ping Zhang, and Xu Sun. 2021c. Contrastive attention for automatic chest x-ray report generation. In _Findings of the Association for Computational Linguistics: ACL/IJCNLP 2021, Online Event, August 1-6, 2021_ , volume ACL/IJCNLP 2021 of _Findings of ACL_ , pages 269–280. Association for Computational Linguistics. 

- Guanxiong Liu, Tzu-Ming Harry Hsu, Matthew B. A. McDermott, Willie Boag, Wei-Hung Weng, Peter Szolovits, and Marzyeh Ghassemi. 2019. Clinically accurate chest x-ray report generation. _CoRR_ , abs/1904.02633. 

- Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In _7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019_ . OpenReview.net. 

- Justin Lovelace and Bobak Mortazavi. 2020. Learning to generate clinically coherent chest X-ray reports. In _Findings of the Association for Computational Linguistics: EMNLP 2020_ , pages 1235–1243, Online. Association for Computational Linguistics. 

- Jiasen Lu, Caiming Xiong, Devi Parikh, and Richard Socher. 2017. Knowing when to look: Adaptive attention via a visual sentinel for image captioning. In _2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017_ , pages 3242–3250. IEEE Computer Society. 

- Yasuhide Miura, Yuhao Zhang, Emily Tsai, Curtis Langlotz, and Dan Jurafsky. 2021. Improving factual completeness and consistency of image-to-text radiology report generation. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 5288–5304, Online. Association for Computational Linguistics. 

- Amit Moryossef, Yoav Goldberg, and Ido Dagan. 2019. Step-by-step: Separating planning from realization in neural data-to-text generation. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics:_ 

_Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 2267–2277, Minneapolis, Minnesota. Association for Computational Linguistics. 

- Toru Nishino, Yasuhide Miura, Tomoki Taniguchi, Tomoko Ohkuma, Yuki Suzuki, Shoji Kido, and Noriyuki Tomiyama. 2022. Factual accuracy is not enough: Planning consistent description order for radiology report generation. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , Online. Association for Computational Linguistics. 

- Farhad Nooralahzadeh, Nicolas Perez Gonzalez, Thomas Frauenfelder, Koji Fujimoto, and Michael Krauthammer. 2021. Progressive transformer-based generation of radiology reports. In _Findings of the Association for Computational Linguistics: EMNLP 2021_ , pages 2824–2832, Punta Cana, Dominican Republic. Association for Computational Linguistics. 

- Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics_ , pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics. 

- Han Qin and Yan Song. 2022. Reinforced cross-modal alignment for radiology report generation. In _Findings of the Association for Computational Linguistics: ACL 2022, Dublin, Ireland, May 22-27, 2022_ , pages 448–458. Association for Computational Linguistics. 

- Steven J. Rennie, Etienne Marcheret, Youssef Mroueh, Jerret Ross, and Vaibhava Goel. 2017. Self-critical sequence training for image captioning. In _2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017_ , pages 1179–1195. IEEE Computer Society. 

- Akshay Smit, Saahil Jain, Pranav Rajpurkar, Anuj Pareek, Andrew Ng, and Matthew Lungren. 2020. Combining automatic labelers and expert annotations for accurate radiology report labeling using BERT. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 1500–1519, Online. Association for Computational Linguistics. 

- Xiao Song, Xiaodan Zhang, Junzhong Ji, Ying Liu, and Pengxu Wei. 2022. Cross-modal contrastive attention model for medical report generation. In _Proceedings of the 29th International Conference on Computational Linguistics_ , pages 2388–2397, Gyeongju, Republic of Korea. International Committee on Computational Linguistics. 

- Yixuan Su, David Vandyke, Sihui Wang, Yimai Fang, and Nigel Collier. 2021a. Plan-then-generate: Controlled data-to-text generation via planning. In _Findings of the Association for Computational Linguistics: EMNLP 2021_ , pages 895–909, Punta Cana, Dominican Republic. Association for Computational Linguistics. 

- Yixuan Su, Yan Wang, Deng Cai, Simon Baker, Anna Korhonen, and Nigel Collier. 2021b. Prototype-tostyle: Dialogue generation with style-aware editing on retrieval memory. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 29:2152– 2161. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In _Proceedings of the 31st International Conference on Neural Information Processing Systems_ , NIPS’17, page 6000–6010, Red Hook, NY, USA. Curran Associates Inc. 

- Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. 2015. Show and tell: A neural image caption generator. In _CVPR_ , pages 3156–3164. IEEE Computer Society. 

- Ronald J. Williams. 1992. Simple statistical gradientfollowing algorithms for connectionist reinforcement learning. _Mach. Learn._ , 8(3–4):229–256. 

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_ , pages 38–45, Online. Association for Computational Linguistics. 

- Shuxin Yang, Xian Wu, Shen Ge, Shaohua Kevin Zhou, and Li Xiao. 2021. Knowledge matters: Radiology report generation with general and specific knowledge. _CoRR_ , abs/2112.15009. 

- Di You, Fenglin Liu, Shen Ge, Xiaoxia Xie, Jing Zhang, and Xian Wu. 2021. Aligntransformer: Hierarchical alignment of visual regions and disease tags for medical report generation. In _Medical Image Computing and Computer Assisted Intervention - MICCAI 2021 - 24th International Conference, Strasbourg, France, September 27 - October 1, 2021, Proceedings, Part III_ , volume 12903 of _Lecture Notes in Computer Science_ , pages 72–82. Springer. 

## **A Appendices** 

## **A.1 Observation Statistics** 

There are 14 categories of observations: _No Finding_ , _Enlarged Cardiomediastinum_ , _Cardiomegaly_ , _Lung Lesion_ , _Lung Opacity_ , _Edema_ , _Consolidation_ , _Pneumonia_ , _Atelectasis_ , _Pneumothorax_ , _Pleural Effusion_ , _Pleural Other_ , _Fracture_ , and _Support Devices_ . Table 5 lists the observation distributions annotated by CheXbert(Smit et al., 2020) in the train/valid/test split of two benchmarks. 

|**#Observation**|**IU X-RAY**|**MIMIC-CXR**|
|---|---|---|
|_No Finding_/POS<br>_No Finding_/NEG|744/108/318<br>1,325/188/272|64,677/514/229<br>206,133/1,616/3,629|
|_Cardiomegaly_/POS<br>_Cardiomegaly_/NEG|244/38/61<br>1,375/198/386|70,561/514/1,602<br>85,448/714/801|
|_Pleural Effusion_/POS<br>_Pleural Effusion_/NEG|60/13/15<br>1,559/230/452|56,972/477/1,379<br>170,989/1,310/1,763|
|_Pneumothorax_/POS<br>_Pneumothorax_/NEG|9/2/5<br>1,528/231/449|8,707/62/106<br>190,356/1,495/2,338|
|_Enlarged Card._/POS<br>_Enlarged Card._/NEG|159/29/28<br>1,200/161/384|49,806/413/1,140<br>129,360/1,006/868|
|_Consolidation_/POS<br>_Consolidation_/NEG|17/1/3<br>763/117/210|14,449/119/384<br>97,197/788/964|
|_Lung Opacity_/POS<br>_Lung Opacity_/NEG|295/35/57<br>331/49/82|67,714/497/1,448<br>8,157/73/125|
|_Fracture_/POS<br>_Fracture_/NEG|84/6/15<br>137/22/50|11,070/59/232<br>9,632/72/53|
|_Lung Lesion_/POS<br>_Lung Lesion_/NEG|85/14/17<br>92/10/30|11,717/123/300<br>1,972/21/11|
|_Edema_/POS<br>_Edema_/NEG|28/2/7<br>119/17/31|33,034/257/899<br>51,639/409/669|
|_Atelectasis_/POS<br>_Atelectasis_/NEG|143/15/37<br>3/0/0|68,273/515/1,210<br>563/5/9|
|_Support Devices_/POS<br>_Support Devices_/NEG|89/20/16<br>1/0/0|60,455/450/1,358<br>1,081/7/11|
|_Pneumonia_/POS<br>_Pneumonia_/NEG|20/2/1<br>68/9/25|23,945/184/503<br>21,976/165/411|
|_Pleural Other_/POS<br>_Pleural Other_/NEG|32/4/7<br>0/0/0|7,296/70/184<br>63/0/0|



Table 5: Observation distribution in train/valid/test split of two benchmarks. _Enlarged Card._ refers to _Enlarged Cardiomediastinum_ . 

## Enlarged Cardiomediastinum/NEG 

- cardiomediastinal silhouette is unremarkable 

- cardiomediastinal contours are normal 

- cardiomediastinal silhouette is normal 

## Enlarged Cardiomediastinum/POS 

- cardiomediastinal contours are stable 

- mediastinal contours are unchanged 

- mediastinal contours are stable 

## Consolidation/NEG 

- focal consolidation effusion 

- consolidation effusion 

- without focal consolidation 

## Consolidation/POS 

- new focal consolidation 

## **A.2 Observation-aware N-grams** 

Here are some of the observation-aware n-grams we use in our experiments, as shown in Figure 6. These categories are _Enlarged Cardiomediastinum_ , _Consolidation_ , and _Cardiomegaly_ . 

- new consolidation 

- new focal 

## Cardiomegaly/NEG 

- heart is normal 

- heart size is normal 

- heart size is within 

## Cardiomegaly/POS 

- moderate cardiomegaly is unchanged 

- cardiomegaly is stable 

- mild cardiomegaly is unchanged 

Figure 6: Observation-aware n-grams. 

