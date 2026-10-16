# **Align and Aggregate: Compositional Reasoning with Video Alignment and Answer Aggregation for Video Question-Answering** 

Zhaohe Liao[1][*] 

Jiangtong Li[2][*] 

Li Niu[1][†] 

Liqing Zhang[1][†] 

1 Shanghai Jiao Tong University 

_{_ zhaoheliao, ustcnewly, zhang-lq _}_ @sjtu.edu.cn 

2 Tongji University jiangtongli@tongji.edu.cn 

## **Abstract** 

_Despite the recent progress made in Video QuestionAnswering (VideoQA), these methods typically function as black-boxes, making it difficult to understand their reasoning processes and perform consistent compositional reasoning. To address these challenges, we propose a modelagnostic Video Alignment and Answer Aggregation (VA_[3] _) framework, which is capable of enhancing both compositional consistency and accuracy of existing VidQA methods by integrating video aligner and answer aggregator modules. The video aligner hierarchically selects the relevant video clips based on the question, while the answer aggregator deduces the answer to the question based on its sub-questions, with compositional consistency ensured by the information flow along question decomposition graph and the contrastive learning strategy. We evaluate our framework on three settings of the AGQA-Decomp dataset with three baseline methods, and propose new metrics to measure the compositional consistency of VidQA methods more comprehensively. Moreover, we propose a large language model (LLM) based automatic question decomposition pipeline to apply our framework to any VidQA dataset. We extend MSVD and NExT-QA datasets with it to evaluate our VA_[3] _framework on broader scenarios. Extensive experiments show that our framework improves both compositional consistency and accuracy of existing methods, leading to more interpretable real-world VidQA models._ 

## **1. Introduction** 

Video Question-Answering (VidQA) has emerged as a popular research topic in recent years, with potential applications in interactive artificial intelligence and recognition science. 

> *These two authors contributed equally to this work. 

> †The corresponding authors. 

With the development of representing video, question and their alignment, numerous works [6, 14, 21, 23, 34, 35, 49] have achieved considerable success in both open-ended VidQA [22, 53] and multi-choice VidQA [22, 30, 48]. 

However, existing VidQA methods often function as black-box models, making it difficult to understand the reasoning process behind their predictions and leading to inconsistent compositional reasoning. For example, in Figure 1, HQGA [49] can answer the question “ _Is a phone the first object that the person is touching after taking a picture?_ ” as “ _Yes_ ”. However, HQGA can neither clearly identify the video clips that contain “ _touch a phone_ ” or “ _take a picture_ ” nor predict all the sub-questions correctly. Therefore, the lack of reasoning transparency can lead to poor compositional consistency, which reveals limited compositional reasoning ability, and further limits the accuracy of VidQA models, particularly on questions that involve temporal relations and multiple visual clues [12]. 

To tackle this issue, we introduce the Video Alignment and Answer Aggregation (VA[3] ) framework, which addresses these challenges by improving their compositional consistency and accuracy. This framework is _modelagnostic_ and can be applied to various VidQA methods, such as memory-based [9, 14], graph-based [6, 17, 24, 37, 38, 44, 46], and hierarchy-based [7, 18, 29, 39, 40, 49, 50] methods. In detail, our VA[3] framework includes two additional modules, the video aligner and answer aggregator. The video aligner hierarchically aligns the question with the video clips from the object-level, appearance-level to motion-level. The answer aggregator takes the questions from the same Question Decomposition Graph (QDG) as input and deduces their answers based on their videoquestion joint representation. To the enhance compositional consistency, we further explore a contrastive learning strategy on the edge type of QDG. Overall, the VA[3] framework improves both compositional consistency and accuracy of existing VidQA methods, provides a more transparent compositional reasoning process, and further leads to more in- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0002-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Question Decompose Graph (QDG)<br>Q: Was a phone the first thing they were putting down while  A: Yes,    Prediction: Yes<br>Equals      laughing at a picture?<br>Q: Does a phone exist?  A: Yes,    Prediction: Yes<br>First Q: Which is the first object that the person is putting down      while laughing at a picture?       A: Phone,    Prediction: Bottom<br>Q: What is the person putting down while laughing  A: Phone,    Prediction: Phone<br>Object      at a picture?<br>Q: Does a person exist while laughing at a picture? A: Yes,    Prediction: No<br>While<br>Q: Does a person exist?       A: Yes,    Prediction: Yes<br>Q: Is the person laughing at a picture?  A: Yes,    Prediction: No<br>Interaction<br>Q: Does a person exist? A: Yes,    Prediction: Yes<br>Q: Is the person laughing at something? A: Yes,    Prediction: Yes<br>Q: Does a picture exist? A: Yes,    Prediction: No<br>While Q: Is the person putting down something while      laughing at a picture? A: Yes,    Prediction: No<br>Q: Is the person putting down something? A: Yes,    Prediction: Yes<br>Q: Is the person laughing at a picture? A: Yes,    Prediction: No<br>Interaction<br>Q: Does a person exist? A: Yes,    Prediction: Yes<br>Q: Is the person laughing at something? A: Yes,    Prediction: Yes<br>Q: Does a picture exist? A: Yes,    Prediction: No<br>**----- End of picture text -----**<br>


Figure 1. The Question Decomposition Graph (QDG) of a question from AGQA-Decomp [12]. The predicted answer to each question is from HQGA [49]. Green ( _resp._ , Red) represents the predicted answer is right ( _resp._ , wrong) 

terpretable VidQA models in real-world applications. 

As for the evaluation metrics, AGQA-Decomp [12] proposes the compositional accuracy (CA), right for the wrong reasons (RWR), and delta (CA _−_ RWR) system to evaluate the compositional consistency of VidQA methods. However, these metrics only focus on reasoning failure based on the sub-questions correctness without considering the main question correctness, leading to asymmetric and unstable problems. To address this, we extend it to provide a symmetric and stable measurement for compositional consistency. In detail, our metrics include consistency precision (cP), consistency recall (cR), and consistency F1 (c-F1) along with their negative versions. These metrics can evaluate the compositional consistency of VidQA methods from a balanced viewpoint. More details are in Section 4. 

We conduct the experiments on the AGQA-Decomp dataset [12] to verify the effectiveness of our framework. This dataset decomposes the questions into the subquestions and the directed acyclic graphs, _i.e_ . the QDGs, making it applicable to evaluate the compositional consistency for VidQA methods. To validate the effectiveness and compositional consistency of our VA[3] framework, we conduct comprehensive experiments with three baseline methods: HME [9], HGA [24], and HQGA [49], which are the representations of the memory-based, graph-based and hierarchy-based methods, in three different settings: balanced, novel compositions, and more compositional steps. Moreover, we propose an automatic question decomposition pipeline for VidQA datasets with the help of large 

language models (LLMs) to generalize our framework to datasets that do not have QDGs ( _e.g_ ., MSVD [52] and NExT-QA [48]) to verify the applicability of our framework. Additionally, we visualize the aligned video clips and the variation of predicted answers while equipping video aligner and answer aggregator successively to the backbone model on QDG to verify the interpretability of our framework. Our contribution can be summarized as: 

- Dataset: We propose an automated question decomposition pipeline for any VidQA dataset to generate the QDGs and the sub-questions with the help of LLMs and further extend MSVD and NExT-QA dataset with it. 

- Framework: We propose a _model-agnostic_ VA[3] framework, which provides a more transparent compositional reasoning process and increases both the interpretability and the accuracy of existing VidQA models. 

- Metric: We extend the compositional consistency metrics as consistency precision (cP), consistency recall (cR) and consistency F1 (c-F1) along with their negative versions for a more balanced and comprehensive evaluation. 

- Experiments: Comprehensive experiments with three baselines on five benchmark settings of three datasets reveal that our framework significantly boosts these baselines in compositional consistency and accuracy. 

## **2. Related Work** 

## **2.1. Video Question-Answering** 

While the architecture of VidQA methods has undergone significant changes over the years, the essential components of these methods remain the same: video representation, question representation, and video-question aligned representation. For the video representation, appearance features [20] and motion features [51] were commonly used, then the object-level representation was introduced [25]. For the question representation, most existing works relied on word embeddings [41] with RNNs, while BERT [8] features became widely used in more recent works [34, 49]. In the early research, the video-question alignment was implemented using cross-modal attention [15, 32] or memory networks [9, 14], then graph reasoning [6, 17, 24, 37, 38] became popular. Recently, the natural hierarchy in video representation [18, 29, 40, 49, 50] received more attention. 

Despite these advancements, existing methods still face challenges in achieving satisfactory levels of compositional consistency [12]. To address these challenges, in this paper, we propose a _model-agnostic framework_ for compositional reasoning by combining the visual alignment and the answer aggregation to improve current VidQA methods. 

## **2.2. Compositional Reasoning in VidQA** 

The practice of decomposing a complex question into simpler questions has been observed in various tasks [4, 54]. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0003-00.png)


**----- Start of picture text -----**<br>
down while thing they were puttingWas a phone the firstpicture?laughing at a Model-agnostic VidQA Framework AggregatorAnswer c Grounding Score Transformer Transformer c Video Aligner<br>Motion Feature<br>Does a phone exist? Question Decompose Graph c Attention Pooling Transformer cTransformerc c QuestionFeature<br>What is the person c c c c c Concatenate<br>putting down while Appearance Feature Object Feature<br>laughing at a picture? Attention Pooling Appearance FeatureMotion Feature<br>Transformer Frame<br>Does a picture exist? Clip<br>Object Feature Video<br>VidQA<br>Video Aligner Hierarchical Backbone<br>**----- End of picture text -----**<br>


Figure 2. Our _model-agnostic_ Video Alignment and Answer Aggregation (VA[3] ) framework. ( _v, q_ ) is a video-question pair, where _qm_ denotes main-question and _qs_ 1 _, · · · , qsn_ denote the _n_ sub-questions derived from _qm_ . _vm_ and _{vs_ 1 _, · · · , vsn }_ denote the aligned videos according to corresponding questions. _**f** vm,qm_ and _{_ _**f** vs_ 1 _,qs_ 1 _, · · · ,_ _**f** vsn ,qsn }_ denote the video-question joint features. _Gqm_ is the question decomposition graph (QDG) associated with _qm_ , which is a direct acyclic graph describing the compositional relationship among questions. Moreover, _Gqm_ stores in which manner the questions are decomposited ( _i.e_ ., the operators in the decomposition program) as the attribute of edges. _a_ ˆ _m_ denotes the predicted answer for _qm_ . 

In VidQA, most of earlier efforts [16] broke down questions into modular programs that were defined in a neural modular network [42] to answer the question. AGQA [16] explored spatio-temporal scene graphs to represent the programs for VidQA. However, such a reasoning program cannot be directly used by existing VidQA methods. To address this issue, AGQA-Decomp [12] transferred each reasoning program into several sub-questions and QDG to evaluate the compositional consistency of existing VidQA methods. 

The Neural Modular Network (NMN)-based methods ( _e.g_ ., DSTN [42]) modularized the VidQA task into multiple modules ( _e.g._ , _FindObj_ , _TemporalFilter_ , _etc._ ) and generated the reasoning program through a modular policy. Although the NMN-based approaches provide perfect interpretability, there still exist three main challenges: 1) the basic modulars and logic rules have to be pre-defined, making any novel modulars and logic rules incompatible; 2) compared to conventional neural networks, training NMNs can be more challenging because the learning process optimizes the composition strategy other than the individual modules; 3) as the number of modules increases, the search space for optimal modules grows exponentially, which hinders its scalability to more complex tasks or larger datasets. 

## **2.3. Video Grounding** 

Video grounding [2, 13] seeks to identify the most relevant moment in a video based on language queries [31, 36, 45, 55], and has received growing attention from downstream video-language tasks [1, 33–35]. Previous works such as IGV [35] and EIGV [34] have focused on differentiating between causal and environment clips in VidQA through a simple grounding indicator and encouraging sensitivity to semantic changes in the causal scene, respectively. In contrast to these approaches, our framework hierarchically aligns the question with video clips from objectlevel, appearance-level, to motion-level to provide a more refined video context along with an answer aggregator. This 

approach enhances both the generalization and compositional reasoning abilities of existing VidQA methods, leading to more effective and accurate models. 

## **3. Our Method** 

As described in Section 1, current VidQA models suffer from insufficient compositional reasoning ability. Therefore, we propose our VA[3] framework, consisting a hierarchical video aligner and a QDG-based answer aggregator. 

## **3.1. Model-agnostic VidQA Framework** 

Our VidQA framework is illustrated in Figure 2. For each original question in dataset, it is decomposed (either oracularly or with our question decomposition pipeline) into several sub-questions. Formally, the original question (regarded as main question) _qm_ and its decomposed sub-questions _qs[m] i_[form][a][question][cluster][on][correspond-] ing QDG _Gqm_ . First, we introduce a hierarchical video aligner, which selects the question-related video clips by hierarchically interacting the video clips with the question among the object-level, appearance-level and motion-level. After that, the questions in the cluster and their corresponding video clips are send into a VidQA model, regarded as _F_ : ( _v, q_ ) _→_ _**f** v,q ∈_ R _[h]_ , where _h_ is the hidden dimension for joint feature space, to generate the joint feature _**f** v,q_ . The answer aggregator takes all the joint features from the questions cluster and associates each joint feature with regard to the question node in the QDG. By aggregating the information in joint features of each node through QDG, we adjust the question representations and predict the answers. 

## **3.2. Video Alignment** 

The structure of video aligner is shown in Figure 2. The video aligner takes video-question pair ( _v, q_ ) as input, and gives the video clips that are most relevant to the question. Former research on localizing video clips with ques- 

tions [34, 35] failed to use the natural hierarchy of video features, thus limited the representation ability of the aligner. 

In our video aligner, a video _v_ is represented within three-level features: object feature _**F** o ∈_ R _[n][c][×][n][f][ ×][n][o][×][h][v]_ , appearance feature _**F** a ∈_ R _[n][c][×][n][f][ ×][h][v]_ and motion feature _**F** m ∈_ R _[n][c][×][h][v]_ , where _nc_ , _nf_ and _no_ represents the number of clips per video, frames per clip and objects per frame respectively, and _hv_ is the hidden dimension for each feature vector. As these three-level features naturally follow a hierarchical relationship, we designed a hierarchical video aligner to capture them for a better alignment. 

Our hierarchical video aligner follows a bottom-up video-question interaction scheme. Starting from _**F** o_ , we aggregate its information among objects with the condition of question, concatenate it with corresponding _**F** a_ , and further aggregate it with other frames in control of question. Then, such cross-frame representation is concatenated with _**F** m_ , and interacted with question feature to generate the grounding score. During each aggregation, we fuse the video feature and question feature through a transformer layer, where the video feature is regarded as query while question feature serves as the key and value. Formally, the aggregation from object feature to appearance feature is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0004-03.png)


where _σni_ is the softmax function along _ni_ dimension, TF is the transformer encoder layer, _**W** o_ and _**b** o_ are trainable parameters, and [ _·||·_ ] denotes the concatenation operator. The updated appearance feature _**F** a[c]_[is][used][to][produce][aggre-] gated motion feature _**F** m[c]_[in][a][similar][manner.][Further,][we] use the _**F** m[c]_[to generate the binary indicator of relevant clip] for the video, which can be formulated as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0004-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0004-06.png)


where MLP is the multi-layer linear projection. 

Since the ground-truth of aligned video is not applicable in VidQA dataset, we exploit the contrastive learning [34] to guide this module. Formally, given a video-question pair ( _v, q_ ) in training data, the indicator specifies the relevant video clips _v_ ˆ _r_ and irrelevant video clips _v_ ˆ _c_ within _v_ . Thus, the anchor _**f** v_ ˆ _r,q_ , positive sample _**f** v_ ˆ _′,q_ , and negative sample _**f** v_ ˆ _c,q_ of the contrastive loss is presented as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0004-09.png)


where ˆ _v[′]_ is acquired by replacing ˆ _vc_ in _v_ with random sampled clips. Therefore, the contrastive loss is defined as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0004-11.png)


Moreover, the answer prediction can be formulated as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0004-13.png)


where _σ_ is softmax function and _**W** o_ 1 and _**b** o_ 1 are the trainable parameters. Therefore, the total loss of each videoquestion pair ( _v, q_ ) for video aligner can be formulated as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0004-15.png)


where CE refers to cross entropy and _a_ represents the ground-truth answer for the video-question pair ( _v, q_ ). 

## **3.3. Answer Aggregation** 

Existing VidQA methods predict the answers of different questions independently, which ignores the correlations among questions from the same cluster, leading to insufficient compositional consistency. Therefore, we introduce an answer aggregator to supplement the main-question with sub-questions and enhance the compositional consistency. Specifically, assume _{_ _**f** qi,vi|i ∈{s_ 1 _, · · · , sn, m}}_ is the video-question joint feature extracted by backbone VidQA model for all questions in QDG _Gqm_ = ( _Vqm, Eqm_ ). We explore the graph attention network (GAT) to aggregate the joint feature along the given QDG. Formally, given the _k_ - th layer of the GAT, the main question _qm_ and its subquestions _{qs_ 1 _, · · · , qsn}_ , the information aggregation for node associated with _qi_ is formulates as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0004-19.png)


where _i, j ∈{s_ 1 _, · · · , sn, m}_ , and _**a** k_ , _**W** s[k]_[and] _**[ W]** g[ k]_[are the] trainable parameters for the _k_ -th layer of GAT. Finally, the outputs of all layers is concatenated together and projected to predict the answer, which is formulated as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0004-21.png)


where _**W** o_ 2 and _**b** o_ 2 are trainable parameters. 

Moreover, to enhance the compositional consistency, we introduce an additional contrastive training scheme. As the type of relation ( _i.e._ , edge) between the questions provides crucial clues in question decompositing and compositional reasoning, we introduce a heuristic prior, where edges with the same type shall have similar representations, and the distance between different types of edges shall be relatively large. By leveraging this prior, we can raise the level of abstraction for more accurate and consistent answer reasoning. Formally, _{_ _**f**[e]_ = _**W** e_ [ _fq[a] i,vi[||][f] q[ a] j ,vj_[] +] _**[ b]**[e][|][e][∈][E][q] m[}]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0005-00.png)


**----- Start of picture text -----**<br>
AGQA<br>Dataset<br>Selection Prompt Decomposition Prompt<br>Candidate Rearrange the questions Suppose that you are an<br>Quesiton Set in the following set K-shot Examples expert in linguistic and logic.<br>Class1:object-relation Question 1-1 according to the similaritywith <Question>. Decomposition ... Decompose the question<Question>.<br>Question 1-2 Graphs<br>......<br>The most related Decomposed<br>Class: object-action Question 2-1Question 2-2 LLM K questions inAGQA LLM Subquestiions &QDG<br>......<br>......<br>**----- End of picture text -----**<br>


Figure 3. The automatic question decomposition pipeline. The question to be decomposed is denoted as _<_ Question _>_ . 

denotes the set of node relation representation for edges in graph _Gqm_ , where _**W** e_ and _**b** e_ are the trainable parameters. Moreover, we use _te ∈ T_ to denote the class of edge _e_ , where _T_ is the set of edge types. For _t ∈ T_ , we use _t[c]_ = _T/{t}_ to represent the complementary set of _t_ , and use _et_ to denote a random edge sampled from all edges with type _t_ . Thus, the triplet loss for main question _qm_ is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0005-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0005-04.png)


where _d_ ( _·, ·_ ) is Eular distance and _m_ is margin. Thus, the answer aggregation loss for question cluster _Q_ = _{qm, qs_ 1 _, · · · , qsn}_ is formulated as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0005-06.png)


where _D_ = _{_ ( _v, q_ ) _|q ∈ Q}_ . Both _Lag_ and _Lal_ are consequently applied during training. 

## **3.4. Automatic Question Decomposition Pipeline** 

For some VidQA dataset ( _e.g_ ., MSVD and NExT-QA), the QDGs are not applicable as they only provide the main questions. To address this issue, we explore an automatic question decomposition pipeline for VidQA data using the knowledge in LLMs. Since directly asking the LLM to decompose questions may result in poor results, and random examples could cause unstable quality as the chosen examples may have different compositional structure with the queried question, we proposed the decomposition pipeline as shown in Figure 3. Firstly, we construct a candidate example set based on the AGQA-Decomp dataset manually, in which each subset consists of a few main questions chosen with a main question type in AGQA-Decomp dataset to cover as many types of main questions as we can. Then, we construct a selection prompt which ask the LLM to select the most similar K question, and they form K-shot examples with their QDGs. Finally, these K-shot examples are provided to LLM with a decomposition prompt asking the LLM to decompose the target question, resulting in the decomposed sub-questions and corresponding QDGs with better quality. More details and explanations of our pipeline and prompts are in the supplementary material. 

## **4. Metrics** 

Compositional consistency measures whether a method can provide the correct answer for the right reason. AGQADecomp [12] propose compositional accuracy (CA), right for the wrong reasons (RWR), and Delta (CA-RWA), which offer some insight on it. However, as we are to illustrate in Section 4.1, they cannot fully reveal the reasoning capabilities, leading to asymmetric and unstable problem. To address this issue, we extend them with compositional precision (cP), recall (cR), and F1 (c-F1), along with their negative versions, providing a more comprehensive assessment of reasoning consistency. We name _qj_ as parent question, and _qi, qk_ as children question of _qj_ for QDG edges like _qi ← qj → qk_ . Note that we only consider the **1-st** order parent-children relation, and the intermediate sub-questions can be both parent or children regarding the viewpoint. 

## **4.1. Another View of Existing Metrics** 

CA and RWR are designed to evaluate the accuracy of the parent questions, conditional on whether all their children questions are correct. Let N[+] _−_[denote][the][number][of][cor-] rectly answered main-question with any sub-question incorrect, while N _[−]_ +[denotes the number of falsely answered main] question with all sub-questions correct. N _[−] −_[and N][+] +[is sim-] ilarly defined. Thus, CA and RWR is formulated as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0005-14.png)


and Delta = RWR _−_ CA. Therefore, both CA and 1 _−_ RWR can be viewed as **“precisions”** , as the conditions only inspect the correctness of the children questions and missed out the correctness of the parent questions, leading to asymmetric and unstable problems illustrated in Table 1. 

For row 1 to row 4, the corresponding model shall have the same compositional consistency, because in all 210 parent questions, 110 of them can be answered for the right reasoning and the opposite. However, the CA, RWR and Delta varies significantly among these models, which indicate that the CA-RWR-Delta metrics cannot treat N[+] +[, N] _−[−]_[,] N _[−]_ +[and][N][+] _−_[asymmetrically][and][cannot][correctly][identify] the compositional consistency. Moreover, such failure also lead to instability when facing imbalanced child question accuracy distribution, as shown in row 5 to row 7. These models have similar compositional consistency, but the CA, RWR and Delta may vary to the extreme opposite value. 

## **4.2. Our Metrics** 

As described in Section 4.1, CA and 1 _−_ RWR can both be viewed as **“precisions”** , thus we denote them as **consistency precision (cP)** and **negative consistency precision (NcP)** respectively to simplify the following formulation. For a more comprehensive view on compositional consistency, we are to introduce the corresponding “recalls”. 

||Data Count<br>_N_ +<br>+<br>_N −_<br>+<br>_N_ +<br>_−_<br>_N −_<br>_−_|Existing Metrics<br>CA<br>RWR<br>Delta<br>Acc.|Our Metrics<br>c-F1<br>Nc-F1|
|---|---|---|---|
|1<br>2<br>3<br>4|100<br>100<br>0<br>10<br>10<br>100<br>0<br>100<br>100<br>0<br>100<br>10<br>10<br>0<br>100<br>100|50.00<br>0.00<br>-50.00<br>47.61<br>9.09<br>0.00<br>-9.09<br>4.76<br>100.00<br>90.91<br>-9.09<br>95.23<br>100.00<br>50.00<br>-50.00<br>52.38|66.67<br>16.67<br>16.67<br>66.67<br>66.67<br>16.67<br>16.67<br>66.67|
|5<br>6<br>7|99<br>100<br>1<br>0<br>100<br>99<br>0<br>1<br>99<br>99<br>1<br>1|49.75<br>100.00<br>-50.25<br>50.00<br>50.25<br>0.00<br>50.25<br>50.00<br>50.00<br>50.00<br>0.00<br>50.00|66.22<br>0.00<br>66.88<br>0.99<br>66.44<br>0.98|



Table 1. The conterexamples. Acc. is parent question accuracy. 

**Definition 1 (Consistency Recalls)** _Given a VidQA model M , the consistency recall (cR) and negative consistency recall (NcR) is defined as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0006-03.png)


These metrics condition on the correctness of main questions. Clearly, neither cP and cR can represent the compositional consistency solely, since they only take part of the condition into consideration. To combine both perspectives, we introduce their weighted harmonic mean, _i.e_ ., consistency F-scores for a robust and symmetric representation. 

**Definition 2 (Consistency F-Score)** _Given a VidQA model M , the consistency F-score (c-Fβ) and negative consistency F-score (Nc-Fβ) is defined as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0006-06.png)


In such definition, _β_ is a hyper-parameter to balance cP and cR. To measure the two kinds of error in a equal weight, we set _β_ = 1, and thus use c-F1 to measure the compositional consistency of models. Such metric considers the compositional consistency in a symmetric manner, raising a comprehensive evaluation on model’s ability. As shown in Table 1, our c-F1 and Nc-F1 metrics raises more reasonable evaluations (row 1 to row 4), and is also more stable facing extreme cases (row 5 to row 7). Note that although c-F1 and Nc-F1 provides a balanced view of compositional consistency, we still need to the accuracy, cP, and cR for a detailed analysis regarding prediction ability and compositional bias. **More analysis and comparison between our metrics and original ones are in the supplementary.** 

## **5. Experiments** 

## **5.1. Experiment Setting** 

**Dataset** As described in Section 1, we conduct our experiment on AGQA-Decomp benchmark, which extends AGQA 2.0 by decomposing each question into several subquestions with a QDG, and evaluates the compositional 

reasoning ability by supplying extensive challenging complex questions with their decomposed sub-questions and answers. Moreover, we test the improvement on MSVD and NExT-QA dataset to verify the applicability of our VA[3] framework and question decomposition pipeline. 

**Baselines and Metrics** We conduct experiment on all three categories of VidQA mothods, _i.e_ ., memory-based, graph-based and hierarchy-based methods. Specifically, we choose HME [9], HGA [24] and HQGA [49] as the baseline methods from each category respectively. For the evaluation metrics, we measure the open-ended, binary, and overall accuracy for main questions and sub-questions. Moreover, to illustrate the improvement in terms of compositional consistency, we evaluate the cR, cP, c-F1, NcR, NcP and Nc-F1. More detailed settings are in supplementary material. 

## **5.2. Main Results** 

The result of our framework with various baselines is shown in Table 2. Compared the 1-st to 3-rd row with the 4-th to 6-th row correspondingly, our framework outperforms all baseline models, including memory-, graph- and hierarchybased models, significantly in terms of both accuracy and compositional consistency. For accuracy, the overall main question accuracy improves 1.23% to 2.71%, while the subquestion accuracy raises 3.16% to 3.29%. The accuracy improvement on sub-questions are usually more than that on main questions. For video aligner, it is more hard to align the corresponding video clips for main questions, besides, for answer aggregator, it is also more challenge to aggregate all the sub-questions to deduce the main question, leading to such improvement gap. Moreover, the accuracy improvements on binary and open-ended questions are not equal. The reasons include the the following two aspects: 1) the video aligner helps open-ended questions more since they are more sensitive to irrelevant clips as they have to choose answer from a much larger candidate set than binary questions, and may be mislead by the actions in irrelevant clips more easily; 2) the open-ended questions provide and receive more severe information in answer aggregation, making the answer aggregator contribute more on them. 

Moreover, the compositional consistency also raises significantly compared to the baseline models. Specifically, the c-F1 significantly improves 2.97% to 3.54%, while the Nc-F1 raises 0.11% to 0.64%. The c-F1 indicates how much the model answers **correctly** with **correct** inference, therefore, the c-F1 is the most important overall measurement for the reasoning ability of models. The significant improvement in terms of c-F1 further indicates that our framework does help the VidQA models reasoning correctly and consistently. And the Nc-F1, which measures if the model is still consistent even when the answer is incorrect, shows that our framework also slightly helps improve the overall consistency on the incorrect main questions. Furthermore, 

||Main Accuracy<br>Open<br>Binary<br>**All**|Sub Accuracy<br>Open<br>Binary<br>**All**|Compositational Consistency<br>cP<br>cR<br>**c-F1**<br>NcP<br>NcR<br>**Nc-F1**|
|---|---|---|---|
|HME<br>HGA<br>HQGA<br>VA3(HME)<br>VA3(HGA)<br>VA3(HQGA)|36.29<br>51.41<br>**41.59**<br>41.18<br>56.62<br>**46.61**<br>41.05<br>50.40<br>**44.34**<br>39.91+3_._62 52.26+0_._85 **44.30**+2_._71 <br>43.04+1_._86 56.82+0_._20 **47.88**+1_._27 <br> 42.35+1_._30 51.53+1_._13 **45.57**+1_._23|29.68<br>71.73<br>**56.59**<br>36.03<br>73.11<br>**59.75**<br>33.22<br>70.03<br>**56.77**<br> 35.23+5_._55 73.56+1_._83 **59.75**+3_._16 <br> 42.18+6_._15 74.69+1_._58 **62.98**+3_._23 <br> 35.28+2_._06 74.01+3_._98 **60.06**+3_._29|53.06<br>46.02<br>**49.29**<br>56.57<br>63.33<br>**59.76**<br>64.02<br>57.85<br>**60.78**<br>57.35<br>63.55<br>**60.29**<br>54.49<br>38.55<br>**45.16**<br>53.89<br>69.04<br>**60.53**<br> 57.78+4_._72 48.66+2_._64 **52.83**+3_._54 55.81_−_0_._76 64.58+1_._25 **59.87**+0_._11<br> 67.52+3_._50 60.38+2_._53 **63.75**+2_._97 57.47+0_._12 64.83+1_._28 **60.93**+0_._64<br> 56.02+1_._53 42.50+3_._95 **48.33**+3_._17 55.26+1_._37 68.04_−_1_._00 **60.99**+0_._46|



Table 2. The comparison with baseline methods on AGQA-Decomp [12]. The overall measurements are highlighted in bold, and the improvements of our framework are highlighted as superscript. 

||Novel Comp. Setting<br>Accuracy<br>c-F1<br>Nc-F1|More Comp. Step Setting<br>Accuracy<br>c-F1<br>Nc-F1|
|---|---|---|
|HME<br>HGA<br>HQGA<br>VA3(HME)<br>VA3(HGA)<br>VA3(HQGA)|31.54<br>35.88<br>68.94<br>33.40<br>35.44<br>65.18<br>34.21<br>38.91<br>67.96<br>33.45+1_._91 38.26+2_._38 69.08+0_._14 <br>35.27+1_._87 40.47+5_._03 65.33+0_._15 <br> 36.33+2_._12 40.76+1_._85 68.20+0_._24|44.28<br>48.32<br>63.71<br>47.26<br>48.42<br>63.45<br>46.64<br>50.65<br>62.97<br> 46.07+1_._79 49.87+1_._55 64.01+0_._30<br> 48.38+1_._12 51.08+2_._66 63.58+0_._13<br> 47.91+1_._27 51.75+1_._10 63.26+0_._29|



Table 3. The comparison with baseline methods on the AGQADecomp [12] _novel composition_ setting and _more composition step_ setting [16]. Comp. is the abbreviation for compositional. The improvements of our framework are highlighted as superscript. 

||Main Accuracy<br>Open Binary<br>All|Sub Accuracy|Consistency|
|---|---|---|---|
|||Open Binary<br>All|c-F1 Nc-F1|
|HME|36.29 51.41 41.59|29.68 71.73 56.59 49.29 59.76||
|VA. EIGV [34]<br>Our Aligner|38.84 51.48 43.29 <br> 39.49 51.50 43.72|31.77 72.42 57.77 48.05 59.31<br> 32.27 72.81 58.20 49.21 59.71||
|AA. +AA.<br>+AA. +_Lqm_<br>_c_|38.21 51.35 42.83 <br>38.81 52.05 43.46|33.78 72.33 58.44 52.19 59.73<br> 34.12 72.91 58.93 53.49 60.05||
|VA3(HME)|39.91 52.26 44.30|35.23 73.56 59.75 52.83 59.87||



Table 4. The ablation study on our Video Aligner and Answer Aggregator. VA. is video aligner and AA. is answer aggregator. 

we can also find that the the improvement on c-F1 are naturally more significant that on Nc-F1 as our constraint on answer aggregation mainly focus on correctly deducing the main question based on the correct sub-questions. 

## **5.3. Generalization Ability** 

We further test the improvement of our framework when generalizing to new situations on two extra settings, _i.e_ . _novel composition_ and _more composition step_ [16]. The _Novel composition_ setting tests if models can generalize to unseen composition types, and the _more composition step_ setting tests the generalization ability when facing more complex questions than training. The results are in Table 3. 

When facing novel composition setting, the clear accuracy drop compared to Table 2 indicates generalizing to novel composition setting is much harder for VidQA models than the standard setting. However, our framework is still capable of significantly boosting baseline methods under this challenging setting. The main question accuracy raises 1.87% to 2.12%, while the c-F1 improves 1.85% to 5.03% and the Nc-F1 improves 0.14% to 0.24%, on different baselines, implying that our framework provides better generalization ability on unseen composition types, in terms of both accuracy and compositional consistency. 

For the more composition step setting, our framework still significantly improves the baseline methods on both the accuracy and compositional consistency, indicating our framework is effective when generalizing to more complex questions. In detail, there is a 1.12% to 1.76% accuracy 

improvement for main questions, while the c-F1 improves 1.10% to 2.66% while the Nc-F1 raises 0.13% to 0.30%. 

## **5.4. Ablation Study** 

To measure the effectiveness and necessity of our modules, we conduct ablation studies in this section. To clearly reveal the contribution of each module and loss, we choose HME [9] as the baseline method in this section, since the improvement over HME is the largest among all three baselines. For the video aligner, we test its improvement over the EIGV [34] aligner brought by the hierarchy structure. For the answer aggregator, we test how much it boosts the original model even without our contrastive loss _L[q] c[m]_[, then] measure the improvement of applying _L[q] c[m]_ on answer aggregator. The results are summarized in Table 4. 

By comparing row 3 with rows 1 and 2, we can conclude that the video aligner substantially improves the accuracy of both main questions and sub-questions, but helps little on compositional consistency. This is reasonable since video aligners do not exploit the relations between main questions and sub-questions, thus cannot improve the compositional consistency among them. Moreover, the comparison between row 3 and row 2 shows that our aligner outperforms the video grounding module in EIGV due to its hierarchical structure, which could effectively extract information from different level of video feature and align them with the question. As for the answer aggregation module, by comparing row 4 with row 1, we can infer that answer aggregator, even without the contrastive loss _L[q] c[m]_[, can signifi-] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering_images/cvpr_2024_Align_and_Aggregate_Compositional_Reasoning_with_Video_Alignment_and_Answer_Aggregation_for_Video_Question-Answering.pdf-0008-00.png)


**----- Start of picture text -----**<br>
In the video, did they snuggle both a<br>pillow and the object they were taking?<br>Question Decompose Graph HME HME + Video Aligner VA (HME)<br>Is the person snuggling the object<br>Yes Bed Yes Blanket Yes Blanket that they are taking?Is the person snuggling a pillow?<br>Yes No No<br>No Yes Yes Yes Yes Yes Yes Yes No What is the person taking?<br>No No Yes Is the person snuggling something?<br>Yes Yes Yes Does a pillow exist?<br>Is the person taking something?<br>Visualization of Video Aligner Visualization of Answers and Compositional Consistency<br>Query Video Does a person exist?<br>**----- End of picture text -----**<br>


Figure 4. Quantitive results of Video Aligner and the visualization of improvements on accuracy and compositional consistency brought by our modules. Best viewed in color and zoom in. **More visualizations and explanations are in the supplementary material.** 

cantly improve the accuracy and compositional consistency, as the information exchange along main-sub questions relation helps both correct answering and ensuring their consistency. Further, as the comparison between row 4 and row 5 implies, the contrastive loss _L[q] c[m]_ further improves the accuracy and compositional consistency. Finally, the improvement of row 6 over row 3 and 5 indicates that our combination of video aligner and answer aggregator is mutually beneficial on both accuracy and compositional consistency, proving the effectiveness of our framework. 

## **5.5. Quailititave Study** 

In Figure 4, we firstly visualize the result of video aligner. As the left part of Figure 4 shows, our video aligner successfully aligns the related video clips (denoted by the dotted boxes) with the corresponding questions. Therefore, the VidQA backbones can use more accurate information to improve the accuracy of both main questions and subquestions. Moreover, we also visualize the predicted answers of HME, HME with video aligner, and VA[3] (HME) respectively. The original model failed to answer the main question correctly due to several factual errors in subquestions and the potential reasoning failure. With the help of video aligner, we may reduce the irrelevant noise by only processing the most correlated clips, thus eliminating several factual errors ( _i.e_ ., _qs_ 1 _, qs_ 3 and _qs_ 6), but may not help on the main question since the whole video is fatal in generating its answer. Moreover, the video aligner may not correct the compositional reasoning failure as it does not use interquestion relation information. However, with the help of the answer aggregator, we can correct the reasoning failure by aggregating the information from sub-questions ( _e.g_ ., correct the answer of _qs_ 2 by aggregating video-question joint features of _qs_ 4 _, qs_ 5 and _qs_ 6), and deduce the correct answer of main questions with higher compositional consistency. 

## **5.6. Applicability** 

To verify that our framework is generally applicable, we further apply our framework on MSVD [52] and NExTQA [48] datasets extended by our automatic question de- 

||HME|HGA|HQGA|VA3(HME)|VA3(HGA)|VA3(HQGA)|
|---|---|---|---|---|---|---|
|MSVD|33.75|36.71|41.23|**38.51**+4_._76|**41.24**+4_._53|**44.46**+3_._23|
|NExT-QA|48.72|50.04|51.65|**53.23**+4_._51|**54.11**+4_._07|**55.23**+3_._58|



Table 5. The comparison with baseline methods on MSVD and NExT-QA datasets. Improvements are highlighted as superscripts. 

composition pipeline. The results are summarized in Table 5. By comparing the 1st to 3rd columns with 4th to 6th columns in the table, we could find that our framework, with our automatic decomposition pipeline, significantly boosts the performance of baselines on both MSVD and NExT-QA dataset. Specifically, the overall accuracy improves 3.23% to 4.76% on MSVD, while improves 3.58% to 4.51% on NExT-QA, which indicates that with the help of our question decomposition pipeline, our framework can still raise the performance significantly even on datasets which originally do not have QDGs, further implying the applicability of our framework in real world scenarios. 

## **6. Conclusion** 

In this work, we have focused on the VidQA from interpretability and proposed a model-agnostic align-andaggregate framework for VidQA. It firstly aligns the video representation towards both main question and subquestions, then aggregates the video-question joint representation through the QDG. Further, we have revisited the compositional consistency metrics and have proposed more comprehensive c-F scores. Extensive experiments on various VidQA models have revealed that our framework improves both compositional consistency and accuracy significantly, leading to more interpretable VidQA models. 

## **Acknowledgments** 

The work was supported by the National Natural Science Foundation of China (Grant No. 62076162), the Shanghai Municipal Science and Technology Major/Key Project, China (Grant No. 2021SHZDZX0102). 

## **References** 

- [1] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko S¨underhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In _CVPR_ , pages 3674–3683, 2018. 3 

- [2] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In _ICCV_ , pages 5803– 5812, 2017. 3 

- [3] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In _NeurIPS_ , 2020. 

- [4] Qingxing Cao, Xiaodan Liang, Bailing Li, Guanbin Li, and Liang Lin. Visual question reasoning on general dependency tree. In _CVPR_ , pages 7249–7257, 2018. 2 

- [5] David L Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In _ACL_ , pages 190–200, 2011. 

- [6] Anoop Cherian, Chiori Hori, Tim K. Marks, and Jonathan Le Roux. (2.5+1)D spatio-temporal scene graphs for video question answering. In _AAAI_ , pages 444–453, 2022. 1, 2 

- [7] Long Hoang Dang, Thao Minh Le, Vuong Le, and Truyen Tran. Hierarchical object-oriented spatio-temporal reasoning for video question answering. In _IJCAI_ , pages 636–642, 2021. 1 

- [8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In _NAACL-HLT_ , pages 4171–4186, 2019. 2 

- [9] Chenyou Fan, Xiaofan Zhang, Shu Zhang, Wensheng Wang, Chi Zhang, and Heng Huang. Heterogeneous memory enhanced multimodal attention model for video question answering. In _CVPR_ , pages 1999–2007, 2019. 1, 2, 6, 7 

- [10] Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu. VIOLET: Endto-End Video-Language Transformers with Masked Visualtoken Modeling. In _arXiv:2111.1268_ , 2021. 

- [11] Tsu-Jui Fu*, Linjie Li*, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu. An Empirical Study of End-to-End Video-Language Transformers with Masked Visual Modeling. In _CVPR_ , 2023. 

- [12] Mona Gandhi, Mustafa Omer Gul, Eva Prakash, Madeleine Grunde-McLaughlin, Ranjay Krishna, and Maneesh Agrawala. Measuring compositional consistency for video question answering. In _CVPR_ , pages 5046–5055, 2022. 1, 2, 3, 5, 7 

- [13] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In _ICCV_ , pages 5267–5275, 2017. 3 

- [14] Jiyang Gao, Runzhou Ge, Kan Chen, and Ram Nevatia. Motion-appearance co-memory networks for video question answering. In _CVPR_ , pages 6576–6585, 2018. 1, 2 

- [15] Lianli Gao, Pengpeng Zeng, Jingkuan Song, Yuan-Fang Li, Wu Liu, Tao Mei, and Heng Tao Shen. Structured twostream attention network for video question answering. In _AAAI_ , pages 6391–6398, 2019. 2 

- [16] Madeleine Grunde-McLaughlin, Ranjay Krishna, and Maneesh Agrawala. Agqa: A benchmark for compositional spatio-temporal reasoning. In _CVPR_ , pages 11287–11297, 2021. 3, 7 

- [17] Mao Gu, Zhou Zhao, Weike Jin, Richang Hong, and Fei Wu. Graph-based multi-interaction network for video question answering. _IEEE Trans. Image Process._ , 30:2758–2770, 2021. 1, 2 

- [18] Zhicheng Guo, Jiaxuan Zhao, Licheng Jiao, Xu Liu, and Lingling Li. Multi-scale progressive attention network for video question answering. In _ACL_ , pages 973–978, 2021. 1, 2 

- [19] Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh. Can spatiotemporal 3d cnns retrace the history of 2d cnns and imagenet? In _CVPR_ , pages 6546–6555, 2018. 

- [20] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _CVPR_ , pages 770–778, 2016. 2 

- [21] Deng Huang, Peihao Chen, Runhao Zeng, Qing Du, Mingkui Tan, and Chuang Gan. Location-aware graph convolutional networks for video question answering. In _AAAI_ , pages 11021–11028, 2020. 1 

- [22] Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. TGIF-QA: Toward spatio-temporal reasoning in visual question answering. In _CVPR_ , pages 1359–1367, 2017. 1 

- [23] Jianwen Jiang, Ziqiang Chen, Haojie Lin, Xibin Zhao, and Yue Gao. Divide and Conquer: Question-guided spatiotemporal contextual attention for video question answering. In _AAAI_ , pages 11101–11108, 2020. 1 

- [24] Pin Jiang and Yahong Han. Reasoning with heterogeneous graph alignment for video question answering. In _AAAI_ , pages 11109–11116, 2020. 1, 2, 6 

- [25] Weike Jin, Zhou Zhao, Mao Gu, Jun Yu, Jun Xiao, and Yueting Zhuang. Multi-interaction network with object relation for video question answering. In _ACM MM_ , pages 1193– 1201, 2019. 2 

- [26] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 

- [27] Dohwan Ko, Ji Soo Lee, Miso Choi, Jaewon Chu, Jihwan Park, and Hyunwoo J Kim. Open-vocabulary video question answering: A new benchmark for evaluating the generalizability of video question answering models. In _ICCV_ , 2023. 

- [28] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In _NeurIPS_ , 2023. 

- [29] Thao Minh Le, Vuong Le, Svetha Venkatesh, and Truyen Tran. Hierarchical conditional relation networks for video question answering. In _CVPR_ , pages 9969–9978, 2020. 1, 2 

- [30] Jiangtong Li, Li Niu, and Liqing Zhang. From Representation to Reasoning: Towards both evidence and commonsense reasoning for video question-answering. In _CVPR_ , pages 21241–21250, 2022. 1 

- [31] Juncheng Li, Junlin Xie, Long Qian, Linchao Zhu, Siliang Tang, Fei Wu, Yi Yang, Yueting Zhuang, and Xin Eric Wang. Compositional temporal grounding with structured variational cross-graph correspondence learning. In _CVPR_ , pages 3032–3041, 2022. 3 

- [32] Xiangpeng Li, Jingkuan Song, Lianli Gao, Xianglong Liu, Wenbing Huang, Xiangnan He, and Chuang Gan. Beyond RNNs: Positional self-attention with co-attention for video question answering. In _AAAI_ , pages 8658–8665, 2019. 2 

- [33] Yicong Li, Xun Yang, Xindi Shang, and Tat-Seng Chua. Interventional video relation detection. In _ACM MM_ , 2021. 3 

- [34] Yicong Li, Xiang Wang, Junbin Xiao, and Tat-Seng Chua. Equivariant and invariant grounding for video question answering. In _ACM MM_ , pages 4714–4722, 2022. 1, 2, 3, 4, 7 

- [35] Yicong Li, Xiang Wang, Junbin Xiao, Wei Ji, and Tat-Seng Chua. Invariant grounding for video question answering. In _CVPR_ , pages 2918–2927, 2022. 1, 3, 4 

- [36] Daizong Liu, Xiaoye Qu, Jianfeng Dong, Pan Zhou, Yu Cheng, Wei Wei, Zichuan Xu, and Yulai Xie. Context-aware biaffine localizing network for temporal sentence grounding. In _CVPR_ , pages 11235–11244, 2021. 3 

- [37] Fei Liu, Jing Liu, Weining Wang, and Hanqing Lu. HAIR: hierarchical visual-semantic relational reasoning for video question answering. In _ICCV_ , pages 1678–1687, 2021. 1, 2 

- [38] Jungin Park, Jiyoung Lee, and Kwanghoon Sohn. Bridge To Answer: Structure-aware graph interaction network for video question answering. In _CVPR_ , pages 15526–15535, 2021. 1, 2 

- [39] Liang Peng, Shuangji Yang, Yi Bin, and Guoqing Wang. Progressive graph attention network for video question answering. In _ACM MM_ , pages 2871–2879, 2021. 1 

- [40] Min Peng, Chongyang Wang, Yuan Gao, Yu Shi, and XiangDong Zhou. Multilevel hierarchical network with multiscale sampling for video question answering. In _IJCAI_ , pages 1276–1282, 2022. 1, 2 

moment localization via language query. In _CVPR_ , pages 7026–7035, 2021. 3 

   - [46] Jianyu Wang, Bing-Kun Bao, and Changsheng Xu. Dualvgr: A dual-visual graph reasoning unit for video question answering. _IEEE Trans. Multim._ , 24:3369–3380, 2022. 1 

   - [47] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In _NeurIPS_ , 2022. 

   - [48] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. NExT-QA: Next phase of question-answering to explaining temporal actions. In _CVPR_ , pages 9777–9786, 2021. 1, 2, 8 

   - [49] Junbin Xiao, Angela Yao, Zhiyuan Liu, Yicong Li, Wei Ji, and Tat-Seng Chua. Video as conditional graph hierarchy for multi-granular question answering. In _AAAI_ , pages 2804– 2812, 2022. 1, 2, 6 

   - [50] Junbin Xiao, Pan Zhou, Tat-Seng Chua, and Shuicheng Yan. Video graph transformer for video question answering. In _ECCV_ , 2022. 1, 2 

   - [51] Saining Xie, Ross B. Girshick, Piotr Doll´ar, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In _CVPR_ , pages 5987–5995, 2017. 2 

   - [52] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In _ACM MM_ , pages 1645–1653, 2017. 2, 8 

   - [53] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. MSR-VTT: A large video description dataset for bridging video and language. In _CVPR_ , pages 5288–5296, 2016. 1 

   - [54] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In _EMNLP_ , pages 2369– 2380, 2018. 2 

   - [55] Zhu Zhang, Zhijie Lin, Zhou Zhao, Jieming Zhu, and Xiuqiang He. Regularized two-branch proposal networks for weakly-supervised moment retrieval in videos. In _ACM MM_ , pages 4098–4106, 2020. 3 

- [41] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. GloVe: Global vectors for word representation. In _EMNLP_ , pages 1532–1543, 2014. 2 

- [42] Zi Qian, Xin Wang, Xuguang Duan, Hong Chen, and Wenwu Zhu. Dynamic spatio-temporal modular network for video question answering. In _ACM MM_ , pages 4466–4477, 2022. 3 

- [43] Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun. Faster R-CNN: towards real-time object detection with region proposal networks. In _NeurIPS_ , pages 91–99, 2015. 

- [44] Ahjeong Seo, Gi-Cheon Kang, Joonhan Park, and ByoungTak Zhang. Attend what you need: Motion-appearance synergistic networks for video question answering. In _ACL_ , pages 6167–6177, 2021. 1 

- [45] Hao Wang, Zheng-Jun Zha, Liang Li, Dong Liu, and Jiebo Luo. Structured multi-level interaction network for video 

