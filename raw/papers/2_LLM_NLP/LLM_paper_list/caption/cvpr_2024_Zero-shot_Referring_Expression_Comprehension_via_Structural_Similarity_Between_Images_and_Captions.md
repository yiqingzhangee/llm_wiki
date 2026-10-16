# **Zero-shot Referring Expression Comprehension via Structural Similarity Between Images and Captions** 

Zeyu Han[1] , Fangrui Zhu[1] , Qianru Lao[2] , Huaizu Jiang[1] 1Northeastern University 2Harvard University 

_{_ han.zeyu,zhu.fang,h.jiang _}_ @northeastern.edu, estherbear17@gmail.com 

## **Abstract** 

_**Zero-shot referring expression comprehension** aims at localizing bounding boxes in an image corresponding to provided textual prompts, which requires: (i) a fine-grained disentanglement of complex visual scene and textual context, and (ii) a capacity to understand relationships among disentangled entities. Unfortunately, existing large visionlanguage alignment (VLA) models,_ e.g _., CLIP, struggle with both aspects so cannot be directly used for this task. To mitigate this gap, we leverage large foundation models to disentangle both images and texts into triplets in the format of_ ( _subject, predicate, object_ ) _. After that, grounding is accomplished by calculating the structural similarity matrix between visual and textual triplets with a VLA model, and subsequently propagate it to an instancelevel similarity matrix. Furthermore, to equip VLA models with the ability of relationship understanding, we design a triplet-matching objective to fine-tune the VLA models on a collection of curated dataset containing abundant entity relationships. Experiments demonstrate that our visual grounding performance increase of up to 19.5% over the SOTA zero-shot model on RefCOCO/+/g. On the more challenging Who’s Waldo dataset, our zero-shot approach achieves comparable accuracy to the fully supervised model. Code is available at https://github. com/Show-han/Zeroshot_REC._ 

## **1. Introduction** 

Visual grounding is a fundamental task across computer vision and natural language processing, where the goal is to find the correspondences between image content and textual descriptions. It has broad applications in image captioning [20, 55], visual question answering [45, 68], vision-language navigation [13], etc. Collecting detailed grounding annotations to train specialist models, however, is cumbersome. Therefore, **zero-shot visual grounding** [32, 40, 56] is an attractive alternative. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions_images/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions.pdf-0001-08.png)


**----- Start of picture text -----**<br>
Cat touching a paper roll<br>A man watches as a man shoots over a man<br>**----- End of picture text -----**<br>


Figure 1. **Illustration of how we disambiguate visual entities based on their interactions with other entities.** The same entity or relationships in the image and caption are in the same color. 

As a visual grounding task, the essence of referring expression comprehension (REC) is the alignment of text queries with corresponding image regions. To achieve this goal, it is critical for a grounding model to understand the relationships of entities[30], both within and cross different modalities (visual _vs._ textual), when identifying referred entities within an image. As shown in the upper part of Fig. 1, the touching relationship is the key to resolve the ambiguity of identifying the correct cat. Similarly, the lower part of Fig. 1 illustrates a more complex situation where multiple entities are engaged in various interactions. Both scenarios highlight the importance of _relationship understanding within both the image and caption_ , where entities are not merely isolated elements but interact dynamically with others in the scene. In a zero-shot learning context, the task of understanding these relationships can be more challenging, as the model lacks exposure to specific training instances that could aid in interpretation. 

Recent advances in zero-shot REC [32, 56, 67] have been largely driven by the integration of large-scale visionlanguage aligned (VLA) models such as CLIP [49] and FLAVA [54], which serve as bridges connecting text and image domain. These approaches, however, fall short in relationship understanding. On the one hand, in the textual domain, existing approaches adopt hand-crafted lan- 

guage parsers [56, 67] to decompose the input caption into a set of phrases, which are fragile and do not generalize well to long, complex captions in real-world applications, as shown in the bottom of Fig. 1. On the other hand, the visual relationship understanding capability of VLA models is inherently not good enough. Recent studies have revealed that VLA models behave like “bags-of-words” [69], and demonstrated that they fail to perform beyond chance level at simple tasks requiring compositional understanding [7, 8, 14, 59, 69]. Some effort have been dedicated to mitigate this issue by generating hard negative prompts through word replacement [8, 22, 69], caption augmentation [7, 8], and feature augmentation [24]. These rule-based methods, however, are limited in producing diverse samples and have potential bias of design pattern, consequently restricting their generalization capabilities. 

In this paper, we focus on the REC task by explicitly modeling the entity relations within both images and captions using the _structural similarity_ between them to solve the zero-shot visual grounding problem. Specifically, we decompose the image and caption into two sets of triplets in the form of (subject _,_ predicate _,_ object), in which each triplet captures a pair of potential entities with their interrelation. By considering the similarity of subject, object, and predicate jointly, we can find better matchings of object proposals and their referrings. Compared with existing work [56], our approach is more principled and eliminates the ad-hoc post-processing spatial relation resolver. More importantly, to improve the relationship understanding in the caption, we resort to ChatGPT and leverage its powerful _in-context learning_ capability [58, 61] for the triplet decomposition to find all possible relation triplets given a sentence. In contrast to the dependency parser in [56], our parsing works better when dealing with long captions and do not restrict to spatial relationships ( _e.g_ ., to the left of), which can fully capture the rich compositional semantics present in actions or interactions, such as walking or talking to. 

To address the limitation of a VLA model’s visual relationship understanding, we harness a curated collection of data sources rich in relational knowledge, which include human-object interaction datasets [3, 48] and image scene graph dataset [29]. Similar to our grounding pipeline, we isolate visual entities and construct triplets in both visual and textual sides and then implement a triplet-level contrastive learning objective to fine-tune the VLA model. Compared with the existing rule-based negative prompts construction, this design has two unique advantages. First, by decomposing a single image into multiple triplets, we can obtain more training instances and improve the diversity of the training data than simply using the entire image for fine-tuning. Furthermore, the isolation of entities removes the distraction of other content in the image, pro- 

viding more useful supervision for the model fine-tuning. We fine-tune a VLA model in a parameter-efficient manner [17] using LoRA [21], improving its visual relationship understanding while preserving its powerful generic feature representations learned from large-scale data. Our resulting model is called **VR-VLA** ( **V** isual **R** elationship VLA). 

Experimental results show that on the standard RefCOCO/g/+ datasets [44, 65], we can surpass the SOTA zeroshot baseline [56] up to 19.5%, and an average of 9.7%. On the challenging Who’s Waldo dataset [6], whose captions are much longer and depict much richer interactions of humans, our zero-shot method significantly outperforms [56], achieving comparable accuracy to supervised methods. We also conduct ablation studies validating the effectiveness of our model design. 

To summarize, our main contributions are three-folded. (1) A novel zero-shot visual grounding model, where we harness the powerful capabilities of foundation models ( _e.g_ ., ChatGPT and CLIP) to explicitly model the structural similarity between entities. (2) A novel recipe to improve a VLA model’s visual relationship understanding by efficiently incorporating the supervision from a collection of curated data sources. (3) We report SOTA zero-shot visual grounding results on the REC datasets and also show promising results on the Who’s Waldo dataset, where our zero-shot approach achieves comparable accuracy to the fully supervised method. 

## **2. Related Work** 

**Visual grounding.** Based on the focus of the grounding task, visual grounding diverges into two main categories. The first emphasizes the noun properties of the query text. Precise understanding of the noun’s meaning enables locating the corresponding grounding box in the image. Representative datasets contain MS-COCO [39], Object365 [51] (fixed-category), Flickr30K Entities [47] (open-vocabulary), etc. The second category emphasizes comprehending the interrelations among entities to localize the correct visual entity (potentially from among many similar ones) corresponding to the query text. Representative datasets contain RefCOCO/+/g [44, 65], Who’s Waldo [6], etc. Our research of interest belongs to the second category, where relations are strong grounding clues. 

Based on the use of grounding data, visual grounding can be classified into supervised and zero-shot categories. The majority of research [25, 37, 75] focuses on supervised visual grounding, where models are specifically designed and trained with grounding data for this purpose. Conversely, zero-shot visual grounding methods [32, 40, 56] adapt pre-existing vision-language models for grounding tasks. Our work is situated within this zero-shot visual grounding paradigm. 

**Visual relationship understanding.** Images and texts are 

constructed using fundamental elements — objects in images and noun spans in texts — along with their interactions and relationships. For a model to understand relationship, it must not only detect individual entities but also establish relational links between them. Research communities such as human-object interaction [3, 11, 26, 27, 36, 42, 48, 71– 73, 77, 79] and visual scene graph [29, 35, 57, 62, 63, 70, 74, 78] emphasize the relational aspect for visual tasks. 

Visual relationship understanding is also highly relevant to the _compositional reasoning_ [8, 9, 14, 19, 22, 24, 69] ability for VLA models. Although we anticipate that VLA models, trained via contrastive learning on extensive image-text pairs, would inherently develop a capacity for compositional reasoning, the reality is somewhat different. Most SOTA VLA models behave like “bags-of-words” [69]. They are capable of matching textual entities with corresponding visual elements, but falling short in interpreting their relationships or attributes. To address this, many studies have implemented strategies such as introducing negative text prompts [7, 8, 69] in training batches — including noun substitutions, attribute and verb modifications, caption augmentation — or integrating scene graphs [19] during training or inference. In our work, we show that finetune the VLA models on the visual relationship datasets can alleviate this problem. Furthermore, the explicitly defined structural representation also help strengthen the compositional reasoning for visual grounding tasks. 

**Language parsing.** In the language side, structure prediction [1, 5, 46, 52] is well studied and aims for solving several problems including entity recognition [10, 12, 34, 53], relation classification [16, 76], semantic role labeling [2, 18], event extraction [23, 33], coreference resolution [31, 60], etc. The acquired structural representation can be illustrated as either a language parsing tree [56] or a set of labels that indicate the respective roles of each word [52]. 

## **3. Proposed Approach** 

Our proposed grounding pipeline contains two stages. First, we decouple image and text entities and construct triplets in the format of (subject _,_ predicate _,_ object). Second, we calculate triplet-level similarity matrix and propagate it to the instance-level and then obtain the bounding box with the highest similarity score. The primary focus in our matching pipeline is to accurately model the relationship between entities, which is achieved by the the tripletlevel structural similarity, as shown in Fig. 2. We also provide a novel recipe to equip the VLA models with better compositional understanding ability. 

## **3.1. Constructing Triplets** 

Given a caption, denoted by _C_ , and its associated image, denoted by _I_ , we postulate that both _C_ and _I_ comprise sets 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions_images/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions.pdf-0003-07.png)


**----- Start of picture text -----**<br>
grounded triplets<br>input<br>Cat touching a paper roll<br>Image Encoder Text Encoder<br>Cat touching a paper roll structuralsimilarity = + +<br>subject  predicate  object<br>similarity similarity similarity<br>**----- End of picture text -----**<br>


Figure 2. **Illustration of the triplet-level structural similarity.** Visual and textual triplets are encoded by image encoder and text encoder, respectively. Then the structural similarity is calculated as the sum of cosine similarities between subject, predicate, and object. 

of entities[1] , denoted by _ET_ = _{e[T] i[}][M] i_ =1[for the text and] _[ E][I]_[=] _{e[I] i[}][N] i_ =1[for the image, where] _[ M]_[and] _[ N]_[represent the total] number of entities in _C_ and _I_ , respectively. Interrelation of an entity pair is represented by _r[T]_ ( _·_ ) for the text and _r[I]_ ( _·_ ) for the image. In this stage, our objective is to construct entity-relation triplets for both modalities. 

For text, we denote the triplets as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions_images/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions.pdf-0003-11.png)


For image, we denote the triplets as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions_images/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions.pdf-0003-13.png)


The cardinality of the above two sets are defined as _M[′]_ for text and _N[′]_ for image, respectively. 

**Textual triplets construction.** Large language models have exhibited a powerful capacity for a range of downstream tasks. Here, we leverage its powerful in-context learning capability to parse a caption _C_ into triplets _TT_ . Specifically, We design a prompt to instruct the ChatGPT to parse the caption text _C_ . Fig. 3 provides an overview on how we design prompt for RefCOCO/+/g dataset, and further details are elaborated as follows. Note that the prompts can vary depending on datasets to accommodate different distributions of the data. 

As shown in Fig. 3, the prompt can be divided into four parts: _(i) general instruction_ ; _(ii) supporting details_ ; _(iii) in-context learning examples_ , and _(iv) task instruction_ , followed by _LLM completion_ , which yields the output of the LLM in the specified format and content. In part (i), we define a clear and general instruction for specific task. Then, we elaborate supporting details in part (ii), including the expected output format, essential elements, and preferences for what should or should not be included, etc. In part (iii), we curate several in-context learning examples to guide the 

> 1We use “entities” to denote objects to differentiate from object in a triplet. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions_images/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions.pdf-0004-00.png)


**----- Start of picture text -----**<br>
General Instruction Given a sentence, first determine the main entity (with its attributes) that the<br>sentence is describing. Then analyze and extract all the spacial relation/action<br>between the determined main entity with other entities.<br>Supporting  Return in the JSON format: {"entity": "xxx", "relations": [["xxx", "yy", "zz"], ["aa",<br>Details "bb", "xxx"], ...]}. All the returned relations must be a triplet containing exactly<br>three elements. The first… The second… The third… You can make some guess if…<br>Your response must accurately follow the above instruction in both content and<br>format, akin to the examples provided below, without any extra explanatory text.<br>In-context Learning  ##Examples##<br>Examples INPUT: a woman wearing blue jeans sitting on a chair with a baby sitting in her lap<br>OUTPUT: {"entity": "a woman", "relations": [["a woman", "wearing", "blue jeans"], ["a woman", "sitting on", "a<br>chair"], ["a baby", "sitting in", "a woman"]]}<br>INPUT: ……<br>OUTPUT: ……<br>Task Instruction ##Your Task##<br>INPUT:  [user input, e.g., one small girl in white t-shirt is touching the elephant]<br>OUTPUT:<br>LLM completion {"entity": "one small girl", "relations": [["one small girl", "wearing", "white t-shirt"], ["one small<br>girl", "is touching", "the elephant"]]}<br>**----- End of picture text -----**<br>


Figure 3. **Illustration of leveraging ChatGPT’s powerful in-context learning capability to parse a caption into triplets.** 

LLM, which is immediately followed by part (iv), where we append the input caption _T_ . Finally, we feed the above input into LLM, and then decoupled textual triplets will be generated through the LLM completion. We also do a simple format check after the completion. 

**Visual triplets construction.** In an image, entities are represented by bounding boxes, each enclosing an individual object. These boxes might be predefined by the dataset or extracted using a pre-trained object detector. Without prior knowledge about how these entities are related, we assume potential interactions can happen between every pair of entities. Therefore, we generate visual pairs using a Cartesian product, which includes all possible combinations of entities. A notable case is when a pair consists of the same entity (box) twice. This represents a self-relation, suggesting the entity’s own attributes, such as color ( _e.g_ ., red) or self-actions ( _e.g_ ., walking). Then we use the union area of two entity boxes to represent _r[I]_ ( _e[I] i[, e][I] j_[)][, the interrelation] between entities. 

Here, we derive triplets set _TT_ and _TI_ from caption and image, respectively. Before moving on, we filter out redundant triplets _t[I] ij[∈T][I]_[based on heuristic rules similar to Re-] CLIP [56]. Specifically, given a textual triplet _t[T] ij_[, where its] predicate contains keywords that reflect some spacial relationships, _e.g_ ., to the left of. In such a case, we filter out visual box pairs where the central point of the former box ( _i.e_ ., subject) is in the right of the latter one ( _i.e_ ., object). This approach is much simpler than building complicated spatial semantic trees like in ReCLIP, yet it effectively adds spatial context and improves performance. 

## **3.2. Grounding Based on the Structural Similarity** 

With the triplets derived from the image and caption, we can use them to solve the visual grounding problem, which 

allows us to leverage the structural similarity between both modalities to more accurately link the textual descriptions of their corresponding image regions. 

We consider two grounding directions: text _→_ image and image _→_ text, each serving unique task requirements. The text _→_ image grounding is applied when identifying specific image regions based on a textual description ( _e.g_ ., RefCOCO/+/g [44, 65] dataset). Conversely, image _→_ text grounding involves locating relevant textual descriptions for a given image region ( _e.g_ ., Who’s Waldo [6] dataset). Given their symmetrical nature, this section will primarily focus on the text _→_ image grounding scenario. 

**Triplet-level grounding.** Given a text triplet _t[T] ij_ = ( _e[T] i[, r][T]_[ (] _[e][T] i[, e][T] j_[)] _[, e][T] j_[)][,][we][separately][feed] _[e][T] i_[,] _[r][T]_[ (] _[e][T] i[, e][T] j_[)][,] and _e[T] j_[into a VLA text encoder to obtain three text embed-] dings, denoted as ( **t** _i,_ **t** _i,j,_ **t** _j_ ). Similarly, for a image triplet _t[I] kl_[=][(] _[e][I] k[, r][I]_[(] _[e][I] k[, e][I] l_[)] _[, e][I] l_[)][,][we][derive][three][image][embed-] dings ( **v** _k,_ **v** _k,l,_ **v** _l_ ) using the image encoder. The similarity between these two triplets is then given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions_images/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions.pdf-0004-10.png)


where cos( _·, ·_ ) denotes the cosine similarity function. **S** _∈_ R _[M][ ′][×][N][′]_ is the similarity matrix between all text triplets with all image triplets. We subsequently get a binary indicator matrix **B** _∈{_ 0 _,_ 1 _}[M][ ′][×][N][ ′]_ by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions_images/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions.pdf-0004-12.png)


Here, for each text triplet _t[T] ij_[, the binary indicator matrix] **[ B]** assigns the value of 1 to the most similar image triplet _t[I] kl_ and 0 to all others. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions_images/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions.pdf-0005-00.png)


**----- Start of picture text -----**<br>
triplet-level grounding instance-level grounding<br>visual entities<br>visual triplets textual<br>textual … triplets<br>triplets 0.8 0 0<br>0.8 0.4 0.7 0.2 … 1 0 0 0 […]<br>0 0.8 0<br>Cat touching a paper roll 𝐒𝐒 𝐁𝐁 𝐑𝐑<br>visual triplets visual entities<br>textual … textual<br>triplets triplets<br>0.6 0.3 0.1 … 1 0 0 … 0.6+0.7 0 0<br>0.1 0.7 0.1 … 0 1 0 … 0 0.6+0.5 0<br>0.3 0.2 0.5 … 0 0 1 … 0 0 0.7+0.5<br>𝐒𝐒 𝐁𝐁 𝐑𝐑<br>A man watches as a man shoots over a man<br>**----- End of picture text -----**<br>


Figure 4. **Illustration of propagating the similarity scores from grounded triplets to the instance level.** Via the aggregation of the similarity scores from multiple grounded triplets, it helps find the instance-level correspondences more accurately. For instance, in the lower part, the referring expression a man and the blue bounding box appear in two different triplets, acting as the subject and object, respectively. Such structural similarity provide more useful cues to improve the instance-level grounding. (Best viewed in color.) 

**Instance-level grounding.** Another substantial challenge in the visual grounding problem is that both the subject and object in a triplet may have multiple interactions with other entities. To this end, we design a novel method to propagate the triplet-level grounding results to the instances. 

Specifically, based on the triplet-level grounding results, we can compute the instance-level structure-aware similarity matrix **R** as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions_images/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions.pdf-0005-04.png)


The two terms in Eq. 4 consider both the cases where _e[T] i_ and _e[I] k_[appear][as][the][subject][and][object][in][different] triplets, as shown in lower part of Fig. 4. By aggregating the similarity scores from multiple grounded triplets, it helps find the instance-level correspondences more accurately. 

Finally, for each text entity, we compute the most relevant image entity as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions_images/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions.pdf-0005-07.png)


where _e_ ˆ _[I] i_[denotes][the][corresponding][visual][entity][for] _[e][T] i_[.] Notably, our approach can easily extend to the one-to-many grounding scenario if we implement a threshold-based selection in place of the argmax function in Eq. 4 and Eq. 5. We limit our discussion to one-to-one grounding for clearer understanding. 

## **3.3. Enhanced Relational Understanding** 

In Equation 3, the interaction between two entities is represented by the term cos( **t** _i,j,_ **v** _k,l_ ). This term is crucial 

as it attempts to quantify the relationship between entities through cosine similarity, under the assumption that VLA models can adequately grasp these relationships. Nevertheless, as indicated by other studies [56, 69], this assumption often falls short in practice. To address this issue, we finetuned VLA models using a combination of datasets rich in relational knowledge. These datasets include HICO-det [3], SWiG [48], and Visual Genome (VG) [29]. Notably, in the case of the VG dataset, we excluded all the images from COCO to maintain the integrity of the zero-shot protocol, aligning with our experiments based on RefCOCO/+/g. 

The datasets mentioned provide annotation bounding boxes for objects together with their textual descriptions and relationships with other objects. So we can easily follow what we have done in the triplet-level grounding stage to create visual-textual triplets, and then utilize a contrastive learning loss on these triplets. To clarify, we use the same notation in Eq. 3 to calculate the similarity between two triplets. Assume _t[T] ij_[and] _[ t][I] kl_[are two corresponding triplets,] we define the contrastive loss as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions_images/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions.pdf-0005-13.png)


Through this refined approach, we aim to enhance the VLA models’ ability to understand and accurately score the relationship between entities, thereby enhancing the zero-shot grounding capability. 

Compared with the existing rule-based negative prompts construction [7, 8, 69], this design has two unique advan- 

|**Model**|RefCOCOg<br>**Val**<br>**Test**|RefCOCOg<br>**Val**<br>**Test**|RefCOCO+<br>**Val**<br>**TestA**<br>**TestB**|RefCOCO+<br>**Val**<br>**TestA**<br>**TestB**|RefCOCO+<br>**Val**<br>**TestA**<br>**TestB**|RefCOCO<br>**Val**<br>**TestA**|**TestB**|
|---|---|---|---|---|---|---|---|
|Random|18.12<br>19.10||16.29|13.57<br>19.60||15.73<br>13.51|19.20|
|Supervised SOTA[37]|88.73<br>89.37||85.24|89.63<br>79.79||92.64<br>94.33|91.46|
|CPT-Blk w/ VinVL [64]<br>CPT-Segw/ VinVL[64]|32.10<br>32.30<br>36.70<br>36.50||25.40<br>31.90|25.00<br>27.00<br>35.20<br>28.80||26.90<br>27.50<br>32.20<br>36.10|27.40<br>30.30|
|**CLIP (ViT-B/32)**<br>CPT-adapted [56]<br>GradCAM [50]<br>ReCLIP [56]<br>**Ours**<br>**Ours+VR-CLIP**|21.77<br>22.78<br>49.51<br>48.53<br>56.96<br>56.15<br>57.60<br>56.64<br>59.87<br>59.90||23.46<br>44.64<br>45.34<br>45.64<br>**55.52**|21.73<br>26.32<br>50.73<br>39.01<br>48.45<br>42.71<br>47.59<br>42.79<br>**62.56**<br>45.69||23.79<br>22.87<br>42.29<br>49.04<br>45.77<br>46.99<br>48.24<br>48.40<br>**60.62**<br>**66.52**|26.03<br>36.68<br>45.24<br>49.15<br>**54.86**|
|**FLAVA**<br>**Ours**<br>**Ours+VR-FLAVA**|60.95<br>59.99<br>**61.25**<br>**60.86**||48.89<br>50.79|50.02<br>46.86<br>53.35<br>**47.62**||49.37<br>47.76<br>52.46<br>52.66|51.68<br>52.92|



Table 1. **Accuracy on the RefCOCOg, RefCOCO+ and RefCOCO datasets.** Ours represents leveraging our triplet-to-instance pipeline for grounding. Ours+VR-CLIP/VR-FLAVA further replaces the original VLA model with our relationship-enhanced model. Except for the supervised method, the best results are highlighted in **bold** , and second-best results are underlined. 

tages. First, by decomposing a single image into multiple triplets, we can obtain more training instances and improve the diversity of the training data than simply using the entire image for fine-tuning. Furthermore, the isolation of entities removes the distraction of other content in the image, providing more useful supervision for the model fine-tuning. 

## **4. Experiments** 

## **4.1. Setup** 

**RefCOCO/RefCOCO+[44]/RefCOCOg[65]** are collected from MS-COCO [38]. RefCOCO includes 19,994 images with 142,210 referring expressions. RefCOCO+ has 19,992 images and 141,564 expressions. RefCOCOg contains 26,771 images with 104,560 expressions. In RefCOCO and RefCOCO+, expressions are shorter, averaging 1.6 nouns and 3.6 words. In RefCOCOg, expressions are longer, averaging 2.8 nouns and 8.4 words. 

**Who’s Waldo [6]** introduces a person-centric visual grounding task, where all names in the captions are masked, forcing models to link boxes and the masked [NAME] tokens through attributes and interactions between visual entities. The captions are long and contain complex scene descriptions. We use its test split for evaluation, which contains 6741 images. Each caption contains about 30 words. **Evaluation metrics.** On RefCOCO/+/g, we follow previous work [56] to use accuracy as the grounding results, _i.e_ ., if the IoU (Intersection over Union) value between the predicted box and ground truth region is larger than 0.5, it is a correct prediction. On Who’s Waldo, following previous work [6], given the grounding results of person in textual descriptions to bounding boxes in the image, we report the accuracy against ground-truth links on the test set. 

## **4.2. Implementation Details** 

**RefCOCO/+/g** On RefCOCO/+/g datasets, we adopt the test-time augmentation strategy outlined in ReCLIP [56], where they use both cropping and blurring for isolated visual entities. When blurring the union region, we separately process each box in the box pair, isolating only the area where the box is located, rather than directly using the union area, which helps to minimize distraction from other visual objects. We use the whole caption instead of the decoupled main entities before feeding into the text encoder since it produces better results. All the heuristic rules mentioned in Sec. 3.1 are also adopted from ReCLIP. 

**VLA fine-tuning** We fine-tune the CLIP model based on the code from [8], where LoRA rank _r_ = 4, batch size is 1024, learning rate is 5 _e −_ 6, and epoch is 20. We use the huggingface PEFT [43] for fine-tuning FLAVA, where we set lora rank _r_ = 16 for 10 epochs. Since the SWiG dataset [48] does not contains triplet annotations, we use ChatGPT to convert the existing annotation into triplets. 

**Box generator** Following ReCLIP, we use bounding boxes generated from MAttNet [66] as the box proposals on RefCOCO/+/g. On the Who’s Waldo dataset, we use the box proposals provided in the annotations. 

## **4.3. Main Results** 

**RefCOCO/+/g** We benchmark our approach against various zero-shot visual grounding models, including Colorful Prompt Tuning (CPT) [64], GradCAM [50], and ReCLIP [56]. CPT-adapted is introduced and adapted by [56]. ReCLIP represents the latest SOTA in zero-shot REC methods. 

As shown in Table 1, compared to other models utilizing the same CLIP architecture, our proposed method consis- 

|**Method**|**Test Accuracy**|
|---|---|
|**Supervised**||
|_Who’s Waldo_ [6]|**63.5**|
|**Pretrained on grounding data**||
|Gupta et al. [15] (COCO)|35.6|
|Gupta et al. [15] (Flickr30K)|38.2|
|SL-CCRF [41]|46.4|
|MAttNet [66]|24.1|
|UNITER[4]|34.2|
|**CLIP**||
|ReCLIP [56]|29.4|
|**Ours**|60.8|
|**Ours+VR-CLIP**|61.3|
|**FLAVA**||
|**Ours**|59.6|
|**Ours+VR-FLAVA**|59.8|



Table 2. **Accuracy on the Who’s Waldo dataset.** The best results are highlighted in **bold** , and second-best results are underlined. 

tently outperforms all competitors across all splits. Specifically, our model exhibits a performance improvement of up to 19.53% over ReCLIP, with an average enhancement of 9.74%. Remarkably, even without fine-tuning the backbone CLIP model, our method can surpass the ReCLIP by up to 3.91%, with an average of 1.05%, showing that the structural similarity based on ChatGPT’s parsing also contributes to the relational understanding. 

In addition, we also extend our methodology to another VLA model, FLAVA [54], to verify the generalizability of our approach. Not surprisingly, when integrated into our matching pipeline, FLAVA demonstrates superior performance compared to the CLIP model. This can be attributed to FLAVA’s inherently more robust architecture. After finetune the FLAVA, the resulting VR-FLAVA consistently improve the performance across all dataset splits, reinforcing the effectiveness of our method in enhancing the relationship understanding of various VLA models. 

**Who’s Waldo** We compare our approach with the models trained on grounding dataset, which include Gupta et al. [15], SL-CCRF [41], MAttNet [66] and UNITER [4]. The _Who’s Waldo_ method serves as a supervised baseline, as reported in its original paper [6]. Additionally, we adapte ReCLIP [56] for our dataset, utilizing their language parser to identify potential referring expressions, followed by grounding using their original approach[2] . 

As shown in Table 2, our approach outperforms all models trained on the grounding dataset, with a notable margin. Among these, SL-CCRF was the closest competitor, yet it falls behind our method by 14.9%. When compared to the supervised _Who’s Waldo_ method, our zero-shot setting only shows a 2.2% reduction in performance. This highlights 

|Triplet|VR-CLIP|**Val**|**Test**|
|---|---|---|---|
|||||
|✗<br>✗<br>✓<br>✓|✗<br>✓<br>✗<br>✓|55.35<br>56.90<br>57.60<br>**59.87**|54.33<br>56.81<br>56.64<br>**59.90**|



Table 3. **Effectiveness of each component of our grounding pipeline on RefCOCOg.** Triplet means whether we use the triplet-to-instance grounding (instead of scoring-and-ranking). VR-CLIP represents whether we use fine-tuned VR-CLIP instead of the original CLIP model. 

our method’s effectiveness in visual grounding, especially in processing long and complex captions. Notably, performance of ReCLIP is no better than random choice. This is because their language parser fails to handle the real-world complex captions as in Who’s Waldo. It validates our design choice of using the in-context learning capability of LLMs for better generalization ability. 

We show visualization results in Fig. 5. The left two columns display results from RefCOCOg, while the rightmost column shows results from Who’s Waldo. ReCLIP failed in all examples listed. Our approach, which explicitly models relationships (indicated by arrows in the images), provides more helpful information for grounding. Additionally, it is observed that ChatGPT consistently excels in parsing complex captions, such as those in Who’s Waldo. 

## **4.4. Ablation Studies** 

In this section, we conduct ablation studies on RefCOCOg. This dataset is particularly suitable for our evaluation because it has longer captions and rich entity interactions, making it an ideal testbed for assessing each component. **Effectiveness of components in the grounding pipeline.** We explore two key variations: _Triplet_ and _VR-CLIP_ . The _Triplet_ variation examines the impact of utilizing tripletto-instance matching as opposed to a basic scoring-andranking approach, i.e., scoring each isolated boxes using CLIP, than select one with the highest similarity. The _VRCLIP_ variant assesses the performance differences between the fine-tuned VR-CLIP and the original CLIP model. 

As shown in Table 3, substituting the CLIP model with the VR-CLIP yields superior results because of the enhanced relational capability. Note that although we do not explicitly use the structural similarity in this context, we are still using the whole caption as the referring expressions, which inherently convey the relationship information. Further analysis reveals that if we replace the scoringand-ranking with our proposed triplet-to-instance matching pipeline, we can get better results through the relationship modeling. By combining both, we can achieve best results. **Effectiveness of triplet components.** We separately remove the subject, object and predicate terms in 

2The experiment is conducted using the authors’ released code. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions_images/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions.pdf-0008-00.png)


**----- Start of picture text -----**<br>
the catcher U.S. President person<br>a flower<br>person<br>with<br>shake hands with<br>green vase Indian Minister for Human<br>other vases other vases Resource Development Shri person<br>the ball<br>green vase on the left with a flower in it. the catcher waiting for the ball Indian Minister for Human Resource Development Shri [NAME] looks on as U.S. President [NAME] shakes hands with [NAME]<br>at the U.S.-India Strategic Dialogue reception at the the U.S.<br>Department of State in Washington, D.C., on June 3, 2010.<br>a  woman in<br>a purple coat<br>adjust<br>a cake pile of brush giraffe Army Gen. person<br>Navy Adm. person<br>Navy Adm. [NAME]. [NAME], embraces Army Gen. [NAME] during the<br>change-of-command ceremony for U.S. Special Operations Command in<br>a woman in a purple coat adjusts a cake. giraffe bent over pile of brush. Tampa, Fla., Aug. 28, 2014. [NAME] assumed command from [NAME].<br>**----- End of picture text -----**<br>


Figure 5. **Zero-shot visual grounding results.** Left two columns are results from RefCOCO, where our predictions are in green box, distraction objects are in red box. The rightmost column shows results from Who’s Waldo, where predicted annotation links are in the same color. Arrows represent relationships between visual objects, and the text on the images are the parsed triplets. 

||**Val**|**Test**|
|---|---|---|
||||
|full<br>w/osubject<br>w/oobject<br>w/opredicate|**59.87**<br>48.35<br>56.92<br>56.90|**59.90**<br>47.83<br>57.05<br>56.81|



Table 4. **Effectiveness of each triplet component on RefCOCOg.** Each removal is done by set the corresponding term in Eq. 3 to 0. For w/o predicate, we also turn off the box pair filter to remove any spatial information provided in the caption. 

Eq. 3 to explore their effectiveness in grounding performance. As shown in Table 4, the absence of subject or object leads to the loss of most noun information and its attributes, resulting in a clear accuracy drop. This impact is particularly significant for subject removal, since most main entities in RefCOCOg are represented as subject. Without predicate, meaning no interrelation between entities is considered, the accuracy degrades by about 3 points. It emphasizes the importance of our structural similarity in modeling the entity relationships. 

**Effectiveness of triplet- to instance-level grounding .** We separately remove the first and second terms in Eq. 4 to validate our design of triplet- to instance-level grounding. The results are reported in Table 5, which shows that utilizing both pieces of information yields the best performance. 

## **4.5. Limitations** 

While our method enhances the visual relationship understanding of VLA models, it sacrifices the model’s zero-shot 

||**Val**|**Test**|
|---|---|---|
||||
|full<br>w/o 1st term<br>w/o 2nd term|**59.87**<br>59.54<br>59.11|**59.90**<br>59.37<br>59.26|



Table 5. **Effectiveness of different terms in Eq. 4 for triplet- to instance-level grounding on RefCOCOg.** 

capabilities as generalist models and downgrades them to specialist ones. For example, after fine-tuning, CLIP’s zeroshot image classification accuracy decreases from 0.63 to 0.50 and the R@5 for image retrieval on COCO decreases from 0.57 to 0.54. A promising future direction is to apply large-scale unlabeled image-caption pairs during VLA finetuning, and probably we can generate region-text triplets using ChatGPT and SAM (Segment Anything Model) [28] as pseudo ground-truths. 

## **5. Conclusion** 

In this paper, we proposed a novel zero-shot referring expression comprehension model by resorting to powerful capabilities of foundation models ( _e.g_ ., ChatGPT and CLIP) to explicitly model the structural similarity between entities and then find their correspondences by propagating the similarity from triplets to instances. We also introduced a novel recipe to improve a VLA model’s visual relationship understanding by training from a collection of curated data sources. Experimental results on RefCOCO/+/g and Who’s Waldo validate the effectiveness of our approach. 

## **A. Implementation Details** 

**Special Cases for Textual Triplets.** Not all caption can be perfectly decoupled into textual triplets. Often, such a caption is just a single noun or lacks an explicit subject. For example, the caption red apple would be decoupled into ("red apple", "", ""), and person walking into ("person", "walking", ""). In these instances, we fill the blank spaces ( _i.e_ ., missing items in the triplet) with the subject string, resulting in ("red apple", "red apple", "red apple") and ("person", "walking", "person"), respectively, for the previous examples. This approach ensures our grounding pipeline can manage such simplified cases. For instance, ("red apple", "red apple", "red apple") will be matched three times with the visual entity of a red apple, which means that it degenerates to the naive score-andranking strategy. 

Additionally, before feeding the predicate into the text encoder, on the RefCOCO/+/g dataset, we form a complete sentence by concatenating the subject, predicate and object, _e.g_ ., "vase on top of table" instead of "on top of". On the Who’s Waldo dataset, we add a person before and after the predicate, _e.g_ ., "a person looking at a person" instead of "looking at". This is because, in most cases, a single predicate like "on top of" is semantically meaningless. Instead, a complete phrase like "vase on top of table" offers more contextual information. 

**Dataset for VLA Fine-tuning.** Our dataset for VLA finetuning is obtained from HICO-det [3], SWiG [48] and Visual Genome (without COCO images) [29]. Each datapoint in this dataset consists of multiple image-text triplet pairs with the same text triplet, with two examples illustrated in Fig. 6. Notably, “multiple” is because we group image triplets corresponding to the same textual triplet into a single datapoint. Consequently, a single datapoint may comprise several distinct image triplets paired with the same textual triplet. For training purposes, we randomly select one image triplet from each category per epoch. This strategy is adopted to avoid scenarios in a single batch where an image triplet is forced to simultaneously pull in and push away from the same textual triplet due to the contrastive learning objective. 

## **B. Additional Experiment Results** 

This section provides additional experiment results tested on RefCOCO/+/g. Table. 6 shows the full results with different box proposal variants, _i.e_ ., using a bounding box size prior (filter our objects smaller than 5% of the image), and use the groundtruth bounding boxes as box proposals. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions_images/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions.pdf-0009-06.png)


**----- Start of picture text -----**<br>
a photo of person<br>a photo of bear person watch bear<br>a photo of a soccer player<br>a photo of people<br>people hoists a soccer player<br>**----- End of picture text -----**<br>


Figure 6. **Two examples in the dataset for VLA fine-tuning.** 

## **C. Additional Visualization Results** 

In this section, we provide additional visualization results for RefCOCO, RefCOCO+, RefCOCOg, and Who’s Waldo, illustrated in Fig.7, Fig.8, Fig.9, and Fig.10, respectively. Fig. 7, 8, 9 highlight examples where ReCLIP failed but our grounding approach yielded correct results. The textual triplets parsed by ChatGPT are also displayed in the images. Next, we will discuss some selected examples to illustrate the advantages of our approach. 

As shown in Fig. 9, on the RefCOCOg dataset, our approach is able to successfully differentiate multiple instances of the same object category by understanding their relationships with others. For instance, in the first example in Fig. 9, the blue suitcase can be accurately grounded among other ones. For the example of “a zebra that is standing”, the curved arrow in image represents self-action, where no object is involved, which is a special case of our grounding pipeline. 

ChatGPT plays an important role in improving the robustness of our grounding approach. In Fig. 7, as for “rt bottom chair”, ChatGPT understands that “rt” stands for “right”, allowing us to accurately generate triplets as depicted in the image. Similarly, in Fig. 8, it is worth highlighting the example “rider of the gray elephant”. Here, ChatGPT made some reasonable deduction that rider is “on top of” the elephant. With longer captions, as shown in Figure 8, ChatGPT can consistently parse each entity, along with its complex attributes, affiliations, and inter-entity interactions, which are vital for accurate grounding. These 

|RefCOCOg<br>RefCOCO+<br>RefCOCO<br>**Model**<br>**Val**<br>**Test**<br>**Val**<br>**TestA**<br>**TestB**<br>**Val**<br>**TestA**<br>**TestB**|RefCOCOg<br>RefCOCO+<br>RefCOCO<br>**Model**<br>**Val**<br>**Test**<br>**Val**<br>**TestA**<br>**TestB**<br>**Val**<br>**TestA**<br>**TestB**|RefCOCOg<br>RefCOCO+<br>RefCOCO<br>**Model**<br>**Val**<br>**Test**<br>**Val**<br>**TestA**<br>**TestB**<br>**Val**<br>**TestA**<br>**TestB**|
|---|---|---|
|Random<br>18.12<br>19.10<br>Random(w/groundtruth boxproposal)<br>20.18<br>20.34|16.29<br>13.57<br>19.60<br>16.73<br>12.57<br>22.13|15.73<br>13.51<br>19.20<br>16.37<br>12.45<br>21.32|
|Supervised SOTA[37]<br>88.73<br>89.37|85.24<br>89.63<br>79.79|92.64<br>94.33<br>91.46|
|CPT-Blk w/ VinVL [64]<br>32.10<br>32.30<br>CPT-Segw/ VinVL[64]<br>36.70<br>36.50|25.40<br>25.00<br>27.00<br>31.90<br>35.20<br>28.80|26.90<br>27.50<br>27.40<br>32.20<br>36.10<br>30.30|
|**CLIP**<br>CPT-adapted [56]<br>21.77<br>22.78<br>GradCAM [50]<br>49.51<br>48.53<br>ReCLIP [56]<br>56.96<br>56.15<br>**Ours**<br>57.60<br>56.64<br>**Ours+VR-CLIP**<br>**59.87**<br>**59.90**|23.46<br>21.73<br>26.32<br>44.64<br>50.73<br>39.01<br>45.34<br>48.45<br>42.71<br>45.64<br>47.59<br>42.79<br>**55.52**<br>**62.56**<br>**45.69**|23.79<br>22.87<br>26.03<br>42.29<br>49.04<br>36.68<br>45.77<br>46.99<br>45.24<br>48.24<br>48.40<br>49.15<br>**60.62**<br>**66.52**<br>**54.86**|
|**CLIP (w/ box size prior)**<br>CPT-adapted [56]<br>28.98<br>30.14<br>GradCAM [50]<br>52.29<br>51.28<br>ReCLIP [56]<br>**60.85**<br>**61.05**<br>**Ours**<br>58.52<br>57.95<br>**Ours+VR-CLIP**<br>58.95<br>59.55|26.64<br>25.13<br>27.27<br>49.41<br>59.66<br>38.62<br>55.07<br>60.47<br>47.41<br>52.38<br>57.65<br>45.65<br>**58.65**<br>**68.32**<br>**47.42**|26.08<br>25.38<br>28.03<br>44.65<br>53.49<br>36.19<br>54.04<br>58.60<br>49.54<br>56.10<br>58.97<br>52.23<br>**62.92**<br>**69.90**<br>**55.19**|
|**CLIP (w/ groundtruth box proposal)**<br>CPT-adapted [56]<br>24.16<br>24.70<br>GradCAM [50]<br>54.00<br>54.01<br>ReCLIP [56]<br>**65.48**<br>64.38<br>**Ours**<br>64.99<br>64.03<br>**Ours+VR-CLIP**<br>65.11<br>**66.00**|25.07<br>22.28<br>28.68<br>48.00<br>52.13<br>43.85<br>49.20<br>50.23<br>48.58<br>49.75<br>50.18<br>49.77<br>**58.65**<br>**64.78**<br>**53.98**|25.12<br>23.39<br>28.42<br>45.41<br>50.13<br>41.47<br>49.69<br>48.08<br>52.50<br>52.82<br>49.90<br>57.29<br>**65.60**<br>**68.59**<br>**63.51**|
|**FLAVA**<br>**Ours**<br>60.95<br>59.99<br>**Ours+VR-FLAVA**<br>**61.25**<br>**60.86**|48.89<br>50.02<br>46.86<br>**50.79**<br>**53.35**<br>**47.62**|49.37<br>47.76<br>51.68<br>**52.46**<br>**52.66**<br>**52.92**|
|**FLAVA (w/ box size prior)**<br>**Ours**<br>60.40<br>60.73<br>**Ours+VR-FLAVA**<br>**60.48**<br>**61.28**|54.82<br>59.73<br>**48.25**<br>**55.00**<br>**61.13**<br>48.17|57.22<br>59.61<br>55.05<br>**57.80**<br>**60.86**<br>**55.33**|
|**FLAVA (w/ groundtruth box proposal)**<br>**Ours**<br>67.71<br>66.11<br>**Ours+VR-FLAVA**<br>**67.97**<br>**67.25**|52.17<br>51.73<br>54.33<br>**54.66**<br>**55.78**<br>**54.82**|55.75<br>50.68<br>62.10<br>**58.22**<br>**55.83**<br>**62.47**|



Table 6. **Accuracy on the RefCOCOg, RefCOCO+ and RefCOCO datasets.** Ours represents leveraging our triplet-to-instance pipeline for grounding. Ours+VR-CLIP/VR-FLAVA further replaces the original VLA model with our relationship-enhanced model. Results excluding object boxes smaller than 5% of the image size are denoted as w/ box size prior. Results using groundtruth box proposals are indicated as w/ groundtruth box proposal. For every combination of model and box proposal type, the best results are highlighted in **bold** . 

examples demonstrate ChatGPT’s superior robustness compared to ReCLIP’s language parsing method, especially in challenging scenarios. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions_images/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions.pdf-0011-00.png)


**----- Start of picture text -----**<br>
other<br>people<br>other people<br>person<br>guy<br>in the<br>back of<br>bench geishas guy<br>person on left of bench guy on right turning to look at geishas left guy in back<br>other chairs<br>other oranges<br>orange<br>to the to the<br>left of other oranges right of<br>orange<br>chair<br>leftmost orange orange on right rt bottom chair<br>**----- End of picture text -----**<br>


Figure 7. **Zero-shot visual grounding results on RefCOCO.** Our predictions are in green box, distraction objects are in red box. Arrows represent relationships between visual objects, and the text on the images are the parsed triplets. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions_images/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions.pdf-0011-02.png)


**----- Start of picture text -----**<br>
guy<br>snowboard<br>rider<br>playing with<br>to the right of<br>on top of to the right of<br>other snowboards<br>dog the gray elephant<br>guy playing with dog rider of the gray elephant rightiest snowboard<br>elephant<br>bed<br>above is bigger than<br>sandwich<br>under other elephants<br>person the other one<br>bed under person sandwich above the other one big elephant<br>**----- End of picture text -----**<br>


Figure 8. **Zero-shot visual grounding results on RefCOCO+.** Our predictions are in green box, distraction objects are in red box. Arrows represent relationships between visual objects, and the text on the images are the parsed triplets. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions_images/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions.pdf-0012-00.png)


**----- Start of picture text -----**<br>
a man<br>cycles<br>is sitting on<br>a flower pot<br>a blue suitcase the horse on top of<br>the table<br>a blue suitcase that a man is sitting on the horse looking at the cycles. a flower pot on the table<br>woman<br>a zebra<br>the ball<br>is standing<br>throwing<br>the chair<br>the man<br>the man throwing the ball from the picther's mound. a zebra that is standing woman sitting in the chair<br>**----- End of picture text -----**<br>


Figure 9. **Additional zero-shot visual grounding results on RefCOCOg.** Our predictions are in green box, distraction objects are in red box. Arrows represent relationships between visual objects, and the text on the images are the parsed triplets. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions_images/cvpr_2024_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_Captions.pdf-0012-02.png)


**----- Start of picture text -----**<br>
listens to<br>Secretary<br>of State<br>person President person<br>2005, of the 2005 Summit of the Americas in Mar del [NAME]President  during the opening session Friday, Nov. 4, [NAME] listens to Secretary of State  addresses reporters<br>Plata, Argentina.<br>met with<br>Lt. Gen. person,<br>commanding general,<br>Marine Corps Combat<br>Development Command<br>person, former first lady<br>Afghanistan  Gen. person,<br>commanding general, Marine Corps Combat Development [NAME], former first lady, is escorted by Lt. Gen. [NAME],  President  person Central CommandCommander U.S.<br>Command during the 100th birthday anniversary  Afghanistan President [NAME] addresses reporters<br>celebration of former president [NAME], Feb. 6. [NAME] Portuguese President  President  at a press conference held at MacDill Air Force<br>greeted the crowd of more than 1, 500 actors, musicians,  person person Base, Tampa. Gen. [NAME], Commander U.S.<br>former advisors, friends and Camp Pendleton Marines  Portuguese President [NAME] presenting to  Central Command, shown here right, and coalition<br>during the celebration and ceremonial wreath laying. President [NAME] instructions compiled by [NAME]. partners met with [NAME] while he was at MacDill.<br>**----- End of picture text -----**<br>


Figure 10. **Additional zero-shot visual grounding results on Who’s Waldo.** Predicted annotation links are in the same color. Arrows represent relationships between visual objects, and the text on the images are the parsed triplets. 

## **References** 

- [1] David Belanger and Andrew McCallum. Structured prediction energy networks. In _International Conference on Machine Learning_ , pages 983–992. PMLR, 2016. 3 

- [2] Xavier Carreras and Llu´ıs M`arquez. Introduction to the conll-2005 shared task: Semantic role labeling. In _Proceedings of the ninth conference on computational natural language learning (CoNLL-2005)_ , pages 152–164, 2005. 3 

- [3] Yu-Wei Chao, Zhan Wang, Yugeng He, Jiaxuan Wang, and Jia Deng. Hico: A benchmark for recognizing human-object interactions in images. In _Proceedings of the IEEE international conference on computer vision_ , pages 1017–1025, 2015. 2, 3, 5, 9 

- [4] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter: Universal image-text representation learning. In _European conference on computer vision_ , pages 104–120. Springer, 2020. 7 

- [5] Despina Christou and Grigorios Tsoumakas. Improving distantly-supervised relation extraction through bert-based label and instance embeddings. _IEEE Access_ , 9:62574– 62582, 2021. 3 

- [6] Yuqing Cui, Apoorv Khandelwal, Yoav Artzi, Noah Snavely, and Hadar Averbuch-Elor. Who’s waldo? linking people across text and images. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1374– 1384, 2021. 2, 4, 6, 7 

- [7] Sivan Doveh, Assaf Arbelle, Sivan Harary, Amit Alfassy, Roei Herzig, Donghyun Kim, Raja Giryes, Rogerio Feris, Rameswar Panda, Shimon Ullman, et al. Dense and aligned captions (dac) promote compositional reasoning in vl models. _arXiv preprint arXiv:2305.19595_ , 2023. 2, 3, 5 

- [8] Sivan Doveh, Assaf Arbelle, Sivan Harary, Eli Schwartz, Roei Herzig, Raja Giryes, Rogerio Feris, Rameswar Panda, Shimon Ullman, and Leonid Karlinsky. Teaching structured vision & language concepts to vision & language models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 2657–2668, 2023. 2, 3, 5, 6 

- [9] Yue Fan, Jing Gu, Kaiwen Zhou, Qianqi Yan, Shan Jiang, Ching-Chen Kuo, Xinze Guan, and Xin Eric Wang. Muffin or chihuahua? challenging large vision-language models with multipanel vqa. _arXiv preprint arXiv:2401.15847_ , 2024. 3 

- [10] Jenny Rose Finkel and Christopher D Manning. Nested named entity recognition. In _Proceedings of the 2009 conference on empirical methods in natural language processing_ , pages 141–150, 2009. 3 

- [11] Chen Gao, Jiarui Xu, Yuliang Zou, and Jia-Bin Huang. DRG: Dual relation graph for human-object interaction detection. In _ECCV_ , 2020. 3 

- [12] Jing Gu and Zhou Yu. Data annealing for informal language understanding tasks. In _Findings of the Association for Computational Linguistics: EMNLP 2020_ , pages 3153– 3159, 2020. 3 

- [13] Jing Gu, Eliana Stefani, Qi Wu, Jesse Thomason, and Xin Eric Wang. Vision-and-language navigation: A sur- 

vey of tasks, methods, and future directions. _arXiv preprint arXiv:2203.12667_ , 2022. 1 

- [14] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 14953–14962, 2023. 2, 3 

- [15] Tanmay Gupta, Arash Vahdat, Gal Chechik, Xiaodong Yang, Jan Kautz, and Derek Hoiem. Contrastive learning for weakly supervised phrase grounding. In _European Conference on Computer Vision_ , pages 752–768. Springer, 2020. 7 

- [16] Xu Han, Hao Zhu, Pengfei Yu, Ziyun Wang, Yuan Yao, Zhiyuan Liu, and Maosong Sun. Fewrel: A large-scale supervised few-shot relation classification dataset with stateof-the-art evaluation. _arXiv preprint arXiv:1810.10147_ , 2018. 3 

- [17] Zeyu Han, Chao Gao, Jinyang Liu, Sai Qian Zhang, et al. Parameter-efficient fine-tuning for large models: A comprehensive survey. _arXiv preprint arXiv:2403.14608_ , 2024. 2 

- [18] Luheng He, Kenton Lee, Mike Lewis, and Luke Zettlemoyer. Deep semantic role labeling: What works and what’s next. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 473–483, 2017. 3 

- [19] Roei Herzig, Alon Mendelson, Leonid Karlinsky, Assaf Arbelle, Rogerio Feris, Trevor Darrell, and Amir Globerson. Incorporating structured representations into pretrained vision & language models using scene graphs. _arXiv preprint arXiv:2305.06343_ , 2023. 3 

- [20] MD Zakir Hossain, Ferdous Sohel, Mohd Fairuz Shiratuddin, and Hamid Laga. A comprehensive survey of deep learning for image captioning. _ACM Computing Surveys (CsUR)_ , 51(6):1–36, 2019. 1 

- [21] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_ , 2021. 2 

- [22] Yufeng Huang, Jiji Tang, Zhuo Chen, Rongsheng Zhang, Xinfeng Zhang, Weijie Chen, Zeng Zhao, Tangjie Lv, Zhipeng Hu, and Wen Zhang. Structure-clip: Enhance multimodal language representations with structure knowledge. _arXiv preprint arXiv:2305.06152_ , 2023. 2, 3 

- [23] Heng Ji and Ralph Grishman. Refining event extraction through cross-document inference. In _Proceedings of ACL08: Hlt_ , pages 254–262, 2008. 3 

- [24] Kenan Jiang, Xuehai He, Ruize Xu, and Xin Eric Wang. Comclip: Training-free compositional image and text matching. _arXiv preprint arXiv:2211.13854_ , 2022. 2, 3 

- [25] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetrmodulated detection for end-to-end multi-modal understanding. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1780–1790, 2021. 2 

- [26] Bumsoo Kim, Junhyun Lee, Jaewoo Kang, Eun-Sol Kim, and Hyunwoo J Kim. HOTR: End-to-end human-object interaction detection with transformers. In _CVPR_ , 2021. 3 

- [27] Sanghyun Kim, Deunsol Jung, and Minsu Cho. Relational context learning for human-object interaction detection. In _CVPR_ , 2023. 3 

- [28] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 4015–4026, 2023. 8 

- [29] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _International journal of computer vision_ , 123:32–73, 2017. 2, 3, 5, 9 

- [30] Ranjay Krishna, Ines Chami, Michael Bernstein, and Li FeiFei. Referring relationships. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 6867–6876, 2018. 1 

- [31] Kenton Lee, Luheng He, and Luke Zettlemoyer. Higherorder coreference resolution with coarse-to-fine inference. _arXiv preprint arXiv:1804.05392_ , 2018. 3 

- [32] Jiahao Li, Greg Shakhnarovich, and Raymond A Yeh. Adapting clip for phrase localization without further training. _arXiv preprint arXiv:2204.03647_ , 2022. 1, 2 

- [33] Qi Li, Heng Ji, and Liang Huang. Joint event extraction via structured prediction with global features. In _Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 73–82, 2013. 3 

- [34] Xiaoya Li, Jingrong Feng, Yuxian Meng, Qinghong Han, Fei Wu, and Jiwei Li. A unified mrc framework for named entity recognition. _arXiv preprint arXiv:1910.11476_ , 2019. 3 

- [35] Yikang Li, Wanli Ouyang, Bolei Zhou, Kun Wang, and Xiaogang Wang. Scene graph generation from objects, phrases and region captions. In _ICCV_ , 2017. 3 

- [36] Yue Liao, Aixi Zhang, Miao Lu, Yongliang Wang, Xiaobo Li, and Si Liu. Gen-vlkt: Simplify association and enhance interaction understanding for hoi detection. In _CVPR_ , 2022. 3 

- [37] Fangjian Lin, Jianlong Yuan, Sitong Wu, Fan Wang, and Zhibin Wang. Uninext: Exploring a unified architecture for vision recognition. _arXiv preprint arXiv:2304.13700_ , 2023. 2, 6, 10 

- [38] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _ECCV_ , 2014. 6 

- [39] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13_ , pages 740–755. Springer, 2014. 2 

- [40] Xuyang Liu, Siteng Huang, Yachen Kang, Honggang Chen, and Donglin Wang. Vgdiffzero: Text-to-image diffusion models can be zero-shot visual grounders. _arXiv preprint arXiv:2309.01141_ , 2023. 1, 2 

- [41] Ye Liu, Junsong Yuan, and Chang Wen Chen. Consnet: Learning consistency graph for zero-shot human-object interaction detection. In _Proceedings of the 28th ACM International Conference on Multimedia_ , 2020. 7 

- [42] Shuailei Ma, Yuefeng Wang, Shanze Wang, and Ying Wei. Fgahoi: Fine-grained anchors for human-object interaction detection. _arXiv preprint arXiv:2301.04019_ , 2023. 3 

- [43] Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. Peft: State-of-the-art parameter-efficient fine-tuning methods. https://github.com/huggingface/peft, 2022. 6 

- [44] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 11–20, 2016. 2, 4, 6 

- [45] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In _Proceedings of the IEEE/cvf conference on computer vision and pattern recognition_ , pages 3195–3204, 2019. 1 

- [46] Giovanni Paolini, Ben Athiwaratkun, Jason Krone, Jie Ma, Alessandro Achille, Rishita Anubhai, Cicero Nogueira dos Santos, Bing Xiang, and Stefano Soatto. Structured prediction as translation between augmented natural languages. In _ICLR_ , 2021. 3 

- [47] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In _Proceedings of the IEEE international conference on computer vision_ , pages 2641–2649, 2015. 2 

- [48] Sarah Pratt, Mark Yatskar, Luca Weihs, Ali Farhadi, and Aniruddha Kembhavi. Grounded situation recognition. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16_ , pages 314–332. Springer, 2020. 2, 3, 5, 6, 9 

- [49] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pages 8748–8763. PMLR, 2021. 1 

- [50] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In _Proceedings of the IEEE international conference on computer vision_ , pages 618–626, 2017. 6, 10 

- [51] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 8430–8439, 2019. 2 

- [52] Peng Shi and Jimmy Lin. Simple bert models for relation extraction and semantic role labeling. _arXiv preprint arXiv:1904.05255_ , 2019. 3 

- [53] Takashi Shibuya and Eduard Hovy. Nested named entity recognition via second-best sequence learning and decoding. _Transactions of the Association for Computational Linguistics_ , 8:605–620, 2020. 3 

- [54] Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. Flava: A foundational language and vision alignment model. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15638–15650, 2022. 1, 7 

- [55] Matteo Stefanini, Marcella Cornia, Lorenzo Baraldi, Silvia Cascianelli, Giuseppe Fiameni, and Rita Cucchiara. From show to tell: A survey on deep learning-based image captioning. _IEEE transactions on pattern analysis and machine intelligence_ , 45(1):539–559, 2022. 1 

- [56] Sanjay Subramanian, William Merrill, Trevor Darrell, Matt Gardner, Sameer Singh, and Anna Rohrbach. Reclip: A strong zero-shot baseline for referring expression comprehension. In _ACL_ , 2022. 1, 2, 3, 4, 5, 6, 7, 10 

- [57] Kaihua Tang, Hanwang Zhang, Baoyuan Wu, Wenhan Luo, and Wei Liu. Learning to compose dynamic tree structures for visual contexts. In _CVPR_ , 2019. 3 

- [58] Jerry Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao Liu, Da Huang, Denny Zhou, et al. Larger language models do in-context learning differently. _arXiv preprint arXiv:2303.03846_ , 2023. 2 

- [59] Size Wu, Wenwei Zhang, Sheng Jin, Wentao Liu, and Chen Change Loy. Aligning bag of regions for openvocabulary object detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15254–15264, 2023. 2 

- [60] Wei Wu, Fei Wang, Arianna Yuan, Fei Wu, and Jiwei Li. Corefqa: Coreference resolution as query-based span prediction. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 6953–6963, 2020. 3 

- [61] Binfeng Xu, Xukun Liu, Hua Shen, Zeyu Han, Yuhan Li, Murong Yue, Zhiyuan Peng, Yuchen Liu, Ziyu Yao, and Dongkuan Xu. Gentopia: A collaborative platform for toolaugmented llms. _arXiv preprint arXiv:2308.04030_ , 2023. 2 

- [62] Danfei Xu, Yuke Zhu, Christopher B Choy, and Li Fei-Fei. Scene graph generation by iterative message passing. In _CVPR_ , 2017. 3 

- [63] Jingkang Yang, Yi Zhe Ang, Zujin Guo, Kaiyang Zhou, Wayne Zhang, and Ziwei Liu. Panoptic scene graph generation. In _European Conference on Computer Vision_ , pages 178–196. Springer, 2022. 3 

- [64] Yuan Yao, Ao Zhang, Zhengyan Zhang, Zhiyuan Liu, TatSeng Chua, and Maosong Sun. Cpt: Colorful prompt tuning for pre-trained vision-language models. _arXiv preprint arXiv:2109.11797_ , 2021. 6, 10 

- [65] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In _Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14_ , pages 69–85. Springer, 2016. 2, 4, 6 

- [66] Licheng Yu, Zhe Lin, Xiaohui Shen, Jimei Yang, Xin Lu, Mohit Bansal, and Tamara L Berg. Mattnet: Modular attention network for referring expression comprehension. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 1307–1315, 2018. 6, 7 

- [67] Seonghoon Yu, Paul Hongsuck Seo, and Jeany Son. Zeroshot referring image segmentation with global-local context features. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 19456– 19465, 2023. 1, 2 

- [68] Zhou Yu, Jun Yu, Yuhao Cui, Dacheng Tao, and Qi Tian. Deep modular co-attention networks for visual question answering. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 6281–6290, 2019. 1 

- [69] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why visionlanguage models behave like bags-of-words, and what to do about it? In _The Eleventh International Conference on Learning Representations_ , 2022. 2, 3, 5 

- [70] Rowan Zellers, Mark Yatskar, Sam Thomson, and Yejin Choi. Neural motifs: Scene graph parsing with global context. In _CVPR_ , 2018. 3 

- [71] Aixi Zhang, Yue Liao, Si Liu, Miao Lu, Yongliang Wang, Chen Gao, and Xiaobo Li. Mining the benefits of two-stage and one-stage hoi detection. In _NeurIPS_ , 2021. 3 

- [72] Frederic Z Zhang, Dylan Campbell, and Stephen Gould. Spatially conditioned graphs for detecting human-object interactions. In _ICCV_ , 2021. 

- [73] Frederic Z Zhang, Dylan Campbell, and Stephen Gould. Efficient two-stage detection of human-object interactions with a novel unary-pairwise transformer. In _CVPR_ , 2022. 3 

- [74] Hanwang Zhang, Zawlin Kyaw, Jinyang Yu, and Shih-Fu Chang. Ppr-fcn: Weakly supervised visual relation detection via parallel pairwise r-fcn. In _ICCV_ , 2017. 3 

- [75] Haotian Zhang, Pengchuan Zhang, Xiaowei Hu, Yen-Chun Chen, Liunian Li, Xiyang Dai, Lijuan Wang, Lu Yuan, JenqNeng Hwang, and Jianfeng Gao. Glipv2: Unifying localization and vision-language understanding. _Advances in Neural Information Processing Systems_ , 35:36067–36080, 2022. 2 

- [76] Yuhao Zhang, Victor Zhong, Danqi Chen, Gabor Angeli, and Christopher D Manning. Position-aware attention and supervised data improve slot filling. In _Conference on Empirical Methods in Natural Language Processing_ , 2017. 3 

- [77] Yong Zhang, Yingwei Pan, Ting Yao, Rui Huang, Tao Mei, and Chang-Wen Chen. Exploring structure-aware transformer over interaction proposals for human-object interaction detection. In _CVPR_ , 2022. 3 

- [78] Yiwu Zhong, Jing Shi, Jianwei Yang, Chenliang Xu, and Yin Li. Learning to generate scene graph from natural language supervision. In _ICCV_ , 2021. 3 

- [79] Fangrui Zhu, Yiming Xie, Weidi Xie, and Huaizu Jiang. Diagnosing human-object interaction detectors. _arXiv preprint arXiv:2308.08529_ , 2023. 3 

