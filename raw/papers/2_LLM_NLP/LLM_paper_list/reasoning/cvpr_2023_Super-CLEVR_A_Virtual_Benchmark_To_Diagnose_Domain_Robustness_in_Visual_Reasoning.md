# **Super-CLEVR: A Virtual Benchmark to Diagnose Domain Robustness in Visual Reasoning** 

Zhuowan Li[1] Xingrui Wang[2] Elias Stengel-Eskin[1] Adam Kortylewski[3] _[,]_[ 4] Wufei Ma[1] Benjamin Van Durme[1] Alan Yuille[1] 1 Johns Hopkins University 2 University of Southern California 3 Max Planck Institute for Informatics 4 University of Freiburg 

## **Abstract** 

_Visual Question Answering (VQA) models often perform poorly on out-of-distribution data and struggle on domain generalization. Due to the multi-modal nature of this task, multiple factors of variation are intertwined, making generalization difficult to analyze. This motivates us to introduce a virtual benchmark, Super-CLEVR, where different factors in VQA domain shifts can be isolated in order that their effects can be studied independently. Four factors are considered: visual complexity, question redundancy, concept distribution and concept compositionality. With controllably generated data, Super-CLEVR enables us to test VQA methods in situations where the test data differs from the training data along each of these axes. We study four existing methods, including two neural symbolic methods NSCL [45] and NSVQA [59], and two non-symbolic methods FiLM [50] and mDETR [29]; and our proposed method, probabilistic NSVQA (P-NSVQA), which extends NSVQA with uncertainty reasoning. P-NSVQA outperforms other methods on three of the four domain shift factors. Our results suggest that disentangling reasoning and perception, combined with probabilistic uncertainty, form a strong VQA model that is more robust to domain shifts. The dataset and code are released at https://github.com/Lizw14/ Super-CLEVR._ 

## **1. Introduction** 

Visual question answering (VQA) is a challenging task that assesses the reasoning ability of models to answer questions based on both visual and linguistic inputs. Current VQA methods are typically developed on standard benchmarks like VQAv2 [16] or GQA [25], with the implicit assumption that testing data comes from the same underlying distribution as training data. However, as has been widely studied in computer vision [15, 36, 51], algorithms trained on one domain often fail to generalize to other domains. 

Moreover, having learned the distributional prior of training data, models often struggle on out-of-distribution tests. This has been studied in VQA from the perspective of domain transfer [8,57,62], dataset bias [2,11,48], counter-factual diagnosis [9, 47], and out-of-distribution benchmarking [30]. 

The multi-modal nature of VQA gives rise to multiple intertwined factors of variation, making domain shift an especially difficult problem to study. For example, [8] suggests that VQA domain shifts are a combination of differences in images, questions or answers; and [39] reveals a gap between synthetic and real VQA datasets by differences in the over-specification of questions and the underlying distribution of concepts. However, despite a wealth of research on domain generalization in VQA [3,26,57,62], there is no systematic analysis of the contributing factors in domain shifts. 

To this end, we introduce a virtual benchmark, SuperCLEVR, which enables us to test VQA algorithms in situations where the test data differs from the training data. We decompose the domain shift into a set of isolated contributing factors, so that their effects can be diagnosed independently. We study four factors: visual complexity, question redundancy, concept distribution, and concept compositionality. These are illustrated in Fig. 1 and described in Sec. 3.1. With controllable data generation using our SuperCLEVR virtual benchmark, we are able to isolate the different factors in VQA domain shifts so that their effects can be studied independently. Compared with the original CLEVR dataset [27], Super-CLEVR contains more complicated visual components and has better controllability over the domain shift factors. As shown in Fig. 1, the Super-CLEVR dataset contains images rendered from 3D graphical vehicle models in the UDA-Part dataset [40], paired with questions and answers automatically generated from templates. The objects and questions are sampled based on the specified underlying probability distribution, which can be controlled to produce distribution shifts in different factors. 

With Super-CLEVR, we diagnose the domain robustness of current VQA models. Four representative models are studied: for the classic two-stream feature fusing 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning_images/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Domain A<br>Visual<br>Complexity<br>Super-CLEVR easy middle hard<br>-  redundancy “What color is the bus?”<br>Question<br>standard “What color is the large bus?”<br>Redundancy<br>+  redundancy “What color is the large bus behind the cyan car?”<br>Concept<br>“What color is the bus?” Distribution<br>balanced  unbalanced  long-tail<br>Concept<br>Compositionality<br>Domain B<br>well-composed   correlated<br>**----- End of picture text -----**<br>


Figure 1. We decompose VQA domain shifts into four contributing factors: visual complexity, question redundancy, concept distribution and concept compositionality. The domain shifts along each factor can be independently studied with the proposed Super-CLEVR dataset. 

architecture, we choose FiLM [50]; for a large-scale pretrained model we take mDETR [29]; we use NSCL [45] and NSVQA [59] as representative neuro-symbolic methods. We observe that all these models suffer from domain shifts to varying degrees of sensitivity. We analyze each factor separately to examine the influence of different model designs. Specifically, we find that the step-bystep design of neural modular methods enhances their robustness to changes in question redundancy compared with non-modular ones; however, the non-modular models are more robust to visual complexity. Furthermore, thanks to its decomposed reasoning and perception, NSVQA is more robust to concept distribution shifts. 

While existing models suffer from domain shifts with different characteristics, we make a technical improvement over NSVQA which enables it to significantly outperform existing models on three of the four factors. In particular, we inject probabilities into the deterministic symbolic executor of NSVQA, empowering it to take into account the uncertainty of scene understanding. We name our model _probabilistic NSVQA_ (P-NSVQA), and show that its performance improvement in both the in-domain and outof-domain settings. With superior results of P-NSVQA, we suggest that disentangling reasoning from vision and language understanding, together with probabilistic uncertainty, gives a strong model that is robust to domain shifts. 

Our contributions are as follows. (1) We introduce the Super-CLEVR benchmark to diagnose VQA robustness along four different factors independently. This benchmark can also be used for part-based reasoning. (2) We enhance a neural-symbolic method by taking the uncertainty of visual understanding into account in reasoning. (3) We conduct detailed analysis of four existing methods, as well as our novel approach to study the influence of model designs on 

distinct robustness factors. We conclude that disentangled reasoning and perception plus explicit modeling of uncertainty leads to a more robust VQA model. 

## **2. Related work** 

**Visual question answering (VQA).** Popular VQA methods fall into three categories. Two-stream methods extract features for image and questions using CNN and LSTM respectively, then enable interaction between the two modalities with different feature fusing methods [4, 13, 24, 32, 37, 50, 60]. Neural symbolic methods, on the other hand, use a parse-then-execute pipeline where the question is parsed into a functional program, which is then executed on the image using neural modules [45, 59]. Recently, transformers-based models have achieved impressive performance on various vision-and-language tasks by pretraining on large scale dataset then finetuning for downstream tasks [29, 38, 44, 55, 63]. We choose FiLM [50], NSCL [45] and mDETR [29] as category representatives. 

**VQA datasets.** Datasets containing real images and human-written questions have been widely used to benchmark VQA models, _e.g_ . VQA [5], VQAv2 [16], Visual 7w [65], VizWiz [18], Visual Genome [35], COCO QA [52], etc. However, subsequent work has revealed the strong prior and bias in those datasets which might be exploited by models to correctly predict the answers without reasoning [1, 16, 17, 30, 31, 47, 48]. Attempts to address this problem include better balancing datasets [2] and creating counterfactual examples [9, 11]. To assess a model’s true reasoning ability, the CLEVR dataset [27] proposes to generate complex multi-step questions on synthetic images, which is then extended to various vision-and-language tasks [6,23,34,41,53,58,61]. The GQA dataset [25] extends CLEVR-style questions to real images. Our benchmark is 

distinct from existing ones because we introduce more complex visual scenes into CLEVR and provide controllability to study domain robustness on isolated factors. 

**Domain shift in VQA.** Domain shift is a long-standing challenge in computer vision, explore in prior works in domain adaptation [14, 15, 22, 43] and domain generalization [36, 51]. Recent works have focused on domain shifts in VQA. [8, 57] improves model adaptation between datasets by feature learning. [62] analyze domain shifts between nine popular VQA datasets and proposes an unsupervised method to bridge the gaps. [39] generalize symbolic reasoning from synthetic to real dataset. [3] introduce a questionanswer generation module that simulates the domain shifts. [26] propose a training scheme X-GGM to improve out-ofdistribution generalization. [6] assess models generalization on the CLOSURE of linguistic components. In contrast to prior works we study each of the different domain shift factors independently with our virtual benchmark. 

## **3. Super-CLEVR** 

## **3.1. Motivation: domain shift factors** 

**Visual complexity.** A major difference between different VQA datasets is visual complexity. For example, in the CLEVR dataset, objects are simple, atomic shapes while in real-world data, objects are more complex and have hierarchical parts. While hard to quantify, visual complexity is related to various factors, such as object variety, object size, background, texture, lighting, occlusion, view point, etc. In our work, we control visual complexity by introducing more challenging objects that can have distinct attributes associated with their parts, and by optionally pasting various textures onto objects. Examples of generated images with different complexity levels are shown in Fig. 1. 

**Question redundancy.** Question redundancy refers to the amount of over-specified or redundant information in the question, which can be in the form of either _attributes_ or _relationships_ . For example, in Fig. 1, “what color is the large bus behind the cyan car”, large ( _attributes_ ) and behind the cyan car ( _relationship_ ) are redundant because there is only one bus in the image. As observed in linguistics and cognitive science [12, 33, 49, 54], human speakers may include over-specified information when identifying a target object, which has also been studied in referring expression generation [46]. For VQA, as analyzed in [39], a significant difference between synthetic and real datasets is that real questions contain some redundant information, which sometimes is a distraction leading to model prediction errors. Therefore, in this work, we generate questions with different redundancy levels and study the effect of question redundancy on model behaviors. 

**Concept distribution.** The distributions of _concept_ , _i.e_ . _objects_ ( _e.g_ . car) and _attributes_ ( _e.g_ . large), are distinct 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning_images/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning.pdf-0003-07.png)


**----- Start of picture text -----**<br>
Object with parts:<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning_images/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning.pdf-0003-08.png)


**----- Start of picture text -----**<br>
car bus airplane<br>bike motorcycle<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning_images/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning.pdf-0003-09.png)


**----- Start of picture text -----**<br>
Texture:  dotted, checkered, stripped, none<br>Color:  green, gray, brown, yellow, red, purple, cyan, blue<br>Size:  large, small<br>Material:  rubber, metal<br>**----- End of picture text -----**<br>


Figure 2. Super-CLEVR contains 21 vehicle models belonging to 5 categories, with controllable attributes. 

across different VQA datasets. For example, while colors are well-balanced in CLEVR dataset, in the GQA dataset, the color distribution is long-tailed where “white” appears _>_ 50 times more frequently than “gold”. Long-tailed distributions have been a challenge in many computer vision tasks [20, 28, 42, 56, 64]. In VQA, the long-tailed concept distribution not only hinders the learning of infrequent concepts due to few training samples, but also introduces strong biases and priors in the dataset that may mislead the models. For example, “tennis” is the correct answer to most questions with “what sport is ...” [2]. With strong priors in data, it is hard to assess the true reasoning capacity of current models. While previous works address this problem by carefully re-balancing datasets [2], in our work, we controllably vary the concept distribution in our dataset and study model robustness to concept distribution shifts. 

**Concept compositionality.** Concept compositionality refers to how different concepts (shapes, attributes) compose and co-occur with each other, _e.g_ . roses are usually red while violets are usually blue [30]. Concept compositionality can be viewed as a conditional concept distribution in the context of other concepts. Shifts in concept compositionality impede the generalization of VQA models. For example, the model may fail to recognize a _green_ banana because most bananas are _yellow_ in the training data [7]. Previous works evaluate the out-of-distribution performance by collecting counterfactual testing examples [47]. In our work, we control the compositionality of _shapes_ and _colors_ with an intuitive motivation: if, for example, in the training data, bicycles are red and cars are blue, will the models be able to recognize blue bicycles and red cars in testing? 

## **3.2. Dataset generation** 

Super-CLEVR follows a similar data generation pipeline as CLEVR, but with more complex visual components and better control of domain gap factors. We describe the generation procedure below. 

**Objects with parts.** To improve the visual complexity of CLEVR scenes, we replace the simple shapes ( _e.g_ ., _cube_ , _sphere_ ) in CLEVR dataset with vehicles from UDAPart dataset [40]. There are 21 vehicle models, belonging to 5 categories: _car_ , _motorbike_ , _aeroplane_ , _bus_ , and _bicycle_ . Each 3D model comes with part annotations, _e.g_ ., _left front wheel_ or _left right door_ for _car_ . Examples for the vehicle models are shown in Fig. 2. We remove or merge small parts from the original annotations to avoid severe difficulty in visual understanding. The full object and parts list is in the supplementary material. 

**Attributes.** Besides the attributes in the original CLEVR dataset, _i.e_ . _color_ , _material_ , _size_ , we optionally add _texture_ as an additional attribute to increase visual complexity. Note that in order to enable part-based questions, the attributes (color or material) of object parts can be different from that of the object. For example, a blue car can have a red wheel or a green door. In this case, the attribute of the holistic object refers to the attribute of its main body ( _e.g_ . the blue car has blue frame). 

**Scene rendering.** Following CLEVR, each scene contains 3 to 10 objects. The objects are placed onto the ground plane with random position and orientation. When placing the objects, we ensure that the objects do not overlap with each other and we avoid severe occlusion by thresholding the number of visible pixels for each object. Random jitters are added to lamp and camera positions. When rendering, we also save the ground-truth bounding boxes and segmentation masks for each of the objects and their parts, which are required when training some of the models. 

**Question generation.** Super-CLEVR follows similar question generation pipeline in CLEVR, which instantiates question templates using the underlying reasoning program that can be operated on the scene graph. For example, the program select ~~s~~ hape(truck) _→_ query ~~c~~ olor( _·_ ) can be instantiated as question “what is the color of the truck”. Therefore, redundancy level of questions can be controlled by removing or adding redundant reasoning steps in the underlying reasoning program. 

## **3.3. Controlling the dataset** 

To study domain generalization, we generate several variants of the dataset for each of the domain shift factors. The variants of the datasets serve as different data domains to test the model robustness. Here we describe the method for controllably generating the dataset variants. 

**Visual complexity.** We generate three variants of the dataset with different levels of visual complexity: _easy_ , _mid_ (middle) and _hard_ . The only difference between the 3 versions is visual complexity: for the easy version, objects with different sizes, colors and materials are placed into the scene; for the middle version, we choose 3 parts on each object that are visible and randomly change their attributes; 

for the hard version, we further add random textures to the objects and parts. An example of the 3 dataset versions can be found in Fig. 1. Note that the scene layout and the questions are shared, so that the influence of visual complexity can be isolated and studied independently. 

**Question redundancy.** Three variants of the dataset with different redundancy levels are generated: _rd-_ , _rd_ (default), _rd+_ . By default ( _rd_ ), as in original CLEVR dataset, the questions contain some redundant _attributes_ resulting from random sampling, while all redundant _relationships_ are removed. In _rd-_ , we also remove all redundant attributes from the questions, leading to no redundancy in the questions. In _rd+_ , we add all possible attributes and relationships into the question, so that questions contain a high level of redundancy. For all the variants, the questions are ensured to be valid. 

**Concept distribution.** We generate three dataset variants with different concept distributions: _bal_ (balanced), _slt_ (slightly unbalanced) and _long_ (long-tail distributed). More specifically, we change the distribution of _shapes_ , _colors_ and _materials_ while the distribution of _size_ is kept fixed in order to keep visual complexity consistent, since objects with smaller sizes are visually harder to recognize. By default ( _bal_ ), the shapes and attributes are randomly sampled, leading to a balanced distribution. For _slt_ and _long_ , the concept distribution **d** is generated by _di_ = _a[−][i]_ , where _i_ is the index of the concept. _a_ is a hyper-parameter controlling the length of the tail. A larger _a_ leads to more imbalanced distribution and _a_ = 1 leads to flat distribution (cf. Fig. 1). For _slt_ , _a_ = 1 _._ 3; for _long_ , _a_ = 2 _._ 0. In addition, to better analyze the performance on the frequent and rare concepts, we generate three variants for testing purpose only: _head_ (frequent concepts in the long-tail distribution), _tail_ (infrequent/rare concepts), and _oppo_ (opposite to the longtail distribution). We test each model on those three variants to analyze the performance on concepts with different degrees of frequency. 

**Concept compositionality.** We generate 3 versions of the dataset, _co-0_ , _co-1_ and _co-2_ , with different compositions of the 21 _shapes_ (from 5 categories) and the 8 _colors_ . The compositionality of the dataset is controlled with the codistribution matrix _M ∈_ R[21] _[×]_[8] , where each entry _Mij_ is the probability an object of the _i_ -th shape has the _j_ -th color. Entries in each row of _M_ sum up to 1. In the version _co-0_ , _M_ is a flat matrix so that the shapes and colors are randomly composed. In _co-1_ , each shape in one category has a different color distribution, _e.g_ . truck and sedan, while shapes from different categories may share the same color distribution _e.g_ . sedan and airliner. Oppositely, in _co-2_ , we make the shapes in same category have the same color distribution, whiles shapes from different categories have different distributions. The motivation is that since shapes from the same category are visually similar, the difference in _co-1_ 

and _co-2_ will help analyze the difference in model predictions on visually similar objects and dissimilar objects when composed with different color distributions. 

**Dataset Statistics.** Every dataset variant contains 30k images, including 20k for training, 5k for validation and 5k for testing. Each image is paired with 10 object-based and 10 part-based questions. By _default_ , the dataset refers to the version with _mid_ visual complexity level ( _i.e_ . objects are untextured and has up to 3 parts with distinct attributes), _rd_ redundancy level, balanced ( _bal_ ) concept distribution and random ( _co-0_ ) compositionality. More dataset statistics are in supplementary materials. 

## **4. Evaluated methods** 

## **4.1. Simple baselines** 

**Random, Majority.** These simple baselines pick a random or the most frequent answer for each question type in the training set as the predicted answer. 

**LSTM.** This question-only baseline encodes the question word embeddings with LSTM [21] and predicts answer with an MLP on top of the final hidden states of the LSTM. 

**CNN+LSTM.** The image is represented with features extracted by CNN and question is encoded by LSTM. An MLP predicts answer scores based on the concatenation of image and question features. 

## **4.2. Existing models** 

**FiLM.** We choose _Feature-wise Linear Modulation_ [50] as a representative of classic two-stream feature merging methods. The question features extracted with GRU [10] and image features extracted with CNN are fused with the proposed FiLM module. 

**mDETR.** mDETR [29] is a transformer-based detector trained to detects objects in an image conditioned on a text query. The model is pretrained with 1.3M image and text pairs and can be finetuned for various downstream tasks like referring expression understanding or VQA. 

**NSCL.** The _Neuro-Symbolic Concept Learner_ [45] is a representative neural symbolic method. NSCL executes neural modules on the scene representation based on the reasoning program, during which the modules learns embeddings of each concept with the answer supervision. 

**NSVQA.** _Neural-Symbolic VQA_ [59] is a neural symbolic method composed of three components: A scene parser (Mask-RCNN [19]) that segments an input image and recovers a structural scene representation, a question parser that converts a question from natural language into a program and a program executor that runs the program on the structural scene representation to obtain the answer. Notably, compared to NSCL, the individual components of NSVQA can be learned separately, hence, for example the scene parser can be learned from data that does not necessarily have Visual-Question annotations. 

## **4.3. Probabilistic NSVQA (P-NSVQA)** 

Since the program executor in NSVQA is a collection of deterministic, generic functional modules, it can be augmented with a probabilistic reasoning process that takes into account the confidence of the predictions of the scene parser. This allows the model to execute the program that has the largest joint likelihood, instead of only taking the maximal likelihood execution at each step of the program. The experiment results demonstrate a significant performance improvement of this probabilistic approach over the deterministic NSVQA model proposed in [59]. 

In particular, we interpret the confidence of the MaskRCNN output as a likelihood function for all detected object classes _pobject_ and their attributes _patt_ . Moreover, we define a likelihood _pspatial_ for the spatial relations between objects (behind, in front, left, right) that is proportional to the distance between the centers of two bounding boxes. Given a reasoning program containing multiple reasoning steps, we execute each step based on the scene parsing likelihood and produce an step-wise output with confidence. Finally, we use a factorized model, multiplying the output for all the steps to get the final answer prediction. We refer readers to the Appendix for more details. 

## **4.4. Implementation details** 

Training mDETR requires ground-truth grounding of question tokens to image regions, which is available in Super-CLEVR. NSCL requires bounding box of objects, which can predicted using a trained Faster RCNN, and the reasoning program, which can be parsed using a trained parser. Similarly, ground-truth programs are used for training NSVQA and P-NSVQA. Note that we empirically find that the question-to-program parsing is a relatively easy task ( _>_ 99% accuracy using a simple LSTM), so we focus more on models’ reasoning ability in our analysis. 

Unless specified, the models are trained with default setting as in the official implementation. FiLM is trained for 100k iterations with batch size 256. mDETR is trained for 30 epochs with batch size 64 using 2 GPUs for both the grounding stage and the answer classification stage. NSCL is trained for 80 epochs with batch size 32. For NSVQA and P-NSVQA, we first train the object parser (Mask RCNN [19]) for 30k iterations with batch size 16, then train the attribute extraction model (using the Res50 backbone) for 100 epochs with batch size 64. For P-NSVQA, when counting the objects or determining whether objects exist in the scene, we use a threshold (0.7) to obtain the final selected objects. Early stopping is used based on validation accuracy. All the models are trained with 200k questions. We repeat experiments on _default_ split for 3 times with different random seeds and get _std_ of _±_ 0 _._ 10 (P-NSVQA) and _±_ 0 _._ 40 (NSVQA), showing the statistical significance of our results, then we only run other experiments once. 

## **5. Results and analysis** 

In this section, we first show evaluation results for indomain setting, then provide results and analysis on out-ofdomain evaluation. Finally, we describe additional studies and future works. 

## **5.1. In-domain results** 

In-domain evaluation refers to the setting where the training and testing data come from the same domain (the _default_ dataset variant in this case). We compare the indomain results on Super-CLEVR and CLEVR. The results are shown in Fig. 3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning_images/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning.pdf-0006-04.png)


Figure 3. Comparison of models’ accuracy on Super-CLEVR and the original CLEVR dataset. 

For all the models, the performance is lower on SuperCLEVR than CLEVR, suggesting that Super-CLEVR is a more challenging benchmark. The scenes with vehicles are much harder visually for the models to understand compared with the simpler shapes from CLEVR. Note that the performance gap on two datasets for simple baselines (Random, Majority, LSTM, CNN+LSTM) is smaller than for the other better models. This is because Super-CLEVR contains more object types, and therefore the performance of simply guessing is lower than on CLEVR. 

When comparing the performance of different models, we find that the neural modular methods, _i.e_ . NSCL, NSVQA, P-NSVQA, perform much better than nonmodular ones. This is not surprising given their nearly perfect performance on the original CLEVR dataset, which shows its strong ability to model synthetic images. The large-scale pretrained grounding model mDETR, which is a leading model on both real and synthetic images, also achieves good performance (82.7%) on Super-CLEVR. The two-stream method FiLM does not achieve very strong performance (53.2%), but is still much better than the other simple baselines. 

Our proposed P-NSVQA outperforms all the other models. In particular, on Super-CLEVR, it outperforms its de- 

terministic counterpart, NSVQA, by 2.96%. This shows the advantage of taking into account the probabilities when the scenes are challenging thus the model’s uncertainty of predictions can be utilized. 

## **5.2. Out-of-domain results** 

In this section, we train and test the five models (FiLM, mDETR, NSCL, NSVQA and P-NSVQA) on different dataset variants, and diagnose their domain robustness on each of the four domain shift factors. Please refer to Sec. 3.3 for a description of different variants. The validation accuracy is used for analysis here and the results are shown in Tab. 1. 

All the methods suffer from domain shifts. The results show that the best performance mostly occurs in situations where the model is tested on the same dataset variant as it is trained on, _i.e_ . the bold or underlined numbers fall mostly on the diagonals in Tab. 1. 

We compare the domain robustness of the five models by measuring the relative performance decrease when the testing data differs from the training data, _i.e_ . smaller performance drop on different testing domains means better robustness. Based on this intuition, for easier understanding of Tab. 1, we propose a measurement metric for domain robustness named _**Relative Degrade (RD)**_ to better analyze the results. We define _Relative Degrade_ as the the percentage of accuracy decrease when the model is tested under a domain shift, _i.e_ . the accuracy drop divided by the in-domain accuracy. Specifically, if a model gets accuracy _a_ under in-domain testing ( _i.e_ . testing with the same dataset variant as training) and accuracy _b_ under out-ofdomain testing ( _i.e_ . testing with a different dataset variant from training), then _RD_ = ( _a − b_ ) _/a_ . Since we train each model on three data variants, the _RD_ ’s for the three models are averaged to measure its domain robustness.[1] 

Tab. 2 shows the _Relative Degrade_ of the five models on the four factors. We see that P-NSVQA outperforms other models by a significant margin on three of the four factors, indicating that it has better overall domain robustness. In the following, we take a closer look at the results on each of the factors separately, to diagnose the influence of different model designs. 

**Question redundancy.** Neural modular methods are much more robust to question redundancy shifts than nonmodular ones. The relative degrades for modular methods are less than 2%, while one-modular ones degrade for around 20%. Due to the step-by-step design of the reasoning in modular methods, each reasoning step is independent of the others so that the models are less likely to learn the spurious correlation between question and answers. There- 

> 1For concept distributions, we compute relative degrade with a slight change: we compute the accuracy drop from _head_ to _tail_ and the drop from _long_ to _oppo_ , take their average, and divide by the accuracy on _bal_ . 

||||**FiLM**|||**mDETR**|**mDETR**|**NSCL**|**NSCL**|**NSCL**|**NSCL**|**NSCL**|**NSCL**|**NSVQA**|**NSVQA**|**Prob NSVQA**|**Prob NSVQA**|**Prob NSVQA**|**Prob NSVQA**|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||**Visual Complexity**||||||||||||
|||easy|mid|hard||easy|mid|hard||easy<br>mid<br>hard||||easy|mid|hard|easy|mid|hard|
|easy<br>mid<br>hard||**59.96**<br>**57.41**<br>**55.95**|53.95<br>53.28<br>53.11|50.66<br>50.18<br>50.47||**93.36**<br>**83.34**<br>79.71|84.30<br>82.36<br>79.94|82.97<br>81.27<br>**80.71**||**95.13**<br>92.31<br>90.81<br>84.5<br>**89.10**<br>86.33<br>76.85<br>78.66<br>**85.08**||||**95.19**<br>81.99<br>73.11|94.19<br>92.80<br>79.71|94.09<br>**96.76**<br>**93.78**<br>86.25<br>**92.65**<br>79.81||95.98<br>**95.76**<br>86.47|96.37<br>95.11<br>**95.36**|
|||||||||**Question Redundancy**||||||||||||
|||rd-|rd|rd+||rd-|rd|rd+||rd-|rd|rd+||rd-|rd|rd+|rd-|rd|rd+|
|rd-<br>rd<br>rd+||51.42<br>50.39<br>46.14|52.54<br>53.28<br>52.30|**53.51**<br>**54.78**<br>**71.47**||**83.94**<br>**82.77**<br>78.48|80.37<br>82.36<br>84.05|66.28<br>70.36<br>**90.42**||88.64<br>88.45<br>87.94|88.82<br>89.10<br>88.34|**90.33**<br>**91.45**<br>**91.16**||**92.95**<br>91.19<br>91.38|92.94<br>**92.78**<br>91.96|92.67<br>95.66<br>92.14<br>94.87<br>**92.80**<br>94.88||**95.72**<br>**95.72**<br>95.47|95.43<br>95.43<br>**95.72**|
|||||||||**Concept Distribution**||||||||||||
|||bal|slt|long||bal|slt|long||bal<br>slt||long||bal|slt|long|bal|slt|long|
|bal<br>long||50.47<br>49.43|53.04<br>54.75|**54.35**<br>**62.96**||**80.71**<br>79.06|75.79<br>80.29|74.54<br>**90.66**||**85.08**<br>83.79<br>85.33<br>89.42||75.10<br>**91.10**||**92.65**<br>92.73|90.82<br>**93.38**|83.74<br>92.53|**95.36**<br>96.31|94.89<br>**96.32**|89.88<br>95.25|
|head<br>tail<br>oppo||48.60<br>**51.80**<br>**49.06**|58.06<br>48.70<br>48.93|**61.60**<br>50.08<br>46.68||80.75<br>**81.50**<br>**79.13**|79.60<br>70.88<br>68.37|**87.46**<br>60.94<br>56.98||84.58<br>88.39<br>**86.10**<br>80.27<br>**85.07**<br>77.86||**90.19**<br>60.55<br>55.14||93.87<br>**90.26**<br>**91.22**|**94.82**<br>89.20<br>88.65|92.48<br>96.42<br>75.32<br>**94.08**<br>71.32<br>**95.76**||**96.80**<br>93.20<br>94.09|95.92<br>82.68<br>79.74|
|||||||||**Concept Compositionality**||||||||||||
|||co-0|co-1|co-2||co-0|co-1|co-2|co-0||co-1|co-2||co-0|co-1|co-2|co-0|co-1|co-2|
|co-0<br>co-1<br>co-2||53.28<br>52.41<br>52.96|**57.00**<br>**60.57**<br>57.37|56.1<br>56.67<br>**60.53**||**83.36**<br>79.46<br>80.03|77.03<br>82.45<br>77.41|82.43<br>**83.93**<br>**87.24**|**89.1**<br>78.89<br>78.40||82.52<br>**87.18**<br>81.55|83.77<br>84.2<br>**88.84**||**92.80**<br>78.74<br>77.85|90.11<br>89.99<br>89.28|91.59<br>**90.67**<br>**92.23**|**95.76**<br>87.12<br>87.19|94.02<br>94.53<br>93.49|95.12<br>**94.78**<br>**95.61**|



Table 1. Accuracy of models trained and tested on different domains. Column headings indicate _training_ settings, while rows indicate the dataset variant for _testing_ . The best performance in each row ( _i.e_ . the best training setting) is marked in **bold** and best performance in each column ( _i.e_ . the best testing setting) is underlined. Description for different splits is in Sec. 3.3 and analysis is in Sec. 5.2. 

||**Visual**|**Redund.**|**Dist.**|**Comp.**|
|---|---|---|---|---|
|**FiLM**|**4.03**|21.33|28.46|9.04|
|**mDETR**|9.81|19.05|36.34|9.45|
|**NSCL**|15.57|0.92|37.44|15.40|
|**NSVQA**|17.48|1.72|20.92|11.44|
|**Prob NSVQA**|12.88|**0.84**|**13.72**|**7.00**|



Table 2. _Relative Degrade_ under domain shifts, _i.e_ . the percentage of accuracy decrease when the model is tested with domain that differs with training. Lower _RD_ means better robustness. 

fore the modular methods are less vulnerable to change in question/program length. 

**Visual complexity.** Different from our findings on question redundancy, for domain shifts in visual complexity, non-modular methods are more robust compared to modular ones. As shown in Tab. 2, while FiLM and mDETR gets less than 10% degrade, NSCL and (P-)NSVQA degrade for more than 12%. The reason might be that the simple reasoning modules in modular methods can not process the visual signals as well as the dense non-modular models. 

Comparing P-NSVQA with NSVQA, we find that in- 

jecting probability into deterministic symbolic reasoning greatly improves the robustness on visual complexity (4 _._ 04% decrease in _RD_ ). This suggests that some errors in visual understanding can be corrected and recovered by taking into account the uncertainty of visual parsing and combining the results of each reasoning step with probability. 

**Concept distribution.** While all the four existing models suffer a lot (larger than 20% _RD_ ) on domain shifts in concept distribution, we see that the symbolic method NSVQA is better than the other three (by more than 7 _._ 5%). With the disentangled reasoning and visual understanding components in NSVQA, the distribution priors in the images and the programs/answers cannot intertwine with each other, which prevent the model heavily relying on the priors. With uncertainty, we can further boost the robustness of NSVQA with a large margin (from 21% to 14% _RD_ ). 

Moreover, the head-tail results suggests that the overall accuracy, which is commonly used to measure VQA performance, should be taken with cautious. When the testing split is imbalanced, the seemingly high accuracy is misleading because the head concepts dominates the testing while the tail ones are not well-reflected. For example, for NSCL, although it gets high accuracy (91%) on the long-tailed 

data, its performance is only 60.6% on the tail concepts. In real-world datasets, the data are usually not well-balanced, which suggests the value of synthetic testing. 

**Concept compositionality.** Comparing the existing methods, we find that the non-modular methods seems to be more robust than modular methods NSCL or NSVQA. However, with uncertainty, P-NSVQA improves the result of NSVQA, which even outperforms the non-modular methods. This suggest the large potential of better robustness of modular methods by improving current models. 

In summary, while non-modular methods are more robust to visual complexity shifts, the modular symbolic methods (improved with uncertainty) are more robust on the other three factors. By disentangling reasoning with visual understanding, separately executing every each reasoning step then merging the results of the steps using probabilities based on uncertainty, our P-NSVQA outperforms all the existing models in question redundancy, concept distribution and compositionality. Therefore, we suggest that symbolic reasoning with uncertainty leads to strong VQA models that are robust to domain shifts. 

## **5.3. More analysis and future work** 

**Synthetic-to-real transfer.** We provide an additional proof-of-concept study to show that the findings drawn from Super-CLEVR dataset can transfer to real datasets. In the following experiments, we show our finding that neurosymbolic methods (NSCL, NSVQA, P-NSVQA) are more robust than mDETR on question redundancy also holds true on the real GQA dataset [25]. More precisely, we progressively removed the redundant operations from the reasoning program in GQA testdev split, and then regenerated questions using a program-to-question generator. Using the change of models’ testing accuracy as the redundant operations are removed, we can evaluate the models’ robustness towards question redundancy. The results are show in Tab. 3.[2] We observe that the performance drop of mDETR is much larger than neuro-symbolic methods as the redundant information is progressively removed, which indicates that symoblic methods have better question redundancy than mDETR on GQA dataset. This is consistent with our findings on Super-CLEVR. 

**Reasoning with part-object hierarchies.** In addition to evaluating domain generalization, Super-CLEVR can be extended for broader purposes, _e.g_ . part-based reasoning. We can ask questions like “what is the color of the front wheel of the bike?”, “what is the color of the vehicle that has a yellow wheel”, etc. Those questions require the model to correctly understanding the part-object hierarchy, which 

2For implementation of NSCL on a real-word dataset, we use the model in [39] (the version without calibration). The model accuracies on the original not-perturbed GQA testdev split are as following: mDETR (61.67%), NSCL (56.13%), NSVQA (39.58%), P-NSVQA (39.66%). 

||**0%**|**14%**|**32%**|**70%**|**91%**|**100%**|
|---|---|---|---|---|---|---|
|**mDETR**|0|-4.82|-8.46|-13.16|-13.88|-14.56|
|**NSCL**|0|-0.14|-0.34|-1.09|-1.71|-2.59|
|**NSVQA**|0|-3.47|-4.80|-7.01|-7.02|-7.02|
|**P-NSVQA**|0|-1.93|-3.15|-5.73|-5.91|-5.78|



Table 3. Accuracy drop on the GQA dataset when redundant information is progressively removed. 

is an ability that current VQA models lack. 

**Limitations.** The main limitations of our work lie in the synthetic nature of our dataset. Future efforts can be made in collecting better controlled and balanced real datasets for model diagnosis. We emphasize that the purpose of the dataset is for model diagnosis and that models should also be tested on real data. 

## **6. Conclusion** 

We diagnose domain shifts in visual reasoning using a proposed virtual benchmark, Super-CLEVR, where distinct factors can be independently studied with controlled data generation. We evaluate four existing methods and show that all of them struggle with domain shifts, highlighting the importance of out-of-domain testing. Among the evaluated methods, neural modular methods are more robust towards question redundancy. In particular, NSVQA with disentangled perception and reasoning shows better robustness towards distribution and compositionality shifts. We further propose P-NSVQA, which improves NSVQA with uncertainty in the reasoning modules. We show that P-NSVQA outperforms all the existing methods in both in-domain testing and out-of-domain testing. With detailed analysis, our study suggests that disentangling reasoning and perception, combined with probabilistic uncertainty, form a strong VQA model that is more robust to domain shifts. We hope our analysis may facilitate better understanding of strengths and weaknesses of VQA models and, more broadly, future work might explore using the Super-CLEVR benchmark for other tasks like part-based reasoning. 

## **Acknowledgements** 

We would like to thank Nils Holzenberger, Kate Sanders, Chenxi Liu, Zihao Xiao, Qing Liu, Reno Kriz, and David Etter for their helpful comments and suggestions, as well as the anonymous reviewers. Zhuowan Li is supported by ONR N00014-21-1-2812 and grant IAA80052272 from the Institute for Assured Autonomy at JHU. Elias StengelEskin is supported by an NSF GRFP. A. Kortylewski acknowledges support via his Emmy Noether Research Group funded by the German Science Foundation (DFG) under Grant No.468670075. 

## **References** 

- [1] Aishwarya Agrawal, Dhruv Batra, and Devi Parikh. Analyzing the behavior of visual question answering models. _arXiv preprint arXiv:1606.07356_ , 2016. 2 

- [2] Aishwarya Agrawal, Dhruv Batra, Devi Parikh, and Aniruddha Kembhavi. Don’t just assume; look and answer: Overcoming priors for visual question answering. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 4971–4980, 2018. 1, 2, 3 

- [3] Arjun Akula, Soravit Changpinyo, Boqing Gong, Piyush Sharma, Song-Chun Zhu, and Radu Soricut. Crossvqa: scalably generating benchmarks for systematically testing vqa generalization. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 2148–2166, 2021. 1, 3 

- [4] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 6077–6086, 2018. 2 

- [5] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In _Proceedings of the IEEE international conference on computer vision_ , pages 2425– 2433, 2015. 2 

- [6] Dzmitry Bahdanau, Harm de Vries, Timothy J O’Donnell, Shikhar Murty, Philippe Beaudoin, Yoshua Bengio, and Aaron Courville. Closure: Assessing systematic generalization of clevr models. _arXiv preprint arXiv:1912.05783_ , 2019. 2, 3 

- [7] Remi Cadene, Corentin Dancette, Matthieu Cord, Devi Parikh, et al. Rubi: Reducing unimodal biases for visual question answering. _Advances in neural information processing systems_ , 32, 2019. 3 

- [8] Wei-Lun Chao, Hexiang Hu, and Fei Sha. Cross-dataset adaptation for visual question answering. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 5716–5725, 2018. 1, 3 

- [9] Long Chen, Xin Yan, Jun Xiao, Hanwang Zhang, Shiliang Pu, and Yueting Zhuang. Counterfactual samples synthesizing for robust visual question answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10800–10809, 2020. 1, 2 

- [10] Kyunghyun Cho, Bart Van Merri¨enboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. _arXiv preprint arXiv:1406.1078_ , 2014. 5 

- [11] Corentin Dancette, R´emi Cad`ene, Damien Teney, and Matthieu Cord. Beyond question-based biases: Assessing multimodal shortcut learning in visual question answering. _2021 IEEE/CVF International Conference on Computer Vision (ICCV)_ , pages 1554–1563, 2021. 1, 2 

- [12] William Ford and David Olson. The elaboration of the noun phrase in children’s description of objects. _Journal of Experimental Child Psychology_ , 19(3):371–382, 1975. 3 

- [13] Akira Fukui, Dong Huk Park, Daylen Yang, Anna Rohrbach, Trevor Darrell, and Marcus Rohrbach. Multimodal compact bilinear pooling for visual question answering and visual grounding. _arXiv preprint arXiv:1606.01847_ , 2016. 2 

- [14] Yaroslav Ganin and Victor Lempitsky. Unsupervised domain adaptation by backpropagation. In _International conference on machine learning_ , pages 1180–1189. PMLR, 2015. 3 

- [15] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, Franc¸ois Laviolette, Mario Marchand, and Victor Lempitsky. Domain-adversarial training of neural networks. _The journal of machine learning research_ , 17(1):2096–2030, 2016. 1, 3 

- [16] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering. In _Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2017. 1, 2 

- [17] Vipul Gupta, Zhuowan Li, Adam Kortylewski, Chenyu Zhang, Yingwei Li, and Alan Yuille. Swapmix: Diagnosing and regularizing the over-reliance on visual context in visual question answering. _arXiv preprint arXiv:2204.02285_ , 2022. 2 

- [18] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 3608– 3617, 2018. 2 

- [19] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick. Mask r-cnn. In _Proceedings of the IEEE international conference on computer vision_ , pages 2961–2969, 2017. 5 

- [20] Ruifei He, Jihan Yang, and Xiaojuan Qi. Re-distributing biased pseudo labels for semi-supervised semantic segmentation: A baseline investigation. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 6930–6940, 2021. 3 

- [21] Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory. _Neural computation_ , 9(8):1735–1780, 1997. 5 

- [22] Judy Hoffman, Eric Tzeng, Taesung Park, Jun-Yan Zhu, Phillip Isola, Kate Saenko, Alexei Efros, and Trevor Darrell. Cycada: Cycle-consistent adversarial domain adaptation. In _International conference on machine learning_ , pages 1989– 1998. PMLR, 2018. 3 

- [23] Yining Hong, Li Yi, Josh Tenenbaum, Antonio Torralba, and Chuang Gan. Ptr: A benchmark for part-based conceptual, relational, and physical reasoning. _Advances in Neural Information Processing Systems_ , 34, 2021. 2 

- [24] Drew A Hudson and Christopher D Manning. Compositional attention networks for machine reasoning. 2018. 2 

- [25] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 6700–6709, 2019. 1, 2, 8 

- [26] Jingjing Jiang, Ziyi Liu, Yifan Liu, Zhixiong Nan, and Nanning Zheng. X-ggm: Graph generative modeling for out-ofdistribution generalization in visual question answering. In 

_Proceedings of the 29th ACM International Conference on Multimedia_ , pages 199–208, 2021. 1, 3 

- [27] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2901–2910, 2017. 1, 2 

- [28] Lie Ju, Xin Wang, Lin Wang, Tongliang Liu, Xin Zhao, Tom Drummond, Dwarikanath Mahapatra, and Zongyuan Ge. Relational subsets knowledge distillation for long-tailed retinal diseases recognition. In _International Conference on Medical Image Computing and Computer-Assisted Intervention_ , pages 3–12. Springer, 2021. 3 

- [29] Aishwarya Kamath, Mannat Singh, Yann LeCun, Ishan Misra, Gabriel Synnaeve, and Nicolas Carion. Mdetr– modulated detection for end-to-end multi-modal understanding. _arXiv preprint arXiv:2104.12763_ , 2021. 1, 2, 5 

- [30] Corentin Kervadec, Grigory Antipov, Moez Baccouche, and Christian Wolf. Roses are red, violets are blue... but should vqa expect them to? In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 2776–2785, 2021. 1, 2, 3 

- [31] Corentin Kervadec, Theo Jaunet, Grigory Antipov, Moez Baccouche, Romain Vuillemot, and Christian Wolf. How transferable are reasoning patterns in vqa? _2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 4205–4214, 2021. 2 

- [32] Jin-Hwa Kim, Jaehyun Jun, and Byoung-Tak Zhang. Bilinear attention networks. _Advances in Neural Information Processing Systems_ , 31, 2018. 2 

- [33] Ruud Koolen, Martijn Goudbeek, and Emiel Krahmer. Effects of scene variation on referential overspecification. In _Proceedings of the Annual Meeting of the Cognitive Science Society_ , volume 33, 2011. 3 

- [34] Satwik Kottur, Jos´e MF Moura, Devi Parikh, Dhruv Batra, and Marcus Rohrbach. Clevr-dialog: A diagnostic dataset for multi-round reasoning in visual dialog. _arXiv preprint arXiv:1903.03166_ , 2019. 2 

- [35] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _International journal of computer vision_ , 123(1):32–73, 2017. 2 

- [36] Da Li, Jianshu Zhang, Yongxin Yang, Cong Liu, Yi-Zhe Song, and Timothy M Hospedales. Episodic training for domain generalization. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1446–1455, 2019. 1, 3 

- [37] Linjie Li, Zhe Gan, Yu Cheng, and Jingjing Liu. Relationaware graph attention network for visual question answering. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 10313–10322, 2019. 2 

- [38] Xiujun Li, Xi Yin, Chunyuan Li, Xiaowei Hu, Pengchuan Zhang, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, Yejin Choi, and Jianfeng Gao. Oscar: Object- 

semantics aligned pre-training for vision-language tasks. _ECCV 2020_ , 2020. 2 

- [39] Zhuowan Li, Elias Stengel-Eskin, Yixiao Zhang, Cihang Xie, Quan Hung Tran, Benjamin Van Durme, and Alan Yuille. Calibrating concepts and operations: Towards symbolic reasoning on real images. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , pages 14910–14919, October 2021. 1, 3, 8 

- [40] Qing Liu, Adam Kortylewski, Zhishuai Zhang, Zizhang Li, Mengqi Guo, Qihao Liu, Xiaoding Yuan, Jiteng Mu, Weichao Qiu, and Alan Yuille. Learning part segmentation through unsupervised domain adaptation from synthetic vehicles. In _CVPR_ , 2022. 1, 4 

- [41] Runtao Liu, Chenxi Liu, Yutong Bai, and Alan L Yuille. Clevr-ref+: Diagnosing visual reasoning with referring expressions. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 4185– 4194, 2019. 2 

- [42] Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, and Stella X Yu. Large-scale long-tailed recognition in an open world. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 2537–2546, 2019. 3 

- [43] Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Michael I Jordan. Conditional adversarial domain adaptation. _Advances in neural information processing systems_ , 31, 2018. 3 

- [44] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. _Advances in neural information processing systems_ , 32, 2019. 2 

- [45] Jiayuan Mao, Chuang Gan, Pushmeet Kohli, Joshua B. Tenenbaum, and Jiajun Wu. The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision. In _International Conference on Learning Representations_ , 2019. 1, 2, 5 

- [46] Margaret Mitchell, Kees Van Deemter, and Ehud Reiter. Generating expressions that refer to visible objects. In _Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 1174–1184, 2013. 3 

- [47] Yulei Niu, Kaihua Tang, Hanwang Zhang, Zhiwu Lu, Xiansheng Hua, and Ji-Rong Wen. Counterfactual vqa: A causeeffect look at language bias. _2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 12695–12705, 2021. 1, 2, 3 

- [48] Yulei Niu, Kaihua Tang, Hanwang Zhang, Zhiwu Lu, XianSheng Hua, and Ji-Rong Wen. Counterfactual vqa: A causeeffect look at language bias. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 12700–12710, June 2021. 1, 2 

- [49] Thomas Pechmann. Incremental speech production and referential overspecification. 1989. 3 

- [50] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 32, 2018. 1, 2, 5 

- [51] Fengchun Qiao, Long Zhao, and Xi Peng. Learning to learn single domain generalization. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 12556–12565, 2020. 1, 3 

- [52] Mengye Ren, Ryan Kiros, and Richard Zemel. Exploring models and data for image question answering. _Advances in neural information processing systems_ , 28, 2015. 2 

- [53] Leonard Salewski, A Koepke, Hendrik Lensch, and Zeynep Akata. Clevr-x: A visual reasoning dataset for natural language explanations. In _International Workshop on Extending Explainable AI Beyond Deep Models and Classifiers_ , pages 69–88. Springer, 2022. 2 

   - [64] Xiao Zhang, Zhiyuan Fang, Yandong Wen, Zhifeng Li, and Yu Qiao. Range loss for deep face recognition with longtailed training data. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 5409–5418, 2017. 3 

   - [65] Yuke Zhu, Oliver Groth, Michael Bernstein, and Li Fei-Fei. Visual7w: Grounded question answering in images. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 4995–5004, 2016. 2 

- [54] Susan Sonnenschein. The development of referential communication skills: Some situations in which speakers give redundant messages. _Journal of Psycholinguistic Research_ , 14(5):489–508, 1985. 3 

- [55] Hao Tan and Mohit Bansal. Lxmert: Learning crossmodality encoder representations from transformers. _arXiv preprint arXiv:1908.07490_ , 2019. 2 

- [56] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The inaturalist species classification and detection dataset. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 8769–8778, 2018. 3 

- [57] Yiming Xu, Lin Chen, Zhongwei Cheng, Lixin Duan, and Jiebo Luo. Open-ended visual question answering by multimodal domain adaptation. _arXiv preprint arXiv:1911.04058_ , 2019. 1, 3 

- [58] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. _arXiv preprint arXiv:1910.01442_ , 2019. 2 

- [59] Kexin Yi, Jiajun Wu, Chuang Gan, Antonio Torralba, Pushmeet Kohli, and Joshua B Tenenbaum. Neural-Symbolic VQA: Disentangling Reasoning from Vision and Language Understanding. In _Advances in Neural Information Processing Systems (NIPS)_ , 2018. 1, 2, 5 

- [60] Zhou Yu, Jun Yu, Yuhao Cui, Dacheng Tao, and Qi Tian. Deep modular co-attention networks for visual question answering. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 6281– 6290, 2019. 2 

- [61] Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and SongChun Zhu. Raven: A dataset for relational and analogical visual reasoning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 5317– 5327, 2019. 2 

- [62] Mingda Zhang, Tristan Maidment, Ahmad Diab, Adriana Kovashka, and Rebecca Hwa. Domain-robust vqa with diverse datasets and methods but no target labels. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 7046–7056, 2021. 1, 3 

- [63] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Making visual representations matter in visionlanguage models. _CVPR 2021_ , 2021. 2 

## **A. Dataset statistics** 

Fig. 4 shows the distribution of question types in the Super-CLEVR dataset. The question type is determined by the type of the last operation in the question program. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning_images/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning.pdf-0012-02.png)


Figure 4. Distribution of question types in Super-CLEVR. 

## **B. List of objects** 

Super-CLEVR contains 21 objects from 5 categories: airplane, bicycle, bus, car and motorcycle. They are shown in Fig. 5 and Tab. 4. 

|**catetory**|**objects**|
|---|---|
|**airplane**|airliner, biplane, jet, fghter|
|**bicycle**|utility bike, tandem bike, road bike, mountain bike|
|**bus**|articulated bus, double bus, regualr bus, school bus|
|**car**|truck, suv, minivan, sedan,wagon|
|**motorcycle**|chopper, scooter, cruiser, dirtbike|



Table 4. Super-CLEVR dataset contains 21 objects from 5 categories. 

## **C. Dataset controlling** 

Fig. 6 shows the concept distribution for dataset variants _bal_ , _slt_ and _long_ ; and variants _head_ , _tail_ and _oppo_ for testing purpose. Fig. 7 shows the concept co-distribution Matrix _M_ for controlling the concept compositionality (for variants _co-0_ , _co-1_ and _co-2_ ). The descriptions for the variants are in Sec. 3.3. 

## **D. Definition of Relative Degrade** 

Here we describe _Relative Degrade_ for each domain shift factors. We use _A[i] j_[to][denote][the][accuracy][of][model][trained] using data variant _i_ and tested with data variant _j_ . Visual complexity: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning_images/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning.pdf-0012-12.png)


Question redundancy, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning_images/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning.pdf-0013-01.png)


Concept distribution, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning_images/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning.pdf-0013-03.png)


Concept compositionality, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning_images/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning.pdf-0013-05.png)


## **E. More details about P-NSVQA** 

Given a image containing _n_ objects, we maintain a vector of probability **p** = [ _p_[1] _, p_[2] _, . . . , p[n]_ ], where _p[k]_ means the probability that object _k_ is selected. We update **p** when executing the reasoning operations step by step. In the following, we describe how to compute **p** for each kind of operations. 

- _scene_ 

Initialize all the values in **p** to 1. 

- _filteridentifier_ [ _attribute_ ] ( _e.g_ . _filtercolor_ [ _red_ ]) For object _k_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning_images/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning.pdf-0013-11.png)


Here _Pattribute[k]_[is the probability of object] _[ k]_[ having the] _[ attribute]_[, which is predicted by the visual scene parsing model.] 

- _relate_ ~~_s_~~ _pacial_ (including _relate_ ~~_b_~~ _ehind_ , _relate_ ~~_f_~~ _ront_ , _relate_ ~~_r_~~ _ight_ , _relate_ ~~_l_~~ _eft_ ) 

The output of the _relate_ operation is the probabilities of each object being on the _spacial_ side of the given object. For example, _relate_ ~~_l_~~ _eft_ ( _i_ ) computes the probabilities of objects to be on the left side of the given object _i_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning_images/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning.pdf-0013-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning_images/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning.pdf-0013-16.png)


Here _i_ is the input object and ( _xi, yi_ ) is the center of it. _a_ and _b_ are hyperparameters. In our experiments, we set _a_ = 20 and _b_ = 0 _._ 02. 

- _same_ ~~_c_~~ _olor_ , _same_ ~~_s_~~ _hape_ , _same_ ~~_s_~~ _ize_ , _same_ ~~_m_~~ _aterial_ The _same_ operation returns the probabilities of each object having the same attribute as the given object _i_ . For example, for object _k_ and attribute _color_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning_images/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning.pdf-0013-19.png)


- _intersect_ , _union_ 

Given two probability vectors **p1** _,_ **p2** , we calculate their intersection or union: 

_Intersection_ : **p** = **p1** _⊙_ **p2** 

_Union_ : **p** = 1 _−_ (1 _−_ **p1** ) _⊙_ (1 _−_ **p2** ) 

Here _⊙_ is the pointwise product. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning_images/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning.pdf-0015-00.png)


**----- Start of picture text -----**<br>
airliner biplane jet fighter<br>utility bike tandem bike road bike mountain bike<br>chopper scooter cruiser dirtbike<br>articulated bus double bus regular bus school bus<br>truck suv minivan sedan<br>wagon<br>**----- End of picture text -----**<br>


Figure 5. There are 21 objects belonging to 5 categories in the Super-CLEVR dataset. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning_images/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning.pdf-0016-00.png)


**----- Start of picture text -----**<br>
color<br>material<br>shape<br>(a)  bal (b)  slt (c)  long<br>(balanced) (slightly unbalanced) (long-tail)<br>color<br>material<br>shape<br>(d)  head (e)  tail (f)  oppo<br>(opposite)<br>**----- End of picture text -----**<br>


Figure 6. Concept distribution for dataset variants _bal_ , _slt_ , _long_ , _head_ , _tail_ and _oppo_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning_images/cvpr_2023_Super-CLEVR_A_Virtual_Benchmark_To_Diagnose_Domain_Robustness_in_Visual_Reasoning.pdf-0017-00.png)


**----- Start of picture text -----**<br>
(a)  co-0 (b)  co-1 (c)  co-2<br>**----- End of picture text -----**<br>


Figure 7. Concept co-distribution matrix _M_ for dataset variants _co-0_ , _co-1_ and _co-2_ . 

