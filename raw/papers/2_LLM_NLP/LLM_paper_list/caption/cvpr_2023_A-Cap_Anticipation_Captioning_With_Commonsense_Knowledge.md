## **A-CAP: Anticipation Captioning with Commonsense Knowledge** 

Duc Minh Vo The University of Tokyo, Japan 

Quoc-An Luong The Graduate University for Advanced Studies, Japan lqan@nii.ac.jp 

vmduc@nlab.ci.i.u-tokyo.ac.jp 

Akihiro Sugimoto National Institute of Informatics, Japan sugimoto@nii.ac.jp 

Hideki Nakayama The University of Tokyo, Japan 

nakayama@ci.i.u-tokyo.ac.jp 

## **Abstract** 

_Humans possess the capacity to reason about the future based on a sparse collection of visual cues acquired over time. In order to emulate this ability, we introduce a novel task called_ _**Anticipation Captioning** , which generates a caption for an unseen oracle image using a sparsely temporally-ordered set of images. To tackle this new task, we propose a model called_ A-CAP _, which incorporates commonsense knowledge into a pre-trained vision-language model, allowing it to anticipate the caption. Through both qualitative and quantitative evaluations on a customized visual storytelling dataset,_ A-CAP _outperforms other image captioning methods and establishes a strong baseline for anticipation captioning. We also address the challenges inherent in this task._ 

## **1. Introduction** 

When humans observe the real world, we not only capture visual information (e.g. objects), but also forecast the future from past and current observations. For example, in Fig. 1, given some photos of an attack in a hockey game, we can predict without a doubt that “the athlete will shoot the puck toward the goalie”. In fact, anticipatory ability aids us in surviving in a world of volatility. This ability necessitates a significant shift from visual to cognitive understanding, which extends far beyond the scope of tasks that primarily use visible visual data, such as object detection, action recognition, and existing image captioning. As a result, a variety of new tasks have been proposed to emulate humans’ anticipatory ability, such as generating future images [12,29], and action prediction [22,37]. Despite their great success, the aforementioned tasks frequently involve densely temporal information (i.e., video), which can be difficult to acquire at times, and their outcomes are not friendly to everyone, particularly those with visual impairments. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0001-12.png)


**----- Start of picture text -----**<br>
a) b)<br>c) d)<br>Sparsely temporally-ordered images Oracle image<br>( Input for all tasks ) ( for reference )<br>Task Output(s) for  Output for<br>input images oracle image<br>a) A man standing with a hockey stick.<br>Image  b) A man standing on an ice rink. ( N/A )<br>captioning c) A man holding a hockey stick.<br>d) A group of men playing a game of hockey.<br>a, b, c, d) This breakaway was the first threat to<br>score. The wingman took the puck to the goal<br>Story but a nice play by the goalie saved the goal.  ( N/A )<br>telling<br>Finally, the other team gets the puck deep into<br>red zone. He is now within 20 feet of the goal.<br>He shoots, he<br>Anticipation  scores and the<br>captioning  ( N/A )<br>game ends one<br>(Ours)<br>to nothing.<br>**----- End of picture text -----**<br>


Figure 1. Given a set of sparsely temporally-ordered images (a, b, c, d), image captioning [38] and storytelling [35] tasks generate captions for those images, while our anticipation captioning task anticipates what happens afterward. To illustrate the potential future, we show their related oracle image. It should be noted that our task only receives the same inputs as others. 

In this work, we hope to dislodge the time constraints imposed by previous tasks while also looking for a more user-friendly output format. Needless to say, textual description is a potential candidate because generating text from images has been successfully explored in a variety of ways [6, 14, 21, 33, 35, 38], showing a number of applications. Furthermore, we can easily leverage recent advances in text-to-image [28] or text-to-sound [36] as a flexible transformation that will benefit other downstream tasks, 

allowing everyone to fully grasp our output in their own way. With this in mind, we go beyond the immediately visible content of the images, proposing a new task of image captioning problems, called **anticipation captioning** . Anticipation captioning is to generate a caption for an unseen image (referred to as the oracle image) that is the future of a given set of sparsely temporally-ordered images. The term “sparse” means that two consecutive images are not required to be as close in time as those in a video, allowing the scene to change freely as long as the change does not disrupt the information flow of the image sequence, as seen in Fig. 1. Our task is a new branch of the image captioning problems [6, 14, 21, 35, 38]; it is to predict only captions in the future. As an example, we depict the outputs of generic image captioning, visual storytelling, and our task in Fig. 1. The image captioning model [38] generates a caption for each individual image, whereas visual storytelling [35] connects all images in a story. Our task, on the other hand, produces a caption for the oracle image that is similar to human anticipation: “he shoots, he scores, and the game ends one to nothing”. Unlike [12, 22, 29, 37], anticipation captioning does not require strictly temporal information while producing a more informative output. In theory, the greater the success of this task, the greater the deductive ability of the intelligent system. Meanwhile, other applications such as incident prevention or behavior prediction can be launched. 

Additionally, we propose a baseline model, A-CAP, to solve this new task rather than simply using current image captioning models, given their failures in predicting the future. We hypothesize that under common thinking, the future can be predicted from observable concepts (e.g., objects, events) appearing in the input images, implying that the future cannot be dramatically changed to the “football scene” from the “hockey scene”, for instance. As a result, we make full use of commonsense knowledge to connect all detected concepts in terms of a graph while expanding the graph toward forecasted ones, creating a knowledge graph. The term “forecasted concept” refers to a concept that is not visible in the given image but related to another concept visible in the image (we can infer the forecasted concept from the related concept using common thinking). Technically, each node in our constructed graph is either a detected concept in given inputs or a forecasted one explored using the ConceptNet [30], and nodes are connected if and only if they have corresponding ConceptNet relations. After aggregating all node information with a graph neural network, we use prompt learning [39, 40] to integrate the enriched nodes into a frozen pre-trained vision-language (VL) model, successfully generating the anticipated caption. The following are our primary contributions. 

- We introduce a novel task of anticipation captioning, which predicts a caption for the future from a given set of sparsely temporally-ordered images. 

- For anticipation captioning, we establish a strong baseline model A-CAP, which incorporates commonsense knowledge into a pre-trained VL model. 

We evaluate the effectiveness of A-CAP in both qualitative and quantitative ways, using a customized VIST dataset [14]. Extensive experiments show that A-CAP successfully generates captions for oracle images that are more accurate, descriptive, and reasonable than those generated by other captioning methods [35, 38]. 

## **2. Related work** 

**Future forecasting** has long been studied in computer vision. Some attempts [12, 16, 29, 34] have been made to generate future images/frames from a given video (i.e., dense time-series images). Meanwhile, some methods [22, 37] use past observations to predict future events. These methods heavily rely on the dense temporal-structure to learn visual representations, implying that such representations are different from those for sparsely temporally-ordered images. Furthermore, generated images/frames are not always of high quality [12,16,29,34], and the set of predicted future events is limited [22, 37], making them difficult to apply to downstream tasks. Our method, on the other hand, accepts only sparsely temporal information as long as we can detect objects/events. Furthermore, our method is designed to generate textual descriptions that are easier to interpret than outputs by other methods [12, 16, 22, 29, 34, 37]. 

In NLP, there are also several approaches to predict the future: story ending generation [7,18], temporal order anticipation [23, 24]. Though those methods use texts as inputs while our method uses images, we can think of story ending generation as an indirect way to solve our problem because we can generate a story first and then predict its ending. 

**Image captioning** is a long-standing problem with numerous methods developed to address various purposes. Captioning models [6, 21] in an early stage aim to generate generic descriptions for given images. They are then evolved in various directions to generate dense captions [15], novel object captions [33], controllable captions [9], or visual story telling [8, 14, 35]. Anticipation captioning belongs to the image captioning family, with the exception that we predict a caption for the future. Furthermore, our method is based on recent methods [33, 38], which use a vision-language model to generate better captions. Rather than fine-tuning or retraining the model, we use prompt learning [39, 40] to replace the object tags used in the concatenated sequence of words—object tags—ROIs of VinVL [38] with our detected and forecasted concepts. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Input<br>Pre-processing detectionConcept  … He [MASK], he [MASK] and the [MASK] ends<br>one to nothing.<br>Image Feature<br>Knowledge Graph Construction Extraction Tokenizer<br>Embedding<br>layer<br>Model Graph neural  context feature … …<br>network<br>… Multi-layer Transformers<br>enriched nodes<br>Frozen VinVL<br>He shoots, he scores and the<br>Output<br>game ends one to nothing.<br>**----- End of picture text -----**<br>


Figure 2. The overall pipeline of our proposed A-CAP. The preprocessing step is used to build the knowledge graph, extract image features and tokenize the input words. In the knowledge graph construction, blue nodes represent the detected concepts obtained from concept detection while brown nodes represent the forecasted concepts obtained from the ConceptNet. Our network consists of a trainable graph neural network and a frozen pre-trained VinVL [38]. The outputs of the graph neural network are the enriched nodes of the knowledge graph. During inference time, the dash-dotted red part is removed. 

## **3. Our approach** 

## **3.1. Problem statement** 

Our input is a set of _k_ sparsely temporally-ordered images _I_ 1 _, I_ 2 _, . . . , Ik_ . It is worth noting that _Ii_ and _Ii_ +1 are not necessarily strongly temporal as illustrated in Fig. 1. We assume that an image _Ik_ +1 is an oracle image that continues the set of _k_ images, and that a caption _Ck_ +1 corresponds to _Ik_ +1 which is a future of _I_ 1 _, I_ 2 _, . . . , Ik_ . Obviously, the oracle image is sparsely temporally-ordered with respect to the input images as we intentionally seek to anticipate the future. 

Our task is to generate caption _Ck_ +1 using given _k_ images. The task is formally defined as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0003-06.png)


where CAPTION( _·_ ) is a captioning system that will be discussed later. Note that we produce neither captions for each input image _I_ 1 _, . . . , Ik_ nor oracle image _Ik_ +1. 

## **3.2. Proposed** A-CAP 

## **3.2.1 Design of A-CAP** 

Given the progress of vision-language models in image captioning tasks, we choose VinVL [38] as our base architecture. VinVL takes a concatenated sequence of words– concepts–ROIs as input (note that words are not used during inference time; object tags are used instead of concepts in the original paper [38]). The core idea is the usage of 

concepts, which allows better alignment between the vision and language spaces. The above observation suggests that incorporating forecasted concepts into VinVL is critical in allowing the model to generate the anticipated caption. However, simply using VinVL is not wise because it detects only concepts appearing in images. We thus find forecasted concepts based on the detected concepts. Under normal circumstances, forecasted concepts should be related to current observable concepts. Therefore, to retrieve forecasted concepts, we use commonsense knowledge, which consists of many popular concepts and their relationships. 

VinVL [38] is trained on a very large dataset, making fine-tuning or re-training difficult. To avoid this difficulty, we use the prompt learning technique to train the concept embeddings only while other parameters are fixed. In what follows, we detail our model. 

## **3.2.2 Network architecture** 

We base A-CAP on the VinVL [38] architecture. As discussed above, we use concepts as a prompt to allow the model to generate a desired caption. We can then focus on learning the embeddings for all detected and forecasted concepts. To this end, we first retrieve the forecasted concepts using the detected ones and then construct the knowledge graph that connects all concepts. This is because the graph structure is effective for learning the interactions between concepts. We use an undirected graph for simplicity where two concepts are connected as long as their relationship exists. The concept embeddings are then enhanced using a graph neural network. Next, the enriched concept embeddings are injected into a frozen VinVL to generate the caption. Fig. 2 depicts our simple yet effective A-CAP. 

## **3.2.3 Modules of A-CAP** 

**Pre-processing.** The input images are pre-processed to (i) construct the knowledge graph and (ii) extract image features. We also tokenize the ground-truth captions used to train the model during training. We obtain _N_ features (ROIs) with the size of 1 _×_ 2054 each after image feature extraction using Faster-RCNN [27] trained on the COCO dataset. Each image feature is fed into VinVL’s embedding layer to reduce its size to 1 _×_ 768. We then take the average of all image features[¯] **f** = _N_ 1 � _Ni_ =1 **[f]** _[i]_[to construct a context] feature (1 _×_ 768) which will be used later. Simultaneously, we obtain _L_ word embeddings of the caption _{_ **w** _i}[L] i_ =1[, each] of which has a size of 1 _×_ 768. For more information on image feature extraction and tokenizer, see VinVL [38]. 

We now detail knowledge graph construction. We follow Chen et al. [8] to detect concepts for each input image. Specifically, we use clarifai [1] to obtain the top-ten concepts _{ci}_[10] _i_ =1[for each image.][As a result, we detect] _[ k][ ×]_[ 10] concepts in total. Then, using ConceptNet [30], we use each 

detected concept as a query to heuristically retrieve forecasted concepts with 2-hop neighbors of the query. Since the number of forecasted concepts is large ( _>_ 400) and many of them are unrelated to input images, we employ a filtering process to retain only the informative concepts. 

Let _c[f] i_[be a forecasted concept.][Using a pre-trained lan-] guage model RoBERTa [20], we compute a relevance score between the forecasted concept and image context as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0004-02.png)


where **c** _[f] i_[=][BERT][(] _[c][f] i_[)][ is an embedding vector of the con-] cept _c[f] i_[extracted by a pre-trained BERT [][10][],][ [] _[·]_[;] _[ ·]_[]][ denotes] the concatenation operator, _f_ enc is the encoder part of the language model while _f_ head is a softmax layer. This score indicates the probability of **c** _[f] i_[related to][ ¯] **[f]**[.] 

We keep _M_ forecasted concepts having high relevance scores. In total, we have _k ×_ 10 detected concepts _{ci}[k] i_ =1 _[×]_[10] and _M_ forecasted concepts _{c[f] i[}] i[M]_ =1[in our knowledge graph] ( _k ×_ 10 + _M_ nodes). If two concepts are related in the ConceptNet [30], an undirected edge is given to connect them. For simplicity, we do not use a specific relation (e.g., has, IsA). Furthermore, a concept in _Ii_ is connected to its related concepts in the adjacent images _Ii−_ 1 and _Ii_ +1 to ensure information flow and the awareness of the temporal order of the images. Hereafter, we use the same notation to refer to detected and forecasted concepts _{ci}[k] i_ =1 _[×]_[10+] _[M]_ . **Graph neural network** is used to update the node embeddings through iterative messages passing between neighbors on the graph. We use graph attention network [32] to build our graph neural network. To produce the input for the graph network, we first employ pre-trained BERT [10] to embed each concept into an embedding with the size of 1 _×_ 768. To be more specific, each node embedding is calculated as **e** _i_ = BERT( _ci_ ). To strengthen the connection between concepts and image context, we concatenate the node embedding and the context feature as **e** _i_ = [ **e** _i_ ; **[¯] f** ]. Brevity, we summarize the entire computation in each graph layer: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0004-05.png)


where _l_ indicates the current graph layer while _l−_ 1 does the previous one, GNN( _·_ ) represents a graph layer. In detail, each node is updated by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0004-07.png)


where **W** _q,_ **W** _k,_ **W** _v,_ **W** _o ∈_ R _[D][×][D]_ are learnable matrices, _Ni_ represents the neighbors of node _i_ , _D_ = 768 + 768 = 

1536, SOFTMAX and LAYERNORM are the softmax function and the batch normalization, respectively. We note that **e**[(0)] _i_ is the initial node embedding (i.e., [ **e** _i_ ; **[¯] f** ]). 

In practice, we use 2 graph layers. After the graph attention network, we add two more fully connected layers to reduce the size of each **˜e** _i_ to 1 _×_ 768. **Frozen VinVL.** As discussed above, the concept embeddings learned from the graph neural network are used as a prompt to generate the caption. To this end, we inject all _{_ **˜e** _i}[k] i_ =1 _[×]_[10+] _[M]_ into a frozen pre-trained VinVL [38]. As a result, the input of VinVL is changed to _{_ **w** 1 _, · · · ,_ **w** _L,_ [ `SEP` ] _,_ **˜e** 1 _, · · · ,_ **˜e** _k×_ 10+ _M ,_ [ `SEP` ] _,_ **f** 1 _, · · · ,_ **f** _N }_ . We note that [ `SEP` ] is a special token used to distinguish different types of tokens. We do not feed **w** _i_ to the network during inference time, but instead, create _L ×_ [ `MASK` ] as pseudo words. Formally, Eq. 1 becomes 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0004-11.png)


**Loss function.** Following previous works, we simply use cross entropy between the generated and the ground-truth captions to train the network. We do not use CIDEr optimization because the pre-trained VinVL has been welltrained on a large text–image corpus. 

## **4. Experiments** 

## **4.1. Dataset and training details** 

**Dataset.** We use the visual storytelling dataset (VIST) [14] with a modification to evaluate our method because there is no dataset tailored for our task. The original VIST includes 210,819 photos from 10,117 Flickr albums. Given five input temporally ordered images from the same event, the corresponding five human-annotated sentences are provided as ground-truths. There are 4,098, 4,988, and 5,050 samples for training, validation, and test sets, respectively. We use the first four images of each sample as input ( _k_ = 4) and the last sentence of each sample as the ground-truth caption. We keep the last image of each sample as an oracle image for reference. The training, validation, and test sets all have the same number of samples as the original dataset. **Dataset verification.** We investigate the correlation between _Ck_ +1 and _C_ 1 _, C_ 2 _, . . . , Ck_ (corresponding captions to _I_ 1 _, I_ 2 _, . . . , Ik_ ) in two ways. First, we compute the sentence cosine similarity sim(S( _Ck_ +1) _,_ S( _Ci_ )) ( _i_ = 1 _, . . . , k_ ) and then test whether those similarities monotonically increase (i.e., sim(S( _Ck_ +1) _,_ S( _Ci_ )) _<_ sim(S( _Ck_ +1) _,_ S( _Ci_ +1)) (S( _·_ ) is a pre-trained SentenceTransformer model [2], outputting an embedding vector for a given sentence). We confirm that 72.69% of samples follow monotonic increasing, 10.32% have only one sentence similarity that violates monotonic increasing, and only 4.4% do not comply with the monotonicity. As the second, we use a pre-trained BERT model [10] to figure out whether _Ci_ +1 is the next 

|Sparsely temporally-ordered images|Oracle<br>image (for<br>reference)|VinVL|VinVL +<br>Oracle image|AREL + BART|**A-CAP**|Ground-truth|
|---|---|---|---|---|---|---|
|||after the<br>ceremony, the<br>teams got to eat<br>outside.|the defense was<br>able to close<br>out the game<br>and had a great<br>time.|i was getting<br>ready to leave<br>the game and i<br>took a picture<br>of the players<br>on the field<br>before the<br>game.|the goalie<br>caught the puck<br>as it passed the<br>goalie.|he shoots, he<br>scores and the<br>game ends one<br>to nothing.|
|||she let the<br>crowd ask<br>questions in the<br>end.|we got to meet<br>the people<br>behind the<br>company's<br>logo.|he welcomed to<br>the stage his<br>new assistant|at the end of the<br>show, the<br>audience<br>enjoyed<br>themselves.|they were all<br>about<br>preserving the<br>internet|
|||the llamas were<br>very curious.|the competition<br>ended with a<br>bang.|they had a great<br>time.|it was a great<br>time for the<br>horse racers.|he thought he<br>was going to<br>cry|
|||the vice<br>president closed<br>the meeting by<br>thanking all the<br>workers of the<br>company.|the party went<br>on well into the<br>night.|everyone was<br>having a great<br>time.|they ended the<br>night with a<br>speech.|eventually the<br>winner was<br>announced, and<br>he was very<br>grateful|



Figure 3. Examples of generated captions obtained by all compared methods. We show the oracle images and ground-truth captions for reference purposes. VinVL [38] generates captions that are out of context with the input images. VinVL [38] + Oracle image sometimes generates reasonable captions. AREL [35] + BART [17] tends to generate a general ending for the sequence of images. In contrast, our method A-CAP predicts more accurate, descriptive, and plausible captions than others. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0005-02.png)


**----- Start of picture text -----**<br>
VinVL VinVL +  AREL + BART A-CAP Ground-truth<br>Oracle image<br>**----- End of picture text -----**<br>


Figure 4. The generated images obtained by using stable diffusion model [28] to generate an image from each generated caption in Fig. 3. The order of images is the same as the order of captions in Fig. 3. The images generated using our captions are close to the ground-truth ones while those by other methods are not. 

sentence of _Ci_ . We see that 77.34% of the samples satisfy the next sentence condition (i.e., _Ci_ +1 is always the next 

sentence of _Ci_ for all sentences in the sequence), 17.78% have only one sentence that does not meet the condition, and 0.06% do not satisfy the condition (i.e., _Ci_ +1 is never the next sentence of _Ci_ ). The above verification shows that the VIST dataset mostly meets our assumption. 

**Training details.** We set the length of the word sequence _L_ = 35, the number of ROIs _N_ = 100 (25 ROIs for each image), the number of forecasted concepts _M_ = 60 (the number of concepts is 4 _×_ 10 + 60 = 100 in total). 

We build A-CAP using PyTorch, in which we use the pre-trained VinVL model published by its authors [3]. We remark that we freeze all the parameters of VinVL during training time. Given the small size of our used dataset, we train the model for only 10 epochs with a batch size of 16 and a learning rate of 3e-5. It takes four hours to train our model on a single GTX-3090 GPU. 

## **4.2. Compared methods and evaluation metrics** 

**Compared methods.** We carefully design methods that can be straightforwardly applied to our task. For a fair comparison, all compared methods are fine-tuned on VIST. To avoid over-tuning, we only train the methods for a few epochs and select their best checkpoints. 

VinVL [38] is a cutting-edge image captioning model. We strictly adhere to its settings, but instead of a single im- 

age, we use the input as our method. Comparing our method to VinVL will demonstrate the advancement of our method over the conventional image captioning model. 

VinVL [38] + Oracle image is the method where VinVL uses the ground-truth oracle image in training and testing. Since we do not successfully generate oracle images using existing methods, we may regard this method as a method that sequentially generates the oracle image and caption. 

AREL [35] + BART [17] is a combination of visual storytelling (AREL [35]) and story ending generation (BART [17]). Particularly, we generate a story for the input and then generate the ending sentence for that story. We compare the ending sentence to the caption by our method. **Evaluation metrics.** Since our problem is an open domain generation like dialogue generation, we follow [11] to use automatic metrics to quantitatively evaluate all the methods in two aspects: _accuracy_ and _descriptiveness_ . For accuracy evaluation, we report referenced metrics including BLEU [25], CIDEr [31]. Since those metrics are sensitive to the whole sentence structure [19], we also report SPICE [5], CLIPScore, and RefCLIPScore [13] to overcome the structural dependency. For descriptiveness evaluation, we adopt a self-retrieval strategy, drawing on prior work. This strategy is based on the observation that more descriptive captions with significant details frequently lead to more precise self-retrieval, i.e., retrieving the target image from a set of similar images given the generated caption. We report the refined R@1, R@5, and R@10 scores using CLIP [26] as the retriever. 

## **4.3. Qualitative comparisons** 

In Fig. 3, we show some randomly selected examples of captions generated by our method as well as others. Despite its enormous success in image captioning, VinVL [38] is unable to generate the expected captions. We can see that the captions generated by VinVL are completely out of context with the input images. This observation suggests that the current image captioning model is inadequate for our task. VinVL [38] + Oracle image generates reasonable captions to some extent when the oracle images are close enough to the input images (see first and second samples). However, if the temporal information is too sparse as in the third and fourth samples, it fails to generate captions that are linked to the inputs. These results imply that even if we can generate a high-quality unseen oracle image, the model struggles to complete the task. We notice that AREL [35] + BART [17] generates a general ending for the story (e.g., having a great time). On the contrary, our method produces more accurate and reasonable captions that reflect the inputs’ future. In most cases, we can see that our method accurately predicts what is likely to happen, which is close to the ground-truth captions. When we examine the third sample in greater detail, we can see that our caption is in- 

correct because we failed to detect the concept “falling” in the second image. However, we believe that the generated caption is still plausible under ordinary situations. 

To have a better understanding of the generated captions, we use the stable diffusion model [28] implemented on the Huggingface platform [4] with the default settings to generate an image from each generated caption, and choose the first generated image for each method as shown in Fig. 4. The images obtained from our generated captions are similar to the ground-truth ones, indicating that our method generates correct anticipated captions. Furthermore, Fig. 4 demonstrates the benefits of our task to downstream tasks, specifically future image generation in this case. 

## **4.4. Quantitative comparisons** 

The quantitative scores are summarized in Table 1, first four rows. We first assess all methods based on their accuracy. All of the results in Table 1 support the advantage of our method over the other methods. Though our method obtains the highest scores, we notice that it does not significantly outperform the other methods on referenced metrics (BLEU and CIDEr). The reason for this observation is that those metrics are calculated using ground-truth captions. Because our task is an open-domain generation, it is difficult to generate a caption that is nearly identical to the ground-truth one. However, based on the qualitative comparison in Figs. 3 and 4, we can conclude that our method outperforms the others. SPICE and the unreferenced metrics (CLIPScore, RefCLIPScore) also justify our conclusion. We see substantial improvements in these metrics, indicating that our generated captions accurately reflect the oracle images. Notably, as shown in Fig. 3, our generated captions are, without a doubt, the future of input images. 

The descriptiveness of generated captions is then assessed using R@1, R@5, and R@10 scores. In comparison to VinVL [38] and AREL [35] + BART [17], our method outperforms them significantly. This is thanks to the fact that captions generated by our method are close to the ground-truth images, whereas those obtained by the other methods are not. Our method and VinVL [38] + Oracle image achieve the same level. This is not surprising, given that VinVL [38] + Oracle image generates captions directly from oracle images. 

We conclude that our method is more promising than the other methods in solving the anticipation captioning task. Furthermore, the experiments highlight the shortcomings of using image captioning and story ending models in our task. 

## **4.5. Detailed analysis** 

**Ablation study.** To validate the plausibility of our model design, we investigate two ablated models: A-CAP w/o GNN and A-CAP w/o context. A-CAP w/o GNN denotes the model that does not use a graph neural network (instead, 

Table 1. Quantitative comparison against other methods. For accuracy evaluation, we report referenced metrics (BLEU [25] (B-1, B-4), CIDEr [31]), SPICE [5], and unreferenced metrics (CLIPScore and RefCLIPScore [13]). For descriptiveness evaluation, we report top-1, top-5 and top-10 retrieval accuracy (R@1, R@5, R@10, respectively). Our method outperforms others on all metrics. Higher scores are better. Gray background indicates results obtained by our method, and ∆ indicates the improvement over compared methods. 

|Method|Accuracy<br>B-1<br>B-4<br>CIDEr<br>SPICE<br>CLIPScore<br>RefCLIPScore|Descriptiveness<br>R@1<br>R@5<br>R@10|
|---|---|---|
||||
|VinVL [38]<br>31.7<br>3.1<br>2.6<br>13.8<br>40.7<br>42.8<br>1.3<br>6.5<br>10.8<br>VinVL [38] + Oracle image<br>34.9<br>3.8<br>4.3<br>16.9<br>57.9<br>61.3<br>8.1<br>17.2<br>31.1<br>AREL [35] + BART [17]<br>30.9<br>2.0<br>3.1<br>11.4<br>37.8<br>39.7<br>1.1<br>5.9<br>9.3|||
|A-CAP|**37.2**<br>**6.9**<br>**4.7**<br>**20.1**<br>**65.2**<br>**70.2**|**8.7**<br>**18.9**<br>**31.5**|
|A-CAPw/o GNN|34.8<br>5.2<br>3.7<br>14.5<br>38.2<br>47.3|3.6<br>8.7<br>15.4|
|A-CAPw/o context|36.1<br>6.2<br>4.2<br>13.9<br>39.8<br>46.9|4.1<br>9.5<br>16.1|
||||
|∆|2.3_↑_<br>3.1_↑_<br>0.4_↑_<br>3.2_↑_<br>7.3_↑_<br>8.9_↑_|0.6_↑_<br>1.7_↑_<br>0.4_↑_|




![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0007-02.png)


**----- Start of picture text -----**<br>
Oracle  A-CAP  A-CAP  A-CAP Ground-<br>image (for  w/o GNN w/o  (full) truth<br>Sparsely temporally-ordered images reference) context<br>the  we walked  we ended  we stopped<br>downtown  around a  the night  at a<br>streets  bit more  by  souvenir<br>were lined  before  shopping  store to get<br>with  heading  in the  some<br>people  home. center of  things<br>enjoying  the city. before<br>themselves finally<br>heading<br>back to the<br>hotel.<br>the  we had a  it was a  the place<br>nightlife  great time  great night  was ready<br>was just  walking  and i can't  to close<br>amazing to  around. wait to go  and we had<br>look at. back next  to leave.<br>year.<br>**----- End of picture text -----**<br>


Figure 5. Examples of generated captions by two ablated models: A-CAP w/o GNN, A-CAP w/o context, and full model A-CAP. We select two inputs where the detected concepts almost overlap. A-CAP w/o GNN generates captions that most likely describe the inputs. A-CAP w/o context generates captions that are far from the inputs and similar to each other. 

we directly feed the concept embeddings **˜e** _i_ = BERT( _ci_ ) to the pre-trained VinVL). A-CAP w/o context is the model in which we do not concatenate the node embeddings and the context feature (we instead use only the node embeddings as graph neural network inputs). We also drop the two fully connected layers on top of the graph neural network because reducing the size of embeddings is no longer required. 

The last two rows of Table 1 quantify the performance of the two ablated models. When we simplify the model, the performance scores are degraded. In the case of A-CAP w/o GNN, the concept embeddings are insufficient to guide the model to generate the expected caption. As a result, the caption most likely describes the inputs as depicted in Fig. 5. The graph neural network enriches and connects concept embeddings, making them more powerful as a prompt to the model. Similarly, A-CAP w/o context breaks the connections between concepts and the context of images in gen- 

Table 2. Impact of the number of forecasted concepts on the performance of our model. Using either a large number of concepts or no concepts drops the performance drastically. 

|Number of forecasted concepts|Accuracy<br>SPICE<br>CLIPScore<br>RefCLIPScore|Descriptiveness<br>R@1<br>R@5<br>R@10|
|---|---|---|
|_M_ = 400<br>_M_ = 200<br>_M_ = 100<br>_M_ = 60(used model)<br>_M_ = 0|5.8<br>15.3<br>12.1<br>5.4<br>16.7<br>13.0<br>15.7<br>48.6<br>52.4<br>20.1<br>65.2<br>70.2<br>14.2<br>43.1<br>44.7|1.1<br>3.7<br>7.6<br>0.9<br>4.2<br>7.1<br>6.2<br>15.7<br>26.6<br>8.7<br>18.9<br>31.5<br>1.9<br>7.3<br>11.2|



eral, resulting in captions that are far from the inputs and similar to each other if the detected concepts are similar (Fig. 5). This indicates that the context feature compensates for the concepts in order to make the correct prediction. In contrast, the full model generates plausible captions. 

We do not investigate the model where all the parameters are trainable since the training collapsed despite our best efforts. The reason for this failure is that the training data is too small in comparison with the one used to train VinVL. 

**Impact of the number of forecasted concepts.** As stated above, when we search for concepts on ConceptNet, we usually have more than 400 forecasted concepts. We empirically retain _M_ = 60 forecasted concepts to eliminate irrelevant concepts and balance the number of concepts and image features. We now investigate how the number of forecasted concepts affects the captions generated. To this end, we run our method through a series of scenarios using the number of forecasted concepts at 400, 200, 100, and 0. 

Table 2 shows the results of all tested scenarios on accuracy and descriptiveness. We can see that retrieving a large number of concepts ( _M_ = 400 or _M_ = 200) degrades performance. The reason is obvious because when we include a larger number of irrelevant concepts, the input becomes too noisy, preventing the model from selecting essential information. The model with _M_ = 100 forecasted concepts 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0008-00.png)


**----- Start of picture text -----**<br>
Oracle  A-CAP Ground-truth<br>image (for<br>Sparsely temporally-ordered images reference)<br>the bride and  night settles on<br>groom are about  this wonderous<br>to cut the cake. day and everyone<br>heads home.<br>**----- End of picture text -----**<br>


Figure 6. A case study of samples with low scores. Though our method generates a plausible caption, it is far from the groundtruth caption. The reason is that the oracle image changes significantly from the inputs. 

comes close to our best performance ( _M_ = 60). Finally, we examine an extreme case where no forecasted concept is employed ( _M_ = 0). The performance drops to the same level as that of VinVL [38] (first row in Table 1). This is due to the fact that the inputs to the two models are nearly identical. This experiment confirms that the number of forecasted concepts has an effect on our performance, implying that retrieving a sufficient number of concepts results in improved effectiveness. 

**A case study of samples with low scores.** While our method produces promising quantitative results, we notice a relatively small number of samples with low scores when delving into each sample in detail. We thus manually check those samples, as shown in Fig. 6. Given what is happening in the inputs, our generated caption is reasonable because the next step of the wedding party is “cutting a wedding cake”. The ground-truth caption, in contrast, is completely different because the scene shifts from “wedding” to “nighttime”. We recall that our hypothesis is that the scene does not change significantly, but in this case, it does. Though our method fails to predict the far future, it does correctly predict the near future. We may ignore such failures because they contradict our hypothesis. In fact, when we exclude those failure samples from quantitative comparison, our outperformance becomes more significant than before. **Limitations.** First, our method is heavily reliant on concept detection (here, clarifai). When we are unable to detect important concepts, our method is unable to predict the correct caption, as seen in Fig. 4, third example. Second, as shown in Table 2, the performance of our method is dependent on the number of forecasted concepts from commonsense knowledge. We use a simple filtering process in this paper, namely, computing the relevance score between concept and image context and empirically retaining _M_ = 60 forecasted concepts. Our strategy is effective, but it may not be optimal. To improve this issue, it is necessary to learn how to determine a suitable number of concepts. One possible solution is to learn concept selection while training the model. This is left for our future work. 

## **5. Discussions** 

We now discuss the potential negative societal impacts of our task. While we believe our introduced task will push 

more applications to make our lives safer and benefit downstream tasks, we have noticed that it has the potential to be abused. One of the concerns is that it will be used to predict behavior for nefarious purposes, such as criminal activity. 

Our task still has some difficulties. First, to the best of our knowledge, no suitable dataset exists to serve as a benchmark. Though our used VIST dataset [14] is useful to some extent, it is originally designed for the visual storytelling task, so it does not always meet task requirements, as already seen. As a result, a new dataset for this task is required, which should cover various scenarios such as near future, far future, abnormal thinking, and rationale. We should note that owing to the labor cost of creating a dataset, we are currently using the customized VIST to assess the performance of our method. Second, evaluating the task is difficult. Although appropriate evaluation metrics for the open domain are still unavailable, our used metrics are partially effective in our task. This is because, as we do not account for the diversity of potential futures, generating a caption close to the ground-truth (BLEU, CIDEr) is a valid indicator of the model’s predictive capability. Moreover, considering the dataset that we employed, CLIP-based scores are suitable for evaluating the degree of similarity between the generated captions and the oracle images, which are presumed to represent the future of the input images. In fact, our experiments show that the current metrics cannot evaluate the task thoroughly. User study may compensate for the automatic metrics, but it is expensive and subjective, as is customary. We believe that new metrics for this task can capitalize on the advantages of the vision-language space, such as CLIP [26]. Furthermore, new metrics should emphasize the rationale, which explains the reason why the model generates that caption but not another. 

## **6. Conclusion** 

We introduced a new task, called anticipation captioning, that generates a caption for an unseen oracle image, given a sparsely temporally-ordered set of images. For this new task, we proposed a baseline model (A-CAP), which incorporates commonsense knowledge into the off-the-shelf vision-language model VinVL. We evaluated A-CAP on a customized VIST dataset, showing that A-CAP outperforms other image captioning methods. We also addressed the potential positive and negative impacts of the task as well as its challenges, in order to encourage further research. 

**Acknowledgement.** This work was supported by the Institute of AI and Beyond of the University of Tokyo, JSPS/MEXT KAKENHI Grant Numbers JP19H04166, JP22H05015, and 22K17947, and the commissioned research (No. 225) by the National Institute of Information and Communications Technology (NICT), Japan. 

## **References** 

- [1] https://github.com/soodoku/clarifai. 3 

- [2] https://www.sbert.net/. 4 

- [3] https://github.com/microsoft/Oscar. 5 

- [4] https : / / huggingface . co / spaces / stabilityai / stable - diffusion. 6, 12, 13, 14, 15 

- [5] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In _ECCV_ , 2016. 6, 7 

- [6] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In _CVPR_ , 2018. 1, 2 

- [7] Gang Chen, Yang Liu, Huanbo Luan, Meng Zhang, Qun Liu, and Maosong Sun. Learning to generate explainable plots for neural story generation. _IEEE/ACM Trans. Audio, Speech and Lang. Proc._ , 2020. 2 

- [8] Hong Chen, Yifei Huang, Hiroya Takamura, and Hideki Nakayama. Commonsense knowledge aware concept selection for diverse and informative visual storytelling. In _AAAI_ , 2021. 2, 3 

- [9] Long Chen, Zhihong Jiang, Jun Xiao, and Wei Liu. Humanlike controllable image captioning with verb-specific semantic roles. In _CVPR_ , 2021. 2 

- [10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In _NAACL_ , 2019. 4, 11 

- [11] Sarah E. Finch and Jinho D. Choi. Towards unified dialogue system evaluation: A comprehensive analysis of current evaluation protocols. In _Proceedings of the 21th Annual Meeting of the Special Interest Group on Discourse and Dialogue_ , pages 236–245, 1st virtual meeting, July 2020. Association for Computational Linguistics. 6 

- [12] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. _ICLR_ , 2019. 1, 2 

- [13] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: a reference-free evaluation metric for image captioning. In _EMNLP_ , 2021. 6, 7 

- [14] Ting-Hao K. Huang, Francis Ferraro, Nasrin Mostafazadeh, Ishan Misra, Jacob Devlin, Aishwarya Agrawal, Ross Girshick, Xiaodong He, Pushmeet Kohli, Dhruv Batra, et al. Visual storytelling. In _NAACL_ , 2016. 1, 2, 4, 8 

- [15] Justin Johnson, Andrej Karpathy, and Li Fei-Fei. Densecap: Fully convolutional localization networks for dense captioning. In _CVPR_ , 2016. 2 

- [16] Vincent Le Guen and Nicolas Thome. Disentangling physical dynamics from unknown factors for unsupervised video prediction. In _CVPR_ , 2020. 2 

- [17] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. BART: Denoising sequence-tosequence pre-training for natural language generation, translation, and comprehension. In _ACL_ , 2020. 5, 6, 7, 12, 13, 14, 15 

- [18] Zhongyang Li, Xiao Ding, and Ting Liu. Story ending prediction by transferable bert. In _IJCAI_ , 2019. 2 

- [19] Lixin Liu, Jiajun Tang, Xiaojun Wan, and Zongming Guo. Generating diverse and descriptive image captions using visual paraphrases. In _ICCV_ , 2019. 6 

- [20] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized BERT pretraining approach. _CoRR_ , 2019. 4 

- [21] Jiasen Lu, Jianwei Yang, Dhruv Batra, and Devi Parikh. Neural baby talk. In _CVPR_ , 2018. 1, 2 

- [22] Luk´aˇs Neumann, Andrew Zisserman, and Andrea Vedaldi. Future event prediction: If and when. In _CVPRW_ , 2019. 1, 2 

- [23] Qiang Ning, Zhili Feng, Hao Wu, and Dan Roth. Joint reasoning for temporal and causal relations. In _ACL_ , 2018. 2 

- [24] Qiang Ning, Hao Wu, Rujun Han, Nanyun Peng, Matt Gardner, and Dan Roth. TORQUE: A reading comprehension dataset of temporal ordering questions. In _EMNLP_ , 2020. 2 

- [25] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In _ACL_ , 2002. 6, 7 

- [26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 6, 8 

- [27] Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In _NIPS_ , 2015. 3 

- [28] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In _CVPR_ , 2022. 1, 5, 6, 11 

- [29] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari, go, chess and shogi by planning with a learned model. _Nature_ , 2020. 1, 2 

- [30] Robyn Speer, Joshua Chin, and Catherine Havasi. Conceptnet 5.5: An open multilingual graph of general knowledge. In _AAAI_ , 2017. 2, 3, 4, 11, 18, 19 

- [31] Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In _CVPR_ , 2015. 6, 7 

- [32] Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li`o, and Yoshua Bengio. Graph attention networks. _ICLR_ , 2017. 4 

- [33] Duc Minh Vo, Hong Chen, Akihiro Sugimoto, and Hideki Nakayama. NOC-REK: novel object captioning with retrieved vocabulary from external knowledge. In _CVPR_ , 2022. 1, 2 

- [34] Carl Vondrick and Antonio Torralba. Generating the future with adversarial transformers. In _CVPR_ , 2017. 2 

- [35] Xin Wang, Wenhu Chen, Yuan-Fang Wang, and William Yang Wang. No metrics are perfect: Adversarial reward learning for visual storytelling. In _ACL_ , 2018. 1, 2, 5, 6, 7, 12, 13, 14, 15 

- [36] Dongchao Yang, Jianwei Yu, Helin Wang, Wen Wang, Chao Weng, Yuexian Zou, and Dong Yu. Diffsound: Discrete diffusion model for text-to-sound generation. _arXiv e-prints_ , 2022. 1 

- [37] Kuo-Hao Zen, William B. Shen, De-An Huang, Min Sun, and Juan Carlos Niebles. Visual forecasting by imitating dynamics in natural sequences. In _ICCV_ , 2017. 1, 2 

- [38] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Making visual representations matter in visionlanguage models. In _CVPR_ , 2021. 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15 

- [39] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In _CVPR_ , 2022. 2, 11 

- [40] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. _IJCV_ , 2022. 2, 11 

## **Supplemental Material** 

## **I. Prompt Learning** 

Prompt learning was developed by NLP research. It considers pre-trained language models such as BERT [10], as knowledge-based sources of useful information for downstream tasks. The key idea is to create a prompt (template) that can guide the pre-trained model through the adaptation process to a new task. It should be noted that the prompt format should be the same as the input format learned by the pre-trained model. Furthermore, the parameters of the pre-trained model are not updated during the training process; instead, we train the layers to learn prompt embeddings. The concept of prompt learning has recently been explored in computer vision [39, 40], where the contextword-generated prompt is converted into a set of learnable vectors and fed into a pre-trained vision-language model to solve downstream tasks. 

In our method, we use prompt learning in the same way as recent methods [39, 40]. We see that the key idea of VinVL [38] is the usage of concepts (object names), which allows better alignment between vision and language spaces, leading to the appearance of concepts in the caption. If we add forecasted concepts to the model, the model will be able to generate the caption based on the forecasted concepts. In our method, we combine detected and forecasted concepts to create the prompt. To this end, we change the VinVL’s input to words–(detected, forecasted)concepts– ROIs because the format of the prompt should be familiar to the pre-trained model (i.e., sequence of words–concepts– ROIs). During the training time, by using cross-entropy loss, we update the graph neural network to learn the embeddings for the concepts to ensure that the pre-trained model can understand the prompt embeddings. After training, the pre-trained model can easily generate the desired captions from the input. 

The observations from the additional examples support our conclusion that our method is better suited to the anticipation captioning task than the other methods and ablated models. 

## **III. Visualization of Knowledge Graph** 

We visualize the knowledge graphs corresponding to the examples in Fig.3 (main paper) in Figs. K, L, M, and N to better understand the contributions of forecasted concepts in the anticipated captions. The left graph in each figure is the full knowledge graph, which contains all detected and forecasted concepts. We see nodes in the graph are densely connected, meaning most nodes are related. We remark that the number of nodes is 100 (= 4 _×_ 10+60) and the number of edges is 6000 on average. 

The right graph, on the other hand, is the portion of the knowledge graph that is extracted using only the forecasted concepts (brown nodes) appearing in the anticipated caption and the detected concepts (blue nodes) related to the forecasted ones. We can see that our method successfully retrieves forecasted concepts from ConceptNet [30], which are the future of detected concepts. More importantly, our method can include forecasted concepts in the final caption thanks to our usage of prompt learning. 

## **II. More Examples** 

We randomly select more examples of captions generated by our method and our compared methods. They are shown in Figs. A, C, E, and G. We also show their corresponding generated images obtained by using stable diffusion model [28] in Figs. B, D, F, and H. Along with Fig. 3 in the main paper, these figures consistently demonstrate that our method generates captions that are more accurate, descriptive, and plausible than the other methods. 

In addition, Figs. I and J show the captions generated by ablated models: A-CAP w/o GNN, A-CAP w/o context, and our full model. We can see that, as stated in the main paper, the captions generated by A-CAP w/o GNN most likely describe the inputs, whereas those generated by A- CAP w/o context are far from the inputs. Meanwhile, our full model can produce plausible captions. 

|Sparsely temporally-ordered images|Oracle<br>image (for<br>reference)|VinVL|VinVL +<br>Oracle image|AREL + BART|**A-CAP**|Ground-truth|
|---|---|---|---|---|---|---|
|||everyone had a<br>great time.|it was a great<br>night with lots<br>of great food.|they were all<br>excited for the<br>event to start.|[ female ] was<br>excited to see<br>what was going<br>on.|this is [female]<br>watching as her<br>dad scores a<br>strike to win the<br>game.|
|||the couple was<br>wed and the<br>day was merry.|the bride's<br>parents were<br>especially<br>happy.|at the end of the<br>night, they all<br>took a group<br>photo to<br>remember the<br>day.|the bride and<br>groom made a<br>toast to their<br>new life<br>together.|the pastor<br>presenting them<br>as husband and<br>wife for the<br>first time.|
|||they finished<br>the night with a<br>glass of wine<br>and some good<br>food.|the lights were<br>still on by the<br>time we were<br>done.|we had a great<br>time and<br>enjoyed it.|after a long day<br>of drinking,<br>they went home<br>for the night.|later they all<br>left, full and<br>happy|
|||the band played<br>with purple<br>lights, stunning<br>the audience.|at the end of the<br>day, we all<br>went home.|the audience<br>was having a<br>great time.|the stage was<br>set up and<br>everyone was<br>ready for the<br>show to begin.|Afterwards the<br>graduates<br>congratulated<br>each other and<br>discussed the<br>directions their<br>lives were now<br>going in<br>beyond school.|



Figure A. Examples of generated captions obtained by all compared methods. We show the oracle images and ground-truth captions for reference purposes. VinVL [38] generates captions that are out of context with the input images. VinVL [38] + Oracle image sometimes generates reasonable captions. AREL [35] + BART [17] tends to generate a general ending for the sequence of images. On the other hand, our method A-CAP predicts more accurate, descriptive, and plausible captions than others. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0012-02.png)


**----- Start of picture text -----**<br>
Sparsely temporally-ordered images VinVL VinVL +  AREL + BART A-CAP Ground-truth<br>Oracle image<br>**----- End of picture text -----**<br>


Figure B. The generated images obtained by using stable diffusion model [4] to generate an image from each generated caption in Fig. A. The order of images is the same as the order of captions in Fig. A. The images generated using our captions are close to the ground-truth ones while those by other methods are not. 

|Sparsely temporally-ordered images|Oracle<br>image (for<br>reference)|VinVL|VinVL +<br>Oracle image|AREL + BART|**A-CAP**|Ground-truth|
|---|---|---|---|---|---|---|
|||and, to make it<br>less boring,<br>goof around<br>with some of<br>the products.|and of course,<br>there was no<br>food left for the<br>people to eat.|he wasn't sure if<br>he wanted to<br>buy it, but he<br>didn't want to<br>buy it at the<br>store.|finally, we got<br>to grab some<br>lunch at the<br>convenience<br>store.|time to check<br>out the food.|
|||they were there<br>to protest<br>against laws<br>they didn't<br>accept|and they paid<br>their respects to<br>those who had<br>lost their lives.|i had a great<br>time.|the government<br>officials also<br>honored the<br>fallen soldiers.|It was easy to<br>tell that he was<br>impressed<br>about the<br>special event<br>for veterans'<br>day|
|||this is the stuff I<br>got to keep in<br>the museum.|the jack - o -<br>lantern<br>competition had<br>some very<br>impressive<br>contestants.|they have a<br>great time and<br>are ready to go<br>home.|at the end of the<br>night, they all<br>made their own<br>pumpkins.|while others<br>made their own,<br>unique<br>creations.|
|||the stands<br>selling<br>sculptures are<br>also very<br>popular.<br>we went to the<br>waffle shop to<br>get some food.<br>i really enjoyed<br>all of the fresh<br>fruits and<br>vegetables.<br>we got to buy<br>some fresh fruit<br>and vegetables.<br>after lunch, we<br>found a small<br>candy shop and<br>got some<br>dessert.|||||



Figure C. Examples of generated captions obtained by all compared methods. We show the oracle images and ground-truth captions for reference purposes. VinVL [38] generates captions that are out of context with the input images. VinVL [38] + Oracle image sometimes generates reasonable captions. AREL [35] + BART [17] tends to generate a general ending for the sequence of images. On the other hand, our method A-CAP predicts more accurate, descriptive, and plausible captions than others. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0013-02.png)


**----- Start of picture text -----**<br>
Sparsely temporally-ordered images VinVL VinVL +  AREL + BART A-CAP Ground-truth<br>Oracle image<br>**----- End of picture text -----**<br>


Figure D. The generated images obtained by using stable diffusion mode [4] to generate an image from each generated caption in Fig. C. The order of images is the same as the order of captions in Fig. C. The images generated using our captions are close to the ground-truth ones while those by other methods are not. 

|Sparsely temporally-ordered images|Oracle<br>image (for<br>reference)|VinVL|VinVL +<br>Oracle image|AREL + BART|**A-CAP**|Ground-truth|
|---|---|---|---|---|---|---|
|||the celebration<br>began to wind<br>down as the<br>night dragged<br>on.|we had a great<br>time.|the men are<br>waiting for the<br>food to be<br>served.|the night ended<br>with people<br>camping in<br>tents on the side<br>of the road.|everyone made<br>friends.|
|||we had a great<br>time watching<br>the game.|the owner of<br>the car was<br>awarded a star.|afterward there<br>were a lot of<br>rides.|the arena was<br>packed with<br>people.|self driving cars<br>are shown to be<br>the future of<br>transportation.|
|||after the<br>ceremony, the<br>teams got to eat<br>outside.|and got to<br>throw the ball.|the game was<br>very exciting.|we had a great<br>time at the<br>baseball game.|he was tired by<br>the end of the<br>day, but it is a<br>day he will<br>remember<br>forever.|
|||the night ended<br>with an<br>amazing<br>fireworks<br>display.<br>the weather was<br>nice and we had<br>a great time.<br>they had a lot<br>of fun.<br>we had a great<br>time at the<br>concert.<br>when the show<br>starts<br>everybody is<br>cooking, eating<br>and waiting for<br>the speaker to<br>begin.|||||



Figure E. Examples of generated captions obtained by all compared methods. We show the oracle images and ground-truth captions for reference purposes. VinVL [38] generates captions that are out of context with the input images. VinVL [38] + Oracle image sometimes generates reasonable captions. AREL [35] + BART [17] tends to generate a general ending for the sequence of images. On the other hand, our method A-CAP predicts more accurate, descriptive, and plausible captions than others. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0014-02.png)


**----- Start of picture text -----**<br>
Sparsely temporally-ordered images VinVL VinVL +  AREL + BART A-CAP Ground-truth<br>Oracle image<br>**----- End of picture text -----**<br>


Figure F. The generated images obtained by using stable diffusion model [4] to generate an image from each generated caption in Fig. E. The order of images is the same as the order of captions in Fig. E. The images generated using our captions are close to the ground-truth ones while those by other methods are not. 

|Sparsely temporally-ordered images|Oracle<br>image (for<br>reference)|VinVL|VinVL +<br>Oracle image|AREL + BART|**A-CAP**|Ground-truth|
|---|---|---|---|---|---|---|
|||the park was<br>beginning to<br>close because it<br>was getting<br>late, so they<br>left.|we had a great<br>time and can't<br>wait to come<br>back next year.|they won a<br>prize at a prize.|it was a great<br>night and<br>everyone had a<br>good time.|the lobsters are<br>then place in<br>the tubs for the<br>guest to choose<br>from.|
|||my husband<br>loves a good<br>organization.|the principal<br>gave a<br>presentation for<br>the students.|they had a great<br>time at the<br>wedding.|[ male ] is tired<br>from all the<br>school day and<br>is ready to go<br>home.|he studied<br>throughout the<br>day, making<br>sure to finish<br>his homework.|
|||this is the stuff i<br>got to keep in<br>the museum.|everyone had a<br>great time.|after that we<br>played in the<br>water.|the day ended<br>with a picnic in<br>the park.|there were also<br>games for<br>children.|
|||the view from<br>our room is<br>breathtaking.|the boy was<br>really glad he<br>got out of the<br>car.|he rode his bike<br>down the road<br>and stopped for<br>a bite to eat.|as the sun went<br>down, we<br>headed back<br>home.|I think I’ll take<br>a nap and ride<br>my bike some<br>more tomorrow.|
||||||||



Figure G. Examples of generated captions obtained by all compared methods. We show the oracle images and ground-truth captions for reference purposes. VinVL [38] generates captions that are out of context with the input images. VinVL [38] + Oracle image sometimes generates reasonable captions. AREL [35] + BART [17] tends to generate a general ending for the sequence of images. On the other hand, our method A-CAP predicts more accurate, descriptive, and plausible captions than others. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0015-02.png)


**----- Start of picture text -----**<br>
Sparsely temporally-ordered images VinVL VinVL +  AREL + BART A-CAP Ground-truth<br>Oracle image<br>**----- End of picture text -----**<br>


Figure H. The generated images obtained by using stable diffusion model [4] to generate an image from each generated caption in Fig. G. The order of images is the same as the order of captions in Fig. G. The images generated using our captions are close to the ground-truth ones while those by other methods are not. 

|Sparsely temporally-ordered images|Oracle<br>image (for<br>reference)|A-CAP<br>w/o GNN|A-CAP<br>w/o<br>context|**A-CAP**<br>(full)|Ground-<br>truth|
|---|---|---|---|---|---|
|||everyone<br>had a good<br>time<br>carving<br>pumpkins.|they made<br>pumpkins<br>for<br>everyone<br>to eat.|at the end<br>of the<br>night, we<br>got to see<br>some scary<br>pumpkins.|what a<br>harvest at<br>halloween.|
|||it was a<br>great day<br>for both<br>teams.|we had a<br>great time.|after the<br>game, the<br>mascots<br>posed for a<br>picture<br>with their<br>fans.|the first<br>inning was<br>over and<br>the pitcher<br>was<br>practicing.|
|||they had a<br>lot of fun<br>in the car,<br>and they<br>had a lot of<br>fun.|the driver<br>of the<br>yellow car<br>happily<br>takes a<br>picture<br>with the<br>camera<br>crew to<br>celebrate<br>his win|the driver<br>of the red<br>and yellow<br>car took<br>the lead in<br>a race.|it is the<br>final lap<br>and the<br>lowe’s car<br>is in the<br>lead, the<br>crowd goes<br>wide.|
|||there were<br>many<br>people<br>walking<br>around in<br>the parade.|we had a<br>great time.|the parade<br>ended in<br>the center<br>of the city.|all in all, it<br>was a fun<br>celebration<br>.|



Figure I. Examples of generated captions by two ablated models: A-CAP w/o GNN, A-CAP w/o context, and full model A-CAP. We select two inputs where the detected concepts almost overlap. A-CAP w/o GNN generates captions that most likely describe the inputs. A-CAP w/o context generates captions that are far from the inputs and similar to each other. 

|Sparsely temporally-ordered images|Oracle<br>image (for<br>reference)|A-CAP<br>w/o GNN|A-CAP<br>w/o<br>context|**A-CAP**<br>(full)|Ground-<br>truth|
|---|---|---|---|---|---|
|||i had a<br>great time<br>there.|the family<br>of the bride<br>and groom<br>were happy<br>to be<br>together.|the family<br>and friends<br>gathered<br>for a group<br>photo.|however in<br>the end<br>seeing his<br>happy face<br>brought a<br>smile to<br>mine|
|||the family<br>was<br>incredibly<br>proud of<br>her<br>achieveme<br>nt.|the family<br>and friends<br>pose for<br>one last<br>photo<br>before<br>heading off<br>to the<br>hospital.|after the<br>ceremony,<br>the family<br>and friends<br>posed for a<br>picture on<br>the steps.|many<br>friends and<br>family<br>have come<br>to show<br>their<br>support.|
|||the food<br>was great<br>and i had a<br>great time.|the fruit<br>and<br>vegetables<br>were very<br>colorful.|we ended<br>the night<br>with a ride<br>on the<br>horse and<br>buggy.|our<br>evening<br>ended with<br>a carriage<br>ride back<br>to our<br>hotel.|
|||the<br>downtown<br>streets<br>were lined<br>with<br>people<br>enjoying<br>themselves|we ended<br>the day<br>with a ride<br>on a boat<br>on the<br>river.|then we<br>headed<br>back to the<br>market to<br>buy some<br>fresh fruit.|they all<br>looked so<br>good and<br>fresh.|



Figure J. Examples of generated captions by two ablated models: A-CAP w/o GNN, A-CAP w/o context, and full model A-CAP. We select two inputs where the detected concepts almost overlap. A-CAP w/o GNN generates captions that most likely describe the inputs. A-CAP w/o context generates captions that are far from the inputs and similar to each other. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0018-00.png)


Figure K. Visualization of knowledge graph of the first example in Fig. 3 in the main paper. The full graph is shown on the left, while the detected concepts (blue nodes) and forecasted concepts (brown nodes) that contribute to the caption are shown on the right. We can see that our method successfully retrieves forecasted concepts from ConceptNet [30], which are the future of detected concepts. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0018-02.png)


Figure L. Visualization of knowledge graph of the second example in Fig. 3 in the main paper. The full graph is shown on the left, while the detected concepts (blue nodes) and forecasted concepts (brown nodes) that contribute to the caption are shown on the right. We can see that our method successfully retrieves forecasted concepts from ConceptNet [30], which are the future of detected concepts. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0019-00.png)


Figure M. Visualization of knowledge graph of the third example in Fig. 3 in the main paper. The full graph is shown on the left, while the detected concepts (blue nodes) and forecasted concepts (brown nodes) that contribute to the caption are shown on the right. We can see that our method successfully retrieves forecasted concepts from ConceptNet [30], which are the future of detected concepts. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge_images/cvpr_2023_A-Cap_Anticipation_Captioning_With_Commonsense_Knowledge.pdf-0019-02.png)


Figure N. Visualization of knowledge graph of the fourth example in Fig. 3 in the main paper. The full graph is shown on the left, while the detected concepts (blue nodes) and forecasted concepts (brown nodes) that contribute to the caption are shown on the right. We can see that our method successfully retrieves forecasted concepts from ConceptNet [30], which are the future of detected concepts. 

