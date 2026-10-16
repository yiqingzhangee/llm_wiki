## DENSE VIDEO OBJECT CAPTIONING FROM DISJOINT SUPERVISION 

**Xingyi Zhou[*] Anurag Arnab[*] Chen Sun Cordelia Schmid** Google DeepMind 

## ABSTRACT 

We propose a new task and model for _dense video object captioning_ – detecting, tracking and captioning trajectories of objects in a video. This task unifies spatial and temporal localization in video, whilst also requiring fine-grained visual understanding that is best described by natural language. We propose a unified model, and demonstrate how our end-to-end approach is more accurate and temporally coherent than a multi-stage pipeline combining state-of-the-art detection, tracking, and captioning models. Moreover, we propose a training strategy based on a mixture of disjoint tasks, which allows us to leverage diverse, large-scale datasets which supervise different parts of our model. Although each pretraining task only provides weak supervision, they are complementary and, when combined, result in noteworthy zero-shot ability and serve as strong initialization for additional finetuning to further improve accuracy. We carefully design new metrics capturing all components of our task, and show how we can repurpose existing video grounding datasets ( _e.g_ . VidSTG and VLN) for our new task. We show that our model improves upon a number of strong baselines for this new task. Furthermore, we can apply our model to the task of spatial grounding, outperforming prior state-of-the-art on VidSTG and VLN, without explicitly training for it. Code is available at https://github.com/google-research/scenic. 

## 1 INTRODUCTION 

Powered by gigantic datasets and models, _language_ is becoming the output modality of the most capable artificial intelligence models (Team et al., 2023; Alayrac et al., 2022; Ouyang et al., 2022; Li et al., 2023; Liu et al., 2023; Tong et al., 2024; Li et al., 2024a). Language unifies different tasks with the same output space (Raffel et al., 2020; Chen et al., 2023a), is more descriptive than discrete class labels (Wu et al., 2022a; Long et al., 2023), and naturally facilitates zero-shot prediction of novel tasks (Radford et al., 2021; Brown et al., 2020). Inspired by advances in natural language understanding, the vision community has explored language in a number of tasks including image captioning (Chen et al., 2015), dense image captioning (Krishna et al., 2017b), question answering (Antol et al., 2015), video captioning (Monfort et al., 2021) and representation learning (Radford et al., 2021). However, likely due to the scarcity of large-scale, aligned training data, we are not aware of any existing single vision-language model that unifies both fine-grained **spatial** - (by detecting objects) and **temporal** - (by reasoning across time in videos) understanding. 

In this paper, we propose a new task and model for _dense video object captioning_ (Dense VOC) – the task of generating captions of trajectories of all objects from video (Fig. 1). Dense VOC requires understanding across space, time, and language (Fig. 2), and is therefore a superset of existing vision tasks, namely object detection (Everingham et al., 2015; Lin et al., 2014), multi-object tracking (Dendorfer et al., 2021; Dave et al., 2020) and captioning (Chen et al., 2015). 

A prominent challenge for training our model is that datasets with captioned trajectories are scarce. However, annotations for each sub-task, or even each combination of the sub-tasks, are abundant. For example, we can train our object proposal component using image-level object detection labels from COCO (Lin et al., 2014), and the captioning component from video-level captioning datasets like SMiT (Monfort et al., 2021). These disjoint training tasks are complementary, and in combination supervise our entire model. This enables us to perform our Dense VOC task in a zero-shot manner, and we show that we can achieve noteworthy performance despite not having access to any full, 

> *Equal contribution. {zhouxy, aarnab}@google.com 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision_images/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision.pdf-0002-00.png)


**----- Start of picture text -----**<br>
t = 1 t = 2 t = 3 t = 4<br>A child in gray bounces a basketball outdoors There is a bench in the background<br>A child in blue clothes plays basketball with another child A brown basketball<br>**----- End of picture text -----**<br>


Figure 1: **Overview of the dense video object captioning (Dense VOC) task.** Given a video, we predict object trajectories (identities denoted by colors) and their natural language description. We show a video from the VidSTG (Zhang et al., 2020) validation set. 

captioned object trajectories during training. Furthermore, this pretraining serves as a powerful initialization for finetuning on the full Dense VOC task, where limited annotations are available. 

Another challenge in our task is to produce holistic and consistent captions for objects across frames. Note that a baseline of applying a strong, dense image captioning model per-frame, and then linking objects together is poorly suited to this scenario: the captions at each frame are likely to be different due to subtle appearance changes across frames. This motivates our end-to-end trained model, which includes a novel end-to-end tracking algorithm that aggregates features of the same object across time, enabling the subsequent captioner to leverage global features to produce coherent captions. 

Although we are the first to our knowledge to study Dense VOC, we can still repurpose existing video grounding datasets for evaluation and domain-specific finetuning. We use VidSTG (Zhang et al., 2020) and VLN (Voigtlaender et al., 2023), originally designed for spatiotemporal sentence grounding: Instead of finding an object tube given a sentence query (grounding), we predict object trajectories directly and use the sentence queries as the ground truth captions. In addition, we show that our generative model trained for Dense VOC can perform grounding by simply 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision_images/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision.pdf-0002-05.png)


**----- Start of picture text -----**<br>
Object detection<br>Space<br>Object tracking<br>Object captioning Dense VOC<br>Video recognition<br>Time<br>Language<br>Image Captioning  Video captioning<br>**----- End of picture text -----**<br>


Figure 2: **Overview of Dense VOC** . Our problem involves understanding across space, time, and language, and thus encompasses other vision tasks, which typically consider one or two of these axes. We show these subtasks are complementary, and pretraining on them enables zero-shot generalization to Dense VOC. 

selecting the bounding boxes with the maximum likelihood of producing the query sentence. We also develop a new metric that jointly measures captioning, detection and tracking accuracy by extending HOTA (Luiten et al., 2021), the most popular metric for multi-object tracking. 

Experiments show that our end-to-end trained Dense VOC model outperforms baselines consisting of strong, per-task models by a substantial margin, producing more accurate and inherently temporally consistent captions. Moreover, we achieve significant improvements from our disjoint, multi-dataset training. We additionally show how we can readily apply our model to related domain-specific datasets: by finetuning our model on a recent person tracking and captioning dataset, BenSMOT (Li et al., 2024b), we outperform prior work by 18 _._ 2 points. Furthermore, by applying our generative captioning model to the discriminative grounding task, we are able to outperform dedicated spatial grounding models on both VidSTG and VLN. In summary, we propose the following contributions: 

1. We propose the new task of Dense Video Object Captioning. We propose novel evaluation metrics, and repurpose existing grounding datasets for evaluation. 

2. We design an end-to-end architecture for our task, with a novel tracking algorithm and feature aggregator that ensures temporally consistent captions. Unlike conventional offline trackers, our tracker is trained end-to-end with the model and produces long-term trajectory features for subsequent captioning. 

3. We show our model can be trained without full annotations for the task, with a mixture of disjoint datasets which supervise different parts of our model. 

4. We further show how our models generalize to downstream video grounding tasks, achieving state-of-the-art results on two datasets, without explicitly being trained for grounding. 

5. Moreover, we significantly improves the state-of-the-art on the BenSMOT dataset Li et al. (2024b) for Semantic Multi-Object Tracking. 

2 

## 2 RELATED WORK 

**Image captioning** (Chen et al., 2015; Anderson et al., 2018; Xu et al., 2015; Rennie et al., 2017) describes the content of an image with language. State-of-the-art methods map the input image to output text by using multi-modal models (Jiang et al., 2020; Desai & Johnson, 2021; Li et al., 2020; Zhang et al., 2021a; Li et al., 2023; Yu et al., 2022) pretrained on large datasets (Sharma et al., 2018; Radford et al., 2021). For example, GIT (Wang et al., 2022) simple forwards vision tokens from a ViT encoder (Dosovitskiy et al., 2021) to an auto-regressive language decoder (Vaswani et al., 2017; Devlin et al., 2019). Similar ideas apply to **video captioning** (Xu et al., 2016; Zhou et al., 2018; Monfort et al., 2021), by concatenating (Wang et al., 2022) or pooling (Yan et al., 2022) features from each frame, before feeding them to an auto-regressive text decoder. Our work builds on existing captioning architectures (Wang et al., 2022), and extends them to trajectory captioning using our end-to-end model and weak supervision (Monfort et al., 2021; Krishna et al., 2017b; Lin et al., 2014). 

**Dense object captioning** in contrast, detects objects in an image and describes them with text (Johnson et al., 2016; Li et al., 2019; Shao et al., 2022; Wu et al., 2022a). It was popularized by the Visual Genome (Krishna et al., 2017b) dataset, which contains full annotations for the task. Early work, DenseCap (Johnson et al., 2016) used a one-stage detector (Redmon et al., 2016) followed by an LSTM text decoder (Hochreiter & Schmidhuber, 1997) on dense feature maps. Most recently, GRiT (Wu et al., 2022a) built upon the state-of-the-art image captioning architecture of GIT (Wang et al., 2022), and generated object captions, also with a transformer decoder (Vaswani et al., 2017), from RoI-pooled (He et al., 2017) image features. Our model advances architectures like GRiT to videos and incorporates end-to-end tracking. We also note that **dense video captioning** in the literature refers to the task of localizing and captioning multiple events _temporally_ in videos (Krishna et al., 2017a; Zhou et al., 2018; Wang et al., 2021a; Yang et al., 2023a). Our task, in contrast, involves tracking and captioning objects in a video, and therefore requires _spatial_ localization, which is why we name our task “dense video object captioning”. 

**Multi-object tracking** detects objects and track them with a consistent identity label. The predominant approach is tracking-after-detection (Bewley et al., 2016; Zhang et al., 2021b; Du et al., 2021), _i.e_ . first running detectors on each frame and then using a separate tracker to link them. While this works well for existing benchmarks with only a few classes (Dendorfer et al., 2021; Geiger et al., 2012; Yang et al., 2019), it is more challenging in our case: we need tracks _before_ captioning to have a single, consistent textual output for the whole trajectory. Thus, our work follows **end-to-end multi-object tracking** (Cheng et al., 2022; Li et al., 2022a; Wang et al., 2021c; Zhou et al., 2022b). We adopt a global tracker GTR (Zhou et al., 2022b), which casts tracking as pairwise association among all objects within a video. Whilst GTR applies a sliding-window-based identity association algorithm during inference as a post-processing step, we design an efficient algorithm to perform this process end-to-end. This is necessary for our task, since our trajectory features are used by a subsequent captioning module which is trained jointly. We are not aware of prior work which efficiently assigns object identities and corresponding features to tracks, and trains end-to-end through this process. Finally, note that **video object tracking and segmentation** (Yang et al., 2021; 2023b; Yang & Yang, 2022; Cheng & Schwing, 2022; Cheng et al., 2024) focuses on following only a _single_ object which is given in the first frame (Perazzi et al., 2016; Xu et al., 2018). This is therefore a different setting from our task of detecting, tracking and captioning multiple objects. 

**Video object grounding** (Zhang et al., 2020; Voigtlaender et al., 2023) finds a spatio-temporal tube given a video and query sentence as inputs. Existing, discriminative methods (Zhang et al., 2020; Yang et al., 2022; Jin et al., 2022; Su et al., 2021) co-embed visual and text inputs, and use the sentence feature to find the corresponding object. In contrast, we use our generative language model for this task by selecting the object with the highest likelihood of producing the query. To our knowledge, we are the first work to explore the alternate paradigm of generative models for this task. Finally, we note that these tasks are also related to video-referring segmentation (Bellver et al., 2020; Wu et al., 2022b; Yu et al., 2016) which grounds textual queries to segmentation masks. Segmentation, however, is not the focus of our work. 

Concurrent to our work, BeyondMOT (Li et al., 2024b) proposes an video object tracking and captioning benchmark and model. We highlight two differences: 1. Li et al. (2024b) uses a frameby-frame tracker similar to our baselines (Tab. 2), and we propose a novel end-to-end tracker. 2. Our work aims to track and caption **all objects** in the video, while Li et al. (2024b) handles **only** 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision_images/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Tracking Trajectory captioning<br>Class-agnostic  Grouping Aggregator Language<br>detector (Sec. 3.2) (Alg. 1) (Sec. 3.3) Decoder<br>RoI features Association matrix Identity Trajectory features Object captions<br>Video<br>… A dog picking up a toy<br>… A toy on the<br>ground<br>**----- End of picture text -----**<br>


Figure 3: **Overview of our model.** Our end-to-end model has three modules: First it produces object proposals per-frame using a class-agnostic detector (left, trained with detection loss, _Lobject_ ). These object proposals are then passed to an end-to-end tracking module that groups objects into trajectories (middle, trained with association loss, _Lassoc_ ). The identities produced by the tracking module are used to aggregate features which are then fed to a language decoder to produce the final caption (right, trained with caption loss _Lcaption_ ). Our model can be trained end-to-end with partial supervision on different and disjoint datasets to provide zero-shot Dense VOC capabilities. 

**persons** . As a result, our task is much more challenging, and we show our model yields superior performance on their benchmark. OW-VISCap (Choudhuri et al., 2024) on the other hand augments a video segmentation model, Cheng et al. (2022), with a language model (OPT with 2.7 billion parameters (Zhang et al., 2022a)) head for video segmentation and captioning. In contrast, our model is trained flexibly using our disjoint pretraining, which enables us to achieve better detection and tracking performance whilst still using a substantially smaller model. 

## 3 METHOD 

As shown in Fig. 3, our end-to-end model consists of interlinked heads for object proposal, tracking and captioning the resulting trajectories. Before introducing our novel components, we review prior techniques for captioning and dense object captioning in images (Wu et al., 2022a; Wang et al., 2022). 

## 3.1 BACKGROUND 

Image captioning maps an input image, **I** _∈_ R _[H][×][W][ ×]_[3] , to a caption _c_ = ( _y_ 1 _, y_ 2 _, . . . , ynt_ ) which is a sequence of up to _nt_ text tokens from a given vocabulary. The minimal set of components is an image encoder, followed by a text decoder (Vaswani et al., 2017). The encoder maps the input image **I** , to a feature representation **f** _∈_ R _[n][v][×][d]_ consisting of _nv_ tokens with dimensionality _d_ . The subsequent text decoder is auto-regressive (Graves, 2013) – it predicts the next text token, _yi_ , as a function of both the image features, **f** , and previously generated text tokens, **y** 0: _i−_ 1, denoted by _yi_ = Decode( **f** _,_ **y** 0: _i−_ 1). Note that the first step of decoding begins with _y_ 0 = BOS, a special beginning-of-sentence token, and the caption ends when the end-of-sentence token, EOS, is output by the model. This simple image captioning model has been demonstrated to be effective and scalable by GIT (Wang et al., 2022), achieving state-of-the-art results across a number of captioning datasets. 

GRiT (Wu et al., 2022a) extends the approach further to dense object captioning of images: Here, the authors use an object proposal network (Zhou et al., 2019) to produce a set of _K_ class-agnostic bounding boxes, _b_ 1 _, b_ 2 _, . . . , bK_ . Features corresponding to each of these objects are obtained using RoIAlign (He et al., 2017), resulting in a localized feature, _fk ∈_ R _[r][×][r][×][d]_ where _r_ = 7 is the output resolution of RoIAlign. Each of these grid features is flattened into _fk ∈_ R _[r]_[2] _[×][d]_ and decoded independently by the text decoder, as done in GIT. Therefore, the loss used to train a GRiT model consists of _L_ = _Lobject_ + _Lcaption_ where _Lcaption_ is a cross-entropy loss over all text tokens in the vocabulary, and _Lobject_ consists of bounding box regression and objectness terms, as standard in object detection literature (Zhou et al., 2019; Ren et al., 2015; Lin et al., 2017). 

We now describe how we extend object captioning to videos by tracking object proposals over time (Sec. 3.2) and aggregating trajectory features and captioning them (Sec. 3.3) in an end-to-end fashion. Section 3.4 explains how we train our model, whilst Sec. 3.5 describes how we apply our model directly to video object grounding tasks. 

3.2 END-TO-END TRACKING 

As shown in Fig. 3 (left), we first produce object proposals separately for each frame. Tracking then aims to assign each object in each frame a unique trajectory identity _δ ∈_ N. We define **f** _k[t][∈]_[R] _[r]_[2] _[×][d]_[ as] the ROI feature of object proposal _k_ in frame _t_ , **F** = [ **f** _k[t]_[]] _[T,K] t_ =1 _,k[t]_ =1[as the concatenation of all object] 

4 

**Algorithm 1:** Identity assignment from association matrix. This greedy algorithm can be implemented efficiently on accelerators, enabling end-to-end training. 

|**Algorithm 1:**<br>Identity assignment from association matrix. This greedy algorithm can be<br>implemented effcientlyon accelerators, enablingend-to-end training.|**Algorithm 1:**<br>Identity assignment from association matrix. This greedy algorithm can be<br>implemented effcientlyon accelerators, enablingend-to-end training.|
|---|---|
|**Input**<br>**:** Association Matrix**A**_∈_R**TK**_×_**TK** //_T_: num. frames. _K_: num. objects per frame.||
|**Hyperparameters :**Association|score threshold_θ_|
|**Output**<br>**:** Identities for each object_δ ∈_N_T K_||
|_M ←T × K_|// Number of total objects.|
|_A ←_preprocess(_A_)|// Preprocess_A_to ensure object pairs in the same frame have a score of0.|
|ˆ_A ←_(_A ≥θ_)_._astype(bool)|// Binary matrix for possible merges.|
|_δ ←zeros_(_M_)|// Initialize output identities, shape(_M,_)|
|id_count_←_0|// Initialize ID count.|
|**while** ˆ_A.any() > 0_**do**||
|track_len_←_ˆ_A_.sum(axis=1)|// Number of objects in each merge.|
|i_←_track_len.argmax()|// Find the longest track to merge.|
|id_count_←_id_count + 1|// Create a new identity.|
|_δ ←δ_+id_count * ˆ_Ai_|// Assign the current track a new ID using ˆ_Ai_.|
|ˆ_A ←_ˆ_A −_ˆ_Ai·|_ˆ_A·i_|// Remove merged indices. “_|_” is logical or.|
|**end**||
|**return**_δ_||



features in the video. Let _M_ = _|_ **F** _|_ =[�] _[T] t_ =1 _[K][t]_[ as the total number of objects in all frames, where] _Kt_ is the number of object proposals at the _t_[th] frame. Thus, we have **F** _∈_ R _[M][×][r]_[2] _[×][d]_ . 

From these object features, **F** , we predict a global association matrix, **A** _∈_ R _[M][×][M]_ , where **A** _ij_ = 1 if the objects denoted by the _i[th]_ row and _j[th]_ column are from the same trajectory (Fig. 3 middle). Otherwise, **A** _ij_ = 0 means that they are from different trajectories, or one of them is the background. 

We use a transformer module, H, with two self-attention layers, similar to Zhou et al. (2022b), to predict the association matrix **A** = _σ_ (H( **F** )), where _σ_ is the sigmoid activation. Given the object trajectory annotations, we construct the ground truth association matrix **A[¯]** for **A** , where **A[¯]** _ij_ = 1 if and only if row _i_ and column _j_ of **A** are matched to the same ground truth trajectory using an Intersection over Union (IoU) criteria of 0.5. The training loss _Lassoc_ for this module is then a binary cross entropy between **A** and **A[¯]** , _Lassoc_ = _M_ 1 � _ij_[BCE][(] _[A][ij][,][A]_[¯] _[ij]_[)][.] 

After constructing our association matrix, **A** , we need to aggregate object-level features according to identities _δ_ = [ _δk[t]_[]] _[T,K] t_ =1 _,k[t]_ =1[, to generate trajectory-level captions for the next captioning stage.][Here,] _δk[t]_[denotes the identity of the] _[ k]_[-th object proposal in the] _[ t]_[-th frame.][We design a greedy grouping] algorithm (Alg. 1) operating on **A** to obtain _δ_ . Concretely, we greedily extract the longest trajectory from untracked objects, until there are no possible associations left (indicated by the association score being above a threshold _θ_ ). This guarantees each trajectory has at most one object in each frame. This algorithm can be implemented efficiently on accelerators, allowing us to backpropagate through it. 

As aforementioned, prior trackers (Zhang et al., 2021b; Zhou et al., 2020; 2022a) do not explicitly perform identity assignment within the model, but rather as a post-processing step since tracking is the final output for such methods. Our work efficiently assigns object identities to tracks in an end-to-end trainable network, which enables us to perform joint trajectory-level captioning training as described next. 

## 3.3 TRAJECTORY CAPTIONING 

Our end-to-end tracking module produces object features, **f** _k_ (we omit the frame index _t_ below for clearer notation), paired with their identities, _δk_ , which denote their correspondence over time. We now describe two methods for aggregating features along this trajectory in order to caption it. 

**Soft aggregation.** A straightforward way to leverage object features over time is to compute a weighted sum to combine them into a single, global trajectory feature. We observe that the association matrix, **A** (Sec. 3.2), already serves as a summation weight. Specifically, we set **G** = _||_ **AA** _||[·]_ **[ F]**[,] where _·_ denotes matrix multiplication, and _|| · ||_ normalizes **A** by rows. Each row of **G** , **g** _k ∈_ R _[r]_[2] _[×][d]_ , therfore denotes an aggregated feature over its trajectory for object _k_ . 

**Hard aggregation.** An alternative to weighted temporal averaging is to concatenate and construct new trajectory features. Let **f** _τ_ = _{_ **f** _k′}δk′_ = _τ_ be the set of all object features with identity _τ_ . We note **f** _τ_ can be as long as the entire video, and thus it may be expensive to directly use **f** _τ_ . Therefore, we uniformly 

5 

|Dataset|Annotation type|Train set size (103)_Lobject _|Train set size (103)_Lobject _|_Lassoc Lcaption_|
|---|---|---|---|---|
|COCO (Lin et al., 2014)|Image detection|118|✓||
|VG (Krishna et al., 2017b)|Image object captioning|70|✓|✓|
|SMiT (Monfort et al., 2021)|Video captioning|480||✓|
|Aug-COCO (Lin et al., 2014)|Video object tracking|118|✓|✓|



Table 1: **Datasets for pretraining.** We supervise different losses based on available annotations. 

sample a subset of object features from the trajectory, denoted as **g** _τ_ = UniformSample( **f** _δ, m_ ), where **g** _τ ∈_ R _[mr]_[2] _[×][d]_ , inspired by Wang et al. (2022). _m_ is the number of sampled frames, and we set _m_ = 6 following ablations in Appendix C.2. 

The trajectory-aggregated features for each object, **g** _k_ , are then autoregressively decoded into output captions for each object, **y** _k_ . This follows Sec. 3.1, where _yk,i_ = Decode( **g** _k,_ **y** _k,_ 0: _i−_ 1). Note that the language decoder has the same parameters as in single-frame object captioning, but processes more input tokens. Therefore, we train it in the same manner with a softmax cross-entropy loss over the vocabulary of text tokens, denoted by _Lcaption_ . 

## 3.4 PRETRAINING WITH DISJOINT SUBTASKS 

As shown in Fig. 3, our model is trained with the loss function, _L_ = _Lobject_ + _Lassoc_ + _Lcaption_ . When we have full Dense VOC annotations, which supervise each component of our model we can train our entire model end-to-end. However, to leverage more weakly-labeled data, we can also decompose Dense VOC into subtasks, and use each subtask to supervise the relevant part of our model using the available annotations as shown in Tab. 1. This approach also enables us to perform our final task in a zero-shot manner ( _i.e_ . without training on any full Dense VOC annotations). 

**Object detection.** Using detection datasets for images, we can train the object proposal generator with _Lobject_ . We use COCO (Lin et al., 2014) as it is the most popular dataset for this task. 

**Dense captioning in images.** Dense object captioning datasets of images allow us to train both the object proposal generator and the text decoder, by supervising _Lobject_ and _Lcaption_ . Here, we use Visual Genome (Krishna et al., 2017b), the largest dataset for this task. 

**Global video captioning.** Video captioning datasets help us to reduce the domain gap to our final task by also training on video. In particular, we use Spoken Moments in Time (SMiT) (Monfort et al., 2021) which is the largest dataset for this task and contains narrations for short clips (roughly 3 seconds). As there are no object annotations, but only video-level captions, we construct an object proposal from the entire frame and caption that with our text decoder, applying _Lcaption_ . This approach is inspired by prior work on weakly-supervised object detection (Zhou et al., 2022a; Bilen & Vedaldi, 2016; Arnab et al., 2020). 

**Tracking.** Training the tracking module of our network (Sec. 3.2) requires annotations that associate detections of an object identity throughout the video. We found that existing tracking datasets either have too limited vocabularies for general objects (MOT (Dendorfer et al., 2021), KITTI (Geiger et al., 2012), YouTube VIS (Yang et al., 2019)), or are too small (TAO (Dave et al., 2020) and UVO (Wang et al., 2021b) label 600 and 5 000 videos respectively), and thus giving unsatisfactory results in our setting (Appendix C.3). As a result, following existing work (Zhang et al., 2021b; Zhou et al., 2020), we instead augment image datasets into tracking ones by applying two different data augmentations to the same image, and then linearly interpolating the frames in between to form a pseudo-video. In particular, we augment COCO (referred to as Aug-COCO (Zhou et al., 2020)). This enables us to apply _Lassoc_ and _Lobject_ when training our model. 

## 3.5 APPLICATION TO VIDEO OBJECT GROUNDING 

The task of video object grounding (Zhang et al., 2020; Voigtlaender et al., 2023) consists of two inputs: a video, **V** , and a sentence query, _c_ ¯. The output is a sequence of bounding boxes, [ _b[s] , b[s]_[+1] _, . . . , b[e]_ ], corresponding to the sentence query, where _s_ and _e_ are the indices of the start and end frames respectively. 

Our model, however, generates captions, _c_ , at the output, rather than requiring it as an input. To apply our model to grounding, we follow an analogous approach to prior works that performed closed-set image classification with captioning models (Alayrac et al., 2022; Chen et al., 2023b): we evaluate the likelihood (i.e., exponential negative cross-entropy loss) of the sentence query, _c_ ¯, for each of the object trajectories produced by our model. In practice, we find that instead of just taking the 

6 

object trajectory with the highest sentence-likelihood, we achieve higher accuracy by weighting the likelihood by the detection score, _sk_ , from our object proposal module. Thus, given bounding boxes, trajectory features and detection scores, _{_ ( _b[t] k[, s][t] k[,]_ **[ g]** _[k]_[)] _[}][T,K] t_ =1 _,k[t]_ =1[, we choose the bounding boxes with] the highest weighted likelihood: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision_images/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision.pdf-0007-01.png)


## 4 EXPERIMENTAL EVALUATION 

As we are proposing a new task, there is no dedicated dataset or evaluation metric for dense video object captioning for all objects. Fortunately, existing video grounding datasets (Zhang et al., 2020; Voigtlaender et al., 2023) have annotations for object trajectories and their captions, allowing us to repurpose them for Dense VOC, as defined next. We also report results on concurrent person-focused video object tracking and captioning benchmark, BenSMOT(Li et al., 2024b). 

## 4.1 DATASETS 

**VidSTG (Zhang et al., 2020)** was originally created for spatio-temporal sentence grounding, but can be used for Dense VOC: Each video annotates multiple textual queries and their corresponding spatio-temporal tubes. By aggregating these across all videos, we obtain the paired trajectory-caption annotations that we need for training and evaluating our model. 

VidSTG has exhaustive trajectory ( _i.e_ . bounding box and tracking) annotations for all objects (Shang et al., 2019), but not all objects are used in grounding, and thus not all objects have captions. We account for this fact in both training and testing. Specifically, we do not compute _Lcaption_ on objects without caption annotations, and also exclude them during evaluation (see Sec. 4.2). In particular, when a prediction is matched to a ground truth without caption annotations, we do not evaluate its captioning metrics, but still evaluate detection metrics. The dataset contains 5,436 training videos and 602 validation videos, with each video being at most 200 frames long. We use the declarative annotations from the dataset containing 19,000 captioned trajectories for training. 

**Video Localized Narratives (VLN) (Voigtlaender et al., 2023)** augments existing datasets by narrating the “actors” in a video. We therefore use these narrations as our target captions. We use the subset from the UVO dataset (Wang et al., 2021b) as UVO has exhaustive detection and tracking annotations for all objects. Like VidSTG, the captions are not exhaustive for all objects, so we exclude objects without captions in both training and evaluating the captioning module. Each video has bounding box annotations for 3 sparsely sampled frames, and thus we train and evaluate on these frames. The dataset contains a total of 5,136 training and 2,451 validation videos. 

**BenSMOT (Li et al., 2024b)** contains person bounding boxes trajectories and their manuallyannotated captions for 3292 YouTube videos. The dataset has in average 2.2 trajectories per video. 

## 4.2 EVALUATION METRICS 

**Captioned-HOTA (CHOTA).** Our primary metric, CHOTA, builds on Higher Order Tracking Accuracy (HOTA) (Luiten et al., 2021) – which is now the most popular metric in multi-object tracking – by adding a captioning term. HOTA decomposes tracking into two subproblems: detection and association, with the final score being the geometric mean of detection accuracy (DetA) and _|T P |_ Association Accuracy (AssA): HOTA = _√_ DetA _·_ AssA. Here, DetA = _|T P |_ + _|F P |_ + _|F N |_[, and AssA] averages the “Association IoU” over true-positives, as AssA = _|T P_ 1 _|_[(][�] ( _x,y_ ) _∈T P_[Ass-IoU(x, y))][,] where ( _x, y_ ) are the matched prediction-ground truth box pairs in each frame. Note that HOTA computes the DetA and AssA for each detection in each frame, rather than for each trajectory, as the overall trajectory performance is implicitly measured by the association of detections over time. Moreover, it considers all possible trajectory matches that can be made simultaneously (Sec. 7 of Luiten et al. (2021)). 

Our task consists of captioning, detection and association. Therefore, we also define an additional “Captioning Accuracy” (CapA) term as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision_images/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision.pdf-0007-12.png)


7 

|#||CHOTA|DetA|AssA|CapA|Consistent captions|
|---|---|---|---|---|---|---|
|1|Per-frame cap. w. IOU tracker|49.9|**64.4**|52.2|37.1|✗|
|2|Per-frame cap. w. FairMOT Zhang et al. (2021b)|51.2|63.4|57.2|37.0|✗|
|3|Per-frame cap. w. ByteTrack Zhang et al. (2022b)|52.3|64.2|60.2|37.1|✗|
|4|Middle-frame cap. w. ByteTrack Zhang et al. (2022b) 50.7||64.2|60.2|33.8|✓|
|5|Ours, soft aggregation|54.6|**64.4**|**65.9**|38.4|✓|
|6|Ours, hard aggregation|**54.9**|64.2|**65.9**|**39.1**|✓|



Table 2: **Comparison of our end-to-end model to per-task baselines on VidSTG validation.** Our models are based on #2 of Tab. 3 right. The image dense captioning models used in the baselines (rows #1-#4) are trained on the same datasets, and run off-the-shelf trackers as post-processing. Our end-to-end approach improves across all metrics, and produces temporally consistent captions. 

which uses three popular image-captioning metrics (Chen et al., 2015), and _TP[′]_ are the true-positive detection pairs that have caption annotations (as discussed in Sec. 4.1). Note that for compatibility with HOTA, we follow DetA and AssA and compute CapA separately per-object on each frame. The final metric is then CHOTA = _[√]_ 3 DetA _·_ AssA _·_ CapA, effectively adding a captioning term to the HOTA metric. We include further details and code in Appendix B, along with results using the image dense object captioning metrics, mAP-METEOR (Appendix C.1). 

## 4.3 IMPLEMENTATION DETAILS 

Our implementation is based on the public release of GRiT (Wu et al., 2022a). GRiT uses a ViTDetBase (Li et al., 2022b) backbone (initialized with CLIP (Radford et al., 2021)), a CenterNet (Zhou et al., 2019) object proposal network and RoI Head, and a randomly-initialized text decoder. 

We first train our model for general Dense VOC on large-scale disjoint datasets (Sec. 3.4). During disjoint pretraining, we sample batches from different datasets with an even ratio, (1: 1: 1: 1), thus avoiding additional hyperparameters. For video datasets, we sample 8 frames for a video and use a local batch size of 1. For image datasets, we use a local batch size of 8. We train our model on 32 GPUs, which means we have an effective batch size of 256 images or 32 videos. 

We then evaluate the models on the two fully-annotated datasets (Sec. 4.1) in both zero-shot and full-finetuning setups. For VidSTG, we sample 16 frames during training, and then run on all 200 frames during testing. For VLN, we use all 3 annotated frames in both training and evaluation. In both cases, we use an input size of 384 _×_ 384. During inference, we threshold the outputs of our object proposal module with a score of 0 _._ 5, and only track the remaining objects. We include exhaustive implementation details and hyperparameters in Appendix B.2 with the full code. 

## 4.4 ANALYSIS OF END-TO-END TRACKING 

We first study the benefits of our end-to-end model in Tab. 2. We do this by comparing to multiple, strong baseline models running in sequence. Concretely, we use the state-of-the-art image-dense object captioning model Wu et al. (2022a) followed by tracking as a post-processing step. We use trackers ranging from a simple IoU-based tracker Wu et al. (2019) to more recent, sophisticated methods like FairMOT (Zhang et al., 2021b), and ByteTrack (Zhang et al., 2022b). 

As the baseline predicts captions independently on each frame, the caption is not consistent over the entire trajectory. Therefore, we consider an additional baseline where we only use the caption from the middle frame of the trajectory. Finally, note that as our baseline captioner is pretrained on Visual Genome, and then finetuned on individual frames of VidSTG, it has been trained on identical data to our model, allowing us to make fair comparisons. 

As shown in Tab. 2, per-frame captioners followed by offline trackers produce temporally inconsistent captions (#1-#3). Naively selecting the caption from the middle frame as the trajectory-level caption produces temporally consistent captions, but comes at the cost of captioning accuracy, as a single frame may not be representative of the entire event (#4). Both variants of our model (#5 and #6) improve tracking quality substantially, as shown by their large improvement on AssA, demonstrating the benefits of end-to-end training and incorporating temporal information. Our model improves on CapA too, showing that improved object trajectories provide better features for subsequent captioning. Finally, we note that the quality of the initial detections at each frame, measured by DetA, does not really change between the baselines and our method. This does, however, show that training our model jointly with multiple loss functions does not compromise performance on individual tasks. 

Overall, our end-to-end model (#6) improves the CHOTA by 2.6 points over the best baseline (#3). As hard aggregation performs slightly better, we use it in our following experiments. 

8 

|#COCO VG SMiT Aug<br>COCO|VidSTG (zero-shot)<br>VLN (zero-shot)<br>VidSTG (fnetuned)<br>VLN (fnetuned)<br>CHOTA DetA AssA CapA CHOTA DetA AssA CapA CHOTA DetA AssA CapA CHOTA DetA AssA CapA<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>47.8<br>54.6 57.8 34.5<br>29.7<br>35.3 85.4<br>8.7<br>-<br>48.9<br>-<br>-<br>-<br>27.8<br>-<br>-<br>52.3<br>64.9 63.0 34.9<br>31.8<br>43.9 88.7<br>8.2<br>-<br>17.8<br>-<br>7.8<br>-<br>12.1<br>-<br>7.4<br>54.9<br>64.2 65.9 39.1<br>40.6<br>**45.1** 88.4 16.7<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>45.4<br>51.9 56.9 31.6<br>37.4<br>41.2 87.7 14.5<br>-<br>19.1<br>-<br>8.5<br>-<br>14.3<br>-<br>8.5<br>55.2<br>64.0 67.1 39.2<br>41.0<br>44.2 88.4 **17.8**<br>-<br>49.9<br>-<br>8.1<br>-<br>28.0<br>-<br>7.8<br>55.6<br>65.7 68.9 38.4<br>40.9<br>44.1 88.8 17.4<br>-<br>50.4<br>-<br>4.9<br>-<br>28.7<br>-<br>7.5<br>54.4<br>64.9 63.9 38.8<br>35.6<br>43.7 88.5 11.6<br>-<br>51.3<br>-<br>9.1<br>-<br>**29.9**<br>-<br>9.0<br>56.5<br>**65.8** 68.2<br>**40.1**<br>41.1<br>44.2 88.9<br>17.7<br>**31.1**<br>**51.4 59.6**<br>**9.8**<br>**29.2**<br>29.1 **88.0**<br>**9.7**<br>**56.9**<br>**65.8 70.4** 39.7<br>**41.3**<br>44.3<br>**89.5** 17.7|VidSTG (zero-shot)<br>VLN (zero-shot)<br>VidSTG (fnetuned)<br>VLN (fnetuned)<br>CHOTA DetA AssA CapA CHOTA DetA AssA CapA CHOTA DetA AssA CapA CHOTA DetA AssA CapA<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>47.8<br>54.6 57.8 34.5<br>29.7<br>35.3 85.4<br>8.7<br>-<br>48.9<br>-<br>-<br>-<br>27.8<br>-<br>-<br>52.3<br>64.9 63.0 34.9<br>31.8<br>43.9 88.7<br>8.2<br>-<br>17.8<br>-<br>7.8<br>-<br>12.1<br>-<br>7.4<br>54.9<br>64.2 65.9 39.1<br>40.6<br>**45.1** 88.4 16.7<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>45.4<br>51.9 56.9 31.6<br>37.4<br>41.2 87.7 14.5<br>-<br>19.1<br>-<br>8.5<br>-<br>14.3<br>-<br>8.5<br>55.2<br>64.0 67.1 39.2<br>41.0<br>44.2 88.4 **17.8**<br>-<br>49.9<br>-<br>8.1<br>-<br>28.0<br>-<br>7.8<br>55.6<br>65.7 68.9 38.4<br>40.9<br>44.1 88.8 17.4<br>-<br>50.4<br>-<br>4.9<br>-<br>28.7<br>-<br>7.5<br>54.4<br>64.9 63.9 38.8<br>35.6<br>43.7 88.5 11.6<br>-<br>51.3<br>-<br>9.1<br>-<br>**29.9**<br>-<br>9.0<br>56.5<br>**65.8** 68.2<br>**40.1**<br>41.1<br>44.2 88.9<br>17.7<br>**31.1**<br>**51.4 59.6**<br>**9.8**<br>**29.2**<br>29.1 **88.0**<br>**9.7**<br>**56.9**<br>**65.8 70.4** 39.7<br>**41.3**<br>44.3<br>**89.5** 17.7|VidSTG (zero-shot)<br>VLN (zero-shot)<br>VidSTG (fnetuned)<br>VLN (fnetuned)<br>CHOTA DetA AssA CapA CHOTA DetA AssA CapA CHOTA DetA AssA CapA CHOTA DetA AssA CapA<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>47.8<br>54.6 57.8 34.5<br>29.7<br>35.3 85.4<br>8.7<br>-<br>48.9<br>-<br>-<br>-<br>27.8<br>-<br>-<br>52.3<br>64.9 63.0 34.9<br>31.8<br>43.9 88.7<br>8.2<br>-<br>17.8<br>-<br>7.8<br>-<br>12.1<br>-<br>7.4<br>54.9<br>64.2 65.9 39.1<br>40.6<br>**45.1** 88.4 16.7<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>45.4<br>51.9 56.9 31.6<br>37.4<br>41.2 87.7 14.5<br>-<br>19.1<br>-<br>8.5<br>-<br>14.3<br>-<br>8.5<br>55.2<br>64.0 67.1 39.2<br>41.0<br>44.2 88.4 **17.8**<br>-<br>49.9<br>-<br>8.1<br>-<br>28.0<br>-<br>7.8<br>55.6<br>65.7 68.9 38.4<br>40.9<br>44.1 88.8 17.4<br>-<br>50.4<br>-<br>4.9<br>-<br>28.7<br>-<br>7.5<br>54.4<br>64.9 63.9 38.8<br>35.6<br>43.7 88.5 11.6<br>-<br>51.3<br>-<br>9.1<br>-<br>**29.9**<br>-<br>9.0<br>56.5<br>**65.8** 68.2<br>**40.1**<br>41.1<br>44.2 88.9<br>17.7<br>**31.1**<br>**51.4 59.6**<br>**9.8**<br>**29.2**<br>29.1 **88.0**<br>**9.7**<br>**56.9**<br>**65.8 70.4** 39.7<br>**41.3**<br>44.3<br>**89.5** 17.7|VidSTG (zero-shot)<br>VLN (zero-shot)<br>VidSTG (fnetuned)<br>VLN (fnetuned)<br>CHOTA DetA AssA CapA CHOTA DetA AssA CapA CHOTA DetA AssA CapA CHOTA DetA AssA CapA<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>47.8<br>54.6 57.8 34.5<br>29.7<br>35.3 85.4<br>8.7<br>-<br>48.9<br>-<br>-<br>-<br>27.8<br>-<br>-<br>52.3<br>64.9 63.0 34.9<br>31.8<br>43.9 88.7<br>8.2<br>-<br>17.8<br>-<br>7.8<br>-<br>12.1<br>-<br>7.4<br>54.9<br>64.2 65.9 39.1<br>40.6<br>**45.1** 88.4 16.7<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>45.4<br>51.9 56.9 31.6<br>37.4<br>41.2 87.7 14.5<br>-<br>19.1<br>-<br>8.5<br>-<br>14.3<br>-<br>8.5<br>55.2<br>64.0 67.1 39.2<br>41.0<br>44.2 88.4 **17.8**<br>-<br>49.9<br>-<br>8.1<br>-<br>28.0<br>-<br>7.8<br>55.6<br>65.7 68.9 38.4<br>40.9<br>44.1 88.8 17.4<br>-<br>50.4<br>-<br>4.9<br>-<br>28.7<br>-<br>7.5<br>54.4<br>64.9 63.9 38.8<br>35.6<br>43.7 88.5 11.6<br>-<br>51.3<br>-<br>9.1<br>-<br>**29.9**<br>-<br>9.0<br>56.5<br>**65.8** 68.2<br>**40.1**<br>41.1<br>44.2 88.9<br>17.7<br>**31.1**<br>**51.4 59.6**<br>**9.8**<br>**29.2**<br>29.1 **88.0**<br>**9.7**<br>**56.9**<br>**65.8 70.4** 39.7<br>**41.3**<br>44.3<br>**89.5** 17.7|VidSTG (zero-shot)<br>VLN (zero-shot)<br>VidSTG (fnetuned)<br>VLN (fnetuned)<br>CHOTA DetA AssA CapA CHOTA DetA AssA CapA CHOTA DetA AssA CapA CHOTA DetA AssA CapA<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>47.8<br>54.6 57.8 34.5<br>29.7<br>35.3 85.4<br>8.7<br>-<br>48.9<br>-<br>-<br>-<br>27.8<br>-<br>-<br>52.3<br>64.9 63.0 34.9<br>31.8<br>43.9 88.7<br>8.2<br>-<br>17.8<br>-<br>7.8<br>-<br>12.1<br>-<br>7.4<br>54.9<br>64.2 65.9 39.1<br>40.6<br>**45.1** 88.4 16.7<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>45.4<br>51.9 56.9 31.6<br>37.4<br>41.2 87.7 14.5<br>-<br>19.1<br>-<br>8.5<br>-<br>14.3<br>-<br>8.5<br>55.2<br>64.0 67.1 39.2<br>41.0<br>44.2 88.4 **17.8**<br>-<br>49.9<br>-<br>8.1<br>-<br>28.0<br>-<br>7.8<br>55.6<br>65.7 68.9 38.4<br>40.9<br>44.1 88.8 17.4<br>-<br>50.4<br>-<br>4.9<br>-<br>28.7<br>-<br>7.5<br>54.4<br>64.9 63.9 38.8<br>35.6<br>43.7 88.5 11.6<br>-<br>51.3<br>-<br>9.1<br>-<br>**29.9**<br>-<br>9.0<br>56.5<br>**65.8** 68.2<br>**40.1**<br>41.1<br>44.2 88.9<br>17.7<br>**31.1**<br>**51.4 59.6**<br>**9.8**<br>**29.2**<br>29.1 **88.0**<br>**9.7**<br>**56.9**<br>**65.8 70.4** 39.7<br>**41.3**<br>44.3<br>**89.5** 17.7|
|---|---|---|---|---|---|
|0<br>1<br>✓<br>2<br>✓<br>3<br>✓|-<br>-<br>-<br>-<br>-<br>48.9<br>-<br>-<br>-<br>17.8<br>-<br>7.8<br>-<br>-<br>-<br>-|-<br>-<br>-<br>-<br>-<br>27.8<br>-<br>-<br>-<br>12.1<br>-<br>7.4<br>-<br>-<br>-<br>-|47.8<br>54.6 57.8 34.5<br>52.3<br>64.9 63.0 34.9<br>54.9<br>64.2 65.9 39.1<br>45.4<br>51.9 56.9 31.6|29.7<br>35.3 <br>31.8<br>43.9 <br>40.6<br>**45.1** <br>37.4<br>41.2|85.4<br>8.7<br> 88.7<br>8.2<br> 88.4 16.7<br> 87.7 14.5|
|4<br>✓<br>✓<br>5<br>✓<br>✓<br>6<br>✓<br>✓<br>7<br>✓<br>✓<br>✓<br>8<br>✓<br>✓<br>✓<br>✓|-<br>19.1<br>-<br>8.5<br>-<br>49.9<br>-<br>8.1<br>-<br>50.4<br>-<br>4.9<br>-<br>51.3<br>-<br>9.1<br>**31.1**<br>**51.4 59.6**<br>**9.8**|-<br>14.3<br>-<br>8.5<br>-<br>28.0<br>-<br>7.8<br>-<br>28.7<br>-<br>7.5<br>-<br>**29.9**<br>-<br>9.0<br>**29.2**<br>29.1 **88.0**<br>**9.7**|55.2<br>64.0 67.1 39.2<br>55.6<br>65.7 68.9 38.4<br>54.4<br>64.9 63.9 38.8<br>56.5<br>**65.8** 68.2<br>**40.1**<br>**56.9**<br>**65.8 70.4** 39.7|41.0<br>44.2 <br>40.9<br>44.1 <br>35.6<br>43.7 <br>41.1<br>44.2 <br>**41.3**<br>44.3|88.4 **17.8**<br> 88.8 17.4<br> 88.5 11.6<br> 88.9<br>17.7<br>**89.5** 17.7|



Table 3: **Zero-shot (left) and finetuning (right) evaluation of our disjoint trained models with varying datasets.** We show results on VidSTG (Zhang et al., 2020) and VLN (Voigtlaender et al., 2023). Each row is a model pretrained on the specified datasets for zero-shot evaluation and then finetuned on the downstream datasets. #0 is finetuned from a CLIP checkpoint. For models without tracking supervision (#1–7), we cannot report their zero-shot association accuracy (AssA). Our full model (#8) gains full Dense VOC ability from disjoint training, and shows good performance on all metrics with or without finetuning, on both datasets. Detailed captioning metrics are in Appendix C.4. 

## 4.5 ANALYSIS OF DISJOINT TRAINING 

**Zero-shot evaluation.** We first pretrain on multiple disjoint datasets (Sec. 3.4), and evaluate zero-shot on our target datasets, VidSTG and VLN, without training on them in Tab. 3 left. Zero-shot evaluation is simple to perform for captioning models compared to classification, thanks to their open vocabulary. 

As mentioned in Tab. 1 and Fig. 3, each dataset supervises different parts of our model. For example, a model that is only trained on COCO (#1 in Tab. 3), is only trained with _Lobject_ , meaning that it only produces object proposals which we can evaluate with the Detection Accuracy component of our CHOTA metric. Visual Genome (VG) can supervise both the object proposal and captioning heads of our model. However, there is a large domain gap between the captions in VG and our target datasets, since the captions in VG are for single images and focus on very different vocabularies. Furthermore, VG tends to annotate bounding boxes around object parts rather than entire objects. Consequently, our zero-shot DetA is low when training only on VG (#2). To mitigate the differences in the type of bounding boxes annotated by VG, we ignore _Lobject_ on it when using it in conjunction with COCO. Note that we cannot evaluate a model trained only on SMiT, as it does not produce bounding boxes. 

We observe in Tab. 3 (left) that the different datasets have complementary properties: Adding COCO improves detection accuracy (#2 to #5, #4 to #7), and adding SMiT improves the captioning accuracy (#2 to #4, #5 to #7) even though SMiT only captions at a video-level. Finally, training with AugCOCO allows us to also supervise _Lassoc_ and thus the tracking module of our model. A model trained on all the datasets (#8) can therefore perform the full Dense VOC task, and shows good performance on all individual metrics compared to models trained on fewer datasets. Notably, we observe our final model with tracking improves captioning ability (CapA) without adding captioning training data. Similar to Tab. 2, the improvements are likely from our ability to leverage temporal information. 

**Finetuning evaluation.** We now finetune each of the pretrained models from Tab. 3 left and show results in the right. We also include a baseline (#0) which initializes from only a CLIP-pretrained checkpoint, observing that this model performs poorly. Once again, we observe that different pretraining datasets are complementary, as adding either SMiT or COCO (#2 to #4, #2 to #5, #1 to #6) improves results further. Adding more pretraining datasets improves results further (#7), and we achieve the best results with our model pretrained on all pretraining datasets (#8), which outperforms the best single-dataset pretrained model by 2.0 CHOTA on VidSTG, and 0.7 CHOTA on VLN. The improvement over only a CLIP-pretrained checkpoint is even larger, by 9.1 CHOTA and 11.6 CHOTA on the two respective datasets. Qualitative visualizations are shown in the supplement. 

## 4.6 COMPARISON TO CONCURRENT WORKS BENSMOT AND OW-VISCAP. 

We compare to the concurrent work Li et al. (2024b), which focuses on person category rather than all classes like our task. We finetune our model on the training set using the same hyper-parameters as our VidSTG experiments. Our full model achieved 90.19 HOTA and 0.254 CIDEr on this benchmark, significantly outperforming Li et al. (2024b). There are two major advantages of our model: our disjoint pretraining, and our use of a larger backbone (ViT-B vs. DLA-34 in BeyondMOT). We further break down the improvements by removing these two components in Tab. 4. The results show: 1. Our pretraining provides consistent gains on Li et al. (2024b) benchmark, improving especially captioning metrics. 2. With a small backbone and no pretraining, our model still outperforms Li et al. (2024b) on tracking and captioning metrics, showing the advantages of our end-to-end architecture. 

9 

||HOTA|DetA|AssA|CIDEr||CHOTA<br>DetA<br>AssA|CapA|
|---|---|---|---|---|---|---|---|
|Li et al. (2024b)|71.98|80.79|73.71|0.087|OW-VisCap 53.0<br>60.1<br>54.0||43.9|
|Ours|**90.19**|**90.79**|**89.59**|**0.254**|Ours|56.9<br>65.8<br>70.4|39.7|
|- Pretrain<br>- Backbone|88.56<br>86.55|89.38<br>84.33|87.74<br>89.19|0.150<br>0.129|Table 5: <br>|**Compare with concurrent work OW**<br>||



Table 5: **Compare with concurrent work OWVisCap (Choudhuri et al., 2024) on VidSTG.** The results are from Choudhuri et al. (2024) Tab. 2 and our Tab. 3 #8 under the same setting. Our model are better at detection and tracking, with lower captioning accuracy due to a smaller langauge head (46M vs. 2.7B params.). 

Table 4: **State-of-the-art comparison on BenSMOT (Li et al., 2024b).** Our model outperforms Li et al. (2024b) on comparable setting (no extra data, small backbone), and our full model improved 18 _._ 2 HOTA and 0.167 CIDEr. 

|Backbone<br>Rec. & Prec._>_0_._5<br>ReferFormer<br>ResNet50<br>48.3<br>GRiT<br>ViT-B<br>62.1<br>Ours<br>ViT-B<br>**65.1**|Finetuned<br>Zero-shot|
|---|---|
||STVGBert<br>47.3<br>-<br>TubeDETR<br>59.0<br>-<br>STCAT<br>61.7<br>-|
||Ours<br>**61.9**<br>54.1|



Table 6: **State-of-the-art comparison of spatial grounding on VLN Location-QA Voigtlaender et al. (2023).** We report the official metric, which evaluates if bounding box recall and precision are both above 0.5. We compare to the ReferFormer baseline Voigtlaender et al. (2023), GRiT Wu et al. (2022a), and our model (#8 of Tab. 3 left). 

Table 7: **State-of-the-art comparison of spatial grounding on the VidSTG** with STVGBert Su et al. (2021), TubeDETR Yang et al. (2022), and STCAT Jin et al. (2022). All models use ground truth temporal localization. 

We compare to OW-VISCap which uses a Mask2Former architecture with video object queries. Tab. 5 shows an improved overall performance in CHOTA. Our largest improvement is in Association Accuracy, showing that our end-to-end tracking module (Sec. 3.2) outperforms the Mask2Former counterparts. OW-VisCap gets a higher captioning accuracy as they used a substantially larger, 2.7 billion parameter OPT language decoder (Zhang et al., 2022a), whilst we used a smaller, 46 million parameter language model as in GIT (Wang et al., 2022). 

## 4.7 STATE-OF-THE-ART COMPARISON ON VIDEO GROUNDING 

As introduced in Sec. 3.5, Dense VOC models can be directly used for sentence grounding, by finding the proposals with the maximum likelihood of generating the query. We evaluate spatial grounding on the VLN Location-QA (Voigtlaender et al., 2023) and VidSTG (Zhang et al., 2020) benchmarks. 

**VLN Location-QA** consists of questions starting with “Where is”, and requires the model to produce a bounding box at each frame in the video. The task is therefore effectively a sentence grounding problem, and indeed, the ReferFormer (Wu et al., 2022b) baseline used by Voigtlaender et al. (2023) performs sentence grounding after removing “Where is” from the question. We also remove this prefix before grounding following Sec. 3.5 for both our final model, and an additional GRiT baseline. 

In this dataset, only one annotated frame (unknown at inference time) is evaluated, and this benchmark therefore effectively does not involve temporal localization. As the annotation of this dataset is based on mouse traces instead of bounding boxes, the evaluation metric considers bounding box coverage (recall) and precision (full details in Voigtlaender et al. (2023)). As shown in Tab. 6, we improve substantially over ReferFormer and our GRiT (Wu et al., 2022a) baseline. 

**VidSTG** requires producing a sequence of bounding boxes for a given sentence query. The evaluation metric is the average of the Intersection over Union (IoU) at each frame, between the predicted and ground truth bounding boxes for the target object. We compare to other prior works on this dataset in Tab. 7, assuming that the input video is already trimmed temporally to the objects of interest. Our model achieves the best IoU, outperforming models designed specifically for grounding, thereby showing that our generative framework can be used effectively in the discriminative grounding task. We also evaluate zero-shot without training on VidSTG, and still perform competitively. This emphasizes the efficacy of our disjoint pretraining. We provide more results in Appendix C.5. 

## 5 CONCLUSION 

We proposed the new task of dense video object captioning. Although this task requires expensive annotations across space, time and language, we show that we can train a model on existing largerscale datasets for disjoint subtasks. We show our proposed end-to-end architecture is important for producing more accurate and coherent captions. 

10 

## REFERENCES 

- Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In _NeurIPS_ , 2022. 

- Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In _CVPR_ , 2018. 

- Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In _ICCV_ , 2015. 

- Anurag Arnab, Chen Sun, Arsha Nagrani, and Cordelia Schmid. Uncertainty-aware weakly supervised action detection from untrimmed videos. In _ECCV_ , 2020. 

- Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In _ACL Workshops_ , 2005. 

- Miriam Bellver, Carles Ventura, Carina Silberer, Ioannis Kazakos, Jordi Torres, and Xavier Giro-i Nieto. Refvos: A closer look at referring expressions for video object segmentation. _arXiv:2010.00263_ , 2020. 

- Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, and Ben Upcroft. Simple online and realtime tracking. In _ICIP_ , 2016. 

- Hakan Bilen and Andrea Vedaldi. Weakly supervised deep detection networks. In _CVPR_ , 2016. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In _NeurIPS_ , 2020. 

- Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, et al. Pali-x: On scaling up a multilingual vision and language model. In _arXiv preprint arXiv:2305.18565_ , 2023a. 

- Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointly-scaled multilingual language-image model. In _ICLR_ , 2023b. 

- Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. _arXiv:1504.00325_ , 2015. 

- Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Maskedattention mask transformer for universal image segmentation. In _CVPR_ , 2022. 

- Ho Kei Cheng and Alexander G. Schwing. XMem: Long-term video object segmentation with an atkinson-shiffrin memory model. In _ECCV_ , 2022. 

- Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, and Alexander Schwing. Putting the object back into video object segmentation. In _CVPR_ , 2024. 

- Anwesa Choudhuri, Girish Chowdhary, and Alexander G Schwing. Ow-viscap: Open-world video instance segmentation and captioning. _NeurIPS_ , 2024. 

- Achal Dave, Tarasha Khurana, Pavel Tokmakov, Cordelia Schmid, and Deva Ramanan. Tao: A large-scale benchmark for tracking any object. In _ECCV_ , 2020. 

- Patrick Dendorfer, Aljosa Osep, Anton Milan, Konrad Schindler, Daniel Cremers, Ian Reid, Stefan Roth, and Laura Leal-Taixé. Motchallenge: A benchmark for single-camera multiple target tracking. _IJCV_ , 2021. 

- Karan Desai and Justin Johnson. Virtex: Learning visual representations from textual annotations. In _CVPR_ , 2021. 

11 

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In _NAACL_ , 2019. 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_ , 2021. 

Yuming Du, Wen Guo, Yang Xiao, and Vincent Lepetit. 1st place solution for the uvo challenge on video-based open-world segmentation 2021. _arXiv:2110.11661_ , 2021. 

- Mark Everingham, SM Ali Eslami, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes challenge: A retrospective. _IJCV_ , 2015. 

- Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In _CVPR_ , 2012. 

Alex Graves. Generating sequences with recurrent neural networks. _arXiv:1308.0850_ , 2013. 

Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In _ICCV_ , 2017. 

Lisa Anne Hendricks, Kaylee Burns, Kate Saenko, Trevor Darrell, and Anna Rohrbach. Women also snowboard: Overcoming bias in captioning models. In _ECCV_ , 2018. 

Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. _Neural computation_ , 1997. 

- Huaizu Jiang, Ishan Misra, Marcus Rohrbach, Erik Learned-Miller, and Xinlei Chen. In defense of grid features for visual question answering. In _CVPR_ , 2020. 

Yang Jin, Zehuan Yuan, Yadong Mu, et al. Embracing consistency: A one-stage approach for spatio-temporal video grounding. In _NeurIPS_ , 2022. 

- Justin Johnson, Andrej Karpathy, and Li Fei-Fei. Densecap: Fully convolutional localization networks for dense captioning. In _CVPR_ , 2016. 

Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In _ICCV_ , 2017a. 

- Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _IJCV_ , 2017b. 

- Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. _arXiv:2408.03326_ , 2024a. 

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _ICML_ , 2023. 

- Xiangtai Li, Wenwei Zhang, Jiangmiao Pang, Kai Chen, Guangliang Cheng, Yunhai Tong, and Chen Change Loy. Video k-net: A simple, strong, and unified baseline for video segmentation. In _CVPR_ , 2022a. 

- Xiangyang Li, Shuqiang Jiang, and Jungong Han. Learning object context for dense captioning. In _AAAI_ , 2019. 

- Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In _ECCV_ , 2020. 

Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. Exploring plain vision transformer backbones for object detection. In _ECCV_ , 2022b. 

Yunhao Li, Hao Wang, Qin Li, Xue Ma, Jiali Yao, Shaohua Dong, Heng Fan, and Libo Zhang. Beyond mot: Semantic multi-object tracking. _ECCV_ , 2024b. 

12 

- Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _ECCV_ , 2014. 

- Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In _ICCV_ , 2017. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In _NeurIPS_ , 2023. 

- Yanxin Long, Youpeng Wen, Jianhua Han, Hang Xu, Pengzhen Ren, Wei Zhang, Shen Zhao, and Xiaodan Liang. Capdet: Unifying dense captioning and open-world detection pretraining. In _CVPR_ , 2023. 

- Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixé, and Bastian Leibe. Hota: A higher order metric for evaluating multi-object tracking. _IJCV_ , 2021. 

- Mathew Monfort, SouYoung Jin, Alexander Liu, David Harwath, Rogerio Feris, James Glass, and Aude Oliva. Spoken moments: Learning joint audio-visual representations from video descriptions. In _CVPR_ , 2021. 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In _NeurIPS_ , 2022. 

- Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In _CVPR_ , 2016. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _JMLR_ , 2020. 

- Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In _CVPR_ , 2016. 

- Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In _NeurIPS_ , 2015. 

- Steven J Rennie, Etienne Marcheret, Youssef Mroueh, Jerret Ross, and Vaibhava Goel. Self-critical sequence training for image captioning. In _CVPR_ , 2017. 

- Xindi Shang, Donglin Di, Junbin Xiao, Yu Cao, Xun Yang, and Tat-Seng Chua. Annotating objects and relations in user-generated videos. In _ICMR_ , 2019. 

- Zhuang Shao, Jungong Han, Demetris Marnerides, and Kurt Debattista. Region-object relation-aware dense captioning via transformer. _IEEE Transactions on Neural Networks and Learning Systems_ , 2022. 

- Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _ACL_ , 2018. 

- Rui Su, Qian Yu, and Dong Xu. Stvgbert: A visual-linguistic transformer based framework for spatio-temporal video grounding. In _ICCV_ , 2021. 

- Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv:2312.11805_ , 2023. 

- Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. In _NeurUPS_ , 2024. 

13 

- Generalized Intersection Over Union. A metric and a loss for bounding box regression. In _CVPR_ , 2019. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _NeurIPS_ , 2017. 

- Paul Voigtlaender, Soravit Changpinyo, Jordi Pont-Tuset, Radu Soricut, and Vittorio Ferrari. Connecting vision and language with video localized narratives. In _CVPR_ , 2023. 

- Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lijuan Wang. Git: A generative image-to-text transformer for vision and language. _TMLR_ , 2022. 

Teng Wang, Ruimao Zhang, Zhichao Lu, Feng Zheng, Ran Cheng, and Ping Luo. End-to-end dense video captioning with parallel decoding. In _ICCV_ , 2021a. 

- Weiyao Wang, Matt Feiszli, Heng Wang, and Du Tran. Unidentified video objects: A benchmark for dense, open-world segmentation. In _ICCV_ , 2021b. 

- Yuqing Wang, Zhaoliang Xu, Xinlong Wang, Chunhua Shen, Baoshan Cheng, Hao Shen, and Huaxia Xia. End-to-end video instance segmentation with transformers. In _CVPR_ , 2021c. 

- Jialian Wu, Jianfeng Wang, Zhengyuan Yang, Zhe Gan, Zicheng Liu, Junsong Yuan, and Lijuan Wang. Grit: A generative region-to-text transformer for object understanding. _arXiv:2212.00280_ , 2022a. 

Jiannan Wu, Yi Jiang, Peize Sun, Zehuan Yuan, and Ping Luo. Language as queries for referring video object segmentation. In _CVPR_ , 2022b. 

- Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2. https://github.com/facebookresearch/detectron2, 2019. 

Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In _CVPR_ , 2016. 

- Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhudinov, Rich Zemel, and Yoshua Bengio. Show, attend and tell: Neural image caption generation with visual attention. In _ICML_ , 2015. 

- Ning Xu, Linjie Yang, Yuchen Fan, Jianchao Yang, Dingcheng Yue, Yuchen Liang, Brian Price, Scott Cohen, and Thomas Huang. Youtube-vos: Sequence-to-sequence video object segmentation. In _ECCV_ , 2018. 

Shen Yan, Tao Zhu, Zirui Wang, Yuan Cao, Mi Zhang, Soham Ghosh, Yonghui Wu, and Jiahui Yu. Video-text modeling with zero-shot transfer from contrastive captioners. _arXiv:2212.04979_ , 2022. Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Tubedetr: Spatiotemporal video grounding with transformers. In _CVPR_ , 2022. 

- Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning. In _CVPR_ , 2023a. 

Linjie Yang, Yuchen Fan, and Ning Xu. Video instance segmentation. In _ICCV_ , 2019. 

- Zongxin Yang and Yi Yang. Decoupling features in hierarchical propagation for video object segmentation. In _NeurIPS_ , 2022. 

- Zongxin Yang, Yunchao Wei, and Yi Yang. Associating objects with transformers for video object segmentation. In _NeurIPS_ , 2021. 

- Zongxin Yang, Xiaohan Wang, Jiaxu Miao, Yunchao Wei, Wenguan Wang, and Yi Yang. Scalable video object segmentation with identification mechanism. _arXiv preprint arXiv:2203.11442_ , 2023b. 

14 

Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. _TMLR_ , 2022. 

- Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In _ECCV_ , 2016. 

- Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Revisiting visual representations in vision-language models. In _CVPR_ , 2021a. 

- Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. _arXiv preprint arXiv:2205.01068_ , 2022a. 

- Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, and Wenyu Liu. Fairmot: On the fairness of detection and re-identification in multiple object tracking. _IJCV_ , 2021b. 

- Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. Bytetrack: Multi-object tracking by associating every detection box. In _ECCV_ , 2022b. 

- Zhu Zhang, Zhou Zhao, Yang Zhao, Qi Wang, Huasheng Liu, and Lianli Gao. Where does it exist: Spatio-temporal video grounding for multi-form sentences. In _CVPR_ , 2020. 

- Dora Zhao, Angelina Wang, and Olga Russakovsky. Understanding and evaluating racial biases in image captioning. In _ICCV_ , 2021. 

- Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In _AAAI_ , 2018. 

Xingyi Zhou, Dequan Wang, and Philipp Krähenbühl. Objects as points. _arXiv:1904.07850_ , 2019. 

Xingyi Zhou, Vladlen Koltun, and Philipp Krähenbühl. Tracking objects as points. In _ECCV_ , 2020. 

- Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Krähenbühl, and Ishan Misra. Detecting twenty-thousand classes using image-level supervision. In _ECCV_ , 2022a. 

- Xingyi Zhou, Tianwei Yin, Vladlen Koltun, and Philipp Krähenbühl. Global tracking transformers. In _CVPR_ , 2022b. 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision_images/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision.pdf-0016-00.png)


**----- Start of picture text -----**<br>
A brown dog chasing an adult on<br>the grass<br>An adult woman in white holds a<br>white ball<br>An adult woman in gray holds an<br>yellow bag<br>An adult in blue clothes pulls a<br>dog<br>There is a white goat towards<br>another black goat on the ground<br>There is a dog behind another<br>dog outdoors<br>There is a brown sheep chasing a<br>white goat on the ground<br>**----- End of picture text -----**<br>


Figure 4: **Qualitative results on VidSTG.** Our model captures motion (1st row) and handles crowded scenes (2nd row). However, it may misrecognize objects (2nd row, “dog” should be “goat”) and action boundaries (2nd row, “chasing” before it occurs). 

APPENDICES 

We present further qualitative results (App. A), additional experimental details (App. B), additional experimental analysis (App. C), and broader impact and potential negative impact (App. E). 

## A QUALITATIVE RESULTS 

We show example qualitative visualizations Fig. 4 and discuss typical failure cases. 

## B ADDITIONAL EXPERIMENTAL AND IMPLEMENTATION DETAILS 

## B.1 CODE 

Our code is available at https://github.com/google-research/scenic. 

Our CHOTA evaluation code is in file “code/chota.py”. This evaluation code is based on the official HOTA implementation[*] . The original code is under an MIT license. 

## B.2 FULL TRAINING DETAILS 

As mentioned in Sec. 4.3, our model is based on GRiT Wu et al. (2022a). The original GRiT code[*] is released under an MIT license. Following GRiT, we use a ViTDet-Base Dosovitskiy et al. (2021); Li et al. (2022b) backbone, a CenterNet Zhou et al. (2019) region proposal network and RoI Head, and a randomly-initialized text decoder following that of GIT Wang et al. (2022). The text decoder consists of 6 self-attention layers with casual feature masks Wang et al. (2022). All model architecture parameters follow the defaults from GRiT Wu et al. (2022a). 

The original GRiT uses an MAE pretrained checkpoint, while in our case we found a CLIP pretrained checkpoint Radford et al. (2021) performs better on our task. To fit more frames into memory for both training and evaluation, we use a 384 _×_ 384 input size instead of the original 1024 _×_ 1024. This choice moderately decreases dense image captioning performance on Visual Genome (from 17.3 AP _M_ to 15.7 AP _M_ ). 

During disjoint multi-dataset pretraining, we sample batches from different datasets in an even ratio (1 : 1 : 1 : 1). For image datasets, a batch is composed of different images; for video datasets, we put the time dimension in batches and always guarantee images in the same mini-batch are from the same video. We use a local batch size of either 1 video (consisting of 8 sampled frames), or 8 images. As we use 32 GPUs, this means that our global batch size is either 32 videos or 256 images. We use the AdamW optimizer with a learning rate of 2 _×_ 10 _[−]_[4] , weight decay of 0 _._ 05, and a layerwise learning rate decay of 0 _._ 7 Li et al. (2022b); Wu et al. (2022a). We train for 22 _._ 5 _×_ 10[3] iterations per dataset, decreasing the learning rate by a factor of 10 after 90% and 97 _._ 5% of the training schedule Wu et al. 

> *https://github.com/JonathonLuiten/TrackEval 

> *https://github.com/JialianW/GRiT 

16 

(2022a). For pretraining on all the 4 datasets in Sec. 3.4, this corresponds to a total of 90 _×_ 10[3] iterations, which took approximately 20 hours on 32, 16GB V100 GPUs. 

For VidSTG Zhang et al. (2020) finetuning, we sample 16 frames in training, and run on all 200 frames in testing. For VLN Voigtlaender et al. (2023) finetuning, we use the 3 annotated frames in both training and evaluation. For finetuning experiments on both datasets, we use a video batch size 16 and train for 11 _._ 25 _×_ 10[3] iterations, with a learning rate of 10 _[−]_[5] , weight decay of 0 _._ 05, and layerwise-learning decay of 0 _._ 7 Li et al. (2022b). The finetuning took approximately 6 hours on 16, 16GB GPUs for VidSTG, and about 2 hours on 16, 16GB GPUs for VLN. Inference on VidSTG requires 32GB of GPU memory to fit 200 frames. 

**Training losses.** Training our model involves a detection loss _Lobject_ , a tracking loss _Lassoc_ , and a captioning loss _Lcaption_ , that is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision_images/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision.pdf-0017-03.png)


For completeness, we detail these three terms next: 

The detection loss Zhou et al. (2019) involves a center heatmap loss, a bounding box regression loss, and a classification and bounding box refinement loss in the RoI head: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision_images/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision.pdf-0017-06.png)


The heatmap loss is defined on the predicted heatmap _Y_ ¯ _∈_ R _[H][×][W]_ : _Y ∈_ R _[H][×][W]_ and the ground truth heatmap 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision_images/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision.pdf-0017-08.png)


where _n_ is the number of objects in the image, _α_ = 2 and _β_ = 4 are the focal loss weights Lin et al. (2017). 

_Lreg_ is a gIoU loss Union (2019): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision_images/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision.pdf-0017-11.png)


where _B_ and _B_[¯] are the predicted and the ground truth bounding boxes of the _n_ annotated objects, _Ci_ is the enclosing convex hull of _Bi_ and _B_[¯] _i_ , and _| · |_ computes the area. 

_Lroi_ - _cls_ is a softmax classification loss on each RoI box, defined on the predicted class logits **c** _∈_ R[2] and the ground truth label ¯ _c ∈{_ 0 _,_ 1 _}_ . Here we only have foreground or background classification. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision_images/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision.pdf-0017-14.png)


_Lroi_ - _reg_ is an L1 loss between the predicted boxes _B_ and the ground truth boxes _B_[¯] , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision_images/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision.pdf-0017-16.png)


The tracking loss is a per-element binary cross-entropy loss between the predicted association matrix _A_ and the ground truth binary matrix _A_[¯] : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision_images/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision.pdf-0017-18.png)


The captioning loss is a softmax on each predicted word over the entire vocabulary, with a label smoothing co-efficient of 0 _._ 1 following GIT Wang et al. (2022). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision_images/iclr_2025_Dense_Video_Object_Captioning_from_Disjoint_Supervision.pdf-0017-20.png)


where _y_ ¯ is the ground truth caption, _L_ is the groud-truth sentence length, and _f_ is the object feature. 

17 

|#COCO|#COCO|VG|SMiT|Aug<br>COCO|VidSTG (zero-shot)<br>AP_M_|VLN (zero-shot)<br>AP_M_|VidSTG (fnetuned)<br>AP_M_|VLN (fnetuned)<br>AP_M_|
|---|---|---|---|---|---|---|---|---|
|0|||||-|-|54.1|35.1|
|1|✓||||-|-|69.1|36.3|
|2||✓|||17.1|9.9|68.7|45.9|
|3|||✓||-|-|54.8|38.0|
|4||✓|✓||18.2|12.7|68.9|47.2|
|5|✓|✓|||37.4|19.7|70.8|46.1|
|6|✓||✓||36.7|18.1|69.4|41.3|
|7|✓|✓|✓||38.2|19.0|71.2|**48.2**|
|8|✓|✓|✓|✓|**39.5**|**20.1**|**71.5**|**48.2**|



Table 8: **Zero-shot (left) and finetuning (right) evaluation of our disjoint trained models with varying datasets using image dense captioning metric AP** _M_ **.** We show results on VidSTG Zhang et al. (2020) and VLN Voigtlaender et al. (2023). Each row is a model pretrained on the specified datasets for zero-shot evaluation and then finetuned on the downstream datasets following Tab. 3. The results are consistent with the CHOTA metric: our models trained on joint datasets perform the best. 

|_m_|CHOTA(_↑_)|DetA(_↑_)|AssA(_↑_)|CapA(_↑_)|AP_M_(_↑_)|
|---|---|---|---|---|---|
|1|53.6|64.3|**66.3**|36.1|68.7|
|2|54.1|64.3|66.2|37.3|68.7|
|4|54.4|64.3|65.7|37.8|68.7|
|6|**54.9**|64.2|65.9|**39.1**|**69.1**|
|8|54.6|64.3|66.1|38.4|**69.1**|



Table 9: **Hyper-parameter sweep for number of sampled frames,** _m_ **, for hard-tracking** . We show results on VidSTG Zhang et al. (2020) validation. The models are based on #2 of Tab. 3 right on VidSTG. Results with hard-feature aggregation get improved with more frames and get saturated with 6 frames. 

## C ADDITIONAL EXPERIMENTAL ANALYSIS 

- C.1 AP _M_ EVALUATION. 

mAP-METEOR is the official evaluation metric used in Visual Genome Krishna et al. (2017b) dataset for dense image object captioning. This metric evaluates predictions in each frame separately, without evaluating the tracking output. 

mAP-METEOR is based on the Average Precision used in object detection Lin et al. (2014); Everingham et al. (2015), but includes a caption similarity criteria for determining true positives: _i.e_ . a prediction is a true positive if the Intersection over Union (IoU) with the ground truth bounding box is above a threshold, _and_ if the METEOR score Banerjee & Lavie (2005) is above another threshold. We follow the same implementation and thresholds as the Visual Genome dataset[*] . _i.e_ . IoU thresholds of (0.3, 0.4, 0.5, 0.6, 0.7) and METEOR thresholds of (0.0, 0.05, 0.1, 0.15, 0.2). 

In our case, some objects in the datasets Zhang et al. (2020); Voigtlaender et al. (2023) only have bounding box annotations and no caption annotations (Sec. 4.1). For these objects without caption annotations, we allow any caption prediction (and therefore ignore it) by setting its METEOR score to the maximum of 1. For brevity, we abbreviated this metric as AP _M_ . We report AP _M_ following Tab. 3 in Tab. 8. The improvements are consistent with CHOTA. 

## C.2 ABLATION OF HARD TRACKING SAMPLING. 

We analyze the effect of the number of sampled frames, _m_ , in hard-aggregation (Sec. 3.3) in Tab. 9. With hard-aggregation, the captioning accuracy benefits from a larger number of frames _m_ , thanks to longer input-sequence length. However, this also costs more GPU memory in both training and testing. We use _m_ = 6 in our ablation experiments (Tab. 2) as it achieves the best accuracy. It also follows the default number of frames used in the GIT Wang et al. (2022) video captioning model. 

> *https://github.com/jcjohnson/densecap/blob/master/eval/eval_utils.lua 

18 

|Tracking dataset|CHOTA|DetA|AssA|CapA|AP_M_|
|---|---|---|---|---|---|
|_Zero-shot_||||||
|UVO|24.8|41.2|52.1|7.1|33.8|
|Aug-COCO|**31.1**|**51.4**|**59.6**|**9.8**|**39.5**|
|_Finetuning_||||||
|UVO|50.9|65.2|53.4|37.9|70.4|
|Aug-COCO|**56.9**|**65.8**|**70.4**|**39.7**|**71.5**|



Table 10: **Results using UVO Wang et al. (2021b) as the tracking dataset.** We show both zero-shot results (top) and finetuning results (bottom) on VidSTG datasets. For reference, we also include our results of using Aug-COCO (#8 of Tab. 3). Aug-COCO performs better in both settings, motivating our choice. 

|#COCO VG SMiT Aug-<br>COCO|VidSTG<br>CapA CIDEr METEOR SPICE|VLN<br>CapA CIDEr METEOR SPICE|
|---|---|---|
|1<br>✓<br>2<br>✓<br>3<br>✓|-<br>-<br>-<br>-<br>7.8<br>4.2<br>7.1<br>12.1<br>-<br>-<br>-<br>-|-<br>-<br>-<br>-<br>7.4<br>2.7<br>7.4<br>12.1<br>-<br>-<br>-<br>-|
|4<br>✓<br>✓<br>5<br>✓<br>✓<br>6<br>✓<br>✓<br>7<br>✓<br>✓<br>✓<br>8<br>✓<br>✓<br>✓<br>✓|8.5<br>4.4<br>7.7<br>13.4<br>8.1<br>4.0<br>7.2<br>13.0<br>4.9<br>3.3<br>7.2<br>4.2<br>9.1<br>5.2<br>8.3<br>**13.7**<br>**9.8**<br>**7.0**<br>**9.1**<br>13.3|8.5<br>3.1<br>8.7<br>13.8<br>7.8<br>3.1<br>7.8<br>12.4<br>7.5<br>3.7<br>9.4<br>9.6<br>9.0<br>3.9<br>9.2<br>13.9<br>**9.7**<br>**4.6**<br>**9.9**<br>**14.6**|



Table 11: **Detailed captioning metrics of our** _**zero-shot evaluation**_ **(Tab. 3 left).** We show the individual captioning metrics CIDEr, METEOR, and SPICE for each row on both datasets. 

## C.3 USING THE UVO DATASET FOR DISJOINT PRETRAINING 

For the disjoint pretraining of our model (Sec. 3.4), we used Augmented COCO as our tracking dataset. Another alternative would have been to use UVO Wang et al. (2021b), which contains real-world videos, but is relatively small at only 5000 videos. 

Table 10 compares Aug-COCO and UVO under the setting of Tab. 3 #8, both using a default multidataset sampling ratio 1 : 1 : 1 : 1. We observe that disjoint pretraining with Aug-COCO consistently performs better than UVO in both zero-shot and finetuning scenarios, thus motivating our choice to use Aug-COCO for our experiments in the main paper. 

## C.4 DETAILED CAPTIONING RESULTS 

The Captioning Accuracy (CapA) component of our CHOTA metric is the average of the CIDEr, METEOR and SPICE metrics. For completeness, we report each of these captioning metrics individually in Tabs. 11 and 12, for zero-shot and full-finetuning evaluation, respectively. 

## C.5 VIDSTG SPATIO-TEMPORAL GROUNDING EVALUATION 

Table 7 of the main paper compared to prior methods on the spatial-video grounding task on VidSTG (where the input videos were assumed to already be temporally trimmed). In Tab. 13, we report results for spatial-, temporal- and spatio-temporal grounding by reporting the Spatial IoU (sIoU), Temporal IoU (tIoU) and Video IoU (vIoU) respectively. 

The sIoU assumes the video is already temporally trimmed before evaluation, thus evaluating spatial localization. Similarly, the tIoU assumes that the video is already cropped spatially around the object of interest, and only the temporal extent of the query sentence needs to be determined, thereby evaluating temporal localization. The vIoU evaluates both spatial and temporal localization. 

Our model was designed for the Dense VOC task, and not grounding, and we were able to perform grounding by selecting the bounding boxes with the highest likelihood of generating the target sentence (Sec. 3.5). As shown in Tab. 13, this approach works well for spatial-grounding, outperforming prior works in terms of the sIoU. However, as our model first generates object trajectories, without taking the input query into account, it struggles more at temporal localization. Nevertheless, it still achieves competitive results compared to prior works for both the tIoU and vIoU, although our model 

19 

|#COCO VG SMiT Aug-<br>COCO|VidSTG<br>CapA CIDEr METEOR SPICE|VLN<br>CapA CIDEr METEOR SPICE|
|---|---|---|
|0<br>1<br>✓<br>2<br>✓<br>3<br>✓|35.8<br>43.2<br>29.0<br>35.0<br>34.8<br>41.7<br>27.5<br>35.5<br>39.1<br>49.4<br>30.3<br>37.6<br>31.6<br>33.8<br>27.2<br>33.6|8.7<br>3.9<br>14.8<br>7.3<br>8.2<br>6.7<br>14.0<br>3.9<br>16.7<br>13.9<br>20.1<br>16.1<br>14.5<br>9.6<br>19.6<br>14.2|
|4<br>✓<br>✓<br>5<br>✓<br>✓<br>6<br>✓<br>✓<br>7<br>✓<br>✓<br>✓<br>8<br>✓<br>✓<br>✓<br>✓|39.2<br>49.9<br>30.3<br>37.5<br>38.4<br>48.3<br>30.0<br>36.9<br>38.8<br>49.6<br>29.9<br>37.1<br>**40.1**<br>**51.5**<br>**30.8**<br>38.1<br>39.7<br>51.0<br>30.6<br>**37.7**|**17.8**<br>13.9<br>21.4<br>**17.9**<br>17.4<br>**15.1**<br>20.5<br>16.5<br>11.6<br>8.2<br>17.3<br>9.5<br>17.7<br>14.6<br>**21.5**<br>17.1<br>17.7<br>14.3<br>**21.5**<br>17.5|



Table 12: **Detailed captioning metrics of our** _**finetuning evaluation**_ **(Tab. 3 right).** We show the individual captioning metrics CIDEr, METEOR, and SPICE for each row on both datasets. 

||Validation set<br>sIoU tIoU vIoU|Test set<br> sIoU tIoU vIoU<br>38.0 48.5 19.8<br>47.3<br>-<br>24.0<br> 59.0 48.1 30.4<br>61.7 50.8 33.1|
|---|---|---|
|STGRN Zhang et al. (2020)<br>STVGBert Su et al. (2021)<br>TubeDETR Yang et al. (2022) <br>STCAT Jin et al. (2022)|-<br>-<br>-<br>-<br>-<br>-<br> 56.4 47.2 28.7 <br>-<br>-<br>-||
|Ours (zero-shot)<br>Ours|51.8 40.0 22.0 <br>58.7 41.8 25.9|54.1 40.2 22.5<br> 61.9 41.1 26.3|



Table 13: **State-of-the-art comparison of spatial-temporal grounding on VidSTG.** “-” means the numbers are not reported in the paper. Our model performs competitively at this task, although it was not actually designed for it. As our model generates object trajectories without conditioning on the input query, it struggles at temporal localization, denoted by the tIoU. The spatial localization performance, denoted by the sIoU, outperforms dedicated methods for this task. 

was not designed specifically for this task like the other methods in Tab. 13 which include explicit temporal localization modules within the network. 

## D LIMITATIONS 

Currently, our model produces a single caption for each trajectory, and in future work, we aim to caption potentially multiple action segments within a trajectory. Also, we repurposed existing grounding datasets for our task, as annotating a new captioning dataset can be subjective. We leave annotating a Dense VOC dataset with rigorous protocols and richer captioning as a future work. 

## E BROADER IMPACT AND POTENTIAL NEGATIVE IMPACT 

Our work presents a new task and model for dense video object captioning. This task represents a general technology with a wide range of potential applications. Whilst we are unaware of all potential applications of such models, it is important to be cognizant that each application has its own merits and societal implications depending on the intentions of the individuals building and using the system. For example, we believe that the Dense VOC models can be used as part of systems to improve video search and retrieval, though they could also be used in video surveillance systems too. Additionally, we note that training datasets, especially for captioning Hendricks et al. (2018); Zhao et al. (2021), can contain biases that may render models trained on them unsuitable for deployment. 

20 

