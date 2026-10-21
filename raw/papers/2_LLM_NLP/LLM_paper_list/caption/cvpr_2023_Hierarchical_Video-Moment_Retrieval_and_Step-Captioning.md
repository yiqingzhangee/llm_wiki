# **Hierarchical Video-Moment Retrieval and Step-Captioning** 

Abhay Zala[*][ 1] Jaemin Cho[*][ 1] Satwik Kottur[2] Xilun Chen[2] Barlas Oguz[2] Yashar Mehdad[2] Mohit Bansal[1] UNC Chapel Hill[1] Meta AI[2] 

_{_ jmincho, aszala, mbansal _}_ @cs.unc.edu 

_{_ skottur, xilun, barlaso, mehdad _}_ @fb.com 

https://hirest-cvpr2023.github.io 

## **Abstract** 

_There is growing interest in searching for information from large video corpora. Prior works have studied relevant tasks, such as text-based video retrieval, moment retrieval, video summarization, and video captioning in isolation, without an end-to-end setup that can jointly search from video corpora and generate summaries. Such an endto-end setup would allow for many interesting applications,_ e.g _., a text-based search that finds a relevant video from a video corpus, extracts the most relevant moment from that video, and segments the moment into important steps with captions. To address this, we present the_ HIREST _(_ _**HI** erarchical_ _**RE** trieval and_ _**ST** ep-captioning) dataset and propose a new benchmark that covers hierarchical information retrieval and visual/textual stepwise summarization from an instructional video corpus._ HIREST _consists of 3.4K text-video pairs from an instructional video dataset, where 1.1K videos have annotations of moment spans relevant to text query and breakdown of each moment into key instruction steps with caption and timestamps (totaling 8.6K step captions). Our hierarchical benchmark consists of video retrieval, moment retrieval, and two novel moment segmentation and step captioning tasks. In moment segmentation, models break down a video moment into instruction steps and identify start-end boundaries. In step captioning, models generate a textual summary for each step. We also present starting point task-specific and end-to-end joint baseline models for our new benchmark. While the baseline models show some promising results, there still exists large room for future improvement by the community._ 

## **1. Introduction** 

Encouraged by the easy access to smartphones, recording software, and video hosting platforms, people are increasingly accumulating videos of all kinds. To fuel the 

> *equal contribution 

subsequent growing interest in using machine learning systems to extract and summarize important information from these large video corpora based on text queries, progress has been made in video retrieval [2, 17, 18, 41, 42], moment retrieval [10, 16, 17], video summarization [9, 24, 33, 34], and video captioning [13, 20, 41, 42]. Previous works have generally focused on solving these tasks independently; however, all these tasks share the common goal of retrieving information from a video corpus, at different levels of scales and via different modalities. Hence, in this work, we introduce a new hierarchical benchmark that combines all four tasks to enable novel and useful real-world applications. For example, a text-based search service that finds a relevant video from a large video corpus, extracts the most relevant moment from that video, segments the moment into important steps, and captions them for easy indexing and retrieval. To support this, we introduce HIREST, a hierarchical instructional video dataset for a holistic benchmark of information retrieval from a video corpus (see Sec. 3). HIREST consists of four annotations: 1) 3 _._ 4 _K_ pairs of text query about open-domain instructions ( _e.g_ ., _‘how to make glow in the dark slime’_ ) and videos, 2) relevant moment timestamps inside the 1 _._ 1 _K_ videos, where only a part of the video ( _<_ 75%) is relevant to the text query, 3) moment breakdown in several instructional steps with timestamps (7 _._ 6 steps per video, total 8 _._ 6 _K_ steps), and, 4) an manually curated English caption for each step ( _e.g_ . _‘pour shampoo in container’_ ). We collect fine-grained step-wise annotations of HIREST in a two-step annotation process with online crowdworkers on instructional text-video pairs from the HowTo100M [23] dataset (see Sec. 3.1). The instructional videos often come with clear step-by-step instructions, allowing fine-grained segmentation of the videos into short steps. While there are existing video datasets with step annotations, they are based on a small number of predefined task names [36, 46] (thus step captions are not diverse), or are limited to a single topic ( _e.g_ . cooking [45]). HIREST covers various domains and provides diverse step captions with timestamps written by human annotators (see Table 1), 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning_images/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Query: “How to Make Glow in the Dark Slime”<br>1) Video Retrieval<br>2) Moment Retrieval Retrieved Moment<br>0s 44s 119s 162s<br>Segmented Moment<br>3) Moment Segmentation<br>... ... ... ...<br>4) Step Captioning 48s Pour shampoo in container 63s 67s Break glow stick 77s 90s Pour glow stick in shampoo 104s<br>**----- End of picture text -----**<br>


Figure 1. Overview of four hierarchical tasks of our HIREST dataset (Sec. 3). 1) Video retrieval: find a video that is most relevant to a given text query. 2) Moment retrieval: choose the relevant span of the video, by trimming the parts irrelevant to the text query. 3) Moment segmentation: break down the span into several steps and identify the start-end boundaries of each step. 4) Step captioning: generate step-by-step textual summaries of the moment. 

presenting new challenging and realistic benchmarks for hierarchical video information retrieval. 

Using the HIREST dataset, we benchmark four tasks: 1) video retrieval, 2) moment retrieval, 3) moment segmentation, and 4) step captioning (see Fig. 1 and Sec. 3.3). In the video retrieval task, models have to identify a video that is most relevant to a given text query. In the moment retrieval task, models have to select the relevant span of the video, by trimming the parts irrelevant to the text query (blue boundary in Fig. 1). In the moment segmentation task, models have to break down the relevant portion into several instructional steps and identify the start-end boundaries of each step (green boundaries in Fig. 1). Finally, in the step captioning task, models have to generate step captions ( _e.g_ . _‘spray the warm water on carpet’_ ) of the instructional steps. To provide good starting points to the community for our new task hierarchy, we show the performance of recent baseline models on HIREST. For baselines, we use strong models including CLIP [27], EVA-CLIP [8], Frozenin-Time [2], BMT [13], and SwinBERT [20]. On all four tasks, we find that finetuning models on HIREST improve performance; however, there exists a large room to improve performance. 

We summarize our contributions in this paper: 1) We present HIREST dataset and propose a new benchmark that covers hierarchy in information retrieval and visual/textual summarization from an instructional video corpus. 2) Unlike existing video datasets with step captions based on predefined task names or limited to a single topic, our HIREST provides diverse, high-quality step captions with timestamps written by human annotators. 3) We provide a joint baseline model that can perform moment retrieval, moment segmentation, and step captioning with a single ar- 

chitecture. 4) We provide comprehensive dataset analyses and show experiments with baseline models for each task, where there is a large room to improve model performance. We hope that HIREST can foster future work on end-to-end systems for holistic information retrieval and summarization on large video corpus. In addition, our manually annotated step captions can also be a good source for training and testing the step-by-step reasoning of large multimodal language models [40, 44]. 

## **2. Related Work** 

## **2.1. Text-based Information Retrieval from Video** 

With growing interest in building machine learning systems to search for useful information from large video corpora via text searches, several lines of work have been proposed. In text-to-video retrieval, a system finds the most relevant videos from a list of videos with a given text query [2, 17, 18, 41, 42]. In moment retrieval, a system finds the most relevant moments (usually a few seconds of frame spans), from a single video [10, 16, 17]. In query-focused video summarization, which is a text-conditional version of generic video summarization [9, 34], a system finds the most relevant frames from a video with text query [24, 33]. In video captioning, a system generates a short textual description of a given video [13, 20, 41, 42]. While all these tasks share common goals, information retrieval and summarization from a video corpus, previous works have focused on systems that are specialized in a single task. In this work, we introduce a holistic setup that combines video retrieval, moment retrieval, query-focused video summarization (called moment segmentation), and generating a stepwise textual summary of short clip (called step captioning), 

2 

so that users can search for the most relevant video, the most relevant moment inside the video, and get the stepwise text summarization of the moment. 

## **2.2. Instructional Video Datasets** 

Recently, there have also been several efforts towards creating instructional video datasets [15, 23, 31, 36, 38, 45, 46]. While many of these datasets do a good job of providing high-quality instructional videos, they primarily only target a single domain [15, 31, 38, 45]. There have been recent strong efforts towards developing more diverse instructional datasets [23, 36, 46]. Datasets like HowTo100M [23] provide diverse instructional videos but lack specific stepby-step annotations. Some previous works such as [36, 46] provide step-level annotations for open domain videos, however, are restricted to a set of predefined steps that are reapplied across several videos. Our HIREST dataset provides step annotations on diverse instructional videos, where all step captions are manually written to answer the input text query by human annotators (see Table 1). 

## **3. HIREST: Hierarchical Retrieval and StepCaptioning Dataset** 

We present HIREST, a video dataset consisting of 3.4K text-video pairs, 1.8K moments, and 8.6K step caption annotations. It covers the hierarchy of video/moment retrieval and stepwise captioning from a diverse instructional video corpus. Previous step annotations in video datasets used predefined task descriptions with small vocabulary [36, 46] or limited to a single domain ( _e.g_ . cooking [45]). In contrast, the step captions of HIREST are manually written by human annotators and cover diverse domains with a large vocabulary (see Table 1). We describe the data collection process (Sec. 3.1), dataset analysis (Sec. 3.2), and four hierarchical tasks that stem from our dataset (Sec. 3.3). 

## **3.1. Dataset Collection** 

In the following, we describe the two-stage data collection process. In the appendix, we provide screenshots of the data collection interface for each stage and worker qualification process. 

**Stage 1: Video and Moment Retrieval.** We collect the pairs of text queries and relevant videos from the HowTo100M [23] dataset. Since videos were originally automatically collected from YouTube, we ensure that all videos are actually relevant to the query through human annotation. We employ crowdworkers from Amazon Mechanical Turk[1] and ask them to label whether or not the video correctly answers/solves the associated text query. 

If the video is labeled as relevant to the text query, then we collect relevant ‘moment’ annotation from the video, by 

> 1https://www.mturk.com 

asking the crowdworkers to trim the video to the parts that are directly associated with the text ( _i.e_ . remove video parts unrelated to the text query, such as intro or other topics). We define a video as _clippable_ to a moment, if the moment relevant to the query is less than 75% of the original video length. A system that can retrieve moments from videos would help people directly watch the video portion they are interested in and save time. For the retrieved moments, we collect more fine-grained annotations by dividing the moment into steps and captioning each step. We explain the moment annotation below. 

**Stage 2: Moment Segmentation and Step Captions.** In this stage, we collect fine-grained, stepwise annotations of the retrieved moments. We ask crowdworkers to watch retrieved moments, divide them into several steps and mark the start timestamp of each step. Then, for each of the marked moment segments, they are asked to write a _step caption_ that describes the specific step to complete ( _e.g_ . “add crayons to the candle”, “melt it in bowl with hot water”, “stir it well until dry”). Our text queries from HowTo100M [23] are instructional questions starting with “how to”, and we want the step captions to serve as short textual summaries of moments/steps. we ask crowdworkers to start each caption with an action verb ( _e.g_ . “add”, “apply”) and limit the length of the captions to seven words. 

## **3.2. Dataset Analysis** 

**Task Category Distribution.** Our videos and text queries are collected from the HowTo100M [23] dataset, and hence our category labels match theirs. As shown in Fig. 2, the most frequently occurring categories (for all text-video pairs and just videos with step captions) are “Hobbies and Crafts”, “Food and Entertaining”, and “Home and Garden”. While these are the most common categories (similar to HowTo100M’s most common categories), other categories still have a presence in our dataset. 

**Dataset Statistics.** We collected a total of 3.4K text-video pairs, which are 287 seconds long on average, with a total duration of 270 hours. Out of 3.4K videos, 1.8K videos are _clippable_ to a moment; _i.e_ ., only a short clip ( _<_ 75% of the original video) is relevant to the text query. The average moment length is 148 seconds, which is 55% of the original videos. Out of the 1.8K moments, we provide moment segmentation and step caption annotations for the randomly chosen 1.1K moments. The 1.1K moments are broken down to 7.6 steps on average, totaling 8.6K steps. Each step is annotated with a start-end timestamp and a step caption. The step captions are on average 4.42 words long and have 633 unique starting verbs with 3382 unique words. Fig. 4 shows the most frequent starting verbs and the most frequent words in the step captions (not counting the starting word and stop words). Fig. 3 shows the first three words of 

3 

|Dataset|Domain|Step caption|# Videos / # Steps|# Steps<br>per Moment|# Words<br>per Caption|# Unique<br>Captions|Avg. Duration (s)<br>Video / Step|
|---|---|---|---|---|---|---|---|
|COIN [36]|Open|Predefned steps|11.8K / 46K|3.9|4.8|0.8K|142 / 14.9|
|CrossTask [46]|Open|Predefned steps|4.7K / 21K|7.4|2.4|0.1K|297 / 9.6|
|YouCook2 [45]|Cooking|Manually written|2K / 14K|7.7|8.8|13K|316 / 19.7|
|HIREST (Ours)|Open|Manually written|3.4K (1.1K w/ steps) / 8.6K|7.6|4.4|7.9K|263 / 18.9|



Table 1. Comparison of HIREST and other video datasets with step annotations. While smaller in terms of the total number of videos than other datasets, HIREST covers various open-domain videos with many step annotations per video and high-quality step captions written by human annotators. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning_images/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning.pdf-0004-02.png)


**----- Start of picture text -----**<br>
Home and Garden (10.62%) Cars & Other Vehicles (6.70%) Home and Garden (10.83%) Cars & Other Vehicles (7.31%)<br>Holidays and Traditions (6.16%) Holidays and Traditions (6.43%)<br>Education and Communications (3.68%) Personal Care and Style (5.28%)<br>Personal Care and Style (3.29%) Education and Communications (3.87%)<br>Other (5.76%) Other (4.66%)<br>Food and Entertaining (24.33%) Food and Entertaining (28.52%)<br>Hobbies and Crafts (39.46%) Hobbies and Crafts (33.10%)<br>(a) all videos (b) videos with step caption annotations<br>**----- End of picture text -----**<br>


Figure 2. Task category distribution of HIREST text queries. There are a wide variety of categories for our videos. The most frequent categories are “Hobbies and Crafts”, “Food and Entertaining”, and “Home and Garden”. The task categories are from HowTo100M [23]. 

50 random step caption samples (ignoring stop words). As shown in the visualizations, the manually written step captions of HIREST cover open domain instruction steps and have a diverse vocabulary. 

**Comparisons to Other Datasets with Step Captions.** Table 1 compares our HIREST dataset to other video datasets with step annotations. HIREST covers various opendomain videos with many step annotations per video and high-quality step captions written by human annotators. While COIN [36] and CrossTask [46] also provide steplevel annotations for open-domain videos, however, they are restricted to a set of predefined steps. In contrast, all the step captions of HIREST are manually written to answer the input text query. 

**Data Splits.** Since there are cases where multiple videos are retrieved from the same query, we split our dataset into train/val/test splits by query instead of video. We split our queries into 546/292/546 (1507/477/1391 videos) for train/val/test splits, respectively. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning_images/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning.pdf-0004-07.png)


**----- Start of picture text -----**<br>
nails<br>pan<br>plate<br>add<br>more apply<br>paste paper<br>paint brown sugar<br>this wrap cilantro<br>polybag<br>shake snow<br>up<br>top<br>bag<br>very<br>fake<br>cut tight<br>go<br>line<br>transfer<br>fold<br>tie<br>ice<br>tape locker<br>another cubes<br>comb<br>ink<br>squeeze/shake<br>above<br>soil bomb<br>fart<br>plate<br>keep<br>same<br>remover<br>card<br>palm<br>board<br>oil<br>spray<br>name<br>note<br>pot<br>title<br>half other<br>side<br>take repeat<br>spread<br>from<br>two<br>raw<br>pinch<br>sticks<br>salt<br>remove<br>design<br>pull<br>back<br>place hook<br>turn<br>light<br>two<br>uv<br>mesh<br>pour<br>egg<br>ingredients<br>dump<br>positions<br>one<br>brown<br>pick<br>sleeve<br>flowers<br>fry<br>adjust<br>put<br>again paste<br>smooth piece<br>grain other<br>another<br>cloth<br>open<br>end top<br>dough<br>water part<br>bringcheck<br>white<br>join<br>pvc<br>key overpipe<br>create<br>using<br>line up<br>cook<br>slowly<br>dry<br>drape<br>bowl<br>get<br>rainbow<br>covered ready<br>find fourteen<br>line<br>center desert<br>are<br>**----- End of picture text -----**<br>


Figure 3. Distribution of HIREST step captions by their first three words for 50 random samples. Words are often related to actions or objects. We remove stop words ( _e.g_ . ‘the’, ‘it’, _etc_ .). 

## **3.3. Hierarchical Tasks Enabled by** HIREST 

In the following, we introduce four tasks connected in a hierarchy based on our HIREST dataset. See Fig. 1 for an overview and visual examples of the tasks. 

**Video Retrieval.** This task gives models an instructional text query ( _e.g_ . “How to make a memory jar”), and the models need to determine which videos are relevant and retrieve 

the top results. The models must retrieve videos among 4.2K test split videos (1.4K videos paired with text queries + 2.8K distractor videos from HowTo100M [23]). Distractor videos serve as negative examples (hence ‘distractors’), similar to Revaud _et al_ . [30]. We include these distractors to help increase the difficulty of our video retrieval task. 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning_images/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning.pdf-0005-00.png)


**----- Start of picture text -----**<br>
(a) Most Common Starting Verb Frequency<br>Starting Verb<br>(b) Most Common Word Frequency<br>Word<br>**----- End of picture text -----**<br>


Figure 4. (a) Top 10 most common starting verbs in HIREST step captions. (b) Top 10 most common words in HIREST step captions (excluding the starting words and stop words). The top words typically refer to objects ( _e.g_ . water) or quantities ( _e.g_ . all). 

**Moment Retrieval.** In this task, the goal is to extract the portion of the video that is directly relevant to the given text query (i.e. to remove any unnecessary information from the start/end of the video). 

**Moment Segmentation.** In this task, models should identify all relevant key ‘steps’ from the retrieved relevant moment of the video. Models should generate a list of start and end times for every key step in a given video. 

**Step Captioning.** This task requires models to generate short textual step captions for each retrieved step in a video. Models are provided with the source video and start/end times of each step. They should then generate a short instructional step caption for every step. 

## **4. Experiments** 

For all four HIREST tasks, we conduct experiments with task-specific baseline models (Sec. 4.1), a joint baseline model (Sec. 4.2), and evaluate them with different standard metrics (Sec. 4.3). We represent each video as 32 frames with uniform intervals, if not specified. 

## **4.1. Task-specific Models** 

**Video Retrieval.** We experiment with CLIP (ViTB/32) [27], EVA-CLIP (ViT-G/14) [8], Frozen-in-Time [2], and MIL-NCE (S3D) [22], which are pretrained textto-image (CLIP/EVA-CLIP) and text-to-video (Frozen-inTime/MIL-NCE) retrieval models, respectively. For CLIP and EVA-CLIP, we obtain a video embedding by averaging frame embeddings. We compute the matching score by taking the cosine similarity between video and text query embedding. Following the original setup, we use 4 frames for Frozen-in-Time and 32 frames for MIL-NCE. 

**Moment Retrieval.** We experiment with two CLIPbased heuristics methods and the event proposal module of BMT [13], a dense video captioning model pretrained on ActivityNet Captions [14]. With CLIP, we compute the cosine similarity between all frames and the text query and find the frame with the highest score. Then we determine the start/end boundary of a moment with two different heuristics: 1) picking the frames where the similarity score drops from the highest scoring frame by a certain threshold ( _e.g_ ., 0.10); 2) picking the 8 frames to the left and right, totaling up to 17 (= 8+1+8) frames (see appendix for details). Furthermore, we experiment with the BMT [13] event proposal module, which predicts video event proposals with center/length/confidence values. We allow BMT to generate various events and then take the minimum start time and maximum end time across the events as the retrieved moment. For BMT, we give the model the I3D [5] RGB+Flow features and VGGish [11] audio features of the entire video, extracted at 1fps. 

**Moment Segmentation.** We experiment with 1) framewise difference with the Structural Similarity Index Measure (SSIM) [39], and 2) the event proposal module of BMT [13]. For SSIM, if two adjacent frames have an SSIM below a certain threshold ( _e.g_ ., 0.85), we mark that as a step boundary. For BMT, we feed the model I3D and VGGish features (extracted at 1fps) of the entire video and directly use the video event proposal prediction. 

**Step Captioning.** We experiment with BMT and SwinBERT [20], a pretrained video captioning model. For BMT, we use I3D and VGGish features of each step, extracted at 1fps. We do not use its event proposal module for this task, as we give the features within the ground-truth step boundaries. For SwinBERT, we use YouCook2 [45] checkpoint and 32 video frames from each step as input to the model. 

## **4.2. Joint Model** 

We also experiment with an end-to-end joint baseline model that handles moment retrieval, moment segmentation, and step captioning tasks with a single architecture. As shown in Fig. 5, our model is built on four existing pretrained models: EVA-CLIP [8], Whisper [28], 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning_images/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning.pdf-0006-00.png)


**----- Start of picture text -----**<br>
Joint Embedding Moment Retrieval & Step Captioning<br>Video Eva-CLIP Vis … Moment Segmentation<br>* Start End<br>Query Eva-CLIP Text … Step caption<br>Frame Frame<br>+<br>Audio Whisper & … Text Decoder<br>MiniLM<br>Mask … Multimodal Encoder Multimodal Encoder<br>Frozen Updated … … …<br>**----- End of picture text -----**<br>


Figure 5. Illustration of our joint model that handles moment retrieval, moment segmentation, and step captioning tasks (Sec. 4.2). We learn a shallow multimodal transformer encoder layer that adapts the four pretrained models: EVA-CLIP (frozen), Whisper (frozen), MiniLM (frozen), and CLIP4Caption (finetuned). 

MiniLM [29], and CLIP4Caption [35]. EVA-CLIP visual encoder maps a video frame into a visual embedding, EVACLIP text encoder maps a text query into a text embedding, Whisper extracts speech transcription from audio, MiniLM text encoder maps the speech transcription into a text embedding. To adapt the video, text, and audio embeddings, we finetune a two-layer multimodal encoder and a twolayer text decoder, which are initialized from CLIP4Caption (MSRVTT [41] checkpoint). We train the joint model in a multi-task setup in a round-robin fashion, by sampling a batch from one of the data loaders at each step [6]. 

**Input Embedding.** We construct the multimodal input embedding to the transformer by combining 1) EVA-CLIP video frame embedding, 2) EVA-CLIP text query embedding (tiled to the number of video frames), 3) and MiniLM speech transcription embedding (temporally warped into each frame), and 4) task-specific mask embeddings. For moment retrieval and moment segmentation tasks, we feed the same multimodal embeddings while masking out the frames that are outside of interest. 

**Moment Retrieval & Moment Segmentation.** Following the span-based text question answering models [7, 32], we learn linear layers that predict the boundaries of moments and steps. Concretely, we use three linear layers predicting moment start, moment end, and step boundaries. For the moment retrieval, our joint start and end predictor predicts the moment boundary in parallel, and we do not mask out the video inputs. For the moment segmentation, our joint model autoregressively predicts each step’s boundaries with masking; _i.e_ ., we mask out 1) frames that are outside of the moment and 2) frames that are included in the previous steps. For both tasks, we feed the video in 1fps. 

**Step Captioning.** Following CLIP4Caption [35], we sample 20 frames from each step. The autoregressive text decoder attends to the multimodal encoder output via crossattention and generates each step caption independently. 

|Model|Frames|FT|R@1|R@5|R@10|
|---|---|---|---|---|---|
|CLIP-B/32|1||11.4|20.7|27.3|
|CLIP-B/32|4||12.5|28.8|37.4|
|CLIP-B/32|10||13.0|31.7|39.9|
|CLIP-B/32|20||13.0|33.3|41.2|
|CLIP-B/32|32||12.6|33.0|41.8|
|Frozen-in-Time|4||7.0|19.4|26.7|
|MIL-NCE (S3D)|32||13.9|31.1|41.4|
|CLIP-B/32|1|✓|11.5|22.7|27.1|
|CLIP-B/32|4|✓|13.9|29.5|39.4|
|CLIP-B/32|10|✓|11.4|31.3|41.4|
|CLIP-B/32|20|✓|12.3|31.7|41.6|
|CLIP-B/32|32|✓|13.0|32.1|41.9|
|EVA-CLIP-G/14|1||18.9|32.6|37.5|
|EVA-CLIP-G/14|4||20.7|43.6|53.7|
|EVA-CLIP-G/14|10||26.0|48.5|58.8|
|EVA-CLIP-G/14|20||**26.4**|**51.1**|**61.5**|
|EVA-CLIP-G/14|32||26.0|50.0|61.4|



Table 2. Video retrieval results on HIREST test split. CLIP/EVACLIP results are based on temporal average pooling. _FT: finetuning on_ HIREST _, R@k: Recall@k._ MIL-NCE was trained on the HowTo100M dataset, which is the video source of HIREST. 

## **4.3. Metrics** 

**Video Retrieval.** Following previous work [2, 17–19, 42], We evaluate models on Recall@k metrics: R@1, R@5, and R@10. 

**Moment Retrieval.** Following previous work [16, 17], we evaluate model outputs against the ground-truth (GT) moment spans with Recall@1 with Intersection over Union (IoU) thresholds (0.5 and 0.7). 

**Moment Segmentation.** Following previous work [16,17], we evaluate models on how similar the generated step spans are to the GT spans using IoU. We then compute the recall and precision with IoU thresholds (0.5 and 0.7). 

6 

|Model|FT|R@0.5|R@0.7|
|---|---|---|---|
|CLIP-B/32 (threshold=0.05)||21.01|9.02|
|CLIP-B/32 (8 frames left/right)||34.02|15.72|
|EVA-CLIP-G/14 (threshold=0.10)||19.33|7.86|
|EVA-CLIP-G/14 (8 frames left/right)||38.27|19.33|
|BMT||43.56|10.57|
|BMT|✓|71.91|**39.18**|
|Joint (Ours)|✓|**73.32**|32.60|



Table 3. Moment retrieval results on HIREST test split. CLIP (threshold): determines the start/end frames, by picking the frames where the similarity score drops from the highest scoring frame with a certain threshold ( _e.g_ ., 0.05). CLIP (8 frames left/right): determines the start/end frames by eight frames to the left and to the right of the highest scoring frame. _FT: Finetuning on_ HIREST _, R@IoU: Recall@1 with a threshold of IoU_ . 

|Model<br>FT|Recall@IoU<br>0.5<br>0.7|Precision@IoU|
|---|---|---|
|||0.5<br>0.7|
|SSIM@0.75 (32 frames)<br>SSIM@0.85 (32 frames)<br>BMT (1fps)|12.24<br>5.27<br>25.03<br>9.79<br>8.24<br>3.71|26.32<br>10.05<br>**37.38**<br>**13.80**<br>20.95<br>7.96|
|BMT (1 fps)<br>✓<br>Joint (Ours) (1 fps)<br>✓|34.07<br>12.35<br>**37.50**<br>**14.76**|24.71<br>8.93<br>28.52<br>10.84|



Table 4. Moment segmentation results on HIREST test split. We perform zeroshot evaluation with BMT, and then also provide results of using SSIM. SSIM is given 32 frames. _FT: Finetuning on_ HIREST _, Recall/Precision@IoU: Recall@1/Precision with a threshold of IoU, SSIM@k: SSIM with a score threshold of k._ 

**Step Captioning.** Following previous work [13, 19, 20, 41], we evaluate with the N-gram metrics: CIDEr [37], METEOR [3], and SPICE [1] with the language ~~e~~ valuation package.[2] We also report two sentence-level embeddingbased metrics BERTScore [43] and CLIPScore [12]. 

For BERTScore, we use the RoBERTa-Large [21]. For CLIPScore, we CLIP ViT-B/32 [27] and report the average of frame-caption cosine similarities using 4 frames uniformly sampled from each step. In addition, we compute the entailment of generated sentences to the GT sentences using the ELMo [26]-based Decomposable Attention model [25] pretrained on SNLI [4] with 3 labels: _{_ entailment, contradict, neutral _}_ .[3] We use the ratio of entailment prediction as the entailment score. 

## **5. Results and Discussions** 

In the following, we present the experiment results on the four tasks and the visualization of the pipelined model predictions. Our baseline models show promising initial re- 

> 2https://github.com/bckim92/language-evaluation 

> 3https : / / docs . allennlp . org / models / main / models / pair _ classification / models / decomposable _ attention/ 

|Model|FT|METEOR|CIDEr|SPICE|Entail. (%)|BERT-S|CLIP-S|
|---|---|---|---|---|---|---|---|
|BMT||2.23|1.04|1.41|1.17|0.83|0.21|
|SwinBERT||5.12|13.31|4.65|5.86|0.85|**0.23**|
|BMT|✓|3.84|6.72|1.05|30.68|0.82|0.20|
|SwinBERT|✓|**5.94**|**24.66**|**6.67**|35.09|**0.86**|**0.23**|
|Joint (Ours)|✓|4.13|23.01|3.54|**43.88**|**0.86**|**0.23**|



Table 5. Step captioning results on HIREST test split. We finetune each model on HIREST and evaluate them on our test split. _FT: Finetuning on_ HIREST _, Entail: Entailment, BERT-S: BERTScore, CLIP-S: CLIPScore_ . 

sults, but there exists some gap between the current model performance and the upper bound accuracies, leaving large room for future improvements. 

**Video Retrieval.** Table 2 shows the video retrieval results. Increasing input frames increases the recall until 20 frames. Although CLIP was not trained on a video dataset, CLIP outperforms Frozen-in-Time (4 frames) shows comparable performance with MIL-NCE (32 frames). This is likely due to the fact that CLIP was trained on a much larger dataset than Frozen-in-Time. Finetuning CLIP on HIREST does not show a big difference. EVA-CLIP, a larger CLIP architecture with 1B parameters, outperforms all the other models with a big margin. Thus, we use EVA-CLIP as our video retrieval model and use its features for the three downstream tasks for our joint model. 

**Moment Retrieval.** Table 3 shows the results for moment retrieval. Among the cosine similarity-based zeroshot methods, the 8-frame left/right method outperforms the similarity score drop difference method for both CLIP and EVA-CLIP. BMT achieves better R@0.5 than the zero-shot methods, and the finetuning improves both recall metrics. Our joint model outperforms finetuned BMT on the R@0.5, while finetuned BMT achieves a higher score on R@0.7. 

**Moment Segmentation.** Table 4 shows the results for the moment segmentation task. In the zero-shot setting, BMT fails to adapt the span distribution of HIREST, and simple SSIM methods could outperform the BMT model on both recall and precision. But after finetuning, BMT shows significant improvement over its zero-shot version and SSIM methods on recall metrics. Our joint model achieves a better performance than BMT on both recall and precision. 

**Step Captioning.** Table 5 shows the results of the step captioning task. For both BMT and SwinBERT, zeroshot inference did not result in a good result in N-gram ( _e.g_ ., CIDEr) and entailment metrics, indicating the domain gap between their pretraining datasets (ActivityNet caption and YouCook2) and HIREST is not negligible. For example, their captions are longer than step captions of HIREST. Finetuning brings a performance boost to BMT and SwinBERT in N-gram and entailment metrics but not in sentence-level embedding-based metrics (BERTScore and 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning_images/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning.pdf-0008-00.png)


**----- Start of picture text -----**<br>
Put ingredients ... Blend it mixtureAddthe ... Pour halfand half Mix it ... LayersMake Pat withhands ... Put themin pan ...<br>Ground Truth Moment<br>mix it . mix it cup sugartake one ... transferplateinto mix it ... mix it ... the processrepeat on topput it in plateput it<br>Retrieved Moment<br>0 14 36 63 71 84 87 94 106 117 130 152 159 179 198 208 213 300<br>Time (s)<br>Ground Truth Our Joint Model<br>**----- End of picture text -----**<br>


Figure 6. Comparison of our joint model prediction and ground truth annotation for moment retrieval, moment segmentation, and step captioning. The video is paired with a text query ‘How to make butter biscuits’. 

|Model<br>FT|Moment Retrieval<br>R@0.5<br>R@0.7|Moment Segmentation<br>R@0.7<br>P@0.7|Step Captioning<br>CIDEr|
|---|---|---|---|
||With A|udio||
|BMT<br>✓<br>Joint (Ours)<br>✓|71.9<br>39.2<br>73.3<br>32.6|12.4<br>8.9<br>14.8<br>10.8|6.7<br>23.0|
||Without|Audio||
|BMT<br>✓<br>Joint (Ours)<br>✓|62.6 (-9.3)<br>32.34 (-6.8)<br>70.7 (-2.6)<br>20.6 (-12.0)|10.4 (-2.0)<br>7.4 (-1.6)<br>13.5 (-1.3)<br>10.0 (-0.8)|6.1 (-0.6)<br>15.2 (--7.8)|



117s). The model also captions ‘take one cup sugar’ during that part where ingredients are added (47-55s). The model makes mistakes by missing the end of the dough cutting and the final cooking process (160-213s) during moment retrieval. In this period, we find that a human instructor stands up and describes the process, making the frames visually very different from the previous batter-making process. 

Table 6. Ablation of using audio inputs. Removing audio input drops the performance of all three tasks for both models. 

## **6. Conclusion** 

CLIPScore). Compared to SwinBERT, our joint model achieves similar CIDEr and sentence-level embeddingbased metrics. Notably, our joint model outperforms SwinBERT significantly on the entailment metric. Future work on our dataset can also hopefully explore the complementary strengths of SwinBERT and our joint model. 

**Audio Ablation.** Table 6 shows the ablation study about using (top rows) and not using (bottom rows) audio input with BMT and our joint model. Overall, both models show a performance drop without audio input. For moment retrieval and moment segmentation, removing audio input significantly drops the scores for both models, indicating that audio is very helpful for the tasks that require models to detect the boundaries of events. For the step captioning task, removing audio input significantly drops the score for our joint model, while BMT does not show a big difference. 

**Visualization of Hierarchical Model Pipelining.** In Fig. 6, we visualize the model prediction results and ground-truth annotation for moment retrieval, moment segmentation, and step captioning tasks on a video associated with a query ‘How to make butter biscuits’. The retrieved moment matches with the video moment about making the batter (36-159s) with the ground truth (GT) annotations. The predicted step boundaries and step captions also show semantic correspondence with GT annotations and the video. For example, the predicted caption ‘mix it’ matches the GT captions ‘add the mixture’ (84-87s) and ‘mix it’ (106- 

In this work, we present the HIREST dataset and propose a new benchmark that covers hierarchy in information retrieval and summarization from an instructional video corpus. Our benchmark consists of four tasks: video retrieval, moment retrieval, and our new moment segmentation and step captioning tasks. Different from existing video datasets with step captions, our HIREST provides unique, diverse, high-quality instruction steps with timestamps written by human annotators. We provide comprehensive dataset analysis and present experiments with several task-specific and end-to-end joint baseline models for each task as starting points. We hope that HIREST can foster future work on multimodal systems for holistic video information retrieval, summarization, and step-by-step reasoning. 

## **Acknowledgments** 

We thank the reviewers for their helpful comments. This work was supported by Meta AI, ARO Award W911NF2110220, DARPA KAIROS Grant FA8750-19-21004, and NSF-AI Engage Institute DRL-211263. The views, opinions, and/or findings contained in this article are those of the authors and not of the funding agency. For this work, the collection of data and the subsequent experiments were performed by the University of North Carolina at Chapel Hill, and not by Meta AI. As a result, both the data and code will be released by UNC Chapel Hill. 

8 

## **References** 

- [1] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In _ECCV_ , 2016. 7 

- [2] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In _IEEE International Conference on Computer Vision_ , 2021. 1, 2, 5, 6 

- [3] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In _IEEvaluation@ACL_ , 2005. 7 

- [4] Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. A large annotated corpus for learning natural language inference. In _EMNLP_ , 2015. 7 

- [5] Jo˜ao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. _2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 4724–4733, 2017. 5 

- [6] Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal. Unifying Vision-and-Language Tasks via Text Generation. In _ICML_ , feb 2021. 6 

- [7] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In _NAACL_ , oct 2019. 6 

- [8] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. In _CVPR_ , 2023. 2, 5, 12 

- [9] Michael Gygli, Helmut Grabner, Hayko Riemenschneider, and Luc Van Gool. Creating summaries from user videos. In _ECCV_ , 2014. 1, 2 

- [10] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan C. Russell. Localizing moments in video with natural language. In _ICCV_ , pages 5804– 5813, 2017. 1, 2 

- [11] Shawn Hershey, Sourish Chaudhuri, Daniel P. W. Ellis, Jort F. Gemmeke, Aren Jansen, R. Channing Moore, Manoj Plakal, Devin Platt, Rif A. Saurous, Bryan Seybold, Malcolm Slaney, Ron J. Weiss, and Kevin W. Wilson. Cnn architectures for large-scale audio classification. _2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 131–135, 2017. 5 

- [12] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: A Reference-free Evaluation Metric for Image Captioning. In _EMNLP_ , 2021. 7 

- [13] Vladimir Iashin and Esa Rahtu. A better use of audio-visual cues: Dense video captioning with bi-modal transformer. In _British Machine Vision Conference (BMVC)_ , 2020. 1, 2, 5, 7, 12, 13 

- [14] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-Captioning Events in Videos. In _ICCV_ , 2017. 5 

- [15] Hilde Kuehne, Ali Bilgin Arslan, and Thomas Serre. The language of actions: Recovering the syntax and semantics of goal-directed human activities. _2014 IEEE Conference on_ 

_Computer Vision and Pattern Recognition_ , pages 780–787, 2014. 3 

- [16] Jie Lei, Tamara L. Berg, and Mohit Bansal. Qvhighlights: Detecting moments and highlights in videos via natural language queries. In _NeurIPS_ , 2021. 1, 2, 6 

- [17] Jie Lei, Licheng Yu, Tamara L Berg, and Mohit Bansal. Tvr: A large-scale dataset for video-subtitle moment retrieval. In _ECCV_ , 2020. 1, 2, 6 

- [18] Linjie Li, Yen-Chun Chen, Zhe Gan Yu Cheng, Licheng Yu, and Jingjing Liu. HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training. In _EMNLP_ , 2020. 1, 2, 6 

- [19] Linjie Li, Jie Lei, Zhe Gan, Licheng Yu, Yen-Chun Chen, Rohit Pillai, Yu Cheng, Luowei Zhou, Xin Eric Wang, William Yang Wang, Tamara Lee Berg, Mohit Bansal, Jingjing Liu, Lijuan Wang, and Zicheng Liu. VALUE: A Multi-Task Benchmark for Video-and-Language Understanding Evaluation. In _NeurIPS_ , pages 1–21, 2021. 6, 7 

- [20] Kevin Lin, Linjie Li, Chung-Ching Lin, Faisal Ahmed, Zhe Gan, Zicheng Liu, Yumao Lu, and Lijuan Wang. Swinbert: End-to-end transformers with sparse attention for video captioning. In _CVPR_ , 2022. 1, 2, 5, 7, 13 

- [21] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A Robustly Optimized BERT Pretraining Approach, 2019. 7 

- [22] Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman. End-to-end learning of visual representations from uncurated instructional videos. In _CVPR_ , 2020. 5 

- [23] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips. In _ICCV_ , 2019. 1, 3, 4, 10 

- [24] Medhini Narasimhan, Arsha Nagrani, Chen Sun, Michael Rubinstein, Trevor Darrell, Anna Rohrbach, and Cordelia Schmid. Tl;dw? summarizing instructional videos with task relevance & cross-modal saliency. In _ECCV_ , volume abs/2208.06773, 2022. 1, 2 

- [25] Ankur P. Parikh, Oscar T¨ackstr¨om, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model for natural language inference. In _EMNLP_ , 2016. 7 

- [26] Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. In _NAACL_ , 2018. 7 

- [27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. _ArXiv_ , abs/2103.00020, 2021. 2, 5, 7 

- [28] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision, 2022. 5 

- [29] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In _Proceedings of_ 

9 

_the 2019 Conference on Empirical Methods in Natural Language Processing_ . Association for Computational Linguistics, 11 2019. 6 

- [30] J´erˆome Revaud, Matthijs Douze, Cordelia Schmid, and Herv´e J´egou. Event retrieval in large video collections with circulant temporal encoding. In _CVPR_ , 2013. 4 

- [31] Marcus Rohrbach, Sikandar Amin, Mykhaylo Andriluka, and Bernt Schiele. A database for fine grained activity detection of cooking activities. In _2012 IEEE Conference on Computer Vision and Pattern Recognition_ , pages 1194– 1201, 2012. 3 

- [32] Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, and Hananneh Hajishirzi. Bi-Directional Attention Flow for Machine Comprehension. In _ICLR_ , pages 1–12, 2017. 6 

- [33] Aidean Sharghi, Jacob S. Laurel, and Boqing Gong. Queryfocused video summarization: Dataset, evaluation, and a memory network based approach. In _CVPR_ , pages 2127– 2136, 2017. 1, 2 

- [34] Yale Song, Jordi Vallmitjana, Amanda Stent, and Alejandro Jaimes. Tvsum: Summarizing web videos using titles. In _CVPR_ , pages 5179–5187, 2015. 1, 2 

- [35] Mingkang Tang, Zhanyu Wang, Zhenhua Liu, Fengyun Rao, DIan Li, and Xiu Li. CLIP4Caption: CLIP for Video Caption. In _ACM MM_ , 2021. 6 

- [36] Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. Coin: A large-scale dataset for comprehensive instructional video analysis. In _IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2019. 1, 3, 4 

- [37] Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. _2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 4566–4575, 2015. 7 

- [38] Weiying Wang, Yongcheng Wang, Shizhe Chen, and Qin Jin. YouMakeup: A large-scale domain-specific multimodal dataset for fine-grained semantic comprehension. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLPIJCNLP)_ , pages 5133–5143, Hong Kong, China, Nov. 2019. Association for Computational Linguistics. 3 

- [39] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. _IEEE Transactions on Image Processing_ , 13(4):600–612, 2004. 5 

- [40] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2022. 2 

- [41] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. MSR-VTT: A Large Video Description Dataset for Bridging Video and Language. In _CVPR_ , 2016. 1, 2, 6, 7 

- [42] Youngjae Yu, Hyungjin Ko, Jongwook Choi, and Gunhee Kim. End-to-end concept word detection for video captioning, retrieval, and question answering. In _CVPR_ , volume 2017-Janua, pages 3261–3269, 2017. 1, 2, 6 

- [43] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. BERTScore: Evaluating Text Generation with BERT. In _ICLR_ , 2019. 7 

- [44] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models, 2023. 2 

- [45] Luowei Zhou, Chenliang Xu, and Jason J. Corso. Towards automatic learning of procedures from web instructional videos. In _AAAI_ , 2018. 1, 3, 4, 5, 13 

- [46] Dimitri Zhukov, Jean-Baptiste Alayrac, Ramazan Gokberk Cinbis, David Fouhey, Ivan Laptev, and Josef Sivic. Crosstask weakly supervised learning from instructional videos. In _2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 3532–3540, 2019. 1, 3, 4 

In the appendix, we include the following content: Data annotation details (Appendix A), CLIP-based moment retrieval method visualization (Appendix B), Model performance analysis in video categories and duration groups (Appendix C), and Evaluation details (Appendix D). 

## **A. Data Annotation Details** 

**Annotation Interface.** In the following, we provide screenshots of the HIREST annotation interface for each stage (Fig. 7 and Fig. 8) and worker qualification process. 

**Worker Qualifications and Pay.** We require crowdworkers to have above a 95% approval rate and have completed at least 1000 or more other tasks before working on ours. We also require that all workers pass a qualification test (separate tests for each stage) before they can work on our tasks. For stage 1, the qualification test is composed of 2 parts. First workers are asked to determine if a video solves/answers the given prompt, then they are shown a relevant video and asked to identify the relevant moment in the video (we provide some leniency with timing). For stage 2, workers are given a short series of videos with multiplechoice questions. The multiple choice answers consist of pre-written step captions for the video. The workers were asked then to identify which set of step captions was the best ( _i.e_ . best covered the video and didn’t violate simple rules). A total of 72 workers passed the qualification test for both tasks. As text queries from the HowTo100M [23] dataset are all in English and all of our collected step captions are in English, we require crowdworkers to be from an English-speaking country. Workers were paid $0.20 for stage 1 and $0.45 for stage 2. We also provide a large bonus for good workers. For stage 1, workers are bonused with $0.05 if the video answers/solves the prompt and if they correctly trim the video. Then for every 25 tasks they complete, their base pay increases by $0.02. A typical worker can earn up to $0.27 per task, which is roughly $12.00 per hour. For stage 2, workers are paid with $0.04 bonus for every high-quality step caption they write complete and for 

10 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning_images/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning.pdf-0011-00.png)


Figure 7. Stage 1 data collection interface for video and moment retrieval. Crowdworkers are presented with a video and text query and asked if the video answers/solves the question. If they select yes, the interface expands to a sliding bar that allows them to trim the video down to just the relevant portion. 

every 10 high-quality tasks they complete, their base pay is increased by $0.02. A typical worker can earn up to $0.81 per task, which is roughly $12.00 per hour. For both tasks, there is a baseline pay of around $12, but oftentimes, workers would complete more than 25 and 10 tasks (respectively, for stages 1 and 2), pushing the pay/hour higher. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning_images/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning.pdf-0011-03.png)


Figure 8. Stage 2 data collection interface moment segmentation and step captions. Crowdworkers are presented with a video and instructional text query and are asked to write all the essential steps in the video along with the timestamp of each step. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning_images/cvpr_2023_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning.pdf-0011-05.png)


**----- Start of picture text -----**<br>
0.10 Threshold<br>Bounds<br>8 Frame Bounds<br>GT Bounds<br>**----- End of picture text -----**<br>


## **B. CLIP-based Moment Retrieval Method** 

In Fig. 9, we illustrate two heuristics that we discuss in the main paper Sec. 4.1. From the frame that scores the highest text-frame cosine similarity, we determine the start/end timestamp of the moment by 1) picking the frames where the similarity score drops from the highest scoring frame by a certain threshold ( _e.g_ ., 0.10); 2) picking the 8 frames to the left and right, totaling up to 17 (= 8+1+8) frames. 

## **C. Model Performance Analysis in Video Categories and Duration** 

As mentioned in the main paper Sec. 3, HIREST videos are collected by text queries with different categories such 

Figure 9. Visualization of image-text cosine similarity-based methods for the moment retrieval task. The 8-frame method (red with dashes and dots) achieves IoU=0.42 while the 0.10 threshold method (blue with dashes) achieves IoU=0.19. Ground Truth bounds are indicated with the solid green lines. EVA-CLIP model was used for the plot. _CLIP Score: cosine similarity between image and text embedding_ . 

as ‘Home and Garden’ and ‘Food and Entertaining’. Also, videos and moments have different durations. In Table 7, we show the distribution of prompts and videos for each category in HIREST test split. In the following, we provide comprehensive evaluation results per category and different video/moment duration groups. 

11 

|Category|# Prompts|# Videos|
|---|---|---|
|Hobbies and Crafts|193|231|
|Food and Entertaining|192|250|
|Home and Garden|69|111|
|Cars and Other Vehicles|28|55|
|Holidays and Traditions|25|47|
|Education and Communications|15|23|
|Personal Care and Style|6|29|
|Pets and Animals|5|6|
|Health|5|13|
|Family Life|4|1|
|Arts and Entertainment|1|1|
|Sports and Fitness|1|8|
|Misc.|2|1|
|All|546|776|



Table 7. Prompt and Video category distributions of HIREST test split. Categories are sorted in descending order by the number of prompts. The number of prompts is smaller than the number of videos since multiple videos were retrieved and paired with some prompts. 

|Category|Model|FT|R@1|R@5|R@10|
|---|---|---|---|---|---|
|Hobbies and Crafts|EVA-CLIP||26.42|52.85|63.73|
|Food and Entertaining|EVA-CLIP||25.52|43.75|53.12|
|Home and Garden|EVA-CLIP||27.54|62.32|71.01|
|Cars and Other Vehicles|EVA-CLIP||14.29|53.57|64.29|
|Holidays and Traditions|EVA-CLIP||44.0|68.0|76.0|
|Education and Communications|EVA-CLIP||20.0|26.67|46.67|
|Personal Care and Style|EVA-CLIP||50.0|66.67|66.67|
|Pets and Animals|EVA-CLIP||0|60.0|80.0|
|Health|EVA-CLIP||60.0|60.0|80.0|
|Family Life|EVA-CLIP||0|75.0|100.0|
|Arts and Entertainment|EVA-CLIP||0|0|0|
|Sports and Fitness|EVA-CLIP||100.0|100.0|100.0|
|All|EVA-CLIP||26.37|51.1|61.54|



Table 8. Video retrieval results per prompt category on our HIREST test split. _FT: finetuning on_ HIREST _, R@k: Recall@k._ 

**Video retrieval.** In Table 8, we show EVA-CLIP [8] (ViTG/14) with 20 frames on each prompt category in our dataset. Among the categories that have many most videos ( _>_ 20 videos), the model is better at ‘Holidays and Traditions’ and ‘Personal Care and Style’ than ‘Cars and Other Vehicles’. 

**Moment retrieval.** In Table 9, we show the zeroshot and finetuning results of BMT [13] proposal module and our joint model on each HIREST video category. Categories like ‘Home and Garden’ and ‘Holidays and Traditions’ see a strong performance increase after finetuning. 

In Table 10, we show Moment retrieval performance on three video duration groups. Before finetuning, BMT performs slightly better on videos of a longer length; however, after finetuning, BMT performs much better on shorterlength videos. In R@0.5, our joint model outperforms BMT 

|Category|Model|FT|R@0.5|R@0.7|
|---|---|---|---|---|
|Hobbies and Crafts|BMT||50.65|11.26|
|Food and Entertaining|BMT||34.40|8.80|
|Home and Garden|BMT||39.64|6.31|
|Cars and Other Vehicles|BMT||60.00|14.55|
|Holidays and Traditions|BMT||36.17|6.38|
|Education and Communications|BMT||47.83|26.09|
|Personal Care and Style|BMT||44.83|10.34|
|Pets and Animals|BMT||33.33|33.33|
|Health|BMT||61.54|30.77|
|Family Life|BMT||0|0|
|Arts and Entertainment|BMT||100|0|
|Sports and Fitness|BMT||62.5|12.5|
|All|BMT||43.56|10.57|
|Hobbies and Crafts|BMT|✓|72.29|39.39|
|Food and Entertaining|BMT|✓|72.80|38.00|
|Home and Garden|BMT|✓|67.57|36.04|
|Cars and Other Vehicles|BMT|✓|74.55|52.72|
|Holidays and Traditions|BMT|✓|72.34|31.91|
|Education and Communications|BMT|✓|60.87|39.13|
|Personal Care and Style|BMT|✓|72.41|34.38|
|Pets and Animals|BMT|✓|66.67|16.67|
|Health|BMT|✓|84.62|69.23|
|Family Life|BMT|✓|100|100|
|Arts and Entertainment|BMT|✓|100|100|
|Sports and Fitness|BMT|✓|87.5|37.5|
|Hobbies and Crafts|Joint (Ours)|✓|75.76|35.5|
|Food and Entertaining|Joint (Ours)|✓|75.2|36.4|
|Home and Garden|Joint (Ours)|✓|63.06|21.62|
|Cars and Other Vehicles|Joint (Ours)|✓|81.82|34.55|
|Holidays and Traditions|Joint (Ours)|✓|72.34|31.91|
|Education and Communications|Joint (Ours)|✓|78.26|26.09|
|Personal Care and Style|Joint (Ours)|✓|68.97|17.24|
|Pets and Animals|Joint (Ours)|✓|33.33|33.33|
|Health|Joint (Ours)|✓|61.54|30.77|
|Family Life|Joint (Ours)|✓|100.0|100.0|
|Arts and Entertainment|Joint (Ours)|✓|100.0|0.0|
|Sports and Fitness|Joint (Ours)|✓|75.0|50.0|
|All|BMT|✓|71.91|39.18|
|All|Joint (Ours)|✓|73.32|32.60|



Table 9. Moment retrieval results per video category on our HIREST test split. _FT: Finetuning on_ HIREST _, R@IoU: Recall@1 with a threshold of IoU_ . 

when the videos are longer. 

**Moment segmentation.** In Table 11, we show the zeroshot and finetuned results of BMT [13] proposal module and our joint model on each individual category in our dataset. Finetuning BMT results show significant improvement in every category. 

In Table 12, we show the moment segmentation performance on three moment duration groups. All models achieve higher performance in shorter moments than in longer moments. Our joint model shows better performance than BMT on shorter moments, while BMT does better on longer moments. 

12 

|Video Duration|Model|FT|R@0.5|R@0.7|
|---|---|---|---|---|
|_<_2 mins|BMT||48.70|10.43|
|2 - 6 mins|BMT||37.47|7.04|
|_>_6 mins|BMT||56.74|20.22|
|All|BMT||43.56|10.57|
|_<_2 mins|BMT|✓|74.78|44.35|
|2 - 6 mins|BMT|✓|72.05|40.37|
|_>_6 mins|BMT|✓|69.66|32.58|
|_<_2 mins|Joint (Ours)|✓|68.10|18.10|
|2 - 6 mins|Joint (Ours)|✓|73.21|28.21|
|_>_6 mins|Joint (Ours)|✓|75.00|40.26|
|All|BMT|✓|71.91|39.18|
|All|Joint (Ours)|✓|73.32|32.60|



Table 10. Moment retrieval results for various durations on our HIREST test split. _FT: Finetuning on_ HIREST _, R@IoU: Recall@1 with a threshold of IoU_ . 

|Category<br>Model<br>FT|Recall@IoU<br>0.5<br>0.7|Precision@IoU|
|---|---|---|
|||0.5<br>0.7|
|Hobbies and Crafts<br>BMT<br>Food and Entertaining<br>BMT<br>Home and Garden<br>BMT<br>Cars and Other Vehicles<br>BMT<br>Holidays and Traditions<br>BMT<br>Education and Communications<br>BMT<br>Personal Care and Style<br>BMT<br>Pets and Animals<br>BMT<br>Health<br>BMT<br>Family Life<br>BMT<br>Arts and Entertainment<br>BMT<br>Sports and Fitness<br>BMT|8.91<br>3.02<br>7.47<br>4.47<br>8.64<br>3.18<br>4.18<br>1.11<br>7.24<br>4.67<br>6.11<br>0<br>9.87<br>6.75<br>40.00<br>10.00<br>13.33<br>10.00<br>0<br>0<br>0<br>0<br>0<br>0|22.44<br>6.17<br>19.06<br>9.36<br>23.33<br>7.62<br>16.16<br>5.05<br>16.11<br>10.50<br>20.00<br>0<br>19.65<br>11.58<br>80.00<br>20.00<br>30.00<br>20.00<br>0<br>0<br>0<br>0<br>0<br>0|
|All<br>BMT|8.24<br>3.71|20.95<br>7.96|
|Hobbies and Crafts<br>BMT<br>✓<br>Food and Entertaining<br>BMT<br>✓<br>Home and Garden<br>BMT<br>✓<br>Cars and Other Vehicles<br>BMT<br>✓<br>Holidays and Traditions<br>BMT<br>✓<br>Education and Communications<br>BMT<br>✓<br>Personal Care and Style<br>BMT<br>✓<br>Pets and Animals<br>BMT<br>✓<br>Health<br>BMT<br>✓<br>Family Life<br>BMT<br>✓<br>Arts and Entertainment<br>BMT<br>✓<br>Sports and Fitness<br>BMT<br>✓|33.09<br>10.13<br>32.25<br>12.02<br>38.21<br>13.89<br>33.09<br>14.79<br>34.57<br>12.49<br>41.47<br>13.00<br>29.79<br>11.47<br>50.00<br>27.50<br>42.52<br>22.30<br>35.71<br>14.29<br>22.22<br>11.11<br>33.62<br>20.13|25.35<br>7.84<br>21.86<br>7.66<br>29.04<br>11.86<br>22.03<br>10.60<br>26.72<br>10.00<br>23.95<br>7.07<br>22.25<br>7.32<br>42.04<br>16.86<br>31.11<br>18.34<br>38.46<br>15.38<br>11.76<br>5.88<br>29.05<br>11.59|
|Hobbies and Crafts<br>Joint (Ours)<br>✓<br>Food and Entertaining<br>Joint (Ours)<br>✓<br>Home and Garden<br>Joint (Ours)<br>✓<br>Cars and Other Vehicles<br>Joint (Ours)<br>✓<br>Holidays and Traditions<br>Joint (Ours)<br>✓<br>Education and Communications<br>Joint (Ours)<br>✓<br>Personal Care and Style<br>Joint (Ours)<br>✓<br>Pets and Animals<br>Joint (Ours)<br>✓<br>Health<br>Joint (Ours)<br>✓<br>Family Life<br>Joint (Ours)<br>✓<br>Arts and Entertainment<br>Joint (Ours)<br>✓<br>Sports and Fitness<br>Joint (Ours)<br>✓|38.11<br>13.63<br>35.43<br>13.56<br>35.28<br>17.63<br>39.68<br>15.23<br>34.52<br>10.92<br>49.71<br>24.86<br>41.38<br>17.57<br>70.0<br>30.0<br>40.58<br>11.48<br>50.0<br>28.57<br>22.22<br>11.11<br>31.71<br>16.85|26.47<br>9.34<br>28.54<br>10.4<br>29.69<br>14.46<br>31.16<br>12.39<br>25.42<br>6.95<br>31.61<br>13.94<br>29.91<br>13.34<br>40.76<br>17.33<br>37.26<br>7.42<br>63.64<br>36.36<br>13.33<br>6.67<br>24.84<br>9.5|
|All<br>BMT<br>✓<br>All<br>Joint (Ours)<br>✓|34.06<br>12.34<br>37.50<br>14.76|24.71<br>8.93<br>28.52<br>10.84|



|Moment<br>Duration<br>Model<br>FT|Recall@IoU<br>0.5<br>0.7|Precision@IoU|
|---|---|---|
|||0.5<br>0.7|
|_<_1.5 mins<br>BMT<br>1.5 - 3 mins<br>BMT<br>_>_3 mins<br>BMT|11.75<br>4.76<br>6.59<br>3.31<br>6.40<br>3.04|26.92<br>9.19<br>17.13<br>7.18<br>19.28<br>7.59|
|All<br>BMT|8.24<br>3.71|20.95<br>7.96|
|_<_1.5 mins<br>BMT<br>✓<br>1.5 - 3 mins<br>BMT<br>✓<br>_>_3 mins<br>BMT<br>✓|38.08<br>14.27<br>33.92<br>13.26<br>29.54<br>8.82|27.75<br>10.87<br>23.20<br>8.74<br>23.23<br>6.90|
|_<_1.5 mins<br>Joint (Ours)<br>✓<br>1.5 - 3 mins<br>Joint (Ours)<br>✓<br>_>_3 mins<br>Joint (Ours)<br>✓|44.32<br>17.81<br>38.04<br>14.70<br>28.72<br>11.27|42.22<br>16.39<br>25.41<br>9.68<br>16.75<br>5.93|
|All<br>BMT<br>✓<br>All<br>Joint (Ours)<br>✓|34.06<br>12.34<br>37.50<br>14.76|24.71<br>8.93<br>28.52<br>10.84|



Table 12. Moment segmentation results for different moment duration groups on our HIREST test split. _FT: Finetuning on_ HIREST _, Recall@IoU: Recall@1 with a threshold of IoU, Precision@IoU: Precision@1 with a threshold of IoU_ . 

on each video category in our dataset. Notably, SwinBERT performs best in the ‘Food and Entertaining’ category, likely because SwinBERT was pretrained on the YouCook2 [45] dataset. 

In Table 14, we show the step captioning performance in different step durations. Both N-gram ( _e.g_ . CIDEr) and sentence-level embedding metrics (BERTScore and CLIPScore) do not show significant differences among different categories. In the entailment metric, finetuned SwinBERT gets better as the steps get longer, while our joint model gets slightly worse for longer steps. 

## **D. Evaluation Details** 

We continue the evaluation details of moment segmentation task in the main paper Sec. 4.3 (Metrics). The BMT [13] model generates up to 100 possible step segments, where many of them are outside of the ground-truth (GT) moment input, overlap each other, and there are also gaps between segments. For evaluation of moment segmentation task, we first remove any segments outside the given ground truth moment and use non-maximum suppression (NMS) to remove any overlapping segments. Then any resulting gaps between steps are also marked as separate steps. 

Table 11. Moment segmentation results per video category on our HIREST test split. _FT: Finetuning on_ HIREST _, Recall@IoU: Recall@1 with a threshold of IoU, Precision@IoU: Precision@1 with a threshold of IoU_ . 

**Step captioning.** In Table 13, we show the zeroshot and finetuned results of SwinBERT [20] and our joint model 

13 

|Category|Model|FT|METEOR|CIDEr|SPICE|Entailment (%)|BERTScore|CLIPScore|
|---|---|---|---|---|---|---|---|---|
|Hobbies and Crafts|SwinBERT||3.18|5.45|1.37|2.11|0.84|0.22|
|Food and Entertaining|SwinBERT||8.15|24.23|9.74|11.62|0.86|0.25|
|Home and Garden|SwinBERT||3.23|6.35|1.34|1.49|0.84|0.21|
|Cars and Other Vehicles|SwinBERT||3.12|4.51|0.17|2.07|0.84|0.20|
|Holidays and Traditions|SwinBERT||4.84|10.55|3.26|2.28|0.84|0.22|
|Education and Communications|SwinBERT||3.16|7.96|2.58|3.41|0.83|0.24|
|Personal Care and Style|SwinBERT||4.48|16.05|4.83|4.69|0.84|0.22|
|Pets and Animals|SwinBERT||2.62|7.17|0|6.25|0.83|0.21|
|Health|SwinBERT||2.68|6.75|0.33|0|0.83|0.19|
|Family Life|SwinBERT||2.46|9.78|0|14.29|0.83|0.20|
|Arts and Entertainment|SwinBERT||1.22|8.09|0|0|0.84|0.20|
|Sports and Fitness|SwinBERT||1.87|3.59|0|6.82|0.84|0.23|
|All|SwinBERT||5.12|13.31|4.65|5.86|0.85|0.23|
|Hobbies and Crafts|SwinBERT|✓|4.54|13.82|4.88|38.95|0.86|0.23|
|Food and Entertaining|SwinBERT|✓|8.08|37.64|9.82|32.60|0.87|0.24|
|Home and Garden|SwinBERT|✓|4.84|18.04|5.26|41.04|0.86|0.22|
|Cars and Other Vehicles|SwinBERT|✓|5.49|21.59|6.64|27.80|0.87|0.22|
|Holidays and Traditions|SwinBERT|✓|5.52|20.56|4.29|32.42|0.86|0.23|
|Education and Communications|SwinBERT|✓|3.45|10.51|1.33|23.86|0.85|0.24|
|Personal Care and Style|SwinBERT|✓|5.86|25.79|6.25|39.84|0.86|0.22|
|Pets and Animals|SwinBERT|✓|4.59|18.74|0|21.25|0.86|0.22|
|Health|SwinBERT|✓|2.27|5.06|0|10.91|0.85|0.20|
|Family Life|SwinBERT|✓|4.96|9.52|3.57|42.86|0.85|0.22|
|Arts and Entertainment|SwinBERT|✓|2.90|13.16|0|11.11|0.85|0.21|
|Sports and Fitness|SwinBERT|✓|2.65|12.78|0.91|54.55|0.86|0.24|
|Hobbies and Crafts|Joint (Ours)|✓|3.98|18.26|3.61|35.31|0.86|0.23|
|Food and Entertaining|Joint (Ours)|✓|4.22|30.35|2.63|57.67|0.86|0.23|
|Home and Garden|Joint (Ours)|✓|4.24|14.88|4.43|34.33|0.86|0.22|
|Cars and Other Vehicles|Joint (Ours)|✓|5.41|22.20|6.65|28.33|0.87|0.23|
|Holidays and Traditions|Joint (Ours)|✓|4.40|22.83|3.48|38.81|0.85|0.22|
|Education and Communications|Joint (Ours)|✓|4.01|19.37|4.17|27.27|0.85|0.23|
|Personal Care and Style|Joint (Ours)|✓|3.37|18.09|4.74|57.03|0.85|0.23|
|Pets and Animals|Joint (Ours)|✓|3.55|13.99|3.12|12.50|0.85|0.23|
|Health|Joint (Ours)|✓|2.33|11.37|2.42|30.91|0.85|0.19|
|Family Life|Joint (Ours)|✓|3.68|3.41|7.14|35.71|0.84|0.22|
|Arts and Entertainment|Joint (Ours)|✓|1.67|0|10.00|11.11|0.84|0.21|
|Sports and Fitness|Joint (Ours)|✓|2.54|17.79|2.65|40.91|0.86|0.23|
|All|SwinBERT|✓|5.94|24.66|6.67|35.09|0.86|0.23|
|All|Joint (Ours)|✓|4.13|23.01|3.54|43.88|0.86|0.23|



Table 13. Step captioning results per video category on our HIREST test split. _FT: Finetuning on_ HIREST. 

14 

|Step Duration|Model|FT|METEOR|CIDEr|SPICE|Entailment (%)|BERTScore|CLIPScore|
|---|---|---|---|---|---|---|---|---|
|_<_8 secs|SwinBERT||5.73|16.72|5.40|6.74|0.84|0.23|
|8 - 18 secs|SwinBERT||5.18|13.95|4.95|6.05|0.85|0.23|
|_>_18 secs|SwinBERT||4.57|10.66|3.66|4.96|0.84|0.23|
|All|SwinBERT||5.12|13.31|4.65|5.86|0.85|0.23|
|_<_8 secs|SwinBERT|✓|6.25|25.32|6.94|25.37|0.86|0.23|
|8 - 18 secs|SwinBERT|✓|6.21|25.92|6.31|32.99|0.86|0.23|
|_>_18 secs|SwinBERT|✓|5.40|23.64|6.83|37.03|0.86|0.23|
|_<_8 secs|Joint (Ours)|✓|4.22|22.49|3.24|48.67|0.85|0.22|
|8 - 18 secs|Joint (Ours)|✓|4.02|22.55|3.36|41.25|0.86|0.23|
|_>_18 secs|Joint (Ours)|✓|4.17|24.83|4.02|41.29|0.86|0.22|
|All|SwinBERT|✓|5.94|24.66|6.67|35.09|0.86|0.23|
|All|Joint (Ours)|✓|4.13|23.01|3.54|43.88|0.86|0.23|



Table 14. Step captioning results for various step durations on our HIREST test split. _FT: Finetuning on_ HIREST. 

15 

