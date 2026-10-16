## **VidChapters-7M: Video Chapters at Scale** 

**Antoine Yang** _[†]_ **, Arsha Nagrani** _[§]_ **, Ivan Laptev** _[†]_ **, Josef Sivic** _[¶]_ **, Cordelia Schmid** _[†]_ 

> _†_ Inria Paris, DI ENS, CNRS, PSL Research University _§_ VGG, University of Oxford 

> _¶_ Czech Institute of Informatics, Robotics and Cybernetics at the Czech Technical University in Prague `https://antoyang.github.io/vidchapters.html` 

## **Abstract** 

Segmenting long videos into chapters enables users to quickly navigate to the information of their interest. This important topic has been understudied due to the lack of publicly released datasets. To address this issue, we present VidChapters-7M, a dataset of 817K user-chaptered videos including 7M chapters in total. VidChapters7M is automatically created from videos online in a scalable manner by scraping user-annotated chapters and hence without any additional manual annotation. We introduce the following three tasks based on this data. First, the video chapter generation task consists of temporally segmenting the video and generating a chapter title for each segment. To further dissect the problem, we also define two variants of this task: video chapter generation given ground-truth boundaries, which requires generating a chapter title given an annotated video segment, and video chapter grounding, which requires temporally localizing a chapter given its annotated title. We benchmark both simple baselines and state-of-the-art video-language models for these three tasks. We also show that pretraining on VidChapters-7M transfers well to dense video captioning tasks in both zero-shot and finetuning settings, largely improving the state of the art on the YouCook2 and ViTT benchmarks. Finally, our experiments reveal that downstream performance scales well with the size of the pretraining dataset. Our dataset, code, and models are publicly available at `https://antoyang.github.io/vidchapters.html` . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_VidChapters-7M_Video_Chapters_at_Scale_images/nips_2023_VidChapters-7M_Video_Chapters_at_Scale.pdf-0001-07.png)


**----- Start of picture text -----**<br>
Intro and new  Hiking Low Tatras  Beautifull ascent  New shoes, la  … Back to the campervan Outro and<br>shoes todays plan Low Tatras Sportiva x4 many thanks<br>0:00 0:42 1:46 2:26 9:58 12:10<br>**----- End of picture text -----**<br>


Figure 1: **A video with user-annotated chapters in VidChapters-7M:** the video is temporally segmented into chapters, which are annotated with a chapter title in free-form natural language. 

## **1 Introduction** 

As online media consumption grows, the volume of video content available is increasing rapidly. While searching for specific videos is already a challenging problem, searching within a long video is an even _less_ explored task. Manual navigation can often be time consuming, particularly for long videos. A compelling solution for organizing content online is to segment long videos into _chapters_ (see Figure 1). Chapters are contiguous, non-overlapping segments, completely partitioning a video. Each chapter is also labeled with a short description of the chapter content, enabling users to quickly 

37th Conference on Neural Information Processing Systems (NeurIPS 2023) Track on Datasets and Benchmarks. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_VidChapters-7M_Video_Chapters_at_Scale_images/nips_2023_VidChapters-7M_Video_Chapters_at_Scale.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Video chapter generation Video chapter generation with  Video chapter grounding<br>Output chapter  ground-truth boundaries Input chapter title<br>Input full video times and titles Back to the campervan<br>0:00 Intro and  Input video clip Output  Output<br>0:42 new shoes chapter title Input full video timestamp<br>Back to the  9:58<br>9:58 Back to the  campervan<br>12:10 campervan<br>… …<br>…<br>**----- End of picture text -----**<br>


Figure 2: **Illustration of the three tasks defined for VidChapters-7M.** 

navigate to areas of interest and easily replay different parts of a video. Chapters also give _structure_ to a video, which is useful for long videos that contain inherently listed content, such as listicles [96], instructional videos [64], music compilations and so on. 

Given the plethora of content already online, our goal is to explore automatic solutions related to video chaptering - generating chapters automatically, and grounding chapter titles temporally in long videos. While the benefits of automatically chaptering videos are obvious, data for this task is scarce. Video captioning datasets (such as WebVid-10M [5] and VideoCC [66]) consist of short videos (10s in length), and hence are unsuitable. Web datasets consisting of longer videos (HowTo100M [64], YT-Temporal-1B [118]) come with aligned speech transcripts (ASR), which are only weakly related to visual content, and if used as chapter titles would tend to over-segment videos. Moment retrieval [24, 33] or dense video captioning [42, 127] datasets are perhaps the most useful, but do not focus on creating explicit _structure_ , and instead describe low-level actions comprehensively. Such datasets are also manually annotated, and hence not scalable and small in size (see Table 1). 

To remedy this, we curate VidChapters-7M, a large-scale dataset of user-annotated video chapters automatically scraped from the Web. Our dataset consists of 7M chapters for over 817K videos. Compared to existing datasets, videos in VidChapters-7M are long (23 minutes on average) and contain rich chapter annotations consisting of a starting timestamp and a title per chapter. Our dataset is also diverse, with 12 different video categories having at least 20K videos each, which itself is the size of existing dense video captioning datasets [29, 36, 42, 127]. On top of this dataset we also define 3 video tasks (see Figure 2): (i) _video chapter generation_ which requires temporally segmenting the video and generating a chapter title for each segment; (ii) _video chapter generation given ground-truth boundaries_ , which requires generating a chapter title given an annotated video segment; and (iii) _video chapter grounding_ , which requires temporally localizing a chapter given the chapter title. All three tasks involve parsing and understanding _long_ videos, and multi-modal reasoning (video and text), and hence are valuable steps towards story understanding. 

For all three tasks, we implement simple baselines as well as recent, state-of-the-art video-text methods [45, 101, 114]. We find that the tasks are far from being solved, demonstrating the value of this problem. Interestingly, we also show that our video chapter generation models trained on VidChapters-7M transfer well to dense video captioning tasks in both zero-shot and finetuning settings, largely improving the state of the art on the YouCook2 [127] and ViTT benchmarks [36]. Moreover, we show that pretraining using both speech transcripts and chapter annotations significantly outperforms the widely used pretraining method based only on speech transcripts [65, 114, 118]. This demonstrates the additional value of our dataset as a generic video-language _pretraining_ set. Interestingly, we also find that the transfer performance scales with the size of the chapter dataset. 

In summary, our contributions are: 

- _(i)_ We present VidChapters-7M, a large-scale dataset of user-annotated video chapters obtained from the Web consisting of 817K videos and 7M chapters; 

- _(ii)_ Based on this dataset, we evaluate a range of simple baselines and state-of-the-art videolanguage models on the tasks of video chapter generation with and without ground-truth boundaries, and video chapter grounding; 

- _(iii)_ We show that video chapter generation models trained on VidChapters-7M transfer well to dense video captioning tasks in both zero-shot and finetuning settings, largely improving the state of the art on the YouCook2 [127] and ViTT benchmarks [36], outperforming prior pretraining methods based on narrated videos [114], and showing promising scaling behavior. 

Our dataset, code and models are publicly available on our website [1]. 

2 

|Dataset|Number of<br>videos|Video<br>duration(min)|Number of<br>descriptions|Annotations|
|---|---|---|---|---|
|HowTo100M [64]<br>YT-Temporal-1B [118]<br>HD-VILA-100M[108]|1M<br>**19M**<br>3M|7<br>6<br>7|136M<br>_∼_**900M**<br>103M|Speech transcripts<br>Speech transcripts<br>Speech transcripts|
|ActivityNet Captions [42]<br>YouCook2 [127]<br>ViTT [36]<br>Ego4D[29]|20K<br>2K<br>8K<br>10K|3<br>6<br>4<br>**23**|100K<br>15K<br>56K<br>4M|Dense Captions<br>Dense Captions<br>Dense Captions<br>Dense Captions|
|VidChapters-7M (Ours)|817K|**23**|7M|**Speech transcripts +**<br>**User-annotated Chapters**|



Table 1: **Comparison of VidChapters-7M with existing datasets** . We consider open-sourced video datasets that contain dense natural language descriptions aligned over time. VidChapters-7M is much larger than current dense video captioning datasets. Compared to datasets with ASR (top 3 rows), it is smaller in the total number of videos but contains longer videos with richer annotations (chapters). 

## **2 Related Work** 

**Large-scale vision-language datasets.** The development of powerful multi-modal models [3, 15, 23, 35, 37, 38, 46, 48–50, 54, 61, 62, 72, 85, 87, 90, 94, 99, 105, 115, 116, 129] has been made possible by pretraining on large-scale image-caption datasets scraped from the Web such as SBU [68], Conceptual Captions [82], Conceptual-12M [12], LAIT [71], Wikipedia-ImageText [86], RedCaps [18] and LAION-5B [78]. Similarly, many strong video-language models [2, 27, 30, 41, 45, 47, 52, 53, 58, 65, 80, 81, 88, 89, 91, 97, 100, 107, 110–112, 126] have been pretrained on Web-scraped video-text datasets. These datasets are largely composed of short videos paired with captions, e.g. WebVid-10M [5] and VideoCC [66], or narrated videos with speech transcripts aligned over time (ASR), e.g. HowTo100M [64], YT-Temporal-1B [117, 118] and HD-VILA-100M [108]. Our proposed VidChapters-7M dataset is also downloaded from the Web, via a scalable pipeline without the need for expensive manual annotation. Unlike these datasets, VidChapters-7M consists of long videos with user-annotated chapters aligned over time (see Table 1), which significantly differ from ASR (see Section 3.3). Furthermore, most videos in VidChapters-7M _also_ contain ASR. Finally, VidChapters-7M is also related to the recent ChapterGen dataset [10], which also consists of user-annotated chapters. However, ChapterGen is several orders of magnitude smaller than VidChapters-7M (10K vs 817K videos) and is not open-sourced at the time of writing. 

**Video tasks.** The video chapter generation task requires temporally segmenting the video into chapters, hence is related to video shot detection [76, 77, 84], movie scene segmentation [14, 75], temporal action localization [13, 16, 59, 83, 120, 121] and temporal action segmentation [8, 21, 26, 43, 55, 104]. However, unlike these tasks, video chapter generation also requires generating a free-form natural language chapter title for each segment. Hence this task is also related to video captioning [25, 57, 63, 69, 98, 102, 125], video title generation [4, 119, 123], generic event boundary captioning [103] and dense video captioning [42, 101, 128]. Most related to video chapter generation, the dense video captioning task requires temporally localizing and captioning all events in an untrimmed video. In contrast, video chapter generation requires temporally _segmenting_ the video (i.e. the start of the chapter _i_ + 1 is the end of chapter _i_ , and the chapters cover the full video), and involves generating a chapter title that is substantially shorter than a video caption. We study in more detail the transfer learning between these two tasks in Section 4.4. Finally, the video chapter grounding task is related to temporal language grounding [33, 34, 44, 45, 67, 113, 122, 124]. However, we here focus on localizing a chapter starting point and not a start-end window. Furthermore, most temporal language grounding methods represent the video only with visual inputs, while we also exhibit the benefits of using speech inputs for localizing chapters in videos (see Section 4.3). 

## **3 VidChapters-7M: a large-scale dataset of user-chaptered videos** 

Our goal is to build a large and diverse set of videos annotated with temporarily localized chapter information, consisting of chapter titles and chapter start times. In detail, chapters are contiguous, non-overlapping segments, completely partitioning a video. However manual annotation of chapters is time consuming and expensive and therefore hard to scale. Hence we automatically scrape chapter information from videos available online, as explained in Section 3.1. Then, we perform several processing steps on this data, e.g., to extract speech transcripts, as described in Section 3.2. The 

3 

outcome is VidChapters-7M, a dataset of 817K videos with 7M chapter annotations provided by real users online. Finally, we analyze VidChapters-7M in Section 3.3. Details are given next. 

## **3.1 Data collection** 

Since early 2020, YouTube users can create chapters for uploaded videos by annotating them in the YouTube description. The YouTube API, however, currently does not enable explicit search for user-chaptered videos. Hence, our data collection procedure consists of: (i) Collecting a large and diverse set of video candidates (characterized by their 11-character YouTube video ID), which do not necessarily contain user-annotated chapters; (ii) For all video candidates, downloading the video description, automatically selecting videos with user-annotated chapters, extracting video chapters and downloading corresponding videos. We next describe the individual steps in more detail. 

**Video candidates.** We start from a large pool of video candidates built from the YT-Temporal-180M dataset [117], which was constructed to be more diverse than prior large video datasets such as HowTo100M [64]. Note that while the released YT-Temporal-180M dataset consists of only 5M videos, the authors collected a larger set of candidates by using YouTube’s recommendation algorithm to suggest related videos. We obtained this extended list of 92 million video IDs directly from the authors. 

**Extracting chapters from descriptions.** In the description, chapters typically constitute a block with consecutive lines following the format `“<Timestamp>: <Chapter Title>”` or `“<Chapter Title>: <Timestamp>”` , where the chapter title is written in free-form natural language and its corresponding start timestamp is written in `MM:SS` format. The video should contain at least two timestamps listed in ascending order. Hence we download the descriptions for all video candidates and use standard regular expression operations to verify whether a given description contains user-annotated chapters and extract them if so. Note that some videos contain chapters that are automatically generated by YouTube algorithms, however, these generated chapters do not appear in the descriptions and, hence, are excluded by our procedure for data collection. Also note that the video content is only downloaded for user-chaptered videos, which is convenient for both the downloading speed and storage constraints. Finally, we obtain 817K user-chaptered videos, making up 0.9% of all video candidates. 

## **3.2 Data processing** 

We describe below how we process the previously obtained user-chaptered videos to facilitate building efficient video chapter generation models. For reproducibility, we publicly release the resulting speech transcripts and the code for extracting visual features. 

**ASR extraction.** We observed that most user-chaptered videos contain speech. Hence, for all videos, we extract speech transcripts aligned in time with the video content (ASR) by applying the Whisper-Large-V2 model [73] on the audio track, using faster-whisper [40] backend for computational efficiency. We found that the Whisper model provides higher-quality ASR compared to the YouTube API ASR service on several data samples from VidChapters-7M. We further use WhisperX [6] to derive accurate word-level timestamps which we use to segment the speech transcript into sentences. For example, the Whisper-Large-V2 model extracts speech segments like _“Right, we’re gonna do the Synthetics Dirty Race. No we’re not. [...] So we’re gonna put two t-shirts and two pairs of jeans in the”_ with timestamps 20.478s and 50.465s, and the corresponding first sentence output by WhisperX is _“Right, we’re gonna do the Synthetics Dirty Race.”_ with timestamps 20.538s and 29.26s. 

**Visual feature extraction.** Training end-to-end deep learning models from RGB inputs on minuteslong videos is computationally expensive. Hence we extract visual features with CLIP ViT-L/14 backbone [20, 72] at resolution 224 _×_ 224 pixels and 1 FPS. This model has been trained to map images to text descriptions with a contrastive loss on 400M Web-scraped image-text pairs. 

## **3.3 Data analysis** 

The result of the previously described pipeline is VidChapters-7M, a dataset of 817,076 user-chaptered videos containing 6,813,732 chapters in total. We randomly split VidChapters-7M in training, validation, and testing splits with 801K, 8.2K, and 8.2K videos, respectively. We analyze VidChapters- 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_VidChapters-7M_Video_Chapters_at_Scale_images/nips_2023_VidChapters-7M_Video_Chapters_at_Scale.pdf-0005-00.png)


**----- Start of picture text -----**<br>
1e6<br>500000 1.2<br>80000<br>400000 1.0<br>60000 0.8<br>300000<br>0.6<br>40000 200000<br>0.4<br>20000 100000<br>0.2<br>0 0 0.0<br>Number of chapters Chapter duration (s) Chapter title length (number of words)<br>(a) Number of chapters per video (b) Video chapter duration (s) (c) Chapter title length (# words)<br>140000<br>120000 10 [5]<br>100000<br>10 [3]<br>80000<br>60000<br>40000<br>20000<br>0<br>Video category Chapter language<br>(d) Video category (e) Chapter languages<br>Count Count Count<br>0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 0 30 60 90 120 150 180 210 240 270 300 330 360 390 420 450 480 510 540 570 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19<br>Count Count (log scale)<br>EnglishGerman French Italian Spanish; CastilianTagalogCatalan; ValencianPortugueseIndonesian Dutch; FlemishAfrikaans Swedish NorwegianRussian Korean Somali Hindi Estonian Danish Polish Vietnamese Swahili Welsh Albanian JapaneseSlovenian Finnish Lithuanian Croatian Turkish HungarianLatvian Czech<br>Howto & Style Education People & Blogs Entertainment Gaming Music Sports Autos & Vehicles News & Politics Film & Animation Travel & Events Pets & Animals Comedy<br>Science & Technology Nonprofits & Activism Romanian; Moldavian; Moldovan<br>**----- End of picture text -----**<br>


Figure 3: **Statistics of the VidChapters-7M dataset.** 

7M below and give examples of annotated videos, more statistics, as well as a datasheet in Appendix Sections A, C, and F, respectively. 

**Statistics.** VidChapters-7M is highly diverse and contains 4,894,855 distinct chapter titles. On average, a video contains 8.3 chapters, start times of adjacent chapters are separated by 142.0s seconds, a chapter title contains 5.4 words and a video lasts 1354 seconds. The most represented video category (in YouTube’s glossary) is HowTo & Style, making up 17.0% of total videos. The distributions for the number of chapters per video, the video chapter duration, the length of the chapter title, and the video category are illustrated in Figure 3, and further show the diversity of VidChapters7M, e.g., there are 12 different video categories with at least 20K videos in VidChapters-7M. 

**ASR vs Chapters.** 97.3% of videos in VidChapters-7M contain speech transcripts (ASR). However, user-annotated chapters significantly differ from speech transcripts: on average, a video with ASR contains 269.8 speech sentences (vs 8.3 chapter titles), a speech sentence lasts 3.9 seconds (vs 142.0 seconds for chapters) in the video and contains 11.5 words (vs 5.4 words for chapters). 

**Biases.** Using the langdetect [17] language detection tool, we find that 92.9%/93.9% of total videos in VidChapters-7M have their chapter titles/ASR in English. However, as shown in Figure 3 (bottom right), the distribution of chapter languages includes a long tail of languages, e.g., 13 languages appear in more than 1K videos of VidChapters-7M. We also use GenBit [79] to measure gender bias in the chapters and ASR. We observe that the percentage of female/male/non-binary gendered words is 19.7%/39.7%/40.7% for the chapters, and 11.6%/35.6%/52.8% for the ASR. 

**Ethical considerations.** We employ several techniques to identify harmful visual or language content. We use a classifier [78] built on top of the previously extracted CLIP features to detect not-safe-forwork (NSFW) visual content (such as pornographic and sexualized content). Moreover, we use a language model [31] to detect toxic content in chapter titles and speech transcripts. These processes flag 5,716 (0.70%) visually NSFW videos, 355 (0.04%) videos with toxic chapter titles and 1,368 (0.17%) videos with toxic ASR. We assume the relatively low number of flagged videos is due to the regulations performed by the Web platform used to collect our dataset. Following [78], we refrain from removing these samples to encourage research in fields such as dataset curation and tag them instead. Note that these automated filtering techniques are not perfect and that harmful content may pass. 

5 

|Type of chapter titles|Percentage|
|---|---|
|Speech and visual<br>Audio and visual<br>Speech-only<br>Visual-only<br>Audio-only|49<br>2<br>26<br>3<br>3|
|Structure-only|14|
|Unrelated|3|



Table 2: **Manual assessment of the informativeness of chapter titles in the VidChapters-7M dataset over a random sample of 100 videos.** Video chapter titles can be based on speech and vision; audio and vision; vision, audio or speech alone; or only on the structure of the video ( _e.g._ "step 1", "step 2" etc). In a small number of cases, video chapters are unrelated to the video content. 

**Manual assessment of the quality of annotations.** While chapter titles are manually written and uploaded by real users, sometimes chapter titles are not informative about the content of the video at the corresponding timestamps. To assess the quality of chapter title annotations in our dataset, we inspected a random sample of 100 videos in VidChapters-7M. For each video, we checked if the titles are related to the content of the video chapter and if so which video modalities (ASR, visual or raw audio) they are related to, or if they only refer to the structure of the video (e.g. chapter titles like "step 1", "step 2" etc). Results are presented in Table 2, and show that 83% of videos have chapters related to one or multiple modalities of the video, 14% of videos have chapters only referring to the structure of the video, and 3% of videos have chapters unrelated to the video content. 

## **4 Experiments** 

In this Section, we present the results of models on VidChapters-7M for the full video chapter generation task in Section 4.1, the task of video chapter generation given ground-truth boundaries in Section 4.2 and the video chapter grounding task in Section 4.3. Finally, we study transfer learning from video chapter generation to dense video captioning tasks in Section 4.4. 

**Evaluation metrics.** To evaluate the quality of the generated chapter titles (without their positions), we use standard metrics used for visual captioning: BLEU [70] (B), CIDEr [95] (C), METEOR [7] (M) and ROUGE-L [56] (RL). To evaluate video chapter generation as a whole, including the locations of the generated chapters, we follow standard protocols used for dense video captioning, given the similar nature of the two tasks. We use the standard evaluation tool [42] which calculates matched pairs between generated events and the ground truth across IoU thresholds of {0.3, 0.5, 0.7, 0.9}, and compute captioning metrics over the matched pairs. However, these metrics do not take into account the story of the video and give high scores to methods generating many redundant chapters. Hence for an overall evaluation, we also use SODA_c [22] (S) which first tries to find a temporally optimal matching between generated and reference chapters to capture the story of a video, then computes METEOR scores for the matching and derives F-measure scores from the METEOR scores to penalize redundant chapters. To separately evaluate chapter localization, we report the recall (R@Ks, R@K) and the precision (P@Ks, P@K) across various thresholds in terms of the distance to the ground-truth start time or IoU with the ground-truth start-end window. We also report the average recall (R) and average precision (P) across IoU thresholds of {0.3, 0.5, 0.7, 0.9}. 

**Implementation details.** Unless stated otherwise, for all models, we use the speech transcripts (ASR) and visual features extracted as explained in Section 3.2. By default, each model is taken from the corresponding official implementation, and all model hyper-parameters are set according to the original papers. We use the Adam optimizer [39] for training and select the final model based on the best validation performance. Our experiments are run on 8 NVIDIA A100 80GB GPUs. More details are included in Appendix Section D. 

## **4.1 Video chapter generation** 

In this Section, we study the task of video chapter generation that requires temporally segmenting the video and generating a chapter title for each segment. 

6 

|Method|Modalities<br>PretrainingData<br>Finetuned|S<br>B1<br>B2 B3 B4<br>C<br>M<br>RL|
|---|---|---|
|Text tiling [32] + Random<br>Text tiling [32] + LLaMA [93]<br>Shot detect [92] + BLIP-2 [51]<br>Vid2Seq [114]|Speech<br>_∅_<br>✗<br>Speech<br>Text mixture<br>✗<br>Visual<br>129M image-texts<br>✗<br>Speech+Visual C4 + HowTo100M<br>✗|0.4<br>0.6<br>0.2 0.1 0.0<br>0.8<br>0.7<br>0.6<br>0.2<br>0.4<br>0.1 0.1 0.0<br>0.5<br>0.3<br>0.4<br>0.6<br>0.7<br>0.3 0.1 0.1<br>0.2<br>0.6<br>0.8<br>0.1<br>0.1<br>0.0 0.0 0.0<br>0.1<br>0.1<br>0.1|
|PDVC [101]<br>Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]|Visual<br>_∅_<br>✓<br>Speech<br>C4<br>✓<br>Speech<br>C4 + HowTo100M<br>✓<br>Visual<br>C4<br>✓<br>Visual<br>C4 + HowTo100M<br>✓<br>Speech+Visual<br>C4<br>✓<br>Speech+Visual C4 + HowTo100M<br>✓|6.8<br>9.4<br>3.7 1.4 0.9 35.8 9.4 11.4<br>10.2<br>9.5<br>6.7 4.0 2.7 48.8 8.5 11.0<br>10.5<br>9.9<br>7.0 4.2 2.9 50.7 8.7 11.4<br>3.1<br>2.3<br>1.5 0.6 0.5 10.9 2.2<br>2.9<br>5.5<br>4.5<br>2.8 1.2 0.9 21.1 4.1<br>5.5<br>10.6<br>9.9<br>7.0 4.2 2.8 51.3 8.8 11.6<br>**11.4 10.9 7.7 4.6 3.1 55.7 9.5 12.6**|



Table 3: **Video chapter generation (global metrics) on VidChapters-7M test set.** Here, finetuned refers to finetuning on the VidChapters-7M train set, and speech refers to transcribed speech (ASR). 

|Method|Modalities<br>PretrainingData<br>Finetuned|R@5s R@3s R@0.5 R@0.7 P@5s P@3s P@0.5 P@0.7|
|---|---|---|
|Text tiling [32]<br>Shot detect [92]<br>Vid2Seq [114]|Speech<br>_∅_<br>✗<br>Visual<br>_∅_<br>✗<br>Speech+Visual C4 + HowTo100M<br>✗|9.4<br>5.8<br>23.6<br>8.9<br>12.6<br>7.9<br>26.0<br>8.8<br>31.2<br>27.4<br>24.9<br>12.5<br>33.2<br>29.7<br>18.0<br>8.7<br>10.7<br>9.5<br>5.8<br>0.2<br>23.3<br>18.5<br>1.9<br>0.8|
|PDVC [101]<br>Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]|Visual<br>_∅_<br>✓<br>Speech<br>C4<br>✓<br>Speech<br>C4 + HowTo100M<br>✓<br>Visual<br>C4<br>✓<br>Visual<br>C4 + HowTo100M<br>✓<br>Speech+Visual<br>C4<br>✓<br>Speech+Visual C4 + HowTo100M<br>✓|21.1<br>17.8<br>31.2<br>22.5<br>**45.3**<br>**40.2**<br>**47.2**<br>**26.9**<br>**37.8**<br>**29.5**<br>44.6<br>26.1<br>29.0<br>23.0<br>38.0<br>23.4<br>36.7<br>28.9<br>46.5<br>27.2<br>29.5<br>23.3<br>40.4<br>24.8<br>35.3<br>26.4<br>23.6<br>8.7<br>17.9<br>13.6<br>17.2<br>7.1<br>33.5<br>25.0<br>33.0<br>14.5<br>19.5<br>14.7<br>26.2<br>12.5<br>36.3<br>28.6<br>45.8<br>26.9<br>29.9<br>23.8<br>40.9<br>24.9<br>36.4<br>28.5<br>**48.2**<br>**28.5**<br>30.3<br>24.0<br>43.1<br>26.4|



Table 4: **Video chapter generation (segmentation metrics) on VidChapters-7M test set** . 

**Models.** For the video chapter segmentation subtask, we evaluate two zero-shot approaches (i.e., that are not trained on VidChapters-7M): speech text tiling [32], which detects subtopic shifts based on the analysis of lexical co-occurrence patterns, and a visual scene change detection algorithm [92] based on the sum of absolute differences. To derive zero-shot baselines for the full video chapter generation task, we combine text tiling and shot detection with various alternatives that can generate text given text or visual input: a random baseline that predicts a random speech sentence spoken inside the predicted boundaries, LLaMA-7B [93] (prompted to summarize the speech transcript spoken inside the predicted boundaries) and BLIP-2 [51] (prompted to describe the middle video frame of the predicted segment). Finally, we also train and evaluate two state-of-the-art end-to-end dense video captioning models on VidChapters-7M: PDVC [101] which consists of a visual-only DETR-style [11] architecture and Vid2Seq [114] which is a multi-modal sequence-to-sequence model pretrained on the C4 text corpus [74] and on narrated videos with ASR ( _e.g._ , YT-Temporal-1B [118]). For Vid2Seq, we also report zero-shot results after pretraining on narrated videos without finetuning on VidChapters-7M. 

**Implementation details.** We use the text tiling implementation from the NLTK library [9] which tokenizes the text into pseudosentences of size 50. We use the shot detection software from the FFMPEG library [92] with a confidence threshold of 0.7. For BLIP-2, we use the 3.4B-parameter variant with FLAN-T5-XL [106] and CLIP ViT-L/14 [20, 72]. We reimplement Vid2Seq [114] (originally released in Jax) in PyTorch, use T5-Base pretrained on C4 [74] for initialization and pretrain Vid2Seq on HowTo100M [64]. More details are included in Appendix Section D. 

**Results.** We report the results for video chapter generation using global metrics and localizationonly metrics in Tables 3 and 4, respectively. We observe that models trained on VidChapters-7M outperform zero-shot baselines, demonstrating the effectiveness of training on VidChapters-7M. In particular, PDVC [101] has the best precision and Vid2Seq [114] achieves the best results in terms of overall generation and recall. We also find that Vid2Seq’s speech-only mode outperforms its visual-only mode and that using both speech and visual inputs leads to the best performance. This demonstrates that video chapter generation is a multi-modal task. Finally, we observe that pretraining using ASR in narrated videos from HowTo100M [64] improves the video chapter generation performance of the Vid2Seq model. Specifically, pretraining on HowTo100M is more beneficial for vision-aware models than for the speech-only model. 

7 

|Method|Modalities<br>PretrainingData<br>Finetuned|B1<br>B2<br>B3<br>B4<br>C<br>M<br>RL|
|---|---|---|
|Random<br>LLaMA [93]<br>BLIP-2 [51]<br>Vid2Seq [114]|Speech<br>_∅_<br>✗<br>Speech<br>Text mixture<br>✗<br>Visual<br>129M image-texts<br>✗<br>Speech+Visual C4 + HowTo100M<br>✗|2.4<br>1.3<br>0.9<br>0.7<br>10.4<br>2.2<br>4.4<br>0.0<br>0.0<br>0.0<br>0.0<br>0.0<br>0.1<br>0.2<br>3.1<br>1.5<br>0.9<br>0.7<br>12.4<br>2.2<br>4.5<br>2.0<br>1.2<br>0.9<br>0.6<br>0.9<br>0.3<br>0.6|
|Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]|Speech<br>C4 + HowTo100M<br>✓<br>Visual<br>C4 + HowTo100M<br>✓<br>Speech+Visual<br>C4<br>✓<br>Speech+Visual C4 + HowTo100M<br>✓|21.0 15.5 12.1 10.0 105.3 11.5 24.5<br>10.1<br>5.6<br>3.5<br>2.4<br>47.1<br>5.1<br>14.7<br>21.6 15.7 12.3 10.0 110.8 11.5 26.0<br>**23.5 17.2 13.4 11.0 120.5 12.6 28.3**|



Table 5: **Chapter title generation given ground-truth boundaries on VidChapters-7M test set.** 

|Method|Modalities<br>PretrainingData<br>Finetuned|R@10s R@5s R@3s R@1s R@0.3 R@0.5 R@0.7 R@0.9|
|---|---|---|
|Random<br>BERT [19]<br>CLIP [72]<br>Moment-DETR[45]|Speech<br>_∅_<br>✗<br>Speech<br>BookCorpus + Wikipedia<br>✗<br>Visual<br>400M image-texts<br>✗<br>Visual<br>5.4K narrated videos[45]<br>✗|3.1<br>1.8<br>1.2<br>0.6<br>0.7<br>0.3<br>0.1<br>0.0<br>9.0<br>6.8<br>5.4<br>2.9<br>0.6<br>0.3<br>0.1<br>0.0<br>8.1<br>5.2<br>3.7<br>1.4<br>10.7<br>5.2<br>2.3<br>0.5<br>3.2<br>1.6<br>1.1<br>0.5<br>11.3<br>3.6<br>0.8<br>0.1|
|Moment-DETR [45]|Visual<br>_∅_<br>✓|**21.8**<br>**15.5**<br>**12.4**<br>**8.3**<br>**37.4**<br>**27.3**<br>**17.6**<br>**6.4**|



Table 6: **Video chapter grounding on VidChapters-7M test set.** 

## **Qualitative examples.** See Appendix Section B. 

## **4.2 Video chapter generation given ground-truth boundaries** 

In this Section, we study the task of generating chapter titles provided correct temporal boundaries of video chapters. This task is a simplification of the previously studied task where we assume perfect temporal segmentation. We adopt the same models and implementation details as previously introduced in Section 4.1. 

**Results.** We report results for video chapter generation given ground-truth boundaries in Table 5. Similar to the full video chapter generation task, we observe that solving the task without training on VidChapters-7M is hard. Indeed, LLaMA [93] struggles to summarize the speech content into a chapter title and underperforms the random baseline. Furthermore, BLIP-2 [51] slightly improves over the random baseline. In addition, Vid2Seq [114] in zero-shot mode underperforms the random baseline due to the large domain gap between ASR and chapter titles (see Section 3.3). In comparison, the performance of models trained on VidChapters-7M is significantly higher. Moreover, Vid2Seq’s speech-only mode outperforms its visual-only mode, and using both speech and visual inputs is beneficial, confirming the benefit of multi-modal reasoning for the task of generating chapter titles. Finally, pretraining on narrated videos from HowTo100M [64] improves the performance of the Vid2Seq model on VidChapters-7M. 

## **4.3 Video chapter grounding** 

In this Section, we study the task of video chapter grounding that requires a model to temporally localize a chapter start time (or start-end window) given an annotated chapter title (query). Hence, compared to the video chapter generation task, we here assume chapter titles to be given and focus on the temporal chapter localization only. 

**Models.** We evaluate three zero-shot alternatives: a random baseline that randomly picks the timestamps of a speech sentence in the video, a BERT [19] baseline that picks the timestamps of the speech sentence that has the closest text embedding with the queried chapter title, and a CLIP [72] baseline picking the frames where the query-frame similarity score drops from the highest scoring frame by a certain threshold _ϵ_ . We also train and evaluate on VidChapters-7M a state-of-the-art end-to-end video grounding model: Moment-DETR [45] which is designed for moment retrieval based on visual inputs. Furthermore, we report zero-shot performance of Moment-DETR obtained with the model checkpoint from Lei et al. [45] pretrained on 5.4K narrated videos with ASR from the QVHighlights dataset [45]. 

**Implementation details.** We use the `[CLS]` token sequence embedding for the BERT baseline and a threshold of _ϵ_ = 0 _._ 05 for the CLIP baseline. More details are provided in Appendix Section D. 

8 

|Method|Modalities<br>Pretraining Data|YouCook2 (val)<br>S<br>C<br>M<br>R<br>P|ViTT (test)<br>S<br>C<br>M<br>R<br>P|
|---|---|---|---|
|PDVC [101]<br>E2ESG [130]<br>Vid2Seq [114]<br>Vid2Seq [114]|V<br>_∅_<br>T+V<br>C4 + WikiHow<br>T+V<br>C4 + HTM<br>T+V<br>C4 + YT-Temporal-1B|4.4<br>22.7<br>4.7<br>—<br>—<br>—<br>25.0<br>3.5<br>20.7 20.6<br>8.3<br>48.3<br>9.5<br>27.1 27.0<br>7.9<br>47.1<br>9.3<br>27.9 27.8|—<br>—<br>—<br>—<br>—<br>—<br>25.0<br>8.1<br>32.2 32.1<br>—<br>—<br>—<br>—<br>—<br>13.5 43.5<br>8.5<br>42.6 46.2|
|PDVC_†_<br>PDVC_†_<br>Vid2Seq_†_<br>Vid2Seq_†_<br>Vid2Seq_†_<br>Vid2Seq_†_<br>Vid2Seq_†_<br>Vid2Seq_†_|V<br>_∅_<br>V<br>VC (Chap.)<br>T+V<br>C4 + HTM<br>T+V<br>C4 + VC (ASR+Chap.)<br>T+V<br>C4 + HTM + VC (ASR)<br>T+V<br>C4 + HTM + 1% of VC (ASR+Chap)<br>T+V<br>C4 + HTM + 10% of VC (ASR+Chap.)<br>T+V<br>C4 + HTM + VC (ASR+Chap.)|4.8<br>28.8<br>5.8<br>22.6 33.1<br>5.9<br>34.7<br>7.5<br>28.8 **36.4**<br>8.6<br>53.2 10.5 29.2 26.2<br>9.8<br>62.9 11.7 32.5 30.1<br>8.4<br>50.1 10.3 29.7 26.3<br>8.8<br>52.7 10.4 29.3 27.6<br>9.9<br>63.9 12.1 32.4 31.4<br>**10.3 67.2 12.3 34.0** 31.2|9.4<br>40.6 **16.5** 19.2 37.4<br>10.1 41.5 16.1 21.3 37.2<br>14.1 44.8<br>8.7<br>43.8 44.5<br>**15.1 50.9**<br>9.6<br>45.1 46.7<br>14.3 45.6<br>8.8<br>43.7 44.9<br>13.5 41.6<br>8.2<br>44.7 42.1<br>14.5 47.4<br>9.2<br>45.3 45.9<br>15.0 50.0<br>9.5<br>**45.5 46.9**|



Table 7: **Comparison with the state of the art on the YouCook2 and ViTT dense video captioning benchmarks.** T: Transcribed speech, V: Visual, HTM: HowTo100M [64], VC: VidChapters-7M, Chap.: Chapters. _[†]_ denote results of our experiments. 

|Method|Modalities<br>Pretraining Data|YouCook2 (val)<br>S<br>C<br>M<br>R<br>P|ViTT (test)<br>S<br>C<br>M<br>R<br>P|
|---|---|---|---|
|Text tiling [32] + Random<br>Text tiling [32] + LLaMA [93]<br>Shot detect[92]+ BLIP-2[51]|T<br>_∅_<br>T<br>Text mixture<br>V<br>129M image-texts|0.3 0.9 0.3 3.8<br>6.6<br>0.2 0.6 0.2 3.8<br>6.6<br>0.6 1.0 0.5 8.9<br>5.5|0.3 0.6 0.6 11.6 24.4<br>0.2 0.6 0.5 11.6 24.4<br>0.2 0.1 0.2 3.1 13.7|
|Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]<br>Vid2Seq [114]|V<br>C4 + VC (ASR)<br>V<br>C4 + VC (Chap.)<br>T+V<br>C4 + HTM<br>T+V<br>C4 + VC (ASR)<br>T+V<br>C4 + VC (Chap.)<br>T+V<br>C4 + VC (ASR+Chap.)<br>T+V<br>C4 + HTM + VC (ASR)<br>T+V<br>C4 + HTM + 1% of VC (ASR+Chap.)<br>T+V<br>C4 + HTM + 10% of VC (ASR+Chap.)<br>T+V<br>C4 + HTM + VC (ASR+Chap.)|0.0 0.0 0.0 0.0<br>0.0<br>0.7 1.1 0.5 21.3 8.6<br>0.0 0.1 0.0 0.5<br>0.6<br>0.1 0.1 0.0 1.1<br>0.9<br>0.1 0.2 0.1 0.7<br>1.4<br>3.2 10.2 2.9 20.6 19.7<br>0.0 0.1 0.0 1.2<br>0.9<br>2.7 7.2 2.1 18.1 17.3<br>3.2 11.5 3.0 19.4 19.2<br>**3.9 13.3 3.4 22.3 20.1**|0.0 0.0 0.0 0.2<br>0.8<br>1.5 1.9 0.6 18.9 10.4<br>0.0 0.0 0.0 0.5<br>1.0<br>0.0 0.0 0.0 0.7<br>0.6<br>0.7 1.1 0.3 14.3 12.8<br>**9.1 30.2 6.7 33.8 40.8**<br>0.0 0.0 0.0 0.8<br>0.7<br>5.5 15.5 4.3 31.3 37.1<br>6.4 21.6 5.3 31.0 38.2<br>9.0 28.0 6.5 33.7 40.1|



Table 8: **Zero-shot dense video captioning on the YouCook2 and ViTT benchmarks.** T: Transcribed speech, V: Visual, HTM: HowTo100M [64], VC: VidChapters-7M, Chap.: Chapters. 

**Results.** We report results for the video chapter grounding task in Table 6. We first observe that the simple zero-shot baselines based on ASR can decently find start times, but struggle to predict start-end windows due to the important domain gap between ASR and video chapters (see Section 3.3). The CLIP [72] baseline slightly underperforms the BERT baseline [19] at retrieving start times, but is much better at finding start-end windows. Furthermore, the Moment-DETR model [45] trained on VidChapters-7M outperform the zero-shot baselines for both localization of start times and start-end windows, which further demonstrates the effectiveness of training on VidChapters-7M. Finally, we note that Moment-DETR cannot handle speech inputs, but hope that our results showing the benefit of this modality on other tasks in VidChapters-7M will foster research in the localization of language queries in untrimmed videos using multi-modal inputs (vision and speech transcripts). 

## **4.4 Transfer learning on dense video captioning** 

In this Section, we investigate the pretraining of video-language models on our new VidChapters-7M. To this end, we adopt video chapter generation models trained on VidChapters-7M (see Section 4.1) to the tasks of dense video captioning with or without finetuning. 

**Datasets.** We use two dense video captioning datasets. **YouCook2** [127] has 2K untrimmed videos of cooking procedures. On average, each video lasts 320s and is annotated with 7.7 temporally-localized sentences. **ViTT** [36] was created to better reflect the distribution of instructional videos in the wild compared to YouCook2, and consists of 8K untrimmed instructional videos. On average, each video lasts 250s and is annotated with 7.1 temporally-localized short tags. For both datasets, we extract speech transcripts and visual features as described in Section 3.2, and follow the standard splits for training, validation and testing. Note that we only use videos available on YouTube at the time of the work, resulting in 10 to 20% less videos than in the original datasets. 

**Implementation details.** See Section 4.1 and Appendix Section D. 

9 

**Results after finetuning.** In Table 7, we show that pretraining for video chapter generation on VidChapters-7M greatly improves the downstream dense video captioning performance compared to training from scratch or pretraining only with ASR data as done in previous work [114]. We also find that pretraining both on HowTo100M [64] and VidChapters-7M results in the best overall performance. In particular, the Vid2Seq model pretrained on both HowTo100M and VidChapters-7M largely improves the state of the art on both the YouCook2 and ViTT benchmarks. In detail, on the YouCook2 benchmark, in the setting with C4 + HowTo100M pretraining, we observe that a boost of about 4.9 points in CIDEr is obtained with our reimplementation of Vid2Seq, and that 14.0 additional points in CIDEr are obtained by pretraining on VidChapters-7M. Finally, we report the results of the Vid2Seq model after pretraining on different fractions of VidChapters-7M for a fixed number of iterations. We construct these subsets such that larger subsets include the smaller ones. These results suggest that the scale of the chapter dataset is an important factor in the downstream dense video captioning performance. We conclude that VidChapters-7M opens a promising avenue for multi-modal pretraining. We further show qualitative examples of dense video captioning in Appendix Section B. 

**Zero-shot dense video captioning.** In Table 8, we report results obtained by directly applying video chapter generation models trained on VidChapters-7M for dense video captioning without finetuning for this task. As far as we know, our work is the first to explore this challenging zero-shot setting where no manual annotation of dense video captions is used for training. The Vid2Seq model trained only using ASR data underperforms the random baseline, due to the large domain difference between speech transcripts and dense captions [114]. In the visual-only setting, the variant trained on chapter annotations is better than the variant trained on ASR annotations. In the visual+speech settings, only using chapter annotations does not perform well, as training only on chapters (i.e., without speech) does not enable the model to learn how to use the input speech modality at inference. However, using both ASR and chapter annotations results in a largely better zero-shot dense video captioning performance and outperforms all baselines not trained on VidChapters-7M, demonstrating the complementary nature of the ASR and chapters annotations. Finally, we also observe the benefits of increasing the size of the pretraining dataset of chapters in this setting. 

## **5 Conclusion, Limitations, and Societal Impacts** 

In this work, we presented VidChapters-7M, a large-scale dataset of user-chaptered videos. Furthermore, we evaluated a variety of baselines on the tasks of video chapter generation with and without ground-truth boundaries and video chapter grounding. Finally, we investigated the potential of VidChapters-7M for pretraining video-language models and demonstrated improved performance on the dense video captioning tasks. VidChapters-7M thus provides a new resource to the research community that can be used both as a benchmark for the video chapter generation tasks and as a powerful means for pretraining generic video-language models. 

**Limitations.** As it is derived from YT-Temporal-180M [117], VidChapters-7M inherits the biases in the distribution of video categories reflected in this dataset. 

**Societal Impacts.** The development of video chapter generation models might facilitate potentially harmful downstream applications, e.g., video surveillance. Moreover, models trained on VidChapters7M might reflect biases present in videos from YouTube. It is important to keep this in mind when deploying, analysing and building upon these models. 

## **Acknowledgements** 

This work was granted access to the HPC resources of IDRIS under the allocation 2023-A0131011670 made by GENCI. The work was funded by Antoine Yang’s Google PhD fellowship, the French government under management of Agence Nationale de la Recherche as part of the "Investissements d’avenir" program, reference ANR-19-P3IA-0001 (PRAIRIE 3IA Institute), the Louis Vuitton ENS Chair on Artificial Intelligence, the European Regional Development Fund under project IMPACT (reg. no. CZ.02.1.01/0.0/0.0/15 003/0000468). We thank Jack Hessel and Rémi Lacroix for helping with collecting the dataset, and Antoine Miech for interesting discussions. 

10 

## **References** 

- [1] VidChapters-7M project webpage. `https://antoyang.github.io/vidchapters.html` . 2, 18, 28, 30, 31 

- [2] Hassan Akbari, Liangzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, and Boqing Gong. VATT: Transformers for multimodal self-supervised learning from raw video, audio and text. _NeurIPS_ , 2021. 3 

- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In _NeurIPS_ , 2022. 3 

- [4] Soheyla Amirian, Khaled Rasheed, Thiab R Taha, and Hamid R Arabnia. Automatic generation of descriptive titles for video clips using deep learning. In _Advances in Artificial Intelligence and Applied Cognitive Computing: Proceedings from ICAI’20 and ACC’20_ , 2021. 3 

- [5] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In _ICCV_ , 2021. 2, 3 

- [6] Max Bain, Jaesung Huh, Tengda Han, and Andrew Zisserman. WhisperX: Time-accurate speech transcription of long-form audio. In _Interspeech_ , 2023. 4 

- [7] Satanjeev Banerjee and Alon Lavie. METEOR: An automatic metric for mt evaluation with improved correlation with human judgments. In _Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization_ , 2005. 6 

- [8] Nadine Behrmann, S Alireza Golestaneh, Zico Kolter, Jürgen Gall, and Mehdi Noroozi. Unified fully and timestamp supervised temporal action segmentation via sequence to sequence translation. In _ECCV_ , 2022. 3 

- [9] Steven Bird, Ewan Klein, and Edward Loper. _Natural language processing with Python: analyzing text with the natural language toolkit_ . O’Reilly Media, Inc., 2009. 7 

- [10] Xiao Cao, Zitan Chen, Canyu Le, and Lei Meng. Multi-modal video chapter generation. In _BMVC_ , 2022. 3 

- [11] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In _ECCV_ , 2020. 7 

- [12] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In _CVPR_ , 2021. 3 

- [13] Yu-Wei Chao, Sudheendra Vijayanarasimhan, Bryan Seybold, David A Ross, Jia Deng, and Rahul Sukthankar. Rethinking the Faster R-CNN architecture for temporal action localization. In _CVPR_ , 2018. 3 

- [14] Shixing Chen, Xiaohan Nie, David Fan, Dongqing Zhang, Vimal Bhat, and Raffay Hamid. Shot contrastive self-supervised learning for scene boundary detection. In _CVPR_ , 2021. 3 

- [15] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. UNITER: Universal image-text representation learning. In _ECCV_ , 2020. 3 

- [16] Feng Cheng and Gedas Bertasius. TALLformer: Temporal action localization with longmemory transformer. In _ECCV_ , 2022. 3 

- [17] Michal Danilák. Language detection library. `https://github.com/Mimino666/ langdetect` , 2021. 5 

- [18] Karan Desai, Gaurav Kaul, Zubin Aysola, and Justin Johnson. RedCaps: Web-curated imagetext data created by the people, for the people. In _NeurIPS Datasets and Benchmarks_ , 2021. 3 

- [19] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In _NAACL-HLT_ , 2019. 8, 9 

11 

- [20] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_ , 2021. 4, 7 

- [21] Yazan Abu Farha and Jurgen Gall. MS-TCN: Multi-stage temporal convolutional network for action segmentation. In _CVPR_ , 2019. 3 

- [22] Soichiro Fujita, Tsutomu Hirao, Hidetaka Kamigaito, Manabu Okumura, and Masaaki Nagata. SODA: Story oriented dense video captioning evaluation framework. In _ECCV_ , 2020. 6 

- [23] Zhe Gan, Yen-Chun Chen, Linjie Li, Chen Zhu, Yu Cheng, and Jingjing Liu. Large-scale adversarial training for vision-and-language representation learning. In _NeurIPS_ , 2020. 3 

- [24] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. TALL: Temporal activity localization via language query. In _ICCV_ , 2017. 2 

- [25] Lianli Gao, Zhao Guo, Hanwang Zhang, Xing Xu, and Heng Tao Shen. Video captioning with attention-based lstm and semantic consistency. _IEEE Transactions on Multimedia_ , 2017. 3 

- [26] Shang-Hua Gao, Qi Han, Zhong-Yu Li, Pai Peng, Liang Wang, and Ming-Ming Cheng. Global2Local: Efficient structure search for video action segmentation. In _CVPR_ , 2021. 3 

- [27] Yuying Ge, Yixiao Ge, Xihui Liu, Dian Li, Ying Shan, Xiaohu Qie, and Ping Luo. Bridging video-text retrieval with multiple choice questions. In _CVPR_ , 2022. 3 

- [28] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé Iii, and Kate Crawford. Datasheets for datasets. _Communications of the ACM_ , 2021. 18, 24 

- [29] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, Miguel Martin, Tushar Nagarajan, Ilija Radosavovic, Santhosh Kumar Ramakrishnan, Fiona Ryan, Jayant Sharma, Michael Wray, Mengmeng Xu, Eric Zhongcong Xu, Chen Zhao, Siddhant Bansal, Dhruv Batra, Vincent Cartillier, Sean Crane, Tien Do, Morrie Doulaty, Akshay Erapalli, Christoph Feichtenhofer, Adriano Fragomeni, Qichen Fu, Christian Fuegen, Abrham Gebreselasie, Cristina Gonzalez, James Hillis, Xuhua Huang, Yifei Huang, Wenqi Jia, Weslie Khoo, Jachym Kolar, Satwik Kottur, Anurag Kumar, Federico Landini, Chao Li, Yanghao Li, Zhenqiang Li, Karttikeya Mangalam, Raghava Modhugu, Jonathan Munro, Tullie Murrell, Takumi Nishiyasu, Will Price, Paola Ruiz Puentes, Merey Ramazanova, Leda Sari, Kiran Somasundaram, Audrey Southerland, Yusuke Sugano, Ruijie Tao, Minh Vo, Yuchen Wang, Xindi Wu, Takuma Yagi, Yunyi Zhu, Pablo Arbelaez, David Crandall, Dima Damen, Giovanni Maria Farinella, Bernard Ghanem, Vamsi Krishna Ithapu, C. V. Jawahar, Hanbyul Joo, Kris Kitani, Haizhou Li, Richard Newcombe, Aude Oliva, Hyun Soo Park, James M. Rehg, Yoichi Sato, Jianbo Shi, Mike Zheng Shou, Antonio Torralba, Lorenzo Torresani, Mingfei Yan, and Jitendra Malik. Ego4D: Around the World in 3,000 Hours of Egocentric Video. In _CVPR_ , 2022. 2, 3 

- [30] Tengda Han, Weidi Xie, and Andrew Zisserman. Temporal alignment networks for long-term video. In _CVPR_ , 2022. 3 

- [31] Laura Hanu and Unitary team. Detoxify. `https://github.com/unitaryai/detoxify` , 2020. 5, 20, 26 

- [32] Marti A Hearst. Text tiling: Segmenting text into multi-paragraph subtopic passages. _Computational linguistics_ , 1997. 7, 9, 24 

- [33] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. _ICCV_ , 2017. 2, 3 

- [34] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with temporal language. In _EMNLP_ , 2018. 3 

- [35] Xiaowei Hu, Zhe Gan, Jianfeng Wang, Zhengyuan Yang, Zicheng Liu, Yumao Lu, and Lijuan Wang. Scaling up vision-language pre-training for image captioning. In _CVPR_ , 2022. 3 

12 

- [36] Gabriel Huang, Bo Pang, Zhenhai Zhu, Clara Rivera, and Radu Soricut. Multimodal pretraining for dense video captioning. In _AACL-IJCNLP_ , 2020. 2, 3, 9 

- [37] Zhicheng Huang, Zhaoyang Zeng, Yupan Huang, Bei Liu, Dongmei Fu, and Jianlong Fu. Seeing out of the box: End-to-end pre-training for vision-language representation learning. In _CVPR_ , 2021. 3 

- [38] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _ICML_ , 2021. 3 

- [39] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _ICLR_ , 2015. 6 

- [40] Guillaume Klein. faster-whisper library. `https://github.com/guillaumekln/ faster-whisper` , 2023. 4 

- [41] Dohwan Ko, Joonmyung Choi, Juyeon Ko, Shinyeong Noh, Kyoung-Woon On, Eun-Sol Kim, and Hyunwoo J Kim. Video-text representation learning via differentiable weak temporal alignment. In _CVPR_ , 2022. 3 

- [42] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Densecaptioning events in videos. In _ICCV_ , 2017. 2, 3, 6 

- [43] Colin Lea, Michael D Flynn, Rene Vidal, Austin Reiter, and Gregory D Hager. Temporal convolutional networks for action segmentation and detection. In _CVPR_ , 2017. 3 

- [44] Jie Lei, Licheng Yu, Tamara L Berg, and Mohit Bansal. TVR: A large-scale dataset for video-subtitle moment retrieval. In _ECCV_ , 2020. 3 

- [45] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. In _NeurIPS_ , 2021. 2, 3, 8, 9, 23 

- [46] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, Mohit Bansal, and Jingjing Liu. Less is more: ClipBERT for video-and-language learning via sparse sampling. In _CVPR_ , 2021. 3 

- [47] Dongxu Li, Junnan Li, Hongdong Li, Juan Carlos Niebles, and Steven CH Hoi. Align and prompt: Video-and-language pre-training with entity prompts. In _CVPR_ , 2022. 3 

- [48] Gen Li, Nan Duan, Yuejian Fang, Ming Gong, Daxin Jiang, and Ming Zhou. Unicoder-VL: A universal encoder for vision and language by cross-modal pre-training. In _AAAI_ , 2020. 3 

- [49] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. In _NeurIPS_ , 2021. 

- [50] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In _ICML_ , 2022. 3 

- [51] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. In _ICML_ , 2023. 7, 8, 9, 20, 23, 24 

- [52] Linjie Li, Yen-Chun Chen, Yu Cheng, Zhe Gan, Licheng Yu, and Jingjing Liu. HERO: Hierarchical encoder for video+language omni-representation pre-training. In _EMNLP_ , 2020. 3 

- [53] Linjie Li, Zhe Gan, Kevin Lin, Chung-Ching Lin, Zicheng Liu, Ce Liu, and Lijuan Wang. LAVENDER: Unifying video-language understanding as masked language modeling. In _CVPR_ , 2023. 3 

- [54] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In _ECCV_ , 2020. 3 

13 

- [55] Zhe Li, Yazan Abu Farha, and Jurgen Gall. Temporal action segmentation from timestamp supervision. In _CVPR_ , 2021. 3 

- [56] Chin-Yew Lin. Rouge: a package for automatic evaluation of summaries. In _Proceedings of the Workshop on Text Summarization Branches Out (WAS)_ , 2004. 6 

- [57] Kevin Lin, Linjie Li, Chung-Ching Lin, Faisal Ahmed, Zhe Gan, Zicheng Liu, Yumao Lu, and Lijuan Wang. SwinBERT: End-to-end transformers with sparse attention for video captioning. In _CVPR_ , 2022. 3 

- [58] Kevin Qinghong Lin, Alex Jinpeng Wang, Mattia Soldan, Michael Wray, Rui Yan, Eric Zhongcong Xu, Difei Gao, Rongcheng Tu, Wenzhe Zhao, Weijie Kong, et al. Egocentric videolanguage pretraining. In _NeurIPS_ , 2022. 3 

- [59] Xiaolong Liu, Qimeng Wang, Yao Hu, Xu Tang, Shiwei Zhang, Song Bai, and Xiang Bai. Endto-end temporal action detection with transformer. In _IEEE Transactions on Image Processing_ , 2022. 3 

- [60] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _ICLR_ , 2019. 23 

- [61] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. ViLBERT: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In _NeurIPS_ , 2019. 3 

- [62] Jiasen Lu, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, and Stefan Lee. 12-in-1: Multi-task vision and language representation learning. In _CVPR_ , 2020. 3 

- [63] Huaishao Luo, Lei Ji, Botian Shi, Haoyang Huang, Nan Duan, Tianrui Li, Xilin Chen, and Ming Zhou. UniViLM: A unified video and language pre-training model for multimodal understanding and generation. _arXiv preprint arXiv:2002.06353_ , 2020. 3 

- [64] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. HowTo100M: Learning a text-video embedding by watching hundred million narrated video clips. In _ICCV_ , 2019. 2, 3, 4, 7, 8, 9, 10, 21 

- [65] Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman. End-to-end learning of visual representations from uncurated instructional videos. In _CVPR_ , 2020. 2, 3 

- [66] Arsha Nagrani, Paul Hongsuck Seo, Bryan Seybold, Anja Hauth, Santiago Manen, Chen Sun, and Cordelia Schmid. Learning audio-video modalities from image captions. In _ECCV_ , 2022. 2, 3 

- [67] Guoshun Nan, Rui Qiao, Yao Xiao, Jun Liu, Sicong Leng, Hao Zhang, and Wei Lu. Interventional video grounding with dual contrastive learning. In _CVPR_ , 2021. 3 

- [68] Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. In _NeurIPS_ , 2011. 3 

- [69] Yingwei Pan, Ting Yao, Houqiang Li, and Tao Mei. Video captioning with transferred semantic attributes. In _CVPR_ , 2017. 3 

- [70] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. BLEU: a method for automatic evaluation of machine translation. In _ACL_ , 2002. 6 

- [71] Di Qi, Lin Su, Jia Song, Edward Cui, Taroon Bharti, and Arun Sacheti. ImageBERT: Cross-modal pre-training with large-scale weak-supervised image-text data. _arXiv preprint arXiv:2001.07966_ , 2020. 3 

- [72] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 3, 4, 7, 8, 9 

- [73] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. _arXiv preprint arXiv:2212.04356_ , 2022. 4, 27 

14 

- [74] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _JMLR_ , 2020. 7, 21 

- [75] Anyi Rao, Linning Xu, Yu Xiong, Guodong Xu, Qingqiu Huang, Bolei Zhou, and Dahua Lin. A local-to-global approach to multi-modal movie scene segmentation. In _CVPR_ , 2020. 3 

- [76] Zeeshan Rasheed and Mubarak Shah. Scene detection in hollywood movies and tv shows. In _CVPR_ , 2003. 3 

- [77] Yong Rui, Thomas S Huang, and Sharad Mehrotra. Exploring video structure beyond the shots. In _IEEE International Conference on Multimedia Computing and Systems_ , 1998. 3 

- [78] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. LAION-5B: An open large-scale dataset for training next generation image-text models. In _NeurIPS_ , 2022. 3, 5, 20, 26 

- [79] Kinshuk Sengupta, Rana Maher, Declan Groves, and Chantal Olieman. Genbit: measure and mitigate gender bias in language datasets. _Microsoft Journal of Applied Research_ , 2021. 5 

- [80] Paul Hongsuck Seo, Arsha Nagrani, and Cordelia Schmid. Look before you speak: Visually contextualized utterances. In _CVPR_ , 2021. 3 

- [81] Paul Hongsuck Seo, Arsha Nagrani, Anurag Arnab, and Cordelia Schmid. End-to-end generative pretraining for multimodal video captioning. In _CVPR_ , 2022. 3 

- [82] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual Captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _ACL_ , 2018. 3 

- [83] Zheng Shou, Dongang Wang, and Shih-Fu Chang. Temporal action localization in untrimmed videos via multi-stage CNNs. In _CVPR_ , 2016. 3 

- [84] Panagiotis Sidiropoulos, Vasileios Mezaris, Ioannis Kompatsiaris, Hugo Meinedo, Miguel Bugalho, and Isabel Trancoso. Temporal video segmentation to scenes using high-level audiovisual features. _IEEE TCSVT_ , 2011. 3 

- [85] Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. FLAVA: A foundational language and vision alignment model. In _CVPR_ , 2022. 3 

- [86] Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork. Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning. In _ACM SIGIR Conference on Research and Development in Information Retrieval_ , 2021. 3 

- [87] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. VL-BERT: Pre-training of generic visual-linguistic representations. In _ICLR_ , 2019. 3 

- [88] Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. VideoBERT: A joint model for video and language representation learning. In _ICCV_ , 2019. 3 

- [89] Yuchong Sun, Hongwei Xue, Ruihua Song, Bei Liu, Huan Yang, and Jianlong Fu. Long-form video-language pre-training with multimodal temporal contrastive learning. In _NeurIPS_ , 2022. 3 

- [90] Hao Tan and Mohit Bansal. LXMERT: Learning cross-modality encoder representations from transformers. In _EMNLP_ , 2019. 3 

- [91] Zineng Tang, Jaemin Cho, Yixin Nie, and Mohit Bansal. TVLT: Textless vision-language transformer. In _NeurIPS_ , 2022. 3 

- [92] Suramya Tomar. Converting video formats with ffmpeg. _Linux Journal_ , 2006. 7, 9, 24 

15 

- [93] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 7, 8, 9, 20, 23, 24 

- [94] Maria Tsimpoukelli, Jacob Menick, Serkan Cabi, SM Eslami, Oriol Vinyals, and Felix Hill. Multimodal few-shot learning with frozen language models. In _NeurIPS_ , 2021. 3 

- [95] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. CIDEr: Consensus-based image description evaluation. In _CVPR_ , 2015. 6 

- [96] Bram Vijgen et al. The listicle: An exploring research on an interesting shareable new media phenomenon. _Studia Universitatis Babes-Bolyai-Ephemerides_ , 59(1):103–122, 2014. 2 

- [97] Alex Jinpeng Wang, Yixiao Ge, Rui Yan, Yuying Ge, Xudong Lin, Guanyu Cai, Jianping Wu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. All in one: Exploring unified video-language pre-training. In _CVPR_ , 2023. 3 

- [98] Bairui Wang, Lin Ma, Wei Zhang, and Wei Liu. Reconstruction network for video captioning. In _CVPR_ , 2018. 3 

- [99] Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lijuan Wang. GIT: A generative image-to-text transformer for vision and language. In _TMLR_ , 2022. 3 

- [100] Jinpeng Wang, Yixiao Ge, Guanyu Cai, Rui Yan, Xudong Lin, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Object-aware video-language pre-training for retrieval. In _CVPR_ , 2022. 3 

- [101] Teng Wang, Ruimao Zhang, Zhichao Lu, Feng Zheng, Ran Cheng, and Ping Luo. End-to-end dense video captioning with parallel decoding. In _ICCV_ , 2021. 2, 3, 7, 9, 21, 24 

- [102] Xin Wang, Wenhu Chen, Jiawei Wu, Yuan-Fang Wang, and William Yang Wang. Video captioning via hierarchical reinforcement learning. In _CVPR_ , 2018. 3 

- [103] Yuxuan Wang, Difei Gao, Licheng Yu, Weixian Lei, Matt Feiszli, and Mike Zheng Shou. GEB+: A benchmark for generic event boundary captioning, grounding and retrieval. In _ECCV_ , 2022. 3 

- [104] Zhenzhi Wang, Ziteng Gao, Limin Wang, Zhifeng Li, and Gangshan Wu. Boundary-aware cascade networks for temporal action segmentation. In _ECCV_ , 2020. 3 

- [105] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. SimVLM: Simple visual language model pretraining with weak supervision. In _ICLR_ , 2022. 3 

- [106] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In _ICLR_ , 2022. 7 

- [107] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. VideoCLIP: Contrastive pre-training for zero-shot video-text understanding. In _EMNLP_ , 2021. 3 

- [108] Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In _CVPR_ , 2022. 3 

- [109] Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. mT5: A massively multilingual pre-trained text-to-text transformer. In _NAACL_ , 2021. 23 

- [110] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Just ask: Learning to answer questions from millions of narrated videos. In _ICCV_ , 2021. 3 

- [111] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Learning to answer visual questions from web videos. _IEEE TPAMI_ , 2022. 

16 

- [112] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Zero-shot video question answering via frozen bidirectional language models. In _NeurIPS_ , 2022. 3 

- [113] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. TubeDETR: Spatio-temporal video grounding with transformers. In _CVPR_ , 2022. 3 

- [114] Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vid2Seq: Large-scale pretraining of a visual language model for dense video captioning. In _CVPR_ , 2023. 2, 7, 8, 9, 10, 21, 23, 24 

- [115] Fei Yu, Jiji Tang, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. ERNIE-ViL: Knowledge enhanced vision-language representations through scene graph. In _AAAI_ , 2020. 3 

- [116] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision. _arXiv preprint arXiv:2111.11432_ , 2021. 3 

- [117] Rowan Zellers, Ximing Lu, Jack Hessel, Youngjae Yu, Jae Sung Park, Jize Cao, Ali Farhadi, and Yejin Choi. MERLOT: Multimodal neural script knowledge models. In _NeurIPS_ , 2021. 3, 4, 10, 25, 27 

- [118] Rowan Zellers, Jiasen Lu, Ximing Lu, Youngjae Yu, Yanpeng Zhao, Mohammadreza Salehi, Aditya Kusupati, Jack Hessel, Ali Farhadi, and Yejin Choi. MERLOT Reserve: Neural script knowledge through vision and language and sound. In _CVPR_ , 2022. 2, 3, 7, 21 

- [119] Kuo-Hao Zeng, Tseng-Hung Chen, Juan Carlos Niebles, and Min Sun. Title generation for user generated videos. In _ECCV_ , 2016. 3 

- [120] Runhao Zeng, Wenbing Huang, Mingkui Tan, Yu Rong, Peilin Zhao, Junzhou Huang, and Chuang Gan. Graph convolutional networks for temporal action localization. In _CVPR_ , 2019. 3 

- [121] Chenlin Zhang, Jianxin Wu, and Yin Li. ActionFormer: Localizing moments of actions with transformers. In _ECCV_ , 2022. 3 

- [122] Hao Zhang, Aixin Sun, Wei Jing, and Joey Tianyi Zhou. Span-based localizing network for natural language video localization. In _ACL_ , 2020. 3 

- [123] Shengyu Zhang, Ziqi Tan, Zhou Zhao, Jin Yu, Kun Kuang, Tan Jiang, Jingren Zhou, Hongxia Yang, and Fei Wu. Comprehensive information integration modeling framework for video titling. In _ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_ , 2020. 3 

- [124] Songyang Zhang, Houwen Peng, Jianlong Fu, and Jiebo Luo. Learning 2d temporal adjacent networks for moment localization with natural language. In _AAAI_ , 2020. 3 

- [125] Ziqi Zhang, Yaya Shi, Chunfeng Yuan, Bing Li, Peijin Wang, Weiming Hu, and Zheng-Jun Zha. Object relational graph with teacher-recommended learning for video captioning. In _CVPR_ , 2020. 3 

- [126] Yue Zhao, Ishan Misra, Philipp Krähenbühl, and Rohit Girdhar. Learning video representations from large language models. In _CVPR_ , 2023. 3 

- [127] Luowei Zhou, Xu Chenliang, and Jason J. Corso. Towards automatic learning of procedures from web instructional videos. In _AAAI_ , 2018. 2, 3, 9 

- [128] Luowei Zhou, Yingbo Zhou, Jason J Corso, Richard Socher, and Caiming Xiong. End-to-end dense video captioning with masked transformer. In _CVPR_ , 2018. 3 

- [129] Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason J Corso, and Jianfeng Gao. Unified vision-language pre-training for image captioning and VQA. In _AAAI_ , 2020. 3 

- [130] Wanrong Zhu, Bo Pang, Ashish Thapliyal, William Yang Wang, and Radu Soricut. End-to-end dense video captioning as sequence generation. In _COLING_ , 2022. 9 

17 

## **Appendix** 

In this Appendix, we present the following items: 

- _(i)_ Additional examples from our VidChapters-7M dataset (Section A). 

- _(ii)_ Qualitative examples of video chapter generation and dense video caption prediction (Section B). 

- _(iii)_ Additional data analysis of our VidChapters-7M dataset (Section C). 

- _(iv)_ Additional implementation details (Section D). 

- _(v)_ Video chapter generation results split by language (Section E). 

- _(vi)_ A datasheet [28] for VidChapters-7M (Section F). Note that in this datasheet, the hosting, licensing, and maintenance plan of VidChapters-7M is presented. 

Note that our code, models and the VidChapters-7M dataset can be found on our website [1]. 

## **A Additional examples from VidChapters-7M** 

In Figure 4, we provide additional examples that complement Figure 1. These examples illustrate the diversity of the data in VidChapters-7M, e.g., our dataset includes review videos, cooking videos, clothing fitting videos, ASMR videos, and videos of conversations. These examples also show the multi-modal nature of the chapter data. Indeed, chapters depict visual events (e.g., the mini chicken burgers that appear in the second video), conversations (see the last video), or events in the raw audio (e.g., the sound of the crinkly plastic bag in the penultimate video) in various scenarios. 

## **B Qualitative examples of video chapter generation and dense video caption prediction** 

We present qualitative results for video chapter generation and dense video captioning in Figures 5 and 6. Compared with the speech-only model, a key advantage of the speech+visual video chapter generation model is that it can generalize to videos that do not have ASR input, as shown in the first example of Figure 5. Compared with the visual-only variant, the multi-modal model can also benefit from speech cues, as seen in the second example in Figure 5. Moreover, we observe that the dense video captioning model pretrained on VidChapters-7M is more accurate and hallucinates less than the variant not pretrained on VidChapters-7M, see Figure 6. 

## **C Additional data analysis of VidChapters-7M** 

We here complement the analysis of the data in VidChapters-7M provided in Section 3.3. In Figure 7, we show a histogram of the most common chapter titles and word clouds[1] of the chapters titles and ASR content in VidChapters-7M. A few generic chapter titles that outline the structure of the video (e.g., _Intro_ , _Introduction_ , _Outro_ , _Conclusion_ and _Start_ ) appear more than 10K times. Besides, we notice that many videos include chapters about _Unboxing_ , _Review_ , or _Tips_ . This is consistent with the fact that there are many vlogs and ’Howto’ videos in VidChapters-7M. We also observe that the most common words in the ASR largely differ from the most common words in the chapter titles, which further shows the difference between these two types of data. To further measure the text-video alignment in the VidChapters-7M dataset, we compute the CLIP cosine similarity between chapter titles and their corresponding video frames and plot the resulting distribution in Figure 8. The average similarity score is 54.6%, and less than 1% of the chapters have a visual-text similarity score below 30%. These statistics demonstrate a good video-text alignment in the VidChapters-7M dataset. 

## **D Additional implementation details** 

In this Section, we present implementation details that complement the information provided in Section 4. We discuss implementation details of our tagging protocol for ethical considerations in 

> 1 To generate the word clouds, we used `https://github.com/amueller/word_cloud` . 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_VidChapters-7M_Video_Chapters_at_Scale_images/nips_2023_VidChapters-7M_Video_Chapters_at_Scale.pdf-0019-00.png)


**----- Start of picture text -----**<br>
1. Intro 2. Unboxing 3. Product Tour Installation  … How does this solve your<br>guide NBN issues<br>0:00 2:13 3:32 5:57 14:06<br>Nutella crepes Mini meatball pasta Caramel Bread  Chicken spread  … Mini chicken burgers<br>popcorns sandwich<br>0:17 1:57 5:13 7:00 11:23<br>intro athleisure leggings/sports bra jackets fitness gear<br>0:00 1:55 6:19 11:23 13:34<br>Examining  Drinking a Glass of Tea Onboard Camera/Tripod  Crinkly Plastic Bag Button Pressing …<br>a List Sounds<br>0:00 0:34 3:04 5:56 9:21<br>For me, working with  obviously I'm not as  … How can I turn that<br>… yeah yeah oh it's  the tennis players,  active as I was before  from like a negative of  I've gone back to the<br>just easier for them to like do it on their own  getting to record their  … missing my sport and  basics. Like bare bones,<br>time um and then if  stuff has been one of  not being able to train  basics of let's get my<br>the biggest  I find is that my hips  in the gym and then flip  body to move with just<br>they need me they just send me videos of it … improvements to my  have been getting pretty  that around into a  his body…<br>coaching. … tight… positive situation …<br>How work has changed  Benefits of recording  Hip mobility Turning being in quarantine  Training fundamentals  …<br>during quarantine your training to a positive thing and body weight training<br>1:33 1:46 2:54 4:38 6:40<br>**----- End of picture text -----**<br>


Figure 4: **Additional examples of videos with user-annotated chapters in VidChapters-7M:** Chapters depict visual events (e.g., the mini chicken burgers that appear in the second video), conversations (see the last video), or events in the raw audio (e.g., the sound of the crinkly plastic bag in the penultimate video) in various scenarios. 

19 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_VidChapters-7M_Video_Chapters_at_Scale_images/nips_2023_VidChapters-7M_Video_Chapters_at_Scale.pdf-0020-00.png)


**----- Start of picture text -----**<br>
Input  Ø Ø Ø Ø Ø Ø Ø Ø<br>Speech<br>Input<br>Frames<br>Ground Introduction Game overview Final thoughts<br>-Truth<br>Vid2Seq (HTM What’s in the box. What’s in the box. … … … … … … What’s in the box.<br>+VC, no<br>vision)<br>Vid2Seq  Introduction. Game overview. Final thoughts.<br>(HTM<br>+VC)<br>Input  If you are looking for the best Nike running  Number 1. Most Popular. Zoom  Unfortunately, the new thin mesh upper has  Number 2. Nike Men's Running Shoes.  The  Number 3. On Women's CloudFlyer  In order to reduce over pronation, the shoe  Number 4. Nike Downshifter Men's 7  Number 5. Nike Men's Trail Running Shoes.<br>Speech shoes, here is a  Pegasus Turbo 2.  A  issues. Its minimal heel  new trend in stability  Running Shoes.  features firmer medial  Running Shoe. The  Made of a breathable<br>collection you have got  souped-up, speed- support means you  shoes is less  Provide your foot with  elements that redirect  Downshifter 7  mesh upper and a<br>to see. oriented version of the  have to cinch the laces  interference, and the  the cushion it deserves  force to the lateral side  Running Shoes from  sturdy EVA sole, these<br>Pegasus, the Peg  down for a secure fit,  Infinity Run follows  with the On  of the runner's foot.  Nike are designed to  Quest running shoes<br>Turbo keeps the  but the tongue isn't  that principle by  CloudFlyer. Utilizing  Paired with an even  be lightweight, sturdy  from Nike should<br>winning combo of  thick or long enough to  providing comfort,  plush clouds built from  stiffer speed board, the  and durable, all the  pretty much be a staple<br>Zoomex and React  prevent the laces from  support, and a smooth  zero-gravity foam and  shoe promotes a  while providing you  in every man's shoe<br>foams found in the  causing irritation. ride without messing  a wider CloudTek  quicker heel-to-toe  with optimum  closet. Fly-wire cables<br>first version. up your natural  platform, this daily  transfer that helps get  performance, making  offer your feet a secure<br>movement. […] trainer provides  the runner through  them a worthy  fit, while the soft yet<br>supreme cushioning in  their pronated phase. investment. […] responsive foam is<br>a more stable package.  supportive […]<br>Input<br>Frames<br>Ground 1. Zoom Pegasus Turbo 2 2. Nike Men's  3. ON Women's Cloudflyer Running Shoes 4. Nike Downshifter  5. Nike Men's Trail<br>-Truth Running Shoes Men's 7 Running Shoe Running Shoes<br>Vid2Seq  1. nike nexus running shoe. 2. nike nexus running  3. nike nexus running shoe. 4. nike nexus  5. nike nexus running<br>(HTM shoe. running shoe. shoe.<br>+VC, no<br>speech)<br>Vid2Seq  1. zoom pegasus turbo 2. 2. nike men’s running  3. on women's cloudflyer running shoes. 4. nike downshifter  5. nike men's trail<br>(HTM shoes. men's 7 running shoe. running shoes.<br>+VC)<br>**----- End of picture text -----**<br>


Figure 5: **Examples of video chapter generation using the Vid2Seq model with different input modalities compared with ground-truth on the test set of VidChapters-7M** . The first example shows that the Vid2Seq variant with both speech and visual modalities "Vid2Seq (HTM+VC)" can predict the structure of the input video without the ASR input, unlike the Vid2Seq speech-only variant "Vid2Seq (HTM+VC, no vision)". The second example shows that the Vid2Seq variant with both speech and visual modalities "Vid2Seq (HTM +VC)" can effectively leverage speech cues to detect the names of the depicted and discussed shoes, unlike the Vid2Seq visual-only variant "Vid2Seq (HTM+VC, no speech)". 

Section D.1, models used for video chapter generation and dense video captioning in Section D.2, models used for video chapter generation with ground-truth boundaries in Section D.3, and models used for video chapter grounding in Section D.4. 

## **D.1 Tagging for ethical considerations** 

In Section 3.3, we explained how we tag videos for ethical considerations. We here give additional details about this procedure. For the NSFW visual content detector [78], we compute the NSFW score at every frame (at 1 FPS) and tag videos with an average score above 0.5. For the toxic content detection model [31], we compute the toxicity score at every chapter title / ASR sentence and tag videos where the chapter titles / ASR have an average toxicity score above 0.5. 

## **D.2 Video chapter generation and dense video captioning** 

**LLaMA [93].** We use the following prompt: `Summarize the following speech transcript in a chapter title. Transcript: <ASR> Chapter title:` where the ASR is the concatenation of all speech sentences spoken during a given video segment. 

**BLIP-2 [51].** We use the following prompt: `Summarize the image in a chapter title. Chapter title:` , and use the middle frame of the predicted video segment. 

20 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_VidChapters-7M_Video_Chapters_at_Scale_images/nips_2023_VidChapters-7M_Video_Chapters_at_Scale.pdf-0021-00.png)


**----- Start of picture text -----**<br>
Okay, so let's get  We'll add in two  Okay, now that the pan  Now that the onions  And then we'll add in  So what we're hoping  Whatever you want to<br>started. So I'm going to  teaspoons of olive oil  has been heated over  are nice and sauteed  our sweet potatoes,  for in that 15 minutes  call it, it smells<br>heat our pan over  to coat the pan. We'll  medium heat, let's add  and smelling delicious,  and then about a  is that these sweet  delicious, it looks<br>Input  medium heat. Once again, keep the  let that heat up, then we're going to add the  in the onions and let’s saute these for about  let's add in our red pepper. tablespoon of water. potatoes will become soft and ready to eat. pretty cool. Add this as a side dish to any<br>Speech temperature low  other ingredients. So  two minutes. Get them  protein source, so a<br>because we are using  come on over. soft, then we'll add in  steak, chicken, or have<br>olive oil. You don't want  the other ingredients. it in the morning with<br>to cook olive oil over  your eggs and you're<br>too high of heat. good to go.<br>Input<br>Frames<br>Ground Add some chopped  Add in red pepper and  Add a spoon of  Cook with lid  Season it with salt<br>-Truth white onions in a pan  sweet potatoes. water. on. and black pepper.<br>under medium heat.<br>Vid2Seq  Heat a pan with  Add 2 tbsp of olive  Add onions to the pan  Add red pepper  Add red pepper and water to the pan. Cook the sweet<br>(HTM) olive oil. oil to the pan. and saute. sweet potatoes and water to the  potatoes in the pan.<br>pan.<br>Vid2Seq  Heat a pan over  Add olive oil to the  Add onions to the  Add red pepper to  Add sweet potatoes  Cover the pan<br>(HTM medium heat. pan. pan. the pan. and water to the  with a lid.<br>+VC) pan.<br>When you're making  Next you want to peel,  Next comes a crucial  You want to salt the  After you bring them  out over very low heat  So now it's time to  Don't worry, it can<br>mashed potatoes, if  rinse, and most  step. You want to cook  water because the salt  to a boil, let them  until they really start  finally add your cream  look like soup at first,<br>you're using the wrong  important, cut your  these potatoes until  starts to disintegrate  simmer long and slow  falling apart, turning  or milk. You want to  but if you keep<br>Input  potato and you're throwing them in a food  potatoes into nice, even chunks. That way  they are falling apart. Next comes a crucial  the potatoes too. But the most important  until the edges start to fall off, and when you  white on the edges and practically turning into  make sure it's warm so it doesn't cool down  stirring, you'll see those potatoes just<br>Speech processor, you're doing  they'll all cook at the  step. You want to cook  thing is that you really  stick a knife in the  mashed potatoes right  your potatoes, and the  drink up all that cream.<br>it all wrong. The first  same rate, and you  these potatoes until  cook those potatoes  middle, it just all falls  then and there in the  most important thing is  The final step for<br>mistake people tend to  won't get any weird  they are falling apart  about 25 minutes. apart. Drain the  pot. To finish the job,  you want to make sure  making great mashed<br>make is they pick the  little nasty hard bits in  more than if you just  potatoes, and then put  you can use a hand  you use lots. […] potatoes is to add a<br>wrong potato. […] your potato. put them in boiling  them right back into  masher or you can use  few fresh herbs, chives<br>water. that hot pot and start to  a ricer for super  are nice […]<br>dry them smooth, silky potatoes.<br>Input<br>Frames<br>Ground Peel and cut  Put in cold  Drain and  Mash the  Add milk and stir the  Season the potatoes<br>-Truth potatoes  water and cook  dry the  potatoes  potatoes. with some chopped<br>into chunks. to a boil and  potatoes. well with  parsley leaves.<br>salt the water. a hand<br>masher.<br>Vid2Seq  Peel rinse and cut  Boil the potatoes in  Boil the potatoes in  Drain the  Add cream and milk to  Add chives<br>(HTM) the potatoes. water. water. potatoes and dry them out  the potatoes. parsley and butter to the<br>in a pot. potatoes.<br>Vid2Seq  Peel rinse and cut the potatoes. Boil the potatoes in  Drain the potatoes and  Add milk and mash the potatoes. Add fresh herbs<br>(HTM water and salt the  dry them out. parsley and butter<br>+VC) water. to the mashed<br>potatoes.<br>**----- End of picture text -----**<br>


Figure 6: **Examples of dense event captioning of the Vid2Seq model pretrained on VidChapters7M (vs. not pre-trained), compared with ground-truth, on the validation set of YouCook2.** We find that the model pretrained on VidChapters-7M "Vid2Seq (HTM+VC)" is more accurate and less prone to hallucination. For instance, in the first example (top), the non-VC-pretrained model "Vid2Seq (HTM)" predicts "Add red pepper sweet potatoes and water to the pan." before the sweet potatoes are actually thrown into the pan. In the second example (bottom), the non-VC-pretrained model "Vid2Seq (HTM)" predicts the event "Boil the potatoes in water." twice and predicts the event "Add chives parsley and butter to the potatoes." before it actually happens. The VC-pretrained model "Vid2Seq (HTM+VC)" produces more accurate predictions. 

**PDVC [101].** We use PDVC’s official codebase. PDVC includes a caption decoder that relies on dataset-specific word vocabularies. To adapt PDVC to VidChapters-7M/YouCook2/ViTT, we construct a vocabulary made with all words that appear at least 50/2/3 times in the dataset (33,598/3,815/1,607 words). For transfer learning from VidChapters-7M to YouCook2/ViTT, we initialize the downstream dataset-specific word embedding layer with the weights of the corresponding word embedding in the pretrained model. We subsample or pad the sequence of frames to 100 frames. For all datasets, we use 100 queries and train with a constant learning rate of 5 _e[−]_[5] , weight decay 1 _e[−]_[4] and batch size 1 on an NVIDIA V100 32GB (as the official codebase is not compatible with higher batch sizes or multi-gpu training) . We train on VidChapters-7M/YouCook2/ViTT for 5/30/30 epochs. The training on VidChapters-7M lasts about a week. 

**Vid2Seq [114].** We reimplement Vid2Seq (originally released in Jax) in PyTorch. For initialization, we use the T5-Base language model pretrained on the C4 text corpus [74]. Vid2Seq is originally pretrained on YT-Temporal-1B [118] using a generative and denoising objective in the speech sequence. Due to computational limitations, we instead pretrain Vid2Seq on the smaller HowTo100M dataset [64] with the same objectives. Then we train Vid2Seq on VidChapters-7M with the next 

21 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_VidChapters-7M_Video_Chapters_at_Scale_images/nips_2023_VidChapters-7M_Video_Chapters_at_Scale.pdf-0022-00.png)


**----- Start of picture text -----**<br>
10 [5]<br>10 [4]<br>10 [3]<br>Chapter title<br>(a) Most common chapter titles<br>0<br>200<br>400<br>600<br>0 500 1000 1500 2000<br>(b) Word clouds of chapter titles.<br>0<br>200<br>400<br>600<br>0 500 1000 1500 2000<br>(c) Word clouds of ASR.<br>Count (log scale)<br>Intro Introduction Outro Conclusion Final Thoughts Start intro Unboxing Overview Final thoughts Summary Rest Welcome INTRO Things You'll Need Game overview Tips & Warnings Closing Recap Performance Q&A Price Review Design outro Interior Wrap Up Features Opening Bloopers Round 2 Verdict Materials Round 1 Wrap up start Ending Round 3 End Demonstration Intro & Summary Specs Setup 1 2 3 Credits 4 Remove old filter Filter location<br>**----- End of picture text -----**<br>


Figure 7: **Additional statistics of the VidChapters-7M dataset.** 

22 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_VidChapters-7M_Video_Chapters_at_Scale_images/nips_2023_VidChapters-7M_Video_Chapters_at_Scale.pdf-0023-00.png)


**----- Start of picture text -----**<br>
1e6<br>1.6<br>1.4<br>1.2<br>1.0<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>Average visual-text similarity<br>Count<br>0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95<br>**----- End of picture text -----**<br>


Figure 8: **Average visual-text similarity between chapter titles and the corresponding video frames as measured by CLIP cosine similarity (rescaled between 0 and 100) in VidChapters-7M.** 

token prediction objective in the chapter sequence and the denoising objective in the speech sequence. Finetuning on YouCook2/ViTT is done with the next token prediction objective in the dense video captioning sequence and the denoising objective in the speech sequence. We subsample or zero-pad the sequence of frames to 100 frames. The text encoder and decoder sequence are truncated or padded to 1000 and 256 tokens, respectively. For all datasets, we use a learning rate of 3 _e[−]_[4] warmed up linearly (from 0) for the first 10% of iterations and following a cosine decay (down to 0) for the remaining 90%. We train for 6/10/40/20 epochs on HowTo100M/VidChapters-7M/YouCook2/ViTT. We use a batch size of 64 videos split on 8 NVIDIA A100 80GB for HowTo100M/VidChapters-7M, and 16 videos split on 8 NVIDIA V100 32GB for YouCook2/ViTT. The training on HowTo100M or VidChapters-7M takes about 2 days. 

## **D.3 Video chapter generation with ground-truth boundaries** 

## **LLaMA [93] and BLIP-2 [51].** See Section D.2. 

**Vid2Seq [114].** To adapt the model pretrained on HowTo100M (see Section D.2) to video chapter generation with ground-truth boundaries, we remove the model weights corresponding to the time tokens (in the token embedding layers and the token prediction layer). We train for 20 epochs on VidChapters-7M using the next token prediction objective in the sequence of tokens corresponding to a single chapter title. We construct training batches by sampling a chapter title with its associated video clip at each iteration (i.e., an epoch corresponds to seeing one chapter title for all videos). The text encoder and decoder sequence are truncated or padded to 256 and 32 tokens, respectively. We use a learning rate of 3 _e[−]_[4] warmed up linearly (from 0) for the first 10% of iterations and following a cosine decay (down to 0) for the remaining 90%. We use a batch size of 512 videos split on 8 NVIDIA A100 80GB for VidChapters-7M. The training takes about a day. 

## **D.4 Video chapter grounding** 

**Moment-DETR [45].** We use Moment-DETR’s official codebase. We train with the AdamW optimizer [60], a constant learning rate of 3 _e[−]_[4] , and a batch size of 256 videos split on 8 NVIDIA A100 80GB. We use a FPS of 1/3 and subsample or zero-pad the sequence of frames to 1200 frames. We use a maximum number of text query tokens of 77. We train for 50 epochs on VidChapters-7M, where an epoch corresponds to seeing one chapter title for all videos, which takes about 2 days. 

## **E Video chapter generation results split by language** 

We report video chapter generation results on the VidChapters-7M dataset split by language for both English and German in Tables 9 and 10, respectively. We find that training on VidChapters-7M is beneficial for both languages. Interestingly, pretraining on HowTo100M (which is a dataset in English) improves results on English as well as German. We also observe that the quantitative results in German are lower than in English. Finally, we report results of the Vid2Seq model with the multi-lingual language model mT5 [109] pretrained on the multi-lingual dataset mC4 [109]. We 

23 

|Method|Modalities<br>PretrainingData<br>Finetuned|S<br>B1<br>B2 B3 B4<br>C<br>M<br>RL|
|---|---|---|
|Text tiling [32] + Random<br>Text tiling [32] + LLaMA [93]<br>Shot detect[92]+ BLIP-2[51]|Speech<br>_∅_<br>✗<br>Speech<br>Text mixture<br>✗<br>Visual<br>129M image-texts<br>✗|0.5<br>0.8<br>0.2 0.1 0.0<br>0.9<br>0.8<br>0.7<br>0.3<br>0.5<br>0.2 0.1 0.0<br>0.5<br>0.4<br>0.4<br>1.3<br>1.5<br>0.7 0.4 0.2<br>4.7<br>1.4<br>1.6|
|PDVC [101]<br>Vid2Seq [114]<br>Vid2Seq [114] w/ mT5<br>Vid2Seq [114]|Visual<br>_∅_<br>✓<br>Speech+Visual<br>C4<br>✓<br>Speech+Visual<br>mC4<br>✓<br>Speech+Visual C4 + HowTo100M<br>✓|6.6<br>9.0<br>3.8 1.5 1.0 36.0 9.1 11.0<br>10.8 10.3 7.6 4.9 3.4 54.8 9.1 11.9<br>10.4<br>9.9<br>7.2 4.7 3.3 52.0 8.7 11.3<br>**11.5 11.1 8.1 5.1 3.6 58.8 9.7 12.8**|



Table 9: **Video chapter generation (global metrics) on the VidChapters-7M test set restricted to videos with English chapter titles and ASR.** Here, finetuned refers to finetuning on the VidChapters7M train set, and speech refers to transcribed speech (ASR). 

|Method|Modalities<br>PretrainingData<br>Finetuned|S<br>B1<br>B2 B3 B4<br>C<br>M<br>RL|
|---|---|---|
|Text tiling [32] + Random<br>Text tiling [32] + LLaMA [93]<br>Shot detect[92]+ BLIP-2[51]|Speech<br>_∅_<br>✗<br>Speech<br>Text mixture<br>✗<br>Visual<br>129M image-texts<br>✗|0.6<br>1.7<br>1.3 1.3 1.1 12.8<br>1.5<br>1.6<br>0.1<br>0.3<br>0.2 0.0 0.0<br>0.0<br>0.2<br>0.2<br>0.6<br>0.4<br>0.2 0.0 0.0<br>1.3<br>0.6<br>0.5|
|PDVC [101]<br>Vid2Seq [114]<br>Vid2Seq [114] w/ mT5<br>Vid2Seq [114]|Visual<br>_∅_<br>✓<br>Speech+Visual<br>C4<br>✓<br>Speech+Visual<br>mC4<br>✓<br>Speech+Visual C4 + HowTo100M<br>✓|5.4<br>**11.6** 0.0 0.0 0.0 29.4 **12.4 14.9**<br>9.1<br>8.4<br>5.2 1.0 0.9 34.1<br>6.1<br>10.1<br>8.8<br>8.1<br>5.9 1.7 1.8 38.4<br>6.1<br>10.1<br>**10.9**<br>9.6<br>**5.4 1.7 1.7 43.2**<br>8.1<br>8.1|



Table 10: **Video chapter generation (global metrics) on the VidChapters-7M test set restricted to videos with German chapter titles and ASR.** Here, finetuned refers to finetuning on the VidChapters7M train set, and speech refers to transcribed speech (ASR). 

find that this variant performs a bit worse on English but slightly better on German compared to the Vid2Seq variant based on T5 pretrained on the C4 corpus. 

## **F Datasheet for VidChapters-7M** 

Datasheets for datasets introduced by Gebru et al. [28] serve as a medium of communication between the creators and users of a dataset. They effectively consolidate the motivation, creation process, composition, and intended uses of a dataset as a series of questions and answers. In this Section, we provide a datasheet for the VidChapters-7M dataset. 

## **Motivation** 

- Q1. **For what purpose was the dataset created?** _Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description._ 

- The VidChapters-7M dataset was created to explore the task of video chapter generation, 

- which enables users to quickly navigate to the information of their interest. 

- Q2. **Who created this dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?** 

- Five researchers have created VidChapters-7M: Antoine Yang (Inria and DI ENS), Arsha 

- Nagrani (VGG, University of Oxford), Ivan Laptev (Inria and DI ENS), Josef Sivic (CIIRC CTU) and Cordelia Schmid (Inria and DI ENS). 

- Q3. **Who funded the creation of the dataset?** _If there is an associated grant, please provide the name of the grantor and the grant name and number._ 

- We collected VidChapters-7M without any monetary costs, since no part of our dataset requires 

- annotations from crowd workers or contractors. This research work was funded by Antoine Yang’s Google PhD fellowship, the French government under management of Agence Nationale de la Recherche as part of the "Investissements d’avenir" program, reference ANR-19-P3IA0001 (PRAIRIE 3IA Institute), the Louis Vuitton ENS Chair on Artificial Intelligence, the European Regional Development Fund under project IMPACT (reg. no. CZ.02.1.01/0.0/0.0/15 003/0000468). However, note that this article solely reflects the opinions and conclusions of its authors and not of its funders. 

24 

## Q4. **Any other comments?** 

- No. 

## **Composition** 

- Q5. **What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?** _Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description._ 

- Each instance in VidChapters-7M represents a YouTube video. 

- Q6. **How many instances are there in total (of each type, if appropriate)?** 

- There are 817K instances in VidChapters-7M. 

- Q7. **Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?** _If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable)._ 

- VidChapters-7M is a small sample drawn from all the data uploaded to YouTube. Millions of 

- videos are uploaded on YouTube every day. We started from a subset of 92 million YouTube video candidates, which consists of videos recommended in videos from the YT-Temporal180M dataset [117]. We selected the videos from this subset (817K instances) that contain user-annotated chapters. Hence, VidChapters-7M data does not fully represent YouTube. 

- Q8. **What data does each instance consist of? “Raw” data (e.g., unprocessed text or images) or features?** _In either case, please provide a description._ 

- Each instance in VidChapters-7M consists of four metadata fields: 

   - `"video_id"` : Unique alphanumeric ID of the video (assigned by YouTube). 

   - `"url"` : Static URL for downloading the video, e.g., `https://www.youtube.com/watch?v=<video_id>` . 

   - `"asr"` : ASR transcripts aligned over time. 

   - `"chapters"` : Chapters aligned over time. 

- Q9. **Is there a label or target associated with each instance?** _If so, please provide a description._ – We use the chapters as labels in this work, though it might be also possible to use auxiliary 

- information (like video titles or tags). 

- Q10. **Is any information missing from individual instances?** _If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text._ 

- No and yes. No, because all the metadata fields for every instance are filled with valid values. 

- Yes, because the `"url"` for some instances may not retrieve the underlying video. This may happen if the YouTube user (author) removes the video from YouTube. Such deletions reduce our dataset size over time, however, video deletions are rare. 

- Q11. **Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?** _If so, please describe how these relationships are made explicit._ 

- Relationships between individual instances (e.g., videos made by the same creator) are not 

- made explicit in our work, though this is a possibility for future work. 

- Q12. **Are there recommended data splits (e.g., training, development/validation, testing)?** _If so, please provide a description of these splits, explaining the rationale behind them._ 

- We randomly split our data into training, validation, and testing sets. The training, validation, 

- and testing sets are meant for training, development, and evaluation, respectively. 

- Q13. **Are there any errors, sources of noise, or redundancies in the dataset?** _If so, please provide a description._ 

- VidChapters-7M is inherently noisy since YouTubers are free to write the chapters that they 

- want. 

25 

- Q14. **Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?** _If it links to or relies on external resources,_ 

   - _(a) Are there guarantees that they will exist, and remain constant, over time?_ 

   - _(b) Are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created)?_ 

   - _(c) Are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate._ 

- We do not distribute videos of our dataset to respect YouTube user privacy and to limit our 

- storage budget. Instead, we provide video URLs ( `"url"` , Q8) that point to videos hosted on YouTube servers. In response to sub-questions: 

   - (a) These video servers ensure stable access unless the YouTube user deletes their video. (b) Yes, YouTube archives all the metadata of submitted videos. For videos, YouTube only archives the URL and not the media content, giving full control of accessibility to the users. 

   - (c) All video URLs are freely accessible. It is unlikely for video servers to restrict access in the future, given their free accessibility over the past decade. 

- Q15. **Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals non-public communications)?** _If so, please provide a description._ 

- No, the videos included in VidChapters-7M do not cover topics that may be considered 

- confidential. All videos were publicly shared on YouTube prior to inclusion in VidChapters-7M. 

- Q16. **Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?** _If so, please describe why._ 

- The scale of VidChapters-7M means that we are unable to manually verify the contents of all 

- videos and chapters. However, YouTube removes videos that contain offensive content or do not follow their community guidelines. Furthermore, we employed additional mitigation techniques on VidChapters-7M: 

   - (a) We tagged all instances whose video frames were predicted as NSFW by an off-the-shelf detector [78]. 

   - (b) We tagged all instances whose chapter titles or speech transcripts were predicted as toxic by a language model [31]. 

- Q17. **Does the dataset relate to people?** _If not, you may skip remaining questions in this section._ – The dataset pertains to people in that people upload videos to YouTube and write descriptions 

- that include chapter annotations. Furthermore, most videos in VidChapters-7M have people speaking and/or appearing. 

- Q18. **Does the dataset identify any subpopulations (e.g., by age, gender)?** _If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset._ 

- VidChapters-7M does not explicitly identify any subpopulations. Since most videos contain 

- people and chapters are free-form natural language written by YouTube users, it is possible that some chapters may identify people appearing in individual videos as part of a subpopulation. 

- Q19. **Is it possible to identify one or more natural persons, either directly or indirectly (i.e., in combination with other data) from the dataset?** _If so, please describe how._ 

- Yes, our data includes celebrities, or other YouTube-famous people. All of the videos that 

- we use are of publicly available data, following the Terms of Service ( `https://www.youtube. com/static?template=terms` ) that users agreed to when uploading to YouTube. 

- Q20. **Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?** _If so, please provide a description._ 

- This is highly unlikely, as YouTube removes videos that contain offensive content or do not 

- follow their community guidelines. 

26 

- Q21. **Any other comments?** 

- No. 

## **Collection Process** 

- Q22. **How was the data associated with each instance acquired?** _Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how._ 

- See Q7 for an explanation of how the candidate video IDs were chosen. These video IDs 

- were provided by the YT-Temporal-180M dataset providers [117] and collected via the YouTube API. The `"video_id"` and `"URL"` are directly observable. The `"chapters"` are extracted from the YouTube description which is directly observable. The `"asr"` is obtained by applying the Whisper-Large-V2 model [73] to the directly observable audio from the video. We found this model to provide higher-quality transcriptions compared to the YouTube API on several data samples from VidChapters-7M. 

- Q23. **What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?** _How were these mechanisms or procedures validated?_ 

- We collected all data using compute resources provided by IDRIS, under the allocation 

- 2023-A0131011670 made by GENCI. The code for querying APIs, extracting ASR, and filtering data are implemented in Python. The code was validated by checking several data samples from VidChapters-7M. 

- Q24. **If the dataset is a sample from a larger set, what was the sampling strategy?** – See Q7. 

- Q25. **Who was involved in data collection process (e.g., students, crowd-workers, contractors) and how were they compensated (e.g., how much were crowd-workers paid)?** 

- Our data collection pipeline is fully automatic and does not require any human annotators. 

- YouTube users have uploaded videos whose metadata is a part of VidChapters-7M – we did not directly interact with these users. 

- Q26. **Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)?** _If not, please provide a description of the timeframe._ 

- VidChapters-7M contains videos that were uploaded to YouTube between 2005–2022. We 

- collected all data in early 2023, which we used to conduct experiments for our NeurIPS 2023 submission. 

- Q27. **Were any ethical review processes conducted (e.g., by an institutional review board)?** _If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation._ 

- We did not conduct a formal ethical review process via institutional review boards. However, 

- as described in Section 3.3 and Q16 we employed several filtering mechanisms to tag instances that could be problematic. 

- Q28. **Does the dataset relate to people?** _If not, you may skip remaining questions in this section._ – Yes, see Q17. 

- Q29. **Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?** 

- We collected data submitted by YouTube users indirectly through the YouTube API. However, 

- users agree with YouTube’s Terms of Service regarding the redistribution of their data by YouTube. 

27 

- Q30. **Were the individuals in question notified about the data collection?** _If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself._ 

- No. YouTube users are not required to share their personal contact information (email, phone 

- numbers, etc.). Hence, the only way to notify the authors of VidChapters-7M videos is by commenting on their videos. This is practically difficult to do manually and will be classified as spam and blocked by YouTube if attempted to programmatically write a templated comment to millions of users. 

- Q31. **Did the individuals in question consent to the collection and use of their data?** _If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented._ 

- Users did not explicitly consent to the use of their data in our dataset. However, by uploading 

- their data on YouTube, they consent that it would appear on the YouTube plaform and will be accessible via the official YouTube API (which we use to collect VidChapters-7M). 

- Q32. **If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?** _If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate)._ 

- Users have full control over the presence of their data in our dataset. If users wish to revoke 

- their consent, they can delete the underlying YouTube video – it will be automatically removed from VidChapters-7M since we distributed videos as URLs. Moreover, we provide an opt-out request form on our dataset website for anybody to request removal of an individual instance if it is potentially harmful (e.g. NSFW, violates privacy, harmful stereotypes, etc.). 

- Q33. **Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?** _If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation._ 

- No. 

- Q34. **Any other comments?** 

- No. 

## **Preprocessing, Cleaning, and/or Labeling** 

- Q35. **Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?** _If so, please provide a description. If not, you may skip the remainder of the questions in this section._ 

- We converted chapter timestamps in HH:MM:SS format to seconds. Refer to Section 3.1 for 

- more details. We also extracted speech transcripts and visual features (see Section 3.2). Finally, we tagged some instances with a focus on ethical considerations, see Q16 for more details. 

- Q36. **Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?** _If so, please provide a link or other access point to the_ 

- _“raw” data._ 

- Yes, the raw descriptions from which chapters are extracted are also released on the dataset 

- website [1]. 

- Q37. **Is the software used to preprocess/clean/label the instances available?** _If so, please provide a link or other access point._ 

- Yes, the data preprocessing code is open-sourced and accessible from the dataset website [1]. 

- Q38. **Any other comments?** 

- 

- No. 

28 

## **Uses** 

- Q39. **Has the dataset been used for any tasks already?** _If so, please provide a description._ 

- We have used our dataset to train deep neural networks that perform video chapter generation, 

- and that can be transferred to dense video captioning tasks (see Sections 4.1 and 4.4). We also trained models for video chapter generation with ground-truth boundaries and video chapter grounding (see Sections 4.2 and 4.3). 

- Q40. **Is there a repository that links to any or all papers or systems that use the dataset?** _If so, please provide a link or other access point._ 

- We do not maintain such a repository. However, citation trackers like Google Scholar and 

- Semantic Scholar would list all future works that cite our dataset. 

- Q41. **What (other) tasks could the dataset be used for?** 

- We anticipate that the dataset could be used for a variety of video-and-language tasks, such as 

- text-to-video retrieval. 

- Q42. **Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?** _For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other undesirable harms (e.g., financial harms, legal risks) If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?_ 

- This is very difficult to anticipate. Future users of our dataset should be aware of YouTube’s 

- user demographics which might subtly influence the types of videos, languages, and ideas that are present in the dataset. Also, note that our dataset is mainly composed of English videos, hence models trained on this dataset might perform worse on videos in other languages. 

- Q43. **Are there any tasks for which the dataset should not be used?** _If so, please provide a description._ 

- Broadly speaking, our dataset should only be used for non-commercial academic research. Our 

- dataset should not be used for any tasks that involve identifying features related to people (facial recognition, gender, age, ethnicity identification, etc.) or making decisions that impact people (mortgages, job applications, criminal sentences; or moderation decisions about user-uploaded data that could result in bans from a website). Any commercial and for-profit uses of our dataset are restricted – it should not be used to train models that will be deployed in production systems as part of a product offered by businesses or government agencies. 

## Q44. **Any other comments?** 

- No. 

## **Distribution** 

- Q45. **Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?** _If so, please provide a description._ 

- 

   - Yes, our dataset is publicly available. 

- Q46. **How will the dataset will be distributed (e.g., tarball on website, API, GitHub)** _Does the dataset have a digital object identifier (DOI)?_ 

- We distribute our dataset as JSON/PICKLE files containing annotations. Users will have to 

- download the videos by themselves by using our data collection code. All uses of VidChapters7M should cite the paper as the reference. 

- Q47. **When will the dataset be distributed?** 

- 

   - The dataset is publicly available as of September 2023. 

- Q48. **Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?** _If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions._ 

29 

- Uses of our dataset are subject to YouTube API terms ( `https://www.youtube.com/` 

- `static?template=terms` ). The data and code are released with an MIT license. 

- Q49. **Have any third parties imposed IP-based or other restrictions on the data associated with the instances?** _If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions._ 

- The videos corresponding to our instances are legally owned by YouTube users. Our dataset 

- users can download them from the URLs we provide in annotation files, but redistributing videos for commercial use is prohibited. 

- Q50. **Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?** _If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation._ 

- 

   - No. 

- Q51. **Any other comments?** 

- 

- No. 

## **Maintenance** 

- Q52. **Who will be supporting/hosting/maintaining the dataset?** 

- The authors will maintain the dataset. The dataset is hosted using Inria servers and Google 

- Drive service. All the information about the dataset, including links to the paper, code, and future announcements will be accessible at the dataset website [1]. 

- Q53. **How can the owner/curator/manager of the dataset be contacted (e.g., email address)?** – The contact emails of authors are available on the dataset website [1]. 

- Q54. **Is there an erratum?** _If so, please provide a link or other access point._ 

- There is no erratum for our initial release. We will version all errata as future releases (Q55) 

- and document them on the dataset website [1]. 

- Q55. **Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?** _If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?_ 

- We will update our dataset once every year and announce it on the dataset website [1]. These 

- future versions would remove instances that were requested to be removed via the opt-out form (Q32). 

- Q56. **If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?** _If so, please describe these limits and explain how they will be enforced._ 

- Rather than directly distributing videos, we distribute URLs that point to the original videos 

- uploaded by YouTube users. This means that users retain full control of their data – any post deleted from YouTube will be automatically removed from VidChapters-7M (see also Q10, Q14, Q31). 

- Q57. **Will older versions of the dataset continue to be supported/hosted/maintained?** _If so, please describe how. If not, please describe how its obsolescence will be communicated to users._ 

- A new version release of VidChapters-7M will automatically deprecate its previous version. 

- We will only support and maintain the latest version at all times. Deprecated versions will remain accessible on the dataset website for a few weeks, after which they will be removed. We decided to deprecate old versions to ensure that any data that is requested to be removed (Q32) will be no longer accessible in future versions. 

30 

- Q58. **If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?** _If so, please provide a description. Will these contributions be verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description._ 

- Anyone can extend VidChapters-7M by using our data collection code (linked on our 

- website [1]). We are open to accepting extensions via personal communication with contributors. Otherwise, our code and data licenses allow others to create independent derivative works (with proper attribution) as long as they are used for non-commercial academic research. 

31 

