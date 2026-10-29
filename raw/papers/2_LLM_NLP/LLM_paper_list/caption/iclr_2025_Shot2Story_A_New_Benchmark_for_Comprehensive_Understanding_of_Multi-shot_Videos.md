Published as a conference paper at ICLR 2025 

## - SHOT2STORY: A NEW BENCHMARK FOR COMPREHEN - SIVE UNDERSTANDING OF MULTI SHOT VIDEOS 

**Mingfei Han**[2] _[,]_[3] _[,]_[5] _[†∗]_ **, Linjie Yang**[1] _[†]_ **, Xiaojun Chang**[3] _[,]_[4] **, Lina Yao**[5] **, Heng Wang**[1] 1Bytedance Inc. 2ReLER Lab, AAII, UTS 3Department of Computer Vision, MBZUAI 4University of Science and Technology of China 5Data61, CSIRO https://github.com/bytedance/Shot2Story 

## ABSTRACT 

A short clip of video may contain progression of multiple events and an interesting story line. A human need to capture both the event in every shot and associate them together to understand the story behind it. In this work, we present a new multishot video understanding benchmark Shot2Story with detailed shot-level captions, comprehensive video summaries and question-answering pairs. To facilitate better semantic understanding of videos, we provide captions for both visual signals and human narrations. We design several distinct tasks including single-shot video captioning, multi-shot video summarization, and multi-shot video question answering. Preliminary experiments show some challenges to generate a long and comprehensive video summary for multi-shot videos. Nevertheless, the generated imperfect summaries can already achieve competitive performance on existing video understanding tasks such as video question-answering, promoting an underexplored setting of video understanding with detailed summaries. 

## 1 INTRODUCTION 

Video captioning is a long-standing video understanding task to facilitate open-world video analysis with the help of human-annotated captions. Since a video may contain multiple events, dense captioning benchmarks (Ego4D (Grauman et al., 2022), YouCook2 (Zhou et al., 2018), ActivityNetCaps (Krishna et al., 2017)) capture information of multiple events in videos ranging from 3-20 minutes. However, even within seconds, more than one event often occurs in daily videos such as news broadcasts, tutorial videos, and movies. Specifically, shot transition, which is a common technique to transfer from one event to another, or to switch the viewpoint of a single event, happens less than every 4s for average English movies after 2010 (Cutting et al., 2011). Although some existing captioning benchmarks (Xu et al., 2016; Krishna et al., 2017; Zhou et al., 2018) already use multi-shot videos, they often annotate the captions in a coarse-grained manner, either providing a holistic caption or asking annotators to subjectively choose the boundary of each event. To better accommodate the multi-shot formation of videos, we believe a new video benchmark with rich textual descriptions based on video shots is favored in the research community. 

On the other hand, multi-shot videos are often accompanied by rich narrations that relate to the different events happening in the video. A model needs to capture both the visual and audio signals to understand the underlying story. Specifically, narrations may contain key information that cannot be inferred from pure visual information only. See Figure 1, without the narration, a viewer is unable to capture the relationship between the man’s action and the avocado product in the first shot. 

In this work, we propose a new benchmark Shot2Story for audio-visual understanding of multi-shot videos. We collect a dataset of 42,958 short videos where the average number of shots in each video is 4.4. For each video shot, we annotate a detailed textual description for the video frames and another textual description for the human speech. We also leverage a state-of-the-art large language model (LLM) GPT-4 (OpenAI) to generate a long textual video summary from the annotated clip descriptions, which are further verified by human annotators. The summary includes additional details such as transitions of different shots, progression of multiple events, and mapping of the subject identities in different scenes. An example of one annotated video is shown in Figure 1. 

> _∗_ Work was done during an internship at Bytedance. _†_ Equal contribution. 

1 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Shot 0: 0-6.5s Shot 1: 6.5-10.5s Shot 2: 10.5-13.5s Shot 3: 13.5-15.7s<br>Caption:  It's a man in a kitchen chopping something with  Caption:  It's a close-up of a person  Caption:  It's a woman in a pink  Caption:  It's a bowl of guacamole<br>a knife. The man is wearing a T-shirt with the number  scooping avocado out with a spoon  shirt with a cast on her arm. She  with chips on the side. There is a<br>19 on it. He seems to hurt himself and appears in pain. The video is in black and white. The yellow words "DON'T RISK IT" appear on the screen. Then the scene  on a wooden cutting board. There is also a white bowl next to it. holds her thumb up with the injured arm. The word "SWEET" in purple is typed on the screen. plastic container of guacamole with a purple package next to it.<br>changes to a bowl of guacamole with chips on the side.  Narration Caption:  The background<br>voice says the product is<br>Narration Caption:  The background voice says don't risk  refrigerated and pre-scooped for<br>getting injured from cutting up avocados. the customers.<br>… … …<br>Video summarization :<br>**----- End of picture text -----**<br>


The video begins with a man in a kitchen, wearing a T-shirt with the number 19 on it. He is chopping something with a knife, but he seems to hurt himself and appears in pain. This scene is presented in black and white for dramatic effect. As a cautionary message, the yellow words "DON'T RISK IT" appear on the screen. The video then transitions to a bowl of guacamole with chips on the side, and the package of the product is also shown. The audio during this segment warns viewers not to risk getting injured from cutting up avocados. The video then shifts to a close-up shot of a person scooping out avocado with a spoon on a wooden cutting board. A white bowl is placed next to the cutting board. The audio during this segment informs viewers that the product is refrigerated and pre-scooped for the customers, implying that it's a safer and more convenient alternative to preparing avocados manually. The next scene features a woman in a pink shirt with a cast on her arm, suggesting that she might have injured herself while preparing food. She holds her thumb up with the injured arm, possibly indicating that she approves of the pre-scooped avocado product. The word "SWEET" in purple is typed on the screen, perhaps to emphasize her satisfaction with the product. The video concludes with a shot of a bowl of guacamole with chips on the side. A plastic container of guacamole with a purple package is also shown, likely the product that has been discussed throughout the video. The video seems to be a promotional material for a pre-scooped, refrigerated avocado product, highlighting the risks of preparing avocados manually and the convenience of their product. 

_**Video question-answering -** Multi-shot_ : _**Video question-answering -** Multi-shot and temporal related_ : Q: What overarching message is the video conveying about the process of preparing avocados? Q: What is the visual narrative structure of the video? A: The video conveys that manual avocado preparation is risky and promotes a safer preA: The video starts with an injury from manual chopping, introduces the prescooped product. scooped product, and ends with an endorsement. 

Figure 1: An annotated example of our Shot2Story with sing-shot visual captions and narration captions. Moreover, we provide coherent and reasonable video summaries, and question-answering pairs to facilitate comprehensive understanding of multi-shot videos. 

To benchmark the advances of multi-modal video understanding, we designed several distinctive tasks using our dataset, including single-shot video captioning, multi-shot video summarization, and multi-shot video question answering. We design and implement several baseline models using a frozen vision encoder and an LLM, by prompting the LLM with frame tokens and ASR (Automatic Speech Recognition) text. Through extensive experiments, we show that: (1) the ASR text is critical to joint understanding of visual and audio content, (2) processing the video as a whole without the shot structure degenerates the model’s capacity of understanding the multi-shot video, (3) the summarization model trained on our multi-shot summaries can be used on the proposed multishot QA benchmark and generalized to other datasets with longer durations (ActivityNet(Krishna et al., 2017)) and out-of-domain topics (MSRVTT(Xu et al., 2016)), validating the quality of our annotated summaries. Without any bells and whistles, we attain competitive results on zero-shot video question-answering by converting the problem into pure text-based QA with the generated video summaries. 

## 2 THE SHOT2STORY BENCHMARK 

Our new benchmark Shot2Story contains 42,958 videos. The length of each video is ranging from 10s to 40s. We first use an off-the-shelf shot detection method TransNetV2 (Souˇcek & Lokoˇc, 2020) to split each video into shots. For each video shot, we annotate captions for both visual and audio information. Then we further annotate video summaries based on the annotated shot captions. Figure 2 shows an overview of our dataset with some key statistics. 

## 2.1 DATA PREPARATION 

We source videos for our dataset from the public video benchmark HDvila100M (Xue et al., 2022). It offers a large collection of narrative videos, comprising 3M YouTube videos segmented into 100M clips, each about 13 seconds long. We choose this data source for its concise yet complex multi-shot formats, diverse topics, and abundant ASR content. Since we prefer videos with both rich visual and ASR information, we design several filtering techniques to exclude those videos with either low visual-ASR correlation or static visual content. 

We start with keeping video clips with durations between 10 to 40 seconds, since we observe that the majority of the video clips from HDvila100M fall in this range. Then we remove videos with more than 8 shots due to the heavy annotation cost. We also notice that the video segments with too many shots in HDvila100M tend to be slideshows or image collages that deviates from our focuses. Further, to harvest videos with rich visual-ASR correlations, we set up a metric between video shots and ASR 

2 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0003-01.png)


**----- Start of picture text -----**<br>
8000 20000 12000 14000<br>10000 12000<br>6000 16000 8000 10000<br>12000 8000<br>4000 6000<br>8000 6000<br>4000<br>2000 4000 2000 40002000<br>0 0 0 0<br>2 3 4 5 6 7 8<br>Number of shots per video Shot duration range Length of shot visual caption Length of shot narration caption<br>Travel & Events,  Autos & Vehicles,<br>16001400 40000 Sta,s,cs on video summary phrasesVideo Progression Sports, 10.8% 4.1% 10.4% Science &<br>1200 30000 Scene TransitionsCamera Shots Education, 8.1% Technology, 9.1%<br>1000 20000 Narration Descriptions Pets & Animals,  Film &<br>800 10000 2.6% Animation,<br>600 People & Blogs,  8.6%<br>400 0 Nonprofits & 4.9%<br>200 Activism, 0.6%News & Politics,  Entertainment, 12.8%<br>0 12.4% Howto & Style,  Comedy, 2.4%<br>Music, 0.6% 9.7% Gaming, 2.9%<br>Length of video summary Distribution of videos<br>video begins withfollowing thisin the final scenescene transitions tovideo shifts tovideo cuts toclose-upcamera pansthe audio revealsthe narrator explainszooms inthe audio suggests<br>Number of videos Number of shots Number of shots Number of shots<br>1-2 3-4 5-6 7-8 9-10 3 18 28 38 48 58 68 78 88 98 108 118 128 138 3 6 11 16 21 26 31 36 41 46 51 56 61 66-91<br>videos Frequency<br>Number of<br>57 87 117 147 177 207 237 267 297 327 357 387<br>**----- End of picture text -----**<br>


Figure 2: Statistics of Shot2Story . Our dataset features detailed visual captions and narration captions, and video summaries, highlighting video progressions, transitions, camera cuts and narration descriptions, with statistics of frequent expressions depicted in the figure. 

texts. Specifically, we uniformly sample 4 video frames for each shot and obtain the cosine similarity score between the video shot embedding and the text embedding using CLIP (Radford et al., 2021) encoders. We only keep the videos containing at least one shot that is visually correlated to ASR with a threshold of 0.25. Next, in order to obtain videos with diverse shot contents, we set up an inter-shot metric to filter out the videos with similar adjacent shots. We compute the cosine similarities between embeddings of adjacent shots and keep the videos with all inter-shot similarity scores smaller than 0.9. Finally, to further remove the videos with static contents, we adopt an intensity-based scene change detector in PySceneDetect[1] with a low threshold of 11 on our segmented shots. If the filter is unable to detect scene changes at this low threshold, it is conceivable that the shot contains static contents. We only keep the video clips in which all shots contain no static content based on our filtering method. 

As a result, from a total of 2.1M sampled video clips from HD-VILA-100M, we obtain 42 _,_ 958 video clips that meet our quality standard. The number of shots in each video is from 2 to 8. These videos are then shared with our annotators for further annotations. 

## 2.2 ANNOTATION OF SINGLE-SHOT CAPTIONS 

After using TransNetV2 to divide the target videos into video shots, we ask annotators to annotate both visual-only captions and audio-related captions for each shot. We split these two caption annotation to facilitate separate modeling of these two types of information source. For visual-only captions, we require annotators to describe the major subjects and events in the video. Since it is an open-world setting, the videos can be quite diverse and hard to describe. In order to reduce the difficulties of annotating a caption from scratch, we generate an initial video caption using MiniGPT-4 (Zhu et al., 2023) by sampling 4 image frames from the video clip and prompting the model using below prompt. 

_###Human:<Img>Frame1</Img ><Img>Frame2</Img><Img>Frame3</Img><Img>Frame4</Im g>Please describe this video. Do not include details that you are not sure of. For example, if there is text in the image, do not include the content of the text if they are not clearly shown. ###Assistant:_ 

Although MiniGPT-4 is originally designed for image understanding, empirically it is able to generate captions for short video clips, both comprehensively and reasonably. It is able to describe different subjects including persons, animals, food, tools, and virtual objects like animated characters. Annotators are first instructed to correct any errors in the original captions. The mistakes include incorrect descriptions of the object categories, attributes, actions, facial expressions, etc. Also, there might be some subjective descriptions generated by MiniGPT-4 such as emotion and atmosphere. We ask annotators to remove all these subjective descriptions. We then ask annotators to supplement the 

> 1https://www.scenedetect.com/ 

3 

Published as a conference paper at ICLR 2025 

Table 1: High-level comparison of our dataset to previous ones. The summary length of ActivityNet and YouCook2 are combined length of captions in one video. M and G denote manual and generated. 

|Dataset|Annotation|Multi-shot<br>Video|Multi-event<br>Descriptions|Detailed<br>Summary|Summary<br>Length|#Videos|Avg.<br>Duration|
|---|---|---|---|---|---|---|---|
|MSRVTT (Xu et al., 2016)|M|✓|✗|✗|-|10K|15s|
|ActivityNet Caps (Krishna et al., 2017)|M|✓|✓|✗|52.4|20K|3min|
|VideoStorytelling (Li et al., 2019)|M|✓|✓|✓|162.6|105|12.5min|
|Ego4D (Grauman et al., 2022)|M|✗|✓|✗|-|10K|23min|
|YouCook2 (Zhou et al., 2018)|M|✓|✓|✗|67.8|2K|6min|
|VAST (Chen et al., 2024)|G|✓|✗|✓|32.4|27M|5_∼_30s|
|Shot2Story|M+G|✓|✓|✓|218.3|43K|17.1s|



information about the major subjects, actions, and backgrounds present in the video. The goal is for the resulting captions to accurately capture the key elements of each video shot. Statistics shows over 80% of single-shot visual captions are manually corrected. For narration captions, annotators should watch the video and interpret the audio content that visually correlates into descriptive narration, including information sources. For example: “According to the woman in white, the room is not very clean.” All narration captions are manually drafted from scratch. An example of this annotation process is shown in Appendix A.1, where the annotator corrects the caption from “standing in front of the car” to “getting close to the car”, and adding a missing detail of “a close-up shot of the front”. In this way, we find the annotation speed significantly faster ( _∼_ 3 _×_ ) compared to writing a caption from scratch. On the other hand, we find the captions generated this way has more coherent style and tend to cover more details of the video. 

In contrast to the traditional video captioning benchmarks (Xu et al., 2016; Krishna et al., 2017; Zhou et al., 2018), we also annotate narration captions in addition to the visual-only captions. Different from existing audio captioning benchmarks (Gemmeke et al., 2017), we focus more on human speeches rather than acoustic events. Annotators are required to associate the human speech with the video content and summarize the main idea of the speech. We require annotators to describe the source of the speech using visual information. For example, if someone is talking, the annotators need to describe which person in the video is talking. If the human speech refers to some object in the video, the annotator is required to describe which object in the video the speaker is referring to. Note that the speaker identity and reference of visual objects are critical information for understanding a video that cannot be trivially obtained using existing algorithms. There are existing research on speaker identification (Kim et al., 2021) and visual grounding (Anne Hendricks et al., 2017; Zhou et al., 2019), but they only work well on constraint scenarios. 

## 2.3 ANNOTATION OF VIDEO SUMMARY 

To create video summaries with annotated video-shot captions, we leverage an LLM-based approach. Specifically, we form a text prompt with incorporating all shot captions and ASR text included, and uses GPT-4 (OpenAI) to generate a cohesive summary. The text prompt we use is shown in Appendix A.2. The quality is assured through further review and correction by our annotators. 

We prompt GPT-4 to produce coherent, fluent text summaries with transition expressions such as “the video begins”, “following this”, and “in the final scene” to connect video-shot descriptions. The generated annotations also encompass a higher-level understanding of shots, using key phrases such as “scene shifts back” and “returns to the scene” to denote recurring scenes across shots. Notably, GPT-4 often identifies and links the same subjects across scenes without relying on explicit re-identification models. It draws on descriptive and attributive text from shot captions like “a newsroom” or “a man wearing a black suit” to infer scene or subject identity. To ensure quality, annotators carefully review and correct any inconsistencies in scene or subject references within summaries. Since our shot-level captions for generating video summaries are manually checked and annotated, the initial video summaries merely have factual errors, with exceptions for some subject identity and scene mismatches. We require annotators to pay more attention to these errors and ensure holistic summary is accurate and comprehensive. Statistics show that over 40% summaries are manually corrected. 

Despite the rigorous verification process, the reliance on automated generation introduces certain inaccuracies and biases. Common pitfalls in the generated summaries include the omission of minor yet contextually important details and a bias towards emphasizing more prominent actions or objects, 

4 

Published as a conference paper at ICLR 2025 

potentially overlooking less conspicuous elements. This is partly due to our dataset predominantly featuring human-centric activities, as a result of our video filtering process that selects videos rich in visually related audio information and sourcing from HDVILA Xue et al. (2022), which primarily curates content related to human activities. Consequently, our annotations tend to highlight salient events and large-scale objects essential to the video’s storyline, mentioning smaller objects only when they directly contribute to the narrative. 

## 2.4 ANNOTATION OF QUESTION-ANSWERING PAIRS 

We annotate the question-answering pairs on videos in validation and testing splits. To construct this benchmark, we begin with the human-annotated video summaries from Section 2.3, which is detailed in video content. We then prompt GPT-4 (Achiam et al., 2023) to generate candidate question-answer pairs in three predefined categories: temporal-related ( _e.g_ ., _Does the woman appearing at the end of the video wear any accessories?_ ), multi-shot holistic understanding ( _e.g_ ., _What is the overarching theme of the video?_ ), and audio content related ( _e.g_ ., _Which specific car model does the background voice mention, and what visual features confirm its identity?_ ). The text prompt we use is shown in Appendix B.3. 

The quality is then assessed through further automatic filtering and manual checks. Annotators are instructed to verify each question-answer pair against the video content and discard any with mistakes. Simultaneously, they are asked to categorize the questions, where a single question might fall into multiple categories, facilitating evaluating different aspects of multishot understanding _i.e_ ., understanding sequences of events and actions (temporal related), integrating information across multiple shots (holistic understanding), correlating audio content with visual elements (audio related), and others (directly discarded). After this verification, annotators carefully review the quality of the questions to ensure they solely correlate with the video content and that the answers are not revealed in the question texts. This step includes a thorough manual check to address potential mistakes and biases from the initial verification. Through these sequential annotation stages, we ensure high-quality annotations. The process starts with detailed manual single-shot captioning, followed by careful verification and correction at each stage. Even with LLMs introduced to reduce workload, the human-involved process and thorough procedures keep the resulting annotations well-aligned with human labeling. 

Subsequently, to optimize clarity and reduce unnecessary information in the QA pairs, we remove QA pairs with questions exceeding 28 words or answers exceeding 20 words. Then, we employ Vicuna-13B (Chiang et al., 2023) to attempt answering the questions without video context and discard those answerable without accessing the videos. Finally, the pairs are tested against popular methods such as Video-LLaVA (Lin et al., 2023), LLaMAVID (Li et al., 2023e), Video-ChatGPT (Maaz et al., 2023), and VideoChat2 (Li et al., 2023c). Questions correctly answered by more than two models are excluded to guarantee that our dataset poses a substantial challenge. Finally, we obtain 4905 QA pairs for validation and 6465 QA pairs for testing set. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0005-06.png)


**----- Start of picture text -----**<br>
1200 Holistic/Temporal All-shared, 0.3% related only, Temporal-<br>1000 shared, 5.4% 12.2%<br>Temporal/Audi<br>800 o shared, 0.5%<br>600 Holistic<br>understanding  Audio-related<br>400 only, 46.9% only, 29.4%<br>200<br>0 Audio/Holistic<br>6 8 10121416182022242628 shared, 5.3%<br>Length of questions Distribution of questions<br>Number of questions<br>**----- End of picture text -----**<br>


Figure 3: Distribution of multi-shot video QA benchmark. Questions from different categories overlap. _All-shared_ means questions fall under all three categories. 

## 2.5 COMPARISON TO EXISTING BENCHMARKS 

Compared to existing video description datasets, our dataset is more challenging due to the explicit modeling of the multi-shot nature of web videos. Our textual description includes both shot-level captions and video-level summaries, combining visual and audio understanding, which provides a unique test bed for multi-modal video understanding. Table 1 shows a high-level comparison of our new dataset with existing video captioning benchmarks. Most existing video captioning benchmarks, such as MSRVTT (Xu et al., 2016), YouCook2 (Zhou et al., 2018) and ActivityNet Caps (Krishna et al., 2017), also use multi-shot videos as annotation source, but they either annotate a holistic caption for the video (MSRVTT) or ask annotators to decide the boundary of different events. In our study, we observe that video shots naturally create a sequence of related events, motivating us to annotate distinct captions for each shot. Ego4D (Grauman et al., 2022) only annotates dense visual captions but not audio captions for relatively long egocentric videos. Video Storytelling (Li et al., 

5 

Published as a conference paper at ICLR 2025 

Table 2: Performance of models on video shot captioning using different modalities, following the settings of VAST(Chen et al., 2024). The models are fine-tuned on Shot2Story video shot captions. V, A and S are abbreviated for vision, audio and subtitle (ASR text) respectively. 

|Model|FT Modality|FT Modality|B4|M|R|C|
|---|---|---|---|---|---|---|
|VAST||V+S|10.7|16.1|30.3|33.8|
|VAST||V+A+S|10.7|16.1|30.4|34.0|
|MiniGPT4-C||V|9.2|14.7|27.9|25.1|
|MiniGPT4-C||V+S|11.8|16.7|30.1|35.9|
|VideoChat2-C||V|8.8|16.1|27.9|23.7|
|VideoChat2-C||V+S|**13.8**|**18.7**|**32.1**|**43.9**|



Table 3: Performance of models on video summarization. The models are fine-tuned on Shot2Story video summaries. V and S are abbreviated for vision and subtitle (ASR text) respectively. 

|Model|FT|Modality|B4|M|R|C|
|---|---|---|---|---|---|---|
|Video-ChatGPT w/o ASR (Maaz et al., 2023)||V|4.8|17.3|21.3|1.5|
|Video-ChatGPT (Maaz et al., 2023)||V+S|3.6|17.8|19.7|1.0|
|MiniGPT4-SUM-holistic||V+S|7.8|16.9|23.4|2.8|
|MiniGPT4-SUM-shot w/o ASR||V|10.4|18.5|25.8|4.8|
|MiniGPT4-SUM-shot||V+S|12.4|19.7|27.6|7.6|
|VideoChat2-SUM-shot||V+S|**12.7**|**19.8**|**28.3**|**9.0**|



2019) is a small-scale dataset with annotations of multiple events in a videos and provides a summary of the video by concatenating all captions. 

A recent work VAST (Chen et al., 2024) feeds generated video and audio captions into an LLM to generate video summary. However, it processes multi-shot video as a whole and lacks the granularity of the events in different shots. Moreover, VAST directly uses predicted captions without any human verification, leading to potentially noisy and biased summaries towards the captioning models. Our dataset stands out from VAST with its accurately annotated visual and narration shot captions. Although our video summary is also generated using an LLM, it is further verified by annotators to make sure there is no hallucinated details from the LLM. Our dataset has an average length of 218 _._ 3 words for the video summary, which is much longer than existing benchmarks, and is longer than the combined length of captions in one video in ActivityNet and YouCook2. 

Furthermore, our Shot2Story-QA introduces unique and complex challenges through its emphasis on shot transitions and multi-event progression, setting it apart from benchmarks like MSRVTT-QA Xu et al. (2017) and ActivityNet-QA Yu et al. (2019). For instance, unlike existing benchmarks Xu et al. (2017); Yu et al. (2019) that assess understanding at a single time point, _e.g_ ., "Who do three judges talk to?" (MSRVTT-QA, in Figure 19), or general inquiries like "What is the person in the video doing?" (ActivityNet-QA, in Figure 20), Shot2Story-QA includes "Temporal-related" and "Multishot Holistic Understanding" questions. A temporal-related question, such as "What is the man’s immediate action after handling the skewer?" shown in Figure 21, requires models to comprehend the sequence and progression of events, linking consecutive actions meaningfully. Similarly, multi-event progression questions like "How does the setting change from the start to the end of the video?" necessitate understanding multiple concurrent events within their temporal context. This provides a more rigorous and nuanced evaluation framework for temporal understanding. 

## 3 TASKS AND EXPERIMENTS 

## 3.1 BASIC SETTINGS 

For all the tasks described in this section, we follow the same training/validation/test split. Specifically, the number of videos for training, validation, and test set are 36951, 1982 and 4025, respectively. We resize the frames to 224 _×_ 224. We adapt two popular VLMs to accomodate our tasks: MiniGPT4 (Zhu et al., 2023) and VideoChat2 (Li et al., 2023c). For MiniGPT-4, we employ ViT-G/14 (Fang et al., 2022) and Q-Former (Li et al., 2023a) as visual encoder, and Vicuna v0-7B (Chiang et al., 2023) as the language model. We load pretrained Q-Former and MLP from MiniGPT-4 (Zhu et al., 2023). In training, we update only Q-Former and MLP parameters, keeping the ViT and LLM frozen. 

6 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0007-01.png)


**----- Start of picture text -----**<br>
This is a video of 4 shots. The first shot is<br>𝐯!! … 𝐯"! !<br>shot 1<br>The second shot is … The third shot is … The<br>fourth shot is<br>𝐯!# … 𝐯"#!<br>In the audio, I hear that:<br>shot 4<br>"Have you suffered an avocado related injury, dont<br>risk it holy guacamole is refrigerated and pre scooped<br>Input to LLM ASR for your protec=on, …"<br>Params updated Visual tokens<br>Params frozen Sing-shot tokens Describe this video in detail.<br>CLIP<br>encoder Q-Former<br>… …<br>CLIP<br>encoder Q-Former<br>Large Language Model<br>Multi-shot Video sumamry<br>**----- End of picture text -----**<br>


Figure 4: Model structure for multi-shot video summarization model SUM-shot. We arrange visual tokens sequentially for each single shot and in a multi-shot format to encapsulate multi-shot information. Additionally, ASR text is incorporated for audio-visual video summarization. 

For VideoChat2, we employ UMT-L(Li et al., 2023d) as backbone and load pretrained Q-Former and MLP from VideoChat2 (Li et al., 2023c). During training, we adopt LoRA(Hu et al., 2021) and AdamW (Loshchilov & Hutter, 2017) with a learning rate of 8e-5. We train both models for 10 epochs with a batch size of 128 for single-shot video captioning. We finetune our video summarization models on the single-shot captioning models with a batch size of 32. For captioning and summarization, we evaluate the models using BLEU@4 (Papineni et al., 2002) (B), METEOR (Denkowski & Lavie, 2014) (M), ROUGE (Lin, 2004) (R), and CIDEr (Vedantam et al., 2015) (C). 

## 3.2 SINGLE-SHOT VIDEO CAPTIONING 

This task involves generating descriptions for individual video shots, where the target description is a concatenation of the visual-only and narration caption for a video shot. This task requires a joint understanding of visual and speech information. Specifically, we adopt a similar structure as we generate pseudo captions for data annotation in Section 2.2. First, we sample _Ns_ frames from a video shot, encode them using a fixed vision encoder, then feed the encoded features to a Q-Former to produce visual tokens. Further, we combine the visual tokens and an optional ASR text into a unified LLM prompt to facilitate both visual and narration understanding. We adapt the framework of MiniGPT-4 (Zhu et al., 2023) and VideoChat2 (Li et al., 2023c), with the two models denoted as MiniGPT4-C and VideoChat2-C for brevity. We compare with baseline model VAST (Chen et al., 2024), which is able to processes audio, vision, and subtitle inputs simultaneously. 

The results are shown in Table 2. Benefiting from extensive pretraining, VAST achieves 34 on C and 30 on R, comparable to MiniGPT4-C. Incorporating an additional audio modality results in a negligible performance difference, indicating that audio content only contributes marginally given the ASR text as input on our benchmark. We then assess variants of our models, MiniGPT4-C and VideoChat2-C, with and without the additional ASR text. It shows that including ASR texts significantly enhances performance across all metrics, with a notable boost in R and C, highlighting the relevance of audio content to our video captions. Furthermore, VideoChat2-C, featuring a superior visual backbone and extensive video pretraining, consistently outperforms MiniGPT4C. This superiority highlights the importance of advanced visual backbone and video pretraining, confirming our benchmark’s robustness. However, despite these advances, the results also indicate room for improvement. Figure 5 (a) showcases output examples of our model’s single-shot video captioning, detailing visual elements and audio content effectively, capturing actions like "gesturing to explain her fear" and secondary elements such as "a red stuffed doll next to her". 

## 3.3 MULTI-SHOT VIDEO SUMMARIZATION 

Multi-shot video summarization is a new task that is distinct from existing video description tasks. It requires the model to understand the shot structure of the given video and to provide a coherently paragraph to describe the progression of events in the different shots. Due to the complexity of this task, we adopt GPT-V (Achiam et al., 2023) to generate a supplementary training set, with video summaries for another 90K videos, sampled in the same approach as described in Section 2.1. Please 

7 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0008-01.png)


**----- Start of picture text -----**<br>
… … … …<br>Human speech : explains to us the fear she has. Human speech : But when you look at something like this, how many man<br>hours have gone into peeling back the matrix?<br>Single-shot  video captioning: Single-shot  video captioning:<br>in the video, a pregnant woman is si<ng on a black couch. she is  in the video, a man in a blue shirt and a man in a hat are looking at a<br>wearing a black halter top. there is a red stuffed doll next to her. she  large rock in a room. the man in the blue shirt is querying how many<br>is gesturing to explain her fear. man hours have spent.<br>(a) Example predic0ons of single-shot video cap0oning.  Videos: dR9jfG9Zr5A.2_516_578 and JPMGjObNQwk.58_161_287<br>… … … … …<br>Human speech : PuHer pocket holds two puHers for quick and easy access in the front. We have a flap that you can close to protect your discs, but theres also<br>velcro in there, so that when youre ready to play, you have easy access to your discs.<br>**----- End of picture text -----**<br>


_**VideoChat2-SUM-shot**_ video summarization: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0008-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0008-04.png)


**----- Start of picture text -----**<br>
the video begins with a  close-up shot  of  a blue and black backpack  resting  on a wooden table . the backpack is  adorned with a logo  and<br>features a putter pocket , as mentioned by  a male voice in the audio . the scene then transitions to a man  sitting at the same wooden table . he<br>is dressed in  a red shirt and black shorts , and he is  surrounded by various backpacks . the man  proceeds to open one of the backpacks ,<br>revealing its contents. as he does this,  the male voice in the audio  explains that  the backpack has a flap that can be closed to protect the discs<br>inside . in the final part of the video, the focus shifts to  a close-up shot  of  the man's hand as he opens the backpack . the backpack is  blue  and<br>black ,  with a logo on the front . the male voice in the audio explains that there is velcro in the backpack, which makes it easy for people to<br>access their discs when they are ready to play (b) Example predic0ons of mul0-shot video summariza0on. . Video: ETyfOQt9vtM.2<br>(b) Example predic0ons of mul0-shot video summariza0on .  Video: ETyfOQt9vtM.2<br>Temporal-related Question : What colors are visible on the backpack that  Multi-shot Holistic Understanding Question : What is the overarching<br>is initially shown behind the frisbee? theme of the video?<br>Ground-truth answer:  Blue, gray, and white. Ground-truth answer:  Showcasing disc storage backpacks.<br>Answer  based on  VideoChat2-SUM-shot summary :  Answer  based on  VideoChat2-SUM-shot summary :<br>The backpack in the video has a pa@ern of blue, black, and white. The overarching theme of the video is the demonstraAon of a<br>backpack with features that make it suitable for playing disc golf.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0008-05.png)


**----- Start of picture text -----**<br>
(c) Example questions and predicted answers of multi-shot video.  Video: ETyfOQt9vtM.2<br>**----- End of picture text -----**<br>


Figure 5: Example predictions of our models. (a) shows single-shot video captioning results of VideoChat2-C, which incorporates audio and visual content correctly (b) shows multi-shot video summarization of VideoChat2-SUM-shot, with accurate descriptions in green and errors in red, illustrating the model’s ability to narrate event sequences (c) shows two sample questions of the video in (b). The answers are based on the generated summary of VideoChat2-SUM-shot. 

check the annotation prompt and data samples in Appendix B.5. First, we finetune an existing video caption model Video-ChatGPT (Maaz et al., 2023) by instruction-tuning it on our video summary data, with and without the additional ASR text input. Then, we experiment with three different architecture designs based on MiniGPT4. The first model MiniGPT4-SUM-holistic uses a similar pipeline as MiniGPT4-C. We uniformly sample 16 frames from the full video clip and prompt the LLM with frame tokens and ASR text. The second model MiniGPT4-SUM-shot w/o ASR, neglecting ASR input, uses a more refined framework by sampling 4 frames in each video shot and prompting the LLM with frame tokens from different shots, as is shown in Figure 4. The third model, MiniGPT4-SUM-shot further incorporates ASR text input as an additional input. Further, we replace the backbone of MiniGPT4-SUM-shot with the more advanced VideoChat2 model, resulting in the VideoChat2-SUM-shot model variant. Compared to SUM-shot, SUM-holistic does not have explicit shot information and relies on the LLM to parse the video shots using the provided frame features. 

Table 3 shows the results of the models. It is shown that MiniGPT4-SUM-holistic is worse than MiniGPT4-SUM-shot, showing the importance of the shot structure in predicting a video summary matching the transition of shots. MiniGPT4-SUM-shot w/o ASR underperforms compared to MiniGPT4-SUM-shot and outperforms MiniGPT4-SUM-holistic, highlighting the significance of both audio information and shot structure in multi-shot video understanding. Compared VideoChat2SUM-shot and MiniGPT4-SUM-shot, the former model achieves the best performance, indicating the benefit of advanced vision backbone and video pretraining. Video-ChatGPT obtains much worse performance comparing to our models, potentially due to their weakness in processing multiple scenes and building the correlation between visual frames and ASR texts. It directly encodes the whole video into a sequence of tokens, potentially losing significant frame details and essential correlation between ASR and visual frames, while ours directly feed frames tokens into the LLM without compressing them. 

8 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0009-01.png)


**----- Start of picture text -----**<br>
… … … … … … …<br>VideoChat2-SUM-shot video summarization:<br>in the opening scene of the video, a man stands on a ping pong table in an indoor sports facility. he is dressed in a black polo shirt with a logo on the left chest, and he holds a  …<br>ping pong racket in his right hand. the background is a vibrant blue, and there's a sign that reads welcome to the world of ping pong. as he speaks, he introduces the topic of the<br>video, which is about the basics of ping pong. transitioning to the second scene, the same man continues to address the audience, now holding a red ping pong ball in his hand.<br>he emphasizes the importance of understanding the basics of ping pong, suggesting that this is a fundamental aspect of the sport. in the third scene, the man is seen again on the<br>ping pong table, this time preparing for a serve. he demonstrates the proper grip and stance for a serve, which is a crucial part of the game. the video then cuts to the fourth<br>scene, where a different man appears on the ping pong court. he is wearing a white t-shirt emblazoned with the website www.pingponginfo.com and the text ping pong<br>information . this man speaks about the importance of proper footwork in ping pong, highlighting the role it plays in the game. finally, in the fifth and last scene, the same man<br>from the first scene reappears. he is still on the ping pong table, and he poses a question to the viewers, asking if they have a question about ping pong. this serves as an<br>invitation for the audience to engage with the content and potentially learn more about the sport. throughout the video, the speakers, including the man on the ping pong table<br>and the man in the white t-shirt, provide informative commentary on the basics of ping pong, emphasizing the importance of understanding the fundamentals of the game.<br>Questions : Answers :<br>Q1: what is the person in the video doing?  GT1: play table tennis<br>Q2: is the person in the video wearing a watch?  GT2: yes<br>Q3: is the person in the video wearing a hat?  GT3: no<br>Q4: what is in front of the person in the video?  GT4: table tennis table<br>Q5: what happened before playing table tennis?  GT5: introduction of pang ball<br>Vicuna answers using summaries  & GPT evaluation :<br>A1: The person in the video is demonstrating the basics of ping pong, including proper grip, stance, serve, and footwork.<br>A2: No, the person in the video is not wearing a watch.<br>A3: No, the person in the video is not wearing a hat.<br>A4: In the video, the person is standing on a ping pong table in an indoor sports facility.<br>A5: Based on the video, it is not clear what happened before the man started playing table tennis.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0009-02.png)


**----- Start of picture text -----**<br>
ActivityNet-QA case: v_mZYWfmsYQPA.mp4  duration: 104 seconds<br>**----- End of picture text -----**<br>


Figure 6: Example predictions of our model on zero-shot question answering an ActivityNet-QA video. More questions and explanations can be found in Appendix C.2. 

Figure 5 (b) showcases predictive capabilities of our VideoChat2-SUM-shot model. The model adeptly narrates event sequences with appropriate emphasis. For instance, it details the backpack’s colour and location, and rationalizes the item in the beginning shot, _i.e_ ., “putter pocket”, aligning with the ASR with “by a male voice in the audio”. However, some predictions that marked in red are erroneous, such as the incorrect “black” colour and the non-existent “with a logo on the front” in the ending shot. These inaccuracies likely stem from the LLM’s tendency to “hallucinate” plausible yet non-factual details. Despite these errors, the model demonstrates proficiency in generating consistent and nuanced summaries, highlighting our model’s potential and the challenges our dataset presents. 

## 3.4 VIDEO QUESTION-ANSWERING WITH VIDEO SUMMARY 

Generated video summaries are supposed to be grounded and detailed, covering rich elements like event progression, holistic topics and audio elements, making them suitable for other vision tasks such as video question-answering. Existing work (Guo et al., 2023; Zhang et al., 2023) uses image or video frame captions as input to an LLM to generate question responses. However, little work has been done for the capacity of video summaries. We directly apply our video summarization model on video QA benchmarks, _i.e_ . MSRVTT-QA (Xu et al., 2017), ActivityNet-QA (Yu et al., 2019) and our Shot2Story-QA. 

Specifically, we first split the testing videos into video shots, and then feed the videos into our SUM-shot models. The generated summaries and the associated questions are then fed into a Vicuna model to derive the answers with the prompt shown in Appendix B.4.1. Note there is no adaptation or finetuning conducted for the Vicuna model. Since the original answers in the QA benchmarks are very short and the responses generated by LLM tend to be long sentences, we leverage the gpt-3.5-turbo model to generate a binary decision on whether the answer is correct, following Video-ChatGPT (Maaz et al., 2023). 

**Zero-shot video question-answering.** As shown in Table 4, our results with VideoChat2-SUMshot surpass 5 out of 6 existing video-VLMs on MSRVTT-QA and 4 out of 6 existing models on ActivityNet-QA. Furthermore, our results are comparable to the SOTA performance on MSRVTT-QA with Video-LLaVA (Lin et al., 2023). Note that these models require extensive instruction-tuning data to learn to directly generate answers from visual features and the text prompt, whereas our model bypasses instruction tuning by distilling the video information into a summary. Our model also follows the zero-shot QA setting since the model only uses Shot2Story as training data. Note that MSRVTT contains a large portion of videos with out-of-domain topics such as TV shows 

9 

Published as a conference paper at ICLR 2025 

Table 4: Performance on video question answering on MSRVTT-QA and ActivityNet-QA. IT means whether the model uses video-text instruction tuning data. All methods follow the zero-shot manner. 

|Model|Training Data|IT|QA<br>Input|MSRVTT<br>QA|ActivityNet<br>QA|
|---|---|---|---|---|---|
|||||||
|VideoChat (Li et al., 2023b)<br>Video-ChatGPT (Maaz et al., 2023)<br>MovieChat (Song et al., 2023)<br>LLaMA-VID (Li et al., 2023e)<br>VideoChat2 (Li et al., 2023c)<br>Video-LLaVA (Lin et al., 2023)|Cap.+QA<br>Cap.+QA<br>Cap.+QA<br>Cap.+QA<br>Cap.+QA<br>Cap.+QA|✓<br>✓<br>✓<br>✓<br>✓<br>✓|V+T<br>V+T<br>V+T<br>V+T<br>V+T<br>V+T|45.0<br>49.3<br>52.7<br>57.7<br>54.1<br>**59.2**|26.5<br>35.2<br>45.7<br>47.4<br>**49.1**<br>45.3|
|||||||
|MiniGPT4-SUM-shot<br>VideoChat2-SUM-shot|Summary<br>Summary|✗<br>✗|T<br>T|57.7<br>58.5|45.6<br>47.1|



Table 5: Benchmark on Shot2Story-QA. IT means usage of video-text instruction tuning data. Summary, Cap. and QA denote video summary, captions and question-answering pairs. 

|Model|Training data|IT|QA<br>Input|Temporal<br>related|Holistic<br>understanding|Audio<br>related|Overall|
|---|---|---|---|---|---|---|---|
|||||||||
|LLaMA-VID (Li et al., 2023e)<br>Video-ChatGPT (Maaz et al., 2023)<br>VideoChat2 (Li et al., 2023c)<br>Video-LLaVA (Lin et al., 2023)|Cap.+QA<br>Cap.+QA<br>Cap.+QA<br>Cap.+QA|✓<br>✓<br>✓<br>✓|V+T<br>V+T<br>V+T<br>V+T|7.9<br>13.1<br>15.1<br>16.4|9.7<br>15.5<br>15.4<br>14.8|11.4<br>14.3<br>13<br>11.7|9.7<br>14.2<br>14.5<br>14.3|
|||||||||
|MiniGPT4-SUM-shot<br>VideoChat2-SUM-shot|Summary<br>Summary|✗<br>✗|T<br>T|28.9<br>36.1|31.9<br>41.5|36.7<br>43.8|32.5<br>40.5|



(e.g., Figure 19) and food, while ActivityNet has much longer videos than our training videos (e.g., Figure 6. This validates the robustness and transferability of our model across different topics and longer videos. This surprisingly good result indicates that a comprehensive and detailed video summary is a high-quality abstraction of the video, facilitating a wide range of tasks including video QA and video-based conversation. Moreover, our model achieves strong results on ActivityNetQA, which predominantly consists of single-shot long videos, validating that models trained with multi-shot videos can effectively generalize to single-shot videos. 

**Multi-shot video question-answering.** As shown in Table 5, we benchmark existing and our proposed video summary models on Shot2Story-QA. Specifically, four popular video-VLMs are compared, _i.e_ ., Video-ChatGPT (Maaz et al., 2023), LLaMA-VID (Li et al., 2023e), VideoChat2 (Li et al., 2023c) and Video-LLaVA (Lin et al., 2023). The predicted summaries of MiniGPT4-SUMshot and VideoChat2-SUM-shot are used to tackle the QA task with the same configuration as zero-shot VQA. Accuracies on temporal-related, multi-shot holistic understanding and audio-related are reported, with the “overall” metric showing the average score from these three sub-tasks. The current video-VLMs present unsatisfying results, potentially due to two factors: (1) The current models does not have audio or ASR as input, lacking capacity with audio-related understanding. (2) Current models do not have training data with detailed descriptions based on multi-shot videos, weakening their performance on holistic understanding and temporal modeling. For our models, VideoChat2-SUM-shot achieves an overall score of 40.5, surpassing the compared models and MiniGPT4-SUM-shot on all three subtasks. This performance underscores the benefits of video pretraining and the advanced visual backbone of VideoChat2. These baseline results highlight the complexities and demanding nature of our Shot2Story-QA task. We show some example predictions in Figure 5(c). Please refer to Appendix C.2 for more qualitative results. 

## 4 CONCLUSION 

In this work, we present Shot2Story, a large-scale benchmark for comprehensive multi-shot video understanding. We provide detailed shot-level captions for both visual signals and human narrations. Furthermore, we provide comprehensive video summaries based on shot-level captions and design a challenging video question-answering benchmark for multi-shot video understanding. With the rich and diverse descriptions, our benchmark serves as a playground for future multi-modal video understanding models, ready to be extended for a range of other video understanding tasks, such as visual grounding and video-based conversation. 

10 

Published as a conference paper at ICLR 2025 

## REFERENCES 

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In _ICCV_ , pp. 5803–5812, 2017. 

- Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, and Jing Liu. Vast: A vision-audio-subtitle-text omni-modality foundation model and dataset. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. URL https: //lmsys.org/blog/2023-03-30-vicuna/. 

- James E Cutting, Kaitlin L Brunick, Jordan E DeLong, Catalina Iricinschi, and Ayse Candan. Quicker, faster, darker: Changes in hollywood film over 75 years. _i-Perception_ , 2(6):569–576, 2011. 

- Michael Denkowski and Alon Lavie. Meteor universal: Language specific translation evaluation for any target language. In _Proceedings of the ninth workshop on statistical machine translation_ , pp. 376–380, 2014. 

Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. _arXiv preprint arXiv:2211.07636_ , 2022. 

- Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In _2017 IEEE international conference on acoustics, speech and signal processing (ICASSP)_ , pp. 776–780. IEEE, 2017. 

- Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In _CVPR_ , 2022. 

- Jiaxian Guo, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Boyang Li, Dacheng Tao, and Steven Hoi. From images to textual prompts: Zero-shot visual question answering with frozen large language models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 10867–10877, 2023. 

- Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_ , 2021. 

- You Jin Kim, Hee-Soo Heo, Soyeon Choe, Soo-Whan Chung, Yoohwan Kwon, Bong-Jin Lee, Youngki Kwon, and Joon Son Chung. Look who’s talking: Active speaker detection in the wild. _Interspeech_ , 2021. 

- Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In _Proceedings of the IEEE international conference on computer vision_ , pp. 706–715, 2017. 

Junnan Li, Yongkang Wong, Qi Zhao, and Mohan S Kankanhalli. Video storytelling: Textual summaries for events. _IEEE Transactions on Multimedia_ , 22(2):554–565, 2019. 

- Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ , 2023a. 

- KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding, 2023b. 

11 

Published as a conference paper at ICLR 2025 

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. _arXiv preprint arXiv:2311.17005_ , 2023c. 

- Kunchang Li, Yali Wang, Yizhuo Li, Yi Wang, Yinan He, Limin Wang, and Yu Qiao. Unmasked teacher: Towards training-efficient video foundation models. _arXiv preprint arXiv:2303.16058_ , 2023d. 

- Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. _arXiv preprint arXiv:2311.17043_ , 2023e. 

- Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. _arXiv preprint arXiv:2311.10122_ , 2023. 

- Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In _Text summarization branches out_ , pp. 74–81, 2004. 

- Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2017. 

- Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. _arXiv preprint arXiv:2306.05424_ , 2023. 

OpenAI. Gpt-4. URL https://chat.openai.com/?model=gpt-4. 

- Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th annual meeting of the Association for Computational Linguistics_ , pp. 311–318, 2002. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pp. 8748–8763. PMLR, 2021. 

- Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Xun Guo, Tian Ye, Yan Lu, Jenq-Neng Hwang, et al. Moviechat: From dense token to sparse memory for long video understanding. _arXiv preprint arXiv:2307.16449_ , 2023. 

- Tomáš Souˇcek and Jakub Lokoˇc. Transnet v2: An effective deep network architecture for fast shot transition detection. _arXiv preprint arXiv:2008.04838_ , 2020. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 

- Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 4566–4575, 2015. 

- Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In _ACM Multimedia_ , 2017. 

- Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. CVPR, June 2016. 

- Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In _CVPR_ , pp. 5036–5045, 2022. 

Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , 2019. 

12 

Published as a conference paper at ICLR 2025 

- Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, and Gedas Bertasius. A simple llm framework for long-range video question-answering. _arXiv preprint arXiv:2312.17235_ , 2023. 

- Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In _AAAI_ , volume 32, 2018. 

- Luowei Zhou, Yannis Kalantidis, Xinlei Chen, Jason J Corso, and Marcus Rohrbach. Grounded video description. In _CVPR_ , pp. 6578–6587, 2019. 

- Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , 2023. 

13 

Published as a conference paper at ICLR 2025 

## A ANNOTATION PROCESS 

In this section, the annotation process for Shot2Story is detailed, including single-shot caption annotation in Sec. A.1, GPT-4 summary generation prompts in Sec. A.2, human correction of summaries in Sec. A.3, and annotation of question-answering pairs in Sec. A.4. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0014-03.png)


**----- Start of picture text -----**<br>
a group of people standing in front of a large  This is  a close-up shot of the front of a car.<br>truck parked in a parking lot. The truck is a  The car is a large, yellow and black commercial<br>large, yellow and black commercial vehicle  vehicle with the words " Thomas " wri=en  on<br>with the words "Ford" wri=en on the side.  Reading Human  the front . Two people are  looking and talking<br>The people are standing in front of the truck,  Correc1on to each other in a parking lot .  They are<br>Genera1ng  looking at it and talking to each other. ge7ng close to the car .<br>ini1al cap1on<br>MiniGPT-4 Watching video<br>Video to be annotated<br>**----- End of picture text -----**<br>


Figure 7: Human annotation process of sing-shot video captions. Texts in bold green represent correct content, while those in red indicate errors. Please find more explanations in Sec. A.1. 

## A.1 HUMAN ANNOTATION OF SINGLE-SHOT CAPTIONS 

Our single-shot video caption annotation process, described in Sec. 2.2, is a two-phase approach designed for high-quality and style-consistent captions. This procedure also accelerates the annotation process by _∼_ 3 times. As depicted in Figure 7, the process begins with MiniGPT-4 generating initial captions from structured prompts. While these captions often correctly identify subjects such as “parking lot” and “vehicle”, they sometimes inaccurately describe actions or locations. Annotators then watch the video shot and revise these captions. For instance, errors like “standing in front of”, depicted in red, are corrected to “getting close to the car”, shown in green. Additionally, annotators enrich captions with key details, such as “a close-up shot of the front of a car.” Our single-shot narration caption annotation process follows a similar approach. Differently, we offer ASR text and videos to the annotators and ask them to write down the visually related content and describe the source of the speech. The process during model deployment has been detailed in Sec. 3.2. 

## A.2 GPT-4 SUMMARIZATION PROMPT 

We utilize GPT-4 to summarize video clips, leveraging our detailed video-shot captions and ASR text. The summarization follows a prompt structure adapted from (Li et al., 2023b), which defines video captions and audio captions for each shot, as depicted in Figure 9. For each video, we organize shot durations, video captions, narration captions, and ASR into a text format (see Figure 8). This arranged content is then fed into GPT-4 for generating the video summary. 

## A.3 HUMAN CORRECTION OF VIDEO SUMMARIES 

Our detailed shot captions enable GPT-4 to effectively identify and link subjects across shots, without requiring extra re-identification modules. However, according to human evaluation, about 30% of video summaries struggle to connect objects and scenes across shots. Our annotators review these summaries alongside the video clips to correct such errors. Figure 10 illustrates this process. While GPT-4 accurately references the same location, such as “the open field” and “the same open field”, it sometimes fails to maintain continuity with elements like “the black car” across scene transitions. Annotators must watch the video and assess the initial summary to make necessary corrections for the final summary. This method ensures the production of high-quality video summaries with efficiency. 

## A.4 ANNOTATION OF VIDEO QUESTION-ANSWERING PAIRS 

As shown in Figure 13, our annotation of question-answering (QA) pairs utilizes a hybrid manualautomatic approach to ensure both diversity and quality. Initially, we employ GPT-4 to generate 

1 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0015-01.png)


**----- Start of picture text -----**<br>
The video has  4  shots. It has  15.8 seconds  in total.<br>The first ac3on segment starts from  0.0  seconds to  6.5  seconds.<br>Visual cap3on of this clip is:  It's a man in a kitchen chopping something with a knife. The man<br>is wearing a tee shirt with the number 19 on it. He seems to hurt himself and appears in pain.<br>The video is in black and white. The yellow words "DON'T RISK IT" appear on the screen. Then<br>the scene changes to a bowl of guacamole with chips on the side. The video also shows the<br>package.<br>The audio cap3on of this clip is:  In the audio, the background voice says don't risk geLng<br>injured from cuLng up avocados.<br>The second ac3on segment starts from  6.5  seconds to  10.5  seconds.<br>Visual cap3on of this clip is:  It's a close-up of a person scooping avocado out with a spoon on a<br>wooden cuLng board. There is also a white bowl next to it.<br>The audio cap3on of this clip is:  In the audio, the background voice says the product is<br>refrigerated and pre-scooped for the customers.<br>The third ac3on segment starts from  10.5  seconds to  13.5  seconds.<br>Visual cap3on of this clip is:  It's a woman in a pink shirt with a cast on her arm. She holds her<br>thumb up with the injured arm. The word "SWEET" in purple is typed on the screen.<br>The audio cap3on of this clip is:<br>The forth ac3on segment starts from  13.5  seconds to  15.8  seconds.<br>Visual cap3on of this clip is:  It's a bowl of guacamole with chips on the side. There is a plasSc<br>container of guacamole with a purple package next to it.<br>The audio cap3on of this clip is:<br>The ASR of the video is:  ASR content<br>**----- End of picture text -----**<br>


Figure 8: Example of textual content for video in Figure 1. Texts in color are specific for input video and are replaced during our generation. 

You are a chatbot that conducts conversa0ons based on video contexts. You mainly answer based on the given cap0on contexts from different sources, such as visual cap0on and audio cap0on. You can also answer the relevant knowledge of the person or object contained in the video. The video has a high-level topic and the video content is supposed to be coherent. The video can have more than one shot and in each shot different ac0on segments and events exist. Visual cap0on is provided by a validated annotater and it usually describes more details in the video. Audio cap0on is the cap0on of audio in the shot, which is relevant to the visual content and it is provided by a validated annotator. Different shots are annotated separately, but the content and topics of different shots are supposed to be coherent. You should refer to the audio cap0on and extend the content accordingly. You don’t create a video plot out of nothing. The video content is: 

## **{video content}** 

Describe the video in detail and coherently, including the ac0ons and scenes. You should not men0on the number of scenes, but you ought to describe the content changes and transi0on. You ought to describe the content in happening and reasoning order. You should not create a video plot out of nothing. 

Figure 9: Prompt template for GPT-4 summarization. 

candidate question-answer pairs based on our specific instructions, focusing on creating temporalrelated, holistic understanding, and audio-related questions. Subsequently, annotators review these pairs while watching the corresponding videos to eliminate simple or incorrect entries. Subsequently, the annotators are asked to categorize the QA pairs into predefined categories, _i.e_ ., temporal-related, holistic understanding and audio-related, and discard the data not under these categories. 

The prompts used for generating candidate QA pairs are detailed in Figure 11 and Figure 12. For different task-specific questions, we adopt different task instructions as shown in Figure 11. During generation, we adopt the template shown in Figure 12. The boldfaced texts, such as “ **{shot_caps}** ”, “ **{video_sum}** ”, “ **{task_inst}** ”, are replaced with shot captions organized as in Figure 8, video summary and task-specific instructions from Figure 11. 

2 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0016-01.png)


**----- Start of picture text -----**<br>
…  an open field  where two men are standing  … an open field where two men are standing<br>... they engage in a conversation about… the  ... they engage in a conversation about… the<br>Generate initial summaries man in the long-sleeved shirt mentions … as  man in the long-sleeved shirt mentions … as<br>as shown in Figure 8-9 the video transitions, we see parked  sitting  inside the car… in the same open fielda black car  .  a person is  Reading CorrectionHuman  the video transitions, we see parked in the same open field.  are sitting inside the car… the black car the persons<br>GPT-4<br>Watching video<br>+ASR<br>Video-shot  Annotate<br>captions shot captions<br>as shown in Figure 7<br>Video to be annotated<br>**----- End of picture text -----**<br>


Figure 10: Human correction process of video summaries. Overlapped text is omitted for clarity. Texts in bold green represent correct content, while those in red indicate errors. Please find more explanations in Sec. A.3. 

Please ask 5 questions only related to visual content. Please remember to avoid audio content. Please answer the questions according to the Holistic Video Summary, or Video Shot Captions, depending on the question types. 

The questions should be related to multiple video shots, rather than a single shot. They can only be answered with multiple video shots and cannot be answered by a single shot. The questions should be logical, reasonable and non-trivial. Please only return json of the questions and answers in the below format. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0016-05.png)


**----- Start of picture text -----**<br>
(a) Temporal-related task instruction<br>Please ask 5 more new questions only related to visual content. Please remember to avoid audio content. Please answer the<br>questions according to the Holistic Video Summary, or Video Shot Captions, depending on the question types.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0016-06.png)


**----- Start of picture text -----**<br>
The questions should be related to holistic video understanding, such as the topic, the progression and the story line. The<br>questions should be logical, reasonable and non-trivial. The questions should only be answered by viewing and understanding<br>the full video content. Please only return json of the questions and answers in the below format.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0016-07.png)


**----- Start of picture text -----**<br>
(b) Holistic understanding task instruction<br>**----- End of picture text -----**<br>


Please ask 2 to 5 new questions that can only be answered together with visual and audio contents. Please remember to avoid audio content. Please answer the questions according to the Holistic Video Summary, or Video Shot Captions, depending on the question types. 

The questions should be that they can not be answered without audio content. The answer can only be obtained by associating visual content and audio content. Easy questions are strictly prohibited. Remember: Objective questions are always preferred. Please avoid general questions and general answers. General questions are questions that are not specific to the video. General answers mean answers can provided without viewing the video content. An example question is: whom in what appearance speaks what content? This is only for illustration. 

- (c) Audio-related task instruction 

Figure 11: Task instruction for candidate question-answering pairs generation. 

## B PROMPTS USED IN OUR MODELS 

In this section, we elaborate on the prompts used for training and testing our models. We detail the prompts for single-shot video captioning and narration captioning in Sec. B.1. For video summarization models like SUM-shot, SUM-holistic, and SUM-text, the prompts are thoroughly explained in Sec. B.2. 

## B.1 PROMPT FOR SINGLE-SHOT VIDEO CAPTIONING 

During the training of our single-shot captioning models, we select a random text prompt for each video shot, with different model variants utilizing distinct prompts. The prompts for the single-shot video captioning model that incorporates both visual signals and ASR are depicted in Figure 14. In the figure, boldfaced text, such as “ **{asr}** ”, is replaced with specific video information. The arrangement of visual tokens and text prompts, as presented in Figure 4, is not included here for brevity. 

3 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0017-01.png)


**----- Start of picture text -----**<br>
You are a professional video creator. You are required to ask valuable and non-trivial questions according to the<br>given video. Please be notified that the video has multiple shots, containing event progression and storyline.<br>There's also a possibility that the video contains the same objects, subjects and scenes in different video shots.<br>START VIDEO!<br>{shot_caps}<br>The video content in the format of Holistic Video Summary:<br>{video_sum}<br>END VIDEO!<br>START QUESTIONING!<br>{task_inst}<br>NOTE:<br>0. Please ask questions that answers are definite. You can ask questions such as major event, theme or topical<br>category. But you should avoid questions that can be easily answered in logic and content.<br>1. Please avoid asking repeated questions.<br>2. Please do not mention the specific video shot index. It is better to describe the starting and ending of a period<br>of content by former and latter actions, events, and objects. Of course, you can use timestamps instead.<br>3. The questions together should be able to be composited to express the original video content, including event<br>content, event progression, subject appearance and topics.<br>The questions can definitely be answered by the textual video content. Please generate answers for the questions<br>based on the provided textual video content. The answers should be grounded on the video content, concisely.<br>**----- End of picture text -----**<br>


Figure 12: Prompt template for candidate question-answering pairs generation. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0017-03.png)


**----- Start of picture text -----**<br>
Q1: What is the overall theme of the video? Q1-A1: Correct<br>A1: The video introduces the different  Human  Category:<br>Generate candidate QA models of the two cars. verification holistic understanding, audio-related<br>as shown in Figure 11-12 Q2: What is the colour of the car shown in  Reading Categorization Q2-A2: Correct<br>the last scene of the video? Category:<br>A2: Black. Temporal-related<br>GPT-4<br>Watching video<br>Video-shot  MiniGPT4 generation<br>captions Human verification<br>as shown in Figure 7<br>Multi-shot  GPT generation<br>summaries Human verification<br>as shown in Figure 10<br>Video to be annotated<br>**----- End of picture text -----**<br>


Figure 13: Human annotation process of QA pairs. Please find more explanation in Sec. A.4. 

## B.2 PROMPT FOR VIDEO SUMMARIZATION 

In Sec. 3.3, we explore different model variants for video summarization, namely MiniGPT4-SUMshot, MiniGPT4-SUM-holistic, and VideoChat2-SUM-shot. For training these models, we use the same text prompt as in single-shot video captioning, shown in Figure 14. The key distinction between MiniGPT4-SUM-shot and MiniGPT4-SUM-holistic lies in the arrangement of visual tokens: MiniGPT4-SUM-shot incorporates shot-specific information such as shot number or index along with visual tokens from each shot, whereas MiniGPT4-SUM-holistic uniformly samples frames across the video. For VideoChat2-SUM-shot, the input prompt is the same to MiniGPT4-SUM-shot. 

## B.3 PROMPT FOR IN-DOMAIN VIDEO QUESTION-ANSWERING 

In our paper, we propose a unique question-answering procedure in which we generate video summaries and prompt an LLM to answer the corresponding question. The text prompt used for LLM is shown in Figure 15. 

4 

Published as a conference paper at ICLR 2025 

- Random one during training: 1. The audio transcripts are: **{asr}** . Describe this video in detail. 2. In the audio, I hear that: **{asr}** . Take a look at this video clip and describe what you notice. 3. Based on the audio, the speech content is: **{asr}** . Please provide a detailed description of the video. 

4. From the audio, I gather the content that: **{asr}** . Could you describe the contents of this video for me? 

Figure 14: Prompt during training for single-shot video captioning. 

descrip/on. You must answer my ques/on concisely. The answer is definitely contained in the provided video descrip/on. Video content is: 

**{video summary}** 

You should answer the ques/on concisely. Based on the video, please answer **{video ques0on}** 

Figure 15: Prompt used for video question answering with summaries. 

## B.4 PROMPTS USED FOR ZERO-SHOT QA 

In this section, we detail the prompts employed for the zero-shot video QA task, which is discussed in Sec. 3.4. Note that we use the same prompt with Vicuna for text summary-based video questionanswering, both zero-shot video QA and in-domain video QA. 

## B.4.1 LLM QA PROMPT 

The text prompt is shown in Figure 15, the same to in-domain video QA. It requires video summaries, which are generated from models trained with our Shot2Story data, which fullfils the definition of zero-shot tasks. To better align with the ground truth answers, we prompt the LLM to generate concise answers solely based on the provided video content. 

## B.4.2 EVALUATION PROMPT 

In our paper, we follow the same evaluation procedure as outlined in (Maaz et al., 2023), using ChatGPT-3.5 to assess the alignment of the generated answers with the given ground truth. 

## B.5 GPTV GENERATION PROMPT 

The prompt used for GPTV generation for another 90K videos is shown in Figure 16. We follow a similar prompt structure as Figure 8 to organize the different shots. In addition to the shot structure, we additionally embed speaker information and ASR texts for visual-audio correlation in generated summaries. Using Whisper-X, we extract the speaker diarization and ASR texts, which are organized in the form of “ _somebody_ speaks _something_ during _when_ to _when_ ”. The speaker diarization descriptions are then appended to the tail of video content in the prompt, in order to have speaker identification information in the generated summaries. We show two samples in Figure 17, which shows the successful speaker assignment to the visual elements and adequate details. Note this subset of data is not verified by human annotators. We only used it in the video summarization experiments in Section 3.3 of the main paper. 

## C ADDITIONAL EXPERIMENT RESULTS 

## C.1 COMPARISON OF DIFFERENT MODELS 

In this subsection, we present an example of video summarization from our Shot2Story testing split in Figure 18, using MiniGPT4-SUM-holistic, MiniGPT4-SUM-shot, and VideoChat2-SUM-shot. Both MiniGPT4-SUM-shot and VideoChat2-SUM-shot, with their access to shot information including shot count and visual tokens in shots, successfully capture the video’s storyline and transitions. For instance, they accurately depict the sequence involving a woman in the kitchen, almonds in an oven, 

5 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0019-01.png)


**----- Start of picture text -----**<br>
You are a chatbot that conducts conversations based on video contexts. These are frames of a short video. You mainly answer<br>based on the given frames. You can also answer the relevant knowledge of the person or object contained in the video. The<br>video has a high-level topic and the video content is supposed to be coherent. The video can have more than one shot and in<br>each shot different action segments and events exist. Do not include details that you are not sure of.<br>Please note that some speakers in the audio appear in the video, of whom the speech content should be described in the shot.<br>Please also note that some speakers may not appear in the video, who may be background voice or camera holders, of whom<br>the speech content should be described as narrator or background voice.<br>The video has  {num_shots}  shots. Each shots may contain multiple actions, scenes and subjects.<br>The  {shot_idx}  shot starts from  {start_duration} s to  {end_duration} s.<br>It contains frames  {frames}<br>The ASR contained in this shot is:  {shot_ASR)<br>…  # more shots in the same structure<br>{speaker_diarization}<br>Please create a detailed description that outlines the key actions and components. Please describe the appearance, clothing,<br>and surrounding environment of the characters. Also, please describe the appearance and characteristics of key objects.<br>When multiple people and objects appear, please describe them all and make their descriptions as unique as possible.<br>You should ensure the description is in narrative style and third-person view. You should describe the video coherently<br>You should describe and include the speech content into each video shot. You should be aware that the speakers in the audio<br>may not appear in the video. If the speakers in the audio does not appear in the video, you should still mention the speech<br>content if it is related to the visual content and topic. The speech content should be properly rephrased from its original ASR<br>texts, if possible. You should ignore the incomplete speech content. Do not mention the words of "ASR". DO NOT include a<br>conclusion of ASR or speakers.<br>You should not mention duration of videos and video shots. You should not mention the number of scenes, but you ought to<br>describe the content changes and transition. You ought to describe the content in happening and reasoning order. You should<br>not create a video plot out of nothing. Please describe the video coherently. You should not include a separate conclusion<br>paragraph.<br>**----- End of picture text -----**<br>


Figure 16: Prompt for video summary generation using GPTV. 

and the woman speaking to the camera. VideoChat2-SUM-shot, with its advanced vision backbone and video pretraining, captures more nuanced action details, like “using a wooden spatula to roast almonds in an oven”. MiniGPT4-SUM-holistic, while effectively identifying major content and events, falls short in accurate scene sequencing due to its lack of shot-structured visual tokens. This leads to errors in narrative order, such as reversing the scenes of “shifts back to the woman in the kitchen” and “in the final scene, the woman is seen using a wooden spatula”. 

## C.2 QA SUMMARY 

In this subsection, we present the results of zero-shot video question-answering using Vicuna v0-13B, based on textual summaries of video samples from MSRVTT-QA (Xu et al., 2017) and ActivityNetQA (Yu et al., 2019). Despite the limitations of our summarization model, which scores 8.6 in CIDEr on the Shot2Story test split (see Table 3), and the inherent challenges of the videos due to out-of-domain topics or extended durations, the summaries generated from our trained SUM-shot model largely succeed in capturing the key elements of the videos and providing relevant information. 

**MSRVTT-QA:** For instance, in Figure 19, video7089 from MSRVTT-QA portrays a TV show outside the domain of Shot2Story . This genre typically features minimal movement within individual shots, and frequent scene transitions, but a restricted variety of scenes. Yet, our generated summary aptly identifies principal elements such as the judges and contestants, actions like “engaged in a conversation” or “picking up a guitar”, and the setting of an American Idol audition. These details equip the summary to competently address questions from MSRVTT-QA. However, some gaps in detail lead to inaccuracies: of the first 10 questions for video7089, 5 are incorrectly answered due to missing information ( _e.g_ ., Q1, Q3), incorrect summary content ( _e.g_ ., Q4), or misalignment with the ground truth ( _e.g_ ., Q6, Q10). 

**ActivityNet-QA:** In Figure 20, we present the video v_mZYWfmsYQPA from ActivityNet-QA. The video’s duration is 104 seconds, which is significantly longer than the average duration in our Shot2Story . Our summary effectively identifies important elements such as the main subject’s clothing described as “dressed in a black polo shirt with a logo”, the actions including “address the 

6 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0020-01.png)


**----- Start of picture text -----**<br>
… … … … …<br>Speaker diarization : There is 1 speaker. SPEAKER_00 speaks during 0.01 seconds to 15.3 seconds.<br>Human speech : what exactly is going on. So there's sDll pressure on the clutch pedal. Like the clutch feels like it works. It's just super, super soJ. I know it's<br>not a throw bearing clip issue. Like we all know Evos have their pole style clutch. So there's<br>GPTV-generated  video summary:<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0020-02.png)


le=ers "ASYN" on it and a grey beanie. He seems puzzled, his brows furrowed as he quesEons what exactly is happening, which implies confusion or concern regarding a situaEon off-camera. TransiEoning into the second shot, focus shiJs to the inside of a vehicle, where the man's hand is interacEng with the clutch pedal, pressing it down and arEculaEng it to test its feel. He expresses that while the clutch pedal sEll has pressure, it feels unusually soJ. The footwell of the vehicle is shown, highlighEng the clutch pedal as a key object of interest. We see the hand pressing and wiggling the clutch pedal, conveying an examinaEon of its funcEonality. In the final shot, the narraEve conEnues with a further detailed explanaEon about the car's mechanical state. There's an asserEon that it is not an issue with the throw bearing clip, a component known to be problemaEc in certain car models, which are alluded to be "evos" known for their pull-style clutches. The focus remains on the vehicle interior and briefly moves to the exterior, suggesEng the man has stepped out, possibly inspecEng other parts of the car relevant to the clutch issue. The final frame ends with the man's hand gesturing, emphasizing his point about the car's mechanical specifics, possibly indicaEng the clutch's pole style. … … … … … _**Speaker diarization**_ : There is 1 speaker. SPEAKER_00 speaks during 0.8 seconds to 21.9 seconds. _**Human speech**_ : Here's what you'll need. Beef rump roast, Roma tomatoes, yellow onion, garlic, oil, a liSle cilantro, some salt and pepper, and last but certainly not least, Herde's Red Guajillo Chile Mexican cooking sauce and roasted pasilla chili sauce. To start, heat oil over medium-high heat. Now take your beef and season _**GPTV-generated**_ video summary: In the first shot of the video, a woman dressed in a professional chef's coat is standing in a kitchen, holding a large dish that appears to be filled with a prepared meal, possibly a salad. She is poised in front of a kitchen counter adorned with various ingredients and cooking utensils. The kitchen features modern appliances and a decoraEve painEng on the wall. As she stands there, an off-screen speaker begins with, "Here's what you'll need," suggesEng that the viewer will be instructed on a recipe or cooking method. TransiEoning to the second shot, the focus shiJs to the ingredients and equipment required for the recipe. Several frames show a close-up of a bowl filled with cut beef rump roast, which is the primary ingredient. The kitchen counter is neatly arranged with Roma tomatoes, a diced yellow onion, minced garlic, and a clear bo=le of oil. Fresh cilantro, salt, pepper, and two jars of Herdez brand cooking sauces, one labeled as Guajillo Chile and the other as Roasted Pasilla Chile, are also displayed. A list of ingredients appears on the screen, detailing the quanEEes and specific items needed, including "2 ½ pounds beef round rump roast, cut into 1-1½ inch cubes" and the other components menEoned. The speaker narrates the list of ingredients, including the beef rump roast, Roma tomatoes, yellow onion, garlic, oil, cilantro, salt and pepper, as well as the specific Herdez Mexican cooking sauces. The video transiEons once more into the third shot, where the cooking process begins. A pot is centered on a stove, and the chef pours oil into it. Following this, the chef seasons the beef with salt, preparing it for cooking. The acEon implies that the beef will soon be browned in the oil. The speaker's instrucEons -- to heat oil over medium-high heat and to season the beef with salt -- align with the visual acEons, demonstraEng the iniEal steps in preparing the dish. The viewer is now both informed of the ingredients and shown the beginnings of the cooking process, paving the way for the conEnuaEon of the recipe.. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0020-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0020-05.png)


Figure 17: Samples of video summaries generated by GPTV. In both cases, GPTV successfully correlates the audio speaker and the person in the video, _i.e_ . “He expresses that” and “an off-screen speaker begins”. Moreover, GPTV-generated summaries capture the overall action and event flow with the help of shot structure, and the essential details in the video. 

audiance” and “demonstrate the proper grip”, and the setting, noted as “a ping pong table in an indoor sports facility”. These comprehensive details enable the Vicuna model to correctly answer most questions from ActivityNet-QA. Despite these insights, some omissions and discrepancies in the summary contribute to inaccuracies in answering. Specifically, four out of ten questions are answered incorrectly due to either a lack of specific details or mismatches with the ground truth, as seen in questions Q2, Q5, Q6 and Q9. 

**Shot2Story-QA:** In Figure 21, we present a video sample from our Shot2Story-QA dataset. The questions address audio-related (Q1, Q4), holistic understanding (Q2, Q3), and temporal-related (Q4) aspects. With shot structure enabled in VideoChat2-SUM-shot, the generated video summary includes actions and transitions, such as _begins with a close-up shot of a person holding a pedal_ , _transitions to a man_ , _then shifts to a wider shot_ , _returns to a close-up shot of the man_ , and _concludes with a wider shot of the man_ . However, the generated summary falls short in capturing detailed information about the actions and their orders (e.g., "Taking out a piece of paper" in Q3), and fails to align visual and audio content (e.g., different components in Q1 which have been mentioned in ASR texts). As a result, Vicuna struggles to answer these questions, highlighting the challenges posed by our Shot2Story-QA benchmark. To further improve the model performance, the summarization 

7 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0021-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0021-02.png)


… … … … … _**Human speech**_ : However, there are a couple of tricks that you need to know in order to be successful. The first is that you must roast your almonds that helps to bring out not only the flavors but the oil as well. _**MiniGPT4-SUM-holistic**_ video summarization: the video begins with a woman in a well-equipped kitchen, addressing the camera with a smile. she is wearing a green t-shirt and is surrounded by brown wooden cabinets and appliances. as she speaks, she reveals that there are a few tricks that people need to know in order to be successful. the scene transi9ons to a close-up of a clear, transparent blender. the woman's voice instructs viewers to roast their almonds to bring out not only the flavors but the oil as well. the video then shiBs back to the woman in the kitchen, now holding the transparent blender. she con9nues to speak to the camera, emphasizing the importance of roas9ng the almonds. in the final scene, the woman is seen using a wooden spatula to scoop up the roasted almonds from a baking tray. she then places the tray in the oven to con9nue roas9ng the nuts. throughout the video, the woman's words and ac9ons provide a comprehensive guide on how to successfully roast almonds. _**MiniGPT4-SUM-shot**_ video summarization: the video begins with a woman in a kitchen se2ng, wearing a blue top and blonde hair. she is holding a clear plas9c container in her hands, which contains almonds. as she speaks to the camera, she men9ons a couple of tricks that people need to know in order to be successful. the scene transi9ons to a close-up of a plate of almonds being cooked in an oven using a shovel. the woman's voice can be heard in the background, explaining that roas9ng the almonds helps to bring out not only the flavors but the oil as well. in the final scene, the woman con9nues to hold the clear plas9c container filled with almonds. she is seated in a well-equipped kitchen, with a blender placed next to her. as she speaks to the camera, she holds a bowl of almonds in her other hand. the video seems to be a tutorial or a cooking demonstra9on, with the woman sharing her knowledge and 9ps on roas9ng almonds. _**VideoChat2-SUM-shot**_ video summarization: in the video, a woman with blonde hair, wearing a light blue top, is standing in a kitchen with wooden cabinets in the background. she is holding a glass bowl filled with almonds and a blender. as she talks to the camera, she men9ons that there are a couple of tricks that people need to know in order to be successful. the scene then transi9ons to a close-up shot of a person using a wooden spatula to roast almonds in an oven. the woman's voice can be heard in the background, explaining that roas9ng the almonds helps to bring out not only the flavors but also the oil. finally, the video returns to the woman in the kitchen, who con9nues to hold the glass bowl filled with almonds and the blender. _**Ground truth**_ video summary: the video begins with a woman, dressed in green short sleeves, standing in front of a kitchen console. she is engaged in a conversa9on, presumably about cooking 9ps, as she stands next to a bowl of almonds and a transparent mixer. as she speaks, she picks up the bowl of almonds, emphasizing the importance of knowing certain tricks for successful cooking. the scene transi9ons smoothly to a close-up of an oven pan filled with almonds. a wooden spoon is used to s9r the almonds around in the pan. the woman's voice con9nues in the background, explaining the benefits of roas9ng almonds. she men9ons that this process helps to enhance the flavors of the almonds, sugges9ng that this is one of the tricks she was referring to earlier. the video concludes with the woman once again standing in front of the kitchen console, con9nuing her discussion. she holds the bowl of almonds in her hands, possibly preparing to demonstrate the next step in her cooking process. throughout the video, the woman's ac9ons and dialogue suggest that she is sharing her knowledge and 9ps about cooking, specifically focusing on the use of almonds. 

Figure 18: Example for video summarization using MiniGPT4-SUM-holistic, MiniGPT4-SUM-shot and VideoChat2-SUM-shot. All three models effectively grasp the video’s main topic and content. Please find explanations in Sec. C.1. 

8 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0022-01.png)


**----- Start of picture text -----**<br>
… … … … …<br>video7089.mp4<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0022-02.png)


**----- Start of picture text -----**<br>
VideoChat2-SUM-shot video summarization:<br>**----- End of picture text -----**<br>


in the first shot of the video, three individuals are seated at a desk in what appears to be a television studio with a cityscape view in the background. the man on the left is wearing a dark shirt, the woman on the right is dressed in a light blue blouse, and the man on the right is wearing a black t-shirt. they are engaged in a lively conversation, with the man on the left gesturing with his hands as he speaks. the woman on the right seems to be listening attentively, while the man on the right listens as well. the conversation seems to revolve around a topic related to the city, with the man on the left expressing his enthusiasm for it. transitioning to the second shot, the scene shifts to a stage where two men are standing. the man on the left is dressed in a light blue blazer and a dark shirt, while the man on the right is wearing a dark shirt and khaki pants. the man on the left is holding a ukulele, and they both appear to be in the middle of a performance or presentation. the man on the left is smiling and seems to be enjoying the moment, while the man on the right stands with his hands on his hips, looking at the audience or perhaps the man with the ukulele. in the third and final shot, the video returns to the studio setting with the three individuals from the first shot. the man on the left is still wearing his dark shirt, the woman on the right is in her light blue blouse, and the man on the right is in his black t-shirt. the man on the left is gesturing with his hands again as he speaks, while the woman on the right and the man on the right listen attentively. the woman on the right seems to be responding to something the man on the left is saying, and the man on the right listen intently. 

**Ques%ons** : **Answers** : Q1: who speaks to tv show judges? GT1: person Q2: who do three judges talk to? GT2: contestant Q3: what is a black male laughing on? GT3: idol Q4: what talks to a performer? GT4: panel Q5: what did the man audiKon for? GT5: idol Q6: who does a panel talk to? GT6: performer Q7: what do judges talk on? GT7: idol Q8: what shows judges of a singing compeKKon? GT8: video Q9: how many contestants are standing before the judges at american idol? GT9: two Q10: what is going on? GT10: reality 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0022-05.png)


**----- Start of picture text -----**<br>
Vicuna answers using summaries  & GPT evaluation :<br>**----- End of picture text -----**<br>


A1: Based on the video description, it is not clear who is speaking to TV show judges. 

A2: The three judges in the video are the man in the black shirt and khaki pants, who claps his hands and smiles, and the woman in the blue and silver-colored shirt, who looks at the camera. 

A3: I'm sorry, but I cannot identify a specific person laughing in the video, as there is no mention of a person laughing in the description. The video appears to be focused on the interactions and performances of the individuals in front of the American Idol backdrop. 

A4: The man in the white shirt and black pants is a performer. 

A5: The man in the white shirt and black pants auditioned for American Idol. 

A6: Based on the video, it appears that a panel talk is happening between the man in the black shirt and the woman in the blue shirt, who are sitting behind a table with a city skyline in the background. The man occasionally claps his hands, and the scene transitions to two men standing next to a large screen with \"American Idol\" written on it. One man, dressed in a white shirt and black pants, picks up a guitar and starts playing, while the other man, wearing a black shirt and khaki pants, claps his hands and smiles. In the final scene, the man in the white shirt and black pants continues to play the guitar, while the woman in the blue and silver-colored shirt looks at the camera. The man in the black shirt and khaki pants claps his hands and smiles. 

A7: Judges talk about performances in front of the American Idol backdrop. A8: The video shows judges of a singing competition. 

A9: Two contestants are standing before the judges at American Idol. 

A10: It appears that the video is a behind-the-scenes look at a recording session for a musical performance, possibly for a talent competition such as American Idol. The individuals shown in the video are a group of musicians who are preparing for a performance in front of a panel of judges. The man in the black shirt and khaki pants is likely a music producer or judge, while the man in the white shirt and black pants is likely a contestant on the show. The woman in the blue and silver-colored shirt is likely another member of the group or a crew member. 

Figure 19: Example for video question-answering using summaries - MSRVTT. We show the zeroshot video summary using our trained VideoChat2-SUM-shot model. The summary can capture most of the video, such as the subject, the action and the transitions. We also show the answers from Vicuna to the first 10 questions and the evaluation results from GPT. Please find more explanations in Sec. C.2. 

9 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0023-01.png)


**----- Start of picture text -----**<br>
… … … … …<br>v_mZYWfmsYQPA.mp4  duration: 104 seconds<br>VideoChat2-SUM-shot video summarization:<br>**----- End of picture text -----**<br>


in the opening scene of the video, a man stands on a ping pong table in an indoor sports facility. he is dressed in a black polo shirt with a logo on the left chest, and he holds a ping pong racket in his right hand. the background is a vibrant blue, and there's a sign that reads welcome to the world of ping pong. as he speaks, he introduces the topic of the video, which is about the basics of ping pong. transitioning to the second scene, the same man continues to address the audience, now holding a red ping pong ball in his hand. he emphasizes the importance of understanding the basics of ping pong, suggesting that this is a fundamental aspect of the sport. in the third scene, the man is seen again on the ping pong table, this time preparing for a serve. he demonstrates the proper grip and stance for a serve, which is a crucial part of the game. the video then cuts to the fourth scene, where a different man appears on the ping pong court. he is wearing a white t-shirt emblazoned with the website www.pingponginfo.com and the text ping pong information . this man speaks about the importance of proper footwork in ping pong, highlighting the role it plays in the game. finally, in the fifth and last scene, the same man from the first scene reappears. he is still on the ping pong table, and he poses a question to the viewers, asking if they have a question about ping pong. this serves as an invitation for the audience to engage with the content and potentially learn more about the sport. throughout the video, the speakers, including the man on the ping pong table and the man in the white t-shirt, provide informative commentary on the basics of ping pong, emphasizing the importance of understanding the fundamentals of the game. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0023-03.png)


**----- Start of picture text -----**<br>
Questions : Answers :<br>Q1: what is the person in the video doing?  GT1: play table tennis<br>Q2: is the person in the video wearing a watch?  GT2: yes<br>Q3: is the person in the video wearing a hat?  GT3: no<br>Q4: what is in front of the person in the video?  GT4: table tennis table<br>Q5: what happened before playing table tennis?  GT5: introduction of pang ball<br>Q6: where is the video taken?  GT6: indoor<br>Q7: how safe is this sport in the video?  GT7: security<br>Q8: what benefit can this exercise bring?  GT8: reaction capacity<br>Q9: how many people are there in video?  GT9: 1<br>Q10: what color of the clothes does the person wear in the video?  GT10: black<br>Vicuna answers using summaries  & GPT evaluation :<br>A1: The person in the video is demonstrating the basics of ping pong, including proper grip, stance, serve, and footwork.<br>A2: No, the person in the video is not wearing a watch.<br>A3: No, the person in the video is not wearing a hat.<br>A4: In the video, the person is standing on a ping pong table in an indoor sports facility.<br>A5: Based on the video, it is not clear what happened before the man started playing table tennis.<br>A6: The video is taken in an indoor sports facility with a vibrant blue background and a sign that reads Welcome to the world of ping pong.<br>A7: Based on the video, ping pong is a safe sport. The video shows people of different ages and skill levels playing the game without any major<br>incidents, and the speakers emphasize the importance of safety in the sport, such as proper grip and stance for serves.<br>A8: The benefit of this exercise is to learn the basics of ping pong, including proper grip and stance for a serve, the importance of footwork in<br>the game, and the importance of understanding the fundamentals of the sport. By watching the video, viewers can learn about the basics of<br>ping pong and potentially gain a deeper appreciation for the sport.<br>A9: There are two people in the video.<br>A10: The person in the video wears a black polo shirt with a logo on the left chest.<br>**----- End of picture text -----**<br>


Figure 20: Example for video question-answering using summaries - ActivityNet. We show the zero-shot video summary using our trained VideoChat2-SUM-shot model. The summary can capture most of the video, such as the subject, the action and the transitions. We also show the answers from Vicuna and the evaluation results from GPT. Please find more explanations in Sec. C.2. 

10 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos_images/iclr_2025_Shot2Story_A_New_Benchmark_for_Comprehensive_Understanding_of_Multi-shot_Videos.pdf-0024-01.png)


**----- Start of picture text -----**<br>
… … … … …<br>[Human speech] [: We get pedals disc for the front wheel, im assuming screws, probably for the disc bell brackets skewer and an instruc:on book hang on.]<br>VideoChat2-SUM-shot video summarization:<br>the video begins with a close-up shot of a person holding a pedal in their hand. the scene then transi7ons to a man in a black t-shirt standing in<br>a garage filled with various items, including a bicycle and a motorcycle. he is engaged in a task at a workbench, where a bicycle is mounted on a<br>stand. as he works, he men7ons that he is using a pedal disc for the front wheel. the video then shi=s to a wider shot of the man in the black t-<br>shirt, s7ll standing in the garage filled with items. he con7nues his task at the workbench, where a bicycle is mounted on a stand. the scene<br>then returns to a close-up shot of the man in the black t-shirt, s7ll working at the workbench. the video concludes with a wider shot of the man<br>in the black t-shirt, s7ll standing in the garage filled with items. he con7nues his task at the workbench, where a bicycle is mounted on a stand.<br>Questions : Answers :<br>Q1: What are the different components the man handles from the  GT1: A pedal, disc brake pad, screws, bell, brackets, skewer, paper,<br>beginning to the end of the video? and instruc7on book.<br>Q2: How does the setting change from the start to the end of the  GT2: The seLng remains the same, in the garage with the half-<br>video?  finished bike.<br>Q3: What is the man's immediate action after handling the skewer? GT3: Taking out a piece of paper.<br>Q4: What is the main activity being depicted in the video? GT4: The items for the bike assembly or prepara7on.<br>Vicuna answers using summaries  & GPT evalua8on :<br>A1: The different components the man handles from the beginning to the end of the video are: a piece of equipment, a white object, and the<br>pedals disc for the front wheel.<br>A2: The seLng changes from a workshop or laboratory to a garage.<br>A3: The man is placing the skewer back into the box.<br>A4: The main ac7vity being depicted in the video is the demonstra7on of tool prepara7on for bikes in a garage seLng.<br>**----- End of picture text -----**<br>


Figure 21: Example for video question-answering using summaries - Shot2Story-QA. We show the in-domain video summary using our trained VideoChat2-SUM-shot model. The summary can capture the shot transitions and major actions. However, as indicated by the QA results, the detailed actions and their order fail to match with the groundtruth, such as whether the immediate action after handing the skewer. Additionally, it fails to provide adequate information regarding the consistent scene between shot transitions, such as “at a workbench” and “in the garage”. Please find explanations in Sec. C.2. 

11 

Published as a conference paper at ICLR 2025 

model could be enhanced with detailed information with denser frames, fine-grained matching of audio and visual cues, and tracking of the same objects across the video. We leave it as future work. 

## D ADDITIONAL IMPLEMENTATION DETAILS 

**Video-Shot Captioning.** For each video shot, we uniformly sample 4 frames. For testing, a consistent text prompt is used, _i.e_ ., “The audio transcripts are asr. Describe this video in detail.”. The maximum number of new tokens generated by the LLM is capped at 180 for both training and inference. 

**MiniGPT4-SUM-shot, MiniGPT4-Holistic and VideoChat2-SUM-shot.** In two SUM-shot models, 4 frames per video shot are sampled uniformly. In MiniGPT4-SUM-holistic, 16 frames per video clip are sampled. The rationale behind sampling 16 frames in a holistic approach is based on our dataset’s average of 4 shots per video, aligning with the SUM-shot approach of 4 frames per shot. For both training and inference, the LLM’s maximum new token count is set at 600. A consistent text prompt, _i.e_ ., “The audio transcripts are {asr}. Describe this video in detail.”, is used during inference. 

For video-shot captioning and video summarization tasks, both models are trained on 8 _×_ 2 A100-80G GPUs using Pytorch. The captioning model is trained for 10 epochs, with the best-performing checkpoint on the validation set used for test performance reporting. To prevent overfitting, text prompts are randomly sampled for each sample, as detailed in Sec. B. 

**VAST.** We tune the model following the official instructions by inputting our video frames, audios and ASR texts. The optimizing target is the concatenation of our single-shot visual caption and single-shot narration caption. During inference, we input the corresponding modalities, in case of two different model versions we trained (visual + audio, or visual + audio + ASR texts), into the model and use the directly generated captions for evaluation. The model is trained on 8 _×_ 4 A100-80G GPUs for 3 epochs, maintaining other hyperparameters at their default values from the original configuration. 

**Video-ChatGPT.** Consistent with SUM-holistic, we uniformly sample 16 frames for both training and inference. Our prompt setup excludes ASR for video summarization. The training is conducted over 3 epochs with a learning rate of 2e-6, and we retain other hyperparameters at their default settings as specified in the original repository. 

**Video Question-Answering with Summary.** For the MSRVTT-QA (Xu et al., 2017) and ActivityNetQA (Yu et al., 2019) datasets, we generate video summaries using our trained SUM-shot model, employing only visual tokens during inference. Upon generating these summaries, we integrate them with individual questions from their corresponding videos into the prompt format displayed in Figure 15. This integrated content is then processed through Vicuna (Chiang et al., 2023) to obtain answers. The evaluation of these results is carried out following the methodology outlined in (Maaz et al., 2023). 

## E BROADER IMPACT 

**Data Limitations and Ethical Considerations.** We provide cropped multi-shot videos instead of the original videos. Users can also turn to download these from original sources. Given HD-VILA100M (Xue et al., 2022)’s long-standing public availability, we assess a low risk of the currently available videos being removed in the near future. Additionally, our meticulous manual annotation process is designed to avoid any ethical or legal violations. Specifically, our videos don’t have personally identifiable information or offensive content, which is ensured by the manual annotation process. The authors will take the responsibility of long-term maintenance. 

**Human Rights in Annotation Process.** We have conscientiously structured the annotation process to ensure fair workloads and equitable compensation for annotators, upholding human rights standards. 

**Scope of Conclusions.** It is important to recognize that experiments and data, including ours, might only represent a subset of universal realities. Nevertheless, given the wide range of categories covered in our videos, we believe our conclusions offer a robust understanding applicable to various multi-shot 

12 

Published as a conference paper at ICLR 2025 

video scenarios and durations. These findings, while specific to our dataset, provide significant insight into the broader field of video analysis. 

**Usage of Language Models.** Our use of the LLaMA model (Touvron et al., 2023) from Meta is authorized for research purposes. Those intending to use our model post-release should ensure that they have the necessary permissions and adhere to usage restrictions. We express deep respect for the work of developers and contributors, recognizing their integral role in advancing language modelling and multi-modal learning. 

**Future Research and Development.** We release both our code and dataset. This is intended to encourage further research and enable others to build upon our work. Although our current experiments require up to 8 _×_ 2 A100-80G GPUs, we are aware this may be a limitation. Consequently, we plan to focus future efforts on adapting these experiments to be compatible with a single node of 8 A100 GPUs. It’s important to note that fitting the experiments within an 8 GPU framework is not the primary focus of this paper, but we consider it a crucial step towards making our research more accessible and inclusive for a wider array of research groups. 

13 

