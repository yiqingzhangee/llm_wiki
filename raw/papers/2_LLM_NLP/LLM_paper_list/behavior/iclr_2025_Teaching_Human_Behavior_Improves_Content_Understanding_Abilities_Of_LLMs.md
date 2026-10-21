# - TEACHING HUMAN BEHAVIOR IMPROVES CON TENT UNDERSTANDING ABILITIES OF VLMS 

**Somesh Singh** _[⋆]_ **, Harini S I** _[⋆]_ **, Yaman K Singla** _[⋆]_ **,** 

**Veeky Baths , Rajiv Ratn Shah , Changyou Chen , Balaji Krishnamurthy** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs_images/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs.pdf-0001-04.png)


Adobe Media and Data Science Research (MDSR), IIITD, SUNY at Buffalo, BITS Pilani behavior-in-the-wild@googlegroups.com 

## ABSTRACT 

Communication is defined as “ _Who_ says _what_ to _whom_ with _**what** effect_ .” A message from a communicator generates downstream receiver effects, also known as behavior. Receiver behavior, being a downstream effect of the message, carries rich signals about it. Even after carrying signals about the message, the behavior signal is often ignored while training vision language models. We show that training VLMs on receiver behavior can actually help improve their content-understanding abilities. We demonstrate that training VLMs to predict receiver behaviors, such as likes, comments, and replay graphs, which are available at scale, enhances the VLM’s performance across a broad range of downstream content understanding tasks. We show this performance increase over 6 types of behavior, 46 different tasks covering image, video, text and audio over 26 benchmark datasets across both 0-shot and fine-tuning settings, outperforming many supervised baselines on diverse tasks ranging from emotion recognition to captioning by upto 150%. We note that since receiver behavior, such as likes, comments, and replay graphs, is collected by default on the internet and does not need any human annotations to be useful, the performance improvement we get after training on this data is essentially free-lunch. We also release BLIFT, our Behaviour-LLaVA IFT dataset comprising of 730k images and videos with their receiver behavior collected from multiple platforms on which we train our models to achieve this. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs_images/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs.pdf-0001-08.png)


**----- Start of picture text -----**<br>
(What) Channel: Receiver Effect<br>Video Content YouTube (with what effect)<br>Arnold and Danny00:52 That ending  are always a put a smile on my face beautiful duet. . Temporal Understanding<br>Receivers  (To whom) :  Character Understanding<br>YouTube Subscribers The sheep part was hilarious<br>Cognitive Understanding<br>Views: 807k Won't make me change insurance companies,<br>but it's a great commercial. User Opinion Understanding<br>Likes: 4.3k Sad that State Farm no longer covers us in World Knowledge<br>Speaker State Farm Insurance (Who says) : Comments: 352Replay graph “Neighbah, Papah, Baaah, Labah, Backstabah”… Lol!!Arnie just can't get that hard R out.! Bay Area! 😂 😂 Context Understanding<br>Most replayed moment is at 0:52<br>**----- End of picture text -----**<br>


Figure 1: The diagram depicts the five factors of communication in the context of an example YouTube video https://www.youtube.com/watch?v=eT8hO4e2iTM and where lies the free lunch. The receiver effect is not used while training Large Vision and Language Models. However, it contains many important signals that can help in understanding the content. The figure shows several comments containing temporal, cognitive, character, context, and user opinion information useful for understanding the video. 

## 1 INTRODUCTION 

Communication is defined by five factors: sender, message, channel, receiver, and behavior (Shannon & Weaver, 1949; Lasswell, 1948; 1971). Lasswell (1948) encoded these five factors in the phrase, “ _Who_ says _what_ to _whom_ with _what effect_ .” Human behavior occurs as a downstream artifact in the process of communication. Behavior is produced by the receiver as a response to the message sent by the sender. Being a downstream effect, behavior can help us infer important signals about the message itself. These signals, if properly harnessed, should be able to increase performance on the 

> _⋆_ Equal Contribution. Contact behavior-in-the-wild@googlegroups.com for questions and suggestions. 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs_images/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Simulate 3 user Simulate the no. 1. I just lost my grandpa a few weeks ago. I sobbed at 00:50. I<br>comments on this of views, likes, would perform for him every time he couldn’t make it. I miss<br>video and comments on him 💔<br>this video Who does the girl 2. The smile on that girl’s face after the final stop always makes<br>in the video me cry. Like she’s proud she could put on a show for the only<br>perform for at the audience that matters. A moment that even money can’t buy<br>end of the video? Behavior 3. Proudly built in GEORGIA!<br>Which primary LLaVA Views: 297k A Girl performs<br>sport activity is Likes: 1.5k for her Ice skating<br>being shown in Comments: 250 grandfather<br>the video?<br>Emotion: Persuasion<br>What are the Affection, Joy, Strategy: Social<br>emotions Pride Identity, Emotion<br>depicted in the<br>Write a 1-line What is thepersuasion video? In a heartwarming gesture, a girl's father uses a Kia EV to drive<br>story for thisvideo.  strategy used inthe video? through heavy snow so she can perform for her wheelchair-bound grandfather, bringing joy to both.<br>**----- End of picture text -----**<br>


Figure 2: Behavior-LLaVA is trained to answer behavioral questions like simulating user comments and likes on the video. The model, once trained, shows superior performance than LLaMA-Vid and other VLMs on content-related tasks like emotion recognition, action recognition, question answering, persuasion strategy classification, _etc_ . The original video was showcased in SuperBowl-2024 and is posted on YouTube on the URL https://www.youtube.com/watch?v=OU7BJc96lI4. The video is titled “Perfect 10: The Kia big game commercial featuring the 2024 Kia EV9” by Kia America. 

message understanding tasks popular in NLP and CV, like question answering, sentiment analysis, topic classification, _etc_ . Despite this, behavior data is considered noise and is ignored while training large language models (Biderman et al., 2022; Penedo et al., 2023) and also large vision and language models (Liu et al., 2023a; Zhu et al., 2023). In this paper, we explore this line of thought more. 

Humans produce two kinds of behavioral signals upon observing a message (Bertenthal, 1996; Prinz, 1997): perceptual signals and actions as behavior. Perceptual signals, like seeing, touching, and hearing, help a receiver primarily sense the world around her, ultimately guiding her actions. Actions are how a receiver acts on the outside world. The signals produced by the human receiver upon receiving a message carry information about the message itself (Fig. 1). For instance, if a person’s heartbeat rises upon watching a movie scene, it can help us infer that perhaps the scene was an exciting scene (Dzedzickis et al., 2020). Similarly, regressing while reading is indicative of important or confusing phrases (Bicknell & Levy, 2011). In these cases, perception behavior helps us derive inferences about content. In a similar vein, the actions a person performs after watching a movie, such as comments and likes, carry signals about the movie (Fig. 1, 2). 

Expanding on these ideas, prior literature has shown that harnessing perceptual signals, like eye movements, saliency, keystrokes, mouse movements, and FMRI, by modeling them together with content understanding tasks can improve both NLP and CV tasks. For instance, integration with perception signals causes performance improvement in tasks like visual and natural language question answering (Patro & Namboodiri, 2018; Khurana et al., 2023; Sood et al., 2020), text and image sentiment analysis (Khurana et al., 2023; Barrett et al., 2018; Fan et al., 2018), natural language inference (Khurana et al., 2023), part-of-speech identification (Barrett et al., 2016a;b), named entity recognition (Hollenstein et al., 2019; Hollenstein & Zhang, 2019), syntactic parsing (Plank, 2016), image captioning (Cornia et al., 2018), and visual object detection (Wang et al., 2018; Kruthiventi et al., 2016). 

While the initial studies show that perceptual signals have much promise for improving downstream content understanding, they have a few significant issues due to which integrating human perception has not seen wide adoption in training LLMs. These perceptual signals can only be collected in lab settings requiring specialized lab equipment and are thus expensive to collect and thus are also limited in number. For example, the largest datasets containing the human processing signals are SALICON (Jiang et al., 2015) and Cheng et al. (2014) for visual saliency (10k images each), CELER (Berzak et al., 2022) and Dundee corpus (Kennedy et al., 2013) containing eye movements over 28k sentences and 20 news articles respectively, and Dhakal et al. (2018) containing keystroke patterns over 1.5k sentences. Clearly, these datasets, while making important contributions, do not scale to the level at which today’s large language models are trained (trillions of natural language and image tokens). 

On the other hand, actions (the other type of behavioral signals produced by a human receiver) are collected at a large scale in the form of digital analytics. Examples of this kind of data are likes, views, shares, comments, and purchase histories on images, tweets, videos, webpages, and other kinds of media. Action data has a much broader representation than is possible in lab settings, is available on more diverse content, and is much cheaper to collect than using specialized lab equipment. At the 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs_images/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Image Memorability Video Emotion<br>Video Persuasion Behaviour-LLaVA<br>Video Memorability LLaMA-VID<br>Ad-LLaVA<br>Ad Understanding<br>Video<br>Moment Retrieval Image<br>LVU Audio<br>Highlight Detection<br>Audio Summarization<br>Video Summarization<br>Dense Caption<br>VQA Image Emotion<br>**----- End of picture text -----**<br>


Figure 3: Behaviour-LLava achieves much higher zero-shot performance compared to Ad-LLaVA and the base model LLaMA-VID across a diverse suite of image, video, and audio benchmarks. 

same time, actions have not been much investigated in the literature for their potential to improve downstream content understanding. 

Therefore, in this paper, we make initial efforts to collect and understand digital analytics at scale with the aim of integrating them with VLMs to improve their downstream content understanding capabilities. We introduce methods for filtering and cleaning behavioral data and then propose tasks for large language and vision models, leading to improvements in language and visual content understanding tasks. For this, we look to Reddit and YouTube as two major sources of visual content and human behavior in the form of viewer comments, likes, replay graphs, and upvotes. From Reddit, we collect 5 million images and videos along with their upvotes and top-upvoted comments from two major subreddits (r/pics and r/videos). Similarly, from YouTube, we collect 2.2 million videos from 30000 channels along with their likes, views, replay graphs, and top user comments. After extensive filtering and cleaning, we are left with 730k samples of videos and images across the two platforms which we use for the next steps. 

After collecting user behavior over image and video content, we design tasks to teach large vision and language models (VLMs) to simulate user behavior. For this, we use an instruction fine-tuning format. Given a video or an image and the other metadata like time of post and channel, we ask the model to simulate user behavior of likes and comments. See Fig 2, Listing 1 for examples. We choose LLaMA-Vid (Li et al., 2023b) as our base model to teach it the user behavior. We call the resultant model Behavior-LLaVA (Large Language and Vision Assistant) (Liu et al., 2023a). We test Behavior-LLaVA on a diverse variety of tasks, evaluating its capabilities on image, video, text, and audio understanding tasks. We compare Behavior-LLaVA against its base model, LLaMA-Vid, and other supervised baselines. Further, to show the impact of behavior, we train another version of LLaMA-Vid, where we train it on the same set of videos and images as Behavior-LLaVA but do not include behavior information. We call this model Ad-LLaVA. 

We make the following contributions with this work: 

1) **Behavior-LLaVA Instruction Fine-Tuning:** We explore the idea of learning human behavior, resulting in better content understanding. We test this for action-level behavior data such as receiver comments, likes, and replay graphs. We collect a dataset called **BLIFT** , consisting of 400k images and 330k videos, along with their receiver behavior. Then, LLaMA-Vid is trained for the task of predicting receiver comments and upvotes given a media (a video or an image) (Listing 1). We show that using this simple task formulation over behavioral data collected in the wild, results in performance improvement over a hierarchy of tasks. We get improvements over the base LLaMA-Vid across 46 tasks over 26 benchmark datasets in both zero-shot and fine-tuned settings. We show this over low-level content understanding tasks like object and activity recognition and also over high-level tasks like topic and emotion detection. Through this, we propose a scalable approach to increase the content understanding abilities of VLMs, requiring minimal cost and no architectural changes. 

2) **AdLLaVA: Disentangling the effect of content and behaviour:** To disentangle the effect of training LLaMA-Vid on additional image and video data from the effect of training on behavior data, we train LLaMA-Vid on BLIFT’s videos and images without including behavior. We call this model Ad-LLaVA. We show that Ad-LLaVA shows equivalent performance as its base model LLaVA-Vid; however, Behavior-LLaVA performs better than both Ad-LLaVA and LLaMA-Vid, thus highlighting the importance of behavior data and instruction fine-tuning on behavior data. 

3 

3) **Perception vs Action:** We also show an ablation of Behavior-LLaVA across different kinds of behavior. We try out the perception behavior of saliency prediction over images and five types of action-level behavior over images and videos. We find that perception-level behavior does not result in significant performance improvements; however, action-level behavior shows improvements across all the tasks. We posit that one reason for this could be due to the scale for which action-level data is available (Table 11). While perception behavior is mostly collected in lab settings, action-level behavior data is diverse, can be collected in a scalable manner automatically and cheaply. 

## 2 METHODOLOGY 

In this section, we introduce our approach to train Behavior-LLaVA. Since no publicly available corpus consists of behavior together with image and video content, we first introduce our instruction fine-tuning dataset, “Behavior-LLaVA Instruction Fine-Tuning dataset” (BLIFT). Next, we introduce our methodology to train Behavior-LLaVA. Finally, we report the results of testing Behavior-LLaVA’s capabilities on a hierarchy of tasks. The tasks cover low-level media understanding tasks like object and activity detection, high-level media understanding tasks like emotion, topic, and persuasion strategy classification. 

## 2.1 BLIFT DATASET 

Given the abundance of media and behavioral data and its accessibility, our data collection relies on two primary sources: Reddit and YouTube. These platforms share similarities in terms of hosting media content (images and videos) and providing user engagement metrics in the form of Reddit upvotes and comments, and YouTube likes, views, comments, and replay graphs. Here, we delineate the process involved in constructing the instruction fine-tuning dataset, which we term as the Behavior-LLaVA Instruction Fine-Tuning (BLIFT) dataset. 

## 2.1.1 DATA FROM REDDIT 

To collect a substantial corpus of diverse images and videos, we targeted at two specific subreddits, namely r/pics and r/videos. Established over 15 years ago, these subreddits had a user base exceeding 20 million during the data collection period, with an average of over 5,000 online users concurrently. Notably, due to stringent content moderation guidelines (Reddit, Inc., 2024; red, 2024a;b;c) and the exclusive focus of these subreddits on media content, they offer a rich variety of content devoid of thematic biases. Our data collection spans until January 2022, during which the activity on these subreddits witnessed a notable decline following several policy adjustments and user protests (Hern, 2023; Economist, 2023). 

To ensure data quality and relevance, we executed a series of filtering steps on the posts and comments from these subreddits. Initially, we excluded posts predating February 2015 from r/pics, coinciding with the implementation of a rule requiring images without digital/overlay text (red, 2024a;b). This filtering step resulted in the exclusion of 3.1 million images and 2 million videos. Subsequently, considering the sustained popularity of both subreddits, with rankings within the top 20 since 2017 and consistent membership exceeding 20 million, we confined our dataset to posts from January 2018 onwards. This selection process yielded 1.4 million images and 1.1 million videos. 

Further refinement of the dataset involved removing posts and comments marked as NSFW, BOTgenerated, or [deleted], along with eliminating duplicate images and videos. This curation step reduced the dataset to 876,000 images and 983,000 videos. To address redundancy in comments, we excluded those comprising fewer than three words and employed TF-IDF-based deduplication with a similarity threshold of 0.6, determined through manual observations. 

Following these steps, posts with fewer than two comments were filtered out, resulting in a dataset comprising 631,000 images and 397,000 videos. Additionally, videos exceeding a duration of 500 seconds were omitted, leaving 221,000 videos for analysis. Notably, images not directly hosted on Reddit were excluded due to scraping and copyright limitations. Similarly, for r/videos, only videos hosted on YouTube were considered. It is pertinent to mention that approximately 51% of YouTube videos collected during this period were either made private/unlisted or removed, resulting in 400,000 images and 80,000 videos, accompanied by 1.5 million and 312,000 comments, respectively. These 

4 

comprehensive filtering steps ensured the construction of a diverse and relevant dataset for fine-tuning instruction-based models. 

Listing 1: Behavior Instruction fine-tuning template for the video: https://www.youtube.com/ watch?v=BKPQkjRF4yY 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs_images/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs.pdf-0005-02.png)


<SYSTEM>You are an AI visual assistant. You are given a detailed description of a media, followed by the actual media. Answer all questions as if you are seeing the media.</ SYSTEM> <USER> The video advertisement is titled "Gatorade | Make Your Own Footsteps with Suni Lee" for the brand Gatorade. The audio in the ad says "[ASR HERE ...]". Analyze this video deeply, then write scene by scene description of the video and answer the following questions. What percentage of viewers would like this video, and what would be the top-5 popular comments on this video? What would the replay graph values for each scene be? <video>...</video> </USER> <ASSISTANT> The scene-by-scene descriptions are: Scene 1: The scene shows a woman looking off into the distance with an orange line going around the outside of the screen. The foreground colors of the scene are black, mud green, gray, dark gray, and the background colors are dark brown, black, dark gray. The dominant tone of the scene is neutral. This scene is categorized by the tags: cosmetic , eyebrow, face, girl, ponytail, stand, string, woman. Scene 2: The scene shows a woman balancing on a skateboard in a yard. The foreground colors of the scene are black, mud green, dark gray, olive, and the background colors are black, dark gray, gray, dark brown.The dominant tone of the scene is neutral. This scene is categorized by the tags: athletic, balance, beam, car, girl, house exterior, hurdle, jog, legging, plank, rail, seesaw, woman, yard. Scene 3: The scene shows a girl jumping over a wooden ramp in the backyard. The foreground colors of the scene are black, dark gray, gray, dark blue, and the background colors are dark brown, dark blue, purple, dark pink, brown. ... >>> BEHAVIOR <<< The video will be liked by 2.0% 1. "Wow. Love it. She’s such an inspiration to the next generation as well as everyone." 2. "Inspiring and great story behind this commercial. Builds meaning and purpose in the hearts and minds of youth. It’s been a while since good, meaningful ads have been made." 3. "She’s an inspiration to the world. Thanks to her,my niece is learning gymnastics. Hopefully someday, she is an inspiration to others as Suni is an to everyone" 4. "Chills watching this. Such an inspiration." 5. "Yooooo, this is straight up!" The replay values for each scene would be: Scene 1: 0.06 Scene 2: 0.23 Scene 3: 0.38 ... </ASSISTANT> 

Figure 4: Behavior Instruction fine-tuning template for the video: https://www.youtube.com/watch?v= BKPQkjRF4yY 

## 2.1.2 DATA FROM YOUTUBE 

Our data collection from YouTube begins with querying Wikidata (Vrandeciˇ c & Kr´ otzsch¨ , 2014) for YouTube IDs to compile a list of channels. Wikidata, derived from Wikipedia, provides a curated selection of renowned channels, automatically filtering out noisy videos commonly found in datasets collected from diverse sources like user-generated videos. This initial step yielded a dataset of 2.2 million videos spanning the period from 2018 to 2023, sourced from approximately 6,000 channels collected from Wikidata. 

To refine the dataset, manual filtering was employed to exclude certain categories deemed less relevant for our purposes. These categories included music and songs, gaming content, non-English videos, sports commentary, anime, memes, channels with disabled comments sections, and news-related content. Furthermore, and only videos with a substantial viewership, defined as greater than 10,000 views, were retained. We observed that these videos usually have less noisy comments and likes. 

Subsequently, the top comments from each video, as ordered by YouTube (i.e., the most liked comments), were selected for inclusion in the dataset. To address redundancy in comments, a TF-IDF filter was applied with a threshold of 0.7, which proved effective in removing duplicate comments prevalent in YouTube data. 

5 

Comments were further filtered to include only those with a minimum of four words and a maximum of 100 words, ensuring a balance between relevance and conciseness. Additionally, to mitigate the presence of NSFW content, a vocabulary specific to NSFW terms (ldn) was employed to filter out inappropriate posts. On average, we finally get 3.1 comments per video, providing a substantial corpus of user-generated content for analysis. After applying these filtering steps, the dataset was reduced to 250,000 videos, ensuring a curated and relevant collection for subsequent analysis and model training. 

## 2.2 INSTRUCTION FINE-TUNING LLAMA-VID 

After collecting Reddit and YouTube media and user behavior, we formulate instruction fine-tuning tasks for training LLaMA-Vid. In the training instruction, given the media content and automatic speech recognition if available, we ask the model to simulate the scene-by-scene description and user likes/views and top-5 comments. This instruction training template is given in Listing 1. To generate the instruction data, first, frames are sampled using the 30-degree rule (SI et al., 2023; Arev et al., 2014; Friedman & Feldman, 2004), then the scene-by-scene description are obtained by concatenating automatically generated captions and tags from LLaVA-13B (Liu et al., 2023a), colors and tone through Qin et al. (2020). This instruction format keeps the instructions similar to the instruction format for other VLMs like LLaVA (Liu et al., 2023a), MiniGPT-4 (Zhu et al., 2023), BLIP (Li et al., 2022), LLaMA-Vid (Li et al., 2023b), _etc._ , while additionally teaching the model to learn behavior. We keep the instruction fine-tuning template similar for both YouTube and Reddit. The complete instruction is given in Listing 1. 

We start with the trained LLaMA-Vid model. The LLaMA-Vid model uses two tokens to represent each frame in the video, which they call content and context tokens. While the context token encodes the overall image context based on user input, the content token encapsulates visual cues in each frame. For learning context tokens, the model uses attention queries that interact with previously generated image features in the designed attention module. To generate content tokens, the image features are average pooled. This dual-token strategy significantly reduces the number of tokens needed to represent videos, thus enabling the model to scale to longer (hour-long) videos. To better support hour-long videos, LLaMA-Vid was trained on a 9k movie-level conversation instruction set containing plot reasoning and detail understanding questions. 

Taking the base LLaMA-Vid model, we finetune it further on behavioral data. In our experiments we observe that the sampling ratio of BLIFT and IFT datasets is an important hyperparameter. We track 4 zero-shot metrics, likes/views, comments perplexity, and empirically find the best results with 1:1 ratio for 2.2 epochs (see Fig 11 and Table 12). For the best checkpoint, the perplexity on comments reduces from 6.22 to 3.05, and the R2 on likes/views goes from -5.1 to 0.45 

We combine 730k instruction pairs from BLIFT with the original instruction tuning dataset consisting of 40K text conversations from ShareGPT, 625K single or multi-turn visual QA pairs, and 98K video QA pairs; all the modules except the Visual Encoder are kept frozen. We ablate on multiple sampling ratios from BLIFT. We train the LLaMA-Vid checkpoints with their original SFT mix along with BLIFT. We ablate different sampling ratios and found 1:1 to be empirically performing the best. We train the model for 2.2 epochs, keeping track of the 0-shot evaluation metrics and perplexity on comments in the eval set Fig 11 and Table 12 show the ablations on different sampling ratios and epochs of training. For the best checkpoint, the perplexity on comments reduces from 6.22 to 3.05, and the _R_[2] on likes/views goes from -5.1 to 0.45. 

**AdLLaVA to show the impact of behavior data:** To disentangle the effect of training on additional data samples from the effect of training on behavioral data, we train LLaMA-Vid on BLIFT with the video and image verbalization and do not include receiver behavior. Then, the overall instruction template consists of scene-by-scene automatically generated verbalization similar to Listing 1 without the likes and comment simulation. We call the LLaMA-Vid fine-tuned on this data, Ad-LLaVA. We compare Behavior-LLaVA with Ad-LLaVA and LLaMA-Vid along with other state-of-the-art literature benchmarks on various tasks (Tables 1-4). 

**Impact of filtering steps on performance:** We use various filtering steps in our data pipeline, including NSFW filters, time filters, bot filters, and gaming and news video filters. To quantify the impact of our filtering steps on the final performance, we compare the performance from training on unfiltered data with filtered data (Figure 11). The figure shows the benefit of our data filtering 

6 

|**Model**|**scene**|**way**|~~**s**~~**peaking**|**relationship**|**like**<br>~~**r**~~**atio**|**view**<br>~~**c**~~**ount**|**director**|**genre**|**writer**|**year**|
|---|---|---|---|---|---|---|---|---|---|---|
|Video4096-GPT-3.5 generated story<br>+ Flan-t5-xxl|60.2||39.07|64.1|0.061|12.84|69.9|58.1|52.4|75.6|
|Video4096-GPT-3.5 generated story<br>+ GPT-3.5 classifer|54.54||32.95|68.42|0.031|12.69|75.26|50.84|32.16|75.96|
|LLaMA-Vid + GPT-3.5 Generated Story|58.12||35.5|60.6|0.314|10.34|65.34|49.77|34.23|72.12|
|Ad-LLaVA|59.05||37.07|61.2|0.319|10.37|66.84|55.13|35.33|77.34|
|Behavior LLaVA + GPT-3.5 Generated Story|66.43||41.03|64.21|0.17|5.12|71.12|63.45|39.4|79.3|
|**Improvement of Behavior LLaVA**<br>**over LLaMA-Vid**|10.48%|15.58%||9.62%|45.86%|50.48%|8.85%|27.49%|15.1%|9.96%|



Table 1: Comparison of various models on the Long Video Understanding benchmark (Wu & Krahenbuhl, 2021) consisting of 9 VQA tasks. We see that Behavior-LLaVA improves on LLaMA-Vid on 9/9 tasks with an average improvement of 21.49%. Further, it outperforms the state-of-the-art in 5/9 tasks. 

process. While training on unfiltered BLIFT also improves the result over baseline performance of Llama-Vid but data filtering adds more improvement on top of it 

**Ablation with perceptual behavior:** As an ablation experiment, we also try teaching the BehaviorLLaVA perceptual signals. For this, we take the largest perception signal dataset in the literature - Salicon10k (Jiang et al., 2015). It consists of 10,000 MS COCO images (Chen et al., 2015) with free-viewing eye gaze data collected through a novel mouse-based interface. The dataset has been widely used in many studies. We formulate two tasks using this data, (1) **Salicon [Object]** : estimating the saliency over the objects in the image and (2) **Salicon [Region]** : estimating the saliency over a region, where the regions tiles formed by breaking the image into a 3x3 grid. For both tasks we try to model two objectives, ranking and predicting, we found ranking to be much more effective. The instruction are given in Listings 5 and 6. 

## 3 RESULTS AND DISCUSSION 

In the experimental results, we aim to showcase the diverse and emergent capabilities of our BehaviorLLaVA model through quantitative numbers on various tasks and qualitative examples. These abilities include generating detailed image and video descriptions, emotion and sentiment analysis, question answering, video understanding tasks like scene and action detection. Additionally, we present the ability of Behavior-LLaVA to transfer learn on other behaviors like memorability of a video - both short-term and long-term. 

## 3.1 EVALUATION 

To test the effectiveness of Behavior-LLaVA, we conduct experiments involving 46 distinct tasks across 26 benchmark datasets. The diversity of tasks and datasets allows us to evaluate the performance and capabilities of Behavior-LLaVA thoroughly. Each of them is covered briefly next: 

1. **Visual Question Answering (VQA)** : We evaluate the performance of visual question answering on the following benchmark datasets: 

   - The Long-Video Understanding (LVU) benchmark by Wu & Krahenbuhl (2021) comprises nine distinct tasks aimed at assessing long video comprehension, incorporating over 1000 hours of video content. These tasks encompass diverse aspects such as content understanding (including relationship, speaking style, scene/place), prediction of user engagement (YouTube like ratio, YouTube popularity), and movie metadata (director, genre, writer, movie release year). 

   - The Holistic Video Understanding (HVU) dataset by Diba et al. (2020) stands as the largest dataset for long video comprehension, comprising 572,000 samples. Encompassing a broad spectrum of semantic elements within videos, HVU tasks involve the classification of scenes, objects, actions, events, attributes, and concepts. Performance evaluation on HVU tasks is conducted using the mean average precision (mAP) metric on the validation set. 

   - We also use MSVD-QA, MSRVTT-QA (Chen & Dolan, 2011; Xu et al., 2016b), and ActivityNetQA (Caba Heilbron et al., 2015) datasets. Their description is given in Appendix A. 

2. **Video and Image Understanding Benchmarks** : We use a wide variety of tasks to evaluate video and image understanding: topic, emotion, and persuasion strategy classification, action and reason retrieval and generation, and emotions. We briefly introduce the benchmarks: 

   - The advertisements dataset by Hussain et al. (2017) contains 3,477 video advertisements and the corresponding annotations for emotion and topic tags and action-reason statements for each 

7 

|**Training**|**Model**<br>**Topic**<br>**Sentiment**<br>**Persuasion**<br>**Action**<br>**Reason**<br>**Clubbed**<br>**All labels**|
|---|---|
|Random|Random<br>2.63<br>3.37<br>14.3<br>8.37<br>3.34<br>3.33|
|Zero-shot<br>VideoChat (Li et al.,2023a)<br>9.07<br>3.09<br>5.1<br>10.28<br>-<br>-<br>Video4096 - GPT-3.5 Generated Story + GPT-3.5 Classifer<br>51.6<br>11.68<br>79.69<br>35.02<br>66.27<br>59.59<br>LCBM (Khandelwal et al.,2024)<br>42.17<br>7.08<br>58.83<br>32.83<br>39.55<br>27.91<br>LLaMA-VID w/ only video<br>10.11<br>3.42<br>5.75<br>12.32<br>29.61<br>24.11<br>LLaMA-VID w/ video + GPT-3.5 Story<br>42.72<br>11.05<br>64.02<br>32.07<br>37.76<br>42.33<br>Behavior-LLaVA w/ only video<br>22.65<br>11.13<br>60.04<br>13.39<br>42.66<br>33.33<br>Behavior-LLaVA w/ video + verbalization<br>46.34<br>11.7<br>64.13<br>33.33<br>52.06<br>52.03<br>Ad-LLaVA w/ video + GPT-3.5 story<br>51.16<br>11.33<br>68.03<br>33.11<br>43.26<br>51.45<br>Behavior-LLaVA w/ video + GPT-3.5 story<br>60.09<br>12.84<br>79.94<br>36.12<br>67.10<br>79.18<br>**Improvement of Behavior-LLaVA over LLaMA-Vid**<br>40.66%<br>16.2%<br>24.86%<br>12.62%<br>77.7%<br>87.05%|VideoChat (Li et al.,2023a)<br>9.07<br>3.09<br>5.1<br>10.28<br>-<br>-<br>Video4096 - GPT-3.5 Generated Story + GPT-3.5 Classifer<br>51.6<br>11.68<br>79.69<br>35.02<br>66.27<br>59.59<br>LCBM (Khandelwal et al.,2024)<br>42.17<br>7.08<br>58.83<br>32.83<br>39.55<br>27.91<br>LLaMA-VID w/ only video<br>10.11<br>3.42<br>5.75<br>12.32<br>29.61<br>24.11<br>LLaMA-VID w/ video + GPT-3.5 Story<br>42.72<br>11.05<br>64.02<br>32.07<br>37.76<br>42.33|
||Behavior-LLaVA w/ only video<br>22.65<br>11.13<br>60.04<br>13.39<br>42.66<br>33.33<br>Behavior-LLaVA w/ video + verbalization<br>46.34<br>11.7<br>64.13<br>33.33<br>52.06<br>52.03<br>Ad-LLaVA w/ video + GPT-3.5 story<br>51.16<br>11.33<br>68.03<br>33.11<br>43.26<br>51.45<br>Behavior-LLaVA w/ video + GPT-3.5 story<br>60.09<br>12.84<br>79.94<br>36.12<br>67.10<br>79.18|
|Finetuned<br>Video4096- Generated Story + Roberta Classifer<br>71.3<br>33.02<br>84.20<br>64.67<br>42.96<br>39.09<br>LLaMA-VID w/ video + verbalization<br>59.13<br>32.11<br>79.15<br>50.93<br>50.32<br>30.13<br>LLaMA-VID w/ video + GPT-3.5 Story<br>63.11<br>35.01<br>84.15<br>55.01<br>57.11<br>45.73<br>Behavior-LLaVA w/ only video<br>58.03<br>22.72<br>84.41<br>26.23<br>59.33<br>51.45<br>Behavior-LLaVA w/ video + verbalization<br>68.32<br>33.92<br>85.93<br>64.72<br>70.89<br>75.34<br>Ad-LLaVA w/ video + GPT-3.5 story<br>66.34<br>36.24<br>84.09<br>58.31<br>68.15<br>78.15<br>Behavior-LLaVA w/ video + GPT-3.5 story<br>71.2<br>39.55<br>86.17<br>65.03<br>80.44<br>81.67<br>**Improvement of Behavior-LLaVA over LLaMA-Vid**<br>12.82%<br>12.97%<br>2.4%<br>18.21%<br>40.85%<br>78.59%|Video4096- Generated Story + Roberta Classifer<br>71.3<br>33.02<br>84.20<br>64.67<br>42.96<br>39.09<br>LLaMA-VID w/ video + verbalization<br>59.13<br>32.11<br>79.15<br>50.93<br>50.32<br>30.13<br>LLaMA-VID w/ video + GPT-3.5 Story<br>63.11<br>35.01<br>84.15<br>55.01<br>57.11<br>45.73|
||Behavior-LLaVA w/ only video<br>58.03<br>22.72<br>84.41<br>26.23<br>59.33<br>51.45<br>Behavior-LLaVA w/ video + verbalization<br>68.32<br>33.92<br>85.93<br>64.72<br>70.89<br>75.34<br>Ad-LLaVA w/ video + GPT-3.5 story<br>66.34<br>36.24<br>84.09<br>58.31<br>68.15<br>78.15<br>Behavior-LLaVA w/ video + GPT-3.5 story<br>71.2<br>39.55<br>86.17<br>65.03<br>80.44<br>81.67|



Table 2: Comparison of various models on two video understanding benchmarks (Hussain et al., 2017; Kumar et al., 2023) consisting of 5 tasks related to video advertisements understanding. The main goal of comparing on these benchmarks is to demonstrate Behavior-LLaVA’s understanding of complex videos. We see that Behavior-LLaVA improves on LLaMA-Vid on 5/5 tasks with an average improvement score of 43.18% in zero-shot and 27.64% in fine-tuned settings. Further, it outperforms the current state-of-the-art on 3/5 tasks in zero-shot and 5/5 in fine-tuned settings. Full results are presented in the Table 6. 

- video. There are a total of 38 topics and 30 unique emotion tags per video. Further, we have 5 action-reason statements for each video for the action-reason generation task. 

- Persuasion strategy dataset (Bhattacharyya et al., 2023) is a dataset consisting of 1002 video advertisements from popular brands and their persuasion strategy labels like social identity, anchoring and comparison, reciprocity, foot-in-the-door, _etc_ . 

- For emotion analysis, we use VideoEmotion-8 (Asur & Huberman, 2010), Ekman-6 (Xu et al., 2016a), CAER (Lee et al., 2019), IAPSa (Mikels et al., 2005), Emotion6 (Peng et al., 2015), EmoSet (Yang et al., 2023), and Abstract (Machajdik & Hanbury, 2010) datasets. A brief description for each of them is given in Appendix A. 

3. **Image Dense Captioning** : Literature image captioning datasets such as MS-COCO (Chen et al., 2015) reduce the inherently rich information and fine-grained semantics to simplistic captions, with very brief statements focussing only on salient objects. Behavior data such as user comments help a model learn much more information such as object and material properties, world knowledge, emotion, character understanding, spatial relationships, aesthetics, _etc._ (see Fig. 1), enhancing the model’s captioning capability. Therefore, we design a captioning task to test this capability and compare it with respect to LLaMA-Vid and LLaVA-34B (a 2.5x larger model). Since we do not have ground truth for this task, following the LLM-as-a-judge paradigm, we use GPT-4V as the judge for all the models. GPT-4V is asked to evaluate the dense captions on three metrics: _Correctness_ (Listing 2) evaluating the factuality and model hallucinations, _Detail_ (Listing 3) evaluating the number and depth of details captured by the generated captions, and _Quality_ (Listing 4) measuring the subjective quality of the concepts chosen to be highlighted by the captioning model and the arrangement, coherence, and the linking of various concepts. 

4. **Image and Video Memorability Simulation** : Behavior-LLaVA is trained on behavior along with the media. To check if training on behavior helps in solving other behavior tasks (Khandelwal et al., 2024), we test it over image and video memorability simulation. For this, we select seven benchmark datasets covering long-term and short-term memorability over images and videos: LaMem (Khosla et al., 2015), SUN (Isola et al., 2011), and MemCat (Goetschalckx & Wagemans, 2019) for images and Memento10k(Newman et al., 2020), VideoMem (Cohendet et al., 2019), MediaEval (Kiziltepe et al., 2021), and LAMBDA (I et al., 2024) for videos. We briefly cover each of them in Appendix A. 5. **Modalities other than videos and images:** Behavior-LLaVA, built on top of LLaMA-Vid and fine-tuned using BLIFT, is pretrained and fine-tuned on image and video datasets. To test if behavior data can improve the results on other modalities as well, we test Behavior-LLaVA’s performance on two tasks across audio and text modalities (Table 13). For audio, we evaluate on the audio summarization task (Han et al., 2023) and for text, we evaluate on the IMDB sentiment benchmark (Maas et al., 2011). 

8 

|**Training**|**Dataset**|**Video Emotion-8**<br>**CAER**<br>**Ekman-6**|
|---|---|---|
|Random|Random|12.5<br>14.28<br>16.67|
|0-Shot<br>LLaMA-Vid<br>Behavior-LLaVA<br>Ad-LLaVA<br>**Improvement of Behavior-LLaVA over LLaMA-Vid**|LLaMA-Vid<br>Behavior-LLaVA<br>Ad-LLaVA|29.7<br>27.2<br>37.33<br>41.35<br>51.0<br>49.33<br>29.8<br>27.3<br>37.66|
|||39.22%<br>84.19%<br>32.14%|
|Finetuned<br>Zhao et al.(2020)<br>Zhang et al.(2023c)<br>eMOTIONS (Wu et al.,2023)<br>Arevalo et al.(2017)<br>Qiu et al.(2020)<br>Xu et al.(2016a)<br>LLaMA-Vid<br>Ad-LLaVA<br>Behavior-LLaVA<br>**Improvement of Behavior-LLaVA over LLaMA-Vid**|Zhao et al.(2020)<br>Zhang et al.(2023c)<br>eMOTIONS (Wu et al.,2023)<br>Arevalo et al.(2017)<br>Qiu et al.(2020)<br>Xu et al.(2016a)<br>LLaMA-Vid<br>Ad-LLaVA<br>Behavior-LLaVA|54.5<br>78.3<br>55.3<br>57.3<br>80.1<br>58.2<br>-<br>-<br>53.12<br>53.7<br>77.3<br>54.2<br>53.3<br>-<br>57.3<br>52.6<br>77.9<br>55.6<br>53.8<br>75.6<br>57.9<br>54.1<br>76.1<br>57.8<br>56.9<br>79.3<br>58.4|
|||5.76%<br>3.57%<br>0.86%|



Table 3: Comparison of various models on three video emotion understanding benchmarks (Video Emotion8 (Jiang et al., 2014), CAER (Lee et al., 2019), Ekman-6 (Xu et al., 2016a)). The main goal of comparing on these benchmarks is to demonstrate Behavior-LLaVA’s understanding of complex tasks like video emotions of long-form videos. We see that Behavior-LLaVA improves on LLaMA-Vid on 3/3 benchmarks with an average improvement score of 51.85% in zero-shot and 3.39% in fine-tuned settings. Further, it outperforms the current state-of-the-art on 3/3 benchmarks in zero-shot and 1/3 in fine-tuned settings. 

|**Training**|**Models**|**Image Datasets**<br>**Lamem**<br>**Memcat**<br>**SUN**|**Video Datasets**<br>**Memento10k**<br>**VideoMem**<br>**MediaEval**<br>**LAMBDA**|
|---|---|---|---|
||Human Consistency|0.68<br>0.78<br>0.75|0.73<br>0.61<br>-<br>0.61|
|Finetuned<br>10-shot in-context learning GPT-3.5<br>ViTMem (Hagen & Espeseth,2023)<br>Henry trained with 25% data (I et al.,2024)<br>Henry trained with 50% data (I et al.,2024)<br>Henry trained with 75% data (I et al.,2024)<br>Henrytrained on all(combined)datasets(I et al.,2024)<br>Ad-LLaVA trained with 50% data<br>Behaviour LLaVA trained with 25% data<br>Behaviour LLaVA trained with 50% data<br>Behaviour LLaVA trained with 75% data<br>Behavior-LLaVA trained on all datasets<br>**Improvement of Behavior-LLaVA over LLaMA-Vid**(25% data)<br>**Improvement of Behavior-LLaVA over LLaMA-Vid** (50% data)|10-shot in-context learning GPT-3.5<br>ViTMem (Hagen & Espeseth,2023)<br>Henry trained with 25% data (I et al.,2024)<br>Henry trained with 50% data (I et al.,2024)<br>Henry trained with 75% data (I et al.,2024)<br>Henrytrained on all(combined)datasets(I et al.,2024)|0.29<br>0.18<br>0.15<br>0.71<br>0.65<br>0.63<br>0.56<br>0.64<br>0.59<br>0.65<br>0.68<br>0.67<br>0.71<br>0.75<br>0.73<br>0.72<br>0.79<br>0.76|0.07<br>0.06<br>0.06<br>0.06<br>0.56<br>0.51<br>-<br>0.08<br>0.62<br>0.49<br>0.32<br>0.28<br>0.69<br>0.55<br>0.44<br>0.40<br>0.74<br>0.62<br>0.49<br>0.47<br>0.72<br>0.60<br>0.48<br>0.52|
||Ad-LLaVA trained with 50% data<br>Behaviour LLaVA trained with 25% data<br>Behaviour LLaVA trained with 50% data<br>Behaviour LLaVA trained with 75% data<br>Behavior-LLaVA trained on all datasets|0.67<br>0.65<br>0.61<br>0.67<br>0.72<br>0.69<br>0.72<br>0.77<br>0.73<br>0.73<br>0.77<br>0.74<br>0.73<br>0.78<br>0.74|0.69<br>0.56<br>0.43<br>0.47<br>0.68<br>0.53<br>0.44<br>0.50<br>0.71<br>0.59<br>0.46<br>0.51<br>0.70<br>0.60<br>0.47<br>0.50<br>0.71<br>0.60<br>0.47<br>0.52|
|||19.64%<br>12.5%<br>16.95%<br>10.77%<br>13.26%<br>8.96%|9.68%<br>8.16%<br>37.5%<br>78.57%<br>2.90%<br>7.27%<br>4.54%<br>27.5%|
|0-shot|LLaMA-Vid<br>Ad-LLaVA<br>Behavior-LLaVA|0.13<br>0.11<br>0.05<br>0.14<br>0.13<br>0.06<br>0.21<br>0.17<br>0.13|0.03<br>0.05<br>0.02<br>0.05<br>0.06<br>0.07<br>0.04<br>0.13<br>0.12<br>0.08<br>0.07<br>0.16|
||**Improvement of Behavior-LLaVA over LLaMA-Vid**|61.5%<br>54.5%<br>160%|300%<br>160%<br>350%<br>219%|



Table 4: Comparison of various models on seven video and image memorability benchmarks (Memento10k (Newman et al., 2020), VideoMem (Cohendet et al., 2019), LaMem (Khosla et al., 2015), SUN (Isola et al., 2011), MemCat (Goetschalckx & Wagemans, 2019), MediaEval (Kiziltepe et al., 2021), LAMBDA (I et al., 2024)). The main goal of comparing on these benchmarks is to demonstrate Behavior-LLaVA’s understanding of complex and high-level tasks like memorability simulation. We see that Behavior-LLaVA improves on LLaMA-Vid on 7/7 benchmarks with an average improvement score of 186.4% in zero-shot and 39% in fine-tuned settings after seeing 25% train data. Further, it performs similarly to the current state-of-the-art on 7/7 benchmarks in the fine-tuned settings while still seeing only 25% data. 

For Tables 1, 10, and 2, we follow the evaluation protocol of Video-4096 (Bhattacharyya et al., 2023), for Table 7 we follow the evaluation protocol of LLaVA and LLaMA-VID. For Tables 3, 8, and 4, for 0-shot evaluation results, we use the logits of the next token from the given task vocabulary. For Table 4, we use the evaluation protocol of SI et al. (2023). 

## 3.2 DISCUSSION 

Tables 1, 10, and 7 contain the results for the visual question answering tasks, Tables 2, 3, 8 contain the results for video and image understanding tasks, Tables 9 contains the results for dense-captioning, Table 4 contains the results for image and video memorability benchmarks, and Table 13 contains the results for the audio and text tasks. A common trend we observe across all the results is that BehaviorLLaVA performs better than the base model LLaMA-Vid and the finetuned model Ad-LLaVA on all tasks, especially in zero-shot settings. In fact, Ad-LLaVA performs very similar to LLaMA-Vid itself. This shows that BLIFT adds meaningful signals on an average rather than noise to the model. Interestingly, the performance gains remain even after fine-tuning on the task dataset (Tables 2, 3, 8). 

9 

|**Model**|**Summarization (3 Shot)**<br>**Moment Retrieval**<br>**Highlight Detection**<br>BLEU<br>ROUGE<br>METEOR<br>mAP (avg)<br>mAP|
|---|---|
|Behavior-LLaVA[Replay-Graphs]<br>Behavior-LLaVA[Mem-Recalls]<br>Behavior-LLaVA|11.2<br>19.1<br>25.3<br>**35.2**<br>**34.9**<br>**11.9**<br>**20.3**<br>**26.9**<br>34.9<br>33.1<br>10.6<br>18.3<br>22.7<br>33.1<br>32.7|
|LLaMA-VID<br>Video-ChatGPT|9.1<br>16.5<br>20.3<br>30.3<br>32.4<br>5.0<br>14.0<br>19.7<br>-<br>-|



Table 5: Improvement on downstream content understanding tasks by introducing more behaviour signals. Brackets [] denote the new behaviour that we include. Replay graphs (Khandelwal et al., 2024). Mem-Recalls (I et al., 2024) Evaluation done on Multi-shot video summarization (Han et al., 2023) and MomentDETR (Lei et al., 2021) 

The performance gains are relatively smaller for low-level tasks of action and object recognition (Tables 10, and 7), but much higher for the more high-level tasks of emotion understanding, sentiment analysis, persuasion strategy classification, and memorability simulation, longform-video understanding and other sub-tasks of Table 10. This indicates that receiver behavior has richer signals for higher-level tasks, infact fine-tuned Behaviour-LLaVA models outperform GPT4-V on image emotion recognition. The gains are observed across both image and video benchmarks. We also observe that classification using a story generated by GPT-3.5 (following Bhattacharyya et al. (2023)) results in better performance than only using the video (Tables 1, 10, 2). To evaluate the generalizability of behavior data across modalities, we extend our evaluation to audio summarization and text sentiment analysis and observe improvements of 19.5% (Table 13). 

Figures 6, 7, 10, and 5, 8, 9, show several randomly sampled qualitative examples for dense captions generated by Behavior-LLaVA over images and videos respectively. It can be noticed that despite not being explicitly trained for this task, the model performs quite well, picking up various artistic, cognitive, and object and material properties. From Table 9, while Behavior-LLaVA shows a decrease in correctness over LLaMA-Vid, it shows significant improvement in other aspects, including detail and quality. On these aspects, it even comes close to 2.5X larger models (LLaVA-1.6 (34B)). 

Next, in Table 11 we compare the signals from behavioral data of perception and action. For this, we compare Behavior-LLaVA trained on BLIFT and Behaviour-LLaVA trained on Salicon salient regions and objects. Further, within BLIFT, we compare the performance from predicting singled out behaviours including likes/views, titles, comments. It can be noted that training on just Salicon results in a performance decrease for the lower-level task of action recognition (MSRVTT-QA) but improves on the higher-level task of Emotion recognition. However, the gains are smaller than those observed with training on BLIFT. 

## 4 CONCLUSION 

In this paper, we explore the idea of learning behavior leading to learning content _better_ . Humans produce behavior in response to content. Hence, logically, behavior should contain signals about content, which, if used as a training task, should help in learning content better. We follow this line of thought and show that training large vision and language models on user behavior data of comments and likes collected from Reddit and YouTube leads to performance improvements across a wide variety of tasks. The gains are higher on higher-level tasks such as emotion recognition, persuasion strategy classification, and question answering and smaller on lower-level tasks like action and object recognition. Further, the gains remain even after fine-tuning the VLMs on those benchmarks, thus demonstrating the importance of learning behavior in understanding content better. 

## REFERENCES 

List of dirty, naughty, obscene, and otherwise bad words. https://github.com/LDNOOBW/ List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words. 13th April, 2024. 6 

- Reddit pics: Rules. https://www.reddit.com/r/pics/wiki/rule8/, 2024a. Accessed: April 11, 2024. 4 

- Reddit pics: Wiki index. https://new.reddit.com/r/pics/wiki/index/, 2024b. Accessed: April 11, 2024. 4 

10 

Reddit videos: Rules. https://www.reddit.com/r/videos/wiki/rules, 2024c. Accessed: April 11, 2024. 4 

- Ido Arev, Hyun Soo Park, Yaser Sheikh, Jessica Hodgins, and Ariel Shamir. Automatic editing of footage from multiple social cameras. _ACM Transactions on Graphics (TOG)_ , 33(4):1–11, 2014. 6 

- John Arevalo, Thamar Solorio, Manuel Montes y Gomez, and Fabio A. Gonz´ alez.´ Gated multimodal units for information fusion, 2017. 9 

- Sitaram Asur and Bernardo A Huberman. Predicting the future with social media. In _2010 IEEE/WIC/ACM international conference on web intelligence and intelligent agent technology_ , volume 1, pp. 492–499. IEEE, 2010. 8, 23 

- Maria Barrett, Joachim Bingel, Frank Keller, and Anders Søgaard. Weakly supervised part-of-speech tagging using eye-tracking data. In _ACL (Volume 2: Short Papers)_ . ACL, August 2016a. doi: 10.18653/v1/P16-2094. URL https://aclanthology.org/P16-2094. 2 

- Maria Barrett, Frank Keller, and Anders Søgaard. Cross-lingual transfer of correlations between parts of speech and gaze features. In _Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers_ , pp. 1330–1339, Osaka, Japan, December 2016b. The COLING 2016 Organizing Committee. URL https://aclanthology.org/C16-1126. 2 

- Maria Barrett, Joachim Bingel, Nora Hollenstein, Marek Rei, and Anders Søgaard. Sequence classification with human attention. In _CNLL_ . Association for Computational Linguistics, October 2018. URL https://aclanthology.org/K18-1030. 2 

- Bennett I Bertenthal. Origins and early development of perception, action, and representation. _Annual review of psychology_ , 47(1):431–459, 1996. 2 

- Yevgeni Berzak, Chie Nakamura, Amelia Smith, Emily Weng, Boris Katz, Suzanne Flynn, and Roger Levy. Celer: A 365-participant corpus of eye movements in l1 and l2 english reading. _Open Mind_ , 2022. 2 

- Aanisha Bhattacharyya, Yaman K Singla, Balaji Krishnamurthy, Rajiv Ratn Shah, and Changyou Chen. A video is worth 4096 tokens: Verbalize videos to understand them in zero shot. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 9822–9839, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.608. URL https://aclanthology.org/2023.emnlp-main.608. 8, 9, 10, 20 

- Klinton Bicknell and Roger Levy. Why readers regress to previous words: A statistical analysis. In _Proceedings of the Annual Meeting of the Cognitive Science Society_ , volume 33, 2011. 2 

- Stella Biderman, Kieran Bicheno, and Leo Gao. Datasheet for the pile. _arXiv preprint arXiv:2201.07311_ , 2022. 2 

- Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In _Proceedings of the ieee conference on computer vision and pattern recognition_ , pp. 961–970, 2015. 7, 23 

- David Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In _Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies_ , pp. 190–200, 2011. 7, 23 

- Sihan Chen, Xingjian He, Longteng Guo, Xinxin Zhu, Weining Wang, Jinhui Tang, and Jing Liu. Valor: Vision-audio-language omni-perception pretraining model and dataset, 2023. URL https://arxiv.org/abs/2304.08345. 22 

- Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollar, and´ C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. _arXiv preprint arXiv:1504.00325_ , 2015. 7, 8 

11 

- Ming-Ming Cheng, Niloy J Mitra, Xiaolei Huang, Philip HS Torr, and Shi-Min Hu. Global contrast based salient region detection. _IEEE transactions on pattern analysis and machine intelligence_ , 37 (3):569–582, 2014. 2 

- Romain Cohendet, Claire-Hel´ ene Demarty, Ngoc QK Duong, and Martin Engilberge.` Videomem: Constructing, analyzing, predicting short-term and long-term video memorability. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 2531–2540, 2019. 8, 9, 23 

- Marcella Cornia, Lorenzo Baraldi, Giuseppe Serra, and Rita Cucchiara. Paying more attention to saliency: Image captioning with saliency and context attention. _ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)_ , 14(2):1–21, 2018. 2 

- Vivek Dhakal, Anna Maria Feit, Per Ola Kristensson, and Antti Oulasvirta. Observations on typing from 136 million keystrokes. In _Proceedings of the 2018 CHI conference on human factors in computing systems_ , pp. 1–12, 2018. 2 

- Ali Diba, Mohsen Fayyaz, Vivek Sharma, Manohar Paluri, Jurgen Gall, Rainer Stiefelhagen, and¨ Luc Van Gool. Large scale holistic video understanding. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part V 16_ , pp. 593–610. Springer, 2020. 7, 21 

- Andrius Dzedzickis, Arturas Kaklauskas,¯ and Vytautas Bucinskas. Human emotion recognition: Review of sensors and methods. _Sensors_ , 20(3):592, 2020. 2 

- The Economist. Why reddit users are protesting against the site’s leadership. _The Economist Explains_ , June 2023. URL https://www.economist.com/the-economist-explains/2023/06/ 26/why-reddit-users-are-protesting-against-the-sites-leadership. Accessed: April 11, 2024. 4 

- Shaojing Fan, Zhiqi Shen, Ming Jiang, Bryan L Koenig, Juan Xu, Mohan S Kankanhalli, and Qi Zhao. Emotional attention: A study of image sentiment and visual attention. In _Proceedings of the IEEE Conference on computer vision and pattern recognition_ , pp. 7521–7531, 2018. 2 

- Doron Friedman and Yishai A Feldman. Knowledge-based cinematography and its applications. In _ECAI_ , volume 16, pp. 256. Citeseer, 2004. 6 

- Lore Goetschalckx and Johan Wagemans. Memcat: a new category-based image set quantified on memorability. _PeerJ_ , 7:e8169, 2019. 8, 9, 23 

- Thomas Hagen and Thomas Espeseth. Image memorability prediction with vision transformers. _arXiv preprint arXiv:2301.08647_ , 2023. 9 

- Mingfei Han, Xiaojun Chang, Heng Wang, and Linjie Yang. Shot2story20k: A new benchmark for comprehensive understanding of multi-shot videos. _arXiv preprint arXiv:2312.10300_ , 2023. 8, 10, 22 

- Alex Hern. Reddit moderator protest spreads across thousands of communities. _The Guardian_ , December 2023. URL https://www.theguardian.com/technology/2023/dec/30/ reddit-moderator-protest-communities-social-media. Accessed: April 11, 2024. 4 

- Nora Hollenstein and Ce Zhang. Entity recognition at first sight: Improving NER with eye movement information. In _NAACL: Human Language Technologies, Volume 1 (Long and Short Papers)_ , Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. 2 

- Nora Hollenstein, Maria Barrett, Marius Troendle, Francesco Bigiolli, Nicolas Langer, and Ce Zhang. Advancing nlp with cognitive language processing signals. _arXiv preprint arXiv:1904.02682_ , 2019. 2 

- Zaeem Hussain, Mingda Zhang, Xiaozhong Zhang, Keren Ye, Christopher Thomas, Zuha Agha, Nathan Ong, and Adriana Kovashka. Automatic understanding of image and video advertisements, 2017. 7, 8, 20 

12 

- Harini S I, Somesh Singh, Yaman K Singla, Aanisha Bhattacharyya, Veeky Baths, Changyou Chen, Rajiv Ratn Shah, and Balaji Krishnamurthy. Long-term ad memorability: Understanding and generating memorable ads, 2024. 8, 9, 10, 23 

- Phillip Isola, Jianxiong Xiao, Antonio Torralba, and Aude Oliva. What makes an image memorable? In _CVPR 2011_ , pp. 145–152. IEEE, 2011. 8, 9, 23 

- Ming Jiang, Shengsheng Huang, Juanyong Duan, and Qi Zhao. Salicon: Saliency in context. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 1072–1080, 2015. 2, 7 

- Yu-Gang Jiang, Baohan Xu, and Xiangyang Xue. Predicting emotions in user-generated videos. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 28, 2014. 9 

- Alan Kennedy, Joel Pynte, Wayne S Murray, and Shirley-Anne Paul.¨ Frequency and predictability effects in the dundee corpus: An eye movement analysis. _Quarterly Journal of Experimental Psychology_ , 66(3):601–618, 2013. 2 

- Ashmit Khandelwal, Aditya Agrawal, Aanisha Bhattacharyya, Yaman Kumar, Somesh Singh, Uttaran Bhattacharya, Ishita Dasgupta, Stefano Petrangeli, Rajiv Ratn Shah, Changyou Chen, and Balaji Krishnamurthy. Large content and behavior models to understand, simulate, and optimize content and behavior. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum?id=TrKq4Wlwcz. 8, 10, 20 

- Aditya Khosla, Akhil S Raju, Antonio Torralba, and Aude Oliva. Understanding and predicting image memorability at a large scale. In _Proceedings of the IEEE international conference on computer vision_ , pp. 2390–2398, 2015. 8, 9, 23 

- Varun Khurana, Yaman Kumar, Nora Hollenstein, Rajesh Kumar, and Balaji Krishnamurthy. Synthesizing human gaze feedback for improved NLP performance. In _Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics_ , pp. 1895–1908, Dubrovnik, Croatia, May 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.eacl-main.139. URL https://aclanthology.org/2023.eacl-main.139. 2 

- Rukiye Savran Kiziltepe, Lorin Sweeney, Mihai Gabriel Constantin, Faiyaz Doctor, Alba Garcia Seco de Herrera, Claire-Helene Demarty, Graham Healy, Bogdan Ionescu, and Alan F. Smeaton. An annotated video dataset for computing video memorability. _Data in Brief_ , 39:107671, dec 2021. doi: 10.1016/j.dib.2021.107671. 8, 9, 23 

- Srinivas SS Kruthiventi, Vennela Gudisa, Jaley H Dholakiya, and R Venkatesh Babu. Saliency unified: A deep architecture for simultaneous eye fixation prediction and salient object segmentation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 5781–5790, 2016. 2 

- Yaman Kumar, Rajat Jha, Arunim Gupta, Milan Aggarwal, Aditya Garg, Tushar Malyan, Ayush Bhardwaj, Rajiv Ratn Shah, Balaji Krishnamurthy, and Changyou Chen. Persuasion strategies in advertisements. _Proceedings of the AAAI Conference on Artificial Intelligence_ , 2023. 8, 20 

- Harold D Lasswell. The structure and function of communication in society. _The communication of ideas_ , 37(1):136–139, 1948. 1 

Harold D Lasswell. _Propaganda technique in world war I_ . MIT press, 1971. 1 

- Jiyoung Lee, Seungryong Kim, Sunok Kim, Jungin Park, and Kwanghoon Sohn. Context-aware emotion recognition networks. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pp. 10143–10152, 2019. 8, 9, 23 

- Jie Lei, Tamara L. Berg, and Mohit Bansal. Qvhighlights: Detecting moments and highlights in videos via natural language queries, 2021. URL https://arxiv.org/abs/2107.09609. 10 

- Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In _ICML_ , 2022. 6 

13 

KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding, 2023a. 8, 20 

- Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. _arXiv preprint arXiv:2311.17043_ , 2023b. 3, 6 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _arXiv preprint arXiv:2304.08485_ , 2023a. 2, 3, 6 

- Ruyang Liu, Chen Li, Yixiao Ge, Ying Shan, Thomas H Li, and Ge Li. One for all: Video conversation is feasible without video instruction tuning. _arXiv preprint arXiv:2309.15785_ , 2023b. 20 

- Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In _Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies_ , pp. 142–150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics. URL http: //www.aclweb.org/anthology/P11-1015. 8, 22 

- Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. _arXiv preprint arXiv:2306.05424_ , 2023. 20 

- Jana Machajdik and Allan Hanbury. Affective image classification using features inspired by psychology and art theory. In _Proceedings of the 18th ACM international conference on Multimedia_ , pp. 83–92, 2010. 8, 20, 23 

- Joseph A Mikels, Barbara L Fredrickson, Gregory R Larkin, Casey M Lindberg, Sam J Maglio, and Patricia A Reuter-Lorenz. Emotional category data on images from the international affective picture system. _Behavior research methods_ , 37:626–630, 2005. 8, 20, 23 

- Anelise Newman, Camilo Fosco, Vincent Casser, Allen Lee, Barry McNamara, and Aude Oliva. Multimodal memorability: Modeling effects of semantics and decay on video memorability. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVI 16_ , pp. 223–240. Springer, 2020. 8, 9, 23 

- Badri Patro and Vinay P Namboodiri. Differential attention for visual question answering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 7680–7688, 2018. 2 

- Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only. _arXiv preprint arXiv:2306.01116_ , 2023. 2 

- Kuan-Chuan Peng, Tsuhan Chen, Amir Sadovnik, and Andrew C Gallagher. A mixed bag of emotions: Model, predict, and transfer emotion distributions. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 860–868, 2015. 8, 20, 23 

- Barbara Plank. Keystroke dynamics as signal for shallow syntactic parsing. In _Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers_ , pp. 609–619, 2016. 2 

- Wolfgang Prinz. Perception and action planning. _European journal of cognitive psychology_ , 9(2): 129–154, 1997. 2 

- Xuebin Qin, Zichen Zhang, Chenyang Huang, Masood Dehghan, Osmar R Zaiane, and Martin Jagersand. U2-net: Going deeper with nested u-structure for salient object detection. _Pattern recognition_ , 106:107404, 2020. 6 

Haonan Qiu, Liang He, and Feng Wang. Dual focus attention network for video emotion recognition. In _2020 IEEE International Conference on Multimedia and Expo (ICME)_ , pp. 1–6. IEEE, 2020. 9 

- Reddit, Inc. Reddit content policy. https://www.redditinc.com/policies/content-policy, 2024. Accessed: April 11, 2024. 4 

14 

- Claude E. Shannon and Warren Weaver. _The mathematical theory of communication._ The mathematical theory of communication. University of Illinois Press, Champaign, IL, US, 1949. 1 

- Harini SI, Somesh Singh, Yaman K Singla, Aanisha Bhattacharyya, Veeky Baths, Changyou Chen, Rajiv Ratn Shah, and Balaji Krishnamurthy. Long-term ad memorability: Understanding and generating memorable ads. _arXiv preprint arXiv:2309.00378_ , 2023. 6, 9 

- Ekta Sood, Simon Tannert, Philipp Muller,¨ and Andreas Bulling. Improving natural language processing tasks with human gaze-guided neural attention. _NeurIPS_ , 2020. 2 

- Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are dataefficient learners for self-supervised video pre-training, 2022. 20 

- Denny Vrandeciˇ c and Markus Kr´ otzsch.¨ Wikidata: A free collaborative knowledgebase. _Commun. ACM_ , 57(10):78–85, sep 2014. ISSN 0001-0782. doi: 10.1145/2629489. URL https://doi. org/10.1145/2629489. 5 

- Wenguan Wang, Jianbing Shen, Xingping Dong, and Ali Borji. Salient object detection driven by fixation prediction. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 1711–1720, 2018. 2 

- Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, Sen Xing, Guo Chen, Junting Pan, Jiashuo Yu, Yali Wang, Limin Wang, and Yu Qiao. Internvideo: General video foundation models via generative and discriminative learning. _arXiv preprint arXiv:2212.03191_ , 2022. 20 

- Chao-Yuan Wu and Philipp Krahenbuhl. Towards long-form video understanding. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 1884–1894, 2021. 7 

- Xuecheng Wu, Heli Sun, Junxiao Xue, Ruofan Zhai, Xiangyan Kong, Jiayu Nie, and Liang He. emotions: A large-scale dataset for emotion recognition in short videos, 2023. 9 

- Baohan Xu, Yanwei Fu, Yu-Gang Jiang, Boyang Li, and Leonid Sigal. Heterogeneous knowledge transfer in video emotion recognition, attribution and summarization. _IEEE Transactions on Affective Computing_ , 9(2):255–270, 2016a. 8, 9, 23 

- Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 5288–5296, 2016b. 7, 23 

- Liwen Xu, Zhengtao Wang, Bin Wu, and Simon Lui. Mdan: Multi-level dependent attention network for visual emotion analysis. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 9479–9488, 2022. 20 

- Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Zero-shot video question answering via frozen bidirectional language models. _Advances in Neural Information Processing Systems_ , 35:124–141, 2022. 20 

- Jingyuan Yang, Jie Li, Xiumei Wang, Yuxuan Ding, and Xinbo Gao. Stimuli-aware visual emotion analysis. _IEEE Transactions on Image Processing_ , 30:7432–7445, 2021. 20 

- Jingyuan Yang, Qirui Huang, Tingting Ding, Dani Lischinski, Danny Cohen-Or, and Hui Huang. Emoset: A large-scale visual emotion dataset with rich attributes. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 20383–20394, 2023. 8, 20, 23 

- Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. _arXiv preprint arXiv:2306.02858_ , 2023a. 20 

- Renrui Zhang, Jiaming Han, Chris Liu, Aojun Zhou, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. Llama-adapter: Efficient fine-tuning of large language models with zero-initialized attention. In _The Twelfth International Conference on Learning Representations_ , 2023b. 20 

- Zhicheng Zhang, Lijuan Wang, and Jufeng Yang. Weakly supervised video emotion detection and prediction via cross-modal temporal erasing network. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 18888–18897, 2023c. 9 

15 

- Sicheng Zhao, Yunsheng Ma, Yang Gu, Jufeng Yang, Tengfei Xing, Pengfei Xu, Runbo Hu, Hua Chai, and Kurt Keutzer. An end-to-end visual-audio attention network for emotion recognition in user-generated videos. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 34, pp. 303–311, 2020. 9 

- Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , 2023. 2, 6 

## APPENDIX 

Listing 2: GPT-4V Prompt to calculate correctness of a image dense caption 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs_images/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs.pdf-0016-04.png)


**----- Start of picture text -----**<br>
You are a great critique for analyzing images and captions .<br>Assess the performance of a dense image captioning model based on the correctness of the captions generated .<br>Please assess the correctness of the provided caption in relation to the image. Consider whether the caption accurately<br>identifies and describes the main subjects or objects depicted in the image. Assess whether the caption correctly<br>interprets the relationships between elements within the image, such as actions , interactions , or spatial arrangements.<br>Focus on the precision and accuracy of the information presented in the caption . Provide a score reflecting the level<br>of correctness , ranging from 1 (low correctness ) to 10 (high correctness ) .<br>Listing 3: GPT-4V Prompt to calculate detail of a image dense caption<br>You are a great critique for analyzing images and captions .<br>Assess the performance of a dense image captioning model based on the detail of the captions generated .<br>Evaluate the level of detail captured in the provided caption . Consider how well the caption describes specific attributes ,<br>features , or aspects of the image, including colors , shapes, textures , sizes , and any other relevant details . Assess<br>whether the caption provides comprehensive information about the scene depicted in the image, covering both prominent<br>and subtle elements. Pay attention to the depth and specificity of the details conveyed in the caption . Provide a score<br>indicating the richness of detail , ranging from 1 (low detail ) to 10 (high detail ) .<br>Listing 4: GPT-4V Prompt to calculate quality of a image dense caption<br>You are a great critique for analyzing images and captions .<br>Assess the performance of a dense image captioning model based on the quality of the captions generated .<br>Assess whether the caption is concise yet descriptive , providing meaningful and engaging information about the image. Evaluate<br>the caption ’s ability to evoke a clear mental image corresponding to the visual content . Additionally , consider if the<br>caption is insightful or imaginative in its description . Provide a score reflecting the overall quality of the caption<br>, ranging from 1 (low quality ) to 10 (high quality ) .<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs_images/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs.pdf-0016-05.png)


Listing 5: Perceptual Signal Instruction fine-tuning template for the image: http://farm6.staticflickr.com/5106/5670500150 ~~e~~ 035dd2d30 ~~z~~ . jpg 

_<_ SYSTEM _>_ You are an AI visual assistant. Answer all questions as you are seeing the media _<_ SYSTEM _><_ USER _>_ The objects in this image in no particular order are car, dog, frisbee . Give me the order of saliency of these objects , start with the most salient object and end with the least salient object , each in a separate line . Give me the objects only and nothing else . _<_ image _> <_ ASSISTANT _>_ dog frisbee car _<_ ASSISTANT _>_ 

16 

Listing 6: Perceptual Signal Instruction fine-tuning template for the image: http://farm6.staticflickr.com/5106/5670500150 ~~e~~ 035dd2d30 ~~z~~ . jpg 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs_images/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs.pdf-0017-01.png)


_<_ SYSTEM _>_ You are an AI visual assistant. Answer all questions as you are seeing the media _<_ SYSTEM _><_ USER _>_ Assume the given image is broken into a 3X3 grid the regions or tiles being named ”upper−left” ”upper−center ”, ”upper−right ”, ”middle− left ”, ”middle−center”, ”middle−right ”, ”bottom−left ”, ”bottom−center”, ” bottom−right ”. Rank these regions or tiles based on their saliency , give me the line separated ranking of all regions in decreasing order . _<_ ASSISTANT _>_ middle−right bottom−center bottom−right upper−center upper−right middle−center upper−left middle−left bottom−left 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs_images/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs.pdf-0017-03.png)


The Volkswagen ad titled begins with a group of three women seated excitedly in a Volkswagen Jetta, the driver sporting a wide smile as she grips the steering wheel. The voiceover sets a nostalgic tone, likening the Jetta to unforgettable first experiences like a first kiss or hearing indie rock for the first time—a symbol of newfound freedom and excitement. 

As the scenes unfold, the camera captures the trio cruising down winding roads, their laughter blending with the music and the wind tousling their hair. The atmosphere inside the car is one of camaraderie and adventure, with the Jetta serving as the backdrop to their shared moments of joy and spontaneity. Transitioning to a wider shot of the Jetta gliding along a scenic highway, surrounded by lush greenery, the ad evokes a sense of exploration and the open road. The visuals seamlessly blend modern-day cruising with vintage footage of classic Volkswagen vehicles, reflecting on the brand’s 75-year history in America, starting with the beloved Beetle. 

The ad concludes with the Volkswagen logo and the tagline ”An American Love Story”, encapsulating the enduring relationship between Volkswagen and its drivers across generations. This phrase serves as a tribute to Volkswagen’s 75-year history in America, beginning with the iconic Type 1 vehicles fondly known as ”The Beetle” Through its nostalgic narrative and captivating visuals, the teaser promises viewers an immersive journey into the essence of Volkswagen—a timeless icon that has been a part of countless cherished memories on the American road. 

Figure 5: Dense caption generated by Behavior-LLaVA for the video of a Volkswagen ad. The original video is posted at URL: https://www.youtube.com/watch?v=kyuGXPNr-T0. The red-colored text highlights the most important aspects of the video captured by Behavior-LLaVA, demonstrating an understanding of aesthetics, characters, world knowledge, emotion, and spatial relationships. More such examples are given in Figs. 6, 7, 10, and Figs. 8, 9 for images and videos respectively. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs_images/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs.pdf-0017-08.png)


In this captivating painting, a majestic tree takes center stage against a stunning backdrop. The sky above is a poetic split between night and day—a deep, starry blue on one side with a crescent moon, and a warm, radiant blend of yellows and pinks on the other, suggesting either a sunrise or sunset. The tree itself embodies the essence of transition, with one side adorned in lush green foliage and the other stripped bare, symbolizing the timeless dance of seasons and life’s perpetual transformations. This artwork not only captures the eye but also speaks to deeper truths. It invites us to contemplate the delicate balance between light and darkness, growth and dormancy, woven into the fabric of existence. The artist’s brushstrokes, rendered with a serene palette and gentle abstraction, evoke a sense of tranquility and introspection. This painting beautifully reminds us to embrace the fleeting beauty of each passing moment, whether bathed in daylight or cloaked in the quiet mystery of night. 

Figure 6: Dense caption generated by Behavior-LLaVA for an artistic drawing of a tree. 

17 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs_images/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs.pdf-0018-00.png)


The image features a pair of Nike Air Force 1 sneakers, which are a classic and iconic design. The sneakers are positioned in the center of the image, with the left shoe slightly in front of the right one, giving a sense of depth. They are predominantly white with green accents, including the Nike swoosh logo on the side and the outsole. The laces are white, and the midsole appears to be a light beige color. The background is a plain, light color, which contrasts with the sneakers and makes them stand out. Overlaid on the image is a stylized text that reads ”JUST DO IT” in a cursive, flowing script. The text is in a light green color that matches the accents on the sneakers, creating a cohesive color scheme. Below the main text, there is a smaller line of text that says ”New collection,” indicating that these sneakers are part of a new line or series from Nike. 

The overall style of the image is clean and modern, with a focus on the product. The use of the ”JUST DO IT” slogan, which is a well-known tagline for Nike, adds a layer of branding and recognition to the image. The text is designed to be eye-catching and to draw attention to the sneakers, which are the main subject of the image. The composition is balanced, with the sneakers centrally placed and the text evenly distributed around them. 

Figure 7: Dense caption generated by Behavior-LLaVA for a Nike ad. The red-colored text highlights the most important aspects of the video captured by Behavior-LLaVA, demonstrating an understanding of aesthetics, characters, world knowledge, emotion, and spatial relationships. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs_images/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs.pdf-0018-04.png)


In a heart-pounding and visually stunning trailer for Red Dead Redemption, we are thrust into the gritty world of the American frontier. The trailer opens with a voiceover, a chilling warning delivered with calm certainty: ”Listen to me, we don’t want to kill any of you... But trust me, we will.” Scenes flash by in quick succession, each more intense than the last. We see a lone figure, silhouetted against a setting sun, riding a magnificent horse through a sprawling, golden field. The rugged beauty of the landscape contrasts sharply with the impending sense of danger. Cut to a dimly lit saloon where a group of hardened men sit around a table, cards in hand, tension thick in the air. The voiceover continues, ”This whole thing is pretty much done. We’re more ghosts than people.” A flurry of action unfolds: a quick draw in a darkened room, bullets slicing through the air with deadly precision. The voiceover reminisces, ”Good old Dutch, my best friend... You know how we met? A pair of hucksters trying to rob each other... Back in ’78 or thereabouts.” The visuals intensify as we witness a robbery in progress, chaos erupting as masked figures burst into a bank. ”Ladies and gentlemen, this is a robbery,” declares one of the outlaws, setting the stage for a clash between lawlessness and order. 

Amidst the chaos, snippets of camaraderie emerge: ”Sons of Dutch. Makes us brothers.” But looming shadows of betrayal and regret cast doubt on these bonds. ”Sometimes, brothers make mistakes,” acknowledges the voiceover, acknowledging the complexities of loyalty and survival in this unforgiving world. The trailer crescendos with a crescendo of gunfights, horseback chases, and impassioned speeches. ”You’ll never change. I know that,” declares a voice, capturing the immutable nature of the human spirit amidst adversity. Throughout, a thematic motif resonates: ”You have to love yourself a fire.” The elemental symbolism underscores the primal nature of existence in a land where survival demands courage and cunning. As the trailer draws to a close, we’re left with the haunting refrain: ”Stay strong. Stay with me.” The screen fades to black, leaving us hungry for more of this epic tale set against the backdrop of the untamed West. With breathtaking visuals and evocative storytelling, this Red Dead Redemption trailer promises an unforgettable journey through a world where danger lurks around every corner, and the line between hero and outlaw blurs in the dust and shadows of the frontier. 

Figure 8: Dense caption generated by Behavior-LLaVA for the video of the official trailer of the game Red Dead Redemption 2. The original video is posted at URL: https://www.youtube.com/watch?v=eaW0tYpxyp0. 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs_images/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs.pdf-0019-00.png)


The scene is set at a packed stadium buzzing with excitement, the air thick with anticipation as Argentina and France face off in what promises to be an epic World Cup final. Lionel Messi steps up confidently to take a crucial penalty early in the match, the tension palpable as he eyes the goal. The commentator’s voice echoes through the stadium, ”He’s got the ground, he’s got a penalty! A heart beats... And Messi!” The crowd holds its breath as Messi strikes the ball cleanly, sending it into the back of the net. The stadium erupts with cheers as Argentina takes the lead, setting the stage for a thrilling encounter. 

As the game unfolds, both teams showcase their attacking prowess. Argentina’s Angel Di Maria emerges as a standout, capitalizing on a swift counterattack to score a spectacular goal. ”Alvarez, here’s McAllister, Di Maria is the spare man, and how Di Maria! Glorious goal! Argentine gold!” The Argentine fans go wild, waving their flags and chanting passionately. However, France fights back fiercely, earning a penalty of their own which they duly convert. 

The drama intensifies in the second half as Francespeed and skill, scoring a breathtaking equalizer. ’s Kylian Mbapp”Mbappe,´ to Lamp,e dazzles with his´ Mbappe!´ Oh wow!” The match swings back and forth, with both teams pushing for victory. In the dying moments of extra time, the score is deadlocked at 3-3 when Argentina’s goalkeeper makes a crucial save, denying France a late winner. The match heads to a nail-biting penalty shootout. With nerves of steel, Argentina’s players step up and convert their penalties flawlessly, culminating in a decisive save by the goalkeeper. ”Argentina, champions of the world!” The stadium erupts once more, as Messi lifts the World Cup trophy amidst a sea of jubilant fans, marking a historic and unforgettable victory for Argentina 

Figure 9: Dense caption generated by Behavior-LLaVA for the video of Argentina vs France FIFA World Cup Qatar 2022 Highlights. The original video is posted at URL: https://www.youtube.com/watch?v= zhEWqfP6V ~~w~~ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs_images/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs.pdf-0019-05.png)


The composition of the image presents a compelling narrative of a soldier immersed in a war-torn landscape. Positioned amidst a backdrop of explosive chaos and dense foliage, the soldier, adorned in traditional military gear, gazes directly at the viewer with a resolute demeanor. The soldier’s face is camouflaged, blending seamlessly with the olive-green helmet—a hallmark of battlefield attire. Surrounding this central figure, a tableau of activity unfolds: a tank looms in the background, its barrel skyward, suggesting recent action. Other soldiers, alert and vigilant, navigate the dense jungle terrain, underscoring the high stakes of the conflict. The foliage, lush yet foreboding, heightens the palpable tension of the scene. 

Foregrounded by the silhouette of another soldier’s helmeted head, the viewer is drawn into the heart of the action, evoking a sense of shared experience amid the perils of warfare. The realism of the depiction accentuates the emotional weight of the moment, capturing the essence of human resilience amidst the ravages of battle. 

Overall, the image transcends mere representation, offering a poignant reflection on the individual’s role in the broader narrative of war—imbued with suspense, authenticity, and a profound exploration of the human condition within conflict. 

Figure 10: Dense caption generated by Behavior-LLaVA for a painting of a soldier. The model captures many qualitative aspects that are usually missed in common captioning tasks. 

19 

|**Training**|**Model**<br>**Topic**<br>**Sentiment**<br>**Persuasion**<br>**Action**<br>**Reason**<br>**Clubbed**<br>**All labels**|
|---|---|
|Random|Random<br>2.63<br>3.37<br>14.3<br>8.37<br>3.34<br>3.33|
|Zero-shot|VideoChat (Li et al.,2023a)<br>9.07<br>3.09<br>5.1<br>10.28<br>-<br>-<br>Video4096 - GPT-3.5 Generated Story + GPT-3.5 Classifer (Bhattacharyya et al.,2023)<br>51.6<br>11.68<br>79.69<br>35.02<br>66.27<br>59.59<br>Video4096 - GPT-3.5 Generated Story + Flan-t5-xxl Classifer (Bhattacharyya et al.,2023)<br>60.5<br>10.8<br>79.10<br>33.41<br>79.22<br>81.72<br>Video4096 - GPT-3.5 Generated Story + Vicuna Classifer (Bhattacharyya et al.,2023)<br>22.92<br>10.8<br>67.35<br>29.6<br>21.39<br>20.89<br>Video4096 - Vicuna Generated Story + GPT-3.5 Classifer (Bhattacharyya et al.,2023)<br>46.7<br>5.9<br>80.33<br>27.54<br>61.88<br>55.44<br>Video4096 - Vicuna Generated Story + Flan-t5-xxl Classifer (Bhattacharyya et al.,2023)<br>57.38<br>9.8<br>76.60<br>30.11<br>77.38<br>80.66<br>Video4096 - Vicuna Generated Story + Vicuna Classifer (Bhattacharyya et al.,2023)<br>11.75<br>10.5<br>68.13<br>26.59<br>20.72<br>21.00<br>LCBM (Khandelwal et al.,2024)<br>42.17<br>7.08<br>58.83<br>32.83<br>39.55<br>27.91<br>LLaMA-VID w/ only video<br>10.11<br>3.42<br>5.75<br>12.32<br>29.61<br>24.11<br>LLaMA-VID w/ video + GPT-3.5 Story<br>42.72<br>11.05<br>64.02<br>32.07<br>37.76<br>42.33|
||Behavior-LLaVA w/ only video<br>22.65<br>11.13<br>60.04<br>13.39<br>42.66<br>33.33<br>Behavior-LLaVA w/ video + verbalization<br>46.34<br>11.7<br>64.13<br>33.33<br>52.06<br>52.03<br>Ad-LLaVA w/ video + GPT-3.5 story<br>51.16<br>11.33<br>68.03<br>33.11<br>43.26<br>51.45<br>Behavior-LLaVA w/ video + GPT-3.5 story<br>60.09<br>12.84<br>79.94<br>36.12<br>67.10<br>79.18|
||**Improvement of Behavior-LLaVA over LLaMA-Vid**<br>40.66%<br>16.2%<br>24.86%<br>12.62%<br>77.7%<br>87.05%|
|Finetuned|VideoMAE (Tong et al.,2022)<br>24.72<br>29.72<br>85.55<br>11.17<br>-<br>-<br>Hussain et al.(2017)<br>35.1<br>32.8<br>-<br>-<br>48.45<br>-<br>Intern-Video (Wang et al.,2022)<br>57.47<br>36.08<br>86.59<br>5.47<br>6.8<br>-<br>Video4096- Generated Story + Roberta Classifer (Bhattacharyya et al.,2023)<br>71.3<br>33.02<br>84.20<br>64.67<br>42.96<br>39.09<br>LLaMA-VID w/ video + verbalization<br>59.13<br>32.11<br>79.15<br>50.93<br>50.32<br>30.13<br>LLaMA-VID w/ video + GPT-3.5 Story<br>63.11<br>35.01<br>84.15<br>55.01<br>57.11<br>45.73|
||Behavior-LLaVA w/ only video<br>58.03<br>22.72<br>84.41<br>26.23<br>59.33<br>51.45<br>Behavior-LLaVA w/ video + verbalization<br>68.32<br>33.92<br>85.93<br>64.72<br>70.89<br>75.34<br>Ad-LLaVA w/ video + GPT-3.5 story<br>66.34<br>36.24<br>84.09<br>58.31<br>68.15<br>78.15<br>Behavior-LLaVA w/ video + GPT-3.5 story<br>71.2<br>39.55<br>86.17<br>65.03<br>80.44<br>81.67|
||**Improvement of Behavior-LLaVA over LLaMA-Vid**<br>12.82%<br>12.97%<br>2.4%<br>18.21%<br>40.85%<br>78.59%|



Table 6: Comparison of various models on two video understanding benchmarks (Hussain et al., 2017; Kumar et al., 2023) consisting of 5 tasks related to video advertisements understanding. The main goal of comparing on these benchmarks is to demonstrate Behavior-LLaVA’s understanding of complex videos. We see that Behavior-LLaVA improves on LLaMA-Vid on 5/5 tasks with an average improvement score of 43.18% in zero-shot and 27.64% in fine-tuned settings. Further, it outperforms the current state-of-the-art on 3/5 tasks in zero-shot and 5/5 in fine-tuned settings. 

|Method<br>LLM|**MSVD-QA**<br>**MSRVTT-QA**<br>**ActivityNet-QA**|
|---|---|
||Acc<br>Score<br>Acc<br>Score<br>Acc<br>Score|
|FrozenBiLM (Yang et al.,2022)<br>DeBERTa-V2<br>VideoLLaMA (Zhang et al.,2023a)<br>Vicuna-7B<br>LLaMA-Adapter (Zhang et al.,2023b)<br>LLaMA-7B<br>VideoChat (Li et al.,2023a)<br>Vicuna-7B<br>Video-ChatGPT (Maaz et al.,2023)<br>Vicuna-7B<br>BT-Adapter (Liu et al.,2023b)<br>Vicuna-7B<br>LLaMA-VID<br>Vicuna-7B<br>LLaMA-VID<br>Vicuna-13B<br>Ad-LLaVA<br>Vicuna-13B<br>Behaviour-LLaVA<br>Vicuna-13B|32.2<br>–<br>16.8<br>–<br>24.7<br>–<br>51.6<br>2.5<br>29.6<br>1.8<br>12.4<br>1.1<br>54.9<br>3.1<br>43.8<br>2.7<br>34.2<br>2.7<br>56.3<br>2.8<br>45.0<br>2.5<br>26.5<br>2.2<br>64.9<br>3.3<br>49.3<br>2.8<br>35.2<br>2.7<br>67.5<br>**3.7**<br>57.0<br>3.2<br>45.7<br>3.2<br>69.7<br>**3.7**<br>57.7<br>3.2<br>47.4<br>3.3<br>70.0<br>3.7<br>58.9<br>3.3<br>47.5<br>3.3<br>70.0<br>3.7<br>59.0<br>3.3<br>47.4<br>3.3<br>70.1<br>3.7<br>59.2<br>3.4<br>47.5<br>3.3|
|**Improvement of Behavior-LLaVA over LLaMA-Vid**|0.14%<br>0%<br>0.5%<br>3%<br>0%<br>0%|



Table 7: Comparison of various models on three conventional video question answering benchmarks consisting of question answers related to action understanding. The main goal of comparing on this benchmark is to show that Behavior-LLaVA does not perform worse on low-level understanding tasks like action recognition. We see that Behavior-LLaVA marginally improves on LLaMA-Vid on 2/3 benchmarks. Further, it performs equivalent to the state-of-the-art in 3/3 benchmarks. 

|**Training**|**Models**|**IAPSa-8**<br>**Abstract**<br>**Emotion6**<br>**Emoset**|
|---|---|---|
|Random|Random|12.5<br>12.5<br>16.67<br>12.5|
|0-shot<br>GPT4-V<br>LLaMA-Vid<br>Ad-LLaVA<br>Behavior-LLaVA<br>**Improvement of Behavior-LLaVA over LLaMA-Vid**|GPT4-V<br>LLaMA-Vid<br>Ad-LLaVA<br>Behavior-LLaVA|83.33<br>71.12<br>65.47<br>79.16<br>43.41<br>43.24<br>40.37<br>45.23<br>43.22<br>43.01<br>43.21<br>44.38<br>57.97<br>64.21<br>49.71<br>50.38|
|||33.54%<br>48.5%<br>23.14%<br>11.39%|
|Finetuned<br>MIDAN (Xu et al.,2022)<br>Stimuli-aware (Yang et al.,2021)<br>LLaMA-VID fnetuned<br>Ad-LLaVA fnetuned<br>Behavior-LLaVA fnetuned<br>**Improvement of Behavior-LLaVA over LLaMA-Vid**|MIDAN (Xu et al.,2022)<br>Stimuli-aware (Yang et al.,2021)<br>LLaMA-VID fnetuned<br>Ad-LLaVA fnetuned<br>Behavior-LLaVA fnetuned|85.96<br>78.34<br>61.66<br>75.75<br>-<br>-<br>61.62<br>78.40<br>84.93<br>71.23<br>62.87<br>80.31<br>85.13<br>71.16<br>62.66<br>79.88<br>87.36<br>81.41<br>72.31<br>83.21|
|||2.86%<br>14.29%<br>15.02%<br>3.61%|



Table 8: Comparison of various models on four image emotion understanding benchmarks (IAPSa-8 (Mikels et al., 2005) Abstract (Machajdik & Hanbury, 2010), Emotion6 (Peng et al., 2015), Emoset (Yang et al., 2023)). The main goal of comparing on these benchmarks is to demonstrate Behavior-LLaVA’s understanding of complex tasks like image emotions. We see that Behavior-LLaVA improves on LLaMA-Vid on 4/4 benchmarks with an average improvement score of 29.14% in zero-shot and 8.95% in fine-tuned settings. Further, it outperforms the current state-of-the-art on 4/4 benchmarks in the fine-tuned settings. 

20 

|**Model**|**Correctness**<br>**Detail**<br>**Quality**<br>**Average**|
|---|---|
|GPT4-V<br>LLaVA-1.6 (34B)<br>LLaMA-Vid (13B)<br>Ad-LLaVA (13B)<br>Behavior-LLaVA(13B)|8.4<br>8.5<br>8.4<br>8.43<br>8.1<br>8.2<br>7.4<br>7.9<br>7.4<br>7.6<br>7.2<br>7.4<br>7.5<br>7.8<br>7.3<br>7.53<br>7.3<br>8.1<br>7.9<br>7.76|
|**Improvement of Behavior-LLaVA over LLaMA-Vid**|-1.3%<br>6.57%<br>9.72%<br>4.8%|



Table 9: Comparison of various models on the image dense captioning task. The main goal of this task is to demonstrate Behavior-LLaVA’s image captioning ability. Despite not being explicitly trained on this task, Behavior-LLaVA performs better than both Ad-LLaVA and LLaMA-Vid on Detail and Quality aspects while losing marginally on correctness. On the aspects of detail and quality, it even outperforms the much larger model of LLaVA-1.6 (34B). 

||**Model**|**Scene**|**Object**|**Action**|**Event**|**Attribute**|**Concept**|**Overall**|
|---|---|---|---|---|---|---|---|---|
|0-shot|LLaMA-VID|47.25|65.26|78.12|28.03|42.33|50.03|51.83|
||Ad-LLaVA|49.10|65.35|77.45|31.45|43.33|50.70|52.91|
||Behavior-LLaVA|52.03|65.33|77.95|32.66|45.67|51.20|54.14|
|**Improvement of Behavior-LLaVA over LLaMA-Vid**||10.12%|0.11%|-0.22%|16.52%|7.89%|2.34%|4.46%|
|0-shot w/ story|Video4096 - GPT-3.5 generated story + Flan-t5-xxl classifer|59.66|98.89|98.96|38.42|67.76|86.99|75.12|
||Video4096 - GPT-3.5 generated story + GPT-3.5 classifer|60.2|99.16|98.72|40.79|67.17|88.6|75.77|
||LLaMA-VID + Generated Story|60.3|99.92|99.01|39.33|66.66|87.33|75.425|
||Behavior-LLaVA + GPT3.5generated story|60.4|99.89|98.23|40.97|67.23|88.33|75.84|
|**Improvement of Behavior-LLaVA over LLaMA-Vid**||0.16%|-0.03%|-0.78%|4.17%|0.85%|1.14%|0.55%|



Table 10: Comparison of various models on the Holistic Video Understanding benchmark (Diba et al., 2020) consisting of 7 VQA tasks. We see that Behavior-LLaVA improves on LLaMA-Vid on 6/7 tasks with an average improvement of 5.88%. Further, it performs equivalent to the state-of-the-art in 6/7 tasks. State of the art is achieved by generating a story and asking Behavior-LLaVA to answer questions based on the generated story. 

|**Task**|**MSRVTT-QA**<br>**CAER**<br>**Emoset**<br>**Comments**<br>**likes/views**|
|---|---|
|Base<br>Salicon [Region]<br>Salicon [Object]<br>BLIFT[Likes/Views]<br>BLIFT[Titles]<br>BLIFT[Comments]<br>BLIFT|58.9<br>75.6<br>45.23<br>6.22<br>-0.1<br>55.6<br>75.8<br>47.25<br>6.25<br>-0.07<br>57.9<br>76.4<br>48.0<br>6.12<br>0.05<br>58.2<br>76.2<br>47.12<br>6.15<br>0.38<br>58.4<br>78.1<br>48.12<br>5.09<br>0.13<br>59.0<br>79.1<br>49.58<br>3.02<br>0.19<br>59.2<br>79.3<br>50.38<br>3.05<br>0.40|



Table 11: Ablation on using comments and/or perception signals from Salicon 

|**Sampling Ratio**<br>**Epoch**|**Likes/Views**_R_2<br>**Comments Perplexity**<br>**Performance**|
|---|---|
|Base-Model<br>0|-0.1<br>6.22<br>0|
|1:1<br>0.5<br>1<br>1.25<br>1.5<br>2<br>2.2|0.11<br>4.71<br>5.49<br>0.22<br>3.95<br>8.23<br>0.33<br>3.19<br>10.97<br>0.35<br>3.13<br>11.79<br>0.38<br>3.08<br>12.31<br>0.4<br>3.05<br>12.57|
|1:2<br>0.5<br>1.05<br>1.45|0.14<br>4.33<br>3.04<br>0.28<br>3.66<br>7.08<br>0.42<br>2.99<br>8.12|
|1:10<br>0.5<br>0.8<br>1<br>1.2|0.15<br>3.43<br>1.38<br>0.31<br>2.78<br>3.44<br>0.38<br>2.46<br>4.48<br>0.46<br>2.13<br>5.51|
|2:1<br>0.5<br>1<br>1.5|0.1<br>5.3<br>3.52<br>0.21<br>4.6<br>7.45<br>0.31<br>3.9<br>9.13|



Table 12: Ablation on different sampling ratios and epochs of training. Sampling ratio is the ratio of behaviour data to multimodal instruct data. Performance is the average increase in 0-shot accuracy on 6 tasks with 250 samples each from the eval set. These tasks include image emotion recognition, video emotion recognition, persuasion strategy classification, MSRVTT, HVU and MSVD-QA 

21 

|**Model**|**Audio Summarization (3 Shot)**<br>**IMDb Sentiment**<br>BLEU<br>ROUGE<br>METEOR<br>0-shot<br>1-shot|
|---|---|
|Behaviour-LLaVA<br>LLaMA-VID<br>VALOR(Chen et al.,2023)|19.0<br>25.1<br>39.3<br>84.1<br>90.2<br>15.1<br>18.3<br>30.7<br>80.3<br>87.9<br>6.6<br>10.0<br>23.9<br>-<br>-|
|**Improvement of Behavior-LLaVA over LLaMA-Vid**|25%<br>37%<br>28.01%<br>4.73%<br>2.61%|



Table 13: Evaluation on audio and text modalities. We evaluate on the audio summarization benchmark(Han et al., 2023) for audio and IMDB sentiment benchmark for text.(Maas et al., 2011) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs_images/iclr_2025_Teaching_Human_Behavior_Improves_Content_Understanding_Abilities_Of_LLMs.pdf-0022-02.png)


Figure 11: Percentage performance improvement over an untrained LLaMA-Vid model, compared across various sampling ratios at different checkpoints. The 1:1 sampling ratio shows the best empirical performance. Performance is averaged over 0-shot accuracy improvements on six tasks with 250 samples each from the evaluation set. These tasks include image emotion recognition, video emotion recognition, and persuasion strategy classification, MSRVTT, HVU and MSVD-QA. The figure also shows the benefit of our data filtering process. While training on unfiltered BLIFT also improves the result over baseline performance of Llama-Vid but data filtering adds more improvement on top of it. 

22 

## A DATASET DESCRIPTIONS 

1. MSVD-QA and MSRVTT-QA: These datasets are based on Microsoft Research Video Description (Chen & Dolan, 2011) and MSR-VTT corpora (Xu et al., 2016b) and are extensively used in many video captioning and question-answering experiments. The MSVD-QA dataset has a total number of 1,970 video clips and 50,505 question-answer pairs. The MSRVTT-QA dataset contains 10K video clips and 243k question-answer pairs. 

2. ActivityNet-QA (Caba Heilbron et al., 2015) is a benchmark primarily for human activity understanding containing 849 hours of video, including 28,000 action instances. 

3. VideoEmotion-8 (Asur & Huberman, 2010) dataset comprises 1,101 user-generated videos sourced from YouTube and Flickr, each containing a minimum of 100 videos per emotional category, as per Plutchik Wheel’s emotion model. 

4. Ekman-6 (Xu et al., 2016a) dataset is compiled from social websites, with each of its 1,637 videos labeled with a single emotion category based on Ekman’s psychological research. 

5. CAER (Lee et al., 2019) dataset, sourced from TV shows, consists of 13,201 clips with an average sequence length of 90, each manually labeled with six basic emotions, aligning with the Ekman-6 dataset. 

6. IAPSa (Mikels et al., 2005) is a subset of IAPS, following the Mikels model with eight emotion categories such as amusement, awe, contentment, excitement, anger, disgust, fear, and sadness. It consists of 395 affective images, marking the first visual emotion dataset with discrete categories. 

7. Emotion6 (Peng et al., 2015) features 1,980 images sourced from Flickr, each labeled by 15 annotators according to the Ekman model, covering six emotion categories: happiness, anger, disgust, fear, sadness, and surprise. 

8. EmoSet (Yang et al., 2023) encompasses a total of 3.3 million images, including 118,102 from social networks and artistic sources, evenly distributed across various emotion categories. Based on the Mikels model, EmoSet is categorized into eight emotion categories. 

9. Abstract (Machajdik & Hanbury, 2010) exclusively consists of color and texture combinations without recognizable objects. Differing from the IAPS dataset where emotions often stem from identifiable objects, the abstract paintings dataset was peer-rated via a web survey, with each image rated approximately 14 times. It comprises 228 images spanning eight categories similar to those in IAPS. 

10. Memento10k(Newman et al., 2020) is a short-term video memorability dataset comprising 10,000 video clips, with 900,000 human memory annotations recorded at various delay intervals. The video clips were, on average, 3s long. 

11. VideoMem (Cohendet et al., 2019) is comprised of 10,000 soundless videos, each lasting 7 seconds, accompanied by memorability scores. Memorability was measured twice: first, shortly after viewing and again 24-72 hours later to capture both short-term and long-term memorability effects. 

12. LaMem (Khosla et al., 2015) dataset is a short-term image memorability dataset comprising of 60000 images. The dataset contains scene-centric images, object-centric images and other types such as images of art, images evoking certain emotions, and other user-generated images. 

13. SUN (Isola et al., 2011) dataset is a short-term image memorability dataset comprising 2222 images that were sourced from the SUN database. 

14. MemCat (Goetschalckx & Wagemans, 2019) dataset is a short-term image memorability dataset comprising 10000 images. It consists of five broader memorability-relevant semantic categories (animal, sports, food, landscapes, vehicles), with 2K exemplars each, further divided into different subcategories (e.g., bear, pigeon, cat, etc. for animal). The images were sourced from existing image sets: ImageNet, COCO, Open Images Dataset, and SUN. 

15. MediaEval (Kiziltepe et al., 2021) utilized publicly available links to short-form video clips, each averaging 6 seconds in duration, with both short-term and long-term memorability scores. Short-term memorability evaluations were conducted on videos viewed within the preceding few minutes, while long-term memorability assessments were based on videos viewed within the previous 24 to 72 hours. 

16. LAMBDA (I et al., 2024) is a long-term memorability consisting of 2205 video ads collected over 1749 participants covering 276 brands. The average video length is 33 seconds and the videos are highly complex, consisting of audio, logos, fast-moving scenes, emotions, _etc_ . 

23 

## B LIMITATIONS 

In this paper, we try to show the hypothesis that training on the behavior modality improves learning of the content modality. We train the models on comments and likes to show this. We test our models on multiple benchmarks and obtain positive results. While we try to cover a wide variety of tasks and while results do conclusively show that the hypothesis is true, yet, we can test on more benchmarks covering even more tasks. Similarly, we can show it with multiple models, other than LLaMA-Vid. 

## C BROADER IMPACTS 

Our paper talks about how behavioral training can positively impact content understanding of VLMs. We think this will be useful in various content understanding applications such as question answering, captioning, etc. 

## D ETHICAL IMPLICATIONS 

This paper demonstrates that training on behavioral modalities enhances the learning of content modalities. Models trained on user interactions such as comments and likes were tested on multiple benchmarks and yielded positive results. While these findings present exciting opportunities for advancing content understanding in AI systems, they also raise important ethical considerations that must be carefully addressed. 

1. No personally identifiable information (PII) is used to improve content understanding. Instead, aggregated behavioral data, including replays, likes, and comments, is utilized, ensuring user privacy. We have implemented rigorous anonymization and aggregation techniques to protect individual user identities. 

2. We explicitly acknowledge the inherent biases that may exist in data sourced from social media platforms such as Reddit and YouTube. Despite our rigorous data filtering and cleaning processes, we recognize that the dataset may still be subject to demographic skews, self-selection bias, and algorithmic influences. These biases could potentially lead to uneven model performance across different user groups or reinforce existing societal biases. To mitigate these issues, we emphasize the importance of considering these broader implications when applying our model and interpreting its results. 

3. We acknowledge the need for greater cultural diversity in our dataset. To address this, we plan to release our artifacts as open-source and encourage community contributions to incorporate multilingual and multicultural data. This could involve expanding the range of subreddits and YouTube channels included in our dataset, with the aim of capturing a more diverse and representative spectrum of receiver behavior across different cultural contexts. By taking this approach, we hope to enhance the ethical considerations and societal impact of our work, providing a more holistic view of behavioral patterns in various cultural settings. However, we also recognize that this approach may introduce new challenges, such as ensuring the quality and reliability of community-contributed data. 

24 

