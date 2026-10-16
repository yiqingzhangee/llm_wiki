## - LARGE CONTENT AND BEHAVIOR MODELS TO UN DERSTAND, SIMULATE, AND OPTIMIZE CONTENT AND BEHAVIOR 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0001-02.png)


**----- Start of picture text -----**<br>
Ashmit Khandelwal [⋆] Aditya Agrawal [⋆] Aanisha Bhattacharyya [⋆] Yaman K Singla [⋆]<br>Somesh Singh Uttaran Bhattacharya Ishita Dasgupta Stefano Petrangeli<br>Rajiv Ratn Shah Changyou Chen Balaji Krishnamurthy<br>Adobe, IIIT-Delhi, State University of New York at Buffalo<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0001-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0001-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0001-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0001-06.png)


**----- Start of picture text -----**<br>
Verbalized Content<br>and Behavior<br>Text Video Image Behavior<br>Behavior I have two ad variants A and B, which one will the user like? Variant A<br>Simulation Today is Jan 1 2023, how many people in Texas will buy this shoeby Jul 1 2023? 224k<br>SimulationContent If my target segment is new parents, how can I modify mymessage such that I get a click through rate of 2.5%? Content-BehaviorLarge  Make the baby face the camerarather than her mother<br>Content What is the color of the wall behind the man? Model (LCBM) Emily was elated to see herWhite<br>Understanding What does the narrator say when Emily gets into the car? grandma after a long time<br>UnderstandingBehavior If 100k people viewed this food ad, why did only 500 peopleclick on it? The ad sells cheese burger but gets shownduring morning time. People usually<br>prefer healthier options during morning.<br>**----- End of picture text -----**<br>


Figure 1: Encoding and predicting content (images, videos, and text) and behavior in the language space. Large Content Behavior Models (LCBMs), once trained, can enable a host of different applications, including behavior simulation, content understanding, content-behavior optimization, and content-behavior understanding. 

## ABSTRACT 

Shannon, in his seminal paper introducing information theory, divided the communication into three levels: _technical_ , _semantic_ , and _effectivenss_ . While the technical level is concerned with accurate reconstruction of transmitted symbols, the semantic and effectiveness levels deal with the inferred meaning and its effect on the receiver. Thanks to telecommunications, the first level problem has produced great advances like the internet. Large Language Models (LLMs) make some progress towards the second goal, but the third level still remains largely untouched. The third problem deals with predicting and optimizing communication for desired receiver behavior. LLMs, while showing wide generalization capabilities across a wide range of tasks, are unable to solve for this. One reason for the underperformance could be a lack of “behavior tokens” in LLMs’ training corpora. Behavior tokens define receiver behavior over a communication, such as shares, likes, clicks, purchases, retweets, _etc_ . While preprocessing data for LLM training, behavior tokens are often removed from the corpora as noise. Therefore, in this paper, we make some initial progress towards reintroducing behavior tokens in LLM training. The trained models, other than showing similar performance to LLMs on content understanding tasks, show generalization capabilities on behavior simulation, content simulation, behavior understanding, and behavior domain adaptation. Using a wide range of tasks on two corpora, we show results on all these capabilities. We call these models Large Content and Behavior Models (LCBMs). Further, to spur more research on LCBMs, we release our new Content Behavior Corpus (CBC), a repository containing communicator, message, and corresponding receiver behavior. 

> _⋆_ Equal Contribution. Contact ykumar@adobe.com for questions and suggestions. 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0002-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0002-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0002-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0002-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0002-04.png)


**----- Start of picture text -----**<br>
(a) (b) (c) (d)<br>**----- End of picture text -----**<br>


Figure 2: Comparison of GPT-3.5, GPT-4, Vicuna-13B, and LCBM-13B on ((a),(b)) Behavior Simulation on two types of behaviors: replay value prediction and likes/views prediction. The task is, given the video content and channel information, to predict replay values corresponding to each scene and the ratio of likes to views. (a) is the negative RMSE scores, and (b) is the accuracy. (c) Content simulation and behavior understanding tasks. The task for content simulation is, given the channel information and scene-level behavior, to predict the scene content. The task for behavior understanding task is to give the channel information and video content, predict what will be the sentiment of commenters on that Youtube video and explain why they would write. To get the predicted sentiment and explanation scores, six evaluators scored the models’ explanations between 0-5 by comparing the ratings and reasons with the user comments. The annotators didn’t know which model gave the reasoning. (d) Content understanding tasks. We evaluate four tasks: emotion, topic, and persuasion strategy prediction, and action-and-reason understanding. It can be noted that on the behavior simulation, content simulation, and behavior understanding tasks, LCBM performs better than 3-shot GPT-3.5 and 10-shot GPT-4. On the content understanding tasks, LCBM although outperforms similar sized Vicuna model but GPT3.5 performs better. It is noteworthy that GPT-3.5 and GPT-4 are at least 12 times larger than LCBM-13B. Further, we show the behavior domain adaptation results in Table 6. 

## 1 INTRODUCTION 

In this paper, we use communication in the same sense that Shannon & Weaver (1949) use in their seminal paper on communication. Communication, according to them, includes all of the procedures by which one mind may affect another. This includes all forms of expression, such as words, gestures, speech, pictures, and musical sounds. They mentioned that the broad problem of communication can be studied at three levels: technical, semantic and effectiveness. 

**Level A - Technical problem** : How accurately can the symbols of communication be transmitted? 

**Level B - Semantic problem** : How precisely do the transmitted symbols convey the desired meaning? 

**Level C - Effectiveness problem** : How effectively does the received meaning affect conduct in the desired way? 

The technical problem is concerned with the _accuracy_ of transference, the semantic with the interpretation of meaning by the receiver, and the effectiveness problem deals with the success with which the meaning conveyed to the receiver leads to the desired receiver conduct. Basically, the effectiveness problem is concerned with influencing the conduct of the receiver. These three levels build on top of each other, and thus, solving communication at Level C necessarily requires solving it at Level B. 

This work was published in 1949, and since then, the field of communication has progressed to a great extent. In the last 70 years, due to contributions from Shannon and other scientists, we have progressed much on the problem proposed in Level A. This has resulted in the development of various communication tools such as telephones, mobile phones, internet, and many more, making communication so much more accurate that today we can think about video calls across continents. 

With the recent advances in large language models (LLMs) such as BERT (Devlin et al., 2018), GPT-3 (Brown et al., 2020), T5 (Raffel et al., 2020), and many more, we have witnessed a significant improvement in the performance of various Natural Language Processing (NLP) tasks. Now, LLMs in zero or few shot settings are able to perform well on tasks such as question answering, summarization, translation, and many more. This has helped us progress towards solving the _semantic problem_ to a large extent where we can decode the semantic meaning of a message. However, the effectiveness problem still remains unsolved. The problem proposed in Level C is about designing and transmitting messages which can fulfil the communicator’s _objective_ . The objective can be, for 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Video Title : Robber hangs<br>from the 23rd-floor window to<br>escape the police in China Question : Would the average sentiment of the comments of<br>Posted by : South China the audience be positive, neutral, or negative? Explain why.<br>Morning Post<br>Link : youtu.be/P7iDWZiT3Kg LCBM : The average sentiment of the comments of the audience watching<br>the video would be negative. This is because the video shows a man<br>Comments from Video: jumping out of a high-rise building to escape the police. The audience is<br>(not shown to LLMs/LCBM) likely to feel sympathy and sadness towards the man's situation.<br>Commenter : I feel bad for the dude. Nobody in<br>the right mind robs a local shop.. To then hang Vicuna:  The audience would likely have a mix of positive and negative<br>out a window. I wonder what he is going through. sentiment towards the video. Viewers may feel sympathetic towards the<br>person trying to escape and praise their bravery.<br>Commenter : Kinda feel sorry for him when he<br>cried, he must have been in a bad situation to<br>do what he did but it does not justify his actions. GPT-3.5:  The video depicts a dangerous and potentially life-threatening<br>situation, which may cause some viewers to feel uneasy or uncomfortable.<br>Commenter : Aw, i felt his desperation. However, others may find the events exciting or thrilling to watch.<br>sorry young man but you did wrong. :(<br>Video Title : The new Porsche<br>718 Spyder. Perfectlyirrational. Question : Given the video, would the average audience<br>Posted by : Porsche sentiment be positive, neutral, or negative? Explain why.<br>Link : youtu.be/573YaSAP5H0<br>LCBM : Positive average sentiment of the audience. The video features a<br>luxury car driving along a scenic coastal road, which is would evoke a<br>Comments from Video: sense of awe and admiration in the viewers. Additionally, the Porsche 718<br>(not shown to LLMs/LCBM) in the video is an iconic sports car that is widely recognized and respected<br>in the automotive community, contributing to the positive sentiment.<br>Commenter : Dream car. Mid engine, natural<br>aspiration, b6, rwd, manual, lightweight, open top Vicuna : The average sentiment of the comments would be positive. The<br>video showcases a beautiful scenic drive through the mountains, which is<br>Commenter : All the makings of a epic sports car 👌 likely to evoke a sense of awe and enjoyment in the viewers. Additionally,<br>the presence of a luxury car in the video may also contribute to a positive<br>Commenter : This car is perfect. I’m in sentiment, as it may be seen as a symbol of success and wealth.<br>love. New dream car, no questions asked!<br>GPT-3.5 : The sentiment of the audience watching the video is positive<br>Commenter : I add my vote and wishes to yours ... because the video showcases a new sports car, which is exciting and<br>a great car I’d love to own and pilot desirable to car enthusiasts.<br>**----- End of picture text -----**<br>


Figure 3: A few examples showing LCBM’s ability to understand and explain human behavior of audience sentiment. We also compare it against other models like Vicuna and GPT-3.5. 

example, to persuade the receiver to buy a product, to convince the receiver to vote for a candidate, or to make the student understand a concept or something else. 

**How do we solve the effectiveness problem?** In order to solve the effectiveness problem, we can take inspiration from how the semantic problem is being solved. Raffel et al. (2020) in their seminal work on T5, mention that the basic idea underlying large language models is to treat every text processing problem as a “text-to-text” problem, _i.e._ , taking text as input and producing new text as output. This framework allows for a direct application of the same model, objective, training procedure, and decoding process to every task we consider. Further, this allows us to pre-train a model on a data-rich task like the next-word prediction, which can then be transferred to downstream tasks. Notably, thanks to the Internet, the next-word prediction task has huge amounts of available data. Consider the Common Crawl project[1] , one common source of data included in most language models; it produces more than 20TB of text per month sampled from random web pages across the internet. 

T5 and other language models like GPT-3, Pythia (Biderman et al., 2023), Llama (Touvron et al., 2023), _etc._ showed that these language models can solve a wide variety of tasks, including the ones for which they were not explicitly trained. For instance, language models trained on the next word prediction task showed generalization capabilities across a wide variety of tasks like questionanswering, summarization, nautral language inference, and translation (Brown et al., 2020). Recently, a series of papers have shown that this generalized “world understanding” captured in LLMs can be leveraged to enable them to “see” (Liu et al., 2023; Li et al., 2023b;a; Zhu et al., 2023; Ge et al., 2023; Zhang et al., 2023; Bhattacharya et al., 2023). This is a significant capability enhancement since a model trained in language only settings can be made to reason about images and videos. These papers follow the same transfer learning approach advocated by T5, where they convert visual information to language space to leverage the “text-to-text” framework. They show that it is possible to teach a large language model, the new modality of vision, without needing to pretrain the model from scratch. Rather, using only a few million tokens, it is possible to scale LLMs’ abilities to vision as well. Following this chain of thought, it might be possible to solve the effectiveness problem by posing the problem as a “text-to-text” problem. This is the paradigm we explore in this work. 

> 1https://commoncrawl.org 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0004-00.png)


Figure 4: A few examples showing LCBM’s ability to understand and explain human behavior of scene replayability. We compare it against human-provided explanations of the same. 

**How can we pose the effectiveness problem as a text-to-text problem?** The problem of effect is to know what the receiver does after receiving the message (Shannon & Weaver, 1949). In general, for a piece of content, other than the content itself, we often have information about _who_ consumes the content and what his _action_ is on consuming the content. The latter is the effect described in Shannon’s three levels of communication. For instance, an email as a message from communicator to receiver, elicits certain actions from the receiver like link-clicks, reply, read-time, _etc_ . While LLMs are trained on trillions of tokens of content, the training doesn’t include the receiver effect. For instance, Enron Email (Klimt & Yang, 2004) is a popular corpus which is included in the training of LLMs like Pythia (Biderman et al., 2023). It contains 600k email content sourced from the Enron corporation using which LLMs learn how to write emails. However, it does not contain data about the receiver effect, such as whether the email was opened, how much time it was read, what was the receiver’s reply, _etc_ . Similarly, while a lot of public blogs and user forums are included in major text corpora used in training LLMs like CommonCrawl, they are stripped off of receiver behavior on forum messages like the number of likes, shares, replies, _etc._ before including them in LLM training (for instance, see (Biderman et al., 2022; Penedo et al., 2023)). In order to pose the effectiveness problem as a text-to-text problem, we can include these _behavior tokens_ in text along with content tokens and train the LLM to model both of those in the same space. This might help the LLM simulate the receiver effect, optimize for it, and reason about it. 

Therefore, in this paper, we show initial experiments to integrate the behavior modality as a new modality to increase the scope of multimodal LLMs from content, to content and behavior. We call this new type of model, Large Content Behavior Model (LCBM). This class of models shows promise in enabling the LLMs to not only reason about content but also reason about and predict human behavior over that content. Further, LCBMs also show potential of behavior domain adaptation where models trained on one type of behavior can generalize on another behavior type (Fig. 2). Behavior simulation can enable many real-world applications, such as content recommendation, customer journey optimization, and A/B testing. To build LCBM, we introduce behavior instruction tuning (§2.4), an attempt to extend the instruction tuning paradigm to behavior space, bringing all five communication factors (communicator, message, channel, receiver, and effect) into the same space (Fig. 5). Similar to Brown et al. (2020); Raffel et al. (2020); Liu et al. (2023); Ge et al. (2023), we do not design best-in-class predictors for any of the downstream tasks. Rather, we show a model which shows generalization capabilities across a wide variety of behavior 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0005-00.png)


**----- Start of picture text -----**<br>
description: a sunny<br>football ground<br>objects: a football,<br>players, grass<br>asr: even after hours of<br>Message { practice, } likes: 15k<br>views: 60M<br>scene replays:<br>Youtube {[73,85,99,...,24]}<br>Communicator { } Receiver Effect<br>Nike Nike Subscribers/<br>{Subscribers: 754K} Channel {General YoutubeAudience }<br>**----- End of picture text -----**<br>


Figure 5: Five factors of communication: Communicator, Message, Channel, Receiver, and Effect. 

and content related tasks. Therefore, our paper makes the following contributions: 

**Large Content Behavior Model (LCBM)** : We develop a large multimodal model that shows capabilities of behavior simulation (given content), content simulation (given behavior), content understanding, and behavior understanding (Fig 1). Following the text-to-text framework, we connect the Vicuna LLM (Touvron et al., 2023; Chiang et al., 2023) with an open-set visual encoder of EVA-CLIP (Sun et al., 2023) and instruction finetune it end-to-end on behavior instruction data. EVA-CLIP and QFormer (Li et al., 2023a) help the model to understand visual content in the language space, making it a Vision Language Model (VLM). During behavior instruction tuning, we teach the model to predict behavior given content and content given behavior using various instruction tasks (§2.4). This helps us teach behavior modality to the VLM while grounding it in the natural language space. We use two datasets to show the performance of LCBM: a public dataset consisting of Youtube videos as the content and corresponding receiver behavior of retention graph, likes, the number of views, and comment sentiment, and an internal dataset of Adobe Marketing Emails and the behavior of click-through rate corresponding to each segment they were sent to. We observe that teaching the LCBM behavior and content simulation improves its capabilities on them (expected), but also the model now shows signs of domain-adaptation in behavior modality (few-shot capability, _unexpected_ ) (Table 6) and improvements in behavior understanding (Figs. 3,4,§3) (zero-shot capability, _unexpected_ ) (Brown et al., 2020). See Fig. 2 for a radar plot of all the capabilities and comparisons of performances across LCBM and large state-of-the-art LLMs - GPT-3.5 and GPT-4. 

**Dataset and Test Benchmark:** To spur research on the topic of large content and behavior models, we release our generated behavior instruction finetuning data from over 40,000 Youtube videos in the public domain. The data contains Youtube video links, automatically extracted key scenes, scene verbalizations, replay graph data, video views, likes, comments, channel name, and subscriber count at the time of collection. We also release a benchmark to test performance on the joint content behavior space (§2.3), introducing two types of tasks in this space: predictive and descriptive. In the predictive benchmark, we test the model’s ability to predict behavior given content and content given behavior. In the descriptive benchmark, we validate its explanation of human behavior by comparing it with ground-truth annotations we obtain from human annotators trying to explain human behavior. See Figs. 3,4 for a few examples. 

## 2 SETUP 

In this section, we introduce our approach to model content and behavior together as a text-to-text problem. Since most publicly available corpora strip-off receiver behavior from content, we first introduce our dataset, “The Content Behavior Corpus (CBC)”, a dataset consisting of content and the corresponding receiver behavior. Next, we introduce our methodology to convert the content and behavior into text and our approach to model it using an LLM. Then, we cover the tasks through which we test various capabilities of LCBM (Fig. 1): content-understanding, behavior understanding, content simulation, behavior simulation, and behavior domain adaptation. 

## 2.1 THE CONTENT BEHAVIOR CORPUS (CBC) 

Much of the progress in the LLM space has been made thanks to the availability of large-scale unlabeled text data set for unsupervised learning. In this paper, we are interested in modeling content 

5 

and the corresponding receiver behavior in the same space. While the literature datasets contain trillions of content tokens (text, images, audio, and videos), unfortunately, the largest sources of these tokens do not contain the receiver effect. To address this, we utilize Youtube, a large publicly available source of content-behavior data, consisting of (a) channel name, channel description, and number of subscribers ( _communicator data_ ), (b) rich content in the form of videos, creator-provided titles and descriptions ( _message_ ), (c) behavior in the form of likes, views, user comments, and replay graph ( _receiver effect_ ). This covers all the five factors of communication (Fig. 5), with the channel being fixed as Youtube and receivers being average channel subscribers and viewers of the communicator. Since video data is in the form of videos, and behavior data is in the form of numbers, to model it using a text-to-text paradigm, we _verbalize_ both of them following the methodology we detail next. 

_Verbalization:_ For the video _V_ , Youtube provides us with 100 average viewer retention values ( _ri{i ∈_ [0 _.._ 100) _}_ ) corresponding to the entire video. The sampling rate 100 is constant and independent of video length ( _T_ ). Replay value _ri_ corresponds to video frames between the timestamps ( _T/_ 100 _∗i, T/_ 100 _∗_ ( _i_ +1)) and denotes how often these frames were replayed compared to the most replayed frames. The metric has a value between 0 and 1 that identifies the video’s relative retention performance at a given point in the video. To accommodate longer video lengths, we merge replay values until _T/_ 100 _∗_ ( _i_ + _j_ ) _− T/_ 100 _∗_ ( _i_ ) _>_ 1 _second, j ∈{i_ + 1 _,_ 100 _}_ . We choose the replay value for this merged group of scenes as _max_ ( _ri, ..., rj_ ). Using this logic, we get replay values, ( _Ri{i ∈_ [0 _..m_ ] _}_ ), where _m_ = _⌊_ 100 _/_ ( _⌈_ 100 _/T ⌉_ ) _⌋_ . Next, we sample two frames randomly corresponding to each _i ∈_ [0 _..m_ ]. We caption the frames using BLIP (Li et al., 2023a). We also obtain the automatic speech recognition for the speech for the video between the timestamps corresponding to replay value _Ri_ using Whisper (Radford et al., 2023). The ASR and BLIP captions are content for scenes and replay values are the behavior corresponding to it. We include the scene content and behavior in the video verbalization (Listing 1). 

We also include video content by encoding video frames through EVA-CLIP (Sun et al., 2023) (explained in §2.2). Other than video embeddings, we include the video title and description as part of video content. Corresponding to the overall video content, we verbalize overall video behavior metrics like video views and the ratio of likes and views. Finally, we append it with communicator information of video channel and subscriber count. The Listing 1 presents the overall verbalization for video and frame level content and behavior. 

Listing 1: Verbalization pattern for inputting content and behavior in the same space 

|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9|Input :<br>_<_video_>_..[Video Tokens] .. _<_/video_>_<br>The video has the following scenes:<br>Scene 1: _{_ASR: Welcome to a quick tutorial , OCR: Adobe Premiere Pro, Captions: A desktop interface , Replays: 60_}_,<br>Scene 2: _{_ASR: on using Premiere Pro to edit , Captions: A computer interface , with an image of a white horse. Objects −<br>Horse, Grass, Fence., Replays: 53_}_,<br>...<br>It was posted by Adobe with the<br>title<br>'Using Premiere Pro like a Pro' on Aug 15 2022. Adobe has 100k subscribers . This<br>video was viewed by 346 thousand people and liked (as a percentage of likes /views) by 2.3%|
|---|---|



## 2.2 MODEL 

To understand both visual and textual contents, we follow a similar approach as was taken by recent models like BLIP, Llava, VideoLlama, and others (Liu et al., 2023; Ge et al., 2023; Li et al., 2023a; Zhu et al., 2023), we use visual encoders to encode visual knowledge and an LLM to encode text and world knowledge. Fig. 6 shows our architecture to encode visual content into the language space. We include video content by encoding video frames through EVA-CLIP (Sun et al., 2023) and Global Multi-Head Relation Aggregator (GMHRA) from Uniformer (Li et al., 2021). GMHRA helps aggregate the information better across the time dimension. The combination of ViT and GMHRA gives us a good representation for visual content. Next, in order to effectively leverage the LLM’s rich language representations, we use Q-Former from BLIP-2 (Li et al., 2023a) with an extra linear layer and additional query tokens to convert from visual tokens to language tokens. Further, similar to Bhattacharya et al. (2023), we find that while encoding visual tokens is powerful, converting visual content to text adds to the downstream performance. Therefore, we include BLIP caption for each scene along with the scene replay graph. 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0007-00.png)


Figure 6: Encoding and predicting content (images, videos, and text) and behavior in the language space. Strategy to behavior instruction finetune (BFT) LLMs to create LCBMs. Visual concepts are captured through the visual encoder (EVA-CLIP) and world knowledge is captured through LLM (Llama). To leverage the rich knowledge of LLMs, we use GMHRA and QFormer to convert visual tokens of ViT to language tokens which Llama can understand. Further, we find that verbalizing the visual stimulus helps Llama to gather information more explicitly than what is provided by ViT+QFormer. We instruction fine-tune the combined model end to end to predict 1) behavior given content and 2) content given behavior. Snowflake and fire symbols denote the frozen and unfrozen parts of the architecture. 

We use the Llama based Vicuna-13B LLM (Touvron et al., 2023; Chiang et al., 2023) as our base LLM. Similar to prior works (Liu et al., 2023; Ge et al., 2023; Li et al., 2023a; Zhu et al., 2023), we follow a two-stage training paradigm where in the first stage, we utilize the WebVid (Bain et al., 2021), COCO caption (Chen et al., 2015), Visual Genome (Krishna et al., 2017), CC3M (Sharma et al., 2018), and CC12M (Changpinyo et al., 2021) datasets to align the visual encoder embeddings with LLM. In the second stage, we train the model with behavior instructions prepared by following the approach described in §2.4. In summary, LCBM takes concatenated inputs of visual tokens, scene ASR, caption, scene behavior of replays, channel information, and video title and behavior metrics of views and a ratio of likes to views. Based on the instruction, we test LCBM’s abilities on various tasks we cover in the next paragraphs. 

## 2.3 CONTENT BEHAVIOR TEST BENCHMARK 

We test the capabilities of large content-behavior models on predictive and descriptive abilities on content and behavior, as illustrated in Fig: 1. We design the following four tasks to test these capabilities: behavior simulation, content simulation, content understanding, behavior understanding, and behavior domain adaptation. We cover each of these tasks next: 

1) **Behavior Simulation** : We test behavior simulation capability on three behaviors: replay-values, the ratio of likes to views, and the number of views. The common task amongst all of them is that, given the video content, title, scene-by-scene descriptions, channel and subscriber count (sender characteristics), and date of posting, predict the behavior. The behavior which is to be predicted is masked or asked as a question to the LLM. Listing 2 lists the verbalization pattern for this task. For replay value prediction, we test the masked behavior in three settings: _Masked Past_ (all replay values of the first 5-20% scenes are masked), _Masked Future_ (all replay values of last 5-20% scenes are masked), and _Random Masks_ (random masking of replay values for 5-20% scenes). 

Listing 2: Verbalization pattern of videos for the Behavior Simulation task: 

- 1 Input : _<_ video _>_ .. _<_ /video _>_ The video has the following scenes: 2 Scene 1: _{_ ASR: Welcome to a quick tutorial , OCR: Adobe Premiere Pro, Captions: A desktop interface , Replays: [MASK] _}_ , 3 Scene 2: _{_ ASR: on using Premiere Pro to edit , Captions: A computer interface , with an image of a white horse. Objects − Horse, Grass, Fence., Replays: [MASK] _}_ , ... 

- 4 It was posted by Adobe with the title 'Using Premiere Pro like a Pro' on Aug 15 2022. Adobe has 100k subscribers . Can you tell the replay values for scenes 2 to 5. How many times will this video be viewed and liked as a percentage of likes /views? 

5 6 Output: Scene 1: _{_ Replay: 60% 

2) **Content Simulation** : In this case, the task is to predict content given receiver behavior (Listing 3). The task here is that given the video content in terms of scene-by-scene descriptions where the content of one group of five consecutive scenes content is masked, behavior values of all scenes, and channel information, choose the masked scene speech amongst a list of 25 options. The options are chosen randomly from the entire test set. 

Listing 3: Verbalization pattern of videos for the content simulation task: 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0008-00.png)


**----- Start of picture text -----**<br>
1 Input : < video >  ..  < /video >  The video has the following scenes: Scene 1: { ASR: [MASK], Replays: 60%<br>2 ...<br>3 Scene 5: { ASR: has never been, Captions: Colour Pallete , Replays: 47%<br>4 Scene 6: { ASR: been easier , Captions: Colour Pallete , Replays: 54%<br>5 ...<br>6 It was posted by Adobe with the title 'Using Premiere Pro like a Pro' on Aug 15 2022. It is viewed 203k times and liked<br>1.2%<br>7<br>8 Option−1: Welcome to a quick tutorial ,<br>9 Option−2: Samsung Galaxy A20 smartphone,<br>10 ...<br>11 Option−25: regulations . We haven't had.<br>12<br>13 Output: 1. ASR: Welcome to a quick tutorial .<br>**----- End of picture text -----**<br>


3) **Behavior Understanding** : The goal of this task is to check if the model can understand and reason about observed or unobserved receiver behavior. To test this task, we could ask the model to explain any behaviors we observe with the content. However, only the receiver comments have ground truth available with the video. Therefore, to test a model’s behavior understanding based on the video content, we ask it to simulate what would be the sentiment of the receiver comments. Further, we then ask it to describe why it thought so. To evaluate the reasons, we ask six annotators to annotate the reasons provided by the model on a scale of 0-5. Zero was given when the LLMs provided no sentiment or reasoning. The annotators were free to rate the LLMs as they seemed fit, with one being noisy reasoning and five being perfect. The annotators were asked to review the video content and the comments to help them evaluate the reasons. We average the ratings of three annotators to get an average rating for every video. Similarly, to review the sentiment correctness, we asked the annotators to judge the predicted sentiment rating with respect to user comments. Without ground truth, we found that other behaviors, such as replay values, likes and views, are difficult to explain by non-experts. 

Listing 4: Verbalization pattern of videos for the behavior understanding task: 

- 1 Input : _<_ video _>_ .. _<_ /video _>_ 2 The video has the following scenes: 3 Scene 1: _{_ ASR: Welcome to a quick tutorial , OCR: Adobe Premiere Pro, Captions: A desktop interface , Replays: 60 _}_ , 4 Scene 2: _{_ ASR: on using Premiere Pro to edit , Captions: A computer interface , with an image of a white horse. Objects − Horse, Grass, Fence., Replays: 53 _}_ , 

- 5 ... 6 It was posted by Adobe with the title 'Using Premiere Pro like a Pro' on Aug 15 2022. Adobe has 100k subscribers . This video was viewed by 346 thousand people and liked (as a percentage of likes /views) by 2.3% 

- 7 8 Output: The scene shows the transformation of the image after the changes. 

4) **Content Understanding** : To check if a model trained on both content and behavior tokens does not forget its original content understanding capabilities, we test on the content understanding tasks proposed by Bhattacharya et al. (2023). They use the following tasks for video-understanding: topic, emotion, persuasion, and action-reason classification. For topic, emotion, and action-reason classification tasks, they use the advertisements dataset by Hussain et al. (2017), which contains 3,477 video advertisements and the corresponding annotations for emotion and topic tags and actionreason statements for each video. There are a total of 38 topics and 30 unique emotion tags per video. Further, we have 5 action-reason statements for each video for the action-reason generation task. For our experiment, we use 1785 videos, due to other videos being unavailable/privated from Youtube. Following Bhattacharya et al. (2023), for the topic and emotion classification task, we evaluate our pipeline using top-1 accuracy as the evaluation metric. Further, we evaluate emotion classification on clubbed emotion labels as well. For action and reason prediction, we evaluate our accuracy on the action and reason retrieval tasks where 29 random options along with 1 ground truth are provided to the model to find which one is the ground truth. 

In the persuasion strategy classification, we use the 1002 persuasion strategy videos and corresponding labels released by Bhattacharya et al. (2023). The task is that given the video, the model has to predict which persuasion strategy does the video convey. Persuasion strategy classification could be an important task for evaluating LCBM since the concept of persuasion in psychology views human communication as the means to change the receiver’s beliefs and actions ( _i.e._ to persuade) (Kumar et al., 2023), and understanding the different strategies present in communication may help understand human behavior better. We evaluate the top-1 accuracy of the model on this task. 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0009-00.png)


Figure 7: Examples of Adobe marketing emails used in the Email dataset. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0009-02.png)


**----- Start of picture text -----**<br>
Date Range April 1, 2022 to June 12, 2023<br>Number of Countries 225<br>Target Products Photoshop, Premiere Pro, Acro-<br>bat Professional, InDesign, XD,<br>After Effects, Express, Creative<br>Cloud, ...<br>Customers Segmented on the Type of use, user expertise, fre-<br>basis of quency of use, and others<br>**----- End of picture text -----**<br>


Figure 8: Details of the Adobe Marketing Email dataset used to evaluate behavior generalization capabilities of the LCBM 

5) **Behavior Domain Adaptation** : We have observed strong generalization capabilities from LLMs. LLMs while being trained on next token prediction, show generalization across tasks like question answering, natural language inference, sentiment analysis, _etc_ . Given this, a natural question is, does LCBM, too show this kind of generalization, where a model trained on one kind of data and task, can show performance on another data and task? 

To check this, we test the model on a different dataset and task than what it was originally trained for. We ask the model to predict the click-through rate for a given target segment of an email, given the email content, subject, and verbalized descriptions of the images in the email. We use the emails sent by Adobe’s marketing team to its subscribers. The emails were sent from April 1, 2022 to June 12, 2023 and covered many of the premiere products of Adobe like Photoshop, Premiere Pro, Acrobat, Indesign, _etc_ . The emails were sent to many customer segments (as defined by the marketing team) across 225 countries (Fig. 8). Listing 5 lists the verbalization format to verbalize emails to input to the LCBM. We use this dataset as a test bed to test the behavior transfer learning capabilities of the model. 

Listing 5: Verbalization pattern of emails for the Behavior Domain Adapation Task: 

|1<br>2<br>3<br>4<br>5<br>6<br>7|Input : Email with Subject : Lock it down before you send it out.<br>Header: Nobody messes with your PDFs.<br>Body text : Add password protection , secure encryption , and restricted<br>editing to your PDFs with Adobe Acrobat Pro DC.<br>Share only what you want and nothing more. A button that says 'Get started '. An image of a laptop , with window<br>open on it . Image text : ” Protect using password”.<br>Foreground colors : grey, blue. Background colors: lavender , white. Image Emotions: security , serious . Image keywords:<br>laptop , protect , password, lock. Aesthetic value: low. Clutter<br>level : medium. The email is created by a Creative<br>Professional , for the product Adobe Acrobat Pro. It is sent to users in the United States , in the commercial market<br>.<br>Specifcally , it is sent to Power users with the intent of Active Use.<br>The email was sent 109 times between 25 August, 2022 and 26 August, 2022, and had a click through rate of [MASK]%<br>Output: 0.037%|
|---|---|



## 2.4 BEHAVIOR INSTRUCTION FINE TUNING (BFT) 

In order to teach an LLM the behavior modality over multimodal content, we convert both the visual tokens and behavior modality in the text format and instruction fine-tune the LLM end to end. This follows a two stage approach: first we teach the LLM the visual modality (described in §2.2), and next, we teach the LLM the behavior modality. We call the latter “Behavior Instruction Finetuning (BFT)” inspired by instruction fine-tuning (IFT) and its variants like visual instruction tuning (Liu et al., 2023). 

We prepare the content-behavior instruction datasets for Youtube videos and Stock images as explained next: 

**Teaching behavior in the forward direction** (predict behavior given content): In this instruction tuning task, we teach the model to predict behavior given the message sent by the communicator. Essentially, this teaches the model to predict behavior in the forward direction (as in Fig. 5). Concretely, we include the following information as part of verbalization - video embedding converted to the text space (using EvaCLiP (Sun et al., 2023)), scene-by-scene verbalization covering automatic 

9 

|**Model**<br>**#Params**<br>**Training**<br>**Past**<br>**Future**<br>**RMSE**<br>**Accuracy**<br>**RMSE**<br>**Accuracy**|**Random**<br>**All Masked**<br>**Window Size**<br>**5**<br>**7**<br>**RMSE**<br>**Accuracy**<br>**RMSE**<br>**Accuracy**<br>**RMSE**<br>**Accuracy**<br>8.55<br>61.41<br>9.91<br>55.10<br>-<br>-<br>8.13<br>64.83<br>9.22<br>60.26<br>31.34<br>17.16<br>9.02<br>59.22<br>10.47<br>53.84<br>XXX<br>XXX<br>10.01<br>55.15<br>10.49<br>52.61<br>XXX<br>XXX<br>22.99<br>26.99<br>27.25<br>17.27<br>38.52<br>14.26<br>17.42<br>38.10<br>21.26<br>28.05<br>37.60<br>13.73<br>24.52<br>24.81<br>26.30<br>18.74<br>38.77<br>13.47<br>23.35<br>25.35<br>24.68<br>21.24<br>37.16<br>13.39<br>34.10<br>10.00<br>34.10<br>10.00<br>34.10<br>10.00|
|---|---|
|**LCBM**<br>13B<br>3<br>8.12<br>55.10<br>15.05<br>42.42<br>**LCBM**<br>5<br>11.53<br>52.06<br>12.02<br>53.06<br>**LCBM**<br>7<br>16.17<br>35.61<br>15.14<br>44.11<br>**LCBM**<br>11<br>18.25<br>30.95<br>15.05<br>41.44||
|**GPT-4**<br>_>_100B0<br>10-shot<br>34.45<br>20.55<br>19.51<br>36.08<br>**GPT-4**<br>2-shot<br>35.05<br>19.34<br>18.07<br>39.33||
|**GPT-3.5**<br>175B<br>3-shot<br>34.10<br>19.06<br>24.71<br>27.14<br>**GPT-3.5**<br>2-shot<br>33.36<br>18.02<br>26.44<br>25.42||
|**Random**<br>-<br>-<br>34.10<br>10.00<br>34.10<br>10.00||



Table 1: **Behavior Simulation** : Mean RMSE and accuracy scores for scene-by-scene predictions of video replay values. Replay values are the normalized replay scores of each scene as provided by YouTube. The normalized scores are considered to 2 decimal places and multiplied by hundred to convert the score to an integer score in the range 0-100. RMSE is calculated for each video in the test set and the mean is calculated for this score and reported. The model is said to classify correctly if the absolute error between the predicted and ground truth value is less than or equal to 5. The scores are calculated in four regimes: past, future, random, and all-masked. In the past (future) regimes, first (last) 5-20% scenes are masked, in the random setting, 5-20% scenes are masked randomly, and in all masked setting, everything is masked. We note that after behavior finetuned LCBM, while being at least 10x smaller than other models, performs the best. Best models are denoted in green and runner-ups in blue . 

speech recognition, scene captions, video title and description, receiver behavior covering replay rates, views, and likes, and communicator information covering channel name and subscriber count. The verbalisation pattern for this task is the same as given in the Listing 2. 

**Teaching behavior in the reverse direction** (predict content given behavior): This task teaches the model to learn about behavior in the reverse direction (Fig. 5). Here, the model learns to simulate content given behavior. The instruction for this task is given in Listing 6. 

Listing 6: Verbalization pattern to teach behavior in the reverse direction (predicting content given behavior): 

- 1 Input : _<_ video _>_ .. _<_ /video _>_ The video has the following scenes: Scene 1: _{_ ASR: [MASK], Replays: 60% 2 ... 

- 3 Scene 5: _{_ ASR: has never been, Captions: Colour Pallete , Replays: 47% 

- 4 Scene 6: _{_ ASR: been easier , Captions: Colour Pallete , Replays: 54% 

- 5 ... 

- 6 It was posted by Adobe with the title 'Using Premiere Pro like a Pro' on Aug 15 2022. It is viewed 203k times and liked 1.2% 

- 7 

- 8 Output: Scene 1: _{_ ASR: Welcome to a quick tutorial . _}_ 

Using the prepared content and behavior instruction datasets consisting of pairs of content and behavior tokens, we treat the content tokens ( **X** _C_ ) as input and behavior tokens ( **X** _B_ ) as output of the language model. We then perform instruction-tuning of the LLM on the prediction tokens, using its original auto-regressive training objective. Specifically, for a sequence of length L, we compute the probability of generating target answers ( **X** _B_ ) by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior_images/iclr_2024_Large_Content_And_Behavior_Models_To_Understand,_Simulate,_And_Optimize_Content_And_Behavior.pdf-0010-13.png)


For the behavior instruction tuning, we keep the visual encoder weights frozen and continue to update the pre-trained weights of the LLM in LLaVA. 

## 3 RESULTS AND DISCUSSION 

Here, we discuss the results for the five tasks we discuss in Section 2.3, namely, behavior simulation, content simulation, behavior understanding, content understanding, and behavior domain adaptation. We compare the behavior fine-tuned model discussed in §2.4 with state-of-the-art content-only 

> 0The exact size of GPT-4 is unknown. 

10 

|**Model**|**#Params**|**Training**|**RMSE**|**RMSE**|**R**2|**Accuracy**|**Accuracy**|
|---|---|---|---|---|---|---|---|
|**LCBM**||3|1.31||0.87||15.89|
|**LCBM**|13B|5|1.48||0.82||19.93|
|**LCBM**||7|1.71||0.78||15.20|
|**LCBM**||11|1.55||0.82||13.94|
|**GPT-4**<br>**GPT-4**|_>_100B0|10-shot<br>2-shot|3.50<br>3.58||-0.01<br>-0.03||7.84<br>5.39|
|**GPT-3.5**<br>**GPT-3.5**|175B|3-shot<br>2-shot|64.40<br>64.88|-256.96<br>-375.83|||2.48<br>1.27|
|**Random**|-|-|4.67||0||3.94|



Table 2: **Behavior Simulation** : RMSE, R[2] , and accuracy scores for like/view ratio prediction task. To calculate accuracy, the model is said to classify correctly if the absolute error between the predicted and ground truth likes/views is less than or equal to 10%. We note that LCBM, while being at least 10x smaller than the other models, performs the best. The best results over four runs are reported for all models. Best models are denoted in green and runner-ups in blue . 

|**Model**<br>**#Params**<br>**Accuracy**<br>Vicuna<br>13B<br>19.30%<br>LCBM<br>13B<br>48.68%<br>GPT-3.5<br>175B<br>34.98%<br>Random<br>-<br>4%|**Model**<br>**#Params**<br>**Sentiment Accuracy**<br>**Reasoning Score**|
|---|---|
||**Vicuna**<br>13B<br>65.66%<br>2.23<br>**LCBM**<br>13B<br>72.73%<br>4.00<br>**GPT-3.5**<br>175B<br>61.62%<br>1.67|



Table 3: **Content Simulation** : In this task, the models have to choose the speech segment from a list of 25 options given the video description, nonmasked scenes. and replay behavior. We see that despite being similar to masked language modeling (which is a content-only task), LCBM performs better than both Vicuna and GPT-3.5. Best models are denoted in green and runner-ups in blue . 

Table 4: **Behavior Understanding** : In this task, the models have to simulate the sentiment of comments that a video would get by looking at only the video. Further, they also have to explain the reason for such sentiment. The responses were annotated by humans on a scale of 0- 5 for the reason, with 0 being no response provided and 5 being the response matches exactly with the (ground truth) comments received on the video. Best models are denoted in green and runner-ups in blue . 

models like GPT-3.5, GPT-4, and Vicuna-13B. This allows us to compare how much including behavior tokens in the training of an LLM helps in improving the LLM’s understanding of behavior and joint content and behavior spaces while retaining its understanding of the content space. 

The results for the five tasks are presented in Tables 1,2,3,4,5, and 6. We note a few general trends. LCBM, while being 10x smaller than GPT-3.5 and 4, performs better than them on all behaviorrelated tasks. Further, we see that there is no significant difference between 10-shot and 2-shot GPT4 or between GPT-3.5 and GPT-4, indicating that unlike other tasks, it is harder to achieve good performance through in-context-learning on the behavior modality. It can be observed that often GPT-3.5 and 4 achieve performance comparable to (or worse than) random baselines. Interestingly, the performance of GPTs on the content simulation task is also substantially behind LCBM. The way we formulate the content simulation task (Listing 3), it can be seen that a substantial performance could be achieved by strong content knowledge, and behavior brings in little variance. We still see a substantial performance gap between the two models. All of this indicates that large models like GPT-3.5 and 4 are not trained on behavior tokens. 

For the content understanding tasks (Table 5), predictably GPT-3.5, being the largest model, achieves the best results. However, we see that BFT helps the LLM to learn most content understanding tasks better than the base LLM. LCBM gets better results than both Vicuna and VideoChat. This indicates that behavior modality might carry additional information about content itself, which might help an LLM to understand content better. Next, we see that LCBM also shows signs of domain adaptation in the behavior modality. We see that on three tasks: comment sentiment prediction, comment sentiment reasoning (Table 4), and email behavior simulation (Table 6). We note that if the LCBM is trained on only email behavior simulation samples, it underperforms than the model trained on both Youtube data and few samples to make the model learn email format. Finally, Figs. 3,4 show few samples where we query LCBM to explain replay and comment behavior and compare it with human explanations. We see that LCBM while verbose, can explain behavior well. 

11 

|**Training**<br>**Model**<br>**#Params**|**Topic**<br>**Emotion**<br>**Persuasion**<br>**Action**<br>**Reason**<br>**All labels**<br>**Clubbed**|
|---|---|
|**Random**<br>**Random**<br>-|2.63<br>3.37<br>14.3<br>8.37<br>3.34<br>3.34|
|**Zero-shot**<br>**GPT-3.5**<br>175B<br>**Vicuna**<br>13B<br>**VideoChat**(Li et al., 2023b)<br>13B<br>**LCBM**<br>13B|51.6<br>11.68<br>79.69<br>35.02<br>66.27<br>59.59<br>11.75<br>10.5<br>68.13<br>26.59<br>20.72<br>21.00<br>9.07<br>3.09<br>5.1<br>10.28<br>-<br>-<br>42.17<br>7.08<br>58.83<br>32.83<br>39.55<br>27.91|



Table 5: **Content Understanding** : Comparison of several models including behavior instruction tuned models before and after BFT. We compare the models across topic, emotion, and persuasion strategy detection tasks as per the framework given by Bhattacharya et al. (2023). We see that our model outperforms similarly sized models (Vicuna, VideoChat) in most tasks. Best models are denoted in green and runner-ups in blue . 

|**LCBM Type**|**Fine-tuned**||**Trained**|**On**|**Tested On**|**RMSE**|**R**2|
|---|---|---|---|---|---|---|---|
||**on**|**Unique**|**Unique**|**Email-Segment**||||
||**Youtube?**|**Emails**|**Segments**|**Pairs**||||
|**Domain Adapted**|Yes|100|10|1k|Different Segment<br>(emails could be same)|14.47|0.64|
|**In Domain**|No|600|560k|350k|Different Segment<br>(emails could be same)|25.28|0.55|
|**Domain Adapted**|Yes|100|10|1k|Different Segments<br>and Different Emails|27.28|0.54|
|**In Domain**|No|600|560k|350k|Different Segments<br>and Different Emails|29.28|0.5|



Table 6: **Behavior Domain Adaptation** : In order to test generalisation capabily of LCBM, we calculate its RMSE and R[2] scores on the Adobe email click-through rate simulation task. We train two versions of LCBM with the Adobe Email Marketing data: one which was trained on Youtube videos and further BFT on a few email samples ( _domain-adapted_ ), the other version was BFT on a larger set ( _in-domain_ ). We notice that the domain-adapted LCBM performs better than the in-domain LCBM in both the settings, showing promise of domain-adaptation in the behavior modality. Best models are denoted in green and runner-ups in blue . 

## 4 RELATED WORK 

**Models of Human Communication:** Communication is the situation in which a source transmits a message to a receiver with conscious intent to affect the latter’s behaviors (Osgood et al., 1957; Miller, 1966). Thus, in the most general terms, communication implies a sender, a channel, a message, a receiver, a relationship between sender and receiver, an effect, a context in which communication occurs and a range of things to which ’messages’ refer (McQuail & Windahl, 2015; Lasswell, 1948). As per this, all of the content produced by humanity is essentially communication from a sender to a receiver over some channel and with some effect. Despite much research on communication in social sciences since the 1900s, there has been little adoption of it in machine learning modeling. A prime artefact of this is that the biggest models in machine learning (LLMs) are trained only on content (messages) and ignore other factors in communication (the intended receiver, channel, and behavior) even when they are available. 

**Prior Efforts To Model Behavior:** While there has been much research in ML to model human behavior, it has been disconnected from language and, sometimes, real-world data. For instance, Agent-based modeling (ABMs), a popular paradigm in Reinforcement Learning, has been employed to model behavior (Bankes, 2002; Romero et al., 2023; Park et al., 2023). Nevertheless, ABMs tend to view humans as rational economic agents who communicate primarily through their actions, neglecting the significance of content in communication. In ABMs, agents strive to maximize their rewards, whereas communication does not always aim to optimize specific, well-defined reward signals. Moreover, the scarcity of large repositories containing extensive records of human actions poses a challenge when training ABMs to learn human behavior. Consequently, existing large models trained on human behavior, such as the ABMs and decision transformer and its variants, often rely on simulated data, such as game environments, rather than real human behavior (Chen et al., 2021). This reliance on artificially generated data introduces biases inherent to the creators of the 

12 

training data, making it difficult to capture authentic human behavior. However, recent advancements have demonstrated the potential of large models trained on real-world tokens encompassing various modalities, like images, videos, audio, and text, as the basis for diverse tasks (Ge et al., 2023; Li et al., 2023a). Notably, LLMs, as exemplars of foundation models, have exhibited impressive performance across a range of tasks, including those they were not explicitly trained for, such as emotion recognition, named entity recognition, and complex tasks like table understanding (Ye et al., 2023; Bhattacharya et al., 2023). 

Further, there has also been much work in modeling behavior using conventional modeling techniques like regression, bagging and boosting (Mazloom et al., 2016; Villarroel Ordenes et al., 2019), neural networks (Ding et al., 2019; Wang et al., 2018; Khosla et al., 2014), transformers (Wu & Krahenbuhl, 2021; Xiao et al., 2022), _etc_ . While these models can certainly model behavior, LLMs show generalization powers which extend to capabilities much beyond just behavior simulation. For instance, once trained on behavior tokens, other than behavior simulation, LLMs can now generate behavior optimized content (Table 3), explain behavior (Table 4), and domain-adapt to other behaviors (Table 6), none of which are shown by other models. The other concurrent works which model behavior using LLMs (Kang et al., 2023) model just behavior (for example, by CTR prediction) by attaching classification or regression heads to LLMs and thereby lose out on the text-to-text paradigm where LLMs show their best performance and generalization capabilities. In addition, similar to non LLM paradigm, this method loses out on other capabilities like generating behavior optimized content and explaining behavior. 

## 5 CONCLUSION 

In this paper, we make initial strides towards solving the effectiveness problem proposed by Shannon in his seminal paper on communication. The effectiveness problem deals with predicting and optimizing communication to get the desired receiver behavior. This can be seen as consisting of a string of capabilities: behavior simulation, content simulation, and behavior domain adaptation. We show that while large language models have great generalization capabilities, are unable to perform well on the effectiveness problem. We posit that the reason for this could be a lack of “behavior tokens” in their training corpora. Next, we train LLMs on behavior tokens to show that other than content understanding tasks, the trained models are now able to have good performance across all the behavior-related tasks as well. We also introduce a new Content Behavior Corpus (CBC) to spur research on these large content and behavior models (LCBMs). 

## REFERENCES 

- Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 1728–1738, 2021. 

- Steven C Bankes. Agent-based modeling: A revolution? _Proceedings of the National Academy of Sciences_ , 99 (suppl ~~3~~ ):7199–7200, 2002. 

- Aanisha Bhattacharya, Yaman K Singla, Balaji Krishnamurthy, Rajiv Ratn Shah, and Changyou Chen. A video is worth 4096 tokens: Verbalize story videos to understand them in zero shot. _arXiv preprint arXiv:2305.09758_ , 2023. 

- Stella Biderman, Kieran Bicheno, and Leo Gao. Datasheet for the pile. _arXiv preprint arXiv:2201.07311_ , 2022. 

- Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In _International Conference on Machine Learning_ , pp. 2397–2430. PMLR, 2023. 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020. 

13 

- Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale imagetext pre-training to recognize long-tail visual concepts. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 3558–3568, 2021. 

- Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. _Advances in neural information processing systems_ , 34:15084–15097, 2021. 

- Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. _arXiv preprint arXiv:1504.00325_ , 2015. 

- Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. URL https://lmsys.org/blog/ 2023-03-30-vicuna/. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_ , 2018. 

- Keyan Ding, Ronggang Wang, and Shiqi Wang. Social media popularity prediction: A multiple feature fusion approach with deep neural networks. In _Proceedings of the 27th ACM International Conference on Multimedia_ , pp. 2682–2686, 2019. 

- Yuying Ge, Yixiao Ge, Ziyun Zeng, Xintao Wang, and Ying Shan. Planting a seed of vision in large language model. _arXiv preprint arXiv:2307.08041_ , 2023. 

- Zaeem Hussain, Mingda Zhang, Xiaozhong Zhang, Keren Ye, Christopher Thomas, Zuha Agha, Nathan Ong, and Adriana Kovashka. Automatic understanding of image and video advertisements, 2017. 

- Wang-Cheng Kang, Jianmo Ni, Nikhil Mehta, Maheswaran Sathiamoorthy, Lichan Hong, Ed Chi, and Derek Zhiyuan Cheng. Do llms understand user preferences? evaluating llms on user rating prediction. _arXiv preprint arXiv:2305.06474_ , 2023. 

- Aditya Khosla, Atish Das Sarma, and Raffay Hamid. What makes an image popular? In _Proceedings of the 23rd international conference on World wide web_ , pp. 867–876, 2014. 

- Bryan Klimt and Yiming Yang. The enron corpus: A new dataset for email classification research. In _European conference on machine learning_ , pp. 217–226. Springer, 2004. 

- Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In _Proceedings of the IEEE international conference on computer vision_ , pp. 706–715, 2017. 

- Yaman Kumar, Rajat Jha, Arunim Gupta, Milan Aggarwal, Aditya Garg, Tushar Malyan, Ayush Bhardwaj, Rajiv Ratn Shah, Balaji Krishnamurthy, and Changyou Chen. Persuasion strategies in advertisements. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 37, pp. 57–66, 2023. 

- Harold D Lasswell. The structure and function of communication in society. _The communication of ideas_ , 37 (1):136–139, 1948. 

- Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models, 2023a. 

- Kunchang Li, Yali Wang, Gao Peng, Guanglu Song, Yu Liu, Hongsheng Li, and Yu Qiao. Uniformer: Unified transformer for efficient spatial-temporal representation learning. In _International Conference on Learning Representations_ , 2021. 

- KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. _arXiv preprint arXiv:2305.06355_ , 2023b. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _arXiv preprint arXiv:2304.08485_ , 2023. 

- Masoud Mazloom, Robert Rietveld, Stevan Rudinac, Marcel Worring, and Willemijn Van Dolen. Multimodal popularity prediction of brand-related social media posts. In _Proceedings of the 24th ACM international conference on Multimedia_ , pp. 197–201, 2016. 

- Denis McQuail and Sven Windahl. _Communication models for the study of mass communications_ . Routledge, 2015. 

14 

Gerald R Miller. On defining communication: Another stab. _Journal of Communication_ , 1966. 

- Charles Egerton Osgood, George J Suci, and Percy H Tannenbaum. _The measurement of meaning_ . Number 47. University of Illinois press, 1957. 

- Joon Sung Park, Joseph C O’Brien, Carrie J Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. _arXiv preprint arXiv:2304.03442_ , 2023. 

- Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only. _arXiv preprint arXiv:2306.01116_ , 2023. 

- Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In _International Conference on Machine Learning_ , pp. 28492–28518. PMLR, 2023. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _The Journal of Machine Learning Research_ , 21(1):5485–5551, 2020. 

- Elena Romero, Manuel Chica, Sergio Damas, and William Rand. Two decades of agent-based modeling in marketing: a bibliometric analysis. _Progress in Artificial Intelligence_ , pp. 1–17, 2023. 

- Claude E. Shannon and Warren Weaver. _The mathematical theory of communication._ The mathematical theory of communication. University of Illinois Press, Champaign, IL, US, 1949. 

- Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 2556–2565, 2018. 

- Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. _arXiv preprint arXiv:2303.15389_ , 2023. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 

- Francisco Villarroel Ordenes, Dhruv Grewal, Stephan Ludwig, Ko De Ruyter, Dominik Mahr, and Martin Wetzels. Cutting through content clutter: How speech and image acts drive consumer sharing of social media brand messages. _Journal of Consumer Research_ , 45(5):988–1012, 2019. 

- Ke Wang, Mohit Bansal, and Jan-Michael Frahm. Retweet wars: Tweet popularity prediction via dynamic multimodal regression. In _2018 IEEE winter conference on applications of computer vision (WACV)_ , pp. 1842–1851. IEEE, 2018. 

- Chao-Yuan Wu and Philipp Krahenbuhl. Towards long-form video understanding. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 1884–1894, 2021. 

- Fanyi Xiao, Kaustav Kundu, Joseph Tighe, and Davide Modolo. Hierarchical self-supervised representation learning for movie understanding. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 9727–9736, 2022. 

- Yunhu Ye, Binyuan Hui, Min Yang, Binhua Li, Fei Huang, and Yongbin Li. Large language models are versatile decomposers: Decompose evidence and questions for table-based reasoning. _arXiv preprint arXiv:2301.13808_ , 2023. 

- Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. _arXiv preprint arXiv:2306.02858_ , 2023. 

- Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , 2023. 

15 

