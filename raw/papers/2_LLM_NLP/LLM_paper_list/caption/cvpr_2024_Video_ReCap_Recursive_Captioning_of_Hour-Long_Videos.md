# **Video ReCap: Recursive Captioning of Hour-Long Videos** 

Md Mohaiminul Islam[1] Ngan Ho[1] Xitong Yang[2] Tushar Nagarajan[2] Lorenzo Torresani[2] Gedas Bertasius[1] 1UNC Chapel Hill 2Meta AI https://sites.google.com/view/vidrecap 

## **Abstract** 

_Most video captioning models are designed to process short video clips of few seconds and output text describing low-level visual concepts (e.g., objects, scenes, atomic actions). However, most real-world videos last for minutes or hours and have a complex hierarchical structure spanning different temporal granularities. We propose Video ReCap, a recursive video captioning model that can process video inputs of dramatically different lengths (from 1 second to 2 hours) and output video captions at multiple hierarchy levels. The recursive video-language architecture exploits the synergy between different video hierarchies and can process hour-long videos efficiently. We utilize a curriculum learning training scheme to learn the hierarchical structure of videos, starting from clip-level captions describing atomic actions, then focusing on segment-level descriptions, and concluding with generating summaries for hour-long videos. Furthermore, we introduce Ego4D-HCap dataset by augmenting Ego4D with 8,267 manually collected longrange video summaries. Our recursive model can flexibly generate captions at different hierarchy levels while also being useful for other complex video understanding tasks, such as VideoQA on EgoSchema. Data, code, and models are publicly available at https://sites.google. com/view/vidrecap._ 

## **1. Introduction** 

Many videos in the real world exhibit a hierarchical information structure that spans human behaviors at different temporal granularities (i.e., atomic actions, intermediate activity steps, long-term goals, etc.). However, most modern video captioning models ignore hierarchical video structure and are specifically tailored for short video inputs, typically limited to 5-15 seconds [3, 13, 21, 33, 37, 38, 42, 45, 49, 50, 56, 62]. These short-range captioning methods capture atomic actions and low-level visual details, such as objects and scenes. Moreover, these models are often prohibitively resource-intensive when applied to longer videos, making them ill-suited for understanding human activities occurring 

over long periods (e.g., several hours) [27, 45, 50, 62]. 

In this paper, we investigate a hierarchical video captioning task requiring generating captions at multiple hierarchy levels given a long video input (e.g., several minutes to several hours). Studies in psychology [8, 10, 15] and social cognitive theories [4] have shown the inherent hierarchical structures of human behavior, consisting of atomic actions at the lowest level, intermediate steps in the middle and overall goals/intents at the highest level of the hierarchy. Inspired by these prior studies, we also assume three levels of hierarchies for our video captioning task. At the most granular level, video captions describe individual frames or short video clips of several seconds, focusing on low-level visual elements such as objects, scenes, and atomic actions. As we move up the hierarchy, the short-term captions coalesce into medium-length video segment descriptions spanning activities extending beyond brief moments, such as the intermediate steps within broader activities (e.g., a single step in a cooking recipe) or short segments or sequences within a more extended storyline (e.g., a several minute-long scene within a movie). Lastly, the top level of the hierarchy encapsulates the long-term human goals in the video, intricate relationships between events and characters, and the overarching purpose behind the video, which can be captured via long-range video summaries (See Figure 1). 

The task of hierarchical video captioning poses several technical challenges. Firstly, it necessitates models capable of handling vastly different input lengths, ranging from a few seconds to several hours. This contrasts with most existing methods, designed for fixed video durations of up to a few minutes. Secondly, long-range videos are highly redundant, requiring the model to aggregate only essential information while discarding unimportant visual cues. Thirdly, another critical challenge is comprehending the hierarchical structure in long videos and leveraging the synergy between distinct hierarchies. 

To address these technical challenges, we propose Video ReCap, a model capable of processing videos of dramatically different lengths where input time spans may differ by up to three orders of magnitude (from a handful of seconds 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos_images/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos.pdf-0002-00.png)


Figure 1. **Hierarchical Video Captioning.** We aim to generate hierarchical captions for a long-range video (e.g., several hours long) at three temporal granularities. First, we generate short clip captions for each few seconds of the video focusing on atomic human actions. Afterward, we produce medium-length segment descriptions for every few minutes of the video, capturing the intermediate steps within a longer activity or a video segment within an extended storyline. Finally, our method generates a summary for a long-range video depicting the overall intent and goals of the actors in the video. 

to a few hours) and generating captions at multiple hierarchy levels. Our model encompasses three key attributes that empower its hierarchical video captioning capability. Firstly, Video ReCap adopts a recursive video-language architecture, allowing it to generate captions across distinct hierarchical tiers. At the first level, the model generates captions from features extracted from short video clips, typically lasting a few seconds. As we move up the hierarchy, the model uses sparsely sampled video features and captions generated at the previous hierarchy level as inputs to produce video captions for the current hierarchy level. Such a recursive design effectively leverages the synergy between different video hierarchies and allows us to handle very long video inputs (e.g., up to 2 hours) efficiently. Moreover, it facilitates our model to leverage the powerful reasoning abilities of modern LLMs. Secondly, we implement a curriculum learning scheme, commencing with training on short video clip captions and progressively incorporating data from higher-level hierarchies, namely medium-length segment descriptions and long-range video summaries. Such a hierarchical curriculum learning strategy allows the model to gradually learn the hierarchical structure of the video, starting from short low-level captions to long high-level video summaries. Thirdly, to mitigate the challenge of limited manually annotated hierarchical captioning data, we use LLMs to generate pseudo-summary data spanning different temporal lengths and then use these pseudo-annotations as additional data to train our model. 

To evaluate Video ReCap, we introduce Ego4D-HCap dataset, a new hierarchical video captioning benchmark that contains long-range egocentric videos lasting up to several hours with manually annotated captions at multiple hierarchical levels. To build Ego4D-HCap benchmark, we utilize Ego4D [19], the largest publicly available long-range 

egocentric video dataset, which provides time-stamped captions and video-segment summaries of up to 5 minutes. We then augment the subset of Ego4D videos with manually annotated 8,267 long-range video summaries, where each video spans up to two hours. Consequently, the Ego4DHCap becomes a rich resource with three levels of hierarchical captions for long untrimmed egocentric videos, encompassing captions for short clips, intermediate descriptions for few-minute video segments, and video-level summaries for long video sequences. 

Our results show that Video ReCap outperforms strong prior video captioning baselines [28, 66] across all three temporal hierarchies by a large margin. We also demonstrate that Video ReCap can be effectively used for other complex video understanding tasks, such as long-form video question-answering on EgoSchema [34] where our approach outperforms the previous best method by a substantial margin ( **+18.13%** ). 

## **2. Related Works** 

**Video Captioning Methods.** Early works in video captioning used template-based approaches [24, 26, 42, 48, 60]. Subsequently, these methods were replaced by deep learning methods built using CNN-RNN encoder-decoder architectures [7, 16, 36, 37, 46, 54, 55, 63]. The recent introduction of Transformer [17, 52] led to a plethora of transformerbased video captioning methods [7, 21, 27, 37, 38, 45, 46, 50, 55, 62]. Though these approaches have shown great success in short clip captioning, most are limited to short videos of a few seconds and cannot generate captions spanning multiple temporal hierarchies for hour-long videos. 

**Video Captioning Datasets.** Most existing video captioning datasets contain short video clip inputs (5-30 sec- 

2 

onds) [12, 41, 57, 59]. There exist several datasets with longer videos of 1-5 minutes [22, 25, 67], but the captions of these datasets still focus on short-term visual concepts (e.g., atomic actions, presence of objects, etc.). Instead, our work aims to develop models and datasets for hierarchical video captioning that spans multiple temporal granularity levels ranging from short clip captions to long-range video summaries. To do this, we introduce Ego4D-HCap dataset by augmenting Ego4D with long-range video summaries of hour-long videos. This leads to a hierarchical video captioning dataset consisting of short clip captions, medium-range segment descriptions, and long-range video summaries. **Hierarchical Video Understanding.** Several recent datasets include hierarchical activity annotations for procedural videos [6, 44, 47, 51, 68]. However, these datasets define a fixed taxonomy for the activity labels of each hierarchy and focus on procedural activity recognition. In contrast, we assume free-form natural language descriptions for multiple levels to capture inherent hierarchical structure in real-world videos (not limited to only instructional videos). Aside from the datasets, several methods [2, 29, 65] learn hierarchical feature embeddings for several-minute-long videos (e.g., 5 minutes). In contrast, our work focuses on generating free-form hierarchical captions for hour-long videos at multiple temporal scales. 

## **3. Technical Approach** 

## **3.1. Problem Overview** 

Given a long, untrimmed video input, we aim to generate textual captions at multiple hierarchy levels of the video. Formally, as our inputs, we consider a long-range video sequence _Vi_ = [ _Ii_[(] _[t]_[)][]] _[t]_[=1] _[,...,T]_[comprised][of] _[T]_[RGB][frames,] denoted by _Ii_[(] _[t]_[)][.][Our][goal][is][then][to][generate][captions][at] three distinct hierarchical levels: _Yi_[(] _[ℓ]_[)] = [ _yi,j_[(] _[ℓ]_[)][]] _j_ =1 _,...,|Yi_[(] _[ℓ]_[)] _|_ 

for _ℓ_ = 1 _,_ 2 _,_ 3, where _yi,j_[(] _[ℓ]_[)][depicts a] _[ j][th]_[word in a caption] _[ i]_ for the hierarchy level _l_ . Each hierarchy of captions is generated sequentially starting with the short-term video clip captions, _Yi_[(1)] , describing fine-grained actions and objects occurring within few seconds intervals throughout the video (e.g., a person picks up an apple in Figure 1). Afterward, the model outputs medium-length segment descriptions _Yi_[(2)] , which capture intermediate steps or summaries unfolding over a few minutes of the video (e.g., a person driving a car and parking it in Figure 1). Finally, the model finishes its generation with long-range video summaries _Yi_[(3)] representing video content for the entire video input. 

## **3.2. Recursive Video-Language Model** 

We now describe the Video ReCap model, which contains three high-level components: a Video Encoder, VideoLanguage Alignment, and a Recursive Text Decoder. We 

illustrate our approach in Figure 2 and describe each component below. 

**Video Encoder.** First, we utilize an off-the-shelf video encoder (e.g., TimeSformer [9]) to extract features from a long-range video. Given a short video clip, the video encoder outputs dense spacetime features. We divide the entire video uniformly and extract a sequence of features _Xi_ = [ _xi,j_ ] _j_ =1 _,...,|C|_ , where _|C|_ is the number of video clips, _x ∈_ R _[F][ ×][H][×][W][ ×][D]_ is the spatiotemporal features of a particular clip, _F_ is the number of frames, _H_ is the height, _W_ is the width, and _D_ is the feature dimension. We use dense spacetime features for short-clip captions so that the model can identify low-level visual cues (i.e., objects and atomic actions); for higher-level captions (e.g., segment descriptions and video summaries), we use global features (e.g., CLS features) to reduce the computational cost and capture the global properties of long video inputs. 

**Video-Language Alignment.** Next, we utilize a VideoLanguage (VL) Alignment module which takes the video features, _Xi_ and the captions generated in the previous hierarchy _Yi_[(] _[ℓ][−]_[1)] as input and outputs a fixed number of embeddings _Zi_ = [ _zi,j_ ] _j_ =1 _,...,|Z|_ , where _z ∈_ R _[D][z]_ , _|Z|_ is the number of embeddings, and _Dz_ is the hidden dimension. The objective of the alignment module is to map the video and text features to the joint feature space so that the subsequent text decoder can jointly process both features as in [28]. Moreover, this scheme enables us to compress a large number of video and text features (e.g., several thousand) into a small set of embeddings (e.g., 256), dramatically reducing the computational cost. In particular, we use a frozen pretrained language model (e.g., DistilBERT [43]) to learn a fixed number of video embeddings from the video features _Xi_ by injecting trainable cross-attention layer inside each transformer block of the LM. We also learn a fixed number of text embeddings from the captions generated at the previous hierarchy _Yi_[(] _[ℓ][−]_[1)] by using a similar frozen LM with trainable cross-attention layers. Finally, we concatenate the video and text embeddings to get the joint embeddings _Zi_ , which is used by the subsequent text decoder for generating captions _Yi_[(] _[ℓ]_[)] . Note that the first hierarchy level (i.e., clip caption) has no text features and uses only video embeddings as _Zi_ . 

**Recursive Text Decoder.** We use a pretrained language model (e.g., GPT2 [40]) as our recursive text decoder for generating captions at multiple hierarchy levels. The decoder takes the video-text embeddings _Zi_ produced by the video-language alignment module (described above) and then generates captions _Yi[ℓ]_[for][the][hierarchy] _[ℓ]_[.][Note][that] we use captions generated at the previous hierarchy level _Yi[ℓ][−]_[1] as one of the inputs (along with video features _Xi_ ), which enables a recursive caption generation pipeline. Note that for short-term caption generation (i.e., _Yi_[1][), the textual] 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos_images/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos.pdf-0004-00.png)


Figure 2. **The Video ReCap model** . **(Left)** First, we generate captions for each short clip (e.g., a few seconds long) of the video using the dense spatiotemporal features extracted by a pretrained video encoder (not shown in the figure). **(Middle)** Then Video ReCap produces segment descriptions for every few minutes of the video using sparsely sampled features (e.g., CLS features) and the previously generated clip captions belonging to a particular segment. **(Right)** Finally, Video ReCap generates the full video summary by utilizing sparsely sampled CLS features from the entire video and the previously generated segment descriptions. The Video-Language (VL) Alignment module maps the video and text features to a joint space so that the subsequent text decoder can jointly process them. Note: the yellow box represents the first segment of the video in each of the three panels, zooming in from right to left. 

feature set is initialized as empty (i.e., the base case of our model’s recursion). Following prior works [1, 66], we insert trainable cross-attention blocks inside each transformer layer of our textual decoder and freeze the remaining layers. The cross-attention layer attends to video-text embeddings of the alignment module. Therefore, the proposed Video ReCap models the likelihood of caption _Y_[(] _[ℓ]_[)] conditioned on the video _X_ and the captions generated at lower-level hierarchy _Y_[(] _[ℓ][−]_[1)] using the following training objective: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos_images/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos.pdf-0004-03.png)


Here, _yk_[(] _[ℓ]_[)] denotes the language token of the caption, _y<k_[(] _[ℓ]_[)] is the set of preceding tokens, and _Y_[(0)] = _∅_ . 

## **3.3. Hierarchical Curriculum Learning** 

Training a recursive video-language model is challenging for several reasons. First, the model must process videos of dramatically different input lengths (i.e., from a few seconds to several hours). Second, there is a significant data imbalance where short-term clip captions vastly outnumber the number of video segment descriptions and long-range summaries. Finally, exploiting the synergy between different hierarchy levels is crucial for generating meaningful and contextually relevant captions. To overcome these chal- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos_images/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos.pdf-0004-07.png)


Figure 3. **Hierarchical Curriculum Learning.** We gradually learn the hierarchical structure of the video, starting from short low-level captions to long high-level video summaries. 

lenges, we draw motivation from classic studies of psychology [4, 8, 10, 15], which show a hierarchical organization of human perception of actions. Just as humans first perceive atomic actions before grasping mid-level actions and then infer goals from mid-level activities, our training strategy unfolds in a similar hierarchical fashion. Specifically, our training begins with samples from the lowest hierarchy level, namely clip captions. Subsequently, we train our model with higher-level captions, e.g., medium-length segment descriptions and long-range video summaries. This strategic progression allows the model to gradually understand the intricate hierarchical structure inherent in videos and maximize the synergy between all hierarchies. Moreover, this strategy effectively handles highly imbalanced training data across different hierarchies. Figure 3 shows 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos_images/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos.pdf-0005-00.png)


Figure 4. **Large Language Model Supervision.** Given shortterm ground truth captions, we use an LLM to generate pseudoground truth annotations for medium-length segment descriptions and long-range video summaries to augment our training data. 

an overview of the proposed curriculum learning strategy. 

## **3.4. Additional Supervision using Language Models** 

Collecting captioning annotations for hour-long videos is time-consuming and costly. Thus, another critical challenge associated with hierarchical video captioning is the scarcity of manually annotated hierarchical captioning data, particularly for medium-length segment descriptions and long-range video summaries. We leverage Large Language Models (LLMs) to mitigate this issue. LLMs can effectively incorporate information from text inputs of varying lengths, which aligns perfectly with our objective of guiding the video model to generate captions across multiple hierarchies. Motivated by these insights, we use LLMs to generate a large number of pseudo-caption annotations for medium-length and long-range videos (i.e., our last two hierarchies). The process involves two main steps. First, given manually annotated hierarchical captions, we finetune an LLM teacher to generate medium-length segment descriptions and long-range video summaries from shortterm clip captions concatenated across varying temporal durations. Afterward, we use such LLM-generated pseudo ground truth caption data as additional training samples to train Video ReCap (see Figure 4). Our experiments indicate that such pseudo ground truth data generated by LLMs effectively complements manually annotated data and significantly improves our model’s captioning ability. 

## **3.5. Implementation Details** 

We use TimeSformer [9] as our video encoder to extract features that take an input clip of 4 RGB frames of 224 _×_ 224. We use GPT2 [40] as our default text-decoder, with a hidden dimension of 768 and 12 transformer blocks. We use Adam optimizer [23] with a learning rate of 3 _[−]_[5] and a weight decay of 0 _._ 01. Our training pipeline also utilized cosine scheduling strategy [32]. Please refer to supplementary materials for additional implementation details. 

|HierarchyLevel|# Samples|Avg. Duration|
|---|---|---|
|Clip Caption|5.27M|0.96 sec|
|Segment Description|17.5K|2.87 min|
|Video Summary|8.3K|28.46 min|



Table 1. **Summary of Ego4D-HCap dataset.** 

length video segment descriptions, and long-range videolevel summaries. To construct Ego4D-HCap, we leverage Ego4D [19], the largest publicly available egocentric video dataset. Ego4D videos have several unique features, making them ideal for the hierarchical video captioning task. First, most videos in Ego4D are orders of magnitude longer (e.g., several hours) than the traditional video captioning datasets. Second, egocentric videos typically contain goaldriven and human activities at different hierarchy levels. Third, Ego4D videos capture human behaviors from various scenarios such as cooking, gardening, assembly, etc. 

While Ego4D comes with time-stamped atomic captions and video-segment descriptions spanning up to 5 minutes, it lacks video-level summaries for longer video durations. To address this issue, we annotate a subset of 8,267 Ego4D videos with long-range video summaries, each spanning up to two hours. This enhancement provides a three-level hierarchy of captions, making it a perfect resource for validating the effectiveness of our model on the hierarchical video captioning task. In Table 1, we provide a detailed summary of our introduced Ego4D-HCap subset. Please refer to our supplementary materials for a more detailed analysis of the Ego4D-HCap dataset. 

Our proposed Ego4D-HCap dataset contains videos that capture diverse scenarios in various contexts, such as household settings, outdoor environments, workplaces, leisure activities, and more, totaling 127 distinct scenarios. The distribution of the most common 50 scenarios is illustrated in Figure 5. The distribution of caption lengths for three hierarchy levels in the Ego4D-HCap dataset is illustrated in Figure 6. Notably, clip captions are generally shorter, averaging 7.74 words per caption. In comparison, segment descriptions display a medium length, averaging 15.79 words, while video summaries are the longest, with an average of 25.59 words. Additionally, we observe that the maximum length for a clip caption is 43 words, while segment descriptions and video summaries can extend to 73 and 172 words, respectively. Our supplementary materials include more details on the dataset and our annotation collection process. 

## **5. Experimental Setup** 

## **5.1. Hierarchical Video Captioning Baselines** 

## **4. Ego4D-HCap Dataset** 

We now describe our introduced Ego4D-HCap dataset, a hierarchical video captioning dataset comprised of a three-tier hierarchy of captions: short clip-level captions, medium- 

Hierarchical video captioning is a relatively unexplored task, so there are no well-established baselines for comparing our work. Thus, we introduce the following videolanguage baselines, which we extend for this task. 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos_images/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos.pdf-0006-00.png)


Figure 5. **Distribution of the most common 50 scenarios in Ego4D-HCap dataset.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos_images/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos.pdf-0006-02.png)


Figure 6. **Distribution of the lengths of three hierarchical captions of the Ego4D-HCap dataset.** 

## • **Zero-Shot Baselines:** 

1. **BLIP2** [28]. A zero-shot baseline for _short-term clip captioning_ that utilizes a state-of-the-art image captioning model. 

2. **BLIP2 + GPT3.5** [11, 28]. A zero-shot text-based baseline for _video segment descriptions_ and _longrange video summaries_ . Given BLIP2-generated captions, it uses GPT3.5 to generate video segment descriptions and long-range video summaries. 

3. **LaViLa + GPT3.5** [11, 66]. Similar to the above, a zero-shot baseline for _video segment_ and _summary_ generation using LaViLa captions fed into GPT3.5. 

- **Finetuned Baselines:** 

1. **LaViLa + GPT2** [40, 66]. A fully-finetuned textbased baseline that takes LaViLa-generated clip captions and finetunes a text-only GPT2 model for _segment description_ and _video summary_ generation while keeping the underlying LaViLa model frozen. 

2. **LaViLa + FLAN-T5** [14, 66]. Similar to the above, a fully-finetuned text-based baseline that uses FLANT5 rather than GPT2 for _segment description_ and _video_ 

_summary_ generation. 

3. **LaViLa** [66]. A video-based baseline, finetuned end-to-end to generate _short-term captions, mediumlength segment descriptions_ , and _long-range video summaries_ directly using video inputs. Note that this baseline uses the same video encoder, text decoder, and other experimental settings as our model. 

## **5.2. Our Model Varients** 

1. **Video ReCap.** This variant of our model uses a shared video encoder but separate text decoders and videolanguage alignment modules to generate captions at different hierarchy levels (i.e., the weights across different hierarchies are not shared). Due to the increased model capacity of having specialized modules for each hierarchy, this variant typically produces the best performance. 

2. **Video ReCap-U.** The unified variant using shared parameters across all hierarchies. Since it has a lot fewer trainable parameters than the previous variant, it is more efficient but performs slightly worse in certain settings. 

6 

|Model|Visual<br>Encoder<br>Text<br>Decoder<br>Train<br>Params|ClipCaption|
|---|---|---|
|||CIDEr<br>ROUGE-L<br>METEOR|
|**Zero-Shot**<br>BLIP2 [28]<br>**Finetuned**<br>LaViLa [66]<br>Video ReCap<br>Video ReCap-U|VIT-G<br>FT5-XL<br>0<br>TSF-B<br>GPT2<br>258M<br>TSF-B<br>GPT2<br>339M<br>TSF-B<br>GPT2<br>113M|8.1<br>7.4<br>12.7<br>88.56<br>47.64<br>28.03<br>**98.35**<br>**48.77**<br>**28.28**<br>92.67<br>47.90<br>28.08|



(a) Results for short-range clip captioning. 

||(a)Results for short-range clip ca|ptioning.||
|---|---|---|---|
|Model|Video<br>Encoder<br>Text<br>Decoder<br>Train<br>Params<br>Pseudo<br>Ann.|Segment Description|Video Summary|
|||C<br>R<br>M|C<br>R<br>M|
|**Zero-Shot**<br>BLIP2 [28] + GPT3.5 [11]<br>LaViLa [66] + GPT3.5 [11]<br>**Finetuned**<br>LaViLa [66] + GPT2 [40]<br>LaViLa [66] + FLANT5 [14]<br>LaViLa [66]<br>Video ReCap<br>Video ReCap<br>Video ReCap-U|VIT-G<br>FT5-XL<br>0<br>✗<br>TSF-B<br>GPT2<br>0<br>✗<br>TSF-B<br>GPT2<br>336M<br>✗<br>TSF-B<br>FT5-XL<br>586M<br>✗<br>TSF-B<br>GPT2<br>258M<br>✗<br>TSF-B<br>GPT2<br>339M<br>✗<br>TSF-B<br>GPT2<br>339M<br>✓<br>TSF-B<br>GPT2<br>113M<br>✓|5.68<br>16.87<br>13.47<br>11.13<br>22.41<br>12.10<br>5.79<br>19.77<br>13.45<br>12.16<br>24.49<br>12.48<br>38.22<br>38.10<br>16.58<br>17.98<br>29.48<br>12.81<br>39.13<br>38.77<br>16.88<br>20.12<br>30.06<br>13.17<br>24.63<br>33.31<br>15.30<br>6.54<br>23.97<br>10.95<br>41.74<br>39.04<br>18.21<br>28.06<br>32.27<br>14.26<br>**46.88**<br>**39.73**<br>**18.55**<br>29.34<br>32.64<br>**14.45**<br>45.60<br>39.33<br>18.17<br>**31.06**<br>**33.32**<br>14.16||



(b) Results for medium-length segment description and long-range video summary generation. 

Table 2. **Main Results on the Ego4D-HCap dataset.** All results are evaluated in standard CIDEr (C), ROUGE-L (R) and METEOR (M) metrics. We observe several interesting trends. First, finetuned methods perform significantly better than the zero-shot baselines. Second, the Video ReCap model achieves the best results in video captioning across all three hierarchies, surpassing strong prior baselines such as LaViLa [66]. Third, using LLM-generated pseudo annotations leads to a significant boost in performance. Lastly, the unified variant of the model produces competitive results while having a significantly smaller number of trainable parameters than our standard variant. 

## **6. Results and Analysis** 

## **6.1. Hierarchical Video Captioning Results** 

In Table 2, we present our main results for hierarchical video captioning. We use standard captioning metrics, including CIDEr [53], ROUGE-L [30], and METEOR [5] to evaluate our model on the hierarchical video captioning task. Based on these results, we observe several interesting trends. First, we note that zero-shot baselines (e.g., BLIP2 [28], BLIP2 + GPT3.5 [11], LaViLa + GPT3.5) perform considerably worse than the fully finetuned approaches (e.g., LaViLa [66], LaViLa + GPT2 [40], LaViLa + FLAN-T5 [14]), underscoring the significance of indomain learning on the Ego4D-HCap dataset. Second, we observe that the best performing fully-finetuned text-based baseline LaViLa + FLAN-T5 [14] falls short of our model by 2.61% CIDEr on video segment description and 9.94% CIDEr on video summary generation, despite using significantly more trainable parameters (586M vs 339M). This indicates the benefits of using hierarchical video and text inputs rather than just text for video segment description and long-range video summary generation. Third, we notice that our best performing Video ReCap variant significantly improves upon the strong LaViLa baseline on clip captioning for Ego4D [19], outperforming it by 9.79% CIDEr while employing the same visual encoder, text decoder, and 

training data as our model. We note that while LaViLa uses a transformer resampler [1, 66], our model utilizes a Language Model-based alignment module (see Section 3.2), which we found very effective for this particular task. 

We also note that the performance of LaViLa drops significantly for segment description and video summary generation, indicating its inability to handle long-range videos. In contrast, Video ReCap maintains strong performance on these longer video inputs, outperforming LaViLa by 17.11% CIDEr on segment description and 21.52% CIDEr on video summary generation. We also note that while Video ReCap uses more training parameters than LaViLa (258M vs. 339M), Video ReCap-U has significantly fewer training parameters (113M) than LaViLa but still outperforms LaViLa by substantial margins (+20.97% and +24.50% in CIDEr for segment description and video summary generation respectively). This indicates that the performance gain of our model comes from the recursive and hierarchical design and not from the larger capacity of the model. Our results also indicate that our model’s performance can be further improved (5.14% CIDEr in segment description and 1.28% CIDEr in video summary) by incorporating LLM-based supervision (see Section 3.4). Lastly, the last two rows of Table 2 highlight the trade-off between the two variants of our model, i.e., Video ReCap achieves the highest performance across two out of three 

7 

|Model|Input<br>Feature|Ego4D<br>Pretrain|QA<br>Acc|
|---|---|---|---|
|Random|-|✗|20.0|
|GPT3.5 [11]|Question|✗|19.57|
|FrozenBiLM [61]|Video|✗|26.9|
|VIOLET [18]|Video|✗|19.9|
|mPLUG-Owl [64]|Video|✗|31.1|
|InternVideo [58]|Video|✗|32.1|
|EgoVLP [31]|Video|✓|34.86|
|EgoVLPv2 [39]|Video|✓|34.12|
|LaViLa [66] + GPT3.5 [11]|Captions|✓|44.27|
|Video ReCap + GPT3.5 [11]|Captions|✓|46.03|
|Video ReCap + GPT3.5 [11]|Hier. Captions|✓|**50.23**|



Table 3. **Long-Range VideoQA on EgoSchema [34]** Our approach achieves state-of-the-art results, outperforming the previous best method, InternVideo, by a substantial margin of 18.13%. Furthermore, leveraging the hierarchical captions produced by our model leads to 4.2% and 5.96% boost in performance compared to short-clip captions generated by Video ReCap or LaViLa [66]. 

hierarchies, while the unified variant, Video ReCap-U, attains the second-best performance with significantly fewer trainable parameters. 

## **6.2. Long-Range VideoQA on EgoSchema** 

In Table 3, we validate the effectiveness of our hierarchical video model on the recently introduced long-range video question-answering (VideoQA) EgoSchma dataset [34]. EgoSchema contains over 5K human-curated multiplechoice question-answer pairs spanning 250 hours of realworld videos, requiring hierarchical reasoning over long videos. We use a simple two-stage approach to perform VideoQA on EgoSchema. First, given long EgoSchema video inputs, we generate hierarchical video captions like before. Afterward, we feed our generated hierarchical video captions as inputs to a text-only GPT3.5 [11] and prompt it to answer a question about a given video in a zero-shot manner. The simple framework performs very well on this benchmark despite the simplicity. We first observe that compared to the variant of our method that uses only shortterm captions as inputs to GPT3.5, the variant that uses hierarchical video captions achieves a significant 4.2% boost in performance. We also compare our method with a similar baseline that uses LaViLa-generated short-term captions rather than our hierarchical video captions as inputs to GPT3.5 and show that our approach outperforms this baseline by 5.96%. This highlights the benefits of hierarchical video cues for long-range videoQA. Our results also indicate that our method outperforms the previous best model, InternVideo [58] by a large margin of 18.13%, setting a new state-of-the-art on this benchmark. We note, however, that since InternVideo was never pretrained on Ego4D, the comparison with our approach might be somewhat unfair. Thus, 

|Input|Segment Description|Video Summary|
|---|---|---|
||C<br>R<br>M|C<br>R<br>M|
|Video<br>Text<br>Video + Text|40.17<br>38.65<br>17.59<br>40.10<br>38.02<br>17.41<br>**41.74**<br>**39.04**<br>**18.21**|25.64<br>29.61<br>13.57<br>23.23<br>29.17<br>13.31<br>**28.06**<br>**32.27**<br>**14.26**|



Table 4. **Video-Language Input Ablation.** Using both sparse video features and recursive text inputs leads to better performance for both segment description and video summary generation. 

|Training Scheme|Segment Description|Video Summary|
|---|---|---|
||C<br>R<br>M|C<br>R<br>M|
|Init_→_Segment<br>Caption_→_Segment|36.81 38.70<br>17.17<br>**41.74 39.04**<br>**18.21**|-<br>-<br>-<br>-<br>-<br>-|
|Init_→_Video<br>Caption_→_Video<br>Caption_→_Segment_→_Video|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|8.62<br>26.33 11.24<br>24.84 30.74 13.25<br>**28.06 32.27 14.26**|



Table 5. **Hierarchical Curriculum Learning.** Using the proposed curriculum learning scheme yields a performance boost of +4.93% in segment description and +19.44% in long-range video summary generation compared to training the model from GPT2 pretrained weights (Init). 

in our comparisons, we also include two recent methods, pretrained on Ego4D, EgoVLP [31] and EgoVLPv2 [39]. Note that for all evaluations, we removed all Ego4D videos used by the EgoSchema benchmark from our training set to avoid data leakage. Compared to EgoVLP and EgoVLP2, our approach still achieves the best results, outperforming these two baselines by a significant margin of 16%, indicating the superiority of our method. 

## **6.3. Ablation Studies** 

**Ablation of Input Modalities.** Our model utilizes both video features and recursive text inputs (generated in the previous hierarchy) for the segment descriptions and video summaries. Note that we do not use any text inputs for clip captions as they define the base case of our recursive video model. Since we need to sparsely sample video features to fit long-range videos into GPU memory, we hypothesize that using text as an intermediate representation should complement the sparse video features. In Table 4, we compare our model with a non-recursive baseline (row 2), which only uses sparse video features and a recursive baseline (row 3), which only uses recursive text features. We observe that combining video and text inputs produces a +1.57% boost relative to video-only and a +1.64% boost compared to text-only baselines in CIDEr for segment description generation. Moreover, combining both inputs is more important for long-range video summary generation, where video+text inputs provide +2.42% and +4.83% gains compared to video-only and text-only variants. These experiments reveal that the recursive design of Video ReCap that utilizes both video and text input modalities is crucial for the hierarchical video captioning task. 

8 

|LLM|Segment Description|Segment Description|Video Summary|
|---|---|---|---|
||C<br>R<br>M||C<br>R<br>M|
|GPT2<br>GPT2-L<br>FLAN-T5-S<br>FLAN-T5-L|96.47<br>46.96<br>23.13<br>104.30<br>47.68<br>23.15<br>95.61<br>46.16<br>22.30<br>**125.67**<br>**50.61**<br>**26.06**||40.06<br>33.06<br>14.76<br>43.18<br>33.86<br>15.00<br>43.27<br>34.19<br>14.69<br>**52.08**<br>**36.99**<br>**19.93**|
||(a) Training an LLM Teacher.|||
|Pseudo<br>Ann.|Segment Description|Video Summary||
||C<br>R<br>M|C<br>R<br>M||
|✗<br>✓|41.74<br>39.04<br>18.21<br>**46.88**<br>**39.73**<br>**18.55**|28.06<br>32.27<br>14.26<br>**29.34**<br>**32.64**<br>**14.45**||



(b) Supervision Using the best LLM Teacher (FLAN-T5-Large). 

Table 6. **Importance of LLM Supervision. Top:** Given groundtruth short-term captions concatenated across varying temporal lengths, FLAN-T5-Large generates the highest quality pseudoannotations for segment description and long-range video summary annotations. Using this LLM Oracle, we produce 100K pseudo-annotations for medium-length segment descriptions and 15K for long-range video summaries. **Bottom:** Combining LLMgenerated annotations with manual annotations during training leads to a performance improvement of 5.14% CIDEr for segment description and 1.28% CIDEr for the video summary. 

**Significance of Hierarchical Curriculum Learning.** Next, we investigate the significance of our hierarchical curriculum learning scheme. Table 5 shows the importance of such a curriculum learning scheme. We observe that if we directly train our model on the segment description from GPT2 pretrained initialization, performance drops by a significant margin of 4.93% CIDEr. Moreover, the performance drop is even more catastrophic (-19.44%) for video summary generation without curriculum learning. Finally, we show that it is useful to progressively incorporate higher-level captions, starting from short-term captions, then transitioning to medium-length segment descriptions, and lastly, finishing with long-range video summaries. The variant that progresses from short-term caption to longrange video summary learning directly exhibits a 3.22% drop in CIDEr performance. 

**Importance of LLM-Based Supervision.** Finally, we study the importance of LLM-based supervision for medium-length segment descriptions and long-range video summaries. In Table 6a, we show the performance of different LLM Teachers (e.g., GPT2 [40], and FLAN-T5 [14]) that we use to generate the pseudo ground truth data. We observe that FLAN-T5-Large achieves the best performance in all metrics. Hence, we use FLAN-T5-Large as our Teacher to generate pseudo-ground truth data for segment descriptions and long-range video summaries. Specifically, we produce 100K pseudo-annotations for segment descriptions and 15K for video summaries. We combine these pseudoannotations with the manually annotated data and train our model. Table 6b shows that utilizing supervision from LLMs provides a substantial performance boost in both seg- 

ment description (+5.14% CIDEr gain) and video summary (+1.28% CIDEr improvement) generation performance. 

## **6.4. Qualitative Results on Ego4D-HCap** 

In Figure 7, we present three instances of hierarchical captions generated by our model. It is evident that clip captions mostly describe atomic actions and objects, such as ‘C closes the tap’ (Figure 7 (a)) and ‘C pushes the trolley’ (Figure 7 (b)). In contrast, segment descriptions focus on intermediate concepts within the video spanning longer durations, i.e., ‘C was in the kitchen, washed utensils’ (Figure 7 (a)), and ‘C arranged the tent and interacted with a woman’ (Figure 7 (c)). Moreover, video summaries aim to encapsulate the overarching content and events of the video. For example, ‘C went to the supermarket. C picked up fruits vegetables, and interacted with other people. C bought groceries and paid at the cashier’ (Figure 7 (b)). 

We also notice that while generating clip captions and segment descriptions is relatively more straightforward, generating video summaries is more challenging. For instance, while the generated video summaries of Figure 7 (a) and Figure 7 (b) are of good quality, the video summary of Figure 7 (c) could be further improved. The video summary of Figure 7 (c) fails to capture some important events of the video and includes repeated words and phrases. These challenges highlight the complexity of summarizing content in long-range videos. We anticipate that future advancements and the use of our released data will contribute to the development of more effective methods and models for this demanding task. 

## **7. Conclusions and Future Work** 

We introduce Video ReCap a recursive video captioning model adept at producing hierarchical captions for videos spanning diverse temporal granularities—from brief clip captions to extensive hour-long summaries. The incorporation of a curriculum learning scheme inspired by human psychology and an LLM-based supervision strategy enhances the model’s efficacy in tackling the hierarchical video captioning problem. Beyond its primary focus, our model’s hierarchical captions also proves advantageous for long-range video question answering. Additionally, the curated Ego4D-HCap dataset will be released, intended to catalyze ongoing progress in video understanding research. Some promising future directions include real-time caption generation, interactive video understanding, and videobased dialoguing. 

**Acknowledgements.** We thank Feng Cheng, Yan-Bo Lin, Ce Zhang, Yue Yang, and Soumitri Chattopadhyay for their helpful discussions. Authors from UNC Chapel Hill were supported by the NIH Award R01HD11107402, Sony Faculty Innovation award, Laboratory for Analytic Sciences via NC State University, and ONR Award N00014-23-1-2356. 

9 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos_images/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos.pdf-0010-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos_images/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos.pdf-0010-01.png)


**----- Start of picture text -----**<br>
(a)<br>(b)<br>(c)<br>**----- End of picture text -----**<br>


Figure 7. **Qualitative Results on Ego4D-HCap .** Generally, clip captions depict atomic actions and objects; segment descriptions focus on intermediate concepts, and video summaries encapsulate the overall content and goals of the videos. While generating clip captions and segment descriptions are often relatively easier tasks, developing a good video summary is often challenging. Our models perform well on video summaries (a) and (b), but the generated video summary (c) could be further improved. 

## **References** 

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, 

Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. _Advances in_ 

10 

_Neural Information Processing Systems_ , 35:23716–23736, 2022. 4, 7, 16, 17 

- [2] Kumar Ashutosh, Rohit Girdhar, Lorenzo Torresani, and Kristen Grauman. Hiervl: Learning hierarchical videolanguage embeddings. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 23066–23078, 2023. 3 

- [3] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. _arXiv preprint arXiv:1409.0473_ , 2014. 1 

- [4] Albert Bandura. Social cognitive theory: An agentic perspective. _Asian journal of social psychology_ , 2(1):21–41, 1999. 1, 4 

- [5] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In _Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization_ , pages 65–72, 2005. 7 

- [6] Siddhant Bansal, Chetan Arora, and CV Jawahar. My view is the best view: Procedure learning from egocentric videos. In _European Conference on Computer Vision_ , pages 657–675. Springer, 2022. 3 

- [7] Lorenzo Baraldi, Costantino Grana, and Rita Cucchiara. Hierarchical boundary-aware neural encoder for video captioning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 1657–1666, 2017. 2 

- [8] R.G. Barker and H.F. Wright. _Midwest and Its Children: The Psychological Ecology of an American Town_ . Row, Peterson, 1954. 1, 4 

- [9] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In _ICML_ , page 4, 2021. 3, 5, 14 

- [10] Matthew Botvinick and David C Plaut. Doing without schema hierarchies: a recurrent connectionist approach to normal and impaired routine sequential action. _Psychological review_ , 111(2):395, 2004. 1, 4 

- [11] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 6, 7, 8 

- [12] David Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In _Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies_ , pages 190–200, 2011. 3 

- [13] Sihan Chen, Xingjian He, Longteng Guo, Xinxin Zhu, Weining Wang, Jinhui Tang, and Jing Liu. Valor: Vision-audiolanguage omni-perception pretraining model and dataset. _arXiv preprint arXiv:2304.08345_ , 2023. 1 

- [14] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. _arXiv preprint arXiv:2210.11416_ , 2022. 6, 7, 9 

- [15] Richard P Cooper and Tim Shallice. Hierarchical schemas and goals in the control of sequential behavior. 2006. 1, 4 

- [16] Jeffrey Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, and Trevor Darrell. Long-term recurrent convolutional networks for visual recognition and description. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2625–2634, 2015. 2 

- [17] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv preprint arXiv:2010.11929_ , 2020. 2 

- [18] Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu. An empirical study of end-to-end video-language transformers with masked visual modeling. _2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 22898–22909, 2022. 8 

- [19] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18995–19012, 2022. 2, 5, 7, 16 

- [20] Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory. _Neural computation_ , 9(8):1735–1780, 1997. 14 

- [21] Chiori Hori, Takaaki Hori, Teng-Yok Lee, Ziming Zhang, Bret Harsham, John R Hershey, Tim K Marks, and Kazuhiko Sumi. Attention-based multimodal fusion for video description. In _Proceedings of the IEEE international conference on computer vision_ , pages 4193–4202, 2017. 1, 2 

- [22] Gabriel Huang, Bo Pang, Zhenhai Zhu, Clara Rivera, and Radu Soricut. Multimodal pretraining for dense video captioning. _arXiv preprint arXiv:2011.11760_ , 2020. 3 

- [23] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _CoRR_ , abs/1412.6980, 2014. 5, 14 

- [24] Atsuhiro Kojima, Takeshi Tamura, and Kunio Fukunaga. Natural language description of human activities from video images based on concept hierarchy of actions. _International Journal of Computer Vision_ , 50:171–184, 2002. 2 

- [25] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In _Proceedings of the IEEE international conference on computer vision_ , pages 706–715, 2017. 3 

- [26] Weiyu Lan, Xirong Li, and Jianfeng Dong. Fluency-guided cross-lingual image captioning. In _Proceedings of the 25th ACM international conference on Multimedia_ , pages 1549– 1557, 2017. 2 

- [27] Jie Lei, Liwei Wang, Yelong Shen, Dong Yu, Tamara L Berg, and Mohit Bansal. Mart: Memory-augmented recurrent transformer for coherent video paragraph captioning. _arXiv preprint arXiv:2005.05402_ , 2020. 1, 2 

- [28] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ , 2023. 2, 3, 6, 7, 17 

11 

- [29] Linjie Li, Yen-Chun Chen, Yu Cheng, Zhe Gan, Licheng Yu, and Jingjing Liu. Hero: Hierarchical encoder for video+language omni-representation pre-training. In _Conference on Empirical Methods in Natural Language Processing_ , 2020. 3 

- [30] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In _Text summarization branches out_ , pages 74–81, 2004. 7 

- [31] Kevin Qinghong Lin, Jinpeng Wang, Mattia Soldan, Michael Wray, Rui Yan, Eric Z XU, Difei Gao, Rong-Cheng Tu, Wenzhe Zhao, Weijie Kong, et al. Egocentric video-language pretraining. _Advances in Neural Information Processing Systems_ , 35:7575–7586, 2022. 8 

- [32] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _International Conference on Learning Representations_ , 2017. 5 

- [33] Huaishao Luo, Lei Ji, Botian Shi, Haoyang Huang, Nan Duan, Tianrui Li, Jason Li, Taroon Bharti, and Ming Zhou. Univl: A unified video and language pre-training model for multimodal understanding and generation. _arXiv preprint arXiv:2002.06353_ , 2020. 1 

- [34] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. _arXiv preprint arXiv:2308.09126_ , 2023. 2, 8, 17 

- [35] Ron Mokady, Amir Hertz, and Amit H Bermano. Clipcap: Clip prefix for image captioning. _arXiv preprint arXiv:2111.09734_ , 2021. 17 

- [36] Pingbo Pan, Zhongwen Xu, Yi Yang, Fei Wu, and Yueting Zhuang. Hierarchical recurrent neural encoder for video representation with application to captioning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 1029–1038, 2016. 2 

- [37] Yingwei Pan, Ting Yao, Houqiang Li, and Tao Mei. Video captioning with transferred semantic attributes. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 6504–6512, 2017. 1, 2 

- [38] Wenjie Pei, Jiyuan Zhang, Xiangrong Wang, Lei Ke, Xiaoyong Shen, and Yu-Wing Tai. Memory-attended recurrent network for video captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8347–8356, 2019. 1, 2 

- [39] Shraman Pramanick, Yale Song, Sayan Nag, Kevin Qinghong Lin, Hardik Shah, Mike Zheng Shou, Rama Chellappa, and Pengchuan Zhang. Egovlpv2: Egocentric video-language pre-training with fusion in the backbone. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 5285–5297, 2023. 8 

- [40] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9, 2019. 3, 5, 6, 7, 9, 14 

- [41] Anna Rohrbach, Atousa Torabi, Marcus Rohrbach, Niket Tandon, Christopher Pal, Hugo Larochelle, Aaron Courville, and Bernt Schiele. Movie description. _International Journal of Computer Vision_ , 123:94–120, 2017. 3 

- [42] Marcus Rohrbach, Wei Qiu, Ivan Titov, Stefan Thater, Manfred Pinkal, and Bernt Schiele. Translating video content to 

   - natural language descriptions. In _Proceedings of the IEEE international conference on computer vision_ , pages 433–440, 2013. 1, 2 

- [43] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: Smaller, faster, cheaper and lighter. arxiv 2019. _arXiv preprint arXiv:1910.01108_ , 2019. 3, 14, 16, 17 

- [44] Fadime Sener, Dibyadip Chatterjee, Daniel Shelepov, Kun He, Dipika Singhania, Robert Wang, and Angela Yao. Assembly101: A large-scale multi-view video dataset for understanding procedural activities. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 21096–21106, 2022. 3 

- [45] Paul Hongsuck Seo, Arsha Nagrani, Anurag Arnab, and Cordelia Schmid. End-to-end generative pretraining for multimodal video captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 17959–17968, 2022. 1, 2 

- [46] Jingkuan Song, Yuyu Guo, Lianli Gao, Xuelong Li, Alan Hanjalic, and Heng Tao Shen. From deterministic to generative: Multimodal stochastic rnns for video captioning. _IEEE transactions on neural networks and learning systems_ , 30 (10):3047–3058, 2018. 2 

- [47] Yale Song, Gene Byrne, Tushar Nagarajan, Huiyu Wang, Miguel Martin, and Lorenzo Torresani. Ego4d goal-step: Toward hierarchical understanding of procedural activities. In _Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track_ , 2023. 3 

- [48] Chen Sun and Ram Nevatia. Semantic aware video transcription using random forest classifiers. In _Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I 13_ , pages 772–786. Springer, 2014. 2 

- [49] Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. Videobert: A joint model for video and language representation learning. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 7464–7473, 2019. 1 

- [50] Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. _Advances in neural information processing systems_ , 27, 2014. 1, 2 

- [51] Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. Coin: A large-scale dataset for comprehensive instructional video analysis. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 1207– 1216, 2019. 3 

- [52] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 2 

- [53] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 4566–4575, 2015. 7 

- [54] Subhashini Venugopalan, Marcus Rohrbach, Jeffrey Donahue, Raymond Mooney, Trevor Darrell, and Kate Saenko. 

12 

Sequence to sequence-video to text. In _Proceedings of the IEEE international conference on computer vision_ , pages 4534–4542, 2015. 2 

- [55] Bairui Wang, Lin Ma, Wei Zhang, and Wei Liu. Reconstruction network for video captioning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 7622–7631, 2018. 2 

- [56] Junke Wang, Dongdong Chen, Zuxuan Wu, Chong Luo, Luowei Zhou, Yucheng Zhao, Yujia Xie, Ce Liu, Yu-Gang Jiang, and Lu Yuan. Omnivl: One foundation model for image-language and video-language tasks. _Advances in neural information processing systems_ , 35:5696–5710, 2022. 1 

_on Computer Vision and Pattern Recognition_ , pages 6586– 6597, 2023. 2, 4, 6, 7, 8, 14, 16, 17 

   - [67] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , 2018. 3 

   - [68] Dimitri Zhukov, Jean-Baptiste Alayrac, Ramazan Gokberk Cinbis, David Fouhey, Ivan Laptev, and Josef Sivic. Crosstask weakly supervised learning from instructional videos. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 3537–3545, 2019. 3 

- [57] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, highquality multilingual dataset for video-and-language research. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 4581–4591, 2019. 3 

- [58] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. _arXiv preprint arXiv:2212.03191_ , 2022. 8 

- [59] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 5288–5296, 2016. 3 

- [60] Ran Xu, Caiming Xiong, Wei Chen, and Jason Corso. Jointly modeling deep video and compositional text to bridge vision and language in a unified framework. In _Proceedings of the AAAI conference on artificial intelligence_ , 2015. 2 

- [61] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Zero-shot video question answering via frozen bidirectional language models. _ArXiv_ , abs/2206.08155, 2022. 8 

- [62] Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10714–10726, 2023. 1, 2 

- [63] Li Yao, Atousa Torabi, Kyunghyun Cho, Nicolas Ballas, Christopher Pal, Hugo Larochelle, and Aaron Courville. Describing videos by exploiting temporal structure. In _Proceedings of the IEEE international conference on computer vision_ , pages 4507–4515, 2015. 2 

- [64] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yi Zhou, Junyan Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qiang Qi, Ji Zhang, and Feiyan Huang. mplug-owl: Modularization empowers large language models with multimodality. _ArXiv_ , abs/2304.14178, 2023. 8 

- [65] Bowen Zhang, Hexiang Hu, and Fei Sha. Cross-modal and hierarchical modeling of video and text. In _European Conference on Computer Vision_ , 2018. 3 

- [66] Yue Zhao, Ishan Misra, Philipp Kr¨ahenb¨uhl, and Rohit Girdhar. Learning video representations from large language models. In _Proceedings of the IEEE/CVF Conference_ 

13 

## **Video ReCap: Recursive Captioning of Hour-Long Videos** Supplementary Material 

Our supplementary materials contain Section S1: Additional Implementation Details, Section S2: Ego4DHCap Data Collection Process, Section S3: Ego4DHCap Dataset Analysis, Section S4: Additional Quantitative Results, and Section S5: Qualitative Results on EgoSchema. 

## **S1. Additional Implementation Details** 

Figure S1 Shows the schematic diagram of the proposed Video ReCap model. 

**Video Encoder.** We employ the TimeSformer model [9] as our video encoder. This model, consisting of 12 transformer layers, is pretrained using a contrastive objective [66]. The input to the encoder comprises 4 RGB frames of size 224 _×_ 224. To process the video, we divide it into 4-second clips and extract features for each clip using the pretrained video encoder. For clip caption, we utilize the dense spatiotemporal features. This allows our model to capture finegrained details. However, we only use the CLS features for segment description and video summary, allowing efficient computation. 

**Video-Language Alignment.** We utilize a pretrained language model DistilBERT [43] as our Video-Language (VL) Alignment module. It is a 6-layer transformer encoder model, where we freeze the self-attention blocks and insert a trainable cross-attention module inside each layer. It takes video features output by the video encoder and captions generated at the previous hierarchy as inputs. Note that there are no text inputs for clip captions. For segment description, we extract clip captions at each 4 seconds of the segment, and for video summary, we extract segment descriptions at each 3 minutes of the video and pass them to the VL alignment module along with corresponding video features. 

**Text Decoder.** We leverage a pretrained GPT2 [40]) as our text decoder. It is a 12-layer transformer model, and we insert a gated cross-attention block inside each transformer layer. We train only the cross-attention modules and freeze the rest of the model. Each cross-attention block contains a cross-attention layer and a feed-forward layer, followed by a tanh gating [20]. The tanh-gating is initialized with an initial value of zero so that the model’s output is the same as the pre-trained LLM at the beginning. As the training progresses, the model gradually learns to attend to the videotext embedding output by the VL-alignment module. 

**Training the Video ReCap Model.** We follow a threestage training pipeline for the Video ReCap model. First, we train our model 5 epoch using a batch size of 128 using clip 

caption data, which only uses video features. Afterward, we employ the trained model from the first stage to extract clip captions within the videos at 4-second intervals. Then, during the second stage, we train the model for 10 epochs using a batch size of 32 using segment description samples, which take as input both video features and text features (clip captions). Finally, in the third stage, we extract segment descriptions every three minutes of the video using the trained model of the second stage and further train the model for 10 epochs using a batch size of 32 using video summary data. We use AdamW optimizer with optimizer [23] with ( _β_ 1 _, β_ 2) = (0 _._ 9 _,_ 0 _._ 999) and weight decay 0.01. We use a learning rate of 3 _[−]_[5] and a cosine scheduling strategy. 

**Training the Video ReCap-U Model.** Training a unified model that shares all parameters across three hierarchies is more challenging. We employ a similar three-stage approach with some additional tricks. In particular, the firststage training is identical to the Video ReCap model. However, during the second stage, we train the Video ReCapU model using both clip captions and segment description samples to prevent catastrophic forgetting of clip captions. One particular challenge is that the clip captions and segment description data are quite different. While clip captions use dense spatiotemporal features, segment descriptions utilize CLS features. Moreover, segment descriptions use video and text features as inputs, while clip captions only use video features. To overcome this challenge, we employ an alternate batching pipeline, where we sample a batch of clip captions and segment descriptions alternatively during the training. Since we have a lot more clip caption data ( _∼_ 4 _M_ ) compared to segment descriptions (100 _K_ including manually annotated and LLM-generated pseudo annotations), we randomly sample 100 _K_ clip captions and only used those during the second stage of training. Finally, we train the model during the third stage using samples from all three hierarchies using a similar alternate batching approach. Since we have only _∼_ 20 _K_ (including manually annotated and LLM-generated pseudo annotations) samples for video summaries, we randomly sample 20 _K_ clip captions and 20K segment descriptions and used those along with video summaries during the third stage of training. This strategy prevents catastrophic forgetting of the model. It allows the training of the Video ReCapU model, which shares all parameters across hierarchies. For Video ReCap-U, We use the same learning rate, batch size, training epoch, optimizer, and scheduler for the Video ReCap (See the previous paragraph). 

**Inference.** During inference, we uniformly sample 4 frames from the corresponding clip and extract spatiotem- 

14 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos_images/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos.pdf-0015-00.png)


Figure S1. **Model Architecture.** 

poral features using the video encoder to use as inputs to generate clip captions. For segment description, we extract CLS features and clip captions every 4 seconds of the segment and use them as inputs to generate segment descriptions. Lastly, we extract segment descriptions at each 3 minutes of the video and use them along with pre-extracted CLS features to generate video summaries. Note that clip boundaries are not given during the inference of segment descriptions, and segment boundaries are not given during the inference of video summaries. 

We will release our code, data, and pretrained models. 

## **S2. Ego4D-HCap Data Collection Process** 

The Ego4D-HCap dataset was collected over the span of 2 months, from April 2023 to May 2023 and from September 2023 to October 2023. We recruited 91 specialized annotators through CloudResearch[1] , a participant-sourcing company. All annotators are based in the United States and are compensated at a rate of 9 dollars per hour, which is above the national minimum wage. 

We utilized Qualtrics and Google Drive to build our data collection interface. Our interface began with an introduction to our project, guidelines for summarizing the videos, and examples of good summaries. It then asked the annotators for their ConnectID and provided them a link to the documents of videos assigned to them. Each document would contain 10-25 videos for the annotators to summarize, along with a prompt and a GIF summarizing the events of each video. The last interfaces contain text boxes for the annotators to put the text summaries for each video and the annotator’s experience with the data collection interface. 

> 1https://www.cloudresearch.com 

We used the latter to improve upon the interface so that the quality of the annotated summaries ultimately became better. Figure S2 shows our data collection interface. 

## **S2.1. Guidelines for Annotators** 

**Overview.** In this project, we aim to develop a model that can automatically summarize long videos. Our model generates text captions for each video describing what happens every 3 minutes. We need your help to summarize those captions into a summary for the entire video. The total length of a video can be between 10 and 100 minutes. **Captions.** 

1. You are given a list of captions for each video. 

2. Each caption describes what is happening every 3 minutes. 

3. C refers to a person in the provided captions. 

4. The captions are generated using a machine learning model, so sometimes, they can be out of order or inaccurate. In that case, you can exclude the events or details that do not make sense in the summary or refer to the GIF provided under the captions. 

5. The captions may also use different terms to refer to the same thing. If only technical terms are used, then use them in your summary. Otherwise, we prefer you to use generic terms. 

## **GIFs.** 

1. Since the videos are very long, we do not provide the full video. Instead, you are also given a GIF for each video. 

2. GIFs created by sparsely sampled frames from the video, which is intended to help you better understand the overall contents of the video along with the captions. 

## **Summaries.** 

1. The summary should be one paragraph long. Try to 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos_images/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos.pdf-0016-00.png)


Figure S2. **Data Collection Interface.** 

maintain a compression factor of 5, i.e., for every five captions, you should summarize it in 1 sentence. However, each summary should be at least one sentence. 

2. The summary should cover the setting, characters, and events that take place in the order of the video. 

3. Avoid using X, Y or other letters to refer to characters other than C. Instead, use woman and man. Refer to examples of good summaries on the next page. 

4. The summary should not have an interpretation of the characters’ personalities or qualities. 

5. The summary should be logically coherent, unambiguous, and understandable. 

6. The summary should be grammatically correct. 

7. Repetition of actions should have an underlying purpose/pattern. 

## **S2.2. Quality Control** 

To control the quality of the annotations, we pre-selected annotators before moving them forward with the official annotation task and manually reviewed the annotations. Before the official annotation task, we paid 171 annotators to complete a preliminary annotation task and selected from this pool annotators who provided desirable annotation quality. We minimized the chances of getting lowquality annotations by pre-selecting high-quality annotators and familiarizing them with an interface similar to the actual annotation task. 

Another quality control method we utilized was to review the annotations ourselves manually. For each annotator, we randomly sampled half of the annotations they provided. We assessed their quality based on whether they followed the expectations outlined in Section S2.1. If less than half of the sampled annotations are of low quality, we would provide annotator feedback and ask them to redo their annotations. If the annotations were of better quality, we would 

replace them with the initial annotation. Otherwise, we would discard both versions and assign them to other annotators. 

## **S2.3. De-identification Process** 

Due to the nature of the dataset and our task, our dataset has already been de-identified. Since all of our videos are sourced from Ego4D, they have undergone sensitive object detection, false positive removal, fast negative correction, and image blurring [19]. They were not modified during the dataset collection process, so the videos remain de-identified. Our annotators are also anonymized, as we recruited, managed, and corresponded with annotators on CloudResearch. Aside from their ConnectID, which we used to revise annotations, we did not collect any of the annotators’ personal information. 

## **S3. Ego4D-HCap Dataset Analysis** 

## **S3.1. Example Video Summaries.** 

Figure S3 Shows examples of annotated video summaries of the Ego4D-HCap dataset. We observe that video summaries are of various lengths and capture diverse scenarios, places, and activities. Typically, each video is annotated with multiple summaries. However, the figure shows only one summary per video for clarity and conciseness. 

## **S4. Additional Quantitative Results** 

**Backbone Design.** In this section, we ablate various aspects of our Video-Language Backbone design. First, we validate the effectiveness of a Language Model-based (LM) [43] Video-Language Alignment module rather than a standard Transformer resampler used in prior works [1, 66]. Table S1 shows that an LM-based Alignment module per- 

16 

|LM<br>Alignment|Trainable<br>CA|ClipCaption|Segment Description|Video Summary|
|---|---|---|---|---|
|||C<br>R<br>M|C<br>R<br>M|C<br>R<br>M|
|✗<br>✓<br>✓<br>✗<br>✓<br>✓||92.56<br>47.64<br>28.03<br>39.41<br>38.62<br>17.71<br>23.04<br>28.33<br>13.72<br>73.88<br>43.17<br>21.67<br>32.16<br>31.67<br>13.33<br>12.16<br>21.06<br>8.22<br>**98.35**<br>**48.77**<br>**28.28**<br>**41.74**<br>**39.04**<br>**18.21**<br>**28.06**<br>**32.27**<br>**14.26**|||



Table S1. **Architecture Ablation.** An LM-based [43] Video Language Alignment module provides significant performance gains compared to the transformer-based resampler used in prior works [1, 66]. Adding trainable cross-attention layers inside the text decoder performs much better than freezing the decoder. 

forms significantly better than the standard transformerbased resampler in all three hierarchies. Second, we inject trainable cross-attention layers [1, 66] in the text decoder to incorporate video features. In contrast, several prior works [28, 35] inject video features only in the input layer while freezing the whole text decoder. Table S1 shows that using trainable cross-attention layers in the textual decoder performs significantly better than using video features in the input layer alone across all three hierarchical levels. 

## **S5. Qualitative Results on EgoSchema** 

Figure S4 illustrates the qualitative outcomes of our long-range video question answering experiment on the EgoSchema [34] dataset. The approach, detailed in Section 6.2, involves the generation of hierarchical captions utilizing the Video ReCap model for videos. Subsequently, these captions are presented to ChatGPT along with questions and answer choices as prompts, enabling the model to select the correct answer. In Figure S4 (a) and Figure S4 (b), it is evident that ChatGPT tends to choose incorrect answers when provided solely with clip captions. However, the model consistently makes correct choices in both scenarios when supplemented with video summaries. This highlights the efficacy of our generated hierarchical captions in enhancing the performance of long-range video question answering tasks. Nevertheless, in certain instances, as depicted in Figure S4 (c), our approach encounters challenges and fails to identify the correct answer. 

17 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos_images/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos.pdf-0018-00.png)


Figure S3. **Examples of annotated video summaries of the Ego4D-HCap dataset.** Due to space limitation and conciseness, we show one frame for each 5 minutes of the video.. 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos_images/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos.pdf-0019-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos_images/cvpr_2024_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos.pdf-0019-01.png)


**----- Start of picture text -----**<br>
(a)<br>(b)<br>(c)<br>**----- End of picture text -----**<br>


Figure S4. **Qualitative Results on EgoSchema.** The baseline method that uses only short-range clip captions as input fails in examples (a) and (b), where our approach succeeds by utilizing hierarchical captions (i.e., clip captions and video summaries). Both models fail in Example (c). 

19 

