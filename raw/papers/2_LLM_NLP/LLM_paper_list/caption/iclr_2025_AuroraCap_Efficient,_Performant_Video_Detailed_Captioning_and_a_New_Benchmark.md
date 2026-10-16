Published as a conference paper at ICLR 2025 

## **AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark** 

**Wenhao Chai**[1,2,] _[∗]_ **, Enxin Song** _[∗]_ **, Yilun Du**[4] **, Chenlin Meng**[2,3] **, Vashisht Madhavan**[2] **, Omer Bar-Tal**[2] **, Jenq-Neng Hwang**[1] **, Saining Xie**[5] **, Christopher D. Manning**[3] 

> 1 University of Washington 

2 Pika Labs 

3 Stanford University 

4 Harvard University 

5 New York University 

```
Link:ProjectPage|VDCLeaderboard|Code&Model
```

**Video detailed captioning is a key task which aims to generate comprehensive and coherent textual descriptions of video content, benefiting both video understanding and generation. In this paper, we propose AuroraCap, a video captioner based on a large multimodal model. We follow the simplest architecture design without additional parameters for temporal modeling. To address the overhead caused by lengthy video sequences, we implement the token merging strategy, reducing the number of input visual tokens. Surprisingly, we found that this strategy results in little performance loss. AuroraCap shows superior performance on various video and image captioning benchmarks, for example, obtaining a CIDEr of 88.9 on Flickr30k, beating GPT-4V (55.3) and Gemini-1.5 Pro (82.2). However, existing video caption benchmarks only include simple descriptions, consisting of a few dozen words, which limits research in this field. Therefore, we develop VDC, a video detailed captioning benchmark with over one thousand carefully annotated structured captions. In addition, we propose a new LLM-assisted metric VDCscore for bettering evaluation, which adopts a divide-and-conquer strategy to transform long caption evaluation into multiple short question-answer pairs. With the help of human Elo ranking, our experiments show that this benchmark better correlates with human judgments of video detailed captioning quality.** 

## **1. Introduction** 

The task of video detailed captioning involves generating comprehensive and coherent textual descriptions of video content, capturing not only the primary actions and objects but also intricate details, contextual nuances, and temporal dynamics. It has emerged as a critical area of research in computer vision and natural language processing, with significant implications for the fields of robotics [32, 148], ego-centric perception [40, 41], embodied agents [29, 30, 163–166], and video editing [13] and generation [7]. The challenges of video detailed captioning compared to past problems include the limited detailed caption data for training and evaluation, and also the lack of a good evaluation metric. 

Before the emergence of Large Language Models (LLMs), previous models could only generate very short and rough descriptions of videos [130, 139, 144, 147]. Although these models have been trained on web-scale video-text datasets ( _e.g.,_ HowTo100M [93] and VideoCC3M [97]), their capabilities remain limited due to their smaller scale and the lack of world knowledge possessed by LLMs. Recently, researchers start to build more powerful large multimodal models (LMMs) upon pretrained LLMs ( _e.g.,_ LLaVA [82], InstructBlip [27], InternVL [23]). These models typically use intermediate components ( _e.g.,_ Q-Former [62] or an MLP) to connect the pre-trained vision transformer (ViT) [31] and the LLM. 

*Equal contributions. 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

**Table 1** : **Benchmark comparison** for video captioning task. Ave. Length indicates the average number of words per caption. Compared to the existing benchmarks, **VDC** has the longest captions. 

|Dataset|Theme|# Video|# Clip|# Caption|# Word|# Vocab.|Ave. Length|
|---|---|---|---|---|---|---|---|
|MSVD [14]||1,970|1,970|70,028|607,339|13,010|8.67|
|MSR-VTT [140]<br>ActivityNet [56]|Open|7,180<br>20,000|10,000<br>100,000|200,000<br>100,000|1,856,523<br>1,340,000|29,316<br>15,564|9.28<br>13.40|
|S-MiT [96]||515,912|515,912|515,912|5,618,064|50,570|10.89|
|M-VAD [126]|Movie|92|48,986|55,905|519,933|18,269|9.30|
|MPII-MD [107]||94|68,337|68,375|653,467|24,549|9.56|
|Youcook2 [168]|Cooking|2,000|15,400|15,400|121,418|2,583|7.88|
|Charades [112]|Human|9,848|10,000|27,380|607,339|13,000|22.18|
|VATEX [133]||41,300|41,300|413,000|4994,768|44,103|12.09|
|**VDC**(ours)|Open|1,027|1,027|1,027|515,441|20,419|500.91|



Expanding from image-based LMMs to video-based LMMs is a natural progression, as videos can be viewed as sequences of frames. While most LMMs start with loading the pre-trained weights from image models and are further fine-tuned with video-text data, we find that LLaVA-like models can be easily adapted to a video one without any additional parameters but only with high-quality video-text instruction data for fine-tuning. 

However, naive treatment of videos as a series of image frames can result in significant computational overhead and may cause a generalization of the length problem [131]. To address these concerns, more specifically, to reduce the number of visual tokens, Video-LLaMA [72] adapts the video Q-former, MovieChat [114] uses a memory bank, LLaMA-VID [73] simply uses global pooling, and FastV [16] drops visual tokens by attention rank within LLM layers. In this paper, we present **AuroraCap** , adapting a simple yet efficient method called Token Merging [10], which is proved to be effective in image and video classification and editing tasks [71]. To be specific, we gradually combine similar tokens in a transformer layer using a bipartite soft matching algorithm to reduce the number of visual tokens. Following this pattern, our experiments show that we can use only 10% to 20% visual tokens compared to the original tokens generated by ViT with a marginal performance drop in various benchmarks. With this technique, it is easier to support higher-resolution and longer video sequence inputs for training and inference. 

We present results on several widely used benchmarks, but find that existing video understanding benchmarks are either question-answer-based [11, 14, 36, 114, 134, 137, 140], which cannot demonstrate detailed descriptive abilities, or they provide descriptions that are too short, with only a few words [11, 140] as shown in Table **1** . Some large-scale datasets focus on specific domains such as ego-centric [41] or contain low-quality videos and annotations [5]. Therefore, we construct the **VDC** ( **V** ideo **D** etailed **C** aptions) benchmark, which contains over one thousand high-quality video-caption pairs spanning a wide range of categories, and the resulting captions encompass rich world knowledge, object attributes, camera movements, and crucially, detailed and precise temporal descriptions of events. We utilize `GPT-4o` as our recaption assistant with a hierarchical prompt design. To preserve as much information as possible from the videos and maintain temporal consistency, we implement a denseframe extraction strategy. Using the dense frames as input, despite description of the whole video, 

2 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

we also generate high-quality captions from different aspects, including objective facts, backgrounds, camera angles and movements. Manual quality inspection is employed to ensure the quality of the video captions. While existing video-caption datasets [19, 51, 61] offer structured captions, **VDC** is the first benchmark focused on detailed video captioning, providing significantly longer and more detailed captions than others as shown in the Table **1** and Section S4. 

We also introduce a novel evaluation metric specifically designed for detailed captioning task. Traditional metrics like METEOR [6], CIDEr [127], and BLEU [99], designed for machine translation or short captions, fail to evaluate detailed captions which contain rich semantic information. On the other side, an LLM-based evaluation metric is commonly used in visual question answering benchmarks [87, 134], especially for those generated by VLMs [114, 115]. However, we observe that the LLM-based evaluation metric still struggles to differentiate the quality of detailed captions and tends to give lower scores. To address these challenges, we propose **VDCscore** , a novel captioning evaluation metric that leverages the reliability of large language models (LLMs) by evaluating short visual question-answer pairs. We first decompose the ground-truth caption into a set of concise question-answer pairs using the LLM, then generate corresponding responses from the predicted caption. Finally, the LLM is used to assess the accuracy of each response to provide an overall score. In particular, our paper makes the following contributions: 

- In Section 2, we illustrate how we can reduce the number of tokens used for image or video input before injecting into LLM with marginal performance drop. Using these insights, we propose **AuroraCap** , a video understanding base model with improved captioning performance, which also demonstrates advanced performance on existing benchmarks. 

- In Section 3, we present **VDC** , the first benchmark for detailed video captioning, featuring over one thousand videos with significantly longer and more detailed captions. We comprehensively evaluate proprietary and open-source models using our proposed **VDCscore** metric. We also conduct extensive ablation studies based on **AuroraCap** . 

## **2. AuroraCap: A Video Detailed Captioning Baseline** 

## **2.1. Architecture** 

**Large multimodal model.** To effectively leverage the capabilities of both the pre-trained LLM and visual model, which is typically CLIP [103] or DINO [98], LLaVA adapt a simple multi-layer perceptron (MLP) as projection layer to connect each patch tokens of image features into the word embedding space. The original LLaVA model is trained by a two-stage instruction-tuning procedure, which first pretraining projection layer for feature alignment and then finetuning end-to-end while freeze the visual encoder. Recent works like Prismatic VLMs [52] and Idefics2 [58] further explore the design space of LLaVA architecture. We adapt some conclusion from those works for training the model. 

**Token merging.** To increase the throughput of existing ViT models, Token Merging [10] is proposed to gradually combines similar tokens in a transformer to reduce the number of tokens passing through ViT models. Token Merging has been proven to be effective on image and video classification tasks even without the need for training. Token Merging is applied between the attention and MLP within each transformer block as: 

1. Alternatively partition the tokens into two sets **A** and **B** of roughly equal size. 

3 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

2. For each token in set **A** , calculate the token similarity with each token in set **B** based on cosine similarity of the _Key_ features in attention block. 

3. Use bipartite soft matching and then select the most similar _r_ pairs. 

4. Merge the tokens using weighted average, record the token size. 

5. Concatenate the two sets **A** and **B** back together again. 

Once the tokens have been merged, they actually carry the features for more than one input patch. Therefore, the merged tokens will have less effect in softmax attention as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0004-06.png)


where **s** is the number of patches the token represents after token merging. We conduct frame-wise token merging in **AuroraCap** , the visualization of token merging can be found in Appendix S5. 

## **2.2. Training Recipe** 

Building upon the exploration in works like Prismatic VLMs [52], Idefics2 [58], and InternVL [23], we further adopt a three-stage training strategy, which can be noted as Pretraining stage, Vision stage and Language stage. The training data used in each stage are shown in Table S5, Table S6 and Table S7. More training details including hyper-parameters selection and data preprocessing operation can be found in Appendix S7. 

**Pretraining stage.** Similar to LLaVA, we first align visual features from the vision encoder ViT with the word embedding space of LLMs. To achieve this, we freeze the pretrained ViT and LLM, training solely the vision-language connector. Consistent with LLaVA-1.5 [60], we employ a two-layer MLP as the projection layer and pretrain on 1.3M image-caption pairs. To optimize performance, we explore various combinations of the pre-trained ViT and LLM in Appendix S6. 

**Vision stage.** Unlike LLaVA, we next unfreeze the pretrained ViT while freezing the LLM during vision stage and train with the public data among various computer vision tasks ( _e.g.,_ captioning, object identification, classification, reasoning, VQA, _etc._ ) to get better generalization [46]. The motivation for doing this is that CLIP ViT usually performs poorly in aspects such as Orientation and Direction, Positional and Relational Context, Quantity and Count [125]. However, since the most of the collected datasets lack high-quality and detailed corresponding language descriptions, the labels often consist of only a few words or a short phrase when converted to text. Therefore, unfreezing the language model at this stage is risky, as it may lead to a degradation in the performance of the language model. 

**Language stage.** Finally, we conduct end-to-end training, which means all the components are trainable, with the most high-quality public data during language stage training. We mix all the data, including images and videos, captions and instructions, into each mini-batch for training. To improve video captioning performance, we duplicate the video captioning datasets twice. We remove all the video training data for training a image-based **AuroraCap** as well for image captioning task. 

## **2.3. Evaluation** 

In this section we evaluate **AuroraCap** on various tasks including image captioning, video captioning, and video question answering. Appendix S8 show detailed evaluation settings. 

4 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

**Table 2** : Comparison of **AuroraCap** with LLM-based SoTA methods on image-captioning benchmarks under a zero-shot setting, except _k_ -shot where _k_ is marked as a superscript. The GPT-4V CIDEr results are from [119], the only reference we know of, but seem oddly low. _∗_ marks fine-tuned results. Scores in **bold** indicate the best performance under zero-shot setting. 

|Model|Flickr (31,784)<br>CI-<br>BLEU<br>BLEU<br>Met-<br>ROU-<br>DEr<br>@1<br>@4<br>eor<br>GE|NoCaps (4,500)<br>CI-<br>BLEU<br>BLEU<br>Met-<br>ROU-<br>DEr<br>@1<br>@4<br>eor<br>GE|COCO-Cap (5,000)<br>CI-<br>BLEU<br>BLEU<br>Met-<br>ROU-<br>DEr<br>@1<br>@4<br>eor<br>GE|
|---|---|---|---|
|InstructBLIP-7B [27]|78.2<br>77.1<br>30.9<br>24.8<br>53.4|118.2<br>88.2<br>47.2<br>30.3<br>62.0|141.3*<br>82.8*<br>41.7*<br>30.9*<br>61.0*|
|InstructBLIP-13B [27]|76.1<br>76.8<br>30.1<br>24.4<br>53.0|116.3<br>88.3<br>46.7<br>29.8<br>61.4|135.0*<br>82.2*<br>39.7*<br>29.8*<br>59.8*|
|LLaVA-1.5-7B [80]|74.9<br>71.7<br>28.4<br>26.1<br>52.8|105.5<br>82.6<br>40.2<br>30.3<br>59.4|110.3<br>73.0<br>29.7<br>29.2<br>55.5|
|LLaVA-1.5-13B [80]|79.4<br>73.6<br>30.2<br>26.6<br>53.9|109.2<br>84.2<br>42.4<br>**30.6**<br>60.3|115.6<br>74.6<br>31.5<br>**29.4**<br>56.5|
|LLaVA-1.6-7B [81]|68.4<br>69.6<br>26.6<br>23.2<br>50.3|88.4<br>73.8<br>34.8<br>25.9<br>54.6|99.9<br>67.7<br>28.4<br>25.5<br>52.4|
|LLaVA-1.6-13B [81]|66.6<br>65.2<br>24.2<br>22.2<br>48.8|88.1<br>68.7<br>34.0<br>25.4<br>54.9|101.8<br>62.2<br>27.5<br>24.6<br>52.1|
|MiniCPM-V-3B [43]|66.8<br>68.0<br>25.1<br>27.2<br>51.0|89.9<br>79.1<br>33.2<br>29.7<br>55.8|94.2<br>69.8<br>23.9<br>28.3<br>52.3|
|DeCap [67]|56.7<br>_−_<br>21.2<br>21.8<br>_−_|42.7<br>_−_<br>_−_<br>_−_<br>_−_|91.2<br>_−_<br>24.7<br>25.0<br>_−_|
|Flamingo-80B [2]|67.2<br>_−_<br>_−_<br>_−_<br>_−_|_−_<br>_−_<br>_−_<br>_−_<br>_−_|84.3<br>_−_<br>_−_<br>_−_<br>_−_|
|<br>Chameleon-34B [119]|74.72<br>_−_<br>_−_<br>_−_<br>_−_<br>|_−_<br>_−_<br>_−_<br>_−_<br>_−_|120.22<br>_−_<br>_−_<br>_−_<br>_−_<br>|
|GPT-4V|55.38<br>_−_<br>_−_<br>_−_<br>_−_|_−_<br>_−_<br>_−_<br>_−_<br>_−_|78.58<br>_−_<br>_−_<br>_−_<br>_−_|
|Gemini-1.5 Pro|82.24<br>_−_<br>_−_<br>_−_<br>_−_|_−_<br>_−_<br>_−_<br>_−_<br>_−_|99.82<br>_−_<br>_−_<br>_−_<br>_−_|
|**AuroraCap**-7B|**88.9**<br>**75.6**<br>**32.8**<br>**26.7**<br>**55.4**|**111.4**<br>**85.6**<br>**44.4**<br>29.9<br>**60.6**|**120.8**<br>**78.0**<br>**35.3**<br>28.6<br>**57.2**|



**Table 3** : Comparison of **AuroraCap** with SoTA methods on existing video captioning benchmarks under zero-shot setting. **AuroraCap** substantially outperforms other recent methods. 

|Model|MSR-VTT (1,000)<br>CI-<br>BLEU<br>BLEU<br>Met-<br>ROU-<br>DEr<br>@1<br>@4<br>eor<br>GE|VATEX (1,000)<br>CI-<br>BLEU<br>BLEU<br>Met-<br>ROU-<br>DEr<br>@1<br>@4<br>eor<br>GE|
|---|---|---|
|ZeroCap [121]|9.6<br>_−_<br>2.9<br>16.3<br>35.4|_−_<br>_−_<br>_−_<br>_−_<br>_−_|
|DeCap [67]|18.6<br>_−_<br>14.7<br>20.4<br>_−_|18.7<br>_−_<br>13.1<br>15.3<br>_−_|
|PaLI-3 [21]|21.3<br>_−_<br>_−_<br>_−_<br>_−_|_−_<br>_−_<br>_−_<br>_−_<br>_−_|
|Ma_et al._[86]|22.1<br>_−_<br>3.5<br>17.3<br>28.7|23.9<br>_−_<br>2.8<br>14.1<br>23.5|
|LLaVA-7B [82]|16.9<br>_−_<br>_−_<br>_−_<br>_−_|_−_<br>_−_<br>_−_<br>_−_<br>_−_|
|Video-LLaMA [155]|2.3<br>_−_<br>4.9<br>16.8<br>_−_|3.8<br>_−_<br>4.3<br>16.3<br>21.8|
|**AuroraCap**-7B<br>**33.1**<br>**58.6**<br>**21.0**<br>**23.9**<br>**49.5**<br>**33.8**<br>**57.1**<br>**18.4**<br>**19.0**<br>**40.8**|||



**Image Captioning.** We evaluate **AuroraCap** using CIDEr (C), BLEU-4 (B@4), BLEU-1 (B@1), METEOR (M), and ROUGE-L (R) metric on Flickr [102], NoCaps [1], and COCO-Cap [78] benchmarks and **2** . compare it with LLM-based state-of-the-art methods as shown in Table We show the performance of image based **AuroraCap** under zero-shot settings. Notice that these benchmarks all contain short captions consisting of a single sentence, so they only partially reflect the model’s performance. The performance mentioned in the rest of this paper refers to video-based **AuroraCap** . 

**Video Captioning.** Although the current video captioning benchmarks only contain one-sentence captions, to compare with prior work, we similarly evaluate on these benchmarks. We evaluate **AuroraCap** using CIDer (C), BLEU-4 (B@4), BLEU-1 (B@1), METEOR (M), and ROUGE-L (R) metric on MSR-VTT [140], VATEX [133] as shown in Table **3** . 

**Video Question Answering.** We evaluate **AuroraCap** on MSVD-QA [138], ActivityNet-QA [152], MSRVTT-QA [138], and iVQA [145] for video question answering tasks as shown in Table **4** . Although **AuroraCap** is primarily a captioning model, it achieves competitive performance in some VQA datasets (ANet, iVQA). For others (MSVD, MSR-VTT) performance is more modest, but still not 

5 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

**Table 4** : Comparison of **AuroraCap** with SoTA methods on video question answering and classification benchmarks under zero-shot setting. The pretrained LLM size is 7B for all listed models. 

|Model|ANet<br>Acc<br>Score|MSVD<br>Acc<br>Score|MSR-VTT<br>Acc<br>Score|iVQA<br>Acc|
|---|---|---|---|---|
|Just Ask [145]|_−_<br>_−_|_−_<br>_−_|_−_<br>_−_|12.2|
|FrozenBiLM [146]|24.7<br>_−_|32.2<br>_−_|16.8<br>_−_|26.8|
|Video-LLaMA [155]|12.4<br>1.1|51.6<br>2.5|29.6<br>1.8|_−_|
|VideoChat [64]|26.5<br>2.2|56.3<br>2.8|45.0<br>2.5|_−_|
|Video-ChatGPT [87]|35.2<br>2.7|64.9<br>3.3|49.3<br>2.8|_−_|
|LLaMA-VID [73]|47.4<br>3.3|69.7<br>3.7|57.7<br>3.2|_−_|
|Video-LLaVA [75]|45.3<br>3.3|70.7<br>3.9|59.2<br>3.5|_−_|
|FreeVA [136]|51.2<br>3.5|73.8<br>**4.1**|**60.0**<br>**3.5**|_−_|
|LLaVA-NeXT-Video [81]|53.5<br>3.2|_−_<br>_−_|_−_<br>_−_|_−_|
|MovieChat [114]|45.7<br>3.4|75.2<br>3.8|52.7<br>2.6|_−_|
|MovieChat+ [115]|48.1<br>3.4|**76.5**<br>3.9|53.9<br>2.7|_−_|
|**AuroraCap**-7B|**61.8**<br>**3.8**|62.6<br>3.6|43.5<br>2.9|**55.2**|



bad. In some failure cases observed in the model, we found that prompting the model to generate a comprehensive caption for the video input can lead to outputs that include the correct answer. This phenomenon may be attributed to a disruption in the model’s instruction-following capabilities during the training. 

## **3. VDC: A Video Detailed Captioning Benchmark** 

## **3.1. Benchmark Dataset Curation** 

## **3.1.1. Video Collection and Processing** 

To ensure the reliability of the benchmark, it is crucial to maintain high video quality, balanced data distribution, and content complexity. Panda-70M [20] offers a high-resolution, open-domain YouTube video dataset with diverse one-minute clips across wildlife, cooking, sports, news, TV shows, gaming, and 3D rendering, ideal for studying complex real-world scenarios. Additionally, a large volume of aesthetically appealing videos from user-uploaded platforms like Mixkit [95], Pixabay [101], and Pexels [100] provides scenic views and visually pleasing human activities with minimal transitions and simpler events. Ego4D [40] complements the video source by focusing on ego-centric human activities and auto-driving scenarios, ensuring comprehensive coverage of real-world scenes. To mitigate content homogeneity among these candidate videos and maintain diversity in the final dataset, inspired by ShareGPT4Video [19], we building **VDC** upon those various video sources. Note that the videos used in **VDC** construction are not included in the training data of **AuroraCap** . To ensure balanced data distribution, we allocate equal proportions of videos from Panda-70M [20], Ego4D [40], Mixkit [95], Pixabay [101], and Pexels [100] as shown in Table **5** . We first split the video into clips and apply dense frame extraction. 

## **3.1.2. Structured Detailed Captions Construction Pipeline** 

We believe that a comprehensive detailed video caption benchmark should encompass various aspects, including main objects, camera movements, and background. However, most existing 

6 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

**Table 5** : The video source distribution of the proposed **VDC** benchamrk including diverse settings such as natural landscapes, human activities, and animal activities. Videos from different sources have a similar proportion in **VDC** , reducing the data bias. 

|Video Source|# Sample|Proportion|Duration (sec.)|Ave. Length (sec.)|Ave. # Keyframe|
|---|---|---|---|---|---|
|Panda-70M [20]|229|22.25%|5,714|24.95|7.18|
|Ego4D [40]|196|19.05%|10,935|55.79|19.46|
|Mixkit [95]|197|19.14%|3,261|16.55|6.58|
|Pixabay [101]|199|19.34%|4,748|23.86|8.99|
|Pexels [100]|208|20.21%|4,343|20.88|7.99|
|Total|1,027|_−_|29,001|28.18|10.43|




![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0007-03.png)


**----- Start of picture text -----**<br>
90 Short Caption: Ave. Length = 26.5<br>40s-50s7% 30s-40s4% 40 Background Caption: Ave. Length = 80.4<br>20s-30s Main Object Caption: Ave. Length = 90.4<br>50s-60s 16% 30 Camera Caption: Ave. Length = 86.6<br>21% Detailed Caption: Ave. Length = 500.9<br>20<br>> 60s 10<br>1%<br>< 10s<br>0<br>13% 38% 50 100 150 350 400 450 500 550 600 650 700<br>10s-20s Caption Length (words)<br>Percentage<br>**----- End of picture text -----**<br>


**Fig. 1** : Video length statistics. 

**Fig. 2** : Distribution of structured caption length. 

benchmarks [19, 61] provide only a single caption for the entire video with less structured details. Therefore, we develop a structured detailed captions construction pipeline to generate extra detailed descriptions from various perspectives, significantly extending the length and enhancing the richness compared to previous ben ~~chma~~ rks. F ~~oll~~ o ~~wing [5~~ 1], th ~~e s~~ tr ~~uctured~~ caption ~~s in~~ ~~**VDC**~~ enco ~~mpass no~~ t only **short** and **detailed captions** but also three additional categories: (1) **main object caption** , offering a comprehensive analysis of the primary subjects’ actions, attributes, interactions, and movements across frames, including variations in posture, expression, and speed; (2) **background caption** , providing detailed descriptions of the background, such as objects, location, weather, time, and dynamic elements; and (3) **camera caption** , which details the camera work, including shot types, angles, movements, transitions, and special effects. 

To generate detailed, fine-grained, and accurate captions, we leverage `GPT-4o` to produce video descriptions. We utilize the dense video frames to obtain captions. We observed that generating all captions in a single conversation round often introduces hallucinations in the detailed captions. To address this, we design a hierarchical prompt strategy to efficiently obtain accurate structured captions and detailed captions in two conversation rounds: (1) structured captions generation and (2) detailed captions integration. In the first round, the prompt briefly introduces the differences between structured captions and uses the dense video frames as input to generate the short caption, main object caption, background caption, camera caption, and the detailed caption. In the second round, the generated captions serve as the reference. The second-round prompt guides `GPT-4o` to enhance the detailed caption based on the initial captions, ensuring consistency without introducing new entities or relations, and producing a vivid, engaging, and informative description. The whole prompt template for the structured detailed captions construction pipeline can be found in Appendix S11. Finally, 

7 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0008-01.png)


**----- Start of picture text -----**<br>
GT caption generated caption<br>① raise Who is the main character of this  ③ ask<br>The video showcases an  video? The video captures a thrilling<br>exhilarating moment as a  moment of a snowboarder in<br>② get<br>snowboarder soars through the  Snowboarder mid-air, performing an<br>air, executing a stunning trick.  ⑤ check impressive trick. The<br>Dressed in a bold red and white  snowboarder, clad in a vibrant<br>④ get<br>jacket, black pants, and a  Snowboarder red and black jacket, black<br>protective helmet. The backdrop  pants, and a protective helmet.<br>to this action-packed scene is a  The snowboarder is holding onto<br>① raise Who is the main character of this  ③ ask<br>breathtaking snowy mountain  a rope with one hand, suggesting<br>landscape. The mountain's peak  video? that they are being pulled up the<br>is visible in the distance. The  ② get mountain by a snowmobile, a<br>overall composition of the video  red and white ⑤ check common practice in<br>suggests a high-speed descent  snowboarding to gain speed and<br>down the mountain … ④ get momentum …<br>red and black<br>**----- End of picture text -----**<br>


**Fig. 3** : Evaluation pipeline with **VDCscore** . Like when humans take reading comprehension tests, we transform the matching between two paragraphs into a set of question-answer pairings. We first generate some questionanswer pairs based on the ground truth captions, then derive corresponding answers one by one from the generated captions, and finally perform matching. The process is automatically evaluated with the LLM involvement in each step. 

we conduct a manual review to correct captions with hallucinations and supplement omitted visual elements. The refined detail structured captions are then used as the ground truth for evaluation. 

## **3.1.3. Comparison on Numerical Statistics** 

Based on the hierarchical scheme, **VDC** can capture a rich variety of details of the video and reduce hallucinations. The visual representation in Figure **1** demonstrates the video duration distribution of **VDC** . Over 87% of the videos exhibit a duration ranging from 10K to 12K frames, while 1% of videos extending beyond 60 seconds. Only 13% of videos have duration less than 10 seconds. As illustrated in Table **1** , the average length of detailed descriptions in **VDC** is significantly longer than in previous benchmarks. Figure **2** shows the length distribution of structured captions in **VDC** , with detailed captions averaging over 500 words. Appendix S13 present more statistics. 

## **3.2. Evaluation Metric Design and Leaderboard** 

## **3.2.1. VDCscore: Evaluating Detailed Captions with LLMs** 

Evaluating video captions requires not only assessing the quality of the captions but also flexibly evaluating the alignment between the video and the caption. While metrics such as BLEU [99], CIDEr [127], and ROUGE-L [76] have been employed for caption evaluation, these metrics are predominantly designed for short captions and rely heavily on word-level frequency-based alignment. Given the advanced semantic understanding capabilities of large language models (LLMs), VideoChatGPT [64] proposes using LLM as an evaluation assistant to directly judge the correctness of the whole predicted captions and assign scores. However, as demonstrated in Table **6** , our experiments indicate that when dealing with detailed captions, the direct application of LLM struggles to accurately distinguish the correctness of various predicted captions, fails to effectively evaluate the precision of 

8 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

**Table 6** : Comparison of **AuroraCap** with LLM-based baseline methods on **VDCscore** and other evaluation metrics under zero-shot setting. For each evaluation metric, we report the average value of the five structured captions in **VDC** . Note that VDD, CIDEr, and BLEU are only the average of background and main object caption, since the values of the others are closed to zero. 

|Model|**VDCscore**<br>Acc<br>Score|VDD<br>Acc<br>Score|CI-<br>BLEU<br>BLEU<br>Met-<br>ROU-<br>DEr<br>@1<br>@4<br>eor<br>GE|Elo|
|---|---|---|---|---|
||||||
|Gemini-1.5 Pro [105]|41.73<br>2.15|49.68<br>3.07|5.97<br>29.72<br>2.63<br>21.21<br>20.19|1,171|
||||||
|LLaMA-VID [72]<br>Video-ChatGPT-7B [87]<br>MovieChat-7B [114]<br>VILA-7B [77]<br>Video-LLaVA-7B [75]<br>LLaVA-1.5-7B [80]<br>LongVA-7B [157]<br>LLaVA-1.5-13B [80]<br>LLaVA-NeXT-V7B [161]<br>LLaVA-1.6-7B [81]<br>LLaVA-1.6-13B [81]<br>ShareGPT4Video-8B [19]<br>LLaVA-OV-7B [60]<br>InternVL-2-8B [23]|30.86<br>1.62<br>31.12<br>1.62<br>31.92<br>1.64<br>32.61<br>1.70<br>32.80<br>1.72<br>33.98<br>1.76<br>34.50<br>1.79<br>34.78<br>1.80<br>35.46<br>1.85<br>35.70<br>1.85<br>35.85<br>1.85<br>36.17<br>1.85<br>37.45<br>1.94<br>37.72<br>1.96|4.63<br>1.63<br>8.57<br>1.84<br>10.24<br>1.86<br>16.27<br>2.02<br>14.14<br>2.00<br>26.71<br>2.33<br>32.65<br>2.69<br>28.26<br>2.36<br>25.62<br>2.34<br>40.16<br>2.69<br>34.55<br>2.51<br>36.44<br>1.85<br>41.83<br>2.70<br>**48.99**<br>**3.03**|1.48<br>17.74<br>1.46<br>8.07<br>17.47<br>2.92<br>17.31<br>2.19<br>11.57<br>16.96<br>5.14<br>14.33<br>3.17<br>13.60<br>14.98<br>8.20<br>19.13<br>2.11<br>5.62<br>16.63<br>4.43<br>17.20<br>2.32<br>10.36<br>17.53<br>6.63<br>29.80<br>2.54<br>22.79<br>20.36<br>4.83<br>18.75<br>2.16<br>13.43<br>14.84<br>3.90<br>20.43<br>2.02<br>26.37<br>17.87<br>2.66<br>20.18<br>2.33<br>**28.17**<br>17.51<br>3.09<br>17.36<br>1.59<br>24.23<br>17.08<br>5.55<br>29.23<br>2.50<br>20.26<br>19.96<br>1.02<br>12.61<br>0.79<br>8.33<br>16.31<br>4.09<br>28.34<br>2.84<br>23.98<br>19.59<br>5.59<br>15.75<br>2.48<br>10.76<br>17.63|859<br>944<br>890<br>1,073<br>1,007<br>825<br>969<br>943<br>1,022<br>846<br>728<br>1,102<br>1,155<br>1,081|
|**AuroraCap**-7B<br>**38.21**<br>**1.98**<br>48.33<br>2.90<br>**9.51**<br>**30.90**<br>**4.06**<br>19.09<br>**21.58**<br>**1,267**|||||



detailed descriptions, and exhibits a tendency to assign disproportionately lower scores. Therefore, we introduce **VDCscore** , a novel quantitative metric that utilizes LLMs to evaluate the similarity between predicted and ground-truth detailed captions through a divide-and-conquer approach. 

The core idea of **VDCscore** is to decompose long detailed captions into multiple short questionanswering pairs, avergae the evaluation of each pair as the final result. We elaborate the design of **VDCscore** in the following parts: (1) ground-truth question-answer pairs extraction, (2) responsed answers generation and (3) answers matching. As illustrated in Figure **3** , we first employ `GPT-4o`[*] to generate question-answer pairs from the detailed ground-truth caption. To ensure that the generated question-answer pairs capture as much information as possible from the original caption and facilitate accurate evaluation by LLMs, we constrain the number of generated pairs and impose specific guidelines: the questions must be open-ended, and the answers should be concise, and directly relevant to the questions. For a fair comparison and to mitigate potential variability arising from generating different question-answer pairs for the same caption, we pre-generate a standardized set of question-answer pairs for all captions in **VDC** , as depicted in Figure **3** . The used prompt templates used along with additional examples, are provided in Appendix S12. 

**VDCscore** subsequently analyzes the predicted captions by leveraging ground-truth questionanswer pairs. We prompt `GPT-4o` to read the detailed predicted captions and generate answers based solely on these captions. To mitigate biases arising from discrepancies in the length between groundtruth and predicted answers, we also impose constraints ensuring that responses are limited to concise sentences or phrases. Consequently, for each pair of ground-truth and predicted captions, we obtain 

> * `GPT-4o` latest version in September, 2024. 

9 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

**Table 7** : Comparison of **AuroraCap** with LLM-based baseline methods on **VDCscore** under zero-shot structured captions setting. We consider the **VDCscore** of detailed captions, short captions, background captions, main object captions and camera captions. We also test the vision-blind case suggesting by [17, 124]. 

|Model|Camera<br>Short<br>Background<br>Main Object<br>Detailed<br>Acc / Score<br>Acc / Score<br>Acc / Score<br>Acc / Score<br>Acc / Score|
|---|---|
|||
|Vicuna-v1.5-7B [25]<br>Llama-3.1-8B [34]|21.68 / 1.12<br>23.06 / 1.17<br>22.02 / 1.15<br>22.64 / 1.16<br>23.09 / 1.20<br>17.83 / 1.00<br>17.90 / 1.02<br>19.52 / 1.10<br>19.57 / 1.10<br>20.10 / 1.22|
|||
|Gemini-1.5 Pro [105]|38.68 / 2.05<br>35.71 / 1.85<br>43.84 / 2.23<br>47.32 / 2.41<br>43.11 / 2.22|
|||
|LLaMA-VID [72]<br>Video-ChatGPT-7B [87]<br>MovieChat-7B [114]<br>VILA-7B [77]<br>Video-LLaVA-7B [75]<br>LLaVA-1.5-7B [80]<br>LongVA-7B [157]<br>LLaVA-1.5-13B [80]<br>LLaVA-NeXT-V7B [161]<br>LLaVA-1.6-7B [81]<br>LLaVA-1.6-13B [81]<br>ShareGPT4Video-8B [19]<br>LLaVA-OV-7B [60]<br>InternVL-2-8B [23]|39.47 / 2.10<br>29.92 / 1.56<br>28.01 / 1.45<br>31.24 / 1.59<br>25.67 / 1.38<br>37.46 / 2.00<br>29.36 / 1.56<br>33.68 / 1.70<br>30.47 / 1.60<br>24.61 / 1.26<br>37.25 / 1.98<br>32.55 / 1.59<br>28.99 / 1.54<br>31.97 / 1.64<br>28.82 / 1.46<br>34.33 / 1.83<br>30.40 / 1.55<br>35.15 / 1.80<br>33.38 / 1.72<br>29.78 / 1.58<br>37.48 / 1.97<br>30.67 / 1.63<br>32.50 / 1.70<br>36.01 / 1.85<br>27.36 / 1.43<br>38.38 / 2.04<br>28.61 / 1.51<br>34.86 / 1.79<br>34.62 / 1.76<br>33.43 / 1.73<br>35.32 / 1.90<br>31.94 / 1.63<br>36.39 / 1.85<br>40.95 / 2.11<br>27.91 / 1.48<br>38.97 / 2.07<br>30.89 / 1.60<br>34.79 / 1.78<br>36.27 / 1.84<br>33.00 / 1.74<br>39.73 / 2.10<br>30.63 / 1.60<br>36.54 / 1.88<br>36.54 / 1.88<br>33.84 / 1.77<br>36.50 / 1.93<br>31.91 / 1.65<br>37.58 / 1.92<br>36.03 / 1.85<br>36.47 / 1.89<br>35.61 / 1.86<br>31.90 / 1.66<br>**38.90**/**1.99**<br>36.65 / 1.87<br>36.18 / 1.89<br>33.28 / 1.76<br>**39.08**/**1.94**<br>35.77 / 1.81<br>37.12 / 1.89<br>35.62 / 1.84<br>37.82 / 2.02<br>32.58 / 1.70<br>37.43 / 1.92<br>38.21 / 1.96<br>41.20 / 2.13<br>39.08 / 2.11<br>33.02 / 1.74<br>37.47 / 1.89<br>**44.16**/**2.22**<br>34.89 / 1.82|
|||
|**AuroraCap**-7B|**43.50**/**2.27**<br>32.07 / 1.68<br>35.92 / 1.84<br>39.02 / 1.97<br>**41.30**/**2.15**|



a set of `<question, correct answer, predicted answer>` triplets. Following Video-ChatGPT [64], we then ask `GPT-4o` to output two scores for each triplet: one for answer correctness and another for answer quality. The final accuracy and score are calculated by averaging the correctness score and quality score respectively. When using two same captions as input, **VDCscore** returns an accuracy of 100%, demonstrating the feasibility and reliability. 

## **3.2.2. Benchmarking Video Detailed Captioning** 

To our knowledge, no standard evaluation benchmarks have been established for detailed video captioning. To advance this field, we assess several baselines on our proposed **VDC** . As illustrated in Table **6** , we present a quantitative comparison between our **AuroraCap** with existing state-ofthe-art LMMs. We compare the **VDCscore** with both rule-based and model-based caption metrics with **AuroraCap** performing well. BLEU [99], CIDEr [127], ROUGE-L [76] and METEOR [6] are included as representative rule-based metrics. For model-based metrics, we also consider Video Detailed Description [64] (VDD), which employs `ChatGPT` as an evaluation assistant to compare. 

Table **7** presents **VDCscore** performance across various sections of structured captions within **VDC** . Following [17], we also incorporate vision-blind baselines. Furthermore, Figure **4** illustrates a schematic diagram of the performance and efficiency of different video training models. Since the comparison of model inference time under different architectures, models, deployment frameworks, and output lengths is unfair, so we used the number of visual tokens as a representation of efficiency. 

10 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0011-01.png)


**----- Start of picture text -----**<br>
41.7 Gemini-1.5-Pro<br>38<br>36<br>34 AuroraCap-7B<br>InternVL-2-8B<br>32 LLaVA-OV-7B<br>ShareGPT4Video-8B<br>LLaVA-Next-V7B<br>30<br>LongVA-7B<br>Video-LLaVA-7B<br>28<br>VILA-7B<br>Video-ChatGPT-7B<br>26 MovieChat-7B<br>LLaMA-VID-7B<br>24<br>0 1000 2000 3000 4000 5000 6000 7000 8000<br># visual tokens<br>VDCScore<br>**----- End of picture text -----**<br>


**Fig. 4** : Comparison between various models with different number of visual tokens input on **VDC** . For Gemini-1.5-Pro, we only report the performance. We manage the number of visual tokens by managing token merging for **AuroraCap** , and manage the number of frames for others. **AuroraCap** achieves a much better **VDCscore** than all other models given a certain compression in the number of visual tokens and approaches the SoTA performance. 

**Table 8** : Comparison of **AuroraCap** with LLM-based baseline methods on VDD [81] under zero-shot structured captions setting. We consider the VDD [81] of detailed captions, short captions, background captions, main object captions and camera captions. 

|Model|Camera<br>Short<br>Background<br>Main Object<br>Detailed<br>Acc / Score<br>Acc / Score<br>Acc / Score<br>Acc / Score<br>Acc / Score|
|---|---|
|||
|Gemini-1.5 Pro [105]|18.89 / 2.115<br>16.91 / 1.572<br>57.73 / 3.263<br>41.64 / 2.886<br>2.581 / 0.330|
|||
|LLaMA-VID [72]<br>Video-ChatGPT-7B [87]<br>VILA-7B [77]<br>Video-LLaVA-7B [75]<br>LLaVA-1.5-7B [80]<br>LongVA-7B [157]<br>LLaVA-1.5-13B [80]<br>LLaVA-NeXT-V7B [161]<br>LLaVA-1.6-7B [81]<br>LLaVA-1.6-13B [81]<br>ShareGPT4Video-8B [19]<br>LLaVA-OV-7B [60]<br>InternVL-2-8B [23]|28.28 / 2.513<br>1.034 / 1.042<br>6.198 / 1.895<br>3.063 / 1.366<br>2.046 / 0.304<br>16.00 / 2.175<br>4.173 / 1.032<br>14.14 / 2.273<br>3.001 / 1.423<br>1.038 / 0.192<br>4.005 / 1.751<br>2.087 / 0.233<br>22.45 / 2.385<br>10.10 / 1.672<br>1.015 / 0.262<br>20.00 / 2.336<br>3.193 / 1.064<br>17.17 / 2.253<br>11.11 / 1.765<br>3.130 / 0.316<br>26.88 / 2.515<br>1.222 / 0.793<br>36.50 / 2.725<br>16.92 / 1.937<br>0.694 / 0.276<br>17.00 / 2.204<br>1.016 / 0.794<br>50.00 / 3.203<br>15.31 / 2.196<br>0.002 / 0.247<br>32.65 / 2.662<br>2.836 / 0.922<br>37.55 / 2.749<br>18.98 / 1.978<br>0.688 / 0.275<br>29.81 / 2.645<br>1.913 / 0.957<br>33.23 / 2.692<br>18.02 / 1.999<br>0.887 / 0.279<br>21.11 / 2.229<br>8.696 / 1.146<br>53.31 / 3.105<br>27.01 / 2.286<br>1.282 / 0.279<br>21.56 / 2.199<br>9.798 / 1.206<br>39.37 / 2.692<br>29.73 / 2.329<br>1.287 / 0.271<br>33.28 / 1.768<br>4.908 / 0.986<br>35.77 / 1.813<br>37.12 / 1.899<br>**3.213**/**0.752**<br>17.11 / 2.086<br>**11.15**/**1.277**<br>55.82 / 3.149<br>27.84 / 2.258<br>1.372 / 0.249<br>29.00 / 2.545<br>4.041 / 1.079<br>**63.64**/**3.446**<br>34.34 /**2.627**<br>3.032 / 0.394|
|**AuroraCap**-7B<br>**49.40**/**3.141**<br>3.313 / 0.886<br>59.52 / 3.261<br>**37.14**/ 2.533<br>1.275 / 0.295||



11 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0012-01.png)


**----- Start of picture text -----**<br>
Pearson Correlation Coefficient r = 0.86 Pearson Correlation Coefficient r = 0.66 Pearson Correlation Coefficient r = 0.12<br>45 22<br>3<br>40<br>20<br>2<br>35<br>18<br>1<br>30<br>16<br>0<br>900 1000 1100 1200 900 1000 1100 1200 900 1000 1100 1200<br>Elo Score Elo Score Elo Score<br>: Pearson correlation analysis among three evaluation metrics— VDCscore , ROUGE-L [76], VDD [8176], VDD [81], VDD [8181]—and<br>human Elo rankings for video models. VDD reflects the score for detailed captions.  VDCscore  demonstrates<br>the highest consistency with expert judgments, thereby reinforcing the reliability. The detailed settings are<br>provided in the Appendix S15. S15..<br>Accuracy on Camera Captions Accuracy on Background Captions<br>44 43.35 43.29 Accuracy on Short Captions<br>42.83<br>42 42.14 41.25 32.532.0 32.07 31.88 36 35.42 35.55<br>40 31.5 31.59 31.5 35 34.98<br>34.61<br>38 31.0<br>36 30.5 30.27 34<br>34 30.0 90% 80% 29.94 90% 80% 33.3233.2 8<br>90% 80% 32.6 29.5 33<br>32<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>Accuracy on Main Object Captions Accuracy on VDC<br>Accuracy on Detailed Captions<br>39 38.51 41.5 41.03 38 37.28 37.34 37.3 38.01<br>38 37.58 41.0 40.61 40.807 37 36.73<br>37 36.62 36.34 37.05 40.5 40.13 40.51 36<br>36 40.0<br>35<br>98 0% 39.57<br>35 98 0% 34.43 39.5 34 90% 80% 33.96<br>34 39.0<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>VDD<br>VDCscore ROUGE-L<br>Accuracy Accuracy Accuracy<br>Accuracy Accuracy Accuracy<br>**----- End of picture text -----**<br>


**Fig. 5** : Pearson correlation analysis among three evaluation metrics— **VDCscore** , ROUGE-L [76], VDD [8176], VDD [81], VDD [8181]—and human Elo rankings for video models. VDD reflects the score for detailed captions. **VDCscore** demonstrates the highest consistency with expert judgments, thereby reinforcing the reliability. The detailed settings are provided in the Appendix S15. S15.. 

**Fig. 6** : Ablation study of token merging on **VDC** . We found that token merging significantly reduces the number of tokens while maintaining minimal performance drop, and even showing improvement in some aspects. We highlight the token merging ratio when achieving 90% and 80% performance with the dash line and filled area. 

**AuroraCap** achieves superior performance in video detailed captioning while utilizing significantly fewer visual tokens than other models, fully highlighting the efficiency of **AuroraCap** . We also show additional experimental results with VDD metric as shown in Table **8** . We also perform a human study using Elo ranking to supplement our evaluation and provide a more intuitive assessment of **AuroraCap** ’s performance. As depicted in Figure **5** , **VDCscore** shows better correlation with human evaluation results than VDD and ROUGE metric. 

**Ablation study on token merging ratio.** As a core strategy of **AuroraCap** , token merging plays a significant role in reducing the number of visual tokens. We further study how the video detailed captioning capability is influenced by token merge ratio. We define the performance percentage as the proportion between the highest and lowest values on the entire performance curve. As shown in 

12 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

Figure **6** , most models maintain satisfactory performance ( _>_ 80%) even with only 0.2 of visual token kept ratio. Since **AuroraCap** focuses on spatial visual token merging, the temporal features introduce additional complexity to explore the token merging laws, resulting in the optimal performance may occurs at a middle level of visual token kept ratio. 

## **4. Conclusion** 

In this paper, we first introduce **AuroraCap** , a efficient video detailed captioner based on large multimodal model. By leveraging the token merging strategy, we significantly reduce the computational overhead without compromising performance. We also present **VDC** , a novel video detailed captioning benchmark designed to evaluate comprehensive and coherent textual descriptions of video content. For better evaluating, We propose **VDCscore** , a new LLM-assisted metric with divide-and-conquer strategy. Our extensive evaluation on various video and image captioning benchmarks demonstrated that **AuroraCap** achieves competitive results, even outperforming state-of-the-art models in some tasks. We also conduct thorough ablation studies to validate the effectiveness of token merging and other aspects of our model. We found that the current model performs poorly in terms of the trade-off between performance and the scale of input tokens. Additionally, there is still room for improvement in camera handling and detailed captioning. 

13 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

## **Acknowledgments** 

We acknowledge Pika Labs for the computing resources support. We are also thankful to Xtuner[*] , lmms-eval[*] , and SGLang[*] for their well-developed and user-friendly codebase. We would like to thank Enxin Song for her contribution to the benchmark processing. We would like to thank Professor Yilun Du, Professor Saining Xie, and Professor Christopher D. Manning for their valuable discussions and revisions of this paper. 

## **Ethics Statement** 

This research on video captioning utilizes publicly available datasets, ensuring that all data complies with privacy regulations. We acknowledge the potential biases that can arise in automatic caption generation, particularly concerning gender, race, or other characteristics. We have taken measures to evaluate and minimize such biases, while remaining committed to further improvements. Additionally, we recognize the potential risks of misuse, such as generating misleading captions, and have checked the training dataset with safeguards against such applications. 

## **Reproducibility Statement** 

We have made several efforts to ensure the reproducibility of our work. All the key implementation details, including the architecture of our model, the training procedures, and hyperparameter settings, are described in supplementary meterial Section S7. The introduction of the used evaluation benchmarks and settings are in Section S8. The introduction of the used evaluation benchmarks and settings are in Section S8. Prompt template of **VDC** generation is in Section S11. Question-answer pairs generation prompt template of **VDCscore** is in Section S12. And calculation details of Elo ranking is in Section S15. We have also outlined any hardware configurations and computation requirement used for our experiments in Figure S14 to further support reproducibility. The code for model training, evaluation, deployment, as well as model weight, benchmark, and training dataset of all the training stages will be released with the paper. 

> *GitHub Repository: `https://github.com/InternLM/xtuner` 

> *GitHub Repository: `https://github.com/EvolvingLMMs-Lab/lmms-eval` 

> *GitHub Repository: `https://github.com/sgl-project/sglang` 

14 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

## **References** 

- [1] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. Nocaps: Novel object captioning at scale. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 8948–8957, 2019. 

- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. _Advances in neural information processing systems_ , 35:23716–23736, 2022. 

- [3] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In _Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part V 14_ , pages 382–398. Springer, 2016. 

- [4] Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Neural module networks. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 39–48, 2016. 

- [5] Max Bain, Arsha Nagrani, Gul Varol, and Andrew Zisserman.¨ Frozen in time: A joint video and image encoder for end-to-end retrieval. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1728–1738, 2021. 

- [6] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In _Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization_ , pages 65–72, 2005. 

- [7] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Yuanzhen Li, Tomer Michaeli, et al. Lumiere: A space-time diffusion model for video generation. _arXiv preprint arXiv:2401.12945_ , 2024. 

- [8] Lucas Beyer, Andreas Steiner, Andre Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel´ Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. _arXiv preprint arXiv:2407.07726_ , 2024. 

- [9] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marc¸al Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 4291–4301, 2019. 

- [10] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your ViT but faster. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- [11] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In _Proceedings of the ieee conference on computer vision and pattern recognition_ , pages 961–970, 2015. 

- [12] Mu Cai, Jianwei Yang, Jianfeng Gao, and Yong Jae Lee. Matryoshka multimodal models. _arXiv preprint arXiv:2405.17430_ , 2024. 

15 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

- [13] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-aware diffusion video editing. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 23040–23050, 2023. 

- [14] David Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In _Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies_ , pages 190–200, 2011. 

- [15] Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. Allava: Harnessing gpt4vsynthesized data for a lite vision-language model. _arXiv preprint arXiv:2402.11684_ , 2024. 

- [16] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. _arXiv preprint arXiv:2403.06764_ , 2024. 

- [17] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? _arXiv preprint arXiv:2403.20330_ , 2024. 

- [18] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. _arXiv preprint arXiv:2311.12793_ , 2023. 

- [19] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. _arXiv preprint arXiv:2406.04325_ , 2024. 

- [20] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 13320–13331, 2024. 

- [21] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointly-scaled multilingual language-image model. _arXiv preprint arXiv:2209.06794_ , 2022. 

- [22] Xin Chen, Yu Zhu, Hua Zhou, Liang Diao, and Dongyan Wang. Chinesefoodnet: A large-scale image dataset for chinese food recognition. _arXiv preprint arXiv:1705.02743_ , 2017. 

- [23] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. _arXiv preprint arXiv:2312.14238_ , 2023. 

- [24] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. _arXiv preprint arXiv:2406.07476_ , 2024. 

- [25] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot 

16 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

impressing gpt-4 with 90%* chatgpt quality. _See https://vicuna. lmsys. org (accessed 14 April 2023)_ , 2(3):6, 2023. 

- [26] Dawei Dai, YuTang Li, YingGe Liu, Mingming Jia, Zhang YuanHui, and Guoyin Wang. 15m multimodal facial image-text dataset, 2024. 

- [27] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose visionlanguage models with instruction tuning. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [28] Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, Jose MF Moura, Devi´ Parikh, and Dhruv Batra. Visual dialog. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 326–335, 2017. 

- [29] Jie Deng, Wenhao Chai, Jianshu Guo, Qixuan Huang, Wenhao Hu, Jenq-Neng Hwang, and Gaoang Wang. Citygen: Infinite and controllable 3d city layout generation. _arXiv preprint arXiv:2312.01508_ , 2023. 

- [30] Jie Deng, Wenhao Chai, Junsheng Huang, Zhonghan Zhao, Qixuan Huang, Mingyan Gao, Jianshu Guo, Shengyu Hao, Wenhao Hu, Jenq-Neng Hwang, et al. Citycraft: A real crafter for 3d city generation. _arXiv preprint arXiv:2406.04983_ , 2024. 

- [31] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations_ , 2020. 

- [32] Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [33] Zhengfang Duanmu, Wentao Liu, Zhongling Wang, and Zhou Wang. Quantifying visual image quality: A bayesian view. _Annual Review of Vision Science_ , 7:437–464, 2021. 

- [34] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

- [35] Desmond Elliott, Stella Frank, Khalil Sima’an, and Lucia Specia. Multi30k: Multilingual englishgerman image descriptions. _arXiv preprint arXiv:1605.00459_ , 2016. 

- [36] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. _arXiv preprint arXiv:2405.21075_ , 2024. 

- [37] Haoyuan Gao, Junhua Mao, Jie Zhou, Zhiheng Huang, Lei Wang, and Wei Xu. Are you talking to a machine? dataset and methods for multilingual image question. _Advances in neural information processing systems_ , 28, 2015. 

17 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

- [38] Roopal Garg, Andrea Burns, Burcu Karagol Ayan, Yonatan Bitton, Ceslee Montgomery, Yasumasa Onoe, Andrew Bunner, Ranjay Krishna, Jason Baldridge, and Radu Soricut. Imageinwords: Unlocking hyper-detailed image descriptions. _arXiv preprint arXiv:2405.02793_ , 2024. 

- [39] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 6904–6913, 2017. 

- [40] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4D: Around the world in 3,000 hours of egocentric video. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18995–19012, 2022. 

- [41] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, et al. Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. _arXiv preprint arXiv:2311.18259_ , 2023. 

- [42] Mingfei Han, Xiaojun Chang, Heng Wang, and Linjie Yang. Shot2story20k: A new benchmark for comprehensive understanding of multi-shot videos. _arXiv preprint arXiv:2312.10300_ , 2023. 

- [43] Jinyi Hu, Yuan Yao, Chongyi Wang, Shan Wang, Yinxu Pan, Qianyu Chen, Tianyu Yu, Hanghao Wu, Yue Zhao, Haoye Zhang, Xu Han, Yankai Lin, Jiao Xue, Dahai Li, Zhiyuan Liu, and Maosong Sun. Large multilingual models pivot zero-shot multimodal learning across languages. _arXiv preprint arXiv:2308.12038_ , 2023. 

- [44] Ting-Hao Huang, Francis Ferraro, Nasrin Mostafazadeh, Ishan Misra, Aishwarya Agrawal, Jacob Devlin, Ross Girshick, Xiaodong He, Pushmeet Kohli, Dhruv Batra, et al. Visual storytelling. In _Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: Human language technologies_ , pages 1233–1239, 2016. 

- [45] Drew A Hudson and Christopher D Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 6700–6709, 2019. 

- [46] Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. The platonic representation hypothesis. _arXiv preprint arXiv:2405.07987_ , 2024. 

- [47] Vladimir Iashin and Esa Rahtu. A better use of audio-visual cues: Dense video captioning with bi-modal transformer. _arXiv preprint arXiv:2005.08271_ , 2020. 

- [48] Jitesh Jain, Jianwei Yang, and Humphrey Shi. Vcoder: Versatile vision encoders for multimodal large language models. _arXiv preprint arXiv:2312.14233_ , 2023. 

- [49] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 13700–13710, 2024. 

18 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

- [50] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2901–2910, 2017. 

- [51] Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions. _arXiv preprint arXiv:2407.06358_ , 2024. 

- [52] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. _arXiv preprint arXiv:2402.07865_ , 2024. 

- [53] Maxime Kayser, Oana-Maria Camburu, Leonard Salewski, Cornelius Emde, Virginie Do, Zeynep Akata, and Thomas Lukasiewicz. e-vil: A dataset and benchmark for natural language explanations in vision-language tasks. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 1244–1254, 2021. 

- [54] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 4015–4026, 2023. 

- [55] Jonathan Krause, Justin Johnson, Ranjay Krishna, and Li Fei-Fei. A hierarchical approach for generating descriptive image paragraphs. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 317–325, 2017. 

- [56] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In _Proceedings of the IEEE international conference on computer vision_ , pages 706–715, 2017. 

- [57] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _International journal of computer vision_ , 123:32–73, 2017. 

- [58] Hugo Laurenc¸on, Leo Tronchon, Matthieu Cord, and Victor Sanh.´ What matters when building vision-language models? _arXiv preprint arXiv:2405.02246_ , 2024. 

- [59] Paul Lerner, Olivier Ferret, Camille Guinaudeau, Herve Le Borgne, Romaric Besan´ c¸on, Jose G´ Moreno, and Jesus Lov´ on Melgarejo.´ Viquae, a dataset for knowledge-based visual question answering about named entities. In _Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval_ , pages 3108–3120, 2022. 

- [60] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-OneVision: Easy visual task transfer. _arXiv preprint arXiv:2408.03326_ , 2024. 

- [61] Boyi Li, Ligeng Zhu, Ran Tian, Shuhan Tan, Yuxiao Chen, Yao Lu, Yin Cui, Sushant Veer, Max Ehrlich, Jonah Philion, et al. Wolf: Captioning everything with a world summarization framework. _arXiv preprint arXiv:2407.18908_ , 2024. 

19 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

- [62] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _International conference on machine learning_ , pages 19730–19742. PMLR, 2023. 

- [63] Junyan Li, Delin Chen, Tianle Cai, Peihao Chen, Yining Hong, Zhenfang Chen, Yikang Shen, and Chuang Gan. Flexattention for efficient high-resolution vision-language models. _arXiv preprint arXiv:2407.20228_ , 2024. 

- [64] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. _arXiv preprint arXiv:2305.06355_ , 2023. 

- [65] Lei Li, Yuwei Yin, Shicheng Li, Liang Chen, Peiyi Wang, Shuhuai Ren, Mukai Li, Yazheng Yang, Jingjing Xu, Xu Sun, et al. M3IT: A large-scale dataset towards multi-modal multilingual instruction tuning. _arXiv preprint arXiv:2306.04387_ , 2023. 

- [66] Qingyun Li, Zhe Chen, Weiyun Wang, Wenhai Wang, Shenglong Ye, Zhenjiang Jin, Guanzhou Chen, Yinan He, Zhangwei Gao, Erfei Cui, et al. Omnicorpus: An unified multimodal corpus of 10 billion-level images interleaved with text. _arXiv preprint arXiv:2406.08418_ , 2024. 

- [67] Wei Li, Linchao Zhu, Longyin Wen, and Yi Yang. Decap: Decoding clip latents for zero-shot captioning via text-only training. _arXiv preprint arXiv:2303.03032_ , 2023. 

- [68] Xiaotong Li, Fan Zhang, Haiwen Diao, Yueze Wang, Xinlong Wang, and Ling-Yu Duan. Densefusion-1m: Merging vision experts for comprehensive multimodal perception. _arXiv preprint arXiv:2407.08303_ , 2024. 

- [69] Xirong Li, Weiyu Lan, Jianfeng Dong, and Hailong Liu. Adding chinese captions to images. In _Proceedings of the 2016 ACM on international conference on multimedia retrieval_ , pages 271–275, 2016. 

- [70] Xirong Li, Chaoxi Xu, Xiaoxu Wang, Weiyu Lan, Zhengxiong Jia, Gang Yang, and Jieping Xu. Coco-cn for cross-lingual image tagging, captioning, and retrieval. _IEEE Transactions on Multimedia_ , 21(9):2347–2360, 2019. 

- [71] Xirui Li, Chao Ma, Xiaokang Yang, and Ming-Hsuan Yang. Vidtome: Video token merging for zero-shot video editing. _arXiv preprint arXiv:2312.10656_ , 2023. 

- [72] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. _arXiv preprint arXiv:2311.17043_ , 2023. 

- [73] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. _arXiv preprint arXiv:2311.17043_ , 2023. 

- [74] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. _arXiv preprint arXiv:2305.10355_ , 2023. 

- [75] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. _arXiv preprint arXiv:2311.10122_ , 2023. 

20 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

- [76] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In _Text summarization branches out_ , pages 74–81, 2004. 

- [77] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. _arXiv preprint arXiv:2312.07533_ , 2023. 

- [78] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar,´ and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13_ , pages 740–755. Springer, 2014. 

- [79] Feng Liu, Tao Xiang, Timothy M Hospedales, Wankou Yang, and Changyin Sun. ivqa: Inverse visual question answering. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 8611–8619, 2018. 

- [80] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. _arXiv preprint arXiv:2310.03744_ , 2023. 

- [81] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024. 

- [82] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _Advances in neural information processing systems_ , 36, 2024. 

- [83] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2017. 

- [84] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. _Advances in Neural Information Processing Systems_ , 35:2507– 2521, 2022. 

- [85] Fan Ma, Xiaojie Jin, Heng Wang, Yuchen Xian, Jiashi Feng, and Yi Yang. Vista-llama: Reliable video narrator via equal distance to visual tokens. _arXiv preprint arXiv:2312.08870_ , 2023. 

- [86] Yunchuan Ma, Laiyun Qing, Guorong Li, Yuankai Qi, Quan Z Sheng, and Qingming Huang. Retrieval enhanced zero-shot video captioning. _arXiv preprint arXiv:2405.07046_ , 2024. 

- [87] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. _arXiv preprint arXiv:2306.05424_ , 2023. 

- [88] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. _Advances in Neural Information Processing Systems_ , 36:46212–46244, 2023. 

- [89] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. _Advances in Neural Information Processing Systems_ , 36, 2024. 

21 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

- [90] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In _Proceedings of the IEEE/cvf conference on computer vision and pattern recognition_ , pages 3195–3204, 2019. 

- [91] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. _arXiv preprint arXiv:2203.10244_ , 2022. 

- [92] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_ , pages 2200–2209, 2021. 

- [93] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 2630–2640, 2019. 

- [94] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In _2019 international conference on document analysis and recognition (ICDAR)_ , pages 947–952. IEEE, 2019. 

- [95] Mixkit. Free stock video clips, motion graphics, & music tracks, 2024. Accessed: 2024-08-24. 

- [96] Mathew Monfort, SouYoung Jin, Alexander Liu, David Harwath, Rogerio Feris, James Glass, and Aude Oliva. Spoken moments: Learning joint audio-visual representations from video descriptions. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 14871–14881, 2021. 

- [97] Arsha Nagrani, Paul Hongsuck Seo, Bryan Seybold, Anja Hauth, Santiago Manen, Chen Sun, and Cordelia Schmid. Learning audio-video modalities from image captions. In _European Conference on Computer Vision_ , pages 407–426. Springer, 2022. 

- [98] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy V Vo, Marc Szafraniec, Vasil Khali-´ dov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. _Transactions on Machine Learning Research_ , 2023. 

- [99] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th annual meeting of the Association for Computational Linguistics_ , pages 311–318, 2002. 

- [100] Pexels. Free imageds and video clips, 2024. Accessed: 2024-08-24. 

- [101] Pixabay. Stunning free images & royalty-free stock, 2024. Accessed: 2024-08-24. 

- [102] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In _Proceedings of the IEEE international conference on computer vision_ , pages 2641–2649, 2015. 

22 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

- [103] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pages 8748–8763. PMLR, 2021. 

- [104] Ruchit Rawal, Khalid Saifullah, Ronen Basri, David Jacobs, Gowthami Somepalli, and Tom Goldstein. Cinepile: A long video question answering dataset and benchmark. In _Synthetic Data for Computer Vision Workshop@ CVPR 2024_ , 2024. 

- [105] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jeanbaptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_ , 2024. 

- [106] Shuhuai Ren, Sishuo Chen, Shicheng Li, Xu Sun, and Lu Hou. Testa: Temporal-spatial token aggregation for long-form video-language understanding. _arXiv preprint arXiv:2310.19060_ , 2023. 

- [107] Marcus Rohrbach, Wei Qiu, Ivan Titov, Stefan Thater, Manfred Pinkal, and Bernt Schiele. Translating video content to natural language descriptions. In _Proceedings of the IEEE international conference on computer vision_ , pages 433–440, 2013. 

- [108] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. _International journal of computer vision_ , 115:211–252, 2015. 

- [109] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In _European Conference on Computer Vision_ , pages 146–162. Springer, 2022. 

- [110] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. _arXiv preprint arXiv:2403.15388_ , 2024. 

- [111] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for image captioning with reading comprehension. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16_ , pages 742–758. Springer, 2020. 

- [112] Gunnar A Sigurdsson, Gul Varol, Xiaolong Wang, Ali Farhadi, Ivan Laptev, and Abhinav Gupta.¨ Hollywood in homes: Crowdsourcing data collection for activity understanding. In _Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14_ , pages 510–526. Springer, 2016. 

- [113] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 8317–8326, 2019. 

- [114] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Xun Guo, Tian Ye, Yan Lu, Jenq-Neng Hwang, et al. Moviechat: From dense token to sparse memory for long video understanding. _arXiv preprint arXiv:2307.16449_ , 2023. 

23 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

- [115] Enxin Song, Wenhao Chai, Tian Ye, Jenq-Neng Hwang, Xi Li, and Gaoang Wang. Moviechat+: Question-aware sparse memory for long video question answering. _arXiv preprint arXiv:2404.17176_ , 2024. 

- [116] Alane Suhr, Mike Lewis, James Yeh, and Yoav Artzi. A corpus of natural language for visual reasoning. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_ , pages 217–223, 2017. 

- [117] Reuben Tan, Ximeng Sun, Ping Hu, Jui-hsien Wang, Hanieh Deilamsalehy, Bryan A Plummer, Bryan Russell, and Kate Saenko. Koala: Key frame-conditioned long video-llm. _arXiv preprint arXiv:2404.04346_ , 2024. 

- [118] Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. Visualmrc: Machine reading comprehension on document images. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pages 13878–13888, 2021. 

- [119] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. _arXiv preprint arXiv:2405.09818_ , 2024. 

- [120] Qwen team. Qwen2-vl. 2024. 

- [121] Yoad Tewel, Yoav Shalev, Idan Schwartz, and Lior Wolf. Zerocap: Zero-shot image-to-text generation for visual-semantic arithmetic. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 17918–17928, 2022. 

- [122] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 5238–5248, 2022. 

- [123] Yunjie Tian, Tianren Ma, Lingxi Xie, Jihao Qiu, Xi Tang, Yuan Zhang, Jianbin Jiao, Qi Tian, and Qixiang Ye. Chatterbox: Multi-round multimodal referring and grounding. _arXiv preprint arXiv:2401.13307_ , 2024. 

- [124] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. _arXiv preprint arXiv:2406.16860_ , 2024. 

- [125] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. _arXiv preprint arXiv:2401.06209_ , 2024. 

- [126] Atousa Torabi, Christopher Pal, Hugo Larochelle, and Aaron Courville. Using descriptive video services to create a large data source for video annotation research. _arXiv preprint arXiv:1503.01070_ , 2015. 

- [127] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 4566–4575, 2015. 

24 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

- [128] Andreas Veit, Tomas Matera, Lukas Neumann, Jiri Matas, and Serge Belongie. Coco-text: Dataset and benchmark for text detection and recognition in natural images. _arXiv preprint arXiv:1601.07140_ , 2016. 

- [129] Han Wang, Yanjie Wang, Yongjie Ye, Yuxiang Nie, and Can Huang. Elysium: Exploring objectlevel perception in videos via mllm. _arXiv preprint arXiv:2403.16558_ , 2024. 

- [130] Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lijuan Wang. Git: A generative image-to-text transformer for vision and language. _arXiv preprint arXiv:2205.14100_ , 2022. 

- [131] Jie Wang, Tao Ji, Yuanbin Wu, Hang Yan, Tao Gui, Qi Zhang, Xuanjing Huang, and Xiaoling Wang. Length generalization of causal transformers without position encoding. _arXiv preprint arXiv:2404.12224_ , 2024. 

- [132] Junke Wang, Lingchen Meng, Zejia Weng, Bo He, Zuxuan Wu, and Yu-Gang Jiang. To see is to believe: Prompting gpt-4v for better visual instruction tuning. _arXiv preprint arXiv:2311.07574_ , 2023. 

- [133] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, high-quality multilingual dataset for video-and-language research. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 4581–4591, 2019. 

- [134] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for longcontext interleaved video-language understanding. _arXiv preprint arXiv:2407.15754_ , 2024. 

- [135] Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal llms. _arXiv preprint arXiv:2312.14135_ , 2023. 

- [136] Wenhao Wu. Freeva: Offline mllm as training-free video assistant. _arXiv e-prints_ , pages arXiv– 2405, 2024. 

- [137] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of questionanswering to explaining temporal actions. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 9777–9786, 2021. 

- [138] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In _Proceedings of the 25th ACM international conference on Multimedia_ , pages 1645–1653, 2017. 

- [139] Haiyang Xu, Qinghao Ye, Ming Yan, Yaya Shi, Jiabo Ye, Yuanhong Xu, Chenliang Li, Bin Bi, Qi Qian, Wei Wang, et al. mplug-2: A modularized multi-modal foundation model across text, image and video. In _International Conference on Machine Learning_ , pages 38728–38748. PMLR, 2023. 

- [140] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 5288–5296, 2016. 

25 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

- [141] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. _arXiv preprint arXiv:2404.16994_ , 2024. 

- [142] Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, and Afshin Dehghan. Slowfast-llava: A strong training-free baseline for video large language models. _arXiv preprint arXiv:2407.15841_ , 2024. 

- [143] Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 5036–5045, 2022. 

- [144] Shen Yan, Tao Zhu, Zirui Wang, Yuan Cao, Mi Zhang, Soham Ghosh, Yonghui Wu, and Jiahui Yu. Videococa: Video-text modeling with zero-shot transfer from contrastive captioners. _arXiv preprint arXiv:2212.04979_ , 2022. 

- [145] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Just ask: Learning to answer questions from millions of narrated videos. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 1686–1697, 2021. 

- [146] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Zero-shot video question answering via frozen bidirectional language models. _Advances in Neural Information Processing Systems_ , 35:124–141, 2022. 

- [147] Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10714–10726, 2023. 

- [148] Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. _arXiv preprint arXiv:2310.06114_ , 2023. 

- [149] Barry Menglong Yao, Aditya Shah, Lichao Sun, Jin-Hee Cho, and Lifu Huang. End-to-end multimodal fact-checking and explanation generation: A challenging dataset and models. In _Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval_ , pages 2733–2743, 2023. 

- [150] Hanrong Ye, Haotian Zhang, Erik Daxberger, Lin Chen, Zongyu Lin, Yanghao Li, Bowen Zhang, Haoxuan You, Dan Xu, Zhe Gan, et al. Mm-ego: Towards building egocentric multimodal llms. _arXiv preprint arXiv:2410.07177_ , 2024. 

- [151] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. _Transactions of the Association for Computational Linguistics_ , 2:67–78, 2014. 

- [152] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynetqa: A dataset for understanding complex web videos via question answering. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 33, pages 9127–9134, 2019. 

26 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

- [153] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. _arXiv preprint arXiv:2311.16502_ , 2023. 

- [154] Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 6720–6731, 2019. 

- [155] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. _arXiv preprint arXiv:2306.02858_ , 2023. 

- [156] Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, et al. Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output. _arXiv preprint arXiv:2407.03320_ , 2024. 

- [157] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. _arXiv preprint arXiv:2406.16852_ , 2024. 

- [158] Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. _arXiv preprint arXiv:2303.16199_ , 2023. 

- [159] Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander Hauptmann, Yonatan Bisk, et al. Direct preference optimization of video large multimodal models from language model reward. _arXiv preprint arXiv:2404.01258_ , 2024. 

- [160] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. In _International Conference on Learning Representations_ , 2019. 

- [161] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model, April 2024. 

- [162] Yue Zhao, Long Zhao, Xingyi Zhou, Jialin Wu, Chun-Te Chu, Hui Miao, Florian Schroff, Hartwig Adam, Ting Liu, Boqing Gong, et al. Distilling vision-language models on millions of videos. _arXiv preprint arXiv:2401.06129_ , 2024. 

- [163] Zhonghan Zhao, Wenhao Chai, Xuan Wang, Li Boyi, Shengyu Hao, Shidong Cao, Tian Ye, Jenq-Neng Hwang, and Gaoang Wang. See and think: Embodied agent in virtual environment. _arXiv preprint arXiv:2311.15209_ , 2023. 

- [164] Zhonghan Zhao, Wenhao Chai, Xuan Wang, Ke Ma, Kewei Chen, Dongxu Guo, Tian Ye, Yanting Zhang, Hongwei Wang, and Gaoang Wang. Steve series: Step-by-step construction of agent systems in minecraft. _Computer Vision and Pattern Recognition Workshop_ , 2024. 

- [165] Zhonghan Zhao, Kewei Chen, Dongxu Guo, Wenhao Chai, Tian Ye, Yanting Zhang, and Gaoang Wang. Hierarchical auto-organizing system for open-ended multi-agent navigation. _arXiv preprint arXiv:2403.08282_ , 2024. 

27 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

- [166] Zhonghan Zhao, Ke Ma, Wenhao Chai, Xuan Wang, Kewei Chen, Dongxu Guo, Yanting Zhang, Hongwei Wang, and Gaoang Wang. Do we really need a complex agent system? distill embodied agent into a single model. _arXiv preprint arXiv:2404.04619_ , 2024. 

- [167] Yinhe Zheng, Guanyi Chen, Xin Liu, and Jian Sun. Mmchat: Multi-modal chat dataset on social media. _arXiv preprint arXiv:2108.07154_ , 2021. 

- [168] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 32, 2018. 

- [169] Xingyi Zhou, Anurag Arnab, Shyamal Buch, Shen Yan, Austin Myers, Xuehan Xiong, Arsha Nagrani, and Cordelia Schmid. Streaming dense video captioning. _arXiv preprint arXiv:2404.01297_ , 2024. 

28 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

## **Supplementary Material** 

The supplementary material is structured as follows: 

- Iiterature review about the related works in Section S1. 

- More results on long video benchmarks in Section S2. 

- N-gram based evaluation metrics in Section S3. 

- Benchmark comparison between **VDC** and other video captioning benchmarks in Section S4. 

- The calculation and the visualization examples of token merging in Section S5. 

- More ablation studies for **AuroraCap** in Section S6. 

- The training settings for **AuroraCap** in Section S7. 

- The introduction of the used evaluation benchmarks and settings in Section S8. 

- More limitations of **AuroraCap** at current stage in Section S10. 

- Additional evaluation metrics analysis in Section S9. 

- Prompt template of **VDC** generation in Section S11. 

- Question-answer pairs generation prompt template of **VDCscore** in Section S12. 

- More statistics information of **VDC** in Section S13. 

- More statistics information of **VDCscore** in Section S14. 

- Calculation details of Elo ranking in Section S15. 

- Case studies among several baselines in Section S16. 

- Prompt template of predicted answers extraction in Section S17. 

- Prompt template of correctness evaluation in Section S18. 

## **S1. Related Works** 

**Video Captioning** The general goal of video captioning is understanding a video and describing it with natural language. Unlike image captioning, video captioning requires the description also on the temporal dimension ( _e.g.,_ human action, camera and object movement). The current datasets including videos in the domain of human activities [11], cooking scene [47, 168], movie [114], and open domain [14]. In Table **1** , we show the summary of current video captioning benchmarks. Most of them are with short caption, which is not suitable for evaluating video detailed captioning task. There are several widely used metrics to evaluate the correctness of generated caption, such as BLEU [99], ROUGE-L [76], CIDEr [127], SPICE [3], and BERTScore [160]. Dense Video Captioning [147, 169] is also a captioning task but further requires localizing the events temporally. In this paper, we focus on the detailed description of a short video clip, where there are no scene changes or camera switches. 

**Large Multimodal Models for Video** With the develop of LLMs and LMMs [8, 12, 24, 49, 63, 72, 87, 110, 114, 115, 120, 155, 158, 162, 169], many recent works have explored adapting them into video understanding field ( _e.g.,_ Video-LLaMA [72], Video-LLaVA [75], VideoChat [64], Vista-LLaMA [85], LLaVA-Hound [159], Koala [117], Elysium [129], and MovieChat [114, 115]). Thanks to this flexible design, the models can combine pretrained knowledge with minimal trainable parameters. Instead of requiring large-scale training, using only a small amount of high-quality training data can even achieve better results. Most of the existing models use additional parameters for temporal modeling. There are also some interesting observations that we can actually build advancing LMMs without additional 

29 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

parameters for temporal modeling [77, 81, 106, 141, 156] or even without further training with videotext data [136]. Recent works ( _e.g.,_ FreeVA [136], LLaVA-Next [81], VLIA [77], and PLLaVA [141]) also find that the vanilla LLaVA-like model pretrained on high-quality image instruction data can also be a strong video understanding model. FreeVA further observe that using existing video instruction tuning data like Video-ChatGPT 100K [87] to tune LMMs may not necessarily lead to improvements. As the concurrent work, we also observe this phenomenon and proceeded to develop a video detailed captioning baseline training based on the LLaVA architecture. Chat-UniVi [49] employ a set of dynamic visual tokens to uniformly represent images and videos. However, for current video inputs, when we sample at 1 FPS, the frames are already sufficiently sparse; frame-to-frame similarity is quite low except in static scenes. On the other hand, if we use keyframes as input, the frame similarity will be even lower. Under these conditions, we believe that strong temporal merging should be avoided, as it is likely to impair the model’s core capabilities, especially since the CLIP vision encoder trained on images. 

As for the benchmark, inspired by LLaVA [82], Video-ChatGPT [87] introduces a 100K video clips with text instructions with the first vLMMs benchmark evaluation system powered by LLMs. MovieChat-1K [114] and CinePile [104] are question-answering based benchmark for long-form video understanding. Shot2Story20K [42] comprises videos with 2 to 8 shots each sourced for our dataset from the public video benchmark HDvila100M [143]. However, currently there is no video benchmark available to evaluate video detailed captioning tasks like IIW [38] and other datasets [66] did in image captioning field. Other works [150] focus on the specific domain like Ego. In this paper, our work fills this gap. As for metric, LLaVA-Hound-DPO [159] uses similar divide-and-conquer strategy to build a preference dataset, ensuring that instructional data remains factually consistent with detailed captions. Our goal for **VDCscore** , however, is to evaluate the accuracy of generated captions through QA pairs generated from ground truth. 

## **S2. More Results on Long-form Video Benchmarks** 

**Fig.** S1: Model performance on MovieChat-1K [114] **Fig.** S2: Model performance on benchmark. Egoschema [89] benchmark. 

|Model|Degign for Long Video|Breakpoint|Global|Model|Design for Long Video Acc.|Design for Long Video Acc.|
|---|---|---|---|---|---|---|
|Video-LLaMA|_×_|39.1|51.7|Random Choice<br>FrozenBLM|_×_<br>_×_|20.0<br>26.9|
|VideoChat|_×_|46.1|57.8|mPLUG-Owl|_×_|31.1|
|TimeChat|✓|46.1|73.8|TimeChat<br>Video-LLAVA|_×_<br>✓|33.0<br>38.4|
|VideoChatGPT|_×_|48.0|47.6|LLAMA-VID|✓|39.5|
|MovieChat|✓|48.3|62.3|LLAVA-NEXT-Video|✓|43.9|
|||||VideoLLAMA2|✓|51.7|
|MovieChat+|✓|49.6|71.2|MovieChat|✓|53.5|
|||||Human Performance||76.2|
|**AuroraCap**(ours)|_×_|52.6|59.7|AuroraCap (ours)|_×_|46.0|



We take MovieChat-1K [114], Egoschema [88] as the representative long video benchmark, and compare with existing models. As shown in the following Table S1 and Table S2, **AuroraCap** achieves comparable performance even without training on long videos and special design. We also plot the visual token kept ratio curve on EgoSchema as Figure S3, demonstrating the stability and effectiveness of our token merging method for longer video sequences. 

30 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0031-01.png)


**----- Start of picture text -----**<br>
Accuracy on Egoschema Pearson Correlation Coefficient r = 0.74<br>46.0<br>46 45.6<br>44.8 7.5<br>44 43.4<br>7.0<br>42<br>41.2<br>40 6.5<br>90% 80% 38.6<br>381.0 0.8 0.6 0.4 0.2 0.0 900 1000 1100 1200<br>Ratio of visual token kept Elo Score<br>Accuracy<br>VDCScore ROUGE<br>**----- End of picture text -----**<br>


**Fig.** S3: Token merging ablation curve on Egoschema [89], with best performance at a visual token kept ratio of 0.6 to 0.7. 

**Fig.** S4: Correlation between n-gram evaluation scores and human elo ranking on the **VDC** . 

## **S3. N-gram Based Evaluation Results** 

A variant of **VDCscore** could be developed to assess final triplets `(<question, correct answer, predicted answer>)` by employing a n-gram based evaluation approach as shown in Table S1 with correlation shown in Figure S4. 

**Table** S1: Evaluation results of various models on **VDC** based on ROUGE for answer matching. 

|Evaluation results|of various models on**VDC**based on ROUGE for answer|
|---|---|
|Model|Avg.<br>Camera<br>Short<br>Background<br>Main Object<br>Detailed|
|||
|LLaMA-VID<br>Video-ChatGPT-7B<br>ViLA-7B<br>Video-LLAVA-7B<br>LLAVA-1.5-7B<br>LongVA-7B<br>LLAVA-1.5-13B<br>LLAVA-NeXT-V7B<br>LLAVA-1.6-7B<br>LLAVA-1.6-13B<br>ShareGPT4Video-8B<br>LLAVA-OV-7B<br>InternVL-2-8B<br>**AuroraCap**-7B<br>Gemini-1.5 Pro|6.66<br>9.53<br>5.12<br>5.30<br>6.07<br>7.29<br>6.85<br>9.20<br>5.37<br>6.40<br>6.29<br>6.97<br>6.68<br>7.88<br>5.16<br>6.29<br>6.35<br>7.74<br>6.79<br>8.67<br>5.25<br>6.09<br>6.64<br>7.28<br>6.34<br>8.31<br>5.40<br>5.80<br>5.52<br>6.65<br>6.70<br>8.37<br>4.99<br>6.11<br>6.79<br>7.23<br>6.34<br>8.64<br>5.55<br>5.57<br>5.15<br>6.79<br>6.69<br>8.61<br>5.30<br>6.35<br>6.27<br>6.93<br>6.45<br>8.08<br>5.38<br>5.79<br>5.82<br>7.20<br>6.18<br>7.31<br>5.28<br>6.00<br>5.29<br>7.01<br>6.78<br>8.46<br>5.73<br>5.82<br>6.54<br>7.36<br>6.59<br>8.09<br>5.66<br>5.86<br>5.58<br>7.75<br>6.82<br>8.14<br>5.63<br>5.95<br>6.53<br>7.86<br>7.60<br>9.55<br>5.66<br>6.79<br>7.42<br>8.58<br>7.59<br>8.78<br>6.25<br>7.29<br>7.70<br>7.94|



31 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

## **S4. Benchmark Comparison** 

We compare our proposed **VDC** with some examples from several video captioning benchmarks as shown in Figure S5, Figure S6, and Figure S7. The corresponding captions are shown as followings: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0032-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0032-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0032-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0032-06.png)


**Fig.** S5: Video example of MSR-VTT [140] benchmark. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0032-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0032-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0032-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0032-11.png)


**Fig.** S6: Video example of VATEX [133] benchmark. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0032-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0032-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0032-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0032-16.png)


**Fig.** S7: Video example of **VDC** benchmark sampled from Ego4D [40]. 

## **Benchmark Caption (for VDC we show only the detailed caption)** 

MSR-VTT S5 (4 words) Teams are playing soccer. 

- VATEX S6 (13 words) A woman instructs and demonstrates how to remove the insides of a pumpkin. 

- **VDC** S7 (618 words) The video opens with an intimate close-up of a surface adorned with vibrant green moss and intricate lichen, initially evoking the serene beauty of a natural landscape. This organic imagery quickly transitions, revealing that the mossy surface is actually part of a motorcycle or vehicle’s engine compartment, creating a striking contrast between the lush textures of nature and the cold, hard lines of mechanical components. As the camera angle shifts, the viewer is drawn deeper into the engine compartment, where the interplay of moss overgrowth on various machinery introduces a fascinating blend of organic life and industrial elements, highlighting the unexpected coexistence of nature and technology. 

   - The perspective then zooms in, accentuating the rich details of the mossy growth, which clings tenaciously to the metallic surfaces, while a dark cavity beneath hints at the complexity of the machinery. The reflective metallic surfaces glint in the light, further enhancing the visual contrast and inviting the viewer to explore this unique juxtaposition. Suddenly, the scene shifts dramatically, with rapid camera motion creating a vibrant blur of colors and shapes, transforming the previously detailed views into a chaotic whirlwind, suggesting a swift movement through the intricate landscape of the engine compartment. 

   - As the motion blur begins to dissipate, the viewer is presented with a clearer image of a lightcolored, textured surface, where blurred mechanical components can be discerned, indicating a 

32 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

deceleration in movement. The camera stabilizes, revealing a rough-textured floor or ground that suggests an indoor or industrial environment, characterized by a sense of organized chaos. The scene transitions to a detailed examination of a cluttered workspace filled with tangled wires, casings, and components in a variety of colors, emphasizing the disorganized state of electronic or mechanical internals, possibly during a maintenance or repair process. 

The perspective shifts once more, showcasing darker, textured surfaces juxtaposed against lighter insulating materials, with hidden metallic elements peeking through, suggesting another angle within this same cluttered interior space. A human hand enters the frame, reaching out to interact with the components, signaling an active workspace filled with purpose. As the scene expands, additional hands join the fray, actively manipulating various objects within the crowded environment, signifying an ongoing task or collaborative effort amidst the complex array of components and materials. 

The atmosphere is imbued with a sense of urgency and engagement, as the camera captures the dynamic interactions of the individuals working together. The camera work remains fluid and dynamic, featuring a mix of close-up shots that highlight the intricate details of the components and wider angles that provide context to the bustling environment. The slightly shaky nature of the shots adds a layer of realism and immersion, drawing the viewer into the heart of the action. The low light conditions create a moody ambiance, with shadows dancing across the surfaces, enhancing the visual depth and interest of the scene. Overall, the video encapsulates a vivid portrayal of the intersection between nature and machinery, as well as the collaborative spirit of those engaged in the intricate task of maintenance and repair within this unique setting. 

## **S5. Token Merging Visualization** 

The number of tokens _r_ merged in each layer of the transformer in ViT is calculated by the following formula: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0033-06.png)


where _W_ , _H_ is the width and the height of the input image, _P_ is the patch size, _L_ is the number of layers in the transformer, ratio is the proportion of tokens to be kept. The merging is evenly distributed across all layers of the transformer. 

We show token merging visualization examples from COCO [78], VG [57], SA-1B [54], and OCRVQA [94] in Figure S8, since we only conduct frame-wise operation. Features extracted by CLIP ViT-H model[*] . We conduct image captioning for the examples given final token numbers by 7B **AuroraCap** as following. We remove line breaks in generated caption for a better reading experience. **# Token Describe this image in detail. (Figure S8 Row 1)** 

- 490 In the heart of a winter wonderland, a skier takes flight, suspended in mid-air against the backdrop of a clear blue sky. The skier, clad in a vibrant red and white jacket, is captured in a moment of pure exhilaration. The skier’s body is angled towards the left side of the image, with their skis crossed in an impressive display of skill and control. The skier’s helmet, a stark contrast to the surrounding snow, is adorned with a pink and white design, adding a touch of color to the otherwise monochrome landscape. The skier’s goggles, reflecting the bright sunlight, are a testament to the intensity of the sport. Beneath the skier, the snow-covered mountain slopes stretch out, dotted with trees that stand as silent spectators to the skier’s daring feat. The image is a freeze-frame of action and adventure, encapsulating the thrill and beauty of winter sports. 

- 154 In the midst of a clear blue sky, a skier is captured in a moment of thrilling action. The skier, clad in a vibrant red and white jacket, is suspended in mid-air, performing an impressive trick. The skier’s body is parallel to the ground, creating a sense of motion and excitement. The skier’s skis, a striking 

> *Huggingface Model: `https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378` 

33 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0034-01.png)


**Fig.** S8: Token merging visualization. From top to bottom, the image IDs are COCO:COCO-train2014000000247906, VG:2331508, SA-1B:sa-393200, OCR-VQA:71789111. From left to right, the number of tokens representing the images are 490, 154, 18, and 6. 

combination of red and white, are crossed in front of them, adding to the dynamic nature of the image. The skier’s helmet, a necessary safety measure, is white and contrasts with the colorful jacket. The backdrop to this action-packed scene is a breathtaking snowy mountain landscape. The mountain’s peak is visible in the distance, adding depth to the image. The overall composition of the image suggests a high-speed descent down the mountain, with the skier skillfully navigating the terrain. This image encapsulates the exhilarating sport of skiing, showcasing not only the skill of the skier but also the stunning beauty of the mountain landscape. 

- 18 In the heart of a snowy mountain, a daring skier takes flight. Dressed in a vibrant red and white jacket, black pants, and a protective white helmet, the skier is captured mid-air, performing an impressive trick. The skier’s body is parallel to the ground, arms outstretched, as if embracing the thrill of the moment. The skier’s skis, a striking combination of red and white, are pointed upwards towards the clear blue sky. The skier’s poles, held firmly in their hands, add to the dynamic nature of the scene. The backdrop to this exhilarating moment is a breathtaking snowy mountain landscape. The mountain’s peak, blanketed in snow, stretches out into the distance. The clear blue sky above is devoid of clouds, adding to the serene beauty of the scene. This image captures not just a moment, but a story 

34 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

of adventure, thrill, and the sheer joy of skiing. 

- 6 The image captures a thrilling moment of a snowboarder in mid-air, performing an impressive trick. The snowboarder, clad in a vibrant red and white jacket, black pants, and a protective helmet, is the central figure in this dynamic scene. The snowboarder is holding onto a rope with one hand, suggesting that they are being pulled up the mountain by a snowmobile, a common practice in snowboarding to gain speed and momentum. The rope extends from the top left corner of the image, creating a sense of movement and direction. The backdrop of the image is a breathtaking snowy mountain landscape, complete with trees and other natural elements. The snowboarder is positioned in the center of the image, drawing the viewer’s attention to their daring stunt. The image is a testament to the exhilarating sport of snowboarding and the skill and courage of the athletes who participate in it. It’s a freeze-frame of an adrenaline-fueled moment, encapsulating the thrill and excitement of snowboarding. 

## **# Token What is the number in the picture? (Figure S8 Row 2)** 

   - 490 The number in the picture is ”45”. It is displayed on the back of the jersey that the person is wearing. 154 The number in the picture is ”45”. 

   - 18 The number in the picture is ”45”. It is displayed on the back of the baseball jersey worn by the player in the image. 

   - 6 The number in the picture is ”45”. It is displayed in a large, bold font on the jersey of the baseball player in the image. 

- **# Token What is the relative position of the white and red cars in the picture? Which one in the left? (Figure S8 Row 3)** 

   - 490 The white car is positioned to the left of the red car in the image. The white car is closer to the foreground and is stopped at the traffic light, while the red car is further back, also stopped at the same traffic light. The relative position of the cars is such that the white car is in the left lane and the red car is in the right lane, both facing the same direction of traffic. 

   - 154 The white car is positioned to the left of the red car in the image. The red car is further back on the road, indicating it is behind the white car. This is determined by the perspective of the image, where the white car is closer to the foreground and the red car is further away, both in the same lane and moving in the same direction. 

   - 18 The white car is on the left side of the image, and the red car is on the right side. The white car is closer to the foreground, while the red car is further away, positioned behind the white car. 

   - 6 The white car is on the left side of the image, while the red car is on the right side. The white car is positioned closer to the foreground, while the red car is further back, near the intersection. 

## **S6. Ablation Studies** 

In this section, we conduct the ablation studies in several aspects including token merging, slowfast inference, efficiency, training strategy, and backbone selection. 

**Token merging.** As a core strategy of **AuroraCap** , token merging plays a significant role in reducing the number of visual tokens. We conduct extensive ablation studies to explore the impact of the token kept ratio _Rvtk_ in terms of performance across multiple tasks including image captioning, visual question answering, video captioning, and video question answering as shown in Figure S9 and Figure S10. We define the performance percentage as the proportion between the highest and lowest values on the entire performance curve. We identify the minimum retention thresholds for achieving 90% and 80% performance. As shown in Figure S9, while the performance of **AuroraCap** generally declines with fewer visual tokens across most benchmarks, it remains relatively stable at higher retention levels. Most models maintain satisfactory performance ( _>_ 80%) even with only 0.4 of _Rvtk_ , highlighting the 

35 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0036-01.png)


**----- Start of picture text -----**<br>
CIDEr on Flickr B@4 on Flickr METEOR on Flickr<br>88.33 87.31 86.76 34 33.32 32.8 32.86 27<br>85 83.55 32 31.75 26 26.08 26.07 25.77<br>25.34<br>80 30 25<br>75 75.35 28.52 24 23.92<br>28<br>23<br>70<br>65 90% 80% 64.09 26 90% 80% 25.0 2221 90% 80% 21.67<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>56 55.01 54.72 ROUGE-L on Flickr54.82 54.08 110 110.5 110.29 CIDEr on NoCaps 108.78 106.91 44 44.76 44.2 B@4 on NoCaps 43.76 43.12<br>54<br>105 42<br>52 51.65 100 100.33 40 40.36<br>50 95 38<br>48 90% 80% 48.66 90 90% 80% 88.54 36 90% 80% 36.14<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>METEOR on NoCaps ROUGE-L on NoCaps CIDEr on COCO-Cap<br>30 29.58 29.46 29.2 6160 60.41 60.31 59.92 59.51 120 119.42 120.0 119.81 119.19<br>29 28.86 59 115<br>28 27.62 58 57.8 112.68<br>27 57 110<br>56<br>2625 90% 80% 25.67 55 90% 80% 54.86 105 98 0% 101.98<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>B@4 on COCO-Cap METEOR on COCO-Cap ROUGE-L on COCO-Cap<br>35 34.55 34.8 34.65 34.89 29 28.6 28.63 28.5 28.37 57 57.01 57.11 56.97 57.09<br>34 33.33 28 27.53 56 55.76<br>33<br>27 55<br>32<br>31 98 0% 30.57 26 90% 80% 25.76 54 98 0% 53.42<br>30 53<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>Accuracy on GQA Accuracy on MMMU Accuracy on VQAv2<br>60 59.11 58.75 58.62 36 36.11 36.0 35.78 76 75.81 75.46 75.24 74.53<br>58 58.02 35.44 74<br>56 55.88 35 34.56 72 71.37<br>34 70<br>54<br>52 90% 80% 52.31 33 90% 80% 33 6866 90% 80% 66.5<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>CIDEr B@4<br>METEOR<br>ROUGE-L CIDEr B@4<br>METEOR ROUGE-L CIDEr<br>B@4<br>METEOR ROUGE-L<br>Accuracy Accuracy Accuracy<br>**----- End of picture text -----**<br>


**Fig.** S9: Ablation study of token merging on image captioning on Flickr [151], NoCaps [1], COCO-Cap [78], visual question answering in GQA [45], MMMU [153], VQAv2 [39]. We found that token merging significantly reduces the number of tokens while maintaining minimal performance drop, and even showing improvement in some tasks. We highlight the token merging ratio when achieving 90% and 80% performance with the dash line and filled area. 

36 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0037-01.png)


**----- Start of picture text -----**<br>
CIDEr on MSR-VTT B@1 on MSR-VTT B@4 on MSR-VTT<br>32 32.12 31.61 58 58.58 21.02 20.55<br>56.57 20<br>30<br>28.34 56 55.22 18.63<br>28 54 18 17.68<br>26 25.18 16.52<br>24 23.31 52 51.53 50.86 16<br>50<br>22 14<br>20 98 0% 19.21 48 90% 80% 47.56 98 0% 12.65<br>12<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>METEOR on MSR-VTT ROUGE-L on MSR-VTT CIDEr on VATEX<br>24 23.91 50 49.48 49.3 48.99 34 33.34<br>49 32<br>23 48 47.34 30 29.28 28.87<br>22 21.48 21.6 21.82 47 28 27.25<br>26.36<br>21 46 26<br>20 20.35 45 44.93 44.03 24<br>19 98 0% 19.07 4443 98 0% 2220 20.24 98 0%<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>B@1 on VATEX B@4 on VATEX METEOR on VATEX<br>56 56.33 18 17.48 19 18.4118.29<br>18<br>5452 51.21 52.23 51.8253.53 16 14.76 15.31 15.51 16.09 17 16.2 17.45 17.03<br>50 14 16<br>48 15<br>12<br>4644 44.11 98 0% 10 10.45 98 0% 14 14.0 98 0%<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>ROUGE-L on VATEX Accuracy on ActivityNet-QA<br>40.76 42.5 Accuracy on MSRVTT-QA 60.72 60.8<br>40 38.48 39.03 39.44 40.0 39.4 38.6 38.8 40.4 60 59.6 59.6<br>38 37.58 37.5 36.67 58 57.6<br>35.0<br>36 32.5 56<br>30.0<br>34 33.31 90% 80% 27.5 90% 80% 27.0 54 53.7 98 0%<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>Score on ActivityNet-QA Accuracy on MSVD-QA62.6 Accuracy on iVQA<br>61.0 54.8<br>3.7 3.63 3.69 3.65 60 59.8 58.84 5452 51.8 53.0 51.4 51.2<br>3.6 3.6 55 53.31 50<br>3.5 3.49 50 48<br>46<br>45<br>3.4 90% 3.37 80% 44<br>3.3 40 90% 80% 37.6 42 98 0% 41.0<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>CIDEr B@1 B@4<br>CIDEr<br>METEOR ROUGE-L<br>B@1 B@4<br>METEOR<br>ROUGE-L Accuracy Accuracy<br>Score<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


**Fig.** S10: Ablation study of token merging on video captioning on MSRVTT [140], VATEX [133], video question answering on MSRVTT-QA [140], ActivityNet-QA [152], MSVD-QA [138], iVQA [79]. We found that token merging significantly reduces the number of tokens while maintaining minimal performance drop, and even showing improvement in some tasks. We highlight the token merging ratio when achieving 90% and 80% performance with the dash line and filled area. 

37 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0038-01.png)


**----- Start of picture text -----**<br>
Setting  Setting  Setting  Setting<br>1.591 1.635 1.119 1.975 2.201 1.892 7.454<br>100%<br>80%<br>60%<br>40%<br>20%<br>0%<br>MSR-VTT VATEX ANet MSVD MSRVTT iVQA VDC<br>**----- End of picture text -----**<br>


**Fig.** S11: Comparison between different inference settings: **A** : _Rvtk_ = 1.0, without SGLang, **B** : _Rvtk_ = 0.1, without SGLang, **C** : _Rvtk_ = 1.0, with SGLang, **D** : _Rvtk_ = 0.1, with SGLang. The number indicates the maximum inference time in seconds for each benchmark. 

efficiency of our approach. Visual token retention thresholds vary by task complexity, with more visually demanding tasks needing higher retention of visual tokens. For instance, CIDEr [127] on COCO-Cap [128] maintains over 90% performance with an _Rvtk_ of 0.3, whereas accuracy on GQA [45] drops to 90% when the _Rvtk_ is reduced to 0.8. Unlike image understanding, the optimal performance across most video understanding benchmarks occurs at a relatively low _Rvtk_ as depicted in Figure S10. And for MSR-VTT [140], VATEX [133], and ActivityNet-QA [152], even achieve better results at extremely low _Rvtk_ ( _<_ 0.1). It indicates that comparing to image, video input have higher redundancy. Note that **AuroraCap** focuses on spatial visual token merging, while the temporal features introduce additional complexity to explore the token merging laws. Appendix S5 shows more calculation details and the visualization results of token merging. 

**Efficiency.** To assess the inference speed, we utilize the inference time per video question-answering pair in seconds (TPV) as an evaluative metric. SGLang is an accelerated serving framework for LLMs and multimodal LLMs. We consider four settings including with or without token merging and SGLang. Figure S11 indicates the minimum TPV achievable in each settings across seven video understanding datasets. Reducing the visual tokens and using SGLang result in excellent inference times per video question-answering pair while all the datasets with short video and question inputs. In contrast, maintaining full visual tokens or omitting the use of SGLang results in comparatively slower performance, demonstrating the superior inference efficiency of **AuroraCap** . For each input, we process the video at a resolution of 378 _×_ 378 and sample 8 frames using single H100 GPU. 

**Slowfast inference.** Inspired by Slowfast-LLaVA [142], we explore whether combining frames with low and high _Rvtk_ can enhance performance. In practice, we don’t conduct token merging in the first frame and concatenate them with the merged tokens from subsequent frames. We apply this strategy to both video captioning and video question answering tasks, comparing performance with and without the inclusion of full first-frame visual tokens. As illustrated in Table S2, slowfast inference brings marginal performance improvement in video question answering tasks or even drop in video captioning tasks but with more computing cost. Therefore, by default, we don’t using slowfast inference for video detailed captioning. 

38 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

**Table** S2: Ablation on slowfast inference for **AuroraCap** -7B. We present the average performance among different token merging ratio on various video understanding benchmarks. We show that slowfast inference brings marginal performance improvement in video question answering tasks or even drop in video captioning tasks but with more computing cost. 

|Setting|C|MSR-VTT<br>B@1<br>B@4<br>M<br>R|VATEX<br>C<br>B@1<br>B@4<br>M<br>R|ANet<br>Acc|MSVD<br>Acc|MSRVTT<br>Acc|
|---|---|---|---|---|---|---|
|w/o slowfast|26.72|53.01<br>17.58<br>21.25<br>46.78|28.03<br>52.28<br>15.22<br>16.95<br>38.30|58.55|56.45|37.26|
|w/ slowfast|26.18|51.68<br>17.00<br>21.20<br>46.16|28.07<br>52.16<br>15.12<br>16.95<br>38.27|59.66|55.65|38.22|
|∆|-0.54|-1.33<br>-0.58<br>-0.05<br>-0.62|+0.04<br>-0.12<br>-0.10<br>-0.01<br>-0.03|+1.11|-0.80|+0.96|




![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0039-03.png)


**----- Start of picture text -----**<br>
Accuracy on Camera Caption Score on Camera Caption Accuracy on Short Caption<br>4442 42.14 42.83 43.35 43.29 41.25 2.3 2.24 2.23 32 31.59 32.07 31.88 31.5<br>2.2<br>40 2.12 2.14 31<br>38 2.1 2.06 30 30.27 29.94<br>36 2.0<br>29<br>34 1.9 1.88<br>32 32.6 1.8 28<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>Score on Short Caption Accuracy on Background Caption Score on Background Caption<br>1.75 36 35.42 35.55 1.9<br>34.98<br>1.701.65 1.64 1.66 1.65 1.64 34 34.61 33.32 33.2 8 1.81.7 1.82 1.8 1.78 1.821 1.71 1.6 9<br>1.60 1.58 32 1.6<br>1.55 1.54<br>30 1.5<br>1.50<br>1.4<br>1.45 28<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>Accuracy on Main Object Caption Score on Main Object Caption Accuracy on Detailed Caption<br>39<br>38 37.58 38.51 2.052.001.95 1.93 1.97 41.040.5 40.61 41.03 40.81 40.51<br>37 36.62 36.34 37.05 1.901.85 1.86 1.87 1.89 40.0 40.13<br>36 1.80 1.77 39.5 39.57<br>1.75<br>35 39.0<br>34.43 1.70<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>Score on Detailed Caption Accuracy on VDC Score on VDC<br>2.8 2.85 38 37.28 37.34 37.3 38.01 2.1 2.08<br>37 36.73<br>2.6<br>2.0<br>36 1.96<br>2.4 35 1.9 1.91 1.91 1.88<br>2.2 2.1 2.12 2.12 2.11 2.0 5 34 33.96 1.8 1.79<br>2.0 33<br>1.7<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>Score<br>Accuracy Accuracy<br>Score Score<br>Accuracy<br>Score<br>Accuracy Accuracy<br>Score Score<br>Accuracy<br>**----- End of picture text -----**<br>


**Fig.** S12: Comparison between performance with and without the inclusion of full first-frame visual tokens during inference on **VDC** . As the visual token kept ratio varies, the blue curve indicates performance with slowfast inference, while the red curve represents performance without slowfast inference. 

39 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0040-01.png)


**----- Start of picture text -----**<br>
CIDEr on MSR-VTT B@4 on MSR-VTT METEOR on MSR-VTT<br>32.5 32.12 31.61 21.02 20.55 24 23.91<br>20<br>30.0 28.34 18.63 23<br>27.5 18 17.68<br>25.0 23.31 25.18 16 16.52 22 21.48 21.6 21.82<br>22.5 21<br>20.0 19.21 14 12.65 20 20.35<br>17.5 12 19 19.07<br>15.0<br>10<br>12.5 18<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>ROUGE-L on MSR-VTT CIDEr on VATEX B@4 on VATEX<br>50 49.48 49.3 48.99 3432 33.34 18 17.48<br>48 47.34 30 29.28 28.8 7 16 15.31 15.51 16.09<br>46 28 27.25 26.36 14.76<br>44.93 26 14<br>44 44.03 24<br>12<br>22<br>42 20 20.24 10 10.45<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>METEOR on VATEX ROUGE-L on VATEX 44 Accuracy on MSRVTT-QA<br>40.76<br>19 42<br>18.41 18.2 9 40 39.44 40.4<br>1817 17.45 17.03 38 37.58 38.48 39.03 403836 39.4 38.6 38.8 36.67<br>16.2<br>16 36 34<br>32<br>15<br>14 14.0 34 33.31 3028 27.0<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>Score on MSRVTT-QA Accuracy on ActivityNey-QA Score on Activityney-QA<br>62 3.8<br>60.72 60.8<br>2.8 2.77 2.78 2.75 2.73 60 59.6 59.6 3.7 3.63 3.69 3.65<br>2.6 2.59 3.6 3.6<br>58 57. 6<br>3.5 3.49<br>2.4<br>56<br>3.4 3.37<br>2.2 2.16 54 53.7 3.3<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>Accurac 62.6 y on MSVD-QA Score on MSVD-QA Accuracy on iVQA<br>60 59.8 61.0 58.84 3.6 3.54 3.61 3.51 54 53.0 54.8<br>3.4 3.41 52 51.8 51.4 51.2<br>55 53.31 3.2 3.2 50<br>50 48<br>3.0<br>46<br>45<br>2.8 44<br>40 37.6 2.6 2.59 42 41.0<br>1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0 1.0 0.8 0.6 0.4 0.2 0.0<br>Ratio of visual token kept Ratio of visual token kept Ratio of visual token kept<br>CIDEr B@4 METEOR<br>CIDEr B@4<br>ROUGE-L<br>METEOR ROUGE-L Accuracy<br>Score Score<br>Accuracy<br>Score<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


**Fig.** S13: Comparison between performance with and without the inclusion of full first-frame visual tokens during inference on video captioning on MSRVTT [140], VATEX [133], video question answering on MSRVTTQA [140], ActivityNet-QA [152], MSVD-QA [138], iVQA [79]. As the visual token kept ratio varies, the blue curve indicates performance with slowfast inference, while the red curve represents performance without slowfast inference. 

40 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

We also present the performance curve with and without the inclusion of full first-frame visual tokens as the visual token kept ratio varies during inference across multiple video understanding tasks. As illustrated in Figure S13 and Figure S12, despite the inclusion of full first-frame visual tokens, slowfast inference does not consistently result in significantly positive effects on performance. In some cases, the incorporation of full first-frame visual tokens even worsens performance degradation as the kept ratio decreases, particularly in video captioning tasks. 

**Training strategy.** Alternative training strategies for the language stage of **AuroraCap** are less frequently explored, which is the primary focus of this section. For a fair comparison, we use the same training datasets across all settings and maintain consistent hyper-parameters. The following training settings are explored: 

- **Setting A:** End-to-end training and set _Rvtk_ to 1.0. 

- **Setting B:** Following Slowfast-LLaVA [142], we retain full visual tokens in the first frame and concatenate with merged tokens using _Rvtk_ of 0.1. 

- **Setting C:** End-to-end training and set _Rvtk_ to 0.1. 

- **Setting D:** Most videos in the training data have no more than 8 key frames, while a subset (mainly from ShareGPT4Video [19]) contains significantly more. We first exclude this subset from end-to-end training and then use it to train solely the LLM, enhancing its ability to handle multi-frame inputs. 

• **Setting E:** Training solely the LLM match the performance of Setting **A** set _Rvtk_ at 0.1. We implement these training strategies, track training costs in H100 hours, and evaluate across various video understanding tasks. As shown in Figure S14, while training with an _Rvtk_ of 1.0 improves performance, it significantly increases training time. Surprisingly, mixing lower and higher visual token ratios during training offers no significant advantage. Training only the LLM under the two settings results in a performance drop, indicating that enhancing long video understanding still requires collaboration with the finetuning the visual encoder. Therefore, we choose Setting **C** as the final training strategy. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0041-08.png)


**----- Start of picture text -----**<br>
Setting  Setting  * Setting  Setting  Setting<br>33.1 39.5 62.9 67.7 45.4 56.2 1120.0<br>100%<br>80%<br>60%<br>40%<br>20%<br>0%<br>MSR-VTT VATEX ANet MSVD MSRVTT iVQA GPU Hours<br>**----- End of picture text -----**<br>


**Fig.** S14: Comparison between different training strategy in Language stage. We take Accuracy for QuestionAnswering tasks and CIDEr for captioning tasks as the evaluation metric and present the performance percentage. We choose Setting **C** as the final training strategy as shown with *. The number shows the maximum value for each benchmark. 

41 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

**Backbone selection.** We use the the training loss among the last ten iterations in original LLaVA alignment pretraining stage to guidance the ViT and LLM backbones selection as shown in Table S3. 

**Table** S3: Final training loss during pretraining stage with original LLaVA pretraining data. 

|**Table**S3: Final trainingloss durin|g pretrain|ingstage with original LLaVApretr|ainingdata|.|
|---|---|---|---|---|
|ViT|ViT Size|LLM|LLM Size|Loss|
|facebook/dinov2-giant|1,136M|microsoft/phi-2|2.7B|3.3021|
|openai/clip-vit-large-patch14-336|428M|Qwen/Qwen1.5-0.5B-Chat|0.5B|3.1001|
|openai/clip-vit-large-patch14-336|428M|microsoft/phi-2|2.7B|2.8067|
|laion/CLIP-ViT-bigG-14-laion2B-39B-b160k|1,845M|microsoft/phi-2|2.7B|2.7124|
|facebook/dinov2-giant|1,136M|lmsys/vicuna-13b-v1.5|13B|2.3895|
|laion/CLIP-ViT-bigG-14-laion2B-39B-b160k|1,845M|internlm/internlm2-chat-7b|7B|2.3437|
|laion/CLIP-ViT-bigG-14-laion2B-39B-b160k|1,845M|internlm/internlm2-chat-20b|20B|2.2745|
|laion/CLIP-ViT-bigG-14-laion2B-39B-b160k|1,845M|deepseek-ai/deepseek-llm-67b-chat|67B|2.1572|
|laion/CLIP-ViT-bigG-14-laion2B-39B-b160k|1,845M|mistralai/Mistral-7B-Instruct-v0.1|7B|2.1569|
|openai/clip-vit-large-patch14-336|428M|mistralai/Mixtral-8x7B-Instruct-v0.1|8x7B|2.0815|
|apple/DFN5B-CLIP-ViT-H-14-378|632M|lmsys/vicuna-13b-v1.5-16k|13B|2.0443|
|laion/CLIP-ViT-bigG-14-laion2B-39B-b160k|1,845M|lmsys/vicuna-7b-v1.5-16k|7B|2.0365|
|laion/CLIP-ViT-bigG-14-laion2B-39B-b160k|1,845M|mistralai/Mixtral-8x7B-Instruct-v0.1|8x7B|1.9889|
|openai/clip-vit-large-patch14-336|428M|lmsys/vicuna-7b-v1.5|7B|1.9762|
|laion/CLIP-ViT-bigG-14-laion2B-39B-b160k|1,845M|meta-llama/Llama-2-13b-chat-hf|13B|1.9708|
|laion/CLIP-ViT-bigG-14-laion2B-39B-b160k|1,845M|lmsys/vicuna-13b-v1.5|13B|1.9412|
|apple/DFN5B-CLIP-ViT-H-14-378|632M|lmsys/vicuna-7b-v1.5-16k|7B|**1.8679**|



## **S7. Detailed Training Settings** 

We use CLIP ViT-H[*] , Vicuna-1.5-7B[*] as the initialization of **AuroraCap** -7B. Training hyper-parameters for both stages are shown in Table S4. For visual data preprocessing, we resize each image or video frame to the short side of 378 while keeping original aspect. Instead of doing center crop, we conduct bilinear position embedding interpolation for each training sample. For video data, we extract frames at 2 FPS uniformly. For token merging, we use constant schedule for each blocks. 

**Table** S4: Training hyper-parameters for **AuroraCap** . 

|Hyper-parameters|Pretraining stage<br>Vision stage<br>Language stage|
|---|---|
|||
|trainable parameters|MLP<br>ViT + MLP<br>ViT + MLP + LLM|
|warmup schedule|linear|
|warmup start factor|1e-5|
|warmup ratio|0.03|
|learning rate schedule|cosine decay|
|optimizer|AdamW [83]<br>|
|optimizer hyper-parameters|_β_1,_β_2 = (0.9, 0.999)|
|weight decay|0.1|
|max norm|1|
|epoch|1|
|peak learning rate|2e-4<br>1e-4<br>2e-5|
|total equivalent batch size|512<br>6,144<br>768|
|token keep proportion|100%<br>100%<br>10%|



- *Huggingface Model: `https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378` 

- *Huggingface Model: `https://huggingface.co/lmsys/vicuna-7b-v1.5-16k` 

42 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

**Table** S5: Summary of datasets used for training **AuroraCap** in Pretraining stage. 

|Task|# Sample|Dataset|
|---|---|---|
|||LAION-CC-SBU-595K [82], ShareGPT4V [18], ALLaVA-|
|Image Captioning|1.3M|Caption-LAION-4V [15], ALLaVA-Caption-VFLAN-4V [15],|
|||DenseFusion [68]|



**Table** S6: Summary of datasets used for training **AuroraCap** in Vision stage. For classification, Reasoning, VQA, and Generation tasks, we adopt the dataset processed by M[3] IT [65] to fit the training objective of language models. 

|Task|# Sample|Dataset|
|---|---|---|
|Captioning|1,925K|ShareGPT4V-PT [18], TextCaps [111], Image-Paragraph-<br>Captioning [55]|
|Object-centric|438K|COST [48], ChatterBox [123], V* [135]|
|||COCO-GOI [78], COCO-Text [128], ImageNet [108], COCO-|
|Classifcation|238K|ITM [78],<br>e-SNLI-VE [53], Mocheg [149], IQA [33]|
|Reasoning|100K|CLEVR [50], NLVR [116], VCR [154], VisualMRC [118],<br>Winoground [122]|
|||VQA v2 [39], Shapes VQA [4], DocVQA [92], OK-VQA [90],|
|||Text-VQA [113], OCR-VQA [94], A-OK-VQA [109], Sci-|
|VQA|3,518K|enceQA [84]<br>ST-VQA [9], ViQuAE [59], LLaVA-OneVision [60]|
|Generation|145K|Visual Storytelling [44], Visual Dialog [28], Multi30k [35]<br>COCO-Caption CN [70], Flickr-8k-Caption CN [69], multi-|
|Chinese|193K|modal Chat [167], FM-IQA [37], ChineseFoodNet [22]|
|Total|6.6M|For all datasets, we uniformly sample without duplication.|



**Table** S7: Summary of datasets used for training **AuroraCap** in Language stage. 

|Task|# Sample|Dataset||
|---|---|---|---|
|Image Captioning|1,779K|ShareGPT4V<br>[18],<br>ALLaVA-Caption-LAION-4V<br>ALLaVA-Caption-VFLAN-4V<br>[15],<br>DenseFusion|[15],<br>[68],|
|||FaceCaption [26]||
|Video Captionin|1,659K|MiraData [51], LLaVA-Hound [159], ShareGPT4Video<br>Private Data|[19],|
|||LLaVA-Mix-665K [80], LVIS-Instruct4V [132], ALLaVA-||
|Image Instruction|9,742K|Instruct-LAION-4V [15], ALLaVA-Instruct-VFLAN-4V<br>Cambrian [124], M4-Instruct [81]|[15],|
|Video Instruction|268k|LLaVA-Hound [159], ShareGPT4Video [19]||
|Language-only|143K|Evol-Intruct-GPT4-Turbo-143K [15]||
|Total|15.0M|We duplicate video captioning and instruction datasets||
|||twice.||



43 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

## **S8. Evaluation Benchmarks and Settings** 

We list all the hyper-parameters and prompt used for evaluation as shown in Table S8. For our proposed **VDC** , we show the settings as following with the max number of the tokens of 1,024: Type Prompt used for evaluation 

Camera Describe any camera zooms, pans, or angle changes. 

Background Summarize the background setting of the video based on these frames. 

Main Object Describe the main subject, including their attributes and movements throughout the video. Detail Imagine the video from these frames and describe it in detail. 

**Table** S8: Evaluation settings summary for each benchmarks. For all benchmarks we set temperature, top p, number of beams to 0, 0, 1 respectively. 

|Benchmark|# Sample|# Tokens|Prompt|
|---|---|---|---|
|Flickr [102]|31,784|64|Provide a one-sentence caption for the provided image.|
|NoCaps [1]|4,500|64|Provide a one-sentence caption for the provided image.|
|COCO-Cap [78]|5,000|64|Provide a one-sentence caption for the provided image.|
|ChartQA [91]|2,500|16|Answer the question with a single word.|
|DocVQA [92]|5,349|32|Answer the question using a single word or phrase.|
|TextCaps [111]|3,166|64|Provide a one-sentence caption for the provided image.|
|GQA [45]|12,578|16|Answer the question using a single word or phrase.|
|POPE [74]|9,000|128|Answer the question using a single word or phrase.|
|MMMU [153]|900|16|Answer with the option letter from the given choices directly.<br>Answer the question using a single word or phrase.|
|VQAv2 [39]|214,354|16|Answer the question using a single word or phrase.|
|MSR-VTT [140]|1,000|64|Provide a one-sentence caption for the provided video.|
|VATEX [133]|4,478|64|Provide a brief single-sentence caption for the last video below.|
|MSVD-QA [138]|1,161|64|Answer the question using a single word or phrase.|
|ActivityNet-QA [152]|8,000|64|Answer the question using a single word or phrase.|
|MSRVTT-QA [138]|6,513|64|Answer the question using a single word or phrase.|
|iVQA [145]|6,000|64|Answer the question using a single word or phrase.|



## **S9. Additional Evaluation Metrics Analysis** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0044-09.png)


**----- Start of picture text -----**<br>
Pearson Correlation Coefficient r = 0.39 Pearson Correlation Coefficient r = 0.69 Pearson Correlation Coefficient r = 0.72<br>7.5<br>7.4<br>0.40<br>7.0 7.2<br>0.35 7.0<br>6.5<br>6.8<br>0.30 6.0<br>900 1000 1100 1200 900 1000 1100 1200 900 1000 1100 1200<br>Elo Score Elo Score Elo Score<br>Clip Similarity<br>VDCScore (5 QA) VDCScore (10 QA)<br>**----- End of picture text -----**<br>


**Fig.** S15: Correlation between the CLIP evaluation scores, different number of QA pairs used in **VDCscore** and human elo ranking on **VDC** . 

In this section, we make the analysis about more evaluation metrics on **VDC** other than **VDCscore** . As shown in Figure S15, we found that the correlation between CLIP score and human ratings is 

44 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

low (0.39), which may be due to the fact that, in a lengthy caption, there may be only one or two fine-grained errors that the CLIP text encoder lacks the capability to detect. Meanwhile, we test the impact of different numbers of QA pairs on the results when calculating **VDCscore** . As shown in the Figure S15, more QA pairs produced more robust test results. **VDCscore** shows over 0.86 of pearson correlation with human when using 20 QA pairs. 

## **S10. Limitations** 

We evaluate **AuroraCap** on various visual question answering benchmarks as shown in Table S9. Since the performance of the VQA task heavily depends on the performance of the LLM, we chose the same LLM for a fair comparison. Also, due the the limitation of OCR-related samples in training dataset, **AuroraCap** does not perform well in OCR at current stage as shown in Table S10. 

**Table** S9: Comparison **AuroraCap** with LLM-based SoTA methods on visual question answering benchmarks under zero-shot setting. 

|Model|LLM|MMMU (900)<br>GQA (12,578)<br>POPE (9,000)<br>VQAv2 (214,354)<br>Acc<br>Acc<br>F1<br>Acc|
|---|---|---|
||||
|LLaVA-1.5-7B|Vicuna-1.5-7B|35.30<br>61.97<br>85.87<br>76.64|
|LLaVA-1.5-13B|Vicuna-1.5-13B|34.80<br>63.24<br>85.92<br>78.26|
|LLaVA-1.6-7B|Vicuna-1.5-7B|35.10<br>64.23<br>86.40<br>80.06|
|LLaVA-1.6-13B|Vicuna-1.5-13B|35.90<br>65.36<br>86.26<br>80.92|
||||
|**AuroraCap**-7B|Vicuna-1.5-7B|36.11<br>59.72<br>83.31<br>75.85|



**Table** S10: Limitation in terms of OCR capability compared with LLaVA models. Appendix S8 shows the introduction and metrics of each benchmark. 

|Model|LLM|ChartQA (2,500)<br>DocVQA (5,349)<br>TextCaps (3,166)<br>Acc<br>Acc<br>Acc|
|---|---|---|
||||
|LLaVA-1.5-7B|Vicuna-1.5-7B|18.24<br>28.08<br>98.15|
|LLaVA-1.5-13B|Vicuna-1.5-13B|18.20<br>30.29<br>103.92|
|LLaVA-1.6-7B|Vicuna-1.5-7B|54.84<br>74.35<br>71.79|
|LLaVA-1.6-13B|Vicuna-1.5-13B|62.20<br>77.45<br>67.39|
||||
|**AuroraCap**-7B|Vicuna-1.5-7B|25.88<br>34.60<br>93.33|



45 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

## **S11. VDC Generation Prompt Template** 

Following [51], we utilize LLM to generate structured detailed captions. Given an input video, LLM return five detailed captions, including camera caption, short caption, background caption, main object caption and detailed caption for the entire video guided by our designed prompt template. The complete prompt is shown as followings: 

## **Type Prompt** 

- SYSTEM You are describing the video. Please provide detailed captions of the video from different aspects. 

   - User Please provide detailed and comprehensive captions for the following content: 1. Short Caption: Summarize the video in one detailed sentence, capturing key actions and the overall mood. 2. Background Caption: Provide a detailed description of the background, including objects, location, weather, time, and any dynamic elements such as movements in the environment. 3. Main Object Caption: Give a thorough description of the main subject’s actions, attributes, interactions, and movements throughout the video frames, including changes in posture, expression, or speed. 4. Camera Caption: Describe the camera work in detail, including shot types, angles, movements, transitions, and any special effects used to enhance the video. 5. Reference Caption: Generate a detailed dense caption for the video that is at least 300 words long. The caption should capture all visible actions, environmental details, and the overall emotional atmosphere in depth. Describe in detail the interactions between the main subjects and their environment, including subtle nuances of their movements or expressions. Elaborate on the sounds, textures, and other sensory experiences depicted in the video. Discuss the camera techniques used extensively, including shot types, angles, movements, and transitions. Highlight the mood and tone of the video throughout, creating a rich narrative that connects viewers emotionally to the scene. Include comprehensive descriptions of background elements that add context and depth, such as weather conditions, time of day, and cultural or historical settings. Make sure to provide a vivid portrayal that is engaging, informative, and rich enough for AI to re-generate the video content. No need to provide summary content. Do not describe each frame individually. Avoid using phrases like ’first frame’. The description should be rich enough for AI to re-generate the video. Please generate the response as a Python dictionary string with keys like ’short caption’. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. These are the frames from the video: [VIDEO KEYFRAME INPUT] 

- `GPT-4o` Short Caption: In a sunny outdoor setting, two workers engage in a collaborative effort to prepare a gravel base for a construction project, showcasing teamwork and focus. Camera Caption: The camera captures the scene from a low angle, providing a unique perspective that emphasizes the workers’ actions and the surrounding environment. The shots are primarily medium close-ups, focusing on the workers’ upper bodies and hands as they interact. The camera occasionally shifts to a wider angle, revealing more of the workspace and background elements. There are smooth transitions between shots, maintaining a steady flow that enhances the viewer’s engagement with the unfolding narrative. The use of a slightly fish-eye lens effect adds depth to the scene, making the environment feel more immersive. Background Caption: The scene is set in a residential area with a well-maintained lawn and a brick wall partially constructed in the background. The ground is covered with a white plastic sheet, and scattered gravel is visible. A few wooden planks lie on the ground, and tools are scattered around, including a black water bottle and a power tool. 

      - Main Object Caption: The main subjects are two workers, one wearing a gray sweatshirt and gloves, and the other dressed in casual work attire. They are actively engaged in conversation, with the first worker gesturing with his hands to emphasize points while explaining the task at hand. His posture is relaxed yet focused, indicating a collaborative atmosphere. The second worker listens attentively, occasionally nodding and responding with gestures, showcasing a dynamic interaction. Both workers exhibit a sense of camaraderie and professionalism as they prepare for the next steps in their project. Reference Caption: The video unfolds ... (300 words) ... through them. 

User The video has been describe from the following aspects:1. short caption: short caption, 2. background 

46 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

caption: background caption, 3. main object caption: main object caption, 4. camera caption: camera caption, 5.Reference Caption: reference caption Please generate a detailed dense caption for the video that is pretty long. You should expand the reference caption upon the information provided in the short caption, background caption, main object caption, and camera caption. Ensure that the detailed caption does not introduce any new entities or relationships that were not mentioned in the previous captions. Make sure to provide a vivid portrayal that is engaging, informative, and rich enough for AI to re-generate the video content. ”Avoid using phrases like ’first frame’, ’short caption’, ’background caption’, ’main object caption’, and ’camera caption’. The description should be rich enough for AI to re-generate the video. 

- `GPT-4o` Detailed Caption: The video opens with ... (500 words) ... literature that surrounds them. User [Manually check and refine] 

## **S12. Question-answer Pairs Generation Prompt Template of VDCscore** 

To decompose the ground-truth structured detailed captions in **VDC** , we utilize `GPT-4o` as the LLM assistant to generate numerous short question-answer pairs for subsequent evaluation. The full prompt and example cases are presented as followings: 

## **Type Prompt** 

- SYSTEM You are an intelligent chatbot designed for generating 20 question-answer pairs given a detailed description of a video or image. You are describing the video. Here’s how you can accomplish the task: INSTRUCTIONS: - Cover the main objects and actions in the video or image. 

      - The questions should be open-ended and start with ’What’, ’Who’, ’Where’, ’When’, ’Why’, ’How’, etc. - The answer should be a short sentence or phrase. 

      - Generate 20 question-answer pairs. 

   - User Please generate 20 question-answer pairs given a detailed description of a video or image: detailed description: The camera work is dynamic, featuring a mix of close-up shots and wider angles that provide context to the environment. The shots are slightly shaky, adding a sense of realism and immersion to the viewer. Transitions between shots are smooth, with the camera often panning or tilting to follow the subject‘s movements. The use of low light creates a moody atmosphere, with shadows enhancing the visual interest of the scene ... 

      - Please generate the response in the form of a Python list of tuple with the question and the corresponding answer. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python list of tuple. For example, your response should look like this: [(the question 1, the answer 1), (the question 2, the answer 2), ...]. 

- `GPT-4o` [”question”: ”What is the style of camera work in the video?”, ”answer”: ”dynamic”, ”question”: ”What is the effect of shaky shots in the video?”, ”answer”: ”realism and immersion”, ”question”: ”What type of transitions are used in the video?”, ”answer”: ”smooth”, ”question”: ”Why is the moody atmosphere created in the video?”, ”answer”: ”through low light”, ”question”: ”What type of camera movement is achieved through panning and tilting?”, ”answer”: ”following the subject’s movements”, ”question”: ”What is the effect of the camera movement on the viewer?”, ”answer”: ”sense of realism and immersion”, ...] 

User [Manually check and refine] 

47 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

## **S13. More Statistics Information of VDC** 

Figure S16 indicates the word distribution of the structured captions in **VDC** . Table S11 illustrates the distribution of the question categories we provided for **VDCscore** calculation. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0048-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0048-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0048-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0048-06.png)


**----- Start of picture text -----**<br>
(a) Camera Caption (b) Short Caption (c) Background Caption<br>(d) Main Object Caption (e) Detailed Caption (f) Total<br>**----- End of picture text -----**<br>


**Fig.** S16: Word cloud of different structured captions in **VDC** , showing the diversity. 

**Table** S11: Distribution of the question categories in **VDCscore** . 

|Type|Background<br>Camera<br>Short<br>Main object<br>Detailed|
|---|---|
|Environment & Scene<br>3,464<br>2,543<br>4,401<br>3,097<br>3,549<br>Character & Object<br>6,272<br>3,779<br>4,834<br>6,035<br>6,440<br>Intent & Outcomes<br>4,585<br>5,105<br>5,067<br>4,474<br>4,640<br>Action<br>2,861<br>1,198<br>1,857<br>3,192<br>1,932<br>Attributes & Relationship<br>944<br>334<br>1,388<br>1,478<br>1,280||



## **S14. More Statistics Information of VDCscore** 

We generate a total of 96,902 question-answer pairs for **VDC** , with an average of 18.87 pairs per detailed caption. As depicted in Figure S17, each section of the structured captions includes a similar number of question-answer pairs. Additionally, Figure S18 presents the distribution of question types generated for **VDC** . To enhance the evaluation of detailed captions, we configure all questions as open-ended. `Environment & Scene` encompasses inquiries about location, environment, atmosphere, scene, and time. `Character & Object` focuses on entities within the caption, such as people, animals, plants, and objects, while `Attribute & Relation` examines their attributes and interrelations. `Intent & Outcomes` addresses deeper interpretative questions regarding methods, purposes, reasons, and outcomes. We further analyze the distribution of generated question types within structured captions. Main object captions predominantly feature `Character & Object` questions, whereas `Environment & Scene` questions are more prominent in background captions, and `Camera` questions constitute a larger proportion in camera. 

48 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0049-01.png)


**----- Start of picture text -----**<br>
Main Object Caption<br>19.93%<br>Camera<br>Caption<br>19.72%<br>Short<br>Caption<br>20.14%<br>Detailed<br>19.77% Caption<br>Background 20.44%<br>Caption<br>**----- End of picture text -----**<br>


**Fig.** S17: Question-answer pairs proportaion in **VDC** . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0049-03.png)


**----- Start of picture text -----**<br>
Attribute & Relation Environment & Scene<br>4.84% 15.22%<br>Action<br>9.85%<br>Camera<br>6.78%<br>Character<br>&<br>Object<br>24.42%<br>28.72% 10.17%<br>Intent & Results Emotion<br>**----- End of picture text -----**<br>


**Fig.** S18: Distribution of question type in **VDCscore** . 

## **S15. Calculation of Elo Ranking** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0049-06.png)


**Fig.** S19: GUI screen for Elo ranking. 

In this section, we present the methodology for evaluating and ranking **AuroraCap** and various models using the Elo rating system as shown in Figure S19. The parameters used in the simulation are summarized in Table S12. 

1. For all models participating in the ELO ranking, we collecte the output captions for each video. 

2. As shown in Figure S19, we develop a frontend tool to randomly select captions generated by two different models for the same video. Without revealing the model IDs, human evaluators then chose the caption they found better. 

3. We record comparison results and calculated the ELO values based on the parameters in Table S12. 

4. In Figure **5** , we calculate the Pearson correlation between different metrics and human ELO values, 

49 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

ultimately showing that our proposed VDC metric is the closest to human evaluations. 

**Table** S12: Elo parameter setting. 

|**Table**S12: Elo parameter s|etting.|
|---|---|
|Parameter|Number|
|initial Elo mean<br>Elo standard deviation<br>base of logarithm<br>scaling factor<br>K-factor<br>minimum Elo rating<br>number of simulated matches|1,000<br>300<br>10<br>400<br>32<br>700<br>2,778|



## **S16. Case Study** 

We perform an extensive case study of **AuroraCap** on a variety of videos for video detailed captioning. As shown as followings, **AuroraCap** is capable of providing excellent detailed captions regarding the camera motion, background and main object with less hallucination. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0050-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0050-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0050-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0050-09.png)


**Fig.** S20: Video example of MSR-VTT [140] benchmark. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0050-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0050-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0050-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0050-14.png)


**Fig.** S21: Video example of VATEX [133] benchmark. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0050-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0050-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0050-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0050-19.png)


**Fig.** S22: Video example. 

50 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-04.png)


**Fig.** S23: Video example. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-09.png)


**Fig.** S24: Video example. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-14.png)


**Fig.** S25: Video example. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-19.png)


**Fig.** S26: Video example. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-21.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-22.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-23.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark_images/iclr_2025_AuroraCap_Efficient,_Performant_Video_Detailed_Captioning_and_a_New_Benchmark.pdf-0051-24.png)


**Fig.** S27: Video example. 

## **Example Caption** 

Figure S20 [short caption] A man with a mustache and a woman with a ponytail are sitting at a table. Figure S21 [short caption] A man plays the drums in a recording studio. 

- Figure S22 [detailed caption] The video depicts a serene beach scene where a young woman stands on the sandy shore, gazing out towards the ocean. She is wearing a black beanie and a pink jacket, adding a pop of color to the otherwise muted scene. The beach, a vast expanse of sand, stretches out in front of her, 

51 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

meeting the ocean at the horizon. The ocean, a vast body of water, is visible in the background. The beach is bathed in a soft, diffused light, creating a dreamy atmosphere. The girl’s gaze is directed towards the horizon, suggesting a sense of wonder or contemplation. The image is slightly blurred, adding a dreamy quality to the scene. The woman’s position on the beach, coupled with the gentle waves of the ocean, suggests a moment of contemplation or admiration. The relative positions of the objects suggest a peaceful day at the beach, with the girl possibly enjoying the serene view of the ocean. The colors are mostly muted, with the girl’s pink jacket standing out against the sandy beach and the blue ocean. The blurred background and the out-of-focus elements, such as the ocean and the sky, contribute to the sense of tranquility and focus on the woman. There is no text present in the video, and the colors are muted, with the exception of the pink jacket, which stands out against the more subdued tones of the surroundings. 

- Figure S23 [detailed caption] The video features a man with a beard and long hair, lying on the ground with his head resting on a tree trunk. He is wearing a colorful shirt with a mix of orange and green patterns. The man’s face is partially obscured by the tree trunk, but his eyes are visible, looking upwards. He appears to be smiling or laughing, with his mouth open. The man’s arms are crossed over his chest, suggesting a relaxed or playful posture. The background is blurred, but it appears to be a natural outdoor setting, possibly a forest or park. In the background, another person is partially visible, their presence subtly hinted at by a hand reaching out from the left side of the frame. The colors in the video are warm, with the man’s shirt standing out against the muted background. The man’s hair and beard are dark, and he has a light complexion. The overall mood of the video is lighthearted and carefree. There are no visible texts or other objects in the video, and the relative positions of the objects remain constant with the man in the foreground and the second person in the background. 

- Figure S24 [background caption] The background features a race track with visible tire marks and barriers, surrounded by grassy areas and a few scattered trees. The track is set in a rural or semi-rural location, with hills in the distance and a cloudy sky overhead, suggesting overcast weather conditions. The track itself appears well-maintained with a smooth surface, designed for high-speed racing. The weather, indicated by the cloudy sky, contributes to the overall ambiance of the scene, enhancing the sense of a cool, possibly early morning or late afternoon setting. 

- Figure S25 [camera caption] The camera maintains a steady, frontal view throughout the video, capturing the woman’s expressions and the intricate details of the bookshelves. The camera starts at the right edge of the bookshelf, moving across the shelf, and ends at the left edge of the bookshelf. The framing is consistent, focusing on the woman and the bookshelves, with the camera positioned at a medium distance to the subject. This movement provides a comprehensive view of the bookshelf, showcasing the variety of books and their arrangement on the shelves. The camera occasionally pans to reveal the depth of the library, showcasing the rows of books and the inviting atmosphere. The use of natural light enhances the visual appeal, creating a warm and inviting tone throughout the video. 

- Figure S26 [camera caption] The view shot remains relatively static, focusing on the children playing in the backyard. The camera angle is at eye level, capturing the scene from a distance that allows both children to be visible. There is minimal camera movement, maintaining a steady focus on the children and their activities. The sequence of video frames suggests a continuous moment of play without significant changes in shooting angles or camera movement. 

- Figure S27 [main object caption] The main subject in the video, a black car, is seen driving down a street that appears to be in a state of disarray. The car moves steadily forward, navigating around obstacles such as a blue car parked on the side of the road. The car’s movement is smooth and continuous, suggesting it is either in motion or has just come to a stop. The environment around the car is chaotic, with debris scattered across the road and signs of destruction, indicating a recent event or disaster. The car’s position remains central in the frame, with the camera angle focused on it from a slightly elevated perspective, possibly from a vehicle or a structure above. 

52 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

## **S17. Predicted Answer Extraction Prompt Template** 

Given the question-answer pairs based on the ground truth caption, we utilize `GPT-4o` to extract predicted answers based on the generated caption by our designed prompt template. The complete prompt is shown as followings: 

## **Type Prompt** 

- SYSTEM You are an intelligent chatbot designed for providing accurate answers to questions related to the content based on a detailed description of a video or image. 

   - Here’s how you can accomplish the task:” 

—— 

## ##INSTRUCTIONS: 

   - Read the detailed description carefully. 

   - Answer the question only based on the detailed description. 

   - The answer should be a short sentence or phrase. 

- User Please provide accurate answers to questions related to the content based on a detailed description of a video or image: 

   - detailed description: The setting is an urban street with a distinctive red and white painted bike lane, which is part of a larger roadway. The lane is marked with white lines and the word ”WALK” is painted in large, bold letters, indicating a pedestrian crossing area. The pavement is a mix of red brick and grey tiles, with a curb separating the bike lane from the adjacent road. The weather appears to be clear and sunny, casting sharp shadows on the ground, suggesting it is daytime. There are no other vehicles or pedestrians in sight, and the road is empty, which could imply a quiet time of day or a less busy area. The presence of a yellow circular object on the handlebar of the bicycle is notable, possibly a bell or a light. 

question: What objects are the cyclist’s hands interacting with? 

   - DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide short but accurate answer. 

- `GPT-4o` The cyclist’s hands are interacting with the bicycle handlebars. 

## **S18. Correctness Evaluation Prompt Template** 

Following [87], we evaluate the correctness and score of the predicted answers with the assistant of `GPT-4o` . Given the question, correct answer, and predicted answer from the generated caption, `GPT-4o` should return the _True_ or _False_ judgement and relative score (0 to 5). The complete prompt is shown as followings: 

## **Type Prompt** 

- SYSTEM You are an intelligent chatbot designed for evaluating the correctness of generative outputs for questionanswer pairs. 

   - Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here’s how you can accomplish the task: 

—— 

## ##INSTRUCTIONS: 

   - Focus on the meaningful match between the predicted answer and the correct answer. 

   - Consider synonyms or paraphrases as valid matches. 

   - Evaluate the correctness of the prediction compared to the answer. 

- User Please evaluate the following video-based question-answer pair: Question: What type of surface can be seen covering the floor? Correct Answer: carpet 

53 

AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark 

Predicted Answer: Carpeted surface. 

Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. 

Please generate the response in the form of a Python dictionary string with keys ’pred’ and ’score’, where value of ’pred’ is a string of ’yes’ or ’no’ and value of ’score’ is in INTEGER, not STRING. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. 

For example, your response should look like this: _{_ ’pred’: ’yes’, ’score’: 4.8 _}_ . 

`GPT-4o` _{_ ’pred’: ’yes’, ’score’: 5 _}_ 

54 

