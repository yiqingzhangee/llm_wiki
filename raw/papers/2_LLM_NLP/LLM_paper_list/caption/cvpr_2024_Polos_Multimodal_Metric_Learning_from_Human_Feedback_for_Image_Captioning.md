# **Polos: Multimodal Metric Learning from Human Feedback for Image Captioning** 

Yuiga Wada 

Kanta Kaneda Daichi Saito Komei Sugiura Keio University https://yuiga.dev/polos 

## **Abstract** 

_Establishing an automatic evaluation metric that closely aligns with human judgments is essential for effectively developing image captioning models. Recent data-driven metrics have demonstrated a stronger correlation with human judgments than classic metrics such as CIDEr; however they lack sufficient capabilities to handle hallucinations and generalize across diverse images and texts partially because they compute scalar similarities merely using embeddings learned from tasks unrelated to image captioning evaluation. In this study, we propose Polos, a supervised automatic evaluation metric for image captioning models. Polos computes scores from multimodal inputs, using a parallel feature extraction mechanism that leverages embeddings trained through large-scale contrastive learning. To train Polos, we introduce Multimodal Metric Learning from Human Feedback (M_[2] _LHF), a framework for developing metrics based on human feedback. We constructed the Polaris dataset, which comprises 131K human judgments from 550 evaluators, which is approximately ten times larger than standard datasets. Our approach achieved state-of-theart performance on Composite, Flickr8K-Expert, Flickr8KCF, PASCAL-50S, FOIL, and the Polaris dataset, thereby demonstrating its effectiveness and robustness._ 

## **1. Introduction** 

Extensive research on image captioning has yielded a wide array of practical applications in society, spanning from assisting people who are blind to facilitating dialog about images and answering questions derived from visual content (e.g., [21, 24, 66]). To effectively advance the development of image captioning models, it is crucial to establish an automatic evaluation metric that closely aligns with human judgment. Previous research has shown that classical automatic evaluation metrics [10, 12, 40, 49, 62] exhibit weak correlation with human judgments [10, 26]. This has prompted the introduction of data-driven automatic evaluation metrics [26, 34, 75, 76]. However, by merely measuring the similarity of embeddings learned from tasks unrelated to image captioning, these metrics potentially misjudge caption quality, which raises concerns about their accuracy in evaluating image captioning models. Further- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0001-08.png)


Figure 1. Our supervised metric Polos computes evaluation scores from multimodal inputs by integrating human feedback within the novel framework M[2] LHF. Polos is capable of modeling intricate relationships within the vector space of text-image pairs as well as text-text pairs, thereby effectively evaluating the depicted samples. 

more, some experiments have identified the limitations of these metrics regarding handling _hallucinations_ adequately. 

Although recent similarity-based [26, 31, 34, 75, 76] and learning-based [35, 53] metrics have demonstrated performance superior to classic metrics, they are still not completely satisfactory. For instance, while the correlation between the state-of-the-art (SOTA) metric and human judgments on the Flickr8K dataset [27] is only about 0.56, the correlation coefficient among human judgments is approximately 0.73 [10]. This discrepancy may emerge because similarity-based metrics merely compute cosine similarity from well-known embeddings (e.g., [16, 19, 50]) learned from tasks unrelated to image captioning evaluation, which potentially leads to a misrepresentation of caption quality. Moreover, many learning-based metrics are specifically 

for limited settings (e.g., closed-vocabulary settings), and therefore fail to accommodate diverse images and texts. 

In this paper, we propose Polos, a supervised automatic evaluation metric for image captioning models. Alongside this, we introduce Multimodal Metric Learning from Human Feedback (M[2] LHF), a framework used to develop a practical supervised metric for image captioning. Fig.1 illustrates our proposed metric Polos and the M[2] LHF framework. As a key contribution, our proposed metric fuses both similarity-based and learning-based approaches. Previous similarity-based approaches merely compute scalar similarities using classic methodologies (e.g., cosine similarity and optimal transport), whereas our metric models intricate relationships in the vector space of text-image pairs and texttext pairs. This is achieved through the _parallel feature extraction mechanism_ that leverages SimCSE [22] and CLIP [50], which we provide details for in Section 3.3. 

To train a metric that embodies the aforementioned characteristics through the M[2] LHF approach, we have constructed the Polaris dataset, which contains a diverse range of human judgments. Compared with existing datasets, Polaris contains a greater diversity of captions and a more extensive range of evaluations. Notably, our dataset comprises 131K human judgments, whereas even the largest existing dataset [1] contains only approximately 12K in total. Regarding the total number of evaluators involved, even the CapEval1K dataset [35], which has the largest number of evaluators to the best of our knowledge, was assessed by only five evaluators. By contrast, our dataset is distinguished by its assessment by a much higher number of evaluators at 550 evaluators. Furthermore, our dataset is superior to existing datasets such as [1, 10, 27, 35] in terms of the inclusion of diverse captions, which were collected from humans and generated from ten image captioning models, including modern models. 

The main contributions of this paper[1] are as follows : 

- We propose Polos[2] , a supervised automatic evaluation metric for image captioning models. 

- We introduce M[2] LHF, a novel framework used to develop a practical metric for image captioning. 

- We introduce a _parallel feature extraction mechanism_ that leverages text embeddings [42] pretrained with SimCSE and vision-language embeddings [50]. 

- We constructed the Polaris dataset, which contains 131,020 human judgments from 550 evaluators. 

- We achieved SOTA performance on image captioning benchmarks including Composite, Flickr8K-Expert, and Flickr8K-CF, PASCAL-50S, FOIL, and Polaris. 

> 1Project page: https://yuiga.dev/polos 

> 2 **P** ractical visi **O** n-and- **L** anguage evaluati **O** n metic for image captioning model **S** 

## **2. Related Work** 

Data-driven metrics can be broadly divided into similaritybased metrics [26, 31, 34, 75] and learning-based metrics [35, 51, 53]. Similarity-based metrics compute similarities using classic approaches such as cosine similarity and optimal transport in an unsupervised manner, whereas learningbased metrics compute scores in a supervised manner. 

**Standard and similarity-based metrics.** Standard automatic metrics for evaluating image captioning models include BLEU [49], ROUGE [40], METEOR [12], CIDEr [62], and SPICE [10], which are primarily based on either _n_ -grams or scene graphs. Extensions to these standard metrics, such as CIDEr-R [48] and JaSPICE [63], have also been proposed. Despite their widespread use, in several studies, researchers have highlighted the limitations of these metrics, indicating their suboptimal performance [26, 34, 51, 53, 54]. This has led to the emergence of datadriven metrics such as BERTScore [75] and MoverScore [76]. Additionally, there are similar metrics that leverage image features directly, such as [26, 29, 31, 34, 35]. 

CLIPScore [26] evaluates captions in an unsupervised manner by computing their similarity with embeddings derived from CLIP [50]. Its distinctive feature is its capacity not only to evaluate based on a reference-with-image manner but also in a reference-free context. Similarly, PACS [53] fine-tunes CLIP on _generated_ image-text pairs and evaluates captions in the same manner. However, these dependences solely on the computation of cosine similarity between CLIP embeddings could potentially constrain its effectiveness and robustness across diverse scenarios. 

MID[31] uses the negative Gaussian cross-mutual information using CLIP features. It was proposed as a bridge between reference-with-image and reference-free metrics and has demonstrated strong performance across multiple image captioning benchmarks. However, it is crucial to highlight that MID relies solely on the text embeddings provided by CLIP for processing text data, which could potentially lead to suboptimal evaluation. This concern stems from the fact that CLIP is predominantly trained on a dataset that comprises mainly short noun phrases, which means that it is possibly suboptimal for evaluating the longer sentences typically generated by image captioning models [53]. 

**Learning-based metrics.** Limited learning-based metrics exist for image captioning models; however multiple learning-based metrics exist in the field of evaluation of text generation [44, 54, 57, 68–70], including RUSE [57] and COMET [51]. COMET is a metric trained using human judgments that has demonstrated robust performance in evaluating machine translations. COMET comprises both an estimator model that directly predicts human judgments and a ranking model that predicts the quality order of the generated translations. 

By contrast, a few learning-based metrics exist that were specifically designed for image captioning [32, 35, 36]. One such metric is UMIC [35], which is among the few learningbased metrics tailored for evaluating image captioning models. UMIC is a fine-tuned UNITER [14] model designed to rank captions against each other using CapEval1K. However, as we argue later, such _ranking_ models have shortcomings when they process multiple potential references, such as varying focal points in captions and subjective variations in expression. Furthermore, UMIC is only a straightforward fine-tuned UNITER model and it cannot adequately handle diverse images and text, such as those associated with an open-vocabulary setting. 

**Datasets and benchmarks.** Standard datasets commonly used for the evaluation of image captioning include Flickr8K-Expert, Flickr8K-CF [27], Composite [1] and PASCAL-50S [62]. The Flickr8K-Expert and Flickr8KCF datasets comprise a significant amount of human judgments on captions provided by humans. However, these datasets do not contain any captions generated by models, which presents an issue from the perspective of the domain gap when using them for training metrics. The Composite dataset [1] encompasses 12K human judgments across images collected from MSCOCO [41], Flickr8k [27], and Flickr30k [71]. Although each image initially contains five references, only one reference was selected for human judgments within the dataset. The CapEval1k dataset was introduced by [35] for training automatic evaluation metrics. We note that the CapEval1K dataset has several limitations: it is a closed dataset, uses outdated models such as [11, 28, 52], and includes only 1K human judgments. In stark contrast, our Polaris dataset offers significant advantages: it is openly accessible, incorporates inferences from modern models [17, 37, 38, 60, 64, 65, 67, 73], and includes a substantial 131K human judgments. A meta-analysis of these datasets can be found in Appendix B.1. 

Standard datasets for image captioning include MSCOCO, nocaps [7], Flickr30K, and CC3M [55]. The nocaps dataset contains a greater diversity of classes than MSCOCO, which enables a more comprehensive evaluation of image captioning models’ ability to generate diverse captions. Our Polaris dataset is built on inferences from MSCOCO and nocaps to ensure caption diversity. 

## **3. Methodology** 

## **3.1. Meta-Analysis** 

When designing automatic evaluation metrics, the decision to use an unsupervised or supervised approach is important. As detailed in previous research [72], this binary distinction can be further decomposed into four categories: _regression_ , _ranking_ , _matching_ , and _generation_ . We will discuss the above categories in the following subsections. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0003-06.png)


**----- Start of picture text -----**<br>
Aggregation<br>Feed-Forward Network<br>Concat & MLP.<br>CLIP CLIP RoBERTa<br>Image Encoder Text Encoder Pretrained with<br>SimCSE<br>**----- End of picture text -----**<br>


Figure 2. Overview of the proposed metric. In alignment with the principles of M[2] LHF, Polos computes the evaluation _y_ ˆ based on multimodal inputs and regresses the human evaluation. The proposed metric extracts effective features for caption evaluation using the difference and Hadamard product of features derived from both CLIP and RoBERTa. 

**Unsupervised approaches.** In the domain of unsupervised approaches, metrics such as BERTScore [75], CLIPScore [26], and BARTScore [72] are prominent. However, each of these metrics has its limitations for evaluating image captioning models. For instance, _matching_ models such as BERTScore measure the similarity between tokens, which makes them ill-suited to explicitly handling images. Simultaneously, CLIPScore, which computes the similarity between sentence embeddings and image embeddings, has a major drawback. Because CLIP was designed to match an entire image to a text description, it has shortcomings regarding capturing the fine-grained alignment between specific image regions and text spans as indicated in [23, 77]. This limitation suggests that there is potential to improve performance, particularly by transitioning to a supervised approach. Similarly, unsupervised _generation_ models such as BARTScore encounter challenges in the evaluation of image captioning models. This difficulty stems from the limited availability of large-scale, pretrained, and lightweight multimodal encoders. 

**Supervised approaches.** Based on the above discussion, we believe that supervised metrics have distinct advantages. Both _regression_ and _ranking_ models are viable options; 

however, we believe that _regression_ models are better suited for evaluating image captioning models that involve multiple references because _ranking_ models cannot adequately handle multiple potential references, such as varying areas of interest in captions and inherent subjectivity in expression. For instance, the comparison of captions with different focal points lacks substantive meaning and ranking subjective expressions is difficult. Although COMET [51] introduces both _ranking_ and _regression_ models, the task of evaluating machine translation with limited interpretation differs significantly from our task. This distinction is a factor that prevents the straightforward adoption of a _ranking_ model. Given these considerations, in this study, we propose a metric based on _regression_ . 

## **3.2. Metric Design** 

**Reference-free vs reference-with-image.** In several studies, researchers have introduced reference-free metrics for image captioning [26, 35, 53]. However, these metrics have not achieved performance comparable with metrics that consider references. Additionally, the performance of reference-free metrics heavily relies on the alignment capabilities between image and language features. This is evident from the fact that, on the FOIL dataset, CLIP-S underperforms compared with traditional metrics such as CIDEr [26]. Such results suggest potential limitations in handling _hallucinations_ , which poses a significant challenge in the field of image captioning. Given these considerations, we chose not to adopt the reference-free problem setting. 

**Limitations of CLIP text embeddings.** As previously highlighted, many data-driven metrics do not have adequate generalization capabilities across diverse image and text types. This inadequacy partly stems from the sole use of text embeddings from the CLIP text encoder. CLIP provides versatile textual and visual features, as demonstrated by their performance in various tasks. However, the CLIP model, pretrained on web-collected image-caption pairs, is likely to be suboptimal for evaluation metrics because these annotations typically lack the richness and descriptiveness necessary for evaluating generated long captions, as indicated in [53]. Consequently, we posit that sentence embeddings pretrained with supervised SimCSE [22] may be a more advantageous approach than CLIP. This is partially supported by the observation that SimCSE outperformed previous sentence embedding techniques in the semantic text similarity task [2–6, 13, 46]. 

## **3.3. Proposed Method: Polos and M[2] LHF** 

We propose Polos, a supervised automatic evaluation metric tailored for image captioning models. Fig.2 shows the overview of the proposed Polos. To enhance robustness and practicality, we also present Multimodal Metric Learning from Human Feedback (M[2] LHF), a novel framework 

for developing metrics based on human feedback. Within M[2] LHF, a metric computes the evaluation _y_ ˆ based on multimodal input _**x**_ and directly regresses the human evaluation _y_ . Our method is inspired by automatic evaluation metrics for machine translation, such as COMET and BLEURT [54], which were explicitly designed to predict human judgments. However, our framework differs from previous works [51, 54, 57] in that it handles both images and text, and learns directly from human judgments based on multimodal inputs. We consider that our approach, which applies regression using both image and language features, is considered to be broadly applicable to automatic evaluation metrics that have learnable parameters. 

We define the input _**x**_ to the model as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0004-08.png)


where _**x**_ cand _∈ {_ 1 _,_ 0 _}[V][ ×][L]_ represents the candidate, _{_ _**x**_[(] ref _[i]_[)] _[}][∈{]_[1] _[,]_[ 0] _[}][N][×][V][ ×][L]_[represents][the] _[i]_[-th][reference][out] of _N_ , and _**x**_ img _∈_ R[3] _[×][H][×][W]_ represents the image. Here, _V, L, N, H_ and _W_ denote vocabulary size, maximum token length, number of reference sentences in one sample, and height and width of the image, respectively. 

As previously emphasized, selecting an appropriate method for sentence embedding necessitates careful consideration. In this study, we use both the CLIP text encoder and RoBERTa trained with supervised SimCSE to obtain sentence embeddings. Initially, using RoBERTa pretrained with supervised SimCSE, we extract sentence embedding _**c**_ rb _∈_ R _[L][×][d][R]_ and _{_ _**r**_ rb[(] _[i]_[)] _[}] i[N]_ =1 _[∈]_[R] _[N][×][L][×][d][R]_[from] _**[ x]**_[cand][and] _{_ _**x**_[(] ref _[i]_[)] _[}] i[N]_ =1[,][respectively.][Note][that] _[d][R]_[represents][the][out-] put dimension of RoBERTa and the sentence embeddings are derived from the [CLS] token of inputs into RoBERTa. Following this, using the CLIP text encoder, we derive language feature _**c**_ clip _∈_ R _[d]_[CLIP] and _**r**_ clip[(] _[i]_[)] _[∈]_[R] _[d]_[CLIP][from] _**x**_ cand and _**x**_[(] ref _[i]_[)][, respectively, where] _[ d]_[CLIP][denotes the out-] put dimension of the CLIP encoders. Additionally, utilizing the pretrained CLIP image encoder (ViT-B/16), we obtain image features _**v** ∈_ R _[d]_[CLIP] from _**x**_ img. 

## **3.3.1 Parallel Feature Extraction Mechanism** 

In alignment with the principles of M[2] LHF, we also propose a _parallel feature extraction mechanism_ . This mechanism serves as a multimodal extension of the RUSE method[51, 57], employing both difference and Hadamard products. It extracts effective features for caption evaluation by utilizing the difference and Hadamard product of features derived from both CLIP and RoBERTa. Given that CLIP is designed to minimize the cosine similarity between corresponding language and image features, the Hadamard product applied to CLIP features is considered to be effective. Additionally, the difference and Hadamard product 

operations generate vectors that encapsulate similarity because each element of the vector can be amplified or attenuated in relation to the others. 

Initially, given the following inputs: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0005-02.png)


the proposed framework first computes _**h**_[(] inter _[i]_[)][as:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0005-04.png)


Here, _F_ is the following function: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0005-06.png)


where _⊙_ denotes the Hadamard product. 

Subsequently, we apply the MLP to _**h**_ inter[(] _[i]_[)][to][compute] _**h**_[(] _[i]_[)] , which effectively captures the similarity among the multimodal features for the _i_ -th reference. Note that we chose MLP because pilot experiments demonstrated its superior performance compared with Transformer [61]. 

Finally, we compute the evaluation score _y_ ˆ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0005-10.png)


where _σ_ denotes the sigmoid function, which scales the output to the range [0 _,_ 1]. In this context, Aggregate denotes an aggregation function. This function can encompass various operations, such as calculating the maximum or average value. Notably, in our experimental setup, we opted for the max function. 

For the loss function, we adopted the mean squared error, which is a standard choice in regression problems because of its effectiveness in quantifying the variance between predicted and human judgments. Our implementation details can be found in Appendix D. 

## **4. Experimental Evaluation** 

## **4.1. Setups** 

**Polaris dataset.** In this study, we introduce the Polaris dataset, which consists of image-caption pairs and human judgments on the appropriateness of the captions. Training supervised models to predict human judgments benefits significantly from a large-scale corpus that contains diverse captions. However, to the best of our knowledge, there are few open datasets with diverse captions. Therefore, we constructed the Polaris dataset, which contains a total of 131,020 human judgments collected from 550 evaluators. 

In comparison with existing datasets, which comprise the inference results of image captioning models and are suitable for metric training, Polaris is distinguished by its exceptional diversity of captions and broader spectrum of 

evaluations. Our dataset, which comprises over 131K human judgments, significantly surpasses the largest dataset suitable for metric training, which contains only around 12K judgments [1]. Regarding the diversity of evaluations, the Polaris dataset had an average of eight evaluators per caption, which provides a more comprehensive judgment than the Flickr8K dataset, which had an average of only three evaluators per caption. Even the CapEval1K dataset, one of the largest in terms of evaluator participation, only involved five evaluators, which underscores the extensive nature of Polaris with its inclusion of 550 evaluators. Moreover, Polaris differentiates itself from existing datasets such as [1, 10, 27, 35] by being openly accessible, constructed with modern models, and incorporating a wide range of diverse captions. 

The Polaris dataset comprises captions generated by the ten standard models: SAT [67], _M_[2] -Transformer [17], VinVL [73], GRIT [60], BLIPbase, BLIPlarge [37], GIT [64], OFA [65], BLIP-2flan, and BLIP-2opt [38]. Within this set, BLIPbase and BLIPlarge refer to versions of BLIP that employ ViT-B and ViT-L as their image encoders. Similarly, BLIP-2flan and BLIP-2opt denote versions of BLIP2 that adopt Flan-T5 [15] and OPT [74] as their Large Language Models (LLMs), respectively. We selected these models because they are standard image captioning models. Additionally, we also chose older models to ensure diversity in the quality of their output sentences. We included inference results for each model, as performed on the MSCOCO [41] and nocaps [7] datasets in the Polaris dataset. We selected MS-COCO because it is the standard dataset for image captioning, whereas we chose nocaps because of its greater diversity of classes compared with MS-COCO. 

For a given image, human evaluators assessed the appropriateness of its caption using a five-point scale, taking into account factors such as fluency, relevance, and descriptiveness. We used a crowdsourcing service to collect these evaluations. In the Polaris dataset, we transformed the human judgments, which were rated on a five-point scale, to values in the range [0 _,_ 1] using min-max normalization. To eliminate unreliable data, we excluded data from evaluators who exhibited suspicious behavior, such as extremely short response times or consistently providing identical values. The statistical information and details of the Polaris dataset can be found in Appendix B. 

**Baseline metrics.** We adopted BLEU [49], ROUGE [40], METEOR [12], CIDEr [62] and SPICE [10] because they are standard metrics for image captioning tasks. Additionally, we included MoverScore [76], BERTScore [75], BARTScore [72], ViLBERTScore [34], TIGEr [29], LEIC [18], CLIPScore [26], MID [31], UMIC [35] and PACS [53] as baseline metrics because they are representative metrics for image captioning. 

|Composite<br>Flickr8K<br>(Expert)<br>Flickr8K<br>(CF)<br>Polaris|Composite<br>Flickr8K<br>(Expert)<br>Flickr8K<br>(CF)<br>Polaris|
|---|---|
|Classic metrics||
|BLEU [49]<br>30.6<br>ROUGE [40]<br>32.4<br>CIDEr [62]<br>37.7<br>METEOR [12]<br>38.9<br>SPICE [10]<br>40.3<br>SPARCS[20]<br>43.1|30.8<br>16.4<br>46.3<br>32.3<br>19.9<br>46.3<br>43.9<br>24.6<br>52.1<br>41.8<br>22.2<br>51.2<br>44.9<br>24.4<br>51.0<br>48.1<br>10.4<br>43.3|
|Similarity-based metrics||
|MoverScore [76]<br>30.1<br>BERTScore [75]<br>30.1<br>BARTScore [72]<br>43.5<br>LEIC [18]<br>–<br>TIGEr [29]<br>45.4<br>ViLBERTScore [34]<br>52.4<br>CLIP-S [26]<br>53.8<br>RefCLIP-S [26]<br>55.4<br>MID[31]<br>55.7|46.7<br>22.8<br>46.4<br>46.7<br>22.8<br>51.6<br>37.8<br>24.3<br>47.3<br>–<br>29.5<br>–<br>–<br>–<br>–<br>50.1<br>–<br>–<br>51.2<br>34.4<br>52.3<br>53.0<br>36.4<br>52.3<br>54.9<br>37.3<br>51.3|
|Learning-based metrics||
|PAC-S [53]<br>55.7<br>UMIC [35]<br>56.1<br>RefPAC-S[53]<br>57.3|54.3<br>36.0<br>52.5<br>46.8<br>30.1<br>49.8<br>55.9<br>37.6<br>56.0|
|**57.6**<br>**Pl (O)**|**56.4**<br>**37.8**<br>**57.8**|
|**oos urs**<br>(+0.3)<br>(+0.5)<br>(+0.2)<br>(+1.8)||



Table 1. Correlation coefficients between various metrics and human judgments. The symbol ‘–’ indicates non-executable code or unavailable data. **Bold** font indicates the highest recorded value and underlining indicates the second-highest value. 

**Benchmarks.** To assess the practicality of a supervised metric, it is essential to evaluate the metric using both indomain and out-of-domain datasets. Particularly in the context of supervised automatic evaluation, cases exist in which supervised metrics seemingly outperform unsupervised metrics on test sets (in-domain). However, as we demonstrate in the following section, this does not inherently imply better performance on out-of-domain data. Given that supervised metrics are frequently applied to outof-domain data, those that lack robustness can be impractical. Therefore, evaluating zero-shot performance in supervised models is paramount. In this study, in addition to the Polaris dataset, we used Composite, Flickr8K, PASCAL50S and FOIL to evaluate zero-shot performance. 

## **4.2. Correlation with Human Judgments** 

## **4.2.1 Caption-level Likert judgments** 

We conducted comparative experiments with baselines across Composite, Flickr8K-Expert, Flickr8K-CF, and the Polaris dataset. The results are presented in Table 1, which details the comparison of Kendall’s _τ_ with the baselines across the aforementioned datasets. Following the methodology of previous research [10, 26, 31], we used _τb_ (Kendall-B) for the Flickr8K-CF dataset and _τc_ (KendallC) for all other datasets. Notably, our proposed metric 

achieved SOTA results with scores of 57.6, 56.4, 37.8, and 57.8 for Composite, Flickr8K-Expert, Flickr8K-CF and the Polaris dataset, respectively. Specifically, our metric outperformed RefPAC-S by margins of 0.3, 0.5, 0.2, and 1.8 points on Composite, Flickr8K-Expert, Flickr8K-CF, and the Polaris dataset, respectively. This indicates that our supervised metric is superior to other supervised metrics such as RefPAC-S. Moreover, we achieved better performance than reference-free metrics such as PAC-S and intermediary metrics (metrics positioned between reference-with-image and reference-free metrics) such as MID. This highlights the critical role of incorporating references. 

Fig.3 shows various examples of the proposed metric for the Polaris dataset (further results are provided in Appendix E). First, Fig.3 (a) shows the sample that human evaluators deemed high quality. In this sample, _**x**_ cand captured the image content suitably, which resulted in a high human judgment of 0.750. CLIP-S assessed it at 0.402, whereas our proposed method aligned more closely with the human evaluation with a score of 0.745. Fig.3 (b) illustrates samples rated as mediocre by human evaluators because of partial accuracy. For instance, the veracity of the phrase “hanging on a tree” could not be confirmed by the image content, which led to a moderate human score of 0.450. RefPACS and RefCLIP-S overestimated the caption with scores of 0.825 and 0.746, respectively, whereas our proposed metric provided a more judicious score of 0.513, thereby reflecting its effectiveness in recognizing “normal” quality captions. Fig.3 (c) shows examples labeled as poor by human evaluators. A discrepancy, such as misidentifying the animal in the reference, resulted in a low human score of 0.071. Again, RefPAC-S and PAC-S assessments overestimated the caption, with scores at 0.903 and 0.856, respectively, whereas our method assigned a more realistic score of 0.173, effectively capturing the _hallucination_ error. These results show that our proposed metric appropriately handled multimodalities and produced evaluation scores close to human judgments. 

Fig.3 (d) illustrates a sample for which the proposed metric did not perform as expected. In Fig.3 (d), _**x**_[(1)] ref[and] _**[ x]**_[(2)] ref were described as “A cardboard box with a brown teddy bear and items in it outside on a curb” and “a box full of stuffed animals and other children’s items,” respectively. Conversely, _**x**_ cand was described as “a brown teddy bear sitting in a cardboard box.” For this sample, the average of human judgments was _**x**_ cand as 0.50. We believe that this was because the box contained various items in addition to the teddy bear. By contrast, our proposed metric evaluated this sample with a score of 0.790, which indicates a notable disparity from human judgment. Similarly, RefPAC-S and RefCLIP-S output scores of 0.847 and 0.751, respectively, which also demonstrates a discrepancy between them and human judgment. Given the failure of our metric and the 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0007-00.png)


**----- Start of picture text -----**<br>
White plate full offries next to a Human 0.750 Polos  0.745 (Ours) A framed pictureof a bunch of Human 0.450 Polos  0.513 (Ours)<br>sandwich. bananas.<br>(a) PAC-S RefPAC-S (b) PAC-S RefPAC-S<br>a close-up of a  0.612  0.670 a bunch of  0.756  0.825<br>sandwich and bananas are<br>some fries. CLIP-S RefCLIP-S hanging on a CLIP-S RefCLIP-S<br> 0.402  0.703 tree .  0.562  0.746<br>A cute little dog Human Polos (Ours) A cardboard box Human Polos (Ours)<br>biting on something  0.071 0.173 with a brown teddy  0.500 0.790<br>(c) a person is holding. PAC-S RefPAC-S (d) bear and items in itoutside on a curb PAC-S RefPAC-S<br>A black and  0.856  0.903  0.812  0.847<br>white cat a brown teddy<br>sleeping on a CLIP-S RefCLIP-S bear sitting in a CLIP-S RefCLIP-S<br>blue blanket .  0.200  0.343 cardboard box.  0.429  0.751<br>**----- End of picture text -----**<br>


Figure 3. Examples of successful and failed cases from the Polaris dataset. Values in **blue** indicate critical errors, and values in **red** represent those closest to the human judgments. Underlined words indicate significant inaccuracies in the candidate captions. These results demonstrate that our proposed metric effectively handled multimodal inputs and yielded evaluation scores that aligned closely with human judgment. Note that _**x**_[(1)] ref[and] _**[ x]**_[cand][denote one of the reference captions and the candidate caption, respectively.] 

|HC|HI|HM<br>MM<br>Mean|
|---|---|---|
|Classic metrics|||
|BLEU [49]<br>60.4<br>METEOR [12]<br>63.8<br>ROUGE [40]<br>63.7<br>SPICE [10]<br>63.6<br>CIDEr[62]<br>65.1|90.6<br>97.7<br>95.3<br>96.3<br>98.1|84.9<br>54.7<br>72.7<br>93.7<br>65.4<br>80.2<br>92.3<br>61.2<br>78.1<br>86.7<br>68.3<br>78.7<br>90.5<br>64.8<br>79.6|
|Similarity-based metrics|||
|ViLBERTS34<br>499|996|931<br>758<br>796|
|core []<br>.<br>BERTScore [75]<br>65.4<br>MoverScore [76]<br>65.1<br>TIGEr [29]<br>56.0<br>CLIP-S [26]<br>56.5<br>RefCLIP-S [26]<br>64.5<br>MID[31]<br>67.0|.<br>98.1<br>97.1<br>**99.8**<br>99.3<br>99.6<br>99.7|.<br>.<br>.<br>96.4<br>60.3<br>80.1<br>93.2<br>65.6<br>80.3<br>92.8<br>74.2<br>80.7<br>96.4<br>70.4<br>80.7<br>95.4<br>72.8<br>83.1<br>97.4<br>76.8<br>85.2|
|Learning-based metrics|||
|PAC-S [53]<br>60.6<br>RefPAC-S [53]<br>67.7<br>UMIC[35]<br>66.1|99.3<br>99.6<br>**99.8**|96.9<br>72.9<br>82.4<br>96.0<br>75.6<br>84.7<br>**98.1**<br>76.2<br>85.1|
|**70.0**<br>|99.6|97.4<br>**79.0**<br>**86.5**|
|**Polos (Ours)**<br>(+3.0)||(+1.2)<br>(+1.3)|



Table 2. Pascal50-S accuracy results (five references). 

CLIPScore family, the primary cause of these failures is likely to be an overemphasis on objects that are prominently visible, which results in overlooking the broader context of the image. This is likely to be caused by CLIP’s shortcomings in capturing the fine-grained alignment between specific image regions and text, as indicated in [23, 77]. 

## **4.2.2 Pairwise ranking on Pascal-50S** 

Pascal-50S [62] introduced an alternative evaluation framework for accuracy, which comprised 4K pairwise preference judgments between two captions. These judgments encompass four distinct scenarios: pairs of HC (human correct) captions, HI pairs (both human-written, with one in- 

correct), HM pairs (one from a human and the other generated by a machine), and MM pairs (both generated by machines). The caption that received the majority vote was deemed the preferred choice, with ties resolved randomly. In our study, following the procedure in [26], we calculated the average over five evaluations randomly selected from 48 candidates. Table 2 shows the accuracy results for PASCAL-50S. In our experiments, we achieved SOTA results with accuracies of 70.0%, 79.0%, and 86.5% for HC, MM, and Mean, respectively. The results in Tables 1 and 2 demonstrate that our proposed metric outperformed the baselines in terms of zero-shot performance. These results provide strong evidence that our metric is both robust and practical, thereby serving as a potent automatic evaluation metric for image captioning. 

## **4.2.3 Sensitivity to hallucination** 

|Previous studies [26, 31]<br>measured how evaluation|||1-ref|4-ref||
|---|---|---|---|---|---|
|metrics handle _hallucina-_<br>_tions_ in captions using||BLEU [49]<br>ROUGE [40]<br>METEOR [12]|66.5<br>71.7<br>78.8|82.6<br>79.3<br>82.6||
|the FOIL (Find One mis-<br>match between Image and<br>Language caption) dataset<br>[56]. Following the pro-<br>cedure used in [26], we<br>evaluated 32K test im-<br>ages with either one or||CIDEr [62]<br>SPICE[10]<br>BARTScore [72]<br>MoverScore [76]<br>BERTScore[75]<br>CLIP-S [26]<br>MID[31]<br>PAC-S [53]|82.5<br>75.5<br>85.3<br>88.4<br>88.6<br>87.2<br>90.5<br>89.9|90.6<br>86.1<br>91.1<br>88.4<br>92.1<br>87.2<br>90.5<br>89.9||
|four references.<br>Subse-<br>quently, we computed the||RefCLIP-S [26]<br>RefPAC-S[53]|91.0<br>**93.7**|92.6<br>94.9||
|accuracy of various met-||**Polos (Ours)**|93.3|**95.4**||
|rics to evaluate their ef-|Table 3. FOIL hallucination pair-|||||
|fectiveness in consistently|wise detection accuracy results.|||||
|awarding higher scores to||||||
|the true candidate compared with the FOIL dataset.||||||



Table 3 presents the accuracy results for the FOIL dataset. Our proposed method outperformed previous metrics, achieving SOTA results in the 4-ref setting. Specifically, it achieved an accuracy of 93.3% in the 1-ref setting and 95.4% in the 4-ref setting. As previously mentioned, CLIP-S lags behind the traditional metric CIDEr, whereas our method outperformed it in both the 1-ref and 4-ref settings. Moreover, compared with reference-with-image metric RefPAC-S as well as the intermediary metric MID, our approach led by 0.6 and 4.9 points in the 4-ref setting. 

## **4.3. Ablation Study** 

We conducted three ablation studies to demonstrate the effectiveness of our proposed method. Table 4 presents the results of the ablation studies. 

**Parallel feature extraction ablation.** We investigated the performance of our _parallel feature extraction mechanism_ by excluding the Hadamard product and difference, specifically by modifying _F_ ( **c** _,_ **r** ) (eq.4) to the function _F[′]_ ( **c** _,_ **r** ) = [ **c** ; **r** ]. The correlation coefficients between Metric (i) and human judgments were lower by 18.3, 15.4, and 6.4 points compared to Metric (vi) for the Composite, Flickr8K, and Polaris, respectively. These indicate that our _parallel feature extraction mechanism_ provided superior performance. 

**M[2] LHF ablation.** We investigated the performance of M[2] LHF by modifying our model to make predictions based solely on text, excluding any of the following: _**x**_ img, ( _**c**_ clip _, {_ _**r**_ clip[(] _[i]_[)] _[}] i[N]_ =1[)][,][or][(] _**[c]**_[rb] _[,][ {]_ _**[r]**_ rb[(] _[i]_[)] _[}] i[N]_ =1[)][.][Initially,][by][omit-] ting _**x**_ img, we assessed the significance of the image feature. Compared to Metric (vi), the correlation coefficients between Metric (ii) and human judgments were lower by 0.4, 1.4, and 0.7 points on Composite, Flickr8K, and Polaris respectively. These results suggest that the image feature played a pivotal role in enhancing the performance of our proposed metric. Subsequently, we evaluated the contribution of each module by excluding either CLIP or RoBERTa. Relative to Metric (vi), the correlation coefficients between Metric (iii) and human judgments decreased by 2.6, 3.2, and 2.4 points. Similarly, the exclusion of RoBERTa led to a decrease in performance. Although we observed that RoBERTa pretrained by SimCSE was found to enhance performance, these results highlighted CLIP as the most influential feature extractor. Overall, these results validated the efficacy of the image feature and each module, underscoring that the introduction of M[2] LHF contributed to the performance improvement of the proposed metric. 

**Aggregation mechanism ablation.** We investigated the impact on performance by setting the Aggregate function to either Max or Mean. The correlation coefficients between Metric (v) and human judgments were lower by 2.5, 1.0, and 5.7 points compared with Metric (vi) for the Composite, Flickr8K, and the Polaris dataset, respectively. 

|Metric|_P_|**_x_**img|CLIP|RoBERTa|Aggregate|Composite|Flickr8K|Polaris|
|---|---|---|---|---|---|---|---|---|
|(i)||✓|✓|✓|Max|39.3|41.0|51.4|
|(ii)|✓||✓|✓|Max|56.8|55.5|57.1|
|(iii)|✓|||✓|Max|55.0|53.2|55.4|
|(iv)|✓|✓|✓||Max|56.0|55.0|57.4|
|(v)|✓|✓|✓|✓|Mean|55.1|55.4|52.1|
|(vi)|✓|✓|✓|✓|Max|**57.6**|**56.4**|**57.8**|



Table 4. Ablation study results: performance of M[2] LHF and the _parallel feature extraction_ and comparison of the effects of using either the Max or Mean within the Aggregate function. ‘ _P_ ’ indicates the use of the _parallel feature extraction mechanism_ 

These results indicate that using the Max function as the Aggregate function provided superior performance. 

## **4.4. Discussion and Limitations** 

Although our metric generated compelling results, our approach has some limitations. As shown in Fig.3, our metric demonstrated a tendency to overestimate captions that lacked intricate details. This phenomenon is likely to be attributed to its excessive focus on prominently visible objects; hence it consequently overlooked the broader context within the image. As discussed in Section 4.2.1, this limitation may be attributed to CLIP’s inherent limitations in capturing the fine-grained alignment between specific image regions and their corresponding textual descriptions. Despite this, we firmly believe that this study represents a significant stride toward the development of a more practical metric for image captioning models. In future work, we plan to extend our metric by enhancing the fine-grained alignment, drawing inspiration from methods such as RegionCLIP [77]. 

## **5. Conclusion** 

In this paper, we introduced Polos, an automatic evaluation metric for image captioning. The contribution of this paper is fourfold: i) the introduction of M[2] LHF, a novel framework used to develop a practical metric for image captioning; ii) the introduction of a _parallel feature extraction mechanism_ that leverages CLIP and RoBERTa pretrained with SimCSE; iii) the construction of the Polaris dataset, containing a total of 131,020 human judgments collected from 550 evaluators; and iv) achieving SOTA performance on image captioning benchmarks including Composite, Flickr8K-Expert, and Flickr8K-CF, PASCAL-50S, FOIL, and the Polaris dataset. 

## **Acknowledgments** 

This work was supported by a grant from Apple, Inc. Any views, opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and should not be interpreted as reflecting the views, policies or position, either expressed or implied, of Apple, Inc. This work was also partially supported by JSPS KAKENHI Grant Number 23H03478, JST CREST, and NEDO. 

## **References** 

- [1] Somak Aditya, Yezhou Yang, Chitta Baral, Cornelia Fermuller, et al. From Images to Sentences through Scene Description Graphs using Commonsense Reasoning and Knowledge. _arXiv preprint arXiv:1511.03292_ , 2015. 2, 3, 5 

- [2] Eneko Agirre, Carmen Banea, Claire Cardie, Daniel Cer, and Mona Diab. SemEval-2014 Task 10: Multilingual Semantic Textual Similarity. In _SemEval_ , pages 81–91, 2014. 4 

- [3] Eneko Agirre, Carmen Banea, et al. SemEval-2015 Task 2: Semantic Textual Similarity, English, Spanish and Pilot on Interpretability. In _SemEval_ , pages 252–263, 2015. 

- [4] Eneko Agirre, Carmen Banea, et al. SemEval-2016 Task 1: Semantic Textual Similarity, Monolingual and CrossLingual Evaluation. In _SemEval_ , pages 497–511, 2016. 

- [5] Eneko Agirre, Daniel Cer, Mona Diab, and Aitor Agirre. SemEval-2012 Task 6: A Pilot on Semantic Textual Similarity. In _SemEval_ , pages 385–393, 2012. 

- [6] Eneko Agirre, Daniel Cer, Mona Diab, Aitor Agirre, and Weiwei Guo. SEM 2013 Shared Task: Semantic Textual Similarity. In _SEM_ , pages 32–43, 2013. 4 

- [7] Harsh Agrawal, Karan Desai, et al. nocaps: Novel Object Captioning at Scale. In _ICCV_ , pages 8948–8957, 2019. 3, 5 

- [8] Hiba Ahsan, Daivat Bhatt, Kaivan Shah, and Nikita Bhalla. Multi-Modal Image Captioning for the Visually Impaired. In _NAACL-HLT_ , pages 53–60, 2021. 1 

- [9] Jean Alayrac, Jeff Donahue, Pauline Luc, et al. Flamingo: A Visual Language Model for Few-shot Learning. In _NeurIPS_ , volume 35, pages 23716–23736, 2022. 1 

- [10] Peter Anderson, Basura Fernando, Mark Johnson, et al. SPICE: Semantic Propositional Image Caption Evaluation. In _ECCV_ , pages 382–398, 2016. 1, 2, 5, 6, 7 

- [11] Peter Anderson, Xiaodong He, et al. Bottom-Up and TopDown Attention for Image Captioning and Visual Question Answering. In _CVPR_ , pages 6077–6086, 2018. 3 

- [12] Satanjeev Banerjee et al. METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments. In _ACL_ , pages 65–72, 2005. 1, 2, 5, 6, 7 

- [13] Daniel Cer, Mona Diab, et al. SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation. In _SemEval_ , pages 1–14, 2017. 4 

- [14] Yen Chen, Linjie Li, Licheng Yu, Ahmed Kholy, Faisal, Zhe Gan, Yu Cheng, et al. UNITER: Universal Image-text Representation Learning. In _ECCV_ , pages 104–120, 2020. 3 

- [15] Hyung Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, et al. Scaling Instruction-finetuned Language Models. _arXiv preprint arXiv:2210.11416_ , 2022. 5 

- [16] Alexis Conneau, Kartikay Khandelwal, et al. Unsupervised Cross-lingual Representation Learning at Scale. In _ACL_ , pages 8440–8451, 2020. 1 

- [17] Marcella Cornia, Matteo Stefanini, Lorenzo Baraldi, et al. Meshed-Memory Transformer for Image Captioning. In _CVPR_ , pages 10578–10587, 2020. 3, 5, 1, 2 

- [18] Yin Cui, Guandao Yang, Andreas Veit, Xun Huang, and Serge Belongie. Learning to Evaluate Image Captioning. In _CVPR_ , pages 5804–5812, 2018. 5, 6 

- [19] Jacob Devlin, Ming-Wei Chang, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In _NAACL-HLT_ , pages 4171–4186, 2019. 1 

- [20] Joshua Feinglass and Yezhou Yang. SMURF: SeMantic and 

   - linguistic UndeRstanding Fusion for Caption Evaluation via Typicality Analysis. In _IJCNLP_ , pages 2250–2260, 2021. 6 

- [21] Adam Fisch, Kenton Lee, Ming Chang, Jonathan Clark, and Regina Barzilay. CapWAP: Image Captioning with a Purpose. In _EMNLP_ , pages 8755–8768, 2020. 1 

- [22] Tianyu Gao, Xingcheng Yao, and Danqi Chen. SimCSE: Simple Contrastive Learning of Sentence Embeddings. In _EMNLP_ , pages 6894–6910, 2021. 2, 4 

- [23] Xiuye Gu, Tsung Lin, Weicheng Kuo, and Yin Cui. Open-vocabulary Object Detection via Vision and Language Knowledge Distillation. In _ICLR_ , 2022. 3, 7 

- [24] Danna Gurari, Yinan Zhao, Meng Zhang, and Nilavra Bhattacharya. Captioning Images Taken by People Who Are Blind. In _ECCV_ , pages 417–434, 2020. 1 

- [25] Simao Herdade, Armin Kappeler, Kofi Boakye, et al. Image Captioning: Transforming Objects into Words. In _NeurIPS_ , volume 32, pages 11137–11147, 2019. 1 

- [26] Jack Hessel, Ari Holtzman, et al. CLIPScore: A Referencefree Evaluation Metric for Image Captioning. In _EMNLP_ , pages 7514–7528, 2021. 1, 2, 3, 4, 5, 6, 7 

- [27] Micah Hodosh, Peter Young, and Julia Hockenmaier. Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics. _JAIR_ , 47:853–899, 2013. 1, 2, 3, 5 

- [28] Lun Huang, Wenmin Wang, Jie Chen, and Xiao Wei. Attention on Attention for Image Captioning. In _ICCV_ , pages 4634–4643, 2019. 3 

- [29] Ming Jiang, Qiuyuan Huang, Lei Zhang, Xin Wang, et al. TIGEr: Text-to-image Grounding For Image Caption Evaluation. In _EMNLP_ , 2019. 2, 5, 6, 7 

- [30] Motonari Kambara et al. Case Relation Transformer: A Crossmodal Language Generation Model for Fetching Instructions. _IEEE RAL_ , 6:8371–8378, 2021. 1 

- [31] Jin Kim et al. Mutual Information Divergence: A Unified Metric for Multimodal Generative Models. In _NeurIPS_ , volume 35, pages 35072–35086, 2022. 1, 2, 5, 6, 7 

- [32] Yongil Kim, Yerin Hwang, Hyeongu Yun, Seunghyun Yoon, et al. PR-MCS: Perturbation Robust Metric for MultiLingual Image Captioning. In _EMNLP_ , pages 12237–12258, 2023. 3 

- [33] Matt Kusner, Yu Sun, Nicholas Kolkin, and Kilian Weinberger. From Word Embeddings To Document Distances. _PMLR_ , 37:957–966, 2015. 1 

- [34] Hwanhee Lee, Seunghyun Yoon, et al. ViLBERTScore: Evaluating Image Caption Using Vision-and-Language BERT. In _Eval4NLP_ , pages 34–39. 1, 2, 5, 6, 7 

- [35] Hwanhee Lee, Seunghyun Yoon, et al. UMIC: An Unreferenced Metric for Image Captioning via Contrastive Learning. In _ACL_ , pages 220–226, 2021. 1, 2, 3, 4, 5, 6, 7 

- [36] Tomer Levinboim, Ashish V. Thapliyal, Piyush Sharma, and Radu Soricut. Quality Estimation for Image Captions Based on Large-scale Human Evaluations. In _NAACL_ , pages 3157– 3166, 2021. 3 

- [37] Junnan Li et al. BLIP: Bootstrapping Language-image Pre-training for Unified Vision-language Understanding and Generation. In _ICML_ , pages 12888–12900, 2022. 3, 5, 2 

- [38] Junnan Li, Dongxu Li, et al. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. In _ICML_ , 2023. 3, 5, 1, 2 

- [39] Jingyu Li, Zhendong Mao, Shancheng Fang, et al. ER-SAN: Enhanced-Adaptive Relation Self-Attention Network for Image Captioning. In _IJCAI_ , pages 1081–1087, 2022. 1 

- [40] Chin Lin. ROUGE: A Package For Automatic Evaluation Of Summaries. In _ACL_ , pages 74–81, 2004. 1, 2, 5, 6, 7 

- [41] Tsung Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, et al. Microsoft COCO: Common Objects in Context. In _ECCV_ , pages 740–755, 2014. 3, 5 

- [42] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, et al. RoBERTa: A Robustly Optimized BERT Pretraining Approach. _arXiv preprint arXiv:1907.11692_ , 2019. 2 

- [43] Yunpeng Luo, Jiayi Ji, Xiaoshuai Sun, Liujuan Cao, et al. Dual-Level Collaborative Transformer for Image Captioning. In _AAAI_ , volume 35, pages 2286–2293, 2021. 1 

- [44] Mounica Maddela, Yao Dou, David Heineman, and Wei Xu. LENS: A learnable evaluation metric for text simplification. In _ACL_ , pages 16383–16408, July 2023. 2 

- [45] Aly Magassouba, Komei Sugiura, and Hisashi Kawai. Multimodal Attention Branch Network for Perspective-Free Sentence Generation. In _CoRL_ , pages 76–85, 2019. 1 

- [46] Marco Marelli, Stefano Menini, Marco Baroni, et al. A SICK Cure for the Evaluation of Compositional Distributional Semantic Models. In _LREC_ , pages 216–223, 2014. 4 

- [47] Yue Ming, Nannan Hu, Chunxiao Fan, Jiangwan Zhou, and Hui Yu. Visuals to Text: A Comprehensive Review on Automatic Image Captioning. _JAS_ , 9(8):1339–1365, 2022. 1 

- [48] Gabriel Oliveira, Esther Colombini, and Sandra Avila. CIDEr-R: Robust Consensus-based Image Description Evaluation. In _W-NUT_ , pages 351–360, 2021. 2 

- [49] Kishore Papineni, Salim Roukos, Todd Ward, and Wei Zhu. BLEU: a Method for Automatic Evaluation of Machine Translation. In _ACL_ , pages 311–318, 2002. 1, 2, 5, 6, 7 

- [50] Alec Radford, Jong Wook Kim, Chris Hallacy, et al. Learning Transferable Visual Models from Natural Language Supervision. In _ICML_ , pages 8748–8763, 2021. 1, 2 

- [51] Ricardo Rei, Craig Stewart, Ana Farinha, and Alon Lavie. COMET: A Neural Framework for MT Evaluation. In _EMNLP_ , pages 2685–2702, 2020. 2, 4 

- [52] Steven Rennie, Etienne Marcheret, Youssef Mroueh, Jerret Ross, and Vaibhava Goel. Self-critical Sequence Training for Image Captioning. In _CVPR_ , pages 7008–7024, 2017. 3 

- [53] Sara Sarto et al. Positive-Augmented Contrastive Learning for Image and Video Captioning Evaluation. In _CVPR_ , pages 6914–6924, 2023. 1, 2, 4, 5, 6, 7 

- [54] Thibault Sellam, Dipanjan Das, and Ankur Parikh. BLEURT: Learning Robust Metrics for Text Generation. In _ACL_ , pages 7881–7892, 2020. 2, 4 

- [55] Piyush Sharma et al. Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset for Automatic Image Captioning. In _ACL_ , pages 2556–2565, 2018. 3 

- [56] Ravi Shekhar, Sandro Pezzelle, Yauhen Klimovich, et al. FOIL it! Find One Mismatch Between Image and Language caption. In _ACL_ , pages 255–265, 2017. 7 

- [57] Hiroki Shimanaka et al. RUSE: Regressor Using Sentence Embeddings for Automatic Machine Translation Evaluation. In _WMT18_ , pages 751–758, 2018. 2, 4 

- [58] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, et al. TextCaps: A Dataset for Image Captioning with Reading Comprehension. In _ECCV_ , pages 742–758, 2020. 1 

      - and Better Image Captioning Transformer Using Dual Visual Features. In _ECCV_ , pages 167–184, 2022. 3, 5, 1, 2 

   - [61] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, et al. Attention Is All You Need. In _NIPS_ , volume 30, pages 5998–6008, 2017. 5 

   - [62] Ramakrishna Vedantam, Lawrence Zitnick, and Devi Parikh. CIDEr: Consensus-based Image Description Evaluation. In _CVPR_ , pages 4566–4575, 2015. 1, 2, 3, 5, 6, 7 

   - [63] Yuiga Wada, Kanta Kaneda, et al. JaSPICE: Automatic Evaluation Metric Using Predicate-Argument Structures for Image Captioning Models. In _CoNLL_ , 2023. 2 

   - [64] Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, et al. GIT: A Generative Image-to-text Transformer for Vision and Language. _TMLR_ , 2022. 3, 5, 2 

   - [65] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, et al. OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-sequence Learning Framework. In _ICML_ , pages 23318–23340, 2022. 3, 5, 2 

   - [66] Julia White, Gabriel Poesia, Robert Hawkins, et al. Opendomain Clarification Question Generation Without Question Examples. In _EMNLP_ , pages 563–570, 2021. 1 

   - [67] Kelvin Xu, Jimmy Ba, Ryan Kiros, et al. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. In _ICML_ , pages 2048–2057, 2015. 3, 5, 1, 2 

   - [68] Wenda Xu, Xian Qian, Mingxuan Wang, et al. SESCORE2: Learning Text Generation Evaluation via Synthesizing Realistic Mistakes. In _ACL_ , pages 5166–5183, 2023. 2 

   - [69] Wenda Xu, Yi-Lin Tuan, Yujie Lu, et al. Not All Errors are Equal: Learning Text Generation Metrics using Stratified Error Synthesis. In _EMNLP_ , pages 6559–6574, 2022. 

   - [70] Wenda Xu, Danqing Wang, et al. INSTRUCTSCORE: Towards Explainable Text Generation Evaluation with Automatic Feedback. In _EMNLP_ , pages 5967–5994, 2023. 2 

   - [71] Peter Young et al. From Image Descriptions to Visual Denotations: New Similarity Metrics for Semantic Inference over Event Descriptions. _TACL_ , 2:67–78, 2014. 3 

   - [72] Weizhe Yuan, Graham Neubig, et al. BARTScore: Evaluating Generated Text as Text Generation. In _NeurIPS_ , volume 34, pages 27263–27277, 2021. 3, 5, 6, 7 

   - [73] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, et al. VinVL: Revisiting Visual Representations in Vision-language Models. In _CVPR_ , pages 5579–5588, 2021. 3, 5, 2 

   - [74] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, et al. OPT: Open Pre-trained Transformer Language Models. _arXiv preprint arXiv:2205.01068_ , 2022. 5 

   - [75] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Weinberger, and Yoav Artzi. BERTScore: Evaluating Text Generation with BERT. In _ICLR_ , 2020. 1, 2, 3, 5, 6, 7 

   - [76] Wei Zhao et al. MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance. In _EMNLP-IJCNLP_ , pages 563–578, 2019. 1, 2, 5, 6, 7 

   - [77] Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, et al. RegionCLIP: Region-based Language-image Pretraining. In _CVPR_ , pages 16793–16803, 2022. 3, 7, 8, 2 

- [59] Matteo Stefanini, Marcella Cornia, Lorenzo Baraldi, et al. From Show to Tell: A Survey on Deep Learning-based Image Captioning. _PAMI_ , 45(1):539–559, 2022. 1 

- [60] Masanori Suganuma, Takayuki Okatani, et al. GRIT: Faster 

## **Polos: Multimodal Metric Learning from Human Feedback for Image Captioning** Supplementary Material 

## **A. Additional Related Work** 

**Image captioning** Numerous studies have been conducted in the domain of image captioning [17, 25, 39, 43, 67]. For instance, GRIT [60] adeptly leverages both grid and region-based features for enhanced image captioning, thereby eliminating the need for conventional CNN-based detectors. BLIP-2 [38] is a novel pre-training approach that efficiently uses LLMs that outperforms models including Flamingo [9] with notably fewer trainable parameters. This field includes applications in various domains, such as aiding persons with vision impairment [8, 24, 58] and robotics [30, 45]. Survey papers such as [47, 59] offer a comprehensive overview of image caption generation, including models, standard datasets, and evaluation metrics. Specifically, they provide a comprehensive summary of various automatic evaluation metrics, including similarity-based and learning-based metrics [33, 75, 76]. 

## **B. Polaris Dataset** 

## **B.1. Meta-Analysis** 

As pointed out in [35], there are issues with utilizing existing datasets such as Flickr8K and Composite for training purposes. Fig.4 shows the score distributions of human judgments in Composite, Flickr8K-Expert, Flickr8K-CF, and our proposed Polaris dataset. For the Flickr8K dataset, the majority of scores fall below 0.4, as the candidate captions were sourced from a reference caption pool through an image retrieval system. Moreover, the Flickr8K datasets do not contain captions generated by models, which presents an issue from the perspective of the domain mismatch because our aim is to build an automatic metric for image captioning. Consequently, we argue that Flickr8K-Expert and Flickr8K-CF are not suitable for training metrics. Furthermore, the human judgments in Flickr8K-CF were provided using a binary scheme, that only allowed responses categorized as “yes” or “no.” This method is problematic because of its lack of granularity and its propensity to force evaluators into making overly simplistic judgments. For instance, a disparity exists in quality for captions that describe content and our method may not be able to adequately evaluate captions of varying quality. 

In the case of the Composite dataset, we note its exceptionally few human judgments. This paucity of data renders it inadequate for developing a practical metric. Moreover, as pointed out in [35], each sample’s score was determined by a single annotator, leading to potentially biased outputs. Upon manual examination of the captions, [35] also pointed out that these captions are often coarsely generated. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0011-07.png)


**----- Start of picture text -----**<br>
Benchmarks<br>40000<br>Polaris<br>35000 Composite<br>Flickr8K-CF<br>Flickr8K<br>30000<br>25000<br>20000<br>15000<br>10000<br>5000<br>0<br>0.0 0.2 0.4 0.6 0.8 1.0<br>**----- End of picture text -----**<br>


Figure 4. Score distributions of human judgments in **Composite** , **Flickr8K-Expert** , **Flickr8K-CF** , and our **Polaris** dataset. All scores were normalized from 0 to 1. Polaris distinguishes itself from other datasets by encompassing a vast collection of captions and integrating a broad spectrum of human judgments. 

## **B.2. Statistics and Details** 

The Polaris dataset includes 13,691 images accompanied by 131,020 generated captions. Additionally, it contains 262,040 references. All sentences are in English. To minimize biases in evaluations and achieve more balanced judgments compared with other datasets, we engaged multiple human evaluators to evaluate each caption. Specifically, each generated caption was evaluated by approximately eight different evaluators. The generated captions encompass a vocabulary of 3,154 unique words, with a total of 1,177,512 words. On average, each caption is composed of 8.99 words. By contrast, the reference captions have a vocabulary of 22,275 unique words, with a word count of 8,309,300. Each reference caption, on average, consists of 10.7 words. 

In the Polaris dataset, the training, validation, and test sets consist of 78,631, 26,269, and 26,123 samples, respectively. We used the training set to train the model, the validation set for hyperparameter tuning, and the test set to evaluate the performance of the model. 

## **B.3. Image Captioning Models** 

The Polaris dataset comprises captions generated by the following 10 standard models. Table 5 summarizes these image captioning models. We selected these models as they are standard image captioning models. Additionally, we also chose older models to ensure diversity in the quality of their output sentences. 

|Year|Venue|
|---|---|
|BLIP-2fan[38]|ICML’23|
|BLIP-2opt[38]|ICML’23|
|GRIT[60]|ECCV’22|
|OFA[65]|ICML’22|
|GIT[64]|TMLR’22|
|BLIPlarge[37]|ICML’22|
|BLIPbase[37]|ICML’22|
|VinVL[73]|CVPR’21|
|_M_2-Transformer[17]|CVPR’20|
|SAT[67]|CVPR’15|



Table 5. Image captioning models used for the Polaris dataset. 

## **C. Error Analysis** 

To investigate the limitations of the proposed metric, we analyzed 100 instances where the method did not perform as expected. We define failed cases as samples that satisfy ˆ the condition _|y − y| ≥ θ_ . In this study, we held _θ_ at a value of 0.25, corresponding to the step size when normalizing a five-level evaluation. Table 6 categorizes the failed cases. The causes of failure can be grouped into seven categories: 

- (i) Overestimation of captions lacking details (OCLD): This category pertains to instances where the proposed metric assigned higher scores to captions that lacked vital details, missing critical aspects of the images. 

- (ii) Overestimation of captions with incorrect details (OCID): This category refers to instances where the proposed metric inaccurately assigned higher scores to captions containing incorrect or misleading details. 

- (iii) Underestimation of captions where the focus areas differ from the references (UCFA): This category refers to instances where the proposed metric assigned lower scores to captions that, although accurate, focused on areas different from the references. 

- (iv) Serious Errors (SE): This category encompasses instances where the evaluation deviated greatly from human judgments, being much higher or lower. 

- (v) Overestimation of captions with grammatical inaccuracies (OSGI): This category refers to instances where the proposed metric erroneously assigned higher scores to captions that, while potentially accurate in content, contained grammatical errors. 

- (vi) Annotation errors (AE): This category pertains to instances where human evaluations proved to be inaccurate. These evaluations were either higher or lower than what could be reasonably expected. 

- (vii) Others: This category encompasses miscellaneous errors that do not fit into the aforementioned categories. 

From Table 6, it can be inferred that the main bottleneck was the overestimation of captions that lacks detail. As mentioned in Section 4.4, a possible solution could be to enhance the fine-grained alignment [77]. 

|Errors|Description|#Error|
|---|---|---|
|OCLD|Overestimation of captions lacking details|29|
|OCID|Overestimation of captions with incorrect|22|
||details||
|UCFA|Underestimation of captions where the fo-|15|
||cus area differs from the reference||
|SE|Serious errors (e.g., assigning a higher|11|
||score to captions with major mistakes)||
|OGI|Overestimation of captions with grammat-|11|
||ical inaccuracies||
|AE|Annotation errors in human judgments|9|
|Others|Miscellaneous or less common errors|3|
|Total|—|100|



Table 6. Categorization of failed samples. 

## **D. Implementation Details** 

Table 7 shows the experimental settings for the proposed method. We trained our model on a Tesla A100 GPU, and the training time was approximately 4.6 hours. To measure the inference time, we tested our metric on a system equipped with a GeForce RTX 3090 and an Intel Core i910900KF. The inference times per sample for SPICE [10], RefPAC-S [53], and Polos were 16 _._ 6 ms, 4 _._ 45 ms, and 6 _._ 91 ms, respectively. Notably, Polos operates at a speed 2 _._ 4 times faster than SPICE. Furthermore, in the 6.91 ms processing time of Polos, CLIP took 4.90 ms, RoBERTa 1.96 ms, and MLP just 0.054 ms. We used early stopping in our model to optimize for the highest Kendall’s _τ_ . At each epoch, we evaluated Kendall’s _τ_ on the validation set. If no improvement was observed over five consecutive epochs, we stopped training. Subsequently, we evaluated the model’s performance using the test set, referring to the epoch where the validation set achieved its best _τ_ value. 

Table 7. Experimental settings for the proposed method. 

|Batch size|64|
|---|---|
|Optimizer|Adam (_β_1 = 0_._9_, β_2 = 0_._98)|
|LearningRate|3_._0_×_10_−_5|



## **E. Additional Qualitative Results** 

Fig.5 and Fig.6 provide additional comparisons between the CLIPScore family and Polos. We observed that the CLIPScore family tends to overestimate scores. Specifically, RefCLIP-S and RefPAC-S may not effectively compare references and a candidate. Although CLIP-S could not show overestimation, this does not imply adequacy in caption evaluation. Rather, it may signal estimation deficiencies, especially for longer captions, stemming from poor alignment between words and images, as CLIP relies heavily on the alignment between image and language features. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0013-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0013-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0013-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0013-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0013-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0013-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0013-06.png)


**----- Start of picture text -----**<br>
x [(1)] [A crying]<br>ref [:]<br>CLIP-S RefCLIP-S<br>woman looking at 0.39 0.69<br>herself in a mirror. Polos Human<br>0.18 0.18<br>x cand: a woman holding PAC-S RefPAC-S<br>a cup with a candle in a 0.81 0.87<br>x [(1)] [a upside down boat]<br>ref [:] CLIP-S RefCLIP-S<br>is on top of a big hil 0.5 0.87<br>Polos Human<br>x cand: a couple of 0.64 0.6<br>boats that are sit- PAC-S RefPAC-S<br>ting in the snow 0.83 0.86<br>x [(1)] [Pink donut]<br>ref [:]<br>with white sprin- CLIP-S RefCLIP-S<br>kles on the top of it. 0.26 0.46 Polos Human<br>0.083 0.031<br>x cand: a blue um- PAC-S RefPAC-S<br>brella with a blue 0.7 0.82<br>and white design<br>x [(1)] [A white car]<br>ref [:] CLIP-S RefCLIP-S<br>is parked in the 0.49 0.86<br>street at night time. Polos Human<br>0.77 0.61<br>x cand: a blurry image PAC-S RefPAC-S<br>of a city street at night 0.89 0.88<br>x [(1)] [a bicycle with]<br>ref [:] CLIP-S RefCLIP-S<br>a basket and a blue<br>0.42 0.73<br>and pink umbrella Polos Human<br>0.35 0.34<br>PAC-S RefPAC-S<br>x cand: a person holding<br>an umbrella in a room 0.9 0.92<br>x [(1)] [A cat is jumping off] CLIP-S RefCLIP-S<br>ref [:]<br>of a stack of suitcases. 0.35 0.61<br>Polos Human<br>0.2 0.25<br>x cand: a blue suitcase is PAC-S RefPAC-S<br>sitting on a blue couch 0.87 0.92<br>**----- End of picture text -----**<br>


Figure 5. Additional examples from the Polaris dataset (the **blue** blocks indicate critical errors and the underlined words represent noteworthy features.) The CLIPScore family tends to overestimate scores. Specifically, reference-with-image metrics such as RefCLIP-S and RefPAC-S may not effectively compare references and a candidate. CLIP-S does not exhibit a tendency to overestimate; however, this does not necessarily imply that it estimates captions adequately. Rather, it may indicate a deficiency in its estimation capabilities, particularly for longer captions. This limitation likely stems from poor alignment between words and images in extended captions, as CLIP heavily relies on the alignment between image and language features. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0014-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0014-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0014-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0014-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0014-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0014-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning_images/cvpr_2024_Polos_Multimodal_Metric_Learning_from_Human_Feedback_for_Image_Captioning.pdf-0014-06.png)


**----- Start of picture text -----**<br>
x [(1)] [A cat standing on]<br>ref [:]<br>CLIP-S RefCLIP-S<br>top of a car trunk next 0.42 0.74<br>to a parked motorcycle. Polos Human<br>0.36 0.31<br>x cand: a dog is sitting PAC-S RefPAC-S<br>on a motorcycle seat 0.88 0.93<br>x [(1)] [A man stand-]<br>ref [:] CLIP-S RefCLIP-S<br>ing next to a dog 0.39 0.68<br>on the ground. Polos Human<br>0.31 0.25<br>PAC-S RefPAC-S<br>x cand: a dog is standing<br>in a wooden bench 0.8 0.85<br>CLIP-S RefCLIP-S<br>x [(1)] [A young boy reach-]<br>ref [:] 0.41 0.72<br>ing up to grab a red apple. Polos Human<br>0.51 0.47<br>x cand: a box of apples PAC-S RefPAC-S<br>0.73 0.78<br>x [(1)] [A banana peel]<br>ref [:]<br>made to look like CLIP-S RefCLIP-S<br>a clothing zipper. 0.41 0.72 Polos Human<br>0.48 0.42<br>x cand: a couple of PAC-S RefPAC-S<br>bananas sitting on 0.78 0.88<br>top of a napkin.<br>x [(1)] [a white stove with]<br>ref [:]<br>a pizza on a pan, a coffee CLIP-S RefCLIP-S<br>0.5 0.88<br>pot and a hot water pot Polos Human<br>0.76 0.61<br>x cand: a stove top PAC-S RefPAC-S<br>oven with a pizza 0.81 0.86<br>sitting on top of it.<br>x [(1)] [A girl in white shirt]<br>ref [:] CLIP-S RefCLIP-S<br>painting a black umbrella. 0.36 0.63<br>Polos Human<br>0.44 0.38<br>x cand: a girl in a<br>room with a chair PAC-S RefPAC-S<br>and a book shelf. 0.81 0.83<br>**----- End of picture text -----**<br>


Figure 6. Additional examples from the Polaris dataset. These are visualized in the same manner as in Fig.5. 

