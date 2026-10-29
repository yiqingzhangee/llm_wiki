# **A Hitchhikers Guide to Fine-Grained Face Forgery Detection Using Common Sense Reasoning** 

**Niki Maria Foteinopoulou**[1] **Enjie Ghorbel**[1] _[,]_[2] **Djamila Aouada**[1] 1CVI2, SnT, University of Luxembourg 

2Cristal Laboratory, National School of Computer Sciences, University of Manouba `{niki.foteinopoulou, enjie.ghorbel, djamila.aouada}@uni.lu` 

## **Abstract** 

Explainability in artificial intelligence is crucial for restoring trust, particularly in areas like face forgery detection, where viewers often struggle to distinguish between real and fabricated content. Vision and Large Language Models (VLLM) bridge computer vision and natural language, offering numerous applications driven by strong common-sense reasoning. Despite their success in various tasks, the potential of vision and language remains underexplored in face forgery detection, where they hold promise for enhancing explainability by leveraging the intrinsic reasoning capabilities of language to analyse fine-grained manipulation areas. For that reason, few works have recently started to frame the problem of deepfake detection as a Visual Question Answering (VQA) task, nevertheless omitting the realistic and informative open-ended multi-label setting. With the rapid advances in the field of VLLM, an exponential rise of investigations in that direction is expected. As such, there is a need for a clear experimental methodology that converts face forgery detection to a Visual Question Answering (VQA) task to systematically and fairly evaluate different VLLM architectures. Previous evaluation studies in deepfake detection have mostly focused on the simpler binary task, overlooking evaluation protocols for multi-label fine-grained detection and text-generative models. We propose a multi-staged approach that diverges from the traditional binary evaluation protocol and conducts a comprehensive evaluation study to compare the capabilities of several VLLMs in this context. In the first stage, we assess the models’ performance on the binary task and their sensitivity to given instructions using several prompts. In the second stage, we delve deeper into fine-grained detection by identifying areas of manipulation in a multiplechoice VQA setting. In the third stage, we convert the fine-grained detection to an open-ended question and compare several matching strategies for the multi-label classification task. Finally, we qualitatively evaluate the fine-grained responses of the VLLMs included in the benchmark. We apply our benchmark to several popular models, providing a detailed comparison of binary, multiple-choice, and open-ended VQA evaluation across seven datasets. `https://nickyfot.github. io/hitchhickersguide.github.io/` 

## **1 Introduction** 

Recent developments in deep generative modelling have resulted in hyper-realistic synthetic images/videos with no clear visible artefacts, making the viewers question whether they can still trust their eyes. Unfortunately, despite its relevance in a wide range of applications, such technology poses a threat to society as it can be used for malicious activities [16]. In a world where synthetic images of a person, known as deepfakes, can easily be generated, it becomes crucial to fight misinformation not 

38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks. 

only by identifying manipulated images/videos in an automated manner but also by explaining the decision behind this classification to reinstate trust in Artificial Intelligence. 

Numerous successful works on deepfake detection have been proposed in the literature to tackle the risks of face forgery [37, 77, 52, 7, 81]. Existing methods primarily rely on deep binary classifiers, resulting in black-box models that predict whether an input is real or fake. Consequently, explaining why those predictions are being made is not straightforward. To handle this issue, a few interpretable deepfake detection methods have been introduced by examining attention maps or weight activations [88, 72] to identify fine-grained areas of manipulation; however, these are based on a post-hoc analysis and thereby do not intrinsically incorporate an explainable mechanism. On the other hand, Vision Large Language Models (VLLMs) have emerged as a pioneering branch of generative Artificial Intelligence (AI), showcasing advancements in common sense reasoning and an inherent explainability that arises from the intrinsic nature of language [22]. They have demonstrated impressive capabilities in tasks such as Visual Question Answering (VQA) [47] and the generation of descriptive content for downstream applications [71], hence bridging the gap between vision-language understanding and contextual reasoning. However, despite these achievements, the explainable power of VLLMs remains under-explored in the field of deepfake detection, with only a handful of works mostly exploring the vision and language capabilities for the binary classification of fake/real images [26, 8, 31, 70] all of which are evaluated on different benchmarks and metrics. To the best of our knowledge, Zhang _et al_ . [87] is the only work employing a VQA approach in deepfake detection by proposing to extend the FF++ dataset with captions in natural language generated by humans. However, this work targets only one manipulated region at once, while deepfakes can incorporate several stacked manipulations [63, 68]. Moreover, the provided augmented FF++ dataset does not allow for cross-dataset evaluation in a VQA setting without an extensive annotation effort, making it difficult to investigate the generalisation aspect. In addition, the fine-grained evaluation in previous works is limited as the more challenging open-ended VQA task is not explored. 

Explainable fine-grained detection –that is, identifying manipulation beyond the binary fake/real decision– in natural language is still in its infancy. However, as VLLM works for deepfake detection are expected to appear following the overwhelming success of foundation models in other tasks, two research questions need to be addressed: 1) _“To what extent can existing VLLMs detect deepfake images and what rationale supports the decision?”_ and 2) _“How do we fairly and comprehensively evaluate VLLMs in the fine-grained task?”_ . In deepfake detection, benchmarks have mainly focused on binary or multi-class decisions and discriminative networks [82, 87], making them unsuitable to answer these research questions. Indeed, they do not propose a unified method to match the generated responses to fine-grained multi-label categories. Similarly, existing benchmarks in Visual Question Answering (VQA) [19, 10] primarily address multi-class tasks, which may not be suitable for the multi-label nature of fine-grained deepfake detection as highlighted in [63]. 

In this work, our objective is to conduct a thorough quantitative and qualitative evaluation of VLLMs for the task of fine-grained multi-label deepfake detection in a systematic and scientific approach, employing a multi-stage protocol without costly human captioning efforts. In the first stage, we assess the models’ performance on the binary task using various prompts while also evaluating the model’s sensitivity to the provided instructions. In the second stage, we delve into multi-label fine-grained detection, aiming to identify areas of manipulation within a multiple-choice Visual Question Answering (VQA) framework, i.e. what areas from a given list are identified as manipulated. Subsequently, in the third stage, we extend our investigation by converting the fine-grained detection task into an open-ended question –that is, identifying areas without instructing the model to select from a list of categories. Here, as the task is a multi-label problem, we compare two matching strategies: a) using the cosine similarity between the generated text and ground truth labels and b) counting the occurrence of the class name in the generated text. Finally, we qualitatively evaluate the fine-grained responses generated by the VLLMs included in our benchmark, providing nuanced and new insights into their performance. 

The main contributions of this work can be thus summarised as follows: 

- We introduce a novel evaluation protocol for deepfake detection under the Visual Question Answering (VQA) **multi-label setting** and without the use of human annotations. This is different from [87] which is based on a succession of yes/no questions for fine-grained areas, resulting in a binary classification setting. It also relies on a relatively small dataset that cannot be extended without costly annotation efforts. In addition, our multi-stage protocol 

2 

allows for open-ended VQA evaluation, which is a more challenging task. To the best of our knowledge, this is the first work to do so in the field of deepfake detection, offering a fresh perspective on explainability through fine-grained multi-label analysis. 

- We present a systematic, unified evaluation study of current state-of-the-art (SOTA) VLLMs, facilitating consistent assessment across different models. This framework is designed to be extendable to any existing or future deepfake dataset, ensuring fair and comprehensive comparisons with future models, thus promoting transparency and reproducibility in the evaluation process. 

- Through extensive comparison and analysis of the tested models and an ensemble of models, our study yields new insights into the capabilities and limitations of VLLMs in the context of deepfake detection. We will use these insights to advance research in the domain and hope to inform future developments and optimisations in model design and evaluation strategies. 

## **2 Related Work** 

**DeepFake generation** encompasses various forms of facial manipulation, including face reenactment [6, 1], face swapping [24, 45], and entirely generated face images [32, 69]. **Deepfake detection** algorithms classify samples as real or fake [7, 81, 77, 37], relying on artefacts left by manipulation methods, often analysed qualitatively for explainability [21, 91]. However, this qualitative analysis happens on a secondary stage and primarily depends on human observers. While generative methods often use natural language instructions [54, 75, 58, 55, 78], explaining manipulations in natural language –a natural extension of the generation process to detection– is still an emerging area. 

With the rise of VLLMs, recent works [8, 31, 70, 30, 73] explore vision and language for face forgery detection, primarily focusing on binary detection in a retrieval setting, with fewer [70, 52, 91] examining fine-grained areas usually as a secondary task. The latter have relied on generated pseudofake datasets to improve generalisation [70, 52, 91], which have a major drawback –that is, the use of pseudo-fake datasets hampers fair comparisons and does not reflect the current state-of-the-art in deepfake generation. 

Several **VLLMs** foundation models [4, 15, 35, 44, 43], bridge the gap between vision and language. These are typically trained on very large datasets with general knowledge. As the computational and data resources needed to train VLLMs from scratch are very high, numerous works leverage the pre-trained networks in three main directions: a) exploring the latent feature space [14, 57] of vision and language, b) parameter efficient training in a downstream task [39, 74, 18] and c) evaluating foundation models in new domains [65, 76]. 

**Benchmarks** for classification tasks [19, 38, 2] in a VQA setting typically address the multi-class paradigm, which may not be appropriate for addressing explainability in DeepFake detection by adopting a multi-label fine-grained strategy, as several areas can be manipulated at once. A few preliminary works in DeepFake detection [3, 26, 87] benchmark ChatGPT4[1] and Gemini[2] ; however, these have primarily focused on the more straightforward binary task and did not explore the reasoning capabilities of VLLMs for fine-grained labels. Furthermore, both these works focused on VLLMs that are not open-sourced with limited information regarding their training set and architecture; thus, it is not possible to assess whether the benchmarks are, in fact, zero-shot or whether they have been trained on deepfake-related image-language pairs. Zhang _et al_ . [87] propose extending FF++ annotations with captions in natural language using human annotators. However, this method is limited to binary decisions, while a given deepfake image can undergo several manipulations. Furthermore, it does not explore the open-ended VQA setting and does not offer a method for cross-dataset evaluation without the need for a costly annotation process. Within DeepFake detection, the vast majority of benchmarking works [60, 65, 46, 82, 34] have focused on binary discriminative networks and are therefore not fit to evaluate the capabilities of generative models such as VLLM, particularly for fine-grained labels. 

In a nutshell, the main novelty of this benchmark compared to previous works [19, 82, 38, 2, 60, 65, 46, 34] is threefold: 1) it converts the multi-label classification task of face forgery detection to a VQA task so that VLLM’s common sense reasoning capabilities can be evaluated, 2) it systematically 

> 1 `https://openai.com/gpt-4` 

> 2 `https://gemini.google.com` 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0004-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Is this image manipulated? a) Yes b) No<br>a)<br>a) Yes<br>Of the areas in the list Nose, Eyebrows,<br>b) Eyes, Lip, Hair, which ones are<br>manipulated?<br>In the image, the nose, eye, eyebrow,<br>lip, and hair of the blonde woman are<br>manipulated. The other areas, such as<br>ear, chin, and neck, are not altered.<br>c) What areas are manipulated?<br>In the image, the area that is<br>manipulated is the woman's face. The<br>photo has been digitally altered to<br>enhance her features, such as her<br>eyes, lips, and cheekbones, making<br>her look more attractive<br>**----- End of picture text -----**<br>


Figure 1: Overview of the proposed benchmarking method, using multiple stages to evaluate the performance of VLLMs in the context of deepfake detection. In the first stage (a), we assess the binary classification performance of VLLMs. In the second stage (b), we perform a fine-grained classification using multiple-choice instruction. In the third and final stage (c), we ask the model to identify fine-grained areas in open-ended VQA. The image example[3] is a sample from the SeqDeepFake dataset [63], and responses are generated using Llava-1.5 [43] 

and consistently assesses VLLM capabilities on nine binary and three fine-grained benchmarks and 3) is offering an open source and extendable framework for future zero-shot or task-specific VLLMs, that ensures a fair comparison. 

## **3 Common Sense Reasoning for Face Forgery Detection** 

**Preliminaries:** We formalise the language generation process of VLLM architectures, akin to standard VQA models, where the model is prompted with an image and a query to produce an auto-regressive answer. Given an image **X** _v ∈_ R _[H][×][W][ ×][C]_ and a text prompt **X** _t ∈_ R _[L][×][d]_ as input, a sentence _ψ_ is generated represented as a sequence of word tokens. The generation can be represented by the function _p_ ( _ψ|_ **X** _v,_ **X** _t_ ) =[�] _[|] j[ψ]_ =0 _[|][p]_[(] _[ψ][j][|][ψ][<j][,]_ **[ X]** _[v][,]_ **[ X]** _[t]_[)][, where] _[ H][×][ W][×][ C]_[represent the image] dimension, _L_ is the number of tokens, _d_ is the embedding dimension, _ψ_ = ( _ψ_ )0 _≤j<|ψ|_ is the generated sentence, and _|.|_ the cardinality. In VQA tasks, the model response aims to match human annotations. This task differs from typical classification problems due to the diverse semantic nature of questions and answers in natural language. The evaluation protocol is outlined for the binary case in Section 3.1 and for open-ended evaluation and multiple-choice of fine-grained labels in Section 3.2. An overview of the proposed method is shown in Figure 1. 

## **3.1 Binary Classification to VQA** 

In binary classification, the task is to predict whether the image sample is a product of face forgery. We create a benchmark to assess VLLM capability in binary Deepfake detection by transforming the discriminative task into a VQA problem. We consider only the positive category for each image **X** _v_ to generate the relevant instruction; that is, we limit the prompt to identifying whether an image is a Deepfake and not whether it is a genuine sample. The prompt used is in the form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0004-08.png)


where **s** _i ∈S_ denotes a set of standard terms used to describe deep fakes in the English language. The synonyms are employed to assess the reasoning capabilities of each tested model by investigating whether the understanding of the model is robust to the given instruction. 

> 3Ground truth: Hair, Nose, Lip, Eye 

4 

## **3.2 Fine-Grained Labels:** 

Fine-grained labels typically refer to manipulation areas. Predicting them necessitates the use of multi-label classification, as multiple areas can be manipulated at once. Following the initial binary prompt, a follow-up prompt to explain what areas of manipulation are identified is given to the VLLM with the same image, as shown in Fig. 1. We propose using two versions of follow-up prompts, one as an open-ended question and one as a multiple-choice. Specifically, the open-ended follow-up prompt follows the template: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0005-02.png)


For the multiple-choice instruction, we follow the template: 

**X** _t_ = `‘‘Of the areas in the list` [ **cls** 0 _, . . . ,_ **cls** _|C|_ ] `, which ones are` [ **s** _i_ ] `?’’` (3) where **cls** _i ∈C_ is the class name of the _i_ -th class from the set of target classes _C_ . 

## **3.3 Matching Strategies:** 

To evaluate the generated responses, we employ several matching methods depending on the task. The stricter one uses an _Exact Match (EM)_ approach, that estimates whether the generated sentence _ψ_ is exactly equal to the class name _clsi_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0005-07.png)


where _y_ ˆ _i_ is the prediction for the _i_ -th class. In the given task, an answer is considered correct only if the model output exactly matches the class names or ‘Yes’/‘No’ in the binary case. As the responses tend to be longer for fine-grained classification and reflect reasoning in natural language for a multi-label problem, a more appropriate matching strategy is to consider a response correct if the class name is _Contained_ in the response, as proposed by Xu _et al_ . [76] – that is _p_ ( ˆ _yi_ ) = 1 _,_ if **cls** _i ∈ ψ_ and 0 otherwise. We extend this to include synonyms of class names, as several ways exist to describe some areas (e.g. “Bangs” could also be described as “Hair”). Finally, we propose adapting the text-to-text score ( _CLIP distance_ ) proposed by Conti _et al_ . [14] for the multi-label task. This is done by using a sigmoid function over the cosine similarity matrix of the prediction embeddings and class name embeddings (obtained with a CLIP [59] text encoder), using an empirical temperature _t_ of 0 _._ 5 so that _p_ ( ˆ _yi_ ) = _σ_ ( _< ψ,_ **cls** _i >_[1] _t_[)][.][The symbol] _[ <][., .][>]_[ denotes the cosine similarity of the text] embeddings and _σ_ ( _._ ) is the sigmoid function. 

## **4 Experimental Set-Up** 

## **4.1 Tested VLLMs** 

We select four open-source state-of-the-art VLLMs to be included in this benchmark; specifically, we include LlaVa-1.5 [43] (an improved version of the LlaVa architecture [44]), BLIP2 [35] and finally InstructBlip [15] with Flan-T5 and Flan-T5-xxl language generators [13]. Finally, for the binary task, we include the CLIP [59] performance as a baseline and compare it against an ensemble of BLIP2 [35] and LlaVa-1.5 [43] following the ensembling strategies for VQA tasks [5], and GPT4v as an upper bound[4] . Experiments using GPT4v are performed on a subset of 5k samples selected from each dataset, and thus, the results may be susceptible to some degree of bias from the sampling, which needs to be considered when comparing the models. The selection of the VLLM is guided by three factors. First, we select architectures with publicly available weights and training methods to ensure transparency and fairness in the evaluation. Second, we include methods that generate output in Natural Language rather than a set of features or classification predictions. Finally, we select methods that have achieved state-of-the-art performance on several zero-shot tasks. Additional model details, such as the number of parameters and pre-training information of the tested models, can be found in Appendix A. 

> 4https://openai.com/index/gpt-4v-system-card/ 

5 

## **4.2 Datasets** 

We evaluate the performance of our method on seven published challenging benchmarks and one pseudo-fake dataset; more specifically, seven datasets for binary detection and two for the fine-grained task. All are evaluated at the frame level, as in previous image works [80, 52, 49]. **FF++:** [60] consists of over 20k images of DeepFake images from 1000 videos, using four types of manipulation methods: Deepfakes, Face2Face, FaceSwap and NeuralTextures. The dataset is split into train, validation, and test with an 80%, 10%, and 10% split, respectively. **DFDC:** [17] is composed of 5k videos of real and manipulated faces split into 4,464 unique training clips and 780 unique test clips. **Celeb-DF:** [41] includes 590 original videos collected from YouTube with subjects of different ages, ethnic groups and genders, and 5639 corresponding manipulated videos. **WildDeepFake:** [90] is a challenging dataset for in-the-wild detection, which consists of 7,314 face sequences extracted from 707 videos that are collected completely from the internet. **StyleGAN:** Two sub-sets consist of curated images generated with StyleGAN3 [29] and StyleGAN2 [28] along with original ones for binary detection of facial images. **SeqDeepFake:** [63] dataset consists of 85k sequential manipulated face images based on two representative facial manipulation techniques, facial components manipulation [32] and facial attributes manipulation [27]. The labels include annotations of manipulation sequences with diverse sequence lengths. **R-splicer:** Augmenting real data by generating pseudo-fake images is a common practice in deepfake detection [36, 49, 11, 66, 89, 40]. Such methods simulate characteristic face-swap artefacts using simplistic operations on a predefined set of regions. In this work, we use a spliced dataset of 59k images to evaluate fine-grained labels of five regions –entire face, mouth, nose, eyes, eyebrows– as implemented by Mejri _et al_ . [49]. 

## **4.3 Metrics** 

**Accuracy** and the Area Under the Receiver Operating Characteristics (ROC) Curve ( **AUC** ) are the most common metrics used in DeepFake detection [82, 52, 65]. However, as the datasets in the task are massively imbalanced, we also use the harmonic mean of Precision and Recall ( **F1-score** ) for the binary task. Furthermore, we note that as AUC is a measure of the classifier’s performance at different thresholds, it has very limited value in the VQA task where matching strategies result in polarised decisions; however, we include it for reference. In the fine-grained task, we use mean Average Precision ( **mAP** ), AUC and F1-score as indicators of classification performance. 

## **5 Results** 

## **5.1 Binary Classification** 

**Robustness to different prompts:** We use seven synonyms for the positive class: “manipulated”, “deepfake”, “synthetic”, “altered”, “fabricated”, “face forgery” and “falsified”. As the binary task is simple and the instruction format is a ‘Yes’ or ‘No’ question, we use _EM_ as a matching approach in this evaluation. In Fig. 2, we see the performance of each tested model under the binary detection setting on the two sub-sets of SeqDeepFake [63] and the R-splicer dataset using the three best performing synonyms: “manipulated”, “synthetic” and “altered”. The first observation is that no VLLM clearly outperforms others across all datasets and metrics. However, we see that BLIP-2 [35] has the most robust performance to the given instruction, even though it is the smallest in terms of parameters. Furthermore, the additional parameters of T5-xxl [13] do not seem to aid the task compared to the base InstructBLIP [15] with T5 generator, as the base model performs comparably better across most benchmarks. We theorise that as the VLLMs have not been explicitly trained on image-language pairs of manipulated images, a large number of parameters on the language generation leads to more hallucinations [25, 79] for this simple but abstract task. Compared to the CLIP [59], models appear to have competitive performance with the exception of InstructBLIP with T5xxl LLM. When both base models, i.e. BLIP-2 [35] and LlaVa [43], have relatively good performance, the ensemble shows marginal improvement, particularly in terms of Accuracy and F1; however, this is not consistent therefore we do not continue the investigation to fine-grained labels. The detailed performance of all models and synonyms across all datasets and additional analysis on CLIP [59] features can be found in the Appendix. 

**Overall model performance:** We average the performance of the three best-performing synonyms on all nine benchmarks in Fig. 3. No model clearly outperforms others across all metrics and 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0007-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0007-01.png)


**----- Start of picture text -----**<br>
(a) SeqDeepFake [63] attributes (b) SeqDeepFake [63] components (c) R-splicer dataset<br>Accuracy<br>(d) SeqDeepFake [63] attributes (e) SeqDeepFake [63] components (f) R-splicer dataset<br>AUC<br>(g) SeqDeepFake [63] attributes (h) SeqDeepFake [63] components (i) R-splicer dataset<br>F1-score<br>**----- End of picture text -----**<br>


Figure 2: Exact Match _(EM)_ Performance of each VLLM in terms of Accuracy (top), AUC (mid) and F1-score (bottom) for the top 3 synonyms “manipulated”, “synthetic” and “altered” 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0007-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0007-04.png)


**----- Start of picture text -----**<br>
(a) Accuracy (b) AUC (c) F1-score<br>**----- End of picture text -----**<br>


Figure 3: Exact Match _(EM)_ Performance of each VLLM on all nine benchmarks 

datasets; however, we can observe competitive performance from BLIP-2 [35] on the binary task, even though it is the smallest model in terms of parameters. We also see that all models struggle with the more challenging in-the-wild datasets, such as CelebDF [41], which highlights the need for further development to achieve adequate generalisation. Performance of GPT4v should be treated as an upper bound as we cannot assess whether the model has been trained on samples of the selected datasets. We see that GPT4v vastly outperforms the selected VLLMs on three benchmarks and has comparable performance on the rest, with the exception of FF++. 

**Vision Encoder Finetuning:** We finetune contrastively the vision encoder of LlaVa on FF++ using a sigmoid loss [83] over an ensemble of prompts for the real/fake categories, and evaluate as described in the previous section. Training details for the vision encoder can be found in Appendix C. The 

7 

architecture with the fine-tuned vision encoder shows improved within dataset and cross-dataset performance as shown in Tab. 1. Even without detailed captions or updating the LLM weights, we see there are still gains from a task specific vision encoder, particularly in terms of F1-score with an average improvement of nearly 4% within dataset and nearly 2% cross-dataset (for SeqDeepFake, CelebDF and StyleGAN2). 

Table 1: Binary performance of LlaVa-1.5 [43] with a fine-tuned Vision Encoder against the zero-shot baseline. 

|LlaVa Baseline<br>Acc.<br>F1|LlaVa w. fne-tuned Vision Encoder<br>Acc.<br>F1|
|---|---|
|||
|FF++<br>64.54<br>73.10|64.57 (+0.03%)<br>76.83 (+3.73%)|
|||
|SeqDeepfake Attr.<br>58.92<br>66.15<br>SeqDeepfake Comp.<br>84.62<br>90.57<br>R-Splicer<br>87.11<br>92.50<br>DFDC<br>54.02<br>58.24<br>CelebDF<br>35.67<br>41.81<br>DFW<br>53.35<br>61.56<br>StyleGAN2<br>33.67<br>38.62<br>StyleGAN3<br>39.80<br>45.66|61.22 (+2.30%)<br>68.03 (+1.88%)<br>84.24 (-0.37%)<br>90.20 (-0.38%)<br>87.31 (+0.20%)<br>92.62 (+0.12%)<br>53.86 (-0.16%)<br>58.65 (+0.41%)<br>37.60 (+1.93%)<br>43.53 (+1.72%)<br>53.45 (+0.10%)<br>61.90 (+0.34%)<br>35.20 (+1.53%)<br>39.72 (+1.10%)<br>39.30 (-0.50%)<br>45.74 (+0.08%)|



**Metrics:** In terms of the selected metrics, following the initial intuition, there is limited information we can get from the standard Accuracy and AUC used in the binary task. Both are heavily skewed by the label distribution, which is typically imbalanced in deepfake datasets; however, the latter may also not be fit for VLLMs as AUC measures performance at different thresholds, which are not present with _EM_ and _contains_ matching strategies. As such, we argue that for the task at hand, the F1-score –and consequently robust to imbalance metrics– are more appropriate. 

## **5.2 Finegrained Evaluation** 

For the fine-grained task, we evaluate the performance of the selected models in the open and closed vocabulary settings. The fine-grained labels are evaluated on samples where the ground truth is positive – i.e., on DeepFake samples. 

**Open-Ended VQA:** We first evaluate the selected VLLMs under the open vocabulary VQA setting on the three fine-grained datasets. The results using _contains_ and _CLIP distance_ matching are reported in Tab. 2a and Tab. 2b respectively. An _EM_ strategy is not possible in multi-label tasks, so no such evaluation is performed. No model clearly outperforms others across all metrics and datasets. In fact, we can observe that, in most cases, they have comparable performance. This holds true for both _contains_ and _CLIP distance_ metrics. In terms of matching strategy, using the _CLIP distance_ consistently and greatly improves recall, as is evident by the improvement in the F1-score and explicitly shown in Appendix H. This matching approach slightly lowers the mAP and AUC scores compared to the _contains_ metrics; however, using the cosine distance to match the open-ended responses to the class categories semantically may offer a more reliable output for the class of interest, as seen by the F1-score. 

**Multiple choice VQA:** The performance of the VLLMs on the multiple-choice instruction is shown in Fig. 4. Even though the open-ended setting is theoretically more challenging, the performances of all tested models are comparable to each other and worse on the multiple-choice instruction for both mAP and AUC. Regarding the F1-score, however, LlaVa [43] consistently performs better than other models. Under the multiple-choice setting, we observe that the models tend to mention all label names, which raises the number of False Positives –a significant limitation of the multiple choice setting– or respond with “All of them” or “None of them”, which makes matching of any sort more challenging and is reflected even more in the lower F1 score. Appendix G includes detailed metrics for each category. 

8 

Table 2: Model performance on open-ended fine-grained detection using a) _contains_ and b _CLIP_ matching 

|matching|||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||BLIP-2||InstructBLIP|||InstructBLIP-xxl||||LlaVa-1.5||
||mAP|AUC|F1|mAP|AUC|F1|mAP|AUC|F1|mAP|AUC|F1|
|SeqDeepFake [63]|61_._8|51_._0|20_._4|61_._3|50_._4|18_._3|**63**_._**1**|**53**_._**6**|37_._5|61_._7|51_._1|**40**_._**0**|
|attributes|||||||||||||
|SeqDeepFake [63]|59_._5|50_._5|14_._7|59_._2|50_._0|4_._1|**60**_._**2**|**51**_._**8**|**26**_._**2**|59_._0|49_._6|17_._1|
|components|||||||||||||
|R-Splicer|55_._8|55_._6|31_._3|52_._3|53_._2|23_._5|53_._8|54_._0|31_._1|**58**_._**7**|**57**_._**5**|**41**_._**6**|



|(a) Assessment|of model performance|of model performance|of model performance|during|open-ended evaluation|open-ended evaluation|open-ended evaluation|with_contains_|with_contains_|distance matching.|distance matching.|distance matching.|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||BLIP-2||InstructBLIP|||InstructBLIP-xxl||||LlaVa-1.5||
||mAP|AUC|F1|mAP|AUC|F1|mAP|AUC|F1|mAP|AUC|F1|
|SeqDeepFake [63]|**63**_._**0**|**53**_._**6**|73_._5|59_._9|50_._9|74_._0|60_._4|50_._7|55_._5|61_._0|51_._3|**74**_._**1**|
|attributes|||||||||||||
|SeqDeepFake [63]|58_._8|52_._7|71_._0|55_._5|49_._0|71_._7|**59**_._**9**|**55**_._**7**|59_._8|56_._1|49_._6|**71**_._**7**|
|components|||||||||||||
|R-Splicer|54_._3|55_._3|66_._2|48_._5|49_._3|66_._5|54_._0|53_._1|60_._3|**56**_._**7**|**57**_._**4**|**66**_._**5**|



(b) Assessment of model performance during open-ended evaluation with _CLIP_ distance matching. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0009-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0009-05.png)


**----- Start of picture text -----**<br>
(a) mAP (b) AUC (c) F1<br>**----- End of picture text -----**<br>


Figure 4: Assessment of model performance in multiple-choice settings, in terms of a) mAP, b) AUC and c) F1 during multiple-choice evaluation with _contains_ matching. 

## **5.3 Qualitative Evaluation** 

As the BertScore [86] is shown to correlate with human evaluation, we first present the Bertscore precision, recall, and F1 scores achieved by each model for the fine-grained open-ended responses compared with ground truth references that have been formatted using the prompt: “ `The areas that are` [ **s** _i_ ] `are` [ **c** _lsi_ ]”. The results of this evaluation, along with the score of human annotators [19, 10] on a subset of the R-Splicer dataset, are shown in Tab. 3. As in previous sections, no model clearly outperforms others across all benchmarks; however, we see that Llava-1.5 [43] has the most competitive performance for most benchmarks, closely followed by InstructBLIP [15]. This is consistent with qualitative evaluations on VQA tasks [19, 44]. 

**Overall performance:** In the simpler binary setting, BLIP-2 is more robust to instruction than other models with more parameters; however, when it comes to fine-grained evaluation, larger models show an advantage in reasoning and identifying areas of manipulation in the open-ended and multiple-choice settings. It is, however, worth noting that no model clearly outperforms others across all metrics and datasets. All of the results presented are based on zero-shot evaluations, where models are tested without being specifically trained for deepfake detection. Despite this, the models are able to leverage a semantic mapping between language and visual input from their very vast pre-training, giving them an inherent concept of “real” versus “fake”. This capability suggests that these models possess some degree of understanding when it comes to identifying deepfakes. However, this general understanding falls far behind that of task-specific models. When we fine-tune the vision encoder, there is a notable improvement in performance. Vision-language models can 

9 

Table 3: Open-ended qualitative evaluation with human annotators in Tab. a and BertScore [86] in Tab. b- d 

|Model<br>Human Eval. Score|Model<br>Precision<br>Recall<br>F1|
|---|---|
|BLIP-2<br>0_._35<br>InstructBLIP<br>0_._36<br>InstructBLIP-xxl<br>0_._33<br>LlaVa-1.5<br>**0**_._**38**|BLIP-2<br>79_._77<br>78_._75<br>79_._24<br>InstructBLIP<br>**86**_._**53**<br>83_._22<br>84_._81<br>InstructBLIP-xxl<br>80_._73<br>81_._78<br>81_._25<br>LlaVa-1.5<br>84_._86<br>**85**_._**31**<br>**85**_._**08**|
|(a) Average score of Human Evaluation (R-splicer)|(b) SeqDeepFake [63] attributes|
|Model<br>Precision<br>Recall<br>F1|Model<br>Precision<br>Recall<br>F1|
|BLIP-2<br>79_._87<br>79_._72<br>79_._61<br>InstructBLIP<br>81_._12<br>83_._89<br>**86**_._**90**<br>InstructBLIP-xxl<br>82_._57<br>81_._77<br>81_._01<br>LlaVa-1.5<br>**87**_._**40**<br>**86**_._**37**<br>85_._39|BLIP-2<br>79_._55<br>79_._76<br>80_._04<br>InstructBLIP<br>83_._47<br>85_._34<br>**87**_._**39**<br>InstructBLIP-xxl<br>82_._53<br>81_._87<br>81_._23<br>LlaVa-1.5<br>**85**_._**94**<br>**86**_._**33**<br>86_._74|



(c) SeqDeepFake [63] components (d) R-splice 

better capture details and nuances in input data, which enhances their deepfake detection capabilities. Nevertheless, due to the scarcity of high-quality captions and large-scale vision-language datasets tailored to deepfake detection, the improvements remain limited and only in the binary task. Overall, addressing these limitations—by creating specialised datasets and foundation models—could lead to substantial advancements in this area. 

**Limitations and Future Work:** As the models in this work are all evaluated under zero-shot settings, their performance is below that of purpose-build networks seen in previous works [82], particularly for more challenging in-the-wild datasets. This further highlights the need for task-specific models and more fine-grained deepfake datasets, which is a key finding of the experiments conducted in this work. A significant limitation is the lack of detailed language descriptions in datasets, making qualitative evaluation harder. Additionally, current datasets lack fine-grained labels, restricting assessments of manipulations to pseudo-fakes and SeqDeepFake [63]. Furthermore, as both the pertaining and evaluation datasets are not unbiased, the performance of all VLLMs is susceptible to the bias of the datasets, which is not addressed in this or previous benchmarks [82]. Identifying these shortcomings is important for future works on the task, particularly as VLLMs gain traction. 

## **6 Conclusion** 

In conclusion, our proposed benchmark has several contributions; first and foremost, we propose a method to transform deepfake detection into a VQA problem beyond binary classification to leverage common sense reasoning as an inherent explainability mechanism. We show how this can be achieved in both a multiple-choice and open-ended VQA –with the latter being the most important use-case for new and unknown face forgery methods. This approach is used to evaluate a multi-label problem that is not typical of classic VQA. By doing so, we can systematically and consistently evaluate the common sense reasoning capabilities of current and future VLLMs in fine-grained deepfake detection. 

Our selection of metrics and matching strategies allows for a fair evaluation of the proposed task. In particular, we include metrics that are robust to imbalance in both the binary and multi-label fine-grained tasks. Even though VLLMs in a zero-shot evaluation do not outperform purposebuilt methods, the generated responses include reasoning, therefore holding promise for significant contributions in explainable deepfake detection, confirming the initial motivation behind examining the use of such models for the task and understanding the current capabilities. Moreover, as this benchmark can be extended in terms of models and datasets, it allows for a systematic and fair comparison of new language generation methods for explainable deepfake detection. 

**Ethics statement:** The authors of this paper acknowledge the crucial role of ethical considerations in AI research and development. Our dedication lies in upholding principles of fairness and impartiality. Recognising the societal implications of generative technology (including VLLMs), we commit to transparency by openly communicating our findings and advancements with the research community. 

10 

## **Acknowledgments and Disclosure of Funding** 

This work is supported by the Luxembourg National Research Fund, under the BRIDGES2021/IS/16353350/FaKeDeTeR, and by POST Luxembourg. Experiments were performed on the Luxembourg national supercomputer MeluXina. The authors gratefully acknowledge the LuxProvide teams for their expert support. 

## **References** 

- [1] Madhav Agarwal, Rudrabha Mukhopadhyay, Vinay P Namboodiri, and CV Jawahar. Audio-visual face reenactment. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , 2023. 

- [2] Arjun Akula, Soravit Changpinyo, Boqing Gong, Piyush Sharma, Song-Chun Zhu, and Radu Soricut. Crossvqa: Scalably generating benchmarks for systematically testing vqa generalization. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , 2021. 

- [3] Omar Mustafa Al-Janabi, Osamah Mohammed Alyasiri, and Elaf Ayyed Jebur. GPT-4 versus Bard and Bing: LLMs for Fake Image Detection. In _2023 3rd International Conference on Intelligent Cybernetics Technology & Applications (ICICyTA)_ , December 2023. 

- [4] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. _Advances in neural information processing systems_ , 2022. 

- [5] Lisa Alazraki, Lluis Castrejon, Mostafa Dehghani, Fantine Huot, Jasper Uijlings, and Thomas Mensink. How (not) to ensemble lvlms for vqa. _arXiv preprint arXiv:2310.06641_ , 2023. 

- [6] Stella Bounareli, Christos Tzelepis, Vasileios Argyriou, Ioannis Patras, and Georgios Tzimiropoulos. Hyperreenact: one-shot reenactment via jointly learning to refine and retarget faces. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 2023. 

- [7] Junyi Cao, Chao Ma, Taiping Yao, Shen Chen, Shouhong Ding, and Xiaokang Yang. End-to-end reconstruction-classification learning for face forgery detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2022. 

- [8] You-Ming Chang, Chen Yeh, Wei-Chen Chiu, and Ning Yu. AntifakePrompt: Prompt-Tuned VisionLanguage Models are Fake Image Detectors. _arXiv preprint arXiv:2310.17419_ , November 2023. 

- [9] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 3558–3568, 2021. 

- [10] Anthony Chen, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. Mocha: A dataset for training and evaluating generative reading comprehension metrics. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2020. 

- [11] Liang Chen, Yong Zhang, Yibing Song, Lingqiao Liu, and Jue Wang. Self-Supervised Learning of Adversarial Example: Towards Good Generalizations for Deepfake Detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2022. 

- [12] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. URL `https://lmsys.org/blog/ 2023-03-30-vicuna/` . 

- [13] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. _arXiv preprint arXiv:2210.11416_ , 2022. 

- [14] Alessandro Conti, Enrico Fini, Massimiliano Mancini, Paolo Rota, Yiming Wang, and Elisa Ricci. Vocabulary-free image classification. _Advances in Neural Information Processing Systems_ , 2023. 

- [15] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. _Advances in Neural Information Processing Systems_ , 2023. 

11 

- [16] Audrey de Rancourt-Raymond and Nadia Smaili. The unethical use of deepfakes. _Journal of Financial Crime_ , 2023. 

- [17] Brian Dolhansky, Russ Howes, Ben Pflaum, Nicole Baram, and Cristian Canton Ferrer. The deepfake detection challenge (dfdc) preview dataset, 2019. 

- [18] Niki Maria Foteinopoulou and Ioannis Patras. Emoclip: A vision-language method for zero-shot video facial expression recognition. In _2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG)_ , pages 1–10. IEEE, 2024. 

- [19] Simon Ging, Maria Alejandra Bravo, and Thomas Brox. Open-ended VQA benchmarking of visionlanguage models by exploiting classification datasets and their semantic hierarchy. In _The Twelfth International Conference on Learning Representations (ICLR)_ , 2024. 

- [20] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 2017. 

- [21] Ijaz Ul Haq, Khalid Mahmood Malik, and Khan Muhammad. Multimodal neurosymbolic approach for explainable deepfake detection. _ACM Transactions on Multimedia Computing, Communications and Applications_ , 2023. 

- [22] Feijuan He, Yaxian Wang, Xianglin Miao, and Xia Sun. Interpretable visual reasoning: A survey. _Image and Vision Computing_ , 2021. 

- [23] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In _International Conference on Learning Representations_ , 2021. 

- [24] Baojin Huang, Zhongyuan Wang, Jifan Yang, Jiaxin Ai, Qin Zou, Qian Wang, and Dengpan Ye. Implicit identity driven deepfake face swapping detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2023. 

- [25] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. _ACM Computing Surveys_ , 2023. 

- [26] Shan Jia, Reilin Lyu, Kangran Zhao, Yize Chen, Zhiyuan Yan, Yan Ju, Chuanbo Hu, Xin Li, Baoyuan Wu, and Siwei Lyu. Can ChatGPT Detect DeepFakes? A Study of Using Multimodal Large Language Models for Media Forensics. _arXiv preprint arXiv:2403.14077_ , March 2024. arXiv:2403.14077 [cs]. 

- [27] Yuming Jiang, Ziqi Huang, Xingang Pan, Chen Change Loy, and Ziwei Liu. Talk-to-edit: Fine-grained facial editing via dialog. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , 2021. 

- [28] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , June 2020. 

- [29] Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. In _Advances in Neural Information Processing Systems_ , 2021. 

- [30] Mamadou Keita, Wassim Hamidouche, Hassen Bougueffa, Abdenour Hadid, and Abdelmalik TalebAhmed. Harnessing the power of large vision language models for synthetic image detection. _arXiv preprint arXiv:2404.02726_ , 2024. 

- [31] Sohail Ahmed Khan and Duc-Tien Dang-Nguyen. CLIPping the Deception: Adapting Vision-Language Models for Universal Deepfake Detection. _arXiv preprint arXiv:2402.12927_ , February 2024. 

- [32] Hyunsu Kim, Yunjey Choi, Junho Kim, Sungjoo Yoo, and Youngjung Uh. Exploiting spatial dimensions of latent in gan for real-time image editing. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , 2021. 

- [33] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _International journal of computer vision_ , 123:32–73, 2017. 

- [34] Chuqiao Li, Zhiwu Huang, Danda Pani Paudel, Yabin Wang, Mohamad Shahbazi, Xiaopeng Hong, and Luc Van Gool. A continual deepfake detection benchmark: Dataset, methods, and essentials. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ , January 2023. 

12 

- [35] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _International Conference on Machine Learning_ . PMLR, 2023. 

- [36] Lingzhi Li, Jianmin Bao, Ting Zhang, Hao Yang, Dong Chen, Fang Wen, and Baining Guo. Face X-Ray for More General Face Forgery Detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2020. 

- [37] Lingzhi Li, Jianmin Bao, Ting Zhang, Hao Yang, Dong Chen, Fang Wen, and Baining Guo. Face X-Ray for More General Face Forgery Detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2020. 

- [38] Linjie Li, Jie Lei, Zhe Gan, and Jingjing Liu. Adversarial vqa: A new benchmark for evaluating the robustness of vqa models. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 2021. 

- [39] Rongjie Li, Yu Wu, and Xuming He. Learning by Correction: Efficient Tuning Task for Zero-Shot Generative Vision-Language Reasoning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , April 2024. 

- [40] Yuezun Li and Siwei Lyu. Exposing deepfake videos by detecting face warping artifacts. _arXiv preprint arXiv:1811.00656_ , 2018. 

- [41] Yuezun Li, Xin Yang, Pu Sun, Honggang Qi, and Siwei Lyu. Celeb-df: A large-scale challenging dataset for deepfake forensics. In _IEEE Conference on Computer Vision and Patten Recognition (CVPR)_ , 2020. 

- [42] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13_ , 2014. 

- [43] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023. 

- [44] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In _Advances in Neural Information Processing Systems_ , 2023. 

- [45] Zhian Liu, Maomao Li, Yong Zhang, Cairong Wang, Qi Zhang, Jue Wang, and Yongwei Nie. Fine-grained face swapping via regional gan inversion. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2023. 

- [46] Yuhang Lu and Touradj Ebrahimi. Towards the Detection of AI-Synthesized Human Face Images. _arXiv preprint arXiv:2402.08750_ , February 2024. 

- [47] Jie Ma, Pinghui Wang, Dechen Kong, Zewei Wang, Jun Liu, Hongbin Pei, and Junzhou Zhao. Robust visual question answering: Datasets, methods, and future challenges. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2024. 

- [48] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In _Proceedings of the IEEE/cvf conference on computer vision and pattern recognition_ , 2019. 

- [49] Nesryne Mejri, Enjie Ghorbel, and Djamila Aouada. UNTAG: Learning Generic Features for Unsupervised Type-Agnostic Deepfake Detection. In _ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , 2023. 

- [50] Sachit Menon and Carl Vondrick. Visual Classification via Description from Large Language Models, December 2022. arXiv:2210.07183 [cs]. 

- [51] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In _2019 international conference on document analysis and recognition (ICDAR)_ . IEEE, 2019. 

- [52] Dat Nguyen, Nesryne Mejri, Inder Pal Singh, Polina Kuleshova, Marcella Astrid, Anis Kacem, Enjie Ghorbel, and Djamila Aouada. Laa-net: Localized artifact attention network for high-quality deepfakes detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2024. 

- [53] Yunsheng Ni, Depu Meng, Changqian Yu, Chengbin Quan, Dongchun Ren, and Youjian Zhao. Core: Consistent representation learning for face forgery detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 12–21, 2022. 

13 

- [54] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In _International Conference on Machine Learning_ . PMLR, 2022. 

- [55] James Oldfield, Christos Tzelepis, Yannis Panagakis, Mihalis Nicolaou, and Ioannis Patras. Parts of speech–grounded subspaces in vision-language models. _Advances in Neural Information Processing Systems_ , 2024. 

- [56] Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. _Advances in neural information processing systems_ , 24, 2011. 

- [57] Yassine Ouali, Adrian Bulat, Brais Matinez, and Georgios Tzimiropoulos. Black box few-shot adaptation for vision-language models. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 2023. 

- [58] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Text-driven manipulation of stylegan imagery. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 2021. 

- [59] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International Conference on Machine Learning_ . PMLR, 2021. 

- [60] Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, and Matthias Nießner. FaceForensics++: Learning to detect manipulated facial images. In _International Conference on Computer Vision (ICCV)_ , 2019. 

- [61] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. _arXiv preprint arXiv:2111.02114_ , 2021. 

- [62] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A- okvqa: A benchmark for visual question answering using world knowledge. In _European Conference on Computer Vision_ , pages 146–162. Springer, 2022. 

- [63] Rui Shao, Tianxing Wu, and Ziwei Liu. Detecting and recovering sequential deepfake manipulation. In _European Conference on Computer Vision (ECCV)_ , 2022. 

- [64] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 2556–2565, 2018. 

- [65] Yichen Shi, Yuhao Gao, Yingxin Lai, Hongyang Wang, Jun Feng, Lei He, Jun Wan, Changsheng Chen, Zitong Yu, and Xiaochun Cao. SHIELD : An Evaluation Benchmark for Face Spoofing and Forgery Detection with Multimodal Large Language Models. _arXiv preprint arXiv:2402.04178_ , February 2024. 

- [66] Kaede Shiohara and Toshihiko Yamasaki. Detecting deepfakes with self-blended images. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2022. 

- [67] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for image captioning with reading comprehension. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16_ , 2020. 

- [68] Inder Pal Singh, Nesryne Mejri, Van Dat Nguyen, Enjie Ghorbel, and Djamila Aouada. Multi-label deepfake classification. In _2023 IEEE 25th International Workshop on Multimedia Signal Processing (MMSP)_ , pages 1–5. IEEE, 2023. 

- [69] Michał Stypułkowski, Konstantinos Vougioukas, Sen He, Maciej Zi˛eba, Stavros Petridis, and Maja Pantic. Diffused heads: Diffusion models beat gans on talking-face generation. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ , January 2024. 

- [70] Ke Sun, Shen Chen, Taiping Yao, Haozhe Yang, Xiaoshuai Sun, Shouhong Ding, and Rongrong Ji. Towards General Visual-Linguistic Face Forgery Detection. _arXiv preprint arXiv:2307.16545_ , February 2024. 

- [71] Dídac Surís, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. _Proceedings of IEEE International Conference on Computer Vision (ICCV)_ , 2023. 

14 

- [72] Loc Trinh, Michael Tsang, Sirisha Rambhatla, and Yan Liu. Interpretable and trustworthy deepfake detection via dynamic prototypes. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ , 2021. 

- [73] Yabin Wang, Zhiwu Huang, Zhiheng Ma, and Xiaopeng Hong. Linguistic profiling of deepfakes: An open database for next-generation deepfake detection. _arXiv preprint arXiv:2401.02335_ , 2024. 

- [74] Alexandros Xenos, Niki Maria Foteinopoulou, Ioanna Ntinou, Ioannis Patras, and Georgios Tzimiropoulos. VLLMs Provide Better Context for Emotion Understanding Through Common Sense Reasoning. _arXiv:2404.07078 [cs]_ , April 2024. 

- [75] Weihao Xia, Yujiu Yang, Jing-Hao Xue, and Baoyuan Wu. Tedigan: Text-guided diverse face image generation and manipulation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2021. 

- [76] Peng Xu, Wenqi Shao, Kaipeng Zhang, Peng Gao, Shuo Liu, Meng Lei, Fanqing Meng, Siyuan Huang, Yu Qiao, and Ping Luo. Lvlm-ehub: A comprehensive evaluation benchmark for large vision-language models. _arXiv preprint arXiv:2306.09265_ , 2023. 

- [77] Yuting Xu, Jian Liang, Gengyun Jia, Ziming Yang, Yanhao Zhang, and Ran He. TALL: Thumbnail Layout for Deepfake Video Detection. In _2023 IEEE/CVF International Conference on Computer Vision (ICCV)_ , Paris, France, October 2023. IEEE. 

- [78] Zipeng Xu, Enver Sangineto, and Nicu Sebe. Stylerdalle: Language-guided style transfer using a vectorquantized tokenizer of a large-scale generative model. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 2023. 

- [79] Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli. Hallucination is Inevitable: An Innate Limitation of Large Language Models. _arXiv:2401.11817 [cs]_ , January 2024. 

- [80] Zhiyuan Yan, Yong Zhang, Yanbo Fan, and Baoyuan Wu. Ucf: Uncovering common features for generalizable deepfake detection. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , October 2023. 

- [81] Zhiyuan Yan, Yong Zhang, Yanbo Fan, and Baoyuan Wu. Ucf: Uncovering common features for generalizable deepfake detection. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 2023. 

- [82] Zhiyuan Yan, Yong Zhang, Xinhang Yuan, Siwei Lyu, and Baoyuan Wu. DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection. In _Advances in Neural Information Processing Systems_ , 2023. 

- [83] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 11975–11986, 2023. 

- [84] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [85] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. _arXiv preprint arXiv:2205.01068_ , 2022. 

- [86] Tianyi Zhang*, Varsha Kishore*, Felix Wu*, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. In _International Conference on Learning Representations_ , 2020. 

- [87] Yue Zhang, Ben Colman, Ali Shahriyari, and Gaurav Bharaj. Common sense reasoning for deep fake detection. _arXiv preprint arXiv:2402.00126_ , 2024. 

- [88] Cairong Zhao, Chutian Wang, Guosheng Hu, Haonan Chen, Chun Liu, and Jinhui Tang. Istvt: Interpretable spatial-temporal video transformer for deepfake detection. _IEEE Transactions on Information Forensics and Security_ , 2023. 

- [89] Tianchen Zhao, Xiang Xu, Mingze Xu, Hui Ding, Yuanjun Xiong, and Wei Xia. Learning self-consistency for deepfake detection. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 2021. 

- [90] Bojia Zi, Minghao Chang, Jingjing Chen, Xingjun Ma, and Yu-Gang Jiang. Wilddeepfake: A challenging real-world dataset for deepfake detection. In _Proceedings of the 28th ACM International Conference on Multimedia_ , 2020. 

15 

- [91] Dragos, -Constantin T, ânt,aru, Elisabeta Oneat,˘a, and Dan Oneat,˘a. Weakly-supervised deepfake localization in diffusion-generated images. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , 2024. 

16 

## **Appendix** 

## **A Model Zoo** 

**LlaVa-1.5** [43] is an extension of the LlaVa [44] model. We use the variant with CLIP-ViT-L-14 as a vision encoder and Vicuna-7b [12] language model. **BLIP-2** [35] uses a QFormer architecture to bridge frozen language and vision encoders. We use the variant with CLIP-ViT-L-14 as a vision encoder and the 2.7b OPT [85] language model. **InstructBLIP** [15] is a family of VLLMs that exploits the basic BLIP-2 [35] architecture and advances the task by giving the instruction to both the QFormer and the LLM. We use two variants of the architecture, with two different LLMs from the T5 family [13]; in the base architecture, we use the CLIP-ViT-L-14 as a vision encoder and the T5-xl LLM. **InstructBLIPxxl** uses the same vision encoder and the T5-xxl language model. A comparison of all architectures in terms of parameters and pre-training datasets can be seen in Tab. 4. It is worth noting, that none of the pre-training datasets are related to deepfakes, making the task more challenging. For the **ensemble** , we chose BLIP-2 [35] and LlaVa-15 [43] , based of two main factors: a) they show the most competitive performance on most datasets as seen in Fig. 3 and b) they have the least overlap in terms of pre-training data thus we intuitively expect them having complementary information. The ensembling method adopted in this work is using score fusion with majority voting; in the occasions where the models disagree, the mean is taken. **GPT4v** is used for comparison in the binary tasks, however as the training details of this model are unknown, it should only be treated as an upper bound. 

|Architecture|FLOPS|# Params|Pre-training Data|
|---|---|---|---|
|BLIP-2 [35]|0_._38_T_|3_._74_B_|COCO [42], Visual Genome [33], CC3M [64], CC12M [9],|
||||SBU [56], and115_M_ images from the LAION400M [61]|
|InstructBLIP [15]|0_._33_T_|4_._02_B_|COCO [42], WebCapFilt [35], TextCaps [67], VQAv2 [20],|
||||OK-VQA [48], AOK-VQA [62], OCR-VQA [51], LLaVA-|
||||Instruct-150K [44]|
|InstructBLIP-xxl [15]|0_._53_T_|12_._31_B_|COCO [42], WebCapFilt [35], TextCaps [67], VQAv2 [20],|
||||OK-VQA [48], AOK-VQA [62], OCR-VQA [51], LLaVA-|
||||Instruct-150K [44]|
|LlaVa-1.5 [43]|4_._14_T_|7_._06_B_|LLaVA-Instruct-150K [44], VQAv2 [20], OK-VQA [48],|
||||OCR-VQA [51]|



Table 4: Comparisons of model FLOPS, number of parameters and pre-training datasets for selected VLLMs 

**Implementation Details.** All experiments were conducted using four NVIDIA A100 GPUs, with 40GB of memory. We use the PyTorch deep learning framework for all model evaluation tasks and weights published on HuggingFace[5] . 

## **B Binary Classification Prompts** 

The term deepfake is a colloquial term for a wide range of manipulations using generative models, from altering one small area all the way to fully generated images and videos. As such, the class name itself has several synonyms that can describe it. To assess the model’s robustness to instruction, we first prompt an LLM –specifically ChatGPT3.5 to give us synonyms for a deepfake. This is done as an automation step to incorporate the general consensus into the method without the author’s bias, following previous works [50]. The following synonyms are tested for the positive class: “manipulated”, “deepfake”, “synthetic”, “altered”, “fabricated”, “face forgery” and “falsified”. We show the detailed performance of all models on all synonyms in Tab. 6. To provide context, we also provide the cross-dataset performance of several discriminative SOTA works in Tab. 5. The models show the most consistent performance on synonyms “manipulated”, “synthetic” and “altered”; therefore, we do all subsequent analyses on these prompts. 

> 5 `https://huggingface.co/` 

17 

Table 5: Reported cross-dataset performance of purpose-built discriminative SoTA models. 

||CelebDF [41]|CelebDF [41]|DFW|[90]|DFDC|[17]|
|---|---|---|---|---|---|---|
||AUC|mAP|AUC|mAP|AUC|mAP|
|UCF [81]|82.40|-|-|-|80.50|-|
|X-Ray [37]|79.50|-|-|-|65.50|-|
|Xception [60]|61.18|66.93|65.29|55.37|69.90|91.98|
|REECE [7]|70.93|70.35|68.16|54.41|-|-|
|CORE [53]|74.28|-|-|-|73.41|-|
|LAA-Net [52]|86.28|91.93|57.13|56.89|69.69|93.67|



## **C Vision Encoder Fine-tuning** 

We fine-tune the CLIP-L/14-336 Vision encoder using LoRA [23] adaptors. Specifically, we add 32 adaptors to the queries, keys, values and out projection of the attention heads, with an alpha of 32, a dropout rate of 0 _._ 2 and 4bit quantization. As the available datasets lack sample level descriptions, we use the text embeddings of the synonyms for the positive category and the embeddings of “real”, “original”, “unaltered”, “authentic”, “legitimate”, “genuine”, “bona fide” for the negative. We apply a Sigmoid loss over the cosine similarity [83] of all relevant text embeddings in the batch and train the vision encoder for 50 epochs. The updated weights are then used in the LlaVa-1.5 architecture. 

## **D CLIP Embeddings** 

As most models use CLIP [59] variants as backbone vision encoders, we assess the separability of samples in real and manipulated images in each dataset by visualising them in a two-dimensional plane using t-SNE. The resulting visualisations can be seen in Fig. 5. The samples appear more separable in some datasets. Interestingly, the StyleGAN datasets, where the images are fully generated, seem to have more distinguishable latent representations from real images. The separability is not necessarily reflected in the language generation as seen in Sec. 5.1; to further examine the root cause of this, we first calculate the average image embedding of each class so as to create a class prototype and retrieve the top-10 nearest language token embeddings, seen in Tab. 8. The first observation that can be made is that none of the tokens seems related to the task at hand, which would potentially inform prompt selection, so without the reasoning capabilities of the LLM, the token retrieval on its own is not very informative. Secondly, we see that a number of tokens are repeated across datasets and for both classes, which can be attributed to the much smaller sub-space of the deepfake task compared to the CLIP latent space; so, while the image embeddings are somewhat separable for several datasets retrieving the nearest language tokens shows that the subspace is not very much related to face forgery. From these two observations, we can better understand the zero-shot performance of the tested foundation models. Finally, we show the performance of CLIP on the binary classification task in Tab. 7. We use an ensemble of prompts using the Imagenet prompt templates [59]; for the positive class, we average the embeddings of the prompts for synonyms “manipulated”, “synthetic” and “altered”; for the negative, we use “real”, “original” and “unaltered”. Contrary to previous VQA works [14, 19] that use CLIP retrieval as an upper bound, we see this is not the case in the task. This can be attributed to the more abstract definition of the class in the deepfake detection task, as well as the pre-training dataset, which is more object-oriented. 

## **E Human Evaluation** 

The human evaluation is based upon previous studies in VQA evaluation [19, 10]. The humans are asked to rate model predictions on a scale from 1 to 5, where 1 is completely wrong and 5 is completely correct. 

**Evaluation dataset:** We use a subset of the pseudo-fake R-Splicer dataset for the human evaluation study, as it has artefacts visible to the human eye and is thus easier for the annotators to assess the response quality. 

18 

Table 6: Binary Performance of tested models 

|Synonym|Seq. Deepfake Attr.<br>Acc. AUC<br>F1|Seq. Deepfake Comp.<br>Acc. AUC<br>F1|R-Splicer<br>Acc. AUC<br>F1|FF++<br>Acc. AUC<br>F1|DFDC<br>Acc. AUC<br>F1|CelebDF<br>Acc. AUC<br>F1|DFW<br>Acc. AUC<br>F1|StyleGAN2<br>Acc. AUC<br>F1|StyleGAN3<br>Acc. AUC<br>F1|
|---|---|---|---|---|---|---|---|---|---|
|manipulated<br>altered<br>synthetic|95.36 50.49 97.62<br>96.06 50.25 97.99<br>95.48 51.75 97.68|94.44 50.46<br>97.14<br>94.62 49.93<br>97.23<br>93.95 52.09<br>96.87|95.32 50.59 97.60<br>95.85 50.22 97.88<br>90.45 54.42 94.95|65.42 49.76 79.00<br>65.76 50.01 79.24<br>65.27 51.63 78.15|49.47 49.95 66.13<br>49.50 49.98 66.21<br>49.46 49.93 65.95|52.20 49.87 68.34<br>51.60 49.31 67.73<br>40.80 39.20 56.21|50.40 49.99 66.93<br>50.37 49.96 66.94<br>50.51 50.17 65.05|52.60 50.25 68.69<br>53.20 50.84 69.13<br>54.60 52.37 69.53|50.00 49.29 66.33<br>51.49 50.76 67.66<br>53.73 53.05 68.37|
|deepfake<br>fabricated<br>face forgery<br>falsifed|93.92 51.54 96.86<br>95.00 51.80 97.43<br>96.11 49.98 98.01<br>95.58 52.10 97.73|93.15 54.50<br>96.43<br>94.02 50.56<br>96.92<br>94.72 51.87<br>97.28<br>93.29 51.74<br>96.52|89.47 54.91 94.40<br>91.70 52.53 95.65<br>94.18 51.18 97.00<br>92.03 53.25 95.83|64.66 52.19 77.28<br>64.79 50.58 78.04<br>65.03 50.42 78.36<br>65.42 51.46 78.38|49.35 49.82 65.81<br>49.41 49.89 66.03<br>49.44 49.92 66.01<br>49.40 49.88 65.94|39.40 37.79 55.24<br>46.60 44.64 62.66<br>53.00 50.69 68.79<br>42.80 41.07 58.55|50.48 50.16 64.18<br>50.85 50.49 65.80<br>49.67 49.27 66.19<br>52.92 52.56 67.06|53.80 51.53 69.16<br>53.80 51.51 69.24<br>53.60 51.26 69.31<br>53.80 51.57 68.99|50.75 50.07 66.33<br>51.49 50.80 67.01<br>51.49 50.76 67.66<br>52.99 52.30 68.02|
|(a) BLIP-2 [35] performance on the nine benchmark datasets.||||||||||
|Synonym|Seq. Deepfake Attr.<br>Acc. AUC<br>F1|Seq. Deepfake Comp.<br>Acc. AUC<br>F1|R-Splicer<br>Acc. AUC<br>F1|FF++<br>Acc. AUC<br>F1|DFDC<br>Acc. AUC<br>F1|CelebDF<br>Acc. AUC<br>F1|DFW<br>Acc. AUC<br>F1|StyleGAN2<br>Acc. AUC<br>F1|StyleGAN3<br>Acc. AUC<br>F1|
|manipulated<br>altered<br>synthetic|43.85 57.87 59.50<br>45.10 55.92 60.86<br>74.33 61.96 84.97|84.34 69.39<br>91.25<br>82.10 65.58<br>89.91<br>88.39 70.21<br>93.67|80.95 76.04 89.17<br>81.31 76.41 89.40<br>88.14 62.50 93.61|53.87 60.01 54.01<br>54.32 61.01 53.83<br>59.24 54.51 69.17|55.11 54.90 41.78<br>55.51 55.29 42.39<br>53.49 53.79 64.10|19.40 19.95 9.84<br>19.40 19.94 10.24<br>44.20 42.93 56.61|58.73 58.68 61.10<br>58.37 58.32 61.23<br>51.54 51.23 64.65|17.40 18.05 5.49<br>17.80 18.41 6.80<br>29.60 28.99 38.25|26.12 26.25 19.51<br>27.61 27.70 23.62<br>38.06 37.86 45.75|
|deepfake<br>fabricated<br>face forgery<br>falsifed|20.19 58.62 29.42<br>74.23 58.01 84.96<br>19.90 58.47 28.99<br>19.62 58.33 28.55|62.94 75.23<br>75.93<br>96.08 63.74<br>97.97<br>54.55 74.75<br>68.60<br>57.76 75.13<br>71.51|66.76 78.77 79.26<br>91.92 57.64 95.76<br>71.10 82.75 82.42<br>64.94 81.28 77.80|49.04 60.62 39.09<br>62.85 50.97 75.76<br>51.26 62.73 42.42<br>49.44 61.70 38.24|53.54 53.17 24.19<br>52.18 52.42 61.59<br>54.76 54.42 30.14<br>54.15 53.80 27.19|26.80 28.13 0.54<br>55.80 53.82 69.26<br>26.80 28.13 0.54<br>26.00 27.29 0.54|59.07 59.30 43.77<br>51.35 51.03 65.02<br>56.81 57.03 41.85<br>55.86 56.14 34.03|26.80 28.13 0.54<br>28.60 27.87 38.77<br>27.00 28.32 1.08<br>26.00 27.29 0.54|28.36 28.74 4.00<br>32.84 32.53 44.44<br>32.84 33.29 4.26<br>29.10 29.52 2.06|



(b) InstructBLIP [15] performance on the nine benchmark datasets. 

|Synonym|Seq. Deepfake Attr.<br>Acc. AUC<br>F1|<br>Seq. Deepfake Comp<br>Acc. AUC<br>F1|.<br>R-Splicer<br>Acc. AUC<br>F1|FF++<br>Acc. AUC<br>F1|DFDC<br>Acc. AUC<br>F1|CelebDF<br>Acc. AUC<br>F1|DFW<br>Acc. AUC<br>F1|StyleGAN2<br>Acc. AUC<br>F1|StyleGAN3<br>Acc. AUC<br>F1|
|---|---|---|---|---|---|---|---|---|---|
|manipulated<br>altered<br>synthetic|16.44 56.68 23.57<br>17.21 57.08 24.80<br>17.40 57.18 25.11|58.88 77.04<br>72.42<br>60.98 78.14<br>74.19<br>56.64 77.17<br>70.42|58.49 76.85 72.63<br>60.99 77.24 74.72<br>59.76 75.42 73.74|48.53 60.88 36.70<br>49.04 61.25 37.78<br>49.74 61.62 39.46|53.88 53.54 28.75<br>53.99 53.66 29.59<br>53.79 53.44 27.57|24.80 26.05 0.00<br>23.40 24.52 1.54<br>24.00 25.21 0.00|54.09 54.37 32.36<br><br>54.90 55.15 36.37<br><br>56.85 57.06 43.52|24.80 26.05 0.00<br>23.20 24.33 1.03<br>24.20 25.40 0.52|25.37 25.74 1.96<br><br>38.81 39.39 0.00<br><br>28.36 28.74 4.00|
|deepfake<br>fabricated<br>face forgery<br>falsifed|7.98 52.29<br>8.77<br>21.06 59.07 30.72<br>9.42 53.04 11.47<br>16.73 56.83 24.04|13.43 54.42<br>16.24<br>64.34 79.91<br>76.92<br>30.63 63.48<br>42.46<br>58.74 78.28<br>72.25|41.08 69.28 56.04<br>64.97 77.75 77.90<br>41.83 69.76 56.84<br>63.55 78.75 76.76|42.22 56.31 22.40<br>51.71 62.81 43.75<br>42.83 56.74 23.90<br>49.68 61.77 38.94|52.28 51.85 14.48<br>54.40 54.08 31.55<br>52.24 51.82 16.20<br>53.81 53.47 28.49|35.00 36.76 0.00<br>23.80 24.96 1.04<br>34.20 35.92 0.00<br>24.40 25.57 1.56|51.36 51.73 13.75<br><br>56.00 56.13 48.37<br><br>51.66 52.02 15.64<br><br>54.25 54.49 35.95|35.00 36.76 0.00<br>23.60 24.77 0.52<br>34.20 35.92 0.00<br>24.00 25.19 0.52|38.81 39.39 0.00<br><br>25.37 25.74 1.96<br><br>26.87 27.25 2.00<br><br>26.87 27.25 2.00|



(c) InstructBLIP [15] with Flan-T5-xxl [13] language model performance on the nine benchmark datasets. 

|Synonym|Seq. Deepfake Attr.<br>Acc. AUC<br>F1|Seq. Deepfake Comp.<br>Acc. AUC<br>F1|R-Splicer<br>Acc. AUC<br>F1|FF++<br>Acc. AUC<br>F1|DFDC<br>Acc. AUC<br>F1|CelebDF<br>Acc. AUC<br>F1|DFW<br>Acc. AUC<br>F1|StyleGAN2<br>Acc. AUC<br>F1|StyleGAN3<br>Acc. AUC<br>F1|
|---|---|---|---|---|---|---|---|---|---|
|manipulated<br>altered<br>synthetic|73.05 57.79 84.14<br>89.16 55.66 94.22<br>14.54 54.96 20.09|94.34 53.56<br>97.07<br>94.30 51.65<br>97.06<br>65.21 79.44<br>77.59|94.05 62.20 96.90<br>96.10 52.45 98.01<br>71.17 75.60 82.58|67.52 55.20 79.18<br>66.42 51.41 79.42<br>59.69 65.70 60.69|54.51 54.80 64.63<br>50.67 51.11 65.95<br>56.87 56.66 44.14|38.80 37.47 52.78<br>49.00 46.85 65.31<br>19.20 19.86 7.34|52.92 52.58 66.54<br>51.05 50.65 67.17<br>56.08 56.17 50.97|37.40 36.13 51.17<br>47.20 45.12 63.74<br>16.40 17.19 0.95|47.01 46.46 61.62<br>50.75 50.04 66.67<br>21.64 21.86 8.70|
|deepfake<br>fabricated<br>face forgery<br>falsifed|8.00 52.16<br>8.29<br>12.72 54.31 16.93<br>12.62 54.56 16.72<br>16.85 56.46 23.86|38.46 67.53<br>51.91<br>58.11 75.38<br>71.73<br>56.78 76.25<br>70.50<br>64.83 77.98<br>77.32|51.03 74.43 65.99<br>66.03 78.94 78.70<br>71.86 84.60 82.94<br>78.19 84.70 87.30|46.16 59.25 31.44<br>51.93 63.19 43.73<br>55.75 66.12 50.36<br>60.02 68.31 58.50|54.18 53.79 23.32<br>54.71 54.38 30.14<br>56.38 56.05 33.71<br>56.98 56.72 40.85|34.60 36.33 0.61<br>25.00 26.24 0.53<br>29.40 30.79 2.75<br>23.00 23.93 5.87|53.63 53.98 20.52<br>53.82 54.07 33.94<br>55.02 55.29 35.12<br>57.44 57.62 46.59|34.20 35.92 0.00<br>25.00 26.24 0.53<br>28.60 30.02 0.56<br>21.40 22.42 1.50|37.31 37.88 0.00<br>29.85 30.19 9.62<br>32.84 33.31 2.17<br>26.87 27.18 7.55|



(d) LlaVa-15 [43] performance on the nine benchmark datasets. 

|Synonym|Seq. Deepfake Attr.<br>Acc. AUC<br>F1|Seq. Deepfake Comp.<br>Acc. AUC<br>F1|R-Splicer<br>Acc. AUC<br>F1|FF++<br>Acc. AUC<br>F1|DFDC<br>Acc. AUC<br>F1|CelebDF<br>Acc. AUC<br>F1|DFW<br>Acc. AUC<br>F1|StyleGAN2<br>Acc. AUC<br>F1|StyleGAN3<br>Acc. AUC<br>F1|
|---|---|---|---|---|---|---|---|---|---|
|manipulated<br>altered<br>synthetic|95.38 58.70 97.63<br>95.73 60.44 97.82<br>80.63 54.15 89.02|94.83 67.61<br>97.34<br>94.83 47.48<br>97.34<br>65.31 56.09<br>77.70|96.24 54.77 98.08<br>96.40 52.40 98.17<br>88.48 53.20 93.83|66.33 74.05 79.69<br>66.18 83.08 79.63<br>63.24 56.20 74.45|49.61 54.59 66.18<br>49.57 60.67 66.24<br>51.09 52.60 63.29|52.40 51.21 68.60<br>52.40 50.00 68.77<br>44.20 39.67 57.14|50.50 59.84 67.05<br>50.42 50.00 67.04<br>48.37 46.35 61.71|53.79 58.98 68.39<br>55.36 72.93 69.86<br>58.20 58.93 66.07|50.75 50.00 67.33<br>50.75 50.00 67.33<br>55.22 64.14 68.42|
|deepfake<br>fabricated<br>face forgery<br>falsifed|64.90 51.64 78.13<br>94.62 52.63 97.23<br>75.96 49.16 86.26<br>94.42 52.28 97.13|37.76 53.74<br>51.26<br>13.17 51.97<br>18.13<br>12.60 51.96<br>17.14<br>63.64 55.34<br>76.45|88.17 52.14 93.66<br>91.13 53.02 95.33<br>94.23 54.09 97.02<br>93.91 55.39 96.85|63.12 54.90 75.02<br>64.09 56.23 75.85<br>65.67 58.53 77.33<br>66.58 60.22 78.59|49.72 50.65 64.95<br>49.80 50.66 64.39<br>51.12 56.03 65.74<br>49.72 51.56 65.69|43.40 37.76 57.06<br>51.80 50.20 65.02<br>55.20 69.48 69.81<br>44.40 36.23 59.36|49.84 49.01 63.39<br>49.81 48.39 64.50<br>48.77 38.75 65.08<br>52.09 55.80 66.32|53.15 52.78 61.87<br>55.05 55.36 64.60<br>44.79 42.94 54.55<br>50.63 49.29 61.88|50.75 50.42 64.52<br>55.97 65.32 68.78<br>48.51 44.79 63.10<br>52.99 59.33 67.36|



(e) Ensemble performance on the nine benchmark datasets. 

**Human evaluation:** We select 100 samples from the dataset and generate responses using the open-ended prompt. Each sample is annotated by the three human annotators on a scale from 1 to 5, where 1 is completely wrong, and 5 is completely correct. Each annotator is shown 50 samples. To reduce the workload on human annotators, we assign 1 (completely wrong) to all responses that describe the image content but no areas of manipulation. The annotations are then standardised between 0 and 1. An example of the form shown to human evaluators can be seen in Fig. 6. 

**Annotator agreement:** as in previous works [19, 10] we use Krippendorff’s alpha to assess the inter-annotator agreement and obtain _._ 75, which is a strong agreement given the complexity of the multi-label task. We average the annotator scores to receive the final gold standard for each sample and then obtain the average human score for each model. 

## **F Qualitative Samples** 

By examining a few samples in Fig. 7, we can also see that all models tend to hallucinate or provide a general description of the image content when they fail to identify specific areas of manipulation. Furthermore, while BLIP-2 [35] tends to respond more concisely –which is identified by Liu _et al_ . 

19 

Table 7: CLIP baseline performance on the binary task of the nine selected benchmarks. 

|Dataset|Accuracy|AUC|F1|
|---|---|---|---|
|SeqDeepFake attributes [63]|54_._62|63_._46|69_._63|
|SeqDeepFake components [63]|35_._80|49_._10|50_._38|
|R-splice|17_._14|44_._33|25_._98|
|FF++ [60]|34_._85|46_._50|17_._38|
|DFDC [17]|46_._84|46_._75|40_._93|
|CelebDF [41]|59_._80|60_._95|49_._11|
|DFW [90]|49_._67|49_._90|31_._09|
|StyleGan2 [28].|69_._20|69_._92|65_._16|
|StyleGan3 [29]|69_._40|69_._63|64_._35|



Table 8: Top 10 closest tokens to the class prototypes. Unique to the class tokens are in **bold** . 

|Dataset|Original|DeepFake|
|---|---|---|
|SeqDeepFake attributes|natives, labeling, liz,**saving**, rink,**demon**, pitch, creole, godis, wentz|natives, liz, labeling, rink, wentz,**ronda**, godis, creole,**%)**, melissa|
|SeqDeepFake components|natives,**labeling**, qld, romo, liz,**creole**,**dhoni**,**.**,**hijab**, gaining|natives, qld, liz, gaining, romo,**%)**,**cuomo**,**klo**,**minions**,**cowgirl**|
|R-splice|natives, wentz, anglo, liz, %), qld, anca,**romo**,**ronda**, ural|anglo, %), wentz, liz, natives,**klo**, anca, qld, ural,**weed**|
|FF++|natives, qld, wentz, anglo, cuomo, anca, liz,**labeling**, romo,**melissa**|natives, qld, wentz, anglo, liz, cuomo, anca,**%)**,**weed**, romo|
|DFDC|natives, wentz, anglo, liz, ronda, qld, %), anca, romo, )!!|anglo, wentz, natives, liz, %), ronda, qld, anca, )!!, romo|
|CelebDF|natives,**liz**, wentz,**ronda**, %), anglo,**qld**, romo,**labeling**,**anca**|anglo, natives,**ural**,**klo**,**creole**, %),**weed**, wentz, romo,**minions**|
|DFW|anglo, %), ronda, wentz, ural, natives,**liz**, klo,**rene**, ator|anglo, %), natives, ural, wentz, ator,**ronda**, klo,**anca**, qld|
|StyleGan2|natives, liz, wentz, ronda,**%)**,**anglo**,**qld**, romo, labeling,**anca**|natives, labeling, liz,**rink**, ronda, romo,**creole**,**melissa**, wentz,**saving**|
|StyleGan3|natives, liz, wentz, ronda, romo, labeling,**aa**,**anca**,**%)**,**ural**|natives, labeling, liz, ronda,**rink**, wentz, romo,**saving**,**creole**,**melissa**|



[44] as well–, we note that larger models attempt to provide a justification for their response, which is ultimately the goal of investigating VLLMs for this task. 

As the available datasets lack sample specific descriptions of manipulation areas, we include BertScore of the tested VLLMs on the MagicBrush [84] dataset to assess response quality on the neighbouring task of image editing detection, that is language driven in table Tab. 9. 

## **G Multiple-choice VQA** 

Further to the analysis in Sec. 5.2, we show the performance of all models on the multiple-choice VQA setting in Tab. 10. In the multiple-choice setting, all models have comparable mAP and AUC, however, LlaVa-1.5 [43] shows a clear advantage in terms of Recall and F1. 

## **H Open-ended VQA** 

The detailed performance of each model on the open-ended fine-graned detection can be seen on Tabs. 11 to 14. The matching strategy has a significant impact on Recall, as mentioned in Sec. 5.2, thus also increasing the F1 score. 

Table 9: BertScore of selected VLLMs on MagicBrush dataset 

||Precision|Recall|F1|
|---|---|---|---|
|Blip|85.44|84.37|84.57|
|InstructBlip|84.56|85.09|84.82|
|InstructBlip-xxl|81.07|84.55|81.88|
|LlaVa 1.5|83.08|84.77|83.87|



20 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0021-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0021-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0021-02.png)


**----- Start of picture text -----**<br>
(a) SeqDeepFake [63] attributes (b) SeqDeepFake [63] components<br>(c) R-Spliced (d) FF++<br>(e) DFW [90] (f) DFDC [17]<br>(g) CelebDF [41] (h) StyleGAN2<br>(i) StyleGAN3<br>**----- End of picture text -----**<br>


Figure 5: t-SNE Visualisation of CLIP [59] image embeddings on the test set of the selected datasets (perplexity=50) 

21 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0022-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0022-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0022-02.png)


**----- Start of picture text -----**<br>
(a) Annotator Briefing (b) Annotation Form<br>**----- End of picture text -----**<br>


Figure 6: Example of Briefing( a) and Annotation Form( b) shown to human evaluators. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0022-04.png)


**----- Start of picture text -----**<br>
BLIP-2 InstructBLIP InstructBLIPxxl LlaVa-1.5<br>The face of the man in the  The image features a man in  In the image, the man's face<br>middle of the image has been  The area of the image that is  a suit and tie speaking into a  has been digitally<br>made to look like he has a  manipulated is the man's face. microphone. He is wearing  manipulated to appear as if<br>mouth full of teeth glasses and a tie, and his face he has a mustache.<br>(a) Ground Truth: Mouth, Faceswap<br>The area of the image that is<br>The eyes, nose, and mouth  The image features a man<br>are the only parts of the  The area of the image that is  with a tattoo on his neck and  manipulated is the man's<br>image that have been  manipulated in the image is the man's face. face, with the tattoo visible  eyes. They have been digitally altered to appear as<br>manipulated on the right side of his neck.<br>if they are g...<br>(b) Ground Truth: Eyebrows, Eyes<br>The face of the man in the green shirt The face of the man in the image is manipulated, as it appears to have been retouched. The image features a man with a green background and a distorted facial expression. The man's face is  In the image, the area that is manipulated is the man's face. Specifically, his eyes, nose, and mouth have been<br>the primary focus of the... altered<br>(c) Ground Truth: Nose, Mouth, Faceswap<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0022-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0022-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning_images/nips_2024_A_Hitchhiker's_Guide_to_Fine-Grained_Face_Forgery_Detection_Using_Common_Sense_Reasoning.pdf-0022-07.png)


Figure 7: Samples of generated responses in an open-ended setting. 

22 

Table 10: Performance of tested VLLMs in multiple-choice setting 

|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|
|---|---|---|
|Bangs<br>49.8<br>0.0<br>60.3<br>0.0<br>Eyeglasses49.9<br>8.0<br>63.4<br>4.6<br>Beard<br>49.8<br>1.0<br>61.0<br>0.5<br>Smiling<br>50.0<br>0.0<br>58.3<br>0.0<br>Young<br>50.0<br>0.0<br>62.3<br>0.0|nose<br>50.0<br>0.0<br>55.4<br>0.0<br>eye<br>50.8<br>10.2<br>66.8<br>5.5<br>eyebrow<br>51.0<br>11.0<br>58.3<br>6.0<br>lip<br>50.0<br>0.0<br>67.6<br>0.0<br>hair<br>50.4<br>9.7<br>48.9<br>5.4|nose<br>51.3<br>46.9<br>52.5<br>42.2<br>eyebrows<br>50.4<br>57.0<br>52.0<br>63.1<br>eyes<br>50.2<br>56.2<br>51.5<br>61.9<br>mouth<br>49.9<br>45.4<br>51.9<br>40.9<br>faceswap<br>44.6<br>9.2<br>50.2<br>5.5|
|Total<br>49.9<br>4.5<br>61.1<br>2.5|Total<br>50.4<br>10.3<br>59.4<br>5.7|Total<br>49.3<br>42.9<br>51.6<br>42.7|



## (a) SeqDeepFake attributes [63] (b) SeqDeepFake comp [63] (c) R-Splicer 

## BLIP-2 [35] 

||BLIP-2[35]||
|---|---|---|
|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|
|Bangs<br>50.2<br>1.0<br>60.4<br>0.5<br>Eyeglasses58.4<br>51.3<br>68.0<br>41.6<br>Beard<br>50.0<br>0.3<br>61.0<br>0.2<br>Smiling<br>50.0<br>0.0<br>58.3<br>0.0<br>Young<br>50.4<br>29.0<br>62.6<br>21.4|nose<br>50.0<br>0.0<br>55.4<br>0.0<br>eye<br>52.2<br>11.7<br>67.7<br>6.5<br>eyebrow<br>50.6<br>3.3<br>58.2<br>1.7<br>lip<br>50.0<br>0.0<br>67.6<br>0.0<br>hair<br>51.1<br>24.7<br>49.3<br>23.6|nose<br>56.5<br>38.5<br>56.2<br>30.4<br>eyebrows<br>52.5<br>58.0<br>53.1<br>66.5<br>eyes<br>52.6<br>16.5<br>53.1<br>10.1<br>mouth<br>52.9<br>13.0<br>54.3<br>7.1<br>faceswap<br>50.0<br>0.0<br>51.6<br>0.0|
|Total<br>51.8<br>20.4<br>62.1<br>15.9|Total<br>50.8<br>13.2<br>59.6<br>10.6|Total<br>52.9<br>31.5<br>53.7<br>28.5|



(d) SeqDeepFake attributes [63] (e) SeqDeepFake comp [63] (f) R-Splicer 

||InstructBLIP[15]||
|---|---|---|
|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|
|Bangs<br>50.4<br>18.5<br>60.7<br>13.2<br>Eyeglasses54.3<br>75.9<br>65.7<br>90.3<br>Beard<br>50.0<br>0.1<br>61.0<br>0.1<br>Smiling<br>50.0<br>0.0<br>58.3<br>0.0<br>Young<br>50.0<br>0.0<br>62.3<br>0.0|nose<br>50.0<br>0.0<br>55.4<br>0.0<br>eye<br>57.3<br>39.1<br>70.3<br>31.2<br>eyebrow<br>54.4<br>21.0<br>60.8<br>13.0<br>lip<br>50.0<br>0.0<br>67.6<br>0.0<br>hair<br>50.1<br>38.4<br>48.8<br>44.4|nose<br>53.4<br>59.0<br>53.6<br>64.6<br>eyebrows<br>52.7<br>31.5<br>53.3<br>22.9<br>eyes<br>51.0<br>8.5<br>52.0<br>4.7<br>mouth<br>57.0<br>41.6<br>56.5<br>32.3<br>faceswap<br>50.0<br>1.2<br>51.6<br>0.6|
|Total<br>50.9<br>31.5<br>61.6<br>34.5|Total<br>52.4<br>32.8<br>60.6<br>29.5|Total<br>52.8<br>28.4<br>53.4<br>25.0|
|(g) SeqDeepFake attributes [63]|(h) SeqDeepFake comp [63]|(i) R-Splicer|



## InstructBLIP [15] with T5-xxl [13] LLM 

|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|
|---|---|---|
|Bangs<br>52.2<br>38.1<br>61.4<br>27.3<br>Eyeglasses58.2<br>73.0<br>67.5<br>77.8<br>Beard<br>57.1<br>74.3<br>64.7<br>86.7<br>Smiling<br>54.7<br>36.4<br>61.1<br>25.2<br>Young<br>49.5<br>74.1<br>62.1<br>92.6|nose<br>50.0<br>3.2<br>55.4<br>1.7<br>eye<br>48.8<br>76.2<br>65.9<br>91.0<br>eyebrow<br>49.2<br>71.8<br>57.4<br>96.0<br>lip<br>49.5<br>3.8<br>67.5<br>2.0<br>hair<br>52.2<br>65.4<br>49.9<br>94.9|nose<br>49.6<br>67.6<br>51.6<br>98.2<br>eyebrows<br>50.1<br>67.9<br>51.8<br>98.4<br>eyes<br>50.2<br>67.7<br>51.5<br>98.9<br>mouth<br>50.0<br>68.0<br>51.9<br>98.4<br>faceswap<br>50.4<br>68.2<br>51.8<br>99.8|
|Total<br>54.3<br>59.2<br>63.3<br>61.9|Total<br>49.9<br>44.1<br>59.2<br>57.1|Total<br>50.1<br>67.9<br>51.7<br>98.7|



(j) SeqDeepFake attributes [63] (k) SeqDeepFake comp [63] (l) R-Splicer 

LlaVa-1.5 [43] 

23 

Table 11: BLIP-2 [35] performance with _contains_ and _CLIP_ matching in open-ended VQA 

|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|
|---|---|
|Bangs<br>53.9<br>33.7<br>62.5<br>22.2<br>Eyeglasses<br>56.4<br>40.3<br>67.0<br>28.0<br>Beard<br>50.1<br>2.1<br>61.1<br>1.1<br>Smiling<br>49.9<br>1.0<br>58.3<br>0.5<br>Young<br>44.6<br>24.9<br>60.3<br>18.2|Bangs<br>56.0<br>73.5<br>64.1<br>99.9<br>Eyeglasses<br>60.0<br>75.9<br>69.6<br>99.9<br>Beard<br>59.9<br>74.1<br>66.8<br>99.9<br>Smiling<br>50.2<br>72.0<br>56.4<br>99.9<br>Young<br>51.6<br>75.1<br>61.2<br>100.0|
|Total<br>51.0<br>20.4<br>61.8<br>14.0|Total<br>55.5<br>74.1<br>63.6<br>99.9|



(a) SeqDeepFake attributes [63] with _contains_ matching (b) SeqDeepFake attributes [63] with _CLIP_ matching 

|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|
|---|---|
|nose<br>49.4<br>20.1<br>55.2<br>13.2<br>eye<br>50.1<br>10.5<br>66.6<br>6.0<br>eyebrow<br>49.9<br>2.2<br>57.7<br>1.1<br>lip<br>50.6<br>11.6<br>68.0<br>6.5<br>hair<br>52.6<br>29.3<br>50.3<br>19.8|nose<br>51.6<br>69.0<br>54.8<br>100.0<br>eye<br>49.8<br>77.3<br>64.1<br>99.8<br>eyebrow<br>50.4<br>70.9<br>55.7<br>100.0<br>lip<br>50.0<br>78.2<br>66.2<br>100.0<br>hair<br>49.7<br>63.2<br>46.8<br>99.8|
|Total<br>50.5<br>14.7<br>59.5<br>9.3|Total<br>50.3<br>71.7<br>57.5<br>99.9|



(c) SeqDeepFake comp. [63] with _contains_ matching (d) SeqDeepFake comp. [63] with _CLIP_ matching 

|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|
|---|---|
|nose<br>55.8<br>31.6<br>55.9<br>20.5<br>eyebrows<br>50.4<br>2.5<br>52.0<br>1.2<br>eyes<br>54.0<br>37.3<br>53.7<br>27.8<br>mouth<br>52.7<br>20.5<br>53.7<br>12.4<br>faceswap<br>66.0<br>64.4<br>62.5<br>60.3|nose<br>54.4<br>66.6<br>58.5<br>99.9<br>eyebrows<br>51.7<br>66.6<br>52.7<br>99.9<br>eyes<br>53.5<br>66.3<br>53.7<br>100.0<br>mouth<br>52.8<br>66.8<br>54.1<br>100.0<br>faceswap<br>55.1<br>66.4<br>57.1<br>100.0|
|Total<br>55.8<br>31.3<br>55.6<br>24.5|Total<br>53.5<br>66.5<br>55.2<br>100.0|



(e) R-splicer with _contains_ matching 

(f) R-splicer with _CLIP_ matching 

24 

Table 12: InstructBLIP [15] performance with _contains_ and _CLIP_ matching in open-ended VQA 

|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|
|---|---|
|Bangs<br>51.3<br>9.1<br>61.1<br>5.2<br>Eyeglasses<br>50.5<br>3.0<br>63.7<br>1.6<br>Beard<br>50.2<br>1.2<br>61.1<br>0.6<br>Smiling<br>50.0<br>1.2<br>58.3<br>0.6<br>Young<br>50.0<br>76.8<br>62.3<br>99.9|Bangs<br>49.6<br>73.6<br>58.5<br>99.6<br>Eyeglasses<br>47.2<br>75.7<br>60.5<br>99.1<br>Beard<br>58.0<br>73.9<br>64.7<br>99.3<br>Smiling<br>49.2<br>72.0<br>56.0<br>99.5<br>Young<br>50.5<br>74.9<br>60.0<br>99.5|
|Total<br>50.4<br>18.3<br>61.3<br>21.6|Total<br>50.9<br>74.0<br>59.9<br>99.4|
|(a) SeqDeepFake attributes [63] with_contains_matching|(b) SeqDeepFake attributes [63] with_CLIP_matching|
|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|
|nose<br>50.1<br>0.9<br>55.5<br>0.4<br>eye<br>50.0<br>0.1<br>66.4<br>0.1<br>eyebrow<br>50.0<br>0.0<br>57.7<br>0.0<br>lip<br>49.7<br>0.4<br>67.5<br>0.2<br>hair<br>50.4<br>18.8<br>49.0<br>16.4|nose<br>49.2<br>69.0<br>52.0<br>99.9<br>eye<br>46.9<br>77.3<br>61.7<br>99.8<br>eyebrow<br>49.5<br>70.8<br>54.1<br>99.9<br>lip<br>49.5<br>78.2<br>63.6<br>99.9<br>hair<br>49.6<br>63.3<br>46.1<br>99.9|
|Total<br>50.0<br>5.1<br>59.2<br>4.3|Total<br>48.9<br>71.7<br>55.5<br>99.9|



(c) SeqDeepFake comp. [63] with _contains_ matching (d) SeqDeepFake comp. [63] with _CLIP_ matching 

|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|
|---|---|
|nose<br>54.0<br>16.8<br>55.2<br>9.3<br>eyebrows<br>50.4<br>3.3<br>52.1<br>1.7<br>eyes<br>50.4<br>9.3<br>51.6<br>5.4<br>mouth<br>53.0<br>19.4<br>54.0<br>11.8<br>faceswap<br>53.5<br>68.7<br>53.4<br>96.4|nose<br>50.8<br>66.6<br>52.0<br>99.9<br>eyebrows<br>49.2<br>66.6<br>49.3<br>99.9<br>eyes<br>49.9<br>66.3<br>49.2<br>99.9<br>mouth<br>51.0<br>66.8<br>51.0<br>100.0<br>faceswap<br>41.3<br>66.4<br>45.3<br>100.0|
|Total<br>52.3<br>23.5<br>53.2<br>24.9|Total<br>48.5<br>66.5<br>49.3<br>99.9|



(e) R-splicer with _contains_ matching (f) R-splicer with _CLIP_ matching 

25 

Table 13: InstructBLIP [15] with T5-xxl LLM [13] performance, with _contains_ and _CLIP_ matching in open-ended VQA 

|Table 13: InstructBLIP [15] with T5-xxl LLM [1<br>in open-ended VQA|3] performance, with_contains_and_CLIP_matchin|
|---|---|
|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|
|Bangs<br>54.4<br>27.4<br>62.9<br>16.7<br>Eyeglasses<br>60.6<br>65.6<br>69.1<br>59.7<br>Beard<br>51.0<br>6.3<br>61.6<br>3.3<br>Smiling<br>51.9<br>11.5<br>59.5<br>6.2<br>Young<br>50.0<br>76.8<br>62.3<br>100.0|Bangs<br>45.8<br>46.3<br>56.2<br>40.7<br>Eyeglasses<br>56.1<br>65.3<br>65.7<br>65.9<br>Beard<br>55.0<br>60.3<br>63.4<br>59.2<br>Smiling<br>46.3<br>55.0<br>54.7<br>56.0<br>Young<br>50.2<br>50.6<br>61.9<br>43.7|
|Total<br>53.6<br>37.5<br>63.1<br>37.2|Total<br>50.7<br>55.5<br>60.4<br>53.1|



|(a) SeqDeepFake attributes [63] with_contains_matching|(b) SeqDeepFake attributes [63] with_CLIP_matching|
|---|---|
|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|
|nose<br>50.1<br>0.9<br>55.5<br>0.4<br>eye<br>50.0<br>0.1<br>66.4<br>0.1<br>eyebrow<br>50.0<br>0.0<br>57.7<br>0.0<br>lip<br>49.7<br>0.4<br>67.5<br>0.2<br>hair<br>50.4<br>18.8<br>49.0<br>16.4|nose<br>54.8<br>56.2<br>56.7<br>56.6<br>eye<br>58.2<br>57.7<br>68.5<br>49.4<br>eyebrow<br>51.2<br>56.1<br>55.1<br>56.8<br>lip<br>60.1<br>71.9<br>70.7<br>76.4<br>hair<br>54.1<br>57.3<br>48.3<br>70.4|
|Total<br>50.0<br>5.1<br>59.2<br>4.3|Total<br>55.6<br>59.8<br>59.9<br>61.9|



|(c) SeqDeepFake comp. [63] with_contains_matching|(d) SeqDeepFake comp. [63] with_CLIP_matching|
|---|---|
|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|
|nose<br>53.1<br>18.4<br>54.0<br>10.6<br>eyebrows<br>50.5<br>7.5<br>52.1<br>4.0<br>eyes<br>52.1<br>38.5<br>52.5<br>29.9<br>mouth<br>53.7<br>29.4<br>54.2<br>19.2<br>faceswap<br>59.5<br>61.9<br>57.1<br>63.7|nose<br>53.5<br>57.2<br>53.5<br>63.7<br>eyebrows<br>52.7<br>59.1<br>51.9<br>69.6<br>eyes<br>51.4<br>57.2<br>50.7<br>66.0<br>mouth<br>54.3<br>62.2<br>53.7<br>78.3<br>faceswap<br>58.0<br>65.7<br>55.5<br>90.1|
|Total<br>53.8<br>31.1<br>54.0<br>25.5|Total<br>54.0<br>60.3<br>53.1<br>73.5|



(e) R-splicer with _contains_ matching (f) R-splicer with _CLIP_ matching 

26 

Table 14: LlaVa-1.5 [43] performance, with _contains_ and _CLIP_ matching in open-ended VQA 

|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|
|---|---|
|Bangs<br>48.9<br>56.9<br>59.9<br>55.9<br>Eyeglasses<br>53.6<br>39.0<br>65.2<br>27.4<br>Beard<br>52.5<br>16.7<br>62.5<br>9.5<br>Smiling<br>51.1<br>12.1<br>59.0<br>7.1<br>Young<br>49.3<br>75.3<br>62.0<br>96.1|Bangs<br>46.4<br>73.5<br>56.9<br>99.9<br>Eyeglasses<br>51.9<br>75.9<br>65.9<br>100.0<br>Beard<br>57.9<br>74.1<br>64.0<br>100.0<br>Smiling<br>52.7<br>72.0<br>59.2<br>100.0<br>Young<br>47.5<br>75.1<br>58.8<br>100.0|
|Total<br>51.1<br>40.0<br>61.7<br>39.2|Total<br>51.3<br>74.1<br>61.0<br>100.0|



(a) SeqDeepFake attributes [63] with _contains_ matching (b) SeqDeepFake attributes [63] with _CLIP_ matching 

|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|
|---|---|
|nose<br>48.8<br>19.7<br>54.9<br>12.7<br>eye<br>49.7<br>3.2<br>66.4<br>1.6<br>eyebrow<br>49.9<br>7.1<br>57.7<br>3.8<br>lip<br>50.3<br>12.3<br>67.8<br>6.8<br>hair<br>49.3<br>43.2<br>48.4<br>43.9|nose<br>51.9<br>68.9<br>53.6<br>99.9<br>eye<br>50.0<br>77.4<br>63.1<br>100.0<br>eyebrow<br>48.7<br>70.8<br>54.1<br>100.0<br>lip<br>50.8<br>78.2<br>65.3<br>100.0<br>hair<br>46.8<br>63.3<br>44.5<br>100.0|
|Total<br>49.6<br>17.1<br>59.0<br>13.8|Total<br>49.6<br>71.7<br>56.1<br>100.0|



(c) SeqDeepFake comp. [63] with _contains_ matching (d) SeqDeepFake comp. [63] with _CLIP_ matching 

|AUC<br>F1<br>mAP<br>Recall|AUC<br>F1<br>mAP<br>Recall|
|---|---|
|nose<br>57.6<br>34.8<br>57.4<br>22.5<br>eyebrows<br>53.5<br>19.5<br>54.3<br>11.2<br>eyes<br>57.4<br>43.2<br>56.2<br>33.3<br>mouth<br>55.9<br>36.7<br>55.8<br>25.6<br>faceswap<br>68.8<br>73.8<br>63.6<br>84.6|nose<br>56.8<br>66.6<br>60.0<br>100.0<br>eyebrows<br>54.5<br>66.6<br>56.2<br>100.0<br>eyes<br>56.5<br>66.3<br>56.0<br>100.0<br>mouth<br>55.0<br>66.8<br>55.6<br>100.0<br>faceswap<br>60.7<br>66.4<br>59.3<br>100.0|
|Total<br>58.7<br>41.6<br>57.4<br>35.5|Total<br>56.7<br>66.5<br>57.4<br>100.0|



(e) R-splicer with _contains_ matching 

(f) R-splicer with _CLIP_ matching 

27 

