Published as a conference paper at ICLR 2025 

## VCR: VISUAL CAPTION RESTORATION 

**Tianyu Zhang[1,2,3*] , Suyuchen Wang[1,3*] , Lu Li[4] , Ge Zhang[5] , Perouz Taslakian[2] , Sai Rajeswar[2] , Jie Fu[6] , Bang Liu[1,3] , Yoshua Bengio[1,3]** 

1 Mila, Qu´ebec AI Institute 2 ServiceNow Research 3 Universit´e de Montr´eal 

4 University of Pennsylvania 5 University of Waterloo 6 HKUST 

_{_ tianyu.zhang, yoshua.bengio _}_ @mila.quebec 

_{_ suyuchen.wang, bang.liu _}_ @umontreal.ca 

luli1@upenn.edu ge.zhang@uwaterloo.ca 

_{_ perouz.taslakian, sai.rajeswar _}_ @servicenow.com jiefu@ust.hk 

ABSTRACT 

We introduce Visual Caption Restoration (VCR), a novel vision-language task that challenges models to accurately restore partially obscured texts using pixel-level hints within images. This task stems from the observation that text embedded in images intrinsically differs from common visual elements and natural language due to the need to align the modalities of vision, text, and text embedded in images. While numerous works have integrated text embedded in images into visual question-answering tasks, approaches to these tasks generally rely on optical character recognition or masked language modeling, thus reducing the task to mainly text-based processing. However, text-based processing becomes ineffective in VCR as accurate text restoration depends on the combined information from provided images, context, and subtle cues from the tiny exposed areas of masked texts. We develop a pipeline to generate synthetic images for the VCR task using image-caption pairs, with adjustable caption visibility to control the task difficulty. With this pipeline, we construct a dataset for VCR called VCR-WIKI using images with captions from Wikipedia, comprising 2 _._ 11 _M_ English and 346 _K_ Chinese entities in both _easy_ and _hard_ configurations. Our results reveal that current vision language models significantly lag behind human performance in the VCR task, and merely fine-tuning the models on our dataset does not lead to notable improvements. We release VCR-WIKI and the data construction code to facilitate future research. 

## 1 INTRODUCTION 

Recent advances in large language models, such as ChatGPT (Ouyang et al., 2022; OpenAI et al., 2023) and Llama (Touvron et al., 2023), have spurred significant interest and progress in the field of vision-language models. With models like GPT4V (OpenAI et al., 2023) and LLaVA (Liu et al., 2023a; 2024a; 2023b) blending textual and visual information, the intersection of computer vision and natural language processing has become a vibrant research frontier. These integrated models aim to leverage the potential of vision and language modalities to understand and interpret multimedia content more effectively. 

Amidst this evolving landscape, we introduce VCR, a novel vision-language task designed to challenge existing models uniquely. VCR challenges these models to restore obscured texts within images, which demands an intricate synthesis of text, vision, and text embedded in the image. The VCR task is Figure 1: An example of the VCR grounded in two key insights: (1) text embedded within images, task. with its characteristics different from common visual elements, represents a distinct modality that requires careful alignment of vision, textual data, and the structure of written texts, and (2) neuroscience findings that suggest that humans are proficient in recognizing partially occluded objects through sophisticated visual and cognitive processes (Thines et al., 2013; Pessoa et al., 1998; van´ 

> _∗_ Equal contribution. 

1 

Published as a conference paper at ICLR 2025 

Lier & Gerbino, 2015; Fyall et al., 2017; Li et al., 2023a). By leveraging these insights, VCR seeks to explore how well vision-language models can handle texts embedded within images, aligning visual elements and natural language to mimic human-like multimodal understanding and recognition. 

The Visual Question Answering (VQA) task (Antol et al., 2015; Wang et al., 2018; Mishra et al., 2019b; Singh et al., 2019) has been a popular benchmark in assessing how well models align and interpret visual and linguistic information. Traditional VQA approaches, however, predominantly focus on direct queries about visible elements in images and do not address the nuanced relationship between textual content embedded within the image and the overall image context. This gap underscores the limited capabilities of current models in processing integrated visual-textual data, particularly when the textual component, which plays a critical role, is partially obscured or altered. 

To address these limitations, our VCR task builds on the premise that effective text restoration from images requires an integrated understanding beyond the capabilities of current VQA benchmarks. For example, in extreme cases, models rely on existing Optical Character Recognition (OCR) system to extract text from documents (Singh et al., 2019; Borisyuk et al., 2018). The extracted text is then used as context for generating answers without a true semantic alignment between the text and the visual elements of the document. This approach, while effective in simple scenarios, falls short in more complex settings where text is intricately woven into the visual narrative of the image. 

To develop the VCR task, in this work, we introduce a pipeline for generating synthetic images that allows for manipulation of the visibility of the textual components of the image. This not only enhances the challenge posed by the task but also provides a scalable way to adjust task difficulty. The resulting dataset, VCR-WIKI, comprises 2.11M English data and 346K Chinese data sourced from Wikipedia, featuring captions in both languages across ‘easy’ and ‘hard’ difficulty levels. Our evaluations indicate that existing vision-language models significantly underperform compared to human benchmarks, underscoring the need for novel model architectures and training paradigms specifically geared towards this complex intermodal alignment. 

By releasing VCR-WIKI and the accompanying dataset construction code, we aim to stimulate further research in this area, encouraging the development of models that can more adeptly navigate the nuanced landscape of the restoration of text embedded in images. This effort aligns with the broader goal of advancing vision-language models to achieve a deeper, more intuitive understanding of multimedia content, bridging the gap between human and machine perception. The code is available at https://github.com/tianyu-z/VCR. 

**Contributions** The main contributions of this paper are: 

- **C1** Introduce the VCR task to challenge vision-language models to restore occluded texts in images. 

- **C2** Develop a pipeline for generating synthetic images with embedded text that allows for adjusting the visibility of such text, thus providing a rich testing environment for VCR. 

- **C3** Create and release VCR-WIKI, a dataset with multilingual captions from Wikipedia images, designed to benchmark vision-language models (VLMs) on text restoration tasks. 

- **C4** Conduct empirical evaluations that show significant gaps between current models and human performance on the VCR task. This highlights the effectiveness of VCR for assessing advancements in VLMs, and underscores the necessity for innovative model architectures and training techniques. 

## 2 VCR TASK DESCRIPTION 

In this section, we compare the VCR task with other existing tasks and aim to answer the following questions: 

**Q1** What is the difference between VCR and other visual reconstruction tasks? 

**Q2** Why should we care about VCR? 

For better clarity, we define _text embedded in image (TEI)_ as text incorporated within the image. The term _visual image (VI)_ pertains to the portion of the image that excludes the text embedded in the image. The _string text (ST)_ is not part of the image itself but is associated with it as a distinct 

2 

Published as a conference paper at ICLR 2025 

textual element. It is usually the question prompt in the form of natural language, for example, ‘What are the covered texts in the image? Please only guess the covered texts without outputting an explanation.’. Consequently, an element of a VCR task can be expressed as ( _ST,_ ( _V I, TEI_ )), where _ST_ is represented as a string and both _V I_ and _TEI_ are presented in image form. This notation does not imply that _V I_ and _TEI_ can be physically separated into two distinct image components. Instead, this definition is adopted merely to facilitate a clearer explanation of the concepts involved. Please refer to Figure 3 for an illustration of _V I_ , _TEI_ , and _ST_ . 

**A1** Existing tasks that are similar to VCR are the tasks of VQA and OCR. VQA takes as input images and a natural language question and generates a free-form response. As the ground-truth response is not unique, evaluating VQA poses a major challenge. In contrast to VQA, OCR is a task where the ground-truth responses are unique: OCR takes as input complete characters in image form and outputs a string representing the characters in the image, without considering the image context. Models pretrained with OCR are able to retrieve texts embedded in the input image, even if they are incomplete or vague. However, as the vagueness or occlusion of the textual components of the image increases, retrieving the original text without considering the remaining nontextual image context becomes harder, and OCR is no longer a good approach. VCR bridges the gap between OCR and VQA: it reconstructs the unique text found in the image while also considering the visual context of the rest of the image. 

Figure 3 is an example VCR task in hard mode, and Figure 1 shows an example VCR task in the easy mode. Although humans can still fill the blanks easily in the hard mode, it is nearly impossible for models with only OCR capabilities to recover the covered texts without using the context. This is because the pixel-level hints of single characters no longer correspond to a unique solution. 

## **A2** The proposed VCR task is significant in two aspects. 

The first aspect of importance stems from discoveries in neuroscience about human cognitive abilities to recognize partially occluded objects (Fyall et al., 2017; Li et al., 2023a). Although existing models can recognize objects and texts in images, they often struggle with the complexity of occluded objects due to significant information loss in the images. In contrast, humans excel at filling in missing information using a combination of low-level visual processing and high-level cognitive functions, such as those managed by the prefrontal cortex. This cortical area is known to handle complex cognitive processes such as decision making and memory retention, which are essential for integrating fragmented visual input into coherent objects. We believe that the occlusion restoration task serves as a probe that can effectively distinguish low-level recognition and high-level cognition involving reasoning. In addition, understanding these neural mechanisms can inspire new algorithms capable of mimicking human-like perception and interpretation in dynamic, real-world conditions where occlusion is common. 

The second aspect underscores the unique challenge presented by the VCR task, distinguishing it significantly from existing benchmarks, such as traditional VQA or the occluded object restoration task. By occluding texts instead of common visual objects, VCR targets the models’ text-image alignment capability, which is one of the major challenges for vision-language models. VCR mandates that models align textual and visual information in a manner that replicates human-like understanding involving the utilization of both 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_VCR_Visual_Caption_Restoration_images/iclr_2025_VCR_Visual_Caption_Restoration.pdf-0003-07.png)


Figure 2: How humans would possibly solve a VCR task. 

textual and visual clues. This task requires a deep integration of visual ( _V I_ ), embedded textual ( _TEI_ ), and contextual interpretation across modalities, pushing beyond simple text extraction as performed in OCR tasks. In OCR, the focus is primarily on recognizing visible characters, often without the need to understand their contextual relevance within the image narrative. In contrast, VCR introduces complexity by requiring the model to use available partial texts and the visual context collaboratively to reconstruct the obscured content accurately. This not only tests the model’s ability 

3 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_VCR_Visual_Caption_Restoration_images/iclr_2025_VCR_Visual_Caption_Restoration.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Wikipedia Images +  String text ( 𝑺𝑺𝑻𝑻 )<br>Captions Step 2: Text Processing Step 3: Create  𝑻 𝑻𝑻 "What are the covered texts in the<br>…… Republic Airport is a regional airport in East  (Republic Airport is a regional) Republic Airport is a regional airport in East  image? Please restore the covered texts without outputting the explanations."<br>Farmingdale, New York,  (Airport is a regional airport) Farmingdale, New York,<br>国鉄 80 系電車 … Republic Air… located one mile east of Farmingdale village limits. (is a Regional airport in East)…… located one mile east of Farmingdale village limits. Visual image ( 𝑽𝑽𝑻𝑻 )<br>Language  Step 1: Original Text Find eligible  𝒏𝒏 -grams Easy (less obscured)<br>Safety  Data Filtering Remove texts where all<br>𝑛𝑛-grams contain: Republic Airport is a  Republic Airport is a<br>Image-Caption PairsHigh-Quality ……  PunctuationsDigitsPerson, organization, location, date, time… regional airport in East Farmingdale, New York, located one Farmingdale village mile east of limits. regional airport in East Farmingdale, New York, located one mile east of Farmingdale village limits.<br>Ginger is… Republic Air… Filter instances without masked  𝒏𝒏 -grams Keep masking below 50%Find  𝒏𝒏 -grams to mask Hard (more obscured) Text embedded in image ( 𝑻 𝑻𝑻 )<br>**----- End of picture text -----**<br>


Figure 3: Illustration of the dataset creation pipeline for VCR-WIKI. visual image ( _V I_ ), text embedded in image ( _TEI_ ) and string text ( _ST_ ) in an example of the English Hard configuration of VCR-WIKI. The solid line-enclosed contents ( _V I_ and _TEI_ ) are part of the image, whereas the dotted line-enclosed content ( _ST_ ) is given separately from the image. 

to process _TEI_ - _V I_ modalities effectively but also challenges it to maintain internal consistency, akin to human cognitive processes where context and visual clues guide understanding and response. Besides, the difficulty of the task can be adjusted by altering the extent of text occlusion, offering a scalable and flexible framework for systematically enhancing model capabilities in text-visual alignment and semantic comprehension. This rigorous testing ground will help evolve vision-language models to better grasp the nuanced interplay between text and imagery. We show an example of how humans would solve the VCR task in ‘hard’ difficulty in Figure 2. 

## 3 DATASET CREATION 

The VCR task requires aligning visual images ( _V I_ ) with text embedded in images ( _TEI_ ). Therefore, the dataset creation process relies on a set of highly correlated image-text pairs. We utilize the primary images and their corresponding captions from Wikipedia as the data source[1] to create VCR-WIKI, a Wikipedia-based VCR dataset. The pipeline for creating VCR-WIKI is shown in Figure 3. Before constructing the dataset, we first filter out instances with sensitive content, including NSFW and crime-related terms, to mitigate AI risk and biases. 

The VCR-WIKI dataset is formatted as a VQA task, where each instance includes an image, a question, and a ground-truth answer. The images are synthesized from text-image pairs by stacking the image ( _V I_ ) with its corresponding text description ( _TEI_ ) vertically, mimicking the format of a captioned image. This stacked image is referred to as a stacked _V I_ + _TEI_ image. Each _V I_ + _TEI_ image is resized to a width of 300 pixels. To avoid excessive image height, we truncate _TEI_ to a maximum of five lines. We filter the dataset to exclude instances with _V I_ + _TEI_ images exceeding 900 pixels in height to avoid drastic resolution changes during data pre-processing. 

Within _TEI_ , we use spaCy to randomly select several 5-grams in the caption for masking. To ensure the restoration process is doable by a human without too much professional domain knowledge, the 5-grams do not contain numbers, person names, religious or political groups, facilities, organizations, locations, dates, and time labeled by spaCy, and the total masked token does not exceed 50% of the tokens in the caption. We exclude all instances that do not have a single eligible 5-gram. The selected 5-grams are partially obscured by a white rectangle that reveals only the upper and lower parts of the text, with the proportion of coverage varying to adjust task difficulty. Furthermore, to assess the impact of _V I_ on model performance, we create an ablation for each image, maintaining the resolution of the _V I_ + _TEI_ image, but retaining only the _TEI_ part in the center of the image. 

The VCR task involves a predefined question that prompts the model to produce the obscured text in the image. The ground truth answer corresponds to the caption displayed in the uncovered portion of the stacked image. Due to the extensive availability of vision-language models and a significant user 

> 1Datasource: https://huggingface.co/datasets/wikimedia/wit_base. 

4 

Published as a conference paper at ICLR 2025 

base in both English and Chinese, we have chosen to develop the dataset in these two languages. For each language, we meticulously select the covering proportion to create two task variants: (1) an easy version, where the task is easy for native speakers but open-source OCR models almost always fail, and (2) a hard version, where the revealed part consists of only one to two pixels for the majority of letters or characters, yet the restoration task remains feasible for native speakers of the language. 

We release the dataset under the CC BY-SA 4.0 license. All configurations of VCRWIKI are publicly available at https://huggingface.co/collections/vcr-org/ vcr-visual-caption-recognition-6661393b1761e2aff7b967b9. 

## 3.1 DATASET FORMAT AND STATISTICS 

Table 1: Basic statistics of the dataset. Note that the Easy and Hard configurations for each language share the same statistics. We report the mean, standard deviation, and the 5[th] and 95[th] percentile ( _η._ 5 and _η._ 95) for the stacked image height and the number of obscured text spans. Unit is in pixels. 

||# Train<br># Val<br># Test|_V I_ +_TEI_ Image Height|# Obscured Text Spans|
|---|---|---|---|
|||Mean<br>SD<br>_η._5<br>_η._95|Mean<br>SD<br>_η._5<br>_η._95|
|English<br>Chinese|2095733<br>5000<br>5000<br>336448<br>5000<br>5000|375.52<br>106.01<br>253<br>564<br>360.44<br>102.76<br>239<br>562|1.62<br>0.63<br>1<br>3<br>2.06<br>0.94<br>1<br>4|



The VCR dataset comprises four configurations: English Easy, English Hard, Chinese Easy and Chinese Hard. Each configuration can be further divided into training, validation, and test splits. The validation and test splits contain 5,000 entities each. The training set for English configurations and Chinese configurations contains 2 _,_ 095 _,_ 733 and 336 _,_ 448 instances, respectively, which can be used for model continuous pretraining. We include more detailed statistics of the dataset in Table 1. 

## 4 EXPERIMENTS 

In this section, we report the experiment results of existing state-of-the-art vision-language models on our proposed VCR tasks. The fine-tuning and evaluation of open-source models are conducted on a mix of NVIDIA A100 80G and L40S 48G GPUs in an internal cluster. 

## 4.1 MODELS 

**Closed-source Models.** We evaluate several most advanced proprietary models with their provided APIs. The evaluated models include GPT-4o (gpt-4o-2024-0513), GPT-4 Turbo (gpt-4-turbo-2024-0409), GPT-4V (gpt-4-1106-vision-preview) (Ouyang et al., 2022; OpenAI et al., 2023), Claude 3 Opus (claude-3-opus-20240229), Claude 3.5 Sonnet (claude-3-5-sonnet-20240620) (Anthropic, 2024), Gemini 1.5 pro (gemini-1.5-pro-001) (Team et al., 2024a), Reka Core (reka-core-20240501) (Team et al., 2024b), and Qwen-VL-Max (tested in May 2024) (Bai et al., 2023). 

**Open-source Models.** We evaluate open-source models with the best performance on the OpenVLM Leaderboard[2] and state-of-the-art Chinese VLM models. The evaluated models include Cambrian-1 (Tong et al., 2024), CogVLM2-19B (Hong et al., 2024), DeepSeek-VL-7B-Chat (Lu et al., 2024), DocOwl-1.5-Omni (Hu et al., 2024a), Idefics3-8B (Laurenc¸on et al., 2024), InternLMXComposer2-VL-7B (Dong et al., 2024a), InternLM-XComposer2.5-VL (Zhang et al., 2024), InternLM-XComposer2-VL-4K (Dong et al., 2024b), Llama-3.2 (Dubey et al., 2024), MiniCPMV2.5 (Hu et al., 2024b), Monkey (Liu et al., 2024b; Li et al., 2023b), Pixtral (Mistral, 2024), Qwen-VL-Chat (Bai et al., 2023), Qwen2-VL (Wang et al., 2024), and Yi-VL (01.AI et al., 2024). Out of these models, Cambrian-1, Idefics3 and Llama-3.2 are English-only models, and CogVLM2Llama3-19B-Chat has its Chinese variant, CogVLM2-Llama3-19B-Chinese-Chat. Please refer to Table 6 for the model specifications. 

**Finetuned Models.** To test whether VLMs can learn to conduct VCR via fine-tuning, we select three models from the open-sourced models: CogVLM2-Llama3-19B-Chat, MiniCPM-Llama3-V2.5, and Qwen2-VL-7B-Instruct, and fine-tune them on a subset of VCR’s training set. 

> 2We selected the highest-performing open-source models with fewer than 40 billion parameters from the OpenVLM Leaderboard as of May 23, 2024 and their later versions. Details are available at https: //huggingface.co/spaces/opencompass/open_vlm_leaderboard. 

5 

Published as a conference paper at ICLR 2025 

More specifically, we fine-tune CogVLM2-Llama3-19B-Chat, MiniCPM-Llama3-V2.5, and Qwen2VL-7B-Instruct in the English Hard configuration, and CogVLM2-Llama3-19B-Chinese-Chat, MiniCPM-Llama3-V2.5, and Qwen2-VL-7B-Instruct on the Chinese Hard configuration. The models are finetuned using LoRA (Hu et al., 2022) with _r_ = 8 and _α_ = 32. We adopt the schedule-free AdamW optimizer (Defazio et al., 2024) with a learning rate 2 _e −_ 4. The effective batch size is 64. Each model is trained on the first 16,000 examples of the training set for 1 epoch. All fine-tuning experiments are performed on a single node with 4 NVIDIA L40S 48G GPUs. 

## 4.2 METRICS 

We measure the quality of the model’s restoration of each masked _n_ -gram (where _n_ = 5 in our setting, as specified in Section 3). Due to the variability of different models’ outputs, for each masked _n_ -gram _m ∈_ V _[n] e_[, where][ V] _[e]_[is the vocabulary of the evaluation tokenizer][3][, we extract the most similar] _n_ -gram _m_ ˆ _∈_ V _[n] e_[with the least edit distance in the model’s generation.] 

We report the two metrics below in our experiment section to measure the restoration quality: ˆ ˆ **Exact Match (** _EM_ **)** _≡ EM_ ( _m, m_ ) = I( _m_ = _m_ ), which measures whether the restored _n_ -gram _m_ similarity ofˆ totally matches the ground-truth _m_ ˆ and _m_ as bag-of-words. _m_ ; and **Jaccard Index (** _J_ **)** _≡ |[|] S[S]_ ([(] _m[m]_ )[)] _∪[∩] S[S]_ ( ˆ[(][ ˆ] _m[m]_ )[)] _|[|]_[, which measures the] 

## 4.3 RELATIONSHIP TO OTHER BENCHMARKS. 

We evaluated 38 Vision-Language Models (VLMs) across 23 different benchmarks, using the VLM performance scores as features of each benchmark to compute a correlation matrix. Based on this matrix, in Figure 4, we applied K-Means clustering and visualized the results in 2D by plotting the first two principal components derived from the correlation matrix rows for each benchmark. Additionally, we provide a heatmap of the correlation matrix in Appendix B. 

VCRZH, EASY and VCRZH, HARD were excluded from these processes due to the limited availability of VLMs that support Chinese. From Figure 4, we observe that VCREN, EASY shows a tentative similarity to ChartQA and TextVQA, as all three benchmarks evaluate the ability to extract and reason about text from natural images and documents. However, VCREN, EASY does not exhibit significant similarity to the other benchmarks. On the other hand, VCREN, HARD stands apart from all other benchmarks. We attribute this to the fact that VCREN, HARD emphasizes caption recovery with minimal pixel-level information, a skill not tested by any of the other benchmarks. Therefore, we assert that the VCR series benchmarks assess unique aspects of VLMs that are not covered by any of the other benchmarks in our evaluation. 

Figure 4: Projection of benchmarks onto the first two principal components derived from the correlation matrix of VLM performance scores. Each point represents a benchmark, and proximity indicates higher similarity based on model performance correlation. 

## 4.4 EXPERIMENTAL RESULTS 

Table 2 presents the exact match scores and Jaccard indices for our evaluations. Additionally, Figure 5 illustrates a comparative analysis of our fine-tuned models against their base counterparts across all four VCR settings. We also provide experiment results on smaller test sets of VCR-WIKI for faster benchmarking. Please refer to Table 4 and 5 in the Appendix. 

The VCR task and the VCR-WIKI dataset are specifically designed to advance the development of future VLMs. In this section, we conduct a detailed analysis of the performance of current state-ofthe-art (SOTA) models on the VCR task, highlighting key insights through comparative evaluations. 

> 3 We utilize spaCy’s en ~~c~~ ore ~~w~~ eb ~~s~~ m’s and zh ~~c~~ ore ~~w~~ eb ~~s~~ m’s tokenizer for English and Chinese evaluation, respectively. 

6 

Published as a conference paper at ICLR 2025 

**VCR Remains a Challenging Task for SOTA VLMs.** Despite observing high-performing models on the VCREN, EASY and VCREN, HARD settings, such as Qwen2-VL, the majority of recent models continue to struggle, particularly under the more demanding hard settings where performance metrics approach zero. This underscores the inherent difficulty of the VCR task and indicates that significant advancements are still required to achieve robust performance across all settings. 

**Enhanced OCR Capabilities Do Not Necessarily Translate to Improved VCR Performance.** Our analysis reveals that models proficient in OCR, such as InternLM-XComposer2-VL, and those excelling in image document understanding, like DocOwl 1.5 and Monkey, demonstrate subpar performance across most VCR settings. This discrepancy suggests that while these models can accurately recognize text within images, they lack the advanced reasoning capabilities required to effectively interpret and utilize this information within the context of the VCR task. This finding also highlights a fundamental distinction between OCR tasks and the more complex VCR task. 

**Language-Specific Performance: Need for Enhanced Multilingual Capabilities.** A significant performance degradation is observed when models are evaluated on Chinese configurations, despite assertions of basic English-Chinese bilingual capabilities. This decline is particularly surprising given the logographic nature of Chinese characters, which theoretically offer higher recognizability compared to alphabetic scripts (Wu et al., 2024; Zhao et al., 2022). These results indicate a critical need for targeted improvements in multilingual support to ensure consistent performance across different languages. 

**Model Size Does Not Guarantee Superior Performance.** Comparative analysis between Llama3.2-11B and Llama-3.2-90B models reveals that both exhibit similar performance levels on the VCREN, EASY and VCREN, HARD settings. This observation suggests that merely increasing model size does not inherently enhance VCR performance. Instead, advancements in the cognitive abilities of models, achieved through improved training strategies, reasoning frameworks, or architectural innovations, are essential for meaningful performance gains in VCR tasks. 

**Model Resolution Is Not Directly Correlated with Performance Enhancement.** InternLMXComposer2-VL-4K, despite its higher resolution, demonstrates significantly lower performance on the VCREN, EASY setting compared to its lower-resolution counterparts. This decline may result from more aggressive image partitioning strategies that disrupt the spatial continuity of text or from more intensive pixel or token compression techniques that lead to the loss of crucial local details. Both factors are critical for the accurate interpretation required in VCR tasks. 

**Inclusion of** _V I_ **Input Images Negatively Impacts Performance.** The addition of _V I_ generally results in negative performance changes (∆ _<_ 0), indicating that the image information is not being effectively leveraged by the models. This negative impact may stem from the importance of key information locations, which could be compromised by image partitioning strategies that fail to preserve spatial relationships essential for accurate reasoning. 

**Model Architecture Influences Performance Gains from VCR-WIKI Finetuning.** Finetuning on the VCR-WIKI dataset yields varying performance improvements across different model architectures. Specifically: 1) **CogVLM2** exhibits substantial performance enhancements across all four settings post-finetuning, suggesting that its architectural design is well-suited for the type of image-text reasoning required by the VCR task when provided with appropriate data and training. 2) **MiniCPMV2.5** shows only marginal performance increases from an already low baseline, particularly in the VCRZH, EASY and VCRZH, HARD settings. This limited improvement indicates potential architectural limitations that hinder its ability to effectively address the complexities of the VCR task. 3) **Qwen2VL-7B** maintains relatively high performance both before and after finetuning, implying that the model is sufficiently pre-trained on relevant tasks to perform well on the VCR task without extensive additional training. 

We hope that through controlled variables, the VCR-WIKI dataset delivers _fully comparable_ results across languages, difficulty levels, image inclusion, and fine-tuning stages. Each comparison is intended to guide specific and targeted improvements in VLM development. 

7 

Published as a conference paper at ICLR 2025 

Table 2: Performance of vision language models on the VCR task in English (EN) and Chinese (ZH), for easy and hard modes. We label the best result of each setting with **bold fonts** , and the best open-source model with underline. A superscript of * marks that the model was released after the initial public release of the VCR-WIKI dataset (June 10, 2024). Subscripts show bootstrapped standard deviation. 

|**Language**<br>English|**Mode**|**Open/closed**<br>**source**|**Model name**<br>**Model size**||**Exac**|**t match**(%)_↑_||**Jacca**|**rd index**(%)_↑_|
|---|---|---|---|---|---|---|---|---|---|
||||||_V I_ +_TEI_|_TEI_<br>∆||_V I_ +_TEI_|_TEI_<br>∆|
||Easy|Closed|Claude 3 Opus<br>-<br>Claude 3.5 Sonnet<br>-<br>Gemini 1.5 Pro<br>-<br>GPT-4 Turbo<br>-<br>GPT-4o<br>-<br>GPT-4V<br>-<br>Qwen-VL-Max<br>-<br>Reka Core<br>-||62_._00_._13<br>63_._851_._71<br>62_._731_._66<br>78_._740_._13<br>**91****_._550****_._29**<br>52_._040_._24<br>76_._80_._5<br>66_._461_._64|77_._00_._5<br>-15<br>72_._81_._56<br>-8.94<br>82_._981_._3<br>-20.25<br>81_._940_._25<br>-3.2<br>**94****_._560****_._13**<br>-3.01<br>37_._860_._22<br>14.17<br>85_._530_._19<br>-8.74<br>78_._511_._42<br>-12.05||77_._670_._32<br>74_._651_._33<br>77_._711_._21<br>88_._540_._24<br>**96****_._440****_._11**<br>65_._360_._39<br>85_._710_._28<br>84_._230_._86|88_._410_._39<br>-10.74<br>83_._481_._14<br>-8.83<br>91_._560_._76<br>-13.85<br>92_._180_._3<br>-3.65<br>**97****_._760****_._06**<br>-1.32<br>54_._130_._41<br>11.23<br>91_._450_._29<br>-5.74<br>90_._450_._7<br>-6.22|
|||Open|Cambrian-1*<br>34B<br>CogVLM2<br>19B<br>DeepSeek-VL<br>1.3B<br>DeepSeek-VL<br>7B<br>DocOwl-1.5-Omni<br>8B<br>Idefcs3*<br>8B<br>InternLM-XComposer2-VL<br>7B<br>InternLM-XComposer2-VL-4K<br>7B<br>InternLM-XComposer2.5-VL*<br>7B<br>InternVL-V2*<br>40B<br>Llama-3.2*<br>11B<br>Llama-3.2*<br>90B<br>MiniCPM-V2.5<br>8B<br>Monkey<br>7B<br>Pixtral*<br>12B<br>Qwen-VL<br>7B<br>Qwen2-VL*<br>7B<br>Yi-VL<br>34B<br>Yi-VL<br>6B||79_._690_._43<br>83_._250_._07<br>23_._040_._05<br>38_._010_._12<br>0_._840_._01<br>25_._990_._48<br>46_._640_._1<br>5_._320_._24<br>41_._350_._55<br>84_._670_._40<br>79_._850_._45<br>80_._540_._43<br>31_._810_._08<br>50_._660_._1<br>18_._410_._42<br>49_._710_._17<br>89_._700_._34<br>0_._820_._03<br>0_._750_._01|81_._280_._43<br>-1.59<br>78_._290_._04<br>4.96<br>31_._090_._12<br>-8.04<br>45_._940_._1<br>-7.93<br>1_._550_._02<br>-0.71<br>31_._430_._51<br>-5.44<br>46_._40_._11<br>0.24<br>3_._710_._21<br>1.60<br>25_._370_._51<br>15.97<br>87_._710_._37<br>-3.04<br>67_._530_._53<br>12.32<br>71_._050_._51<br>9.48<br>40_._050_._09<br>-8.25<br>56_._20_._08<br>-5.54<br>11_._600_._36<br>6.81<br>52_._150_._15<br>-2.44<br>93_._440_._26<br>-3.74<br>1_._610_._04<br>-0.79<br>1_._650_._01<br>-0.9||89_._270_._28<br>89_._750_._1<br>46_._840_._07<br>60_._020_._15<br>13_._340_._03<br>47_._220_._42<br>70_._990_._1<br>22_._140_._28<br>63_._040_._42<br>92_._640_._22<br>90_._580_._22<br>89_._810_._26<br>53_._240_._1<br>67_._60_._09<br>41_._250_._37<br>69_._940_._07<br>93_._840_._24<br>5_._590_._04<br>5_._540_._02|92_._540_._19<br>-3.27<br>88_._070_._08<br>1.68<br>52_._360_._06<br>-5.52<br>64_._720_._04<br>-4.7<br>14_._620_._04<br>-1.28<br>54_._000_._39<br>-6.78<br>72_._140_._07<br>-1.14<br>18_._780_._25<br>3.37<br>49_._950_._41<br>13.09<br>95_._100_._16<br>-2.47<br>81_._110_._33<br>9.47<br>84_._220_._30<br>5.59<br>63_._20_._1<br>-9.96<br>72_._820_._08<br>-5.22<br>31_._600_._33<br>9.65<br>72_._280_._08<br>-2.34<br>97_._470_._12<br>-3.62<br>7_._720_._03<br>-2.13<br>7_._760_._03<br>-2.22|
||Hard|Closed|Claude 3 Opus<br>-<br>Claude 3.5 Sonnet<br>-<br>Gemini 1.5 Pro<br>-<br>GPT-4 Turbo<br>-<br>GPT-4o<br>-<br>GPT-4V<br>-<br>Qwen-VL-Max<br>-<br>Reka Core<br>-||37_._80_._28<br>41_._741_._69<br>28_._071_._58<br>45_._150_._28<br>73_._20_._16<br>25_._830_._44<br>41_._650_._32<br>6_._710_._89|50_._00_._33<br>-12.2<br>44_._721_._78<br>-2.98<br>38_._761_._68<br>-10.68<br>48_._640_._57<br>-3.5<br>**82****_._430****_._17**<br>-9.22<br>14_._950_._3<br>10.87<br>52_._720_._2<br>-11.07<br>11_._181_._15<br>-4.47||57_._680_._8<br>56_._151_._46<br>51_._91_._22<br>65_._720_._25<br>**86****_._170****_._21**<br>44_._630_._48<br>61_._180_._35<br>25_._840_._95|70_._160_._64<br>-12.48<br>58_._541_._6<br>-2.4<br>59_._621_._27<br>-7.72<br>67_._860_._2<br>-2.14<br>**92****_._010****_._2**<br>-5.84<br>30_._080_._67<br>14.56<br>70_._190_._37<br>-9.01<br>35_._831_._05<br>-9.99|
|||Open|Cambrian-1*<br>34B<br>CogVLM2<br>19B<br>DeepSeek-VL<br>1.3B<br>DeepSeek-VL<br>7B<br>DocOwl-1.5-Omni<br>8B<br>Idefcs3*<br>8B<br>InternLM-XComposer2-VL<br>7B<br>InternLM-XComposer2-VL-4K<br>7B<br>InternLM-XComposer2.5-VL*<br>7B<br>InternVL-V2*<br>40B<br>Llama-3.2-Vision*<br>11B<br>Llama-3.2-Vision*<br>90B<br>MiniCPM-V2.5<br>8B<br>Monkey<br>7B<br>Pixtral*<br>12B<br>Qwen-VL<br>7B<br>Qwen2-VL*<br>7B<br>Yi-VL<br>34B<br>Yi-VL<br>6B||27_._200_._48<br>37_._980_._18<br>0_._160_._01<br>1_._00_._02<br>0_._040_._0<br>0_._600_._08<br>0_._70_._01<br>0_._210_._05<br>0_._930_._11<br>13_._100_._37<br>14_._090_._40<br>14_._910_._40<br>1_._410_._03<br>1_._960_._04<br>0_._440_._08<br>2_._00_._03<br>**74****_._320****_._47**<br>0_._070_._0<br>0_._060_._0|29_._680_._50<br>-2.48<br>17_._680_._06<br>20.3<br>0_._390_._02<br>-0.23<br>1_._750_._03<br>-0.75<br>0_._020_._0<br>0.01<br>0_._370_._07<br>0.23<br>0_._920_._01<br>-0.22<br>0_._180_._05<br>0.02<br>1_._110_._11<br>-0.18<br>19_._160_._44<br>-6.06<br>6_._920_._27<br>7.17<br>13_._060_._37<br>1.85<br>1_._960_._02<br>-0.55<br>2_._430_._03<br>-0.48<br>0_._640_._09<br>-0.20<br>2_._320_._03<br>-0.32<br>75_._200_._49<br>-0.88<br>0_._050_._0<br>0.02<br>0_._040_._0<br>0.02||50_._040_._40<br>59_._990_._05<br>11_._890_._02<br>15_._90_._08<br>7_._760_._01<br>10_._370_._15<br>12_._510_._02<br>9_._520_._12<br>13_._820_._16<br>33_._640_._36<br>35_._260_._36<br>35_._440_._35<br>11_._940_._02<br>14_._020_._03<br>10_._990_._13<br>15_._040_._05<br>85_._470_._30<br>4_._310_._02<br>4_._460_._02|55_._660_._39<br>-5.62<br>39_._690_._03<br>20.3<br>11_._470_._03<br>0.42<br>17_._20_._04<br>-1.3<br>7_._740_._02<br>0.03<br>9_._590_._13<br>0.79<br>13_._230_._02<br>-0.72<br>9_._520_._12<br>-0.00<br>14_._720_._18<br>-0.89<br>41_._350_._39<br>-7.71<br>26_._350_._29<br>8.90<br>34_._440_._35<br>1.00<br>13_._370_._04<br>-1.43<br>14_._110_._03<br>-0.09<br>11_._450_._15<br>-0.46<br>14_._270_._05<br>0.77<br>87_._630_._27<br>-2.15<br>5_._890_._02<br>-1.58<br>5_._910_._01<br>-1.46|
|Chinese|Easy|Closed|Claude 3 Opus<br>-<br>Claude 3.5 Sonnet<br>-<br>Gemini 1.5 Pro<br>-<br>GPT-4o<br>-<br>GPT-4 Turbo<br>-<br>Qwen-VL-Max<br>-<br>Reka Core<br>-||0_._90_._3<br>1_._00_._31<br>1_._10_._32<br>14_._871_._14<br>0_._20_._14<br>6_._340_._08<br>0_._00_._0|1_._00_._31<br>-0.1<br>0_._80_._28<br>0.2<br>0_._50_._22<br>0.6<br>22_._461_._35<br>-7.58<br>0_._10_._1<br>0.1<br>9_._920_._09<br>-3.58<br>0_._00_._0<br>0||11_._50_._48<br>7_._540_._54<br>11_._10_._56<br>39_._050_._99<br>8_._420_._36<br>13_._450_._41<br>3_._430_._26|10_._00_._49<br>1.49<br>7_._50_._51<br>0.03<br>11_._470_._48<br>-0.37<br>48_._241_._09<br>-9.19<br>6_._970_._29<br>1.45<br>22_._860_._46<br>-9.42<br>3_._150_._2<br>0.28|
|||Open|CogVLM2-Chinese<br>19B<br>DeepSeek-VL<br>1.3B<br>DeepSeek-VL<br>7B<br>DocOwl-1.5-Omni<br>8B<br>InternLM-XComposer2-VL<br>7B<br>InternLM-XComposer2-VL-4K<br>7B<br>InternLM-XComposer2.5-VL*<br>7B<br>InternVL-V2*<br>40B<br>MiniCPM-V2.5<br>8B<br>Monkey<br>7B<br>Qwen-VL<br>7B<br>Qwen2-VL*<br>7B<br>Yi-VL<br>34B<br>Yi-VL<br>6B||33_._240_._04<br>0_._00_._0<br>0_._00_._0<br>0_._00_._0<br>0_._270_._01<br>0_._460_._07<br>0_._460_._07<br>22_._090_._41<br>4_._10_._02<br>0_._620_._01<br>0_._040_._01<br>**59****_._940****_._49**<br>0_._00_._0<br>0_._00_._0|30_._70_._07<br>2.54<br>0_._00_._0<br>0<br>0_._00_._0<br>0<br>0_._00_._0<br>0<br>0_._230_._01<br>0.04<br>0_._460_._07<br>0.00<br>0_._580_._08<br>-0.12<br>17_._260_._39<br>4.84<br>5_._050_._08<br>-0.95<br>1_._440_._01<br>-0.82<br>0_._00_._0<br>0.04<br>**67****_._480****_._47**<br>-7.54<br>0_._00_._0<br>0<br>0_._00_._0<br>0||57_._570_._06<br>6_._560_._01<br>4_._080_._01<br>1_._140_._01<br>12_._320_._02<br>12_._310_._14<br>12_._970_._16<br>47_._620_._34<br>18_._030_._07<br>8_._340_._06<br>1_._50_._01<br>**76****_._950****_._32**<br>4_._440_._01<br>4_._370_._01|53_._660_._04<br>3.91<br>3_._170_._02<br>3.4<br>6_._840_._01<br>-2.76<br>3_._380_._01<br>-2.23<br>12_._280_._03<br>0.04<br>13_._370_._14<br>-1.05<br>14_._990_._17<br>-2.01<br>37_._930_._35<br>9.69<br>22_._940_._04<br>-4.9<br>10_._950_._03<br>-2.61<br>0_._340_._01<br>1.15<br>**82****_._630****_._28**<br>-5.67<br>1_._80_._01<br>2.64<br>1_._760_._0<br>2.6|
||Hard|Closed|Claude 3 Opus<br>-<br>Claude 3.5 Sonnet<br>-<br>Gemini 1.5 Pro<br>-<br>GPT-4o<br>-<br>GPT-4 Turbo<br>-<br>Qwen-VL-Max<br>-<br>Reka Core<br>-||0_._30_._18<br>0_._20_._15<br>0_._70_._26<br>2_._20_._47<br>0_._00_._0<br>0_._890_._06<br>0_._00_._0|0_._10_._1<br>0.2<br>0_._00_._0<br>0.2<br>0_._50_._23<br>0.2<br>1_._80_._4<br>0.4<br>0_._20_._13<br>-0.2<br>1_._380_._1<br>-0.49<br>0_._00_._0<br>0||9_._220_._38<br>4_._00_._33<br>11_._820_._51<br>22_._720_._67<br>8_._580_._3<br>5_._40_._19<br>3_._350_._23|8_._090_._33<br>1.13<br>2_._370_._23<br>1.63<br>11_._750_._44<br>0.07<br>22_._890_._65<br>-0.17<br>6_._870_._28<br>1.72<br>12_._290_._18<br>-6.89<br>2_._970_._2<br>0.38|
|||Open|CogVLM2-Chinese<br>19B<br>DeepSeek-VL<br>1.3B<br>DeepSeek-VL<br>7B<br>DocOwl-1.5-Omni<br>8B<br>InternLM-XComposer2-VL<br>7B<br>InternLM-XComposer2-VL-4K<br>7B<br>InternLM-XComposer2.5-VL*<br>7B<br>InternVL-V2*<br>40B<br>MiniCPM-V2.5<br>8B<br>Monkey<br>7B<br>Qwen-VL<br>7B<br>Qwen2-VL*<br>7B<br>Yi-VL<br>34B<br>Yi-VL<br>6B||1_._340_._03<br>0_._00_._0<br>0_._00_._0<br>0_._00_._0<br>0_._070_._01<br>0_._050_._02<br>0_._110_._04<br>0_._480_._07<br>0_._090_._0<br>0_._120_._01<br>0_._010_._0<br>**18****_._330****_._37**<br>0_._00_._0<br>0_._00_._0|2_._670_._02<br>-1.32<br>0_._00_._0<br>0<br>0_._00_._0<br>0<br>0_._00_._0<br>0<br>0_._090_._0<br>-0.02<br>0_._050_._02<br>0.00<br>0_._120_._04<br>-0.01<br>0_._740_._08<br>-0.26<br>0_._080_._0<br>0.01<br>0_._070_._0<br>0.05<br>0_._010_._0<br>0<br>**27****_._580****_._44**<br>-9.26<br>0_._00_._0<br>0<br>0_._00_._0<br>0||17_._350_._03<br>6_._460_._01<br>5_._110_._01<br>1_._370_._01<br>8_._970_._02<br>7_._670_._10<br>10_._950_._11<br>12_._570_._14<br>7_._390_._02<br>6_._360_._01<br>1_._170_._01<br>**43****_._550****_._34**<br>4_._120_._0<br>4_._00_._01|19_._510_._03<br>-2.16<br>3_._220_._02<br>3.24<br>7_._210_._01<br>-2.1<br>4_._070_._02<br>-2.7<br>8_._510_._01<br>0.46<br>7_._720_._10<br>-0.04<br>11_._430_._12<br>-0.48<br>13_._310_._15<br>-0.74<br>7_._890_._01<br>-0.5<br>6_._680_._03<br>-0.32<br>0_._120_._0<br>1.06<br>**54****_._240****_._34**<br>-10.69<br>1_._810_._01<br>2.31<br>1_._880_._01<br>2.12|



8 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_VCR_Visual_Caption_Restoration_images/iclr_2025_VCR_Visual_Caption_Restoration.pdf-0009-01.png)


**----- Start of picture text -----**<br>
Model Type<br>Original Fine-tuned<br>100 93.27 VCR_EN_EASY 89.70 94.43 VCR_EN_HARD VCR_ZH_EASY VCR_ZH_HARD<br>83.25 84.15<br>80 77.44 74.32<br>60 61.69 59.94 60.55<br>40 31.81 40.96 37.98 33.24 42.11<br>20 13.86 18.33 18.67<br>0 1.41 4.10 7.44 1.34 0.09 1.53<br>CogVLM2 19B MiniCPM-V2.5 8B Qwen2-VL 7B CogVLM2 19B MiniCPM-V2.5 8B Qwen2-VL 7B CogVLM2 19B MiniCPM-V2.5 8B Qwen2-VL 7B CogVLM2 19B MiniCPM-V2.5 8B Qwen2-VL 7B<br>Exact Match (%)<br>**----- End of picture text -----**<br>


Figure 5: Exact Match performance on VCR before and after LoRA fine-tuning for selected models. The results demonstrate varying degrees of improvement across different tasks and models, highlighting the heterogeneous responses to fine-tuning. 

## 4.5 HUMAN EVALUATION 

We recruited 7 volunteers to perform human evaluation on a subset of the samples from our datasets. Two out of the seven evaluators are native English speakers, while five are native Chinese speakers who are also fluent in English[4] . All volunteers have earned postgraduate degrees, majoring in one of the following fields: biology, statistics, computer science, and economics. The evaluations were conducted on a voluntary basis, and participants received no rewards. 

We gave the volunteers the following instructions: (1) We asked the volunteers to focus on the puzzles. Each example in the hard collection may require 30 seconds to 2 minutes of focused attention, and (2) we asked the volunteers to utilize the context rather than directly brute-force the puzzle. 

Every sample is solved by at least 3 volunteers. In English, we release the exact match score in 2 splits: all errors counted (All), and only counting errors not related to dates and person names (Filtered). 

The human evaluation results are shown in Table 3. Although current SOTA models suffer from the challenge, fluent speakers can easily achieve more than 90 percent accuracy across difficulties. Please refer to Table 4 to compare all models with human evaluation results using the same test cases. 

Table 3: Human evaluation results on the VCR task in terms of exact matches. _N_ is the number of puzzles in each language. 

||**VCREN, EASY (N = 169)**<br>**Mean (%)**<br>**SD (%)**|**VCREN, HARD (N = 169)**<br>**Mean (%)**<br>**SD (%)**|**VCRZH, EASY (N = 188)**<br>**Mean (%)**<br>**SD (%)**|**VCRZH, HARD (N = 188)**<br>**Mean (%)**<br>**SD (%)**|
|---|---|---|---|---|
|All<br>Filtered|96.65<br>0.34<br>98.62<br>0.34|91.12<br>1.18<br>97.63<br>2.13|98.58<br>0.31<br>99.47<br>0.00|91.84<br>0.81<br>96.63<br>1.11|



## 5 RELATED WORK 

**Visual Question Answering (VQA).** Several datasets have been proposed for visual question answering VQA Antol et al. (2015); Zhang et al. (2016); Goyal et al. (2017); Mishra et al. (2019b). FVQA Wang et al. (2018) and OK-VQAMarino et al. (2019) are datasets about knowledge-based visual question answering and contains questions that necessitate the usage of external knowledge resources. CLEVR Johnson et al. (2017) is a synthetic VQA dataset that mainly focuses on visual reasoning abilities. Recognizing the need to develop VQA models that can understand text, Text-VQA Singh et al. (2019); Biten et al. (2019); Mishra et al. (2019a); Wang et al. (2020) aims to read and reason about texts embedded within images in the context of image-question answering. Several datasets Singh et al. (2019); Biten et al. (2019); Mishra et al. (2019a) have been developed for the 

> 4The TOEFL scores of the non-native English-speaking participants range from 102/120 to 112/120. 

9 

Published as a conference paper at ICLR 2025 

Text-VQA task, such as the TextVQA dataset Singh et al. (2019) and the ST-VQA dataset Biten et al. (2019) on natural images, the OCR-VQA dataset Mishra et al. (2019b) on book or movie covers, the InfographicVQA Mathew et al. (2022) dataset on infographics, and the DocVQA dataset Mathew et al. (2021) on document images. 

**Vision Language Model.** Vision-language models are designed for tasks that involve understanding and generating content from images and text Sun et al. (2023); Liu et al. (2023b); Laurenc¸on et al. (2023; 2024). For example, models have been developed to combine Llama3 with advanced visionlanguage processing capabilities to handle complex multimodal tasks Yu et al. (2024); Xu et al. (2024); Hu et al. (2023); Yu et al. (2023); Wang et al. (2023b); Dong et al. (2024a). Qwen-VL Bai et al. (2023) enhances visual-linguistic representations for more accurate contextual interpretations, while OpenGVLab-InternVL-Chat Chen et al. (2023; 2024b) merges the InternVL framework with interactive chat capabilities. These studies typically employ a multimodal encoder (Radford et al., 2021; Zhai et al., 2023; Wu et al., 2022) to process multimodal data, which is then mapped to the same input space of the language model. General-purpose models such as the GPT-4 series models (Ouyang et al., 2022; OpenAI et al., 2023), the Claude series models (Anthropic, 2024), the Gemini series models (Team et al., 2024a) and the Reka series models (Team et al., 2024b) have also been adapted for vision-language tasks, demonstrating strong performance in multimodal tasks. Finally, DocLLM Wang et al. (2023a) specializes in document understanding by integrating visual and textual data to enhance the interpretation and generation of document-related content. These models collectively represent significant advancements in vision-language integration, contributing unique capabilities and enhancements to the understanding and generation of multimodal information. 

**Optical Character Recognition (OCR).** OCR Nagy (2000) and its subproblems Howe (2013); Smith (1995); Shafait et al. (2008); Frinken et al. (2011) have been well-studied in the literature in the constrained setting. However, classical OCR methods often cannot perform well on images captured in the wild in an unconstrained setting. Many new methods have been developed for advancing scene-text recognition on camera-captured images Bissacco et al. (2013); Gupta et al. (2016); Huang et al. (2014); Jaderberg et al. (2014); Wang et al. (2012); Shi et al. (2017); Zhou et al. (2017); Lee & Osindero (2016). In addition to the detection and recognition of OCR tasks, visual question answering has emerged as an important downstream task in the OCR literature. With the development of TextVQA, new methods for improving the reading abilities in VQA utilizing OCR have been proposed. For example, LoRRA Singh et al. (2019) extends a VQA model Pythia Jiang et al. (2018) with an OCR module to better handle Text-VQA tasks. TAP Yang et al. (2021) incorporates scene texts that are generated from OCR engines during pretraining to further improve Text-VQA capabilities. 

## 6 CONCLUSION 

In this work, we introduced the VCR task, a novel vision-language challenge aimed at promoting the integration of visual and textual modalities, including text embedded in both natural language tokens and image formats and highly obscured text embedded in the image. We developed a specialized pipeline to create a dataset tailored to this task, utilizing correlated image-text pairs. This task stands out from existing methods by requiring a more profound integration of visual cues and partially obscured text, highlighting its uniqueness and importance in the field. 

We conducted extensive evaluations of state-of-the-art vision-language models (VLMs) in both English and Chinese. The results demonstrated significant room for improvement, suggesting that current models have not yet fully exploited the capabilities necessary for VCR. We selected models representing both the highest and average performance tiers for additional fine-tuning with our dataset. Although fine-tuning exhibited potential for enhancing VCR capabilities, it did not consistently result in significant improvements, indicating the complexity and challenges of adapting models to this task. 

By introducing the VCR task and its specialized dataset, we aim to advance research in visionlanguage interaction. The unique challenges of VCR seek to improve model development and training, extending the limits of multimodal AI. VCR provides a controllable testbed for fine-grained analysis of model behavior across languages, difficulty levels, image inclusion, and fine-tuning stages. We invite the community to utilize our dataset and develop innovative strategies to boost the performance of vision-language models. 

10 

Published as a conference paper at ICLR 2025 

## REFERENCES 

- 01.AI, Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue, Senbin Yang, Shiming Yang, Tao Yu, Wen Xie, Wenhao Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng Nie, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu Gu, Zhiyuan Liu, and Zonghong Dai. Yi: Open foundation models by 01.ai, 2024. 

Anthropic. The claude 3 model family: Opus, sonnet, haiku. 2024. 

- Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. VQA: Visual Question Answering. In _International Conference on Computer Vision (ICCV)_ , 2015. 

- Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. _arXiv preprint arXiv: 2308.12966_ , 2023. 

- Alessandro Bissacco, Mark Cummins, Yuval Netzer, and Hartmut Neven. Photoocr: Reading text in uncontrolled conditions. In _Proceedings of the IEEE International Conference on Computer Vision (ICCV)_ , December 2013. 

- Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marcal Rusinol, Ernest Valveny, C.V. Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , October 2019. 

- Fedor Borisyuk, Albert Gordo, and Viswanath Sivakumar. Rosetta: Large scale system for text detection and recognition in images. In _Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_ , KDD ’18, pp. 71–79, New York, NY, USA, 2018. Association for Computing Machinery. ISBN 9781450355520. doi: 10.1145/3219819. 3219861. URL https://doi.org/10.1145/3219819.3219861. 

- Qiguang Chen, Libo Qin, Jin Zhang, Zhi Chen, Xiao Xu, and Wanxiang Che. M[3] cot: A novel benchmark for multi-domain multi-step multi-modal chain-of-thought. _arXiv preprint arXiv: 2405.16473_ , 2024a. 

- Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. _arXiv preprint arXiv:2312.14238_ , 2023. 

- Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _arXiv preprint arXiv:2404.16821_ , 2024b. 

- Aaron Defazio, Xingyu Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, and Ashok Cutkosky. The road less scheduled, 2024. 

- Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. _arXiv preprint arXiv: 2401.16420_ , 2024a. 

- Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Zhe Chen, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Kai Chen, Conghui He, Xingcheng Zhang, Jifeng Dai, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. _arXiv preprint arXiv:2404.06512_ , 2024b. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, and et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

11 

Published as a conference paper at ICLR 2025 

- Volkmar Frinken, Andreas Fischer, R Manmatha, and Horst Bunke. A novel word spotting method based on recurrent neural networks. _IEEE transactions on pattern analysis and machine intelligence_ , 34(2):211–224, 2011. 

- Amber M Fyall, Yasmine El-Shamayleh, Hannah Choi, Eric Shea-Brown, and Anitha Pasupathy. Dynamic representation of partially occluded objects in primate prefrontal and visual cortex. _eLife_ , 6:e25784, September 2017. ISSN 2050-084X. doi: 10.7554/eLife.25784. 

- Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering. In _Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2017. 

- Ankush Gupta, Andrea Vedaldi, and Andrew Zisserman. Synthetic data for text localisation in natural images. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , June 2016. 

- Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, et al. Cogvlm2: Visual language models for image and video understanding. _arXiv preprint arXiv:2408.16500_ , 2024. 

- Nicholas R Howe. Document binarization with automatic parameter tuning. _International journal on document analysis and recognition (ijdar)_ , 16:247–258, 2013. 

- Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. _arXiv preprint arXiv: 2403.12895_ , 2024a. 

- Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In _International Conference on Learning Representations_ , 2022. URL https://openreview.net/forum? id=nZeVKeeFYf9. 

- Jinyi Hu, Yuan Yao, Chongyi Wang, Shan Wang, Yinxu Pan, Qianyu Chen, Tianyu Yu, Hanghao Wu, Yue Zhao, Haoye Zhang, Xu Han, Yankai Lin, Jiao Xue, Dahai Li, Zhiyuan Liu, and Maosong Sun. Large multilingual models pivot zero-shot multimodal learning across languages. _arXiv preprint arXiv:2308.12038_ , 2023. 

- Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, Xinrong Zhang, Zheng Leng Thai, Kaihuo Zhang, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm: Unveiling the potential of small language models with scalable training strategies, 2024b. 

- Weilin Huang, Yu Qiao, and Xiaoou Tang. Robust scene text detection with convolution neural network induced mser trees. In David Fleet, Tomas Pajdla, Bernt Schiele, and Tinne Tuytelaars (eds.), _Computer Vision – ECCV 2014_ , pp. 497–511, Cham, 2014. Springer International Publishing. ISBN 978-3-319-10593-2. 

- Max Jaderberg, Andrea Vedaldi, and Andrew Zisserman. Deep features for text spotting. In David Fleet, Tomas Pajdla, Bernt Schiele, and Tinne Tuytelaars (eds.), _Computer Vision – ECCV 2014_ , pp. 512–528, Cham, 2014. Springer International Publishing. ISBN 978-3-319-10593-2. 

- Yu Jiang, Vivek Natarajan, Xinlei Chen, Marcus Rohrbach, Dhruv Batra, and Devi Parikh. Pythia v0.1: the winning entry to the vqa challenge 2018, 2018. 

- Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , July 2017. 

- Hugo Laurenc¸on, Lucile Saulnier, Leo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov,´ Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. Obelics: An open web-scale filtered dataset of interleaved image-text documents, 2023. 

12 

Published as a conference paper at ICLR 2025 

- Hugo Laurenc¸on, Andres Marafioti, Victor Sanh, and L´ eo Tronchon.´ Building and better understanding vision-language models: insights and future directions. _arXiv preprint arXiv: 2408.12637_ , 2024. 

- Chen-Yu Lee and Simon Osindero. Recursive recurrent nets with attention modeling for ocr in the wild. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , June 2016. 

- Bao Li, Chi Zhang, Long Cao, Panpan Chen, Tianyuan Liu, Hui Gao, Linyuan Wang, Bin Yan, and Li Tong. Brain Functional Representation of Highly Occluded Object Recognition. _Brain Sciences_ , 13(10):1387, October 2023a. ISSN 2076-3425. doi: 10.3390/brainsci13101387. 

- Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. _arXiv preprint arXiv: 2311.06607_ , 2023b. 

- Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023a. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In _Advances in Neural Information Processing Systems_ , volume 36, pp. 34892–34916. Curran Associates, Inc., 2023b. URL https://proceedings.neurips.cc/paper_files/paper/2023/ file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf. 

- Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024a. URL https: //llava-vl.github.io/blog/2024-01-30-llava-next/. 

- Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. Textmonkey: An ocr-free large multimodal model for understanding document. _arXiv preprint arXiv: 2403.04473_ , 2024b. 

- Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan. Deepseek-vl: Towards real-world vision-language understanding, 2024. 

- Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , June 2019. 

- Minesh Mathew, Dimosthenis Karatzas, and C.V. Jawahar. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ , pp. 2200–2209, January 2021. 

- Minesh Mathew, Viraj Bagal, Ruben Tito, Dimosthenis Karatzas, Ernest Valveny, and C.V. Jawahar.` Infographicvqa. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ , pp. 1697–1706, January 2022. 

- Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In _2019 International Conference on Document Analysis and Recognition (ICDAR)_ , pp. 947–952, 2019a. doi: 10.1109/ICDAR.2019.00156. 

- Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In _2019 International Conference on Document Analysis and Recognition (ICDAR)_ , pp. 947–952, 2019b. doi: 10.1109/ICDAR.2019.00156. 

AI Mistral. Announcing pixtral 12B, September 2024. 

- G. Nagy. Twenty years of document image analysis in pami. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 22(01):38–62, jan 2000. ISSN 1939-3539. doi: 10.1109/34.824820. 

- OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, et al. Gpt-4 technical report. _arXiv preprint arXiv: 2303.08774_ , 2023. 

13 

Published as a conference paper at ICLR 2025 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. _Advances in neural information processing systems_ , 35:27730– 27744, 2022. 

- Luiz Pessoa, Evan Thompson, and Alva Noe.¨ Finding out about filling-in: A guide to perceptual completion for visual science and the philosophy of perception. _Behavioral and Brain Sciences_ , 21 (6):723–748, 1998. doi: 10.1017/S0140525X98001757. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, A. Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and I. Sutskever. Learning transferable visual models from natural language supervision. _International Conference on Machine Learning_ , 2021. 

- Faisal Shafait, Daniel Keysers, and Thomas Breuel. Performance evaluation and benchmarking of sixpage segmentation algorithms. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 30(6):941–954, 2008. 

- Baoguang Shi, Xiang Bai, and Cong Yao. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 39(11):2298–2304, 2017. doi: 10.1109/TPAMI.2016.2646371. 

- Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 8317–8326, 2019. 

- Ray Smith. A simple and efficient skew detection algorithm via text row accumulation. In _Proceedings of 3rd International Conference on Document Analysis and Recognition_ , volume 2, pp. 1145–1148. IEEE, 1995. 

- Zeyi Sun, Ye Fang, Tong Wu, Pan Zhang, Yuhang Zang, Shu Kong, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. Alpha-clip: A clip model focusing on wherever you want. _arXiv preprint arXiv: 2312.03818_ , 2023. 

- Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, Soroosh Mariooryad, Yifan Ding, Xinyang Geng, Fred Alcober, Roy Frostig, Mark Omernick, Lexi Walker, Cosmin Paduraru, Christina Sorokin, Andrea Tacchetti, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv: 2403.05530_ , 2024a. 

- Reka Team, Aitor Ormazabal, Che Zheng, Cyprien de Masson d’Autume, Dani Yogatama, Deyu Fu, Donovan Ong, Eric Chen, Eugenie Lamprecht, Hai Pham, Isaac Ong, Kaloyan Aleksiev, Lei Li, Matthew Henderson, Max Bain, Mikel Artetxe, Nishant Relan, Piotr Padlewski, Qi Liu, Ren Chen, Samuel Phua, Yazheng Yang, Yi Tay, Yuqi Wang, Zhongkai Zhu, and Zhihui Xie. Reka core, flash, and edge: A series of powerful multimodal language models. _arXiv preprint arXiv: 2404.12387_ , 2024b. 

- ´ 

- G. Thines, A. Costall, and G. Butterworth. _Michotte’s Experimental Phenomenology of Perception_ . Routledge Library Editions: Phenomenology. Taylor & Francis, 2013. ISBN 9781134506897. URL https://books.google.com/books?id=nbREAQAAQBAJ. 

- Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs. _arXiv preprint arXiv:2406.16860_ , 2024. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, 

14 

Published as a conference paper at ICLR 2025 

Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv: 2307.09288_ , 2023. 

- Rob van Lier and Walter Gerbino. Perceptual completions. In _The Oxford Handbook of Perceptual Organization_ . Oxford University Press, 08 2015. ISBN 9780199686858. doi: 10. 1093/oxfordhb/9780199686858.013.040. URL https://doi.org/10.1093/oxfordhb/ 9780199686858.013.040. 

- Dongsheng Wang, Natraj Raman, Mathieu Sibue, Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong Pei, Armineh Nourbakhsh, and Xiaomo Liu. Docllm: A layout-aware generative language model for multimodal document understanding. _arXiv preprint arXiv: 2401.00908_ , 2023a. 

- Peng Wang, Qi Wu, Chunhua Shen, Anthony Dick, and Anton van den Hengel. Fvqa: Fact-based visual question answering. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 40 (10):2413–2427, 2018. doi: 10.1109/TPAMI.2017.2754246. 

- Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. _arXiv preprint arXiv: 2409.12191_ , 2024. 

- Tao Wang, David J. Wu, Adam Coates, and Andrew Y. Ng. End-to-end text recognition with convolutional neural networks. In _Proceedings of the 21st International Conference on Pattern Recognition (ICPR2012)_ , pp. 3304–3308, 2012. 

- Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models, 2023b. 

- Xinyu Wang, Yuliang Liu, Chunhua Shen, Chun Chet Ng, Canjie Luo, Lianwen Jin, Chee Seng Chan, Anton van den Hengel, and Liangwei Wang. On the general value of evidence, and bilingual scene-text visual question answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , June 2020. 

- Shilian Wu, Yongrui Li, and Zengfu Wang. Chinese text recognition enhanced by glyph and character semantic information. _International Journal on Document Analysis and Recognition (IJDAR)_ , 27(1):45–56, March 2024. ISSN 1433-2825. doi: 10.1007/s10032-023-00444-9. URL https://doi.org/10.1007/s10032-023-00444-9. 

- Yusong Wu, K. Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and S. Dubnov. Largescale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. _IEEE International Conference on Acoustics, Speech, and Signal Processing_ , 2022. doi: 10.1109/ICASSP49357.2023.10095969. 

- Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, and Gao Huang. LLaVA-UHD: an lmm perceiving any aspect ratio and high-resolution images. _arXiv preprint arXiv:2403.11703_ , 2024. 

- Zhengyuan Yang, Yijuan Lu, Jianfeng Wang, Xi Yin, Dinei Florencio, Lijuan Wang, Cha Zhang, Lei Zhang, and Jiebo Luo. Tap: Text-aware pre-training for text-vqa and text-caption. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 8751–8761, June 2021. 

- Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. _arXiv preprint arXiv:2312.00849_ , 2023. 

15 

Published as a conference paper at ICLR 2025 

Tianyu Yu, Haoye Zhang, Yuan Yao, Yunkai Dang, Da Chen, Xiaoman Lu, Ganqu Cui, Taiwen He, Zhiyuan Liu, Tat-Seng Chua, and Maosong Sun. Rlaif-v: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness. _arXiv preprint arXiv:2405.17220_ , 2024. 

- Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. _IEEE International Conference on Computer Vision_ , 2023. doi: 10.1109/ ICCV51070.2023.01100. 

- Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, Songyang Zhang, Wenwei Zhang, Yining Li, Yang Gao, Peng Sun, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Hang Yan, Conghui He, Xingcheng Zhang, Kai Chen, Jifeng Dai, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output. _arXiv preprint arXiv:2407.03320_ , 2024. 

- Peng Zhang, Yash Goyal, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Yin and Yang: Balancing and answering binary visual questions. In _Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2016. 

- Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. _arXiv preprint arXiv: 2302.00923_ , 2023. 

- Yuliang Zhao, Xinyue Zhang, Boya Fu, Zhikun Zhan, Hui Sun, Lianjiang Li, and Guanglie Zhang. Evaluation and recognition of handwritten chinese characters based on similarities. _Applied Sciences_ , 12(17), 2022. ISSN 2076-3417. doi: 10.3390/app12178521. URL https://www. mdpi.com/2076-3417/12/17/8521. 

- Xinyu Zhou, Cong Yao, He Wen, Yuzhi Wang, Shuchang Zhou, Weiran He, and Jiajun Liang. East: An efficient and accurate scene text detector. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , July 2017. 

16 

Published as a conference paper at ICLR 2025 

A ADDITIONAL EVALUATION RESULTS ON FIRST 100 AND 500 TEST CASES Table 4: Results of various open-source and closed-source vision language models on the VCR task using the first 100 test cases. FT = finetuned on 16,000 samples from the VCR-WIKI training set. The best results among the finetuned models are underlined while the best results among the models without finetuning are highlighted in **bold** . Subscripts are standard deviations obtained from Bootstrap. 

|**Language**<br>English|**Mode**|**Open/closed**<br>**source**|**Model name**<br>**Model size**|**Exact match**(%)_↑_|**Jaccard index**(%)_↑_|
|---|---|---|---|---|---|
|||||_V I_ +_TEI_<br>_TEI_<br>∆|_V I_ +_TEI_<br>_TEI_<br>∆|
||Easy|Closed|Claude 3 Opus<br>-<br>Claude 3.5 Sonnet<br>-<br>Gemini 1.5 Pro<br>-<br>GPT-4 Turbo<br>-<br>GPT-4o<br>-<br>GPT-4V<br>-<br>Qwen-VL-Max<br>-<br>Reka Core<br>-|62_._00_._76<br>82_._00_._63<br>-20<br>70_._413_._46<br>75_._153_._36<br>-4.73<br>71_._013_._4<br>86_._982_._67<br>-15.98<br>78_._470_._22<br>86_._60_._79<br>-8.13<br>**90****_._910****_._36**<br>**95****_._690****_._23**<br>-4.78<br>25_._360_._5<br>18_._180_._54<br>7.18<br>82_._30_._19<br>88_._040_._43<br>-5.74<br>65_._683_._78<br>78_._113_._19<br>-12.43|78_._060_._24<br>91_._120_._13<br>-13.06<br>78_._12_._85<br>86_._52_._18<br>-8.4<br>82_._892_._27<br>94_._211_._32<br>-11.32<br>88_._080_._25<br>94_._150_._2<br>-6.07<br>**96****_._770****_._16**<br>**98****_._450****_._06**<br>-1.68<br>35_._640_._22<br>28_._490_._23<br>7.15<br>89_._730_._32<br>92_._550_._17<br>-2.82<br>83_._142_._04<br>90_._431_._49<br>-7.29|
|||Open|Cambrian-1<br>34B<br>CogVLM2<br>19B<br>CogVLM2-FT<br>19B<br>DeepSeek-VL<br>1.3B<br>DeepSeek-VL<br>7B<br>DocOwl-1.5-Omni<br>8B<br>Idefcs3<br>8B<br>InternLM-XComposer2-VL<br>7B<br>InternLM-XComposer2-VL-4K<br>7B<br>InternLM-XComposer2.5-VL<br>7B<br>InternVL-V2<br>40B<br>Llama-3.2-Vision<br>11B<br>Llama-3.2-Vision<br>90B<br>MiniCPM-V2.5<br>8B<br>MiniCPM-V2.5-FT<br>8B<br>Monkey<br>7B<br>Pixtral<br>12B<br>Qwen-VL<br>7B<br>Qwen2-VL<br>7B<br>Yi-VL<br>34B<br>Yi-VL<br>6B|78_._113_._16<br>82_._842_._86<br>-4.73<br>86_._390_._66<br>84_._620_._92<br>1.78<br>94_._080_._2<br>94_._670_._26<br>-0.59<br>19_._530_._69<br>26_._041_._47<br>-6.51<br>36_._091_._36<br>44_._970_._79<br>-8.88<br>0_._590_._14<br>1_._180_._14<br>-0.59<br>26_._633_._35<br>32_._543_._63<br>-5.92<br>47_._930_._69<br>47_._340_._57<br>0.59<br>4_._141_._54<br>3_._551_._49<br>0.59<br>45_._563_._83<br>28_._993_._50<br>16.57<br>86_._392_._56<br>86_._982_._60<br>-0.59<br>79_._882_._96<br>68_._643_._75<br>11.24<br>79_._293_._14<br>71_._013_._36<br>8.28<br>30_._180_._66<br>36_._090_._34<br>-5.92<br>39_._050_._69<br>46_._750_._59<br>-7.69<br>46_._750_._44<br>48_._520_._41<br>-1.78<br>14_._792_._65<br>13_._022_._62<br>1.78<br>47_._340_._44<br>46_._750_._57<br>0.59<br>**90****_._532****_._25**<br>**96****_._451****_._39**<br>-5.92<br>1_._780_._16<br>1_._180_._11<br>0.59<br>2_._370_._13<br>1_._780_._22<br>0.59|87_._881_._97<br>93_._121_._26<br>-5.24<br>91_._390_._11<br>91_._630_._11<br>-0.24<br>98_._030_._07<br>98_._220_._03<br>-0.2<br>43_._730_._18<br>48_._030_._16<br>-4.3<br>57_._810_._18<br>61_._830_._33<br>-4.01<br>12_._690_._04<br>13_._30_._06<br>-0.61<br>48_._832_._95<br>55_._112_._78<br>-6.29<br>73_._880_._22<br>74_._580_._16<br>-0.7<br>21_._911_._81<br>21_._851_._86<br>0.06<br>67_._702_._79<br>54_._252_._70<br>13.45<br>93_._511_._40<br>94_._351_._24<br>-0.84<br>90_._881_._56<br>82_._912_._11<br>7.97<br>87_._812_._00<br>83_._172_._30<br>4.64<br>53_._10_._18<br>59_._060_._14<br>-5.96<br>63_._050_._28<br>69_._890_._33<br>-6.84<br>67_._820_._22<br>68_._590_._13<br>-0.76<br>39_._002_._49<br>33_._162_._53<br>5.84<br>69_._020_._35<br>69_._190_._37<br>-0.17<br>**94****_._281****_._54**<br>**98****_._820****_._49**<br>-4.54<br>6_._210_._06<br>7_._50_._08<br>-1.3<br>6_._240_._07<br>8_._050_._11<br>-1.81|
||Hard|Closed|Claude 3 Opus<br>-<br>Claude 3.5 Sonnet<br>-<br>Gemini 1.5 Pro<br>-<br>GPT-4 Turbo<br>-<br>GPT-4o<br>-<br>GPT-4V<br>-<br>Qwen-VL-Max<br>-<br>Reka Core<br>-|34_._01_._12<br>51_._00_._5<br>-17<br>46_._753_._58<br>43_._23_._83<br>3.55<br>33_._733_._69<br>43_._793_._74<br>-10.06<br>53_._110_._46<br>57_._420_._5<br>-4.31<br>**74****_._160****_._31**<br>**84****_._690****_._31**<br>-10.53<br>28_._710_._49<br>16_._270_._73<br>12.44<br>40_._670_._38<br>55_._020_._46<br>-14.35<br>7_._12_._01<br>10_._652_._38<br>-3.55|57_._020_._24<br>70_._320_._15<br>-13.31<br>57_._743_._33<br>54_._133_._51<br>3.61<br>57_._092_._67<br>62_._342_._76<br>-5.25<br>71_._750_._19<br>73_._820_._24<br>-2.07<br>**86****_._990****_._09**<br>**93****_._190****_._07**<br>-6.21<br>49_._890_._15<br>33_._640_._16<br>16.25<br>61_._80_._19<br>72_._460_._15<br>-10.66<br>25_._491_._99<br>36_._782_._19<br>-11.29|
|||Open|Cambrian-1<br>34B<br>CogVLM2<br>19B<br>CogVLM2-FT<br>19B<br>DeepSeek-VL<br>1.3B<br>DeepSeek-VL<br>7B<br>DocOwl-1.5-Omni<br>8B<br>Idefcs3<br>8B<br>InternLM-XComposer2-VL<br>7B<br>InternLM-XComposer2-VL-4K<br>7B<br>InternLM-XComposer2.5-VL<br>7B<br>InternVL-V2<br>40B<br>Llama-3.2-Vision<br>11B<br>Llama-3.2-Vision<br>90B<br>MiniCPM-V2.5<br>8B<br>MiniCPM-V2.5-FT<br>8B<br>Monkey<br>7B<br>Pixtral<br>12B<br>Qwen-VL<br>7B<br>Qwen2-VL<br>7B<br>Yi-VL<br>34B<br>Yi-VL<br>6B|27_._813_._29<br>29_._593_._54<br>-1.78<br>44_._970_._83<br>21_._30_._47<br>23.67<br>75_._740_._72<br>67_._460_._64<br>8.28<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._590_._09<br>1_._780_._17<br>-1.18<br>0_._00_._0<br>0_._00_._0<br>0<br>1_._180_._87<br>0_._590_._61<br>0.59<br>0_._00_._0<br>0_._590_._09<br>-0.59<br>0_._000_._00<br>0_._590_._59<br>-0.59<br>0_._590_._58<br>1_._781_._01<br>-1.18<br>12_._432_._54<br>16_._572_._89<br>-4.14<br>10_._652_._33<br>7_._692_._04<br>2.96<br>13_._022_._59<br>15_._382_._75<br>-2.37<br>1_._180_._12<br>1_._780_._12<br>-0.59<br>10_._060_._43<br>13_._020_._54<br>-2.96<br>1_._180_._22<br>3_._550_._18<br>-2.37<br>0_._000_._00<br>0_._590_._61<br>-0.59<br>1_._780_._21<br>2_._960_._12<br>-1.18<br>**75****_._743****_._32**<br>**73****_._963****_._55**<br>1.78<br>0_._590_._09<br>0_._00_._0<br>0.59<br>0_._590_._13<br>0_._00_._0<br>0.59|51_._392_._79<br>54_._002_._76<br>-2.61<br>65_._390_._2<br>43_._860_._27<br>21.53<br>90_._60_._13<br>84_._260_._08<br>6.34<br>11_._170_._03<br>10_._880_._06<br>0.29<br>16_._710_._11<br>18_._090_._13<br>-1.38<br>7_._890_._05<br>8_._280_._05<br>-0.4<br>11_._621_._20<br>10_._281_._00<br>1.34<br>12_._690_._08<br>14_._050_._11<br>-1.35<br>9_._670_._90<br>8_._830_._95<br>0.84<br>14_._091_._04<br>16_._571_._25<br>-2.48<br>33_._742_._40<br>39_._512_._69<br>-5.76<br>33_._502_._28<br>26_._802_._03<br>6.70<br>36_._802_._36<br>39_._852_._52<br>-3.06<br>12_._020_._12<br>12_._410_._07<br>-0.39<br>34_._670_._2<br>36_._430_._19<br>-1.76<br>12_._660_._21<br>15_._970_._08<br>-3.31<br>9_._900_._79<br>12_._561_._09<br>-2.66<br>15_._70_._14<br>15_._060_._19<br>0.63<br>**85****_._912****_._17**<br>**85****_._832****_._04**<br>0.08<br>4_._390_._07<br>5_._490_._08<br>-1.1<br>5_._120_._03<br>5_._50_._06<br>-0.38|
|Chinese|Easy|Closed|Claude 3 Opus<br>-<br>Claude 3.5 Sonnet<br>-<br>Gemini 1.5 Pro<br>-<br>GPT-4o<br>-<br>GPT-4 Turbo<br>-<br>Qwen-VL-Max<br>-<br>Reka Core<br>-|0_._530_._51<br>0_._530_._55<br>0<br>1_._60_._91<br>2_._131_._05<br>-0.53<br>0_._530_._56<br>0_._00_._0<br>0.53<br>**14****_._892****_._51**<br>**21****_._812****_._98**<br>-6.91<br>0_._530_._55<br>0_._00_._0<br>0.53<br>5_._930_._19<br>8_._70_._37<br>-2.77<br>0_._00_._0<br>0_._00_._0<br>0|11_._341_._07<br>9_._140_._93<br>2.2<br>8_._071_._29<br>9_._91_._48<br>-1.84<br>12_._941_._26<br>12_._771_._17<br>0.16<br>**38****_._572****_._46**<br>**48****_._292****_._43**<br>-9.72<br>11_._091_._05<br>7_._510_._65<br>3.58<br>13_._530_._11<br>18_._50_._1<br>-4.97<br>3_._040_._53<br>2_._420_._45<br>0.61|
|||Open|CogVLM2-Chinese<br>19B<br>CogVLM2-Chinese-FT<br>19B<br>DeepSeek-VL<br>1.3B<br>DeepSeek-VL<br>7B<br>DocOwl-1.5-Omni<br>8B<br>InternLM-XComposer2-VL<br>7B<br>InternLM-XComposer2-VL-4K<br>7B<br>InternLM-XComposer2.5-VL<br>7B<br>InternVL-V2<br>40B<br>MiniCPM-V2.5<br>8B<br>MiniCPM-V2.5-FT<br>8B<br>Monkey<br>7B<br>Qwen-VL<br>7B<br>Qwen2-VL<br>7B<br>Yi-VL<br>34B<br>Yi-VL<br>6B|34_._570_._66<br>34_._041_._01<br>0.53<br>66_._490_._74<br>67_._550_._73<br>-1.06<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._00_._0<br>0_._00_._0<br>0<br>1_._060_._09<br>0_._530_._07<br>0.53<br>0_._000_._00<br>0_._000_._00<br>0.00<br>0_._000_._00<br>1_._600_._91<br>-1.60<br>26_._063_._17<br>19_._152_._88<br>6.91<br>4_._790_._16<br>7_._450_._35<br>-2.66<br>6_._910_._33<br>7_._980_._4<br>-1.06<br>1_._060_._12<br>0_._530_._06<br>0.53<br>0_._00_._0<br>0_._00_._0<br>0<br>**67****_._553****_._34**<br>**73****_._403****_._21**<br>-5.85<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._00_._0<br>0_._00_._0<br>0|58_._780_._13<br>57_._260_._12<br>1.52<br>79_._480_._17<br>81_._780_._09<br>-2.3<br>6_._690_._07<br>2_._920_._02<br>3.78<br>3_._990_._07<br>6_._710_._02<br>-2.72<br>1_._230_._04<br>2_._970_._02<br>-1.75<br>13_._10_._03<br>13_._260_._03<br>-0.16<br>13_._491_._01<br>14_._560_._98<br>-1.08<br>11_._940_._88<br>16_._121_._24<br>-4.18<br>48_._982_._61<br>41_._252_._57<br>7.72<br>20_._580_._11<br>25_._380_._13<br>-4.81<br>30_._80_._07<br>31_._460_._52<br>-0.66<br>9_._230_._08<br>12_._290_._13<br>-3.06<br>1_._410_._02<br>0_._660_._03<br>0.76<br>**84****_._631****_._80**<br>**87****_._121****_._76**<br>-2.49<br>4_._530_._03<br>1_._840_._05<br>2.69<br>4_._730_._02<br>1_._550_._02<br>3.18|
||Hard|Closed|Claude 3 Opus<br>-<br>Claude 3.5 Sonnet<br>-<br>Gemini 1.5 Pro<br>-<br>GPT-4o<br>-<br>GPT-4 Turbo<br>-<br>Qwen-VL-Max<br>-<br>Reka Core<br>-|1_._060_._77<br>0_._530_._54<br>0.53<br>0_._530_._51<br>0_._00_._0<br>0.53<br>1_._060_._71<br>1_._060_._77<br>0<br>**2****_._661****_._16**<br>**1****_._60****_._92**<br>1.06<br>0_._00_._0<br>0_._530_._53<br>-0.53<br>1_._190_._12<br>1_._980_._09<br>-0.79<br>0_._00_._0<br>0_._00_._0<br>0|9_._231_._04<br>7_._770_._83<br>1.45<br>4_._110_._84<br>3_._320_._71<br>0.79<br>11_._581_._14<br>13_._341_._2<br>-1.76<br>**23****_._691****_._65**<br>**23****_._691****_._48**<br>0<br>8_._510_._7<br>8_._020_._78<br>0.49<br>6_._190_._1<br>11_._090_._11<br>-4.9<br>3_._220_._51<br>3_._620_._57<br>-0.4|
|||Open|CogVLM2-Chinese<br>19B<br>CogVLM2-Chinese-FT<br>19B<br>DeepSeek-VL<br>1.3B<br>DeepSeek-VL<br>7B<br>DocOwl-1.5-Omni<br>8B<br>InternLM-XComposer2-VL<br>7B<br>InternLM-XComposer2-VL-4K<br>7B<br>InternLM-XComposer2.5-VL<br>7B<br>InternVL-V2<br>40B<br>MiniCPM-V2.5<br>8B<br>MiniCPM-V2.5-FT<br>8B<br>Monkey<br>7B<br>Qwen-VL<br>7B<br>Qwen2-VL<br>7B<br>Yi-VL<br>34B<br>Yi-VL<br>6B|3_._190_._19<br>3_._190_._32<br>0<br>46_._810_._32<br>46_._280_._49<br>0.53<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._000_._00<br>0_._000_._00<br>0.00<br>0_._000_._00<br>0_._000_._00<br>0.00<br>0_._530_._50<br>1_._060_._72<br>-0.53<br>0_._530_._07<br>0_._530_._07<br>0<br>1_._060_._08<br>2_._130_._19<br>-1.06<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._00_._0<br>0_._00_._0<br>0<br>**17****_._552****_._81**<br>**27****_._663****_._21**<br>-10.11<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._00_._0<br>0_._00_._0<br>0|18_._330_._14<br>21_._380_._09<br>-3.05<br>66_._850_._39<br>69_._790_._12<br>-2.95<br>6_._50_._03<br>4_._160_._03<br>2.34<br>5_._220_._04<br>7_._450_._06<br>-2.23<br>1_._350_._02<br>3_._570_._04<br>-2.23<br>8_._170_._03<br>7_._990_._03<br>0.18<br>7_._730_._68<br>8_._120_._79<br>-0.39<br>10_._870_._82<br>10_._540_._84<br>0.32<br>12_._261_._01<br>13_._581_._20<br>-1.32<br>7_._280_._06<br>7_._710_._06<br>-0.43<br>18_._460_._1<br>16_._420_._22<br>2.03<br>6_._150_._11<br>6_._620_._11<br>-0.47<br>1_._10_._04<br>0_._060_._01<br>1.04<br>**43****_._872****_._48**<br>**51****_._992****_._61**<br>-8.12<br>4_._170_._04<br>2_._020_._04<br>2.15<br>4_._150_._06<br>2_._380_._04<br>1.77|



17 

Published as a conference paper at ICLR 2025 

Table 5: Results of various open-source and closed-source vision language models on the VCR task using the first 500 test cases. FT = finetuned on 16,000 samples from the VCR-WIKI training set. The best results among the finetuned models are underlined while the best results among the models without finetuning are highlighted in **bold** . Subscripts are standard deviations obtained from Bootstrap. 

|**Language**<br>English|**Mode**|**Open/closed**<br>**source**|**Model name**<br>**Model size**|**Exact match**(%)_↑_|**Jaccard index**(%)_↑_|
|---|---|---|---|---|---|
|||||_V I_ +_TEI_<br>_TEI_<br>∆|_V I_ +_TEI_<br>_TEI_<br>∆|
||Easy|Closed|Claude 3 Opus<br>-<br>Claude 3.5 Sonnet<br>-<br>Gemini 1.5 Pro<br>-<br>GPT-4 Turbo<br>-<br>GPT-4o<br>-<br>GPT-4V<br>-<br>Qwen-VL-Max<br>-<br>Reka Core<br>-|62_._00_._13<br>77_._00_._5<br>-15<br>63_._851_._71<br>72_._81_._56<br>-8.94<br>62_._731_._66<br>82_._981_._3<br>-20.25<br>78_._740_._13<br>81_._940_._25<br>-3.2<br>**91****_._550****_._29**<br>**94****_._560****_._13**<br>-3.01<br>52_._040_._24<br>37_._860_._22<br>14.17<br>76_._80_._5<br>85_._530_._19<br>-8.74<br>66_._461_._64<br>78_._511_._42<br>-12.05|77_._670_._32<br>88_._410_._39<br>-10.74<br>74_._651_._33<br>83_._481_._14<br>-8.83<br>77_._711_._21<br>91_._560_._76<br>-13.85<br>88_._540_._24<br>92_._180_._3<br>-3.65<br>**96****_._440****_._11**<br>**97****_._760****_._06**<br>-1.32<br>65_._360_._39<br>54_._130_._41<br>11.23<br>85_._710_._28<br>91_._450_._29<br>-5.74<br>84_._230_._86<br>90_._450_._7<br>-6.22|
|||Open|Cambrian-1<br>34B<br>CogVLM2<br>19B<br>CogVLM2-FT<br>19B<br>DeepSeek-VL<br>1.3B<br>DeepSeek-VL<br>7B<br>DocOwl-1.5-Omni<br>8B<br>Idefcs3<br>8B<br>InternLM-XComposer2-VL<br>7B<br>InternLM-XComposer2-VL-4K<br>7B<br>InternLM-XComposer2.5-VL<br>7B<br>InternVL-V2<br>40B<br>Llama-3.2-Vision<br>11B<br>Llama-3.2-Vision<br>90B<br>MiniCPM-V2.5<br>8B<br>MiniCPM-V2.5-FT<br>8B<br>Monkey<br>7B<br>Pixtral<br>12B<br>Qwen-VL<br>7B<br>Qwen2-VL<br>7B<br>Yi-VL<br>34B<br>Yi-VL<br>6B|76_._891_._52<br>80_._251_._36<br>-3.35<br>83_._110_._28<br>79_._630_._33<br>3.48<br>92_._80_._06<br>92_._670_._13<br>0.12<br>21_._860_._17<br>30_._680_._3<br>-8.82<br>37_._760_._42<br>45_._470_._21<br>-7.7<br>0_._620_._06<br>1_._860_._06<br>-1.24<br>26_._711_._57<br>29_._811_._55<br>-3.11<br>46_._090_._35<br>46_._340_._25<br>-0.25<br>5_._220_._80<br>3_._230_._63<br>1.99<br>42_._481_._73<br>25_._841_._53<br>16.65<br>84_._841_._21<br>87_._081_._19<br>-2.24<br>79_._251_._40<br>66_._461_._63<br>12.80<br>80_._871_._37<br>71_._551_._54<br>9.32<br>32_._80_._16<br>36_._770_._25<br>-3.98<br>42_._360_._3<br>45_._340_._35<br>-2.98<br>47_._20_._2<br>54_._160_._41<br>-6.96<br>16_._651_._31<br>11_._801_._13<br>4.84<br>45_._470_._35<br>52_._170_._33<br>-6.71<br>**90****_._061****_._07**<br>**94****_._530****_._82**<br>-4.47<br>0_._870_._06<br>1_._240_._04<br>-0.37<br>1_._120_._03<br>1_._370_._14<br>-0.25|87_._660_._90<br>92_._420_._60<br>-4.76<br>89_._430_._27<br>88_._650_._26<br>0.79<br>97_._510_._24<br>97_._450_._07<br>0.06<br>45_._40_._33<br>52_._020_._73<br>-6.62<br>59_._070_._43<br>64_._260_._57<br>-5.2<br>12_._650_._3<br>14_._090_._12<br>-1.44<br>46_._911_._40<br>51_._841_._30<br>-4.93<br>71_._110_._2<br>71_._760_._67<br>-0.65<br>22_._700_._89<br>18_._670_._79<br>4.03<br>63_._031_._32<br>50_._751_._21<br>12.28<br>93_._130_._69<br>94_._830_._50<br>-1.71<br>89_._980_._77<br>80_._911_._06<br>9.08<br>89_._630_._85<br>84_._340_._98<br>5.29<br>52_._560_._25<br>60_._890_._19<br>-8.32<br>65_._390_._6<br>67_._850_._43<br>-2.46<br>65_._70_._4<br>71_._170_._72<br>-5.47<br>39_._811_._16<br>31_._471_._11<br>8.34<br>66_._810_._74<br>71_._730_._59<br>-4.93<br>**93****_._770****_._76**<br>**97****_._800****_._34**<br>-4.03<br>5_._610_._28<br>7_._630_._42<br>-2.02<br>5_._930_._16<br>7_._330_._23<br>-1.39|
||Hard|Closed|Claude 3 Opus<br>-<br>Claude 3.5 Sonnet<br>-<br>Gemini 1.5 Pro<br>-<br>GPT-4 Turbo<br>-<br>GPT-4o<br>-<br>GPT-4V<br>-<br>Qwen-VL-Max<br>-<br>Reka Core<br>-|37_._80_._28<br>50_._00_._33<br>-12.2<br>41_._741_._69<br>44_._721_._78<br>-2.98<br>28_._071_._58<br>38_._761_._68<br>-10.68<br>45_._150_._28<br>48_._640_._57<br>-3.5<br>**73****_._20****_._16**<br>**82****_._430****_._17**<br>-9.22<br>25_._830_._44<br>14_._950_._3<br>10.87<br>41_._650_._32<br>52_._720_._2<br>-11.07<br>6_._710_._89<br>11_._181_._15<br>-4.47|57_._680_._8<br>70_._160_._64<br>-12.48<br>56_._151_._46<br>58_._541_._6<br>-2.4<br>51_._91_._22<br>59_._621_._27<br>-7.72<br>65_._720_._25<br>67_._860_._2<br>-2.14<br>**86****_._170****_._21**<br>**92****_._010****_._2**<br>-5.84<br>44_._630_._48<br>30_._080_._67<br>14.56<br>61_._180_._35<br>70_._190_._37<br>-9.01<br>25_._840_._95<br>35_._831_._05<br>-9.99|
|||Open|Cambrian-1<br>34B<br>CogVLM2<br>19B<br>CogVLM2-FT<br>19B<br>DeepSeek-VL<br>1.3B<br>DeepSeek-VL<br>7B<br>DocOwl-1.5-Omni<br>8B<br>Idefcs3<br>8B<br>InternLM-XComposer2-VL<br>7B<br>InternLM-XComposer2-VL-4K<br>7B<br>InternLM-XComposer2.5-VL<br>7B<br>InternVL-V2<br>40B<br>Llama-3.2-Vision<br>11B<br>Llama-3.2-Vision<br>90B<br>MiniCPM-V2.5<br>8B<br>MiniCPM-V2.5-FT<br>8B<br>Monkey<br>7B<br>Pixtral<br>12B<br>Qwen-VL<br>7B<br>Qwen2-VL<br>7B<br>Yi-VL<br>34B<br>Yi-VL<br>6B|27_._201_._59<br>30_._191_._55<br>-2.98<br>41_._740_._25<br>16_._770_._22<br>24.97<br>75_._90_._13<br>65_._220_._18<br>10.68<br>0_._370_._02<br>0_._120_._01<br>0.25<br>0_._750_._02<br>1_._610_._1<br>-0.87<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._750_._30<br>0_._500_._25<br>0.25<br>0_._50_._04<br>0_._370_._05<br>0.12<br>0_._000_._00<br>0_._120_._12<br>-0.12<br>0_._750_._31<br>1_._240_._39<br>-0.50<br>14_._161_._22<br>18_._511_._36<br>-4.35<br>13_._911_._25<br>7_._330_._94<br>6.58<br>15_._161_._28<br>12_._171_._13<br>2.98<br>1_._740_._08<br>1_._610_._08<br>0.12<br>11_._430_._11<br>14_._290_._16<br>-2.86<br>1_._370_._05<br>2_._240_._15<br>-0.87<br>0_._250_._19<br>0_._620_._28<br>-0.37<br>1_._610_._03<br>1_._740_._03<br>-0.12<br>**76****_._271****_._49**<br>**75****_._651****_._45**<br>0.62<br>0_._120_._01<br>0_._00_._0<br>0.12<br>0_._120_._02<br>0_._00_._0<br>0.12|49_._961_._36<br>55_._931_._23<br>-5.97<br>62_._560_._33<br>38_._410_._44<br>24.15<br>89_._750_._14<br>82_._710_._27<br>7.04<br>11_._420_._09<br>11_._410_._22<br>0.01<br>15_._80_._29<br>17_._180_._41<br>-1.38<br>7_._340_._06<br>7_._610_._16<br>-0.27<br>10_._440_._49<br>9_._170_._43<br>1.27<br>12_._380_._13<br>13_._220_._11<br>-0.83<br>9_._550_._38<br>9_._180_._38<br>0.37<br>13_._670_._51<br>14_._920_._56<br>-1.25<br>35_._011_._18<br>41_._021_._22<br>-6.02<br>35_._781_._14<br>26_._140_._94<br>9.64<br>37_._571_._13<br>35_._141_._04<br>2.43<br>11_._550_._24<br>11_._690_._38<br>-0.15<br>35_._130_._19<br>36_._650_._68<br>-1.52<br>13_._160_._18<br>14_._450_._24<br>-1.29<br>10_._040_._41<br>11_._210_._45<br>-1.17<br>15_._280_._13<br>14_._430_._54<br>0.85<br>**86****_._561****_._00**<br>**86****_._770****_._93**<br>-0.22<br>4_._310_._08<br>5_._450_._13<br>-1.14<br>4_._490_._05<br>5_._70_._12<br>-1.21|
|Chinese|Easy|Closed|Claude 3 Opus<br>-<br>Claude 3.5 Sonnet<br>-<br>Gemini 1.5 Pro<br>-<br>GPT-4o<br>-<br>GPT-4 Turbo<br>-<br>Qwen-VL-Max<br>-<br>Reka Core<br>-|0_._90_._3<br>1_._00_._31<br>-0.1<br>1_._00_._31<br>0_._80_._28<br>0.2<br>1_._10_._32<br>0_._50_._22<br>0.6<br>**14****_._871****_._14**<br>**22****_._461****_._35**<br>-7.58<br>0_._20_._14<br>0_._10_._1<br>0.1<br>6_._340_._08<br>9_._920_._09<br>-3.58<br>0_._00_._0<br>0_._00_._0<br>0|11_._50_._48<br>10_._00_._49<br>1.49<br>7_._540_._54<br>7_._50_._51<br>0.03<br>11_._10_._56<br>11_._470_._48<br>-0.37<br>**39****_._050****_._99**<br>**48****_._241****_._09**<br>-9.19<br>8_._420_._36<br>6_._970_._29<br>1.45<br>13_._450_._41<br>22_._860_._46<br>-9.42<br>3_._430_._26<br>3_._150_._2<br>0.28|
|||Open|CogVLM2-Chinese<br>19B<br>CogVLM2-Chinese-FT<br>19B<br>DeepSeek-VL<br>1.3B<br>DeepSeek-VL<br>7B<br>DocOwl-1.5-Omni<br>8B<br>InternLM-XComposer2-VL<br>7B<br>InternLM-XComposer2-VL-4K<br>7B<br>InternLM-XComposer2.5-VL<br>7B<br>InternVL-V2<br>40B<br>MiniCPM-V2.5<br>8B<br>MiniCPM-V2.5-FT<br>8B<br>Monkey<br>7B<br>Qwen-VL<br>7B<br>Qwen2-VL<br>7B<br>Yi-VL<br>34B<br>Yi-VL<br>6B|33_._630_._15<br>31_._440_._19<br>2.2<br>63_._970_._55<br>62_._670_._17<br>1.3<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._60_._05<br>0_._20_._04<br>0.4<br>0_._200_._14<br>0_._100_._10<br>0.10<br>0_._300_._17<br>0_._400_._20<br>-0.10<br>22_._751_._36<br>16_._671_._14<br>6.09<br>4_._590_._11<br>4_._890_._09<br>-0.3<br>7_._290_._14<br>7_._090_._12<br>0.2<br>0_._20_._01<br>1_._40_._05<br>-1.2<br>0_._00_._0<br>0_._00_._0<br>0<br>**61****_._081****_._52**<br>**68****_._161****_._46**<br>-7.09<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._00_._0<br>0_._00_._0<br>0|57_._970_._56<br>54_._050_._54<br>3.92<br>79_._710_._41<br>79_._220_._47<br>0.49<br>6_._10_._1<br>3_._250_._05<br>2.85<br>4_._280_._07<br>7_._30_._05<br>-3.02<br>1_._190_._05<br>3_._830_._06<br>-2.63<br>12_._340_._25<br>12_._520_._14<br>-0.18<br>11_._930_._42<br>13_._680_._41<br>-1.74<br>12_._760_._42<br>14_._990_._43<br>-2.23<br>49_._511_._06<br>39_._461_._10<br>10.05<br>18_._120_._33<br>22_._280_._18<br>-4.17<br>29_._360_._39<br>30_._670_._38<br>-1.31<br>7_._890_._3<br>10_._260_._24<br>-2.37<br>1_._250_._03<br>0_._430_._06<br>0.82<br>**78****_._380****_._96**<br>**83****_._480****_._84**<br>-5.10<br>4_._690_._09<br>1_._710_._06<br>2.98<br>4_._280_._06<br>1_._660_._04<br>2.62|
||Hard|Closed|Claude 3 Opus<br>-<br>Claude 3.5 Sonnet<br>-<br>Gemini 1.5 Pro<br>-<br>GPT-4o<br>-<br>GPT-4 Turbo<br>-<br>Qwen-VL-Max<br>-<br>Reka Core<br>-|0_._30_._18<br>0_._10_._1<br>0.2<br>0_._20_._15<br>0_._00_._0<br>0.2<br>0_._70_._26<br>0_._50_._23<br>0.2<br>**2****_._20****_._47**<br>**1****_._80****_._4**<br>0.4<br>0_._00_._0<br>0_._20_._13<br>-0.2<br>0_._890_._06<br>1_._380_._1<br>-0.49<br>0_._00_._0<br>0_._00_._0<br>0|9_._220_._38<br>8_._090_._33<br>1.13<br>4_._00_._33<br>2_._370_._23<br>1.63<br>11_._820_._51<br>11_._750_._44<br>0.07<br>**22****_._720****_._67**<br>**22****_._890****_._65**<br>-0.17<br>8_._580_._3<br>6_._870_._28<br>1.72<br>5_._40_._19<br>12_._290_._18<br>-6.89<br>3_._350_._23<br>2_._970_._2<br>0.38|
|||Open|CogVLM2-Chinese<br>19B<br>CogVLM2-Chinese-FT<br>19B<br>DeepSeek-VL<br>1.3B<br>DeepSeek-VL<br>7B<br>DocOwl-1.5-Omni<br>8B<br>InternLM-XComposer2-VL<br>7B<br>InternLM-XComposer2-VL-4K<br>7B<br>InternLM-XComposer2.5-VL<br>7B<br>InternVL-V2<br>40B<br>MiniCPM-V2.5<br>8B<br>MiniCPM-V2.5-FT<br>8B<br>Monkey<br>7B<br>Qwen-VL<br>7B<br>Qwen2-VL<br>7B<br>Yi-VL<br>34B<br>Yi-VL<br>6B|1_._20_._07<br>2_._30_._09<br>-1.1<br>42_._510_._32<br>45_._910_._23<br>-3.39<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._000_._00<br>0_._000_._00<br>0.00<br>0_._000_._00<br>0_._000_._00<br>0.00<br>0_._400_._20<br>0_._900_._29<br>-0.50<br>0_._20_._03<br>0_._20_._01<br>0<br>1_._20_._03<br>1_._40_._06<br>-0.2<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._00_._0<br>0_._00_._0<br>0<br>**18****_._761****_._22**<br>**26****_._751****_._40**<br>-7.98<br>0_._00_._0<br>0_._00_._0<br>0<br>0_._00_._0<br>0_._00_._0<br>0|16_._830_._22<br>19_._860_._23<br>-3.04<br>65_._790_._24<br>69_._460_._46<br>-3.68<br>6_._870_._09<br>3_._530_._07<br>3.33<br>5_._490_._07<br>7_._570_._05<br>-2.08<br>1_._680_._04<br>4_._420_._07<br>-2.73<br>8_._360_._09<br>7_._920_._09<br>0.44<br>7_._490_._31<br>7_._250_._30<br>0.25<br>10_._830_._31<br>10_._810_._31<br>0.02<br>12_._300_._42<br>13_._800_._48<br>-1.50<br>7_._230_._18<br>7_._60_._13<br>-0.37<br>18_._010_._35<br>15_._250_._25<br>2.76<br>5_._690_._15<br>6_._30_._13<br>-0.61<br>1_._10_._07<br>0_._150_._01<br>0.94<br>**43****_._841****_._10**<br>**53****_._561****_._09**<br>-9.72<br>4_._490_._09<br>1_._730_._1<br>2.76<br>3_._950_._05<br>2_._080_._09<br>1.87|



18 

Published as a conference paper at ICLR 2025 

## B MORE RELATIONSHIP BETWEEN VCR AND OTHER BENCHMARKS 

The heatmap shown in Figure 6 provides a detailed view of the pairwise correlation between 23 different benchmarks used to evaluate 38 VLMs. The scores of these models on each benchmark were utilized to compute this correlation matrix. The color intensity and numerical values represent the degree of correlation, ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation), with warmer colors indicating higher positive correlations and cooler colors indicating weaker or negative correlations. 

We observe that VCREN, HARD is markedly different from other benchmarks in the evaluation set. It demonstrates minimal and even sometimes negative correlation with other benchmarks. This suggests that the skill set required for VCREN, HARD is largely unrelated to those tested by other popular tasks emphasizing more straightforward image-to-text associations or OCR capabilities. Specifically, VCREN, HARD challenges models with tasks that involve high-level reasoning and minimal reliance on pixel-level information, focusing instead on understanding context, commonsense reasoning, and visual narrative interpretation. These features are less critical in other benchmarks, which explains the weak correlation across tasks. VCREN, EASY exhibits a slightly stronger correlation with a few other benchmarks but remains moderately independent of most others. Like VCREN, HARD, VCREN, EASY also evaluates visual commonsense reasoning, but with less stringent requirements, offering models more cues and simpler connections between visual elements and textual understanding. This leads to moderate overlap with benchmarks like TextVQA, which similarly focus on text understanding in a visual context, but VCREN, EASY still emphasizes a higher level of interpretative reasoning than standard text-based vision benchmarks. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_VCR_Visual_Caption_Restoration_images/iclr_2025_VCR_Visual_Caption_Restoration.pdf-0019-04.png)


Figure 6: The heat map of benchmarks displays the correlation between the metric scores of 38 models for each benchmark pair. 

19 

Published as a conference paper at ICLR 2025 

## C INFORMATION OF MODELS EVALUATED 

We provide the specifications of the evaluated models and a link to each model’s weights in Table 6. 

Table 6: Model specifications 

|Model name|Model size|Open-sourced|
|---|---|---|
|Claude 3 Opus|-||
|Claude 3.5 Sonnet|-||
|Gemini 1.5 Pro|-||
|GPT-4 Turbo|-||
|GPT-4o|-||
|GPT-4V|-||
|Qwen-VL-Max|-||
|Reka Core|-||
|Cambrian-1 5|34B|✓|
|CogVLM2 6|19B|✓|
|CogVLM2-Chinese 7|19B|✓|
|DeepSeek-VL 8|1.3B|✓|
|DeepSeek-VL 9|7B|✓|
|DocOwl-1.5-Omni10|8B|✓|
|Idefcs3 11|8B|✓|
|InternLM-XComposer2-VL 12|7B|✓|
|InternLM-XComposer2-VL-4KHD 13|7B|✓|
|InternVL-V2 14|40B|✓|
|Llama-3.2-Vision 15|11B|✓|
|Llama-3.2-Vision 16|90B|✓|
|MiniCPM-V2.5 17|8B|✓|
|Monkey18|7B|✓|
|Pixtral 19|12B|✓|
|Qwen-VL 20|7B|✓|
|Qwen2-VL 21|7B|✓|
|Yi-VL22|34B|✓|
|Yi-VL23|6B|✓|



- 5https://huggingface.co/nyu-visionx/cambrian-34b 

- 6https://huggingface.co/THUDM/CogVLM2-Llama3-chat-19B 

- 7https://huggingface.co/THUDM/cogvlm2-llama3-Chinese-chat-19B 

- 8https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat 

- 9https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat 

- 10https://huggingface.co/mPLUG/DocOwl1.5-Omni 

- 11https://huggingface.co/HuggingFaceM4/Idefics3-8B 

- 12https://huggingface.co/internlm/internlm-xcomposer2-vl-7b 

- 13https://huggingface.co/internlm/internlm-xcomposer2-4khd-7b 

- 14https://huggingface.co/OpenGVLab/InternVL2-40B 

- 15https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct 

- 16https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct 

- 17https://huggingface.co/OpenBMB/MiniCPM-Llama3-V-2 ~~5~~ 

- 18https://huggingface.co/echo840/Monkey-Chat 

19https://huggingface.co/mistralai/Pixtral-12B-2409 

20https://huggingface.co/Qwen/Qwen-VL-Chat 

- 21https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct 

22https://huggingface.co/01-ai/Yi-VL-34B 

23https://huggingface.co/01-ai/Yi-VL-6B 

20 

Published as a conference paper at ICLR 2025 

## D POTENTIAL QA 

**What could be the possible reason that CogVLM performs well in VCR-WIKI series benchmarks?** Many models we tested (DocOwl-1.5, Monkey, MiniCPM-V2.5, InternLM series, InternVL series) follow a similar inference pipeline to adapt to high-resolution application scenarios: 

1. An algorithm divides the input image into segments. 

2. Each segment is encoded into tokens using a CILP-based image encoder. 

3. A filtering mechanism (algorithm/resampler/abstractor) processes the visual tokens. 

4. The filtered tokens are concatenated with language tokens and input to the LLM. 

If, in step 3, pixel-level hints embedded in text within the image (TEL) are disregarded, the model cannot correctly answer the question. Consequently, some of these models may perform better on benchmarks emphasizing global features but struggle on the VCR-WIKI series benchmarks, particularly in the hard partitions. For example, while InternVL2-40B performs best on VCREN, EASY, it does not perform well on VCREN, HARD. As noted in the paper, the easy partition of the benchmark primarily verifies that the VCR task is feasible for the models. In contrast, the hard partition explores the boundaries of VCR capability for both models and human test-takers (who require more time and focus to solve the puzzles in the hard partition). 

The CogVLM2 and Cambrian-1 series, by contrast, do not include step 3 in their inference pipelines. Instead, their image encoders operate at mid-to-high resolutions (1K level), and they resize the input image to match the supported resolution rather than dividing it into segments. The image encoder resolution for CogVLM2 is 1344 _×_ 1344, while Cambrian-1 employs four image encoders, the largest supporting a resolution of 1024 _×_ 1024. This approach may encounter challenges with extremely shaped input images (e.g., 8192 _×_ 1024), but for VCR-WIKI, where images are mostly near-square (on average 300 _×_ 360 for VCRZH, EASY/VCRZH, HARD and 300 _×_ 375 for VCREN, EASY/VCREN, HARD), high-resolution support is not necessary. For instance, InternLM-XComposer2-VL outperforms InternLM-XComposer2-VL-4KHD on this benchmark. 

**What could be the potential way to improve models’ capability on VCR?** To suggest potential avenues for improving VLM performance on VCR, we propose the following: 

1. **Include VCR in VLM pretraining** : Just as OCR parsing tasks are often included in pretraining to improve OCR performance, researchers could consider incorporating VCR tasks during pretraining. We have released vcr ~~t~~ ransform.py[24] to facilitate this process, making it as straightforward as data augmentation. 

2. **Architectural Exploration** : CogVLM2 is the best-performing model on average across the four partitions, and we believe this is largely due to its vision expert architecture. We contacted the CogVLM2 team and learned that GLM-4 and CogVLM2 share the same training data, yet there is a significant performance gap between them on the VCR benchmarks. 

3. **Chain-of-Thought (CoT) Methods** : Researchers could explore multi-modality pipelines based on CoT techniques to improve existing VLMs on VCR tasks Chen et al. (2024a); Zhang et al. (2023). Although a model might not initially focus on the correct visual area (e.g., pixel-level hints in the TEI), CoT-based techniques could help refine its focus over successive rounds. 

**How is the Transferability of VCR-WIKI Finetuning?** In Table 7, we show the transferability of VCR-WIKI by finetuning multiple models on different finetuning datasets’ training sets and testing their performance on a series of benchmarks. 

The analysis of our experimental results highlights the strong transferability of the proposed VCRWIKI dataset across various benchmarks. Notably, models fine-tuned on VCR-WIKI demonstrate significant performance improvements not only within the VCR-WIKI benchmarks themselves, but 

> 24https://github.com/tianyu-z/VCR/blob/main/src/build_dataset/vcr_ transform.py 

21 

Published as a conference paper at ICLR 2025 

also across different language settings. For example, fine-tuning CogVLM2 on VCREN, HARD leads to a substantial increase in performance on the VCRZH, EASY benchmark, elevating the score from 9.15 to 42.55. Similarly, fine-tuning the Chinese version of CogVLM2 on VCRZH, HARD enhances its performance on both the VCREN, EASY and VCREN, HARD benchmarks, with scores rising from 79.9 to 87.57 and from 25.13 to 44.97, respectively. These enhancements indicate that the VCR-wiki dataset facilitates the learning of effective transferable features even when the fine-tuning and evaluation involve different languages. 

Additionally, the consistent achievement of the highest scores within each model’s finetuning variations underscores the robustness of VCR-WIKI in improving model performance across diverse evaluation metrics. This evidence collectively demonstrates that VCR-WIKI serves as a versatile and powerful resource for enhancing model generalization and performance across multiple tasks and linguistic contexts. 

|**Base Model**||**MiniCPM-V2.5**|**MiniCPM-V2.5**|**MiniCPM-V2.5**|**MiniCPM-V2.5**|**CogVLM2**|**CogVLM2**|**CogVLM2**|**CogVLM2-Ch.**|**CogVLM2-Ch.**|**CogVLM2-Ch.**|
|---|---|---|---|---|---|---|---|---|---|---|---|
|**Finetuning**<br>~~**d**~~**ataset**||None|OKVQA-Train|VCREN, HARD|VCRZH, HARD|None|OKVQA-Train|VCREN, HARD|None|OKVQA-Train|VCRZH, HARD|
|**OKVQA**||77.43|72.20|**77.09**|76.38|75.35|71.86|**75.45**|**74.16**|70.57|74.14|
|**VCREN, EASY**||31.81|19.53|**40.96**|28.40|83.25|79.29|**93.27**|79.90|30.18|**87.57**|
|**VCREN, HARD**||1.41|0.00|**13.86**|5.33|37.98|27.22|**77.44**|25.13|3.55|**44.97**|
|**VCRZH, EASY**||4.10|2.12|2.66|**7.44**|9.15|16.49|**42.55**|33.24|16.49|**61.69**|
|**VCRZH, HARD**||0.09|0.53|1.06|**1.53**|0.08|0.00|**1.60**|1.34|0.00|**42.11**|
|**MMstar**||50.20|**51.73**|50.40|50.27|50.50|**51.07**|50.20|52.73|50.87|**54.33**|
|**MMBench**<br>~~**D**~~**EV**|~~**E**~~**N**|**74.54**|74.46|**74.54**|74.30|72.70|**73.53**|72.60|**77.32**|77.09|76.78|
|**MME**||**2024.6**|1996.7|1923.65|1977.13|1869.5|1860.56|**1882.21**|**2040.7**|1898.42|1939.8|
|**MMMU**<br>~~**V**~~**AL**||45.89|**47.78**|46.11|46.56|**42.60**|38.67|38.67|42.44|41.56|**45.00**|
|**AI2D**<br>~~**t**~~**est**||**78.04**|77.85|77.62|77.91|73.40|**74.97**|73.61|**72.64**|70.98|71.31|
|**OCR**<br>**BENCH**||**71.70**|71.60|71.50|71.30|75.40|72.80|**79.80**|77.30|75.20|**79.60**|
|**MMVet**||**53.12**|45.78|51.10|52.66|57.80|45.00|**59.31**|56.38|39.77|**56.88**|
|**MathVista**<br>~~**M**~~**INI**||**54.50**|53.70|52.80|52.70|**38.60**|38.30|35.40|37.80|37.60|**40.20**|
|**ChartQA**||71.80|72.12|71.92|**72.52**|72.80|**80.40**|79.56|63.16|58.92|**65.84**|
|**OCRVQA**||61.85|**62.70**|61.36|61.59|64.90|65.56|**66.11**|32.65|**34.41**|33.24|



Table 7: Performance Comparison of Base Models 

22 

