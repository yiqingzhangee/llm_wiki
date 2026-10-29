# **VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding** 

**Xiang Li Jian Ding Mohamed Elhoseiny** King Abdullah University of Science and Technology `{xiang.li.1,jian.ding,mohamed.elhoseiny}@kaust.edu.sa` 

## **Abstract** 

We introduce a new benchmark designed to advance the development of generalpurpose, large-scale vision-language models for remote sensing images. Although several vision-language datasets in remote sensing have been proposed to pursue this goal, existing datasets are typically tailored to single tasks, lack detailed object information, or suffer from inadequate quality control. Exploring these improvement opportunities, we present a **V** ersatile vision-language **Bench** mark for **R** emote **S** ensing image understanding, termed **VRSBench** . This benchmark comprises 29,614 images, with 29,614 human-verified detailed captions, 52,472 object references, and 123,221 question-answer pairs. It facilitates the training and evaluation of vision-language models across a broad spectrum of remote sensing image understanding tasks. We further evaluated state-of-the-art models on this benchmark for three vision-language tasks: image captioning, visual grounding, and visual question answering. Our work aims to significantly contribute to the development of advanced vision-language models in the field of remote sensing. The data and code can be accessed at `https://vrsbench.github.io` . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2024_VRSBench_A_Versatile_Vision-Language_Benchmark_Dataset_for_Remote_Sensing_Image_Understanding_images/nips_2024_VRSBench_A_Versatile_Vision-Language_Benchmark_Dataset_for_Remote_Sensing_Image_Understanding.pdf-0001-05.png)


**----- Start of picture text -----**<br>
1 Object Referring<br>2 Object ID=1: The small vehicle that is the farthest to the top.<br>0 Object ID=4: The harbor located on the left side of the scene with multiple docks extending into the water.<br>3 Object ID=7: The harbor situated on the right side of the image with a large dock area.<br>Visual Question Answer<br>Question:  How many harbors are visible?  Answer:  3<br>Question:  What is the object located furthest to the top?  Answer:  Small vehicle<br>Question:  Are the visible vehicles near water?  Answer:  No<br>Detailed Captioning<br>7 The high-resolution aerial image from GoogleEarth shows a waterfront scene with residential areas and<br>6 harbor facilities. Three distinct harbors can be seen, one located on the left side and another on the right<br>4 side of the image. Between them, there are homes with different colored rooftops, green lawns, and<br>5 driveways. A ship is docked in the central part of the bottom edge, and the water body exhibits gentle<br>ripples. Various small vehicles are scattered throughout the residential area, parked near the houses.<br>**----- End of picture text -----**<br>


Figure 1: Examples of an image and corresponding annotations in VRSBench dataset. Our annotations include object referring, visual question answering, and detailed captions. 

## **1 Introduction** 

Remote sensing models seek to understand the Earth’s surface using imagery captured from overhead, offering a unique perspective of our physical world. This technique is instrumental in various applications, such as land use mapping, urban planning, precision agriculture, disaster management, etc. In the past few years, the success of large vision-language models (LVLMs)[1, 2, 3, 4, 5] in natural 

Preprint. Under review. 

scenes has inspired a trend of applying LVLMs to remote sensing [6]. Recent efforts have explored LVLMs for various remote sensing image understanding tasks, including scene classification [7, 8], image captioning [9, 10, 11, 12, 13, 14, 15, 16, 17], visual grounding [18, 19, 20], visual question answering (VQA) [21, 22, 23, 24, 25, 26], and general-purpose models [27, 28], etc. 

However, directly applying LVLMs to remote sensing images presents challenges. LVLMs are typically trained on _internet data_ , which differs significantly from remote sensing data. Remote sensing images often feature very small objects (sometimes only 10 pixels) and require complex spatial reasoning from an overhead view. Building effective LVLMs for remote sensing requires large-scale, high-quality datasets tailored to this field. Recent works[27, 28, 29] have attempted to train LVLMs with a combination of existing text-enriched remote sensing data, achieving reasonable performance. However, further improvements are limited by the current vision-language datasets in remote sensing, which have the following limitations: 

**(i)** Existing vision-language datasets primarily cater to single image perception tasks, e.g., image captioning. Recent works explore integrating multiple datasets to accommodate a wider array of tasks [28, 29]. Such integration, while crucial, introduces challenges including inconsistent data annotations, variations in data quality, and the complexity of merging different data formats and sources, all of which can hinder model performance and scalability. 

**(ii)** Most commonly used remote sensing image caption datasets, such as UCM-Captions [30] and RSICD [10], provide only brief descriptions, lacking detailed object information. Recent work RSGPT [27] provides high-quality, human-generated detailed image captions; however, the dataset comprises only 2,585 image-text pairs. This limited scope restricts its potential for training robust vision-language models in remote sensing applications. Although recent works, such as RS5M [31] and RemoteClip [7], introduced large-scale remote sensing image-text pair datasets, these annotations are automatically generated by image caption models and lack human verification. Given the current limitations of automatic captioning technology, such image-text data often suffer from accuracy issues and a lack of quality control. 

**(iii)** Most existing remote sensing visual grounding datasets are designed under simplistic scenarios where the referring objects typically stand alone within their category. For instance, in the widely used DIOR-RSVG [19] datasets, a large portion of objects are unique within the categories, which leads to 38.36% of objects being easily distinguished by the object category alone. Finally, the majority of current VQA datasets in remote sensing employ automated methods for generating question-answer pairs. These automatically generated pairs often encompass a limited variety of unique questions and answers, which may not be sufficiently diverse to facilitate open-ended question-answering in real-world applications. 

In this study, to address these limitations, we introduce a novel versatile benchmark for visionlanguage understanding of remote sensing images. VRSBench comprises 29,614 images, each enriched with human-verified detailed captions, complex object referring, and question-answer pairs, check Table 1 for a detailed comparison with existing datasets. This dataset facilitates the training and evaluation of vision-language models across a spectrum of remote sensing image understanding tasks. Fig. 1 gives an example of a selected image and associated annotations. 

The key contributions of our work are summarized as follows: 

- We introduce a new semi-automatic vision-language data collection pipeline which includes four key steps: object attributes extraction, prompt engineering, GPT-4 inference, and human verification. This pipeline enables a fast collection of large-scale datasets with human-level annotation quality. 

- Based on the semi-automatic data collection pipeline, we collect VRSBench dataset that provides detailed image captioning, visual grounding, and visual question-answer labels in a unified dataset, and therefore, enables a comprehensive evaluation of multiple visionlanguages capabilities based on this dataset. 

- VRSBench provides large-scale human-verified annotations that feature several advantages: 1) it provides a large-scale collection of human-verified, high-quality captions rich in object details; 2) it offers more realistic object refers in which each referring sentence unambiguously identifies an object among multiple similar ones within the same category; 3) it features a diverse collection of open-ended question-answer pairs in natural language. 

2 

- We develop three benchmarks based on our VRSBench dataset, including detailed image caption, visual grounding, and visual question answering, and evaluate the performance of several state-of-the-art LVLMs. 

Table 1: Comparison between existing remote sensing vision-language datasets and our VRSBench dataset. Values in parentheses in the Caption column indicate the average number of words in captions. OBB denotes orientated bounding box. A small portion of question-answer pairs in RSIVQA are annotated by human annotators. 

|Dataset|Year|#Image|Caption<br>Grounding<br>VQA<br>Human<br>#Captions<br>Details<br>#Refers<br>OBB<br>#VQAs<br>Open-ended|
|---|---|---|---|
|UCM-Captions [30]<br>RSICD [10]<br>RS5M [31]<br>RSICap [27]|2016<br>2017<br>2023<br>2023|2,100<br>10,921<br>5M<br>2,585|10,500 (12)<br>�<br>0<br>�<br>0<br>-<br>�<br>54605 (12)<br>�<br>0<br>�<br>0<br>-<br>�<br>5M (49)<br>�<br>0<br>�<br>0<br>-<br>�<br>2,585(60)<br>�<br>0<br>�<br>0<br>-<br>�|
|RSVG [18]<br>DIOR-RSVG [19]<br>RRSIS-D[20]|2022<br>2023<br>2024|4,239<br>17,402<br>17,402|0<br>�<br>7,933<br>�<br>0<br>-<br>�<br>0<br>�<br>38,320<br>�<br>0<br>-<br>�<br>0<br>�<br>17,402<br>�<br>0<br>-<br>�|
|RSVQA-HR [21]<br>RSIVQA [22]<br>VQA-TextRS [24]<br>RSIEval[27]|2020<br>2021<br>2022<br>2023|10,659<br>37,264<br>2,144<br>100|0<br>�<br>0<br>�<br>1,066,316<br>�<br>�<br>0<br>�<br>0<br>�<br>111,134<br>�<br>�<br>0<br>�<br>0<br>�<br>6,245<br>�<br>�<br>0<br>�<br>0<br>�<br>933<br>�<br>�|
|VRSBench|2024|29,614|29,614 (52)<br>�<br>52,472<br>�<br>123,221<br>�<br>�|



## **2 Pipeline** 

To construct our VRSBench dataset, we employed multiple data engineering steps, including attribute extraction, prompting engineering, GPT-4 inference [32], and human verification. These processes are meticulously designed to ensure the integrity and utility of the dataset for remote sensing applications. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2024_VRSBench_A_Versatile_Vision-Language_Benchmark_Dataset_for_Remote_Sensing_Image_Understanding_images/nips_2024_VRSBench_A_Versatile_Vision-Language_Benchmark_Dataset_for_Remote_Sensing_Image_Understanding.pdf-0003-05.png)


**----- Start of picture text -----**<br>
Extracted image and object information: {'source':  image source, 'resolution':  image resolution,<br>Attribute<br>'objects': [{'obj_id':  object id, 'obj_cls':  object category, 'obj_bbox': bounding box, 'is_unique':<br>Extraction<br>unique within category, 'obj_position': object position, 'obj_size': object size, ...}]}<br>Instructions: you are an AI visual assistant tasked with analyzing remote sensing images. Given<br>Prompt<br>an input image and object information. Your job is to create a detailed image caption, referring<br>Engineering<br>sentences for distinct objects, as well as question-answer pairs…<br>Inputs: image, image and object information in JSON format.<br>GPT-4<br>Outputs: {'caption', detailed image caption, 'objects': [{obj_id, referring_sentence},...], 'qa_pairs':<br>Inference<br>['ques_id': question id, 'question': question, 'type': question type, 'answer': answer]}<br>Guidelines: 1) modify uncertain or meaningless elements; 2) modify error referring; 2) remove<br>Human<br>self-answering questions…<br>Verification<br>Outputs: corrected annotations.<br>**----- End of picture text -----**<br>


Figure 2: Dataset creation pipeline. We generate object information from detection labels and use carefully designed instructions to prompt GPT-4 to generate annotations from input images along with object information. All annotations are verified by human annotators. 

## **2.1 Attribute Extraction** 

Initially, we extract image information, including the source and resolution, as well as object information—such as the object category, bounding box, color, position (absolute and relative), and size (absolute and relative)—from existing object detection datasets. We also determine whether an object is unique within its category, which is important for crafting accurate reference sentences. 

3 

In this study, we utilize two prominent open-access object detection datasets, DOTA-v2 [33] and DIOR [34], to develop our VRSBench dataset. Due to the unavailability of test labels for DOTA-v2, we incorporate only its training and validation sets. We divide each image into patches measuring 512 × 512 pixels. Notably, each image patch from DOTA-v2 contains, on average, 14.2 instances, while each patch from the DIOR dataset averages only 3.3 instances. This higher instance density in DOTAv2 offers a more challenging and diverse training environment compared to existing remote sensing visual grounding datasets, such as DIOR-RSVG [19] and RRSIS-D [35], that are typically sourced from the DIOR dataset. Moreover, the visual grounding task predominantly involves identifying horizontal bounding boxes (HBB) based on referential descriptions. By constructing our dataset upon the framework of DOTA-v2, VRSBench facilitates the grounding of objects with orientated bounding boxes (OBB), thereby extending the capabilities of traditional visual grounding methods. 

## **2.2 Prompt Engineering** 

We carefully design the following instructions to prompt GPT-4V to create detailed image captions, object referring, and question-answer pairs. Detailed instructions for each task are provided in the supplementary. 

“You are an AI visual assistant tasked with analyzing remote sensing images. For each image, you receive image meta information and a list of objects in the format of .... Your job is to create a detailed image caption and referring sentences for 1-5 distinct objects, if multiple are present, as well as 3-10 question-answer pairs. Each referring sentence should unambiguously refer to one object. Finally, you need to return a JSON file in the format: {caption: detailed image caption, objects: [obj_id: object id, ref: referring sentence,...], qa_pairs: [ques_id: question id, question: question, type: question type, answer: answer]}. Do not return any notes after the JSON.” 

## **2.3 GPT-4V Inference** 

Given input prompts, we call OpenAI API[1] to automatically generate annotations. We iteratively refine our instructional prompts to generate annotations, meticulously enhancing these instructions to ensure the quality of the annotations. In the responses generated by GPT-4V, undesirable terms, such as “not provide”, “not specified”, and “unknown”, may be present. Should any of the specified excluding phrases appear in GPT-4V’s output, the procedure requires that GPT-4V be recursively invoked to regenerate responses until the output is free of any excluding phrases. This iterative process is attempted a maximum of five times, after which the final response is utilized for generating annotations. Ultimately, any caption sentences, object-referring sentences, or question-answer pairs containing these excluding phrases are excised from final annotations. 

## **2.4 Human Verification** 

With our carefully designed prompts, most of the annotations generated by GPT-4V are accurate. Nevertheless, a significant number of outputs remain suboptimal. This shortfall is likely attributable to the model’s limited exposure to remote sensing imagery, which impedes its capacity to interpret complex structures within these images. Additionally, it is important to note that even advanced language models, such as the GPT-4V system, exhibit a degree of hallucinatory outputs [36]. 

To improve the quality of the dataset, we engage human annotators to validate each annotation generated by GPT-4V. This validation process incorporates domain experts to guarantee that annotators have a comprehensive understanding of the assigned tasks. Initially, domain experts establish detailed guidelines, which include directives such as: 1) eliminate any uncertain or irrelevant elements; 2) ensure each referring sentence unambiguously identifies the intended object; 3) exclude questions that inherently contain their answers. More details about the annotation guidelines can be found in the supplementary. The verification of each image requires approximately 120 seconds, culminating in a total of 1,004 hours devoted to human verification. Each image verification costs around 0.21 USD and leads to a total cost of 6,200 USD for human verification. To enhance the quality of our dataset, we have instituted a secondary validation phase involving a meticulous re-evaluation of 2,000 images. This step is designed to uncover prevalent annotation discrepancies and to refine the annotators’ understanding of the task requirements. 

> 1https://platform.openai.com/docs/api-reference 

4 

## **3 VRSBench Dataset** 

## **3.1 Dataset Overview** 

Our VRSBench dataset contains 29,614 remote sensing images, with high-quality human-verified annotations. It comprises 29,614 caption sentences, 52,472 referring sentences, and 123,221 questionanswer pairs. Each image is of 512 _×_ 512 pixels. Details of each type of annotation are given below. Note that original object detection labels and object attributes are also provided in our annotations. 

## **3.2 Detailed Caption** 

VRSBench captions provide comprehensive descriptions that encompass both abstract image attributes and detailed object-specific information. Each caption initiates with a general overview of the image, subsequently delving into explicit and precise details present within the image. Attributes of the image include the source, resolution, color or panchromatic distinction, and the type of scene depicted. Conversely, object attributes cover object quantity, color, shape, size, and spatial positioning of each object, encompassing both its absolute location within the image and its relative positioning in relation to other objects. Descriptions are confined to manifest features, eschewing any elements that are uncertain or ambiguous. Additionally, captions may incorporate other visually discernible objects not supplied by the source object detection datasets, such as buildings, houses, roads, and trees, if these elements are clear and unambiguous. Each caption typically comprises 3-7 sentences, with an average length of 54 words. A summary of these caption statistics is detailed in Fig. 3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2024_VRSBench_A_Versatile_Vision-Language_Benchmark_Dataset_for_Remote_Sensing_Image_Understanding_images/nips_2024_VRSBench_A_Versatile_Vision-Language_Benchmark_Dataset_for_Remote_Sensing_Image_Understanding.pdf-0005-05.png)


**----- Start of picture text -----**<br>
#images 29,614<br>#vocabulary size 9,588<br>#total words 1,526,338<br>#caption sentences 114,366<br>Avg. #sentences in caption 4<br>Avg. caption length 52<br>(a) PDF of word count (b) PDF of sentence number (c) Statistics of VRSBench captions.<br>**----- End of picture text -----**<br>


Figure 3: Statistics of the VRSBench caption dataset. (a) Probability density function (PDF) of caption length. (b) PDF of the sentence number. (c) Summative statistics. 

## **3.3 Object Referring** 

In VRSBench, each image is analyzed to identify 1-5 distinct objects, and referring sentences are provided for each. These sentences are carefully crafted such that each can independently and unambiguously identify an object without reliance on other sentences. We utilize distinctive features to clearly differentiate the referred objects from others within the image. These features span a variety of object attributes including color, shape, position, size, relative position, and relative size, among others. Note that the original DOTA-v2 and DIOR datasets contain 18 and 20 object categories respectively, which are merged into 26 object categories in our dataset. Please check the supplementary for category merging details. Figure 4 provides a summary of referring sentences of our VRSBench dataset. 

## **3.4 Visual Question Answering** 

Based on all visible elements and object information, we provide 3-10 question-answer pairs about diverse types, including object category, object existence, object quantity, object color, object size, object position, direction, scene characteristics, and complex reasoning, and provide an answer for each question. Instead of only focusing on objects from source detection datasets, we also ask questions about objects that are not provided, such as houses, roads, and trees if they are obvious and non-ambiguous. When collecting annotations, we ensure each question has a definite answer without any ambiguity, and answer each question using a single word or phrase. We show the statistics of question-answer pairs in Figure 5. 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2024_VRSBench_A_Versatile_Vision-Language_Benchmark_Dataset_for_Remote_Sensing_Image_Understanding_images/nips_2024_VRSBench_A_Versatile_Vision-Language_Benchmark_Dataset_for_Remote_Sensing_Image_Understanding.pdf-0006-00.png)


Figure 4: Statistics of object referring sentences of VRSBench dataset. (a) Distribution of the 10 most frequent object categories. (b) Distribution of the word length of referring sentences. (c) Distribution of object size. (d)Word cloud of the top 50 words in referring sentences. (e) Distribution of unique/non-unique objects in each category. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2024_VRSBench_A_Versatile_Vision-Language_Benchmark_Dataset_for_Remote_Sensing_Image_Understanding_images/nips_2024_VRSBench_A_Versatile_Vision-Language_Benchmark_Dataset_for_Remote_Sensing_Image_Understanding.pdf-0006-02.png)


Figure 5: Statistics of question-answer pairs in VRSBench. (a) Distribution of question types. (b) Word cloud of top 50 most frequent words in questions. (c) Word cloud of top 50 most frequent words in answers. 

## **4 Benchmark Evaluation** 

## **4.1 Benchmark Overview** 

Based on VRSBench, we construct three distinct tasks for advancing remote sensing image understanding: 

- VRSBench-Cap: This challenge requires the prediction of a comprehensive description for a given remote sensing image, encapsulating intricate object details and contextual relevance. 

- VRSBench-Ref: The task involves identifying and localizing specific objects from a given remote sensing image based on textual descriptions. 

- VRSBench-VQA: This task aims to answer questions related to visual content in a given remote sensing image. 

To facilitate benchmark evaluation, we partition our VRSBench dataset into two distinct, nonoverlapping splits designated for model training and evaluation. We split the datasets according to official splits of DOTA [33] and DIOR [34] datasets, where their training images are used to build the training set of VRSBench and their validation sets are used as the test set. Table 2 delineates the statistics of two splits. 

6 

For the above three tasks, we benchmark state-of-the-art models, including LLaVA-1.5 [37], MiniGPT-v2 [38], and GeoChat [28], to demonstrate the potential of LVLMs for remote sensing image understanding. LLaVA-1.5 [37], MiniGPTv2 [38], Mini-Gemini [2], and GeoChat [28] are generalist models that are naturally designed for general-proposed image understanding. We, therefore, report the performance of these methods under joint training of all three tasks, i.e., image captioning, visual grounding, and VQA. 

Table 2: VRSBench data split. 

||#Images<br>#Captions<br>#Refers<br>#VQAs|train<br>20,264<br>20,264<br>36,313<br>85,813|test<br>9,350<br>9,350<br>16,159<br>37,408|
|---|---|---|---|



Note that all these comparison methods include a multi-stage training process. To ensure a fair comparison, we reload the models that are initially trained on large-scale image-text alignment datasets, and then finetune each method using the training set of our VRSBench dataset. We employ CLIP-ViT(L-14) [39] as the vision encoder and use the Vicuna-7B model [40] as the Large Language Model (LLM). For LLaVA-1.5 [37], Mini-Gemini [2], and GeoChat [28], we adhere to the original model specifications, utilizing two MLP layers with GeLU activation [41]. For MiniGPT-v2[38], we implement a single MLP layer as described in the original paper. For each comparing method, we finetune the model on the training set of VRSBench dataset for 5 epochs. Following GeoChat [28], we use LoRA [42] finetuning to finetune all comparing methods, with a rank of 64. To understand the benefit of fientuning on VRSBench, we include the baseline GeoChat [28] without training on our VRSBench dataset for comparison. 

We further evaluate the performance of GPT-4V, which is generally known as one of the most powerful close-source vision-language models, on three tasks based on our VRSBench dataset. To achieve this, we directly call GPT-4V API to generate detailed captions, referring object locations, and answers for visual questions, with the following instructions. Note that we do not include object information in this experiment. 

## **4.2 Detailed Image Caption** 

**Evaluation metrics** . For model evaluation, we follow standard practices by utilizing a set of established metrics including BLEU [43], ROUGE_L [44], METEOR [45], and CIDEr [46]. For BLEU, we consider n-gram precision with n values of 1, 2, 3, and 4. We also report average caption lengths to assess the details of generated captions. Furthermore, we note that traditional caption evaluation metrics may not be suitable for long captions. We, therefore, use an LLM-based caption evaluation metric called CHAIR[2] proposed in [47] for our detailed image caption task. 

**Results.** Table 3 shows the comparative performance of different methods in detailed image captioning of our VRSBench dataset. As demonstrated in the table, the baseline GeoChat model, when not finetuned on the VRSBench dataset, exhibits significantly poorer performance compared to models that have been finetuned on VRSBench. The LLaVA-1.5 [37] model that has undergone fine-tuning on VRSBench achieves the highest performance, reaching a BLEU-1 score of 48.1 and a CIDEr score of 33.9. The GPT-4V model shows the best performance on the CHAIR score. This is expected because the CHIAR score itself is calculated using GPT-4 in our experiments. Moreover, the generated captions have an average word length of 49, which closely approximates the average length of ground truth captions. Note that detailed image captioning is a more challenging task than conventional image caption, therefore, the performance falls far behind. More advanced vision-language modeling techniques are desired to handle this challenging task. 

Table 3: Detailed image caption performance on VRSBench dataset. Avg_L denotes the average word length of generated captions. Boldface indicates the best performance. 

|Method|BLEU-1<br>BLEU-2<br>BLEU-3<br>BLEU-4<br>METEOR<br>ROUGE_L<br>CIDEr|CHAIR|Avg_L|
|---|---|---|---|
|GeoChat w/o ft [28]<br>GPT-4V [32]|13.9<br>6.6<br>3.0<br>1.4<br>7.8<br>13.2<br>0.4<br>37.2<br>22.5<br>13.7<br>8.6<br>20.9<br>30.1<br>19.1|0.42<br>**0.83**|36<br>67|
|MiniGPT-v2 [38]<br>LLaVA-1.5 [37]<br>GeoChat [28]<br>Mini-Gemini [48]|36.8<br>22.4<br>13.9<br>8.7<br>17.1<br>30.8<br>21.4<br>**48.1**<br>**31.5**<br>**21.2**<br>**14.7**<br>**21.9**<br>**36.9**<br>**33.9**<br>46.7<br>30.2<br>20.1<br>13.8<br>21.1<br>35.2<br>28.2<br>47.6<br>31.1<br>20.9<br>14.3<br>21.5<br>36.8<br>33.5|0.73<br>0.78<br>0.77<br>0.77|37<br>49<br>52<br>47|



2https://github.com/davidmchan/clair 

7 

## **4.3 Visual Grounding** 

**Benchmark settings** . In this study, we focus on the grounded localization task, which aims to predict bounding boxes for referring objects. In our experiments, we use horizontal bounding boxes for model training and evaluating the grounding performance. Results on OBB visual grounding can be found in the supplementary. 

**Evaluation metrics** . For model evaluation, we employ the metric accuracy@ _τ_ to assess performance. Accuracy is determined by calculating the Intersection over Union (IoU) between the predicted bounding box and the ground-truth box. A prediction is considered accurate if the IoU exceeds the threshold _τ_ . In our experiments, we choose two different IoU thresholds, i.e., 0.5 and 0.7. 

**Results.** Table 3 shows the visual grounding performance of different methods on our VRSBench dataset. From the table, the model finetuned on the VRSBench significantly outperforms the baseline GeoChat model without finetuning. Furthermore, all models demonstrate superior performance in tasks involving unique object referring compared to non-unique object referring. This superiority is expected, as it is generally easier to localize objects uniquely identified within their categories than to differentiate among multiple instances within the same category. Note that even though MiniGPT-v2 gets worse overall grounding performance, it performs better at grounding non-unique objects. 

Furthermore, GPT-4V exhibits markedly inferior performance compared to models specifically trained on image captioning and visual grounding tasks, primarily due to the absence of object information in its prompts. Despite the notable successes of existing closed-source multimodal large language models (LLMs), such as GPT-4, in comprehending natural images, their effectiveness is notably reduced when not fine-tuned on remote sensing imagery. 

More importantly, even the best-performing GeoChat model fails to achieve satisfactory performance levels, with a grounding accuracy of 49.8% at a threshold of 0.5. This shortfall is attributed to the demanding scenarios presented in the VRSBench dataset, which includes multiple instances of the same category as the target object. This highlights the necessity for more advanced vision grounding techniques to effectively tackle these complexities. 

Table 4: Visual grounding performance on VRSBench dataset. Boldface indicates the best performance. 

|Method|Unique<br>Acc@0.5<br>Acc@0.7|Non Unique<br>Acc@0.5<br>Acc@0.7|All|
|---|---|---|---|
||||Acc@0.5<br>Acc@0.7|
|GeoChat w/o ft [28]<br>GPT-4V [32]|20.7<br>5.4<br>8.6<br>2.2|7.3<br>1.7<br>2.5<br>0.4|12.9<br>3.2<br>5.1<br>1.1|
|MiniGPT-v2 [38]<br>LLaVA-1.5 [37]<br>GeoChat [28]<br>Mini-Gemini [48]|40.7<br>18.9<br>51.1<br>16.4<br>**57.4**<br>**22.6**<br>41.1<br>9.6|32.4<br>15.2<br>34.8<br>11.5<br>**44.5**<br>**18.0**<br>22.3<br>4.9|35.8<br>16.8<br>41.6<br>13.6<br>**49.8**<br>**19.9**<br>30.1<br>6.8|



## **4.4 Visual question answering** 

**Evaluation metrics** . We categorize the questions in the test set into 10 distinct types: object category, presence, quantity, color, shape, size, position, direction, scene characteristic, and reasoning. The first eight categories relate to object-level questions, whereas the last two are aligned with scene-level, and reasoning-level questions, respectively. We present the overall accuracy as well as the accuracy for each individual question type. To ensure a robust evaluation, we use GPT-4 to determine for each question whether the answers match ground truth texts, with the prompt: _“Question: {question}, Ground Truth Answer: {ground_truth}, Predicted Answer: {predicted answer}. Does the predicted answer match the ground truth? Answer 1 for match and 0 for not match. Use semantic meaning not exact match. Synonyms are also treated as a match, e.g., pond and swimming pool.”_ . 

**Results.** Table 5 shows the VQA performance of different methods on our VRSBench dataset. As shown in the table, the baseline GeoChat [28] model without finetuning gets an average accuracy of 40.8%. Further finetuning on our VRSBench training set significantly boosts the average accuracy to 76.0%. GPT-4V gets a reasonable performance but still falls a lot behind fine-tuned models, 

8 

suggesting that detailed object information contributes a lot to the visual question answering task on our benchmark. 

Table 5: Visual question answering performance on VRSBench dataset. Boldface indicates the best performance. 

|Method|Category<br>Presence<br>Quantity<br>Color<br>Shape<br>Size<br>Position<br>Direction<br>Scene<br>Reasoning|All|
|---|---|---|
|# VQAs|5435<br>7789<br>6374<br>3550<br>1422<br>1011<br>5829<br>477<br>4620<br>902||
|GeoChat w/o ft [28]<br>GPT-4V [32]|48.5<br>85.9<br>19.2<br>17.0<br>18.3<br>32.0<br>43.4<br>42.1<br>44.2<br>57.4<br>67.0<br>87.6<br>45.6<br>71.0<br>70.8<br>54.3<br>67.2<br>50.7<br>69.8<br>72.4|40.8<br>65.6|
|MiniGPT-v2 [38]<br>LLaVA-1.5 [37]<br>GeoChat [28]<br>Mini-Gemini [48]|61.3<br>26.0<br>46.1<br>51.0<br>41.8<br>11.2<br>17.1<br>12.4<br>49.3<br>21.9<br>86.9<br>91.8<br>58.2<br>69.9<br>72.2<br>**61.5**<br>**69.5**<br>**56.7**<br>**83.9**<br>73.4<br>86.5<br>**92.1**<br>56.3<br>70.1<br>73.8<br>60.4<br>69.3<br>53.5<br>83.7<br>73.5<br>**87.8**<br>92.1<br>**58.8**<br>**74.0**<br>**75.3**<br>58.0<br>68.0<br>56.7<br>83.2<br>**74.4**|38.2<br>76.4<br>76.0<br>**77.8**|



## **5 Related Work** 

## **5.1 Remote Sensing Image Captioning Datasets** 

Image captioning in remote sensing is a well-established task that focuses on creating descriptive text for overhead imagery. Commonly used datasets such as UCM-Captions [30], Syndey-Captions [30], and RSICD [10] have been instrumental by offering brief scene descriptions. However, these datasets typically provide short and less detailed captions that overlook intricate object details. Recent efforts, such as RSGPT [27], have introduced high-quality, human-generated detailed captions, though the dataset is limited to just 2,585 image-text pairs, which hampers its utility for developing robust visionlanguage models in remote sensing. In contrast, RS5M [31] introduced a substantial dataset featuring 5 million detailed captions. However, these captions are generated automatically, resulting in quality that is not guaranteed. In stark contrast, our VRSBench dataset includes 29,614 human-verified captions that are not only of high quality but also rich in detail, ensuring both reliability and practical utility for advanced remote sensing applications. 

## **5.2 Remote Sensing Visual Grounding Datasets** 

Visual grounding in remote sensing has recently emerged as an intriguing field of study. Unlike referring expressions in natural images, those in RSVG frequently involve complex geospatial relationships, and the objects of interest may not be prominently visible. The first RSVG dataset was introduced in [18], featuring 4,239 images from GoogleEarth and 7,993 referring expressions. Subsequently, Zhan et al. [19] introduced the DIOR-RSVG dataset, which includes 17,402 remote sensing images and 38,320 referring expressions across 20 object categories. Recent studies [35, 20] have developed visual grounding datasets for remote sensing that include object segmentation; however, these tend to be smaller in scale. In contrast, our VRSBench dataset incorporates a substantial number of object-referring expressions. 

## **5.3 Remote Sensing Visual Question Answering Datasets** 

RSVQA [21] established the first VQA benchmark dataset for remote sensing images. This dataset comprises RS images sourced from OpenStreetMap, accompanied by automatically generated questions and answers. It includes 772 images with 77,232 question-answer pairs in the low-resolution collection and 10,659 images with 1,066,316 pairs in the high-resolution collection. Zheng et al.[22] launched the RSIVQA dataset, a remote sensing VQA dataset that features approximately 37k images and 110,000 question-answer pairs. A small portion of question-answer pairs in RSIVQA are annotated by human annotators. Al et al.[24] introduced an innovative dataset, VQA-TextRS, which consists of 2,144 RS images and 6,245 question-answer pairs generated and annotated by humans in an open-ended format. More recently, the RSIEval[27] dataset features 936 human-crafted questionanswer pairs from 100 remote sensing images. Similarly, our VRSBench dataset also incorporates open-ended question-answer pairs, created by GPT-4V and validated by human annotators, with 123,221 question-answer pairs. 

9 

## **6 Conclusion and future work** 

In this work, we introduce VRSBench, a versatile vision-language dataset and benchmark for remote sensing image understanding. This comprehensive dataset not only addresses the limitations of previous datasets that either ignore detailed object information or suffer from quality control issues but also enriches the field by providing a diverse range of annotations including detailed captions, object referring, and visual question answering with rich object information and verified by human annotators. Our benchmark challenges, specifically designed around the VRSBench dataset, demonstrate the practical utility of our dataset in advancing the capabilities of vision-language models in the domain of remote sensing. 

Currently, the VRSBench dataset is limited to annotations for RGB images. In future work, we aim to enhance VRSBench by incorporating annotations from a variety of remote sensing data types, including infrared images, multi- and hyperspectral images, Synthetic Aperture Radar (SAR) images, and temporal datasets. This expansion will significantly broaden the dataset’s utility across diverse observation conditions, facilitating more accurate and timely applications in remote sensing. 

## **7 Broader Impact** 

VRSBench provides a comprehensive benchmark for developing and evaluating generalist visionlanguage models in both remote sensing and computer vision. This dataset not only supports the training and evaluation of advanced vision-language models but also boosts their ability to tackle complex real-world scenarios in remote sensing. 

10 

## **References** 

- [1] OpenAI. Chatgpt. `https://www.openai.com/chatgpt` , 2023. Accessed: 2024-04-01. 

- [2] Google. Gemini. `https://gemini.google.com/` , 2023. Accessed: 2024-04-01. 

- [3] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _Advances in neural information processing systems_ , 36, 2024. 

- [4] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- [5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. _arXiv preprint arXiv:2308.12966_ , 2023. 

- [6] Xiang Li, Congcong Wen, Yuan Hu, Zhenghang Yuan, and Xiao Xiang Zhu. Vision-language models in remote sensing: Current progress and future trends. _IEEE Geoscience and Remote Sensing Magazine_ , 2024. 

- [7] Fan Liu, Delong Chen, Zhangqingyun Guan, Xiaocong Zhou, Jiale Zhu, Qiaolin Ye, Liyong Fu, and Jun Zhou. Remoteclip: A vision language foundation model for remote sensing. _IEEE Transactions on Geoscience and Remote Sensing_ , 2024. 

- [8] Xiang Li, Congcong Wen, Yuan Hu, and Nan Zhou. Rs-clip: Zero shot remote sensing scene classification via contrastive vision-language supervision. _International Journal of Applied Earth Observation and Geoinformation_ , 124:103497, 2023. 

- [9] Zhenwei Shi and Zhengxia Zou. Can a machine generate humanlike language descriptions for a remote sensing image? _IEEE Transactions on Geoscience and Remote Sensing_ , 55(6):3623–3634, 2017. 

- [10] Xiaoqiang Lu, Binqiang Wang, Xiangtao Zheng, and Xuelong Li. Exploring models and data for remote sensing image caption generation. _IEEE Transactions on Geoscience and Remote Sensing_ , 56(4):2183– 2195, 2017. 

- [11] Xiangrong Zhang, Xiang Li, Jinliang An, Li Gao, Biao Hou, and Chen Li. Natural language description of remote sensing images based on deep learning. In _2017 IEEE International Geoscience and Remote Sensing Symposium (IGARSS)_ , pages 4798–4801. IEEE, 2017. 

- [12] Xiangrong Zhang, Xin Wang, Xu Tang, Huiyu Zhou, and Chen Li. Description generation for remote sensing images using attribute attention mechanism. _Remote Sensing_ , 11(6):612, 2019. 

- [13] Yangyang Li, Shuangkang Fang, Licheng Jiao, Ruijiao Liu, and Ronghua Shang. A multi-level attention model for remote sensing image captions. _Remote Sensing_ , 12(6):939, 2020. 

- [14] Qi Wang, Wei Huang, Xueting Zhang, and Xuelong Li. Word–sentence framework for remote sensing image captioning. _IEEE Transactions on Geoscience and Remote Sensing_ , 59(12):10532–10543, 2020. 

- [15] Xuelong Li, Xueting Zhang, Wei Huang, and Qi Wang. Truncation cross entropy loss for remote sensing image captioning. _IEEE Transactions on Geoscience and Remote Sensing_ , 59(6):5246–5257, 2020. 

- [16] Rui Zhao, Zhenwei Shi, and Zhengxia Zou. High-resolution remote sensing image captioning based on structured attention. _IEEE Transactions on Geoscience and Remote Sensing_ , 60:1–14, 2021. 

- [17] Usman Zia, M Mohsin Riaz, and Abdul Ghafoor. Transforming remote sensing images to textual descriptions. _International Journal of Applied Earth Observation and Geoinformation_ , 108:102741, 2022. 

- [18] Yuxi Sun, Shanshan Feng, Xutao Li, Yunming Ye, Jian Kang, and Xu Huang. Visual grounding in remote sensing images. In _Proceedings of the 30th ACM International Conference on Multimedia_ , pages 404–412, 2022. 

- [19] Yang Zhan, Zhitong Xiong, and Yuan Yuan. Rsvg: Exploring data and models for visual grounding on remote sensing data. _IEEE Transactions on Geoscience and Remote Sensing_ , 61:1–13, 2023. 

- [20] Sihan Liu, Yiwei Ma, Xiaoqing Zhang, Haowei Wang, Jiayi Ji, Xiaoshuai Sun, and Rongrong Ji. Rotated multi-scale interaction network for referring remote sensing image segmentation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2024. 

- [21] Sylvain Lobry, Diego Marcos, Jesse Murray, and Devis Tuia. Rsvqa: Visual question answering for remote sensing data. _IEEE Transactions on Geoscience and Remote Sensing_ , 58(12):8555–8566, 2020. 

- [22] Xiangtao Zheng, Binqiang Wang, Xingqian Du, and Xiaoqiang Lu. Mutual attention inception network for remote sensing visual question answering. _IEEE Transactions on Geoscience and Remote Sensing_ , 60:1–14, 2021. 

- [23] Christel Chappuis, Vincent Mendez, Eliot Walt, Sylvain Lobry, Bertrand Le Saux, and Devis Tuia. Language transformers for remote sensing visual question answering. In _IGARSS 2022-2022 IEEE International Geoscience and Remote Sensing Symposium_ , pages 4855–4858. IEEE, 2022. 

11 

- [24] Mohamad M Al Rahhal, Yakoub Bazi, Sara O Alsaleh, Muna Al-Razgan, Mohamed Lamine Mekhalfi, Mansour Al Zuair, and Naif Alajlan. Open-ended remote sensing visual question answering with transformers. _International Journal of Remote Sensing_ , 43(18):6809–6823, 2022. 

- [25] Yakoub Bazi, Mohamad Mahmoud Al Rahhal, Mohamed Lamine Mekhalfi, Mansour Abdulaziz Al Zuair, and Farid Melgani. Bi-modal transformer-based approach for visual question answering in remote sensing imagery. _IEEE Transactions on Geoscience and Remote Sensing_ , 60:1–11, 2022. 

- [26] Zhenghang Yuan, Lichao Mou, Qi Wang, and Xiao Xiang Zhu. From easy to hard: Learning languageguided curriculum for visual question answering on remote sensing data. _IEEE Transactions on Geoscience and Remote Sensing_ , 60:1–11, 2022. 

- [27] Yuan Hu, Jianlong Yuan, Congcong Wen, Xiaonan Lu, and Xiang Li. Rsgpt: A remote sensing vision language model and benchmark. _arXiv preprint arXiv:2307.15266_ , 2023. 

- [28] Kartik Kuckreja, Muhammad Sohail Danish, Muzammal Naseer, Abhijit Das, Salman Khan, and Fahad Shahbaz Khan. Geochat: Grounded large vision-language model for remote sensing. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2024. 

- [29] Wei Zhang, Miaoxin Cai, Tong Zhang, Yin Zhuang, and Xuerui Mao. Earthgpt: A universal multi-modal large language model for multi-sensor image comprehension in remote sensing domain. _arXiv preprint arXiv:2401.16822_ , 2024. 

- [30] Bo Qu, Xuelong Li, Dacheng Tao, and Xiaoqiang Lu. Deep semantic understanding of high resolution remote sensing image. In _2016 International conference on computer, information and telecommunication systems (Cits)_ , pages 1–5. IEEE, 2016. 

- [31] Zilun Zhang, Tiancheng Zhao, Yulong Guo, and Jianwei Yin. Rs5m: A large scale vision-language dataset for remote sensing vision-language foundation model. _arXiv preprint arXiv:2306.11300_ , 2023. 

- [32] OpenAI. Gpt-4 technical report, 2023. 

- [33] Jian Ding, Nan Xue, Gui-Song Xia, Xiang Bai, Wen Yang, Michael Ying Yang, Serge Belongie, Jiebo Luo, Mihai Datcu, Marcello Pelillo, et al. Object detection in aerial images: A large-scale benchmark and challenges. _IEEE transactions on pattern analysis and machine intelligence_ , 44(11):7778–7796, 2021. 

- [34] Ke Li, Gang Wan, Gong Cheng, Liqiu Meng, and Junwei Han. Object detection in optical remote sensing images: A survey and a new benchmark. _ISPRS journal of photogrammetry and remote sensing_ , 159:296–307, 2020. 

- [35] Zhenghang Yuan, Lichao Mou, Yuansheng Hua, and Xiao Xiang Zhu. Rrsis: Referring remote sensing image segmentation. _IEEE Transactions on Geoscience and Remote Sensing_ , 2024. 

- [36] Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. A survey on hallucination in large vision-language models. _arXiv preprint arXiv:2402.00253_ , 2024. 

- [37] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. _arXiv preprint arXiv:2310.03744_ , 2023. 

- [38] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: Large language model as a unified interface for vision-language multi-task learning. _arXiv:2310.09478_ , 2023. 

- [39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pages 8748–8763. PMLR, 2021. 

- [40] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, May 2024. 

- [41] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). _arXiv preprint arXiv:1606.08415_ , 2016. 

- [42] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In _International Conference on Learning Representations_ , 2021. 

- [43] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th annual meeting of the Association for Computational Linguistics_ , pages 311–318, 2002. 

- [44] Lin Chin-Yew. Rouge: A package for automatic evaluation of summaries. In _Proceedings of the Workshop on Text Summarization Branches Out, 2004_ , 2004. 

12 

- [45] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In _Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization_ , pages 65–72, 2005. 

- [46] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 4566–4575, 2015. 

- [47] David Chan, Suzanne Petryk, Joseph E Gonzalez, Trevor Darrell, and John Canny. Clair: Evaluating image captions with large language models. In _The 2023 Conference on Empirical Methods in Natural Language Processing_ . 

- [48] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. _arXiv preprint arXiv:2403.18814_ , 2024. 

13 

## **A VRSBench Documentation and Intended Uses** 

## **A.1 Overview** 

VRSBench consists of 29,614 remote sensing images with detailed captions, 52,472 object refers, 123,221 visual question-answer pairs. VRSBench is designed to facilitate the development and evaluation of vision-language models in remote sensing, providing a comprehensive set of annotations including detailed captions, visual grounding, and visual question answering. This section documents the dataset in accordance with best practices to ensure transparency, reproducibility, and ethical usage. 

## **A.2 Data Organizing** 

Our VRSBench dataset is organized as follows. 

```
root/
Images_train.zip
Annotation_train.zip
Images_val.zip
Annotation_val.zip
VRSBench_train.json
VRSBench_EVAL_Cap.json
VRSBench_EVAL_referring.json
VRSBench_EVAL_vqa.json
```

Detailed descriptions for each folder or file are given below. 

- Images_train.zip contains all raw images in the training split. 

- Annotation_train.zip contains all annotations in the training split, one JSON file per image. 

- Images_val.zip contains all raw images in the validation split. 

- Annotation_val.zip contains all annotations in the validation split, one JSON file per image. 

- VRSBench_train.json contains all training annotations following LLaVA in standard JSON format. 

- VRSBench_EVAL_Cap.json contains all evaluation annotations for the captioning task in standard JSON format. 

- VRSBench_EVAL_referring.json contains all evaluation annotations for the visual grounding task in standard JSON format. 

- VRSBench_EVAL_vqa.json contains all evaluation annotations for the VQA task in standard JSON format. 

## **A.3 Intended Uses** 

VRSBench is intended for use in academic and research settings, specifically for: 

- Training and evaluating vision-language models capable of understanding complex visual and textual tasks. 

- Advancing the state-of-the-art in remote sensing image analysis by providing a rich dataset that supports multiple tasks. 

## **A.4 Use Cases** 

- **Academic Research** : VRSBench is ideal for exploring new algorithms in image captioning, visual grounding, and visual question answering within the remote sensing domain. 

- **Model Evaluation** : The dataset can serve as a benchmark for comparing different visionlanguage models’ performance on a standardized set of tasks. 

- **Educational Purposes** : The dataset and its comprehensive annotations can be used in coursework and workshops to teach advanced techniques in machine learning and remote sensing. 

14 

## **A.5 Limitations** 

- **Geographic Diversity** : While VRSBench includes a variety of landscapes, the geographic diversity is limited to the regions covered by the DOTA-v2 and DIOR datasets. 

- **Annotation Bias** : Despite efforts to ensure high-quality annotations through human verification, biases may exist in the interpretations of visual data due to subjective human factors. 

## **A.6 Ethical Considerations** 

- **Privacy and Sensitivity** : The dataset consists of non-sensitive, publicly available satellite images where no individual person or private property can be identified. 

- **Use Restrictions** : Users are encouraged to use VRSBench responsibly and ethically, particularly when developing applications that might impact environmental monitoring and urban planning. 

## **A.7 Documentation and Maintenance** 

- **Versioning** : Detailed version history of the dataset will be maintained to track changes and improvements over time. 

- **Community Involvement** : Feedback from the user community is encouraged to improve the dataset’s quality and applicability in various use cases. 

## **A.8 Statements for NLP** 

We employ GPT-4V [32] to generate initial annotations; for further details, please refer to the main paper. These annotations undergo a manual review by human annotators. 

## **A.9 Accountability Framework** 

To ensure responsible usage and continuous improvement, an accountability framework is established. Users of VRSBench are encouraged to report any issues or biases they encounter, contributing to an ongoing effort to refine the dataset and its annotations. 

## **B Dataset Collection Details** 

- **Source datasets** : Images are sourced from the DOTA-v2 [33] and DIOR [34] datasets and annotated with high-resolution details. We divide each image into patches measuring 512 × 512 pixels and filter out patches with no object annotations. This yields over 20,310 image patches from the DOTA-v2 dataset and 9,304 patches from the DIOR dataset. Statistics are given in Table 6. 

- **Preprocessing** : We extract image-level information and object-level information for all image patches. Note that the original DOTA-v2 and DIOR datasets contain 18 and 20 object categories respectively. We merge shared object categories and also merge small-vehicle and large-vehicle into the vehicle category. After merging, we get 26 object categories, including airplane, airport, baseball-diamond, basketball-court, bridge, chimney, container-crane, dam, expressway-service-area, expressway-toll-station, golf-field, ground-track-field, harbor, helicopter, helipad, overpass, roundabout, ship, soccer-ball-field, stadium, storage-tank, swimming-pool, tennis-court, train-station, vehicle, windmill. 

- **Object attribute extraction** : We then extract object attributes and formulate a JSON file for each image patch, including object category, corner points, bounding box, position, relative position, size, and relative size. We also determine whether each object is unique within its category or not. We do not extract object colors because objects can have complex structures with multiple colors, and we rely on GPT-4V to identify object colors. The code for preprocessing is provided at `https://github.com/lx709/VRSBench` . 

- **GPT-4V annotation generation** : The code for GPT-4V inference is provided at `https: //github.com/lx709/VRSBench` . Detailed instructions are provided in Section F.1. 

15 

Table 6: Statistics of source object detection datasets. 

|Dataset|#Images<br>#Valid Patches<br>#Selected Patches<br>Category|
|---|---|
|DOTA-v2 [33]<br>DIOR [34]|2,423<br>29,910<br>20,310<br>18<br>11,725<br>9,304<br>9,304<br>20|



## **C URL to Data and Metadata** 

The VRSBench dataset can be accessed and downloaded through our dedicated platform, which provides detailed views of the dataset components and their annotations. 

For practical examples and to download the dataset, visit our Huggingface repository ( `https: //huggingface.co/datasets/xiang709/VRSBench` ). Detailed metadata for the dataset is documented using the Croissant metadata framework, ensuring comprehensive coverage and compliance with the MLCommons Croissant standards, check [metadata]( `https://huggingface.co/api/ datasets/xiang709/VRSBench` ). Please check our Huggingface repo for metadata details. 

## **D Author Statement and Data License** 

**Author Responsibility Statement:** The authors bear all responsibilities in case of any violations of rights or ethical concerns regarding the VRSBench dataset. 

**Data License Confirmation:** The dataset is released under the [CC-BY-4.0], which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited. 

## **E Hosting and Accessibility** 

The VRSBench dataset is hosted on Huggingface ( `https://huggingface.co/datasets/ xiang709/VRSBench` ) to ensure reliable and continuous accessibility. 

**Maintenance Plan:** Ongoing maintenance and updates will be managed by the dataset authors, with updates scheduled bi-annually or as significant changes in the data sources occur. 

**Long-term Preservation:** The dataset is archived in Huggingface ( `https://huggingface.co/ datasets/xiang709/VRSBench` ) to ensure long-term availability. 

**Structured Metadata:** The annotation for each image is well-organized in standard JSON format to ensure easy usage. 

## **F Data Creation Details** 

## **F.1 GPT-4V Prompts** 

We carefully design the following instructions to prompt GPT-4V to generate annotations of image captions, referring sentences, and visual question-answering pairs. 

“You are an AI visual assistant tasked with analyzing remote sensing images. For each image, you receive image meta information and a list of objects in the format: {image source: image source, image resolution: image resolution, objects: [obj_id: object id, obj_cls: object category, obj_corner: corner point, obj_coord: object bounding box, is_unique: unique object or not, obj_position: object position, obj_rel_position: object relative position within category, obj_size: object size, obj_rel_size: object relative size within category, flag: refer or not, ...]}. The bounding box coordinates [x1, y1, x2, y2] are floating numbers from 0 to 1, corresponding to the top left x, top left y, bottom right x, and bottom right y. Note that the top-left corner coordinates are (0,0) and the bottom-right corner coordinates are (1,1). 

Your job is to create a detailed image caption and referring sentences for 1-5 distinct objects, if multiple are present, as well as a list of question-answer pairs. Each referring sentence should 

16 

unambiguously refer to one object. Finally, you need to return a JSON file in the format: {caption: detailed image caption, objects: [obj_id: object id, ref: referring sentence,...], qa_pairs: [ques_id: question id, question: question, type: question type, answer: answer]}. Do not return any notes after the JSON. 

Here are further important instructions for referring sentences: 1. Identify 1-5 distinguishable objects and provide referring sentences. Each sentence alone must independently, without seeing others, and unambiguously identify an object. 

2. Select all unique objects (is_unique=True) for creating referring sentences. Do not select objects whose flag=True for referring sentences, but still use them for captioning and question-answering tasks. 

3. Use distinctive features to describe objects. Try to use diverse object attributes such as color, shape, position, size, relative position, and relative size, but avoid specifying size details for small or large vehicles. Some object attributes are not provided, you may need to identify them from the input image. Do not explain why it is distinctive or distinguishable. 

4. For each object category, select only 1-3 most distinguishable objects and ensure the referring sentences can confidently distinguish each of them from other objects of the same category. 5. Avoid ordinal descriptors and references (first-mentioned, aforementioned, or previously mentioned) to prior mentions. Instead, use distinct features to refer back to previously identified objects. 6. If multiple object categories exist, try to include diverse object categories in a balanced manner. 7. For referring sentences, use natural language to describe objects based on their bounding box data, without directly mentioning the coordinates. Do not mention whether the object is distinguishable or not. 

8. You may include roads/bridges running east-west or north-south but do not mention object-facing directions or pointing directions. 

9. Do not mention the noses, vertical stabilizers, tails, or tail fins of planes, airplanes, or aircraft 

10. Do not mention gate numbers when describing airports or airplanes. 

11. Carefully verify each piece of information before finalizing the referring sentences, make sure each referring sentence alone can distinguish one object without any ambiguity. If not, remove this referring object. 

Here are further important instructions for image captioning: 1. Create a detailed caption for the provided image, incorporating all visible elements and object information. Focus on describing the content of the image without mentioning the reference status of objects or their flag status. 

2. Start the caption with an overview of the image. Possibly include the source of the image (if provided), specify whether it is in grayscale or color, and mention the resolution (if provided). Follow this with a description of specific, clear details within the image. Summarize the image’s content in 3-7 sentences, making sure to include counts of prominent objects. 

3. Describe only clear features; avoid uncertainties and unclear elements. Do not mention anything that is unknown or not specified. 

4. Possibly include other visual objects in the image that are not provided as inputs, such as buildings, houses, roads, and trees if they are obvious and non-ambiguous. 

5. Highlight diverse object attributes such as color, shape, position, size, relative position, and relative size. Do not add size details for small or large vehicles. 

6. Exclude imagined details not visible in the image, like weather, people, or object usage. Do not imagine the moving status of airplanes, ships, or vehicles if you are not sure about it. 

7. For roads, include features like shape (straight/curved), width, length, and orientation. 

8. For houses, mention characteristics like density, size, rooftop color, and presence of gardens. 

9. For airports, include details like boarding bridges, terminals, boarding ports, and tarmac. 

10. Carefully verify each piece of information before finalizing the caption. 

11. Do not mention whether the image is taken during the day or night. 

12. Do not mention whether the vehicles are in motion or not. 

Here are questions for visual question answering: 

1. Based on all visible elements and object information, ask 3-10 questions about diverse types, including object category, object existence, object quantity, object color, object shape, object size, object position, object direction, scene type, rural or urban, and reasoning. The category of scene type includes the main structure/type of area. Additionally, the category of reasoning is available for questions that require multifaceted analytical thought processes (e.g., object distribution pattern). 

17 

Possibly include objects that are not provided, such as houses, roads, and trees if they are obvious and non-ambiguous. 

2. Do not mention the object referred or not. Do not mention any flag information in questions and answers. 

3. Ensure each question has a definite answer without any ambiguity, and answer each question using a single word or phrase, no more than 3 words. 

4. When answering questions about the number of objects, take into account all object information. 5. Only ask questions about clear answers; avoid uncertainties or unclear elements, such as unknown, uncertain, some, or several. If the answer is uncertain or unable to be determined, remove this question-answer pair. 

6. Do not use first, second, third, fourth, fifth, first-mentioned, or previously mentioned to refer to objects, use distinguishable features to refer to specific objects mentioned before. 

7. Try to cover diverse types of questions. 

8. Do not ask about the type of view the image was captured from, or whether the image was taken during day or night. 

9. Do not ask the source of the image. 

10. Do not ask facing direction, but you may ask whether roads/bridges running east-west or northsouth. 

11. Do not ask whether the vehicles are in motion or not. 

12. Do not ask whether the image is taken during the day or night”. 

## **F.2 Human verification guidelines** 

Given an input image and associating detailed image caption, check if each piece of provided information is correct or not (no check for image source). If incorrect, correct the information, possible corrections include modifying/removing words/sentences. Modification is preferred to removing. But if a caption sentence, referring sentence, or question-answer pair is totally wrong/ambiguous/uncertain, remove it. 

- For caption annotations: Make sure each piece of information in the caption is correct. Remove uncertain or meaningless elements. Be careful of object counts, take into account all objects, both referred and not referred. 

- For object referring annotations: Make sure each referring sentence can distinguishably identify the correct object (numbered in boxes) without any ambiguity. Be careful of object color/orientation. 

- For VQA annotations: Make sure each question has a clear answer without any ambiguity, and each answer should be correct using a single word or phrase, no more than 3 words. Correct answers that specify objects by object IDs. Remove self-answered question-answer pairs. 

- Include question type for each QA, all possible question types include: object category, object existence, object quantity, object color, object shape, object size, object position, object direction, scene type, and reasoning. The category of scene type includes color or grayscale, main structure/type of area, and rural or urban. Additionally, the category of reasoning is available for questions that require multifaceted analytical thought processes (e.g., object distribution pattern). 

## **G Experimental details** 

## **G.1 Training details** 

In our experimental setup, all comparative methods are trained on a single node equipped with 4 Nvidia A100 GPUs. The batch size is standardized at 32, and each model undergoes training for a duration of five epochs. We initialize the learning rate at 2e-4 and employ a cosine learning rate decay schedule for optimization. The learning rate experiences a warm-up phase, reaching 3% of the total training steps to gradually adapt to the training regime. 

18 

## **G.2 Visual grounding using OBBs** 

**Settings** . In the main paper, horizontal bounding boxes are utilized for both training the model and evaluating its visual grounding capabilities. This section extends the evaluation to incorporate oriented bounding boxes for object localization. Given that GeoChat has demonstrated superior performance in object grounding using bounding boxes, this experiment is exclusively dedicated to exploring the effectiveness of GeoChat under the conditions of oriented bounding boxes. In our experiments, the oriented bounding box is defined by the parameters [ _cx, cy, w, h, θ_ ], where ( _cx, cy_ ) represents the center coordinates, _w_ and _h_ represent the width and height of the bounding box, respectively, and _θ_ indicates the rotation angle. 

**Results** . As shown in Table 7, the GeoChat model achieves an overall grounding accuracy of 24.3% at a threshold of 0.5, which is lower than its counterpart using horizontal bounding boxes, where the grounding accuracy reaches 49.8%. This highlights that visual grounding with oriented bounding boxes presents a greater challenge compared to grounding with horizontal bounding boxes. 

Table 7: Visual grounding performance on VRSBench dataset using orientated bounding boxes for referring object localization. Err_R denotes average angle prediction error. 

|Method|Unique<br>Acc@0.5<br>Acc@0.7|Non Unique<br>Acc@0.5<br>Acc@0.7|All|
|---|---|---|---|
||||Acc@0.5<br>Acc@0.7<br>Err_R|
|GeoChat [28]|32.3<br>12.6|18.5<br>5.7|24.3<br>8.6<br>18.7|



## **G.3 Qualitative results** 

We show qualitative results of detailed image caption in Fig. 6, visual grounding in Fig. 7, and visual question answering in grounding in Fig. 8. 

## **G.4 GPT-4V Evaluation Prompt** 

"You are an AI visual assistant tasked with analyzing remote sensing images. You receive an image and multiple object-referring sentences and visual questions. Your role is to provide a detailed caption for the image, identify object locations for all referring sentences, and answer all visual questions. 

Here are detailed instructions for image caption: Describe the image in detail in 3-7 sentences, making sure to include counts of prominent objects. Describe only clear features; avoid uncertainties and unclear elements. Do not mention anything that is unknown or not specified. Highlight diverse object attributes such as color, shape, position, size, relative position, and relative size. Do not add size details for small or large vehicles. Exclude imagined details not visible in the image, like weather, people, or object usage. Do not imagine the moving status of airplanes, ships, or vehicles if you are not sure about it. For roads, include features like shape (straight/curved), width, length, and orientation. For houses, mention characteristics like density, size, rooftop color, and presence of gardens. For airports, include details like boarding bridges, terminals, boarding ports, and tarmac. Do not mention the image is taken during the day or night. Do not mention whether the vehicles are in motion or not. Carefully verify each piece of information before finalizing the caption. 

For each referring sentence, tell me the location of the referred object in the image, and return its bounding box coordinates in the format of [x1, y1, x2, y2], which denotes the top left x, top left y, bottom right x, and bottom right y. 

Here are detailed instructions for visual question answering: For each question, answer the question based on the image content in a single word or a short phrase. 

Finally, you need to return {caption: detailed image caption, objects: [obj_id, referring_sentence, location,...], qa_pairs: [ques_id: question id, type: question type, question: question, answer: answer]} in JSON format. Do not return any notes after the JSON." 

19 

(a) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2024_VRSBench_A_Versatile_Vision-Language_Benchmark_Dataset_for_Remote_Sensing_Image_Understanding_images/nips_2024_VRSBench_A_Versatile_Vision-Language_Benchmark_Dataset_for_Remote_Sensing_Image_Understanding.pdf-0020-01.png)


**GT** : The high-resolution image from GoogleEarth shows a large house with a complex roof structure, surrounded by a well-manicured lawn with landscaping features. There is a large vehicle parked on the left side. 

**MiniGPT-v2** : 1080p high-resolution aerial image from googleearth showing a residential area with a large house surrounded by lush greenery and a small vehicle parked near the house the house has a distinctive dark roof and a driveway leading to the. 

**LLaVA** : The image features a large house with a dark roof, surrounded by a well-maintained yard. There are two vehicles parked near the house: one is located towards the left side of the image, while the other is positioned more towards the center. The yard is adorned with a variety of potted plants, with some placed closer to the house and others scattered around the area. 

**GeoChat** : The high-resolution image from GoogleEarth shows a residential area with a large house featuring a dark roof. The house is surrounded by a well-maintained lawn and is adjacent to a body of water. There is a small vehicle parked on the driveway near the house. 

## (b) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2024_VRSBench_A_Versatile_Vision-Language_Benchmark_Dataset_for_Remote_Sensing_Image_Understanding_images/nips_2024_VRSBench_A_Versatile_Vision-Language_Benchmark_Dataset_for_Remote_Sensing_Image_Understanding.pdf-0020-07.png)


**GT** : The high-resolution image from Google Earth features a body of water with three small harbors. One harbor is located near the top right corner. Another is situated along the bottom edge. The third harbor is positioned between the other two, encompassing boat slips and a pier. 

**MiniGPT-v2** : 2020 high-resolution image from googleearth showing a waterfront area with multiple harbors and a ship there are three harbors visible, with the largest one located at the top-right corner of the image the ship is positioned towards the. 

**LLaVA** : The image features a high-resolution aerial view of a harbor with a large body of water. There are two harbors visible in the scene, one located towards the right side and another towards the left side of the image. The harbor on the right is larger and extends from the middle to the right edge of the image, while the harbor on the left is smaller and situated more towards the left side. In addition to the harbors, there is a small ship situated in the middle-left part of the image, close to the water's edge. The presence of the ship and the harbors suggests a bustling waterfront area. 

**GeoChat** : The high-resolution image from GoogleEarth shows a waterfront scene with several harbors and a ship. There are three harbors visible, with the largest one located at the bottom of the image, extending from the center to the right edge. The smallest harbor is situated at the top-right corner, and the second harbor is positioned between the two, closer to the center. A unique ship is docked at the bottom-left harbor. 

Figure 6: Selected examples of detailed image caption results. We highlight correct information in green and incorrect information in red. 

20 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2024_VRSBench_A_Versatile_Vision-Language_Benchmark_Dataset_for_Remote_Sensing_Image_Understanding_images/nips_2024_VRSBench_A_Versatile_Vision-Language_Benchmark_Dataset_for_Remote_Sensing_Image_Understanding.pdf-0021-00.png)


**----- Start of picture text -----**<br>
(a) (b) (c) (d)<br>(e) (f) (g) (h)<br>MiniGPT-v2 LLaVA GeoChat GT<br>Figure 7: Selected examples of Visual grounding. (a) The tennis court on the left side of the image,<br>surrounded by a brownish surface. (b) The basketball court located at the right side of the image. (c)<br>The dome stadium situated towards the bottom-right side of the image. (d) The vehicle parked closest<br>to the top edge of the image. (e) The harbor located at the right-most edge of the image. (f) The small<br>ship located towards the bottom-left of the image. (g) The airplane is located towards the bottom of<br>the frame. (h) The left-most storage tank is fully visible and situated on the upper side of the image.<br>(a)<br>How many harbors are visible in the image?<br>GT: 4, MiniGPT-v2: 5, LLaVA: 4, GeoChat: 4<br>Is there a ship near the left harbor?<br>GT: Yes, MiniGPT-v2: 0, LLaVA: Yes, GeoChat: Yes<br>Where is the right-most harbor located?<br>GT: Middle-right, MiniGPT-v2: right edge, LLaVA: Right side, GeoChat: middle-right<br>(b)<br>What structure is located primarily in the bottom-right of the image?<br>GT: Bridge, MiniGPT-v2: Bridge, LLaVA: Highway, GeoChat: Bridge<br>What is the color of the area surrounding the bridge?<br>GT: Green, MiniGPT-v2: Green, LLaVA: Green, GeoChat: Green<br>What is the shape of the bridge?<br>GT: Curved, MiniGPT-v2: Curved, LLaVA: Curved, GeoChat: Curved<br>**----- End of picture text -----**<br>


Figure 7: Selected examples of Visual grounding. (a) The tennis court on the left side of the image, surrounded by a brownish surface. (b) The basketball court located at the right side of the image. (c) The dome stadium situated towards the bottom-right side of the image. (d) The vehicle parked closest to the top edge of the image. (e) The harbor located at the right-most edge of the image. (f) The small ship located towards the bottom-left of the image. (g) The airplane is located towards the bottom of the frame. (h) The left-most storage tank is fully visible and situated on the upper side of the image. 

Figure 8: Selected examples of VQA results. Correct answers are shown in green and incorrect answers are shown in red. 

21 

